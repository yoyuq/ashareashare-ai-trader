"""Offline contracts for truthful, bounded public-source capability probes."""
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import requests

from scripts import probe_source_inventory as inventory


def test_empty_response_never_counts_as_coverage():
    with pytest.raises(ValueError, match="empty response"):
        inventory.summarize_frame(pd.DataFrame(), inventory.SOURCE_MAP["daily_em"])


def test_history_uses_actual_minimum_and_retains_announcement_nonnull_count():
    frame = pd.DataFrame({"REPORT_DATE": ["2020-12-31", "2017-03-31"],
                          "NOTICE_DATE": ["2021-03-05", None]})
    result = inventory.summarize_frame(frame, inventory.SOURCE_MAP["balance"])
    assert result["first"] == "2017-03-31T00:00:00"
    assert result["pit_fields"] == {"NOTICE_DATE": 1}
    assert result["rows"] == 2
    assert len(result["payload_sha256"]) == 64


def test_missing_or_invalid_history_date_is_failure():
    for frame in (pd.DataFrame({"price": [1]}), pd.DataFrame({"日期": ["unknown"]})):
        with pytest.raises(ValueError):
            inventory.summarize_frame(frame, inventory.SOURCE_MAP["daily_em"])


def test_proxy_failure_has_exactly_one_direct_retry(monkeypatch):
    calls = []

    def fail_then_succeed(source, code, pages):
        calls.append(os.environ.get("HTTP_PROXY"))
        if len(calls) == 1:
            raise requests.exceptions.ProxyError("proxy unavailable")
        return pd.DataFrame({"日期": ["2020-01-02"]}), {}

    monkeypatch.setattr(inventory, "fetch", fail_then_succeed)
    result = inventory.probe_sample(inventory.SOURCE_MAP["daily_em"], "600519", "http://localhost:7897", 2, 1)
    assert calls == ["http://localhost:7897", None]
    assert result["status"] == "ok"
    assert [item["mode"] for item in result["attempts"]] == ["proxy", "direct"]


def test_double_failure_is_reported_without_synthetic_rows(monkeypatch):
    def fail(*args):
        raise TimeoutError("upstream unavailable")

    monkeypatch.setattr(inventory, "fetch", fail)
    result = inventory.probe_sample(inventory.SOURCE_MAP["daily_em"], "600519", "http://localhost:7897", 2, 1)
    assert result["status"] == "unavailable"
    assert result["first"] is None
    assert result["rows"] == 0
    assert len(result["attempts"]) == 2


def test_transport_restores_environment_and_request_method_on_failure(monkeypatch):
    monkeypatch.setenv("HTTP_PROXY", "http://original:1234")
    monkeypatch.setenv("ALL_PROXY", "http://other:5678")
    original = requests.sessions.Session.request
    with pytest.raises(RuntimeError):
        with inventory.transport("direct", "http://unused:7897", 2):
            assert "HTTP_PROXY" not in os.environ
            assert "ALL_PROXY" not in os.environ
            raise RuntimeError("failure")
    assert os.environ["HTTP_PROXY"] == "http://original:1234"
    assert os.environ["ALL_PROXY"] == "http://other:5678"
    assert requests.sessions.Session.request is original


def test_dc_pagination_records_truncation_and_checks_security_filter(monkeypatch):
    params_seen = []

    def request(url, params):
        params_seen.append(dict(params))
        payload = {"success": True, "result": {"pages": 3, "count": 1200,
                   "data": [{"SECURITY_CODE": "600519", "REPORT_DATE": "2001-12-31"}]}}
        return SimpleNamespace(raise_for_status=lambda: None, json=lambda: payload)

    monkeypatch.setattr(requests, "get", request)
    frame, evidence = inventory.fetch_dc(inventory.SOURCE_MAP["balance"], "600519", 2)
    assert len(frame) == 2
    assert evidence["truncated"] is True
    assert evidence["server_count"] == 1200
    assert [params["pageNumber"] for params in params_seen] == [1, 2]
    assert params_seen[0]["sortTypes"] == "1"
    assert params_seen[0]["filter"] == '(SECURITY_CODE="600519")'


@pytest.mark.parametrize("row", [
    {"SECURITY_CODE": "000001", "REPORT_DATE": "2020-01-01"},
    {"SECURITY_CODE": "600519", "wrong_date": "2020-01-01"},
])
def test_dc_rejects_wrong_stock_or_missing_sort_field(monkeypatch, row):
    payload = {"success": True, "result": {"pages": 1, "count": 1, "data": [row]}}
    monkeypatch.setattr(requests, "get", lambda *a, **kw: SimpleNamespace(raise_for_status=lambda: None, json=lambda: payload))
    with pytest.raises(ValueError):
        inventory.fetch_dc(inventory.SOURCE_MAP["balance"], "600519", 1)


def test_errors_strip_url_credentials_and_query_values():
    error = ValueError("failed https://user:secret@host.invalid/path?api_key=secret")
    assert "secret" not in inventory.safe_error(error)


@pytest.mark.parametrize("path", ["reports/probe.json", "docs/.env", "docs/.mcp.json"])
def test_rejects_forbidden_output_paths(path):
    with pytest.raises(ValueError):
        inventory.safe_output(Path(path))


def test_sample_validation_and_partial_overwrite_guard():
    with pytest.raises(SystemExit):
        inventory.parse_args(["--codes", "600519"])
    with pytest.raises(SystemExit):
        inventory.parse_args(["--only", "balance"])
    with pytest.raises(SystemExit):
        inventory.parse_args(["--only", "balance", "--output", "docs/data_source_inventory.md"])
    with pytest.raises(SystemExit):
        inventory.parse_args(["--output", "docs/same.md", "--json-output", "docs/same.md"])


def test_deadline_does_not_claim_unfinished_samples_were_tested(monkeypatch):
    def timeout(*args, **kwargs):
        raise inventory.subprocess.TimeoutExpired("worker", 1, output=(json.dumps({"code": "600519", "status": "ok", "rows": 1}) + "\n").encode())

    monkeypatch.setattr(inventory.subprocess, "run", timeout)
    args = SimpleNamespace(codes=list(inventory.SAMPLES), timeout=1, pages=1, delay=0, proxy="http://localhost:7897", source_timeout=1)
    result = inventory.run_source(inventory.SOURCE_MAP["balance"], args)
    assert len(result["samples"]) == 10
    assert result["samples"][0]["status"] == "ok"
    assert all(sample["status"] == "incomplete" and sample["attempts"] == [] for sample in result["samples"][1:])


def test_truncated_worker_record_does_not_discard_completed_evidence(monkeypatch):
    output = json.dumps({"code": "600519", "status": "ok", "rows": 1}) + '\n{"code":"600036"'
    monkeypatch.setattr(inventory.subprocess, "run", lambda *a, **kw: SimpleNamespace(stdout=output, returncode=1))
    args = SimpleNamespace(codes=list(inventory.SAMPLES), timeout=1, pages=1, delay=0, proxy="http://localhost:7897", source_timeout=1)
    result = inventory.run_source(inventory.SOURCE_MAP["balance"], args)
    assert result["samples"][0]["status"] == "ok"
    assert len(result["samples"]) == 10
    assert result["samples"][1]["status"] == "incomplete"
