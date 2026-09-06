"""人工操作单防呆测试。所有行情/持仓均是明确的单测 fixture，不联网。"""
from __future__ import annotations

import copy
import json
from datetime import datetime, timedelta
from decimal import Decimal as D
from pathlib import Path
from unittest.mock import Mock

import pandas as pd
import pytest

from scripts import capture_order_quotes as capture
from scripts import sandbox_order_sheet as sheet

NOW = datetime(2026, 9, 4, 14, 0, tzinfo=sheet.MARKET_TZ)
DAY = NOW.date()
SYM = "sh.600000"


def quote(price="10", **changes):
    result = {"price": price, "observed_at": (NOW - timedelta(seconds=30)).isoformat(),
              "is_st": False, "suspended": False, "limit_up": str(D(price) * D("1.1")),
              "limit_down": str(D(price) * D("0.9")), "status_source": "mock status provider (unit test)",
              "status_observed_at": (NOW - timedelta(seconds=30)).isoformat()}
    result.update(changes)
    return result


def validate(orders=None, quotes=None, cash=D("10000"), positions=None, limits=None, now=NOW, traded=D("0")):
    return sheet.validate_orders(
        orders if orders is not None else [sheet.Order(SYM, "buy", D("10"), 100)],
        quotes if quotes is not None else {"source": "mock quote provider (unit test)", "quotes": {SYM: quote()}},
        cash, positions or {}, DAY, now, limits or sheet.Limits(), traded)


@pytest.mark.parametrize("shares", [0, 150, -100, True])
def test_reject_non_positive_or_non_lot_shares(shares):
    assert not validate([sheet.Order(SYM, "buy", D("10"), shares)]).passed


def test_single_order_limit_inclusive():
    assert validate(limits=sheet.Limits(max_order_amount=D("1000"))).passed
    result = validate(limits=sheet.Limits(max_order_amount=D("999.99")))
    assert any("单笔金额超限" in error for error in result.errors)


def test_daily_limit_includes_sell_buy_and_already_traded():
    other = "sz.000001"
    orders = [sheet.Order(SYM, "sell", D("10"), 100), sheet.Order(other, "buy", D("10"), 100)]
    quotes = {"source": "mock", "quotes": {SYM: quote(), other: quote()}}
    result = validate(orders, quotes, positions={SYM: {"available_shares": 100}},
                      limits=sheet.Limits(max_daily_amount=D("2500")), traded=D("501"))
    assert any("单日总金额超限" in error for error in result.errors)


@pytest.mark.parametrize("price,accepted", [("10.5", True), ("9.5", True), ("10.5001", False), ("9.4999", False)])
def test_price_deviation_five_percent_boundary(price, accepted):
    result = validate([sheet.Order(SYM, "buy", D(price), 100)])
    assert result.passed is accepted


@pytest.mark.parametrize("change,reason", [({"is_st": True}, "ST状态"),
    ({"suspended": True}, "停牌状态"), ({"limit_up": "10"}, "涨停状态"),
    ({"limit_down": "10"}, "跌停状态"), ({"suspended": "false"}, "缺少明确停牌"),
    ({"is_st": None}, "缺少明确ST"), ({"status_source": ""}, "真实来源"),
    ({"limit_up": None}, "涨停价"), ({"price": "NaN"}, "有限"),
    ({"price": "0"}, "正数")])
def test_reject_unexecutable_or_missing_quote_fields(change, reason):
    result = validate(quotes={"source": "mock", "quotes": {SYM: quote(**change)}})
    assert not result.passed
    assert any(reason in error for error in result.errors)


@pytest.mark.parametrize("change", [
    {"observed_at": (NOW - timedelta(seconds=301)).isoformat()},
    {"observed_at": (NOW + timedelta(seconds=1)).isoformat()},
    {"observed_at": (NOW - timedelta(days=1)).isoformat()},
    {"observed_at": "2026-09-04T14:00:00"},
    {"status_observed_at": (NOW - timedelta(seconds=301)).isoformat()},
])
def test_reject_stale_future_and_timezone_less_snapshots(change):
    assert not validate(quotes={"source": "mock", "quotes": {SYM: quote(**change)}}).passed


@pytest.mark.parametrize("hour,minute", [(9, 29), (11, 32), (12, 0), (15, 2)])
def test_reject_outside_execution_session(hour, minute):
    now = NOW.replace(hour=hour, minute=minute)
    q = quote(observed_at=now.isoformat(), status_observed_at=now.isoformat())
    result = validate(quotes={"source": "mock", "quotes": {SYM: q}}, now=now)
    assert any("交易时段" in error for error in result.errors)


def test_reject_weekend_even_with_current_mock_quote():
    now = NOW + timedelta(days=2)
    q = quote(observed_at=now.isoformat(), status_observed_at=now.isoformat())
    assert not sheet.validate_orders([sheet.Order(SYM, "buy", D("10"), 100)],
        {"source": "mock", "quotes": {SYM: q}}, D("10000"), {}, now.date(), now, sheet.Limits()).passed


@pytest.mark.parametrize("quotes", [{}, {"source": "mock", "quotes": {}}, {"quotes": {SYM: quote()}}])
def test_missing_quote_or_source_never_falls_back(quotes):
    assert not validate(quotes=quotes).passed


def test_fee_components_buy_sell_and_minimum():
    limits = sheet.Limits()
    assert sheet.fees(D("100000"), "buy", limits) == {"commission": D("30.00"), "stamp_duty": D("0.00"), "transfer": D("1.00")}
    assert sheet.fees(D("100000"), "sell", limits) == {"commission": D("30.00"), "stamp_duty": D("50.00"), "transfer": D("1.00")}
    assert sheet.fees(D("1000"), "buy", limits)["commission"] == D("5.00")
    assert sheet.fees(D("1500"), "buy", limits)["transfer"] == D("0.02")


def test_cash_must_cover_buy_fees_at_inclusive_boundary():
    assert not validate(cash=D("1005.00")).passed
    assert validate(cash=D("1005.01")).passed


def test_cash_and_amount_limits_use_higher_live_price():
    result = validate(quotes={"source": "mock", "quotes": {SYM: quote(price="10.4")}}, cash=D("1005.01"))
    assert result.buy_required == D("1045.01")
    assert not result.passed


def test_sell_proceeds_cannot_fund_pending_buys():
    other = "sz.000001"
    result = validate([sheet.Order(SYM, "sell", D("10"), 100), sheet.Order(other, "buy", D("10"), 100)],
        {"source": "mock", "quotes": {SYM: quote(), other: quote()}}, cash=D("0"),
        positions={SYM: {"available_shares": 100}})
    assert result.sell_net == D("994.49")
    assert any("现金不足" in error for error in result.errors)


def test_sell_fee_shortfall_also_requires_cash():
    args = {"orders": [sheet.Order(SYM, "sell", D("0.04"), 100)],
            "quotes": {"source": "mock", "quotes": {SYM: quote("0.04")}},
            "positions": {SYM: {"available_shares": 100}}}
    result = validate(**args, cash=D("0"))
    assert result.sell_fee_shortfall == D("1.00")
    assert not result.passed
    assert validate(**args, cash=D("1.00")).passed


@pytest.mark.parametrize("available", [0, 99, None])
def test_sell_requires_actual_available_holdings(available):
    result = validate([sheet.Order(SYM, "sell", D("10"), 100)], positions={SYM: {"available_shares": available}})
    assert not result.passed


def frames():
    rows = []
    for index in range(12):
        for offset, day in enumerate(pd.bdate_range(end=DAY, periods=20)):
            rows.append({"date": day, "symbol": f"sh.{600000 + index}", "isST": "0", "tradestatus": "1",
                         "turn": 1 + index, "amount": (1 + index) * 1000000, "close": 9 + index / 100,
                         "peTTM": 10, "pbMRQ": 1, "pctChg": (index + 1) * (offset % 2)})
    events = pd.DataFrame([{"sym": "sh.609999", "free_date": pd.Timestamp(DAY), "ratio": 0.04}])
    return pd.DataFrame(rows), events


@pytest.fixture
def inputs(tmp_path):
    panel, events = frames()
    panel.to_parquet(tmp_path / "panel.parquet", index=False)
    events.to_parquet(tmp_path / "events.parquet", index=False)
    account = {"edge_id": "cold_lowvol_top5_hold", "entry_date": "2026-08-31", "as_of": NOW.isoformat(),
               "available_cash": "10000", "daily_traded_amount": "0", "positions": []}
    quotes = {"source": "mock quote provider (unit test)", "quotes": {
        f"sh.{600000 + i}": quote(str(9 + i / 100)) for i in range(12)}}
    (tmp_path / "account.json").write_text(json.dumps(account), encoding="utf-8")
    (tmp_path / "quotes.json").write_text(json.dumps(quotes), encoding="utf-8")
    args = ["--panel", str(tmp_path / "panel.parquet"), "--events", str(tmp_path / "events.parquet"),
            "--account", str(tmp_path / "account.json"), "--quotes", str(tmp_path / "quotes.json"),
            "--output", str(tmp_path / "sheet.md"), "--audit-log", str(tmp_path / "audit.jsonl")]
    return tmp_path, args


def audit_rows(directory):
    return [json.loads(line) for line in (directory / "audit.jsonl").read_text(encoding="utf-8").splitlines()]


def test_default_dry_run_prints_without_sheet_and_audits(inputs, capsys):
    directory, args = inputs
    assert sheet.main(args, now=NOW) == 0
    assert not (directory / "sheet.md").exists()
    assert "dry-run：未写出操作单" in capsys.readouterr().out
    row = audit_rows(directory)[-1]
    assert row["status"] == "previewed" and row["validation"]["passed"]
    assert row["time"] and len(row["input_snapshot_hash"]) == 64 and row["output_path"]


def test_confirm_idempotent_bytes_despite_different_invocation_time(inputs):
    directory, args = inputs
    assert sheet.main(args + ["--confirm"], now=NOW) == 0
    content = (directory / "sheet.md").read_bytes()
    assert sheet.main(args + ["--confirm"], now=NOW + timedelta(seconds=1)) == 0
    assert content == (directory / "sheet.md").read_bytes()
    assert audit_rows(directory)[-1]["status"] == "unchanged"
    assert b"SHA-256" in content and b"2026-09-04" in content
    assert b"\r\n" not in content


def test_same_snapshot_dry_run_and_confirm_render_identical_sheet(inputs, capsys):
    directory, args = inputs
    assert sheet.main(args, now=NOW) == 0
    preview = capsys.readouterr().out.split("\ndry-run：", 1)[0]
    assert sheet.main(args + ["--confirm"], now=NOW) == 0
    assert preview.encode("utf-8") == (directory / "sheet.md").read_bytes()


def test_changed_input_changes_hash_and_cannot_confirm_second_daily_plan(inputs):
    directory, args = inputs
    assert sheet.main(args + ["--confirm"], now=NOW) == 0
    first = audit_rows(directory)[-1]["input_snapshot_hash"]
    original = (directory / "sheet.md").read_bytes()
    # 仅改变真实输入的原始字节也属于新快照。
    with (directory / "account.json").open("a", encoding="utf-8") as stream:
        stream.write("\n")
    assert sheet.main(args + ["--confirm", "--output", str(directory / "other.md")], now=NOW) == 2
    latest = audit_rows(directory)[-1]
    assert latest["input_snapshot_hash"] != first
    assert not latest["validation"]["passed"]
    assert "同日已有" in latest["error"]
    assert (directory / "sheet.md").read_bytes() == original
    assert not (directory / "other.md").exists()


def test_output_conflict_preserves_existing_bytes_and_rejects(inputs):
    directory, args = inputs
    (directory / "sheet.md").write_bytes(b"existing user content")
    assert sheet.main(args + ["--confirm"], now=NOW) == 2
    assert (directory / "sheet.md").read_bytes() == b"existing user content"
    assert not audit_rows(directory)[-1]["validation"]["passed"]


def test_rejection_is_audited_without_output(inputs):
    directory, args = inputs
    account = json.loads((directory / "account.json").read_text())
    account["available_cash"] = 0
    (directory / "account.json").write_text(json.dumps(account))
    assert sheet.main(args + ["--confirm"], now=NOW) == 2
    assert not (directory / "sheet.md").exists()
    assert any("现金不足" in e for e in audit_rows(directory)[-1]["validation"]["errors"])


def test_missing_inputs_refuse_instead_of_reading_panel(tmp_path, monkeypatch):
    read = Mock(side_effect=AssertionError("unexpected input read"))
    monkeypatch.setattr(Path, "read_bytes", read)
    assert sheet.main(["--audit-log", str(tmp_path / "audit.jsonl")], now=NOW) == 2
    read.assert_not_called()


@pytest.mark.parametrize("field", ["--panel", "--events", "--account", "--quotes", "--output", "--audit-log"])
@pytest.mark.parametrize("sensitive", ["reports/private.parquet", ".env", ".mcp.json"])
def test_protected_paths_rejected_before_any_file_read(inputs, monkeypatch, field, sensitive):
    directory, args = inputs
    read = Mock(side_effect=AssertionError("must reject before reading"))
    monkeypatch.setattr(Path, "read_bytes", read)
    assert sheet.main(args + [field, str(directory / sensitive)], now=NOW) == 2
    read.assert_not_called()
    assert not (directory / sensitive).exists()


def test_symlink_resolved_target_cannot_enter_protected_location(tmp_path):
    link = tmp_path / "input-link"
    # 不创建/读取 reports；仅造指向禁止位置的悬空符号链接。
    try:
        link.symlink_to(tmp_path / "reports" / "secret.parquet")
    except OSError:
        pytest.skip("Windows 当前账户无 symlink 权限")
    with pytest.raises(sheet.OrderSheetError, match="禁止访问"):
        sheet._safe_path(link)


def test_protected_resolved_path_guard_without_os_symlink_permission(tmp_path, monkeypatch):
    original = Path.resolve
    link = tmp_path / "alias.parquet"

    def resolve(path, *args, **kwargs):
        if path == link:
            return tmp_path / "reports" / "private.parquet"
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", resolve)
    with pytest.raises(sheet.OrderSheetError, match="禁止访问"):
        sheet._safe_path(link)


def test_input_audit_alias_refuses_even_when_other_inputs_missing(tmp_path):
    account = tmp_path / "account.json"
    account.write_bytes(b"user input")
    assert sheet.main(["--account", str(account), "--audit-log", str(account)], now=NOW) == 2
    assert account.read_bytes() == b"user input"


def test_audit_failure_blocks_sheet_creation(inputs, monkeypatch):
    directory, args = inputs
    monkeypatch.setattr(sheet, "_append_audit", Mock(side_effect=OSError("disk failure")))
    assert sheet.main(args + ["--confirm"], now=NOW) == 2
    assert not (directory / "sheet.md").exists()


def test_final_audit_failure_reports_existing_file_and_keeps_reservation(inputs, monkeypatch, capsys):
    directory, args = inputs
    original = sheet._append_audit
    calls = 0

    def fail_after_reservation(path, row):
        nonlocal calls
        calls += 1
        if calls > 1:
            raise OSError("disk full")
        original(path, row)

    monkeypatch.setattr(sheet, "_append_audit", fail_after_reservation)
    assert sheet.main(args + ["--confirm"], now=NOW) == 2
    assert (directory / "sheet.md").exists()
    assert audit_rows(directory)[-1]["status"] == "reserved"
    assert "文件已写出/复用" in capsys.readouterr().err


def test_parallel_lock_blocks_without_unlocked_audit_append(inputs):
    directory, args = inputs
    lock = directory / "audit.jsonl.lock"
    lock.write_bytes(b"other process")
    assert sheet.main(args + ["--confirm"], now=NOW) == 2
    assert lock.read_bytes() == b"other process"
    assert not (directory / "sheet.md").exists()
    assert not (directory / "audit.jsonl").exists()


def test_clock_rechecked_after_expensive_selection(inputs, monkeypatch):
    directory, args = inputs
    selection = sheet.select_orders
    observed_now = NOW

    class ChangingClock(datetime):
        @classmethod
        def now(cls, tz=None):
            return observed_now

    def slow_selection(*values, **kwargs):
        nonlocal observed_now
        result = selection(*values, **kwargs)
        observed_now = NOW + timedelta(minutes=6)
        return result

    monkeypatch.setattr(sheet, "datetime", ChangingClock)
    monkeypatch.setattr(sheet, "select_orders", slow_selection)
    assert sheet.main(args + ["--confirm"]) == 2
    assert not (directory / "sheet.md").exists()
    assert any("过期" in error for error in audit_rows(directory)[-1]["validation"]["errors"])


def test_clock_rechecked_after_audit_before_write(inputs, monkeypatch):
    directory, args = inputs
    append = sheet._append_audit
    observed_now = NOW

    class ChangingClock(datetime):
        @classmethod
        def now(cls, tz=None):
            return observed_now

    def slow_audit(path, record):
        nonlocal observed_now
        append(path, record)
        if record["status"] == "reserved":
            observed_now = NOW.replace(hour=15, minute=1)

    monkeypatch.setattr(sheet, "datetime", ChangingClock)
    monkeypatch.setattr(sheet, "_append_audit", slow_audit)
    assert sheet.main(args + ["--confirm"]) == 2
    assert not (directory / "sheet.md").exists()
    rows = audit_rows(directory)
    assert rows[0]["status"] == "reserved" and rows[-1]["status"] == "rejected"
    assert any("交易时段" in error for error in rows[-1]["validation"]["errors"])


def test_reserved_plan_survives_output_failure_and_rejects_new_hash(inputs, monkeypatch):
    directory, args = inputs
    monkeypatch.setattr(sheet, "_write_once", Mock(side_effect=OSError("write failure")))
    assert sheet.main(args + ["--confirm"], now=NOW) == 2
    assert audit_rows(directory)[0]["status"] == "reserved"
    with (directory / "account.json").open("a", encoding="utf-8") as stream:
        stream.write("\n")
    assert sheet.main(args + ["--confirm"], now=NOW) == 2
    assert "同日已有" in audit_rows(directory)[-1]["error"]


def test_frozen_selection_keeps_buffer_skips_unaffordable_and_uses_real_sell_shares():
    panel, events = frames()
    # 第0名买不起一手；第1名解禁剔除；第2名保留；第11名跌出 KEEP_ZONE。
    panel.loc[panel.symbol == "sh.600000", "close"] = 25
    events.loc[0, ["sym", "ratio"]] = ["sh.600001", 0.05]
    positions = {"sh.600002": {"shares": 300, "available_shares": 300},
                 "sh.600011": {"shares": 700, "available_shares": 700}}
    orders, context = sheet.select_orders(panel, events, positions, 10000, DAY)
    assert context["keep"] == ["sh.600002"]
    assert context["n_blocked"] == 1
    assert orders[0] == sheet.Order("sh.600011", "sell", D("9.11"), 700)
    assert [o.symbol for o in orders if o.side == "buy"] == [f"sh.{i}" for i in range(600003, 600007)]
    assert [o.shares for o in orders if o.side == "buy"] == [200] * 4


def provider_response(timestamp="20260904135930"):
    fields = [""] * 49
    fields[1], fields[2], fields[3] = "浦发银行", "600000", "10"
    fields[30], fields[47], fields[48] = timestamp, "11", "9"
    return 'v_sh600000="' + "~".join(fields) + '";'


def test_capture_uses_real_response_fields_and_keeps_evidence():
    session = Mock()
    session.get.return_value.text = provider_response()
    statuses = {"source": "mock exchange status (unit test)", "statuses": {SYM: {
        "observed_at": NOW.isoformat(), "suspended": False, "is_st": False}}}
    result = capture.capture_quotes(statuses, now=NOW, session=session)
    assert result["quotes"][SYM]["observed_at"] == "2026-09-04T13:59:30+08:00"
    assert result["quotes"][SYM]["limit_up"] == "11"
    assert result["raw_response"] == provider_response()
    assert len(result["raw_response_sha256"]) == 64
    session.get.assert_called_once_with("https://qt.gtimg.cn/q=sh600000", timeout=15)


@pytest.mark.parametrize("response", [provider_response("20260903135930"), provider_response(""), "v_sh600000=\"\";", ""])
def test_capture_missing_or_stale_provider_data_never_fills_in(response):
    session = Mock()
    session.get.return_value.text = response
    statuses = {"source": "mock status", "statuses": {SYM: {
        "observed_at": NOW.isoformat(), "suspended": False, "is_st": False}}}
    with pytest.raises(sheet.OrderSheetError):
        capture.capture_quotes(statuses, now=NOW, session=session)


def test_capture_unavailable_status_does_not_call_network():
    session = Mock()
    statuses = {"source": "mock status", "statuses": {SYM: {"observed_at": NOW.isoformat(), "is_st": False}}}
    with pytest.raises(sheet.OrderSheetError, match="suspended"):
        capture.capture_quotes(statuses, now=NOW, session=session)
    session.get.assert_not_called()


def test_capture_accepts_new_tick_using_clock_after_http_completion(monkeypatch):
    observed_now = NOW

    class ChangingClock(datetime):
        @classmethod
        def now(cls, tz=None):
            return observed_now

    session = Mock()

    def http_response(*args, **kwargs):
        nonlocal observed_now
        observed_now = NOW + timedelta(seconds=2)
        response = Mock()
        response.text = provider_response("20260904140001")
        return response

    session.get.side_effect = http_response
    monkeypatch.setattr(capture, "datetime", ChangingClock)
    statuses = {"source": "mock status", "statuses": {SYM: {
        "observed_at": NOW.isoformat(), "suspended": False, "is_st": False}}}
    result = capture.capture_quotes(statuses, session=session)
    assert result["quotes"][SYM]["observed_at"] == "2026-09-04T14:00:01+08:00"
    assert result["quotes"][SYM]["observed_at"] != observed_now.isoformat()


def test_missing_actual_cash_or_daily_turnover_refuses(inputs):
    directory, args = inputs
    original = json.loads((directory / "account.json").read_text())
    for field in ("available_cash", "daily_traded_amount"):
        account = copy.deepcopy(original)
        del account[field]
        (directory / "account.json").write_text(json.dumps(account))
        assert sheet.main(args + ["--confirm"], now=NOW) == 2
        assert not (directory / "sheet.md").exists()
