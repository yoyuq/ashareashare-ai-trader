"""v5.6 P1-15 全市场快照缓存单一读写入口 + 陈旧标记

覆盖 `data/full_market_cache.py`:
  - write → read 往返 (schema 一致)
  - NaN 在写入侧转为 None (产出合法 JSON, 无 "NaN" 字面量)
  - `market_cache_lag_days` 计算滞后天数 / 解析失败返回 -1
"""
import json
from datetime import date, timedelta

import numpy as np
import pandas as pd

import data.full_market_cache as fmc


def _df():
    return pd.DataFrame({
        "code": ["600519", "000001"],
        "name": ["贵州茅台", "平安银行"],
        "price": [1800.0, 12.5],
        "pe_ttm": [30.5, np.nan],
    })


def test_roundtrip(monkeypatch, tmp_path):
    monkeypatch.setattr(fmc, "CACHE_PATH", tmp_path / "cache.json")
    fmc.write_full_market_cache(_df(), "2026-08-13", "tencent_realtime")
    df, d = fmc.read_full_market_cache()
    assert d == "2026-08-13"
    assert len(df) == 2
    assert list(df.columns) == ["code", "name", "price", "pe_ttm"]
    assert df["code"].tolist() == ["600519", "000001"]


def test_nan_sanitized_to_null(monkeypatch, tmp_path):
    monkeypatch.setattr(fmc, "CACHE_PATH", tmp_path / "cache.json")
    fmc.write_full_market_cache(_df(), "2026-08-13", "akshare")
    raw = (tmp_path / "cache.json").read_text(encoding="utf-8")
    assert "NaN" not in raw          # NaN 不得以字面量污染 JSON
    parsed = json.loads(raw)
    assert parsed["data"][1]["pe_ttm"] is None


def test_read_missing_returns_none(monkeypatch, tmp_path):
    monkeypatch.setattr(fmc, "CACHE_PATH", tmp_path / "nope.json")
    df, d = fmc.read_full_market_cache()
    assert df is None and d is None


def test_lag_days():
    today = date.today().isoformat()
    assert fmc.market_cache_lag_days(today) == 0
    old = (date.today() - timedelta(days=5)).isoformat()
    assert fmc.market_cache_lag_days(old) == 5
    assert fmc.market_cache_lag_days("unknown") == -1
    assert fmc.market_cache_lag_days("") == -1
