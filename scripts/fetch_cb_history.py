"""抓取全市场可转债历史数据 (零模拟, 真实历史, 含已退市债).

数据源 (AKShare, 走代理):
  - bond_zh_cov:              全列表 (含已退市 → 无幸存者偏差, 已验证 113001/110030 在列)
  - bond_zh_hs_cov_daily:     单债日线 OHLCV (新浪, 已验证 2015 起)
  - bond_zh_cov_value_analysis: 单债估值历史 (东财, 转股价值/纯债价值/转股溢价率)

输出 (分片缓存 replay_data/cb_cache/ 支持断点续传):
  replay_data/cb_daily.parquet   (date, symbol, open, high, low, close, volume)
  replay_data/cb_value.parquet   (date, symbol, close_east, conv_value, bond_value, conv_premium)
  replay_data/cb_meta.json       (债券列表+上市/到期日期)

失败即报错退出 (不兜底), 重跑自动续传。
"""
from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import akshare as ak
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent.parent
OUT = ROOT / "replay_data"
CACHE = OUT / "cb_cache" / "daily", OUT / "cb_cache" / "value"
SLEEP = 0.35  # 礼貌限速


def _save_json(obj, fp: Path) -> None:
    fp.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    for d in CACHE:
        d.mkdir(parents=True, exist_ok=True)
    daily_dir, value_dir = CACHE

    # 1) 全列表
    cov = ak.bond_zh_cov()
    cov["债券代码"] = cov["债券代码"].astype(str)
    meta = {
        r["债券代码"]: {
            "name": r["债券简称"],
            "list_date": str(r.get("上市日期", "")),
            "maturity": str(r.get("到期日期", "")),
        }
        for _, r in cov.iterrows()
    }
    _save_json(meta, OUT / "cb_meta.json")
    codes = sorted(meta.keys())
    print(f"total bonds: {len(codes)}", flush=True)

    def _ex(sym: str) -> str:
        return ("sh" if sym.startswith(("11",)) else "sz") + sym

    def _fetch_with_retry(fn, retries: int = 4):
        """瞬时网络错误 (SSL EOF/代理断连) 重试; 数据性错误不重试."""
        import requests
        for i in range(retries):
            try:
                return fn()
            except (requests.exceptions.SSLError, requests.exceptions.ProxyError,
                    requests.exceptions.ConnectionError) as e:
                if i == retries - 1:
                    raise
                print(f"[retry {i + 1}] {type(e).__name__}, 5s 后重试", flush=True)
                time.sleep(5)

    def _blank_or_empty(e: Exception) -> bool:
        if isinstance(e, KeyError) and "date" in str(e):
            return True  # 新浪对无行情债返回异形空表, akshare 内部解析 KeyError('date')
        if isinstance(e, TypeError) and "NoneType" in str(e):
            return True  # 东财对无估值数据债返回 result=None
        s = str(e)
        return "没有数据" in s or "(((9" in s or "记录" in s

    # 2) 日线
    for i, code in enumerate(codes):
        fp = daily_dir / f"{code}.parquet"
        if fp.exists():
            continue
        try:
            df = _fetch_with_retry(lambda: ak.bond_zh_hs_cov_daily(symbol=_ex(code)))
        except Exception as e:
            if _blank_or_empty(e):
                fp.write_bytes(b"")  # 空分片 = 确认无行情, 不再重试
                continue
            raise
        if df is not None and len(df):
            df = df.copy()
            df["symbol"] = code
            df.to_parquet(fp)
        else:
            fp.write_bytes(b"")
        if i % 50 == 0:
            print(f"daily {i}/{len(codes)}", flush=True)
        time.sleep(SLEEP)

    # 3) 估值历史 (转股溢价率)
    for i, code in enumerate(codes):
        fp = value_dir / f"{code}.parquet"
        if fp.exists():
            continue
        try:
            va = _fetch_with_retry(lambda: ak.bond_zh_cov_value_analysis(symbol=code))
        except Exception as e:
            if _blank_or_empty(e):
                fp.write_bytes(b"")
                continue
            raise
        if va is not None and len(va):
            va.to_parquet(fp)
        else:
            fp.write_bytes(b"")
        if i % 50 == 0:
            print(f"value {i}/{len(codes)}", flush=True)
        time.sleep(SLEEP)

    # 4) 合并落盘 (旧分片缺 symbol 列 → 从文件名回填)
    def _collect(d: Path) -> list[pd.DataFrame]:
        frames = []
        for p in d.glob("*.parquet"):
            if not p.stat().st_size:
                continue
            df = pd.read_parquet(p)
            if "symbol" not in df.columns:
                df["symbol"] = p.stem
            frames.append(df)
        return frames

    frames = _collect(daily_dir)
    if frames:
        daily = pd.concat(frames, ignore_index=True)
        daily["date"] = pd.to_datetime(daily["date"])
        daily.to_parquet(OUT / "cb_daily.parquet")
        print(f"cb_daily.parquet: {len(daily)} rows, {daily['symbol'].nunique()} bonds")
    frames = _collect(value_dir)
    if frames:
        value = pd.concat(frames, ignore_index=True)
        value["日期"] = pd.to_datetime(value["日期"])
        value.to_parquet(OUT / "cb_value.parquet")
        print(f"cb_value.parquet: {len(value)} rows, {value['symbol'].nunique()} bonds")
    print("DONE")


if __name__ == "__main__":
    main()
