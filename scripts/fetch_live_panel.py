"""Live 面板抓取 — 前瞻双臂平行 bet 基础设施 (playbook 待办 09-02).

从 Baostock 抓取流动池 (top-1000 by amount, 与 _filter_liquid 同口径) 最近 70 自然日
日 K (close/amount/turn/peTTM/isST), 落 replay_data/live_panel.parquet。
用途: 前瞻期任意交易日点内重建 score = -(turn_rank + std20_rank + log_mkt_rank)
(与回测 harness 口径一致: log_mkt = amount×100/turn/1e8 流通市值代理)。

增量: 已有 parquet 则只抓缺的新日期 (per-symbol 从上次日期续抓)。
零模拟: Baostock 失败重试 2 次, 仍失败该票标 fail 并退出码 2 (不伪造, 不兜底)。
"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from data.full_market_cache import read_full_market_cache  # noqa: E402
from forward_register_cold_lowvol import _filter_liquid  # noqa: E402

PANEL_FP = ROOT / "replay_data" / "live_panel.parquet"
LOOKBACK_DAYS = 70          # 自然日, ≈46 交易日 (std20 需要 20)
TOP_N = 1000
_FIELDS = "date,code,open,high,low,close,volume,amount,turn,pctChg,peTTM,pbMRQ,tradestatus,isST"


def _force_utf8() -> None:
    for name in ("stdout", "stderr"):
        s = getattr(sys, name, None)
        if hasattr(s, "reconfigure"):
            try:
                s.reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass


def _query_one(bs, symbol: str, start: str, end: str) -> pd.DataFrame | None:
    """单票日K, 重试 2 次; None = 重试后仍失败 (零模拟: 不当空数据用)。"""
    for attempt in range(3):
        try:
            rs = bs.query_history_k_data_plus(
                symbol, _FIELDS, start_date=start, end_date=end,
                frequency="d", adjustflag="3",
            )
            rows = []
            while rs.error_code == "0" and rs.next():
                rows.append(rs.get_row_data())
            if rs.error_code != "0":
                raise RuntimeError(f"baostock error {rs.error_code}: {rs.error_msg}")
            if not rows:
                return pd.DataFrame()
            df = pd.DataFrame(rows, columns=rs.fields)
            for c in ("open", "high", "low", "close", "volume", "amount",
                      "turn", "pctChg", "peTTM", "pbMRQ"):
                df[c] = pd.to_numeric(df[c], errors="coerce")
            return df
        except Exception as e:  # noqa: BLE001
            if attempt == 2:
                print(f"[fail] {symbol}: {e}", flush=True)
                return None
            time.sleep(5 * (attempt + 1))
            try:
                bs.login()
            except Exception:  # noqa: BLE001
                pass
    return None


def main() -> int:
    _force_utf8()
    import baostock as bs

    df_cache, cache_date = read_full_market_cache()
    if df_cache is None:
        print("无全市场缓存, 先跑 scripts/refresh_market_cache.py")
        return 1
    liq = _filter_liquid(df_cache).nlargest(TOP_N, "amount")
    # symbol: cache code 6位 → baostock sh./sz.
    syms = [f"sh.{c}" if str(c).startswith("6") else f"sz.{c}" for c in liq["code"].astype(str)]
    print(f"流动池 {len(syms)} 只 (cache {cache_date}, top{TOP_N} by amount)")

    end = pd.Timestamp(cache_date).strftime("%Y-%m-%d")
    start = (pd.Timestamp(cache_date) - pd.Timedelta(days=LOOKBACK_DAYS)).strftime("%Y-%m-%d")

    # 增量: 已有面板则续抓新日期
    existing_max = None
    if PANEL_FP.exists():
        old = pd.read_parquet(PANEL_FP, columns=["date"])
        existing_max = pd.to_datetime(old["date"]).max()
        if existing_max >= pd.Timestamp(cache_date):
            print(f"面板已最新 ({existing_max.date()}), 无需抓取")
            return 0
        if existing_max is not None:
            start = (existing_max + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        print(f"增量模式: {start} → {end}")

    lg = bs.login()
    if lg.error_code != "0":
        print(f"baostock login 失败: {lg.error_code} {lg.error_msg}")
        return 2

    parts, failed = [], []
    try:
        for i, sym in enumerate(syms):
            d = _query_one(bs, sym, start, end)
            if d is None:
                failed.append(sym)
                continue
            if not d.empty:
                d["symbol"] = sym
                parts.append(d)
            if (i + 1) % 100 == 0:
                print(f"  {i + 1}/{len(syms)} (fail {len(failed)})", flush=True)
    finally:
        try:
            bs.logout()
        except Exception:  # noqa: BLE001
            pass

    if failed:
        print(f"零模拟中止: {len(failed)} 只重试后仍失败 ({failed[:5]}...), 不落盘不兜底")
        return 2
    if not parts:
        print("无新增数据 (非交易日或已最新)")
        return 0

    new = pd.concat(parts, ignore_index=True)
    if PANEL_FP.exists() and existing_max is not None:
        old = pd.read_parquet(PANEL_FP)
        old = old[old["date"] < start]  # 防重
        new = pd.concat([old, new], ignore_index=True)
    new.to_parquet(PANEL_FP, index=False)
    print(f"面板已写: {PANEL_FP}  rows={len(new)}  "
          f"日期 {pd.to_datetime(new['date']).min().date()} → {pd.to_datetime(new['date']).max().date()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
