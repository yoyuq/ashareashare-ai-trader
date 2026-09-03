"""分钟级盘中卖出择时测试 — 一进二候选 (新浪分时, 单窗口近期样本, 换数据源).

背景: 打板「一进二」日线回测 (run_first_board_bt.py) 已证伪选股逻辑 (T收盘 −0.42%, 1/11 正窗口 FAIL),
但 T最高(完美卖点) +3.71%, 说明日内摆动大、日线抓不到。本脚本回答「分钟级」的核心问题:
**真实可实现的盘中卖出, 能否抓到比 T收盘(日内 hold) −0.42% 更好的那一截?**

数据源: 新浪 quotes.sina.cn 分时 K 线 (scale=5, datalen=1970 ≈ 最近 41 交易日), 与日线回测的
baostock replay 数据源不同 (换数据源)。东财 push2his 分钟源本轮被限流(RemoteDisconnected),
故用新浪。覆盖区间 = replay 窗口尾段 [2026-06-17, 2026-07-31]。

卖出规则 (真实可实现, 非完美上界):
  close    日内 hold 至收盘 (对齐日线 T收盘口径, 应为 ≈ −0.42%)
  10:00/10:30/11:30/14:30/14:50  固定时点卖出 (该时点 bar close)
  tp2      首个 bar 高点 ≥ 买价×1.02 → 卖 1.02 (止盈), 否则收盘卖
  sl2      首个 bar 低点 ≤ 买价×0.98 → 卖 0.98 (止损), 否则收盘卖
买价 = 首根 bar open (≈ 9:30 开盘); 成本 31bp 单笔扣。

判据 (预注册, 跑完不调): 某一卖出规则的**单笔均净收益** > 0 且 显著优于「close 日内 hold」基线。
若所有真实卖出规则均 ≤ close 基线 (≈ −0.42%), 则分钟级执行也不能挽救打板短线 → 证伪。

用法: python scripts/run_first_board_minute_bt.py
"""
from __future__ import annotations

import json
import os
import socket
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

# 清代理环境变量 (直连, 新浪无需代理)
for k in list(os.environ):
    if "proxy" in k.lower():
        os.environ.pop(k, None)

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

COST_BPS = 31.0
LU_PCT = 9.8
GAP_HI = 0.06
DATALEN = 1970
T0 = "2026-06-17"   # 新浪分钟覆盖起点
T1 = "2026-07-31"   # replay 窗口终点

CACHE = ROOT / "simulation_data" / "minute_cache_sina"
CACHE.mkdir(parents=True, exist_ok=True)
OUT = ROOT / "reports" / "first_board_minute_bt_result.json"

SINA_HDR = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0 Safari/537.36",
    "Referer": "https://finance.sina.com.cn/",
}


def _force_utf8():
    for name in ("stdout", "stderr"):
        s = getattr(sys, name, None)
        if hasattr(s, "reconfigure"):
            try:
                s.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
            except (ValueError, OSError):
                pass


def _sina_symbol(sym: str) -> str:
    return sym.replace(".", "")  # sh.600519 -> sh600519


def _fetch_sina(symbol: str, datalen: int = DATALEN):
    url = ("https://quotes.sina.cn/cn/api/json_v2.php/CN_MarketDataService.getKLineData"
           f"?symbol={_sina_symbol(symbol)}&scale=5&ma=no&datalen={datalen}")
    last = None
    for i in range(4):
        try:
            op = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            req = urllib.request.Request(url, headers=SINA_HDR)
            r = op.open(req, timeout=15)
            data = json.loads(r.read().decode("utf-8", "replace"))
            return data if isinstance(data, list) else None
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(1.0 * (i + 1))
    raise last


def _get_min(symbol: str, fetch: bool = True):
    fp = CACHE / f"{_sina_symbol(symbol)}.json"
    if fp.exists():
        try:
            return json.loads(fp.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            pass
    if not fetch:
        return None
    data = _fetch_sina(symbol)
    if data is not None:
        fp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return data


def _day_bars(rows: list, date: pd.Timestamp) -> pd.DataFrame | None:
    """取某交易日当天的 5 分钟 bars (datetime 索引, open/high/low/close)."""
    day = date.strftime("%Y-%m-%d")
    out = []
    for b in rows:
        t = b["day"]
        if t.startswith(day):
            out.append(b)
    if not out:
        return None
    df = pd.DataFrame(out)
    for c in ("open", "high", "low", "close"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["open", "high", "low", "close"])
    if df.empty:
        return None
    df["t"] = pd.to_datetime(df["day"])
    df = df.sort_values("t").set_index("t")
    return df[["open", "high", "low", "close"]]


FIXED_TIMES = {"10:00": pd.Timestamp("10:00").time(),
               "10:30": pd.Timestamp("10:30").time(),
               "11:30": pd.Timestamp("11:30").time(),
               "14:30": pd.Timestamp("14:30").time(),
               "14:50": pd.Timestamp("14:50").time()}


def _sell_at_time(df: pd.DataFrame, t) -> float | None:
    """在 <= t 的最后一个 bar 的 close 卖出; 若无则 None."""
    sub = df[df.index.time <= t]
    if sub.empty:
        return None
    return float(sub["close"].iloc[-1])


def _sell_tp_sl(df: pd.DataFrame, buy: float, tp: float | None, sl: float | None):
    """逐 bar 遍历, 保守同 bar 内止损优先。返回成交价."""
    for _, bar in df.iterrows():
        if sl is not None and float(bar["low"]) <= sl:
            return sl
        if tp is not None and float(bar["high"]) >= tp:
            return tp
    return float(df["close"].iloc[-1])


def _simulate(df: pd.DataFrame, buy: float) -> dict:
    cost = COST_BPS / 10000.0
    res = {}
    for name, t in FIXED_TIMES.items():
        px = _sell_at_time(df, t)
        res[name] = (px / buy - 1.0 - cost) if px is not None else None
    close_px = float(df["close"].iloc[-1])
    res["close"] = close_px / buy - 1.0 - cost
    res["tp2"] = _sell_tp_sl(df, buy, buy * 1.02, None) / buy - 1.0 - cost
    res["sl2"] = _sell_tp_sl(df, buy, None, buy * 0.98) / buy - 1.0 - cost
    # 日内高点诊断: 最高价出现在第几根 bar (相对开盘), 以及开盘→最高幅度
    high_idx = df["high"].idxmax().strftime("%H:%M") if len(df) else None
    res["_high_time"] = high_idx
    res["_open_to_high"] = float(df["high"].max()) / buy - 1.0
    return res


def _prep_daily(fp: Path) -> pd.DataFrame:
    df = pd.read_parquet(fp)
    df["date"] = pd.to_datetime(df["date"])
    df["isST"] = df["isST"].astype(str)
    df = df[df["is_trade"] == 1]
    df = df[df["isST"] == "0"]
    df = df[df["symbol"].str.startswith("sh.60") | df["symbol"].str.startswith("sz.00")]
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
    g = df.groupby("symbol", sort=False)
    df["prev_pct"] = g["pctChg"].shift(1)
    df["prev2_pct"] = g["pctChg"].shift(2)
    df["prev_close"] = g["close"].shift(1)
    df["next_open"] = g["open"].shift(-1)
    df["first_board"] = (df["prev_pct"] >= LU_PCT) & (df["prev2_pct"] < LU_PCT)
    df["gap"] = df["open"] / df["prev_close"] - 1.0
    return df


def main() -> int:
    _force_utf8()
    df = _prep_daily(ROOT / "replay_data" / "daily_2025-10-08_2026-07-31.parquet")
    sigA = df["first_board"] & (df["gap"] > 0) & (df["gap"] <= GAP_HI)
    pick = df[sigA & df["next_open"].notna() & (df["next_open"] > 0)]
    pick = pick[(pick["date"] >= T0) & (pick["date"] <= T1)]
    pick = pick.sort_values("date").reset_index(drop=True)
    n_trades = len(pick)
    symbols = sorted(pick["symbol"].unique())
    print(f"近期一进二候选: {n_trades} 笔 / 唯一股票 {len(symbols)} 只")
    print(f"抓取新浪分时 (datalen={DATALEN}), 缓存 {CACHE} ...")

    # 抓取 (线程池并行 + 缓存; 每个 symbol 只写自己的文件, 线程安全)
    import threading
    from concurrent.futures import ThreadPoolExecutor

    fetched = {}
    lock = threading.Lock()
    n_ok = n_fail = n_empty = 0

    def _one(sym):
        nonlocal n_ok, n_fail, n_empty
        try:
            data = _get_min(sym, fetch=True)
            with lock:
                if data is None:
                    n_empty += 1
                else:
                    fetched[sym] = data
                    n_ok += 1
        except Exception as e:  # noqa: BLE001
            with lock:
                n_fail += 1
                if n_fail <= 5:
                    print(f"  [warn] {sym} 抓取失败: {type(e).__name__}: {str(e)[:80]}")
        with lock:
            if (n_ok + n_fail + n_empty) % 100 == 0:
                print(f"  进度 {n_ok + n_fail + n_empty}/{len(symbols)} "
                      f"(ok={n_ok} fail={n_fail} empty={n_empty})")

    with ThreadPoolExecutor(max_workers=8) as ex:
        list(ex.map(_one, symbols))
    print(f"抓取完成: ok={n_ok} fail={n_fail} empty={n_empty}")

    # 逐笔模拟
    rules = list(FIXED_TIMES.keys()) + ["close", "tp2", "sl2"]
    rows = []
    n_matched = 0
    open_mismatch = []
    for _, r in pick.iterrows():
        sym, d = r["symbol"], r["date"]
        data = fetched.get(sym)
        if data is None:
            continue
        bars = _day_bars(data, d)
        if bars is None:
            continue
        buy = float(bars["open"].iloc[0])
        if buy <= 0:
            continue
        n_matched += 1
        # 交叉校验: 分钟首 bar open vs 日线 open
        if abs(buy / r["open"] - 1.0) > 0.005:
            open_mismatch.append((sym, str(d.date()), buy, r["open"]))
        sim = _simulate(bars, buy)
        rec = {"symbol": sym, "date": str(d.date()), "buy": buy}
        rec.update(sim)
        rows.append(rec)

    if not rows:
        print("无匹配分钟数据, 终止")
        return 1

    res = pd.DataFrame(rows)
    cost = COST_BPS / 10000.0
    print("\n" + "=" * 100)
    print(f"分钟级卖出择时 — 一进二候选 (新浪分时, {T0}~{T1}, {n_matched}/{n_trades} 笔匹配)")
    print("=" * 100)
    summary = {"n_trades": n_trades, "n_matched": n_matched,
               "n_symbols": len(symbols), "n_ok": n_ok, "n_fail": n_fail, "n_empty": n_empty,
               "cost_bps": COST_BPS, "data_source": "sina_min5"}
    print(f"\n{'规则':<8} {'单笔均净':>9} {'中位':>8} {'胜率':>7} {'正笔数':>7}   vs close")
    base_close = float(res["close"].mean())
    for rule in rules:
        v = pd.to_numeric(res[rule], errors="coerce").dropna()
        if v.empty:
            continue
        m = float(v.mean())
        med = float(v.median())
        wr = float((v > 0).mean())
        pos = int((v > 0).sum())
        delta = m - base_close
        summary[rule] = {"mean": round(m, 5), "median": round(med, 5),
                         "win_rate": round(wr, 5), "n_positive": pos, "n": int(len(v)),
                         "vs_close": round(delta, 5)}
        print(f"{rule:<8} {m*100:+8.3f}% {med*100:+7.3f}% {wr*100:6.1f}% {pos:>6}/{len(v)} "
              f"{delta*100:+7.3f}pp")

    # 日内高点时间分布诊断
    print("\n日内高点出现时点分布 (相对买入首 bar):")
    if res["_high_time"].notna().any():
        hh = res["_high_time"].astype(str).str[:2].astype(int)
        for h in [9, 10, 11, 13, 14]:
            n = int((hh == h).sum())
            print(f"  {h}点: {n:>5} 笔 ({n/len(res)*100:4.1f}%)")
    oh = pd.to_numeric(res["_open_to_high"], errors="coerce").dropna()
    print(f"  开盘→日内最高 (完美上界) 均值: {oh.mean()*100:+.3f}% (日线回测 T最高 +3.71%)")
    print(f"  开盘→收盘 (日内 hold) 均值:   {base_close*100:+.3f}% (日线回测 T收盘 −0.42%)")
    print(f"  分钟 open 与日线 open 偏差>0.5% 的笔数: {len(open_mismatch)}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果已写 {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
