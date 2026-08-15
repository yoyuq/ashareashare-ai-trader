"""构建 point-in-time 基本面面板 (Baostock, 免代理) — 供"成长质量选股/精细排雷"使用。

背景: 上一版只抓 6 个季度快照, 2024/2025-26 窗口只能用 2023Q3 的 2-3 年前旧数据 (问题1)。
本版扩展为 2017Q3~2026Q2 全季度 (36 期), 每条带 pubDate(公告日), 下游 A/B 按 pubDate<=信号日 取
最新报告期, 杜绝未来函数; 供「精细排雷」(连续亏损/利润暴跌/营收收缩) 检验 (问题3)。

字段 (每 (symbol, 季度) 2 次调用):
  - query_profit_data : roeAvg(ROE) / totalShare(总股本) / MBRevenue(营收) / netProfit(净利润)
  - query_growth_data : YOYNI(净利润同比, 直接给)

point-in-time: 每条带 pubDate(公告日), 下游按 pubDate <= 信号日 取最新报告期, 杜绝未来函数。

universe (预注册): 5 核心窗口各自 top-800 流动性可投资票 (isST=0 & is_trade=1 & pe>0 & pb>0,
按窗口内 median amount 排序) 的并集。

节流/重连/续跑: 复用 build_long_replay 的已知安全模式 (0.08s/次 + 每 150 次调用重连), 避免 Baostock 黑名单。
续跑按 (symbol, statDate) 粒度跳过已抓取, 增量补齐新季度。

用法:
  python scripts/build_fundamental_panel.py --limit 20      # 小批量验证 (20 个 (sym,季度) 对)
  python scripts/build_fundamental_panel.py                 # 全量增量 (后台跑, 约 2-2.5h)
输出: replay_data/fundamentals.parquet
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

import baostock as bs  # noqa: E402

ROOT = Path(__file__).parent.parent
WINDOW_FILES = {
    "2018熊": "replay_data/daily_2018-01-01_2018-12-31.parquet",
    "2019牛": "replay_data/daily_2019-01-01_2019-12-31.parquet",
    "2020牛转崩": "replay_data/daily_2020-06-01_2021-02-28.parquet",
    "2024震荡": "replay_data/daily_2024-01-01_2024-12-31.parquet",
    "2025-26现期": "replay_data/daily_2025-10-08_2026-07-31.parquet",
}
OUT = ROOT / "replay_data" / "fundamentals.parquet"

# 预注册: 全 point-in-time 季度 (2017Q3 ~ 2026Q2), 覆盖 5 核心窗口 + 连续亏损/营收YoY 所需历史。
Q_START = (2017, 3)
Q_END = (2026, 2)
TOP_N_LIQUID = 800
THROTTLE = 0.08
RECONNECT_EVERY_CALLS = 150  # 每 150 次 API 调用重连 (2 调用/(sym,季度))

_STAT_MONTH = {"1": "03-31", "2": "06-30", "3": "09-30", "4": "12-31"}


def _quarters(start: tuple[int, int], end: tuple[int, int]) -> list[tuple[str, str]]:
    qs: list[tuple[str, str]] = []
    y, q = start
    while (y, q) <= end:
        qs.append((str(y), str(q)))
        q += 1
        if q > 4:
            q = 1
            y += 1
    return qs


def _stat_date(year: str, q: str) -> str:
    return f"{year}-{_STAT_MONTH[q]}"


def liquid_universe(top_n: int = TOP_N_LIQUID) -> list[str]:
    """5 核心窗口 top-N 流动性可投资票的并集。"""
    syms: set[str] = set()
    for name, fp in WINDOW_FILES.items():
        df = pd.read_parquet(ROOT / fp)
        df = df[df.symbol != "sh.000001"]
        df["isST"] = df["isST"].astype(str)
        inv = df[(df.isST == "0") & (df.is_trade == 1) & (df.peTTM > 0) & (df.pbMRQ > 0)]
        med = inv.groupby("symbol")["amount"].median().sort_values(ascending=False)
        top = med.head(top_n).index.tolist()
        syms.update(top)
        print(f"  {name}: investable={inv.symbol.nunique()} top{top_n} 已并入")
    return sorted(syms)


def _q(fn, sym, year, quarter):
    rs = fn(code=sym, year=year, quarter=quarter)
    rows = []
    while rs.error_code == "0" and rs.next():
        rows.append(rs.get_row_data())
    if not rows:
        return None
    return dict(zip(rs.fields, rows[0]))


def _f(d: dict, k: str):
    try:
        v = d.get(k)
        return float(v) if v not in (None, "", "None") else None
    except (TypeError, ValueError):
        return None


def fetch_one(sym: str, year: str, q: str) -> dict | None:
    """抓单 (sym, 季度) 的 profit + growth。返回一条记录 dict 或 None。"""
    try:
        p = _q(bs.query_profit_data, sym, year, q)
    except Exception:
        p = None
    try:
        g = _q(bs.query_growth_data, sym, year, q)
    except Exception:
        g = None
    if p is None and g is None:
        return None
    return {
        "symbol": sym,
        "statDate": (p or g).get("statDate"),
        "pubDate": (p or g).get("pubDate"),
        "roeAvg": _f(p, "roeAvg") if p else None,
        "totalShare": _f(p, "totalShare") if p else None,
        "MBRevenue": _f(p, "MBRevenue") if p else None,
        "netProfit": _f(p, "netProfit") if p else None,
        "YOYNI": _f(g, "YOYNI") if g else None,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=0, help="只抓前 N 个 (sym,季度) 对 (0=全量)")
    ap.add_argument("--throttle", type=float, default=THROTTLE)
    args = ap.parse_args()

    universe = liquid_universe()
    quarters = _quarters(Q_START, Q_END)
    print(f"universe 并集: {len(universe)} 只 | 季度 {len(quarters)} 期 | 每(sym,季度) 2 次调用")

    # 续跑 (按 (symbol, statDate) 粒度)
    done_pairs: set[tuple[str, str]] = set()
    rows: list[dict] = []
    if OUT.exists():
        try:
            prev = pd.read_parquet(OUT)
            prev = prev.dropna(subset=["statDate"])
            done_pairs = set(zip(prev["symbol"].astype(str), prev["statDate"].astype(str)))
            rows = prev.to_dict("records")
            print(f"续跑: 已有 {len(done_pairs)} 个 (sym,季度) 记录")
        except Exception as e:
            print(f"续跑加载失败, 从零开始: {e}")

    pending: list[tuple[str, str, str]] = []
    for sym in universe:
        for (y, q) in quarters:
            sd = _stat_date(y, q)
            if (sym, sd) not in done_pairs:
                pending.append((sym, y, q))
    if args.limit:
        pending = pending[:args.limit]
    print(f"待抓取: {len(pending)} 个 (sym,季度) 对 | 节流 {args.throttle}s/次")

    def _save():
        if rows:
            pd.DataFrame(rows).to_parquet(OUT, index=False)

    bs.login()
    t0 = time.time()
    ok = 0
    calls = 0
    for i, (sym, y, q) in enumerate(pending):
        try:
            rec = fetch_one(sym, y, q)
            if rec:
                rows.append(rec)
                ok += 1
        except Exception:
            pass
        calls += 2
        if calls >= RECONNECT_EVERY_CALLS:
            calls = 0
            try:
                bs.logout()
            except Exception:
                pass
            bs.login()
        time.sleep(args.throttle)
        if (i + 1) % 500 == 0:
            el = time.time() - t0
            print(f"  {i+1}/{len(pending)} | 成功 {ok} | 已耗时 {el/60:.1f}min | "
                  f"预估剩余 {el/max(i+1,1)*(len(pending)-i-1)/60:.1f}min")
            _save()
    try:
        bs.logout()
    except Exception:
        pass

    if not rows:
        print("未抓到任何数据!")
        return 1
    df = pd.DataFrame(rows)
    df.to_parquet(OUT, index=False)
    print(f"\n完成: 共 {len(df)} 行 | {df.symbol.nunique()} 只 | 新抓 {ok} 条 | 耗时 {time.time()-t0:.0f}s → {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
