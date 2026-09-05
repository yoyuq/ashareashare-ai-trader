"""iter34 市赚率 PR<1 月度整持 — 预注册 reports/agent_loop/prereg_pr_ratio.md (用户指定策略).

PR = peTTM / (roeAvg×100) < 1; 剔 ST/停牌/上市不满1年; 每月首个交易日收盘买入持整月,
月末收盘卖出, 等权全部合格票, 65bp 单边。vs 匹配 universe (同过滤无 PR 筛选)。

覆盖偏差披露: fundamentals 1919 只 (退市缺ROE→幸存者方向高估); FAIL 直接成立,
PASS 须全市场 ROE 重建复测才可入前瞻。输出: reports/agent_loop/pr_ratio_result.json
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import strategy_research_harness as H  # noqa: E402

COST = 0.0065  # 65bp 单边 (预注册冻结)
IPO_MIN_DAYS = 365

WINDOWS = {
    "2018熊": "replay_data/daily_2018-01-01_2018-12-31.parquet",
    "2019牛": "replay_data/daily_2019-01-01_2019-12-31.parquet",
    "2020牛转崩": "replay_data/daily_2020-06-01_2021-02-28.parquet",
    "2021白马转小盘": "replay_data/daily_2021-01-01_2021-12-31.parquet",
    "2022熊": "replay_data/daily_2022-01-01_2022-12-31.parquet",
    "2023震荡": "replay_data/daily_2023-01-01_2023-12-31.parquet",
    "2024震荡": "replay_data/daily_2024-01-01_2024-12-31.parquet",
    "2025-26现期": "replay_data/daily_2025-10-08_2026-07-31.parquet",
}


def _force_utf8() -> None:
    for name in ("stdout", "stderr"):
        s = getattr(sys, name, None)
        if hasattr(s, "reconfigure"):
            try:
                s.reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass


def load_ipo() -> pd.DataFrame:
    """Baostock query_stock_basic → (symbol, ipoDate)。type=1 股票, 全市场一次调用。"""
    cache = ROOT / "replay_data" / "stock_basic_ipo.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    import baostock as bs
    lg = bs.login()
    if lg.error_code != "0":
        raise RuntimeError(f"baostock login 失败: {lg.error_msg}")
    rs = bs.query_stock_basic()
    rows = []
    while rs.error_code == "0" and rs.next():
        rows.append(rs.get_row_data())
    bs.logout()
    df = pd.DataFrame(rows, columns=rs.fields)
    df = df[df["type"] == "1"][["code", "code_name", "ipoDate", "outDate"]].copy()
    df = df[df["ipoDate"] != ""]
    df.to_parquet(cache)
    return df


def month_bounds(dates: list) -> list:
    """返回 (月首日, 月末日) 列表, 按月分组。"""
    s = pd.Series(pd.DatetimeIndex(dates))
    out = []
    for _, g in s.groupby([s.dt.year, s.dt.month]):
        out.append((g.iloc[0], g.iloc[-1]))
    return out


def run_window(df: pd.DataFrame, fund: pd.DataFrame, ipo: pd.DataFrame) -> dict:
    df = df.copy()
    df["symbol"] = df["symbol"].astype(str)
    df["peTTM"] = pd.to_numeric(df["peTTM"], errors="coerce")
    # PIT ROE (pubDate<=T 最新一期)
    df["roe_pit"] = H.pt_fundamental_col(df, fund, "roeAvg")
    # ipo 满一年
    ipo_map = ipo.set_index("code")["ipoDate"]
    df["ipoDate"] = df["symbol"].map(ipo_map)
    df["ipo_ok"] = (pd.to_datetime(df["ipoDate"], errors="coerce")
                    <= df["date"] - pd.Timedelta(days=IPO_MIN_DAYS))
    # 基础过滤 + PR (tradestatus/isST 在面板里是字符串 dtype, 必须 astype(str) 比较)
    base = df[(df["is_trade"] == 1) & (df["tradestatus"].astype(str) == "1")
              & (df["isST"].astype(str) == "0")
              & (df["ipo_ok"])].copy()
    base["pr"] = base["peTTM"] / (base["roe_pit"] * 100.0)
    basket = base[(base["peTTM"] > 0) & (base["roe_pit"] > 0) & (base["pr"] < 1)]

    ret_cc = df.pivot_table(index="date", columns="symbol", values="close").sort_index().pct_change(fill_method=None)
    dates = list(ret_cc.index)
    months = month_bounds(dates)

    def month_port(months_sel_mask) -> pd.Series:
        """逐月等权收益: 首日收盘→末日收盘, 空月=0 (现金), 65bp×2。"""
        segs = []
        for F, L in months:
            elig = base[base["date"] == F]
            elig = elig[months_sel_mask(elig)]
            if len(elig) == 0:
                segs.append(pd.Series([0.0], index=[L]))
                continue
            syms = elig["symbol"].tolist()
            seg_dates = [d for d in dates if F < d <= L]
            if not seg_dates:
                segs.append(pd.Series([0.0], index=[L]))
                continue
            seg = ret_cc.loc[seg_dates, syms].mean(axis=1)
            gross = float((1 + seg).prod())
            segs.append(pd.Series([gross * (1 - COST) ** 2 - 1.0], index=[L]))
        return pd.concat(segs)

    r_strat = month_port(lambda e: (e["peTTM"] > 0) & (e["roe_pit"] > 0) & (e["pr"] < 1))
    r_univ = month_port(lambda e: pd.Series(True, index=e.index))  # 匹配 universe: 同过滤无 PR
    if (r_strat != 0).sum() == 0:
        raise RuntimeError("整窗无合格票 (零模拟, 报错不兜底)")
    return {"strat": r_strat, "univ": r_univ,
            "n_per_month": [int(len(base[(base["date"] == F) & (base["peTTM"] > 0)
                                         & (base["roe_pit"] > 0) & (base["pr"] < 1)]))
                            for F, _ in months]}


def equity(monthly: pd.Series) -> tuple[float, float]:
    """月度收益序列 → (总收益, 最大回撤)。"""
    nav = (1 + monthly.sort_index()).cumprod()
    dd = float((nav / nav.cummax() - 1).min())
    return float(nav.iloc[-1] - 1), dd


def main() -> int:
    _force_utf8()
    fund = H.load_fundamentals()
    if fund.empty:
        raise RuntimeError("fundamentals.parquet 缺失")
    ipo = load_ipo()
    out: dict = {}
    for window, fp in WINDOWS.items():
        df = H.load_base_window(fp, boards=("sh.", "sz.", "bj."))
        if df.empty:
            raise RuntimeError(f"窗口数据为空: {fp}")
        r = run_window(df, fund, ipo)
        ts, ds = equity(r["strat"])
        tu, du = equity(r["univ"])
        n = r["n_per_month"]
        out[window] = {"strat_total": round(ts, 4), "univ_total": round(tu, 4),
                       "diff_pp": round((ts - tu) * 100, 2),
                       "strat_dd": round(ds, 4), "univ_dd": round(du, 4),
                       "n_min": min(n), "n_med": int(np.median(n)), "n_max": max(n)}
        print(f"{window:<12} PR<1={ts*100:>+8.1f}% univ={tu*100:>+8.1f}% "
              f"({out[window]['diff_pp']:>+7.2f}pp) dd={ds*100:>6.1f}%/{du*100:>6.1f}% "
              f"n={out[window]['n_min']}~{out[window]['n_max']}(med {out[window]['n_med']})")

    diffs = [out[w]["diff_pp"] for w in WINDOWS]
    wins = sum(1 for x in diffs if x >= 0)
    dd_s = np.mean([out[w]["strat_dd"] for w in WINDOWS])
    dd_u = np.mean([out[w]["univ_dd"] for w in WINDOWS])
    dd_ok = bool((dd_s - dd_u) >= -0.01)
    gate = {"avg_diff_pp": round(float(np.mean(diffs)), 2), "wins": f"{wins}/8",
            "dd_strat_avg": round(float(dd_s), 4), "dd_univ_avg": round(float(dd_u), 4),
            "dd_ok": dd_ok,
            "required": "avg>=+0.2pp & wins>=5/8 & dd diff>=-1pp",
            "coverage_bias_note": "PASS 不构成实操依据 (fundamentals 覆盖偏差, 须全市场ROE重建复测)",
            "PASS": bool(np.mean(diffs) >= 0.2 and wins >= 5 and dd_ok)}
    print(json.dumps(gate, ensure_ascii=False, indent=2))
    fp = ROOT / "reports" / "agent_loop" / "pr_ratio_result.json"
    fp.write_text(json.dumps({"prereg": "prereg_pr_ratio.md", "windows": out, "summary": gate},
                             ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved -> {fp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
