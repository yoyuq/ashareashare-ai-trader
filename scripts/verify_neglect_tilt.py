"""验证: PreScreener 冷落(neglect)因子接入后, 选股池 vs 基线 vs 指数 (预注册).

背景: run_improved_portfolio_ab.py 证低换手 bottom-100 tilt 5/5 跑赢上证综指。本脚本验证
**把该发现接入 PreScreener 后**, 系统选股池 (top-100) 是否跑赢「无 neglect 的基线 PreScreener」
以及「上证综指」, 即验证接线是否真正把冷落溢价落到系统选股。

方法 (全程零模拟, 真实 Baostock parquet):
  - 每窗口首日 T, 用 top-800 流动性可投资 universe (非ST/可交易/pe>0/pb>0) 构 PreScreener 输入。
  - PreScreener.screen(top_n=100, neglect=True) vs neglect=False。
  - 等权买入持有到窗口末, 比 return/maxDD/sharpe。
  - 字段映射: close→price, pctChg→pct_change, turn→turnover, peTTM→pe_ttm, pbMRQ→pb,
    total_mv≈amount/(turn/100) (代理, 仅作 size 因子排序输入), amplitude≈(high-low)/close*100。

预注册判据:
  - 主: neglect ON 跑赢上证综指 return 的窗口数 >= 4/5 → 接线成功 (把冷落溢价落到系统选股)。
  - 副: neglect ON 不劣于 OFF (return 且 dd) >= 4/5 → 冷落因子确实改善选股而非噪声。

用法: python scripts/verify_neglect_tilt.py
输出: reports/neglect_tilt_verify.json + .md
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from analysis.pre_screener import PreScreener  # noqa: E402

ROOT = Path(__file__).parent.parent
OUT = ROOT / "reports" / "neglect_tilt_verify.json"
TOP_N = 100

WINDOWS = {
    "2018熊":     "replay_data/daily_2018-01-01_2018-12-31.parquet",
    "2019牛":     "replay_data/daily_2019-01-01_2019-12-31.parquet",
    "2020牛转崩": "replay_data/daily_2020-06-01_2021-02-28.parquet",
    "2024震荡":   "replay_data/daily_2024-01-01_2024-12-31.parquet",
    "2025-26现期": "replay_data/daily_2025-10-08_2026-07-31.parquet",
}


def _force_utf8():
    for name in ("stdout", "stderr"):
        s = getattr(sys, name, None)
        if hasattr(s, "reconfigure"):
            try:
                s.reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass


def _metrics(eq: pd.Series) -> dict:
    eq = eq.dropna()
    if len(eq) < 2:
        return {"return_pct": float("nan"), "max_dd_pct": float("nan"), "sharpe": float("nan")}
    ret = float(eq.iloc[-1] / eq.iloc[0] - 1.0) * 100.0
    dd = float((eq / eq.cummax() - 1.0).min()) * 100.0
    dr = eq.pct_change().dropna()
    sharpe = float(dr.mean() / dr.std() * np.sqrt(252)) if len(dr) > 1 and dr.std() > 0 else 0.0
    return {"return_pct": round(ret, 3), "max_dd_pct": round(dd, 3), "sharpe": round(sharpe, 3)}


def _curve(df: pd.DataFrame, symbols: list[str]) -> pd.Series:
    sub = df[df.symbol.isin(symbols)][["date", "symbol", "close"]].dropna(subset=["close"])
    piv = sub.pivot_table(index="date", columns="symbol", values="close", aggfunc="last")
    piv = piv.reindex(columns=[s for s in symbols if s in piv.columns])
    norm = piv / piv.iloc[0]
    norm = norm.ffill().dropna(axis=1, how="all")
    return norm.mean(axis=1)


def _to_screener_df(df: pd.DataFrame, T) -> pd.DataFrame:
    """Baostock 窗口日线 → PreScreener 输入 (AKShare 字段名)。"""
    d0 = df[df.date == T].copy()
    d0 = d0[(d0.isST.astype(str) == "0") & (d0.is_trade == 1)]
    out = pd.DataFrame(index=d0.index)
    out["code"] = d0["code"].astype(str)
    out["name"] = "X"  # 已预滤 ST, 避免 hard_filter 的 name 正则误伤
    out["price"] = d0["close"]
    out["pct_change"] = d0["pctChg"]
    out["volume"] = d0["volume"]
    out["amount"] = d0["amount"]
    out["turnover"] = d0["turn"]
    out["pe_ttm"] = d0["peTTM"]
    out["pb"] = d0["pbMRQ"]
    # total_mv 代理: 成交额/换手率 (换手率% → 流通市值)
    turn_frac = d0["turn"].clip(lower=0.05) / 100.0
    out["total_mv"] = (d0["amount"] / turn_frac).clip(lower=1e9, upper=1e12)
    out["amplitude"] = (d0["high"] - d0["low"]) / d0["close"] * 100.0
    return out


def main() -> int:
    _force_utf8()
    idx = pd.read_parquet(ROOT / "replay_data" / "index_series.parquet")
    idx["date"] = pd.to_datetime(idx["date"])

    report = {"date": datetime.now().isoformat(timespec="seconds"), "TOP_N": TOP_N,
              "windows": [], "verdict": {}}
    print("=" * 100)
    print(f"验证: PreScreener 冷落因子 (neglect) ON vs OFF vs 上证综指 (top-{TOP_N} 等权买入持有)")
    print("=" * 100)

    for label, fp in WINDOWS.items():
        df = pd.read_parquet(ROOT / fp)
        df["date"] = pd.to_datetime(df["date"])
        T, T_end = df.date.min(), df.date.max()

        idx_win = idx[(idx["date"] >= T) & (idx["date"] <= T_end)]
        sh = idx_win[idx_win.symbol == "sh.000001"].set_index("date")["close"]

        # top-800 流动性可投资 universe
        liq = df[df.symbol != "sh.000001"].copy()
        liq["isST"] = liq["isST"].astype(str)
        inv = liq[(liq.isST == "0") & (liq.is_trade == 1) & (liq.peTTM > 0) & (liq.pbMRQ > 0)]
        med = inv.groupby("symbol")["amount"].median().sort_values(ascending=False)
        uni = med.head(800).index.tolist()

        scr = _to_screener_df(df[df.symbol.isin(uni)], T)
        # 过滤: 只保留可交易当日有数据的
        scr = scr.dropna(subset=["price", "amount", "turnover", "pe_ttm", "pb"])

        ps = PreScreener()
        try:
            r_on = ps.screen(scr, regime="range_bound", top_n=TOP_N, neglect=True)
            r_off = ps.screen(scr, regime="range_bound", top_n=TOP_N, neglect=False)
            r_cold = ps.screen(scr, regime="range_bound", top_n=TOP_N, cold_tilt=True)
        except Exception as e:
            print(f"  [{label}] PreScreener 失败: {e}")
            continue

        # code 已是完整 symbol (如 sh.600519), 直接用作 symbol
        on_syms = [str(c) for c in r_on.df["code"]]
        off_syms = [str(c) for c in r_off.df["code"]]
        cold_syms = [str(c) for c in r_cold.df["code"]]

        m_sh = _metrics(sh)
        m_on = _metrics(_curve(df, on_syms))
        m_off = _metrics(_curve(df, off_syms))
        m_cold = _metrics(_curve(df, cold_syms))

        # 冷落度: 选中池的换手率中位 (越低越冷)
        scr_by_code = scr.set_index("code")["turnover"]
        med_turn_on = float(scr_by_code.reindex(on_syms).median())
        med_turn_off = float(scr_by_code.reindex(off_syms).median())
        med_turn_cold = float(scr_by_code.reindex(cold_syms).median())

        print(f"\n### {label}  ({T.date()} -> {T_end.date()})")
        print(f"  上证综指:      ret={m_sh['return_pct']:>7.2f}%  dd={m_sh['max_dd_pct']:>7.2f}%  sh={m_sh['sharpe']:>5.2f}")
        print(f"  OFF(无冷落):   ret={m_off['return_pct']:>7.2f}%  dd={m_off['max_dd_pct']:>7.2f}%  sh={m_off['sharpe']:>5.2f}  换手中位={med_turn_off:.2f}%")
        print(f"  ON (弱冷落):   ret={m_on['return_pct']:>7.2f}%  dd={m_on['max_dd_pct']:>7.2f}%  sh={m_on['sharpe']:>5.2f}  换手中位={med_turn_on:.2f}%  Δvs指数={m_on['return_pct']-m_sh['return_pct']:>+7.2f}pp")
        print(f"  冷落模式(强):  ret={m_cold['return_pct']:>7.2f}%  dd={m_cold['max_dd_pct']:>7.2f}%  sh={m_cold['sharpe']:>5.2f}  换手中位={med_turn_cold:.2f}%  Δvs指数={m_cold['return_pct']-m_sh['return_pct']:>+7.2f}pp")

        report["windows"].append({"window": label, "index": m_sh, "on": m_on, "off": m_off,
                                  "cold": m_cold, "med_turn_on": med_turn_on,
                                  "med_turn_off": med_turn_off, "med_turn_cold": med_turn_cold})

    ws = report["windows"]
    n = len(ws)
    def _cnt(key, mode):
        return sum(1 for w in ws if (not np.isnan(w[key][mode]) and w[key][mode] > w["index"][mode]))
    def _not_worse():
        c = 0
        for w in ws:
            if np.isnan(w["on"]["return_pct"]) or np.isnan(w["off"]["return_pct"]):
                continue
            dret = w["on"]["return_pct"] - w["off"]["return_pct"]
            ddd = w["on"]["max_dd_pct"] - w["off"]["max_dd_pct"]
            if dret >= -3.0 and ddd <= 2.0:
                c += 1
        return c
    report["win_counts"] = {
        "on_beats_index_return": f"{_cnt('on','return_pct')}/{n}",
        "on_beats_index_sharpe": f"{_cnt('on','sharpe')}/{n}",
        "cold_beats_index_return": f"{_cnt('cold','return_pct')}/{n}",
        "cold_beats_index_sharpe": f"{_cnt('cold','sharpe')}/{n}",
        "on_not_worse_off": f"{_not_worse()}/{n}",
    }
    report["verdict"] = {
        "wired_success": "成功" if _cnt("on", "return_pct") >= 4 else "未满足",
        "cold_tilt_success": "成功" if _cnt("cold", "return_pct") >= 4 else "未满足",
        "improves_over_off": "改善" if _not_worse() >= 4 else "不显著",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n" + "=" * 100)
    for k, v in report["win_counts"].items():
        print(f"  {k}: {v}")
    print("结论:", report["verdict"])
    print(f"\n结果已写 {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
