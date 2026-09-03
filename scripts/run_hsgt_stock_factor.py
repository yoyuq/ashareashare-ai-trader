"""北向持股变动个股截面因子 — 预注册 reports/agent_loop/prereg_hsgt_stock_factor.md.

因子: hold% 的 20 日变化, 月末双向 top5, vs 全主板等权 universe, 65bp。
输出: reports/agent_loop/hsgt_stock_factor_result.json
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
from run_dca_vs_lumpsum import WINDOWS  # noqa: E402

COST = 65.0 / 1e4
TOP_K = 5
DIFF_N = 20
SHARD_DIR = ROOT / "replay_data" / "hsgt_stock"
COV_MIN = 0.60  # 窗口内因子覆盖交易日比例下限


def build_factor() -> pd.DataFrame:
    """hold% 面板 → 20日变化因子 (date × code)。"""
    frames = []
    for fp in sorted(SHARD_DIR.glob("*.parquet")):
        if fp.stat().st_size == 0:
            continue
        d = pd.read_parquet(fp, columns=["持股日期", "持股数量占A股百分比"])
        d = d.rename(columns={"持股日期": "date", "持股数量占A股百分比": "hold"})
        d["code"] = fp.stem
        frames.append(d)
    if not frames:
        raise RuntimeError("hsgt_stock 分片库为空")
    panel = pd.concat(frames, ignore_index=True)
    panel["date"] = pd.to_datetime(panel["date"])
    panel = panel.drop_duplicates(subset=["date", "code"])
    wide = panel.pivot_table(index="date", columns="code", values="hold").sort_index()
    # 缺失日 forward-fill (披露口径, 月频调仓不敏感), 之后仍缺保持 NaN
    wide = wide.ffill()
    return wide.diff(DIFF_N)


def main() -> int:
    fac_all = build_factor()
    print(f"factor panel: {fac_all.shape[0]} days × {fac_all.shape[1]} codes", flush=True)
    out = {}
    for wname, fp in WINDOWS.items():
        df = H.load_base_window(fp)
        if df.empty:
            raise RuntimeError(f"窗口数据为空: {fp}")
        close = df.pivot_table(index="date", columns="symbol", values="close").sort_index()
        ret = close.pct_change(fill_method=None)
        codes = {s[3:]: s for s in close.columns}
        fac = fac_all.rename(columns=codes).reindex(columns=close.columns)
        # as-of 对齐到交易日历: 取月末前最近披露值 (≤10天容忍; 停更后超容忍自动 NaN, 防陈旧假覆盖)
        fac = fac.reindex(close.index.union(fac.index)).sort_index().ffill()
        fac = fac.reindex(close.index)

        s = pd.Series(close.index)
        me = s.groupby([s.dt.year, s.dt.month]).apply(lambda x: x.iloc[-1]).tolist()
        covered = fac.reindex(me).notna().any(axis=1)
        cov_ratio = float(covered.mean()) if len(me) else 0.0
        if cov_ratio < COV_MIN:
            out[wname] = {"factor_coverage": round(cov_ratio, 3), "n_valid_months": 0}
            print(f"{wname}: 因子覆盖 {cov_ratio:.0%} < {COV_MIN:.0%}, 剔除", flush=True)
            continue

        arms = {"add": {}, "cut": {}}
        for arm in arms:
            sel = {}
            for T in me:
                row = fac.loc[T].dropna()
                if len(row) < TOP_K * 5:
                    continue
                codes_k = (row.nlargest(TOP_K) if arm == "add" else row.nsmallest(TOP_K)).index.tolist()
                sel[T] = codes_k
            valid = [T for T in me if T in sel]
            excess_days = []
            for i, T in enumerate(valid):
                T_next = valid[i + 1] if i + 1 < len(valid) else close.index[-1]
                idx = ret.index[(ret.index > T) & (ret.index <= T_next)]
                if idx.empty:
                    continue
                port = ret.loc[idx, sel[T]].mean(axis=1)
                counts = ret.notna().sum(axis=1)
                uni = ret.mean(axis=1).where(counts > 50).reindex(idx)
                prev = sel[valid[i - 1]] if i > 0 else []
                replaced = len(set(sel[T]) - set(prev)) / TOP_K
                port = port - (replaced * COST)
                excess_days.append(port - uni)
            if not excess_days:
                arms[arm] = {"n_valid_months": 0}
                continue
            ex = pd.concat(excess_days).dropna()
            arms[arm] = {"n_valid_months": len(valid), "total_excess": round(float(ex.sum()), 4)}

        spread = (arms["add"].get("total_excess", float("nan"))
                  - arms["cut"].get("total_excess", float("nan")))
        out[wname] = {"factor_coverage": round(cov_ratio, 3),
                      "add": arms["add"], "cut": arms["cut"],
                      "spread_add_minus_cut": round(spread, 4) if np.isfinite(spread) else None}
        print(f"{wname}: 加仓 {arms['add'].get('total_excess', 'NA')} "
              f"减仓 {arms['cut'].get('total_excess', 'NA')} "
              f"(有效月 {arms['add'].get('n_valid_months', 0)})", flush=True)

    def gate(arm):
        act = {k: v[arm] for k, v in out.items()
               if isinstance(v.get(arm), dict) and v[arm].get("n_valid_months", 0) >= 3}
        wins = sum(1 for v in act.values() if v.get("total_excess", 0) >= 0)
        avg = float(np.mean([v["total_excess"] for v in act.values()])) if act else float("nan")
        return {"valid_windows": len(act), "excess_wins": f"{wins}/{len(act)}",
                "avg_excess": round(avg, 4), "PASS": bool(len(act) >= 5 and wins >= 5 and avg > 0)}

    summary = {"add_arm": gate("add"), "cut_arm": gate("cut")}
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    fp_out = ROOT / "reports" / "agent_loop" / "hsgt_stock_factor_result.json"
    fp_out.write_text(json.dumps({"windows": out, "summary": summary},
                                 ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved -> {fp_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
