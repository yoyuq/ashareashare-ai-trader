"""两融个股杠杆变动截面因子 — 预注册 reports/agent_loop/prereg_margin_stock_factor.md.

因子: 融资余额20日变化率, 月末双向 top5, vs 全主板等权, 65bp。
输出: reports/agent_loop/margin_stock_factor_result.json
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
SHARD_DIR = ROOT / "replay_data" / "margin_detail"


def build_factor() -> pd.DataFrame:
    """date × code 的融资余额面板 (仅数据自有日期, 不填充)。"""
    frames = []
    for fp in sorted(SHARD_DIR.glob("*.parquet")):
        if fp.stat().st_size == 0:
            continue
        d = pd.read_parquet(fp)
        if d.empty:
            continue
        d["date"] = pd.to_datetime(fp.stem)
        frames.append(d)
    if not frames:
        raise RuntimeError("margin_detail 分片库为空")
    panel = pd.concat(frames, ignore_index=True)
    wide = panel.pivot_table(index="date", columns="code", values="rz_balance").sort_index()
    return wide


def main() -> int:
    bal = build_factor()
    print(f"balance panel: {bal.shape[0]} days × {bal.shape[1]} codes", flush=True)
    fac_all = bal / bal.shift(DIFF_N) - 1.0  # 20日变化率 (数据缺失自然 NaN)
    fac_all = fac_all.where(bal > 0)

    out = {}
    for wname, fp in WINDOWS.items():
        df = H.load_base_window(fp)
        if df.empty:
            raise RuntimeError(f"窗口数据为空: {fp}")
        close = df.pivot_table(index="date", columns="symbol", values="close").sort_index()
        ret = close.pct_change(fill_method=None)
        codes = {s[3:]: s for s in close.columns}
        fac = fac_all.rename(columns=codes).reindex(columns=close.columns)

        s = pd.Series(close.index)
        me = s.groupby([s.dt.year, s.dt.month]).apply(lambda x: x.iloc[-1]).tolist()
        arms = {"lever": {}, "delever": {}}
        for arm in arms:
            sel = {}
            for T in me:
                if T not in fac.index:
                    continue
                row = fac.loc[T].dropna()
                if len(row) < TOP_K * 10:
                    continue
                codes_k = (row.nlargest(TOP_K) if arm == "lever" else row.nsmallest(TOP_K)).index.tolist()
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
            arms[arm] = {"n_valid_months": len(valid), "total_excess": round(float((1 + ex).prod() - 1), 4)}

        spread = (arms["lever"].get("total_excess", float("nan"))
                  - arms["delever"].get("total_excess", float("nan")))
        out[wname] = {"lever": arms["lever"], "delever": arms["delever"],
                      "spread": round(spread, 4) if np.isfinite(spread) else None}
        print(f"{wname}: 加杠杆 {arms['lever'].get('total_excess', 'NA')} "
              f"去杠杆 {arms['delever'].get('total_excess', 'NA')} "
              f"(有效月 {arms['lever'].get('n_valid_months', 0)})", flush=True)

    def gate(arm):
        act = {k: v[arm] for k, v in out.items()
               if isinstance(v.get(arm), dict) and v[arm].get("n_valid_months", 0) >= 3}
        wins = sum(1 for v in act.values() if v.get("total_excess", 0) >= 0)
        avg = float(np.mean([v["total_excess"] for v in act.values()])) if act else float("nan")
        return {"valid_windows": len(act), "excess_wins": f"{wins}/{len(act)}",
                "avg_excess": round(avg, 4), "PASS": bool(len(act) >= 5 and wins >= 5 and avg > 0)}

    summary = {"lever_arm": gate("lever"), "delever_arm": gate("delever")}
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    fp_out = ROOT / "reports" / "agent_loop" / "margin_stock_factor_result.json"
    fp_out.write_text(json.dumps({"windows": out, "summary": summary},
                                 ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved -> {fp_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
