"""F 注入门槛预注册 A/B — gate=0 vs gate=1 真实市场诊断对比 (LEARNED_KNOWLEDGE_GATE)。

方法 (隔离单变量·全程零模拟):
  - 3 个预注册独立历史窗口 (2018熊 / 2020-06→2021-02牛 / 2024震荡), 各取 N 个真实交易日。
  - 每个交易日, 用同一份真实截面 (replay parquet 重建) + 同一 regime/拥挤度/历史序列,
    只把 LEARNED_KNOWLEDGE_GATE 在 0/1 之间切换, 各跑一次真实市场诊断 (deepseek-v4-flash)。
  - 唯一差异 = 已学 verified 规则注入段 (format_learned_for_prompt)。

判据 (预注册, 与 reports/injection_gate_ab_protocol.md §2.2 一致):
  主指标(安全): 诊断 risk_level → 注入组不得比基线更激进 (risk_level 不降低 / position_multiplier 不抬高)。
                次日指数涨跌对两侧相同, 唯差异=暴露, 故"回撤不深于基线" ⇔ "不更激进"。
  副指标(方向): market_phase → 次日全市场中位涨跌方向命中率, 注入组不劣于基线。
  决策规则: 注入组在 ≥2/3 窗口满足"主指标不劣 + 副指标不劣"才翻 gate=1; 否则 gate 保持 0。

用法:
  python scripts/run_injection_ab.py                 # 3 窗口, 每窗口 5 交易日, 每侧 1 次
  python scripts/run_injection_ab.py --samples 8 --runs 2   # 扩大样本抗噪
输出: reports/injection_gate_ab_result.json + reports/injection_gate_ab_result.md
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dotenv import load_dotenv
load_dotenv()

import numpy as np
import pandas as pd
from loguru import logger

from simulation.daily_runner import get_market_diagnostic, _detect_regime
import historical_replay as hr


def _force_utf8():
    for name in ("stdout", "stderr"):
        s = getattr(sys, name, None)
        if hasattr(s, "reconfigure"):
            try:
                s.reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass


# 预注册窗口: 与 run_multiwindow_ab.py 一致的 regime 标注, 数据据 sh.000001 实际走势标定.
WINDOWS = [
    {
        "label": "2018熊",
        "data_file": "replay_data/daily_2018-01-01_2018-12-31.parquet",
        "start": "2018-06-01", "end": "2018-08-31",
        "regime_note": "熊",
    },
    {
        "label": "2020牛转崩",
        "data_file": "replay_data/daily_2020-06-01_2021-02-28.parquet",
        "start": "2020-12-01", "end": "2021-02-26",
        "regime_note": "牛→崩",
    },
    {
        "label": "2024震荡",
        "data_file": "replay_data/daily_2024-01-01_2024-12-31.parquet",
        "start": "2024-05-06", "end": "2024-07-31",
        "regime_note": "震荡",
    },
]


def load_data_dict(data_file: str) -> dict:
    """从 replay parquet 加载 {symbol: df}, 与 historical_replay.run_replay 同一逻辑。"""
    _df = pd.read_parquet(data_file)
    _df["date"] = pd.to_datetime(_df["date"])
    return {s: hr._optimize_dtypes(g.drop(columns="symbol").reset_index(drop=True))
            for s, g in _df.groupby("symbol")}


def build_day_agg(_df: pd.DataFrame) -> dict:
    """一次性聚合每个交易日的全市场统计 (供 history_5d / 次日方向, 免逐日重建截面)。"""
    _df = _df.copy()
    _df["day"] = _df["date"].dt.date
    agg = {}
    for day, grp in _df.groupby("day"):
        pct = pd.to_numeric(grp["pctChg"], errors="coerce")
        pe = pd.to_numeric(grp["peTTM"], errors="coerce")
        amt = pd.to_numeric(grp["amount"], errors="coerce")
        agg[day] = {
            "up_ratio": float((pct > 0).mean()) if len(pct) else 0.5,
            "limit_up": int((pct >= 9.5).sum()),
            "limit_down": int((pct <= -9.5).sum()),
            "med_pe": float(pe.median()) if len(pe.dropna()) else 0.0,
            "total_amt": float(amt.sum()) / 1e8 if len(amt.dropna()) else 0.0,
            "median_pct": float(pct.median()) if len(pct.dropna()) else 0.0,
        }
    return agg


_PHASE_DIR = {
    "trend_up": "bull", "bubble_late": "bear", "turning": "neutral",
    "trend_down": "bear", "range": "neutral", "panic_bottom": "bear",
}


def _next_direction(day_agg: dict, all_days: list, T: str) -> str:
    """次日全市场中位涨跌方向 (bull/bear/flat)。"""
    d = date.fromisoformat(T)
    nxt = None
    for x in all_days:
        if x > d:
            nxt = x
            break
    if nxt is None or nxt not in day_agg:
        return "flat"
    m = day_agg[nxt]["median_pct"]
    return "bull" if m > 0.05 else ("bear" if m < -0.05 else "flat")


async def _run_one(df_cs, regime, crowd, history_5d, T, gate: int) -> dict:
    os.environ["LEARNED_KNOWLEDGE_GATE"] = str(gate)
    diag = await get_market_diagnostic(
        df_cs, regime, crowding=crowd, history_5d=history_5d,
        current_date=T, adv_mode="same", temperature=0.0,
    )
    return {
        "risk_level": int(diag.get("risk_level", 3)),
        "position_multiplier": float(diag.get("position_multiplier", 0.9)),
        "market_phase": diag.get("market_phase", "unknown"),
        "dominant_master": diag.get("dominant_master", "unknown"),
    }


def _pick_dates(all_days: list, start: str, end: str, n: int) -> list:
    d0, d1 = date.fromisoformat(start), date.fromisoformat(end)
    days = [d for d in all_days if d0 <= d <= d1]
    if not days:
        return []
    if len(days) <= n:
        return days
    if n == 1:
        return [days[len(days) // 2]]
    idx = [int(round(i * (len(days) - 1) / (n - 1))) for i in range(n)]
    return [days[i] for i in idx]


async def run_window(w: dict, samples: int, runs: int) -> dict:
    from analysis.crowding import market_crowding

    label = w["label"]
    logger.info(f"\n=== 窗口 {label} ({w['regime_note']}) 数据 {w['data_file']} ===")
    _df = pd.read_parquet(w["data_file"])
    _df["date"] = pd.to_datetime(_df["date"])
    data = load_data_dict(w["data_file"])
    day_agg = build_day_agg(_df)
    all_days = sorted(day_agg.keys())
    dates = _pick_dates(all_days, w["start"], w["end"], samples)
    logger.info(f"采样 {len(dates)} 个交易日: {[str(d) for d in dates]}")

    rows = []
    for T in dates:
        Tstr = T.isoformat()
        df_cs = hr.reconstruct_cross_section(data, {}, Tstr)
        if df_cs.empty:
            logger.warning(f"  {Tstr} 截面为空, 跳过")
            continue
        df_cs = df_cs[df_cs["isST"] != 1]
        regime = _detect_regime(df_cs).get("regime", "range_bound")
        try:
            crowd = market_crowding(df_cs)
        except Exception:
            crowd = {"signal": "unknown", "score": 50.0, "hot_ratio": 0.0}
        # history_5d: 前 5 个交易日的轻量聚合 (同一窗口内, 逐日聚合表已有)
        prior = [x for x in all_days if x < T][-5:]
        history_5d = [day_agg[x] for x in prior if x in day_agg]
        nxt_dir = _next_direction(day_agg, all_days, Tstr)

        for _ in range(runs):
            g0 = await _run_one(df_cs, regime, crowd, history_5d, Tstr, 0)
            g1 = await _run_one(df_cs, regime, crowd, history_5d, Tstr, 1)
            rows.append({
                "date": Tstr, "regime": regime, "next_dir": nxt_dir,
                "gate0": g0, "gate1": g1,
                "d_risk": g1["risk_level"] - g0["risk_level"],
                "d_pos": round(g1["position_multiplier"] - g0["position_multiplier"], 4),
            })
        logger.info(f"  {Tstr} [{regime}] g0={g0['risk_level']}/x{g0['position_multiplier']:.2f} "
                    f"g1={g1['risk_level']}/x{g1['position_multiplier']:.2f} "
                    f"Δrisk={g1['risk_level']-g0['risk_level']:+d} Δpos={g1['position_multiplier']-g0['position_multiplier']:+.2f}")

    # 汇总
    d_risk = [r["d_risk"] for r in rows]
    d_pos = [r["d_pos"] for r in rows]
    med_risk = statistics.median(d_risk) if d_risk else 0
    med_pos = statistics.median(d_pos) if d_pos else 0.0

    def _hit_rate(gate):
        hits = tot = 0
        for r in rows:
            ph_dir = _PHASE_DIR.get(r[gate]["market_phase"], "neutral")
            if ph_dir == "neutral":
                continue
            tot += 1
            if ph_dir == r["next_dir"]:
                hits += 1
        return hits, tot

    h0, t0 = _hit_rate("gate0")
    h1, t1 = _hit_rate("gate1")
    r0 = (h0 / t0) if t0 else 0.0
    r1 = (h1 / t1) if t1 else 0.0

    primary_ok = med_risk >= 0 and med_pos <= 0.02   # 不更激进
    secondary_ok = r1 >= r0 - 1e-9                    # 命中率不劣
    return {
        "label": label, "regime_note": w["regime_note"],
        "n_dates": len(dates), "n_rows": len(rows),
        "median_d_risk": med_risk, "median_d_pos": med_pos,
        "hit_rate_gate0": f"{h0}/{t0}={r0:.2f}", "hit_rate_gate1": f"{h1}/{t1}={r1:.2f}",
        "primary_ok": bool(primary_ok), "secondary_ok": bool(secondary_ok),
        "rows": rows,
    }


def main():
    _force_utf8()
    ap = argparse.ArgumentParser(description="注入门槛 A/B (gate=0 vs gate=1)")
    ap.add_argument("--samples", type=int, default=5, help="每窗口采样交易日数")
    ap.add_argument("--runs", type=int, default=1, help="每侧重复次数(抗噪)")
    ap.add_argument("--windows", type=str, default=None, help="窗口label子串, 逗号分隔")
    args = ap.parse_args()

    windows = WINDOWS
    if args.windows:
        subs = [x.strip() for x in args.windows.split(",") if x.strip()]
        windows = [w for w in WINDOWS if any(s in w["label"] for s in subs)]

    results = []
    for w in windows:
        if not Path(w["data_file"]).exists():
            logger.warning(f"跳过 {w['label']}: 数据未就绪 {w['data_file']}")
            continue
        results.append(asyncio.run(run_window(w, args.samples, args.runs)))

    n_ok = sum(1 for r in results if r["primary_ok"] and r["secondary_ok"])
    verdict = "flip_gate1" if n_ok >= 2 and len(results) >= 2 else "keep_gate0"
    out = {
        "date": pd.Timestamp.now().isoformat(timespec="seconds"),
        "samples": args.samples, "runs": args.runs,
        "verdict": verdict, "windows_ok": n_ok, "windows_total": len(results),
        "results": results,
    }
    json_path = Path("reports/injection_gate_ab_result.json")
    json_path.write_text(json.dumps(out, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    _write_md(out)
    print(f"\n=== A/B 结论: {verdict} (满足 {n_ok}/{len(results)} 窗口) ===")
    for r in results:
        print(f"  {r['label']} ({r['regime_note']}): Δrisk中位={r['median_d_risk']:+.0f} "
              f"Δpos中位={r['median_d_pos']:+.3f} | 命中 g0={r['hit_rate_gate0']} g1={r['hit_rate_gate1']} "
              f"| 主{'✓' if r['primary_ok'] else '✗'} 副{'✓' if r['secondary_ok'] else '✗'}")
    print(f"\n结果已写入: {json_path} 和 reports/injection_gate_ab_result.md")


def _write_md(out: dict):
    lines = [
        "# 注入门槛 A/B 结果 (F, gate=0 vs gate=1)",
        "",
        f"生成日期: {out['date']} | 采样 {out['samples']} 交易日/窗口 × {out['runs']} runs | 隔离单变量 (唯差异=verified 规则注入)",
        "",
        f"## 结论: **{out['verdict']}** ({out['windows_ok']}/{out['windows_total']} 窗口满足主+副不劣)",
        "",
        "| 窗口 | regime | Δrisk中位 | Δpos中位 | 命中 g0 | 命中 g1 | 主不劣 | 副不劣 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in out["results"]:
        lines.append(
            f"| {r['label']} | {r['regime_note']} | {r['median_d_risk']:+.0f} | {r['median_d_pos']:+.3f} | "
            f"{r['hit_rate_gate0']} | {r['hit_rate_gate1']} | "
            f"{'✓' if r['primary_ok'] else '✗'} | {'✓' if r['secondary_ok'] else '✗'} |"
        )
    lines += [
        "",
        "## 判据 (预注册)",
        "",
        "- 主指标(安全): Δrisk = risk_level(gate1)-risk_level(gate0); 中位 ≥0 (不降低风险级) 且 Δpos ≤ +0.02 → 不更激进。",
        "- 副指标(方向): market_phase → 次日全市场中位涨跌方向命中率, gate1 ≥ gate0。",
        "- 决策: ≥2/3 窗口主+副均不劣 → 翻 gate=1, 否则保持 0。",
        "",
        "## 诚实局限",
        "",
        "- A/B 用 temperature=0.0 隔离注入变量 (消除采样噪声); 生产温度 0.4 下诊断本身还有 ±1 级噪声, 不在本 A/B 测量范围。",
        "- 注入内容随 regime 变化 (牛只注入均线系统, 熊注入估值+均线), 非固定文本。",
        "- 次日方向用全市场中位 pctChg 代理 (非 sh.000001 指数), 与回放复盘口径一致。",
        "- history_5d 用逐日轻量聚合 (非逐日全截面重建), 对两侧一致, 不影响 A/B。",
    ]
    Path("reports/injection_gate_ab_result.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
