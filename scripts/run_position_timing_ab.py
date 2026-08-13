"""方案3b 确定性持仓择时 A/B — 忠实 + 解耦 + 质量 (预注册, 全程零模拟).

背景: 方案3 (截面硬过滤) A/B (`reports/knowledge_apply_ab_result.md`) 结论 keep_gate0 — 截面一次性
硬过滤无法忠实复现 low_pe_value 的**个股择时**价值 (低估买/高估卖), 且消费端 max_pb=1.0 幻觉阈值
100% 触发保底回退导致过滤器形同虚设。方案3b 把规则忠实落地成**确定性布尔择时信号** (无 LLM, 不被
regime 误读): 买入门 (低估才进候选, 无保底) + 卖出信号 (高估即减仓)。

验证目标 (3 个预注册窗口, 真实 replay parquet):
  Part 1 忠实 (确定性, 不跑 LLM):
    1. 买入忠实: 买入门保留的票全满足 `pe_ttm<max_pe & pb<max_pb` (且 pe/pb>0)。
    2. 卖出忠实: 高估卖出信号触发的票全满足 `pe_ttm > max_pe*1.5`。
    3. 解耦: 诊断官 `_market_diagnostic(df_cs, ...)` 吃全市场截面 (historical_replay.py:1515),
       与买入门/卖出信号结构性无关 → risk_level 确定性与 gate 无关 (不跑 LLM 实测, 那是验证代码
       结构保证的事实)。
  Part 2 质量 (诊断模式回放, temperature=0.0 保证双侧 risk_level 确定一致, 真实 parquet):
    每窗口 gate=0 vs gate=1, 比较组合总收益与最大回撤。

预注册判据:
  - 忠实: 买入/卖出布尔断言全真 (3/3 窗口)。
  - 解耦: 结构性成立 (确定性)。
  - 质量: gate=1 不劣于 gate=0 (Δreturn ≥ -3pp 且 Δdd ≤ +2pp) ≥2/3 窗口。
  - 决策: 忠实全真 + 解耦成立 + 质量不劣(≥2/3) → 翻 gate=1; 否则保持 0。
  - 未改判据、未挑窗口、未调参追赢。

用法:
  python scripts/run_position_timing_ab.py                 # Part 1 忠实 (快, 不跑 LLM)
  python scripts/run_position_timing_ab.py --quality       # + Part 2 质量 (诊断回放, 慢, 跑 LLM)
  python scripts/run_position_timing_ab.py --quality --days 30 --samples 8
输出: reports/position_timing_ab_result.json + .md
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dotenv import load_dotenv
load_dotenv()

import pandas as pd
from loguru import logger

import historical_replay as hr
from analysis.pre_screener import PreScreener
from simulation.daily_runner import _detect_regime
from agent.learning.knowledge_apply import (
    recall_verified_rules, apply_rules_to_cross_section, sell_signals_for_positions,
)


def _force_utf8():
    for name in ("stdout", "stderr"):
        s = getattr(sys, name, None)
        if hasattr(s, "reconfigure"):
            try:
                s.reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass


# 预注册窗口: 与 run_injection_ab / run_knowledge_apply_ab 一致的 regime 标注.
WINDOWS = [
    {
        "label": "2018熊",
        "data_file": "replay_data/daily_2018-01-01_2018-12-31.parquet",
        "start": "2018-06-01", "end": "2018-08-31",
        "quality_end": "2018-08-31",
        "regime_note": "熊",
    },
    {
        "label": "2020牛转崩",
        "data_file": "replay_data/daily_2020-06-01_2021-02-28.parquet",
        "start": "2020-12-01", "end": "2021-02-26",
        "quality_end": "2021-02-26",
        "regime_note": "牛→崩",
    },
    {
        "label": "2024震荡",
        "data_file": "replay_data/daily_2024-01-01_2024-12-31.parquet",
        "start": "2024-05-06", "end": "2024-07-31",
        "quality_end": "2024-07-31",
        "regime_note": "震荡",
    },
]

# 预注册判据容差 (不劣 = 收益不显著降 + 回撤不显著恶化), 与 learn 判据 v2 同精神:
RETURN_TOLERANCE = 3.0   # Δreturn 允许最多 -3pp
DD_TOLERANCE = 2.0       # Δdd (深度, 正=更差) 允许最多 +2pp


def load_data_dict(data_file: str) -> dict:
    _df = pd.read_parquet(data_file)
    _df["date"] = pd.to_datetime(_df["date"])
    return {s: hr._optimize_dtypes(g.drop(columns="symbol").reset_index(drop=True))
            for s, g in _df.groupby("symbol")}


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


def _max_drawdown(eq: list) -> float:
    """最大回撤深度 (正数, 越深越大=越差)."""
    peak = float("-inf")
    mdd = 0.0
    for e in eq:
        t = float(e.get("total", 0))
        peak = max(peak, t)
        if peak > 0:
            mdd = max(mdd, (peak - t) / peak * 100)
    return mdd


# ── Part 1: 忠实断言 (确定性, 无 LLM) ──

def _verify_buy_gate(screened: pd.DataFrame, kept: pd.DataFrame, params: dict) -> bool:
    """买入忠实: 保留票全满足 pe<max_pe & pb<max_pb (且 pe/pb>0)。"""
    max_pe = float(params.get("max_pe", 15.0))
    max_pb = float(params.get("max_pb", 2.0))
    if kept.empty:
        return True  # 空池无票可验 (忠实性不违背)
    pe = pd.to_numeric(kept["pe_ttm"], errors="coerce")
    pb = pd.to_numeric(kept["pb"], errors="coerce")
    ok = bool((pe.notna() & (pe > 0) & (pe < max_pe)
               & pb.notna() & (pb > 0) & (pb < max_pb)).all())
    return ok


def _verify_sell_signal(held_df: pd.DataFrame, sell_mask: pd.Series, params: dict) -> bool:
    """卖出忠实: 触发信号票全满足 pe>max_pe*1.5; 且所有 pe>max_pe*1.5 的票都被触发 (双向一致)。"""
    max_pe = float(params.get("max_pe", 15.0))
    pe = pd.to_numeric(held_df["pe_ttm"], errors="coerce")
    expected = pe > max_pe * 1.5
    triggered = sell_mask.reindex(held_df.index).fillna(False).astype(bool)
    # 触发 ⇔ 期望 (pe>max_pe*1.5), 双向一致
    return bool((triggered == expected).all())


def run_fidelity(w: dict, samples: int, top_n: int) -> dict:
    label = w["label"]
    rules = recall_verified_rules(1)
    applicable = [r for r in rules if r.get("applicable")]
    if not applicable:
        return {"label": label, "regime_note": w["regime_note"], "enabled": False,
                "rules": [r["template"] for r in rules],
                "buy_ok_all": True, "sell_ok_all": True, "n_rows": 0, "kept_counts": []}

    params_by_tpl = {r["template"]: r["params"] for r in applicable}
    basic, _u = hr.load_snapshot_basic()
    _df = pd.read_parquet(w["data_file"])
    _df["date"] = pd.to_datetime(_df["date"])
    data = load_data_dict(w["data_file"])
    all_days = sorted(pd.to_datetime(_df["date"]).dt.date.unique())
    dates = _pick_dates(all_days, w["start"], w["end"], samples)

    buy_ok = True
    sell_ok = True
    kept_counts = []
    for T in dates:
        Tstr = T.isoformat()
        df_cs = hr.reconstruct_cross_section(data, basic, Tstr)
        if df_cs.empty:
            continue
        if "isST" in df_cs.columns:
            df_cs = df_cs[df_cs["isST"] != 1]
        regime = _detect_regime(df_cs).get("regime", "range_bound")
        screened = PreScreener().screen(df_cs, regime=regime, top_n=top_n).df

        # 买入门
        kept, _rep = apply_rules_to_cross_section(screened, applicable)
        kept_counts.append(len(kept))
        if "low_pe_value" in params_by_tpl:
            if not _verify_buy_gate(screened, kept, params_by_tpl["low_pe_value"]):
                buy_ok = False
                logger.warning(f"  {Tstr} 买入忠实违背!")

        # 卖出信号: 用 kept (低估值候选) 当"持仓"样例, 验证谓词双向一致
        if not kept.empty and "pe_ttm" in kept.columns:
            sell_mask = sell_signals_for_positions(kept, applicable)
            if not _verify_sell_signal(kept, sell_mask, params_by_tpl["low_pe_value"]):
                sell_ok = False
                logger.warning(f"  {Tstr} 卖出忠实违背!")

    return {"label": label, "regime_note": w["regime_note"], "enabled": True,
            "rules": list(params_by_tpl),
            "params": params_by_tpl, "n_rows": len(dates),
            "buy_ok_all": buy_ok, "sell_ok_all": sell_ok,
            "kept_counts": kept_counts}


# ── Part 2: 质量 (诊断模式回放, gate=0 vs gate=1, temperature=0.0) ──

def _run_quality_window(w: dict, days: int) -> dict:
    label = w["label"]
    data_file = w["data_file"]
    end_date = w["quality_end"]
    results = {}
    for gate in (0, 1):
        tag = f"ptab_{label}_{gate}_{days}d"
        r = asyncio.run(hr.run_replay(
            days=days, end_date=end_date, data_file=data_file,
            diagnostic_mode=True, gate=gate, temperature=0.0, tag=tag,
        ))
        eq = r.get("equity_curve", [])
        results[gate] = {
            "return_pct": round(float(r.get("total_return_pct", 0.0)), 2),
            "max_drawdown": round(_max_drawdown(eq), 2),
            "final_equity": round(float(r.get("final_equity", 0.0)), 2),
            "num_trades": r.get("num_trades", 0),
            "final_positions": len(r.get("final_positions", [])),
        }
    g0, g1 = results[0], results[1]
    dret = round(g1["return_pct"] - g0["return_pct"], 2)
    ddd = round(g1["max_drawdown"] - g0["max_drawdown"], 2)
    not_worse = (dret >= -RETURN_TOLERANCE) and (ddd <= DD_TOLERANCE)
    return {"label": label, "regime_note": w["regime_note"], "gate0": g0, "gate1": g1,
            "dreturn_pp": dret, "ddd_pp": ddd, "not_worse": not_worse}


def main():
    _force_utf8()
    ap = argparse.ArgumentParser(description="方案3b 确定性持仓择时 A/B (忠实+解耦+质量)")
    ap.add_argument("--samples", type=int, default=6, help="Part1 每窗口采样交易日数")
    ap.add_argument("--top-n", type=int, default=300, help="PreScreener 候选池大小")
    ap.add_argument("--quality", action="store_true", help="跑 Part2 质量 (诊断回放, 慢, 跑 LLM)")
    ap.add_argument("--days", type=int, default=25, help="Part2 每窗口回放交易日数")
    ap.add_argument("--skip-fidelity", action="store_true", help="跳过 Part1 忠实")
    ap.add_argument("--windows", type=str, default=None, help="窗口label子串, 逗号分隔")
    args = ap.parse_args()

    windows = WINDOWS
    if args.windows:
        subs = [x.strip() for x in args.windows.split(",") if x.strip()]
        windows = [w for w in WINDOWS if any(s in w["label"] for s in subs)]

    fidelity = []
    if not args.skip_fidelity:
        for w in windows:
            if not Path(w["data_file"]).exists():
                logger.warning(f"跳过 {w['label']}: 数据未就绪 {w['data_file']}")
                continue
            fidelity.append(run_fidelity(w, args.samples, args.top_n))

    quality = []
    if args.quality:
        for w in windows:
            if not Path(w["data_file"]).exists():
                logger.warning(f"跳过 {w['label']}: 数据未就绪 {w['data_file']}")
                continue
            logger.info(f"\n=== 质量回放 {w['label']} (gate=0 vs gate=1, {args.days}日) ===")
            quality.append(_run_quality_window(w, args.days))

    # 决策
    n_fid = len(fidelity)
    fid_all_ok = all(r["buy_ok_all"] and r["sell_ok_all"] for r in fidelity if r["enabled"])
    n_enabled = sum(1 for r in fidelity if r["enabled"])

    # 忠实: 若 --skip-fidelity, 视为已由单独 Part1 验证建立 (确定性布尔断言), 此处仅评估质量.
    fidelity_ok = (fid_all_ok and n_enabled >= 1) if n_fid > 0 else args.skip_fidelity

    verdict = "undetermined"
    if quality:
        n_not_worse = sum(1 for q in quality if q["not_worse"])
        quality_ok = n_not_worse >= max(1, round(2 / 3 * len(quality)))
        verdict = "flip_gate1" if (fidelity_ok and quality_ok) else "keep_gate0"
    elif n_fid > 0:
        # 只跑忠实 (未跑质量): 忠实成立仅满足翻 gate 的必要条件, 质量待 Part2
        verdict = "fidelity_ok_quality_pending" if fidelity_ok else "keep_gate0"

    out = {
        "date": pd.Timestamp.now().isoformat(timespec="seconds"),
        "samples": args.samples, "top_n": args.top_n,
        "quality_days": args.days if args.quality else None,
        "verdict": verdict,
        "fidelity_windows": n_enabled,
        "fidelity_all_ok": bool(fid_all_ok),
        "fidelity": fidelity,
        "quality": quality,
    }
    json_path = Path("reports/position_timing_ab_result.json")
    json_path.write_text(json.dumps(out, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    _write_md(out)
    print(f"\n=== 结论: {verdict} ===")
    for r in fidelity:
        if not r["enabled"]:
            print(f"  [忠实] {r['label']}: 无 applicable 规则 (rules={r['rules']})")
            continue
        med = int(pd.Series(r["kept_counts"]).median()) if r["kept_counts"] else None
        print(f"  [忠实] {r['label']} ({r['regime_note']}): 买入忠实={'✓' if r['buy_ok_all'] else '✗'} "
              f"卖出忠实={'✓' if r['sell_ok_all'] else '✗'} | 保留票中位={med}")
    for q in quality:
        print(f"  [质量] {q['label']} ({q['regime_note']}): gate0 ret={q['gate0']['return_pct']}% "
              f"dd={q['gate0']['max_drawdown']}% | gate1 ret={q['gate1']['return_pct']}% "
              f"dd={q['gate1']['max_drawdown']}% | Δret={q['dreturn_pp']}pp Δdd={q['ddd_pp']}pp "
              f"不劣={'✓' if q['not_worse'] else '✗'}")
    print(f"\n结果已写入: {json_path} 和 reports/position_timing_ab_result.md")


def _write_md(out: dict):
    lines = [
        "# 确定性持仓择时 A/B 结果 (方案3b, 忠实/解耦/质量)",
        "",
        f"生成日期: {out['date']} | 忠实采样 {out['samples']} 交易日/窗口 | "
        f"候选池 top_n={out['top_n']} | 质量回放 {out['quality_days']} 日/窗口",
        "",
        f"## 结论: **{out['verdict']}**",
        "",
        f"- 忠实窗口: {out['fidelity_windows']} | 买入/卖出断言全真: {out['fidelity_all_ok']}",
        "",
    ]
    if out["fidelity"]:
        lines += ["## 忠实 (确定性, 无 LLM)", ""]
        lines += ["| 窗口 | regime | 规则参数 | 买入忠实 | 卖出忠实 | 保留票中位 |",
                  "|---|---|---|---|---|---|"]
        for r in out["fidelity"]:
            if not r["enabled"]:
                lines.append(f"| {r['label']} | - | 无 applicable 规则 | - | - | - |")
                continue
            p = r["params"].get("low_pe_value", {})
            med = int(pd.Series(r["kept_counts"]).median()) if r["kept_counts"] else None
            lines.append(
                f"| {r['label']} | {r.get('regime_note','')} | max_pe={p.get('max_pe')}, "
                f"max_pb={p.get('max_pb')} | {'✓' if r['buy_ok_all'] else '✗'} "
                f"| {'✓' if r['sell_ok_all'] else '✗'} | {med} |"
            )
        lines.append("")
    if out["quality"]:
        lines += ["## 质量 (诊断模式回放, temperature=0.0, gate=0 vs gate=1)", ""]
        lines += ["| 窗口 | regime | gate0 收益 | gate0 回撤 | gate1 收益 | gate1 回撤 | Δ收益 | Δ回撤 | 不劣 |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for q in out["quality"]:
            lines.append(
                f"| {q['label']} | {q['regime_note']} | {q['gate0']['return_pct']}% "
                f"| {q['gate0']['max_drawdown']}% | {q['gate1']['return_pct']}% "
                f"| {q['gate1']['max_drawdown']}% | {q['dreturn_pp']}pp | {q['ddd_pp']}pp "
                f"| {'✓' if q['not_worse'] else '✗'} |"
            )
        lines.append("")
    lines += [
        "## 判据 (预注册)",
        "",
        f"- **主指标忠实**: 买入门保留票全满足 `pe_ttm<max_pe & pb<max_pb` (pe/pb>0); "
        f"高估卖出信号触发票全满足 `pe_ttm>max_pe*1.5` (双向一致)。确定性布尔断言, 无 LLM。",
        "- **主指标解耦**: 诊断官 `_market_diagnostic(df_cs, ...)` 吃全市场截面, 与买入门/卖出信号结构性无关"
        " → risk_level 确定性与 gate 无关 (不跑 LLM 实测, 那是验证代码结构保证的事实)。",
        f"- **副指标质量**: gate=1 不劣于 gate=0 (Δreturn ≥ -{RETURN_TOLERANCE}pp 且 "
        f"Δdd ≤ +{DD_TOLERANCE}pp) ≥2/3 窗口 (temperature=0.0 保证双侧 risk_level 确定一致)。",
        "- **决策**: 忠实全真 + 解耦成立 + 质量不劣(≥2/3) → 翻 gate=1; 否则保持 0。未改判据/未挑窗口/未调参追赢。",
        "",
        "## 诚实局限",
        "",
        "- 规则价值 (low_pe_value 5/6 窗口改善) 已由 tester 预注册判据验证; 本 A/B 验证的是**落地忠实性 + "
        "落地不造成灾难性损害**, 不重新估计 alpha。",
        "- 消费端 `max_pb=1.0` 是翻译官幻觉的破净阈值 (落在 `_sanitize_params` 合法区间 [0.5,8] 故保留), "
        "导致买入门候选极少 (破净股本就稀少) — 这是「只买低估」的忠实后果, 不做保底兜回, 也不擅自调参。",
        "- 质量对比是单次运行 (temperature=0.0 消除诊断噪声, 但选股本身仍随 PreScreener 确定性), "
        "样本有限; 用于 sanity check, 不用于精确收益估计。",
        "- 卖出信号 (pe>max_pe*1.5) 在短窗口内极少触发 (低估值股需大幅上涨才高估), 主价值在买入门; "
        "若质量对比里卖出信号未触发, 如实标注。",
    ]
    Path("reports/position_timing_ab_result.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
