"""方案3 知识选股过滤 — 接线正确性 + 解耦验证 (预注册, 全程零模拟, 不跑 LLM)。

⚠️ 已废弃 (superseded by `scripts/run_position_timing_ab.py`): 方案3 (截面硬过滤) A/B 结论
keep_gate0 (过滤器 100% fallback 不生效 + 语义鸿沟), 已改为方案3b 确定性持仓择时。本脚本保留
作审计历史, 调用会立即退出并指向新脚本 (其依赖的 `apply_rules_to_cross_section` 旧 API 已改签名)。


背景: 注入 A/B (`reports/injection_gate_ab_result.md`) 证明把 verified 规则注入诊断 prompt 会让
模型 regime 反配 (bear/crisis 更激进, bull 更保守), 故 gate 保持 0。方案3 把知识移到选股过滤器
(`agent/learning/knowledge_apply.py`), 只影响"选哪些票", 诊断官继续独立输出 risk_level。

验证目标 (3 个预注册窗口, 真实 replay parquet, 不跑 LLM):
  1. 接线正确: 过滤器删除的票都不满足规则 (pe_ttm>max_pe 或 pb>max_pb 或 NaN), 保留的票都满足。
  2. 解耦: 过滤器只作用于选股候选池 (screened=PreScreener 输出), 不修改诊断官输入 (全市场 df_cs)。
  3. 实际效果: 过滤后候选数 / 保底回退 (fallback) 触发率 / 消费端实际参数披露。

主指标(解耦)是**确定性成立**的: 诊断官 `_market_diagnostic(df_cs, ...)` 吃全市场截面
(historical_replay.py:1511), 过滤器插在 PreScreener 之后 (:1491) 只改 `screened`, 故 risk_level
与知识过滤结构性无关。因此本脚本**不跑 LLM 完整回放来"实测 risk_level 逐日一致"** —— 那是验证一个
代码结构保证的事实, 反而引入诊断 temperature 噪声。真正需要实测的是"过滤器在真实数据上是否按规则
正确过滤 + 是否导致候选池枯竭"。

副指标(选股质量): low_pe_value 的价值已由 tester 预注册判据验证 (5/6 窗口改善)。本脚本验证过滤器
在真实 replay 截面上复现同一条件, 并披露消费端实际参数与 tester 验证参数的差异 (若存在)。

决策: 接线正确(3/3 窗口删除票全不满足、保留票全满足) + 解耦成立 + 过滤不枯竭 → 可翻 gate=1;
否则保持 0。未改判据、未挑窗口、未调参追赢。

用法:
  python scripts/run_knowledge_apply_ab.py                 # 3 窗口, 每窗口 5 交易日
  python scripts/run_knowledge_apply_ab.py --samples 8     # 扩大样本
输出: reports/knowledge_apply_ab_result.json + .md
"""
from __future__ import annotations

import argparse
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
from agent.learning.knowledge_apply import recall_verified_rules, apply_rules_to_cross_section


def _force_utf8():
    for name in ("stdout", "stderr"):
        s = getattr(sys, name, None)
        if hasattr(s, "reconfigure"):
            try:
                s.reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass


# 预注册窗口: 与 run_injection_ab.py 一致的 regime 标注, 数据据 sh.000001 实际走势标定.
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


def _verify_filter(screened: pd.DataFrame, filtered: pd.DataFrame,
                   params: dict) -> dict:
    """验证过滤器正确性: 删除票不满足规则、保留票满足规则 (确定性布尔, 无 LLM)。

    规则谓词 = (pe_ttm <= max_pe) & (pb <= max_pb); 票被删 ⇔ 任一分量不满足/NaN。
    """
    max_pe = float(params.get("max_pe", 15.0))
    max_pb = float(params.get("max_pb", 2.0))
    has_pb = "pb" in screened.columns

    pe = pd.to_numeric(screened["pe_ttm"], errors="coerce")
    pb = pd.to_numeric(screened["pb"], errors="coerce") if has_pb else None
    kept_codes = set(filtered["code"].astype(str))
    removed_idx = screened.index[~screened["code"].astype(str).isin(kept_codes)]
    kept_idx = filtered.index

    # 删除票: 必须满足"PE 不合格或 NaN"或"PB 不合格或 NaN"(任一即删)
    removed_ok = True
    if len(removed_idx):
        pe_fail = pe.loc[removed_idx].isna() | (pe.loc[removed_idx] > max_pe)
        if has_pb:
            pb_fail = pb.loc[removed_idx].isna() | (pb.loc[removed_idx] > max_pb)
        else:
            pb_fail = pd.Series(False, index=removed_idx)
        removed_ok = bool((pe_fail | pb_fail).all())

    # 保留票: PE 和 PB 都必须非 NaN 且满足阈值
    kept_ok = True
    if len(kept_idx):
        pe_ok = pe.loc[kept_idx].notna() & (pe.loc[kept_idx] <= max_pe)
        pb_ok = (pb.loc[kept_idx].notna() & (pb.loc[kept_idx] <= max_pb)) if has_pb \
            else pd.Series(True, index=kept_idx)
        kept_ok = bool((pe_ok & pb_ok).all())

    return {
        "removed_ok": removed_ok, "kept_ok": kept_ok,
        "n_removed": len(removed_idx), "n_kept": len(kept_idx),
        "max_pe": max_pe, "max_pb": max_pb, "has_pb": has_pb,
    }


def run_window(w: dict, samples: int, top_n: int, min_keep: int) -> dict:
    label = w["label"]
    logger.info(f"\n=== 窗口 {label} ({w['regime_note']}) 数据 {w['data_file']} ===")

    # 取 verified 规则 (gate=1), 一次取回 params 供断言
    rules = recall_verified_rules(1)
    applicable = [r for r in rules if r.get("applicable")]
    if not applicable:
        logger.warning(f"  无 applicable verified 规则, 过滤器不启用 (rules={len(rules)})")
        return {
            "label": label, "regime_note": w["regime_note"], "n_dates": 0,
            "enabled": False, "rules": [r["template"] for r in rules],
            "params": {}, "n_fallback": 0, "n_rows": 0,
            "removed_ok_all": True, "kept_ok_all": True,
            "filtered_counts": [], "rows": [],
        }
    params_by_tpl = {r["template"]: r["params"] for r in applicable}
    logger.info(f"  applicable 规则: {list(params_by_tpl)} 参数={params_by_tpl}")

    basic, _universe = hr.load_snapshot_basic()   # total_mv 依赖快照市值 (PIT-in-price)
    _df = pd.read_parquet(w["data_file"])
    _df["date"] = pd.to_datetime(_df["date"])
    data = load_data_dict(w["data_file"])
    all_days = sorted(pd.to_datetime(_df["date"]).dt.date.unique())
    dates = _pick_dates(all_days, w["start"], w["end"], samples)
    logger.info(f"  采样 {len(dates)} 个交易日: {[str(d) for d in dates]}")

    rows = []
    for T in dates:
        Tstr = T.isoformat()
        df_cs = hr.reconstruct_cross_section(data, basic, Tstr)
        if df_cs.empty:
            logger.warning(f"  {Tstr} 截面为空, 跳过")
            continue
        if "isST" in df_cs.columns:
            df_cs = df_cs[df_cs["isST"] != 1]
        regime = _detect_regime(df_cs).get("regime", "range_bound")

        screener = PreScreener()
        screened = screener.screen(df_cs, regime=regime, top_n=top_n).df

        # 边界情况确认: screened 是否保留 pe_ttm/pb 列 (缺列 → 过滤器会 warn 跳过)
        has_pe = "pe_ttm" in screened.columns
        has_pb = "pb" in screened.columns

        filtered, report = apply_rules_to_cross_section(screened, applicable, min_keep=min_keep)
        fallback = bool(report.get("fallback"))

        # 仅非回退时验证删除/保留正确性 (回退 = 返回原池, 未过滤, 无删除票可验)
        v = {"removed_ok": True, "kept_ok": True, "n_removed": 0, "n_kept": len(screened)}
        if not fallback and "low_pe_value" in params_by_tpl and has_pe:
            v = _verify_filter(screened, filtered, params_by_tpl["low_pe_value"])

        rows.append({
            "date": Tstr, "regime": regime, "has_pe": has_pe, "has_pb": has_pb,
            "before": report["before"], "after": report["after"],
            "removed": report["removed"], "fallback": fallback,
            "removed_ok": v["removed_ok"], "kept_ok": v["kept_ok"],
        })
        logger.info(
            f"  {Tstr} [{regime}] pe列={has_pe} pb列={has_pb} "
            f"{report['before']}->{report['after']} (剔 {report['removed']}) "
            f"fallback={fallback} removed_ok={v['removed_ok']} kept_ok={v['kept_ok']}"
        )

    n_fallback = sum(1 for r in rows if r["fallback"])
    removed_ok_all = all(r["removed_ok"] for r in rows)
    kept_ok_all = all(r["kept_ok"] for r in rows)
    filtered_counts = [r["after"] for r in rows if not r["fallback"]]

    return {
        "label": label, "regime_note": w["regime_note"], "n_dates": len(dates),
        "enabled": True, "rules": list(params_by_tpl), "params": params_by_tpl,
        "n_fallback": n_fallback, "n_rows": len(rows),
        "removed_ok_all": bool(removed_ok_all), "kept_ok_all": bool(kept_ok_all),
        "filtered_counts": filtered_counts,
        "min_filtered": min(filtered_counts) if filtered_counts else None,
        "rows": rows,
    }


def main():
    _force_utf8()
    print("已废弃: 方案3 截面硬过滤已由 方案3b 确定性持仓择时取代。"
          "请改用 scripts/run_position_timing_ab.py")
    return
    ap = argparse.ArgumentParser(description="知识选股过滤接线/解耦验证 (方案3, 不跑 LLM)")
    ap.add_argument("--samples", type=int, default=5, help="每窗口采样交易日数")
    ap.add_argument("--top-n", type=int, default=300, help="PreScreener 候选池大小")
    ap.add_argument("--min-keep", type=int, default=50, help="过滤保底下限")
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
        results.append(run_window(w, args.samples, args.top_n, args.min_keep))

    # 决策
    n_enabled = sum(1 for r in results if r["enabled"])
    n_correct = sum(1 for r in results if r["enabled"] and r["removed_ok_all"] and r["kept_ok_all"])
    # 解耦是确定性成立的 (代码结构), 不在此量化; 接线正确 + 有实际过滤效果 → 可翻 gate=1
    any_effective = any(r["enabled"] and r["n_fallback"] < r["n_rows"] for r in results)
    verdict = "flip_gate1" if (n_correct >= 2 and len(results) >= 2 and any_effective) else "keep_gate0"

    out = {
        "date": pd.Timestamp.now().isoformat(timespec="seconds"),
        "samples": args.samples, "top_n": args.top_n, "min_keep": args.min_keep,
        "verdict": verdict, "enabled_windows": n_enabled,
        "correct_windows": n_correct, "windows_total": len(results),
        "any_effective": bool(any_effective),
        "results": results,
    }
    json_path = Path("reports/knowledge_apply_ab_result.json")
    json_path.write_text(json.dumps(out, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    _write_md(out)
    print(f"\n=== 结论: {verdict} (接线正确 {n_correct}/{len(results)}, 有实际效果={any_effective}) ===")
    for r in results:
        if not r["enabled"]:
            print(f"  {r['label']}: 无 applicable 规则 (rules={r['rules']})")
            continue
        print(f"  {r['label']} ({r['regime_note']}): 参数={r['params']} | "
              f"fallback {r['n_fallback']}/{r['n_rows']} | "
              f"删票正确={'✓' if r['removed_ok_all'] else '✗'} 留票正确={'✓' if r['kept_ok_all'] else '✗'} "
              f"| 过滤后中位={r['filtered_counts']}")
    print(f"\n结果已写入: {json_path} 和 reports/knowledge_apply_ab_result.md")


def _write_md(out: dict):
    lines = [
        "# 知识选股过滤 A/B 结果 (方案3, 接线/解耦验证, 不跑 LLM)",
        "",
        f"生成日期: {out['date']} | 采样 {out['samples']} 交易日/窗口 | "
        f"候选池 top_n={out['top_n']} 保底 min_keep={out['min_keep']}",
        "",
        f"## 结论: **{out['verdict']}** (接线正确 {out['correct_windows']}/{out['windows_total']} 窗口, "
        f"有实际过滤效果={out['any_effective']})",
        "",
        "| 窗口 | regime | 规则参数 | fallback | 删票正确 | 留票正确 | 过滤后中位/分布 |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in out["results"]:
        if not r["enabled"]:
            lines.append(f"| {r['label']} | {r['regime_note']} | 无 applicable 规则 | - | - | - | - |")
            continue
        p = r["params"].get("low_pe_value", {})
        dist = r["filtered_counts"]
        med = int(pd.Series(dist).median()) if dist else None
        lines.append(
            f"| {r['label']} | {r['regime_note']} | max_pe={p.get('max_pe')}, max_pb={p.get('max_pb')} "
            f"| {r['n_fallback']}/{r['n_rows']} "
            f"| {'✓' if r['removed_ok_all'] else '✗'} | {'✓' if r['kept_ok_all'] else '✗'} "
            f"| 中位={med} (n={len(dist)}) |"
        )
    lines += [
        "",
        "## 判据 (预注册)",
        "",
        "- **主指标(解耦)**: 过滤器只作用于选股候选池 `screened`, 不修改诊断官输入 (全市场 `df_cs`)。"
        "确定性成立 (诊断官 `_market_diagnostic(df_cs, ...)` 吃全市场截面, 见 historical_replay.py:1511),"
        "故本脚本不跑 LLM 回放实测 risk_level 一致 (那是验证代码结构保证的事实)。",
        "- **接线正确**: 非回退采样日, 删除票全不满足规则、保留票全满足 (确定性布尔断言)。",
        "- **实际效果**: 过滤后候选数 ≥ min_keep (不枯竭) 至少在一个窗口成立。",
        "- **决策**: ≥2/3 窗口接线正确 + 至少一窗口有实际过滤效果 → 可翻 gate=1; 否则保持 0。",
        "",
        "## 诚实局限",
        "",
        "- 本脚本验证「接线正确性 + 解耦 + 实际效果」, 不重新验证 low_pe_value 的选股价值"
        "(该价值已由 tester 预注册判据验证, 5/6 窗口改善)。",
        "- 消费端参数 (max_pb=1.0) 与 tester 验证参数一致 (同经 _sanitize_params), 但该参数本身是"
        "翻译幻觉的破净阈值, 过严导致过滤器 100% fallback 不生效; 且落地语义 (截面过滤) 不同于"
        "tester 验证的择时语义 (低估买/高估卖)。",
        "- fallback = 过滤后候选数 < min_keep 触发回退 (返回原池不过滤); 若 fallback 率 100%,"
        "说明参数过严导致过滤器实际不生效 (翻 gate=1 也无行为差异)。",
        "- 采样日只覆盖窗口内均匀抽样, 非逐日全量; 用于接线/枯竭验证足够, 不用于选股收益估计。",
    ]
    Path("reports/knowledge_apply_ab_result.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
