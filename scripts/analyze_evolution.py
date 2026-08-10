"""分析自我进化回放的结果。

不仅看总收益，更重要的是验证系统是否真的在学习进化：
1. 前半段 vs 后半段 表现对比
2. 经验积累曲线
3. 进化周期前后的变化
4. 大师选择分布变化
5. 风险等级偏差变化
"""
from __future__ import annotations

import json
import math
from pathlib import Path


def load_equity_curve(report_path):
    with open(report_path, "r", encoding="utf-8") as f:
        r = json.load(f)
    return r


def calc_metrics(eq_curve):
    """计算收益、回撤、夏普、胜率。"""
    if not eq_curve:
        return {}
    peak = float(eq_curve[0]["total"])
    max_dd = 0
    rets = []
    for i, d in enumerate(eq_curve):
        v = float(d["total"])
        if v > peak:
            peak = v
        dd = (v - peak) / peak * 100
        if dd < max_dd:
            max_dd = dd
        if i > 0:
            prev = float(eq_curve[i-1]["total"])
            rets.append((v - prev) / prev)
    total_ret = (float(eq_curve[-1]["total"]) - float(eq_curve[0]["total"])) / float(eq_curve[0]["total"]) * 100
    if rets:
        mean_r = sum(rets)/len(rets)
        std_r = math.sqrt(sum((x-mean_r)**2 for x in rets)/len(rets))
        sharpe = mean_r/std_r * math.sqrt(250) if std_r > 0 else 0
        win_rate = sum(1 for x in rets if x > 0) / len(rets)
    else:
        sharpe = 0
        win_rate = 0
    return {
        "total_return": total_ret,
        "max_drawdown": max_dd,
        "sharpe": sharpe,
        "win_rate": win_rate,
        "num_days": len(eq_curve),
    }


def split_half_metrics(eq_curve):
    """前半段 vs 后半段。"""
    mid = len(eq_curve) // 2
    first_half = eq_curve[:mid]
    second_half = eq_curve[mid:]
    return {
        "first_half": calc_metrics(first_half),
        "second_half": calc_metrics(second_half),
    }


def analyze_journal(journal_path):
    """分析决策日志。"""
    records = []
    with open(journal_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                pass

    if not records:
        return {}

    # 大师选择分布
    master_counts = {}
    for r in records:
        m = r.get("dominant_master", "unknown")
        master_counts[m] = master_counts.get(m, 0) + 1

    # 风险等级分布
    risk_dist = {}
    for r in records:
        rl = r.get("risk_level", 3)
        risk_dist[rl] = risk_dist.get(rl, 0) + 1

    # 市场阶段分布
    phase_dist = {}
    for r in records:
        p = r.get("market_phase", "unknown")
        phase_dist[p] = phase_dist.get(p, 0) + 1

    # 复盘结果分布
    reviewed = [r for r in records if r.get("review")]
    verdict_dist = {}
    for r in reviewed:
        v = r["review"].get("verdict", "unknown")
        verdict_dist[v] = verdict_dist.get(v, 0) + 1

    # 风险等级偏差
    deviations = []
    for r in reviewed:
        dev = r["review"].get("risk_level_deviation", 0)
        deviations.append(dev)
    avg_dev = sum(abs(d) for d in deviations) / len(deviations) if deviations else 0

    # 前半段 vs 后半段 复盘表现
    mid = len(reviewed) // 2
    if mid > 0:
        first_devs = [abs(r["review"].get("risk_level_deviation", 0)) for r in reviewed[:mid]]
        second_devs = [abs(r["review"].get("risk_level_deviation", 0)) for r in reviewed[mid:]]
        avg_first = sum(first_devs) / len(first_devs)
        avg_second = sum(second_devs) / len(second_devs)
    else:
        avg_first = avg_second = 0

    # 分阶段大师选择变化（前20天 vs 后20天）
    if len(records) >= 40:
        early_masters = {}
        for r in records[:20]:
            m = r.get("dominant_master", "unknown")
            early_masters[m] = early_masters.get(m, 0) + 1
        late_masters = {}
        for r in records[-20:]:
            m = r.get("dominant_master", "unknown")
            late_masters[m] = late_masters.get(m, 0) + 1
        master_shift = {
            "early": early_masters,
            "late": late_masters,
        }
    else:
        master_shift = {}

    return {
        "total_decisions": len(records),
        "reviewed_count": len(reviewed),
        "master_distribution": master_counts,
        "risk_distribution": risk_dist,
        "phase_distribution": phase_dist,
        "verdict_distribution": verdict_dist,
        "avg_risk_deviation": avg_dev,
        "deviation_first_half": avg_first,
        "deviation_second_half": avg_second,
        "deviation_improvement": avg_first - avg_second,  # 正数=进步
        "master_shift": master_shift,
    }


def analyze_memory(memory_path):
    """分析经验记忆库。"""
    with open(memory_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    items = data.get("items", [])

    by_verdict = {}
    by_master = {}
    by_scenario = {}
    for item in items:
        v = item.get("verdict", "unknown")
        by_verdict[v] = by_verdict.get(v, 0) + 1
        m = item.get("master_used", "unknown")
        by_master[m] = by_master.get(m, 0) + 1
        s = item.get("scenario_type", "unknown")
        by_scenario[s] = by_scenario.get(s, 0) + 1

    return {
        "total_experiences": len(items),
        "by_verdict": by_verdict,
        "by_master": by_master,
        "by_scenario": by_scenario,
        "top_confidence": sorted(
            [(item.get("lesson_title", ""), item.get("confidence", 0)) for item in items],
            key=lambda x: x[1], reverse=True
        )[:5],
    }


def analyze_evolution(evolution_path):
    """分析进化总结历史。"""
    with open(evolution_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    snapshots = data.get("snapshots", [])

    evolution_summary = []
    for s in snapshots:
        summary = s.get("summary", {})
        principles = s.get("principles", [])
        evolution_summary.append({
            "period": f"{s.get('period_start', '?')} ~ {s.get('period_end', '?')}",
            "num_decisions": s.get("num_decisions", 0),
            "period_summary": summary.get("period_summary", ""),
            "biases": summary.get("biases_identified", []),
            "num_principles": len(principles),
            "next_focus": summary.get("next_period_focus", ""),
        })

    return {
        "num_evolution_cycles": len(snapshots),
        "cycles": evolution_summary,
    }


def full_analysis(tag: str, replay_dir: str = "replay_data") -> dict:
    """完整分析一个 v4.0 进化回放。"""
    replay_dir = Path(replay_dir)
    report_path = replay_dir / f"replay_report_{tag}.json"
    journal_path = replay_dir / f"journal_{tag}.jsonl"
    memory_path = replay_dir / f"memory_{tag}.json"
    evolution_path = replay_dir / f"evolution_{tag}.json"

    result = {"tag": tag}

    if report_path.exists():
        report = load_equity_curve(report_path)
        eq = report.get("equity_curve", [])
        result["overall"] = calc_metrics(eq)
        result["half_split"] = split_half_metrics(eq)
        result["total_return_pct"] = report.get("total_return_pct")
        result["final_equity"] = report.get("final_equity")
    else:
        result["overall"] = {}
        result["half_split"] = {}

    if journal_path.exists():
        result["journal"] = analyze_journal(journal_path)
    else:
        result["journal"] = {}

    if memory_path.exists():
        result["memory"] = analyze_memory(memory_path)
    else:
        result["memory"] = {}

    if evolution_path.exists():
        result["evolution"] = analyze_evolution(evolution_path)
    else:
        result["evolution"] = {}

    return result


def print_analysis(analysis: dict):
    """打印可读的分析报告。"""
    tag = analysis.get("tag", "?")
    print(f"\n{'='*60}")
    print(f"  自我进化回放分析: {tag}")
    print(f"{'='*60}")

    # 整体表现
    o = analysis.get("overall", {})
    if o:
        print(f"\n【整体表现】")
        print(f"  总收益: {o.get('total_return', 0):+.2f}%")
        print(f"  最大回撤: {o.get('max_drawdown', 0):+.2f}%")
        print(f"  夏普比: {o.get('sharpe', 0):.2f}")
        print(f"  日胜率: {o.get('win_rate', 0):.1%}")
        print(f"  交易天数: {o.get('num_days', 0)}")

    # 前后半段对比
    hs = analysis.get("half_split", {})
    if hs:
        f = hs.get("first_half", {})
        s = hs.get("second_half", {})
        print(f"\n【前半段 vs 后半段】")
        print(f"  {'指标':<12} {'前半段':>10} {'后半段':>10} {'变化':>10}")
        print(f"  {'-'*44}")
        for metric, label in [("total_return", "总收益(%)"), ("sharpe", "夏普"), ("win_rate", "胜率")]:
            fv = f.get(metric, 0)
            sv = s.get(metric, 0)
            diff = sv - fv
            if metric == "win_rate":
                print(f"  {label:<12} {fv:>9.1%} {sv:>9.1%} {diff:>+9.1%}")
            else:
                print(f"  {label:<12} {fv:>+9.2f} {sv:>+9.2f} {diff:>+9.2f}")

        # 判断是否在进步
        ret_improve = s.get("total_return", 0) - f.get("total_return", 0)
        sharpe_improve = s.get("sharpe", 0) - f.get("sharpe", 0)
        if ret_improve > 0 and sharpe_improve > 0:
            print(f"\n  ✅ 系统在进化！后半段收益和夏普都更高")
        elif ret_improve > 0:
            print(f"\n  ⚠️  收益提升了，但夏普没跟上（风险增大了？）")
        elif sharpe_improve > 0:
            print(f"\n  ⚠️  夏普提升了，但收益下降了（更保守了？）")
        else:
            print(f"\n  ❌ 后半段表现更差，系统没有在学习")

    # 决策日志分析
    j = analysis.get("journal", {})
    if j:
        print(f"\n【决策日志分析】")
        print(f"  总决策数: {j.get('total_decisions', 0)}")
        print(f"  已复盘: {j.get('reviewed_count', 0)}")
        print(f"  平均风险等级偏差: {j.get('avg_risk_deviation', 0):.2f} 级")
        print(f"  前半段偏差: {j.get('deviation_first_half', 0):.2f} 级")
        print(f"  后半段偏差: {j.get('deviation_second_half', 0):.2f} 级")
        impr = j.get('deviation_improvement', 0)
        print(f"  偏差改进: {impr:+.2f} 级 {'(进步)' if impr > 0 else '(退步)'}")

        print(f"\n  大师选择分布:")
        for m, c in sorted(j.get("master_distribution", {}).items(), key=lambda x: -x[1]):
            print(f"    {m}: {c}次 ({c/max(j['total_decisions'],1):.1%})")

        print(f"\n  风险等级分布:")
        for rl in sorted(j.get("risk_distribution", {}).keys()):
            c = j["risk_distribution"][rl]
            print(f"    {rl}级: {c}次 ({c/max(j['total_decisions'],1):.1%})")

        print(f"\n  复盘结果分布:")
        for v, c in j.get("verdict_distribution", {}).items():
            print(f"    {v}: {c}次")

        shift = j.get("master_shift", {})
        if shift:
            print(f"\n  大师选择变化（前20天 vs 后20天）:")
            early = shift.get("early", {})
            late = shift.get("late", {})
            all_masters = set(list(early.keys()) + list(late.keys()))
            for m in sorted(all_masters, key=lambda x: -(early.get(x,0) + late.get(x,0))):
                ec = early.get(m, 0)
                lc = late.get(m, 0)
                change = lc - ec
                print(f"    {m}: {ec} → {lc} ({change:+d})")

    # 经验记忆库分析
    m = analysis.get("memory", {})
    if m:
        print(f"\n【经验记忆库】")
        print(f"  总经验数: {m.get('total_experiences', 0)}")
        print(f"  按类型:")
        for v, c in m.get("by_verdict", {}).items():
            label = {"correct": "正确经验", "wrong": "错误教训", "partial": "部分正确"}.get(v, v)
            print(f"    {label}: {c}条")

        top_conf = m.get("top_confidence", [])
        if top_conf:
            print(f"\n  置信度最高的 5 条经验:")
            for i, (title, conf) in enumerate(top_conf, 1):
                print(f"    {i}. [{conf:.0%}] {title}")

    # 进化总结分析
    e = analysis.get("evolution", {})
    if e and e.get("num_evolution_cycles", 0) > 0:
        print(f"\n【进化总结历史】")
        print(f"  进化周期数: {e['num_evolution_cycles']}")
        for i, cyc in enumerate(e.get("cycles", []), 1):
            print(f"\n  第{i}周期: {cyc['period']} ({cyc['num_decisions']}个决策)")
            print(f"    总结: {cyc['period_summary']}")
            if cyc.get("biases"):
                print(f"    识别偏见: {', '.join(cyc['biases'])}")
            print(f"    核心原则: {cyc['num_principles']}条")
            if cyc.get("next_focus"):
                print(f"    下期重点: {cyc['next_focus']}")

    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    import sys
    tag = sys.argv[1] if len(sys.argv) > 1 else "diag_v40_evo"
    analysis = full_analysis(tag)
    print_analysis(analysis)
