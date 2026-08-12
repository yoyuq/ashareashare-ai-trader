# -*- coding: utf-8 -*-
"""飞书"节点7 观点汇总"增强版 — 加入【有效独立观点熵】.

在原有(多空统计+派系加权)基础上, 新增:
  - effective_independent: 2^entropy(观点分布) — 实证"有几个独立观点", 戳破"号称九人/三人"
  - school_divergence:      派系间分歧度 — 分歧集中在仓位还是方向
  - 诚实标注: 若 effective_independent 接近1个 → 说明大师们高度同质(同一模型), 系统如实披露

替换飞书"节点7 观点汇总"的执行代码. 入参/出参不变.
"""
import json
import math
from collections import Counter


def main(arg1: str, arg2: str, arg3: str, arg4: str) -> dict:
    # ===== 派系权重 (按市场阶段; 注意: 这是过拟合风险点, 竞赛可加"影子验证")=====
    SCHOOL_WEIGHTS = {
        "strong_bull": {"trend": 1.6, "value": 0.7, "macro": 1.0},
        "weak_bull": {"trend": 1.3, "value": 0.9, "macro": 1.0},
        "sideways": {"trend": 1.0, "value": 1.0, "macro": 1.0},
        "recovery": {"trend": 1.2, "value": 1.3, "macro": 1.0},
        "weak_bear": {"trend": 1.2, "value": 1.1, "macro": 1.2},
        "strong_bear": {"trend": 1.0, "value": 1.5, "macro": 1.2},
        "panic": {"trend": 0.8, "value": 1.6, "macro": 1.1},
    }
    SCHOOL_LABELS = {"trend": "趋势派", "value": "价值派", "macro": "宏观派"}

    def safe_parse(s):
        if not s:
            return {}
        s = s.strip()
        if s.startswith("```"):
            s = s.strip("`")
            if s.startswith("json"):
                s = s[4:]
            s = s.strip()
        try:
            return json.loads(s)
        except Exception:
            start, end = s.find("{"), s.rfind("}")
            if start >= 0 and end > start:
                try:
                    return json.loads(s[start:end + 1])
                except Exception:
                    pass
        return {}

    opinions = []
    for raw in [arg1, arg2, arg3]:
        op = safe_parse(raw)
        if op:
            opinions.append(op)

    if not opinions:
        opinions = [{"master_name": "利弗莫尔", "school": "trend", "market_view": "neutral",
                     "confidence": 0.5, "risk_level_assessment": 3, "position_suggestion": 1.0,
                     "key_reasons": ["数据解析失败，默认中性"], "key_risks": [], "style": "趋势跟踪",
                     "classic_quote": "", "target_sectors": []}]

    market_data = json.loads(arg4) if arg4 else {}
    regime = market_data.get("regime", "sideways")

    # ===== 多空统计 =====
    bull = sum(1 for o in opinions if o.get("market_view") == "bullish")
    bear = sum(1 for o in opinions if o.get("market_view") == "bearish")
    neutral = sum(1 for o in opinions if o.get("market_view") == "neutral")

    # ===== ★ 有效独立观点熵 (核心创意: 实证独立性) =====
    views = [o.get("market_view") for o in opinions if o.get("market_view")]
    if views:
        c = Counter(views)
        n = len(views)
        entropy = -sum((v / n) * math.log2(v / n) for v in c.values())
        effective_independent = round(2 ** entropy, 2)
    else:
        entropy, effective_independent = 0.0, 0.0

    # ===== 派系分歧度 (分歧集中在方向还是仓位) =====
    by_school = {"trend": [], "value": [], "macro": []}
    for o in opinions:
        if o.get("school") in by_school:
            by_school[o["school"]].append(o)
    school_views = {s: [o.get("market_view") for o in ops]
                    for s, ops in by_school.items() if ops}
    school_divergence = {}
    for s, vs in school_views.items():
        distinct = len(set(vs))
        # 分歧度: 1=完全一致, 0=三档全占
        school_divergence[SCHOOL_LABELS.get(s, s)] = round(distinct / 3, 2)

    # ===== 按派系动态加权 (保留原逻辑) =====
    weights = SCHOOL_WEIGHTS.get(regime, {"trend": 1.0, "value": 1.0, "macro": 1.0})
    weighted_risk = weighted_pos = total_weight = 0.0
    for school, school_ops in by_school.items():
        if not school_ops:
            continue
        school_w = weights.get(school, 1.0)
        per_master_w = school_w / len(school_ops)
        for o in school_ops:
            conf = o.get("confidence", 0.5)
            w = per_master_w * (0.5 + conf * 0.5)
            weighted_risk += o.get("risk_level_assessment", 3) * w
            weighted_pos += o.get("position_suggestion", 1.0) * w
            total_weight += w
    avg_risk = weighted_risk / total_weight if total_weight > 0 else 3.0
    avg_pos = weighted_pos / total_weight if total_weight > 0 else 1.0
    consensus = abs(bull - bear) / len(opinions) if opinions else 0
    dominant = max(opinions, key=lambda x: x.get("confidence", 0))

    school_stats = {}
    for school, ops in by_school.items():
        if not ops:
            continue
        school_stats[school] = {
            "count": len(ops),
            "bull": sum(1 for o in ops if o.get("market_view") == "bullish"),
            "bear": sum(1 for o in ops if o.get("market_view") == "bearish"),
            "neutral": sum(1 for o in ops if o.get("market_view") == "neutral"),
            "avg_risk": round(sum(o.get("risk_level_assessment", 3) for o in ops) / len(ops), 2),
            "avg_pos": round(sum(o.get("position_suggestion", 1.0) for o in ops) / len(ops), 2),
            "avg_conf": round(sum(o.get("confidence", 0.5) for o in ops) / len(ops), 3),
            "weight": weights.get(school, 1.0),
        }

    # ===== 独立性诚实判定 =====
    if effective_independent <= 1.2:
        independence_note = (f"有效独立观点≈{effective_independent}个，大师观点高度同质"
                             "（可能同一模型），系统如实披露，加权价值有限")
    elif effective_independent <= 2.5:
        independence_note = (f"有效独立观点≈{effective_independent}个，存在部分同质化，"
                             "建议提高大师提示词的差异化")
    else:
        independence_note = f"有效独立观点≈{effective_independent}个，观点分布健康"

    result = {
        "total_masters": len(opinions),
        "bull_count": bull, "bear_count": bear, "neutral_count": neutral,
        "consensus_score": round(consensus, 3),
        "avg_risk_level": round(avg_risk, 2),
        "avg_position_multiplier": round(avg_pos, 2),
        "dominant_master": dominant.get("master_name", "未知"),
        "dominant_school": dominant.get("school", "trend"),
        "by_school": school_stats,
        "opinions": opinions,
        "regime": regime,
        # ★ 新增: 有效独立观点熵 + 派系分歧 + 诚实判定
        "effective_independent": effective_independent,
        "view_entropy": round(entropy, 3),
        "school_divergence": school_divergence,
        "independence_note": independence_note,
    }
    return {'summary': json.dumps(result, ensure_ascii=False)}


if __name__ == "__main__":
    # 本地冒烟: 模拟 3 个同质大师(应得低熵≈1) vs 3 个分歧大师(应得高熵)
    mk = '{"regime":"strong_bull"}'
    same_bull = ('{"master_name":"利弗莫尔","school":"trend","market_view":"bullish","confidence":0.8,"risk_level_assessment":2,"position_suggestion":1.3}',
                 '{"master_name":"巴菲特","school":"value","market_view":"bullish","confidence":0.7,"risk_level_assessment":2,"position_suggestion":1.2}',
                 '{"master_name":"索罗斯","school":"macro","market_view":"bullish","confidence":0.75,"risk_level_assessment":2,"position_suggestion":1.3}')
    split = ('{"master_name":"利弗莫尔","school":"trend","market_view":"bullish","confidence":0.8,"risk_level_assessment":2,"position_suggestion":1.3}',
             '{"master_name":"巴菲特","school":"value","market_view":"bearish","confidence":0.7,"risk_level_assessment":4,"position_suggestion":0.6}',
             '{"master_name":"索罗斯","school":"macro","market_view":"neutral","confidence":0.6,"risk_level_assessment":3,"position_suggestion":1.0}')
    for label, args in (("全看多(同质)", same_bull), ("多空分歧", split)):
        r = json.loads(main(*args, mk)["summary"])
        print(f"{label}: 有效独立观点={r['effective_independent']} 熵={r['view_entropy']} "
              f"共识度={r['consensus_score']} | {r['independence_note']}")