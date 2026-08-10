"""反事实验证 — 教训真的有用吗？

这是自我进化系统里"质量控制"的关键模块。

问题: LLM 复盘时说"如果当时我这么做，结果会更好"——但这是真的吗？
      LLM 很擅长编造听起来合理的理由，但真正的因果关系需要验证。

解法: 用历史数据做反事实回放——
      拿失败当天的数据，假设当时按照教训建议做了，
      重新计算收益/回撤，看看教训是不是真的有用。

只有通过反事实验证的教训，才能升级为"高置信度经验"。
这就是 Level 4 失败经验利用（见前沿调研报告）。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class CounterfactualResult:
    """反事实验证结果。"""
    experience_id: str
    date: str
    original_risk_level: int
    original_multiplier: float
    suggested_risk_level: int     # 教训建议的风险等级
    suggested_multiplier: float   # 教训建议的仓位系数
    actual_next_day_return: float  # 次日实际市场收益（%）
    original_pnl: float           # 原决策的次日盈亏（仓位系数 × 市场收益，相对值）
    counterfactual_pnl: float     # 反事实的次日盈亏
    improvement: float            # 改进幅度（counterfactual - original，正数=变好）
    passed: bool                  # 是否通过验证（改进 > 阈值）

    def to_dict(self) -> dict:
        return {
            "experience_id": self.experience_id,
            "date": self.date,
            "original_risk_level": self.original_risk_level,
            "original_multiplier": self.original_multiplier,
            "suggested_risk_level": self.suggested_risk_level,
            "suggested_multiplier": self.suggested_multiplier,
            "actual_next_day_return": self.actual_next_day_return,
            "original_pnl": self.original_pnl,
            "counterfactual_pnl": self.counterfactual_pnl,
            "improvement": self.improvement,
            "passed": self.passed,
        }


def risk_level_to_multiplier(level: int) -> float:
    """风险等级 → 参考仓位系数（和诊断官用的同一套映射）。"""
    mapping = {1: 1.45, 2: 1.20, 3: 0.95, 4: 0.65, 5: 0.40}
    return mapping.get(int(level), 0.95)


# v5.1 波动率归一系数: 阈值 = base_threshold * max(1, |次日收益| * VOL_SCALE).
# 让"小波动日()里的一点点改进"不被过度置信提升; 大波动日的改进才值得加分.
# 反事实天然是后视验证(用已发生的次日收益评判), 只用于置信度微调, 不做硬规则 → 不设自动升级.
VOL_SCALE = 0.5


def verify_counterfactual(
    experience,
    next_day_return: float,
    improvement_threshold: float = 0.001,  # 0.1% 以上才算真的改进
) -> Optional[CounterfactualResult]:
    """验证一条经验教训：如果当时按教训做了，结果会更好吗？

    简化版验证：用仓位系数 × 次日市场收益来估算盈亏。
    （完整的反事实需要重跑整个组合，但代价太高；
      这个简化版足以验证"风险等级判断方向对不对"）

    Args:
        experience: ExperienceItem 经验条目
        next_day_return: 次日市场涨跌幅（%，正数=涨）
        improvement_threshold: 改进阈值（绝对百分比），超过才算通过

    Returns:
        CounterfactualResult，无法验证时返回 None
    """
    try:
        # 解析原始决策
        orig_risk = experience.risk_level_given
        orig_mult = risk_level_to_multiplier(orig_risk)

        # 从教训中推断建议的风险等级
        # （经验条目不直接存建议等级，需要从 verdict 推断）
        # 简化: 如果 verdict=wrong 且偏差为正（给高了），建议降一级；反之亦然
        # 更精确的做法：从 lesson_detail 里让 LLM 提取，但那是下一轮优化的事
        suggestion = _infer_suggestion(experience)
        if suggestion is None:
            return None

        sug_risk = suggestion["risk_level"]
        sug_mult = risk_level_to_multiplier(sug_risk)

        # 计算原决策和反事实的盈亏
        # 用"仓位系数 × 市场收益"作为相对盈亏指标
        orig_pnl = orig_mult * next_day_return / 100.0  # 转化为小数
        cf_pnl = sug_mult * next_day_return / 100.0

        # 改进幅度
        improvement = cf_pnl - orig_pnl
        # 如果市场涨了 → 建议加仓（更激进）是对的 → cf_pnl > orig_pnl
        # 如果市场跌了 → 建议减仓（更保守）是对的 → cf_pnl（更小的负数） > orig_pnl（更大的负数）
        # 两种情况 improvement 都是正数，逻辑自洽

        # v5.1 波动率归一阈值: 小波动日里的一点点改进不轻易通过, 大波动日的改进才值得置信提升.
        # 避免"市场恰好朝建议方向动了一下"就被当作有效教训 (浅层后视).
        _mag = abs(next_day_return or 0.0)
        _threshold = improvement_threshold * max(1.0, _mag * VOL_SCALE)
        passed = improvement > _threshold

        return CounterfactualResult(
            experience_id=experience.id,
            date=experience.date,
            original_risk_level=orig_risk,
            original_multiplier=orig_mult,
            suggested_risk_level=sug_risk,
            suggested_multiplier=sug_mult,
            actual_next_day_return=next_day_return,
            original_pnl=orig_pnl,
            counterfactual_pnl=cf_pnl,
            improvement=improvement,
            passed=passed,
        )

    except Exception:
        return None


def _infer_suggestion(experience) -> Optional[dict]:
    """从经验条目中推断建议的风险等级。

    v5.1 接地版: 优先用复盘的 error_type tag（conservative/aggressive）直接判定,
    而不是靠关键词猜. 复盘 LLM 自己判断的"保守型错误/激进型错误"比关键词启发式可靠得多.
    - 保守型错误 (该上没上) → 建议降级（更激进）
    - 激进型错误 (不该上却上) → 建议升级（更保守）
    - verdict=correct → 已是正确的, 无需反事实验证, 返回 None

    关键词启发式降级为 fallback（旧记忆/无 error_type tag 时仍可用）。
    """
    if experience.verdict == "correct":
        return None  # 正确的经验不需要反事实验证（已经是对的）

    orig = experience.risk_level_given

    # 优先: error_type tag (Module A 已落 conservative/aggressive)
    tags = list(getattr(experience, "tags", []) or [])
    if "conservative" in tags:
        return {"risk_level": max(1, orig - 1)}  # 保守型错误 → 更激进
    if "aggressive" in tags:
        return {"risk_level": min(5, orig + 1)}  # 激进型错误 → 更保守

    # fallback: 关键词启发式（旧经验/没有 bias tag 时）
    text = experience.lesson_title + " " + experience.lesson_detail
    text_lower = text.lower()

    more_aggressive = any(kw in text_lower for kw in [
        "过于保守", "太谨慎", "低估", "风险给太高", "仓位太低",
        "错过了", "踏空", "可以更积极", "过度悲观"
    ])
    more_conservative = any(kw in text_lower for kw in [
        "过于激进", "太乐观", "高估", "风险给太低", "仓位太高",
        "忽视了风险", "应该减仓", "过度自信"
    ])

    if more_aggressive and not more_conservative:
        return {"risk_level": max(1, orig - 1)}  # 降一级 = 更激进
    if more_conservative and not more_aggressive:
        return {"risk_level": min(5, orig + 1)}  # 升一级 = 更保守

    return None


def audit_review_bias(
    review: dict,
    actual_return: float,
    risk_level: int,
) -> dict:
    """第二审计者 — 用实际市场结果独立复核 LLM 复盘的偏差分类.

    v5.2 P1: 打破"自证循环". 复盘的 error_type 是产生错误的同一个模型自己判的,
    同名模型看不到自己的系统性盲区(自我确认). 这里用确定性逻辑从实际结果+当时风险等级
    独立推导"数学上应该是什么偏差", 与 LLM 的自报 error_type 对比.

    判定规则 (基于仓位暴露方向):
    - 明显上涨(>0.5%) 且当时 risk_level>=4 (防御) → 实际偏保守(踏空)
    - 明显下跌(<-0.5%) 且当时 risk_level<=2 (进攻) → 实际偏激进(被套)
    - 其余 → 无法判定(indeterminate)

    Returns:
        {
          'audit_bias': 'conservative'|'aggressive'|'balanced'|'indeterminate',
          'llm_bias': 'conservative'|'aggressive'|None,
          'agrees': bool|None,     # audit 与 llm 是否一致; 无法判定时 None
          'overrides': bool,       # 是否应覆盖 LLM 的自报偏差 (audit 更可信)
          'note': str
        }
    """
    # 1. 独立偏差 (从实际结果 + 当时风险等级推导)
    _r = float(actual_return or 0.0)
    _rl = int(risk_level or 3)
    if _r > 0.5:
        audit_bias = "conservative" if _rl >= 4 else "balanced"
    elif _r < -0.5:
        audit_bias = "aggressive" if _rl <= 2 else "balanced"
    else:
        audit_bias = "indeterminate"

    # 2. LLM 自报偏差
    _et = str(review.get("error_type", "")).lower()
    llm_bias = None
    if "conservative" in _et or "保守" in _et:
        llm_bias = "conservative"
    elif "aggressive" in _et or "激进" in _et:
        llm_bias = "aggressive"
    elif review.get("verdict") == "correct":
        llm_bias = None  # 自报"正确"= 无偏差

    # 3. 对比
    if audit_bias == "indeterminate":
        agrees, overrides = None, False
        note = "结果波动太小, 无法独立审计偏差"
    elif audit_bias == "balanced":
        # 实际平衡: LLM 也自报无偏差 → 一致; LLM 自报了偏差 → 移除错误偏差tag
        if llm_bias is None:
            agrees, overrides = True, False
            note = "审计与实际均平衡, 与LLM一致"
        else:
            agrees, overrides = False, True
            note = f"实际平衡, 但 LLM 自报偏{llm_bias} → 移除偏差tag"
    elif llm_bias is None:
        # LLM 自报"正确/无偏差", 但实际明显偏保守/激进 → 自证盲区, 覆盖
        overrides, agrees = True, False
        note = f"审计发现实际偏{audit_bias}, 但 LLM 自报无偏差 → 覆盖为实际偏差"
    else:
        overrides = (llm_bias != audit_bias)
        agrees = not overrides
        note = ("审计与LLM一致" if agrees
                else f"审计判定偏{audit_bias}, LLM自报偏{llm_bias} → 以审计为准")

    return {
        "audit_bias": audit_bias,
        "llm_bias": llm_bias,
        "agrees": agrees,
        "overrides": overrides,
        "note": note,
    }


def batch_verify(experiences: list, market_returns: dict) -> list[CounterfactualResult]:
    """批量验证一组经验。

    Args:
        experiences: 经验条目列表
        market_returns: {date: next_day_return_pct} 字典，
                       key 是经验发生的日期，value 是次日市场涨跌幅

    Returns:
        通过验证的结果列表
    """
    results = []
    for exp in experiences:
        if exp.date not in market_returns:
            continue
        result = verify_counterfactual(exp, market_returns[exp.date])
        if result is not None:
            results.append(result)
    return results
