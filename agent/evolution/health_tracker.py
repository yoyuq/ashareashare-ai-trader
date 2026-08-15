"""进化质量在线追踪 — 可观测性层.

解决"静默失效": 进化系统此前只落 journal/memory/evolution 三份数据, 没有一份
"当前进化是否健康"的轻量快照. 本模块提供纯函数 + 一个落盘入口, 在每日复盘闭环
末尾写一份 evolution_health.json, 供 dashboard/人工抽查:

- 记忆库健康: 条数 / verdict 分布 / 置信度均值 / 反事实验证通过率 / 场景分布
- regime 注入对齐: 检测"牛市激进经验注入下跌段"这类跨界 (防 v5.2 过拟合复发)
- 每日活动: 注入经验数、反事实验证通过数、audit 覆盖、拖累票回流

纯函数无 IO 依赖, 可单测; 落盘单独放在 record_health 末尾.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from loguru import logger

# 复用 experience_memory 的方向映射 (避免复制一份导致漂移)
from .experience_memory import _SCENARIO_DIR, _REGIME_DIR


def snapshot_memory_health(memory, date: Optional[str] = None) -> dict:
    """记忆库健康快照 (纯函数).

    Args:
        memory: ExperienceMemory 实例 (只用 .items)
        date: 参考日期 (时间衰减用, 可空)

    Returns:
        dict: total / by_verdict / avg_confidence / cf_verified_ratio / cf_failed_ratio / by_scenario
    """
    items = list(getattr(memory, "items", []))
    if not items:
        return {
            "total": 0, "by_verdict": {}, "avg_confidence": 0.0,
            "cf_verified_ratio": 0.0, "cf_failed_ratio": 0.0, "by_scenario": {},
        }

    total = len(items)
    by_verdict = {}
    for v in ("correct", "wrong", "partial"):
        c = sum(1 for it in items if it.verdict == v)
        if c:
            by_verdict[v] = c

    avg_conf = sum(it.confidence for it in items) / total
    cf_verified = sum(1 for it in items if "cf_verified" in it.tags)
    cf_failed = sum(1 for it in items if "cf_failed" in it.tags)

    by_scenario = {}
    for it in items:
        by_scenario[it.scenario_type] = by_scenario.get(it.scenario_type, 0) + 1

    return {
        "total": total,
        "by_verdict": by_verdict,
        "avg_confidence": round(avg_conf, 4),
        "cf_verified_ratio": round(cf_verified / total, 4),
        "cf_failed_ratio": round(cf_failed / total, 4),
        "by_scenario": by_scenario,
    }


def snapshot_daily_activity(
    date: Optional[str] = None,
    injected_count: int = 0,
    cf_verified_count: int = 0,
    cf_total: int = 0,
    audit_count: int = 0,
    drag_count: int = 0,
) -> dict:
    """每日活动快照 (纯函数) — 记录当日进化系统各环节产出量.

    全部字段可选, 调用方按实际值填入; 缺省 0 表示当日无该环节.
    """
    return {
        "date": date,
        "injected_count": injected_count,          # 当日注入诊断 prompt 的经验条数
        "cf_verified_count": cf_verified_count,    # 当日反事实验证通过条数
        "cf_total": cf_total,                      # 当日反事实验证总条数
        "cf_verified_ratio": round(cf_verified_count / cf_total, 4) if cf_total else 0.0,
        "audit_count": audit_count,                # 当日 audit_review_bias 覆盖/确认次数
        "drag_count": drag_count,                  # 当日拖累票回流经验条数
    }


def check_regime_alignment(regime: str, injected_items: list, threshold: float = 0.5) -> dict:
    """regime 注入对齐监测 (纯函数).

    统计注入经验中"跨界"(方向与当前 regime 相反)条数占比. 跨界占比超过 threshold
    时返回 alert=True + 原因 — 正是 v5.2 留出法 A/B 暴露的"牛市激进经验注入下跌段"
    过拟合症状, 一旦复发立即告警.

    Args:
        regime: 当前市场状态 (strong_bull/weak_bear/range_bound...)
        injected_items: 实际注入 prompt 的经验列表 (ExperienceItem)
        threshold: 跨界占比告警阈值, 默认 0.5

    Returns:
        dict: {regime, injected_count, cross_count, cross_ratio, alert, reason}
    """
    _r = _REGIME_DIR.get(regime)
    if not injected_items:
        return {
            "regime": regime, "injected_count": 0, "cross_count": 0,
            "cross_ratio": 0.0, "alert": False, "reason": "无注入经验",
        }

    cross = 0
    for it in injected_items:
        _s = _SCENARIO_DIR.get(it.scenario_type)
        if _r and _s and _s != "neutral" and _s != _r:
            cross += 1

    total = len(injected_items)
    ratio = cross / total
    alert = ratio > threshold
    reason = (
        f"注入经验中 {cross}/{total} 条与当前 regime({regime})方向相反"
        if alert
        else f"注入经验方向基本对齐 regime({regime})"
    )
    if alert:
        logger.warning(f"[进化健康] regime 注入对齐告警: {reason}")
    return {
        "regime": regime, "injected_count": total, "cross_count": cross,
        "cross_ratio": round(ratio, 4), "alert": alert, "reason": reason,
    }


def record_health(
    memory,
    date: str,
    regime: Optional[str],
    path: Optional[str | Path] = None,
    injected_count: int = 0,
    cf_verified_count: int = 0,
    cf_total: int = 0,
    audit_count: int = 0,
    drag_count: int = 0,
) -> dict:
    """汇总一次进化健康快照并落盘 (计算纯函数 + 末尾 IO).

    注入对齐用与诊断注入一致的检索参数 (top_k=6, 带 regime 惩罚), 反映"当日若诊断
    会注入什么", 跨界占比超阈值即告警. 实盘落 simulation_data/evolution_health.json,
    回放落 replay_data/evolution_health_{tag}.json.

    Returns:
        dict: 完整 payload (date/memory_health/regime_align/daily_activity)
    """
    mem_health = snapshot_memory_health(memory, date)
    injected = []
    if memory is not None and regime:
        try:
            injected = memory.retrieve(current_date=date, top_k=6, regime=regime)
        except Exception:
            injected = []
    align = (
        check_regime_alignment(regime, injected)
        if regime
        else {"regime": None, "injected_count": 0, "cross_count": 0,
              "cross_ratio": 0.0, "alert": False, "reason": "无 regime"}
    )
    activity = snapshot_daily_activity(
        date=date, injected_count=injected_count,
        cf_verified_count=cf_verified_count, cf_total=cf_total,
        audit_count=audit_count, drag_count=drag_count,
    )
    payload = {
        "date": date,
        "memory_health": mem_health,
        "regime_align": align,
        "daily_activity": activity,
    }
    if path:
        try:
            p = Path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as e:
            logger.warning(f"[进化健康] 落盘失败: {e}")
    return payload
