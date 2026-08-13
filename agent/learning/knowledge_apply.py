"""知识消费端 — 把 verified 规则确定性翻译成「持仓择时」信号 (方案3b, 与诊断官解耦)。

背景: 注入 A/B (`reports/injection_gate_ab_result.md`) 证明把规则注入诊断 prompt 会让模型在
bear/crisis 里更激进、在 bull 里更保守 (注入方向系统性 regime 反配)。根因是这两条规则本质是
**选股/择时信号**, 却被塞给了**管总仓位风险**的诊断官。

方案3 (截面硬过滤) A/B (`reports/knowledge_apply_ab_result.md`) 又证明: 截面一次性硬过滤无法忠实
复现规则的价值 —— low_pe_value 是**个股择时规则** (低估 `pe<max_pe&pb<max_pb` 买入、高估
`pe>max_pe*1.5` 卖出, 见 `tester.py:287-297`), 它的价值来自"回避高估值 + 动态进出", 不是"候选池
只留低估值"。

方案3b (确定性持仓择时): 把 low_pe_value 忠实落地成**确定性布尔择时信号** (无 LLM, 不会被 regime
误读), 与 tester 的 `_entry_exit` 同构:
- **买入门 (buy gate)**: 候选票须满足 `pe_ttm < max_pe 且 pb < max_pb` (且 pe>0/pb>0) 才进候选;
  **无保底回退** —— "只买低估"是规则本意, 候选池少是特性不是 bug。
- **卖出信号 (sell signal)**: 持仓票 `pe_ttm > max_pe*1.5` → 确定性卖出 (高估即减仓)。

纪律 (贯穿, 与 [[no-simulation-constraint]] 一致):
- 全程零模拟: 只在真实截面 DataFrame 上做布尔谓词; 缺所需列 → 跳过该规则 (不伪造列)。
- 忠实映射: 只落地**截面可执行的择时规则** (纯阈值, 如 low_pe_value); **时序型** (ma_cross 需 MA 历史)
  标记 not_applicable, 不强行语义改写 —— 与 tester 的 `not_yet_testable` 同一纪律。
- 脏参数夹回: 复用 `tester._sanitize_params`, 防 `max_pe=0.5` 这类 LLM 翻译幻觉参数。
- gate 默认 0 (现状不变): `LEARNED_KNOWLEDGE_GATE` 控制; 0=不启用, 1=只 verified, 2=verified+高置信。
"""
from __future__ import annotations

import json
from typing import Optional

import pandas as pd
from loguru import logger

from .knowledge_history import learned_gate
from .tester import _sanitize_params


# ── 可落地成"确定性持仓择时"的规则 → 语义说明 (只收截面型; 时序型不在本映射) ──
# 每条规则同时提供两个谓词: 买入门 (低估才进候选) + 卖出信号 (高估即减仓)。
_APPLICABLE_RULES: dict[str, str] = {
    "low_pe_value": "低估值择时: 买入门 pe_ttm<max_pe&pb<max_pb (pe/pb>0), 卖出 pe_ttm>max_pe*1.5",
    # ma_cross / rsi_reversal / 各类止损 等时序型规则需要历史序列, 截面无 MA/RSI 历史,
    # 无法忠实落地 → recall 时标记 not_applicable (不强行用动量字段语义改写)。
}


def _parse_params(template: str, params_raw) -> dict:
    """解析并夹回 params (metadata 里嵌套 dict 被压成 JSON 字符串)。

    params_raw 可能是 JSON 字符串 / dict / None。统一 json.loads → dict,
    再过 _sanitize_params 把越界/幻觉值夹回模板文档写明的忠实默认。
    """
    if isinstance(params_raw, str):
        try:
            p = json.loads(params_raw)
        except (json.JSONDecodeError, TypeError):
            p = {}
    elif isinstance(params_raw, dict):
        p = params_raw
    else:
        p = {}
    return _sanitize_params(template, p)


def recall_verified_rules(gate: Optional[int] = None) -> list[dict]:
    """从向量库取回 verified 规则, 含 template + params (已夹回脏值)。

    gate<=0 → 返回 [] (现状不变, 零开销不碰 ChromaDB)。
    用 `list_learned` (完整 meta 含 template/params), 不用 `recall_learned` (不含这两个字段)。

    返回 [{concept, template, params, confidence, applicable, reason}];
    applicable=False 表示该规则无法忠实映射成确定性持仓择时 (如时序型 ma_cross)。
    """
    _gate = learned_gate() if gate is None else int(gate)
    if _gate <= 0:
        return []
    try:
        from knowledge.manager import KnowledgeManager
        km = KnowledgeManager("knowledge/")
        items = km.list_learned(categories=["rule"], verdicts=["verified"])
    except Exception as e:
        logger.warning(f"[知识持仓择时] 取回 verified 规则失败, 跳过: {e}")
        return []

    rules = []
    for it in items:
        meta = it.get("meta") or {}
        template = meta.get("template")
        confidence = meta.get("confidence", 0.5)
        # gate=2: 额外要求 confidence >= 0.75
        if _gate >= 2:
            try:
                if float(confidence or 0.5) < 0.75:
                    continue
            except (TypeError, ValueError):
                continue
        if not template:
            rules.append({"concept": meta.get("concept", ""), "template": None,
                          "params": {}, "confidence": confidence,
                          "applicable": False, "reason": "无 template, 无法映射"})
            continue
        params = _parse_params(template, meta.get("params"))
        applicable = template in _APPLICABLE_RULES
        reason = "" if applicable else "时序型(需历史序列), 暂不适用确定性持仓择时"
        rules.append({"concept": meta.get("concept", ""), "template": template,
                      "params": params, "confidence": confidence,
                      "applicable": applicable, "reason": reason})
    return rules


# ── 买入门谓词 (忠实 tester._entry_exit 的 cb) ──

def buy_gate_low_pe_value(df: pd.DataFrame, params: dict) -> Optional[pd.Series]:
    """low_pe_value 买入门: (pe_ttm>0) & (pe_ttm<max_pe) & (pb>0) & (pb<max_pb)。

    忠实 tester.py:293-294 的 cb (pe<max_pe 且 pb<max_pb 且 _valid 即 pe/pb>0 非 NaN)。
    缺 pe_ttm/pb 列 → 返回 None (调用方跳过, 不伪造)。亏损股 (pe<=0) / 破净缺失 (pb<=0) 剔除。
    """
    if "pe_ttm" not in df.columns or "pb" not in df.columns:
        return None
    max_pe = float(params.get("max_pe", 15.0))
    max_pb = float(params.get("max_pb", 2.0))
    pe = pd.to_numeric(df["pe_ttm"], errors="coerce")
    pb = pd.to_numeric(df["pb"], errors="coerce")
    return (pe.notna() & (pe > 0) & (pe < max_pe)
            & pb.notna() & (pb > 0) & (pb < max_pb))


# ── 卖出信号谓词 (忠实 tester._entry_exit 的 cs) ──

def sell_signal_low_pe_value(df: pd.DataFrame, params: dict) -> Optional[pd.Series]:
    """low_pe_value 卖出信号: pe_ttm > max_pe*1.5 (高估即减仓)。

    忠实 tester.py:295-296 的 cs (pe>max_pe*1.5)。tester 的 cs 还含 `not _valid` (pe/pb 变 NaN 也卖),
    但持仓票已通过买入门 (pe/pb>0), 故主触发是 pe>22.5; 这里只保留主触发, 不把"估值变 NaN"误判成
    确定卖出 (避免持股因一次数据缺失被清仓)。
    缺 pe_ttm 列 → 返回 None (调用方跳过)。
    """
    if "pe_ttm" not in df.columns:
        return None
    max_pe = float(params.get("max_pe", 15.0))
    pe = pd.to_numeric(df["pe_ttm"], errors="coerce")
    return pe > max_pe * 1.5


_BUY_GATE_FN = {
    "low_pe_value": buy_gate_low_pe_value,
}
_SELL_SIGNAL_FN = {
    "low_pe_value": sell_signal_low_pe_value,
}


def apply_rules_to_cross_section(df: pd.DataFrame, rules: list[dict]) -> tuple[pd.DataFrame, dict]:
    """对候选截面 df 应用确定性**买入门** (低估才留), 返回 (过滤后 df, report)。

    多规则叠加用 AND (票必须通过所有规则); **无保底回退** —— "只买低估"是规则本意, 候选池少
    是特性不是 bug (方案3 的 min_keep 保底把高估票放回, 正是过滤器失效的根因)。
    缺列/规则不可用 → warn 跳过该规则, 不报错、不模拟。纯确定性布尔运算, 无 LLM, 不碰 risk_level。
    """
    applicable = [r for r in rules if r.get("applicable")]
    report = {
        "enabled": bool(applicable),
        "total_rules": len(rules),
        "applied": len(applicable),
        "per_rule": [],
        "before": int(len(df)),
        "after": int(len(df)),
        "removed": 0,
    }
    if not applicable:
        for r in rules:
            report["per_rule"].append({"template": r.get("template"),
                                       "applied": False, "removed": 0, "kept": len(df),
                                       "reason": r.get("reason", "")})
        return df, report

    mask = pd.Series(True, index=df.index)
    for r in applicable:
        template = r["template"]
        fn = _BUY_GATE_FN.get(template)
        if fn is None:
            report["per_rule"].append({"template": template, "applied": True,
                                       "removed": 0, "kept": len(df),
                                       "reason": "无对应买入门谓词实现"})
            continue
        m = fn(df, r["params"])
        if m is None:
            report["per_rule"].append({"template": template, "applied": True,
                                       "removed": 0, "kept": len(df),
                                       "reason": "缺所需列, 跳过"})
            continue
        kept = int(m.sum())
        report["per_rule"].append({"template": template, "applied": True,
                                   "removed": int((~m).sum()), "kept": kept,
                                   "reason": ""})
        mask &= m

    filtered = df[mask]
    report["after"] = int(len(filtered))
    report["removed"] = int(len(df) - len(filtered))
    logger.info(
        f"[知识买入门] 应用 {len(applicable)} 条 verified 规则: "
        f"{report['before']} -> {report['after']} (剔除 {report['removed']} 只)"
    )
    return filtered, report


def sell_signals_for_positions(df: pd.DataFrame, rules: list[dict]) -> pd.Series:
    """对持仓截面 df 返回**高估卖出信号** (布尔 Series, 索引与 df 一致)。

    多规则叠加用 OR (任一规则触发高估即卖)。无规则/缺 code 列 → 全 False (零开销)。
    纯确定性布尔运算, 无 LLM。
    """
    applicable = [r for r in rules if r.get("applicable")]
    mask = pd.Series(False, index=df.index)
    if not applicable:
        return mask
    for r in applicable:
        fn = _SELL_SIGNAL_FN.get(r["template"])
        if fn is None:
            continue
        m = fn(df, r["params"])
        if m is None:
            continue
        mask = mask | m
    return mask
