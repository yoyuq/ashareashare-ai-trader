"""组合级反事实验证 — 扔掉一只拖累票，组合真的会更好吗？

现有 `counterfactual.py::verify_counterfactual` 是**市场敞口级**反事实：
用 `risk_multiplier × 次日市场收益` 验证"风险等级/仓位系数调整"的价值。
它不回答个股问题 —— "如果那天没买某只票 / 改了某只仓位，组合整体会怎样"。

本模块补上**组合级（个股级）**反事实：
- 用当天持仓快照 + 每只持仓的次日涨跌幅，算每只票对组合次日收益的贡献
  （contribution = weight × day_return_pct）
- 识别"拖累最大"的票（贡献最负），反事实 = 移除它（资金转现金，收益 0）
- 对比实际 vs 反事实组合收益，判断"扔掉这只拖累票"是否真能改善组合

设计准则（与 counterfactual.py 一致）：
- **反事实天然是后视验证**（用已发生的次日结果评判），只用于置信度微调/学习信号，
  不做硬规则，不改变交易方向。
- 纯函数，无 IO，可单测。
- 组合次日收益 = Σ(weight_i × ret_i) + 现金部分(0)。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class StockContribution:
    """单只持仓对组合次日收益的贡献。"""
    symbol: str
    name: str = ""
    weight: float = 0.0          # 组合占比 (占总资产, 0~1)
    day_return_pct: float = 0.0  # 个股次日涨跌幅 (%)
    contribution_pct: float = 0.0  # 对组合次日收益的贡献 (%) = weight * day_return_pct

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol, "name": self.name,
            "weight": round(self.weight, 4),
            "day_return_pct": round(self.day_return_pct, 2),
            "contribution_pct": round(self.contribution_pct, 3),
        }


@dataclass
class PortfolioCounterfactualResult:
    """组合级反事实验证结果。"""
    date: str = ""
    portfolio_return_pct: float = 0.0      # 实际组合次日收益 (%)
    counterfactual_return_pct: float = 0.0  # 移除最差票后组合收益 (%)
    improvement_pct: float = 0.0           # 提升幅度 (反事实 - 实际, 正=变好)
    worst_stock: Optional[StockContribution] = None  # 拖累最大的票
    worst_contribution_pct: float = 0.0    # 最差票的贡献 (%)
    n_positions: int = 0
    n_match: int = 0                       # 有次日收益匹配到的持仓数
    verified: bool = False                 # 移除最差票是否显著改善组合

    def to_dict(self) -> dict:
        return {
            "date": self.date,
            "portfolio_return_pct": round(self.portfolio_return_pct, 3),
            "counterfactual_return_pct": round(self.counterfactual_return_pct, 3),
            "improvement_pct": round(self.improvement_pct, 3),
            "worst_stock": self.worst_stock.to_dict() if self.worst_stock else None,
            "worst_contribution_pct": round(self.worst_contribution_pct, 3),
            "n_positions": self.n_positions,
            "n_match": self.n_match,
            "verified": self.verified,
        }


# 波动率归一: 移除最差票的改善必须超过 max(IMPROVE_BASE, 组合波动*SCALE) 才算显著.
# 避免"最差票恰好小跌一下"就被当作该扔掉 (浅层后视).
IMPROVE_BASE = 0.05     # 基础阈值 (%)
VOL_SCALE = 0.15        # 组合|收益|的归一系数


def compute_stock_contributions(
    positions_snapshot: list,
    stock_returns: dict,
) -> list[StockContribution]:
    """对每只持仓计算次日贡献。

    Args:
        positions_snapshot: journal 的 positions_snapshot，每项含 symbol/name/value/weight。
        stock_returns: {symbol: 次日涨跌幅%}，T 日截面匹配 T-1 持仓。
    Returns:
        StockContribution 列表（只含匹配到次日收益的持仓）。
    """
    contribs: list[StockContribution] = []
    for p in positions_snapshot:
        sym = p.get("symbol")
        if not sym:
            continue
        ret = stock_returns.get(sym)
        if ret is None:
            continue
        weight = float(p.get("weight", 0.0) or 0.0)
        try:
            ret_f = float(ret)
        except (TypeError, ValueError):
            continue
        contribs.append(StockContribution(
            symbol=sym,
            name=str(p.get("name", "")),
            weight=weight,
            day_return_pct=ret_f,
            contribution_pct=weight * ret_f,
        ))
    return contribs


def portfolio_level_counterfactual(
    positions_snapshot: list,
    stock_returns: dict,
    date: str = "",
    improvement_threshold: float = IMPROVE_BASE,
) -> Optional[PortfolioCounterfactualResult]:
    """组合级反事实验证：移除当天拖累最大的持仓，组合收益是否显著改善。

    Args:
        positions_snapshot: 当日持仓快照 [{symbol, name, value, weight}]。
        stock_returns: {symbol: 次日涨跌幅%}。
        date: 决策日期。
        improvement_threshold: 提升阈值(%)，超过才算显著。

    Returns:
        PortfolioCounterfactualResult；无匹配持仓时返回 None。
    """
    contribs = compute_stock_contributions(positions_snapshot, stock_returns)
    if not contribs:
        return None

    # 实际组合次日收益 = Σ(weight_i × ret_i)（现金收益 0）
    actual = sum(c.contribution_pct for c in contribs)

    # 拖累最大的票
    worst = min(contribs, key=lambda c: c.contribution_pct)

    # 反事实：移除 worst（资金转现金，收益 0）→ 只保留其他票贡献
    cf = actual - worst.contribution_pct
    improvement = cf - actual  # = -worst.contribution_pct

    # 波动率归一阈值: 市场波动大时, 需要的改善门槛也更高 (避免小波动日浅层后视)
    _mag = abs(actual)
    _thr = max(improvement_threshold, _mag * VOL_SCALE)
    verified = improvement > _thr and worst.contribution_pct < 0

    return PortfolioCounterfactualResult(
        date=date,
        portfolio_return_pct=actual,
        counterfactual_return_pct=cf,
        improvement_pct=improvement,
        worst_stock=worst,
        worst_contribution_pct=worst.contribution_pct,
        n_positions=len(positions_snapshot),
        n_match=len(contribs),
        verified=verified,
    )


def summarize_verified(records: list[PortfolioCounterfactualResult]) -> dict:
    """把一组组合级反事实结果汇总成统计（供进化诊断/报告）。

    Returns:
        {
          'total': N, 'verified': M, 'verified_rate': 0.x,
          'avg_improvement_pct': 从最差票释放的均值,
          'worst_symbols': 被反复标记为拖累票的 symbol 频次 top-5,
        }
    """
    recs = [r for r in records if r is not None]
    n = len(recs)
    if n == 0:
        return {"total": 0, "verified": 0, "verified_rate": 0.0,
                "avg_improvement_pct": 0.0, "worst_symbols": []}
    verified = [r for r in recs if r.verified]
    from collections import Counter
    sym_freq = Counter()
    for r in recs:
        if r.worst_stock is not None and r.verified:
            sym_freq[r.worst_stock.symbol] += 1
    return {
        "total": n,
        "verified": len(verified),
        "verified_rate": round(len(verified) / n, 3),
        "avg_improvement_pct": round(sum(r.improvement_pct for r in recs) / n, 3),
        "worst_symbols": sym_freq.most_common(5),
    }


# ═══════════════════════════════════════════════════════════════
# v5.5 P1-4: 组合级反事实闭环 — 拖累票信号回流成记忆 (保守, 非硬规则)
# ═══════════════════════════════════════════════════════════════
# 此前 #26 只"记录不回流": verified/worst_symbols 附在 review + 统计, 从不影响后续决策.
# 这里把"反复被验证为拖累的票"累积成历史, 达到阈值后写入经验记忆, 经已有 memory 检索
# 注入诊断 prompt, 让 LLM 对该票降权/谨慎. 是学习信号, 不改交易方向, 不做硬规则.

REPEAT_FLAG_MIN = 2   # 同一票被验证为拖累 >= 2 次才写经验 (防单次偶然)


def accumulate_drag_history(history: dict, verified_results: list, current_date: str) -> dict:
    """把一组 verified 组合级反事实的拖累票计入历史计数 (纯函数, 可测).

    Args:
        history: dict {symbol: {'count': int, 'avg_improvement_pct': float, 'last_name': str}}
        verified_results: PortfolioCounterfactualResult 列表 (仅 verified 的计入).
        current_date: 本次复盘日期 (ISO).
    Returns:
        更新的 history (原地修改并返回).
    """
    for r in verified_results:
        if r is None or not r.verified or r.worst_stock is None:
            continue
        sym = r.worst_stock.symbol
        entry = history.get(sym) or {
            "count": 0, "avg_improvement_pct": 0.0,
            "last_name": r.worst_stock.name, "last_date": current_date,
        }
        entry["count"] += 1
        entry["last_date"] = current_date
        if r.worst_stock.name:
            entry["last_name"] = r.worst_stock.name
        # 累计平均: 用新均值 (count 次里含本次)
        entry["avg_improvement_pct"] = (
            (entry["avg_improvement_pct"] * (entry["count"] - 1) + r.improvement_pct)
            / entry["count"]
        )
        history[sym] = entry
    return history


def drag_experiences(
    history: dict,
    current_date: str,
    min_count: int = REPEAT_FLAG_MIN,
    master: str = "利弗莫尔",
) -> list:
    """把反复(>=min_count)被标记为拖累票的 symbol 转成经验条目 (供记忆注入).

    保守: 只有同票被验证 >= min_count 次才写, 单次偶然不写. 不改交易方向.
    """
    from .daily_review import ExperienceItem
    items = []
    for sym, e in history.items():
        if e.get("count", 0) < min_count:
            continue
        name = e.get("last_name", "") or sym
        count = e["count"]
        avg_imp = e.get("avg_improvement_pct", 0.0)
        items.append(ExperienceItem(
            id=f"pcf-{sym}-{current_date}",
            date=current_date,
            scenario_type="range",   # 组合拖累与市场阶段关系弱, 用中性场景避免误跨界惩罚
            verdict="wrong",
            lesson_title=f"组合拖累票: {sym}({name}) 反复被验证为拖累",
            lesson_detail=(
                f"该票在最近 {count} 次组合级反事实中被验证为拖累 (移除平均释放 {avg_imp:+.2f}pp), "
                f"建议对 {sym} 保持谨慎/降权, 避免高权重孤注一掷."
            ),
            master_used=master,
            risk_level_given=3,
            actual_outcome="移除该票后组合次日收益显著改善",
            confidence=min(0.9, 0.5 + 0.1 * count),
            tags=["组合拖累票", "portfolio_cf", "downweight"],
        ))
    return items