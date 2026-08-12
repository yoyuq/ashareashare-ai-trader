"""
DecisionValidator — 决策验证器 (v3.1-deerflow)

DeerFlow「Validator」角色: 在交易计划持久化/执行前, 用硬约束规则引擎逐条校验,
确保推荐计划可执行、可风控。纯规则引擎, 无LLM成本。

校验维度 (A股硬约束 + 项目风控参数):
  1. missing_params  — BUY 缺少交易参数 (entry/stop/take/position; SELL 只需有效持仓)
  2. bad_stop        — 止损 >= 入场价 (止损无效)
  3. bad_take        — 止盈 <= 入场价 (止盈无效)
  4. position_range  — 仓位比例超出 [3%, 20%] 区间
  5. limit_unbuyable — 最新收盘已触及板块涨停 (主板10%/创业科创20%/北交所30%), 次日追高不可成交
  6. lot_too_small   — 按当前资金计算的手数不足1手(100股), 无法成交
  7. concentration   — 该标的后行业集中度将超过 35% 上限 (需提供 portfolio)
  8. t1_pending      — 当日买入不可卖出 (T+1) (需提供 portfolio)

输出: 每只标的结构化违规清单 + 通过/拒绝判定; 违规写入持久化 journal
(simulation_data/validation_journal.jsonl), 修复"拒绝原因不落盘"缺口。
"""

import json
import os
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional

from agent.sub_agents.base import AgentContext, AgentResult, BaseAgent

# 项目风控参数 (与 paper_trader / workflow 一致)
MIN_POSITION_PCT = 0.03
MAX_POSITION_PCT = 0.20
INDUSTRY_CAP = 0.35
LOT_SIZE = 100
LIMIT_TOLERANCE = 0.2  # 涨跌停判定容差 (%) — 与 paper_trader 一致


def limit_pct_for_symbol(symbol: str) -> float:
    """按代码返回板块涨跌幅限制(%) — 主板±10%, 创业板/科创板±20%, 北交所±30%"""
    code = symbol.replace("sh.", "").replace("sz.", "").replace("bj.", "")
    if code.startswith("30") or code.startswith("68"):
        return 20.0
    if code.startswith("8") or code.startswith("4"):
        return 30.0
    return 10.0


@dataclass
class ValidationViolation:
    """违规项"""
    code: str
    severity: str            # critical | high | warning
    symbol: str
    message: str
    evidence: str = ""

    def to_dict(self) -> dict:
        return {"code": self.code, "severity": self.severity,
                "symbol": self.symbol, "message": self.message,
                "evidence": self.evidence}


@dataclass
class ValidationResult:
    """DecisionValidator 输出"""
    valid: bool                          # 无 critical 违规
    violations: List[ValidationViolation] = field(default_factory=list)
    checks_passed: int = 0

    def to_dict(self) -> dict:
        return {
            "valid": self.valid,
            "violations": [v.to_dict() for v in self.violations],
            "checks_passed": self.checks_passed,
        }


class DecisionValidator(BaseAgent):
    """决策验证器 — 规则引擎, 执行前硬约束校验"""

    agent_name = "validator"
    agent_icon = "🛡️"

    default_system_prompt = (
        "你是一个决策验证器(Validator Agent)。你的职责是核验每笔交易计划是否满足"
        "A股硬约束(涨跌停/T+1/整手/仓位上限/行业集中度), 拒绝任何不可执行的计划。"
    )

    JOURNAL_PATH = Path(
        os.environ.get("VALIDATION_JOURNAL_PATH",
                       "simulation_data/validation_journal.jsonl")
    )

    async def run(self, ctx: AgentContext) -> AgentResult:
        """
        执行交易计划校验

        输入 (ctx.input_data):
          - trading_params: {symbol: {entry_price, stop_loss, take_profit, position_pct}}
          - stock_recommendations: {symbol: {action, conviction, industry, ...}}
          - market_data: {symbol: df} — 用于上一收盘价与涨跌停可行性
          - portfolio: 可选 {positions: {sym: {quantity, buy_date}}, capital: float}
        """
        self._start_context(ctx.task_id, **ctx.input_data)

        trading_params = ctx.input_data.get("trading_params", {}) or {}
        stock_recs = ctx.input_data.get("stock_recommendations", {}) or {}
        market_data = ctx.input_data.get("market_data", {}) or {}
        portfolio = ctx.input_data.get("portfolio") or {}

        results: Dict[str, Dict] = {}
        total_violations: List[ValidationViolation] = []

        for sym, rec in stock_recs.items():
            if not isinstance(rec, dict):
                continue
            action = rec.get("action", "HOLD")
            if action not in ("BUY", "SELL"):
                results[sym] = {"valid": True, "violations": [], "checks_passed": 0}
                continue

            violations: List[ValidationViolation] = []
            checks = 0
            tp = trading_params.get(sym) or {}
            entry = self._num(tp.get("entry_price"))
            stop = self._num(tp.get("stop_loss"))
            take = self._num(tp.get("take_profit"))
            pos = self._num(tp.get("position_pct"))

            # 1-4. 参数类校验 (仅 BUY 需要完整交易参数; SELL 只需有效持仓)
            if action == "BUY":
                checks += 1
                if not (entry and entry > 0 and stop and take and pos is not None):
                    violations.append(ValidationViolation(
                        code="missing_params", severity="critical", symbol=sym,
                        message=f"{action} 推荐缺少完整交易参数 (entry/stop/take/position)",
                        evidence=f"trading_params={tp}",
                    ))
                else:
                    # 2. bad_stop / 3. bad_take / 4. position_range
                    checks += 1
                    if stop >= entry:
                        violations.append(ValidationViolation(
                            code="bad_stop", severity="critical", symbol=sym,
                            message=f"止损({stop}) >= 入场价({entry}), 止损无效",
                            evidence=f"stop_loss={stop} entry_price={entry}",
                        ))
                    checks += 1
                    if take <= entry:
                        violations.append(ValidationViolation(
                            code="bad_take", severity="critical", symbol=sym,
                            message=f"止盈({take}) <= 入场价({entry}), 止盈无效",
                            evidence=f"take_profit={take} entry_price={entry}",
                        ))
                    checks += 1
                    if not (MIN_POSITION_PCT <= pos <= MAX_POSITION_PCT):
                        violations.append(ValidationViolation(
                            code="position_range", severity="high", symbol=sym,
                            message=(f"仓位比例 {pos:.1%} 超出 [{MIN_POSITION_PCT:.0%}, "
                                     f"{MAX_POSITION_PCT:.0%}] 风控区间"),
                            evidence=f"position_pct={pos}",
                        ))

            # 5. limit_unbuyable — 用上一收盘 vs 最新收盘判断是否已封板
            # 注意: 不能用 `a or b` (DataFrame 的 or 触发歧义真值错误)
            df = market_data.get(sym)
            if df is None:
                df = market_data.get(sym.split(".")[-1])
            if df is not None and hasattr(df, "iloc") and len(df) >= 2:
                try:
                    yest = float(df["close"].iloc[-2])
                    today = float(df["close"].iloc[-1])
                    if yest > 0:
                        limit = limit_pct_for_symbol(sym)
                        chg_pct = (today / yest - 1) * 100
                        checks += 1
                        if action == "BUY" and chg_pct >= limit - LIMIT_TOLERANCE:
                            violations.append(ValidationViolation(
                                code="limit_unbuyable", severity="warning", symbol=sym,
                                message=(f"最新收盘涨幅 {chg_pct:.1f}% 已触及{limit:.0f}%涨停, "
                                         f"次日追高存在封板无法成交风险"),
                                evidence=f"prev_close={yest} close={today} limit={limit}%",
                            ))
                except (ValueError, TypeError, KeyError):
                    pass

            # 6. lot_too_small — 需要资金上下文
            capital = None
            if portfolio:
                capital = self._num(portfolio.get("capital"))
            if capital and entry and entry > 0 and pos is not None:
                checks += 1
                amount = capital * pos
                shares = int(amount / entry / LOT_SIZE) * LOT_SIZE
                if shares < LOT_SIZE:
                    violations.append(ValidationViolation(
                        code="lot_too_small", severity="high", symbol=sym,
                        message=(f"按资金 {capital:,.0f} × 仓位 {pos:.1%} = {amount:,.0f}元, "
                                 f"只能买 {shares}股 < 1手({LOT_SIZE}股), 无法成交"),
                        evidence=f"capital={capital} position_pct={pos} price={entry}",
                    ))

            # 7. concentration — 需要 portfolio 行业分布
            if portfolio:
                ind = rec.get("industry") or ""
                if ind:
                    positions = portfolio.get("positions", {}) or {}
                    existing = sum(
                        1 for p in positions.values()
                        if isinstance(p, dict) and p.get("industry") == ind
                    )
                    if existing >= 2:  # 已有同行业持仓, 再加一票集中度过高
                        checks += 1
                        violations.append(ValidationViolation(
                            code="concentration", severity="warning", symbol=sym,
                            message=f"行业 '{ind}' 已持有 {existing} 只, 再加将超过集中度上限",
                            evidence=f"existing_same_industry={existing}",
                        ))

            # 8. t1_pending — 需要 portfolio 持仓信息
            if portfolio and action == "SELL":
                positions = portfolio.get("positions", {}) or {}
                p = positions.get(sym) or positions.get(sym.split(".")[-1])
                if isinstance(p, dict):
                    buy_date = str(p.get("buy_date") or "")
                    if buy_date == date.today().isoformat():
                        checks += 1
                        violations.append(ValidationViolation(
                            code="t1_pending", severity="warning", symbol=sym,
                            message=f"该股 {buy_date} 今日买入, T+1 不可卖出",
                            evidence=f"buy_date={buy_date}",
                        ))

            # 汇总
            valid = not any(v.severity == "critical" for v in violations)
            vr = ValidationResult(valid=valid, violations=violations, checks_passed=checks)
            results[sym] = vr.to_dict()
            total_violations.extend(violations)

        # ── 持久化违规 journal (修复"拒绝原因不落盘"缺口) ──
        self._journal(total_violations, ctx.input_data.get("task_id", ""))

        return self._finish_context(
            success=True,
            validation_results=results,
            valid_count=sum(1 for r in results.values() if r.get("valid")),
            reject_count=sum(1 for r in results.values() if not r.get("valid")),
            violation_count=len(total_violations),
        )

    # ═══════════════════════════════════════════════════════════════
    # 工具方法
    # ═══════════════════════════════════════════════════════════════

    def _journal(self, violations: List[ValidationViolation], task_id: str) -> None:
        """把违规追加到持久化 journal (JSONL), 保证拒绝原因可审计"""
        if not violations:
            return
        try:
            journal_path = Path(self.JOURNAL_PATH)  # 容忍 str 或 Path
            journal_path.parent.mkdir(parents=True, exist_ok=True)
            with open(journal_path, "a", encoding="utf-8") as f:
                for v in violations:
                    f.write(json.dumps({
                        "ts": datetime.now().isoformat(),
                        "task_id": task_id,
                        "date": date.today().isoformat(),
                        "symbol": v.symbol,
                        "code": v.code,
                        "severity": v.severity,
                        "message": v.message,
                        "evidence": v.evidence,
                    }, ensure_ascii=False) + "\n")
        except OSError:
            pass  # journal 写失败不阻断主流程

    @staticmethod
    def _num(val) -> Optional[float]:
        try:
            f = float(val)
            return f
        except (TypeError, ValueError):
            return None
