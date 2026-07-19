"""
模型成本监控与告警 (v2.0+)

功能:
  - 实时追踪各层模型的API调用成本
  - 日/月预算告警(达到80%→告警, 100%→切断非本地调用)
  - 高峰时段自动降级策略
  - 成本优化建议
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from loguru import logger


@dataclass
class CostAlert:
    """成本告警"""
    level: str  # info / warning / critical
    message: str
    timestamp: datetime = field(default_factory=datetime.now)
    metrics: Dict[str, float] = field(default_factory=dict)


class CostMonitor:
    """
    成本监控器

    使用方式:
        monitor = CostMonitor(daily_budget=1.0, monthly_budget=15.0)
        monitor.record_call(tier="flash", input_tokens=5000, output_tokens=1000)

        if monitor.is_budget_exhausted():
            # 强制切换到本地模型
            ...
    """

    def __init__(
        self,
        daily_budget: float = 1.0,
        monthly_budget: float = 15.0,
        alert_threshold: float = 0.8,
    ):
        self.daily_budget = daily_budget
        self.monthly_budget = monthly_budget
        self.alert_threshold = alert_threshold

        self._daily_cost = 0.0
        self._monthly_cost = 0.0
        self._reset_date = datetime.now().date()
        self._call_history: List[Dict] = []
        self._alerts: List[CostAlert] = []

        # DeepSeek V4 定价 (2026-07, ¥/M tokens)
        self._pricing = {
            "flash": {"input": 1.0, "output": 2.0},
            "pro": {"input": 3.0, "output": 6.0},
        }

    # ═══════════════════════════════════════════════════════════════
    # 成本记录
    # ═══════════════════════════════════════════════════════════════

    def record_call(
        self,
        tier: str,
        input_tokens: int,
        output_tokens: int,
        is_peak: bool = False,
        task_type: str = "",
    ):
        """记录一次API调用"""
        cost = self._compute_cost(tier, input_tokens, output_tokens, is_peak)

        # 跨日重置
        today = datetime.now().date()
        if today != self._reset_date:
            self._daily_cost = 0.0
            self._reset_date = today

        self._daily_cost += cost
        self._monthly_cost += cost

        self._call_history.append({
            "timestamp": datetime.now(),
            "tier": tier,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost": cost,
            "is_peak": is_peak,
            "task_type": task_type,
        })

        # 检查告警
        self._check_alerts(cost)

    def _compute_cost(
        self,
        tier: str,
        input_tokens: int,
        output_tokens: int,
        is_peak: bool,
    ) -> float:
        """计算成本"""
        if tier == "local":
            return 0.0

        pricing = self._pricing.get(tier, {"input": 0, "output": 0})
        input_cost = (input_tokens / 1_000_000) * pricing["input"]
        output_cost = (output_tokens / 1_000_000) * pricing["output"]
        cost = input_cost + output_cost

        if is_peak:
            cost *= 2.0

        return cost

    # ═══════════════════════════════════════════════════════════════
    # 预算管理
    # ═══════════════════════════════════════════════════════════════

    def is_budget_exhausted(self) -> bool:
        """日预算是否耗尽"""
        return self._daily_cost >= self.daily_budget

    def is_near_limit(self) -> bool:
        """是否接近日预算上限(>80%)"""
        if self.daily_budget <= 0:
            return False
        return self._daily_cost >= self.daily_budget * self.alert_threshold

    @property
    def daily_remaining(self) -> float:
        return max(0, self.daily_budget - self._daily_cost)

    @property
    def monthly_remaining(self) -> float:
        return max(0, self.monthly_budget - self._monthly_cost)

    # ═══════════════════════════════════════════════════════════════
    # 告警
    # ═══════════════════════════════════════════════════════════════

    def _check_alerts(self, last_cost: float):
        """检查是否需要告警"""
        alerts = []

        # 日预算80%告警
        if not self.is_budget_exhausted() and self.is_near_limit():
            alerts.append(CostAlert(
                level="warning",
                message=(
                    f"日预算已使用 {self._daily_cost:.2f}/{self.daily_budget:.2f}元 "
                    f"({self._daily_cost/self.daily_budget*100:.0f}%)"
                ),
                metrics={
                    "daily_cost": self._daily_cost,
                    "daily_budget": self.daily_budget,
                    "usage_pct": self._daily_cost / self.daily_budget * 100,
                },
            ))

        # 日预算耗尽
        if self.is_budget_exhausted():
            alerts.append(CostAlert(
                level="critical",
                message=(
                    f"日预算已耗尽! {self._daily_cost:.2f}元,"
                    f"所有非本地API调用将被阻断"
                ),
                metrics={"daily_cost": self._daily_cost, "daily_budget": self.daily_budget},
            ))

        # 月预算告警
        if self._monthly_cost > self.monthly_budget * self.alert_threshold:
            alerts.append(CostAlert(
                level="warning",
                message=(
                    f"月预算已使用 {self._monthly_cost:.2f}/{self.monthly_budget:.2f}元 "
                    f"({self._monthly_cost/self.monthly_budget*100:.0f}%)"
                ),
            ))

        for alert in alerts:
            self._alerts.append(alert)
            if alert.level == "critical":
                logger.error(f"💰 {alert.message}")
            else:
                logger.warning(f"💰 {alert.message}")

    def get_alerts(
        self,
        since: Optional[datetime] = None,
    ) -> List[CostAlert]:
        """获取告警"""
        if since is None:
            return self._alerts
        return [a for a in self._alerts if a.timestamp >= since]

    # ═══════════════════════════════════════════════════════════════
    # 报表
    # ═══════════════════════════════════════════════════════════════

    def daily_report(self) -> Dict[str, Any]:
        """日成本报告"""
        today_calls = [
            c for c in self._call_history
            if c["timestamp"].date() == datetime.now().date()
        ]

        by_tier = {}
        for c in today_calls:
            tier = c["tier"]
            if tier not in by_tier:
                by_tier[tier] = {"count": 0, "cost": 0, "tokens": 0}
            by_tier[tier]["count"] += 1
            by_tier[tier]["cost"] += c["cost"]
            by_tier[tier]["tokens"] += c["input_tokens"] + c["output_tokens"]

        return {
            "date": str(datetime.now().date()),
            "daily_cost": round(self._daily_cost, 4),
            "daily_budget": self.daily_budget,
            "monthly_cost": round(self._monthly_cost, 4),
            "monthly_budget": self.monthly_budget,
            "total_calls_today": len(today_calls),
            "by_tier": {
                k: {
                    "count": v["count"],
                    "cost": round(v["cost"], 4),
                    "tokens": v["tokens"],
                }
                for k, v in by_tier.items()
            },
            "alerts_today": len([
                a for a in self._alerts
                if a.timestamp.date() == datetime.now().date()
            ]),
        }

    def monthly_report(self) -> Dict[str, Any]:
        """月成本报告"""
        # 按日汇总
        daily = {}
        for c in self._call_history:
            day = str(c["timestamp"].date())
            if day not in daily:
                daily[day] = 0
            daily[day] += c["cost"]

        return {
            "monthly_cost": round(self._monthly_cost, 4),
            "monthly_budget": self.monthly_budget,
            "total_calls": len(self._call_history),
            "daily_breakdown": daily,
            "avg_daily_cost": round(
                self._monthly_cost / max(len(daily), 1), 4
            ),
        }
