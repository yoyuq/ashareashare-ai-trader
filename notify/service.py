"""
Legacy NotificationService (v3.1-deerflow: 从废弃的 notifications/ 收敛到 notify/)

通道服务实现 — 供 scripts/run_daily_analysis.py 主分析管线使用。
推荐使用 notify.NotificationManager (更简洁的现代 API), 本模块保留
NotificationService/DailySummary 以兼容现有调用, 统一从 notify 包导入。

v3.1-deerflow: 原 notifications/ 包已删除, 本实现迁移到 notify/service.py。
"""

import asyncio
import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import requests
from loguru import logger


# ═══════════════════════════════════════════════════════════════
# 消息模板
# ═══════════════════════════════════════════════════════════════

@dataclass
class SignalAlert:
    """交易信号告警"""
    symbol: str
    symbol_name: str = ""
    strategy: str = ""
    direction: str = "long"
    confidence: str = "B"
    entry_price: float = 0
    stop_loss: float = 0
    take_profit: float = 0
    win_rate: float = 0
    key_reason: str = ""

    def to_markdown(self) -> str:
        return (
            f"## 📊 交易信号\n\n"
            f"**{self.symbol_name}({self.symbol})** — {self.strategy}\n"
            f"> {self.key_reason}\n\n"
            f"- 方向: {'🔴做多' if self.direction == 'long' else '🟢做空'}\n"
            f"- 入场: ¥{self.entry_price:.2f}\n"
            f"- 止损: ¥{self.stop_loss:.2f}\n"
            f"- 止盈: ¥{self.take_profit:.2f}\n"
            f"- 历史胜率: {self.win_rate:.0%}\n"
            f"- 信号评级: **{self.confidence}**\n"
            f"\n⚠️ 历史数据不代表未来表现, 仅供参考。"
        )


@dataclass
class DailySummary:
    """每日分析摘要"""
    date: str
    regime: str
    regime_confidence: float = 0
    total_scanned: int = 0
    recommendations_count: int = 0
    top_picks: List[Dict] = field(default_factory=list)
    debate_direction: str = "neutral"
    debate_confidence: float = 0
    cost_today: float = 0

    def to_markdown(self) -> str:
        regime_emoji = {
            "strong_bull": "🟢", "weak_bull": "🟡",
            "range_bound": "⚪", "weak_bear": "🟠",
            "strong_bear": "🔴", "crisis": "💀",
        }
        emoji = regime_emoji.get(self.regime, "❓")

        lines = [
            f"## 📋 A股每日分析报告",
            f"**{self.date}**",
            f"",
            f"### 市场状态",
            f"{emoji} **{self.regime}** (置信度: {self.regime_confidence:.0%})",
            f"",
            f"### 扫描结果",
            f"扫描: {self.total_scanned} → 过滤后: {self.recommendations_count} 个机会",
            f"多空方向: {self.debate_direction} | 置信度: {self.debate_confidence:.0%}",
            f"API费用: ¥{self.cost_today:.4f}",
        ]

        if self.top_picks:
            lines.append("\n### Top 推荐")
            for i, pick in enumerate(self.top_picks[:5]):
                lines.append(
                    f"{i+1}. **{pick.get('name', '')}**({pick.get('symbol', '')}) "
                    f"— {pick.get('strategy', '')} "
                    f"| 评分{pick.get('score', 0):.0f} "
                    f"| 胜率{pick.get('win_rate', 0):.0%}"
                )

        lines.append("\n⚠️ 以上为AI分析结果，不构成投资建议。")
        return "\n".join(lines)


@dataclass
class CostAlert:
    """费用告警"""
    daily_cost: float = 0
    daily_budget: float = 1.0
    monthly_cost: float = 0
    monthly_budget: float = 15.0

    def to_markdown(self) -> str:
        pct = self.daily_cost / self.daily_budget * 100
        return (
            f"## 💰 API费用告警\n\n"
            f"今日费用: ¥{self.daily_cost:.4f} / ¥{self.daily_budget:.2f} ({pct:.0f}%)\n"
            f"本月费用: ¥{self.monthly_cost:.2f} / ¥{self.monthly_budget:.2f}\n"
            f"\n⚠️ 日预算使用率超80%，已自动切换到本地模型。"
        )


# ═══════════════════════════════════════════════════════════════
# 通道实现
# ═══════════════════════════════════════════════════════════════

class ConsoleChannel:
    """Console 通道 — 通过 loguru 输出"""

    def send(self, content: str, title: str = ""):
        logger.info(f"[通知] {title}\n{content}")


class WeComChannel:
    """企业微信 Webhook 通道 (v2.9: 异步发送)"""

    def __init__(self, webhook_url: str = ""):
        self.webhook_url = webhook_url or os.getenv("WECOM_WEBHOOK", "")

    @property
    def enabled(self) -> bool:
        return bool(self.webhook_url) and "xxx" not in self.webhook_url

    async def send(self, content: str, title: str = "") -> bool:
        """异步发送 (避免阻塞 event loop)"""
        if not self.enabled:
            logger.debug("企业微信Webhook未配置, 跳过")
            return False

        def _post():
            payload = {
                "msgtype": "markdown",
                "markdown": {"content": content},
            }
            return requests.post(self.webhook_url, json=payload, timeout=10)

        try:
            resp = await asyncio.to_thread(_post)
            result = resp.json()
            if result.get("errcode") == 0:
                logger.info(f"[WeCom] 发送成功: {title}")
                return True
            else:
                logger.warning(f"[WeCom] 发送失败: {result}")
                return False
        except Exception as e:
            logger.warning(f"[WeCom] 发送异常: {e}")
            return False


class DingTalkChannel:
    """钉钉机器人通道 (v2.9: 异步发送)"""

    def __init__(self, webhook_url: str = ""):
        self.webhook_url = webhook_url or os.getenv("DINGTALK_WEBHOOK", "")

    @property
    def enabled(self) -> bool:
        return bool(self.webhook_url) and "xxx" not in self.webhook_url

    async def send(self, content: str, title: str = "") -> bool:
        """异步发送 (避免阻塞 event loop)"""
        if not self.enabled:
            logger.debug("钉钉Webhook未配置, 跳过")
            return False

        def _post():
            payload = {
                "msgtype": "markdown",
                "markdown": {
                    "title": title or "A股AI Agent",
                    "text": content,
                },
            }
            return requests.post(self.webhook_url, json=payload, timeout=10)

        try:
            resp = await asyncio.to_thread(_post)
            result = resp.json()
            if result.get("errcode") == 0:
                logger.info(f"[DingTalk] 发送成功: {title}")
                return True
            else:
                logger.warning(f"[DingTalk] 发送失败: {result}")
                return False
        except Exception as e:
            logger.warning(f"[DingTalk] 发送异常: {e}")
            return False


# ═══════════════════════════════════════════════════════════════
# 通知服务
# ═══════════════════════════════════════════════════════════════

class NotificationService:
    """
    通知服务 — 多通道分发 (v2.8, 迁移自 notifications/)

    配置来源: .env + config/settings.yaml

    使用方式:
        ns = NotificationService()
        await ns.send_signal_alert(alert)
        await ns.send_daily_summary(summary)
    """

    def __init__(self):
        self.console = ConsoleChannel()
        self.wecom = WeComChannel()
        self.dingtalk = DingTalkChannel()
        self._rate_limit: Dict[str, float] = {}  # 通道 → 上次发送时间

    @property
    def active_channels(self) -> List[str]:
        channels = ["console"]
        if self.wecom.enabled:
            channels.append("wecom")
        if self.dingtalk.enabled:
            channels.append("dingtalk")
        return channels

    def _should_send(self, channel: str, min_interval: float = 5.0) -> bool:
        """频率限制 — 同一通道最少间隔5秒"""
        now = time.time()
        last = self._rate_limit.get(channel, 0)
        if now - last < min_interval:
            return False
        self._rate_limit[channel] = now
        return True

    async def _broadcast(self, content: str, title: str = ""):
        """广播到所有活跃通道 (v2.9: 异步)"""
        results = {}
        # Console 总是发送
        self.console.send(content, title)
        results["console"] = True

        if self.wecom.enabled and self._should_send("wecom"):
            results["wecom"] = await self.wecom.send(content, title)
        if self.dingtalk.enabled and self._should_send("dingtalk"):
            results["dingtalk"] = await self.dingtalk.send(content, title)

        return results

    # ──── 便捷方法 ────

    async def send_signal_alert(
        self,
        symbol: str,
        strategy: str,
        confidence: str,
        symbol_name: str = "",
        direction: str = "long",
        entry_price: float = 0,
        stop_loss: float = 0,
        take_profit: float = 0,
        win_rate: float = 0,
        key_reason: str = "",
    ):
        """发送交易信号告警"""
        alert = SignalAlert(
            symbol=symbol,
            symbol_name=symbol_name,
            strategy=strategy,
            direction=direction,
            confidence=confidence,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            win_rate=win_rate,
            key_reason=key_reason,
        )
        await self._broadcast(alert.to_markdown(), f"交易信号: {symbol} {strategy}")

    async def send_daily_summary(self, summary: DailySummary):
        """发送每日分析摘要"""
        await self._broadcast(summary.to_markdown(), f"每日报告: {summary.date}")

    async def send_cost_alert(self, daily_cost: float, monthly_cost: float,
                              daily_budget: float = 1.0, monthly_budget: float = 15.0):
        """发送API费用告警"""
        alert = CostAlert(
            daily_cost=daily_cost,
            daily_budget=daily_budget,
            monthly_cost=monthly_cost,
            monthly_budget=monthly_budget,
        )
        await self._broadcast(alert.to_markdown(), "API费用告警")

    async def send_error_alert(self, error: str, source: str = ""):
        """发送系统异常告警"""
        content = (
            f"## 🚨 系统异常\n\n"
            f"**来源**: {source}\n"
            f"**时间**: {datetime.now().isoformat()}\n\n"
            f"```\n{error[:1000]}\n```"
        )
        await self._broadcast(content, f"异常: {source}")

    async def send_data_error(self, error: str, data_source: str = ""):
        """发送数据异常告警"""
        content = (
            f"## 📊 数据异常\n\n"
            f"**数据源**: {data_source}\n"
            f"**时间**: {datetime.now().isoformat()}\n\n"
            f"```\n{error[:800]}\n```"
        )
        await self._broadcast(content, f"数据异常: {data_source}")
