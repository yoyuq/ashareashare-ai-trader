"""
Notification System (v3.1)

Multi-channel notification dispatcher for WeChat Work, DingTalk, and Telegram.
Wires up the previously-unused webhook URLs from .env.

v3.1-deerflow: 废弃的 notifications/ 包已收敛到 notify/ 下 —
  - notify.NotificationManager: 现代简洁 API (推荐)
  - notify.NotificationService / DailySummary / SignalAlert: 旧通道服务
    (迁移自 notifications/, 供 scripts/run_daily_analysis.py 使用)

Usage:
    from notify import NotificationManager
    nm = NotificationManager()
    nm.send_alert("Risk limit triggered!", level="critical")
    nm.send_daily_summary(report_path="reports/report_2026-07-29.md")
"""

import json
import os
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from loguru import logger

# ── 旧通道服务 (收敛自 notifications/, 供主分析管线使用) ──
from notify.service import (  # noqa: E402, F401
    SignalAlert, DailySummary, CostAlert,
    NotificationService, ConsoleChannel, WeComChannel, DingTalkChannel,
)


class NotificationManager:
    """Multi-channel notification dispatcher"""

    def __init__(self):
        self.wecom_webhook = os.getenv("WECOM_WEBHOOK", "")
        self.dingtalk_webhook = os.getenv("DINGTALK_WEBHOOK", "")
        self.telegram_token = os.getenv("TELEGRAM_BOT_TOKEN", "")
        self.telegram_chat_id = os.getenv("TELEGRAM_CHAT_ID", "")
        self.enabled = any([
            self.wecom_webhook and "xxx" not in self.wecom_webhook,
            self.dingtalk_webhook and "xxx" not in self.dingtalk_webhook,
            self.telegram_token and "xxx" not in self.telegram_token,
        ])

    def send(self, message: str, title: str = "", level: str = "info") -> bool:
        """Send to all configured channels"""
        sent = False
        if self.wecom_webhook and "xxx" not in self.wecom_webhook:
            sent |= self._send_wecom(message, title, level)
        if self.dingtalk_webhook and "xxx" not in self.dingtalk_webhook:
            sent |= self._send_dingtalk(message, title, level)
        if self.telegram_token and "xxx" not in self.telegram_token:
            sent |= self._send_telegram(message)
        return sent

    def send_alert(
        self,
        message: str,
        level: str = "warning",
        channels: List[str] = None,
    ) -> bool:
        """Send risk alert notification"""
        prefix = {"critical": "[CRITICAL]", "warning": "[WARNING]", "info": "[INFO]"}.get(level, "[*]")
        full_msg = f"{prefix} {date.today().isoformat()}: {message}"
        return self.send(full_msg, title=f"A-Share AI Alert: {level}", level=level)

    def send_daily_summary(self, summary_text: str = None, report_path: str = None) -> bool:
        """Send daily analysis summary"""
        if report_path:
            try:
                text = Path(report_path).read_text(encoding="utf-8")[:2000]
            except Exception:
                text = summary_text or "Daily report generated"
        elif summary_text:
            text = summary_text[:2000]
        else:
            text = f"Daily analysis completed for {date.today().isoformat()}"

        return self.send(text, title=f"A-Share Daily Report - {date.today().isoformat()}")

    def send_signal_alert(self, symbol: str, action: str, confidence: float, reason: str = "") -> bool:
        """Send trading signal alert"""
        level = "critical" if confidence > 0.8 and action in ("BUY", "SELL") else "info"
        msg = f"Signal: {action} {symbol} (confidence={confidence:.0%})"
        if reason:
            msg += f" - {reason[:200]}"
        return self.send_alert(msg, level=level)

    def _send_wecom(self, message: str, title: str = "", level: str = "info") -> bool:
        """Send via WeChat Work webhook"""
        try:
            payload = {
                "msgtype": "markdown",
                "markdown": {
                    "content": f"## {title}\n\n{message}",
                },
            }
            resp = requests.post(self.wecom_webhook, json=payload, timeout=10)
            ok = resp.status_code == 200
            if not ok:
                logger.warning(f"WeChat notification failed: {resp.status_code}")
            return ok
        except Exception as e:
            logger.warning(f"WeChat notification error: {e}")
            return False

    def _send_dingtalk(self, message: str, title: str = "", level: str = "info") -> bool:
        """Send via DingTalk webhook"""
        try:
            payload = {
                "msgtype": "markdown",
                "markdown": {
                    "title": title or "A-Share Alert",
                    "text": f"## {title}\n\n{message}",
                },
            }
            resp = requests.post(self.dingtalk_webhook, json=payload, timeout=10)
            return resp.status_code == 200
        except Exception as e:
            logger.warning(f"DingTalk notification error: {e}")
            return False

    def _send_telegram(self, message: str) -> bool:
        """Send via Telegram bot"""
        try:
            url = f"https://api.telegram.org/bot{self.telegram_token}/sendMessage"
            payload = {
                "chat_id": self.telegram_chat_id,
                "text": message[:4096],
                "parse_mode": "HTML",
            }
            resp = requests.post(url, json=payload, timeout=10)
            return resp.status_code == 200
        except Exception as e:
            logger.warning(f"Telegram notification error: {e}")
            return False
