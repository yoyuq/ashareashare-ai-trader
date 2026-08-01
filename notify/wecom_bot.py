"""
企业微信消息机器人 (v1.0)

支持:
  1. 命令交互: /持仓 /行情 /分析 600519 /回测 /策略 /帮助
  2. 定时推送: 每日开盘前/收盘后摘要
  3. 风险告警: 触发L1-L8风控时自动推送

用法:
  from notify.wecom_bot import WeComBot
  bot = WeComBot()
  reply = await bot.handle_message("持仓")
  bot.push_daily_summary()
"""

import asyncio
import json
import os
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))


class WeComBot:
    """企业微信机器人 — 消息处理 + 主动推送"""

    def __init__(self):
        self.webhook = os.getenv("WECOM_WEBHOOK", "")
        self.enabled = bool(self.webhook and "xxx" not in self.webhook)
        self.api_base = os.getenv("API_BASE_URL", "http://localhost:8000")

    # ═══════════════════════════════════════════════════════════════
    # 消息处理
    # ═══════════════════════════════════════════════════════════════

    async def handle_message(self, text: str) -> str:
        """处理用户消息, 返回回复文本 (Markdown格式)"""
        text = text.strip()
        cmd = text.split()[0].lower() if text else ""

        handlers = {
            "/持仓": self._cmd_portfolio,
            "/portfolio": self._cmd_portfolio,
            "/行情": self._cmd_market,
            "/market": self._cmd_market,
            "/分析": self._cmd_analyze,
            "/analyze": self._cmd_analyze,
            "/回测": self._cmd_backtest,
            "/bt": self._cmd_backtest,
            "/策略": self._cmd_strategies,
            "/strategy": self._cmd_strategies,
            "/帮助": self._cmd_help,
            "/help": self._cmd_help,
        }

        if cmd in handlers:
            return await handlers[cmd](text)
        # 默认: 交给 AI ChatAgent
        return await self._cmd_chat(text)

    # ═══════════════════════════════════════════════════════════════
    # 命令实现
    # ═══════════════════════════════════════════════════════════════

    async def _cmd_portfolio(self, _: str) -> str:
        """查询当前持仓"""
        try:
            r = requests.get(f"{self.api_base}/api/v1/portfolio/mtm", timeout=5)
            d = r.json()
            s = d.get("summary", {})
            positions = d.get("positions", [])

            lines = [
                "## 模拟持仓",
                f"总资产: ¥{s.get('total_value',0):,.0f}  |  "
                f"现金: ¥{s.get('cash',0):,.0f}  |  "
                f"收益: {s.get('total_return_pct',0):+.2f}%",
                f"持仓: {s.get('position_count',0)}只  |  "
                f"盈利: {s.get('win_count',0)} / 亏损: {s.get('loss_count',0)}",
                "",
            ]
            for p in positions:
                pnl = p.get("unrealized_pnl", 0)
                emoji = "🟢" if pnl > 0 else ("🔴" if pnl < 0 else "⚪")
                lines.append(
                    f"{emoji} **{p['name']}** ({p['symbol'].replace('sh.','').replace('sz.','')})  "
                    f"¥{p['current_price']:.2f}  |  "
                    f"{p['quantity']}股  |  "
                    f"盈亏 {pnl:+.0f} ({p.get('unrealized_pnl_pct',0):+.1f}%)"
                )
            return "\n".join(lines)
        except Exception as e:
            return f"持仓查询失败: {e}"

    async def _cmd_market(self, _: str) -> str:
        """实时行情"""
        try:
            r = requests.get(f"{self.api_base}/api/v1/realtime/market", timeout=5)
            d = r.json()
            indices = d.get("indices", {})
            ms = d.get("market_status", {})

            lines = [
                f"## 实时行情 ({ms.get('detail','未知')})",
                "",
            ]
            for name, idx in indices.items():
                pct = idx.get("pct", 0)
                emoji = "🔴" if pct > 0 else ("🟢" if pct < 0 else "⚪")
                lines.append(f"{emoji} **{name}**: {idx.get('price',0):.2f}  {pct:+.2f}%")

            # Top3
            top_up = d.get("top_up", [])[:3]
            top_down = d.get("top_down", [])[:3]
            if top_up:
                lines.append("\n📈 涨幅前三:")
                for t in top_up:
                    lines.append(f"  {t.get('name','')}({t.get('code','')}) {t.get('pct',0):+.2f}%")
            if top_down:
                lines.append("\n📉 跌幅前三:")
                for t in top_down:
                    lines.append(f"  {t.get('name','')}({t.get('code','')}) {t.get('pct',0):+.2f}%")

            return "\n".join(lines)
        except Exception as e:
            return f"行情查询失败: {e}"

    async def _cmd_analyze(self, text: str) -> str:
        """单股分析: /分析 600519"""
        parts = text.split()
        code = parts[1] if len(parts) > 1 else ""
        if not code:
            return "用法: /分析 <股票代码>  例: /分析 600519"

        # 补全格式
        if code.startswith("6"): sym = f"sh.{code}"
        elif code.startswith(("0","3")): sym = f"sz.{code}"
        else: sym = code

        try:
            r = requests.get(f"{self.api_base}/api/v1/stock/{sym}", timeout=10)
            if r.status_code == 404:
                return f"未找到股票: {code}"
            d = r.json()
            info = d.get("info", d)
            return (
                f"## {info.get('name', code)}({code})\n\n"
                f"最新价: ¥{info.get('price', info.get('close', 0)):.2f}\n"
                f"涨跌幅: {info.get('pct_change', info.get('change', 0)):+.2f}%\n"
                f"RSI(14): {info.get('rsi_14', 'N/A')}\n"
                f"趋势分: {info.get('trend_score', 'N/A')}\n"
            )
        except Exception as e:
            return f"分析失败: {e}"

    async def _cmd_backtest(self, text: str) -> str:
        """查看策略回测排名"""
        try:
            bt_path = Path(__file__).parent.parent / "reports" / "strategy_backtest.json"
            if not bt_path.exists():
                return "暂无回测数据。运行: python -m backtest.strategy_backtest"

            with open(bt_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            ranked = sorted(data["results"].items(),
                           key=lambda x: x[1].get("sharpe", -999), reverse=True)

            lines = ["## 策略回测排名", ""]
            for i, (sid, r) in enumerate(ranked[:8], 1):
                risk = r.get("overfit_risk", "?")
                risk_icon = {"low": "✅", "medium": "⚡", "high": "⚠️"}.get(risk, "?")
                lines.append(
                    f"{i}. {risk_icon} **{r['name']}**  "
                    f"胜率{r['win_rate']:.0%} 夏普{r['sharpe']:+.1f}  "
                    f"验证SR{r.get('val_sr',0):+.1f}"
                )
            return "\n".join(lines)
        except Exception as e:
            return f"回测查询失败: {e}"

    async def _cmd_strategies(self, _: str) -> str:
        """列出可用策略"""
        try:
            from knowledge.manager import KnowledgeManager
            km = KnowledgeManager()
            strats = km.list_strategies()
            lines = ["## 可用策略", ""]
            for s in strats[:10]:
                lines.append(f"- **{s['name']}** ({s['id']}) — {s.get('category','')}")
            return "\n".join(lines)
        except Exception as e:
            return f"策略查询失败: {e}"

    async def _cmd_chat(self, text: str) -> str:
        """通用 AI 对话"""
        try:
            r = requests.post(
                f"{self.api_base}/api/v1/chat",
                json={"message": text},
                timeout=30,
            )
            d = r.json()
            return d.get("reply", d.get("response", "AI 未返回内容"))[:2000]
        except Exception:
            return f"收到问题: '{text[:100]}'\n\nAI 服务暂不可用, 请稍后重试。支持的命令: /持仓 /行情 /分析 /回测 /策略 /帮助"

    async def _cmd_help(self, _: str) -> str:
        return (
            "## A股AI助手 — 命令列表\n\n"
            "**查询类**\n"
            "- `/持仓` — 查看模拟持仓和盈亏\n"
            "- `/行情` — 实时大盘指数和涨跌榜\n"
            "- `/分析 <代码>` — 单股技术面分析 (例: `/分析 600519`)\n\n"
            "**策略类**\n"
            "- `/回测` — 策略历史回测排名\n"
            "- `/策略` — 可用策略列表\n\n"
            "**其他**\n"
            "- 直接输入问题: AI 对话回答\n"
            "- `/帮助` — 显示此帮助"
        )

    # ═══════════════════════════════════════════════════════════════
    # 主动推送
    # ═══════════════════════════════════════════════════════════════

    def push_markdown(self, content: str) -> bool:
        """推送 Markdown 消息到企业微信"""
        if not self.enabled:
            return False
        try:
            r = requests.post(
                self.webhook,
                json={"msgtype": "markdown", "markdown": {"content": content}},
                timeout=10,
            )
            ok = r.status_code == 200 and r.json().get("errcode") == 0
            if ok:
                logger.info("WeCom 推送成功")
            else:
                logger.warning(f"WeCom 推送失败: {r.text[:200]}")
            return ok
        except Exception as e:
            logger.error(f"WeCom 推送异常: {e}")
            return False

    def push_daily_summary(self) -> bool:
        """推送每日摘要"""
        today_str = date.today().isoformat()
        try:
            # 持仓摘要
            r = requests.get(f"{self.api_base}/api/v1/portfolio/mtm", timeout=5)
            pf = r.json().get("summary", {})
            positions = r.json().get("positions", [])

            # 市场状态
            r2 = requests.get(f"{self.api_base}/api/v1/realtime/market", timeout=5)
            rt = r2.json()
            ms = rt.get("market_status", {})

            lines = [
                f"## A股AI日报 {today_str}",
                f"市场: {ms.get('detail','未知')}",
                f"总资产: ¥{pf.get('total_value',0):,.0f}  |  "
                f"收益: {pf.get('total_return_pct',0):+.2f}%  |  "
                f"持仓: {pf.get('position_count',0)}只",
                "",
            ]
            if positions:
                for p in positions[:6]:
                    pnl = p.get("unrealized_pnl", 0)
                    emoji = "🔴" if pnl > 0 else "🟢"
                    lines.append(
                        f"{emoji} {p['name']}: ¥{p['current_price']:.2f}  "
                        f"盈亏{pnl:+.0f} ({p.get('unrealized_pnl_pct',0):+.1f}%)"
                    )

            # 指数
            indices = rt.get("indices", {})
            idx_line = " | ".join(
                f"{n}: {v.get('pct',0):+.2f}%"
                for n, v in list(indices.items())[:5]
            )
            lines.insert(3, f"指数: {idx_line}")

            return self.push_markdown("\n".join(lines))
        except Exception as e:
            logger.error(f"推送日报失败: {e}")
            return False

    def push_alert(self, title: str, detail: str, level: str = "warning") -> bool:
        """推送风险告警"""
        colors = {"critical": "warning", "warning": "comment", "info": "info"}
        color = colors.get(level, "info")
        return self.push_markdown(
            f"## 🚨 {title}\n"
            f"> {detail}\n\n"
            f"时间: {datetime.now().strftime('%H:%M:%S')}  |  级别: <font color=\"{color}\">{level}</font>"
        )


# ═══════════════════════════════════════════════════════════════
# 命令行测试
# ═══════════════════════════════════════════════════════════════

async def _test():
    bot = WeComBot()
    print(f"Bot enabled: {bot.enabled}")

    # 测试命令
    for cmd in ["/行情", "/持仓", "/回测", "/帮助"]:
        reply = await bot.handle_message(cmd)
        print(f"\n>>> {cmd}\n{reply[:300]}...")


if __name__ == "__main__":
    asyncio.run(_test())
