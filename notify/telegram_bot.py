"""
Telegram 个人聊天机器人 — 30秒部署, 永久在线

用法:
  1. Telegram 搜 @BotFather, 发 /newbot, 创建机器人, 获取 token
  2. .env 设置: TELEGRAM_BOT_TOKEN=xxx  TELEGRAM_CHAT_ID=你的chat_id
  3. python -m notify.telegram_bot
  4. 手机 Telegram 跟机器人聊天

特性:
  - /start 开始, /help 帮助
  - 持仓/行情/分析/回测 自动识别
  - 自然语言对话
  - 长轮询 (long polling), 不依赖 webhook
"""

import asyncio
import json
import os
import re
import sys
from datetime import date, datetime
from pathlib import Path

import requests
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))
from dotenv import load_dotenv; load_dotenv()


class TelegramBot:
    """Telegram 个人聊天机器人"""

    def __init__(self):
        self.token = os.getenv("TELEGRAM_BOT_TOKEN", "")
        self.api_base = os.getenv("API_BASE_URL", "http://localhost:8000")
        self.running = False

    @property
    def enabled(self) -> bool:
        return bool(self.token and "xxx" not in self.token)

    def _api(self, method: str, data: dict = None) -> dict:
        """调用 Telegram API"""
        url = f"https://api.telegram.org/bot{self.token}/{method}"
        try:
            r = requests.post(url, json=data, timeout=15) if data else requests.get(url, timeout=15)
            return r.json()
        except Exception as e:
            logger.error(f"Telegram API error: {e}")
            return {}

    def send_message(self, chat_id: int, text: str) -> bool:
        """发送消息 (自动截断长文本)"""
        if len(text) > 4000:
            text = text[:3900] + "\n\n...(已截断)"
        result = self._api("sendMessage", {
            "chat_id": chat_id,
            "text": text,
            "parse_mode": "Markdown",
        })
        if not result.get("ok"):
            # 降级: 不解析 Markdown
            result = self._api("sendMessage", {
                "chat_id": chat_id,
                "text": text,
            })
        return result.get("ok", False)

    async def handle_message(self, chat_id: int, text: str) -> str:
        """处理消息, 返回回复文本"""
        text = text.strip()

        # ── 命令 ──
        if text in ["/start", "hi", "你好", "在吗"]:
            return (
                "🤖 A股AI助手已上线!\n\n"
                "可以问我:\n"
                "• 持仓 / 盈亏 — 查看模拟持仓\n"
                "• 行情 / 大盘 — 实时市场数据\n"
                "• 回测 / 策略 — 策略排名\n"
                "• 分析 600519 — 单股技术面\n"
                "• 直接提问 — AI 对话\n\n"
                "发送 /help 查看更多命令"
            )

        if text in ["/help", "帮助"]:
            return (
                "📋 命令列表\n\n"
                "/持仓 — 模拟持仓 & 盈亏\n"
                "/行情 — 大盘指数 & 涨跌榜\n"
                "/分析 <代码> — 单股技术面\n"
                "/回测 — 策略回测排名\n"
                "/策略 — 可用策略列表\n"
                "/日报 — 今日摘要推送\n\n"
                "直接输入问题进行AI对话"
            )

        if text in ["/持仓", "持仓", "盈亏", "我的持仓"]:
            return self._get_portfolio()

        if text in ["/行情", "行情", "大盘", "今天行情"]:
            return self._get_market()

        if text in ["/回测", "回测", "策略排名"]:
            return self._get_backtest()

        if text in ["/策略", "策略"]:
            return self._get_strategies()

        if text in ["/日报", "日报"]:
            return self._get_daily_summary()

        if text.startswith("/分析 ") or text.startswith("分析 "):
            code = text.split()[-1] if text.split() else ""
            return self._get_stock_analysis(code)

        # ── 股票代码自动识别 ──
        code_match = re.search(r'(60\d{4}|00\d{4}|30\d{4}|688\d{3})', text)
        if code_match:
            return self._get_stock_analysis(code_match.group(1))

        # ── 语义路由 ──
        if any(w in text for w in ["赚", "亏", "收益", "持仓"]):
            return self._get_portfolio()
        if any(w in text for w in ["跌", "涨", "指数", "大盘"]):
            return self._get_market()

        # ── AI 对话 ──
        return self._get_ai_reply(text)

    # ═══════════════════════════════════════════════════════════════
    # 内部方法
    # ═══════════════════════════════════════════════════════════════

    def _get_portfolio(self) -> str:
        try:
            r = requests.get(f"{self.api_base}/api/v1/portfolio/mtm", timeout=5)
            d = r.json()
            s = d.get("summary", {})
            ps = d.get("positions", [])

            lines = [
                f"💰 总资产: ¥{s.get('total_value', 0):,.0f}",
                f"📊 收益: {s.get('total_return_pct', 0):+.2f}%",
                f"💵 现金: ¥{s.get('cash', 0):,.0f}",
                f"📦 持仓: {s.get('position_count', 0)}只",
                "",
            ]
            for p in ps:
                pnl = p.get("unrealized_pnl", 0)
                e = "🟢" if pnl > 0 else ("🔴" if pnl < 0 else "➖")
                lines.append(
                    f"{e} *{p['name']}* ¥{p['current_price']:.2f}  "
                    f"{pnl:+.0f} ({p.get('unrealized_pnl_pct', 0):+.1f}%)"
                )
            return "\n".join(lines)
        except Exception as e:
            return f"持仓查询失败: {e}"

    def _get_market(self) -> str:
        try:
            r = requests.get(f"{self.api_base}/api/v1/realtime/market", timeout=5)
            d = r.json()
            idx = d.get("indices", {})
            ms = d.get("market_status", {})

            lines = [f"📈 *{ms.get('detail', '未知')}*", ""]
            for n, v in idx.items():
                p = v.get("pct", 0)
                e = "🔴" if p > 0 else ("🟢" if p < 0 else "➖")
                lines.append(f"{e} *{n}*: {v.get('price', 0):.2f}  {p:+.2f}%")

            top = d.get("top_up", [])[:3]
            if top:
                lines.append("\n📈 涨幅前三:")
                for t in top:
                    lines.append(f"  {t.get('name','')} {t.get('pct',0):+.2f}%")

            return "\n".join(lines)
        except Exception as e:
            return f"行情查询失败: {e}"

    def _get_backtest(self) -> str:
        try:
            bp = Path(__file__).parent.parent / "reports" / "strategy_backtest.json"
            if not bp.exists():
                return "暂无回测数据。运行: python -m backtest.strategy_backtest"

            with open(bp, encoding="utf-8") as f:
                data = json.load(f)
            ranked = sorted(data["results"].items(),
                           key=lambda x: x[1].get("sharpe", -999), reverse=True)

            lines = ["🏆 *策略排名 (夏普比率)*", ""]
            of_icon = {"low": "✅", "medium": "⚡", "high": "⚠️"}
            for i, (sid, r) in enumerate(ranked[:6], 1):
                risk = r.get("overfit_risk", "?")
                lines.append(
                    f"{i}. {of_icon.get(risk,'?')} *{r['name']}*  "
                    f"胜率{r['win_rate']:.0%} 夏普{r['sharpe']:+.1f}"
                )
            return "\n".join(lines)
        except Exception as e:
            return f"回测查询失败: {e}"

    def _get_strategies(self) -> str:
        try:
            from knowledge.manager import KnowledgeManager
            km = KnowledgeManager()
            strats = km.list_strategies()
            lines = ["📚 *可用策略*", ""]
            for s in strats[:8]:
                lines.append(f"• *{s['name']}* ({s.get('category','')}) — {s.get('description','')[:30]}")
            return "\n".join(lines)
        except Exception as e:
            return f"策略查询失败: {e}"

    def _get_stock_analysis(self, code: str) -> str:
        if not code or not re.match(r'(60|00|30|688)\d{4,5}', code):
            return "请输入有效的A股代码, 如: 600519"

        prefix = "sh" if code.startswith("6") else "sz"
        try:
            r = requests.get(f"{self.api_base}/api/v1/stock/{prefix}.{code}", timeout=10)
            if r.status_code != 200:
                return f"未找到 {code} 的数据"

            d = r.json()
            info = d.get("info", d)
            return (
                f"📈 *{info.get('name', code)}* ({code})\n\n"
                f"最新价: ¥{info.get('price', info.get('close', 0)):.2f}\n"
                f"涨跌: {info.get('pct_change', info.get('change', 0)):+.2f}%\n"
                f"RSI(14): {info.get('rsi_14', 'N/A')}\n"
                f"趋势: {info.get('trend_score', 'N/A')}"
            )
        except Exception as e:
            return f"分析失败: {e}"

    def _get_daily_summary(self) -> str:
        try:
            pf = self._get_portfolio()
            mk = self._get_market()
            return f"📅 *{date.today().isoformat()} 日报*\n\n{mk}\n\n---\n\n{pf}"
        except:
            return "日报生成失败"

    def _get_ai_reply(self, text: str) -> str:
        try:
            r = requests.post(
                f"{self.api_base}/api/v1/chat",
                json={"message": text},
                timeout=30,
            )
            d = r.json()
            return d.get("reply", "")[:2000] or "收到你的问题, AI 暂时无法回答"
        except:
            # Fallback 规则
            return (
                "收到你的问题。AI 服务暂不可用, 请使用命令:\n"
                "持仓 | 行情 | 回测 | 分析 <代码>"
            )

    # ═══════════════════════════════════════════════════════════════
    # 主循环
    # ═══════════════════════════════════════════════════════════════

    async def run(self):
        """长轮询主循环"""
        if not self.enabled:
            logger.error(
                "Telegram Bot Token 未配置!\n"
                "1. Telegram 搜 @BotFather, 发 /newbot\n"
                "2. 获取 token\n"
                "3. 在 .env 中设置 TELEGRAM_BOT_TOKEN=你的token"
            )
            return

        logger.info("Telegram 机器人启动...")
        self.running = True
        offset = 0

        while self.running:
            try:
                updates = self._api("getUpdates", {
                    "offset": offset,
                    "timeout": 30,
                    "allowed_updates": ["message"],
                })

                if updates.get("ok"):
                    for upd in updates.get("result", []):
                        offset = upd["update_id"] + 1
                        msg = upd.get("message", {})
                        chat_id = msg.get("chat", {}).get("id")
                        text = msg.get("text", "")

                        if chat_id and text:
                            logger.info(f"[{msg.get('chat',{}).get('username','?')}] {text[:80]}")
                            reply = await self.handle_message(chat_id, text)
                            self.send_message(chat_id, reply)

            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"轮询异常: {e}")
                await asyncio.sleep(5)

        logger.info("Telegram 机器人已停止")


async def main():
    bot = TelegramBot()
    await bot.run()


if __name__ == "__main__":
    asyncio.run(main())
