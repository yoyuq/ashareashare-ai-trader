"""
微信好友机器人 — 扫码登录后自动回复消息

原理: uiautomation 控制 Windows 微信客户端, 监听新消息并调用 AI 回复

用法:
  1. 电脑登录微信 (Windows客户端)
  2. python -m notify.wechat_friend
  3. 别人给你发微信, 机器人自动回复

首次使用: 运行后会提示你用手机微信给机器人发 "test", 确认窗口绑定成功
"""

import re
import sys
import time
import threading
from datetime import datetime
from pathlib import Path

import uiautomation as auto

sys.path.insert(0, str(Path(__file__).parent.parent))
from loguru import logger


class WeChatFriend:
    """微信好友机器人"""

    def __init__(self):
        self.wx = None
        self.running = False
        self.replied_msgs = set()  # 已回复消息ID, 防止重复
        self.auto_reply_enabled = True

    def connect(self) -> bool:
        """连接微信窗口"""
        try:
            self.wx = auto.WindowControl(Name="微信", searchDepth=1)
            if not self.wx.Exists(maxSearchSeconds=3):
                logger.error("未找到微信窗口, 请确认微信已登录")
                return False
            logger.info(f"已连接微信: {self.wx.Name}")
            return True
        except Exception as e:
            logger.error(f"连接微信失败: {e}")
            return False

    def get_unread_chats(self) -> list:
        """获取有未读消息的聊天列表"""
        try:
            session_list = self.wx.ListControl(Name="会话")
            if not session_list.Exists():
                return []

            unread = []
            for item in session_list.GetChildren():
                name = item.Name
                # 检查是否有未读标记(红点/数字)
                if self._has_unread(item):
                    unread.append({"name": name, "element": item})
            return unread
        except Exception:
            return []

    def _has_unread(self, item) -> bool:
        """检查会话是否有未读消息"""
        try:
            for child in item.GetChildren():
                # 未读数字或红点
                if child.ControlTypeName == "TextControl" and child.Name:
                    if child.Name.isdigit() or "条" in child.Name:
                        return True
            return False
        except:
            return False

    def get_last_message(self, chat_name: str) -> str | None:
        """获取指定聊天的最后一条消息"""
        try:
            # 点击进入聊天
            self.wx.EditControl(Name="搜索").Click()
            auto.SendKeys(chat_name)
            time.sleep(0.5)

            # 点击第一个搜索结果
            result = self.wx.ListItemControl(Name=chat_name)
            if result.Exists():
                result.Click()
                time.sleep(0.3)

            # 读取最后一条消息
            msg_list = self.wx.ListControl(Name="消息")
            if msg_list.Exists():
                items = msg_list.GetChildren()
                if items:
                    last = items[-1]
                    msg_text = last.Name
                    # 跳过自己发的消息
                    for child in last.GetChildren():
                        if "自己" in child.Name or self._is_self_sent(last):
                            return None
                    return msg_text
        except Exception as e:
            logger.debug(f"读取消息失败: {e}")
        return None

    def _is_self_sent(self, msg_element) -> bool:
        """判断消息是否是自己发的 (右侧绿色气泡)"""
        try:
            # 自己发的消息通常在右侧, 可以通过位置判断
            rect = msg_element.BoundingRectangle
            # 简单判断: 如果消息框的左边距大于窗口一半宽度, 大概率是自己发的
            wx_rect = self.wx.BoundingRectangle
            if rect.left > wx_rect.left + wx_rect.width() * 0.5:
                return True
        except:
            pass
        return False

    def send_message(self, chat_name: str, text: str) -> bool:
        """向指定聊天发送消息"""
        try:
            # 切换到该聊天
            self.wx.EditControl(Name="搜索").Click()
            time.sleep(0.2)
            auto.SendKeys(chat_name)
            time.sleep(0.5)

            result = self.wx.ListItemControl(Name=chat_name)
            if result.Exists():
                result.Click()
                time.sleep(0.3)

            # 定位输入框并输入
            input_box = self.wx.EditControl(Name="输入")
            if not input_box.Exists():
                # 备用: 点击窗口底部区域
                wx_rect = self.wx.BoundingRectangle
                auto.Click(wx_rect.left + 100, wx_rect.bottom - 50)
                time.sleep(0.2)

            # 清空并输入
            auto.SendKeys("{Ctrl}a")
            time.sleep(0.1)

            # 分段发送长消息
            lines = text.split("\n")
            for i, line in enumerate(lines):
                if i > 0:
                    auto.SendKeys("{Shift}{Enter}")  # 微信内换行
                    time.sleep(0.05)
                # 清理特殊字符
                clean = re.sub(r'[#*]', '', line)
                auto.SendKeys(clean)
                time.sleep(0.05)

            time.sleep(0.3)
            auto.SendKeys("{Enter}")  # 发送
            logger.info(f"已回复 {chat_name}: {text[:50]}...")
            return True
        except Exception as e:
            logger.error(f"发送消息失败: {e}")
            return False

    def get_ai_reply(self, text: str) -> str:
        """调用本地 AI 获取回复"""
        from notify.wecom_bot import WeComBot
        import asyncio

        try:
            bot = WeComBot()
            loop = asyncio.new_event_loop()
            reply = loop.run_until_complete(bot.handle_message(text))
            loop.close()
            return reply
        except Exception:
            # Fallback: 简单规则
            return self._fallback_reply(text)

    def _fallback_reply(self, text: str) -> str:
        """本地规则回复 (无 API 时兜底)"""
        import requests

        try:
            # 持仓
            if any(w in text for w in ["持仓", "仓位", "赚", "亏", "盈亏"]):
                r = requests.get("http://localhost:8000/api/v1/portfolio/mtm", timeout=5)
                d = r.json()
                s = d.get("summary", {})
                ps = d.get("positions", [])
                lines = [f"[总资产] {s.get('total_value',0):,.0f}  收益 {s.get('total_return_pct',0):+.2f}%"]
                for p in ps:
                    pnl = p.get("unrealized_pnl", 0)
                    e = "+" if pnl > 0 else ""
                    lines.append(f"{e}{p['name']} {p.get('current_price',0):.2f} ({pnl:+.0f})")
                return "\n".join(lines)

            # 行情
            if any(w in text for w in ["行情", "大盘", "指数"]):
                r = requests.get("http://localhost:8000/api/v1/realtime/market", timeout=5)
                d = r.json()
                idx = d.get("indices", {})
                ms = d.get("market_status", {})
                lines = [f"[{ms.get('detail','')}]"]
                for n, v in idx.items():
                    p = v.get("pct", 0)
                    lines.append(f"{n}: {v.get('price',0):.2f} ({p:+.2f}%)")
                return "\n".join(lines)

            # 回测
            if any(w in text for w in ["回测", "策略"]):
                import json as _json
                bp = Path(__file__).parent.parent / "reports" / "strategy_backtest.json"
                if bp.exists():
                    with open(bp) as f:
                        data = _json.load(f)
                    ranked = sorted(data["results"].items(), key=lambda x: x[1]["sharpe"], reverse=True)
                    lines = ["[策略排名]"]
                    for i, (sid, r) in enumerate(ranked[:5], 1):
                        lines.append(f"{i}. {r['name']} 胜率{r['win_rate']:.0%} 夏普{r['sharpe']:+.1f}")
                    return "\n".join(lines)
                return "暂无回测数据"

            # 股票分析
            code_match = re.search(r'(60\d{4}|00\d{4}|30\d{4}|688\d{3})', text)
            if code_match:
                code = code_match.group(1)
                prefix = "sh" if code.startswith("6") else "sz"
                r = requests.get(f"http://localhost:8000/api/v1/stock/{prefix}.{code}", timeout=5)
                if r.status_code == 200:
                    d = r.json()
                    info = d.get("info", d)
                    return (f"{info.get('name', code)}({code})\n"
                            f"现价: {info.get('price', info.get('close', 0)):.2f}\n"
                            f"涨跌: {info.get('pct_change', info.get('change', 0)):+.2f}%\n"
                            f"RSI: {info.get('rsi_14', 'N/A')}")

        except Exception:
            pass

        return "收到! 支持:\n· 持仓/行情/回测\n· 分析<股票代码>\n· 直接提问"

    def run(self):
        """主循环: 监听消息并回复"""
        if not self.connect():
            print("请先登录电脑版微信!")
            return

        logger.info("微信机器人已启动, 监听新消息中...")
        self.running = True

        while self.running:
            try:
                chats = self.get_unread_chats()
                for chat in chats:
                    name = chat["name"]
                    if name in ("微信", "微信团队", "文件传输助手", "订阅号", "腾讯新闻"):
                        continue

                    msg = self.get_last_message(name)
                    if msg and msg not in self.replied_msgs and len(msg) > 1:
                        logger.info(f"[{name}] {msg[:100]}")
                        self.replied_msgs.add(msg)

                        reply = self.get_ai_reply(msg)
                        if reply:
                            self.send_message(name, reply)

                        # 清理旧消息ID (保留最近200条)
                        if len(self.replied_msgs) > 200:
                            self.replied_msgs = set(list(self.replied_msgs)[-100:])

                time.sleep(2)  # 每2秒扫描一次

            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"循环异常: {e}")
                time.sleep(5)

        logger.info("微信机器人已停止")

    def stop(self):
        self.running = False


# ═══════════════════════════════════════════════════════════════
# 命令行入口
# ═══════════════════════════════════════════════════════════════

def main():
    import argparse
    parser = argparse.ArgumentParser(description="微信好友机器人")
    parser.add_argument("--test", action="store_true", help="测试: 检查微信连接并发送测试消息")
    args = parser.parse_args()

    bot = WeChatFriend()

    if args.test:
        if bot.connect():
            print("微信连接成功!")
            # 测试: 给自己发一条
            bot.send_message("文件传输助手", "🤖 A股AI助手已上线!\n发送 持仓/行情/回测 试试吧")
            print("已发送测试消息到文件传输助手")
        else:
            print("连接失败, 请确认微信已登录")
        return

    bot.run()


if __name__ == "__main__":
    main()
