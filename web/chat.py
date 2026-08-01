"""
A股AI助手 — 微信风格聊天页面
访问: http://localhost:8502
"""

import streamlit as st
import requests
import os
from datetime import datetime

st.set_page_config(page_title="A股AI助手", page_icon="🤖", layout="centered")

API = os.getenv("API_BASE_URL", "http://localhost:8000")

# ── 微信风格 CSS ──
st.markdown("""
<style>
    .stApp { background: #ededed; }
    .chat-container { max-width:600px; margin:0 auto; padding:10px 0 70px 0; }

    .msg-row { display:flex; margin:8px 12px; align-items:flex-start; }
    .msg-row.user { justify-content:flex-end; }
    .msg-row.bot { justify-content:flex-start; }

    .msg-avatar { width:36px; height:36px; border-radius:4px; font-size:20px;
                  text-align:center; line-height:36px; flex-shrink:0; }
    .user .msg-avatar { background:#07c160; margin-left:8px; }
    .bot .msg-avatar { background:#1989fa; margin-right:8px; }

    .msg-bubble { max-width:75%; padding:10px 14px; border-radius:8px;
                  font-size:15px; line-height:1.5; word-break:break-word; }
    .user .msg-bubble { background:#95ec69; color:#000; }
    .bot .msg-bubble { background:#fff; color:#333; box-shadow:0 1px 2px rgba(0,0,0,0.1); }

    .msg-time { text-align:center; font-size:12px; color:#b2b2b2; margin:12px 0; }

    .quick-btns { display:flex; gap:6px; flex-wrap:wrap; padding:8px 12px; }
    .quick-btn { background:#fff; border:1px solid #07c160; color:#07c160;
                 padding:6px 14px; border-radius:16px; font-size:13px; cursor:pointer; }

    .header-bar { background:#ededed; padding:10px 16px; text-align:center;
                  border-bottom:1px solid #d9d9d9; position:sticky; top:0; z-index:10; }
    .header-bar h3 { margin:0; font-size:17px; color:#000; }
    .header-bar span { font-size:12px; color:#999; }

    .input-bar { position:fixed; bottom:0; left:0; right:0; background:#f7f7f7;
                 padding:8px 12px; border-top:1px solid #d9d9d9; max-width:600px;
                 margin:0 auto; display:flex; gap:8px; align-items:center; }

    /* Mobile optimization */
    @media (max-width:600px) {
        .msg-bubble { font-size:16px; }
    }
</style>
""", unsafe_allow_html=True)

# ── 初始化 ──
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "bot", "text": "你好！我是A股AI助手 🤖\n\n可以问我：\n• 今天行情怎么样\n• 我的持仓赚了还是亏了\n• 分析一下贵州茅台\n• 最近有什么好策略\n\n试试点击下方快捷按钮 👇", "time": datetime.now().strftime("%H:%M")}
    ]

# ── 顶部栏 ──
with st.container():
    st.markdown("""
    <div class="header-bar">
        <h3>A股AI助手</h3>
        <span>在线 · AI驱动量化分析</span>
    </div>
    """, unsafe_allow_html=True)

# ── 消息列表 ──
st.markdown('<div class="chat-container">', unsafe_allow_html=True)
for msg in st.session_state.messages:
    role = msg["role"]
    avatar = "🧑" if role == "user" else "🤖"
    st.markdown(f"""
    <div class="msg-row {role}">
        <div class="msg-avatar">{avatar}</div>
        <div class="msg-bubble">{msg['text']}</div>
    </div>
    """, unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)


# ── AI 回复 ──
def get_ai_reply(text: str) -> str:
    """调用 ChatAgent 获取回复"""
    # 命令路由
    cmd = text.strip()
    try:
        if cmd in ["持仓", "我的持仓", "portfolio"]:
            r = requests.get(f"{API}/api/v1/portfolio/mtm", timeout=5)
            d = r.json()
            s = d.get("summary", {})
            ps = d.get("positions", [])
            lines = [f"💰 总资产 ¥{s.get('total_value',0):,.0f}  收益 {s.get('total_return_pct',0):+.2f}%"]
            for p in ps:
                pnl = p.get("unrealized_pnl", 0)
                e = "📈" if pnl > 0 else ("📉" if pnl < 0 else "➖")
                lines.append(f"{e} {p['name']} ¥{p['current_price']:.2f}  {pnl:+.0f}")
            return "\n".join(lines)

        if cmd in ["行情", "今天行情", "大盘"]:
            r = requests.get(f"{API}/api/v1/realtime/market", timeout=5)
            d = r.json()
            idx = d.get("indices", {})
            ms = d.get("market_status", {})
            lines = [f"📊 {ms.get('detail','')}"]
            for n, v in idx.items():
                p = v.get("pct", 0)
                e = "🔴" if p > 0 else "🟢"
                lines.append(f"{e} {n}: {v.get('price',0):.2f}  {p:+.2f}%")
            return "\n".join(lines)

        if cmd in ["回测", "策略排名"]:
            import json
            from pathlib import Path
            bp = Path(__file__).parent.parent / "reports" / "strategy_backtest.json"
            if bp.exists():
                with open(bp) as f:
                    data = json.load(f)
                ranked = sorted(data["results"].items(), key=lambda x: x[1]["sharpe"], reverse=True)
                lines = ["🏆 策略排名 (夏普)"]
                for i, (sid, r) in enumerate(ranked[:5], 1):
                    lines.append(f"{i}. {r['name']} 胜率{r['win_rate']:.0%} 夏普{r['sharpe']:+.1f}")
                return "\n".join(lines)
            return "暂无回测数据"

        # 分析某只股票
        import re
        code_match = re.search(r'(60\d{4}|00\d{4}|30\d{4}|688\d{3})', text)
        if code_match:
            code = code_match.group(1)
            prefix = "sh" if code.startswith("6") else "sz"
            try:
                r = requests.get(f"{API}/api/v1/stock/{prefix}.{code}", timeout=10)
                if r.status_code == 200:
                    d = r.json()
                    info = d.get("info", d)
                    return (
                        f"📈 {info.get('name', code)}({code})\n"
                        f"最新价: ¥{info.get('price', info.get('close', 0)):.2f}\n"
                        f"涨跌: {info.get('pct_change', info.get('change', 0)):+.2f}%\n"
                        f"RSI(14): {info.get('rsi_14', 'N/A')}\n"
                        f"趋势: {info.get('trend_score', 'N/A')}"
                    )
            except:
                pass
            return f"未找到 {code} 的数据"

        # Fallback: AI 对话
        try:
            r = requests.post(f"{API}/api/v1/chat", json={"message": text}, timeout=30)
            return r.json().get("reply", "让我想想...")[:1500]
        except:
            pass

        return "收到你的问题了。目前AI服务暂时不可用，请稍后再试。\n\n可以试试这些快捷命令：\n• 持仓\n• 行情\n• 回测\n• 分析 600519"
    except Exception as e:
        return f"出错了: {e}"


# ── 快捷按钮 ──
st.markdown('<div class="quick-btns">', unsafe_allow_html=True)
cols = st.columns(4)
quick_cmds = ["持仓", "行情", "回测", "分析 600519"]
for i, cmd in enumerate(quick_cmds):
    if cols[i].button(cmd, key=f"q{i}", use_container_width=True):
        st.session_state.messages.append({"role": "user", "text": cmd, "time": datetime.now().strftime("%H:%M")})
        with st.spinner("..."):
            reply = get_ai_reply(cmd)
        st.session_state.messages.append({"role": "bot", "text": reply, "time": datetime.now().strftime("%H:%M")})
        st.rerun()
st.markdown('</div>', unsafe_allow_html=True)

# ── 输入框 ──
with st.form("chat_input", clear_on_submit=True):
    user_input = st.text_input("", placeholder="输入消息...", label_visibility="collapsed", key="msg_input")
    submitted = st.form_submit_button("发送")

if submitted and user_input:
    now = datetime.now().strftime("%H:%M")
    st.session_state.messages.append({"role": "user", "text": user_input, "time": now})
    with st.spinner("🤔"):
        reply = get_ai_reply(user_input)
    st.session_state.messages.append({"role": "bot", "text": reply, "time": now})
    st.rerun()
