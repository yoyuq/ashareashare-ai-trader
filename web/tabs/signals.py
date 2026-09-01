"""Tab: 📋 AI信号 — ChatAgent 对话式信号生成"""
import streamlit as st

from web.data import _run_async, analyzer, knowledge, router


def render():
    st.subheader("AI交易信号")

    # Chat-style signal generation
    if "signal_chat" not in st.session_state:
        st.session_state.signal_chat = []

    for msg in st.session_state.signal_chat:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("向AI提问，如: 'Scan for buy signals' or 'Analyze 600519 with MACD strategy'"):
        st.session_state.signal_chat.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Analyzing..."):
                try:
                    from agent.chat_agent import ChatAgent
                    agent = ChatAgent(router=router, knowledge=knowledge, analyzer=analyzer)

                    async def c():
                        return await agent.chat(prompt, session_id="dashboard")
                    reply = _run_async(c())
                    st.markdown(reply)
                    st.session_state.signal_chat.append({"role": "assistant", "content": reply})
                except Exception as e:
                    st.error(str(e))

    if st.session_state.signal_chat:
        if st.button("清除历史"):
            st.session_state.signal_chat = []
            st.rerun()
