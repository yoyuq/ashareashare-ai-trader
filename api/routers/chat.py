"""APIRouter: 对话式AI (v6.0 拆分自 server.py)"""
from fastapi import APIRouter, HTTPException
from loguru import logger

from api.deps import get_chat_agent
from api.schemas import ChatHistoryResponse, ChatRequest, ChatResponse

router = APIRouter(prefix="/api/v1", tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """
    🆕 v2.8 对话式AI端点

    支持多轮对话: 传入相同的 session_id 即可保持上下文。
    新会话: 不传 session_id 或传新值。

    示例:
      curl -X POST http://localhost:8000/api/v1/chat \
        -H "Content-Type: application/json" \
        -d '{"message": "茅台怎么样？", "session_id": "user1"}'
    """
    try:
        agent = get_chat_agent()
        reply = await agent.chat(
            user_message=req.message,
            session_id=req.session_id,
        )

        # 获取工具调用记录 (最后一条assistant消息的tool_calls)
        tool_calls = []
        conv = agent.conversations.get(req.session_id, [])
        for msg in reversed(conv):
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                tool_calls = [tc.get("function", {}).get("name", "unknown")
                             for tc in msg.get("tool_calls", [])]
                break

        return ChatResponse(
            reply=reply,
            session_id=req.session_id,
            tool_calls=tool_calls,
        )
    except Exception as e:
        logger.error(f"Chat失败: {e}")
        raise HTTPException(500, f"对话处理失败: {e}")


@router.get("/chat/history", response_model=ChatHistoryResponse)
async def chat_history(session_id: str = "default"):
    """获取会话历史"""
    agent = get_chat_agent()
    conv = agent.conversations.get(session_id, [])
    # 只返回用户和assistant消息, 不返回system/tool
    history = [
        {"role": m["role"], "content": m.get("content", "")[:500]}
        for m in conv
        if m["role"] in ("user", "assistant") and m.get("content")
    ]
    return {
        "session_id": session_id,
        "message_count": len(history),
        "history": history[-20:],  # 最近20条
    }


@router.delete("/chat/history")
async def clear_chat_history(session_id: str = "default"):
    """清除会话历史"""
    agent = get_chat_agent()
    if session_id in agent.conversations:
        # 保留system prompt
        system_msgs = [m for m in agent.conversations[session_id]
                       if m["role"] == "system"]
        agent.conversations[session_id] = system_msgs
        return {"status": "cleared", "session_id": session_id}
    return {"status": "not_found", "session_id": session_id}
