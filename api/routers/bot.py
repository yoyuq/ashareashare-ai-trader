"""APIRouter: 企业微信机器人 (v6.0 拆分自 server.py)"""
from fastapi import APIRouter, Request
from loguru import logger

router = APIRouter(prefix="/api/v1", tags=["bot"])


@router.post("/bot/wecom")
async def wecom_bot_webhook(request: Request):
    """
    企业微信机器人消息回调

    接收: {"msgtype": "text", "text": {"content": "/持仓"}}
    返回: {"msgtype": "markdown", "markdown": {"content": "..."}}
    """
    try:
        body = await request.json()
        msgtype = body.get("msgtype", "text")

        if msgtype == "text":
            text = body.get("text", {}).get("content", "")
        else:
            text = ""

        if not text:
            return {"msgtype": "text", "text": {"content": "请输入命令, 如 /帮助"}}

        from notify.wecom_bot import WeComBot
        bot = WeComBot()
        reply = await bot.handle_message(text)

        return {"msgtype": "markdown", "markdown": {"content": reply}}

    except Exception as e:
        logger.error(f"WeCom webhook 异常: {e}")
        return {"msgtype": "text", "text": {"content": f"处理失败: {e}"}}


@router.post("/bot/push/daily")
async def wecom_push_daily():
    """手动触发每日推送"""
    from notify.wecom_bot import WeComBot
    bot = WeComBot()
    ok = bot.push_daily_summary()
    return {"status": "ok" if ok else "failed", "channel": "wecom"}


@router.post("/bot/push/alert")
async def wecom_push_alert(title: str = "", detail: str = "", level: str = "warning"):
    """手动触发告警推送"""
    from notify.wecom_bot import WeComBot
    bot = WeComBot()
    ok = bot.push_alert(title, detail, level)
    return {"status": "ok" if ok else "failed", "channel": "wecom"}
