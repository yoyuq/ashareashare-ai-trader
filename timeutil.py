"""A股交易日时区工具 — 统一按 Asia/Shanghai (UTC+8) 判定交易日边界。

A股市场固定 UTC+8 且无夏令时, 直接使用固定偏移即可 (无需 tzdata, Windows 也可靠)。
此前各处散用 `datetime.now()`/`date.today()` 取宿主机时区, 非 UTC+8 主机上
"今日"、缓存 TTL 窗口、高峰定价时段、T+1 判定会整体偏移 8 小时。
"""

from datetime import date, datetime, timedelta, timezone

CN_TZ = timezone(timedelta(hours=8))  # 固定 UTC+8, 无夏令时


def now_cn() -> datetime:
    """当前北京时间 (时区感知)"""
    return datetime.now(CN_TZ)


def now_cn_naive() -> datetime:
    """当前北京时间 (naive, 无 tzinfo) — 兼容既有 datetime.now() 调用点的比较"""
    return datetime.now(CN_TZ).replace(tzinfo=None)


def today_cn() -> date:
    """当前北京日期 (交易日边界)"""
    return datetime.now(CN_TZ).date()
