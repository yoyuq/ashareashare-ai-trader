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


# 2026年中国法定节假日 (A股休市日, 仅含主要长假) — 与 api/server.py 的日历保持一致
CN_HOLIDAYS = {
    date(2026, 1, 1), date(2026, 1, 2),      # 元旦
    date(2026, 2, 16), date(2026, 2, 17), date(2026, 2, 18), date(2026, 2, 19), date(2026, 2, 20),  # 春节
    date(2026, 4, 6),                        # 清明节
    date(2026, 5, 1), date(2026, 5, 4), date(2026, 5, 5),  # 劳动节
    date(2026, 6, 22),                       # 端午节
    date(2026, 9, 28),                       # 中秋节
    date(2026, 10, 1), date(2026, 10, 2), date(2026, 10, 5), date(2026, 10, 6), date(2026, 10, 7),  # 国庆节
}


def is_trading_day(d: date) -> bool:
    """是否 A 股交易日 (周末 + 已知法定节假日休市)。"""
    return d.weekday() < 5 and d not in CN_HOLIDAYS


def last_trade_date(from_date: date | None = None) -> date:
    """最近一个 A 股交易日 (含 from_date 当天, 默认北京今日)。

    周末/节假日向后回退到最近交易日; 用于 full_market_cache 快照日期, 避免
    周末把 `date.today()` 当交易日写入缓存 (导致陈旧标记失真)。
    """
    d = from_date or today_cn()
    while not is_trading_day(d):
        d -= timedelta(days=1)
    return d
