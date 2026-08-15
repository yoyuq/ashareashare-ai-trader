"""A股代码 → 交易所前缀 单一来源 (v5.6 P2-2)。

此前 `prefix = "sh" if code.startswith("6") else "sz"` 散落 15+ 处, 且普遍丢北交所 bj
(8xxxxx/4xxxxx/920xxx)。统一到本模块, 供 dashboard/refresh_market_cache 等复用。

约定 (与 CLAUDE.md 的 `{market}.{code}` 一致):
  - 6xxxxx/68xxxxx → sh (上交所主板/科创板); 9xxxxx 默认 sh (900 沪B), 但 920xxx → bj (北交所新码)
  - 8xxxxx / 4xxxxx → bj (北交所)
  - 0xxxxx / 3xxxxx → sz (深交所主板/创业板)
"""
from __future__ import annotations


def market_prefix(code) -> str:
    """根据 A 股代码推断交易所前缀: `sh` / `sz` / `bj`。"""
    c = str(code).strip().split(".")[-1]  # 兼容 "sh.600519" / "600519" / "600519.SH"
    if c.startswith("920"):              # 北交所新代码段 (2024+)
        return "bj"
    if c.startswith(("8", "4")):         # 北交所 8xxxxx / 4xxxxx
        return "bj"
    if c.startswith(("6", "9")):         # 上交所 6xxxxx / 68xxxxx (900 沪B)
        return "sh"
    return "sz"                          # 深交所 0xxxxx / 3xxxxx


def to_symbol(code) -> str:
    """幂等: 裸代码或已带前缀的 symbol → 标准 `{market}.{code}`。"""
    s = str(code).strip()
    if s.startswith(("sh.", "sz.", "bj.")):
        return s
    c = s.split(".")[-1]
    return f"{market_prefix(c)}.{c}"
