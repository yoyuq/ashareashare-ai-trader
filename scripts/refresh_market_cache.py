"""用腾讯行情重建全市场缓存 (full_market_cache.json)

EastMoney/AKShare 常被墙/需代理; 腾讯 qt.gtimg.cn 免代理稳定。
从现有缓存读股票列表 → 腾讯批量拉今日实时 → 重建缓存 (date=今天)。

用法: python scripts/refresh_market_cache.py
"""

import sys
import time
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.full_market_cache import (  # noqa: E402
    CACHE_PATH, read_full_market_cache, write_full_market_cache,
)
from timeutil import last_trade_date  # noqa: E402


def fetch_tencent(symbols: list, batch: int = 50) -> dict:
    """批量拉腾讯行情 → {symbol: {price, pct, volume, amount, turnover, pe, pb, mv}}

    腾讯 qt.gtimg.cn 为国内直连源 (免代理); requests 会吃 HTTP_PROXY 环境变量,
    代理挂掉时每批全超时 → 0 只。强制 trust_env=False 直连。
    """
    out = {}
    sess = requests.Session()
    sess.trust_env = False
    i = 0
    while i < len(symbols):
        chunk = symbols[i:i + batch]
        codes = [s.replace("sh.", "sh").replace("sz.", "sz").replace("bj.", "bj") for s in chunk]
        text = None
        for attempt in range(3):
            try:
                r = sess.get(f"https://qt.gtimg.cn/q={','.join(codes)}", timeout=8,
                             headers={"User-Agent": "Mozilla/5.0",
                                      "Referer": "https://finance.qq.com/"})
                r.encoding = "gbk"
                # 限速哨兵: 响应只有 v_pv_none_match="1" 无真实行情 → 退避重试 (分钟配额限速)
                if "v_pv_none_match" in r.text and r.text.count('="') < 2:
                    wait = 5.0 * (attempt + 1)
                    print(f"  批次 {i // batch} 被限速, 退避 {wait:.0f}s (attempt {attempt + 1})")
                    time.sleep(wait)
                    continue
                text = r.text
                break
            except Exception as e:
                print(f"  批次 {i // batch} 拉取失败: {type(e).__name__}: {e}")
                time.sleep(3.0)
        if text is None:
            i += batch
            continue
        for line in text.strip().split("\n"):
            if "=" not in line or "~" not in line:
                continue
            try:
                f = line.split("=", 1)[1].strip('"').split("~")
                if len(f) < 40:
                    continue
                code = f[2]
                prefix = "sh" if code.startswith("6") else ("bj" if code.startswith(("8", "4")) else "sz")
                sym = f"{prefix}.{code}"
                out[sym] = {
                    "name": f[1], "code": code,
                    "price": float(f[3]) if f[3] else 0,
                    "pct_change": float(f[32]) if len(f) > 32 and f[32] else 0,
                    "volume": int(float(f[6])) if f[6] else 0,
                    "amount": float(f[37]) * 1e4 if len(f) > 37 and f[37] else 0,  # 万元→元
                    "turnover": float(f[38]) if len(f) > 38 and f[38] else 0,
                    "pe_ttm": float(f[39]) if len(f) > 39 and f[39] else None,
                    "total_mv": float(f[44]) * 1e8 if len(f) > 44 and f[44] else None,  # 亿→元
                    "pb": float(f[46]) if len(f) > 46 and f[46] else None,
                }
            except (ValueError, IndexError):
                continue
        i += batch
        time.sleep(0.2)  # 腾讯限速: 间隔过短批量请求会返回 v_pv_none_match
    return out


MIN_QUOTES = 500  # 低于此数视为网络性失败, 禁止写缓存 (零模拟: 不得用空数据覆盖好缓存)


def bootstrap_symbols() -> list:
    """缓存为空时从 Baostock 全市场清单自举 (真数据源, 无代理依赖)。"""
    import baostock as bs

    lg = bs.login()
    if lg.error_code != "0":
        raise RuntimeError(f"baostock 登录失败: {lg.error_msg}")
    rows = []
    try:
        # 当日清单可能未生成 (盘后延迟), 往前回溯至最近有数据的交易日
        d = pd.Timestamp(last_trade_date())
        for _ in range(10):
            rs = bs.query_all_stock(day=d.date().isoformat())
            rows = []
            while rs.error_code == "0" and rs.next():
                rows.append(rs.get_row_data())
            if rows:
                print(f"清单日期: {d.date()}")
                break
            d -= pd.Timedelta(days=1)
    finally:
        bs.logout()
    if not rows:
        raise RuntimeError("query_all_stock 无数据, 中止 (零模拟: 不写缓存)")
    # 返回裸 6 位代码 (与旧缓存 code 列口径一致; main 会统一加前缀)
    codes = [r[0].split(".")[1] for r in rows if r[0].startswith(("sh.6", "sz.0", "sz.3", "bj.8", "bj.4"))]
    print(f"Baostock 自举股票列表: {len(codes)} 只")
    return codes


def main():
    # GBK 控制台兼容 (✅ 等字符在 Windows 默认代码页下崩溃 → 定时任务非零退出)
    for _s in (sys.stdout, sys.stderr):
        if hasattr(_s, "reconfigure"):
            try:
                _s.reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass
    _old_df, _old_date = read_full_market_cache()
    if _old_df is not None and len(_old_df) > MIN_QUOTES:
        symbols = [str(d) for d in _old_df["code"].tolist()]
        print(f"从旧缓存读取 {len(symbols)} 只股票")
    else:
        symbols = bootstrap_symbols()
    print(f"从旧缓存读取 {len(symbols)} 只股票")

    # 转成带前缀 symbol
    syms = []
    for c in symbols:
        c = str(c)
        prefix = "sh" if c.startswith("6") else ("bj" if c.startswith(("8", "4")) else "sz")
        syms.append(f"{prefix}.{c}")

    t0 = time.time()
    quotes = fetch_tencent(syms)
    print(f"腾讯拉取完成: {len(quotes)}/{len(syms)} 只, 耗时 {time.time()-t0:.0f}s")
    if len(quotes) < MIN_QUOTES:
        print(f"❌ 拉取 {len(quotes)} 只 < 最低阈值 {MIN_QUOTES}, 网络性失败 — 不写缓存 (保护旧数据)")
        raise SystemExit(1)

    # 重建 data (保留旧缓存里没拉到的基本面? 简单起见只保留拉到的)
    new_data = []
    for sym, q in quotes.items():
        if q["price"] <= 0:
            continue
        new_data.append(q)

    # 快照日期用最近 A 股交易日 (非宿主机 date.today()), 周末/节假日刷新时陈旧标记不失真
    trade_date = last_trade_date().isoformat()
    write_full_market_cache(pd.DataFrame(new_data), trade_date, "tencent_realtime")
    print(f"✅ 缓存已更新: {trade_date} (最近交易日) | {len(new_data)} 只 | {CACHE_PATH}")
    if new_data:
        print("样例:", new_data[0]["name"], new_data[0]["price"], f"涨跌{new_data[0]['pct_change']}%")


if __name__ == "__main__":
    main()
