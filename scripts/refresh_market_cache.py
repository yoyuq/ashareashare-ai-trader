"""用腾讯行情重建全市场缓存 (full_market_cache.json)

EastMoney/AKShare 常被墙/需代理; 腾讯 qt.gtimg.cn 免代理稳定。
从现有缓存读股票列表 → 腾讯批量拉今日实时 → 重建缓存 (date=今天)。

用法: python scripts/refresh_market_cache.py
"""

import json
import sys
import time
from datetime import date
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).parent.parent))

CACHE = Path("simulation_data/full_market_cache.json")


def fetch_tencent(symbols: list, batch: int = 50) -> dict:
    """批量拉腾讯行情 → {symbol: {price, pct, volume, amount, turnover, pe, pb, mv}}"""
    out = {}
    for i in range(0, len(symbols), batch):
        chunk = symbols[i:i + batch]
        codes = [s.replace("sh.", "sh").replace("sz.", "sz").replace("bj.", "bj") for s in chunk]
        try:
            r = requests.get(f"https://qt.gtimg.cn/q={','.join(codes)}", timeout=8,
                             headers={"User-Agent": "Mozilla/5.0",
                                      "Referer": "https://finance.qq.com/"})
            r.encoding = "gbk"
            for line in r.text.strip().split("\n"):
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
        except Exception:
            pass
        time.sleep(0.05)
    return out


def main():
    if not CACHE.exists():
        print("无现有缓存, 无法获取股票列表")
        return
    old = json.loads(CACHE.read_text(encoding="utf-8"))
    symbols = [d["code"] for d in old.get("data", [])]
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

    # 重建 data (保留旧缓存里没拉到的基本面? 简单起见只保留拉到的)
    new_data = []
    for sym, q in quotes.items():
        if q["price"] <= 0:
            continue
        new_data.append(q)

    new_cache = {
        "date": date.today().isoformat(),
        "count": len(new_data),
        "source": "tencent_realtime",
        "data": new_data,
    }
    CACHE.write_text(json.dumps(new_cache, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"✅ 缓存已更新: {new_cache['date']} | {len(new_data)} 只 | {CACHE}")
    if new_data:
        print("样例:", new_data[0]["name"], new_data[0]["price"], f"涨跌{new_data[0]['pct_change']}%")


if __name__ == "__main__":
    main()
