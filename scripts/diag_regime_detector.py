"""最终设计验证: 指数25d动量为主 + 极端快速回调/恐慌覆盖. 在两个regime窗口验证防过拟合."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
REPLAY_DIR = Path("replay_data")

def load(cache):
    big = pd.read_parquet(REPLAY_DIR / cache)
    pv = big.pivot_table(index="date", columns="symbol", values="pctChg").apply(pd.to_numeric, errors="coerce")
    idx = big[big["symbol"] == "sh.000001"].set_index("date").sort_index()["close"].astype(float)
    return pv, idx

def detect(day, ic):
    """改进版: 指数25d动量主 + 极端覆盖. day=cross-section, ic=index closes up to T."""
    avg = day.mean(); ed = (day < -5).mean()
    m5 = (ic.iloc[-1] / ic.iloc[-6] - 1) * 100 if len(ic) >= 6 else 0.0
    m25 = (ic.iloc[-1] / ic.iloc[-26] - 1) * 100 if len(ic) >= 26 else m5
    # 极端: 截面恐慌 OR 指数5d崩
    if ed > 0.1 or m5 < -6:
        return "crisis"
    if avg < -1.5 and m25 < -1:
        return "strong_bear"
    if m25 < -2:
        return "strong_bear" if m5 < -1 else "weak_bear"
    if m25 < 1.5:
        return "weak_bear" if m5 < 0 else "range_bound"
    if m25 < 4:
        return "range_bound" if m5 < 0.5 else "weak_bull"
    if m25 < 6:
        return "weak_bull" if m5 < 1 else "strong_bull"
    return "strong_bull"

def run(cache, start, end, label):
    pv, ic_all = load(cache)
    window = [d for d in pd.to_datetime(pv.index) if pd.Timestamp(start) <= d <= pd.Timestamp(end)]
    print(f"\n=== {label} ({start}→{end}) ===")
    print(f"{'日期':<12}{'reg':<12}{'m5':>7}{'m25':>8}   {'20d动量':<8}")
    print("-" * 52)
    stab = {}
    for d in window:
        day = pv.loc[d]
        ic = ic_all.loc[:d]
        if len(ic) < 26:
            continue
        reg = detect(day, ic)
        m5 = (ic.iloc[-1]/ic.iloc[-6]-1)*100
        m25 = (ic.iloc[-1]/ic.iloc[-26]-1)*100
        mom20 = (ic.iloc[-1]/ic.iloc[-21]-1)*100
        stab[reg] = stab.get(reg, 0) + 1
        print(f"{str(d)[:10]:<12}{reg:<12}{m5:>+6.1f}{m25:>+7.1f}   {mom20:>+7.1f}%")
    print("分布:", stab)

if __name__ == "__main__":
    # 窗口1: 2021 牛转熊 (原始A/B窗口)
    run("daily_2020-06-01_2021-02-28_idx.parquet", "2020-12-29", "2021-02-26", "2021 抱团牛→调整")
    # 窗口2: 2025-2026 牛熊切换 (另一个regime, 防过拟合)
    run("daily_2025-10-08_2026-07-31.parquet", "2026-02-01", "2026-06-30", "2026 牛→调整")