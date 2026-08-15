"""冷落 beta 实盘名单生成 — 从全市场快照算 bottom-100 低换手票。

与 run_cold_tilt_rebalance.py 同口径 (11/11 窗口验证的被动篮子):
  - 过滤: 非ST / 可交易(换手>0) / pe>0 / pb>0  (对应回测 isST==0 & is_trade==1 & pe>0 & pb>0)
  - 池子: top-800 流动性 (按成交额 amount 降序)
  - 选票: bottom-100 换手率 (按 turnover 升序)

用法 (先刷新快照, 再生成名单):
  python scripts/refresh_market_cache.py      # 腾讯免代理, ~1分钟, 日期=最近交易日
  python scripts/cold_tilt_live.py            # 生成当日 bottom-100 名单

输出: stdout (全 100 只) + reports/cold_tilt_live_{date}.md

诚实边界: 这是月频再平衡被动篮子 (不是每日信号), 等权各买一份, 持有到下次再平衡。
机制是系统性 beta (大盘tilt + 冷落溢价) 非选股alpha, 见 reports/cold_tilt_rebalance.md 的 caveats。
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent
CACHE = ROOT / "simulation_data" / "full_market_cache.json"
K = 100
UNIVERSE_N = 800


def _force_utf8():
    for name in ("stdout", "stderr"):
        s = getattr(sys, name, None)
        if hasattr(s, "reconfigure"):
            try:
                s.reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass


def main() -> int:
    _force_utf8()
    cache = json.loads(CACHE.read_text(encoding="utf-8"))
    date = cache.get("date", "?")
    df = pd.DataFrame(cache.get("data", []))
    if df.empty:
        print("缓存为空, 先跑 scripts/refresh_market_cache.py")
        return 1
    df["name"] = df["name"].astype(str)

    n0 = len(df)
    n_st = int(df["name"].str.contains("ST", na=False).sum())
    df = df[~df["name"].str.contains("ST", na=False)]      # 非ST
    df = df[df["turnover"] > 0]                            # 可交易(未停牌)
    df = df[df["amount"] > 0]
    df = df[(df["pe_ttm"] > 0) & (df["pb"] > 0)]           # pe>0 & pb>0 (亏损股 PE 无意义)

    print(f"快照日期: {date} | 原始 {n0} 只 | 剔除 ST {n_st} 只 | 过滤后 {len(df)} 只")

    top800 = df.nlargest(UNIVERSE_N, "amount")
    bottom100 = top800.nsmallest(K, "turnover").sort_values("turnover")

    # 汇总统计
    print(f"top-800 流动性池: 换手中位数 {top800['turnover'].median():.2f}% | "
          f"成交额中位数 {top800['amount'].median()/1e8:.2f} 亿")
    print(f"bottom-100 冷落票: 换手中位数 {bottom100['turnover'].median():.2f}% | "
          f"换手范围 {bottom100['turnover'].min():.2f}% ~ {bottom100['turnover'].max():.2f}%")

    out = bottom100[["code", "name", "price", "turnover", "amount", "pe_ttm", "pb"]].copy()
    out["amount_yi"] = (out["amount"] / 1e8).round(2)      # 成交额(亿)
    out["pe_ttm"] = out["pe_ttm"].round(1)
    out["pb"] = out["pb"].round(2)
    out["turnover"] = out["turnover"].round(2)

    # 打印全表 (code 名前缀补齐方便下单)
    print("\n" + "=" * 90)
    print(f"冷落 beta 实盘名单 ({date}) — bottom-100 低换手, 等权各买一份")
    print("=" * 90)
    for i, (_, r) in enumerate(out.iterrows(), 1):
        code = str(r["code"])
        prefix = "sh" if code.startswith("6") else ("bj" if code.startswith(("8", "4")) else "sz")
        print(f"{i:>3}. {prefix}.{code:<10} {r['name']:<8} 价 {r['price']:>7.2f}  "
              f"换手 {r['turnover']:>5.2f}%  成交 {r['amount_yi']:>7.2f}亿  "
              f"PE {r['pe_ttm']:>6.1f}  PB {r['pb']:>5.2f}")

    # 落盘
    out_path = ROOT / "reports" / f"cold_tilt_live_{date}.md"
    lines = [
        f"# 冷落 beta 实盘名单 ({date})",
        "",
        f"快照来源: 腾讯实时 (tencent_realtime) | 生成 {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"口径: top-800 成交额 → bottom-100 换手率 (与 11 窗口回测同口径)",
        "",
        "| # | 代码 | 名称 | 价格 | 换手% | 成交额(亿) | PE | PB |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for i, (_, r) in enumerate(out.iterrows(), 1):
        code = str(r["code"])
        prefix = "sh" if code.startswith("6") else ("bj" if code.startswith(("8", "4")) else "sz")
        lines.append(f"| {i} | {prefix}.{code} | {r['name']} | {r['price']} | {r['turnover']} | "
                     f"{r['amount_yi']} | {r['pe_ttm']} | {r['pb']} |")
    lines += [
        "",
        "## 操作纪律 (重要)",
        "",
        "- **月频再平衡**, 不是每日信号: 等权买入这 100 只, 持有约 1 个月, 再按当日新名单换仓。",
        "- **不要周频追逐**: 冷落/换手是慢变量, 周频换仓已被回测证伪 (5日 4/11)。",
        "- **这是 beta 不是 alpha**: 跑赢上证靠大盘tilt+冷落溢价, 非选股能力; 回撤不占优 (见 cold_tilt_rebalance.md)。",
        "- **幸存者偏差**: 股票池来自当前快照, 历史回测退市股缺失; 2015-2017 老窗口 Δreturn 或轻微高估。",
        "",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n已写 {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
