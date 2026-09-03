"""短线纸面 bet 周度复盘 — shortline_intraday_momo (只读分析, 不改冻结规则).

产出 reports/agent_loop/shortline_review.md:
  - 完成回合统计 (净收益 mean/median/胜率/左尾) + 滚动 10 笔
  - "钱在日内"缺口追踪: 开→高 (不可实现上限) vs 10:00→14:55 (可实现) 逐日
  - fillability 剔除量, 触发率, 管线健康 (分时/盘口文件覆盖)
  - 与 41 天先验 (中位 −0.16%) 的轨迹对照
进化纪律: 复盘只报告, 不改规则; 规则变更只发生在判定日之后的新预注册。

用法: python scripts/shortline_review.py [--out 路径]
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from minute_collector import (ENTRY_MIN, EXIT_MIN, MINUTE_DIR, BOOK_DIR,   # noqa: E402
                              JOURNAL_FP, COST_RT)

REVIEW_FP = ROOT / "reports" / "agent_loop" / "shortline_review.md"
PRIOR_MEDIAN = -0.0016   # 41 天窗口先验 (10:00→尾盘, 冻结引用)


def _force_utf8() -> None:
    for name in ("stdout", "stderr"):
        s = getattr(sys, name, None)
        if hasattr(s, "reconfigure"):
            try:
                s.reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass


def load_journal() -> list[dict]:
    if not JOURNAL_FP.exists():
        return []
    return [json.loads(l) for l in JOURNAL_FP.read_text(encoding="utf-8").splitlines() if l.strip()]


def round_trips(jrn: list[dict]) -> list[dict]:
    ins = {(r["date"], r["symbol"]): r for r in jrn
           if r["event"] == "in" and r.get("triggered")}
    outs = {(r["date"], r["symbol"]): r for r in jrn if r["event"] == "out"}
    rt = []
    for k, o in outs.items():
        i = ins.get(k)
        if i:
            rt.append({**i, "exit_price": o["price"], "net_ret": o["net_ret"],
                       "unfillable": i["gain30"] >= 0.095})
    return rt


def intraday_gap(day_fp: Path) -> dict | None:
    """单日: 开→高均值 (不可实现) vs 10:00→14:55 均值 (可实现口径近似)。"""
    m = pd.read_parquet(day_fp)
    gaps = []
    for sym, g in m.groupby("symbol"):
        g = g.sort_values("time")
        e = g[g["time"] <= ENTRY_MIN]
        x = g[g["time"] >= EXIT_MIN]
        if e.empty or x.empty:
            continue
        p10 = e["price"].iloc[-1]
        hi = g["price"].max()
        px = x["price"].iloc[-1]
        gaps.append({"sym": sym, "open_to_high": hi / g["price"].iloc[0] - 1,
                     "p10_to_exit": px / p10 - 1})
    if not gaps:
        return None
    d = pd.DataFrame(gaps)
    return {"open_to_high": float(d["open_to_high"].mean()),
            "p10_to_exit": float(d["p10_to_exit"].mean()), "n": len(d)}


def main() -> int:
    _force_utf8()
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REVIEW_FP))
    args = ap.parse_args()

    jrn = load_journal()
    rt = round_trips(jrn)
    fillable = [r for r in rt if not r["unfillable"]]
    days = sorted({r["date"] for r in jrn})

    lines = [f"# 短线纸面 bet 周度复盘 ({datetime.now().strftime('%Y-%m-%d %H:%M')})",
             "",
             f"- 跟踪天数: {len(days)} | 判定 {len(jrn)} 条 | 触发 {len(ins_t:=[r for r in jrn if r.get('triggered')])} "
             f"| 完成回合 {len(rt)} (可成交 {len(fillable)})",
             ""]

    if fillable:
        nets = np.array([r["net_ret"] for r in fillable])
        lines += [f"## 可成交回合 ({len(nets)} 笔)",
                  f"- mean {nets.mean()*100:+.2f}% | median {np.median(nets)*100:+.2f}% "
                  f"| 胜率 {(nets>0).mean()*100:.0f}% | 最差 {nets.min()*100:+.2f}% "
                  f"| p5 {np.percentile(nets,5)*100:+.2f}%",
                  f"- 先验对照 (41天中位 {PRIOR_MEDIAN*100:+.2f}%): "
                  f"{'劣于' if np.median(nets) < PRIOR_MEDIAN else '优于/持平'}",
                  f"- 滚动最近10笔 mean {nets[-10:].mean()*100:+.2f}%",
                  ""]

    ins_all = [r for r in jrn if r["event"] == "in"]
    if ins_all:
        n_unfil = sum(1 for r in ins_all if r.get("triggered") and r["gain30"] >= 0.095)
        lines += [f"## 触发结构",
                  f"- 判定→触发率 {len([r for r in ins_all if r.get('triggered')])/len(ins_all)*100:.0f}% "
                  f"| 触发中不可成交(已涨停) {n_unfil} 笔",
                  ""]

    # 钱在日内缺口
    lines += ["## 「钱在日内」缺口 (开→高 vs 10:00→14:55, 全 watchlist)"]
    for day in days[-5:]:
        fp = MINUTE_DIR / f"{day}.parquet"
        if not fp.exists():
            continue
        g = intraday_gap(fp)
        if g:
            lines.append(f"- {day}: 开→高 {g['open_to_high']*100:+.2f}% "
                         f"(n={g['n']}) | 10:00→14:55 {g['p10_to_exit']*100:+.2f}% "
                         f"→ 缺口 {(g['open_to_high']-g['p10_to_exit'])*100:+.2f}pp")
    # 管线健康
    lines += ["", "## 管线健康"]
    for day in days[-5:]:
        mfp = MINUTE_DIR / f"{day}.parquet"
        bfp = BOOK_DIR / f"{day}.parquet" if BOOK_DIR.exists() else None
        ms = f"分时{pd.read_parquet(mfp).symbol.nunique()}票" if mfp.exists() else "分时缺失"
        bs = f"盘口{len(pd.read_parquet(bfp))}快照" if bfp and bfp.exists() else "盘口缺失"
        lines.append(f"- {day}: {ms} | {bs}")
    lines += ["", "> 复盘只报告不改规则; 规则变更仅限判定日后新预注册 (prereg_shortline_intraday.md)"]

    text = "\n".join(lines)
    Path(args.out).write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
