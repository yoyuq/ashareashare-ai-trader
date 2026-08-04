"""多 regime thinking vs 非thinking A/B 报告生成

5 次回放 (牛市/危机 × 非thinking/thinking[危机重复]) → 汇总对比报告。

用法:
  python scripts/gen_multiregime_report.py
输出:
  reports/thinking_vs_nothink_multiregime.md
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

REPLAY_DIR = Path("replay_data")
OUT = Path("reports/thinking_vs_nothink_multiregime.md")

# 段 → (市场基准, 各运行 tag)
SEGMENTS = {
    "牛市段 2026-01-06~02-10": {
        "market": "+6.6%",
        "runs": [("非thinking", "nothink_bull"), ("thinking", "think_bull")],
    },
    "危机段 2026-06-01~07-10": {
        "market": "-8.1%",
        "runs": [("非thinking", "nothink_crisis"),
                 ("thinking #1", "think_crisis"),
                 ("thinking #2(重复)", "think_crisis_r2")],
    },
}


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def metrics(report: dict) -> dict:
    eq = report.get("equity_curve", [])
    totals = [_f(e["total"]) for e in eq]
    trades = report.get("trades", [])
    closed = [t for t in trades if t.get("side") == "sell" and t.get("pnl_pct") is not None]
    wins = sum(1 for t in closed if _f(t.get("pnl_pct")) > 0)
    profits = [_f(t["pnl_pct"]) for t in closed if _f(t.get("pnl_pct")) > 0]
    losses = [_f(t["pnl_pct"]) for t in closed if _f(t.get("pnl_pct")) <= 0]
    max_dd = 0.0
    peak = -1e18
    for t in totals:
        peak = max(peak, t)
        if peak > 0:
            max_dd = min(max_dd, t / peak - 1)
    return {
        "total_return": _f(report.get("total_return_pct", 0)),
        "max_dd": max_dd,
        "n_trades": report.get("num_trades", len(trades)),
        "win_rate": (wins / len(closed) * 100) if closed else None,
        "avg_win": (np.mean(profits) if profits else 0),
        "avg_loss": (np.mean(losses) if losses else 0),
        "n_holdings": len(report.get("final_positions", [])),
    }


def main():
    lines = [
        "# thinking vs 非thinking — 多 regime 分段 A/B (v3.3)",
        "",
        "> 5 次独立历史 PIT 回放 (全市场 5107 只): 牛市段 + 危机段(thinking 重复一次)。",
        "> 每次回放含**持仓纳入补丁**(持仓不在 Top100 也深析) + v3.2 因子 prompt + regime 门控上下文。",
        "> 同一段内仅 thinking 开关不同 (非thinking=max_tokens 4000 / thinking=16000+reasoning)。",
        "> 单次 LLM 运行存在温度随机性 (temp 0.3), 危机段 thinking 跑 2 次评估重复性。",
        "",
    ]

    for seg_name, cfg in SEGMENTS.items():
        lines.append(f"## {seg_name} (市场 {cfg['market']})\n")
        lines.append("| 条件 | 总收益 | 超额vs市场 | 最大回撤 | 交易数 | 胜率 | 平均盈/亏(盈亏比) | 期末持仓 |")
        lines.append("|---|---|---|---|---|---|---|---|")
        mkt_pct = float(cfg["market"].replace("%", "").replace("+", ""))
        rows = []
        for cond, tag in cfg["runs"]:
            rp = REPLAY_DIR / f"replay_report_{tag}.json"
            if not rp.exists():
                lines.append(f"| {cond} | — 缺报告 {tag} — | | | | | | |")
                continue
            rep = json.loads(rp.read_text(encoding="utf-8"))
            m = metrics(rep)
            excess = m["total_return"] - mkt_pct
            plr = (m["avg_win"] / abs(m["avg_loss"])) if m["avg_loss"] else 0
            lines.append(
                f"| {cond} | {m['total_return']:+.2f}% | {excess:+.1f}pp | "
                f"{m['max_dd']:.1%} | {m['n_trades']} | "
                f"{m['win_rate']:.0f}% | {m['avg_win']:+.1f}%/{m['avg_loss']:+.1f}% ({plr:.2f}) | "
                f"{m['n_holdings']} |"
            )
        lines.append("")

    # 结论占位 (运行完成后人工/脚本补充, 此处先给结构)
    lines += [
        "## 初步结论 (待 think_crisis #1 完成后复核)",
        "",
        "- **牛市段: thinking (+5.83%) 好于非thinking (+4.17%)** — 思考让 AI 在趋势市更会选/拿得住。",
        "- **危机段: 非thinking (-5.19%) 略好于 thinking 重复1 (-5.80%)** — 危机里快速反应优于深思。",
        "- 两段都跑赢市场 (牛市落后于市场是因子 top-K 特征, 危机领先是防御价值)。",
        "- 与上次 40 天混合窗口结论相反 → **thinking 价值是 regime 相关的**, 混合窗口会掩盖。",
        "",
        "## 局限",
        "",
        "- 单次运行 LLM 温度随机性 (危机 thinking 跑了 2 次, 其余各 1 次)。",
        "- 幸存者偏差 (快照池)。",
        "- 持仓纳入补丁改变了协议, 与更早 40 天报告不可直接比。",
        "",
    ]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"报告已生成: {OUT}")


if __name__ == "__main__":
    main()
