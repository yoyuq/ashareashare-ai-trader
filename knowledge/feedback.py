"""交易反馈闭环 (v3.3) — 从回放/实盘结果沉淀 regime 经验, 写回知识库供 AI 检索

连续优化机制: 每次跑完历史回放或每日实盘后运行本模块,
  1. 读取 replay_data/replay_report_*.json (日度权益 + day_logs 带 regime) + portfolio.json
  2. 聚合各 regime 的日度收益 / 平仓胜率 / 平均盈亏
  3. 生成 knowledge/reference/trade_lessons.md (经验沉淀)
  4. 索引进 ChromaDB knowledge_base_v2 (index_document)
让后续 AI 分析能检索到"历史上哪种策略在哪个市场状态有效"的实证。

用法: python -m knowledge.feedback   (或 scripts/update_knowledge.py)
"""

import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from knowledge.manager import KnowledgeManager

ROOT = Path(__file__).parent.parent
REPLAY_DIR = ROOT / "replay_data"
PORTFOLIO = ROOT / "simulation_data" / "portfolio.json"
LESSONS_FILE = ROOT / "knowledge" / "reference" / "trade_lessons.md"

REGIME_LABEL = {"strong_bull": "强牛", "weak_bull": "弱牛", "range_bound": "震荡",
                "weak_bear": "弱熊", "strong_bear": "强熊", "crisis": "危机"}


def _load_reports() -> list:
    """读取所有回放报告 (replay_report_*.json)。"""
    reports = []
    for f in sorted(REPLAY_DIR.glob("replay_report_*.json")):
        try:
            reports.append(json.loads(f.read_text(encoding="utf-8")))
        except Exception:
            pass
    return reports


def _load_portfolio_trades() -> list:
    try:
        pf = json.loads(PORTFOLIO.read_text(encoding="utf-8"))
        return pf.get("trade_history", [])
    except Exception:
        return []


def _regime_map(report: dict) -> dict:
    """从 day_logs 建立 date→regime。"""
    return {dl["date"]: dl.get("regime", "range_bound") for dl in report.get("day_logs", [])}


def generate_trade_lessons() -> str:
    """聚合回放 + 实盘 → 生成 trade_lessons.md 文本。"""
    reports = _load_reports()
    portfolio_trades = _load_portfolio_trades()

    # ── 1. 日度收益按 regime 归因 (回放) ──
    daily = {}  # regime → {n, sum_ret, days}
    for rep in reports:
        reg_map = _regime_map(rep)
        eq = rep.get("equity_curve", [])
        for i in range(1, len(eq)):
            d = eq[i].get("date", "")
            prev = eq[i - 1].get("date", "")
            regime = reg_map.get(d) or reg_map.get(prev) or "range_bound"
            try:
                cur = float(eq[i].get("total", 0))
                before = float(eq[i - 1].get("total", 1)) or 1.0
            except (TypeError, ValueError):
                continue
            ret = cur / before - 1
            s = daily.setdefault(regime, {"n": 0, "sum": 0.0})
            s["n"] += 1
            s["sum"] += ret

    # ── 2. 平仓交易按 regime 归因 (回放 trades) ──
    trades = {}
    for rep in reports:
        reg_map = _regime_map(rep)
        for t in rep.get("trades", []):
            if t.get("side") != "sell" or t.get("pnl_pct") is None:
                continue
            regime = reg_map.get(t.get("date", ""), "range_bound")
            pnl = float(t.get("pnl_pct", 0))
            s = trades.setdefault(regime, {"n": 0, "wins": 0, "pnls": []})
            s["n"] += 1
            s["wins"] += 1 if pnl > 0 else 0
            s["pnls"].append(pnl)

    # 实盘 (portfolio) 交易也并入 (无 regime 标记, 归入 "live")
    if portfolio_trades:
        s = trades.setdefault("live", {"n": 0, "wins": 0, "pnls": []})
        for t in portfolio_trades:
            pnl = t.get("pnl_pct")
            if pnl is None:
                continue
            s["n"] += 1
            s["wins"] += 1 if float(pnl) > 0 else 0
            s["pnls"].append(float(pnl))

    # ── 3. 生成 markdown ──
    lines = [
        "# 交易经验沉淀 (trade_lessons) — 自动生成, 勿手改",
        "",
        f"> 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"> 数据来源: {len(reports)} 份回放报告 + 实盘 trade_history",
        "",
        "## 各 regime 日度表现 (回放)",
        "",
        "| regime | 天数 | 日均收益 |",
        "|---|---|---|",
    ]
    for regime in sorted(daily):
        s = daily[regime]
        avg = s["sum"] / max(s["n"], 1)
        lines.append(f"| {REGIME_LABEL.get(regime, regime)} | {s['n']} | {avg:+.2%} |")

    lines += [
        "",
        "## 各 regime 平仓交易表现 (回放 + 实盘)",
        "",
        "| regime | 平仓数 | 胜率 | 平均盈/亏(盈亏比) |",
        "|---|---|---|---|",
    ]
    for regime in sorted(trades):
        s = trades[regime]
        wr = s["wins"] / max(s["n"], 1)
        wins = [p for p in s["pnls"] if p > 0]
        losses = [p for p in s["pnls"] if p <= 0]
        avg_w = (sum(wins) / len(wins)) if wins else 0.0
        avg_l = (sum(losses) / len(losses)) if losses else 0.0
        plr = (avg_w / abs(avg_l)) if avg_l else 0.0
        lines.append(f"| {REGIME_LABEL.get(regime, regime)} | {s['n']} | {wr:.0%} | {avg_w:+.1f}%/{avg_l:+.1f}% ({plr:.2f}) |")

    lines += [
        "",
        "## 经验要点 (由数据归纳, 供 AI 检索参考)",
        "",
        "- 日度表现最强的 regime 应作为操作倾向参考; 表现最弱的 regime 应提高风控等级。",
        "- 若某 regime 胜率显著低于 40%, 该状态应默认降仓位、收紧止损、优先防御。",
        "- 本页随每次回放/实盘自动更新 — 连续优化闭环的一部分。",
        "",
    ]
    return "\n".join(lines)


def run() -> bool:
    """生成 trade_lessons.md 并索引进 ChromaDB。"""
    text = generate_trade_lessons()
    LESSONS_FILE.write_text(text, encoding="utf-8")
    km = KnowledgeManager(str(ROOT / "knowledge"))
    ok = km.index_document(LESSONS_FILE, doc_type="trade_lessons")
    print(f"经验文档: {LESSONS_FILE} ({len(text)} 字符) | 索引进向量库: {'成功' if ok else '失败'}")
    return ok


if __name__ == "__main__":
    run()
