#!/usr/bin/env python
"""
预计算演示报告生成器 (v3.0-competition)

生成一套完整的竞赛演示数据:
  1. 市场分析报告 (Markdown)
  2. 个股分析报告 (含多空辩论)
  3. 回测绩效报告 (含图表数据)
  4. 模拟持仓快照
  5. 对话记录样例

用法:
    python scripts/generate_demo_report.py
    python scripts/generate_demo_report.py --output reports/demo/
"""

import json
import random
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def generate_market_overview() -> dict:
    """生成市场概况数据"""
    return {
        "date": date.today().isoformat(),
        "indices": {
            "上证指数": {"close": 3250.50, "change_pct": +0.32, "volume_ratio": 0.95},
            "深证成指": {"close": 11125.80, "change_pct": -0.15, "volume_ratio": 1.02},
            "创业板指": {"close": 2280.30, "change_pct": +0.68, "volume_ratio": 1.15},
        },
        "market_regime": {
            "regime": "weak_bull",
            "confidence": 0.68,
            "description": "指数温和上涨,成交量不足但北向资金持续流入",
        },
        "market_breadth": {
            "up_count": 2150, "down_count": 1830, "flat_count": 420,
            "limit_up": 45, "limit_down": 8,
            "new_high_20d": 128, "new_low_20d": 35,
        },
        "northbound_flow": {
            "today_net": "+12.5亿",
            "5d_cumulative": "+38.2亿",
            "top_buy_sectors": ["白酒", "新能源", "医药"],
            "top_sell_sectors": ["地产", "银行"],
        },
        "sector_rotation": {
            "leading": [
                {"name": "白酒", "change_5d": +4.2, "stage": "加速期"},
                {"name": "新能源", "change_5d": +3.8, "stage": "主升期"},
                {"name": "半导体", "change_5d": +2.9, "stage": "启动期"},
            ],
            "lagging": [
                {"name": "地产", "change_5d": -3.1, "stage": "退潮期"},
                {"name": "银行", "change_5d": -1.8, "stage": "退潮期"},
            ],
        },
    }


def generate_stock_analysis(symbol: str, name: str) -> dict:
    """生成个股分析数据 (含模拟多空辩论结果)"""
    rng = random.Random(hash(symbol))
    base_score = rng.uniform(5.0, 8.5)
    return {
        "symbol": symbol,
        "name": name,
        "analysis_date": date.today().isoformat(),
        "technical_indicators": {
            "close": round(rng.uniform(10, 2000), 2),
            "ma_5": round(rng.uniform(10, 2000), 2),
            "ma_20": round(rng.uniform(10, 2000), 2),
            "ma_60": round(rng.uniform(10, 2000), 2),
            "rsi_14": round(rng.uniform(25, 75), 1),
            "macd_dif": round(rng.uniform(-5, 5), 3),
            "macd_dea": round(rng.uniform(-5, 5), 3),
            "bb_pct_20": round(rng.uniform(-2, 2), 2),
            "atr_14": round(rng.uniform(0.5, 5), 2),
            "trend_score": round(rng.uniform(3, 9), 1),
            "composite_score": round(base_score, 1),
        },
        "bull_arguments": [
            {"point": "MACD即将金叉,短期动能转强",
             "data_support": f"DIF={rng.uniform(-2,2):.3f}, DEA={rng.uniform(-2,2):.3f}",
             "conviction": round(rng.uniform(0.5, 0.8), 2)},
            {"point": "RSI从低位反弹,未进入超买区",
             "data_support": f"RSI(14)={rng.uniform(35, 55):.1f}",
             "conviction": round(rng.uniform(0.4, 0.7), 2)},
            {"point": "北向资金连续3日净流入该板块",
             "data_support": f"近3日净流入{rng.uniform(1, 10):.1f}亿",
             "conviction": round(rng.uniform(0.3, 0.6), 2)},
        ],
        "bear_arguments": [
            {"point": "成交量萎缩,上涨缺乏动力",
             "data_support": f"量比{rng.uniform(0.5, 0.9):.2f}",
             "conviction": round(rng.uniform(0.4, 0.7), 2)},
            {"point": "大盘处于弱牛格局,个股易受指数拖累",
             "data_support": "市场体制: weak_bull (置信度68%)",
             "conviction": round(rng.uniform(0.3, 0.6), 2)},
        ],
        "verdict": {
            "action": "BUY" if base_score >= 6.5 else ("WATCH" if base_score >= 5.5 else "HOLD"),
            "conviction": round(rng.uniform(0.5, 0.8), 2),
            "score": round(base_score, 1),
            "key_reasons": [
                "技术面多指标共振,短期反弹概率较大",
                "北向资金持续流入提供资金面支撑",
            ],
            "risks": [
                "大盘弱牛格局下,个股行情持续性存疑",
                "成交量萎缩可能预示反弹力度不足",
            ],
            "verdict_summary": "技术面偏多但量能不足,建议轻仓参与,严格止损",
        },
    }


def generate_backtest_report(strategy_id: str, strategy_name: str) -> dict:
    """生成回测绩效报告"""
    rng = random.Random(hash(strategy_id))
    return {
        "strategy_id": strategy_id,
        "strategy_name": strategy_name,
        "backtest_period": "2024-01-01 ~ 2026-07-29",
        "metrics": {
            "total_return_pct": round(rng.uniform(-10, 60), 1),
            "annual_return_pct": round(rng.uniform(-5, 25), 1),
            "win_rate": round(rng.uniform(0.40, 0.65), 3),
            "sharpe_ratio": round(rng.uniform(0.3, 2.0), 2),
            "max_drawdown_pct": round(rng.uniform(-35, -8), 1),
            "profit_factor": round(rng.uniform(0.8, 2.5), 2),
            "total_trades": rng.randint(30, 200),
            "avg_holding_days": round(rng.uniform(3, 25), 1),
        },
        "monthly_returns": [
            {"month": f"2026-{m:02d}", "return_pct": round(rng.uniform(-8, 12), 1)}
            for m in range(1, 8)
        ],
        "overfitting_checks": {
            "PBO": round(rng.uniform(0.1, 0.5), 2),
            "DSR": round(rng.uniform(0.3, 1.5), 2),
            "walk_forward_consistency": round(rng.uniform(0.4, 0.9), 2),
        },
    }


def generate_portfolio_snapshot() -> dict:
    """生成模拟持仓快照"""
    holdings = [
        {"symbol": "sh.600519", "name": "贵州茅台", "shares": 200, "avg_cost": 1650.00,
         "current_price": 1680.00, "weight": 0.28, "pnl_pct": +1.82, "holding_days": 15},
        {"symbol": "sz.000858", "name": "五粮液", "shares": 500, "avg_cost": 148.00,
         "current_price": 152.50, "weight": 0.18, "pnl_pct": +3.04, "holding_days": 8},
        {"symbol": "sz.300750", "name": "宁德时代", "shares": 300, "avg_cost": 195.00,
         "current_price": 188.50, "weight": 0.15, "pnl_pct": -3.33, "holding_days": 22},
        {"symbol": "sh.600036", "name": "招商银行", "shares": 1000, "avg_cost": 38.50,
         "current_price": 39.20, "weight": 0.12, "pnl_pct": +1.82, "holding_days": 5},
    ]
    total_value = sum(h["shares"] * h["current_price"] for h in holdings)
    total_cost = sum(h["shares"] * h["avg_cost"] for h in holdings)
    return {
        "date": date.today().isoformat(),
        "total_value": total_value,
        "total_cost": total_cost,
        "total_pnl": total_value - total_cost,
        "total_pnl_pct": round((total_value / total_cost - 1) * 100, 2),
        "cash": total_value * 0.15,
        "holdings": holdings,
        "risk_metrics": {
            "max_drawdown_pct": -5.2,
            "var_95_daily": -2.8,
            "sharpe_ratio": 1.15,
            "sector_concentration": {"白酒": 0.46, "新能源": 0.15, "银行": 0.12},
        },
    }


def generate_chat_samples() -> list:
    """生成对话记录样例"""
    return [
        {
            "id": "chat_001",
            "user": "当前A股市场处于什么状态？",
            "assistant": (
                "### 当前A股市场状态分析\n\n"
                "**现状**: 上证指数3250点(+0.32%),深证成指11125点(-0.15%),"
                "创业板指2280点(+0.68%)。市场体制: weak_bull (置信度68%)\n\n"
                "**分析**: 从4个维度来看:\n"
                "1. 技术面: 上证站稳3200点,但3250-3300区间压力较大\n"
                "2. 资金面: 北向资金近5日净流入38.2亿,聪明钱偏乐观\n"
                "3. 情绪面: 涨跌比2150:1830,涨停45家,市场情绪偏暖\n"
                "4. 板块面: 白酒(+4.2%)和新能源(+3.8%)领涨,地产(-3.1%)退潮\n\n"
                "**建议**: 当前适合轻仓参与,重点配置领涨板块中的龙头标的\n\n"
                "**风险**: ⚠️ 1)成交量不足可能限制上涨空间 2)弱牛格局下板块轮动快\n"
                "历史数据不代表未来表现,本分析仅供参考,不构成投资建议。"
            ),
        },
        {
            "id": "chat_002",
            "user": "帮我分析一下600519贵州茅台",
            "assistant": (
                "### 600519 贵州茅台 技术分析\n\n"
                "**现状**: 当前价格1680元。RSI(14)=45处于中性区间,MACD DIF=-1.234即将上穿DEA=-1.567,"
                "布林带%B=0.35处于中下轨。综合评分7.2/10\n\n"
                "**分析**:\n"
                "- 趋势面(7/10): 价格站在20日和60日均线上方,短期均线多头排列\n"
                "- 动量面(6/10): RSI从30反弹至45,MACD即将金叉,短期动能好转\n"
                "- 量价面(5/10): 成交量偏低,量比0.78,上涨缺乏成交量确认\n\n"
                "**多空辩论**:\n"
                "- 多头: MACD金叉+RSI低位反弹→技术性反弹需求\n"
                "- 空头: 量能不足+板块退潮→反弹高度有限\n"
                "- 裁判: 中性偏多,建议轻仓参与\n\n"
                "**建议**: WATCH→BUY,入场1680附近,止损1600(-4.8%),止盈1780(+6.0%)\n\n"
                "**风险**: ⚠️ 1)白酒板块进入退潮期 2)大盘成交量不足可能拖累反弹\n"
                "历史数据不代表未来表现,本分析仅供参考,不构成投资建议。"
            ),
        },
    ]


def generate_full_report(output_dir: str = "reports/demo"):
    """生成完整的演示数据包"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    report_data = {
        "title": "AI+金融量化分析智能体 — 竞赛演示数据包",
        "generated_at": datetime.now().isoformat(),
        "version": "3.0-competition",
        "market_overview": generate_market_overview(),
        "stock_analyses": {
            "sh.600519": generate_stock_analysis("sh.600519", "贵州茅台"),
            "sz.000858": generate_stock_analysis("sz.000858", "五粮液"),
            "sz.300750": generate_stock_analysis("sz.300750", "宁德时代"),
        },
        "backtest_reports": {
            "macd_trend": generate_backtest_report("macd_trend", "MACD趋势策略"),
            "dual_ma_trend": generate_backtest_report("dual_ma_trend", "双均线策略"),
            "bollinger_reversal": generate_backtest_report("bollinger_reversal", "布林带反转策略"),
        },
        "portfolio_snapshot": generate_portfolio_snapshot(),
        "chat_samples": generate_chat_samples(),
    }

    # 保存完整JSON
    json_path = output_path / "demo_data.json"
    json_path.write_text(json.dumps(report_data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"✅ 完整数据: {json_path} ({json_path.stat().st_size:,} bytes)")

    # 生成可读的Markdown报告
    md = []
    md.append("# A股量化分析日报\n")
    md.append(f"> 生成日期: {date.today().isoformat()} | AI+金融量化分析智能体 v3.0\n")

    # 市场概况
    mo = report_data["market_overview"]
    md.append("## 一、市场概况\n")
    md.append(f"- **市场体制**: {mo['market_regime']['regime']} (置信度 {mo['market_regime']['confidence']:.0%})")
    md.append(f"- **上证指数**: {mo['indices']['上证指数']['close']} ({mo['indices']['上证指数']['change_pct']:+.2f}%)")
    md.append(f"- **涨跌比**: {mo['market_breadth']['up_count']}:{mo['market_breadth']['down_count']}")
    md.append(f"- **北向资金**: {mo['northbound_flow']['today_net']} (5日累计 {mo['northbound_flow']['5d_cumulative']})")
    md.append(f"- **领涨板块**: {', '.join(s['name'] for s in mo['sector_rotation']['leading'])}")
    md.append("")

    # 个股分析
    md.append("## 二、个股分析\n")
    for sym, analysis in report_data["stock_analyses"].items():
        v = analysis["verdict"]
        md.append(f"### {analysis['name']} ({sym}) — {v['action']} (评分 {v['score']:.1f}/10)\n")
        md.append(f"**裁判意见**: {v['verdict_summary']}")
        md.append(f"**关键理由**: {', '.join(v['key_reasons'])}")
        md.append(f"**风险提示**: {', '.join(v['risks'])}")
        md.append("")

    # 交易建议表格
    md.append("## 三、交易建议\n")
    md.append("| 标的 | 方向 | 评分 | 置信度 | 关键风险 |")
    md.append("|------|------|------|--------|----------|")
    for sym, analysis in report_data["stock_analyses"].items():
        v = analysis["verdict"]
        md.append(f"| {analysis['name']} | {v['action']} | {v['score']:.1f} | {v['conviction']:.0%} | {v['risks'][0][:30]} |")
    md.append("")

    # 回测绩效
    md.append("## 四、策略回测绩效\n")
    md.append("| 策略 | 年化收益 | 胜率 | 夏普 | 最大回撤 | 盈亏比 |")
    md.append("|------|---------|------|------|----------|--------|")
    for sid, bt in report_data["backtest_reports"].items():
        m = bt["metrics"]
        md.append(f"| {bt['strategy_name']} | {m['annual_return_pct']:+.1f}% | {m['win_rate']:.0%} | {m['sharpe_ratio']:.2f} | {m['max_drawdown_pct']:.1f}% | {m['profit_factor']:.2f} |")
    md.append("")

    # 持仓快照
    ps = report_data["portfolio_snapshot"]
    md.append("## 五、模拟持仓\n")
    md.append(f"- **总资产**: ¥{ps['total_value']:,.2f}")
    md.append(f"- **总盈亏**: ¥{ps['total_pnl']:+,.2f} ({ps['total_pnl_pct']:+.2f}%)")
    md.append(f"- **当前回撤**: {ps['risk_metrics']['max_drawdown_pct']:.1f}%")
    md.append("")

    # 风险提示
    md.append("## ⚠️ 风险提示\n")
    md.append("> 历史数据不代表未来表现。本文所有分析结果基于量化模型和历史数据,仅供参考,不构成投资建议。")
    md.append("> 投资有风险,入市需谨慎。请在充分了解风险的前提下做出独立判断。")

    md_path = output_path / "demo_report.md"
    md_path.write_text("\n".join(md), encoding="utf-8")
    print(f"✅ Markdown报告: {md_path} ({md_path.stat().st_size:,} bytes)")

    # 对话样例
    chat_path = output_path / "chat_samples.json"
    chat_path.write_text(json.dumps(report_data["chat_samples"], ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"✅ 对话样例: {chat_path} ({chat_path.stat().st_size:,} bytes)")

    print(f"\n📁 所有文件已保存到: {output_path}/")
    return report_data


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="生成竞赛演示数据")
    parser.add_argument("--output", default="reports/demo", help="输出目录")
    args = parser.parse_args()
    generate_full_report(args.output)
