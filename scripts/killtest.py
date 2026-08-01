"""
kill-test CLI — 三路信号 (LLM/规则/随机) 前向收益对比

验证核心问题: 信号是否有真实 alpha (N日前向收益是否显著优于随机/全市场基线)。

用法:
    python scripts/killtest.py --history 250 --arms rule,random,llm --lookahead 5
    python scripts/killtest.py --universe sh.600519,sz.000858 --arms rule,random --history 500
    python scripts/killtest.py --seed 7 --data-dir killtest_data

说明:
    - rule   : 纯技术规则, 可回溯立即出结果
    - random : 有种子随机, 与规则臂同数量 (控制基准)
    - llm    : 读取 reports/data_{date}.json, 需系统日跑积累 (历史日期无文件则无信号)
"""

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    ap = argparse.ArgumentParser(description="kill-test 三路信号对比")
    ap.add_argument("--history", type=int, default=250, help="回看交易日数 (默认250)")
    ap.add_argument("--universe", type=str, default="",
                    help="股票池, 逗号分隔 (默认内置20只流动性股)")
    ap.add_argument("--arms", type=str, default="rule,random,llm",
                    help="对比臂: rule,random,llm 逗号分隔")
    ap.add_argument("--lookahead", type=int, default=5, help="N日后回看 (默认5)")
    ap.add_argument("--seed", type=int, default=42, help="随机臂种子")
    ap.add_argument("--data-dir", type=str, default="killtest_data", help="信号/结果持久化目录")
    ap.add_argument("--report-dir", type=str, default="reports", help="报告输出目录")
    args = ap.parse_args()

    symbols = [s.strip() for s in args.universe.split(",") if s.strip()] or None
    arms = tuple(a.strip() for a in args.arms.split(",") if a.strip())

    print(f"kill-test: 窗口={args.history}日 lookahead={args.lookahead} "
          f"arms={arms} seed={args.seed} 股票池={len(symbols) if symbols else 20}只")

    from killtest.runner import run_killtest

    result = asyncio.run(run_killtest(
        history_days=args.history, symbols=symbols, arms=arms,
        lookahead=args.lookahead, seed=args.seed,
        data_dir=args.data_dir, report_dir=args.report_dir,
    ))

    if "error" in result:
        print(f"❌ {result['error']}")
        sys.exit(1)

    print("\n" + result["report"])
    print(f"\n新发出信号: {result['signal_counts']} | 全市场基线均值: "
          f"{result['baseline_mean']*100:.2f}% (n={result['baseline_n']})")
    print(f"报告已写入: {Path(args.report_dir) / 'killtest_report.md'}")


if __name__ == "__main__":
    main()
