"""OPRO 提示词优化 — 用历史决策日志优化诊断官系统提示词。

用法:
    python -m scripts.optimize_diagnostic_prompt --rounds 5 --samples 20
    python -m scripts.optimize_diagnostic_prompt --journal replay_data/journal_diag_v40_evo.jsonl
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()


def build_eval_samples(journal_path: str, max_samples: int = 30) -> list[dict]:
    """从决策日志构造评估样本。

    每条已复盘的记录 = 一个评估样本:
    - user_msg: 当时的市场形势（用于调用诊断官）
    - optimal_risk: 最优风险等级（原风险 + 偏差的相反数）
    - direction: 次日市场方向（up/down/flat）
    """
    from agent.evolution.decision_journal import DecisionJournal

    journal = DecisionJournal(journal_path)
    reviewed = [r for r in journal._cache.values() if r.review]

    samples = []
    for rec in reviewed[:max_samples]:
        snap = rec.market_snapshot or {}
        # v5.1 对齐生产 user_msg 格式 (结构与 closing 与 get_market_diagnostic 的 _diagnostic_user_msg 一致),
        # 避免在非代表性输入上调优 → 优化出的 prompt 在真实决策路径上最优.
        snapshot_txt = (
            f"【市场广度 · 单日快照】\n"
            f"  今日: 上涨{snap.get('up_ratio', 0.5):.1%} "
            f"涨停{snap.get('limit_up', 0)}家 跌停{snap.get('limit_down', 0)}家 "
            f"中位PE{snap.get('med_pe', 0):.1f} 成交额{snap.get('total_amt_yi', snap.get('total_amt', 0)):.0f}亿\n"
            f"\n【中期位置】\n"
            f"市场状态: {rec.regime}\n"
            f"拥挤度: {snap.get('crowding_signal', 'unknown')} (score {snap.get('crowding_score', 50.0):.2f})"
        )
        from simulation.daily_runner import _diagnostic_user_msg
        user_msg = _diagnostic_user_msg(snapshot_txt)

        dev = rec.review.get("risk_level_deviation", 0)
        # dev > 0: 当时给高了（踏空）→ 最优应该更低（更激进）
        # dev < 0: 当时给低了（被套）→ 最优应该更高（更保守）
        optimal = max(1, min(5, rec.risk_level - int(dev)))

        # 方向判断（从偏差反推）
        if dev > 0:
            direction = "up"      # 给高了 = 市场涨了
        elif dev < 0:
            direction = "down"    # 给低了 = 市场跌了
        else:
            direction = "flat"

        samples.append({
            "date": rec.date,
            "user_msg": user_msg,
            "optimal_risk": optimal,
            "direction": direction,
            "original_risk": rec.risk_level,
            "deviation": dev,
            "master": rec.dominant_master,
        })

    return samples


async def main():
    parser = argparse.ArgumentParser(description="OPRO 优化诊断官提示词")
    parser.add_argument("--journal", type=str,
                        default="replay_data/journal_diag_v40_evo.jsonl",
                        help="决策日志路径")
    parser.add_argument("--rounds", type=int, default=5, help="优化轮数")
    parser.add_argument("--samples", type=int, default=20, help="每轮评估样本数")
    parser.add_argument("--output", type=str, default="replay_data/prompt_optimization",
                        help="输出目录")
    args = parser.parse_args()

    # 1. 构造评估样本
    print(f"从 {args.journal} 加载评估样本...")
    all_samples = build_eval_samples(args.journal, max_samples=args.samples * 2)
    # 打乱，取前 N 个（避免都是同一段行情的偏差）
    import random
    random.seed(42)
    random.shuffle(all_samples)
    eval_samples = all_samples[:args.samples]
    print(f"  评估样本数: {len(eval_samples)}")

    # 统计样本方向分布
    dirs = {}
    for s in eval_samples:
        d = s["direction"]
        dirs[d] = dirs.get(d, 0) + 1
    print(f"  方向分布: {dirs}")

    # 2. 加载初始提示词（当前诊断官的系统提示词）
    from simulation.daily_runner import _DIAGNOSTIC_SYSTEM_PROMPT
    initial_prompt = _DIAGNOSTIC_SYSTEM_PROMPT
    print(f"\n初始提示词长度: {len(initial_prompt)} 字符")

    # 3. 运行 OPRO 优化
    from agent.evolution.prompt_optimizer import PromptOptimizer

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    optimizer = PromptOptimizer(
        initial_prompt=initial_prompt,
        eval_samples=eval_samples,
        output_dir=str(out_dir),
        max_rounds=args.rounds,
        variants_per_round=3,
    )

    print(f"\n开始 {args.rounds} 轮 OPRO 优化...")
    history = await optimizer.optimize()

    # 4. 输出结果
    print(f"\n{'='*60}")
    print(f"优化完成!")
    print(f"  最佳得分: {history.best_score:.3f} (第{history.best_round}轮)")
    print(f"  最佳提示词已保存: {out_dir / 'best_prompt.txt'}")
    print(f"  优化历史已保存: {out_dir / 'optimization_history.json'}")

    # 显示各轮对比
    print(f"\n各轮得分:")
    for rnd in history.rounds:
        print(f"  第{rnd['round']}轮: 最佳={rnd['best_score']:.3f} ({rnd['best_this_round']})")

    # 显示最佳提示词的前 500 字
    print(f"\n最佳提示词预览:")
    print(history.best_prompt[:500])
    print("...")


if __name__ == "__main__":
    asyncio.run(main())
