# -*- coding: utf-8 -*-
"""v6.1 清理: 删除审计报告确认的死码文件/目录 + 陈旧 pyc。跑一次即弃。"""
import glob
import os
import shutil

FILES = [
    # Batch 2 死码 (audit 删除清单)
    "backtest/overfitting_check.py",
    "analysis/composite_scorer.py",
    "analysis/factor_factory.py",
    "analysis/factor_evaluator.py",
    "data/cache.py",
    "data/providers/alternative.py",
    "knowledge/feedback.py",
    "scripts/flylark_crossnode_probe.py",
    "scripts/flylark_persist_probe.py",
    "scripts/flylark_platform_probe.py",
    "scripts/probe_datapro_mcp.py",
    "scripts/probe_signal_sources.py",
    "scripts/probe_other_signal_sources.py",
    "scripts/restart_fetch.py",
    "scripts/rebalance_today.py",
    "scripts/generate_demo_report.py",
    "scripts/deep_buy.py",
    "scripts/quick_screen.py",
    "scripts/rehearse_ai_input.py",
    "scripts/verify_portfolio_cf.py",
    "scripts/analyze_evolution.py",
    "scripts/analyze_evolution_noise.py",
    "scripts/analyze_master_diversity.py",
    "scripts/optimize_diagnostic_prompt.py",
    "scripts/run_user_strategy_bt.py",
    # 竞赛遗留模块 (用户 2026-09-04 确认删除; 引用点已在 server.py/schemas.py 摘除)
    "api/routers/competition.py",
    "agent/competition_agent.py",
    "scripts/competition_benchmark.py",
    "tests/test_competition_quality.py",
    "tests/competition_questions.json",
]

DIRS = ["data/storage", "agent/prompts"]

# 陈旧 pyc (源文件已删)
PYC_GLOBS = [
    "agent/sub_agents/__pycache__/bull_researcher*.pyc",
    "agent/sub_agents/__pycache__/judge*.pyc",
    "agent/sub_agents/__pycache__/bear_researcher*.pyc",
]

n = 0
for f in FILES:
    if os.path.exists(f):
        os.remove(f)
        n += 1
        print("rm   ", f)
    else:
        print("skip ", f)

for d in DIRS:
    if os.path.isdir(d):
        shutil.rmtree(d)
        n += 1
        print("rmdir", d)
    else:
        print("skip ", d)

for g in PYC_GLOBS:
    for p in glob.glob(g):
        os.remove(p)
        n += 1
        print("rm   ", p)

print(f"\nremoved: {n} items")
print("下一步: pytest tests/ -m \"not network\" → git add -A && git commit && git push")
