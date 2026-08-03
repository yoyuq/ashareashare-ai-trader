"""因子研究模块 (v3.2) — 评判哪些因子真正提高 AI 判断

方法: 因子 IC 快速筛选 — 对历史窗口每个交易日, 计算每只股票的因子分,
与 N 日前向收益的 Spearman 相关 (IC)。IC>0 且显著 → 因子有预测力, 值得注入 LLM;
IC≈0 或负 → 剔除。比全量 LLM A/B 便宜且可复现。

结构:
  sources.py            单股截面因子注册表 (baseline 8 + wave1 13) + 因子分组
  panels.py             向量化因子面板计算 (跨时间轴, 评估端用)
  evaluate.py           截面 IC 评估 CLI
  market_situation.py   市场形势因子 (全市场级信号 + 等权市场指数)
  evaluate_market.py    市场形势因子时间序列评估 CLI
  market_sentiment.py   市场情绪温度计 (恐慌/贪婪复合, 可复算代理) + live 快照版
  evaluate_sentiment.py 情绪温度计评估 CLI (时间序列相关 + 极端区制条件收益)
  regime_analysis.py    regime 自适应验证: 合成指数判定 regime → 分桶 IC 方向稳定性
  run_comparison.py     基线 vs 第二波对比 → reports/factor_comparison.md
  backtest.py           因子选股回测 (AShareBroker 次日开盘成交) → reports/factor_backtest.md
                        --replay 可选长窗口多 regime 回放

用法:
  python -m factors.evaluate --days 60 --group all
  python -m factors.evaluate_market --days 60
  python -m factors.evaluate_sentiment --days 80
  python -m factors.regime_analysis --replay replay_data/daily_2025-10-08_2026-07-31.parquet
  python -m factors.run_comparison --days 60
  python -m factors.backtest --topk 20 --rebalance 5 --warmup 25
  python -m factors.backtest --replay replay_data/daily_2025-10-08_2026-07-31.parquet
"""

from factors.sources import compute_factors, factor_list  # noqa: F401
