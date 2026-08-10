# 自我进化系统 (v4.0)

让交易诊断官从自己的决策历史中学习，持续优化。

## 架构

```
事实层 → 反思层 → 知识层 → 智慧层
  ↑                              ↑
  │         反事实验证           │
  └────── (质量控制层) ──────────┘
```

| 层级 | 模块 | 作用 | 更新频率 |
|------|------|------|----------|
| 事实层 | `decision_journal.py` | 记录每天的诊断决策 | 每日 |
| 反思层 | `daily_review.py` | 次日回看，自我批评，提取经验 | 每日 |
| 知识层 | `experience_memory.py` | 经验记忆库，动态注入提示词 | 每日（追加） |
| 智慧层 | `weekly_evolution.py` | 定期汇总，提炼长期原则 | 每 10 天 |
| 质量层 | `counterfactual.py` | 反事实验证教训是否真有用 | 按需 |
| 优化层 | `prompt_optimizer.py` | OPRO式自动优化系统提示词 | 离线（人工触发） |

## 数据流

```
每日决策 → DecisionJournal（存盘）
   ↓
次日复盘 → review_decision() → 生成复盘结论
   ↓
提取经验 → extract_experience() → ExperienceItem
   ↓
存入记忆库 → ExperienceMemory.add()（自动去重/合并）
   ↓
次日诊断前 → 检索相关经验 → 注入系统提示词
   ↓
每10天 → EvolutionManager.evolve() → 提炼核心原则
   ↓
后续诊断 → 核心原则注入提示词，指导长期决策
```

## 使用方法

### 在历史回放中启用

```bash
python scripts/historical_replay.py --diagnostic --diag-top-n 30 --evolution --tag my_evo_run
```

开启后会生成三个文件（在 `replay_data/` 目录）：
- `journal_{tag}.jsonl` — 决策日志（每日一条）
- `memory_{tag}.json` — 经验记忆库
- `evolution_{tag}.json` — 进化总结历史

### 在实盘/每日运行中启用

```python
from agent.evolution.decision_journal import DecisionJournal
from agent.evolution.experience_memory import ExperienceMemory
from agent.evolution.weekly_evolution import EvolutionManager

# 初始化
journal = DecisionJournal("simulation_data/diag_journal.jsonl")
memory = ExperienceMemory("simulation_data/diag_memory.json")
evolution = EvolutionManager("simulation_data/diag_evolution.json")

# 诊断时传入（让诊断官能看到历史经验）
diagnosis = await _market_diagnostic(
    df_cs, regime, crowd,
    memory=memory, evolution=evolution, current_date=today_str
)

# 诊断后存日志
journal.record(DecisionRecord(date=today_str, ...))

# 第二天：复盘昨天的决策
from agent.evolution.daily_review import review_decision, extract_experience
review = await review_decision(yesterday_record, today_stats, today_market_move)
journal.update_review(yesterday_date, review)
experience = extract_experience(yesterday_record, review)
if experience:
    memory.add(experience)

# 每10天：进化总结
if evolution.should_evolve(today_str, len(journal)):
    reviewed = [d for d in journal.load_range(start, end) if d.review]
    await evolution.evolve(reviewed, memory.items)
```

### 提示词自动优化（离线）

```python
from agent.evolution.prompt_optimizer import PromptOptimizer

# 从决策日志构建验证集
# samples = [...]  # 每个样本含 user_msg + optimal_risk + direction

optimizer = PromptOptimizer(
    initial_prompt=base_prompt,
    eval_samples=samples[:50],  # 选50个代表性样本
    output_dir="replay_data/prompt_optim/",
    max_rounds=10,
    variants_per_round=3,
)
history = await optimizer.optimize()
print(f"最佳得分: {history.best_score:.3f}")
```

## 设计原则

### 1. PIT 严格性
- 复盘只能用 T+1 及以后的数据
- 经验记忆只在"下一次决策前"注入，不影响已做出的决策
- 进化总结也是用已有的历史数据，不偷看未来

### 2. 结构化 > 自由文本
所有经验、复盘、原则都是结构化的（JSON），不是散文。好处：
- 便于检索和过滤
- 便于量化评估
- 避免 LLM 输出空洞的鸡汤

### 3. 记忆衰减
越老的经验权重越低，半衰期 60 天。防止过时经验主导决策。

### 4. 正反经验都存
成功的经验（verdict=correct）和失败的教训（verdict=wrong）同样重要。
LLM 从成功中学到"该怎么做"，从失败中学到"不该怎么做"。

### 5. 可解释性
每条经验都能追溯到原始决策记录和复盘结论。
进化不是黑箱，每一步都有依据。

## 与前沿研究的对应

| 研究方向 | 本系统中的实现 |
|----------|---------------|
| Reflexion (Shinn et al., 2023) | `daily_review.py` 每日复盘 |
| Experience Replay / Memory Bank | `experience_memory.py` 经验记忆库 |
| OPRO / PromptBreeder | `prompt_optimizer.py` 提示词自动优化 |
| Hindsight Experience Replay | `counterfactual.py` 反事实验证 |
| Voyager 终身学习 Agent | 整体架构（记忆+技能+进化） |

## 已知局限

1. **复盘依赖 LLM 的归因能力** — LLM 可能错误归因，导致学错教训。
   对策：反事实验证（`counterfactual.py`）验证教训是否真有用。

2. **记忆容量有限** — 提示词窗口有限，最多注入 5-6 条经验。
   对策：精准检索（按场景匹配 + 时间衰减 + 置信度排序）。

3. **进化速度慢** — 日级决策，积累足够经验需要几周甚至几个月。
   对策：先用历史回测"预训练"经验库，上线后继续学习。

4. **过拟合风险** — 在一段行情上学到的经验换到另一段可能失效。
   对策：时间衰减 + 定期遗忘 + 多 regime 验证。

## 未来方向

- [ ] 反事实验证集成到每日复盘流程（自动验证，自动升级置信度）
- [ ] 高置信度教训自动转化为硬规则（代码层面强制执行）
- [ ] 程序层记忆（把经验转化为可配置参数，不是文字）
- [ ] 多目标进化（同时优化收益、回撤、夏普，不是单一指标）
- [ ] 策略生成（不只是调仓，还能生成新的选股策略）
