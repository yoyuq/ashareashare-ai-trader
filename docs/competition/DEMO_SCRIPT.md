# 90分钟比赛演示剧本

> AI+金融量化分析智能体 — CompetitionAgent
>
> 版本: v1.0 | 2026-07-28

---

## 时间分配总览

| 时间段 | 模块 | 时长 | 内容 |
|--------|------|------|------|
| 0:00-0:15 | 模块1 | 15min | AI智能体架构设计展示 |
| 0:15-0:35 | 模块2 | 20min | AI知识库搭建展示 |
| 0:35-1:05 | 模块3 | 30min | AI智能体搭建展示 |
| 1:05-1:25 | 模块4 | 20min | AI应用测试展示 |
| 1:25-1:30 | 总结 | 5min | 总结与展望 |

---

## 详细剧本

### 模块1: AI智能体架构设计 (15分钟)

**0:00-0:03 (3min) — 开场**
- 自我介绍 + 赛道说明
- 一句话概括: "我们构建了一个AI+金融量化分析智能体,通过7个专业Agent协作,为A股投资者提供从市场扫描到交易执行的全链路量化决策支持"

**0:03-0:10 (7min) — 架构展示**
- 打开 `agent/ARCHITECTURE.md` → 展示Mermaid架构图
- 讲解 Multi-Agent 协作模式:
  - "ChatAssistant负责自然语言理解,将用户问题转化为工具调用"
  - "TechnicalAnalyst负责130+指标计算和解读"
  - "Bull/Bear/Judge三人辩论组负责对抗论证"
  - "RiskAssessor负责8层风控检查"
- 展示7节点LangGraph分析流水线

**0:10-0:15 (5min) — 业务价值说明**
- 需求分析: 3类用户画像 (个人投资者/量化研究员/投资顾问)
- 核心场景: 每日盘后分析 → 选股决策 → 风险预警 → 知识学习
- 输出成果: 6级信号 + 结构化日报 + 回测绩效 + 风控指标

---

### 模块2: AI知识库搭建 (20分钟)

**0:15-0:18 (3min) — 知识库架构**
- 展示三层知识体系: YAML规则 → MD文档 → ChromaDB向量库
- "知识不是硬编码在代码里,而是结构化存储在知识库中"
- "提示词通过 {placeholder} 自动注入,修改知识库文件即时生效"

**0:18-0:25 (7min) — 知识库内容展示**
- 打开 `knowledge/rules/trading_rules.yaml` → A股交易规则
- 打开 `knowledge/rules/indicator_guide.yaml` → 指标手册
- 打开 `knowledge/strategies/registry.yaml` → 9种策略注册表
- 打开 `knowledge/reference/glossary.md` → 45个A股术语
- 打开 `knowledge/reference/fundamental_analysis.md` → 基本面估值阈值
- 打开 `knowledge/reference/competition_rules.md` → 竞赛场景约束

**0:25-0:32 (7min) — 向量检索演示**
- 演示 ChromaDB 向量检索功能
- "你可以问'历史上有没有类似现在的K线形态',系统会在向量库中检索最相似的形态"
- 展示 RAG 检索: 语义搜索 + 关键词fallback

**0:32-0:35 (3min) — Few-shot样例**
- 展示4个Few-shot场景: 信号复核/恐慌抛售/板块轮动/止损执行
- "Few-shot让LLM在面临相似场景时能够参考历史决策"

---

### 模块3: AI智能体搭建 (30分钟) [最高分]

**0:35-0:41 (6min) — 提示词工程**
- 展示8个系统提示词 (每个Agent独立角色定义)
- "每个提示词都有YAML版本头,支持版本管理和变更追踪"
- 展示 `knowledge/prompts/CHANGELOG.md`
- "提示词不是一次写死的,而是通过评估框架持续优化的"
- 打开 `knowledge/prompts/evaluator.py` → 4维度评估体系

**0:41-0:50 (9min) — LLM配置与模型路由**
- 展示3层模型路由架构
- "60%的任务由本地Ollama免费完成,30%走DeepSeek Flash(¥1-2/M),10%走DeepSeek Pro(¥3-6/M)"
- "日预算控制在¥1以内,月预算¥15"
- "高峰时段自动降级,确保服务稳定"
- 打开 `config/model_config.yaml` 展示配置

**0:50-1:00 (10min) — 工作流编排**
- 运行一次完整的分析流水线:
  ```python
  await agent.run_analysis(symbols=["sh.600519"])
  ```
- 展示7个节点依次执行的过程
- 特别展示 Bull/Bear/Judge 对抗辩论结果
- 展示最终生成的综合报告

**1:00-1:05 (5min) — 代码即推理演示**
- 展示 `agent/tools/code_executor.py`
- "LLM生成计算代码 → 沙箱执行 → 数字锁定 → LLM叙事 → 安全检查"
- "从根本上杜绝LLM数值幻觉"

---

### 模块4: AI应用测试 (20分钟)

**1:05-1:15 (10min) — 预设问题测试**
- 依次测试3个核心问题:
  1. "当前A股市场处于什么状态？"
  2. "分析一下600519贵州茅台,给出操作建议"
  3. "什么是T+1制度？对我的交易有什么影响？"

- 每个问题展示: 工具调用过程 + 数据来源 + 回复内容 + 风险提示

**1:15-1:22 (7min) — 质量评估展示**
- 打开 `tests/competition_questions.json` → 20个测试问题
- 打开 `tests/test_competition_quality.py` → 自动化评估
- 展示评估维度: 格式合规 + 数值准确性 + 逻辑一致性 + 风险提示
- 展示自评分数: 平均质量分 X/10

**1:22-1:25 (3min) — 端到端Benchmark**
- 运行 `python scripts/competition_benchmark.py --quick`
- 展示4模块评分汇总

---

### 总结 (5分钟)

**1:25-1:28 (3min) — 回顾**
- 快速回顾4大模块亮点
- "我们用AI+金融的方式,构建了一个完整的量化分析智能体"

**1:28-1:30 (2min) — 展望**
- 未来计划: 实时行情、基本面Agent、组合优化、移动端
- 感谢评委,开放Q&A

---

## 备用问题库 (应对评委提问)

1. **"为什么不用现成的量化平台?"**
   → 自研能完全控制分析逻辑和风控规则,且可以自由扩展

2. **"LLM幻觉问题怎么解决?"**
   → Code-as-Reasoning: 所有数字由代码计算,LLM只负责叙事,NumericSafetyChecker校验

3. **"成本如何控制?"**
   → 3层模型路由,60%走本地免费模型,日均成本<¥1

4. **"知识库如何更新?"**
   → 修改YAML/MD文件即时生效,提示词支持版本管理和A/B测试

5. **"策略回测有未来函数吗?"**
   → 事件驱动引擎逐日模拟,PIT数据处理器消除幸存者偏差,6层过拟合守卫

6. **"如何保证分析客观性?"**
   → Bull/Bear/Judge三方辩论,双方基于同一份数据,Judge独立裁决

---

## 演示前检查清单

- [ ] DeepSeek API Key 已配置
- [ ] Ollama 服务可正常连接 (或跳过)
- [ ] 知识库文件完整 (8个提示词 + 4个任务 + 4个few-shot + 2个参考文档)
- [ ] `agent/ARCHITECTURE.md` Mermaid图可渲染
- [ ] `competition_benchmark.py --quick` 可正常运行
- [ ] Dashboard 可正常启动 (`streamlit run web/dashboard.py`)
- [ ] 网络稳定 (数据源需要网络)
- [ ] 演示电脑已安装所有依赖 (`pip install -r requirements.txt`)
