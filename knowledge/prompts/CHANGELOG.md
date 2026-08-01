# 提示词变更日志

记录所有系统提示词的版本迭代历史。

## 版本格式

采用语义化版本: `MAJOR.MINOR.PATCH`
- MAJOR: 提示词框架/角色定义变更
- MINOR: 新增分析维度/输出格式变更
- PATCH: 措辞优化/示例更新/Bug修复

---

## 2026-07-28 — 竞赛初始化

### v1.0.0 初始版本

**新建提示词:**
- `chat_assistant.txt` — A股智能分析助手系统提示词 (从 chat_agent.py 硬编码提取)
- `bull_researcher.txt` — 多头研究员提示词 (从 workflow.py 硬编码提取)
- `bear_researcher.txt` — 空头研究员提示词 (从 workflow.py 硬编码提取)
- `judge.txt` — 策略裁判提示词 (从 workflow.py 硬编码提取)

**已有提示词添加版本管理:**
- `technical_analyst.txt` — 技术分析Agent (6维度分析框架)
- `market_scanner.txt` — 市场扫描Agent (4维扫描)
- `synthesis.txt` — 综合研判Agent (多空辩论 + 报告生成)
- `risk_assessor.txt` — 风控评估Agent (8层风控)

**变更说明:**
- 所有提示词从代码硬编码迁移至 `knowledge/prompts/system/` 目录
- 统一添加 YAML frontmatter 版本头 (version/date/author/changes)
- 通过 `KnowledgeManager.get_system_prompt()` 统一加载

---

## 2026-07-29 — v1.1.0 提示词迭代优化

### v1.1.0 chat_assistant 优化

**修改文件:**
- `chat_assistant.txt` → `chat_assistant_v1.1.txt` — 系统性迭代优化

**变更说明:**
- 增加人格化身份「小A」,增强用户信任感
- 增加结构化输出模板 (现状/分析/建议/风险 四段式)
- 增加用户意图识别 (分析类/知识类/预测类/紧急类)
- 增加知识库使用指南,引导LLM调用工具获取数据
- 增加示例对话,Few-shot效应提升格式一致性

**评估结果 (PromptEvaluator 4维评分):**
- 格式合规: 6.0 → 8.5 (+2.5)
- 数值准确性: 5.5 → 7.0 (+1.5)
- 逻辑一致性: 6.0 → 8.0 (+2.0)
- 风险提示: 7.0 → 8.5 (+1.5)
- 总分: 24.5 → 32.0 (+7.5, +30.6%)

**详细对比**: 见 `docs/competition/PROMPT_AB_COMPARISON.md`

```markdown
## YYYY-MM-DD — 简短标题

### vX.Y.Z 版本说明

**修改文件:**
- `filename.txt` — 修改内容摘要

**变更说明:**
- 具体变更1
- 具体变更2
```
