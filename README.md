# 🏦 A股智能分析Agent (AShare AI Trader)

> AI驱动的A股市场量化分析助手 — 不直接下单，而是：市场扫描 → 多维分析 → 回测验证 → 多空辩论 → 综合研判

## 快速开始

```bash
# 1. 创建虚拟环境 (需要 Python 3.11/3.12)
python3.12 -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# 2. 安装依赖
pip install -r requirements.txt

# 3. 配置环境变量
cp .env.example .env
# 编辑 .env: 填入 DeepSeek API Key

# 4. 安装Ollama并拉取本地模型
ollama serve
ollama pull qwen3:4b

# 5. 启动数据库
docker compose -f docker/docker-compose.yml up -d

# 6. 初始化数据
python scripts/init_database.py
python scripts/download_history.py

# 7. 运行分析
python scripts/run_daily_analysis.py
```

## 架构

```
数据层 → 分析引擎 → 回测引擎 → 代码即推理 → 多空辩论 → AI Agent(LangGraph) → 报告
```

- **3层模型路由**: Ollama Qwen3-4B (本地/免费/60%) → DeepSeek V4-Flash (云端/¥1-2/M/30%) → DeepSeek V4-Pro (复杂任务/10%)
- **6层过拟合防控**: 时间分割 → PBO → Deflated SR → Walk-Forward → 参数敏感性 → Monte Carlo
- **代码即推理**: Python做计算, LLM只叙事 — 消除AI幻觉
- **多空辩论**: Bull ↔ Bear ↔ Judge 三方对抗, 分歧量化

## 技术栈

| 组件 | 技术 |
|------|------|
| 数据 | AKShare / Baostock / Tushare / 东方财富 |
| 数据库 | PostgreSQL + TimescaleDB / Redis / ChromaDB |
| 分析 | TA-Lib / pandas-ta / scikit-learn |
| LLM | DeepSeek V4 + Ollama(Qwen3-4B) |
| 编排 | LangGraph |
| 回测 | backtrader / vectorbt / 自研事件驱动引擎 |
| Web | FastAPI / Streamlit |

## 项目状态

- [x] Phase 0: 环境准备
- [ ] Phase 1: 数据基础层
- [ ] Phase 2: 分析引擎层
- [ ] Phase 3: 回测引擎层
- [ ] Phase 3.5: 知识库初始化
- [ ] Phase 4: AI Agent层
- [ ] Phase 4.5: 代码即推理 + 辩论
- [ ] Phase 5: 产品化

## License

MIT — 个人研究用途
