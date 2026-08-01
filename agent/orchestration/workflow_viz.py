"""
Workflow Visualizer — 工作流可视化 (v3.0-competition)

将 LangGraph 7节点分析流水线可视化为 Mermaid 拓扑图,
展示每个节点的输入/输出/模型选择/预估耗时。

用于:
  - 比赛模块1 (架构展示)
  - 比赛模块3 (工作流编排展示)
  - 总决赛PPT素材
"""

from typing import Dict, List, Optional


# 工作流节点定义
WORKFLOW_NODES = [
    {
        "id": "data_preparation",
        "label": "数据采集",
        "icon": "📡",
        "description": "多源数据获取 + 市场体制检测",
        "inputs": ["股票代码列表"],
        "outputs": ["K线数据", "市场体制", "体制置信度"],
        "model_tier": "LOCAL",
        "est_time": "10-30s",
        "parallel": False,
    },
    {
        "id": "technical_analysis",
        "label": "技术分析",
        "icon": "📊",
        "description": "130+指标计算 + LLM解读",
        "inputs": ["K线数据 (DataFrame)"],
        "outputs": ["指标字典", "LLM分析叙事"],
        "model_tier": "FLASH",
        "est_time": "5-15s/股",
        "parallel": True,
    },
    {
        "id": "market_scanner",
        "label": "市场扫描",
        "icon": "🔍",
        "description": "全市场4维评分排序",
        "inputs": ["市场体制", "股票列表"],
        "outputs": ["Top-N评分榜", "扫描叙事"],
        "model_tier": "FLASH",
        "est_time": "5-10s",
        "parallel": False,
    },
    {
        "id": "strategy_matching",
        "label": "策略匹配",
        "icon": "🎯",
        "description": "市场体制→策略映射 + 适配度计算",
        "inputs": ["市场体制", "技术指标"],
        "outputs": ["Top-3策略", "fit_score"],
        "model_tier": "FLASH",
        "est_time": "1-3s/股",
        "parallel": True,
    },
    {
        "id": "backtest_verification",
        "label": "回测验证",
        "icon": "⏮️",
        "description": "事件驱动回测 (T+1/涨跌停模拟)",
        "inputs": ["Top-2策略", "历史K线"],
        "outputs": ["胜率", "夏普比率", "最大回撤", "盈亏比"],
        "model_tier": "LOCAL",
        "est_time": "3-10s/策略",
        "parallel": True,
    },
    {
        "id": "adversarial_debate",
        "label": "对抗辩论",
        "icon": "⚔️",
        "description": "Bull/Bear/Judge三方对抗论证",
        "inputs": ["量化数据上下文", "回测结果"],
        "outputs": ["Verdict JSON", "看涨/看跌论据"],
        "model_tier": "PRO",
        "est_time": "10-30s/股",
        "parallel": False,
    },
    {
        "id": "synthesis",
        "label": "综合报告",
        "icon": "📝",
        "description": "汇总→辩论评估→结构化报告生成",
        "inputs": ["所有前序节点输出"],
        "outputs": ["最终报告 (Markdown)"],
        "model_tier": "PRO",
        "est_time": "5-15s",
        "parallel": False,
    },
]

# 模型路由决策表
MODEL_ROUTING = {
    "LOCAL": {
        "model": "Ollama Qwen3-4B",
        "cost": "免费",
        "target_load": "60%",
        "color": "#4ade80",  # green
    },
    "FLASH": {
        "model": "DeepSeek V4-Flash",
        "cost": "¥1-2/M tokens",
        "target_load": "30%",
        "color": "#facc15",  # yellow
    },
    "PRO": {
        "model": "DeepSeek V4-Pro",
        "cost": "¥3-6/M tokens",
        "target_load": "10%",
        "color": "#f87171",  # red
    },
}


def generate_mermaid_flowchart() -> str:
    """生成工作流 Mermaid 流程图"""
    lines = ["```mermaid", "graph LR"]
    lines.append("    START((开始)) --> DP[📡 数据采集<br/>LOCAL]")
    lines.append("    DP --> TA[📊 技术分析<br/>FLASH]")
    lines.append("    TA --> MS[🔍 市场扫描<br/>FLASH]")
    lines.append("    MS --> SM[🎯 策略匹配<br/>FLASH]")
    lines.append("    SM --> BV[⏮️ 回测验证<br/>LOCAL]")
    lines.append("    BV --> AD[⚔️ 对抗辩论<br/>PRO]")
    lines.append("    AD --> SY[📝 综合报告<br/>PRO]")
    lines.append("    SY --> END((结束))")
    lines.append("")
    lines.append("    style DP fill:#4ade80")
    lines.append("    style TA fill:#facc15")
    lines.append("    style MS fill:#facc15")
    lines.append("    style SM fill:#facc15")
    lines.append("    style BV fill:#4ade80")
    lines.append("    style AD fill:#f87171")
    lines.append("    style SY fill:#f87171")
    lines.append("```")
    return "\n".join(lines)


def generate_mermaid_agent_collaboration() -> str:
    """生成 Multi-Agent 协作 Mermaid 图"""
    lines = ["```mermaid", "graph TD"]
    lines.append("    U[用户] --> CA[ChatAssistant<br/>对话入口]")
    lines.append("    U --> WF[AnalysisWorkflow<br/>分析流水线]")

    lines.append("    CA --> MR[ModelRouter<br/>模型路由]")
    lines.append("    CA --> TE[ToolExecutor<br/>7工具调用]")

    lines.append("    WF --> BULL[BullResearcher<br/>多头研究 PRO]")
    lines.append("    WF --> BEAR[BearResearcher<br/>空头研究 PRO]")
    lines.append("    BULL --> JUDGE[Judge<br/>策略裁判 PRO]")
    lines.append("    BEAR --> JUDGE")

    lines.append("    MR --> L[Qwen3-4B<br/>免费 60%]")
    lines.append("    MR --> F[DeepSeek Flash<br/>低价 30%]")
    lines.append("    MR --> P[DeepSeek Pro<br/>深度 10%]")

    lines.append("    TE --> DATA[(DataRouter<br/>4数据源)]")
    lines.append("    TE --> IND[TechnicalAnalyzer<br/>130+指标]")
    lines.append("    TE --> BT[StrategyBacktester<br/>9策略回测]")

    lines.append("    KM[(KnowledgeManager<br/>知识库+ChromaDB)] -.-> CA")
    lines.append("    KM -.-> WF")
    lines.append("```")
    return "\n".join(lines)


def generate_node_table() -> str:
    """生成节点详情Markdown表格"""
    headers = "| 节点 | 图标 | 输入 | 输出 | 模型 | 耗时 | 并行 |"
    sep = "|------|------|------|------|------|------|------|"

    rows = []
    for node in WORKFLOW_NODES:
        parallel_str = "✅" if node["parallel"] else "❌ (串行)"
        row = (
            f"| {node['label']} "
            f"| {node['icon']} "
            f"| {', '.join(node['inputs'])} "
            f"| {', '.join(node['outputs'])} "
            f"| {MODEL_ROUTING[node['model_tier']]['model']} "
            f"| {node['est_time']} "
            f"| {parallel_str} |"
        )
        rows.append(row)

    return "\n".join([headers, sep] + rows)


def generate_model_routing_table() -> str:
    """生成模型路由决策表"""
    headers = "| 层级 | 模型 | 成本 | 目标负载 | 适用任务 |"
    sep = "|------|------|------|----------|----------|"

    task_map = {
        "LOCAL": "指标读取/K线描述/简单问答/数据格式化",
        "FLASH": "技术分析/策略匹配/信号验证/体制分析",
        "PRO": "综合研判/对抗辩论/策略优化/风险评估",
    }

    rows = []
    for tier, info in MODEL_ROUTING.items():
        row = (
            f"| {tier} "
            f"| {info['model']} "
            f"| {info['cost']} "
            f"| {info['target_load']} "
            f"| {task_map[tier]} |"
        )
        rows.append(row)

    return "\n".join([headers, sep] + rows)


def generate_full_viz() -> str:
    """生成完整的工作流可视化文档"""
    sections = []

    sections.append("# AI+金融量化分析智能体 — 工作流可视化\n")
    sections.append("## 1. 工作流拓扑\n")
    sections.append(generate_mermaid_flowchart())
    sections.append("\n## 2. Multi-Agent协作图\n")
    sections.append(generate_mermaid_agent_collaboration())
    sections.append("\n## 3. 节点详情\n")
    sections.append(generate_node_table())
    sections.append("\n## 4. 模型路由策略\n")
    sections.append(generate_model_routing_table())
    sections.append("\n## 5. 风控体系 (8层)\n")
    sections.append("| 层级 | 名称 | 触发条件 | 动作 |")
    sections.append("|------|------|----------|------|")
    sections.append("| L1 | 回撤熔断 | 单日-8%/-15% | 减仓50%/清仓 |")
    sections.append("| L2 | ATR动态止损 | 价格触及2x ATR | 平仓 |")
    sections.append("| L3 | 移动止盈 | 从高点回落6% | 止盈 |")
    sections.append("| L4 | 仓位限制 | 单票>25%/行业>35% | 限制开仓 |")
    sections.append("| L5 | 踩踏防护 | 跌停无法卖出 | 次日竞价挂单 |")
    sections.append("| L6 | 持仓天数 | >20交易日未盈利 | 强制评估 |")
    sections.append("| L7 | 跌停检测 | 持仓触及跌停 | 立即评估 |")
    sections.append("| L8 | 相关性监控 | 组合相关性>0.8 | 预警 |")

    return "\n".join(sections)


if __name__ == "__main__":
    print(generate_full_viz())
