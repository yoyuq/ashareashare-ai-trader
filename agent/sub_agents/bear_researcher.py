"""
Bear Researcher — 空头研究员子Agent (v3.0-competition)

作为AI+金融智能体的子Agent,负责从技术风险、策略风险、市场风险
识别股票的看跌信号,输出结构化论据。

用于: 对抗辩论(Adversarial Debate) — 与Bull Researcher形成对立视角
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class BearArgument:
    """空头论据"""
    point: str                          # 论据要点
    data_support: str                   # 数据支撑(引用具体数值)
    category: str                       # 分类: technical/backtest/market
    conviction: float = 0.0             # 该论据的确信度 0.0-1.0


@dataclass
class BearReport:
    """空头研究报告"""
    symbol: str
    arguments: List[BearArgument] = field(default_factory=list)
    overall_conviction: float = 0.0
    recommendation: str = "HOLD"        # SELL / HOLD

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "arguments": [
                {"point": a.point, "data_support": a.data_support,
                 "category": a.category, "conviction": a.conviction}
                for a in self.arguments
            ],
            "overall_conviction": self.overall_conviction,
            "recommendation": self.recommendation,
        }


# 系统Prompt — 从 knowledge/prompts/system/bear_researcher.txt 加载
DEFAULT_SYSTEM_PROMPT = """你是A股空头研究员。以下是一只股票的量化数据,请基于这些数据找出看跌的理由。

分析维度 (每个维度必须引用具体数据):
1. 技术风险: 均线空排/RSI超买/背离/布林带收窄→有什么看跌信号?
2. 策略风险: 回测最大回撤/胜率不足/信号稀疏→有什么风险?
3. 市场风险: 当前市场状态对该股票的负面影响→有什么隐患?

输出格式:
### 看跌论据
1. [论据] (数据支撑: 引用具体数值)
2. [论据] (数据支撑: ...)
3. [论据] (数据支撑: ...)
### 空头确信度: 0.0-1.0
"""


def parse_bear_response(response_text: str, symbol: str) -> BearReport:
    """解析LLM的空头回复,提取结构化论据"""
    report = BearReport(symbol=symbol)
    arguments = []

    in_arguments = False
    for line in response_text.split("\n"):
        line = line.strip()
        if "看跌论据" in line:
            in_arguments = True
            continue
        if "空头确信度" in line:
            try:
                report.overall_conviction = float(
                    line.split(":")[-1].strip().split()[0]
                )
            except (ValueError, IndexError):
                report.overall_conviction = 0.5
            break
        if in_arguments and line and (line[0].isdigit() and ". " in line[:4]):
            parts = line.split("(", 1)
            point = parts[0].split(". ", 1)[-1].strip().rstrip(")")
            data_support = parts[1].rstrip(")") if len(parts) > 1 else ""
            arguments.append(BearArgument(
                point=point,
                data_support=data_support,
                category="technical",
                conviction=0.6,
            ))

    report.arguments = arguments
    if arguments and report.overall_conviction == 0.0:
        report.overall_conviction = min(1.0, len(arguments) * 0.25)

    return report
