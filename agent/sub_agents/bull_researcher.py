"""
Bull Researcher — 多头研究员子Agent (v3.0-competition)

作为AI+金融智能体的子Agent,负责从技术面、回测面、市场适配面
识别股票的看涨信号,输出结构化论据。

用于: 对抗辩论(Adversarial Debate) — 与Bear Researcher形成对立视角
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class BullArgument:
    """多头论据"""
    point: str                          # 论据要点
    data_support: str                   # 数据支撑(引用具体数值)
    category: str                       # 分类: technical/backtest/market
    conviction: float = 0.0             # 该论据的确信度 0.0-1.0


@dataclass
class BullReport:
    """多头研究报告"""
    symbol: str
    arguments: List[BullArgument] = field(default_factory=list)
    overall_conviction: float = 0.0
    recommendation: str = "HOLD"        # BUY / HOLD

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


# 系统Prompt — 从 knowledge/prompts/system/bull_researcher.txt 加载
# 如果文件不存在,使用此默认Prompt
DEFAULT_SYSTEM_PROMPT = """你是A股多头研究员。以下是一只股票的量化数据,请基于这些数据找出看涨的理由。

分析维度 (每个维度必须引用具体数据):
1. 技术面: 均线排列/金叉死叉/RSI位置/布林带位置→有什么看涨信号?
2. 回测面: 策略胜率/夏普比率/盈亏比→是否支持做多?
3. 市场适配: 当前市场状态下的策略适配度→是否有利于该股票?

输出格式:
### 看涨论据
1. [论据] (数据支撑: 引用具体数值)
2. [论据] (数据支撑: ...)
3. [论据] (数据支撑: ...)
### 多头确信度: 0.0-1.0
"""


def parse_bull_response(response_text: str, symbol: str) -> BullReport:
    """解析LLM的多头回复,提取结构化论据"""
    report = BullReport(symbol=symbol)
    arguments = []

    # 简单解析: 按行提取编号论据
    in_arguments = False
    for line in response_text.split("\n"):
        line = line.strip()
        if "看涨论据" in line:
            in_arguments = True
            continue
        if "多头确信度" in line:
            try:
                report.overall_conviction = float(
                    line.split(":")[-1].strip().split()[0]
                )
            except (ValueError, IndexError):
                report.overall_conviction = 0.5
            break
        if in_arguments and line and (line[0].isdigit() and ". " in line[:4]):
            # 格式: "1. [论据] (数据支撑: ...)"
            parts = line.split("(", 1)
            point = parts[0].split(". ", 1)[-1].strip().rstrip(")")
            data_support = parts[1].rstrip(")") if len(parts) > 1 else ""
            arguments.append(BullArgument(
                point=point,
                data_support=data_support,
                category="technical",
                conviction=0.6,
            ))

    report.arguments = arguments
    if arguments and report.overall_conviction == 0.0:
        report.overall_conviction = min(1.0, len(arguments) * 0.25)

    return report
