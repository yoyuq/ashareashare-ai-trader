"""
🆕 v2.0 Agent共享记忆

问题: v1.0中每个Agent独立运行,上一个Agent的推理过程没有传递给下一个
改进: 引入共享记忆体,关键中间结论显式传递

支持:
- 会话级记忆: 一次分析流水线内共享
- 跨日记忆: 历史分析结论从DB读取
- 向量记忆: 相似历史场景检索
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class AgentMemory:
    """Agent间共享记忆体"""

    # 会话级记忆 (一次分析流水线内)
    session_findings: List[Dict[str, Any]] = field(default_factory=list)

    # 跨日记忆 (历史分析结论,从DB读取)
    historical_context: Dict[str, Any] = field(default_factory=dict)

    # 向量记忆 (相似历史场景的embedding)
    similar_scenarios: List[Dict[str, Any]] = field(default_factory=list)

    def add_finding(self, agent_name: str, finding: Dict[str, Any]):
        """记录某个Agent的发现"""
        self.session_findings.append({
            "agent": agent_name,
            "timestamp": datetime.now().isoformat(),
            "finding": finding,
        })

    def get_agent_findings(self, agent_name: str) -> List[Dict]:
        """获取特定Agent的所有发现"""
        return [
            f for f in self.session_findings
            if f["agent"] == agent_name
        ]

    def get_latest_finding(self, agent_name: str) -> Optional[Dict]:
        """获取特定Agent的最新发现"""
        findings = self.get_agent_findings(agent_name)
        return findings[-1] if findings else None

    def get_findings_summary(self) -> str:
        """生成所有发现的摘要(注入下游Agent的Prompt)"""
        if not self.session_findings:
            return "暂无前序Agent的分析结果。"

        lines = []
        for f in self.session_findings:
            finding = f["finding"]
            if isinstance(finding, dict):
                key_points = finding.get("key_points", finding.get("summary", str(finding)))
                lines.append(f"- [{f['agent']}]: {key_points}")
            else:
                lines.append(f"- [{f['agent']}]: {finding}")

        return "\n".join(lines)

    def load_historical(self, symbol: str, days: int = 5):
        """
        从数据库加载某标的近N天分析历史

        在实际使用中调用 SignalTracker.get_recent_analyses()
        """
        # TODO: 实现DB读取
        self.historical_context[symbol] = {
            "recent_signals": [],
            "recent_reports": [],
        }

    def clear_session(self):
        """清理当前会话记忆"""
        self.session_findings.clear()
        self.similar_scenarios.clear()
