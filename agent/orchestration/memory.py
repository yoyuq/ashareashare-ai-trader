"""
Agent Shared Memory (v3.1, connected to DecisionLogger)

Problem: In v1.0, each agent ran independently and prior reasoning was not passed downstream.
Improvement: Shared memory object carries key intermediate conclusions across agents.

Features:
- Session-level memory: shared within one analysis pipeline
- Cross-day memory: historical analysis loaded from DecisionLogger (SQLite)
- Vector memory: similar historical scenarios from ChromaDB
- Reflection injection: reflection context injected into downstream agent prompts

v3.1: load_historical() is now connected to DecisionLogger (SQLite).
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from agent.orchestration.decision_log import DecisionLogger


@dataclass
class AgentMemory:
    """Agent shared memory (v3.1, connected to DecisionLogger)"""

    # Session-level memory (within one analysis pipeline)
    session_findings: List[Dict[str, Any]] = field(default_factory=list)

    # Cross-day memory (historical analysis from SQLite DecisionLogger)
    historical_context: Dict[str, Any] = field(default_factory=dict)

    # Vector memory (similar historical scenarios from ChromaDB)
    similar_scenarios: List[Dict[str, Any]] = field(default_factory=list)

    # v3.1: DecisionLogger reference (lazy injection)
    _decision_logger: Optional["DecisionLogger"] = None

    def set_decision_logger(self, decision_logger: "DecisionLogger"):
        """Inject DecisionLogger reference"""
        self._decision_logger = decision_logger

    def add_finding(self, agent_name: str, finding: Dict[str, Any]):
        """Record a sub-agent's finding"""
        self.session_findings.append({
            "agent": agent_name,
            "timestamp": datetime.now().isoformat(),
            "finding": finding,
        })

    def get_agent_findings(self, agent_name: str) -> List[Dict]:
        """Get all findings from a specific agent"""
        return [
            f for f in self.session_findings
            if f["agent"] == agent_name
        ]

    def get_latest_finding(self, agent_name: str) -> Optional[Dict]:
        """Get the most recent finding from a specific agent"""
        findings = self.get_agent_findings(agent_name)
        return findings[-1] if findings else None

    def get_findings_summary(self) -> str:
        """Generate a summary of all findings (for injecting into downstream agent prompts)"""
        if not self.session_findings:
            return "No prior agent analysis results available."

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
        Load recent analysis history for a symbol from DecisionLogger (SQLite).

        v3.1: Connected to DecisionLogger, reads real decision records.
        """
        if self._decision_logger is None:
            self.historical_context[symbol] = {
                "recent_signals": [],
                "recent_reports": [],
                "note": "DecisionLogger not injected",
            }
            return

        records = self._decision_logger.get_history(symbol, days=days)
        self.historical_context[symbol] = {
            "recent_signals": [
                {
                    "date": r.analysis_date,
                    "signal": r.final_signal,
                    "confidence": r.confidence,
                    "realized_return": r.realized_return,
                    "is_correct": r.is_correct,
                }
                for r in records
            ],
            "recent_reports": [
                {
                    "date": r.analysis_date,
                    "reasoning": r.reasoning[:500] if r.reasoning else "",
                    "bull_score": r.bull_score,
                    "bear_score": r.bear_score,
                }
                for r in records[:3]
            ],
            "reflection": self.get_reflection(symbol, days),
        }

    def get_reflection(self, symbol: str, days: int = 5) -> str:
        """
        Generate reflection context for Synthesis Agent prompt injection.

        v3.1: Uses DecisionLogger for structured reflection.
        """
        if self._decision_logger is None:
            return ""

        return self._decision_logger.get_reflection_context(symbol, days=days)

    def clear_session(self):
        """Clear current session memory"""
        self.session_findings.clear()
        self.similar_scenarios.clear()
        self.historical_context.clear()
