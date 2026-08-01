"""
DecisionLogger — 决策日志持久化器 (v3.1)

将每次分析的完整推理链持久化到 SQLite,支持:
  - 全量决策记录: bull/bear论点, judge裁决, 最终信号, 置信度
  - 历史回溯: 按标的+时间范围查询过往决策
  - 反思注入: 生成反射上下文,注入下游 Agent prompt
  - 结果闭环: 记录实际收益,与预测对比,校准置信度

用法:
    logger = DecisionLogger("data/decisions.db")
    log_id = logger.log_analysis(symbol="600519", date="2026-07-29",
                                  debate_result={...}, final_signal="BUY", ...)
    # N天后...
    logger.log_outcome(log_id, realized_return=0.052, benchmark_return=0.010)

    # 下次分析同一标的时:
    reflection = logger.get_reflection_context("600519", days=5)
    # → "你5天前推荐买入600519(置信度75%),实际收益+5.2%"
"""

import json
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger


@dataclass
class DecisionRecord:
    """单条决策记录"""
    log_id: int = 0
    symbol: str = ""
    analysis_date: str = ""          # ISO date
    task_id: str = ""
    market_regime: str = "unknown"
    regime_confidence: float = 0.0

    # 多空辩论
    bull_score: float = 0.0
    bear_score: float = 0.0
    debate_result: str = ""          # bull_win | bear_win | tie
    bull_key_points: str = ""        # JSON string
    bear_key_points: str = ""        # JSON string

    # 最终决策
    final_signal: str = "HOLD"       # STRONG_BUY/BUY/HOLD/SELL/STRONG_SELL
    confidence: float = 0.0
    entry_price: float = 0.0
    stop_loss: float = 0.0
    take_profit: float = 0.0
    strategy_id: str = ""
    reasoning: str = ""

    # 风控审查
    risk_level: str = "medium"
    risk_flaws: int = 0
    robustness_score: float = 100.0

    # 合成评分
    composite_score: float = 0.0
    composite_grade: str = "C"

    # 结果追踪 (事后填写)
    realized_return: Optional[float] = None
    benchmark_return: Optional[float] = None
    is_correct: Optional[bool] = None
    review_date: Optional[str] = None
    review_notes: str = ""

    def to_dict(self) -> dict:
        return {
            "log_id": self.log_id,
            "symbol": self.symbol,
            "analysis_date": self.analysis_date,
            "final_signal": self.final_signal,
            "confidence": self.confidence,
            "bull_score": self.bull_score,
            "bear_score": self.bear_score,
            "debate_result": self.debate_result,
            "realized_return": self.realized_return,
            "is_correct": self.is_correct,
        }


class DecisionLogger:
    """决策日志 SQLite 持久化器"""

    def __init__(self, db_path: str = "data/decisions.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _init_db(self):
        conn = self._get_conn()
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS decisions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    analysis_date TEXT NOT NULL,
                    task_id TEXT DEFAULT '',
                    market_regime TEXT DEFAULT 'unknown',
                    regime_confidence REAL DEFAULT 0.0,

                    -- 多空辩论
                    bull_score REAL DEFAULT 0.0,
                    bear_score REAL DEFAULT 0.0,
                    debate_result TEXT DEFAULT '',
                    bull_key_points TEXT DEFAULT '[]',
                    bear_key_points TEXT DEFAULT '[]',

                    -- 最终决策
                    final_signal TEXT DEFAULT 'HOLD',
                    confidence REAL DEFAULT 0.0,
                    entry_price REAL DEFAULT 0.0,
                    stop_loss REAL DEFAULT 0.0,
                    take_profit REAL DEFAULT 0.0,
                    strategy_id TEXT DEFAULT '',
                    reasoning TEXT DEFAULT '',

                    -- 风控审查
                    risk_level TEXT DEFAULT 'medium',
                    risk_flaws INTEGER DEFAULT 0,
                    robustness_score REAL DEFAULT 100.0,

                    -- 合成评分
                    composite_score REAL DEFAULT 0.0,
                    composite_grade TEXT DEFAULT 'C',

                    -- 结果追踪 (事后填写)
                    realized_return REAL,
                    benchmark_return REAL,
                    is_correct INTEGER,
                    review_date TEXT,
                    review_notes TEXT DEFAULT '',

                    created_at TEXT DEFAULT (datetime('now', 'localtime'))
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_decisions_symbol_date
                ON decisions(symbol, analysis_date DESC)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_decisions_signal
                ON decisions(final_signal, analysis_date DESC)
            """)
            conn.commit()
            logger.info(f"DecisionLogger 初始化完成: {self.db_path}")
        finally:
            conn.close()

    # ── 写入 ──

    def log_analysis(
        self,
        symbol: str,
        analysis_date: str,
        task_id: str = "",
        market_regime: str = "unknown",
        regime_confidence: float = 0.0,
        bull_score: float = 0.0,
        bear_score: float = 0.0,
        debate_result: str = "",
        bull_key_points: Any = None,
        bear_key_points: Any = None,
        final_signal: str = "HOLD",
        confidence: float = 0.0,
        entry_price: float = 0.0,
        stop_loss: float = 0.0,
        take_profit: float = 0.0,
        strategy_id: str = "",
        reasoning: str = "",
        risk_level: str = "medium",
        risk_flaws: int = 0,
        robustness_score: float = 100.0,
        composite_score: float = 0.0,
        composite_grade: str = "C",
    ) -> int:
        """记录一次完整分析决策,返回 log_id"""
        conn = self._get_conn()
        try:
            cursor = conn.execute(
                """INSERT INTO decisions (
                    symbol, analysis_date, task_id,
                    market_regime, regime_confidence,
                    bull_score, bear_score, debate_result,
                    bull_key_points, bear_key_points,
                    final_signal, confidence, entry_price, stop_loss, take_profit,
                    strategy_id, reasoning,
                    risk_level, risk_flaws, robustness_score,
                    composite_score, composite_grade
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    symbol, analysis_date, task_id,
                    market_regime, regime_confidence,
                    bull_score, bear_score, debate_result,
                    json.dumps(bull_key_points or [], ensure_ascii=False) if not isinstance(bull_key_points, str) else bull_key_points,
                    json.dumps(bear_key_points or [], ensure_ascii=False) if not isinstance(bear_key_points, str) else bear_key_points,
                    final_signal, confidence, entry_price, stop_loss, take_profit,
                    strategy_id, reasoning,
                    risk_level, risk_flaws, robustness_score,
                    composite_score, composite_grade,
                ),
            )
            conn.commit()
            log_id = cursor.lastrowid
            logger.info(f"决策已记录: [{log_id}] {symbol} {final_signal} (置信度{confidence:.0%}, "
                       f"多头{debate_result})")
            return log_id
        except Exception as e:
            logger.error(f"决策记录失败: {e}")
            return -1
        finally:
            conn.close()

    def log_outcome(
        self,
        log_id: int,
        realized_return: float,
        benchmark_return: float = 0.0,
        review_date: str = "",
        notes: str = "",
    ) -> bool:
        """记录决策的实际盈亏结果"""
        if not review_date:
            review_date = date.today().isoformat()

        # 判断预测是否正确
        is_correct = None
        # 信号为BUY类 + 正收益 = 正确; SELL类 + 负收益 = 正确
        # 简化判断: realized_return > 0 且是买入信号,或 realized_return < 0 且是卖出信号
        conn = self._get_conn()
        try:
            row = conn.execute("SELECT final_signal FROM decisions WHERE id=?", (log_id,)).fetchone()
            if row:
                signal = row["final_signal"]
                if signal in ("STRONG_BUY", "BUY"):
                    is_correct = 1 if realized_return > 0 else 0
                elif signal in ("STRONG_SELL", "SELL"):
                    is_correct = 1 if realized_return < 0 else 0
                else:
                    is_correct = 1 if abs(realized_return) < 0.01 else 0  # HOLD: 接近0即正确

            conn.execute(
                """UPDATE decisions SET
                    realized_return = ?, benchmark_return = ?,
                    is_correct = ?, review_date = ?, review_notes = ?
                WHERE id = ?""",
                (realized_return, benchmark_return, is_correct,
                 review_date, notes, log_id),
            )
            conn.commit()
            alpha = realized_return - benchmark_return
            correct_label = "✅" if is_correct == 1 else "❌"
            logger.info(f"决策[{log_id}]结果已记录: 收益{realized_return:+.2%} "
                       f"(α={alpha:+.2%}) {correct_label}")
            return True
        except Exception as e:
            logger.error(f"结果记录失败: {e}")
            return False
        finally:
            conn.close()

    # ── 读取 ──

    def get_history(self, symbol: str, days: int = 30) -> List[DecisionRecord]:
        """获取某标的的历史决策"""
        cutoff = (date.today() - timedelta(days=days)).isoformat()
        conn = self._get_conn()
        try:
            rows = conn.execute(
                "SELECT * FROM decisions WHERE symbol=? AND analysis_date >= ? ORDER BY analysis_date DESC",
                (symbol, cutoff),
            ).fetchall()
            return [self._row_to_record(r) for r in rows]
        finally:
            conn.close()

    def get_recent_decisions(self, days: int = 5) -> List[DecisionRecord]:
        """获取所有标的近N天决策"""
        cutoff = (date.today() - timedelta(days=days)).isoformat()
        conn = self._get_conn()
        try:
            rows = conn.execute(
                "SELECT * FROM decisions WHERE analysis_date >= ? ORDER BY analysis_date DESC",
                (cutoff,),
            ).fetchall()
            return [self._row_to_record(r) for r in rows]
        finally:
            conn.close()

    def get_unreviewed(self) -> List[DecisionRecord]:
        """获取未回顾的决策"""
        conn = self._get_conn()
        try:
            rows = conn.execute(
                "SELECT * FROM decisions WHERE is_correct IS NULL AND analysis_date <= date('now', '-5 days')"
            ).fetchall()
            return [self._row_to_record(r) for r in rows]
        finally:
            conn.close()

    def get_reflection_context(self, symbol: str, days: int = 5) -> str:
        """
        生成反思上下文 — 注入下游 Agent 的 Prompt

        返回格式:
            ## 📊 历史决策回顾
            - [2026-07-25] 推荐BUY (置信度75%, 多头得分8.2) → 实际+5.2% ✅
            - [2026-07-20] 推荐HOLD (置信度60%) → 实际-1.3%
            ## 📈 模式总结
            近期2次信号中正确1次(50%), 平均α=+1.95%
        """
        records = self.get_history(symbol, days=days)
        # 只取有结果的
        reviewed = [r for r in records if r.realized_return is not None]

        if not records:
            return "## 📊 历史决策回顾\n暂无该标的近期分析记录。\n"

        lines = ["## History Review (last {} days)".format(days)]
        lines.append("")
        for r in records[:5]:
            realized = f"actual={r.realized_return:+.2%}" if r.realized_return is not None else "pending"
            icon = "[OK]" if r.is_correct == 1 else ("[X]" if r.is_correct == 0 else "[?]")
            lines.append(
                f"- [{r.analysis_date}] **{r.final_signal}** (confidence={r.confidence:.0%}, "
                f"bull={r.bull_score:.1f}/bear={r.bear_score:.1f}) -> {realized} {icon}"
            )

        if reviewed:
            lines.append("")
            lines.append("## Pattern Summary")
            correct_count = sum(1 for r in reviewed if r.is_correct == 1)
            total = len(reviewed)
            avg_alpha = sum((r.realized_return or 0) - (r.benchmark_return or 0)
                           for r in reviewed) / max(total, 1)
            lines.append(f"Recent {total} signals: {correct_count}/{total}({correct_count/max(total,1):.0%}) correct, "
                        f"avg alpha={avg_alpha:+.2%}")
            lines.append("Please calibrate confidence of this analysis based on historical accuracy.")

        return "\n".join(lines)

    def get_stats(self, days: int = 30) -> Dict[str, Any]:
        """获取决策统计"""
        conn = self._get_conn()
        try:
            total = conn.execute(
                "SELECT COUNT(*) as n FROM decisions WHERE analysis_date >= date('now', ?)",
                (f"-{days} days",)
            ).fetchone()["n"]

            reviewed = conn.execute(
                """SELECT COUNT(*) as n FROM decisions
                WHERE analysis_date >= date('now', ?) AND is_correct IS NOT NULL""",
                (f"-{days} days",)
            ).fetchone()["n"]

            if total == 0:
                return {"total": 0, "reviewed": 0}

            correct = conn.execute(
                """SELECT COUNT(*) as n FROM decisions
                WHERE analysis_date >= date('now', ?) AND is_correct = 1""",
                (f"-{days} days",)
            ).fetchone()["n"]

            signal_dist = {}
            for row in conn.execute(
                """SELECT final_signal, COUNT(*) as n FROM decisions
                WHERE analysis_date >= date('now', ?) GROUP BY final_signal""",
                (f"-{days} days",)
            ).fetchall():
                signal_dist[row["final_signal"]] = row["n"]

            return {
                "total": total,
                "reviewed": reviewed,
                "correct": correct,
                "accuracy": correct / max(reviewed, 1) if reviewed > 0 else None,
                "signal_distribution": signal_dist,
            }
        finally:
            conn.close()

    # ── 清理 ──

    def delete_old(self, days: int = 365):
        """删除N天前的旧记录"""
        conn = self._get_conn()
        try:
            cutoff = (date.today() - timedelta(days=days)).isoformat()
            deleted = conn.execute(
                "DELETE FROM decisions WHERE analysis_date < ?", (cutoff,)
            ).rowcount
            conn.commit()
            if deleted:
                logger.info(f"已清理 {deleted} 条过期决策记录")
        finally:
            conn.close()

    def _row_to_record(self, row: sqlite3.Row) -> DecisionRecord:
        return DecisionRecord(
            log_id=row["id"],
            symbol=row["symbol"],
            analysis_date=row["analysis_date"],
            task_id=row["task_id"] or "",
            market_regime=row["market_regime"] or "unknown",
            regime_confidence=row["regime_confidence"] or 0.0,
            bull_score=row["bull_score"] or 0.0,
            bear_score=row["bear_score"] or 0.0,
            debate_result=row["debate_result"] or "",
            bull_key_points=row["bull_key_points"] or "[]",
            bear_key_points=row["bear_key_points"] or "[]",
            final_signal=row["final_signal"] or "HOLD",
            confidence=row["confidence"] or 0.0,
            entry_price=row["entry_price"] or 0.0,
            stop_loss=row["stop_loss"] or 0.0,
            take_profit=row["take_profit"] or 0.0,
            strategy_id=row["strategy_id"] or "",
            reasoning=row["reasoning"] or "",
            risk_level=row["risk_level"] or "medium",
            risk_flaws=row["risk_flaws"] or 0,
            robustness_score=row["robustness_score"] or 100.0,
            composite_score=row["composite_score"] or 0.0,
            composite_grade=row["composite_grade"] or "C",
            realized_return=row["realized_return"],
            benchmark_return=row["benchmark_return"],
            is_correct=bool(row["is_correct"]) if row["is_correct"] is not None else None,
            review_date=row["review_date"],
            review_notes=row["review_notes"] or "",
        )
