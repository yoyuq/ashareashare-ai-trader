"""
CheckpointManager — LangGraph 工作流断点续跑 (v3.1)

基于 LangGraph SqliteSaver, 在每个节点完成后自动保存状态。
当工作流因网络/API错误中断时,可从最后一个成功的节点恢复执行。

用法:
    from langgraph.checkpoint.sqlite import SqliteSaver
    from agent.orchestration.checkpoint import CheckpointManager

    cpm = CheckpointManager()
    graph = workflow.compile(checkpointer=cpm.get_saver())
    config = {"configurable": {"thread_id": task_id}}
    result = await graph.ainvoke(initial_state, config=config)
    # 如果中断,可用相同 thread_id 恢复
"""

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger


class CheckpointManager:
    """
    LangGraph 断点管理器

    封装 SqliteSaver, 提供 checkpoint 的查询、清理和状态摘要。
    """

    def __init__(self, db_path: str = "data/checkpoints.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._saver = None

    def get_saver(self):
        """懒加载 SqliteSaver"""
        if self._saver is None:
            try:
                from langgraph.checkpoint.sqlite import SqliteSaver
                self._saver = SqliteSaver.from_conn_string(str(self.db_path))
                logger.info(f"CheckpointManager 初始化: {self.db_path}")
            except ImportError:
                logger.warning("langgraph.checkpoint.sqlite 不可用,使用内存模式")
                self._saver = None
        return self._saver

    def list_checkpoints(self, thread_id: str) -> List[Dict[str, Any]]:
        """列出某线程的所有检查点"""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(
                "SELECT checkpoint_id, checkpoint_ns, created_at, metadata "
                "FROM checkpoints WHERE thread_id=? ORDER BY created_at DESC",
                (thread_id,),
            ).fetchall()
            return [dict(r) for r in rows]
        except sqlite3.OperationalError:
            return []
        finally:
            conn.close()

    def delete_thread(self, thread_id: str) -> bool:
        """删除某线程的全部检查点"""
        conn = sqlite3.connect(str(self.db_path))
        try:
            conn.execute("DELETE FROM checkpoints WHERE thread_id=?", (thread_id,))
            conn.execute("DELETE FROM checkpoint_writes WHERE thread_id=?", (thread_id,))
            conn.commit()
            return True
        except sqlite3.OperationalError:
            return False
        finally:
            conn.close()

    def cleanup_old(self, days: int = 30):
        """清理过期检查点"""
        conn = sqlite3.connect(str(self.db_path))
        try:
            conn.execute(
                "DELETE FROM checkpoints WHERE created_at < datetime('now', ?)",
                (f"-{days} days",),
            )
            conn.commit()
        except sqlite3.OperationalError:
            pass
        finally:
            conn.close()

    def get_stats(self) -> Dict[str, Any]:
        """获取检查点统计"""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            total = conn.execute("SELECT COUNT(*) as n FROM checkpoints").fetchone()["n"]
            threads = conn.execute(
                "SELECT COUNT(DISTINCT thread_id) as n FROM checkpoints"
            ).fetchone()["n"]
            return {"total_checkpoints": total, "active_threads": threads}
        except sqlite3.OperationalError:
            return {"total_checkpoints": 0, "active_threads": 0}
        finally:
            conn.close()
