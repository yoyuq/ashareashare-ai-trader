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
                # v5.6 P1-7: SqliteSaver 由独立包 langgraph-checkpoint-sqlite 提供
                # (langgraph 1.x 将 checkpoint 后端拆分为 namespace 包), 未安装时退化为
                # 内存模式 (工作流仍可运行, 仅失去断点续跑能力)。
                logger.warning(
                    "langgraph.checkpoint.sqlite 不可用 (需安装 langgraph-checkpoint-sqlite), "
                    "断点续跑退化为内存模式"
                )
                self._saver = None
        return self._saver

    def list_checkpoints(self, thread_id: str) -> List[Dict[str, Any]]:
        """列出某线程的所有检查点。

        v5.6 P1-7: 对齐 LangGraph SqliteSaver 真实表结构。其 `checkpoints` 表
        无 `created_at` 列 (时间戳封装在 pickled 的 `metadata` BLOB 中, 无法用 SQL
        直接读取), 故仅返回纯文本列; `checkpoint_id` 为时间可排序的 UUIDv6,
        按它降序即近似时间倒序。
        """
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(
                "SELECT thread_id, checkpoint_ns, checkpoint_id, parent_checkpoint_id, type "
                "FROM checkpoints WHERE thread_id=? ORDER BY checkpoint_id DESC",
                (thread_id,),
            ).fetchall()
            return [dict(r) for r in rows]
        except sqlite3.OperationalError:
            return []
        finally:
            conn.close()

    def delete_thread(self, thread_id: str) -> bool:
        """删除某线程的全部检查点 (含写入缓冲与分片 blob)。"""
        conn = sqlite3.connect(str(self.db_path))
        try:
            conn.execute("DELETE FROM checkpoints WHERE thread_id=?", (thread_id,))
            conn.execute("DELETE FROM checkpoint_writes WHERE thread_id=?", (thread_id,))
            # checkpoint_blobs 无 thread_id 列 (主键为 thread_id, checkpoint_ns,
            # channel, version), 按 thread_id 前缀匹配删除
            conn.execute(
                "DELETE FROM checkpoint_blobs WHERE thread_id LIKE ?",
                (f"{thread_id}%",),
            )
            conn.commit()
            return True
        except sqlite3.OperationalError:
            return False
        finally:
            conn.close()

    def cleanup_old(self, days: int = 30):
        """清理过期检查点。

        v5.6 P1-7: LangGraph SqliteSaver 的 `checkpoints` 表无 `created_at` 列,
        无法用 SQL 做时间条件删除; 过期时间戳封装在 pickled metadata 中, 反序列化
        成本高且依赖 langgraph serde。因此本方法退化为按 thread_id 精确删除 (无
        时间语义), 时间级清理需上层 (如 daily_runner) 记录并主动调用 delete_thread。
        """
        logger.warning(
            f"cleanup_old(days={days}) 不可用: LangGraph checkpoint 表无 created_at 列, "
            f"请改用 delete_thread(thread_id) 按线程精确删除"
        )

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
