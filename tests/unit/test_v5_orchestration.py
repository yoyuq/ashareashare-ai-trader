"""v5.6 P1 编排层测试 (P1-7 可观测性 / P1-8 注册表一致性 / P1-9 数值安全)

覆盖:
  P1-7 checkpoint 表结构对齐 LangGraph SqliteSaver (无 created_at 列)
  P1-8 硬编码 AGENT_REGISTRY 与 config/agents.yaml 无漂移
  P1-9 NumericSafetyChecker 按量级绝对容差 + 多参数豁免
"""

import sqlite3

import pytest
import yaml

from agent.sub_agents import AGENT_REGISTRY
from agent.tools.code_executor import ComputedNumber, NumericSafetyChecker
from agent.orchestration.checkpoint import CheckpointManager


# ═══════════════════════════════════════════════════════════════
# P1-8: 双注册表一致性 — 硬编码 AGENT_REGISTRY 与 YAML 对齐
# ═══════════════════════════════════════════════════════════════

def test_registry_keys_match_agents_yaml():
    """硬编码注册表键集合必须与 config/agents.yaml 完全一致 (无漂移)"""
    with open("config/agents.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    yaml_keys = set(cfg["agents"].keys())
    hardcoded_keys = set(AGENT_REGISTRY.keys())
    assert hardcoded_keys == yaml_keys, (
        f"注册表漂移 — 仅硬编码: {hardcoded_keys - yaml_keys}, "
        f"仅YAML: {yaml_keys - hardcoded_keys}"
    )


def test_registry_class_paths_resolve_to_same_classes():
    """YAML 的 class 路径必须解析到与硬编码注册表相同的类对象"""
    with open("config/agents.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    for name, cls in AGENT_REGISTRY.items():
        class_path = cfg["agents"][name]["class"]
        mod_path, cls_name = class_path.rsplit(".", 1)
        import importlib
        mod = importlib.import_module(mod_path)
        resolved = getattr(mod, cls_name)
        assert resolved is cls, (
            f"{name}: YAML class {class_path} 解析到 {resolved!r}, "
            f"与硬编码 {cls!r} 不一致"
        )


# ═══════════════════════════════════════════════════════════════
# P1-9: NumericSafetyChecker — 按量级绝对容差 + 多参数豁免
# ═══════════════════════════════════════════════════════════════

def _checker(*values):
    computed = {f"v{i}": ComputedNumber(name=f"v{i}", value=v) for i, v in enumerate(values)}
    return NumericSafetyChecker(computed)


def test_multi_param_indicator_exempt():
    """MACD(12,26,9) 中的 12/26/9 是指标参数, 不应被判为未溯源数字"""
    checker = _checker(0.85)  # 只有一个合法计算值
    report = "MACD(12,26,9) 出现金叉, RSI(14)=0.85"
    is_safe, violations = checker.validate_report(report)
    assert is_safe, f"多参数指标豁免失败: {violations}"


def test_magnitude_tolerance_large_value():
    """大数字按相对容差: 1000.0 与 1002 差值 2, 在 tol=5.05 内, 应通过"""
    checker = _checker(1000.0)
    is_safe, _ = checker.validate_report("当前成交额约 1002 元")
    assert is_safe


def test_magnitude_tolerance_small_value():
    """小数字按绝对下限容差: 0.85 与 0.86 差值 0.01, 在 tol=0.054 内, 应通过"""
    checker = _checker(0.85)
    is_safe, _ = checker.validate_report("RSI(14) 约 0.86")
    assert is_safe


def test_unsourced_large_number_flagged():
    """无溯源的离群大数字仍应被拦截"""
    checker = _checker(1.0)
    is_safe, violations = checker.validate_report("预计目标价 9999 元")
    assert not is_safe
    assert any("9999" in v for v in violations)


def test_negative_and_fraction_numbers():
    """负数与带单位的小数应能正确匹配"""
    checker = _checker(-0.42)
    is_safe, _ = checker.validate_report("动量回落 -0.42")
    assert is_safe


# ═══════════════════════════════════════════════════════════════
# P1-7: checkpoint 表结构对齐 LangGraph SqliteSaver
# ═══════════════════════════════════════════════════════════════

_LANGGRAPH_SCHEMA = """
CREATE TABLE checkpoints (
    thread_id TEXT NOT NULL,
    checkpoint_ns TEXT NOT NULL DEFAULT '',
    checkpoint_id TEXT NOT NULL,
    parent_checkpoint_id TEXT,
    type TEXT,
    checkpoint BLOB,
    metadata BLOB,
    PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id)
);
CREATE TABLE checkpoint_writes (
    thread_id TEXT NOT NULL,
    checkpoint_ns TEXT NOT NULL DEFAULT '',
    checkpoint_id TEXT NOT NULL,
    task_id TEXT NOT NULL,
    idx INTEGER NOT NULL,
    channel TEXT NOT NULL,
    type TEXT,
    blob BLOB,
    PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id, task_id, idx)
);
CREATE TABLE checkpoint_blobs (
    thread_id TEXT NOT NULL,
    checkpoint_ns TEXT NOT NULL DEFAULT '',
    channel TEXT NOT NULL,
    version TEXT NOT NULL,
    type TEXT,
    blob BLOB,
    PRIMARY KEY (thread_id, checkpoint_ns, channel, version)
);
"""


def test_checkpoint_list_against_langgraph_schema(tmp_path):
    """list_checkpoints 必须适配无 created_at 列的真实 LangGraph 表结构"""
    db = tmp_path / "cp.db"
    conn = sqlite3.connect(str(db))
    conn.executescript(_LANGGRAPH_SCHEMA)
    conn.execute(
        "INSERT INTO checkpoints (thread_id, checkpoint_ns, checkpoint_id, parent_checkpoint_id, type) "
        "VALUES (?, '', ?, NULL, 'default')",
        ("t1", "1efc0010-0000-0000-0000-000000000001"),
    )
    conn.commit()
    conn.close()

    cpm = CheckpointManager(db_path=str(db))
    rows = cpm.list_checkpoints("t1")
    assert len(rows) == 1
    assert rows[0]["thread_id"] == "t1"
    assert rows[0]["checkpoint_id"] == "1efc0010-0000-0000-0000-000000000001"
    # 返回结构不再依赖不存在的 created_at 列
    assert "created_at" not in rows[0]


def test_checkpoint_stats_and_delete_against_langgraph_schema(tmp_path):
    """get_stats 与 delete_thread 在真实 schema 上可用"""
    db = tmp_path / "cp.db"
    conn = sqlite3.connect(str(db))
    conn.executescript(_LANGGRAPH_SCHEMA)
    conn.execute(
        "INSERT INTO checkpoints (thread_id, checkpoint_ns, checkpoint_id) VALUES ('t1','','c1'), ('t2','','c2')"
    )
    conn.commit()
    conn.close()

    cpm = CheckpointManager(db_path=str(db))
    stats = cpm.get_stats()
    assert stats["total_checkpoints"] == 2
    assert stats["active_threads"] == 2

    assert cpm.delete_thread("t1") is True
    assert cpm.get_stats()["total_checkpoints"] == 1
