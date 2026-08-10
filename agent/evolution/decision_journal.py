"""决策日志 — 事实层。

记录每天的诊断决策：选了哪位大师、风险等级、仓位系数、理由、关键风险。
纯客观记录，不做评价，供后续复盘使用。

存储格式: JSON Lines，每行一条记录。
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from datetime import date
from pathlib import Path
from typing import Optional


@dataclass
class DecisionRecord:
    """单日诊断决策记录。"""
    date: str                          # YYYY-MM-DD
    market_phase: str                  # trend_up / trend_down / range / ...
    dominant_master: str               # 主导大师
    secondary_master: str              # 次选大师
    risk_level: int                    # 1-5
    position_multiplier: float         # 0.3-1.6
    max_positions_adj: int             # -8 ~ +8
    key_risks: list[str] = field(default_factory=list)
    diagnosis: str = ""                # 诊断理由（200字以内）

    # v5.2 对抗票 — 主导大师 vs 对抗大师 独立评分与分歧裁决记录
    adversarial_risk_level: Optional[int] = None   # 对抗大师独立风险评分
    adversarial_applied: Optional[str] = None      # 分歧裁决: conservative/dampen/None
    adversarial_divergence: Optional[int] = None   # 分歧度 |主导-对抗|

    # 环境快照（用于事后复盘时知道当时的输入）
    market_snapshot: dict = field(default_factory=dict)  # 上涨占比、涨跌停、PE、成交额等
    regime: str = ""                   # 市场状态（来自regime检测器）
    crowding_score: float = 0.0        # 拥挤度分数
    crowding_signal: str = ""          # 拥挤度信号

    # v5.4 组合级反事实 — 当日持仓快照 (PIT 正确, 供次日复盘算个股级贡献)
    # 每项: {symbol, name, qty, price(T日收盘), value, weight(占总资产)}
    positions_snapshot: list = field(default_factory=list)
    total_value: float = 0.0           # T日组合总资产 (现金+持仓+指数书)

    # 复盘结果（T+1填写）
    review: Optional[dict] = None      # {verdict, actual_outcome, correct_factors, missed_factors, lesson}

    def to_json(self) -> str:
        # numpy 标量 (float32 等) → 原生 float, 保证任意数值可 JSON 序列化
        def _default(o):
            if hasattr(o, "item"):
                try:
                    return o.item()
                except Exception:
                    pass
            raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")
        return json.dumps(asdict(self), ensure_ascii=False, default=_default)

    @classmethod
    def from_json(cls, s: str) -> "DecisionRecord":
        data = json.loads(s)
        return cls(**data)


class DecisionJournal:
    """决策日志存储器。

    用法:
        journal = DecisionJournal("simulation_data/diag_journal.jsonl")
        journal.record(DecisionRecord(date="2021-01-05", ...))
        records = journal.load_range("2021-01-01", "2021-01-31")
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, DecisionRecord] = {}
        self._load_all()

    def _load_all(self):
        if not self.path.exists():
            return
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = DecisionRecord.from_json(line)
                    self._cache[rec.date] = rec
                except json.JSONDecodeError:
                    pass

    def record(self, rec: DecisionRecord) -> None:
        """写入一条决策记录（追加模式）。"""
        self._cache[rec.date] = rec
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(rec.to_json() + "\n")

    def update_review(self, date_str: str, review: dict) -> bool:
        """给某天的记录补上复盘结果。"""
        if date_str not in self._cache:
            return False
        self._cache[date_str].review = review
        # 重写整个文件（因为要修改中间行）
        self._rewrite_all()
        return True

    def _rewrite_all(self):
        tmp = self.path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            for d in sorted(self._cache.keys()):
                f.write(self._cache[d].to_json() + "\n")
        os.replace(tmp, self.path)

    def get(self, date_str: str) -> Optional[DecisionRecord]:
        return self._cache.get(date_str)

    def load_range(self, start: str, end: str) -> list[DecisionRecord]:
        """加载日期范围内的记录，按日期排序。"""
        return [
            self._cache[d]
            for d in sorted(self._cache.keys())
            if start <= d <= end
        ]

    def unreviewed(self, before: str) -> list[DecisionRecord]:
        """找出所有还没复盘、且日期在 before 之前的记录。"""
        return [
            self._cache[d]
            for d in sorted(self._cache.keys())
            if d < before and self._cache[d].review is None
        ]

    def __len__(self) -> int:
        return len(self._cache)
