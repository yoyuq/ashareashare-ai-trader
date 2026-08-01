"""
kill-test 信号跟踪器 — 持久化信号 + N日前向收益结算

设计: 信号持久化到 killtest_data/signals/{arm}.jsonl, 结算结果到
killtest_data/outcomes/{arm}.jsonl。跨多次运行累积, 已结算的不重复结算。

结算口径 (信号质量, 隔离执行/手续费):
  BUY : fwd_return  = close[T+N]/close[T] - 1      (做多收益)
  SELL: directional = -(close[T+N]/close[T] - 1)   (做空收益, 正确=正)
  is_correct = directional > 0
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from killtest.arms import Signal


@dataclass
class Outcome:
    """一条已结算的信号"""
    date: str
    symbol: str
    action: str
    arm: str
    conviction: float
    entry_price: float
    exit_price: float
    fwd_return: float        # 方向化收益: BUY=做多收益, SELL=做空收益
    is_correct: bool

    def to_dict(self) -> dict:
        return {"date": self.date, "symbol": self.symbol, "action": self.action,
                "arm": self.arm, "conviction": self.conviction,
                "entry_price": round(self.entry_price, 3),
                "exit_price": round(self.exit_price, 3),
                "fwd_return": round(self.fwd_return, 4),
                "is_correct": self.is_correct}


class KillTestTracker:
    def __init__(self, data_dir: str = "killtest_data"):
        self.data_dir = Path(data_dir)
        self.signals_dir = self.data_dir / "signals"
        self.outcomes_dir = self.data_dir / "outcomes"

    # ── 信号持久化 ──

    def emit(self, signal: Signal) -> bool:
        """写入一条信号 (按 key 去重, 幂等)"""
        self.signals_dir.mkdir(parents=True, exist_ok=True)
        path = self.signals_dir / f"{signal.arm}.jsonl"
        key = signal.key()
        if self._has_key(path, key):
            return False
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps({**signal.to_dict(), "key": key}, ensure_ascii=False) + "\n")
        return True

    def emit_many(self, signals: List[Signal]) -> int:
        return sum(1 for s in signals if self.emit(s))

    def load_signals(self, arm: Optional[str] = None) -> List[dict]:
        records = []
        paths = (self.signals_dir.glob(f"{arm}.jsonl") if arm
                 else self.signals_dir.glob("*.jsonl"))
        for path in sorted(paths):
            for line in path.read_text(encoding="utf-8").strip().splitlines():
                if line.strip():
                    records.append(json.loads(line))
        return records

    def load_outcomes(self, arm: Optional[str] = None) -> List[Outcome]:
        records = []
        paths = (self.outcomes_dir.glob(f"{arm}.jsonl") if arm
                 else self.outcomes_dir.glob("*.jsonl"))
        for path in sorted(paths):
            for line in path.read_text(encoding="utf-8").strip().splitlines():
                if line.strip():
                    records.append(Outcome(**json.loads(line)))
        return records

    def _has_key(self, path: Path, key: str) -> bool:
        if not path.exists():
            return False
        try:
            for line in path.read_text(encoding="utf-8").strip().splitlines():
                if line.strip() and json.loads(line).get("key") == key:
                    return True
        except (json.JSONDecodeError, OSError):
            pass
        return False

    # ── N日前向收益结算 ──

    def settle(self, arm: str, price_data: Dict[str, pd.DataFrame],
               lookahead: int = 5, force: bool = False) -> List[Outcome]:
        """
        结算某臂所有未结算信号

        Args:
            arm: 臂名
            price_data: {symbol: df} — df 需含 date/close 列 (已 standardize)
            lookahead: N 个交易日后回看
            force: 重算已结算的 (默认跳过)
        """
        self.outcomes_dir.mkdir(parents=True, exist_ok=True)
        out_path = self.outcomes_dir / f"{arm}.jsonl"
        settled_keys = set() if force else {
            o.to_dict()["date"] + o.symbol + o.action
            for o in self.load_outcomes(arm)
        }

        results = []
        for sig in self.load_signals(arm):
            skey = sig["date"] + sig["symbol"] + sig["action"]
            if skey in settled_keys:
                continue
            # 注意: 不能用 `a or b` (DataFrame 的 or 触发歧义真值错误)
            df = price_data.get(sig["symbol"])
            if df is None:
                df = price_data.get(sig["symbol"].split(".")[-1])
            out = self._forward_outcome(sig, df, lookahead)
            if out is None:
                continue  # 数据不足, 保持未结算
            with open(out_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(out.to_dict(), ensure_ascii=False) + "\n")
            results.append(out)
        return results

    @staticmethod
    def _forward_outcome(sig: dict, df: Optional[pd.DataFrame],
                         lookahead: int) -> Optional[Outcome]:
        """计算一条信号的方向化前向收益"""
        if df is None or df.empty or "close" not in df.columns:
            return None
        dates = pd.to_datetime(df["date"]) if "date" in df.columns else pd.to_datetime(df.index)
        closes = pd.to_numeric(df["close"], errors="coerce")

        # 入场: 信号日及之后第一个有效交易日
        mask = dates >= pd.Timestamp(sig["date"])
        if not mask.any():
            return None
        i = int(np.argmax(mask.values))
        entry = closes.iloc[i]
        if entry is None or not np.isfinite(entry) or entry <= 0:
            return None
        # 出场: 入场后第 lookahead 个交易日
        j = i + lookahead
        if j >= len(closes):
            return None
        exit_px = closes.iloc[j]
        if exit_px is None or not np.isfinite(exit_px) or exit_px <= 0:
            return None

        fwd = exit_px / entry - 1.0
        if sig["action"] == "SELL":
            fwd = -fwd
        return Outcome(
            date=sig["date"], symbol=sig["symbol"], action=sig["action"],
            arm=sig["arm"], conviction=float(sig.get("conviction", 0.5)),
            entry_price=float(entry), exit_price=float(exit_px),
            fwd_return=float(fwd), is_correct=bool(fwd > 0),
        )
