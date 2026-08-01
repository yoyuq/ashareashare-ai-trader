"""
kill-test 三路信号臂 (v3.1-deerflow)

回答核心问题: LLM 全管道 / 纯规则引擎 / 随机信号, 谁有真实 alpha?

  - RuleArm  : 确定性技术规则 (基于指标, 无 LLM) — 可回溯历史立即出结果
  - RandomArm: 有种子可复现的随机信号 — 与控制臂同数量, 作为基准
  - LLMArm   : 读取每日分析产物 (reports/data_{date}.json), 随系统日跑积累

所有臂共享同一个股票池和信号日, 保证对比公平。
"""

import json
import random
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


@dataclass
class Signal:
    """一次交易信号"""
    date: str            # YYYY-MM-DD 信号日
    symbol: str          # sh.600519
    action: str          # BUY / SELL
    conviction: float    # 0-1
    arm: str             # rule / random / llm
    price: float = 0.0   # 信号日收盘 (结算用 fallback)

    def key(self) -> str:
        return f"{self.arm}:{self.date}:{self.symbol}:{self.action}"

    def to_dict(self) -> dict:
        return {"date": self.date, "symbol": self.symbol, "action": self.action,
                "conviction": round(self.conviction, 3), "arm": self.arm,
                "price": round(self.price, 2)}


def _f(row: pd.Series, col: str, default: float = 0.0) -> float:
    """安全取指标值 (列缺失 → default)"""
    if col not in row.index:
        return default
    try:
        v = float(row[col])
        return v if v == v else default  # NaN → default
    except (TypeError, ValueError):
        return default


def _asof_row(ind_df: pd.DataFrame, asof: str) -> Optional[pd.Series]:
    """取指标 DataFrame 中 asof 当日及之前的最后一行 (因果, 无未来泄漏)"""
    if ind_df is None or ind_df.empty:
        return None
    date_col = "date" if "date" in ind_df.columns else ind_df.index.name
    dates = pd.to_datetime(ind_df[date_col])
    mask = dates <= pd.Timestamp(asof)
    if not mask.any():
        return None
    return ind_df[mask].iloc[-1]


class RuleArm:
    """纯规则信号臂 — 确定性技术规则, 无 LLM, 可回溯验证"""

    name = "rule"

    def generate(
        self,
        indicator_dfs: Dict[str, pd.DataFrame],
        asof: str,
    ) -> List[Signal]:
        """
        对每只股票按 asof 当日指标产生 BUY/SELL 信号

        规则 (经典且确定性):
          BUY : trend_score>0.5 (多头排列) and 50<rsi_14<72 (未超买) and vol_ratio_5>1.2 (放量)
          SELL: rsi_14>75 (超买) 或 (trend_score<-0.5 且 macd_death_cross>0) (空头+死叉)
        """
        signals = []
        for sym, ind_df in indicator_dfs.items():
            row = _asof_row(ind_df, asof)
            if row is None:
                continue
            ts = _f(row, "trend_score")
            rsi = _f(row, "rsi_14")
            vr = _f(row, "vol_ratio_5")
            mdc = _f(row, "macd_death_cross")
            close = _f(row, "close")

            if ts > 0.5 and 50 < rsi < 72 and vr > 1.2:
                # 多头确信度: 趋势强 + 放量 → 高
                conv = min(0.9, 0.5 + abs(ts) * 0.3 + vr * 0.05)
                signals.append(Signal(asof, sym, "BUY", round(conv, 3),
                                      self.name, price=close))
            elif rsi > 75 or (ts < -0.5 and mdc > 0):
                conv = min(0.85, 0.5 + (rsi - 75) * 0.02 if rsi > 75 else abs(ts) * 0.3)
                signals.append(Signal(asof, sym, "SELL", round(conv, 3),
                                      self.name, price=close))
        return signals


class RandomArm:
    """随机信号臂 — 与规则臂同一天、同数量的 BUY/SELL, 有种子可复现"""

    name = "random"

    def __init__(self, seed: int = 42):
        self.seed = seed

    def generate(
        self,
        universe: List[str],
        asof: str,
        n_buy: int,
        n_sell: int,
    ) -> List[Signal]:
        rng = random.Random(f"{self.seed}:{asof}")  # 按日派生, 跨日可复现
        signals = []
        for action, n in (("BUY", n_buy), ("SELL", n_sell)):
            pool = list(universe)
            rng.shuffle(pool)
            for sym in pool[:n]:
                signals.append(Signal(
                    asof, sym, action,
                    round(rng.uniform(0.3, 0.9), 3), self.name,
                ))
        return signals


class LLMArm:
    """LLM 信号臂 — 读取每日分析产物 (reports/data_{date}.json)

    随系统日跑积累信号; 历史日期若文件不存在则无信号 (诚实标注: LLM 臂需要
    每天真实跑分析才能积累, 初始对比以 rule/random 为主)。
    """

    name = "llm"

    def __init__(self, report_dir: str = "reports"):
        self.report_dir = Path(report_dir)

    def load_daily(self, asof: str) -> List[Signal]:
        path = self.report_dir / f"data_{asof}.json"
        if not path.exists():
            return []
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return []
        recs = data.get("stock_recommendations", {})
        signals = []
        for sym, rec in recs.items():
            if not isinstance(rec, dict):
                continue
            action = rec.get("action", "HOLD")
            if action not in ("BUY", "SELL"):
                continue
            signals.append(Signal(
                asof, sym, action,
                round(float(rec.get("conviction", 0.5)), 3), self.name,
                price=float(rec.get("trade_params", {}).get("entry_price", 0) or 0),
            ))
        return signals
