"""前瞻纸面验证闭环 (阶段2 #108): 把「回测验证过的 edge」在真实未来行情上纸面跟踪。

背景 (roadmap §3.1「前瞻验证闭环(最该补, 最容易)」): 冷落 beta 回测 11/11, 但**从未被未来真实行情
验证过** —— 回测是 in-sample, 纸面前瞻才是 out-of-sample。本模块是那个「任何验证过的 edge 都能自动
纸面跟踪」的 harness, 核心两件事:

  1. **预注册 (register)**: 冻结某 edge 的构建口径 + 入场快照 + 成功/失效判据, 在**未来数据到达前**
     一次性写死 (满足「禁止事后调参追赢」—— 判据先注册, 之后只读不写)。
  2. **跟踪 (track)**: 任意未来日期, 用当前行情算 edge 篮子 vs 匹配 universe vs 上证综指的前瞻收益、
     滚动偏离、判断 edge 是否失效 (滚动偏离阈值预先注册)。

第一个试点 = 冷落 beta (bottom-100 低换手等权, 冻结入场名单持有), 见
`scripts/forward_register_cold_tilt.py` (注册) 与 `scripts/forward_track.py` (跟踪)。

设计要点 (诚实边界):
  - 前瞻跟踪用**价格收益** (full_market_cache 无分红); 基准纪律 #107 已证分红在篮子 vs universe 之间
    大致抵消 (价格/全收益两口径结论一致, 8/11), 故价格收益是公平近似, 注册时显式标注 `total_return: false`。
  - 停牌/退市票 ffill (价格冻结贡献 0 收益), 与回测 `matched_universe_curve` 口径一致。
  - 判据字段 `criterion` 注册后只读; `tracking` 数组由 track 脚本追加, 不覆盖 `criterion`/`entry`。
"""

from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict

# 注册表持久化位置 (git 忽略? 否 —— 这是「判据已注册」的证据, 应随仓保存, 见 reports)
ROOT = Path(__file__).resolve().parent.parent
REGISTRY_PATH = ROOT / "simulation_data" / "forward_validation" / "registry.json"

REGISTRY_VERSION = 1


# --------------------------------------------------------------------------- #
# 纯函数: 等权篮子收益 + 状态判定 (无 I/O, 便于单测)
# --------------------------------------------------------------------------- #

def equalweight_return(entry: Dict[str, float], current: Dict[str, float]) -> float:
    """等权篮子的累计收益 (入场价 → 当前价)。

    口径与回测 `matched_universe_curve` 一致:
      - 停牌/退市 (current 缺失) → ffill, 用入场价 (贡献 0 收益), 不失真也不伪造。
      - 等权 = 各票收益的简单平均 (入场即等权, 持有期不漂移重配)。

    Args:
        entry: {code: 入场价} (仅含篮子在籍票)。
        current: {code: 当前价} (可缺失部分票, 缺失按 ffill)。

    Returns:
        累计收益 (e.g. 0.052 → +5.2%); entry 为空返回 0.0 (空篮子不产生收益也不报错)。
    """
    if not entry:
        return 0.0
    rets = []
    for code, px0 in entry.items():
        if px0 is None or px0 <= 0:
            continue
        px1 = current.get(code, px0)  # 缺失 → ffill 入场价 (0 收益)
        if px1 is None or px1 <= 0:
            px1 = px0
        rets.append(px1 / px0 - 1.0)
    if not rets:
        return 0.0
    return float(sum(rets) / len(rets))


def trading_days_between(start: date, end: date) -> int:
    """start(不含) → end(含) 之间的 A 股交易日数 (周末 + 法定节假日剔除)。

    用于「60 交易日 horizon」判定。依赖 timeutil.is_trading_day 的节假日日历。
    """
    from timeutil import is_trading_day

    if end <= start:
        return 0
    n = 0
    d = start
    while d < end:
        d = d.fromordinal(d.toordinal() + 1)
        if is_trading_day(d):
            n += 1
    return n


def compute_status(
    bet: Dict[str, Any],
    current_basket: Dict[str, float],
    current_universe: Dict[str, float],
    current_index: float,
    current_date: str,
) -> Dict[str, Any]:
    """给定前瞻 bet 与当前行情, 算前瞻收益 + 应用预注册判据。

    Args:
        bet: 注册表里的一个 bet (含 entry.basket_prices / universe_prices / sh_index_close,
             construction.cost_bps, criterion)。
        current_basket: {code: 当前价} (篮子)。
        current_universe: {code: 当前价} (匹配 universe)。
        current_index: 上证综指当前收盘。
        current_date: 当前数据日 (ISO `YYYY-MM-DD`)。

    Returns:
        status dict (含收益、滚动偏离、edge_failing、horizon_reached、primary/secondary_success)。
        判据字段只读自 bet["criterion"], 本函数不改写判据。
    """
    entry = bet["entry"]
    cost_bps = float(bet.get("construction", {}).get("cost_bps", 0.0))

    basket_ret = equalweight_return(entry["basket_prices"], current_basket)
    universe_ret = equalweight_return(entry["universe_prices"], current_universe)
    entry_idx = float(entry["sh_index_close"])
    index_ret = (float(current_index) / entry_idx - 1.0) if entry_idx > 0 else float("nan")

    # 篮子扣费 (cost_bps 一次性全额往返上界, 与 run_cold_tilt_rebalance.py「持有」口径一致)
    cost = cost_bps / 10000.0
    basket_net = basket_ret - cost

    selection_alpha = basket_net - universe_ret
    vs_index = basket_net - index_ret

    entry_date = date.fromisoformat(str(entry.get("entry_date", current_date)))
    try:
        cur_date = date.fromisoformat(str(current_date))
    except ValueError:
        cur_date = entry_date
    elapsed = trading_days_between(entry_date, cur_date)

    crit = bet.get("criterion", {})
    horizon = int(crit.get("horizon_trading_days", 60))
    fail_thr = float(crit.get("failure_threshold_pp", -10.0)) / 100.0

    horizon_reached = elapsed >= horizon
    edge_failing = bool(selection_alpha <= fail_thr)

    status: Dict[str, Any] = {
        "current_date": current_date,
        "elapsed_trading_days": elapsed,
        "basket_return_pct": round(basket_net * 100.0, 3),
        "basket_gross_return_pct": round(basket_ret * 100.0, 3),
        "universe_return_pct": round(universe_ret * 100.0, 3),
        "sh_index_return_pct": round(index_ret * 100.0, 3),
        "selection_alpha_pp": round(selection_alpha * 100.0, 3),
        "vs_index_pp": round(vs_index * 100.0, 3),
        "edge_failing": edge_failing,
        "horizon_reached": horizon_reached,
    }

    # 主/副判据仅在 horizon 到达时落定 (未到达 → None = 尚无结论, 不是失败)
    status["primary_success"] = None
    status["secondary_success"] = None
    if horizon_reached:
        status["primary_success"] = bool(selection_alpha >= 0.0)
        status["secondary_success"] = bool(vs_index >= 0.0)

    return status


# --------------------------------------------------------------------------- #
# 注册表 I/O (唯一读写入口, 避免多点手写 schema 漂移)
# --------------------------------------------------------------------------- #

def load_registry() -> Dict[str, Any]:
    """读注册表 → {"version", "bets": {...}}; 不存在/损坏返回空结构。"""
    if not REGISTRY_PATH.exists():
        return {"version": REGISTRY_VERSION, "bets": {}}
    try:
        reg = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
        if not isinstance(reg, dict) or "bets" not in reg:
            return {"version": REGISTRY_VERSION, "bets": {}}
        return reg
    except (ValueError, OSError):
        return {"version": REGISTRY_VERSION, "bets": {}}


def save_registry(reg: Dict[str, Any]) -> None:
    """原子写注册表 (先写临时文件再 rename, 避免中途崩溃写坏判据)。"""
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = REGISTRY_PATH.with_suffix(".json.tmp")
    reg.setdefault("version", REGISTRY_VERSION)
    reg.setdefault("bets", {})
    tmp.write_text(json.dumps(reg, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    tmp.replace(REGISTRY_PATH)


def register_bet(bet: Dict[str, Any]) -> None:
    """注册 (或覆盖更新) 一个 bet —— 幂等: 已存在则覆盖 entry/criterion (重新注册即重新冻结)。"""
    reg = load_registry()
    bets = reg.setdefault("bets", {})
    bet = dict(bet)
    bet.setdefault("tracking", [])
    bet["registered_at"] = datetime.now().isoformat(timespec="seconds")
    bets[bet["edge_id"]] = bet
    save_registry(reg)


def append_tracking(edge_id: str, status: Dict[str, Any]) -> None:
    """往某 bet 的 tracking 数组追加一次前瞻快照 (不改判据/入场快照)。"""
    reg = load_registry()
    bet = reg.get("bets", {}).get(edge_id)
    if bet is None:
        raise KeyError(f"未注册的 edge_id: {edge_id}")
    bet.setdefault("tracking", []).append(status)
    save_registry(reg)
