"""测试器 — 自主学习闭环的"留/删"判定。

两条真实测试路径(全程零模拟, 缺数据即报错):
- fact 型: 与项目规则库(RAG 检索)做一致性比对。
- rule 型: 翻译成预置模板 → 真实回放数据上回测 A/B(带规则 vs 买入持有基线),
  用预注册判据决定留/删。

v5.9 忠实测试: 模板库按"规则类别"忠实映射(趋势/均值回归/低估值/动量排序/
固定止损/移动止损/量能突破/支撑位止损/时间止损), 不再把所有规则硬塞进 RSI/MA 代理。
规则无法被任一模板忠实表达时, 诚实输出 not_yet_testable(测不了就说测不了), 不做假验证。

v5.10 判据 v2 (预注册, 禁止事后调参追赢): "改善"分两条通道 —
(a) 风险调整改善 (夏普提升且回撤不显著恶化); (b) 降险改善 (回撤收窄 >1pp 且收益牺牲 ≤3pp)。
(b) 让降险类规则(止损等)有机会被保留, 不必跑赢买入持有。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date as _date
from typing import Optional

import pandas as pd
from loguru import logger

from .researcher import KnowledgeCandidate


@dataclass
class TestResult:
    verdict: str                    # verified | rejected | inconclusive | not_yet_testable
    reason: str = ""
    metric_delta: dict = field(default_factory=dict)
    windows_tested: list[str] = field(default_factory=list)
    template: Optional[str] = None  # rule 型: 命中的忠实模板 (供滚动重测复用, 避免 LLM 重译漂移)
    params: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = {"verdict": self.verdict, "reason": self.reason,
             "metric_delta": self.metric_delta, "windows_tested": self.windows_tested}
        if self.template:
            d["template"] = self.template
            d["params"] = self.params
        return d


# ── 预注册判据 (禁止事后调参追赢): 至少 KEEP_MIN_WINDOWS 个独立窗口同向改善才留 ──
KEEP_MIN_WINDOWS = 2
DD_IMPROVE_MIN = 1.0      # 预注册: 最大回撤收窄 > 1pp 才算"降险改善" (v2 新增)
RETURN_TOLERANCE = -3.0   # 预注册: 降险可接受的总收益牺牲上限 (回报差 > -3pp, v2 新增)

# 预注册参数规格: (忠实默认, 下界, 上界)。越界视为 LLM 翻译幻觉 → 回落到忠实默认。
# 例: 低估值模板默认 max_pe=15, 但翻译官曾幻觉出 max_pe=0.5 (PE<0.5 实为空组合, sharpe 崩到 -12M)。
# 这不是事后调参 (区间预注册), 只是把无意义参数挡回模板文档写明的默认值。
_PARAM_SPECS: dict = {
    "ma_cross":        {"fast": (5, 2, 60), "slow": (20, 10, 250)},
    "rsi_reversal":    {"period": (14, 5, 40), "oversold": (30, 10, 45), "overbought": (70, 55, 90)},
    "low_pe_value":    {"max_pe": (15.0, 5.0, 100.0), "max_pb": (2.0, 0.5, 8.0)},
    "momentum_rank":   {"lookback": (20, 5, 120), "top_n": (5, 1, 30), "rebalance_days": (5, 1, 60)},
    "stop_loss_fixed": {"stop_pct": (0.08, 0.01, 0.30)},
    "trailing_stop":   {"trail_pct": (0.10, 0.02, 0.40)},
    "volume_breakout": {"vol_mult": (2.0, 1.2, 6.0)},
    "support_stop":    {"support_lookback": (20, 5, 120)},
    "time_stop":       {"max_hold_days": (20, 3, 250)},
}


def _sanitize_params(template: str, params: dict) -> dict:
    """把翻译出的参数夹到预注册合理区间 (防幻觉, 如 max_pe=0.5)。

    缺失/非数值/越界 → 回落到模板文档写明的忠实默认; 区间内原样保留。
    关系约束: ma_cross 须 fast<slow; rsi_reversal 须 oversold<overbought。
    """
    out = dict(params or {})
    for key, (dflt, lo, hi) in _PARAM_SPECS.get(template, {}).items():
        try:
            v = float(out[key])
        except (TypeError, ValueError, KeyError):
            out[key] = dflt
            continue
        out[key] = v if lo <= v <= hi else dflt
    if template == "ma_cross" and out.get("fast", 0) >= out.get("slow", 0):
        out["fast"], out["slow"] = 5, 20
    if template == "rsi_reversal" and out.get("oversold", 0) >= out.get("overbought", 0):
        out["oversold"], out["overbought"] = 30, 70
    return out


def apply_keep_criterion(metric_deltas: list[dict]) -> TestResult:
    """预注册留/删判据 v2 (纯函数, 可单测, 禁止事后调参追赢)。

    两种"同向改善" (任一满足即计 1 个改善窗口):
    (a) 风险调整改善: sharpe_delta>0 且回撤不显著恶化 (dd_delta >= -max(1.0, |s|*2))。
    (b) 降险改善: dd_delta > DD_IMPROVE_MIN (回撤收窄 >1pp) 且 return_delta > RETURN_TOLERANCE
        (总收益牺牲 ≤3pp)。—— 让"降险类规则"(止损等)有机会被保留, 不必跑赢买入持有。

    改善窗口数 ≥ KEEP_MIN_WINDOWS 才留; 0 窗口删; 1 窗口证据不足。
    v2 是 v1 的严格超集 (仅新增 (b) 通道), 故原判据下 verified 的规则在 v2 下仍 verified。
    """
    improved = 0
    risk_reduction = 0
    for d in metric_deltas:
        s = d.get("sharpe_delta", 0.0) or 0.0
        dd = d.get("dd_delta", 0.0) or 0.0  # dd_delta>0 = 规则回撤更浅 (改善)
        ret = d.get("return_delta", 0.0) or 0.0
        adj_win = s > 0 and dd >= -max(1.0, abs(s) * 2)
        dd_win = dd > DD_IMPROVE_MIN and ret > RETURN_TOLERANCE
        if adj_win or dd_win:
            improved += 1
            if dd_win and not adj_win:
                risk_reduction += 1
    windows = [d.get("window", "") for d in metric_deltas]
    total = len(metric_deltas)
    if improved >= KEEP_MIN_WINDOWS:
        return TestResult("verified", f"{improved}/{total} 窗口改善 (其中 {risk_reduction} 个纯降险)",
                          {"improved_windows": improved, "total_windows": total,
                           "risk_reduction_windows": risk_reduction}, windows)
    if improved == 0:
        return TestResult("rejected", f"0/{total} 窗口改善, 规则无增益",
                          {"improved_windows": 0, "total_windows": total,
                           "risk_reduction_windows": 0}, windows)
    return TestResult("inconclusive", f"仅 {improved}/{total} 窗口改善, 证据不足",
                      {"improved_windows": improved, "total_windows": total,
                       "risk_reduction_windows": risk_reduction}, windows)


# ── fact 型: 规则库一致性 ──

_FACT_PROMPT = """你是交易知识核验官。判断一条"外部知识陈述"与系统规则库是否一致。

三选一 (严格区分"冲突"与"未覆盖"):
- consistent: 规则库有相同维度的表述, 且方向一致 (可能更具体/重复表述)。
- contradicts: 规则库在**同一维度**上做了**方向相反**的明确表述。
- not_covered: 规则库**没有提及**该维度/细节。

判据红线 (务必遵守):
1. "规则库没写" = not_covered, 不是 contradicts。沉默永远不构成冲突。
2. contradicts 必须能指出规则库里那条"反着说"的具体条目; 指不出 → not_covered。
3. 一条陈述同时含"一致部分"和"未覆盖部分"时, 以**未覆盖**为准 (not_covered), 不要因部分一致就忽略未覆盖的细节。
4. 拿不准 contradicts 还是 not_covered 时, 选 not_covered (保守, 避免误报冲突)。

严格输出 JSON (无其他文字): {"verdict": "consistent|contradicts|not_covered", "reason": "一句话依据"}
"""


async def test_fact(candidate: KnowledgeCandidate, km=None) -> TestResult:
    """fact 型: 与规则库一致性比对 (真实, 无市场模拟)。"""
    hits = []
    if km is not None:
        try:
            hits = km.retrieve(candidate.claim, top_k=5, mode="hybrid")
        except Exception as e:
            logger.warning(f"[测试器] 规则库检索失败: {e}")
    ref_txt = "\n".join(f"- [{h.get('source', '')}] {h.get('doc', '')[:200]}" for h in hits) or "(规则库无相关条目)"
    user_msg = f"【外部知识】{candidate.claim}\n【规则库相关条目】\n{ref_txt}\n请判定一致性。"
    from models.router import get_shared_router
    result = await get_shared_router().route(
        messages=[{"role": "system", "content": _FACT_PROMPT},
                  {"role": "user", "content": user_msg}],
        task_type="external_fact_check", temperature=0.2, max_tokens=300,
        extra_body={"thinking": {"type": "disabled"}},
    )
    data = _parse_json((result.response or "").strip())
    v = str(data.get("verdict", "not_covered")).lower()
    mapping = {"consistent": "verified", "contradicts": "rejected", "not_covered": "inconclusive"}
    return TestResult(mapping.get(v, "inconclusive"), data.get("reason", ""))


# ── rule 型: 忠实模板库 + 真实回测 A/B ──

# 忠实模板集: 每个模板 = 一类可真实执行的规则, 映射到 broker 动作, 而非指标代理。
TEMPLATES = {
    "ma_cross":          "均线趋势 (fast/slow 交叉进出场)",
    "rsi_reversal":      "RSI 均值回归 (超卖买/超买卖)",
    "low_pe_value":      "低估值价值 (PE/PB 阈值筛选)",
    "momentum_rank":     "动量/趋势排序 (N日动量取 top_n 定期再平衡)",
    "stop_loss_fixed":   "固定止损 (买入持有 + X% 止损)",
    "trailing_stop":     "移动止损 (买入持有 + X% 回撤止盈)",
    "volume_breakout":   "量能突破 (放量买, 持有)",
    "support_stop":      "支撑位止损 (买入持有 + 跌破近N日前低止损)",
    "time_stop":         "时间止损 (买入持有 + 持有超N日离场)",
}

_RULE_TRANSLATE_PROMPT = """你是策略翻译官。把一条自然语言交易规则**忠实**翻译成"模板+参数"。

可用模板 (只能选一个):
- ma_cross: 均线趋势 (params: fast=5, slow=20)
- rsi_reversal: RSI 均值回归 (params: period=14, oversold=30, overbought=70)
- low_pe_value: 低估值价值 (params: max_pe=15, max_pb=2.0)  — 仅当规则讲 PE/PB/估值
- momentum_rank: 动量/趋势排序 (params: lookback=20, top_n=5, rebalance_days=5)  — 仅当规则讲动量/强势/排序
- stop_loss_fixed: 固定止损 (params: stop_pct=0.08)  — 仅当规则讲固定止损
- trailing_stop: 移动止损 (params: trail_pct=0.10)  — 仅当规则讲移动止损/回撤止盈
- volume_breakout: 量能突破 (params: vol_mult=2.0)  — 仅当规则讲放量/量能
- support_stop: 支撑位止损 (params: support_lookback=20)  — 仅当规则讲支撑位/跌破前低止损
- time_stop: 时间止损 (params: max_hold_days=20)  — 仅当规则讲持有期限/时间止损

**重要**: 规则本质属于哪一类就选哪一类; 若无法被以上任一模板忠实表达(如纯仓位管理、
心理偏差、需要非量价数据的复杂多因子), 必须选 not_yet_testable, 不要硬套模板造假验证。

严格输出 JSON (无其他文字):
{"template": "ma_cross|rsi_reversal|low_pe_value|momentum_rank|stop_loss_fixed|trailing_stop|volume_breakout|support_stop|time_stop|not_yet_testable", "params": {...}, "reason": "一句话"}
"""


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, 1e-10)
    return 100.0 - 100.0 / (1.0 + rs)


def _compute_indicators(df: pd.DataFrame, template: str, params: dict) -> pd.DataFrame:
    df = df.sort_values("date").copy()
    if template == "rsi_reversal":
        df["rsi"] = _rsi(df["close"], int(params.get("period", 14)))
    elif template == "ma_cross":
        df["ma_fast"] = df["close"].rolling(int(params.get("fast", 5))).mean()
        df["ma_slow"] = df["close"].rolling(int(params.get("slow", 20))).mean()
    elif template == "momentum_rank":
        df["mom"] = df["close"].pct_change(int(params.get("lookback", 20))) * 100
    elif template == "volume_breakout":
        df["vol_ma20"] = df["volume"].rolling(20).mean()
    elif template == "support_stop":
        # 支撑位 = 近 N 日前低 (shift 使今日 close 可跌破前低)
        df["support"] = df["low"].rolling(int(params.get("support_lookback", 20))).min().shift(1)
    # low_pe_value / stop_loss_fixed / trailing_stop / time_stop: 用原始列(peTTM/pbMRQ)或持仓成本/峰值/持有天数, 无需新列
    return df


def _qty(price: float, position_cash: float) -> int:
    if price is None or price <= 0 or pd.isna(price):
        return 0
    q = int(position_cash / price / 100) * 100
    return max(q, 100)


def _is_holding(broker, sym) -> bool:
    pos = broker.account.positions.get(sym)
    return pos is not None and pos.quantity > 0


def _buyhold_strategy(position_cash: float):
    """基线: 首日买入持有 (只买一次)。"""
    def strategy(today, bars, broker):
        for sym, bar in bars.items():
            close = bar.get("close")
            if not _is_holding(broker, sym) and close is not None and not pd.isna(close):
                broker.buy(sym, _qty(close, position_cash))
    return strategy


def _make_strategy(template: str, params: dict, position_cash: float):
    """按模板生成规则策略 (真实 broker 动作)。基线统一为 _buyhold_strategy。"""
    state: dict = {}  # 跨交易日状态 (trailing_stop 峰值 / momentum_rank 交易日计数)

    def _entry_exit(today, bars, broker, cond_buy, cond_sell):
        for sym, bar in bars.items():
            close = bar.get("close")
            if close is None or pd.isna(close):
                continue
            holding = _is_holding(broker, sym)
            if not holding and cond_buy(sym, bar):
                broker.buy(sym, _qty(close, position_cash))
            elif holding and cond_sell(sym, bar):
                broker.sell(sym, broker.account.positions[sym].quantity)

    if template == "rsi_reversal":
        oversold = float(params.get("oversold", 30)); overbought = float(params.get("overbought", 70))
        def cb(sym, bar):
            r = bar.get("rsi"); return r is not None and not pd.isna(r) and r < oversold
        def cs(sym, bar):
            r = bar.get("rsi"); return r is not None and not pd.isna(r) and r > overbought
        return lambda today, bars, broker: _entry_exit(today, bars, broker, cb, cs)

    if template == "ma_cross":
        def cb(sym, bar):
            f, s = bar.get("ma_fast"), bar.get("ma_slow")
            return f is not None and s is not None and not pd.isna(f) and not pd.isna(s) and f > s
        def cs(sym, bar):
            f, s = bar.get("ma_fast"), bar.get("ma_slow")
            return f is not None and s is not None and not pd.isna(f) and not pd.isna(s) and f < s
        return lambda today, bars, broker: _entry_exit(today, bars, broker, cb, cs)

    if template == "low_pe_value":
        max_pe = float(params.get("max_pe", 15)); max_pb = float(params.get("max_pb", 2.0))
        def _valid(bar):
            pe, pb = bar.get("peTTM"), bar.get("pbMRQ")
            return (pe is not None and pb is not None and not pd.isna(pe) and not pd.isna(pb)
                    and pe > 0 and pb > 0)
        def cb(sym, bar):
            return _valid(bar) and bar.get("peTTM") < max_pe and bar.get("pbMRQ") < max_pb
        def cs(sym, bar):
            return (not _valid(bar)) or bar.get("peTTM") > max_pe * 1.5
        return lambda today, bars, broker: _entry_exit(today, bars, broker, cb, cs)

    if template == "momentum_rank":
        lookback = int(params.get("lookback", 20)); top_n = int(params.get("top_n", 5))
        reb = max(int(params.get("rebalance_days", 5)), 1)
        state["day"] = 0
        def strategy(today, bars, broker):
            state["day"] += 1
            if state["day"] % reb != 1:
                return
            ranked = sorted(
                ((s, bar.get("mom")) for s, bar in bars.items()
                 if bar.get("mom") is not None and not pd.isna(bar.get("mom"))),
                key=lambda kv: kv[1], reverse=True)
            top = {s for s, _ in ranked[:top_n]}
            for sym in list(broker.account.positions.keys()):
                if sym not in top:
                    broker.sell(sym, broker.account.positions[sym].quantity)
            for sym in top:
                if not _is_holding(broker, sym):
                    close = bars.get(sym, {}).get("close")
                    if close is not None and not pd.isna(close):
                        broker.buy(sym, _qty(close, position_cash))
        return strategy

    if template == "stop_loss_fixed":
        stop_pct = float(params.get("stop_pct", 0.08))
        def strategy(today, bars, broker):
            for sym, bar in bars.items():
                close = bar.get("close")
                if close is None or pd.isna(close):
                    continue
                if not _is_holding(broker, sym):
                    broker.buy(sym, _qty(close, position_cash))
                else:
                    pos = broker.account.positions[sym]
                    if close <= pos.avg_cost * (1 - stop_pct):
                        broker.sell(sym, pos.quantity)
        return strategy

    if template == "trailing_stop":
        trail_pct = float(params.get("trail_pct", 0.10))
        def strategy(today, bars, broker):
            for sym, bar in bars.items():
                close = bar.get("close")
                if close is None or pd.isna(close):
                    continue
                if not _is_holding(broker, sym):
                    broker.buy(sym, _qty(close, position_cash))
                    state[sym] = close
                else:
                    peak = max(state.get(sym, close), close)
                    state[sym] = peak
                    if close <= peak * (1 - trail_pct):
                        broker.sell(sym, broker.account.positions[sym].quantity)
                        state.pop(sym, None)
        return strategy

    if template == "volume_breakout":
        vol_mult = float(params.get("vol_mult", 2.0))
        def cb(sym, bar):
            v, vm = bar.get("volume"), bar.get("vol_ma20")
            return (v is not None and vm is not None and not pd.isna(vm) and vm > 0 and v > vol_mult * vm)
        return lambda today, bars, broker: _entry_exit(today, bars, broker, cb, lambda s, b: False)

    if template == "support_stop":
        def strategy(today, bars, broker):
            for sym, bar in bars.items():
                close = bar.get("close")
                if close is None or pd.isna(close):
                    continue
                if not _is_holding(broker, sym):
                    broker.buy(sym, _qty(close, position_cash))
                else:
                    sup = bar.get("support")
                    if sup is not None and not pd.isna(sup) and close < sup:
                        broker.sell(sym, broker.account.positions[sym].quantity)
        return strategy

    if template == "time_stop":
        max_days = int(params.get("max_hold_days", 20))
        state["hold_days"] = {}
        def strategy(today, bars, broker):
            for sym, bar in bars.items():
                close = bar.get("close")
                if close is None or pd.isna(close):
                    continue
                if not _is_holding(broker, sym):
                    broker.buy(sym, _qty(close, position_cash))
                    state["hold_days"][sym] = 0
                else:
                    d = state["hold_days"].get(sym, 0) + 1
                    state["hold_days"][sym] = d
                    if d >= max_days:
                        broker.sell(sym, broker.account.positions[sym].quantity)
                        state["hold_days"].pop(sym, None)
        return strategy

    # 未知模板: 回退到买入持有 (不应到达; test_rule 已拦截 not_yet_testable)
    return _buyhold_strategy(position_cash)


def resolve_template(parsed: dict):
    """纯函数: 解析翻译结果 → (template, params, reason)。

    template 为 None 表示 not_yet_testable (无法被现有模板忠实表达, 不做代理假验证)。
    """
    template = str(parsed.get("template", "not_yet_testable"))
    params = parsed.get("params") or {}
    reason = parsed.get("reason", "")
    if template == "not_yet_testable" or template not in TEMPLATES:
        return None, params, (reason or "规则无法被现有模板忠实表达, 暂不可测")
    return template, params, reason


def _load_universe(data_file: str, universe_size: int) -> tuple[list[str], pd.DataFrame]:
    """加载真实 replay_data parquet, 返回 (universe symbols, 全量 df)。缺文件 → 报错。"""
    import os
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"回放数据不存在: {data_file} (全程零模拟: 缺数据即报错)")
    df = pd.read_parquet(data_file)
    if "symbol" not in df.columns or "date" not in df.columns:
        raise ValueError(f"回放数据缺 symbol/date 列: {data_file}")
    symbols = sorted(df["symbol"].unique().tolist())[:universe_size]
    return symbols, df


def _run_backtest(symbols, df, template, params, position_cash, rule_on, cfg_dates):
    """在给定 universe 上跑一次回测, 返回 (total_return, sharpe, max_drawdown)。"""
    from backtest.engine import BacktestConfig, EventDrivenBacktestEngine

    cfg = BacktestConfig(initial_capital=100000.0, start_date=cfg_dates[0], end_date=cfg_dates[1])
    engine = EventDrivenBacktestEngine(cfg, deferred_execution=True)
    for sym in symbols:
        sdf = df[df["symbol"] == sym].copy()
        if "pctChg" in sdf.columns:
            sdf = sdf.rename(columns={"pctChg": "pct_change"})
        sdf = _compute_indicators(sdf, template, params)
        engine.load_data(sym, sdf)
    strat = _make_strategy(template, params, position_cash) if rule_on else _buyhold_strategy(position_cash)
    res = engine.run(strat, progress_bar=False)
    return res.total_return, res.sharpe_ratio, res.max_drawdown


def _run_rule_deltas(template: str, params: dict, data_files: list[str],
                     universe_size: int = 20) -> list[dict]:
    """在给定窗口上跑规则 vs 基线回测, 返回 per-window delta 列表。

    供 test_rule 与滚动重测 (revalidate) 复用: 同一模板/参数只换数据窗口, 避免重复代码。
    """
    params = _sanitize_params(template, params)  # 防 LLM 翻译幻觉 (如 max_pe=0.5)
    deltas = []
    for df_path in data_files:
        window = df_path.split("daily_")[-1].replace(".parquet", "")
        try:
            symbols, df = _load_universe(df_path, universe_size)
        except (FileNotFoundError, ValueError) as e:
            logger.warning(f"[测试器] 窗口 {df_path} 跳过: {e}")
            continue
        d = pd.to_datetime(df["date"])
        cfg_dates = (_date(int(d.min().year), int(d.min().month), int(d.min().day)),
                     _date(int(d.max().year), int(d.max().month), int(d.max().day)))
        position_cash = 100000.0 * 0.9 / max(len(symbols), 1)
        try:
            base_ret, base_sharpe, base_dd = _run_backtest(symbols, df, template, params, position_cash, False, cfg_dates)
            rule_ret, rule_sharpe, rule_dd = _run_backtest(symbols, df, template, params, position_cash, True, cfg_dates)
        except Exception as e:
            logger.warning(f"[测试器] 窗口 {window} 回测失败: {e}")
            continue
        deltas.append({
            "window": window,
            "sharpe_delta": rule_sharpe - base_sharpe,
            # max_drawdown 为负值(如 -15.0), 规则回撤更浅 → rule_dd 更接近 0
            "dd_delta": rule_dd - base_dd,      # >0 = 规则回撤更浅 (改善)
            "return_delta": rule_ret - base_ret,
        })
    return deltas


async def test_rule(candidate: KnowledgeCandidate, data_files: list[str],
                    universe_size: int = 20) -> TestResult:
    """rule 型: 真实回测 A/B (规则 vs 买入持有), 预注册判据。"""
    from models.router import get_shared_router
    result = await get_shared_router().route(
        messages=[{"role": "system", "content": _RULE_TRANSLATE_PROMPT},
                  {"role": "user", "content": candidate.testable}],
        task_type="external_rule_translate", temperature=0.2, max_tokens=300,
        extra_body={"thinking": {"type": "disabled"}},
    )
    data = _parse_json((result.response or "").strip())
    # 忠实测试红线: 无法忠实映射 → 诚实输出 not_yet_testable, 不做代理假验证
    template, params, reason = resolve_template(data)
    if template is None:
        return TestResult("not_yet_testable", reason)

    deltas = _run_rule_deltas(template, params, data_files, universe_size)
    if not deltas:
        return TestResult("inconclusive", "无可用窗口完成回测 (数据缺失/回测失败)",
                          template=template, params=params, windows_tested=[])
    verdict = apply_keep_criterion(deltas)
    verdict.metric_delta["per_window"] = deltas
    verdict.template = template
    verdict.params = params
    return verdict


def _parse_json(content: str) -> dict:
    content = content.strip()
    if content.startswith("```"):
        content = content.strip("`")
        if content.lower().startswith("json"):
            content = content[4:].strip()
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        s, e = content.find("{"), content.rfind("}")
        if s >= 0 and e > s:
            try:
                return json.loads(content[s:e + 1])
            except json.JSONDecodeError:
                pass
    return {}
