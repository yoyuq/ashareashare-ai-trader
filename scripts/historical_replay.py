"""
历史 PIT 回放引擎 — 全市场 → LLM初筛 → LLM分析 → 持仓规划 → T+1开盘执行

复现 daily_runner 的真实流程, 但严格 Point-In-Time (PIT):
  - 决策日 T 只用 ≤T 的数据 (指标因果、截面用 T 日行情)
  - 决策在 T 收盘, 成交在 T+1 开盘 (真实 A 股规则)
  - 绝不使用 >T 的任何信息

流程 (每交易日 T):
  Baostock 全市场日K (含 peTTM/pbMRQ/turn/isST)
    → PIT 截面重建 (price/pct/PE/PB/市值/换手/60日涨跌/量比)
    → PreScreener 规则筛 300
    → LLM Flash 精筛 100
    → LLM 深度分析 (技术/基本面/风险 → BUY/HOLD/SELL)
    → 持仓规划 (卖出信号/止损止盈 + 买入排序/仓位/现金)
    → T+1 开盘成交 → T+1 收盘市值快照

用法:
  python scripts/historical_replay.py --days 5 --window-before 40   # 小验证
  python scripts/historical_replay.py --days 40 --universe full     # 40日试点
"""

import argparse
import asyncio
import json
import os
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))

REPLAY_DIR = Path("replay_data")
REPLAY_DIR.mkdir(exist_ok=True)

# 复用 daily_runner 的 LLM 函数 (忠实复现分析逻辑)
from simulation.daily_runner import (  # noqa: E402
    _flash_screen, _deepseek_analyze, _detect_regime, build_market_ctx, load_macro_context,
)


# ═══════════════════════════════════════════════════════════════
# Phase 0: 全市场日K数据构建 (Baostock, 含基本面字段, PIT 安全)
# ═══════════════════════════════════════════════════════════════

def load_snapshot_basic() -> tuple:
    """
    从 full_market_cache.json (最近交易日快照) 构建基础信息:
      basic: {symbol: {name, total_mv_now, close_now}} — 市值按 close_T/close_now 缩放 (PIT-in-price)
      universe: 全部 symbol 列表
    """
    cache = Path("simulation_data/full_market_cache.json")
    if not cache.exists():
        logger.error(f"缺少快照 {cache} (需先跑一次实时行情缓存)")
        return {}, []
    d = json.loads(cache.read_text(encoding="utf-8"))
    basic, universe = {}, []
    for item in d.get("data", []):
        code = str(item.get("code", ""))
        if not code or not code.isdigit():
            continue
        prefix = "sh" if code.startswith("6") else ("bj" if code.startswith(("8", "4")) else "sz")
        sym = f"{prefix}.{code}"
        basic[sym] = {
            "name": item.get("name", sym),
            "total_mv_now": float(item.get("total_mv", 0) or 0) * 1e8,  # 亿→元
            "close_now": float(item.get("price", 0) or 0),
        }
        universe.append(sym)
    logger.info(f"快照基础: {len(universe)} 只 (日期 {d.get('date')})")
    return basic, universe


def _bs_query(symbol: str, start: str, end: str) -> pd.DataFrame:
    """单只股票日K (含 peTTM/pbMRQ/turn/isST/pctChg), 不复权"""
    import baostock as bs
    rs = bs.query_history_k_data_plus(
        symbol,
        "date,code,open,high,low,close,volume,amount,turn,pctChg,peTTM,pbMRQ,tradestatus,isST",
        start_date=start, end_date=end, frequency="d", adjustflag="3",
    )
    rows = []
    while rs.error_code == '0' and rs.next():
        rows.append(rs.get_row_data())
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows, columns=rs.fields)
    for c in ["open", "high", "low", "close", "volume", "amount", "turn",
              "pctChg", "peTTM", "pbMRQ"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["is_trade"] = df["tradestatus"].map(lambda x: 1 if x == "1" else 0)
    return df


_NUMERIC_COLS = ["open", "high", "low", "close", "volume", "amount", "turn",
                 "pctChg", "peTTM", "pbMRQ"]


def _optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """高效 dtype: date→datetime64, symbol→category, 数值→float32 (内存 ~5倍↓)"""
    if df.empty:
        return df
    df = df.copy()
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    if "symbol" in df.columns:
        df["symbol"] = df["symbol"].astype("category")
    for c in _NUMERIC_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("float32")
    return df


def build_daily_data(universe: list, start: date, end: date,
                     force: bool = False) -> dict:
    """
    构建全市场日K缓存 (baostock 全局锁, 串行).
    返回 {symbol: df}, 每个 df 用高效 dtype (date→datetime64, 数值→float32),
    大幅降低内存驻留 (约 500MB→~150MB), 缓解系统内存压力下的 OOM 被杀.
    """
    cache = REPLAY_DIR / f"daily_{start.isoformat()}_{end.isoformat()}.parquet"
    if cache.exists() and not force:
        logger.info(f"加载缓存: {cache}")
        return {s: _optimize_dtypes(g.drop(columns="symbol").reset_index(drop=True))
                for s, g in pd.read_parquet(cache).groupby("symbol")}

    import baostock as bs
    bs.login()
    all_dfs = []
    t0 = time.time()
    try:
        for i, sym in enumerate(universe):
            try:
                df = _bs_query(sym, start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
                if not df.empty:
                    df["symbol"] = sym
                    all_dfs.append(df)
            except Exception:
                pass
            if (i + 1) % 500 == 0:
                logger.info(f"  已拉取 {i+1}/{len(universe)}, {time.time()-t0:.0f}s")
    finally:
        bs.logout()
    big = pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()
    big = _optimize_dtypes(big)
    big.to_parquet(cache, index=False)
    logger.info(f"数据构建完成: {len(big)} 行, {len(big['symbol'].unique())} 只, 缓存 {cache}")
    return {s: g.drop(columns="symbol").reset_index(drop=True)
            for s, g in big.groupby("symbol")}


async def _ensure_index_data(data: dict, start: date, end: date) -> dict:
    """确保 data 包含 sh.000001 (上证指数, 真实收盘) 用于择时信号.

    用数据路由器 (Tencent 兜底, 免代理) 获取 sh.000001 日K.
    失败时返回原 data (调用方回退等权代理, 不中断回放).
    """
    if "sh.000001" in data:
        return data
    logger.info("sh.000001 不在缓存中, 从数据路由器获取...")
    try:
        from data.router import get_data_router
        from data.providers.base import DataFrequency, DataRequest
        router = get_data_router()
        req = DataRequest("sh.000001", start, end, DataFrequency.DAILY, adjust="qfq")
        res = await asyncio.wait_for(router.get_daily_kline(req), timeout=20)
        df = res.data
        if df is not None and not df.empty and "close" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            df = _optimize_dtypes(df)
            data["sh.000001"] = df.reset_index(drop=True)
            logger.info(f"sh.000001 已加载: {len(df)} 行")
        else:
            logger.warning("sh.000001 获取失败 (空结果), 使用等权代理")
    except Exception as e:
        logger.warning(f"sh.000001 获取失败 ({e}), 使用等权代理")
    return data


# ═══════════════════════════════════════════════════════════════
# PIT 截面重建 — 每交易日 T 的"全市场快照"
# ═══════════════════════════════════════════════════════════════

def reconstruct_cross_section(data: dict, basic: dict, T: str) -> pd.DataFrame:
    """
    从缓存日K重建 T 日全市场截面 (spot_em 风格, PIT):
      price/close, pct_change, volume, amount, turnover, pe_ttm, pb,
      total_mv(快照市值×close_T/close_now, 股本视为常量), pct_60d, vol_ratio, amplitude, isST
    """
    rows = []
    for sym, df in data.items():
        if sym == "sh.000001":
            continue  # 指数不是个股, 跳过截面重建 (用于择时信号, 不在选股池)
        d = pd.to_datetime(df["date"])
        mask = d <= pd.Timestamp(T)
        if not mask.any():
            continue
        row = df[mask].iloc[-1]
        # v3.1 修复: argmax 返回第一个 True (恒0); 需最后一个 ≤T 的索引做历史回看
        idx = int(np.flatnonzero(mask.values)[-1]) if mask.any() else -1
        close = row["close"]
        if close is None or not np.isfinite(close) or close <= 0:
            continue
        if row["is_trade"] != 1:
            continue  # T 日停牌, 不可交易

        # 60日涨跌 (需 ≥61 行历史)
        pct_60d = np.nan
        if idx >= 60:
            base = df["close"].iloc[idx - 60]
            if base and base > 0:
                pct_60d = (close / base - 1) * 100
        # v3.4 126日涨跌 (RPS 相对强弱基础, 研究: 126日窗口是 A 股平衡灵敏度/稳定性的黄金点)
        pct_126d = np.nan
        if idx >= 126:
            base126 = df["close"].iloc[idx - 126]
            if base126 and base126 > 0:
                pct_126d = (close / base126 - 1) * 100
        # v3.4 拥挤度: 当日换手在自身60日内的分位 (0-1, 高=极端活跃/过热)
        turn_pct_60d = np.nan
        if idx >= 60:
            _turn_hist = pd.to_numeric(df["turn"].iloc[idx - 60:idx], errors="coerce").dropna()
            _cur_turn = row["turn"]
            if np.isfinite(_cur_turn) and _cur_turn > 0 and len(_turn_hist) >= 20:
                turn_pct_60d = float((_turn_hist < _cur_turn).mean())
        # 量比 = 当日量 / 前5日均量
        vol_ratio = np.nan
        if idx >= 5:
            avg5 = df["volume"].iloc[idx - 5:idx].mean()
            if avg5 and avg5 > 0:
                vol_ratio = row["volume"] / avg5
        # 振幅
        pre_close = df["close"].iloc[idx - 1] if idx >= 1 else close
        amplitude = ((row["high"] - row["low"]) / pre_close * 100) if pre_close and pre_close > 0 else 0

        bi = basic.get(sym, {})
        close_now = bi.get("close_now", 0)
        total_mv = np.nan
        if close_now and close_now > 0:
            total_mv = bi.get("total_mv_now", 0) * close / close_now  # PIT 市值近似

        # v3.2 因子注入 (供 --variant v32 的 LLM prompt 使用):
        #   ep/bp 估值, pe_pct_20d/pb_pct_20d 相对自身20日历史, sharpe_20 风险调整动量, reversal_1d
        pe = row["peTTM"] if pd.notna(row["peTTM"]) else np.nan
        pb = row["pbMRQ"] if pd.notna(row["pbMRQ"]) else np.nan
        ep = float(1.0 / np.clip(pe, 0.1, 200)) if pe > 0 else 0.0
        bp = float(1.0 / np.clip(pb, 0.05, 50)) if pb > 0 else 0.0
        pe_pct_20d, pb_pct_20d = 0.5, 0.5
        if idx >= 10:
            pe_hist = pd.to_numeric(df["peTTM"].iloc[max(0, idx - 20):idx], errors="coerce")
            pe_hist = pe_hist[(pe_hist > 0) & pe_hist.notna()]
            if len(pe_hist) >= 5 and pe > 0:
                pe_pct_20d = float((pe_hist < pe).mean())
            pb_hist = pd.to_numeric(df["pbMRQ"].iloc[max(0, idx - 20):idx], errors="coerce")
            pb_hist = pb_hist[(pb_hist > 0) & pb_hist.notna()]
            if len(pb_hist) >= 5 and pb > 0:
                pb_pct_20d = float((pb_hist < pb).mean())
        sharpe_20 = np.nan
        if idx >= 20:
            rets = df["close"].iloc[max(0, idx - 20):idx + 1].pct_change().dropna()
            r5 = (close / df["close"].iloc[idx - 5] - 1) if (idx >= 5 and df["close"].iloc[idx - 5] > 0) else 0.0
            std = rets.tail(20).std()
            sharpe_20 = float(r5 / (std * np.sqrt(5))) if std and std > 0 else np.nan
        reversal_1d = float(-(row["pctChg"] / 100.0)) if pd.notna(row["pctChg"]) else 0.0

        rows.append({
            "code": sym.split(".")[-1], "symbol": sym, "name": bi.get("name", sym),
            "price": close, "pct_change": row["pctChg"] if pd.notna(row["pctChg"]) else 0,
            "volume": row["volume"], "amount": row["amount"],
            "turnover": row["turn"] if pd.notna(row["turn"]) else 0,
            "pe_ttm": row["peTTM"] if pd.notna(row["peTTM"]) else np.nan,
            "pb": row["pbMRQ"] if pd.notna(row["pbMRQ"]) else np.nan,
            "total_mv": total_mv,
            "pct_60d": pct_60d, "pct_126d": pct_126d,
            "turn_pct_60d": turn_pct_60d,
            "vol_ratio": vol_ratio, "amplitude": amplitude,
            "isST": row["isST"], "is_trade": 1,
            "ep": ep, "bp": bp, "pe_pct_20d": pe_pct_20d, "pb_pct_20d": pb_pct_20d,
            "sharpe_20": sharpe_20, "reversal_1d": reversal_1d,
        })
    df = pd.DataFrame(rows)
    if len(df) and "pct_126d" in df.columns:
        # v3.4 RPS: 126日涨幅的截面百分位 (0-100). 抱团/普涨牛里捕捉强势龙头,
        # 由 PreScreener 的 regime 权重门控 (strong_bull 动量权重 0.35, 强熊 0.00).
        df["rps_126"] = pd.to_numeric(df["pct_126d"], errors="coerce").rank(pct=True) * 100
    return df


# ═══════════════════════════════════════════════════════════════
# 回放主循环
# ═══════════════════════════════════════════════════════════════

class ReplayPortfolio:
    """回放持仓 (轻量): {symbol: {qty, entry_price, entry_date, stop, take}}
    v3.3 混合结构: index_units 是"市场指数书" (risk_on 时部分资金持市场代理)."""
    def __init__(self, capital=100000.0):
        self.capital = capital
        self.cash = capital
        self.positions = {}
        self.equity_curve = []       # [{date, total, cash, position_value}]
        self.trades = []
        self.max_positions = 10
        self.index_units = 0.0       # 分配给市场指数书的资金
        self.index_base = 0.0        # 进入指数书时的市场代理水平
        self.index_entry_date = ""

    def to_dict(self) -> dict:
        return {"capital": self.capital, "cash": self.cash,
                "positions": self.positions, "trades": self.trades,
                "equity_curve": self.equity_curve,
                "index_units": self.index_units, "index_base": self.index_base,
                "index_entry_date": self.index_entry_date}

    @classmethod
    def from_dict(cls, d: dict) -> "ReplayPortfolio":
        pf = cls(d.get("capital", 100000.0))
        pf.cash = float(d.get("cash", pf.capital))
        pf.positions = d.get("positions", {}) or {}
        # v3.1.1 修复: 旧 checkpoint 用 default=str 把数值字符串化, 载入后
        # px1<=stop 等比较会 float32<=str 崩溃. 强转持仓数值为 float.
        for _sym, _p in pf.positions.items():
            for _k in ("qty", "entry_price", "stop", "take"):
                if _k in _p and not isinstance(_p[_k], (int, float)):
                    try:
                        _p[_k] = float(_p[_k])
                    except (TypeError, ValueError):
                        _p[_k] = 0.0
        pf.trades = d.get("trades", []) or []
        pf.equity_curve = d.get("equity_curve", []) or []
        pf.index_units = float(d.get("index_units", 0.0) or 0.0)
        pf.index_base = float(d.get("index_base", 0.0) or 0.0)
        pf.index_entry_date = d.get("index_entry_date", "") or ""
        return pf

    def index_value(self, index_level: float) -> float:
        """指数书当前市值 (按市场代理涨跌)."""
        if self.index_units <= 0 or self.index_base <= 0 or index_level <= 0:
            return 0.0
        return self.index_units * (index_level / self.index_base)

    def buy_index(self, amount: float, index_level: float, date_str: str) -> bool:
        """risk_on 时把部分资金投入市场指数书 (持市场代理)."""
        if amount <= 0 or amount > self.cash or index_level <= 0:
            return False
        self.cash -= amount
        if self.index_units <= 0:
            self.index_base = index_level
            self.index_entry_date = date_str
        self.index_units += amount
        return True

    def sell_index(self, index_level: float, date_str: str) -> float:
        """risk_off 时清空指数书, 落袋. 返回实现金额."""
        amount = self.index_value(index_level)
        self.cash += amount
        self.trades.append({"date": date_str, "symbol": "MARKET_INDEX",
                            "side": "sell", "price": round(index_level, 3),
                            "qty": 1, "pnl_pct": round((amount / max(self.index_units, 1e-9) - 1) * 100, 2),
                            "reason": "择时risk_off清指数"})
        self.index_units = 0.0
        self.index_base = 0.0
        return amount

    def total_value(self, price_map, index_level=None):
        pos_val = sum(p["qty"] * price_map.get(sym, p["entry_price"]) for sym, p in self.positions.items())
        idx_val = self.index_value(index_level) if index_level else 0.0
        return self.cash + pos_val + idx_val

    def buy(self, symbol, name, price, qty, date_str, stop=None, take=None):
        if symbol in self.positions or qty <= 0 or price <= 0:
            return False
        cost = price * qty
        if cost > self.cash:
            return False
        self.cash -= cost
        self.positions[symbol] = {"qty": qty, "entry_price": price, "entry_date": date_str,
                                  "name": name, "stop": stop, "take": take}
        self.trades.append({"date": date_str, "symbol": symbol, "side": "buy",
                            "price": price, "qty": qty})
        return True

    def sell(self, symbol, price, date_str, reason=""):
        pos = self.positions.pop(symbol, None)
        if pos is None:
            return False
        self.cash += pos["qty"] * price
        self.trades.append({"date": date_str, "symbol": symbol, "side": "sell",
                            "price": price, "qty": pos["qty"],
                            "pnl_pct": (price / pos["entry_price"] - 1) * 100, "reason": reason})
        return True


def _next_trading_day(data, T: str) -> str:
    """T 之后第一个交易日 (用于 T+1 开盘成交)"""
    dates = sorted({str(pd.Timestamp(d).date()) for df in data.values()
                    for d in pd.to_datetime(df["date"])})
    t_ts = pd.Timestamp(T)
    for d in dates:
        if pd.Timestamp(d) > t_ts:
            return d
    return None


# v3.5: 市场风险决策槽 — LLM 基于当日市场快照, 显式承诺一个量化风险目标与持仓上限.
# 这是把"知识"变成"动作"的强制机制: LLM 必须输出 risk_target 和 max_positions,
# 且该数字会被自由模式实际执行 (而非泛泛的"注意风险"). 失败时保守回退满配 (不改变行为).
async def _market_risk_decision(market_ctx_txt: str) -> dict:
    from openai import AsyncOpenAI
    import os
    try:
        client = AsyncOpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
            timeout=60.0,
        )
        sys = (
            "你是A股市场风险官。基于给定市场快照, 输出当日的风险目标(risk_target, 0.0-1.0, "
            "1.0=满仓进攻, 0.0=空仓避险)和建议持仓上限(max_positions, 1-20只). "
            "硬规则: 若指数跌破MA20 且 拥挤度hot 且 估值处于60日高位 → 必须 risk_target≤0.3 且 max_positions≤5; "
            "若强牛/量价健康/拥挤度cool → risk_target≥0.8 且 max_positions≥10. "
            "只返回JSON:{'risk_target':0.0,'max_positions':0,'reason':文字}"
        )
        prompt = (f"【今日市场快照】\n{market_ctx_txt}\n\n"
                  f"依据快照中的拥挤度/估值分位/指数位置, 输出今日风险目标与持仓上限 (JSON)。")
        resp = await client.chat.completions.create(
            model=os.getenv("DEEPSEEK_FLASH_MODEL", "deepseek-v4-flash"),
            messages=[{"role": "system", "content": sys}, {"role": "user", "content": prompt}],
            temperature=0.0, max_tokens=300,
            extra_body={"thinking": {"type": "disabled"}},
        )
        content = (resp.choices[0].message.content or "").strip()
        if content.startswith("```"):
            content = content.split("\n", 1)[1].rsplit("```", 1)[0]
        d = json.loads(content)
        rt = float(np.clip(d.get("risk_target", 1.0), 0.0, 1.0))
        mp = int(np.clip(d.get("max_positions", 999), 1, 30))
        logger.info(f"[市场风险决策] risk_target={rt:.2f} max_positions={mp} reason={str(d.get('reason',''))[:80]}")
        return {"risk_target": rt, "max_positions": mp, "reason": d.get("reason", "")}
    except Exception as e:
        logger.warning(f"市场风险决策失败(保守回退满配): {e}")
        return {"risk_target": 1.0, "max_positions": 999, "reason": f"fallback:{e}"}


def _compute_day_stat(df_cs: pd.DataFrame) -> dict:
    """计算单日截面的市场统计 (用于诊断官历史队列)."""
    n = len(df_cs)
    if n == 0:
        return {"n": 0, "up_ratio": 0.5, "limit_up": 0, "limit_down": 0,
                "med_pe": 0.0, "med_pb": 0.0, "total_amt": 0.0}
    pct_col = "pctChg" if "pctChg" in df_cs.columns else ("pct_change" if "pct_change" in df_cs.columns else None)
    pe_col = "peTTM" if "peTTM" in df_cs.columns else ("pe_ttm" if "pe_ttm" in df_cs.columns else None)
    pb_col = "pbMRQ" if "pbMRQ" in df_cs.columns else ("pb" if "pb" in df_cs.columns else None)

    if pct_col and pct_col in df_cs.columns:
        pct = pd.to_numeric(df_cs[pct_col], errors="coerce")
        up_r = float((pct > 0).mean())
        lu = int((pct >= 9.5).sum())
        ld = int((pct <= -9.5).sum())
    else:
        up_r, lu, ld = 0.5, 0, 0
    if pe_col and pe_col in df_cs.columns:
        pe = pd.to_numeric(df_cs[pe_col], errors="coerce").dropna()
        mpe = float(pe.median()) if len(pe) else 0.0
    else:
        mpe = 0.0
    if pb_col and pb_col in df_cs.columns:
        pb = pd.to_numeric(df_cs[pb_col], errors="coerce").dropna()
        mpb = float(pb.median()) if len(pb) else 0.0
    else:
        mpb = 0.0
    if "amount" in df_cs.columns:
        amt = pd.to_numeric(df_cs["amount"], errors="coerce").dropna()
        tamt = float(amt.sum()) / 1e8
    else:
        tamt = 0.0
    return {"n": n, "up_ratio": up_r, "limit_up": lu, "limit_down": ld,
            "med_pe": mpe, "med_pb": mpb, "total_amt": tamt}


async def _market_diagnostic(df_cs: pd.DataFrame, regime: str, crowd: dict,
                             _macro_txt: str = None,
                             history_5d: list = None,
                             memory = None,
                             evolution = None,
                             current_date: str = None,
                             prev_risk_level: int = None) -> dict:
    """v4.0 自我进化市场诊断官.

    进化历程:
    v3.5: 简单诊断官 (规则+LLM调仓)
    v3.6: 趋势感知 (5天历史队列)
    v3.7: 多人格自适应 (3个人格, 自己选框架)
    v3.8: 5位大师版 (名片级, 选角色比硬编码规则强)
    v3.9: 深度蒸馏版 (失败, 规则太多反而束缚手脚)
    v4.0: 自我进化版 (从历史决策中学习, 经验记忆库 + 周期进化总结)

    返回: {risk_level, position_multiplier, max_positions_adj, key_risks, diagnosis, market_phase, dominant_master, secondary_master}
    """
    from openai import AsyncOpenAI
    import os
    default_diag = {
        "risk_level": 3, "position_multiplier": 0.9, "max_positions_adj": 0,
        "key_risks": [], "diagnosis": "LLM 诊断失败, 使用默认中性值",
    }
    try:
        n = len(df_cs)
        pct_col = "pctChg" if "pctChg" in df_cs.columns else ("pct_change" if "pct_change" in df_cs.columns else None)
        pe_col = "peTTM" if "peTTM" in df_cs.columns else ("pe_ttm" if "pe_ttm" in df_cs.columns else None)
        pb_col = "pbMRQ" if "pbMRQ" in df_cs.columns else ("pb" if "pb" in df_cs.columns else None)
        amt_col = "amount"

        def _day_stats(df):
            m = len(df)
            if m == 0:
                return None
            if pct_col and pct_col in df.columns:
                pct = pd.to_numeric(df[pct_col], errors="coerce")
                up_r = float((pct > 0).mean())
                lu = int((pct >= 9.5).sum())
                ld = int((pct <= -9.5).sum())
            else:
                up_r, lu, ld = 0.5, 0, 0
            if pe_col and pe_col in df.columns:
                pe = pd.to_numeric(df[pe_col], errors="coerce").dropna()
                mpe = float(pe.median()) if len(pe) else 0.0
            else:
                mpe = 0.0
            if pb_col and pb_col in df.columns:
                pb = pd.to_numeric(df[pb_col], errors="coerce").dropna()
                mpb = float(pb.median()) if len(pb) else 0.0
            else:
                mpb = 0.0
            if amt_col and amt_col in df.columns:
                amt = pd.to_numeric(df[amt_col], errors="coerce").dropna()
                tamt = float(amt.sum()) / 1e8
            else:
                tamt = 0.0
            return {"n": m, "up_ratio": up_r, "limit_up": lu, "limit_down": ld,
                    "med_pe": mpe, "med_pb": mpb, "total_amt": tamt}

        today = _day_stats(df_cs)

        # 构建近5天序列 (如果提供了 history)
        hist_lines = []
        if history_5d and len(history_5d) >= 2:
            # history_5d 是前几天的 _day_stats 结果, [0]=最早, [-1]=昨天
            for i, h in enumerate(history_5d):
                day_label = f"T-{len(history_5d)-i}"
                hist_lines.append(
                    f"  {day_label}: 上涨{h['up_ratio']:.1%} 涨停{h['limit_up']}家 "
                    f"跌停{h['limit_down']}家 中位PE{h['med_pe']:.1f} 成交额{h['total_amt']:.0f}亿"
                )
            # 今天
            hist_lines.append(
                f"  T  : 上涨{today['up_ratio']:.1%} 涨停{today['limit_up']}家 "
                f"跌停{today['limit_down']}家 中位PE{today['med_pe']:.1f} 成交额{today['total_amt']:.0f}亿"
            )
            # 计算变化趋势
            if len(history_5d) >= 3:
                up_trend = today["up_ratio"] - history_5d[-3]["up_ratio"]
                amt_trend = (today["total_amt"] / history_5d[-3]["total_amt"] - 1) if history_5d[-3]["total_amt"] > 0 else 0
                trend_note = (
                    f"\n近3日变化: 上涨占比{'上升' if up_trend > 0 else '下降'}{abs(up_trend):.1%}, "
                    f"成交额{'放量' if amt_trend > 0 else '缩量'}{abs(amt_trend):.1%}"
                )
            else:
                trend_note = ""
        else:
            hist_lines.append(
                f"  今日: 上涨{today['up_ratio']:.1%} 涨停{today['limit_up']}家 "
                f"跌停{today['limit_down']}家 中位PE{today['med_pe']:.1f} 成交额{today['total_amt']:.0f}亿"
            )
            trend_note = ""

        # 60日强度
        p60_col = "pct_60d"
        if p60_col in df_cs.columns:
            p60 = pd.to_numeric(df_cs[p60_col], errors="coerce").dropna()
            p60_pos_ratio = float((p60 > 0).mean()) if len(p60) else 0.5
            p60_above20 = float((p60 > 20).mean()) if len(p60) else 0.0
        else:
            p60_pos_ratio = 0.5
            p60_above20 = 0.0

        snapshot_txt = (
            f"【市场广度 · 近{len(hist_lines)}日序列】\n"
            + "\n".join(hist_lines)
            + trend_note
            + f"\n\n【中期位置】\n"
            f"60日上涨占比: {p60_pos_ratio:.1%}  60日涨幅>20%占比: {p60_above20:.1%}\n"
            f"拥挤度: {crowd.get('signal','unknown')} (score {crowd.get('score',0):.2f}, "
            f"极端活跃 {crowd.get('hot_ratio',0):.1%})\n"
            f"市场状态: {regime}"
        )

        sys_prompt = """你是A股市场的【风险诊断官】。

你体内住着五位投资大师的灵魂 — 他们来自不同时代、不同流派，但都在市场中证明了自己。
每天你要先判断当前市场处于什么阶段，然后请出最适合的那位大师来主导今天的仓位决策。

## 你的五位大师顾问

========================================================================
### 1. 利弗莫尔 (趋势投机之王) — 深度蒸馏版
========================================================================

【核心理念】
价格总是沿着阻力最小的方向运动；市场只有一个方向，不是多头也不是空头，而是正确的方向；
华尔街没有新鲜事，因为人性从未改变 — 投机像山岳一样古老。

【入场规则 — 伯利恒钢铁法】
- 不在底部抄底，不在顶部逃顶，只吃"中间最肥的一段"
- 关键点突破: 价格突破前期高点（阻力位）才入场，不提前埋伏
- 量能确认: 突破当日成交量至少放大30%以上，无量突破是假突破
- 市场配合: 70%以上的股票跟随上涨（广度确认），独涨不参与
- 不做反弹: 下跌趋势中的反弹一律不碰，只做明确反转后的新趋势

【仓位管理 — 金字塔加仓法】
- 试探仓: 首次建仓只用1/4的计划仓位，验证判断
- 盈利加仓: 只有当试探仓盈利（证明你是对的）才加仓，亏损时绝对不加仓摊平
- 加码递减: 每次加仓量比上一次少（100→60→30→15），防止顶部重仓
- 满仓时机: 趋势明确、连续盈利3次加仓后，才允许满仓
- 离场规则: 趋势线跌破、关键支撑失守、或出现"反转信号日"（高开低走巨量阴线），立刻全清

【止损铁律】
- 单笔亏损不超过总资金的10%（这是利弗莫尔三次破产换来的教训）
- 股价跌回关键点下方 — 说明突破失败，立即止损
- 不要给亏损找理由，不要等"反弹再卖"，市场告诉你错了就立刻认错

【适用/不适用行情】
- ✅ 擅长: 明确的单边上升趋势、主升浪、突破行情
- ❌ 不擅长: 震荡市（会被反复打脸）、熊市反弹（假突破太多）
- ⚠️ 警惕: V型反转（来不及建仓就走完了）

【经典失败教训（一定要记住！）】
1. 1907年抄底棉花期货，逆势加仓，一天亏了几百万美元 — 教训：亏损时绝对不要加仓
2. 1929年大崩盘前虽然看空，但中间忍不住反手做多，又亏回去 — 教训：不要轻易改变既定判断
3. 听信内幕消息买了伯利恒钢铁以外的股票，亏了 — 教训：只做你自己研究透的标的
4. 过度交易 — 他说"赚大钱靠的是坐着等，不是靠频繁操作"

【A股适配调整】
- A股涨跌停制度 = 天然的"关键点"（涨停板就是最强阻力突破）
- A股T+1 = 不能日内纠错，所以买点确认要更严格，宁可错过不可做错
- A股散户多 = 假突破更多，量能确认更重要
- 利弗莫尔的10%止损在A股可以放宽到12-15%（因为波动更大）
- 涨停潮次日容易分化，追高要格外谨慎

========================================================================
### 2. 巴菲特 (价值投资之父) — 名片级
========================================================================
- 核心理念: 价格是你付出的，价值是你得到的；安全边际是一切的基石
- 仓位哲学: 价格低于内在价值时越跌越买，价格远高于价值时越涨越卖；别人恐惧我贪婪，别人贪婪我恐惧
- 识别信号: 整体估值水平（PE/PB分位）、市场情绪极端度
- 经典: "在别人恐惧时贪婪，在别人贪婪时恐惧"
- 最擅长: 估值极端时刻（大底/大顶）

========================================================================
### 3. 索罗斯 (反身性大师) — 名片级
========================================================================
- 核心理念: 市场总是错的，趋势不是直线而是加速—衰竭—反转的S曲线；认知和现实相互作用形成泡沫和崩溃
- 仓位哲学: 先于拐点识别泡沫形成并顺势做多（因为你知道这是泡沫但可以参与），在狂欢顶点信号出现时果断反手；最危险的时刻是所有人都相信"这次不一样"的时候
- 识别信号: 上涨家数持续收窄但指数还在涨（背离=泡沫末期）、涨停潮+散户涌入+媒体狂热
- 经典: "世界经济史是一部基于假象和谎言的连续剧"
- 最擅长: 泡沫形成期和拐点判断

========================================================================
### 4. 达利欧 (全天候与经济机器) — 名片级
========================================================================
- 核心理念: 经济是一台简单的机器，由生产率、短期债务周期、长期债务周期驱动；现金是垃圾，但流动性是生命线
- 仓位哲学: 分散是免费的午餐，永远不要all in；持有流动性好的资产，确保在最困难的时候也能活下来；风险和回报是对称的
- 识别信号: 流动性变化（成交额缩量=风险）、市场结构健康度、信用环境
- 经典: "痛苦+反思=进步"
- 最擅长: 震荡市、流动性风险、全天候防守

========================================================================
### 5. 缠中说禅 (A股本土技术分析宗师) — 深度蒸馏版
========================================================================

【核心理念】
走势终完美 — 任何级别的上涨/下跌走势类型终会完成；
没有预测，只有应对 — 市场走出来什么就是什么，不猜顶不猜底；
买点买、卖点卖 — 操作只有两个动作，其余都是等待。

【核心概念（必须理解）】
- 走势级别: 日线是大级别，60分钟是中级别，5分钟是小级别 — 大级别定方向，小级别找买点
- 中枢: 连续三段走势重叠的区域，就是中枢；中枢是"引力场"，价格总有回到中枢的冲动
- 走势类型: 趋势（有两个以上不重叠中枢）vs 盘整（只有一个中枢）
- 背驰: 趋势力度衰竭 = 后一段走势的力度（面积/高度/成交量）比前一段小，就是背驰
  - 顶背驰 = 上涨趋势力度衰竭，要卖
  - 底背驰 = 下跌趋势力度衰竭，要买
- 三类买卖点:
  - 第一类买卖点（1买/1卖）: 背驰点，最安全但需要提前判断
  - 第二类买卖点（2买/2卖）: 1买后回调不创新低，确认反转，最稳妥
  - 第三类买卖点（3买/3卖）: 突破中枢后回踩不回到中枢内，最强但追高

【入场规则】
- 大级别向上（日线在60均线上方），只做多，不做空
- 小级别出现底背驰（1买）或回踩不破前低（2买），进场
- 第三类买点（突破中枢回踩）是最强势的介入点，但只在大级别趋势中做
- 永远不要在"走势还在形成中"时预判，等背驰确认了再动
- 级别联立: 日线看方向，30分钟找买卖点，5分钟精确定位 — 三重确认后出手

【仓位管理 — 按级别分批】
- 1买（底背驰）: 轻仓试探（30%），因为背驰可能判断失误
- 2买（确认不创新低）: 加仓到60%，这是最安全的买点
- 3买（突破回踩）: 加仓到满仓（100%），趋势确认最强
- 卖点反过来: 1卖减30%，2卖减到30%，3卖清仓
- 如果级别判断错了，按小级别止损，不要扛到下一个大级别

【止损与持有的边界】
- 买入后，如果走势跌破"前低"（2买的低点），说明判断错了，立刻止损
- 只要走势还在"中枢上方"运行，就持有，不要被小波动洗出去
- 出现顶背驰信号，先减仓一半，剩下的设移动止损
- "走势完美"的信号: 最后一段走势力度明显减弱 + 成交量萎缩 + 指标背离 = 该走了

【适用/不适用行情】
- ✅ 擅长: 所有行情（缠论是完全分类系统，理论上通吃）
- ✅ 特别擅长: 趋势背驰后的反转、震荡市的高抛低吸、个股走势分析
- ❌ 不擅长: 极度情绪化的行情（连续一字板/连续跌停，没有结构）
- ⚠️ 难点: 级别和中枢的划分主观性强，不同人看同一张图结论可能不同

【经典陷阱（缠师反复提醒的！）】
1. "背了又背" — 背驰后还有背驰，尤其是在强趋势中，不要因为一次背驰就猜顶/底
   （正确做法: 第一次背驰减仓，第二次背驰清仓，不一次all in/out）
2. "小级别转大级别" — 本来只是5分钟的回调，结果演变成日线级别的下跌
   （正确做法: 3买失败就立刻走，不要从小亏扛成大亏）
3. "中枢扩展" — 以为是3买突破，结果又拉回中枢扩展了
   （正确做法: 突破一定要等回踩确认，没回踩之前都不算数）
4. "贪便宜" — 总想在最低点买、最高点卖，结果错过了真正的行情
   （正确做法: 买在2买，卖在1卖，吃中间最确定的一段）

【A股适配（缠师就是A股的，完全贴合！）】
- 缠论本身就是用A股数据写的，100%适配
- A股T+1: 买点尽量在下午尾盘（2:30以后），防止买入后当天不能卖
- A股涨跌停: 一字板不算有效走势（没有成交量），开板后才开始算结构
- A股散户多: 小级别波动噪音大，尽量看30分钟以上级别
- 板块轮动快: 用缠论看板块指数比看个股更准，板块走好了个股才有机会

========================================================================
## 决策流程
========================================================================

1. **判断市场阶段** — 先看5天序列，判断现在是：明确上升趋势 / 明确下跌趋势 / 震荡 / 泡沫末期 / 恐慌底部 / 拐点附近
2. **选主导大师** — 根据阶段特征，从五位中选一位最适合的
3. **对照大师规则检查清单** — 用这位大师的具体规则一条条过，不要只凭感觉
4. **交叉验证** — 找一位观点不同的大师做对手盘检查，如果两位大师矛盾，偏向保守
5. **给出结论** — 风险等级、仓位系数、持仓调整

【利弗莫尔模式检查清单】
  □ 趋势是否明确？（连续3天以上同向，且有广度配合）
  □ 是否有关键点位突破？（前期高点/平台）
  □ 量能是否确认？（成交额放大）
  □ 是突破行情还是反弹行情？（反弹一律pass）
  □ 止损位在哪里？（跌破关键点就走）

【缠中说禅模式检查清单】
  □ 大级别方向是什么？（60日均线之上/之下）
  □ 当下是趋势还是盘整？（有几个中枢）
  □ 有没有背驰信号？（指数新高但上涨家数没新高=顶背驰）
  □ 是第几类买卖点？（1买轻仓/2买重仓/3买追涨）
  □ 走势完美了吗？（最后一段力度是否衰竭）

## 输出格式 (严格JSON，不要多余文字)
{
  "market_phase": "trend_up / trend_down / range / bubble_late / panic_bottom / turning 六选一",
  "dominant_master": "利弗莫尔 / 巴菲特 / 索罗斯 / 达利欧 / 缠中说禅 五选一",
  "secondary_master": "次选大师 (用来交叉验证)",
  "risk_level": 1~5的整数,
  "position_multiplier": 0.3~1.6之间的浮点数,
  "max_positions_adj": -8~+8之间的整数,
  "key_risks": ["风险1", "风险2"],
  "diagnosis": "200字以内, 先讲市场阶段+为什么选这位大师, 再讲结论"
}

## 仓位系数参考 (基准=1.0)
- 1级(进攻): 1.3~1.6  — 趋势明确+量价配合+广度确认
- 2级(偏多): 1.1~1.3  — 趋势尚可但有隐忧，或刚启动信号不全
- 3级(中性): 0.8~1.1  — 震荡市、信号矛盾、看不清方向
- 4级(谨慎): 0.5~0.8  — 有风险信号但还没确认，或背驰初现
- 5级(防御): 0.3~0.5  — 明确的下跌趋势、顶背驰确认、极端拥挤

## 重要
- 选对大师比算准系数更重要
- 不要因为今天涨了就选利弗莫尔，要判断的是"阶段特征"不是"单日涨跌"
- 当多位大师指向同一方向时，信心更高；当他们矛盾时，偏向保守（达利欧模式）
- 深度版的利弗莫尔和缠中说禅有具体的检查清单，一定要对照着用，不要回到"凭感觉"
- **要从历史经验中学习** — 下面的"历史经验"和"核心原则"是你从过去的错误和成功中总结出来的，务必认真对待

## ⚠️ 大师选择稳定性原则 (v4.2)
不要每天换大师。大师的选择应该反映"市场处于什么阶段"，而不是"昨天我用对了/用错了"。

- 如果市场阶段没有本质变化，主导大师也不应该变
- 连续两天选不同的大师，说明你在"赌"而不是在"判断"
- 大师切换只能因为：市场阶段变了（trend→range、bubble→crash等）
- 大师切换不能因为：昨天选错了/昨天复盘结果不好/想试试别的

稳定的大师选择 + 精细的仓位调节 > 每天换大师碰运气

"""

        # ── v4.0 自我进化: 动态注入经验记忆和核心原则 ──
        evolution_sections = []

        # 1. 核心原则（来自周期进化总结）
        if evolution is not None:
            principles_text = evolution.get_latest_principles_text()
            if principles_text:
                evolution_sections.append(principles_text)

            master_tips_text = evolution.get_master_tips_text()
            if master_tips_text:
                evolution_sections.append(master_tips_text)

        # 2. 相关历史经验（来自动态记忆库）
        #    注意: 这里我们还没判断市场阶段，所以给所有场景的代表性经验
        if memory is not None:
            mem_text = memory.format_for_prompt(
                current_date=current_date,
                top_k=6,  # 最多6条，避免提示词太长
            )
            if mem_text:
                evolution_sections.append(mem_text)
            # v4.2: 元认知校准 — 告诉 LLM 它自己近期的表现
            if current_date:
                meta_text = memory.get_metacognition_summary(current_date, lookback_days=15)
                if meta_text:
                    evolution_sections.append(meta_text)

        if evolution_sections:
            sys_prompt += "\n".join(evolution_sections) + "\n"
            sys_prompt += "请结合以上历史经验、核心原则以及对你自己近期表现的认知，做出今天的判断。\n\n"
        user_msg = f"市场诊断数据:\n{snapshot_txt}"
        if _macro_txt:
            user_msg += f"\n\n【宏观背景】\n{_macro_txt}"
        user_msg += "\n\n先判断市场阶段，选最适合的大师主导，再给出诊断。"

        client = AsyncOpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
            timeout=45.0,
        )
        resp = await client.chat.completions.create(
            model=os.getenv("DEEPSEEK_FLASH_MODEL", "deepseek-v4-flash"),
            messages=[{"role": "system", "content": sys_prompt},
                      {"role": "user", "content": user_msg}],
            temperature=0.4, max_tokens=800,
            extra_body={"thinking": {"type": "disabled"}},
        )
        content = (resp.choices[0].message.content or "").strip()
        if content.startswith("```"):
            content = content.strip("`")
            if content.lower().startswith("json"):
                content = content[4:].strip()
        diag = None
        try:
            diag = json.loads(content)
        except json.JSONDecodeError:
            s, e = content.find("{"), content.rfind("}")
            if s >= 0 and e > s:
                try:
                    diag = json.loads(content[s:e+1])
                except json.JSONDecodeError:
                    pass
        if diag is None:
            return default_diag

        risk_level = max(1, min(5, int(diag.get("risk_level", 3))))
        pos_mult = max(0.2, min(1.7, float(diag.get("position_multiplier", 0.9))))
        max_adj = max(-10, min(10, int(diag.get("max_positions_adj", 0))))

        # v4.2: 风险等级稳定性约束 — 单日变化不超过±1级
        if prev_risk_level is not None:
            orig_risk = risk_level
            max_change = 1
            if risk_level > prev_risk_level + max_change:
                risk_level = prev_risk_level + max_change
            elif risk_level < prev_risk_level - max_change:
                risk_level = prev_risk_level - max_change
            if risk_level != orig_risk:
                target_ratio = (risk_level - prev_risk_level) / (orig_risk - prev_risk_level) if orig_risk != prev_risk_level else 1.0
                pos_mult = 1.0 + (pos_mult - 1.0) * max(0.0, min(1.0, target_ratio))
                max_adj = int(max_adj * max(0.0, min(1.0, target_ratio)))

        key_risks = diag.get("key_risks", [])
        if not isinstance(key_risks, list):
            key_risks = []
        diagnosis = str(diag.get("diagnosis", ""))[:200]

        result = {
            "risk_level": risk_level, "position_multiplier": pos_mult,
            "max_positions_adj": max_adj, "key_risks": key_risks,
            "diagnosis": diagnosis,
            "market_phase": diag.get("market_phase", "unknown"),
            "dominant_master": diag.get("dominant_master", "unknown"),
            "secondary_master": diag.get("secondary_master", ""),
        }
        logger.info(
            f"[市场诊断] 风险={risk_level}/5  仓位×{pos_mult:.2f}  "
            f"持仓{max_adj:+d}  风险{len(key_risks)}个"
        )
        return result
    except Exception as e:
        logger.warning(f"市场诊断官失败 ({e}), 使用默认保守值")
        return default_diag


async def run_replay(days: int = 40, universe=None, top_n: int = 300, final_n: int = 100,
                     capital: float = 100000.0, force_data: bool = False,
                     thinking: bool = False, max_days: int = 0,
                     variant: str = "baseline",
                     end_date: str = "2026-07-31",
                     data_file: str = None,
                     tag: str = None,
                     timing_overlay: bool = False,
                     hybrid_pct: float = 0.0,
                     offensive: bool = False,
                     auto_structure: bool = False,
                     crowding_overlay: bool = False,
                     ma_window: int = 20,
                     free_mode: bool = False,
                     diagnostic_mode: bool = False,
                     diagnostic_top_n: int = 20,
                     diagnostic_dynamic: bool = False,
                     flash_diag_mode: bool = False,
                     evolution_mode: bool = False) -> dict:
    """
    max_days: 本次最多处理的新天数 (0=不限). 用于分小段跑, 每段干净退出
              (checkpoint 落盘), 避免后台任务被杀窗口浪费.
    variant: "baseline" (原版) 或 "v32" (v3.2 因子注入: 相对估值分位/风险调整
             动量/反转 + regime 门控市场上下文).
    end_date: 回放窗口终点 (YYYY-MM-DD). 用于多 regime 分段跑.
    data_file: 显式指定缓存 parquet (长窗口全市场). 提供时跳过 build_daily_data
              的按文件名缓存查找, 直接从该文件加载 (warmup 由文件内历史提供).
    diagnostic_mode: v3.5 诊断模式 — LLM 不直接选股, 只做市场风险诊断并调节仓位.
                     选股完全交给 PreScreener 规则系统, LLM 是"监督者"不是"交易员".
    """
    """执行历史 PIT 回放"""
    end = date.fromisoformat(end_date)  # 最近完整交易日 (08-02 休市)
    # 回放只需 ~60交易日 warmup (算 60日涨跌) + 前瞻缓冲, 无需 250日
    start = end - timedelta(days=int(days * 1.5) + 100)

    # ── 0. 快照基础 + 全市场数据 ──
    basic, snapshot_universe = load_snapshot_basic()
    if universe is None or universe == "full":
        universe = snapshot_universe
    basic = {sym: basic.get(sym, {}) for sym in universe}

    logger.info(f"股票池: {len(universe)} 只, 数据窗口 {start}~{end}")
    if data_file:
        logger.info(f"从显式缓存加载: {data_file}")
        _df = pd.read_parquet(data_file)
        _df["date"] = pd.to_datetime(_df["date"])
        _df = _df[_df["date"] <= pd.Timestamp(end)]
        data = {s: _optimize_dtypes(g.drop(columns="symbol").reset_index(drop=True))
                for s, g in _df.groupby("symbol")}
    else:
        data = build_daily_data(universe, start, end, force=force_data)

    # v3.4 sh.000001 上证指数 (真实指数, 非等权代理): 用于择时信号 + 指数书
    mkt_proxy = None
    try:
        data = await _ensure_index_data(data, start, end)
        if "sh.000001" in data:
            _g = data["sh.000001"].copy()
            _g["date"] = pd.to_datetime(_g["date"])
            mkt_proxy = _g.set_index("date")["close"].sort_index()
            logger.info("市场择时使用 sh.000001 上证指数 (真实收盘) 作为基准")
        else:
            # fallback: 等权股票均值 (旧版, 可能滞后)
            _frames = []
            for _s, _gd in data.items():
                if _s == "sh.000001":
                    continue
                _gg = _gd.copy()
                _gg["date"] = pd.to_datetime(_gg["date"])
                _frames.append(_gg.set_index("date")["close"].rename(_s))
            mkt_proxy = pd.concat(_frames, axis=1).mean(axis=1).sort_index()
            logger.warning("sh.000001 不可用, 回退等权代理")
    except Exception as _e:
        logger.warning(f"市场代理构建失败(择时Overlay禁用): {_e}")

    # ── 交易日历 ──
    all_dates = sorted({str(pd.Timestamp(d).date()) for df in data.values()
                        for d in pd.to_datetime(df["date"])})
    window = [d for d in all_dates if d <= end.isoformat()][-days:]
    if len(window) < days:
        logger.warning(f"可用交易日 {len(window)} < {days}, 按实际窗口跑")

    pf = ReplayPortfolio(capital)
    day_logs = []
    total_llm_calls = 0
    t0 = time.time()

    # ── 断点续跑: 加载已完成的日期 + 组合状态 (tag 区分同段不同条件, 避免续跑串档) ──
    _tag_suffix = f"_{tag}" if tag else ""
    ckpt_path = REPLAY_DIR / f"checkpoint_{window[0]}_{window[-1]}{_tag_suffix}.json" if window else None
    completed = []
    if ckpt_path and ckpt_path.exists():
        try:
            ck = json.loads(ckpt_path.read_text(encoding="utf-8"))
            completed = ck.get("completed_dates", [])
            if ck.get("portfolio"):
                pf = ReplayPortfolio.from_dict(ck["portfolio"])
            day_logs = ck.get("day_logs", [])
            total_llm_calls = ck.get("llm_calls", 0)
            logger.info(f"断点恢复: 已完成 {len(completed)} 天, 跳过继续")
        except Exception as e:
            logger.warning(f"断点加载失败: {e}")
    window = [d for d in window if d not in completed]
    if max_days and len(window) > max_days:
        window = window[:max_days]  # 分小段跑: 本次最多 max_days 天, 干净退出

    def _save_ckpt():
        if not ckpt_path:
            return
        def _js_default(o):
            # numpy 标量 → Python float (避免 default=str 把数值字符串化)
            if hasattr(o, "item"):
                try:
                    return o.item()
                except Exception:
                    pass
            return str(o)
        try:
            ckpt_path.write_text(json.dumps({
                "completed_dates": completed, "portfolio": pf.to_dict(),
                "day_logs": day_logs, "llm_calls": total_llm_calls,
            }, ensure_ascii=False, default=_js_default), encoding="utf-8")
            logger.debug(f"checkpoint 已保存 ({len(completed)} 天)")
        except Exception as _e:
            logger.error(f"checkpoint 保存失败: {_e}")

    _struct_history: list = []  # v3.3 市场结构滚动 (5日多数平滑)
    _pe_hist: list = []  # v3.5 中位 PE 60日滚动 (估值分位)
    _pb_hist: list = []  # v3.5 中位 PB 60日滚动
    _recently_sold: dict = {}  # v3.5 近期卖出 code->日期 (再买冷却, 降换手)
    _diag_history: list = []  # v3.6 诊断官近5日市场统计队列 (用于趋势判断)

    # ── v4.0 自我进化系统初始化 ──
    _journal = None
    _memory = None
    _evolution = None
    _diag_count = 0  # 累计诊断次数（用于进化周期判断）
    if evolution_mode and (diagnostic_mode or flash_diag_mode):
        from agent.evolution.decision_journal import DecisionJournal
        from agent.evolution.experience_memory import ExperienceMemory
        from agent.evolution.weekly_evolution import EvolutionManager
        _tag_str = tag or "evo"
        _journal = DecisionJournal(REPLAY_DIR / f"journal_{_tag_str}.jsonl")
        _memory = ExperienceMemory(REPLAY_DIR / f"memory_{_tag_str}.json", max_items=80)
        _evolution = EvolutionManager(REPLAY_DIR / f"evolution_{_tag_str}.json", period_days=10)
        _diag_count = len(_journal)
        logger.info(f"[自我进化] 已加载: {len(_journal)}条决策, {len(_memory.items)}条经验, "
                    f"{len(_evolution.snapshots)}次进化总结")

    for i, T in enumerate(window):
        logger.info(f"[{i+1}/{len(window)}] T={T} 回放...")
        try:
            # ── 1. PIT 截面 + 体制 ──
            df_cs = reconstruct_cross_section(data, basic, T)
            if df_cs.empty:
                continue
            df_cs = df_cs[df_cs["isST"] != 1]  # 剔除 ST
            # v5.3: 传入指数 PIT 收盘序列 → 指数趋势 regime (修复单日截面误判跳变)
            _idx_pt = mkt_proxy[mkt_proxy.index <= pd.Timestamp(T)] if mkt_proxy is not None else None
            regime = _detect_regime(df_cs, index_close=_idx_pt).get("regime", "range_bound")
            # v3.3 市场结构识别: 抱团/动量 → 进攻; 轮动/普涨/熊 → 防御/均衡
            # 广度式 regime 在窄幅抱团牛误判成熊, 需结构维度修正 (见 analysis/market_structure.py)
            # v3.3 市场结构: 始终计算 (便宜), 用于初筛 regime 调整 + 进攻/防御切换
            _structure = "震荡"
            try:
                from analysis.market_structure import market_structure, screening_regime
                _structure = market_structure(df_cs)
                _struct_history.append(_structure)
                _struct_history = _struct_history[-5:]  # 5日多数平滑
                _structure = max(set(_struct_history), key=_struct_history.count)
                # 结构 → 初筛 regime: 抱团动量用 strong_bull 权重 (让龙头进前300)
                screen_regime = screening_regime(_structure, regime)
                if screen_regime != regime:
                    logger.info(f"市场结构 {_structure} → 初筛 regime 用 {screen_regime} (原 {regime})")
            except Exception as _e:
                logger.warning(f"市场结构识别失败: {_e}")
                screen_regime = regime
            # v3.4 全市场拥挤度 (动量崩溃预警): 极端活跃占比 → hot/warm/cool
            _crowd = {"score": 50.0, "signal": "cool", "hot_ratio": 0.0}
            try:
                from analysis.crowding import market_crowding, format_crowding
                _crowd = market_crowding(df_cs)
                if crowding_overlay:
                    logger.info(format_crowding(_crowd))
            except Exception as _e:
                logger.warning(f"拥挤度信号失败: {_e}")
            _off = offensive or (auto_structure and _structure == "抱团动量")

            # ── v4.0 自我进化: 复盘昨天的决策 ──
            # PIT 正确: 用今天(T日)的已知数据，复盘昨天(T-1日)的诊断判断
            if _journal is not None and _diag_count > 0:
                # 找昨天的记录（最近一条未复盘的）
                _unreviewed = _journal.unreviewed(before=T)
                if _unreviewed:
                    _yesterday_rec = _unreviewed[-1]  # 最近的那条
                    # 今天的市场统计（作为"次日实际结果"）
                    _today_stats = _compute_day_stat(df_cs)
                    # 计算市场涨跌幅（用指数代理或截面均值）
                    pct_col = "pctChg" if "pctChg" in df_cs.columns else "pct_change"
                    if pct_col in df_cs.columns:
                        _mkt_move = float(pd.to_numeric(df_cs[pct_col], errors="coerce").mean())
                    else:
                        _mkt_move = 0.0
                    try:
                        from agent.evolution.daily_review import review_decision, extract_experience
                        _review = await review_decision(
                            _yesterday_rec, _today_stats, _mkt_move
                        )
                        if _review is not None:
                            # v5.4 组合级反事实: 用 T-1 持仓快照 + T 日个股涨跌幅,
                            # 验证"移除当天拖累最大的持仓"是否真能改善组合. 后视信号, 附入 review.
                            try:
                                from agent.evolution.portfolio_counterfactual import portfolio_level_counterfactual
                                _pcf_col = "pctChg" if "pctChg" in df_cs.columns else "pct_change"
                                _stock_ret = {}
                                if _pcf_col in df_cs.columns and "symbol" in df_cs.columns:
                                    _sub = df_cs[["symbol", _pcf_col]].dropna(subset=[_pcf_col])
                                    _stock_ret = dict(zip(_sub["symbol"],
                                                          pd.to_numeric(_sub[_pcf_col], errors="coerce")))
                                _pcf = portfolio_level_counterfactual(
                                    _yesterday_rec.positions_snapshot, _stock_ret,
                                    date=_yesterday_rec.date)
                                if _pcf is not None:
                                    _review["portfolio_cf"] = _pcf.to_dict()
                            except Exception as _pce:
                                logger.warning(f"组合级反事实失败: {_pce}")
                            _journal.update_review(_yesterday_rec.date, _review)
                            total_llm_calls += 1
                            # 提取经验存入记忆库
                            _exp = extract_experience(_yesterday_rec, _review)
                            if _exp is not None:
                                # v4.1: 反事实验证 — 验证教训是否真的有用
                                _cf_result = None
                                try:
                                    from agent.evolution.counterfactual import verify_counterfactual
                                    _cf_result = verify_counterfactual(_exp, _mkt_move)
                                    if _cf_result is not None:
                                        if _cf_result.passed:
                                            # 通过验证: 提高置信度 + 打标记
                                            _exp.confidence = min(0.95, _exp.confidence + 0.15)
                                            _exp.tags.append("cf_verified")
                                            _cf_note = f"反事实✅(+{_cf_result.improvement*100:.2f}%)"
                                        else:
                                            # 没通过: 降低置信度
                                            _exp.confidence = max(0.2, _exp.confidence - 0.1)
                                            _exp.tags.append("cf_failed")
                                            _cf_note = f"反事实❌({_cf_result.improvement*100:.2f}%)"
                                    else:
                                        _cf_note = "反事实N/A"
                                except Exception as _cfe:
                                    _cf_note = f"反事实err"

                                _is_new = _memory.add(_exp)
                                logger.info(
                                    f"[复盘] {_yesterday_rec.date} → {_review.get('verdict','?')} "
                                    f"偏差{_review.get('risk_level_deviation',0):+d}级 "
                                    f"{'新经验' if _is_new else '经验合并'} "
                                    f"[{_cf_note}]"
                                )
                    except Exception as _re:
                        logger.warning(f"复盘失败: {_re}")

                # ── v4.0 周期性进化总结 ──
                if (_evolution is not None
                        and _evolution.should_evolve(T, _diag_count)
                        and _diag_count >= 10):
                    try:
                        from agent.evolution.decision_journal import DecisionRecord
                        # 取最近一个周期的决策
                        _all_decisions = _journal.load_range(
                            start=_evolution.snapshots[-1].period_end if _evolution.snapshots else "2000-01-01",
                            end=T
                        )
                        if len(_all_decisions) >= 5:
                            # 只取有复盘的
                            _reviewed = [d for d in _all_decisions if d.review is not None]
                            if len(_reviewed) >= 5:
                                _mem_items = [
                                    it for it in _memory.items
                                    if it.date >= _all_decisions[0].date
                                ]
                                _snap = await _evolution.evolve(_reviewed, _mem_items)
                                total_llm_calls += 1
                                if _snap:
                                    logger.info(
                                        f"[进化总结] 完成! "
                                        f"原则{len(_snap.principles)}条, "
                                        f"偏见{len(_snap.summary.get('biases_identified', []))}个"
                                    )
                    except Exception as _ee:
                        logger.warning(f"进化总结失败: {_ee}")

            # v3.5: 中位估值 60日分位 (PE/PB 截面中位数, 判断全市场贵贱)
            _med_pe = _med_pb = _pe_pctl = _pb_pctl = 0.0
            if "pe_ttm" in df_cs.columns and "pb" in df_cs.columns:
                _pe_s = pd.to_numeric(df_cs["pe_ttm"], errors="coerce")
                _pb_s = pd.to_numeric(df_cs["pb"], errors="coerce")
                _med_pe = float(_pe_s[_pe_s > 0].median()) if (_pe_s > 0).any() else 0.0
                _med_pb = float(_pb_s[_pb_s > 0].median()) if (_pb_s > 0).any() else 0.0
                _pe_hist.append(_med_pe); _pe_hist = _pe_hist[-60:]
                _pb_hist.append(_med_pb); _pb_hist = _pb_hist[-60:]
                _pe_pctl = float((np.array(_pe_hist) <= _med_pe).mean() * 100) if _pe_hist else 50.0
                _pb_pctl = float((np.array(_pb_hist) <= _med_pb).mean() * 100) if _pb_hist else 50.0
            # v3.5: 指数动量 (真指数 vs MA{ma_window} 位置, 供 LLM 判断是否 risk_off)
            _idx_mom = ""
            if mkt_proxy is not None:
                try:
                    _tpq = pd.Timestamp(T)
                    _past_idx = mkt_proxy[mkt_proxy.index <= _tpq]
                    if len(_past_idx) >= ma_window:
                        _maI = float(_past_idx.iloc[-ma_window:].mean())
                        _pxI = float(_past_idx.iloc[-1])
                        _devI = _pxI / _maI - 1.0
                        _idx_mom = (f"指数动量: 现{_pxI:.3f} vs MA{ma_window}={_maI:.3f} "
                                    f"({_devI:+.1%}), {'上方/risk_on' if _devI >= 0 else '下方/risk_off'}")
                except Exception as _e:
                    logger.warning(f"指数动量计算失败: {_e}")
            # v3.2: 构建市场状态/情绪/操作原则上下文 (仅 v32 变体注入)
            if _off:
                # v3.3 进攻: 抱团/窄幅动量市 — 广度失真, 指令忽略 regime 判熊
                regime_ctx = ("今日为抱团/动量市: 强势龙头持续走强, 普通股票普跌。"
                              "操作原则: 优先选择动量最强/相对强度最高/放量突破的龙头强势股, "
                              "回避低估值但趋势下跌的防御股。"
                              f" (structure={_structure}, regime检测={regime})")
            else:
                regime_ctx = build_market_ctx(df_cs, regime) if variant == "v32" else None

            # v3.5 自由模式: 总是注入完整市场知识 (拥挤度/估值分位/指数动量 + 宏观新闻),
            # 让 LLM 深析有决策依据, 并驱动当日的市场风险决策槽.
            # PIT 守卫: 宏观缓存是"当前日期"快照, 仅当回放窗口与该日期接近才注入,
            # 否则把未来新闻喂进历史回放 = lookahead 偏差 (2020 回放不注入 2026 新闻).
            _macro_ctx = None
            try:
                _mc_path = Path("knowledge/macro_context_latest.json")
                if _mc_path.exists():
                    _mc_d = json.loads(_mc_path.read_text(encoding="utf-8")).get("date", "")
                    if _mc_d and abs((pd.Timestamp(T) - pd.Timestamp(_mc_d)).days) <= 45:
                        _macro_ctx = load_macro_context()
            except Exception as _e:
                logger.warning(f"宏观上下文PIT守卫失败: {_e}")
            _market_ctx_txt = None
            if free_mode:
                _base_mkt = build_market_ctx(df_cs, regime)
                _enrich = (
                    f" | 拥挤度: {_crowd['signal']}(score {_crowd['score']}, "
                    f"极端活跃 {_crowd['hot_ratio']:.1%})"
                    f" | 中位估值: PE{_med_pe:.0f}(60日{_pe_pctl:.0f}%) "
                    f"PB{_med_pb:.1f}(60日{_pb_pctl:.0f}%)"
                    + (f" | {_idx_mom}" if _idx_mom else "")
                    + (f"\n宏观/新闻: {_macro_ctx}" if _macro_ctx else "")
                )
                regime_ctx = f"{_base_mkt}{_enrich}"
                _market_ctx_txt = regime_ctx
            # v3.5 自由模式: 每日市场风险决策槽 — LLM 显式承诺风险目标与持仓上限
            _risk_target, _llm_max_pos = 1.0, 999
            if free_mode and _market_ctx_txt:
                _mr = await _market_risk_decision(_market_ctx_txt)
                _risk_target = _mr["risk_target"]
                _llm_max_pos = _mr["max_positions"]
                total_llm_calls += 1

            # ── 2. 规则初筛 → 300 (v3.4 回退: RPS/拥挤度惩罚 A/B 净负, 默认关闭) ──
            from analysis.pre_screener import PreScreener
            screener = PreScreener()
            screened = screener.screen(df_cs, regime=screen_regime, top_n=top_n,
                                       structure=_structure).df
            if screened.empty:
                continue

            # ── 3-4. LLM 选股 或 诊断模式 ──
            _diag = None
            if diagnostic_mode:
                # ── v3.6 诊断模式: 跳过 LLM 选股, 只做市场风险诊断 ──
                # 选股完全交给 PreScreener 规则排序, LLM 只输出仓位调节系数
                # v3.6 升级: 传入近5天历史, 让诊断官能看趋势和拐点
                # v4.2: 传入昨日风险等级做稳定性锚点
                _prev_risk = None
                if _journal is not None and _diag_count > 0:
                    _sorted_dates = sorted(_journal._cache.keys())
                    if _sorted_dates:
                        _prev_rec = _journal._cache[_sorted_dates[-1]]
                        _prev_risk = _prev_rec.risk_level

                _diag = await _market_diagnostic(df_cs, regime, _crowd,
                                                 _macro_txt=_macro_ctx,
                                                 history_5d=_diag_history,
                                                 memory=_memory,
                                                 evolution=_evolution,
                                                 current_date=T,
                                                 prev_risk_level=_prev_risk)
                total_llm_calls += 1
                # 存今日统计入历史队列 (保留近5天)
                _today_stat = _compute_day_stat(df_cs)
                _diag_history.append(_today_stat)
                if len(_diag_history) > 5:
                    _diag_history.pop(0)
                deep = []  # 不用 LLM 选股结果
                # 规则精筛: 取 composite_score 最高的 N 只
                # 动态集中度模式: 风险等级越低越集中, 风险越高越分散
                if diagnostic_dynamic:
                    _rl = int(_diag.get("risk_level", 3))
                    _dyn_map = {1: 10, 2: 15, 3: 20, 4: 30, 5: 40}
                    _top_k = min(_dyn_map.get(_rl, 20), len(screened))
                else:
                    _top_k = min(diagnostic_top_n, len(screened))
                _diag["_top_k"] = _top_k  # 传给后面仓位调整block用
                _rule_buys = []
                for _, _row in screened.head(_top_k).iterrows():
                    _score = float(_row.get("composite_score", _row.get("rule_score", 50)))
                    # conviction: 按分数平滑映射, 60分→0.6, 80分→0.8, 100分→0.95
                    _conv = min(0.95, max(0.4, _score / 100))
                    _rule_buys.append({
                        "code": str(_row["code"]),
                        "name": str(_row.get("name", "")),
                        "action": "BUY",
                        "conviction": _conv,
                        "final_score": _score,
                    })
                deep = _rule_buys
                logger.info(
                    f"诊断模式: 规则精筛Top{len(deep)}只 → BUY候选. "
                    f"诊断风险={_diag['risk_level']}/5 仓位×{_diag['position_multiplier']:.2f}  "
                    f"分数范围[{_rule_buys[-1]['final_score']:.1f}~{_rule_buys[0]['final_score']:.1f}]"
                    if _rule_buys else "诊断模式: 无候选股")
            elif flash_diag_mode:
                # ── v3.6 Flash诊断模式: Flash精筛 + 诊断官 (无深度分析) ──
                # 中等成本方案: 保留 LLM 选股判断 (比纯规则强), 但砍掉昂贵的深度分析
                _diag = await _market_diagnostic(df_cs, regime, _crowd,
                                                 _macro_txt=_macro_ctx,
                                                 history_5d=_diag_history,
                                                 memory=_memory,
                                                 evolution=_evolution,
                                                 current_date=T)
                total_llm_calls += 1
                # 存今日统计入历史队列
                _today_stat = _compute_day_stat(df_cs)
                _diag_history.append(_today_stat)
                if len(_diag_history) > 5:
                    _diag_history.pop(0)
                df_top = await _flash_screen(screened, top_k=final_n, regime=regime,
                                             offensive=_off)
                total_llm_calls += (len(screened) + 24) // 25
                # Flash精筛结果直接当 BUY 候选 (无深析, 无 SELL 建议)
                deep = []
                for _, _row in df_top.head(final_n).iterrows():
                    _score = float(_row.get("composite_score", _row.get("flash_score", 60)))
                    _conv = min(0.9, max(0.5, _score / 120))
                    deep.append({
                        "code": str(_row["code"]),
                        "name": str(_row.get("name", "")),
                        "action": "BUY",
                        "conviction": _conv,
                        "final_score": _score,
                    })
                logger.info(
                    f"Flash诊断: Flash精筛Top{len(deep)}只 → BUY候选. "
                    f"诊断风险={_diag['risk_level']}/5 仓位×{_diag['position_multiplier']:.2f}")
            else:
                # ── 3. LLM Flash 精筛 → 100 ──
                df_top = await _flash_screen(screened, top_k=final_n, regime=regime,
                                             offensive=_off)
                total_llm_calls += (len(screened) + 24) // 25

                # ── 4. LLM 深度分析 (thinking 可开关: 开=更深推理, 关=快速) ──
                # v3.3 持仓纳入: 持仓股不在当日 Top100 也纳入深析 (避免持仓盲区 —
                # 否则持仓只靠止损/止盈触发, LLM 的 SELL 信号永远看不到它)
                df_deep = df_top.head(final_n).copy()
                _held_codes = {_sym.split(".")[-1] for _sym in pf.positions}
                if _held_codes:
                    _top_codes = set(df_deep["code"].astype(str))
                    _missing_held = [c for c in _held_codes if c not in _top_codes]
                    if _missing_held:
                        _held_rows = df_cs[df_cs["code"].astype(str).isin(_missing_held)]
                        if not _held_rows.empty:
                            df_deep = pd.concat([df_deep, _held_rows], ignore_index=True)
                            logger.info(f"  持仓纳入深析: {len(_held_rows)} 只不在Top{final_n}")
                deep = await _deepseek_analyze(df_deep, thinking=thinking,
                                               variant=variant, regime_ctx=regime_ctx,
                                               macro_context=_macro_ctx)
                _bs = 10 if thinking else 20
                total_llm_calls += (len(df_deep) + _bs - 1) // _bs
                # v3.1.1 修复: LLM JSON 偶尔把数值返成字符串 ("conviction":"0.6"),
                # 后续比较/运算会崩 (Float32 vs Str). 归一化为 float.
                for _r in deep:
                    for _k in ("conviction", "final_score", "score", "win_rate"):
                        if _k in _r and not isinstance(_r[_k], (int, float)):
                            try:
                                _r[_k] = float(_r[_k])
                            except (TypeError, ValueError):
                                _r[_k] = 0.0

            # ── 5. 持仓规划 (T 收盘决策) ──
            # 卖出: SELL 信号 + 止损/止盈 (用 T+1 开盘价判断)
            next_T = _next_trading_day(data, T)
            t1_open = {}
            t1_close = {}
            if next_T:
                for sym, df in data.items():
                    d = pd.to_datetime(df["date"])
                    mask = d <= pd.Timestamp(next_T)
                    if not mask.any():
                        continue
                    r1 = df[mask].iloc[-1]
                    if r1["is_trade"] == 1:
                        t1_open[sym] = r1["open"]
                        t1_close[sym] = r1["close"]

            sell_signals = {str(r.get("code")): r for r in deep if r.get("action") == "SELL"}
            for sym in list(pf.positions.keys()):
                pos = pf.positions[sym]
                code = sym.split(".")[-1]
                px1 = t1_open.get(sym)
                if px1 is None or px1 <= 0:
                    continue
                reason = None
                if code in sell_signals:
                    reason = "SELL信号"
                elif pos.get("stop") and px1 <= pos["stop"]:
                    reason = "止损"
                elif pos.get("take") and px1 >= pos["take"]:
                    reason = "止盈"
                if reason:
                    pf.sell(sym, px1, next_T, reason)
                    _recently_sold[code] = pd.Timestamp(T)  # v3.5 记录卖出日, 再买冷却

            # 买入: BUY 信号按 final_score 排序, 仓位/现金/上限
            buy_candidates = [r for r in deep if r.get("action") == "BUY"
                              and r.get("conviction", 0) >= 0.5]
            buy_candidates.sort(key=lambda r: r.get("final_score", 0), reverse=True)
            regime_mult = {"strong_bull": 1.0, "weak_bull": 0.8, "range_bound": 0.5,
                           "weak_bear": 0.3, "strong_bear": 0.1, "crisis": 0.0}.get(regime, 0.5)

            # v3.5 自由模式: 去除所有硬约束, 让 LLM 自主决定仓位和持仓数.
            # regime_mult/timing_mult/crowding_mult 全部取消, 由 conviction 直接驱动.
            # 择时/拥挤度信号仍记录在日志中供 LLM 参考 (已在 regime_ctx 中), 但不强制压仓.
            if free_mode:
                regime_mult = 1.0
                # 记录择时/拥挤度信号供参考, 但不改变仓位行为

            # v3.3 市场择时 Overlay (--timing): 市场代理 < MA{ma_window} (risk_off) 时防御 —
            # 仓位×0.3, 最多2只。用回放自己的等权代理 ≤T 算 MA (PIT 安全, 与择时回测同信号)
            timing_mult, timing_max_pos = 1.0, pf.max_positions
            _risk_on = True
            if timing_overlay and mkt_proxy is not None:
                try:
                    _tp = pd.Timestamp(T)
                    past = mkt_proxy[mkt_proxy.index <= _tp]
                    if len(past) >= ma_window + 5:
                        _ma = past.iloc[-ma_window:].mean()
                        px = past.iloc[-1]
                        # v3.3 迟滞带: 价格在 MA±2% 带内不切换, 只有深跌破才 risk_off,
                        # 深升破才 risk_on — 吸收 02-03 卖/02-05 买 这类 whipsaw 抖动.
                        _band = 0.02
                        if pf.index_units > 0:
                            # 已持指数: 仅当深跌破 MA×(1-2%) 才清
                            _risk_on = not (px < _ma * (1 - _band))
                        else:
                            # 未持指数: 仅当深升破 MA×(1+2%) 才视为 risk_on
                            _risk_on = px > _ma * (1 + _band)
                        if not _risk_on:
                            timing_mult, timing_max_pos = 0.3, min(pf.max_positions, 2)
                            logger.warning(
                                f"市场择时 risk_off (代理{px:.3f} < MA{ma_window}×{(1-_band):.2f}={_ma*(1-_band):.3f}) "
                                f"→ 防御: 仓位×0.3, 最多{timing_max_pos}只")
                        else:
                            logger.info(
                                f"市场择时 risk_on (代理{px:.3f} vs MA{ma_window} {_ma:.3f})")
                except Exception as e:
                    logger.warning(f"市场择时信号失败(保持原仓位): {e}")

            # v3.5 诊断模式 / Flash诊断模式: 用 LLM 诊断系数调节仓位和持仓上限
            if (diagnostic_mode or flash_diag_mode) and _diag is not None:
                _diag_mult = float(_diag.get("position_multiplier", 1.0))
                _diag_adj = int(_diag.get("max_positions_adj", 0))
                _rl = int(_diag.get("risk_level", 3))
                timing_mult *= _diag_mult

                if diagnostic_dynamic:
                    # 动态集中度: 风险等级直接决定持仓上限 (低风险集中, 高风险分散)
                    _pos_map = {1: 8, 2: 12, 3: 15, 4: 20, 5: 25}
                    _base_pos = _pos_map.get(_rl, 15)
                    timing_max_pos = max(1, min(30, _base_pos + _diag_adj))
                    _tk = int(_diag.get("_top_k", 0))
                    logger.info(
                        f"[诊断模式-动态] 风险{_rl}/5 → 候选{_tk}只 "
                        f"上限{timing_max_pos}只  仓位×{_diag_mult:.2f}"
                    )
                else:
                    timing_max_pos = max(1, min(30, timing_max_pos + _diag_adj))
                    logger.info(
                        f"[诊断模式] 仓位×{_diag_mult:.2f}  持仓上限{_diag_adj:+d}→{timing_max_pos}只  "
                        f"风险等级{_rl}/5"
                    )
                if _diag.get("key_risks"):
                    for _kr in _diag["key_risks"][:3]:
                        logger.info(f"  ⚠ {_kr}")

                # ── v4.0 自我进化: 写入决策日志 ──
                if _journal is not None:
                    from agent.evolution.decision_journal import DecisionRecord
                    _snap = _compute_day_stat(df_cs)
                    # v5.4 组合级反事实: 记录当日持仓快照 (用 T 日 t1_close PIT 正确).
                    # total = cash + Σ持仓市值 (指数书是市场代理, 不参与个股权衡, 略去).
                    _pf_snap = []
                    _pf_pos_val = 0.0
                    for _sym, _p in pf.positions.items():
                        _px = t1_close.get(_sym, _p.get("entry_price", 0.0))
                        _val = float(_p.get("qty", 0)) * float(_px)
                        _pf_pos_val += _val
                        _pf_snap.append({
                            "symbol": _sym, "name": _p.get("name", ""),
                            "qty": int(_p.get("qty", 0)),
                            "price": round(float(_px), 3),
                            "value": round(_val, 2), "weight": 0.0,
                        })
                    _pf_total = pf.cash + _pf_pos_val
                    for _s in _pf_snap:
                        _s["weight"] = round(_s["value"] / _pf_total, 4) if _pf_total > 0 else 0.0
                    _rec = DecisionRecord(
                        date=T,
                        market_phase=_diag.get("market_phase", "unknown"),
                        dominant_master=_diag.get("dominant_master", "unknown"),
                        secondary_master=_diag.get("secondary_master", ""),
                        risk_level=_rl,
                        position_multiplier=_diag_mult,
                        max_positions_adj=_diag_adj,
                        key_risks=_diag.get("key_risks", []),
                        diagnosis=_diag.get("diagnosis", ""),
                        market_snapshot=_snap,
                        regime=regime,
                        crowding_score=float(_crowd.get("score", 0)),
                        crowding_signal=str(_crowd.get("signal", "")),
                        positions_snapshot=_pf_snap,
                        total_value=_pf_total,
                    )
                    _journal.record(_rec)
                    _diag_count += 1

            # v3.5 自由模式: 择时/拥挤度信号仅记录日志, 不强制压仓
            if free_mode:
                timing_mult, timing_max_pos = 1.0, 999
                if _crowd.get("signal") in ("hot", "warm"):
                    logger.info(f"择时:{_risk_on} | 拥挤度:{_crowd['signal']} (score {_crowd['score']}) → 自由模式仅记录, 不压仓")
                elif timing_overlay:
                    logger.info(f"择时:{_risk_on} → 自由模式仅记录, 不压仓")

            # v3.4 拥挤度 Overlay (--crowding): 市场过热 (极端活跃占比高) 时, 动量崩溃
            # 风险 → 收紧新买入仓位 (hot×0.5 / warm×0.8). 独立于择时, 可叠加.
            if crowding_overlay and _crowd.get("signal") in ("hot", "warm"):
                _crowd_mult = 0.5 if _crowd["signal"] == "hot" else 0.8
                timing_mult *= _crowd_mult
                timing_max_pos = min(timing_max_pos, 6 if _crowd["signal"] == "hot" else 8)
                logger.warning(
                    f"拥挤度 {_crowd['signal']} (score {_crowd['score']}, 极端活跃 "
                    f"{_crowd['hot_ratio']:.1%}) → 新买仓位×{_crowd_mult}, 最多{timing_max_pos}只")

            # v3.3 混合结构: risk_on 时按 hybrid_pct 买入市场指数书 (持市场代理吃满牛市),
            #               risk_off 时清仓 (落袋避熊). 剩余现金做进攻/防御选股.
            if hybrid_pct > 0 and mkt_proxy is not None and next_T:
                try:
                    _t1 = pd.Timestamp(next_T)
                    _idx_now = float(mkt_proxy.loc[_t1]) if _t1 in mkt_proxy.index \
                        else float(mkt_proxy[mkt_proxy.index <= _t1].iloc[-1])
                    if _risk_on and pf.index_units <= 0:
                        _amt = capital * hybrid_pct
                        if pf.buy_index(_amt, _idx_now, next_T):
                            logger.info(f"混合结构: risk_on 买指数书 ¥{_amt:,.0f} (指数{_idx_now:.3f})")
                    elif not _risk_on and pf.index_units > 0:
                        pf.sell_index(_idx_now, next_T)
                        logger.info(f"混合结构: risk_off 清指数书 @指数{_idx_now:.3f}")
                except Exception as e:
                    logger.warning(f"混合结构指数书失败: {e}")

            for r in buy_candidates:
                if free_mode:
                    if len(pf.positions) >= _llm_max_pos or pf.cash < capital * 0.05:
                        break
                elif len(pf.positions) >= timing_max_pos or pf.cash < capital * 0.05:
                    break
                code = str(r.get("code"))
                sym = next((s for s in data if s.split(".")[-1] == code), None)
                if sym is None or sym in pf.positions:
                    continue
                # v3.5 再买冷却: 5日内卖出的不立刻回补 (降换手/降反复止损)
                if code in _recently_sold and (pd.Timestamp(T) - _recently_sold[code]).days < 5:
                    continue
                px1 = t1_open.get(sym)
                if px1 is None or px1 <= 0:
                    continue
                # v3.5 自由模式: 仓位 = 确信度 × 市场风险目标 (LLM每日显式承诺), 封顶20%
                if free_mode:
                    # conviction 0.5→7.5%, 1.0→15%, 封顶 20%. 高确信就重仓, 低确信轻仓
                    pct = min(0.20, r.get("conviction", 0.5) * 0.15) * _risk_target
                else:
                    # 旧模式: 等权 × 确信度 × 体制 × 择时, 封顶 10%
                    pct = min(0.10, 0.03 + r.get("conviction", 0.5) * 0.05) * regime_mult * timing_mult
                min_pct = 100 * px1 / pf.cash  # 1 手占资金比例
                if pct < min_pct:
                    pct = min(min_pct, 0.10)   # 资金允许则至少买 1 手
                qty = int(pf.cash * pct / px1 / 100) * 100
                if qty < 100 or pf.cash * pct > pf.cash * 0.25:
                    continue
                # v3.3 ATR 缩放止损: 高波动股放宽, 低波动股收紧 (2×ATR, 限5-10%)
                _g = data.get(sym)
                _stop_pct = 0.07
                if _g is not None and "high" in _g.columns and "close" in _g.columns:
                    _h, _l, _c = (_g["high"], _g["low"], _g["close"])
                    _pc = _c.shift(1)
                    _tr = pd.concat([_h - _l, (_h - _pc).abs(), (_l - _pc).abs()], axis=1).max(axis=1)
                    _atr = float(_tr.rolling(20).mean().iloc[-1]) if len(_tr) >= 20 else 0.0
                    if _atr > 0 and px1 > 0:
                        _stop_pct = float(np.clip(2 * _atr / px1, 0.05, 0.10))
                stop = px1 * (1 - _stop_pct)
                # v3.3 回退: 止盈A/B证明 regime 自适应净负, 回到固定 +12%
                take = px1 * (1 + 0.12)
                if pf.buy(sym, r.get("name", sym), px1, qty, next_T, stop=stop, take=take):
                    pass

            # ── 6. T+1 收盘市值快照 (含指数书) ──
            if next_T:
                _idx_lv = None
                if mkt_proxy is not None:
                    _t1 = pd.Timestamp(next_T)
                    _idx_lv = float(mkt_proxy.loc[_t1]) if _t1 in mkt_proxy.index \
                        else float(mkt_proxy[mkt_proxy.index <= _t1].iloc[-1])
                total = pf.total_value(t1_close, index_level=_idx_lv) if t1_close else pf.cash
                pf.equity_curve.append({"date": next_T, "total": round(total, 2),
                                        "cash": round(pf.cash, 2),
                                        "positions": len(pf.positions),
                                        "index_book": round(pf.index_value(_idx_lv), 2) if _idx_lv else 0.0})
            day_logs.append({"date": T, "regime": regime, "screened": len(df_cs),
                             "top": len(deep),
                             "buy": sum(1 for r in deep if r.get("action") == "BUY"),
                             "sell": sum(1 for r in deep if r.get("action") == "SELL"),
                             "hold": sum(1 for r in deep if r.get("action") == "HOLD"),
                             "positions": len(pf.positions)})
            # 断点: 每天完成后立即持久化 (被杀不丢进度)
            completed.append(T)
            _save_ckpt()
            logger.info(f"  ✔ [{i+1}/{len(window)}] {T} 完成, 持仓{len(pf.positions)}")
        except Exception as e:
            import traceback as _tb
            logger.warning(f"[{T}] 回放异常: {e}\n{_tb.format_exc()[-1500:]}")

    elapsed = time.time() - t0
    logger.info(f"回放完成: {len(window)}日, {total_llm_calls}次LLM, {elapsed:.0f}s")

    # ── 报告 ──
    result = _build_report(pf, window, elapsed, day_logs, tag=tag)
    return result


def _build_report(pf: ReplayPortfolio, window, elapsed, day_logs=None, tag: str = None) -> dict:
    eq = pf.equity_curve
    # v3.1.1 修复: checkpoint 用 default=str 保存时 numpy float 变字符串, 加载后除法崩溃
    final_total = float(eq[-1]["total"]) if eq else float(pf.capital)
    ret = (final_total / float(pf.capital) - 1) * 100
    # 基准: 全窗口简单收益率 (用窗口内平均日收益近似 — 这里用持仓等权)
    report = {
        "window": [window[0], window[-1]] if window else [],
        "final_equity": round(final_total, 2),
        "total_return_pct": round(ret, 2),
        "num_trades": len(pf.trades),
        "final_positions": list(pf.positions.keys()),
        "elapsed_seconds": round(elapsed, 1),
        "equity_curve": eq,
        "trades": pf.trades[-30:],
        "llm_calls": None,
        "day_logs": day_logs or [],
    }
    _tag_suffix = f"_{tag}" if tag else ""
    out = REPLAY_DIR / f"replay_report{_tag_suffix}.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    # 控制台表格
    print("\n=== 历史 PIT 回放结果 ===")
    print(f"窗口: {report['window']} | 最终权益: CNY {final_total:,.2f} | 总收益: {ret:+.2f}%")
    print(f"交易: {len(pf.trades)} 笔 | 期末持仓: {len(pf.positions)} 只 | 耗时: {elapsed:.0f}s")
    if eq:
        print("\n权益曲线 (每5日):")
        for s in eq[::5]:
            print(f"  {s['date']}: CNY {s['total']:,.2f} ({s['positions']}只)")
    print(f"\n报告: {out}")
    return report


def main():
    ap = argparse.ArgumentParser(description="历史 PIT 回放 (全市场→LLM→持仓)")
    ap.add_argument("--days", type=int, default=40, help="回放交易日数")
    ap.add_argument("--universe", type=str, default="full",
                    help="股票池: full=全市场, 或逗号分隔 symbol 列表")
    ap.add_argument("--top-n", type=int, default=300, help="规则初筛数量")
    ap.add_argument("--final-n", type=int, default=100, help="LLM 精筛数量")
    ap.add_argument("--capital", type=float, default=100000.0, help="初始资金")
    ap.add_argument("--force-data", action="store_true", help="强制重建数据缓存")
    ap.add_argument("--thinking", action="store_true",
                    help="深度分析启用 thinking 模式 (更深推理, 更慢更贵; 初筛始终禁用)")
    ap.add_argument("--max-days", type=int, default=0,
                    help="本次最多处理的新天数 (0=不限; 分小段跑避免后台被杀)")
    ap.add_argument("--variant", type=str, default="baseline",
                    choices=["baseline", "v32"],
                    help="baseline=原版; v32=注入 v3.2 因子(相对估值分位/风险调整动量/反转) + regime 门控市场上下文")
    ap.add_argument("--end", type=str, default="2026-07-31",
                    help="回放窗口终点 YYYY-MM-DD (多 regime 分段跑)")
    ap.add_argument("--data-file", type=str, default=None,
                    help="显式缓存 parquet (长窗口全市场), 跳过按名缓存查找")
    ap.add_argument("--tag", type=str, default=None,
                    help="运行标签, 用于区分同段不同条件 (checkpoint/报告文件名后缀, 防止续跑串档)")
    ap.add_argument("--timing", action="store_true",
                    help="启用市场择时 Overlay (指数<MA20 时防御仓位, 验证双动量)")
    ap.add_argument("--hybrid", type=float, default=0.0,
                    help="混合结构: risk_on 时把该比例资金投入市场指数书 (0~1, 如 0.5=一半持指数)")
    ap.add_argument("--offensive", action="store_true",
                    help="强制进攻模式: 初筛/深析优先强势龙头动量 (抱团/窄幅动量牛, 广度失真时用)")
    ap.add_argument("--auto-structure", action="store_true",
                    help="自动市场结构识别: 抱团动量→进攻, 轮动普涨/熊→防御 (5日多数平滑)")
    ap.add_argument("--crowding", action="store_true",
                    help="启用拥挤度 Overlay (v3.4): 市场过热时收紧新买仓位, 避动量崩盘")
    ap.add_argument("--ma-window", type=int, default=20,
                    help="择时均线窗口 (研究: 更长窗口更稳, 但迟滞大; 默认20)")
    ap.add_argument("--free-mode", action="store_true",
                    help="动态持仓自由模式: 去除硬约束 (持仓上限/择时压仓/regime压仓), 让 LLM 自主决定仓位和持仓数")
    ap.add_argument("--diagnostic", action="store_true",
                    help="诊断模式 (v3.5): LLM不选股, 只做市场风险诊断并调节仓位; 选股完全交给规则系统")
    ap.add_argument("--diag-top-n", type=int, default=20,
                    help="诊断模式下规则精筛的候选股数 (默认20; 值越集中进攻性越强)")
    ap.add_argument("--diag-dynamic", action="store_true",
                    help="诊断模式动态集中度: 风险等级越低,持仓越集中;风险越高越分散 (覆盖--diag-top-n)")
    ap.add_argument("--flash-diag", action="store_true",
                    help="Flash诊断模式 (v3.5): Flash精筛选股 + 诊断官调仓, 无深度分析 (中等成本)")
    ap.add_argument("--evolution", action="store_true",
                    help="v4.0 自我进化模式: 诊断官每天复盘昨日决策, 积累经验记忆, 每10天进化总结")
    args = ap.parse_args()

    universe = None if args.universe == "full" else [s.strip() for s in args.universe.split(",") if s.strip()]
    asyncio.run(run_replay(days=args.days, universe=universe, top_n=args.top_n,
                           final_n=args.final_n, capital=args.capital,
                           force_data=args.force_data, thinking=args.thinking,
                           max_days=args.max_days, variant=args.variant,
                           end_date=args.end, data_file=args.data_file, tag=args.tag,
                           timing_overlay=args.timing, hybrid_pct=args.hybrid,
                           offensive=args.offensive, auto_structure=args.auto_structure,
                           crowding_overlay=args.crowding, ma_window=args.ma_window,
                           free_mode=args.free_mode,
                           diagnostic_mode=args.diagnostic,
                           diagnostic_top_n=args.diag_top_n,
                           diagnostic_dynamic=args.diag_dynamic,
                           flash_diag_mode=args.flash_diag,
                           evolution_mode=args.evolution))


if __name__ == "__main__":
    main()
