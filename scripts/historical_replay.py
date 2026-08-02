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
    _flash_screen, _deepseek_analyze, _detect_regime,
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

        rows.append({
            "code": sym.split(".")[-1], "symbol": sym, "name": bi.get("name", sym),
            "price": close, "pct_change": row["pctChg"] if pd.notna(row["pctChg"]) else 0,
            "volume": row["volume"], "amount": row["amount"],
            "turnover": row["turn"] if pd.notna(row["turn"]) else 0,
            "pe_ttm": row["peTTM"] if pd.notna(row["peTTM"]) else np.nan,
            "pb": row["pbMRQ"] if pd.notna(row["pbMRQ"]) else np.nan,
            "total_mv": total_mv,
            "pct_60d": pct_60d, "vol_ratio": vol_ratio, "amplitude": amplitude,
            "isST": row["isST"], "is_trade": 1,
        })
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════
# 回放主循环
# ═══════════════════════════════════════════════════════════════

class ReplayPortfolio:
    """回放持仓 (轻量): {symbol: {qty, entry_price, entry_date, stop, take}}"""
    def __init__(self, capital=100000.0):
        self.capital = capital
        self.cash = capital
        self.positions = {}
        self.equity_curve = []       # [{date, total, cash, position_value}]
        self.trades = []
        self.max_positions = 10

    def to_dict(self) -> dict:
        return {"capital": self.capital, "cash": self.cash,
                "positions": self.positions, "trades": self.trades,
                "equity_curve": self.equity_curve}

    @classmethod
    def from_dict(cls, d: dict) -> "ReplayPortfolio":
        pf = cls(d.get("capital", 100000.0))
        pf.cash = d.get("cash", pf.capital)
        pf.positions = d.get("positions", {}) or {}
        pf.trades = d.get("trades", []) or []
        pf.equity_curve = d.get("equity_curve", []) or []
        return pf

    def total_value(self, price_map):
        pos_val = sum(p["qty"] * price_map.get(sym, p["entry_price"]) for sym, p in self.positions.items())
        return self.cash + pos_val

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


async def run_replay(days: int = 40, universe=None, top_n: int = 300, final_n: int = 100,
                     capital: float = 100000.0, force_data: bool = False,
                     thinking: bool = False, max_days: int = 0) -> dict:
    """
    max_days: 本次最多处理的新天数 (0=不限). 用于分小段跑, 每段干净退出
              (checkpoint 落盘), 避免后台任务被杀窗口浪费.
    """
    """执行历史 PIT 回放"""
    end = date(2026, 7, 31)  # 最近完整交易日 (08-02 休市)
    # 回放只需 ~60交易日 warmup (算 60日涨跌) + 前瞻缓冲, 无需 250日
    start = end - timedelta(days=int(days * 1.5) + 100)

    # ── 0. 快照基础 + 全市场数据 ──
    basic, snapshot_universe = load_snapshot_basic()
    if universe is None or universe == "full":
        universe = snapshot_universe
    basic = {sym: basic.get(sym, {}) for sym in universe}

    logger.info(f"股票池: {len(universe)} 只, 数据窗口 {start}~{end}")
    data = build_daily_data(universe, start, end, force=force_data)

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

    # ── 断点续跑: 加载已完成的日期 + 组合状态 ──
    ckpt_path = REPLAY_DIR / f"checkpoint_{window[0]}_{window[-1]}.json" if window else None
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

    for i, T in enumerate(window):
        logger.info(f"[{i+1}/{len(window)}] T={T} 回放...")
        try:
            # ── 1. PIT 截面 + 体制 ──
            df_cs = reconstruct_cross_section(data, basic, T)
            if df_cs.empty:
                continue
            df_cs = df_cs[df_cs["isST"] != 1]  # 剔除 ST
            regime = _detect_regime(df_cs).get("regime", "range_bound")

            # ── 2. 规则初筛 → 300 ──
            from analysis.pre_screener import PreScreener
            screener = PreScreener()
            screened = screener.screen(df_cs, regime=regime, top_n=top_n).df
            if screened.empty:
                continue

            # ── 3. LLM Flash 精筛 → 100 ──
            df_top = await _flash_screen(screened, top_k=final_n)
            total_llm_calls += (len(screened) + 24) // 25

            # ── 4. LLM 深度分析 (thinking 可开关: 开=更深推理, 关=快速) ──
            deep = await _deepseek_analyze(df_top.head(final_n), thinking=thinking)
            _bs = 10 if thinking else 20
            total_llm_calls += (len(df_top.head(final_n)) + _bs - 1) // _bs
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

            # 买入: BUY 信号按 final_score 排序, 仓位/现金/上限
            buy_candidates = [r for r in deep if r.get("action") == "BUY"
                              and r.get("conviction", 0) >= 0.5]
            buy_candidates.sort(key=lambda r: r.get("final_score", 0), reverse=True)
            regime_mult = {"strong_bull": 1.0, "weak_bull": 0.8, "range_bound": 0.5,
                           "weak_bear": 0.3, "strong_bear": 0.1, "crisis": 0.0}.get(regime, 0.5)
            for r in buy_candidates:
                if len(pf.positions) >= pf.max_positions or pf.cash < capital * 0.05:
                    break
                code = str(r.get("code"))
                sym = next((s for s in data if s.split(".")[-1] == code), None)
                if sym is None or sym in pf.positions:
                    continue
                px1 = t1_open.get(sym)
                if px1 is None or px1 <= 0:
                    continue
                # 仓位: 等权 × 确信度 × 体制, 封顶 10%; 但至少买得起 1 手 (100股)
                pct = min(0.10, 0.03 + r.get("conviction", 0.5) * 0.05) * regime_mult
                min_pct = 100 * px1 / pf.cash  # 1 手占资金比例
                if pct < min_pct:
                    pct = min(min_pct, 0.10)   # 资金允许则至少买 1 手
                qty = int(pf.cash * pct / px1 / 100) * 100
                if qty < 100 or pf.cash * pct > pf.cash * 0.12:
                    continue
                stop = px1 * (1 - 0.07)
                take = px1 * (1 + 0.12)
                if pf.buy(sym, r.get("name", sym), px1, qty, next_T, stop=stop, take=take):
                    pass

            # ── 6. T+1 收盘市值快照 ──
            if next_T:
                total = pf.total_value(t1_close) if t1_close else pf.cash
                pf.equity_curve.append({"date": next_T, "total": round(total, 2),
                                        "cash": round(pf.cash, 2),
                                        "positions": len(pf.positions)})
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
            logger.warning(f"[{T}] 回放异常: {e}")

    elapsed = time.time() - t0
    logger.info(f"回放完成: {len(window)}日, {total_llm_calls}次LLM, {elapsed:.0f}s")

    # ── 报告 ──
    result = _build_report(pf, window, elapsed, day_logs)
    return result


def _build_report(pf: ReplayPortfolio, window, elapsed, day_logs=None) -> dict:
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
    out = REPLAY_DIR / "replay_report.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    # 控制台表格
    print("\n=== 历史 PIT 回放结果 ===")
    print(f"窗口: {report['window']} | 最终权益: ¥{final_total:,.2f} | 总收益: {ret:+.2f}%")
    print(f"交易: {len(pf.trades)} 笔 | 期末持仓: {len(pf.positions)} 只 | 耗时: {elapsed:.0f}s")
    if eq:
        print("\n权益曲线 (每5日):")
        for s in eq[::5]:
            print(f"  {s['date']}: ¥{s['total']:,.2f} ({s['positions']}只)")
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
    args = ap.parse_args()

    universe = None if args.universe == "full" else [s.strip() for s in args.universe.split(",") if s.strip()]
    asyncio.run(run_replay(days=args.days, universe=universe, top_n=args.top_n,
                           final_n=args.final_n, capital=args.capital,
                           force_data=args.force_data, thinking=args.thinking,
                           max_days=args.max_days))


if __name__ == "__main__":
    main()
