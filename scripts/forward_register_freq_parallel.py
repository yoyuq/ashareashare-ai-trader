"""前瞻注册 — 双周频 vs 月频平行纸面 bet (iter27 PASS 产物的 OOS 确证, playbook 待办).

在双周换仓日 (半月末最后一个交易日) 运行。两臂规则逐字一致 (buffer keep-zone=10,
top5, 等权, 基础版无解禁叠层 — 与 iter27 回测口径一致), 唯一差异 = 换仓日程:
- freq_monthly 臂: 月频日程选券 (注册日持 = 上月末选券, 不动)
- freq_biweekly 臂: 双周日程选券 (注册日 = 半月末, 执行换仓)

评分点内重建: live_panel.parquet (Baostock 真历史) + 当日 cache (turn/amount/收盘价,
15:00 后即真实收盘)。buffer 历史从面板首日起重建 (空仓起步, 逐期 keep-zone 交接)。
两臂同 entry_date (=注册日) 定价起算, 同 universe 基准, 判据同 CRITERION (60td, vs
universe, 失效 -10pp)。iter27 的 OOS 判据: 双臂 60td 后比较 selection_alpha_pp,
B 臂 ≥ M 臂 + 0 且 B vs universe 不劣于 -10pp → 双周频 OOS 确证, 切实操换仓频率。

用法: 先 python scripts/refresh_market_cache.py, 再 python scripts/fetch_live_panel.py,
最后本脚本 (仅在半月末/月末交易日运行, 否则拒绝)。
输出: simulation_data/forward_validation/registry.json
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from analysis.forward_validation import register_bet  # noqa: E402
from data.full_market_cache import read_full_market_cache  # noqa: E402
from forward_register_cold_lowvol import (  # noqa: E402
    CRITERION, K, UNIVERSE_N, _filter_liquid, _price_map, fetch_index,
)

PANEL_FP = ROOT / "replay_data" / "live_panel.parquet"
KEEP_ZONE = 10


def _force_utf8() -> None:
    for name in ("stdout", "stderr"):
        s = getattr(sys, name, None)
        if hasattr(s, "reconfigure"):
            try:
                s.reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass


def load_panel() -> pd.DataFrame:
    if not PANEL_FP.exists():
        raise RuntimeError("live_panel.parquet 不存在, 先跑 scripts/fetch_live_panel.py")
    d = pd.read_parquet(PANEL_FP)
    d["date"] = pd.to_datetime(d["date"])
    return d


def score_at(panel: pd.DataFrame, day: pd.Timestamp) -> pd.Series:
    """点内 score: std20 (面板滚动) + turn/log_mkt (该日截面, 面板口径)。"""
    g = panel[panel["date"] == day].copy()
    if g.empty:
        raise RuntimeError(f"面板无 {day.date()} 截面 (先刷新面板)")
    g = g[g["tradestatus"].astype(str) == "1"]
    g = g[(g["turn"] > 0) & (g["amount"] > 0) & (g["peTTM"] > 0) & (g["pbMRQ"] > 0)]
    g["name_flag"] = g["isST"].astype(str)
    g = g[g["name_flag"] == "0"]  # 非ST
    g["code6"] = g["symbol"].str.split(".").str[-1]
    g = g[g["code6"].str.startswith(("60", "00"))]  # 主板
    # std20: 每 symbol 截至该日的 rolling std 末值
    sub = panel[panel["date"] <= day].sort_values(["symbol", "date"])
    std20 = sub.groupby("symbol")["close"].rolling(20).std().reset_index(level=0, drop=True)
    tmp = sub[["symbol", "date"]].copy()
    tmp["std20"] = std20.values
    tmp = tmp.dropna(subset=["std20"]).groupby("symbol").tail(1).set_index("symbol")["std20"]
    g["std20"] = g["symbol"].map(tmp)
    g = g.dropna(subset=["std20"])
    g["log_mkt"] = np.log1p(g["amount"] * 100.0 / g["turn"] / 1e8)
    g["score"] = -(g["turn"].rank(pct=True) + g["std20"].rank(pct=True)
                   + g["log_mkt"].rank(pct=True))
    return g.set_index("code6")["score"]


def buffer_selections(panel: pd.DataFrame, rl: list[pd.Timestamp]) -> dict[pd.Timestamp, set]:
    """定型 buffer 选券 (从面板首个可用换仓日空仓起步)。"""
    held: set = set()
    sel: dict = {}
    for T in rl:
        row = score_at(panel, T)
        if len(row) < 50:
            sel[T] = set(held)
            continue
        ranked = row.sort_values(ascending=False)
        top5 = ranked.head(K).index.tolist()
        keepzone = ranked.head(KEEP_ZONE).index.tolist()
        new = [c for c in held if c in keepzone]
        for c in top5:
            if len(new) >= K:
                break
            if c not in new:
                new.append(c)
        sel[T] = set(new)
        held = set(new)
    return sel


def schedule_ends(dates: list[pd.Timestamp], freq: str) -> list[pd.Timestamp]:
    """每期最后交易日: M=月末, 2W=半月末 (与 iter27 回测 period_ends 同口径)。"""
    groups: dict = {}
    for d in dates:
        if freq == "M":
            key = (d.year, d.month)
        elif freq == "2W":
            key = (d.year, d.month, 0 if d.day <= 15 else 1)
        else:
            raise ValueError(freq)
        groups.setdefault(key, []).append(d)
    return sorted(max(v) for v in groups.values())


def is_true_period_end(T: pd.Timestamp, freq: str) -> bool:
    """Baostock 真交易日历确认: (T, 期末] 内无交易日 → T 是真期末。

    防面板/缓存截断造成的伪期末 (09-02 事故: 面板止于 09-01 被误判为期末)。
    """
    import baostock as bs
    if freq == "2W":
        half_end = (pd.Timestamp(year=T.year, month=T.month, day=15) if T.day <= 15
                    else T + pd.offsets.MonthEnd(0))
    else:
        half_end = T + pd.offsets.MonthEnd(0)
    lg = bs.login()
    if lg.error_code != "0":
        raise RuntimeError(f"baostock login 失败: {lg.error_code}")
    try:
        rs = bs.query_trade_dates(
            start_date=(T + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
            end_date=half_end.strftime("%Y-%m-%d"))
        while rs.error_code == "0" and rs.next():
            d = dict(zip(rs.fields, rs.get_row_data()))
            if d.get("is_trading_day") == "1":
                return False
        if rs.error_code != "0":
            raise RuntimeError(f"交易日历查询失败: {rs.error_code} {rs.error_msg}")
    finally:
        try:
            bs.logout()
        except Exception:  # noqa: BLE001
            pass
    return True


def main() -> int:
    _force_utf8()
    df_cache, cache_date = read_full_market_cache()
    if df_cache is None:
        print("无全市场缓存, 先跑 scripts/refresh_market_cache.py")
        return 1
    T = pd.Timestamp(cache_date)
    if T != pd.Timestamp.today().normalize():
        print(f"cache 日期 {T.date()} 非今日 — 先刷新缓存再注册 (拒绝用陈旧截面)。")
        return 1

    panel = load_panel()
    dates = sorted(panel["date"].unique())
    if dates[-1] != T:
        print(f"面板最新 {dates[-1].date()} != 注册日 {T.date()} — 先刷新面板 (须含当日收盘)。")
        return 1
    # 注册日必须是真换仓日 (半月末/月末, 交易日历确认)
    if not is_true_period_end(T, "2W"):
        print(f"今天 {T.date()} 不是真半月末换仓日 (交易日历确认)。拒绝注册 (不伪造时点)。")
        return 1

    rl_bw = schedule_ends(dates, "2W")
    rl_m = schedule_ends(dates, "M")
    sel_bw = buffer_selections(panel, rl_bw)
    sel_m = buffer_selections(panel, rl_m)
    basket_bw = sorted(sel_bw[T])
    last_me = max(d for d in rl_m if d <= T)
    basket_m = sorted(sel_m[last_me])
    print(f"注册日 {T.date()} | 双周臂选券 {basket_bw} | 月频臂持仓 (= {last_me.date()} 选券) {basket_m}")

    # cache 截面 (当日 turn/amount/收盘价 — 15:00 后为真实收盘)
    liq = _filter_liquid(df_cache)
    liq["code"] = liq["code"].astype(str)
    universe = liq.nlargest(UNIVERSE_N, "amount")

    def px_map(codes: list[str]) -> dict[str, float]:
        sub = liq[liq["code"].isin(codes)]
        m = _price_map(sub)
        missing = [c for c in codes if c not in m]
        if missing:
            raise RuntimeError(f"cache 缺价格: {missing} (零模拟, 不兜底)")
        return m

    px_bw, px_m = px_map(basket_bw), px_map(basket_m)
    sh_close = fetch_index("sh000001")
    base = {
        "entry_date": str(T.date()),
        "universe_symbols": [str(c) for c in universe["code"].tolist()],
        "universe_prices": _price_map(universe),
        "sh_index_close": sh_close,
    }
    common = {
        "edge_name": None, "source": "live_panel (baostock) + full_market_cache (收盘后) "
                                      "+ qt.gtimg.cn 上证; 规则=iter27 基础 buffer (无解禁叠层)",
        "total_return": False,
        "construction": {
            "universe_filter": "非ST / 主板(60/00) / turn>0 / amount>0 / pe>0 / pb>0",
            "universe": f"top-{UNIVERSE_N} by amount (主板)",
            "selection": "top-5 by score=-(turn_rank+std20_rank+log_mkt_rank), buffer keep-zone=10",
            "weighting": "equalweight",
            "rebalance": None,
            "cost_bps": 65.0,
        },
        "criterion": CRITERION,
    }

    bet_bw = {**common, "edge_id": "cold_lowvol_freq_biweekly",
              "edge_name": "双周频臂 (iter27 PASS 的 OOS 确证, 60td 后与月频臂比 alpha)",
              "entry": {**base, "basket_symbols": basket_bw, "basket_prices": px_bw}}
    bet_bw["construction"]["rebalance"] = "双周 (注册日 = 半月末换仓后持仓, 冻结至 horizon)"
    bet_m = {**common, "edge_id": "cold_lowvol_freq_monthly",
             "edge_name": "月频臂 (iter27 对照, 60td 后与双周臂比 alpha)",
             "entry": {**base, "basket_symbols": basket_m, "basket_prices": px_m}}
    bet_m["construction"]["rebalance"] = f"月频 (持 {last_me.date()} 月末选券不动, 冻结至 horizon)"

    register_bet(bet_bw)
    register_bet(bet_m)
    print(f"已注册平行前瞻 bet: cold_lowvol_freq_biweekly / cold_lowvol_freq_monthly "
          f"(entry {T.date()}, 上证 {sh_close:.2f})")
    print("  之后用 scripts/forward_track.py 跟踪; 60td 后比较两臂 selection_alpha_pp。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
