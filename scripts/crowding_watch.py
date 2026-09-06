"""拥挤度预警基线 (S5) — 测量层只读, 绝不改规则 (预注册纪律).

用途: 前瞻期持续监测定型 arm (冷落低波精选 top5+解禁叠层) 的三类失效前兆:
  1. 相对弱势: 篮子 vs universe (主板 top-800 by amount) 滚动 20td 净差 + 连续跑输天数
  2. 注意力拥挤: 篮子换手率在 universe 内的截面分位 (冷落溢价本体=低注意, 分位抬升=溢
     价来源衰减)
  3. 规则相关性: 用最新面板点内重建冻结选券, 与当前 bet 持仓的重合度 (规则漂移/冻结篮
     子老化信号)

数据: replay_data/live_panel.parquet (AITraderPanelRefresh 每日维护) + 解禁时间表 +
forward_validation/registry.json 当前 bet 持仓。全程点内, 无未来函数, 零模拟 (缺数据
报错不兜底)。

输出:
  reports/agent_loop/crowding_watch.json                  (最新快照)
  simulation_data/forward_validation/crowding_watch.jsonl (追加历史)

**任何 warn 只进报告, 不触发任何规则变更/调参/换线** (触发器≠处置权)。
用法: python scripts/crowding_watch.py
"""
from __future__ import annotations

import json
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

PANEL_FP = ROOT / "replay_data" / "live_panel.parquet"
REGISTRY_FP = ROOT / "simulation_data" / "forward_validation" / "registry.json"
SNAPSHOT_FP = ROOT / "reports" / "agent_loop" / "crowding_watch.json"
HISTORY_FP = ROOT / "simulation_data" / "forward_validation" / "crowding_watch.jsonl"

LOOKBACK_TD = 120     # 计算窗
MIN_TD = 30           # 最低历史 (20td 滚动窗 + 10td 缓冲)
ROLL_TD = 20          # 滚动相对弱势窗
UNIVERSE_N = 800      # 冻结 universe: 主板 top-800 by amount
BET_IDS = ("cold_lowvol_top5_hold", "cold_lowvol_top5_unlock_screen")

# warn 阈值 (只报告): 失效阈值参考 -10pp 来自预注册判据, 此处 -5pp 提前示警
WARN_DIFF_PP = -5.0
WARN_STREAK_DAYS = 10
WARN_TURN_PCTILE = 0.90


def _force_utf8() -> None:
    for name in ("stdout", "stderr"):
        s = getattr(sys, name, None)
        if hasattr(s, "reconfigure"):
            try:
                s.reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass


def _panel_universe(df: pd.DataFrame, date) -> pd.DataFrame:
    """冻结 universe 口径 (主板 60/00, 非ST, 正常交易, turn/amount/pe/pb>0), top-800 by amount."""
    d = df[df["date"] == date].copy()
    d = d[d["symbol"].str.startswith(("sh.6", "sz.0"))]
    d = d[(d["isST"].astype(str) == "0") & (d["tradestatus"].astype(str) == "1")]
    for c in ("turn", "amount", "close", "peTTM", "pbMRQ"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d[(d["turn"] > 0) & (d["amount"] > 0) & (d["peTTM"] > 0) & (d["pbMRQ"] > 0)]
    return d.nlargest(UNIVERSE_N, "amount")


def _fresh_selection(df: pd.DataFrame, date, blocked: set) -> tuple[pd.DataFrame, int]:
    """点内重建冻结选券: score = -(turn_rank+std20_rank+log_mkt_rank), 解禁叠层剔除.

    std20 由面板逐票 pctChg 滚动 20td 计算 (与回测 harness 同口径, 非外部 std20 文件)。
    返回 (带 score 的候选表, 解禁剔除数)。
    """
    uni = _panel_universe(df, date)
    hist = df[df["date"] <= date].copy()
    hist["pctChg"] = pd.to_numeric(hist["pctChg"], errors="coerce")
    std20 = hist.groupby("symbol")["pctChg"].rolling(20, min_periods=20).std().groupby("symbol").last()
    uni["std20"] = uni["symbol"].map(std20)
    uni = uni[uni["std20"].notna()].copy()
    n_before = len(uni)
    uni = uni[~uni["symbol"].isin(blocked)].copy()
    r_turn = uni["turn"].rank(pct=True)
    r_std = uni["std20"].rank(pct=True)
    r_mkt = np.log1p(uni["amount"] * 100.0 / uni["turn"] / 1e8).rank(pct=True)
    uni["score"] = -(r_turn + r_std + r_mkt)
    return uni, n_before - len(uni)


def _load_blocked(latest) -> set:
    """解禁叠层: 未来60自然日解禁≥5% 的票 (与 forward_register_cold_lowvol_overlay 同规则)."""
    from run_restricted_screen import load_events
    ev = load_events()
    big = ev[ev["ratio"] >= 0.05]
    t0 = pd.Timestamp(latest)
    syms = set(big[(big["free_date"] >= t0) & (big["free_date"] <= t0 + pd.Timedelta(days=60))]["sym"])
    return syms


def _bet_holdings() -> dict:
    """registry 中最新 cold_lowvol bet 的冻结持仓 (code 6位列表 + 入场日)."""
    if not REGISTRY_FP.exists():
        raise RuntimeError(f"registry.json 缺失: {REGISTRY_FP} (零模拟, 不兜底)")
    reg = json.loads(REGISTRY_FP.read_text(encoding="utf-8"))
    raw = reg.get("bets", {}) if isinstance(reg, dict) else {}
    bets = [b for b in (raw.values() if isinstance(raw, dict) else raw)
            if isinstance(b, dict) and b.get("edge_id") in BET_IDS]
    if not bets:
        raise RuntimeError(f"registry 无 {BET_IDS} bet (零模拟, 不兜底)")
    b = bets[-1]
    return {"edge_id": b["edge_id"], "entry_date": b["entry_date"],
            "basket": [str(c) for c in b["entry"]["basket_symbols"]]}


def main() -> int:
    _force_utf8()
    if not PANEL_FP.exists():
        raise RuntimeError(f"live_panel 缺失: {PANEL_FP} (零模拟, 不兜底)")
    df = pd.read_parquet(PANEL_FP)
    df["date"] = pd.to_datetime(df["date"])
    dates = sorted(df["date"].unique())
    if len(dates) < MIN_TD:
        raise RuntimeError(f"面板历史不足: {len(dates)} td < {MIN_TD}")
    dates = dates[-LOOKBACK_TD:]
    df = df[df["date"].isin(dates)]

    bet = _bet_holdings()
    basket_syms = {("sh." if c.startswith("6") else "sz.") + c for c in bet["basket"]}
    missing = basket_syms - set(df["symbol"].unique())
    if missing:
        raise RuntimeError(f"bet 持仓不在面板: {sorted(missing)} (零模拟, 不兜底)")
    # 测量窗从篮子票数据完整日起 (补票历史晚于面板起点, 非缺口)
    b_first = df[df["symbol"].isin(basket_syms)].groupby("symbol")["date"].min().max()
    dates = [d for d in dates if d >= b_first]
    if len(dates) < MIN_TD:
        raise RuntimeError(f"篮子数据窗不足: {len(dates)} td < {MIN_TD}")

    blocked = _load_blocked(pd.Timestamp(dates[-1]))

    # ---- 日度序列: 篮子/universe 等权日收益 (pctChg), 换手分位 ----
    # 注意: 冷落票本就不在 top-800 by amount 内 (策略本体), 篮子收益直接取面板,
    # 不要求篮子 ∈ universe; 换手分位 = universe 中换手低于篮子票的比例。
    basket_px = df[df["symbol"].isin(basket_syms)]
    rows = []
    for d in dates:
        uni = _panel_universe(df, d)
        b = basket_px[basket_px["date"] == d]
        if len(b) < len(basket_syms):
            raise RuntimeError(f"{d.date()}: 篮子 {len(b)}/{len(basket_syms)} 只缺行情 (零模拟)")
        b_pct = pd.to_numeric(b["pctChg"], errors="coerce")
        if b_pct.notna().sum() == 0:
            raise RuntimeError(f"{d.date()}: 篮子 pctChg 全空 (零模拟)")
        u_turn = pd.to_numeric(uni["turn"], errors="coerce").dropna()
        b_turn = pd.to_numeric(b["turn"], errors="coerce").dropna()
        pctiles = [float((u_turn < t).mean()) for t in b_turn] if len(u_turn) else []
        rows.append({
            "date": d,
            "basket_ret": float(b_pct.mean()) / 100.0,
            "univ_ret": float(pd.to_numeric(uni["pctChg"], errors="coerce").mean()) / 100.0,
            "basket_turn_pctile": float(np.mean(pctiles)) if pctiles else float("nan"),
            "basket_med_amount_rank": float("nan"),
        })
    ser = pd.DataFrame(rows).set_index("date")
    ser["rel"] = ser["basket_ret"] - ser["univ_ret"]

    nav_b, nav_u = (1 + ser["basket_ret"]).cumprod(), (1 + ser["univ_ret"]).cumprod()
    rel20 = float(((nav_b / nav_b.shift(ROLL_TD)) - (nav_u / nav_u.shift(ROLL_TD))).iloc[-1] * 100)
    streak = int(0)
    for v in ser["rel"].iloc[::-1]:
        if v < 0:
            streak += 1
        else:
            break
    turn_pctile = float(ser["basket_turn_pctile"].tail(5).mean())
    total_diff = float(nav_b.iloc[-1] / nav_u.iloc[-1] - 1) * 100

    # ---- 规则相关性: 冻结选券点内重建 vs 当前持仓 ----
    latest = dates[-1]
    uni_f, n_blocked = _fresh_selection(df, latest, blocked)
    top5 = uni_f.nlargest(5, "score")
    fresh_codes = {s[3:] for s in top5["symbol"]}
    overlap = sorted(fresh_codes & {c for c in bet["basket"]})

    warns = []
    if rel20 <= WARN_DIFF_PP:
        warns.append(f"20td 相对弱势 {rel20:+.2f}pp <= {WARN_DIFF_PP}pp")
    if streak >= WARN_STREAK_DAYS:
        warns.append(f"连续跑输 {streak}td >= {WARN_STREAK_DAYS}")
    if turn_pctile >= WARN_TURN_PCTILE:
        warns.append(f"篮子换手分位 {turn_pctile:.2f} >= {WARN_TURN_PCTILE} (冷落属性衰减)")

    # ---- 第三信号: 关注度 (C2, attention_collector 产物; 缺文件只降级不报错) ----
    attention = {"status": "unavailable (先跑 scripts/attention_collector.py)"}
    hotrank_fp = ROOT / "replay_data" / "attention_live" / "hot_rank.parquet"
    comment_fp = ROOT / "replay_data" / "attention_live" / "comment_snapshot.parquet"
    if hotrank_fp.exists() and comment_fp.exists():
        try:
            hr = pd.read_parquet(hotrank_fp)
            hr["code"] = hr["code"].astype(str).str.zfill(6)
            hr_b = hr[hr["code"].isin({c for c in bet["basket"]})]
            last5 = hr_b[hr_b["date"] >= hr_b["date"].max() - pd.Timedelta(days=7)]
            b_rank = float(last5["rank"].median()) if len(last5) else None
            cm = pd.read_parquet(comment_fp)
            cm["code"] = cm["code"].astype(str).str.zfill(6)
            cm_last = cm[cm["date"] == cm["date"].max()]
            m_rank = float(pd.to_numeric(cm_last["目前排名"], errors="coerce").median())
            n_all = int(len(cm_last))
            pctile = round(b_rank / n_all, 3) if (b_rank is not None and n_all) else None
            attention = {"basket_median_popularity_rank_5d": b_rank,
                         "market_median_rank_snapshot": m_rank,
                         "market_n": n_all,
                         "basket_rank_percentile": pctile,
                         "note": "人气排名分位高=冷落保持; 分位快速下移=注意力涌入预警"}
            if pctile is not None and pctile < 0.30:
                warns.append(f"篮子人气分位 {pctile:.2f} < 0.30 (冷落属性衰减, 注意力涌入)")
        except Exception as e:  # noqa: BLE001
            attention = {"status": f"degraded: {type(e).__name__}: {str(e)[:120]}"}

    snap = {
        "as_of": str(pd.Timestamp(latest).date()),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "bet": {"edge_id": bet["edge_id"], "entry_date": bet["entry_date"], "basket": bet["basket"]},
        "relative_strength": {
            "total_diff_pp": round(total_diff, 2),
            "rolling20td_diff_pp": round(rel20, 2),
            "underperf_streak_td": streak,
            "window_td": len(ser),
        },
        "attention": {
            "basket_turn_percentile_5d": round(turn_pctile, 3),
            "popularity": attention,
        },
        "rule_relevance": {
            "fresh_top5": sorted(fresh_codes),
            "overlap_with_basket": overlap,
            "fresh_selection_n": int(len(uni_f)),
            "unlock_blocked_n": int(n_blocked),
            "note": "冻结篮子 vs 点内重建选券; 重合度低=篮子老化/池内排序漂移, 只测不改",
        },
        "warns": warns,
        "disclaimer": "测量层只读: warn 仅报告, 不触发规则变更/调参/换线 (预注册纪律)",
    }
    SNAPSHOT_FP.parent.mkdir(parents=True, exist_ok=True)
    SNAPSHOT_FP.write_text(json.dumps(snap, ensure_ascii=False, indent=2), encoding="utf-8")
    with HISTORY_FP.open("a", encoding="utf-8") as f:
        f.write(json.dumps(snap, ensure_ascii=False) + "\n")

    print(f"拥挤度监测 @{snap['as_of']} (窗 {len(ser)}td, bet={bet['edge_id']})")
    print(f"  相对: 总差{total_diff:+.2f}pp | 20td差{rel20:+.2f}pp | 连续跑输{streak}td")
    print(f"  注意力: 换手分位{turn_pctile:.2f}")
    print(f"  相关性: 点内重建top5={sorted(fresh_codes)} 与持仓重合{len(overlap)}/5")
    if warns:
        print("  ⚠ WARN (仅报告): " + " | ".join(warns))
    else:
        print("  ✓ 无预警")
    print(f"  快照 -> {SNAPSHOT_FP.name}, 历史 -> {HISTORY_FP.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
