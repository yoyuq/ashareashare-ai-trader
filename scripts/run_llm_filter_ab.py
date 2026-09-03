"""② LLM 定性过滤接入冷落池 — 预注册 A/B/C (见 preregistration.md §七).

三臂 (同窗口同成本同基准):
  A 原始         = 冷落低波精选 (score=−低换手rank−低波rank−小市值rank, top5 月再平衡)
  B 确定性过滤   = A 的候选里机械剔除「最新 netProfit<0 或 peTTM<0 或 roeAvg<0」(无LLM, 无前视)
  C LLM 过滤     = 每再平衡日取 A 的 top-20 候选, 喂 deepseek-v4-flash 按 rubric keep/drop,
                    从 keep 集按 score 取 top5 (前视风险已披露, 见 §七)

判据(预注册, 跑完不调): C 相对 A 有价值 ⇔ 主判据①不破(非纯牛8窗≥7/8) 且
  (11窗均 vs_universe 提升≥+1.0pp 或 主判据②翻PASS)。否则=无增量价值。

用法: python scripts/run_llm_filter_ab.py
"""
from __future__ import annotations

import asyncio
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import strategy_research_harness as H  # noqa: E402

NAME_MAP_PATH = ROOT / "simulation_data" / "stock_name_map.json"
OBJECTIVE = {
    "2015牛转股灾": "熊/崩", "2016熔断震荡": "震荡", "2017漂亮50": "震荡",
    "2018熊": "熊/崩", "2019牛": "纯牛", "2020牛转崩": "纯牛",
    "2021白马转小盘": "震荡", "2022熊": "熊/崩", "2023震荡": "震荡",
    "2024震荡": "纯牛", "2025-26现期": "震荡",
}
LLM_POOL = 20  # 每个再平衡日取 top-20 候选喂 LLM

SYSTEM = (
    "你是A股基本面定性过滤器。你收到一批「冷落池」候选股(低换手+低波动+小市值)在某历史时点的快照。"
    "只依据给定快照判断, 不得使用你对这些公司未来走势的知识。"
    "逐只判断是否属于「冷而烂」应剔除, 输出严格 JSON: {\"<symbol>\": \"keep\"|\"drop\", ...}。"
    "无法判断的保守 keep(不误杀)。"
)

RUBRIC = (
    "drop 条件(任一命中即 drop): "
    "D1 亏损且恶化(净利负 且 净利同比<0); "
    "D2 ROE<0; "
    "D3 营收同比 < -20%; "
    "D4 TTM亏损(peTTM<0); "
    "D5 名称/行业暗示壳或空心(壳/投资/控股/资源且无主业) 或 夕阳强周期(煤炭/钢铁/玻璃/水泥 且 ROE 持续走低)。"
)


def _force_utf8():
    for name in ("stdout", "stderr"):
        s = getattr(sys, name, None)
        if hasattr(s, "reconfigure"):
            try:
                s.reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass


# --------------------------------------------------------------------------- #
# 因子 (与 run_own_strategies.py 一致)
# --------------------------------------------------------------------------- #
def _factors(df):
    out = df[["date", "symbol", "turn", "amount", "close", "peTTM", "pbMRQ"]].copy()
    s = df.groupby("symbol")["close"].rolling(20).std().reset_index(level=0, drop=True)
    out["std20"] = s.reindex(df.index)
    fmkt = df["amount"] * 100.0 / df["turn"].replace(0, np.nan) / 1e8
    out["log_mkt"] = np.log1p(fmkt)
    return out


def _cold_lowvol(df):
    f = _factors(df)
    r_turn = f["turn"].groupby(df["date"]).rank(pct=True)
    r_std = f["std20"].groupby(df["date"]).rank(pct=True)
    r_mkt = f["log_mkt"].groupby(df["date"]).rank(pct=True)
    return -(r_turn + r_std + r_mkt)


# --------------------------------------------------------------------------- #
# 基本面 point-in-time (含营收同比)
# --------------------------------------------------------------------------- #
def _fund_pt(df, fund):
    """返回与 df 对齐的 pt 基本面 DataFrame: roeAvg / yoyni / netProfit / revYoY."""
    if fund.empty:
        return pd.DataFrame(index=df.index, columns=["roeAvg", "yoyni", "netProfit", "revYoY"])
    f = fund.copy()
    f["symbol"] = f["symbol"].astype(str)
    f = f.sort_values(["symbol", "statDate"])
    # 营收同比: 同季度一年前 (statDate + 1年 匹配)
    prev = f[["symbol", "statDate", "MBRevenue"]].copy()
    prev["statDate"] = prev["statDate"] + pd.DateOffset(years=1)
    m = f.merge(prev, on=["symbol", "statDate"], suffixes=("", "_prev"), how="left")
    f["revYoY"] = m["MBRevenue"] / m["MBRevenue_prev"] - 1.0
    cols = {"roeAvg": "roeAvg", "yoyni": "YOYNI", "netProfit": "netProfit", "revYoY": "revYoY"}
    out = pd.DataFrame(index=df.index)
    for dst, src in cols.items():
        out[dst] = H.pt_fundamental_col(df, f, src)
    return out


# --------------------------------------------------------------------------- #
# LLM 分类
# --------------------------------------------------------------------------- #
def _snapshot_lines(df, fund_pt, name_map, symbols, T):
    """把 date==T 的候选行格式化成 LLM 输入行。"""
    sub = df[df["date"] == T].set_index("symbol")
    fp = fund_pt.loc[df["date"] == T].set_index(df[df["date"] == T]["symbol"])
    lines = []
    for sym in symbols:
        r = sub.loc[sym]
        nm = name_map.get(sym, {}).get("name", "?")
        ind = name_map.get(sym, {}).get("industry", "")
        frow = fp.loc[sym] if sym in fp.index else pd.Series(dtype=float)
        pe = r["peTTM"] if r["peTTM"] == r["peTTM"] else float("nan")
        pb = r["pbMRQ"] if r["pbMRQ"] == r["pbMRQ"] else float("nan")
        roe = frow.get("roeAvg", float("nan"))
        yoy = frow.get("yoyni", float("nan"))
        rev = frow.get("revYoY", float("nan"))
        np_ = frow.get("netProfit", float("nan"))
        turn = r["turn"] if r["turn"] == r["turn"] else float("nan")
        std = r["std20"] if r["std20"] == r["std20"] else float("nan")
        def g(x):
            return "?" if x != x else f"{x:.1f}"
        np_e = (np_ / 1e8) if np_ == np_ else np_
        lines.append(
            f"{sym} | {nm} | 行业:{ind or '?'} | pe:{g(pe)} | pb:{g(pb)} | ROE:{g(roe)}% | "
            f"净利同比:{g(yoy)}% | 营收同比:{g(rev)}% | 净利(亿):{g(np_e)} | 换手:{g(turn)}% | 20日波:{g(std)}"
        )
    return lines


async def _llm_classify(router, lines):
    user = "候选快照(每行一只):\n" + "\n".join(lines) + "\n\n" + RUBRIC + "\n只输出一个 JSON 对象, 不要任何解释或前言。"
    try:
        res = await router.route(
            messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
            task_type="llm_cold_filter",
            temperature=0.0,
            max_tokens=4000,
            extra_body={"thinking": {"type": "disabled"}},
        )
        txt = res.response
    except Exception as e:
        print(f"    [LLM调用失败] {e}", flush=True)
        return None
    # 解析 JSON: 先找所有单层 {...} 块, 取第一个能 parse 成 keep/drop dict 的 (抗前言污染)
    for m in re.finditer(r"\{[^{}]*\}", txt):
        try:
            obj = json.loads(m.group(0))
        except ValueError:
            continue
        if isinstance(obj, dict) and any(v in ("keep", "drop") for v in obj.values()):
            return {k: v for k, v in obj.items() if v in ("keep", "drop")}
    # 兜底: 更大范围匹配
    m = re.search(r"\{.*\}", txt, re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict):
                return {k: v for k, v in obj.items() if v in ("keep", "drop")}
        except ValueError:
            pass
    print(f"    [JSON解析失败] {txt[:150]!r}", flush=True)
    return None


def _apply_llm_mask(score, df, kept_symbols, T):
    """把 date==T 但不在 keep 集的 symbol 的 score 置 NaN。"""
    mask = (df["date"] == T) & (~df["symbol"].isin(kept_symbols))
    score = score.copy()
    score[mask.values] = np.nan
    return score


def _apply_deterministic_mask(score, df, fund_pt):
    """B: 机械剔除 最新 netProfit<0 或 peTTM<0 或 roeAvg<0 (未知=保留)。"""
    loss = (
        (df["peTTM"] < 0)
        | (fund_pt["netProfit"] < 0)
        | (fund_pt["roeAvg"] < 0)
    )
    score = score.copy()
    score[loss.values] = np.nan
    return score


async def _run_arm_c(router, df, fund_pt, name_map, score, freq="monthly"):
    """C: 每再平衡日 top-20 候选 → LLM keep/drop → 从 keep 集取 top5。"""
    dates = H.pick_rebalance_dates(df["date"].tolist(), freq)
    score = score.copy()
    n_calls = 0
    for T in dates:
        day = df[df["date"] == T]
        s = score[day.index].dropna().nlargest(LLM_POOL)
        if s.empty:
            continue
        syms = df.loc[s.index, "symbol"].tolist()
        lines = _snapshot_lines(df, fund_pt, name_map, syms, T)
        keep = await _llm_classify(router, lines)
        n_calls += 1
        if keep is None:
            continue  # 解析失败 → 保守全保留 (不 mask)
        kept = [k for k in syms if keep.get(k) != "drop"]
        dropped = set(syms) - set(kept)
        if dropped:
            mask = (df["date"] == T) & (df["symbol"].isin(dropped))
            score[mask.values] = np.nan
    return score, n_calls


async def main_async():
    _force_utf8()
    name_map = json.load(open(NAME_MAP_PATH, encoding="utf-8")) if NAME_MAP_PATH.exists() else {}
    fund = H.load_fundamentals()
    idx = H.load_index()

    from models.router import get_shared_router
    router = get_shared_router()

    results = {"A": {}, "B": {}, "C": {}, "meta": {"llm_pool": LLM_POOL, "rubric": RUBRIC}}
    t0 = time.time()
    for window, fp in H.WINDOWS.items():
        df = H.load_base_window(fp)
        if df.empty:
            continue
        fac = _factors(df)
        df = df.copy()
        df["std20"] = fac["std20"].values
        score = _cold_lowvol(df)
        fund_pt = _fund_pt(df, fund)

        # A 原始
        m_a = H.run_window_ranking(df, idx, _cold_lowvol, 5, "monthly", window)
        results["A"][window] = m_a

        # B 确定性过滤
        score_b = _apply_deterministic_mask(score, df, fund_pt)
        m_b = H.run_window_ranking(df, idx, lambda d: score_b, 5, "monthly", window)
        results["B"][window] = m_b

        # C LLM 过滤
        score_c, n_calls = await _run_arm_c(router, df, fund_pt, name_map, score)
        m_c = H.run_window_ranking(df, idx, lambda d: score_c, 5, "monthly", window)
        m_c["llm_calls"] = n_calls
        results["C"][window] = m_c

        print(f"  [{window}] A={m_a['total']*100:+.1f}%({m_a['vs_universe']*100:+.1f}pp) "
              f"B={m_b['total']*100:+.1f}%({m_b['vs_universe']*100:+.1f}pp) "
              f"C={m_c['total']*100:+.1f}%({m_c['vs_universe']*100:+.1f}pp) "
              f"llm_calls={n_calls} {time.time()-t0:.0f}s", flush=True)

    H.dump_json(results, ROOT / "reports" / "llm_filter_ab_result.json")
    _print_verdict(results)
    return 0


def _print_verdict(results):
    print("\n" + "=" * 90)
    print("② LLM 定性过滤 A/B/C — 预注册判据 verdict")
    print("=" * 90)
    for arm in ("A", "B", "C"):
        ws = results[arm]
        act = {w: m for w, m in ws.items() if "vs_universe" in m and m["vs_universe"] == m["vs_universe"]}
        nonniu = {w: m for w, m in act.items() if OBJECTIVE[w] != "纯牛"}
        niu = {w: m for w, m in act.items() if OBJECTIVE[w] == "纯牛"}
        c1_win = sum(1 for m in nonniu.values() if m["vs_universe"] >= 0)
        c1 = c1_win >= 7
        c2 = all(m["total"] > 0 and m["maxdd"] >= -0.30 for m in niu.values())
        mean_vu = float(np.mean([m["vs_universe"] for m in act.values()]))
        mean_total = float(np.mean([m["total"] for m in act.values()]))
        print(f"\n[{arm}] 均vs_univ={mean_vu*100:+.2f}pp  均total={mean_total*100:+.1f}%  "
              f"主判据①非纯牛赢={c1_win}/8→{'PASS' if c1 else 'FAIL'}  "
              f"主判据②={c2 and 'PASS' or 'FAIL'}")
        for w in H.WINDOWS:
            m = ws.get(w, {})
            if "vs_universe" not in m or m["vs_universe"] != m["vs_universe"]:
                continue
            print(f"    {w:<14} {OBJECTIVE[w]:<4} total={m['total']*100:+7.1f}% "
                  f"vs_univ={m['vs_universe']*100:+7.2f}pp maxdd={m['maxdd']*100:+6.1f}%")

    # 预注册判据: C vs A
    a = results["A"]; c = results["C"]
    a_act = {w: m for w, m in a.items() if "vs_universe" in m and m["vs_universe"] == m["vs_universe"]}
    c_act = {w: m for w, m in c.items() if "vs_universe" in m and m["vs_universe"] == m["vs_universe"]}
    c_nn = {w: m for w, m in c_act.items() if OBJECTIVE[w] != "纯牛"}
    c_niu = {w: m for w, m in c_act.items() if OBJECTIVE[w] == "纯牛"}
    c1 = sum(1 for m in c_nn.values() if m["vs_universe"] >= 0) >= 7
    c2 = all(m["total"] > 0 and m["maxdd"] >= -0.30 for m in c_niu.values())
    dv = float(np.mean([c_act[w]["vs_universe"] for w in c_act])) - float(np.mean([a_act[w]["vs_universe"] for w in a_act]))
    added = c1 and (dv >= 0.01 or c2)
    print("\n--- 预注册判据 (C vs A) ---")
    print(f"  C 主判据① 不破: {'是' if c1 else '否'}")
    print(f"  C 均 vs_universe 提升: {dv*100:+.2f}pp (门槛 +1.0pp)")
    print(f"  C 主判据② 翻 PASS: {'是' if c2 else '否'}")
    print(f"  → 结论: LLM 过滤 {'有价值 (预注册判据达成)' if added else '无增量价值 (维持原始)'}")


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main_async()))
