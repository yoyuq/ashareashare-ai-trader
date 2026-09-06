"""关注度采集器 (C2, PHASE2) — 千股千评全市场快照 + 前瞻持仓人气历史 (日度, 自建历史).

用途: crowding_watch 第三信号 (篮子关注度 universe 分位, 冷落属性衰减哨兵) 的数据
基础。仅测量层, 不做选券因子 (低关注度 tilt ≈ 低换手同族; 定性净化有害已证)。

每次调用 (维护 tick 工作日收盘后):
  1. stock_comment_em() 全市场快照 (5196 只: 代码/关注指数/目前排名/综合得分/更新日期)
     → upsert replay_data/attention_live/comment_snapshot.parquet (键: 更新日期+代码, 幂等)。
  2. registry 前瞻持仓逐票 stock_hot_rank_detail_em → 人气排名历史 (≈1年日度)
     → upsert attention_live/hot_rank.parquet (键: 代码+日期, 幂等)。
  3. 无新数据 (节假日/已最新) → no-op。

强制直连: 全局 HTTP_PROXY env 可能指向半死代理 (2026-09-06 事故), 本脚本 pop 后直连
(东财直连已验证 200)。失败策略: 快照全失败 exit 2; 单票失败记日志跳过 (零模拟,
缺口=缺口, 不伪造)。
用法: python scripts/attention_collector.py
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent.parent
OUT_DIR = ROOT / "replay_data" / "attention_live"
REGISTRY_FP = ROOT / "simulation_data" / "forward_validation" / "registry.json"
COMMENT_FP = OUT_DIR / "comment_snapshot.parquet"
HOTRANK_FP = OUT_DIR / "hot_rank.parquet"


def _force_utf8() -> None:
    for name in ("stdout", "stderr"):
        s = getattr(sys, name, None)
        if hasattr(s, "reconfigure"):
            try:
                s.reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass


def _registry_baskets() -> list[str]:
    if not REGISTRY_FP.exists():
        return []
    reg = json.loads(REGISTRY_FP.read_text(encoding="utf-8"))
    raw = reg.get("bets", {})
    codes: set[str] = set()
    for b in (raw.values() if isinstance(raw, dict) else raw):
        if isinstance(b, dict):
            codes.update(str(c) for c in (b.get("entry") or {}).get("basket_symbols", []) or [])
    # 仅 A 股个股 (转债 11x/12x 代码不适用股票人气接口, cb bet 持仓排除)
    return sorted(c for c in codes if c[:2] in ("60", "00", "30", "68", "43", "83", "87", "92"))


def _upsert(new: pd.DataFrame, keys: list[str], fp: Path) -> int:
    old_n = 0
    merged = new
    if fp.exists():
        old = pd.read_parquet(fp)
        old_n = len(old)
        merged = pd.concat([old, new], ignore_index=True)
    merged = merged.drop_duplicates(subset=keys, keep="last")
    merged.to_parquet(fp, index=False)
    return len(merged) - old_n  # 净新增行数


def main() -> int:
    _force_utf8()
    # 半死代理防护: pop 全局代理 env, 强制直连 (东财直连已验证)
    import os
    for k in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
        os.environ.pop(k, None)
    import akshare as ak

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ok_any = False

    # ---- 1. 全市场千股千评快照 ----
    try:
        c = ak.stock_comment_em()
        if c is not None and len(c):
            c = c.rename(columns={str(col): str(col).strip() for col in c.columns})
            date_col = next((col for col in c.columns if "更新日期" in col or "日期" in col), None)
            if date_col is None:
                raise RuntimeError(f"快照无日期列: {list(c.columns)}")
            keep = [x for x in ["代码", "名称", "关注指数", "目前排名", "综合得分", "换手率", date_col]
                    if x in c.columns]
            c = c[keep].copy()
            c[date_col] = pd.to_datetime(c[date_col], errors="coerce")
            c = c.dropna(subset=[date_col])
            c = c.rename(columns={date_col: "date"})
            c["code"] = c["代码"].astype(str).str.zfill(6)
            added = _upsert(c, ["date", "code"], COMMENT_FP)
            last = pd.to_datetime(c["date"]).max().date()
            print(f"[comment] 快照 {len(c)} 行 (数据日 {last}), 新增 {added} 行")
            ok_any = True
        else:
            print("[comment] 空快照 (非交易日?), 跳过")
    except Exception as e:  # noqa: BLE001
        print(f"[comment] FAIL: {type(e).__name__}: {str(e)[:160]}")

    # ---- 2. 前瞻持仓人气历史 (单票失败跳过) ----
    codes = _registry_baskets()
    if codes:
        parts, fail = [], []
        for code in codes:
            sym = ("SH" if code.startswith("6") else "SZ") + code
            try:
                h = ak.stock_hot_rank_detail_em(symbol=sym)
                if h is None or len(h) == 0:
                    fail.append(code)
                    continue
                h = h.rename(columns={str(col): str(col).strip() for col in h.columns})
                tcol = next((col for col in h.columns if "时间" in col or "日期" in col), None)
                if tcol is None:
                    fail.append(code)
                    continue
                h = h.rename(columns={tcol: "date"})
                h["date"] = pd.to_datetime(h["date"], errors="coerce")
                h = h.dropna(subset=["date"])
                h["code"] = code
                rank_col = next((col for col in h.columns if "排名" in col), None)
                if rank_col:
                    h = h.rename(columns={rank_col: "rank"})
                    keep = [x for x in ["date", "code", "rank"]
                            + [col for col in h.columns if "粉丝" in col] if x in h.columns]
                    parts.append(h[keep])
                ok_any = True
                time.sleep(0.4)  # 限频礼貌间隔
            except Exception as e:  # noqa: BLE001
                print(f"[hotrank] {code} FAIL: {type(e).__name__}: {str(e)[:120]}")
                fail.append(code)
        if parts:
            hr = pd.concat(parts, ignore_index=True)
            added = _upsert(hr, ["date", "code"], HOTRANK_FP)
            print(f"[hotrank] {len(codes) - len(fail)}/{len(codes)} 票成功, "
                  f"{len(hr)} 行, 新增 {added} 行")
        if fail:
            print(f"[hotrank] 失败票 (缺口=缺口, 不伪造): {fail}")

    if not ok_any:
        print("全部数据源失败 (exit 2, 不兜底)")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
