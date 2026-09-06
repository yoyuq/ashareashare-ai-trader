"""C2 数据门探测 — 关注度数据 (拥挤度测量层第三信号) 历史可得性.

目的: PHASE2 队列 C2 — 探测 akshare 关注度接口的 (a) 可用性 (b) 列语义
(c) 历史深度 (hot_rank_detail_em 人气历史最深多久) (d) 限频表现。
用途仅 crowding_watch 测量层 (篮子关注度分位哨兵), 不做选券因子。
输出: reports/agent_loop/probe_attention_result.json
用法: python scripts/probe_attention_data.py [--proxy http://127.0.0.1:7897]
"""
from __future__ import annotations

import argparse
import inspect
import json
import os
import sys
import warnings
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutTimeout
from datetime import datetime
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent.parent
OUT_FP = ROOT / "reports" / "agent_loop" / "probe_attention_result.json"
CALL_TIMEOUT = 90


def _force_utf8() -> None:
    for name in ("stdout", "stderr"):
        s = getattr(sys, name, None)
        if hasattr(s, "reconfigure"):
            try:
                s.reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass


def _try_call(fn, kwargs: dict) -> tuple[str, object]:
    with ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(fn, **kwargs)
        try:
            df = fut.result(timeout=CALL_TIMEOUT)
        except FutTimeout:
            return "timeout", None
        except Exception as e:  # noqa: BLE001
            return f"error: {type(e).__name__}: {str(e)[:200]}", None
    if df is None or len(df) == 0:
        return "empty", None
    return "ok", df


def _describe(df: pd.DataFrame) -> dict:
    out: dict = {"rows": int(len(df)), "columns": [str(c) for c in df.columns],
                 "sample": df.head(3).astype(str).to_dict(orient="records")}
    for c in df.columns:
        cl = str(c).lower()
        if "date" in cl or "日期" in str(c) or "时间" in str(c):
            s = df[c].astype(str)
            out[f"date_range:{c}"] = [str(s.min())[:19], str(s.max())[:19]]
    return out


TARGETS = [
    # (name, kwargs 尝试列表) — 人气历史趋势 / 千股千评快照 / 日度参与意愿 / 百度热搜
    ("stock_hot_rank_detail_em", [{"symbol": "SH600519"}]),
    ("stock_comment_em", [{}]),
    ("stock_comment_detail_scrd_desire_daily_em", [{"symbol": "SH600519"}]),
    ("stock_comment_detail_scrd_focus_em", [{"symbol": "SH600519"}]),
    ("stock_hot_search_baidu", [{"symbol": "A股", "date": "20260904", "time": "今日"}]),
]


def main() -> int:
    _force_utf8()
    ap = argparse.ArgumentParser()
    ap.add_argument("--proxy", type=str, default="")
    args = ap.parse_args()
    if args.proxy:
        os.environ["HTTP_PROXY"] = os.environ["HTTPS_PROXY"] = args.proxy

    import akshare as ak

    results: dict = {"probed_at": datetime.now().isoformat(timespec="seconds"),
                     "call_timeout_s": CALL_TIMEOUT, "targets": {}}
    for name, attempt_list in TARGETS:
        fn = getattr(ak, name, None)
        if fn is None:
            results["targets"][name] = {"status": "absent_in_akshare"}
            print(f"\n== {name}: 接口不存在于当前 akshare 版本")
            continue
        try:
            params = {p: str(v.default) for p, v in inspect.signature(fn).parameters.items()}
        except (ValueError, TypeError):
            params = {}
        entry: dict = {"signature": params, "attempts": []}
        print(f"\n== {name} (params: {params})")
        for kw in attempt_list:
            status, payload = _try_call(fn, kw)
            line = {"kwargs": kw, "status": status}
            if status == "ok":
                line.update(_describe(payload))
            entry["attempts"].append(line)
            print(f"   {kw} -> {status}"
                  + (f" rows={line.get('rows')} ranges={{k: v for k, v in line.items() if k.startswith('date_range')}}"
                     if status == "ok" else ""))
            if status == "ok":
                break
        results["targets"][name] = entry

    ok = [n for n, e in results["targets"].items()
          if any(a.get("status") == "ok" for a in e.get("attempts", []))]
    results["verdict"] = {
        "callable_ok": ok,
        "gate": "PASS→评估历史深度与限频, 写 attention_collector.py (basket+全市场快照日采集); "
                "仅当日快照且无历史→仍可自建前瞻序列 (与 minute_collector 同模式), 但 crowding "
                "基线需等积累",
    }
    print("\n== 结论 ==")
    print(f"可得接口: {ok}")
    OUT_FP.write_text(json.dumps(results, ensure_ascii=False, indent=2, default=str),
                      encoding="utf-8")
    print(f"saved -> {OUT_FP}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
