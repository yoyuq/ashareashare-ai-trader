"""S4 数据门探测 — 减持计划 (shareholder reduction plan) 预披露历史可得性.

目的: iter34 队列继承候选「减持计划规避 screen」在写 prereg/回测前, 先诚实探测
akshare 是否有 (a) 计划预披露类 (非仅已实施变动) (b) 历史深度 ≥2018 (c) 个股可关联
的数据源。数据门不过 → 记录「数据不可得」封存, 不硬造。

流程: 枚举 akshare 候选接口 (ggcg/share_hold/hold_change/reduc 关键词) → 逐个限时
调用 → 报告列/时间范围/行数/字段语义。只探测, 不建库不回测。
输出: reports/agent_loop/probe_reduction_plan_result.json
用法: python scripts/probe_reduction_plan.py [--proxy http://127.0.0.1:7897]
"""
from __future__ import annotations

import argparse
import inspect
import json
import sys
import warnings
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutTimeout
from datetime import datetime
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent.parent
OUT_FP = ROOT / "reports" / "agent_loop" / "probe_reduction_plan_result.json"
KEYWORDS = ("ggcg", "share_hold", "hold_change", "reduc", "jiantao", "detain")
CALL_TIMEOUT = 100  # 秒


def _force_utf8() -> None:
    for name in ("stdout", "stderr"):
        s = getattr(sys, name, None)
        if hasattr(s, "reconfigure"):
            try:
                s.reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass


def _try_call(fn, kwargs: dict) -> tuple[str, object]:
    """限时调用; 返回 (status, payload)。status: ok/empty/timeout/error。"""
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


def _describe(df) -> dict:
    out: dict = {"rows": int(len(df)), "columns": [str(c) for c in df.columns]}
    for c in df.columns:
        cl = str(c)
        if any(k in cl for k in ("日期", "时间", "date", "公告")):
            s = df[c].astype(str)
            out[f"date_range:{cl}"] = [str(s.min())[:19], str(s.max())[:19]]
    out["sample"] = df.head(3).astype(str).to_dict(orient="records")
    return out


def main() -> int:
    _force_utf8()
    ap = argparse.ArgumentParser()
    ap.add_argument("--proxy", type=str, default="")
    args = ap.parse_args()
    if args.proxy:
        import os
        os.environ["HTTP_PROXY"] = os.environ["HTTPS_PROXY"] = args.proxy

    import akshare as ak

    cands = [n for n in dir(ak) if any(k in n.lower() for k in KEYWORDS)
             and callable(getattr(ak, n))]
    print(f"候选接口 {len(cands)} 个: {cands}")

    results: dict = {"probed_at": datetime.now().isoformat(timespec="seconds"),
                     "call_timeout_s": CALL_TIMEOUT, "candidates": {}}
    for name in cands:
        fn = getattr(ak, name)
        try:
            sig = inspect.signature(fn)
            params = {p: v.default for p, v in sig.parameters.items()}
        except (ValueError, TypeError):
            params = {}
        # 逐签名尝试: 无必填→默认; 有 symbol→常见值
        attempts: list[dict] = [{}]
        if "symbol" in params:
            attempts = [{"symbol": s} for s in ("全部", "股东", "高管", "特定股东")]
        entry: dict = {"params": {k: str(v) for k, v in params.items()}, "attempts": []}
        print(f"\n== {name} (params: {entry['params']})")
        for kw in attempts:
            status, payload = _try_call(fn, kw)
            line = {"kwargs": kw, "status": status}
            if status == "ok":
                line.update(_describe(payload))
            entry["attempts"].append(line)
            print(f"   {kw} -> {status}"
                  + (f" rows={line.get('rows')}" if status == "ok" else ""))
            if status == "ok":
                break  # 该接口已验证可得
        results["candidates"][name] = entry

    ok_any = [n for n, e in results["candidates"].items()
              if any(a["status"] == "ok" for a in e["attempts"])]
    results["verdict"] = {
        "callable_ok": ok_any,
        "gate": "PASS→先核计划预披露语义+历史深度再写 prereg; 仅变动类/历史<2018→数据门不过, 封存",
    }
    print("\n== 结论 ==")
    for n in ok_any:
        for a in results["candidates"][n]["attempts"]:
            if a["status"] == "ok":
                rngs = {k: v for k, v in a.items() if k.startswith("date_range")}
                print(f"{n}: rows={a['rows']} {rngs}")
                break
    OUT_FP.write_text(json.dumps(results, ensure_ascii=False, indent=2, default=str),
                      encoding="utf-8")
    print(f"saved -> {OUT_FP}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
