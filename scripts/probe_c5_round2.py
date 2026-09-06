"""C5 数据门补探 (round 2): pop 代理直连 + stock_add_stock 实测 + LOF 现价/价格史 + bj 经东财."""
from __future__ import annotations

import json
import os
import socket
import sys
import warnings
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent.parent
OUT_FP = ROOT / "reports" / "agent_loop" / "probe_c5_result.json"


def _force_utf8() -> None:
    for name in ("stdout", "stderr"):
        s = getattr(sys, name, None)
        if hasattr(s, "reconfigure"):
            try:
                s.reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass


def probe_timeout(fn, args, timeout=60):
    with ThreadPoolExecutor(max_workers=1) as ex:
        try:
            return fn(*args)
        except Exception as e:  # noqa: BLE001
            return {"__error__": f"{type(e).__name__}: {str(e)[:150]}"}


def summarize(df):
    if isinstance(df, dict) and "__error__" in df:
        return df
    if df is None or len(df) == 0:
        return {"rows": 0}
    info = {"rows": int(len(df)), "columns": [str(c) for c in df.columns][:22]}
    date_col = next((c for c in df.columns if "日期" in str(c) or "date" in str(c).lower()), None)
    if date_col:
        d = pd.to_datetime(df[date_col], errors="coerce").dropna()
        if len(d):
            info["date_min"], info["date_max"] = str(d.min().date()), str(d.max().date())
    return info


def main() -> int:
    _force_utf8()
    for k in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
        os.environ.pop(k, None)
    socket.setdefaulttimeout(20)
    import akshare as ak

    result = json.loads(OUT_FP.read_text(encoding="utf-8"))
    round2 = {}

    # ① LOF: 现价快照 + 二级市场价格史 (直连重试)
    round2["fund_lof_spot_em"] = summarize(probe_timeout(ak.fund_lof_spot_em, ()))
    round2["fund_lof_hist_em(160105)"] = summarize(probe_timeout(ak.fund_lof_hist_em, ("160105",)))
    round2["fund_lof_hist_em(501001)"] = summarize(probe_timeout(ak.fund_lof_hist_em, ("501001",)))

    # ② 定增: stock_add_stock 实测 (cninfo)
    fn = getattr(ak, "stock_add_stock", None)
    round2["stock_add_stock"] = summarize(probe_timeout(fn, (), timeout=90)) if fn else "absent"
    for extra in ("stock_add_plan_cninfo", "stock_research_report_em"):
        if getattr(ak, extra, None):
            round2[extra] = "exists(untested)"

    # ③ 北交所: 东财日线 (stock_zh_a_hist, bj 股票代码)
    round2["stock_zh_a_hist(bj.871981)"] = summarize(probe_timeout(
        ak.stock_zh_a_hist, ("871981", "daily", "20220101", "20260906", "")))
    # 东财 bj 实时
    round2["stock_zh_a_spot_em含bj"] = "checked-in-spot" if getattr(ak, "stock_zh_a_spot_em", None) else "absent"

    result["round2_direct"] = round2
    OUT_FP.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps(round2, ensure_ascii=False, indent=2, default=str)[:3500])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
