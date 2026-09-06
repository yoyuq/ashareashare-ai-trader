"""C9 财报披露时滞 — 数据门探针 v2 (零模拟, 只读探测不建库).

akshare stock_yysj_em 列名清单过时 (29列 vs 22列 Length mismatch) → 直接调
底层东财 API RPT_PUBLIC_BS_APPOIN, columns=ALL, 动态检视日期字段。
探针问题:
  1) 历史报告期 (2018/2020/2023 年报) 是否可查 + 行数;
  2) 实际披露日期覆盖率;
  3) 首次预约 → 实际披露时滞 (变更频率), PIT 表达力 = 首次预约在披露前可知。
输出 probe_c9_result.json
"""
from __future__ import annotations

import json
import os
import socket
import sys
import time
import warnings
from pathlib import Path

import pandas as pd
import requests

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent.parent
OUT_FP = ROOT / "reports" / "agent_loop" / "probe_c9_result.json"

URL = "https://datacenter-web.eastmoney.com/api/data/v1/get"
PERIODS = ["20181231", "20201231", "20231231"]  # 年报 (20081231 起有史)


def fetch_period(period: str) -> pd.DataFrame:
    params = {
        "sortColumns": "FIRST_APPOINT_DATE,SECURITY_CODE",
        "sortTypes": "1,1",
        "pageSize": "500",
        "pageNumber": "1",
        "reportName": "RPT_PUBLIC_BS_APPOIN",
        "columns": "ALL",
        "source": "WEB",
        "client": "WEB",
        "filter": (f'(SECURITY_TYPE_CODE in ("058001001","058001008"))'
                   f'(TRADE_MARKET_CODE!="069001017")'
                   f"(REPORT_DATE='{period[:4]}-{period[4:6]}-{period[6:]}')"),
    }
    rows: list[dict] = []
    page = 1
    while True:
        params["pageNumber"] = str(page)
        r = requests.get(URL, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()["result"]
        rows.extend(data["data"])
        if page >= data["pages"]:
            break
        page += 1
        time.sleep(0.3)
    return pd.DataFrame(rows)


def main() -> int:
    for k in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
        os.environ.pop(k, None)
    socket.setdefaulttimeout(30)

    result: dict = {"probe": "C9 财报披露时滞 数据门 v2 (直连 RPT_PUBLIC_BS_APPOIN)",
                    "periods": {}}
    for period in PERIODS:
        df = fetch_period(period)
        info: dict = {"rows": int(len(df))}
        if df.empty:
            result["periods"][period] = info
            continue
        if period == PERIODS[0]:
            result["all_columns"] = list(df.columns)
        date_cols = [c for c in df.columns
                     if any(k in c.upper() for k in ("APPOINT", "PUBLISH", "RELEASE"))]
        info["date_like_cols"] = date_cols
        for c in date_cols:
            s = pd.to_datetime(df[c], errors="coerce")
            if s.notna().any():
                info[c] = {"covered": int(s.notna().sum()),
                           "range": [str(s.min().date()), str(s.max().date())]}
        first = next((c for c in date_cols if "FIRST_APPOINT" in c), None)
        actual = next((c for c in date_cols if "ACTUAL" in c.upper()), None)
        if first and actual:
            f = pd.to_datetime(df[first], errors="coerce")
            a = pd.to_datetime(df[actual], errors="coerce")
            lag = (a - f).dt.days.dropna()
            if len(lag):
                info["lag_days_median"] = float(lag.median())
                info["lag_days_p90"] = float(lag.quantile(0.9))
                info["changed_ratio"] = float((lag != 0).mean())
                info["neg_lag_ratio"] = float((lag < 0).mean())
        result["periods"][period] = info
        print(f"[{period}] rows={info['rows']} "
              f"lag_med={info.get('lag_days_median')} changed={info.get('changed_ratio')}",
              flush=True)

    OUT_FP.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved -> {OUT_FP}")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
