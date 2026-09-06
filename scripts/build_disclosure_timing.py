"""财报披露预约时间全历史建库 — 东财 RPT_PUBLIC_BS_APPOIN (C9 数据门 PASS).

按报告期 REPORT_DATE 分批拉取 (每期 7-11 页), 冻结为
replay_data/disclosure_schedule_history.parquet。增量: 已有库只拉缺失报告期。
零模拟: 网络失败即报错退出; API 返回空期视为确认无数据 (跳过并记录)。
"""
from __future__ import annotations

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
OUT_FP = ROOT / "replay_data" / "disclosure_schedule_history.parquet"

URL = "https://datacenter-web.eastmoney.com/api/data/v1/get"
KEEP_COLS = ["SECURITY_CODE", "REPORT_DATE", "REPORT_TYPE", "FIRST_APPOINT_DATE",
             "FIRST_CHANGE_DATE", "SECOND_CHANGE_DATE", "THIRD_CHANGE_DATE",
             "ACTUAL_PUBLISH_DATE", "APPOINT_CHANGE", "IS_PUBLISH"]
PERIODS = [d.strftime("%Y%m%d") for d in pd.date_range("2008-12-31", "2025-12-31", freq="YE")]


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
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df[[c for c in KEEP_COLS if c in df.columns]].copy()
    df["REPORT_DATE"] = pd.to_datetime(df["REPORT_DATE"], errors="coerce")
    for c in ("FIRST_APPOINT_DATE", "FIRST_CHANGE_DATE", "SECOND_CHANGE_DATE",
              "THIRD_CHANGE_DATE", "ACTUAL_PUBLISH_DATE"):
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    return df


def main() -> int:
    for k in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
        os.environ.pop(k, None)
    socket.setdefaulttimeout(30)

    existing = None
    done = set()
    if OUT_FP.exists():
        existing = pd.read_parquet(OUT_FP)
        done = set(existing["REPORT_DATE"].dropna().dt.strftime("%Y%m%d").unique())
        print(f"已有库: {len(existing)} rows, {len(done)} 个报告期")
    todo = [p for p in PERIODS if p not in done]
    print(f"待拉取报告期: {len(todo)} 个")

    frames = []
    for i, p in enumerate(todo, 1):
        df = fetch_period(p)
        print(f"[{i}/{len(todo)}] {p} -> {len(df)} rows", flush=True)
        if not df.empty:
            frames.append(df)
        time.sleep(0.4)

    if frames:
        new = pd.concat(frames, ignore_index=True)
        if existing is not None:
            new = pd.concat([existing, new], ignore_index=True)
        new = new.drop_duplicates(subset=["SECURITY_CODE", "REPORT_DATE"], keep="last")
        new = new.sort_values(["REPORT_DATE", "SECURITY_CODE"]).reset_index(drop=True)
        new.to_parquet(OUT_FP, index=False)
        print(f"冻结 -> {OUT_FP}: {len(new)} rows, {new['SECURITY_CODE'].nunique()} codes, "
              f"{new['REPORT_DATE'].min().date()} -> {new['REPORT_DATE'].max().date()}")
    else:
        print("无新数据, 库保持不变")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
