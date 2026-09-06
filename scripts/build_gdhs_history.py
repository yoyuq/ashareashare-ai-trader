"""股东户数全历史建库 — 东财 RPT_HOLDERNUM_DET (C8 数据门 PASS).

按季度末 END_DATE 过滤分批拉取 (每批 1-5 页, 快), 冻结为
replay_data/gdhs_history.parquet。支持增量: 已有库则只拉缺失季度末。
零模拟: 拉取失败即报错退出, 不兜底。
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
OUT_FP = ROOT / "replay_data" / "gdhs_history.parquet"

URL = "https://datacenter-web.eastmoney.com/api/data/v1/get"
BASE_PARAMS = {
    "sortColumns": "HOLD_NOTICE_DATE,SECURITY_CODE",
    "sortTypes": "-1,-1",
    "pageSize": "500",
    "reportName": "RPT_HOLDERNUM_DET",
    "columns": "SECURITY_CODE,SECURITY_NAME_ABBR,END_DATE,INTERVAL_CHRATE,AVG_MARKET_CAP,"
               "AVG_HOLD_NUM,TOTAL_MARKET_CAP,TOTAL_A_SHARES,HOLD_NOTICE_DATE,HOLDER_NUM,"
               "PRE_HOLDER_NUM,HOLDER_NUM_CHANGE,HOLDER_NUM_RATIO,PRE_END_DATE",
    "source": "WEB",
    "client": "WEB",
}
COLMAP = {
    "SECURITY_CODE": "code",
    "END_DATE": "period_end",
    "HOLD_NOTICE_DATE": "announce_date",
    "HOLDER_NUM": "holder_num",
    "PRE_HOLDER_NUM": "holder_num_prev",
    "HOLDER_NUM_CHANGE": "holder_num_change",
    "HOLDER_NUM_RATIO": "holder_num_ratio_pct",
    "AVG_MARKET_CAP": "avg_mkt_cap",
    "AVG_HOLD_NUM": "avg_hold_num",
    "TOTAL_MARKET_CAP": "total_mkt_cap",
    "TOTAL_A_SHARES": "total_shares",
}
# 季度末序列 2013-03-31 → 2026-09-30 (东财股东户数统计起点≈2013-01)
QES = pd.date_range("2013-03-31", "2026-09-30", freq="QE").tolist()


def fetch_quarter(end_date: str) -> pd.DataFrame:
    params = dict(BASE_PARAMS)
    params["filter"] = f"(END_DATE='{end_date}')"
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
        time.sleep(0.4)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df[list(COLMAP)].rename(columns=COLMAP)
    df["period_end"] = pd.to_datetime(df["period_end"], errors="coerce")
    df["announce_date"] = pd.to_datetime(df["announce_date"], errors="coerce")
    for c in ("holder_num", "holder_num_prev", "holder_num_change",
              "holder_num_ratio_pct", "avg_mkt_cap", "avg_hold_num",
              "total_mkt_cap", "total_shares"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def main() -> int:
    for k in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
        os.environ.pop(k, None)
    socket.setdefaulttimeout(30)

    existing = None
    if OUT_FP.exists():
        existing = pd.read_parquet(OUT_FP)
        done_qes = set(existing["period_end"].dropna().dt.date.unique())
        print(f"已有库: {len(existing)} rows, {existing['code'].nunique()} codes, "
              f"{len(done_qes)} 个季度末已入库")
    else:
        done_qes = set()

    todo = [q for q in QES if q.date() > max(done_qes, default=pd.Timestamp("2013-01-01").date())]
    todo = [q for q in todo if q.date() not in done_qes and q <= pd.Timestamp.today()]
    print(f"待拉取季度末: {len(todo)} 个")

    frames = []
    for i, q in enumerate(todo, 1):
        df = fetch_quarter(q.strftime("%Y-%m-%d"))
        # 空季度 (罕见) 允许跳过 — API 确认该期无数据行, 非网络失败
        print(f"[{i}/{len(todo)}] {q.date()} -> {len(df)} rows", flush=True)
        if not df.empty:
            frames.append(df)
        time.sleep(0.5)

    if frames:
        new = pd.concat(frames, ignore_index=True)
        if existing is not None:
            new = pd.concat([existing, new], ignore_index=True)
        new = new.drop_duplicates(subset=["code", "period_end", "announce_date"], keep="last")
        new = new.sort_values(["announce_date", "code"]).reset_index(drop=True)
        new.to_parquet(OUT_FP, index=False)
        print(f"冻结 -> {OUT_FP}: {len(new)} rows, {new['code'].nunique()} codes, "
              f"{new['announce_date'].min().date()} -> {new['announce_date'].max().date()}")
    else:
        print("无新数据, 库保持不变")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
