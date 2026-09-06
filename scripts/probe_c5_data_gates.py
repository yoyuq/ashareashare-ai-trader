"""C5 三候选数据门探测 (PHASE2 §2) — 零模拟: 只枚举真实接口与历史深度, 不造代理.

候选:
  ① LOF 折溢价收敛: 需 LOF 现价/净值/折溢价 真历史 (≥3年, 可个股关联)。
  ② 定增倒挂: 需 定增实施价+日期+解禁 历史 (≥2018, 个股可关联)。
  ③ 北交所冷落低波: baostock 对 bj.8xxxxx 的日K支持 (turn/peTTM/amount)。

输出: reports/agent_loop/probe_c5_result.json
"""
from __future__ import annotations

import json
import socket
import sys
import warnings
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent.parent
socket.setdefaulttimeout(20)
OUT_FP = ROOT / "reports" / "agent_loop" / "probe_c5_result.json"


def _force_utf8() -> None:
    for name in ("stdout", "stderr"):
        s = getattr(sys, name, None)
        if hasattr(s, "reconfigure"):
            try:
                s.reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass


def probe_timeout(fn, args, timeout=45):
    with ThreadPoolExecutor(max_workers=1) as ex:
        try:
            fut = ex.submit(fn, *args)
            return fut.result(timeout=timeout)
        except Exception as e:  # noqa: BLE001
            return {"__error__": f"{type(e).__name__}: {str(e)[:150]}"}


def probe_lof() -> dict:
    import akshare as ak
    out: dict = {}
    for name, fn, args in [
        ("fund_lof_spot_em", ak.fund_lof_spot_em, ()),
        ("fund_lof_hist_em", ak.fund_lof_hist_em, ("160105",)),  # 南方积极配置, 2004 老 LOF
        ("fund_open_fund_info_em", ak.fund_open_fund_info_em, ("160105", "单位净值走势")),
    ]:
        r = probe_timeout(fn, args)
        if isinstance(r, dict) and "__error__" in r:
            out[name] = r
        elif r is None or len(r) == 0:
            out[name] = {"rows": 0}
        else:
            df = r
            info = {"rows": int(len(df)), "columns": [str(c) for c in df.columns]}
            date_col = next((c for c in df.columns if "日期" in str(c)), None)
            if date_col:
                d = pd.to_datetime(df[date_col], errors="coerce").dropna()
                info["date_min"], info["date_max"] = str(d.min().date()), str(d.max().date())
            out[name] = info
    return out


def probe_private_placement() -> dict:
    import akshare as ak
    out: dict = {}
    # akshare 定增相关候选
    for name in dir(ak):
        if "add" in name.lower() and "stock" in name.lower():
            out["candidate:" + name] = "exists"
    for name, fn, args in [
        ("stock_add_stock_cninfo", getattr(ak, "stock_add_stock_cninfo", None), ()),
    ]:
        if fn is None:
            continue
        r = probe_timeout(fn, args, timeout=60)
        if isinstance(r, dict) and "__error__" in r:
            out[name] = r
        elif r is None or len(r) == 0:
            out[name] = {"rows": 0}
        else:
            out[name] = {"rows": int(len(r)), "columns": [str(c) for c in r.columns][:25]}
    # 大宗交易 (定增相关流动性旁证, 顺带)
    return out


def probe_bj_baostock() -> dict:
    import baostock as bs
    lg = bs.login()
    if lg.error_code != "0":
        return {"login": f"FAIL {lg.error_msg}"}
    try:
        out = {}
        # 北交所: baostock 代码前缀 bj. (文档支持有限, 实测)
        for sym, tag in [("bj.835185", "锦好医疗"), ("bj.871981", "曙光数创")]:
            rs = bs.query_history_k_data_plus(
                sym, "date,code,close,volume,amount,turn,pctChg,peTTM,pbMRQ,tradestatus,isST",
                start_date="2022-01-01", end_date="2026-09-06", frequency="d", adjustflag="3")
            rows = []
            while rs.error_code == "0" and rs.next():
                rows.append(rs.get_row_data())
            if rs.error_code != "0":
                out[sym] = f"error {rs.error_code}: {rs.error_msg[:80]}"
            elif not rows:
                out[sym] = "empty"
            else:
                df = pd.DataFrame(rows, columns=rs.fields)
                out[sym] = {"rows": len(rows),
                            "date_min": df["date"].min(), "date_max": df["date"].max(),
                            "turn_nonnull": int((pd.to_numeric(df["turn"], errors="coerce") > 0).sum()),
                            "pe_nonnull": int((pd.to_numeric(df["peTTM"], errors="coerce") > 0).sum())}
        return out
    finally:
        try:
            bs.logout()
        except Exception:  # noqa: BLE001
            pass


def main() -> int:
    _force_utf8()
    result = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "lof": probe_lof(),
        "private_placement": probe_private_placement(),
        "bj_baostock": probe_bj_baostock(),
    }
    OUT_FP.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str)[:4000])
    print(f"\nsaved -> {OUT_FP}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
