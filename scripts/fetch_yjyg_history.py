"""业绩预告全历史抓取 — ak.stock_yjyg_em 按报告期, 分片落 replay_data/yjyg/{period}.parquet.

- 74 期 (20081231 → 20260630); 断点续传: 已有非空分片跳过。
- 只保留归母净利润指标行, 丢大文本列 (业绩变动原因)。
- 零模拟: 接口失败/空结果如实记 0 字节哨兵 (确认无数据), 不造数。
"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent.parent
OUT = ROOT / "replay_data" / "yjyg"
OUT.mkdir(parents=True, exist_ok=True)

PERIODS = [f"{y}1231" for y in range(2008, 2026)] + \
          [f"{y}0331" for y in range(2009, 2027)] + \
          [f"{y}0630" for y in range(2009, 2027)] + \
          [f"{y}0930" for y in range(2009, 2027)]
PERIODS = sorted(set(PERIODS))


def _force_utf8() -> None:
    for name in ("stdout", "stderr"):
        s = getattr(sys, name, None)
        if hasattr(s, "reconfigure"):
            try:
                s.reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass


def fetch_period(period: str) -> pd.DataFrame | None:
    import akshare as ak
    d = ak.stock_yjyg_em(date=period)
    if d is None or d.empty:
        return None
    keep = [c for c in ("股票代码", "股票简称", "预测指标", "预测数值", "业绩变动幅度",
                        "预告类型", "上年同期值", "公告日期") if c in d.columns]
    d = d[keep].copy()
    d["period"] = period
    return d


def main() -> int:
    _force_utf8()
    todo = [p for p in PERIODS
            if not ((OUT / f"{p}.parquet").exists() and (OUT / f"{p}.parquet").stat().st_size > 0)]
    print(f"待抓 {len(todo)}/{len(PERIODS)} 期", flush=True)
    ok, empty, fail = 0, 0, 0
    for i, p in enumerate(todo):
        try:
            d = fetch_period(p)
        except Exception as e:
            print(f"  {p} FAIL: {type(e).__name__}: {str(e)[:80]}", flush=True)
            fail += 1
            time.sleep(2.0)
            continue
        fp = OUT / f"{p}.parquet"
        if d is None or d.empty:
            fp.write_bytes(b"")  # 确认无数据哨兵
            empty += 1
            print(f"  {p} 无数据 (哨兵)", flush=True)
        else:
            # 只留归母净利润指标行 (每股收益/扣非行干扰同票去重)
            m = d["预测指标"].astype(str).str.contains("归属于上市公司股东的净利润", na=False) & \
                ~d["预测指标"].astype(str).str.contains("扣除非经常性损益", na=False)
            d2 = d[m]
            d2.to_parquet(fp)
            ok += 1
            print(f"  {p} {len(d2)}/{len(d)} 行 (归母过滤后)", flush=True)
        time.sleep(0.8)
    print(f"完成: 成功{ok} 空{empty} 失败{fail} / {len(todo)}", flush=True)
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
