"""抓因子 A/B 所需的 lookback 日线 (Baostock, 免代理)。

背景: 横截面因子 (短期反转/动量/低波) 需窗口起点 T 之前的历史收益/换手, 但 replay_data
窗口文件从 T 才开始。2019 窗口已有 daily_2018-10-01_2019-12-31.parquet 提供 lookback,
本脚本补 2020 与 2024 两段。

lookback (预注册):
  2020牛 (T=2020-06-01): 抓 2020-03-01 ~ 2020-06-01 (~3个月/60交易日)
  2024震荡 (T=2024-01-02): 抓 2023-11-01 ~ 2024-01-02 (~2个月/40交易日)
universe: 各窗口 top-800 流动性可投资票 (与 run_factor_ab 一致)。
输出: replay_data/lookback_2020.parquet, replay_data/lookback_2024.parquet

用法:
  python scripts/fetch_factor_lookback.py --jobs 2020,2024
"""
from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.build_long_replay import robust_fetch  # noqa: E402

ROOT = Path(__file__).parent.parent

JOBS = {
    "2020": {
        "win": "replay_data/daily_2020-06-01_2021-02-28.parquet",
        "start": date(2020, 3, 1),
        "end": date(2020, 6, 1),
    },
    "2024": {
        "win": "replay_data/daily_2024-01-01_2024-12-31.parquet",
        "start": date(2023, 11, 1),
        "end": date(2024, 1, 2),
    },
}


def liquid_universe(window_fp: str, top_n: int = 800) -> list[str]:
    df = pd.read_parquet(ROOT / window_fp)
    df = df[df.symbol != "sh.000001"]
    df["isST"] = df["isST"].astype(str)
    inv = df[(df.isST == "0") & (df.is_trade == 1) & (df.peTTM > 0) & (df.pbMRQ > 0)]
    med = inv.groupby("symbol")["amount"].median().sort_values(ascending=False)
    return med.head(top_n).index.tolist()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--jobs", default="2020,2024", help="逗号分隔的 job key (2020/2024)")
    args = ap.parse_args()

    for key in args.jobs.split(","):
        key = key.strip()
        if key not in JOBS:
            print(f"未知 job: {key}")
            continue
        cfg = JOBS[key]
        out = ROOT / f"replay_data/lookback_{key}.parquet"
        if out.exists() and pd.read_parquet(out).symbol.nunique() > 0:
            print(f"[{key}] lookback 已存在, 跳过")
            continue
        uni = liquid_universe(cfg["win"])
        print(f"[{key}] universe={len(uni)} 只, lookback {cfg['start']}~{cfg['end']}")
        robust_fetch(uni, cfg["start"], cfg["end"], out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
