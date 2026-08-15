"""全市场快照缓存 (full_market_cache.json) 的单一读写入口 (v5.6 P1-15 统一写缓存)。

此前 `web.dashboard.get_full_market` 与 `scripts/refresh_market_cache.py` 各自手写
`json.dump`/`json.load`, schema 略不一致 (`source` 一个硬编码 `eastmoney`, 一个 `tencent_realtime`)。
统一到本模块: 单一 `CACHE_PATH`、单一 schema (`date`/`count`/`source`/`data`)、单一读写函数,
避免多点手写造成的字段漂移。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

# 快照文件相对仓库根目录
CACHE_PATH = Path(__file__).resolve().parent.parent / "simulation_data" / "full_market_cache.json"


def write_full_market_cache(df: pd.DataFrame, date: str, source: str) -> None:
    """把全市场 DataFrame 写入快照缓存 (schema: date/count/source/data)。

    Args:
        df: 已标准化列名 (code/name/price/pct_change/...) 的全市场行情。
        date: 数据日期 (ISO `YYYY-MM-DD`)。
        source: 数据来源标识 (e.g. "eastmoney" / "akshare" / "tencent_realtime")。
    """
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    records = df.to_dict(orient="records")
    # NaN/Inf → None (NaN 不是合法 JSON, 会污染跨语言消费者, 统一在写入侧清理)
    records = [
        {k: (None if isinstance(v, float) and pd.isna(v) else v) for k, v in r.items()}
        for r in records
    ]
    cache_data = {
        "date": date,
        "count": int(len(df)),
        "source": source,
        "data": records,
    }
    CACHE_PATH.write_text(json.dumps(cache_data, ensure_ascii=False, default=str),
                          encoding="utf-8")


def read_full_market_cache() -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """读取快照缓存 → (DataFrame, date)。

    Returns:
        (DataFrame, date) — 文件不存在、数据为空或损坏时返回 (None, None)。
    """
    if not CACHE_PATH.exists():
        return None, None
    try:
        cache = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
        df = pd.DataFrame(cache.get("data", []))
        date = cache.get("date", "") or None
        return (df if not df.empty else None), str(date)
    except (ValueError, TypeError, OSError):
        return None, None


def market_cache_lag_days(date: str) -> int:
    """缓存数据相对今天的自然日滞后天数 (解析失败返回 -1 表示未知)。"""
    from datetime import date as _date_cls
    try:
        return (_date_cls.today() - _date_cls.fromisoformat(str(date))).days
    except (ValueError, TypeError):
        return -1
