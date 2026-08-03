"""市场广度快照 (v3.1.2) — 供 scanner LLM 判断"当前形势"

从全市场股票列表实时快照聚合广度/情绪/估值水平信号, 注入
`_market_scanner_node` 的 LLM 上下文。任何失败返回 {} → 调用方保持原提示词,
零行为变化 (仅新增信息)。

数据源: DataRouter.get_stock_list() — AKShare 全市场实时快照 (含 pe_ttm/pb)。
"""

import json
from typing import Dict

import pandas as pd
from loguru import logger


async def live_market_snapshot(router=None) -> Dict[str, float]:
    """全市场广度快照。失败返回 {} (调用方降级为原提示词)。

    Returns:
        {n_stocks, pct_up, avg_pct, limit_up, limit_down,
         median_pe, median_pb, above_ma20_pct}
    """
    try:
        if router is None:
            from data import get_data_router
            router = get_data_router()
        df = await router.get_stock_list()
        if df is None or df.empty or "error" in df.columns:
            return {}

        snap: Dict[str, float] = {"n_stocks": int(len(df))}

        # 涨跌分布 (不同 provider 列名不同)
        pct_col = next((c for c in ("pct_change", "pctChg", "涨跌幅") if c in df.columns), None)
        if pct_col:
            pct = pd.to_numeric(df[pct_col], errors="coerce").dropna()
            snap["pct_up"] = round(float((pct > 0).mean()), 3)
            snap["avg_pct"] = round(float(pct.mean()), 3)
            snap["limit_up"] = int((pct >= 9.5).sum())
            snap["limit_down"] = int((pct <= -9.5).sum())

        # 估值水平 (中位数对异常值稳健)
        pe_col = next((c for c in ("pe_ttm", "peTTM", "市盈率-动态", "市盈率") if c in df.columns), None)
        if pe_col:
            pe = pd.to_numeric(df[pe_col], errors="coerce")
            pos = pe[pe > 0]
            if len(pos):
                snap["median_pe"] = round(float(pos.median()), 2)
        pb_col = next((c for c in ("pb", "pbMRQ", "市净率") if c in df.columns), None)
        if pb_col:
            pb = pd.to_numeric(df[pb_col], errors="coerce")
            pos = pb[pb > 0]
            if len(pos):
                snap["median_pb"] = round(float(pos.median()), 2)

        return snap
    except Exception as e:
        logger.warning(f"市场广度快照获取失败: {e}")
        return {}


def format_snapshot(snap: Dict[str, float]) -> str:
    """把快照格式化为一行可读文本。空快照返回空串。"""
    if not snap:
        return ""
    return json.dumps(snap, ensure_ascii=False, sort_keys=True)
