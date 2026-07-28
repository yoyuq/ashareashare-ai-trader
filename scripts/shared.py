"""
共享模块 — 股票池加载、名称映射、工具函数
避免 run_daily_analysis / evening_summary / morning_buy 之间的重复代码

v2.9: COMMON_PRICES 加入 TTL 过期检查 + 最后更新日追踪
"""

import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

# ═══════════════════════════════════════════════════════════════
# 完整的名称映射 (60+只)
# ═══════════════════════════════════════════════════════════════

NAME_MAP: Dict[str, str] = {
    # 金融
    "sh.600036": "招商银行", "sh.601318": "中国平安", "sh.600030": "中信证券",
    "sh.600048": "保利发展",
    # 消费
    "sh.600519": "贵州茅台", "sz.000858": "五粮液",
    "sz.000333": "美的集团", "sz.000651": "格力电器",
    "sh.601933": "永辉超市", "sh.600754": "锦江酒店", "sh.603605": "珀莱雅",
    # 科技/TMT
    "sz.002415": "海康威视", "sh.603501": "韦尔股份",
    "sz.002230": "科大讯飞", "sh.688111": "金山办公",
    "sz.300502": "新易盛", "sh.600050": "中国联通",
    "sz.002027": "分众传媒", "sz.300413": "芒果超媒", "sz.300308": "中际旭创",
    # 新能源
    "sz.300750": "宁德时代", "sh.601012": "隆基绿能",
    "sz.002594": "比亚迪", "sh.601127": "赛力斯",
    "sz.002459": "晶澳科技", "sz.300014": "亿纬锂能",
    # 半导体
    "sh.688981": "中芯国际", "sz.002049": "紫光国微",
    "sh.688012": "中微公司", "sh.603986": "兆易创新", "sh.688256": "寒武纪",
    # 高端制造
    "sz.300124": "汇川技术", "sh.600031": "三一重工",
    "sh.600760": "中航沈飞", "sz.002179": "中航光电",
    "sh.688017": "绿的谐波", "sz.002572": "索菲亚",
    # 资源/周期
    "sh.600309": "万华化学",
    "sh.601899": "紫金矿业", "sz.002460": "赣锋锂业",
    "sh.600019": "宝钢股份",
    "sh.601088": "中国神华", "sh.601225": "陕西煤业",
    "sh.600028": "中国石化", "sh.600585": "海螺水泥",
    # 基建/公用
    "sh.601668": "中国建筑",
    "sh.600009": "上海机场", "sh.601111": "中国国航",
    "sh.600900": "长江电力", "sh.601985": "中国核电", "sz.300070": "碧水源",
    # 医药/农业
    "sh.600276": "恒瑞医药", "sz.300760": "迈瑞医疗",
    "sz.002714": "牧原股份", "sh.603877": "太平鸟",
    # 其他
    "sh.600895": "张江高科", "sz.300033": "同花顺", "sz.300624": "万兴科技",
    "sz.300433": "蓝思科技",
}

# 常用价格兜底 (最后更新日: 2026-07-26)
# ⚠️ 这些价格每天都会过时,只能用于离线兜底,不能静默使用过期数据
_PRICE_LAST_UPDATED: str = "2026-07-26"
_PRICE_MAX_AGE_DAYS: int = 3  # 超过3天未更新的价格视为过期

COMMON_PRICES: Dict[str, float] = {
    "sh.600009": 24.21,
    "sh.600030": 28.55,
    "sh.600031": 19.79,
    "sh.600036": 38.82,
    "sh.600276": 54.17,
    "sh.600309": 75.37,
    "sh.600519": 1302.3,
    "sh.600760": 43.13,
    "sh.600900": 28.89,
    "sh.601012": 12.61,
    "sh.601088": 45.53,
    "sh.601127": 56.21,
    "sh.601318": 54.09,
    "sh.601668": 4.68,
    "sh.601899": 33.58,
    "sh.603605": 58.11,
    "sh.688111": 236.88,
    "sh.688256": 1248.33,
    "sh.688981": 145.33,
    "sz.000333": 84.79,
    "sz.000651": 40.74,
    "sz.000858": 74.56,
    "sz.002230": 40.2,
    "sz.002415": 35.55,
    "sz.002594": 92.38,
    "sz.002714": 39.55,
    "sz.300124": 61.28,
    "sz.300502": 496.2,
    "sz.300750": 379.54,
    "sz.300760": 150.78,
}


def update_fallback_prices(analysis_prices: Dict[str, dict]) -> int:
    """
    🆕 v2.10: 分析完成后自动更新兜底价格

    从 analysis_prices 中提取 close 价格, 更新 COMMON_PRICES 和 _PRICE_LAST_UPDATED。
    只更新已有 key (不新增, 避免污染未经验证的标的)。

    Returns:
        更新的标的数量
    """
    updated = 0
    for sym, info in analysis_prices.items():
        close = info.get("close", 0) if isinstance(info, dict) else 0
        if close > 0 and sym in COMMON_PRICES:
            COMMON_PRICES[sym] = round(close, 2)
            updated += 1

    if updated > 0:
        global _PRICE_LAST_UPDATED
        _PRICE_LAST_UPDATED = date.today().isoformat()

    return updated


def get_fallback_price(symbol: str) -> Tuple[Optional[float], Optional[str]]:
    """
    获取兜底价格 + 过期警告

    Returns:
        (price, warning_message)
        - price: 兜底价格,过期或不存在时返回 None
        - warning_message: 若过期则返回警告,正常则 None
    """
    from loguru import logger

    price = COMMON_PRICES.get(symbol)
    if price is None:
        return None, None  # 不在兜底列表中

    try:
        last_updated = date.fromisoformat(_PRICE_LAST_UPDATED)
        age = (date.today() - last_updated).days
    except ValueError:
        age = 999

    if age > _PRICE_MAX_AGE_DAYS:
        msg = (
            f"⚠️ {symbol} 兜底价格已过期{age}天(最后更新{_PRICE_LAST_UPDATED}),"
            f" 当前使用价格¥{price}可能不准确,请尽快更新 COMMON_PRICES"
        )
        logger.warning(msg)
        return price, msg

    return price, None


def resolve_name(symbol: str) -> str:
    """解析股票名称"""
    return NAME_MAP.get(symbol, symbol.split(".")[-1] if "." in symbol else symbol)


# ═══════════════════════════════════════════════════════════════
# 🆕 v2.12: 腾讯行情刷新兜底价格 + 统一价格获取
# ═══════════════════════════════════════════════════════════════

async def refresh_fallback_prices_from_tencent() -> int:
    """
    从腾讯行情API刷新 COMMON_PRICES 中的所有兜底价格。

    在任何脚本获取兜底价格之前调用此函数, 可确保兜底价格不
    过期。只要腾讯行情可用, 兜底价格就可以自愈。

    Returns:
        成功更新的标的数量
    """
    try:
        from data.providers.tencent_provider import TencentFinanceProvider
        tp = TencentFinanceProvider()
        symbols = list(COMMON_PRICES.keys())
        quotes = await tp.get_realtime_quotes(symbols)

        updated = 0
        for sym, info in quotes.items():
            price = info.get("price", 0)
            if price > 0 and sym in COMMON_PRICES:
                COMMON_PRICES[sym] = round(price, 2)
                updated += 1

        if updated > 0:
            global _PRICE_LAST_UPDATED
            _PRICE_LAST_UPDATED = date.today().isoformat()
            from loguru import logger
            logger.info(f"🔄 兜底价格已刷新: {updated}只 (来源: 腾讯行情)")

        return updated
    except Exception:
        return 0


def get_best_price(symbol: str, analysis_prices: dict,
                   rt_quotes: dict = None) -> float:
    """
    统一价格获取 (v2.12): 分析价 > 腾讯实时 > 兜底价 > 0

    消除 morning_buy 和 evening_sell 中的重复价格获取逻辑。

    Returns:
        最佳可用价格, 无可用价格时返回 0
    """
    rt_quotes = rt_quotes or {}

    # 1. 分析报告中的收盘价
    ap = analysis_prices.get(symbol, {})
    close = ap.get("close", 0) if isinstance(ap, dict) else 0
    if close > 0:
        return close

    # 2. 腾讯实时行情
    rt = rt_quotes.get(symbol, {})
    rt_price = rt.get("price", 0)
    if rt_price > 0:
        return rt_price

    # 3. 兜底价格 (shared.py COMMON_PRICES)
    fb_price, _ = get_fallback_price(symbol)
    if fb_price is not None and fb_price > 0:
        return fb_price

    return 0.0


# ═══════════════════════════════════════════════════════════════
# 🆕 v2.13: 量化最佳实践 — 凯利公式 + ATR自适应止损 + 移动止盈
# ═══════════════════════════════════════════════════════════════

def check_rsi_filter(rsi: float, trend_score: float) -> Tuple[bool, str]:
    """
    RSI超买过滤 (v2.11)

    Returns:
        (should_skip: bool, reason: str)
    """
    if rsi > 75:
        return True, f"RSI={rsi:.0f}极度超买, 大概率回调"
    if rsi > 70:
        if trend_score <= 0.5:
            return True, f"RSI={rsi:.0f}超买且趋势偏弱(趋势{trend_score:.2f}), 回调风险高"
        else:
            return False, f"RSI={rsi:.0f}超买但趋势强劲(趋势{trend_score:.2f}), 允许买入(降级警告)"
    return False, ""


def calc_kelly_position_pct(win_rate: float, conviction: float,
                            payoff_ratio: float = 2.0,
                            sharpe: float = 0.0) -> float:
    """
    凯利公式仓位计算 (v2.13)

    f* = (win_rate × payoff_ratio - loss_rate) / payoff_ratio

    行业实践采用"半凯利/四分之一凯利"避免过拟合:
      - 高确信(>70%): f* × 0.5 (半凯利)
      - 中确信(50-70%): f* × 0.33 (三分之一凯利)
      - 低确信(<50%): f* × 0.25 (四分之一凯利)

    Args:
        win_rate: 策略回测胜率 (0-1)
        conviction: 当前确信度 (0-1)
        payoff_ratio: 盈亏比 (avg_win/avg_loss, 默认2.0, 可用夏普/2近似)
        sharpe: 夏普比率 (用于微调payoff)

    Returns:
        凯利建议仓位比例 (0.0-0.25)
    """
    # 胜率来源: 使用历史回测胜率, 以确信度作为折扣 (v2.14: min而非max, 避免乐观偏差)
    # 历史胜率是客观基准, 确信度是主观判断 — 取较低值以保守
    effective_wr = min(win_rate, max(conviction, 0.30))
    effective_wr = min(effective_wr, 0.95)
    loss_rate = 1.0 - effective_wr

    # 盈亏比: 如有夏普, 用夏普/2 + 1.5 作为更保守的近似
    if sharpe > 0:
        payoff_ratio = max(payoff_ratio, sharpe / 2.0 + 1.0)

    # 凯利公式核心
    # f* = (p × b - q) / b
    if payoff_ratio <= 0:
        return 0.0
    kelly_f = (effective_wr * payoff_ratio - loss_rate) / payoff_ratio

    # 凯利为负 → 不参与
    if kelly_f <= 0:
        return 0.0

    # 分数凯利 (行业标准: 按确信度分级)
    if conviction > 0.70:
        fractional_f = kelly_f * 0.50   # 半凯利
    elif conviction > 0.50:
        fractional_f = kelly_f * 0.33   # 三分之一凯利
    else:
        fractional_f = kelly_f * 0.25   # 四分之一凯利

    # 行业上限: 单只不超过25%
    return round(min(fractional_f, 0.25), 4)


def calc_atr_stop_loss(entry_price: float, atr: float, regime: str = "range_bound",
                       rsi: float = 50.0) -> Tuple[float, float]:
    """
    ATR 自适应止损止盈 (v2.13)

    替代固定百分比止损。止损距离随波动率伸缩:
      - 高波动 → 宽止损 (避免被噪音扫出)
      - 低波动 → 窄止损 (保护利润)

    Args:
        entry_price: 入场价
        atr: 14日ATR值 (如无数据, 用 entry_price × 0.03 近似)
        regime: 市场状态
        rsi: RSI值 (超买时收紧)

    Returns:
        (stop_loss, take_profit)
    """
    if atr <= 0:
        atr = entry_price * 0.03  # 默认波动率3%

    atr_pct = atr / entry_price  # 波动率百分比

    # ATR乘数: 趋势市放宽, 震荡市标准
    regime_multipliers = {
        "strong_bull": 3.0,
        "weak_bull": 2.5,
        "range_bound": 2.0,
        "weak_bear": 2.5,
        "strong_bear": 3.0,
        "crisis": 4.0,  # 危机中波动极大, 大幅放宽
    }
    atr_mult = regime_multipliers.get(regime, 2.0)

    # RSI超买收紧止损
    if rsi > 75:
        atr_mult -= 0.5
    elif rsi > 70:
        atr_mult -= 0.25

    # 止损 = 入场价 × (1 - atr_mult × ATR%)
    sl_pct = atr_mult * atr_pct
    # 硬限制: 止损在3%-12%之间 (避免过紧或过松)
    sl_pct = max(0.03, min(sl_pct, 0.12))
    stop_loss = round(entry_price * (1 - sl_pct), 2)

    # 止盈: 1.5× 止损距离 (风险回报 1:1.5)
    tp_pct = sl_pct * 1.5
    tp_pct = min(tp_pct, 0.20)  # 止盈上限20%
    take_profit = round(entry_price * (1 + tp_pct), 2)

    return stop_loss, take_profit


def calc_trailing_stop(entry_price: float, current_price: float,
                       highest_since_entry: float, atr: float,
                       regime: str = "range_bound") -> Tuple[bool, float, str]:
    """
    ATR 移动止盈检查 (v2.13)

    当价格从高点回落超过 ATR_mult × ATR 时触发卖出。
    这比固定止盈价更灵活, 能跟随趋势锁定更多利润。

    Args:
        entry_price: 入场价
        current_price: 当前价
        highest_since_entry: 持仓期间最高价
        atr: 14日ATR
        regime: 市场状态

    Returns:
        (should_sell: bool, trailing_level: float, detail: str)
    """
    if atr <= 0:
        atr = current_price * 0.03

    atr_pct = atr / current_price

    # 震荡市用更紧的trail, 趋势市放宽
    trail_mult = {"range_bound": 2.0, "weak_bull": 2.5, "weak_bear": 2.5,
                  "strong_bull": 3.0, "strong_bear": 3.0, "crisis": 4.0}
    mult = trail_mult.get(regime, 2.0)

    trailing_level = highest_since_entry * (1 - mult * atr_pct)

    # 只在盈利时检查 (trailing_level > entry_price)
    if trailing_level <= entry_price:
        return False, trailing_level, ""

    if current_price <= trailing_level:
        pnl_pct = (trailing_level / entry_price - 1) * 100
        detail = (f"移动止盈触发: 高点{highest_since_entry:.2f}回落"
                  f" → {current_price:.2f}≤{trailing_level:.2f}"
                  f" 锁定+{pnl_pct:.1f}%")
        return True, trailing_level, detail

    return False, trailing_level, ""


def _fetch_index_symbols(source_name: str) -> List[str]:
    """从指数名称获取成分股列表"""
    import akshare as ak
    index_map = {
        "hs300": "000300", "csi500": "000905", "csi1000": "000852",
        "chinext": "399006", "star_market": "000688",
    }
    code = index_map.get(source_name, source_name)
    try:
        df = ak.index_stock_cons_weight_csindex(symbol=code)
        if df.empty:
            return []
        return [
            f"sh.{c}" if c.startswith("6") else f"sz.{c}"
            for c in df["成分券代码"].tolist()
        ]
    except Exception:
        return []


def load_watchlist(pool: str = "all_industries") -> List[str]:
    """从配置加载股票池 — 支持 watchlist(硬编码) + stock_pools(动态指数)"""
    try:
        import yaml
        cfg = yaml.safe_load(
            open(Path(__file__).parent.parent / "config" / "symbols.yaml", encoding="utf-8")
        )

        # 1. 先查 watchlist (硬编码精选池)
        watchlists = cfg.get("watchlist", {})
        if pool in watchlists:
            pool_data = watchlists[pool]
            if isinstance(pool_data, list):
                return pool_data
            elif isinstance(pool_data, dict) and "symbols" in pool_data:
                return pool_data["symbols"]

        # 2. 再查 stock_pools (支持 source 动态加载)
        stock_pools = cfg.get("stock_pools", {})
        if pool in stock_pools:
            pool_cfg = stock_pools[pool]
            symbols = []
            for sym in pool_cfg.get("symbols", []):
                symbols.append(sym)
            for src in pool_cfg.get("source", []):
                if src == "all_a_share":
                    for s in ["hs300", "csi500", "csi1000"]:
                        symbols.extend(_fetch_index_symbols(s))
                else:
                    symbols.extend(_fetch_index_symbols(src))
            if symbols:
                return list(set(symbols))

        # 3. 兜底
        return watchlists.get("default", watchlists.get("all_industries", {}).get("symbols", []))
    except Exception:
        return [
            "sh.600519", "sh.600036", "sz.000858", "sz.300750",
            "sh.601318", "sz.002415", "sh.600276", "sz.000333",
        ]