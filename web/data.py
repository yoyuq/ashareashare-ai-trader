"""Dashboard 数据层 (v6.0 拆分自 dashboard.py)

缓存加载器 (@st.cache_data / @st.cache_resource) + 全市场抓取 + 行业猜测。
组件单例 (router/analyzer/detector/knowledge) 在模块导入时初始化一次,
各 tab 经 `from web.data import ...` 使用。
"""
import asyncio
import concurrent.futures
import json as _json
import os
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import streamlit as st
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.symbols import market_prefix, to_symbol  # noqa: E402
from scripts.shared import NAME_MAP  # noqa: E402


# ═══════════════════════════════════════════════════════════════
# Utility
# ═══════════════════════════════════════════════════════════════
def _run_async(coro):
    try:
        asyncio.get_running_loop()
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            return executor.submit(asyncio.run, coro).result()
    except RuntimeError:
        return asyncio.run(coro)


@st.cache_data(ttl=3600, show_spinner=False)
def load_stocks():
    s = {}
    try:
        cfg = yaml.safe_load(open(Path(__file__).parent.parent / "config" / "symbols.yaml", encoding="utf-8"))
        for sym in cfg.get("watchlist", {}).get("default", []):
            s[sym.replace("sh.", "").replace("sz.", "")] = sym
    except Exception:
        pass
    if len(s) < 5:
        s.update({"600519": "sh.600519", "300750": "sz.300750", "600036": "sh.600036",
                  "002594": "sz.002594", "000858": "sz.000858", "601318": "sh.601318"})
    return s


@st.cache_data(ttl=600, show_spinner=False)
def load_full_market_stocks():
    """从全市场缓存加载标的列表 {代码: 完整symbol}, 用于selectbox"""
    result = {}
    try:
        cache_p = Path(__file__).parent.parent / "simulation_data" / "full_market_cache.json"
        if cache_p.exists():
            with open(cache_p, "r", encoding="utf-8") as f:
                market_data = _json.load(f)
            for item in market_data.get("data", []):
                code = item.get("code", "")
                if code:
                    result[code] = to_symbol(code)  # v5.6 P2-2: 补北交所 bj 前缀
    except Exception:
        pass
    if len(result) < 10:
        result = {k.replace("sh.", "").replace("sz.", ""): v for k, v in load_stocks().items()}
    return result


@st.cache_resource(show_spinner=False)
def init_components():
    c = {}
    try:
        from data.router import get_data_router
        c["router"] = get_data_router()
    except Exception:
        c["router_err"] = "init failed"
    try:
        from analysis.indicators import TechnicalAnalyzer
        c["analyzer"] = TechnicalAnalyzer()
    except Exception:
        pass
    try:
        from analysis.regime import MarketRegimeDetector
        c["detector"] = MarketRegimeDetector()
    except Exception:
        pass
    try:
        from knowledge.manager import KnowledgeManager
        c["knowledge"] = KnowledgeManager()
    except Exception:
        pass
    return c


comps = init_components()
router = comps.get("router")
analyzer = comps.get("analyzer")
detector = comps.get("detector")
knowledge = comps.get("knowledge")


@st.cache_data(ttl=300, show_spinner=False)
def get_regime():
    try:
        from data.providers.base import DataFrequency, DataRequest
        today = date.today()

        async def f():
            req = DataRequest("sh.000300", today - timedelta(days=365), today, DataFrequency.DAILY)
            r = await router.get_daily_kline(req)
            return detector.detect(r.data)
        return _run_async(f())
    except Exception:
        import traceback
        traceback.print_exc()
        return None


@st.cache_data(ttl=120, show_spinner=False)
def get_stock_quick(sym, days=90):
    try:
        from data.providers.base import DataFrequency, DataRequest
        today = date.today()

        async def f():
            req = DataRequest(sym, today - timedelta(days=days), today, DataFrequency.DAILY)
            r = await router.get_daily_kline(req)
            d = r.data
            if d.empty:
                return None
            ind = analyzer.compute_all(d, symbol=sym)
            return {"data": d, "indicators": ind, "last": ind.to_dataframe().iloc[-1],
                    "close": float(d["close"].iloc[-1]),
                    "change": float(d.get("pct_change", pd.Series([0])).iloc[-1]) if "pct_change" in d.columns else 0,
                    "name": NAME_MAP.get(sym, sym.split(".")[-1] if "." in sym else sym)}
        return _run_async(f())
    except Exception:
        import traceback
        traceback.print_exc()
        return None


@st.cache_data(ttl=60, show_spinner=False)
def get_portfolio_state():
    """持仓快照 (60s 缓存) — 本地结构 + API MTM 实时价覆盖, 静态渲染不随 fragment 刷"""
    try:
        from simulation.portfolio import PortfolioManager
        from simulation.paper_trader import PaperTradingEngine
        from web.api_client import api_get
        m = PortfolioManager()
        e = PaperTradingEngine(m)
        s = e.get_summary()
        # 用 API MTM 实时价覆盖 (否则 current_price=成本, 盈亏0)
        try:
            d = api_get("/api/v1/portfolio/mtm", timeout=5)
            live_pos = {p.get("symbol"): p for p in d.get("positions", [])}
            for p in s.get("positions", []):
                lp = live_pos.get(p.get("symbol"))
                if lp:
                    p["current_price"] = lp.get("current_price", p.get("current_price"))
                    p["market_value"] = lp.get("market_value", p.get("market_value"))
                    p["unrealized_pnl"] = lp.get("unrealized_pnl", 0)
                    p["unrealized_pnl_pct"] = lp.get("unrealized_pnl_pct", 0)
            s["total_value"] = d.get("summary", {}).get("total_value", s.get("total_value"))
            s["total_return"] = d.get("summary", {}).get("total_return", s.get("total_return"))
            s["total_return_pct"] = d.get("summary", {}).get("total_return_pct", s.get("total_return_pct"))
        except Exception:
            pass
        return s
    except Exception:
        return None


@st.cache_data(ttl=300, show_spinner="正在加载全市场数据...")
def get_full_market():
    """获取全A股行情。AKShare → 本地缓存 → 东方财富直连 → 分析报告, 闭市时自动回退到缓存数据。"""
    from data.full_market_cache import (
        write_full_market_cache, read_full_market_cache, market_cache_lag_days,
    )

    source = None
    _provider = None
    data_date = date.today().isoformat()
    df = None

    # ── 尝试 1: AKShare (交易时段最佳) ──
    try:
        import akshare as ak
        df = ak.stock_zh_a_spot_em()
        source = "live"
        _provider = "akshare"
    except Exception:
        pass

    # ── 尝试 2: 本地缓存 (优先于网络请求, 避免闭市时段 API 限流) ──
    if df is None or df.empty:
        _cached_df, _cached_date = read_full_market_cache()
        if _cached_df is not None:
            df = _cached_df
            source = "cached"
            data_date = _cached_date or "unknown"

    # ── 尝试 3: 东方财富 API 直连 (闭市仍可获取前一日收盘数据, 带限速) ──
    if df is None or df.empty:
        try:
            df = _fetch_eastmoney_full_market()
            if df is not None and not df.empty:
                source = "live"
                _provider = "eastmoney"
        except Exception:
            pass

    # ── 回退 4: 最新分析报告 ──
    if df is None or df.empty:
        import glob as _glob
        report_files = sorted(_glob.glob(str(Path(__file__).parent.parent / "reports" / "data_*.json")), reverse=True)
        if report_files:
            try:
                with open(report_files[0], "r", encoding="utf-8") as f:
                    rpt = _json.load(f)
                prices = rpt.get("analysis_prices", {})
                rows = []
                for sym, info in prices.items():
                    code_short = sym.replace("sh.", "").replace("sz.", "").replace("bj.", "")
                    rows.append({
                        "code": code_short, "name": info.get("name", sym),
                        "price": info.get("close", 0), "pct_change": info.get("pct_change", 0),
                        "volume": info.get("volume", 0), "amount": info.get("amount", 0),
                        "turnover": info.get("turnover", 0), "pe_ttm": info.get("pe_ttm", 0),
                        "pb": info.get("pb", 0), "total_mv": info.get("total_mv", 0),
                    })
                if rows:
                    df = pd.DataFrame(rows)
                    source = "report"
                    data_date = rpt.get("date", "unknown")
            except Exception:
                pass

    if df is None or df.empty:
        st.session_state._market_source = None
        return None

    # ── 标准化列名 (akshare 用中文列名) ──
    col_map = {
        "代码": "code", "名称": "name", "最新价": "price", "涨跌幅": "pct_change",
        "涨跌额": "change_amt", "成交量": "volume", "成交额": "amount",
        "振幅": "amplitude", "最高": "high", "最低": "low", "今开": "open", "昨收": "prev_close",
        "量比": "vol_ratio", "换手率": "turnover", "市盈率-动态": "pe_ttm",
        "市净率": "pb", "总市值": "total_mv", "流通市值": "float_mv",
        "60日涨跌幅": "pct_60d", "年初至今涨跌幅": "pct_ytd",
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

    # 过滤无效数据
    if "price" in df.columns:
        df = df[df["price"].notna() & (df["price"] > 0)]
    if "name" in df.columns:
        df = df[df["name"].notna() & (df["name"] != "")]

    # 预计算搜索索引 (v5.6 P1-14): name 小写 + code, 避免每次交互 astype(str) 全量扫描
    df["_search_key"] = df["name"].str.lower().str.strip() + " " + df["code"].astype(str)

    # 添加交易所标识 (v5.6 P2-2: 补北交所 BJ, 不再把 bj 误归 SZ)
    df["exchange"] = df["code"].apply(lambda x: market_prefix(str(x)).upper())

    # 市值分位标记
    if "total_mv" in df.columns:
        mv = df["total_mv"].dropna()
        if len(mv) > 100:
            q80 = mv.quantile(0.8); q50 = mv.quantile(0.5); q20 = mv.quantile(0.2)
            df["mv_tier"] = df["total_mv"].apply(
                lambda x: "🟢大盘" if x > q80 else ("🟡中盘" if x > q50 else ("🟠小盘" if x > q20 else "🔴微盘"))
            )

    # ── 缓存成功的全市场数据 (统一写缓存, v5.6 P1-15) ──
    if source == "live" and len(df) > 1000:
        try:
            write_full_market_cache(df, data_date, _provider or "live")
        except Exception:
            pass

    st.session_state._market_source = {
        "source": source, "date": data_date, "count": len(df),
        "lag_days": market_cache_lag_days(data_date),
    }
    return df


def _fetch_eastmoney_full_market():
    """东方财富全市场行情 API 直连 — 闭市时段分页拉取(带重试), 降级时返回部分数据"""
    import requests as _req
    import time as _time

    field_map = {
        "f2": "price", "f3": "pct_change", "f4": "change_amt",
        "f5": "volume", "f6": "amount", "f7": "amplitude", "f8": "turnover",
        "f9": "pe_ttm", "f10": "vol_ratio", "f12": "code", "f14": "name",
        "f15": "high", "f16": "low", "f17": "open", "f18": "prev_close",
        "f20": "total_mv", "f21": "float_mv",
    }
    fields = ",".join(field_map.keys())

    all_items = []
    page_size = 200
    max_pages = 28  # ~5500 / 200

    for page in range(1, max_pages + 1):
        url = (
            f"https://push2.eastmoney.com/api/qt/clist/get?"
            f"pn={page}&pz={page_size}&po=1&np=1&fltt=2&invt=2"
            f"&fid=f3&fs=m:0+t:6,m:0+t:80,m:1+t:2,m:1+t:23"
            f"&fields={fields}"
        )
        try:
            resp = _req.get(url, timeout=15,
                            headers={"User-Agent": "Mozilla/5.0", "Referer": "https://quote.eastmoney.com/"})
            data = resp.json().get("data", {})
            items = data.get("diff") or []
            if not items:
                break
            for item in items:
                row = {}
                for api_key, col_name in field_map.items():
                    val = item.get(api_key)
                    if val == "-" or val is None:
                        val = 0
                    row[col_name] = val
                all_items.append(row)
            if len(items) < page_size:
                break
        except Exception:
            break
        # 闭市时段放慢速度, 避免触发反爬
        _time.sleep(0.3 if len(all_items) > 2000 else 0.15)

    if not all_items:
        return None
    return pd.DataFrame(all_items)


# 行业板块映射 (申万一级 — 基于代码区间)
def _guess_sector(code: str) -> str:
    code_str = str(code)
    if code_str.startswith(("8", "4", "920")):  # 北交所 (v5.6 P2-2)
        return "北交所"
    if code_str.startswith(("60", "68")):  # 上交所
        num = int(code_str[:3]) if len(code_str) >= 3 else 0
        if 36 <= num <= 39: return "银行"
        if 48 == num: return "券商"
        if code_str.startswith("688"): return "科创板"
        if num in (16, 19): return "能源"
        if num in (11, 15): return "交运"
        if num in (17, 18): return "材料"
        if num in (10, 58): return "工业"
        if num in (50, 51): return "消费"
        if num in (55, 56): return "医药"
        if num in (53, 54): return "地产"
        if num in (57, 59): return "可选消费"
        if num in (60, 61): return "金融"
        if num in (63, 64): return "科技"
        if num in (65, 66): return "公用事业"
        return "其他"
    else:  # 深交所
        num = int(code_str[:3]) if len(code_str) >= 3 else 0
        if num in (1, 2): return "主板"
        if num == 3: return "创业板"
        return "深市"
