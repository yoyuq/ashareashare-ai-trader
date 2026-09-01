"""APIRouter: 行情/市场状态/成本/实时市场 (v6.0 拆分自 server.py)

路由 path 与原 server.py 逐字一致。
"""
import asyncio
from datetime import date, datetime, time, timedelta

from fastapi import APIRouter, HTTPException

from api.deps import get_analyzer, get_detector, get_router
from api.schemas import RegimeResponse, StockInfoResponse
from analysis.regime import get_regime_parameters

router = APIRouter(prefix="/api/v1", tags=["market"])


@router.get("/stock/{symbol}", response_model=StockInfoResponse)
async def get_stock_info(symbol: str):
    """获取单只股票的行情与技术指标"""
    data_router = get_router()
    analyzer = get_analyzer()

    from data.providers.base import DataFrequency, DataRequest

    try:
        req = DataRequest.recent(symbol, days=365)
        result = await data_router.get_daily_kline(req)

        if result.data.empty:
            raise HTTPException(404, f"未找到{symbol}的数据")

        # 计算指标
        indicator_result = analyzer.compute_all(result.data, symbol=symbol)
        last_row = indicator_result.to_dataframe().iloc[-1]

        # 关键指标
        key_metrics = {
            "symbol": symbol,
            "date": str(result.data["date"].iloc[-1]),
            "close": float(last_row.get("close", 0)),
            "ma_5": float(last_row.get("ma_5", 0)),
            "ma_20": float(last_row.get("ma_20", 0)),
            "ma_60": float(last_row.get("ma_60", 0)),
            "rsi_14": float(last_row.get("rsi_14", 0)),
            "macd_dif": float(last_row.get("macd_dif", 0)),
            "macd_hist": float(last_row.get("macd_hist", 0)),
            "bias_ma20": float(last_row.get("bias_ma20", 0)),
            "atr_14": float(last_row.get("atr_14", 0)),
            "trend_score": float(last_row.get("trend_score", 0)),
            "composite_score": float(last_row.get("composite_score", 0)),
            "vol_ratio": float(last_row.get("vol_ratio_5", 0)),
        }

        # 形态检测
        patterns = {
            k: int(v.iloc[-1]) if hasattr(v, 'iloc') else 0
            for k, v in indicator_result.patterns.items()
        }
        active_patterns = [k for k, v in patterns.items() if v > 0]
        key_metrics["active_patterns"] = active_patterns[:5]

        return key_metrics

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"查询失败: {e}")


@router.get("/market/regime", response_model=RegimeResponse)
async def market_regime():
    """获取当前市场状态"""
    data_router = get_router()
    detector = get_detector()

    from data.providers.base import DataFrequency, DataRequest

    try:
        req = DataRequest.recent("sh.000300", days=365)
        result = await data_router.get_daily_kline(req)

        if result.data.empty:
            return {"regime": "unknown", "error": "无法获取沪深300数据"}

        regime = detector.detect(result.data)
        params = get_regime_parameters(regime.regime)

        return {
            "regime": regime.regime.value,
            "confidence": regime.confidence,
            "details": regime.details,
            "suggested_params": params,
        }
    except Exception as e:
        raise HTTPException(500, f"市场状态检测失败: {e}")


@router.get("/cost/summary")
async def cost_summary():
    """模型API成本汇总"""
    try:
        from models.cost_monitor import CostMonitor
        monitor = CostMonitor()
        return monitor.daily_report()
    except Exception as e:
        return {"error": str(e)}


# ═══════════════════════════════════════════════════════════════
# 市场时间工具
# ═══════════════════════════════════════════════════════════════

# A股交易时间 (北京时间)
_MORNING_OPEN = time(9, 30)
_MORNING_CLOSE = time(11, 30)
_AFTERNOON_OPEN = time(13, 0)
_AFTERNOON_CLOSE = time(15, 0)

# 2026年中国法定节假日 (A股休市日, 仅含主要长假)
_CN_HOLIDAYS_2026 = {
    date(2026, 1, 1), date(2026, 1, 2),      # 元旦
    date(2026, 2, 16), date(2026, 2, 17), date(2026, 2, 18), date(2026, 2, 19), date(2026, 2, 20),  # 春节 (2.17除夕)
    date(2026, 4, 6),                           # 清明节
    date(2026, 5, 1), date(2026, 5, 4), date(2026, 5, 5),  # 劳动节
    date(2026, 6, 22),                          # 端午节
    date(2026, 9, 28),                          # 中秋节 (9.27中秋)
    date(2026, 10, 1), date(2026, 10, 2), date(2026, 10, 5), date(2026, 10, 6), date(2026, 10, 7),  # 国庆节
}


def _is_market_open(dt: datetime | None = None) -> dict:
    """检测当前是否为A股交易时间, 返回 {is_open, status, detail, last_close_date}"""
    if dt is None:
        dt = datetime.now()

    now_date = dt.date()
    now_time = dt.time()
    weekday = now_date.weekday()  # 0=Mon, 6=Sun

    # 1. 检查是否周末
    if weekday >= 5:
        # 计算最近一个交易日
        days_back = weekday - 4 if weekday >= 5 else 0
        last_trade = now_date - timedelta(days=days_back)
        return {"is_open": False, "status": "weekend", "detail": "周末休市", "last_trade_date": last_trade.isoformat()}

    # 2. 检查是否节假日
    if now_date in _CN_HOLIDAYS_2026:
        last_trade = now_date - timedelta(days=1)
        while last_trade.weekday() >= 5 or last_trade in _CN_HOLIDAYS_2026:
            last_trade -= timedelta(days=1)
        return {"is_open": False, "status": "holiday", "detail": "节假日休市", "last_trade_date": last_trade.isoformat()}

    # 3. 检查交易时段
    if _MORNING_OPEN <= now_time < _MORNING_CLOSE:
        return {"is_open": True, "status": "morning_session", "detail": "早盘交易中 9:30-11:30", "last_trade_date": now_date.isoformat()}
    elif _AFTERNOON_OPEN <= now_time < _AFTERNOON_CLOSE:
        return {"is_open": True, "status": "afternoon_session", "detail": "午盘交易中 13:00-15:00", "last_trade_date": now_date.isoformat()}
    elif now_time < _MORNING_OPEN:
        return {"is_open": False, "status": "pre_open", "detail": f"未开盘, 9:30开盘", "last_trade_date": _last_trade_date(now_date)}
    elif _MORNING_CLOSE <= now_time < _AFTERNOON_OPEN:
        return {"is_open": False, "status": "lunch_break", "detail": "午间休市, 13:00开盘", "last_trade_date": _last_trade_date(now_date)}
    else:
        # after 15:00
        return {"is_open": False, "status": "closed", "detail": "已收盘", "last_trade_date": now_date.isoformat()}


def _last_trade_date(from_date: date) -> str:
    """找到最近的交易日"""
    d = from_date - timedelta(days=1)
    while d.weekday() >= 5 or d in _CN_HOLIDAYS_2026:
        d -= timedelta(days=1)
    return d.isoformat()


# ═══════════════════════════════════════════════════════════════
# 实时行情端点 — 纯JSON, 供前端JS轮询
# ═══════════════════════════════════════════════════════════════

@router.get("/realtime/market")
async def realtime_market():
    """实时市场数据 — 大盘指数 + 涨跌Top5 + 重点股票 + 市场状态"""
    import requests

    market_status = _is_market_open()
    result = {
        "indices": {}, "top_up": [], "top_down": [], "watchlist": [],
        "ts": datetime.now().isoformat(),
        "market_status": market_status,
    }

    idx_codes = {
        "上证指数": "sh000001", "深证成指": "sz399001", "创业板指": "sz399006",
        "科创50": "sh000688", "沪深300": "sh000300",
    }
    watch = [
        "sh600519","sz000858","sh601318","sz300750","sh600036","sz000333","sz000651",
        "sh601088","sh600900","sh601899","sh600031","sz002594","sz002415","sh600276",
        "sz300760","sh601012","sh600030","sz002714","sh600009","sh688111",
    ]

    # v3.1.2: 4 个请求并行 (was sequential ~10s → ~3s)
    # 腾讯直连 (trust_env=False, 免代理快); EastMoney 走代理 (HTTP_PROXY)
    _direct = requests.Session()
    _direct.trust_env = False

    async def _fetch_indices():
        out = {}
        try:
            resp = await asyncio.to_thread(_direct.get,
                f"https://qt.gtimg.cn/q={','.join(idx_codes.values())}", timeout=5,
                headers={"User-Agent": "Mozilla/5.0", "Referer": "https://finance.qq.com/"})
            resp.encoding = "gbk"
            for line in resp.text.strip().split("\n"):
                if "=" not in line or "~" not in line: continue
                try:
                    code = line.split("=", 1)[0].replace("v_", "").strip()
                    fields = line.split("=", 1)[1].strip('"').split("~")
                    if len(fields) < 10: continue
                    name = {v: k for k, v in idx_codes.items()}.get(code, code)
                    out[name] = {
                        "price": round(float(fields[3]), 2) if fields[3] else 0,
                        "pct": round(float(fields[32]), 2) if len(fields) > 32 and fields[32] else 0,
                    }
                except (ValueError, IndexError): continue
        except Exception: pass
        return out

    async def _fetch_movers(fid):
        out = []
        # 1. Sina 优先 (免代理, 稳定快) — v3.1.2: EastMoney 依赖不稳代理
        try:
            asc = "0" if fid == "f3" else "1"  # f3=涨幅榜(降序), f32=跌幅榜(升序)
            url = (f"https://vip.stock.finance.sina.com.cn/quotes_service/api/json_v2.php/"
                   f"Market_Center.getHQNodeData?page=1&num=5&sort=changepercent&asc={asc}"
                   f"&node=hs_a&_s_r_a=init")
            resp = await asyncio.to_thread(requests.get, url, timeout=4,
                headers={"User-Agent": "Mozilla/5.0"})
            import json as _json
            data = _json.loads(resp.text)
            if data:
                return [{"code": str(i.get("code", "")), "name": i.get("name", ""),
                         "price": float(i.get("trade", 0) or 0), "pct": float(i.get("changepercent", 0) or 0)}
                        for i in data[:5]]
        except Exception:
            pass
        # 2. EastMoney 兜底 (经代理)
        try:
            url = (f"https://push2.eastmoney.com/api/qt/clist/get?pn=1&pz=5&po=1&np=1"
                   f"&fltt=2&invt=2&fid={fid}&fs=m:0+t:6,m:0+t:80,m:1+t:2,m:1+t:23"
                   f"&fields=f2,f3,f12,f14")
            resp = await asyncio.to_thread(requests.get, url, timeout=3,
                headers={"User-Agent": "Mozilla/5.0", "Referer": "https://quote.eastmoney.com/"})
            items = (resp.json().get("data", {}).get("diff") or [])
            if items:
                return [{"code": i.get("f12", ""), "name": i.get("f14", ""),
                         "price": i.get("f2", 0), "pct": i.get("f3", 0)} for i in items[:5]]
        except Exception:
            pass
        return out

    async def _fetch_watch():
        out = []
        try:
            resp = await asyncio.to_thread(_direct.get,
                f"https://qt.gtimg.cn/q={','.join(watch)}", timeout=5,
                headers={"User-Agent": "Mozilla/5.0", "Referer": "https://finance.qq.com/"})
            resp.encoding = "gbk"
            for line in resp.text.strip().split("\n"):
                if "=" not in line or "~" not in line: continue
                try:
                    fields = line.split("=", 1)[1].strip('"').split("~")
                    if len(fields) < 10: continue
                    out.append({"name": fields[1], "code": fields[2],
                        "price": round(float(fields[3]), 2) if fields[3] else 0,
                        "pct": round(float(fields[32]), 2) if len(fields) > 32 and fields[32] else 0,
                        "volume": int(fields[6]) if fields[6] else 0})
                except (ValueError, IndexError): continue
        except Exception: pass
        return out

    indices, top_up, top_down, watchlist = await asyncio.gather(
        _fetch_indices(), _fetch_movers("f3"), _fetch_movers("f32"), _fetch_watch(),
    )
    result["indices"] = indices
    result["top_up"] = top_up
    result["top_down"] = top_down
    result["watchlist"] = watchlist
    return result
