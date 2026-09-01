"""
FastAPI REST API — A股智能分析Agent (v6.0 结构重构)

本文件: app 创建 + CORS + 认证/限频中间件 + /health + 路由注册 + schema re-export。
  api/schemas.py      — Pydantic 请求/响应模型 (此处 re-export 保持 `server.XXX` 兼容)
  api/deps.py         — 组件单例工厂 (get_router/get_knowledge/...)
  api/routers/*.py    — 领域路由 (market/analysis/chat/portfolio/decisions/competition/bot)

端点总览:
  GET  /health                       — 健康检查
  GET  /api/v1/stock/{symbol}        — 单股行情
  POST /api/v1/analyze               — 发起分析
  GET  /api/v1/analyze/{task_id}     — 查询分析结果
  POST /api/v1/backtest              — 发起回测
  GET  /api/v1/strategies            — 策略列表
  GET  /api/v1/market/regime         — 当前市场状态
  GET  /api/v1/realtime/market       — 实时行情 (免鉴权白名单)
  GET  /api/v1/portfolio/mtm         — 持仓实时估值
  ...                                — 详见 /docs
"""

import hmac
import os
from datetime import datetime, timedelta
from typing import Dict, List

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger
from dotenv import load_dotenv

# 必须在读取任何环境变量 (API_KEY / CORS_ORIGINS 等) 之前加载 .env
load_dotenv()

# ═══════════════════════════════════════════════════════════════
# 应用初始化
# ═══════════════════════════════════════════════════════════════

app = FastAPI(
    title="A股智能分析Agent API",
    version="3.0-competition",
    description="AI驱动的A股量化分析助手 — REST API (第八届AI智能体开发应用赛)",
)

# CORS: 白名单收敛 (v5.6 P0-8) — 未配置 CORS_ORIGINS 时默认关闭跨域 (生产安全)。
# 仅显式配置才放开; 显式 "*" 会打警告 (应仅用于可信内网/开发环境)。
_CORS_ORIGINS = [o.strip() for o in os.getenv("CORS_ORIGINS", "").split(",") if o.strip()]
if "*" in _CORS_ORIGINS:
    logger.warning("⚠️  CORS_ORIGINS 含通配 '*' — 任意站点可跨域访问, 请仅用于开发环境")

app.add_middleware(
    CORSMiddleware,
    allow_origins=_CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 🔐 API Key 认证 (v5.6 P0-8: 多 key 轮换 + fail-closed)
#   - 支持逗号分隔多 key: API_KEY=keyA,keyB — 任一有效 key 均可通过, 便于滚动轮换
#     (添加新 key 部署 → 客户端切换 → 移除旧 key), 不再静态共享单一密钥。
#   - 常量时间比较 (hmac.compare_digest), 仅接受 X-API-Key 头, 不支持 query string 传 key。
#   - 未配置 API_KEY: 默认拒绝所有非白名单请求 (403);
#     仅当显式设置 API_ALLOW_INSECURE_NO_AUTH=true 时放开无认证开发模式。
_API_KEYS = {k.strip() for k in os.getenv("API_KEY", "").split(",") if k.strip()}
# v3.0: 敏感端点(持仓/推送)移出白名单, 必须 X-API-Key 鉴权 — 此前 portfolio/mtm 与 bot/*
# 未鉴权可读持仓/触发推送, 叠加 CORS 通配后可被任意站点窃取。仅保留健康检查/文档/公开行情。
_AUTH_WHITELIST = {"/health", "/docs", "/openapi.json", "/redoc", "/api/v1/realtime/market"}
_ALLOW_INSECURE_NO_AUTH = os.getenv("API_ALLOW_INSECURE_NO_AUTH", "").lower() in ("1", "true", "yes")

if not _API_KEYS:
    if _ALLOW_INSECURE_NO_AUTH:
        logger.warning("⚠️  API_KEY 未配置 — 处于无认证开发模式 (API_ALLOW_INSECURE_NO_AUTH=true), 请勿用于生产!")
    else:
        logger.error(
            "🔒 API_KEY 未配置 — 已启用 fail-closed 认证: 除白名单外所有请求返回 403。"
            "请设置 API_KEY 环境变量; 本地开发可显式设置 API_ALLOW_INSECURE_NO_AUTH=true 放开。"
        )
else:
    logger.info(f"🔐 API 认证已启用: {len(_API_KEYS)} 个 key (支持滚动轮换)")


# ── v5.6 P0-8: 简单滑动窗口速率限制 (内存态, 无需外部依赖) ──
_RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "60"))
_RATE_LIMIT_BURST = max(_RATE_LIMIT_PER_MINUTE, int(os.getenv("RATE_LIMIT_BURST", "120")))
_rate_hits: Dict[str, List[datetime]] = {}  # client_ip → 最近请求时间戳 (60s 滑动窗口)


def _client_ip(request: Request) -> str:
    """取真实客户端 IP (优先 X-Forwarded-For 首个, 反代场景)"""
    xff = request.headers.get("X-Forwarded-For", "")
    if xff:
        return xff.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def _allow_request(key: str) -> bool:
    """60s 滑动窗口限频: 窗口内请求数 >= burst 则拒绝 (429)"""
    now = datetime.now()
    cutoff = now - timedelta(seconds=60)
    hits = [t for t in _rate_hits.get(key, []) if t > cutoff]
    if len(hits) >= _RATE_LIMIT_BURST:
        _rate_hits[key] = hits
        return False
    hits.append(now)
    _rate_hits[key] = hits
    # 防内存膨胀 (多客户端/反代场景的兜底; 内部 API 通常不会触发)
    if len(_rate_hits) > 10000:
        _rate_hits.clear()
    return True


@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    """API Key 认证 (fail-closed) + 速率限制 (v5.6 P0-8)

    - 仅接受 X-API-Key 头; 支持多 key 轮换; 常量时间比较防时序侧信道。
    - 认证通过后按客户端 IP 限频 (滑动窗口), 超限返回 429。
    """
    if request.url.path in _AUTH_WHITELIST:
        return await call_next(request)

    if not _API_KEYS:
        if _ALLOW_INSECURE_NO_AUTH:
            return await call_next(request)
        return JSONResponse(
            status_code=403,
            content={"error": "Forbidden",
                     "detail": "服务端未配置 API_KEY; 请设置 API_KEY 环境变量后重启"},
        )

    api_key = request.headers.get("X-API-Key", "")
    if not api_key or not any(hmac.compare_digest(api_key, k) for k in _API_KEYS):
        return JSONResponse(
            status_code=401,
            content={"error": "Unauthorized", "detail": "Invalid or missing API key (use X-API-Key header)"},
        )

    if not _allow_request(_client_ip(request)):
        return JSONResponse(
            status_code=429,
            content={"error": "RateLimited", "detail": "请求过于频繁, 请稍后再试"},
        )
    return await call_next(request)


# ═══════════════════════════════════════════════════════════════
# Schema re-export (v6.0: 模型迁至 api/schemas.py, 此处保持 `server.XXX` 兼容)
# ═══════════════════════════════════════════════════════════════

from api.schemas import (  # noqa: E402
    AnalyzeRequest,
    AnalyzeResponse,
    BacktestRequest,
    BacktestResponse,
    ChatHistoryResponse,
    ChatRequest,
    ChatResponse,
    CompetitionBenchmarkResponse,
    DecisionsResponse,
    HealthResponse,
    RegimeResponse,
    StockInfoResponse,
    StrategiesResponse,
    TaskResult,
)

__all__ = [
    "app",
    "AnalyzeRequest", "AnalyzeResponse", "BacktestRequest", "BacktestResponse",
    "ChatHistoryResponse", "ChatRequest", "ChatResponse",
    "CompetitionBenchmarkResponse", "DecisionsResponse", "HealthResponse",
    "RegimeResponse", "StockInfoResponse", "StrategiesResponse", "TaskResult",
]


# ═══════════════════════════════════════════════════════════════
# 健康检查 (须在中间件之后、路由注册处定义; 白名单放行)
# ═══════════════════════════════════════════════════════════════

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """系统健康检查"""
    status = {
        "status": "healthy",
        "version": "2.14.0",
        "timestamp": datetime.now().isoformat(),
    }

    # 检查数据源
    try:
        from api.deps import get_router
        router = get_router()
        ds_status = router.status
        status["data_sources"] = ds_status
    except Exception as e:
        status["data_sources"] = {"error": str(e)}

    # 模型状态 (v3.0 统一 DeepSeek V4-Flash)
    try:
        from models import DEEPSEEK_FLASH_MODEL
        status["model"] = {
            "provider": "deepseek",
            "model": DEEPSEEK_FLASH_MODEL,
            "configured": bool(os.getenv("DEEPSEEK_API_KEY", "")),
        }
    except Exception:
        status["model"] = {"configured": False}

    return status


# ═══════════════════════════════════════════════════════════════
# 领域路由注册 (path 与拆分前逐字一致)
# ═══════════════════════════════════════════════════════════════

from api.routers import analysis, bot, chat, competition, decisions, market, portfolio  # noqa: E402

app.include_router(market.router)
app.include_router(analysis.router)
app.include_router(chat.router)
app.include_router(portfolio.router)
app.include_router(decisions.router)
app.include_router(competition.router)
app.include_router(bot.router)


# 启动
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
