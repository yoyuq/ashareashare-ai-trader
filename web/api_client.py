"""Dashboard ↔ REST API 共享 HTTP 层 (v6.0 拆分自 dashboard.py)

统一 X-API-Key 头 + 超时 + JSON 解析, 替换原先 8 处手拼 requests 样板。
白名单端点 (如 /api/v1/realtime/market) 带上 key 也无害, 因此全部请求统一带鉴权头。
"""
import os

import requests

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")


def api_get(path: str, timeout: float = 10):
    """GET {API_BASE_URL}{path} → json; 非 2xx 抛异常 (调用方按需 try/except)"""
    r = requests.get(f"{API_BASE_URL}{path}", timeout=timeout,
                     headers={"X-API-Key": os.getenv("API_KEY", "")})
    r.raise_for_status()
    return r.json()


def api_post(path: str, payload: dict | None = None, timeout: float = 10):
    """POST {API_BASE_URL}{path} → json; 非 2xx 抛异常"""
    r = requests.post(f"{API_BASE_URL}{path}", json=payload, timeout=timeout,
                      headers={"X-API-Key": os.getenv("API_KEY", "")})
    r.raise_for_status()
    return r.json()
