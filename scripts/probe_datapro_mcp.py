"""datapro MCP HTTP endpoint 协议探测 (JSON-RPC over POST).

只探测协议可达性 + 一条真实查询, 不落研究数据。
"""
import json
import os
import sys
from pathlib import Path

import requests

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")
KEY = os.getenv("AGENT_PLAN_KEY", "")
if not KEY:
    print("FATAL: AGENT_PLAN_KEY missing in .env")
    sys.exit(1)

URL = "https://datapro.hqd.cn-beijing.volces.com/mcp"
HEADERS = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream",
           "X-Agent-Plan-Key": KEY}
sid = None
_id = 0


def rpc(method, params=None, notify=False):
    global _id, sid
    h = dict(HEADERS)
    if sid:
        h["Mcp-Session-Id"] = sid
    payload = {"jsonrpc": "2.0", "method": method}
    if not notify:
        _id += 1
        payload["id"] = _id
    if params is not None:
        payload["params"] = params
    r = requests.post(URL, headers=h, data=json.dumps(payload), timeout=30)
    new_sid = r.headers.get("Mcp-Session-Id")
    if new_sid:
        sid = new_sid
    body = r.content.decode("utf-8", errors="replace")  # 无 charset 响应 requests 会猜错编码
    # SSE 帧解析
    if body.startswith("event:") or "\ndata:" in body or body.startswith("data:"):
        lines = [ln for ln in body.splitlines() if ln.startswith("data:")]
        body = lines[-1][5:].strip() if lines else body
    try:
        return r.status_code, json.loads(body)
    except (json.JSONDecodeError, ValueError):
        return r.status_code, body[:500]


def main():
    # 1. initialize
    code, resp = rpc("initialize", {
        "protocolVersion": "2024-11-05",
        "capabilities": {},
        "clientInfo": {"name": "ashare-probe", "version": "0.1"},
    })
    print("initialize:", code, str(resp)[:300])
    if code != 200:
        sys.exit(2)
    rpc("notifications/initialized", notify=True)

    # 2. tools/list
    code, resp = rpc("tools/list")
    tools = resp.get("result", {}).get("tools", []) if isinstance(resp, dict) else []
    print("tools:", [t.get("name") for t in tools])
    if not tools:
        print("raw:", str(resp)[:500])
        sys.exit(3)
    tool_name = tools[0]["name"]
    schema = tools[0].get("inputSchema", {})
    print("tool:", tool_name, "schema keys:", list(schema.get("properties", {}).keys()))

    # 3. tools/call — 一条真实查询 (单票 10 期, 验证返回形状)
    q = ("贵州茅台(600519.SH) 公募基金持股市值 "
         "2016一季报 2016中报 2016三季报 2016年报 2017一季报 "
         "2017中报 2017三季报 2017年报 2018一季报 2018中报")
    code, resp = rpc("tools/call", {"name": tool_name, "arguments": {"query": q}})
    print("call:", code)
    result = resp.get("result", {}) if isinstance(resp, dict) else {}
    text = "".join(c.get("text", "") for c in result.get("content", []) if isinstance(c, dict))
    try:
        data = json.loads(text)
        items = data.get("items", [])
        it0 = items[0] if items else {}
        print("keys of items[0]:", list(it0.keys()))
        if "records" in it0:
            recs = [r for r in it0["records"] if r.get("indicator_name") == "公募基金持股市值"]
            print("records periods:", [(r["period"]["period_id"], r["value"]) for r in recs])
        if "table" in it0:
            tbl = it0["table"]
            for k, v in tbl.items():
                if "公募" in k:
                    print("table key:", k, "->", v)
    except (json.JSONDecodeError, ValueError, KeyError, IndexError) as e:
        print("parse failed:", e)
        print(text[:800])


if __name__ == "__main__":
    main()
