"""抓取冷落候选池的公募基金持股市值历史 — 预注册 reports/agent_loop/prereg_inst_cold_intersection.md.

流程:
1. 用 harness 面板 + 定型 score, 对 ≥2016 的每个窗口每月末取 score 前 30 候选池, 汇总 unique symbols;
2. 逐票调 datapro MCP (HTTP JSON-RPC) 查公募基金持股市值全部报告期;
3. 落 replay_data/inst_holdings.parquet + 失败日志 (重试2次后按无覆盖计, 计数披露)。

断点续跑: 已抓票跳过。全程零模拟 — 端点不可用即报错退出。
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

import strategy_research_harness as H  # noqa: E402

OUT_DIR = ROOT / "replay_data"
OUT_PARQUET = OUT_DIR / "inst_holdings.parquet"
CKPT = OUT_DIR / "inst_holdings_checkpoint.json"
URL = "https://datapro.hqd.cn-beijing.volces.com/mcp"
KEY = os.getenv("AGENT_PLAN_KEY", "")
if not KEY:
    print("FATAL: AGENT_PLAN_KEY missing in .env")
    sys.exit(1)

WINDOWS_FROM_2016 = [w for w in H.WINDOWS if not w.startswith("2015")]
TOPN_POOL = 30  # 预注册候选池 = score 前 30


def _force_utf8() -> None:
    for name in ("stdout", "stderr"):
        s = getattr(sys, name, None)
        if hasattr(s, "reconfigure"):
            try:
                s.reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass


def _score(df: pd.DataFrame) -> pd.Series:
    out = df[["date", "symbol", "turn", "amount", "close"]].copy()
    s = df.groupby("symbol")["close"].rolling(20).std().reset_index(level=0, drop=True)
    out["std20"] = s.reindex(df.index)
    fmkt = df["amount"] * 100.0 / df["turn"].replace(0, np.nan) / 1e8
    out["log_mkt"] = np.log1p(fmkt)
    r_turn = out["turn"].groupby(df["date"]).rank(pct=True)
    r_std = out["std20"].groupby(df["date"]).rank(pct=True)
    r_mkt = out["log_mkt"].groupby(df["date"]).rank(pct=True)
    return -(r_turn + r_std + r_mkt)


def candidate_symbols() -> set[str]:
    """各窗每月末 score 前 30 的 unique symbols。"""
    syms: set[str] = set()
    for window in WINDOWS_FROM_2016:
        df = H.load_base_window(H.WINDOWS[window])
        if df.empty:
            raise RuntimeError(f"窗口数据为空: {H.WINDOWS[window]}")
        score = _score(df)
        d = df.copy()
        d["score"] = score.values
        dates = sorted(d["date"].unique())
        s = pd.Series(pd.DatetimeIndex(dates))
        month_ends = s.groupby([s.dt.year, s.dt.month]).apply(lambda x: x.iloc[-1]).tolist()
        for T in month_ends:
            g = d[d["date"] == T].dropna(subset=["score"])
            if len(g) < 50:
                continue
            top = g.sort_values("score", ascending=False).head(TOPN_POOL)
            syms.update(top["symbol"].tolist())
        print(f"[pool] {window}: cum unique={len(syms)}")
    return syms


def to_sec_code(symbol: str) -> str:
    """sh.600519 -> 600519.SH"""
    code = symbol.split(".")[1]
    mkt = "SH" if symbol.startswith("sh.") else ("BJ" if symbol.startswith("bj.") else "SZ")
    return f"{code}.{mkt}"


class MCPClient:
    def __init__(self):
        self.headers = {"Content-Type": "application/json",
                        "Accept": "application/json, text/event-stream",
                        "X-Agent-Plan-Key": KEY}
        self.sid: str | None = None
        self._id = 0
        self.tool: str | None = None

    def _parse_body(self, text: str):
        if text.startswith("event:") or "\ndata:" in text or text.startswith("data:"):
            lines = [ln for ln in text.splitlines() if ln.startswith("data:")]
            text = lines[-1][5:].strip() if lines else text
        return json.loads(text)

    def rpc(self, method: str, params: dict | None = None, notify: bool = False):
        h = dict(self.headers)
        if self.sid:
            h["Mcp-Session-Id"] = self.sid
        payload: dict = {"jsonrpc": "2.0", "method": method}
        if not notify:
            self._id += 1
            payload["id"] = self._id
        if params is not None:
            payload["params"] = params
        r = requests.post(URL, headers=h, data=json.dumps(payload), timeout=60)
        new_sid = r.headers.get("Mcp-Session-Id")
        if new_sid:
            self.sid = new_sid
        if r.status_code not in (200, 202):  # 202 Accepted = 通知的正常响应
            raise RuntimeError(f"MCP HTTP {r.status_code}: {r.text[:200]}")
        body = r.content.decode("utf-8", errors="replace").strip()
        if not body:  # 通知无响应体
            return {}
        return self._parse_body(body)

    def init(self):
        resp = self.rpc("initialize", {
            "protocolVersion": "2024-11-05", "capabilities": {},
            "clientInfo": {"name": "ashare-fetch", "version": "0.1"},
        })
        if "result" not in resp:
            raise RuntimeError(f"initialize failed: {str(resp)[:300]}")
        self.rpc("notifications/initialized", notify=True)
        resp = self.rpc("tools/list")
        tools = resp.get("result", {}).get("tools", [])
        if not tools:
            raise RuntimeError(f"no tools: {str(resp)[:300]}")
        self.tool = tools[0]["name"]

    def query(self, sec_code: str, periods: list[str]) -> dict:
        """显式列期查询 (实测"历年全部报告期"措辞只回最近4期)。

        返回 {period_id: value_元} (公募基金持股市值)。
        """
        labels = {"1": "一季报", "2": "中报", "3": "三季报", "4": "年报"}
        pl = " ".join(f"{p.split('Q')[0]}{labels[p.split('Q')[1]]}" for p in periods)
        q = f"{sec_code} 公募基金持股市值 {pl}"
        resp = self.rpc("tools/call", {"name": self.tool, "arguments": {"query": q}})
        result = resp.get("result", {})
        content = result.get("content", [])
        text = "".join(c.get("text", "") for c in content if isinstance(c, dict))
        data = json.loads(text)
        if data.get("code") != 0:
            raise RuntimeError(f"api code={data.get('code')}: {data.get('msg', '')[:120]}")
        out: dict = {}
        for item in data.get("items", []):
            for rec in item.get("records", []):
                if rec.get("indicator_name") != "公募基金持股市值":
                    continue
                period = (rec.get("period") or {}).get("period_id")
                if period:
                    out[period] = rec.get("value")
        return out

    def query_all_periods(self, sec_code: str, wanted: list[str]) -> dict:
        """42 期分块 (10期/次, 实测上限) 逐块查询。"""
        out: dict = {}
        for i in range(0, len(wanted), 10):
            chunk = wanted[i:i + 10]
            got = self.query(sec_code, chunk)
            out.update(got)
            time.sleep(0.3)
        return out


def main() -> int:
    _force_utf8()
    if not OUT_DIR.exists():
        OUT_DIR.mkdir(parents=True)
    syms = sorted(candidate_symbols())
    print(f"total unique candidates: {len(syms)}")
    ckpt: dict = {}
    if CKPT.exists():
        ckpt = json.loads(CKPT.read_text(encoding="utf-8"))
        print(f"checkpoint: {len(ckpt)} already fetched")

    client = MCPClient()
    client.init()
    print(f"mcp tool: {client.tool}")

    records: list[dict] = []  # rows: symbol, period, value
    failures: dict[str, str] = {}
    wanted = [f"{y}Q{q}" for y in range(2016, 2027) for q in (1, 2, 3, 4)]
    n_new = 0
    consec_fail = 0
    # 限速退避: 失败不落 checkpoint (fail≠无覆盖, 杜绝污染), 退避重试, 连续失败熔断
    BACKOFFS = [20, 60, 180, 300]
    for i, sym in enumerate(syms):
        if sym in ckpt:
            continue
        sec = to_sec_code(sym)
        periods = None
        last_err = ""
        for attempt, wait in enumerate([0] + BACKOFFS):
            if wait:
                print(f"[retry] {sym} attempt={attempt + 1} wait={wait}s ({last_err[:80]})")
                time.sleep(wait)
            try:
                periods = client.query_all_periods(sec, wanted)
                last_err = ""
                break
            except Exception as e:
                last_err = str(e)[:200]
        if periods is None:
            # 仍失败: 不落 checkpoint, 计数; 连续失败熔断 (服务端封锁时止损, 断点续跑)
            failures[sym] = last_err
            consec_fail += 1
            print(f"[fail] {sym} consec={consec_fail} ({last_err[:100]})")
            if consec_fail >= 8:
                print("ABORT: 连续8票失败 — 服务端疑似封锁, 保存进度退出 (断点续跑)")
                _save(records, ckpt, failures)
                (OUT_DIR / "inst_holdings_failures.json").write_text(
                    json.dumps(failures, ensure_ascii=False, indent=2), encoding="utf-8")
                return 1
            continue
        consec_fail = 0
        for p, v in periods.items():
            records.append({"symbol": sym, "period": p, "value": v})
        ckpt[sym] = "ok" if periods else "empty"
        n_new += 1
        if n_new % 25 == 0:
            print(f"[{i + 1}/{len(syms)}] new={n_new} ok={sum(1 for v in ckpt.values() if v == 'ok')}"
                  f" empty={sum(1 for v in ckpt.values() if v == 'empty')} fail={len(failures)}")
            _save(records, ckpt, failures)  # 增量落盘 (断点续跑)
        time.sleep(1.0)  # 限速礼貌间隔 (限速升级后从0.4s放宽)

    _save(records, ckpt, failures)
    n_ok = sum(1 for v in ckpt.values() if v == "ok")
    n_empty = sum(1 for v in ckpt.values() if v == "empty")
    print(json.dumps({"total": len(syms), "ok": n_ok, "empty": n_empty, "fail": len(failures)},
                     ensure_ascii=False))
    if failures:
        (OUT_DIR / "inst_holdings_failures.json").write_text(
            json.dumps(failures, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0 if not failures else 1


def _save(records: list[dict], ckpt: dict, failures: dict) -> None:
    df = pd.DataFrame(records, columns=["symbol", "period", "value"])
    if not df.empty:
        df.to_parquet(OUT_PARQUET, index=False)
    CKPT.write_text(json.dumps(ckpt, ensure_ascii=False), encoding="utf-8")
    print(f"[save] rows={len(df)} fetched={len(ckpt)} @{datetime.now().strftime('%H:%M:%S')}")


if __name__ == "__main__":
    raise SystemExit(main())
