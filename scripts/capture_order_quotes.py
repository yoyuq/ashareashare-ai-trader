"""只读获取腾讯真实报价，合并已核实的实时交易状态；不接券商，不下单。

python scripts/capture_order_quotes.py --status-snapshot <真实状态.json> --output <行情.json>
状态输入契约见 docs/order_sheet_usage.md。缺失/过期数据一律拒绝；输出禁止覆盖。
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from datetime import datetime
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.sandbox_order_sheet import (  # noqa: E402
    MARKET_TZ, SYMBOL_RE, OrderSheetError, _decimal, _json_bytes, _load_json,
    _safe_path, _timestamp, _write_once,
)


def capture_quotes(statuses: dict, *, now: datetime | None = None, session=None) -> dict:
    """保留 provider 行情时间与原响应，绝不用 HTTP 接收时刻替代行情时刻。"""
    fixed_now = now
    now = now or datetime.now(MARKET_TZ)
    source = statuses.get("source")
    if not isinstance(source, str) or not source.strip():
        raise OrderSheetError("状态快照必须说明真实来源 source")
    states = statuses.get("statuses")
    if not isinstance(states, dict) or not states:
        raise OrderSheetError("状态快照 statuses 不能为空")
    for symbol, state in states.items():
        if not SYMBOL_RE.fullmatch(symbol) or not isinstance(state, dict):
            raise OrderSheetError("状态快照证券代码/对象无效")
        for flag in ("suspended", "is_st"):
            if type(state.get(flag)) is not bool:
                raise OrderSheetError(f"{symbol} 缺少明确的 {flag} 状态")
        at = _timestamp(state.get("observed_at"), f"{symbol} 状态时点")
        if at.date() != now.astimezone(MARKET_TZ).date() or not 0 <= (now - at).total_seconds() <= 300:
            raise OrderSheetError(f"{symbol} 交易状态过期或来自未来")
    url = "https://qt.gtimg.cn/q=" + ",".join(s.replace(".", "") for s in sorted(states))
    own_session = session is None
    if own_session:
        session = requests.Session()
        session.trust_env = False
    try:
        response = session.get(url, timeout=15)
        response.raise_for_status()
        response.encoding = "gbk"
        raw = response.text
    finally:
        if own_session:
            session.close()
    now = fixed_now or datetime.now(MARKET_TZ)
    quotes = {}
    for line in raw.strip().splitlines():
        if '="' not in line:
            continue
        key, value = line.strip().split('="', 1)
        code = key.removeprefix("v_")
        symbol = code[:2] + "." + code[2:]
        if symbol not in states:
            continue
        fields = value.rstrip('";').split("~")
        if len(fields) < 49 or fields[2] != code[2:]:
            raise OrderSheetError(f"{symbol} 腾讯原始行情字段缺失/不匹配")
        if symbol in quotes:
            raise OrderSheetError(f"{symbol} 腾讯响应重复")
        price = _decimal(fields[3], f"{symbol} 实时价", positive=True)
        upper = _decimal(fields[47], f"{symbol} 涨停价", positive=True)
        lower = _decimal(fields[48], f"{symbol} 跌停价", positive=True)
        try:
            observed = datetime.strptime(fields[30], "%Y%m%d%H%M%S").replace(tzinfo=MARKET_TZ)
        except ValueError:
            raise OrderSheetError(f"{symbol} 缺少真实行情时间戳") from None
        if observed.date() != now.astimezone(MARKET_TZ).date() or not 0 <= (now - observed).total_seconds() <= 300:
            raise OrderSheetError(f"{symbol} 腾讯报价过期或来自未来: {observed.isoformat()}")
        if lower >= upper or not lower <= price <= upper:
            raise OrderSheetError(f"{symbol} 腾讯价格/涨跌停字段无效")
        state = states[symbol]
        status_at = _timestamp(state["observed_at"], f"{symbol} 状态时点")
        if status_at.date() != now.astimezone(MARKET_TZ).date() or not 0 <= (now - status_at).total_seconds() <= 300:
            raise OrderSheetError(f"{symbol} HTTP 请求完成时交易状态已过期")
        if ("ST" in fields[1].upper()) != state["is_st"]:
            raise OrderSheetError(f"{symbol} 腾讯名称与状态快照的 ST 标记不一致")
        quotes[symbol] = {"price": str(price), "observed_at": observed.isoformat(),
                          "limit_up": str(upper), "limit_down": str(lower), "name": fields[1],
                          "suspended": state["suspended"], "is_st": state["is_st"],
                          "status_source": source, "status_observed_at": state["observed_at"]}
    missing = set(states) - set(quotes)
    if missing:
        raise OrderSheetError(f"腾讯缺少真实行情: {', '.join(sorted(missing))}")
    return {"source": url, "raw_response": raw,
            "raw_response_sha256": hashlib.sha256(raw.encode("utf-8")).hexdigest(),
            "status_snapshot_sha256": hashlib.sha256(_json_bytes(statuses)).hexdigest(), "quotes": quotes}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--status-snapshot", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args(argv)
    try:
        source = _safe_path(args.status_snapshot)
        output = _safe_path(args.output)
        if source == output or (source.exists() and output.exists() and source.samefile(output)):
            raise OrderSheetError("行情输出不能覆盖状态输入")
        captured = capture_quotes(_load_json(source.read_bytes(), "交易状态"))
        _write_once(output, _json_bytes(captured) + b"\n")
        print(f"已保存真实行情 {len(captured['quotes'])} 只 -> {output}")
        return 0
    except Exception as exc:
        print(f"真实行情不可用，未生成替代数据: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
