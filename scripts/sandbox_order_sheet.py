"""冻结选券链路的人工操作单。默认 dry-run，只有 --confirm 才写操作单。

所有输入必须是真实文件；缺数据即拒绝，不连接券商、不执行交易。
输入契约、真实行情准备与使用方法见 docs/order_sheet_usage.md。
dry-run 只打印操作单，但每次调用仍追加审计日志。
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import sys
import tempfile
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, time, timedelta, timezone
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
TOP_K = 5
KEEP_ZONE = 10
COST_BPS = 65.0  # 冻结研究成本口径；现金校验另按真实费用。
UNIVERSE_N = 800
LOT = 100
FRESH_MAX_NATDAYS = 7
MARKET_TZ = timezone(timedelta(hours=8))
CENT = Decimal("0.01")
SYMBOL_RE = re.compile(r"(?:sh|sz|bj)\.\d{6}\Z")


class OrderSheetError(ValueError):
    """缺失数据或安全检查不通过，不可继续生成操作单。"""


@dataclass(frozen=True)
class Limits:
    max_order_amount: Decimal = Decimal("5000")
    max_daily_amount: Decimal = Decimal("10000")
    max_quote_age_seconds: int = 300
    max_price_deviation: Decimal = Decimal("0.05")
    min_commission: Decimal = Decimal("5")


@dataclass(frozen=True)
class Order:
    symbol: str
    side: str
    price: Decimal
    shares: int

    @property
    def amount(self) -> Decimal:
        return self.price * self.shares


@dataclass
class Validation:
    errors: list[str] = field(default_factory=list)
    checks: list[dict[str, Any]] = field(default_factory=list)
    buy_required: Decimal = Decimal("0")
    sell_fee_shortfall: Decimal = Decimal("0")
    sell_net: Decimal = Decimal("0")
    gross_amount: Decimal = Decimal("0")

    @property
    def passed(self) -> bool:
        return not self.errors


def _decimal(value: Any, label: str, *, positive: bool = False) -> Decimal:
    try:
        if isinstance(value, bool):
            raise InvalidOperation
        result = Decimal(str(value))
    except (InvalidOperation, ValueError):
        raise OrderSheetError(f"{label}: 必须是有效数值") from None
    if not result.is_finite() or result < 0 or (positive and result == 0):
        raise OrderSheetError(f"{label}: 必须是有限{'正' if positive else '非负'}数")
    return result


def _shares(value: Any, label: str) -> int:
    if type(value) is not int or value < 0:
        raise OrderSheetError(f"{label}: 必须是非负整数股数")
    return value


def _timestamp(value: Any, label: str) -> datetime:
    try:
        result = datetime.fromisoformat(str(value))
    except ValueError:
        raise OrderSheetError(f"{label}: 缺少有效 ISO 时间戳") from None
    if result.tzinfo is None:
        raise OrderSheetError(f"{label}: 时间戳必须含时区")
    return result.astimezone(MARKET_TZ)


def _safe_path(value: str | Path) -> Path:
    original = Path(value).absolute()
    resolved = original.resolve()
    for path in (original, resolved):
        if "reports" in {p.casefold() for p in path.parts} or path.name.casefold() in {".env", ".mcp.json"}:
            raise OrderSheetError("禁止访问 reports/、.env 或 .mcp.json")
    return resolved


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                      allow_nan=False, default=str).encode("utf-8")


def _load_json(raw: bytes, label: str) -> dict:
    def reject_constant(value: str):
        raise OrderSheetError(f"{label}: 禁止非有限数值 {value}")

    try:
        value = json.loads(raw.decode("utf-8-sig"), parse_constant=reject_constant)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise OrderSheetError(f"{label}: 无效 JSON ({type(exc).__name__})") from None
    if not isinstance(value, dict):
        raise OrderSheetError(f"{label}: 顶层必须是对象")
    return value


def fees(amount: Decimal, side: str, limits: Limits) -> dict[str, Decimal]:
    """双向过户费；逐项精确到分，佣金默认最低 5 元。"""
    return {
        "commission": max(amount * Decimal("0.0003"), limits.min_commission).quantize(CENT, ROUND_HALF_UP),
        "stamp_duty": (amount * (Decimal("0.0005") if side == "sell" else Decimal("0"))).quantize(CENT, ROUND_HALF_UP),
        "transfer": (amount * Decimal("0.00001")).quantize(CENT, ROUND_HALF_UP),
    }


def validate_orders(orders: list[Order], quotes: dict, cash: Decimal,
                    positions: dict[str, dict], rebalance_date: date, now: datetime,
                    limits: Limits, daily_traded_amount: Decimal = Decimal("0")) -> Validation:
    """不缩量、不改选券，不假定卖出已成交来提供买入资金。"""
    result = Validation()
    cash = _decimal(cash, "可用现金")
    market_now = now.astimezone(MARKET_TZ)
    if rebalance_date != market_now.date():
        result.errors.append("换仓日必须是当前中国市场日期，禁止回填或预写真单")
    local_time = market_now.time()
    if market_now.weekday() >= 5 or not (time(9, 30) <= local_time <= time(11, 30)
                                       or time(13) <= local_time <= time(15)):
        result.errors.append("当前不在工作日交易时段 09:30–11:30 / 13:00–15:00，禁止生成可执行操作单")
    if not isinstance(quotes.get("source"), str) or not quotes["source"].strip():
        result.errors.append("行情缺少真实来源 source")
    quote_map = quotes.get("quotes")
    if not isinstance(quote_map, dict):
        result.errors.append("行情 quotes 映射缺失")
        quote_map = {}
    seen: set[str] = set()
    for order in orders:
        issues: list[str] = []
        try:
            if not SYMBOL_RE.fullmatch(order.symbol):
                raise OrderSheetError("股票代码必须是市场.六位代码")
            if order.symbol in seen:
                raise OrderSheetError("同一股票有重复指令，拒绝重复买卖")
            seen.add(order.symbol)
            if order.side not in {"buy", "sell"}:
                raise OrderSheetError("方向只能是 buy/sell")
            shares = _shares(order.shares, "指令股数")
            if shares == 0 or shares % LOT:
                issues.append("整数手校验失败：必须是 100 股的正整数倍")
            price = _decimal(order.price, "生成价", positive=True)
            quote = quote_map.get(order.symbol)
            if not isinstance(quote, dict):
                raise OrderSheetError("缺少真实实时行情")
            live_price = _decimal(quote.get("price"), "实时价", positive=True)
            observed = _timestamp(quote.get("observed_at"), "行情 observed_at")
            age = (now - observed).total_seconds()
            if observed.date() != rebalance_date or age < 0 or age > limits.max_quote_age_seconds:
                issues.append(f"实时行情过期或来自未来：observed_at={observed.isoformat()}")
            deviation = abs(price / live_price - 1)
            if deviation > limits.max_price_deviation:
                issues.append(f"生成价偏离实时行情超过 ±5%：{deviation:.2%}")
            for key, label in (("suspended", "停牌"), ("is_st", "ST")):
                if type(quote.get(key)) is not bool:
                    issues.append(f"缺少明确{label}状态，禁止推断为可交易")
                elif quote[key]:
                    issues.append(f"{label}状态：不可执行")
            status_source = quote.get("status_source")
            if not isinstance(status_source, str) or not status_source.strip():
                issues.append("缺少交易状态的真实来源 status_source")
            status_at = _timestamp(quote.get("status_observed_at"), "交易状态 status_observed_at")
            status_age = (now - status_at).total_seconds()
            if status_at.date() != rebalance_date or not 0 <= status_age <= limits.max_quote_age_seconds:
                issues.append("交易状态过期或来自未来")
            lower = _decimal(quote.get("limit_down"), "跌停价", positive=True)
            upper = _decimal(quote.get("limit_up"), "涨停价", positive=True)
            if lower >= upper:
                issues.append("涨跌停价格无效")
            if live_price >= upper:
                issues.append("涨停状态：不可执行")
            if live_price <= lower:
                issues.append("跌停状态：不可执行")
            if not lower < price < upper:
                issues.append("生成价触及或超出涨跌停范围：不可执行")
            # 两种报价中的较高金额控制上限和买入现金，避免低估最新报价。
            conservative_amount = max(price, live_price) * shares
            result.gross_amount += conservative_amount
            if conservative_amount > limits.max_order_amount:
                issues.append(f"单笔金额超限：{conservative_amount:.2f} > {limits.max_order_amount:.2f}")
            if order.side == "buy":
                result.buy_required += conservative_amount + sum(fees(conservative_amount, "buy", limits).values())
            else:
                held = positions.get(order.symbol, {})
                available = _shares(held.get("available_shares"), "持仓可卖股数")
                if shares > available:
                    issues.append(f"可卖持仓不足（含 T+1 限制）：{shares} > {available}")
                conservative_sell_amount = min(price, live_price) * shares
                net = conservative_sell_amount - sum(fees(conservative_sell_amount, "sell", limits).values())
                result.sell_net += net
                result.sell_fee_shortfall += max(-net, Decimal("0"))
        except (OrderSheetError, TypeError) as exc:
            issues.append(str(exc))
        result.checks.append({"symbol": order.symbol, "side": order.side,
                              "status": "passed" if not issues else "rejected", "errors": issues})
        result.errors.extend(f"{order.symbol}: {issue}" for issue in issues)
    if result.gross_amount + daily_traded_amount > limits.max_daily_amount:
        result.errors.append(f"单日总金额超限（买卖双向合计）：本单 {result.gross_amount:.2f} + "
                             f"已成交 {daily_traded_amount:.2f} > {limits.max_daily_amount:.2f}")
    needed_cash = result.buy_required + result.sell_fee_shortfall
    if needed_cash > cash:
        result.errors.append(f"可用现金不足（含买入费用及卖出费差额，不预支待卖收入）：需 {needed_cash:.2f}，可用 {cash:.2f}")
    return result


def _account(raw: dict, edge: str, rebalance_date: date, now: datetime) -> tuple[Decimal, dict[str, dict]]:
    if raw.get("edge_id") != edge:
        raise OrderSheetError("账户 edge_id 与 --edge 不一致")
    try:
        if date.fromisoformat(raw.get("entry_date", "")) > rebalance_date:
            raise OrderSheetError("持仓 entry_date 晚于换仓日")
    except (ValueError, TypeError):
        raise OrderSheetError("账户 entry_date 无效或晚于换仓日") from None
    as_of = _timestamp(raw.get("as_of"), "账户 as_of")
    if as_of.date() != rebalance_date or as_of > now:
        raise OrderSheetError("账户现金/持仓必须为换仓日已核实的真实快照")
    cash = _decimal(raw.get("available_cash"), "账户 available_cash")
    if not isinstance(raw.get("positions"), list):
        raise OrderSheetError("账户 positions 必须明确提供列表（真实空仓填 []）")
    positions = {}
    for position in raw["positions"]:
        if not isinstance(position, dict) or not SYMBOL_RE.fullmatch(str(position.get("symbol"))):
            raise OrderSheetError("持仓必须明确提供 symbol=市场.六位代码")
        symbol = position["symbol"]
        if symbol in positions:
            raise OrderSheetError(f"持仓重复: {symbol}")
        shares = _shares(position.get("shares"), f"{symbol} shares")
        available = _shares(position.get("available_shares"), f"{symbol} available_shares")
        if shares <= 0 or available > shares:
            raise OrderSheetError(f"{symbol}: 持仓数量或可卖数量无效")
        positions[symbol] = {"shares": shares, "available_shares": available}
    return cash, positions


def select_orders(df: pd.DataFrame, ev: pd.DataFrame, positions: dict[str, dict],
                  capital: float, rebalance_date: date) -> tuple[list[Order], dict]:
    """冻结原脚本的 universe、叠层、三因子、buffer、slot/整数手数值逻辑。"""
    required = {"date", "symbol", "isST", "tradestatus", "turn", "amount", "close", "peTTM", "pbMRQ", "pctChg"}
    if df.empty or not required.issubset(df.columns):
        raise OrderSheetError(f"真实面板为空或缺列: {sorted(required - set(df.columns))}")
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    if df["date"].isna().any() or df.duplicated(["symbol", "date"]).any():
        raise OrderSheetError("面板含缺失日期或重复 symbol/date")
    T = df["date"].max()
    stale = (pd.Timestamp(rebalance_date) - T).days
    if stale < 0 or stale > FRESH_MAX_NATDAYS:
        raise OrderSheetError(f"面板过期或来自未来: {T.date()}, 间隔 {stale} 天")
    latest = df[df["date"] == T].copy()
    d = latest.copy()
    d["symbol"] = d["symbol"].astype(str)
    d = d[d["symbol"].str.startswith(("sh.6", "sz.0"))]
    d = d[(d["isST"].astype(str) == "0") & (d["tradestatus"].astype(str) == "1")]
    for c in ("turn", "amount", "close", "peTTM", "pbMRQ", "pctChg"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d[(d["turn"] > 0) & (d["amount"] > 0) & (d["peTTM"] > 0) & (d["pbMRQ"] > 0)]
    universe = d.nlargest(UNIVERSE_N, "amount").copy()

    hist = df[df["date"] <= T].sort_values(["symbol", "date"]).copy()
    hist["std20"] = hist.groupby("symbol")["pctChg"].rolling(20, min_periods=20).std().reset_index(level=0, drop=True)
    last_std = hist.dropna(subset=["std20"]).groupby("symbol")["std20"].last()
    universe["std20"] = universe["symbol"].map(last_std)
    universe = universe[universe["std20"].notna()].copy()

    big = ev[ev["ratio"] >= 0.05]
    ev_in = big[(big["free_date"] >= T) & (big["free_date"] <= T + pd.Timedelta(days=60))]
    blocked = set(ev_in["sym"])
    n_before = len(universe)
    universe = universe[~universe["symbol"].isin(blocked)].copy()
    n_blocked = n_before - len(universe)
    if universe.empty:
        raise OrderSheetError("无可用真实候选：检查面板历史及解禁数据，不生成空白替代单")

    r_turn = universe["turn"].rank(pct=True)
    r_std = universe["std20"].rank(pct=True)
    r_mkt = np.log1p(universe["amount"] * 100.0 / universe["turn"] / 1e8).rank(pct=True)
    universe["score"] = -(r_turn + r_std + r_mkt)
    universe = universe.sort_values("score", ascending=False).reset_index(drop=True)
    ranked_syms = universe["symbol"].tolist()
    rank_of = {s: i for i, s in enumerate(ranked_syms)}
    px_of = dict(zip(universe["symbol"], universe["close"]))

    held = set(positions)
    unknown = held - set(rank_of)
    drop_syms_oop = sorted(unknown)
    held = held - unknown
    keep = sorted((s for s in held if rank_of[s] < KEEP_ZONE), key=rank_of.get)
    drop_syms = sorted(held - set(keep), key=rank_of.get)
    slot = capital / TOP_K
    max_px = slot / LOT
    buys = []
    for s in ranked_syms:
        if len(keep) + len(buys) >= TOP_K:
            break
        if s in keep or s in {b[0] for b in buys}:
            continue
        px = float(_decimal(px_of[s], f"{s} 面板收盘价", positive=True))
        if px > max_px:
            continue
        buys.append((s, px))

    # 原脚本没有真实卖出股数；仅取账户实际数量，禁止“一手示意”。
    px_latest = dict(zip(latest["symbol"], latest["close"]))
    orders = [Order(s, "sell", _decimal(px_latest.get(s), f"{s} 卖出生成价", positive=True),
                    positions[s]["shares"]) for s in drop_syms_oop + drop_syms]
    for s, px in buys:
        shares = int(slot / px // LOT) * LOT
        orders.append(Order(s, "buy", Decimal(str(px)), shares))
    return orders, {"panel_date": str(T.date()), "stale_days": stale, "keep": keep,
                    "n_before": n_before, "n_blocked": n_blocked, "n_after": len(universe),
                    "slot": slot, "max_price": max_px, "rank_of": rank_of}


def _events(sources: list[bytes]) -> pd.DataFrame:
    frames = [pd.read_parquet(io.BytesIO(raw)) for raw in sources]
    if not frames or all(frame.empty for frame in frames):
        raise OrderSheetError("真实解禁数据为空")
    ev = pd.concat(frames, ignore_index=True)
    if {"股票代码", "解禁时间", "占解禁前流通市值比例"}.issubset(ev.columns):
        ev["code"] = ev["股票代码"].astype(str).str.zfill(6)
        ev = ev[ev["code"].str.startswith(("60", "00"))].copy()
        ev["free_date"] = pd.to_datetime(ev["解禁时间"], errors="coerce")
        ev["ratio"] = pd.to_numeric(ev["占解禁前流通市值比例"], errors="coerce")
        ev["sym"] = ev["code"].map(lambda c: ("sh." if c.startswith("6") else "sz.") + c)
    elif {"sym", "free_date", "ratio"}.issubset(ev.columns):
        ev["free_date"] = pd.to_datetime(ev["free_date"], errors="coerce")
        ev["ratio"] = pd.to_numeric(ev["ratio"], errors="coerce")
    else:
        raise OrderSheetError("解禁数据缺少 sym/free_date/ratio 或对应原始中文列")
    if ev.empty or ev[["sym", "free_date", "ratio"]].isna().any().any() or not np.isfinite(ev["ratio"]).all():
        raise OrderSheetError("解禁数据缺失/无效，不能当作没有解禁事件")
    return ev.drop_duplicates(subset=["sym", "free_date"])


def render_sheet(orders: list[Order], context: dict, account: dict, quotes: dict,
                 validation: Validation, snapshot_hash: str, rebalance_date: date,
                 limits: Limits) -> bytes:
    if not validation.passed:
        raise OrderSheetError("校验失败，禁止渲染可执行操作单")
    lines = [f"# 人工操作单 — {rebalance_date.isoformat()}", "",
             f"- 生成日期 / 换仓日: {rebalance_date.isoformat()}（中国市场日期）",
             f"- 输入快照 SHA-256: {snapshot_hash}",
             f"- 面板 as_of: {context['panel_date']}；账户 as_of: {account['as_of']}",
             f"- 持仓: {account['edge_id']} / entry {account['entry_date']}",
             f"- 实时行情来源: {quotes['source']}",
             f"- 冻结规则: TOP_K={TOP_K}, KEEP_ZONE={KEEP_ZONE}, universe={UNIVERSE_N}, "
             f"解禁未来60自然日≥5%, 研究成本口径={COST_BPS}bp/单边（不作为现金费用）",
             f"- 候选: {context['n_before']} → 剔除解禁 {context['n_blocked']} → {context['n_after']}；"
             f"slot=¥{context['slot']:.2f}；一手价上限=¥{context['max_price']:.2f}",
             f"- 保留持仓: {', '.join(context['keep']) or '无'}", "",
             "| 方向 | 股票 | 股数 | 生成参考价 | 实时价 | 行情时点 | 参考金额 | 费用 | 状态 |",
             "|---|---|---:|---:|---:|---|---:|---:|---|"]
    for order in orders:
        quote = quotes["quotes"][order.symbol]
        cost = sum(fees(order.amount, order.side, limits).values())
        lines.append(f"| {'买入' if order.side == 'buy' else '卖出'} | {order.symbol} | {order.shares} | "
                     f"{order.price:.2f} | {Decimal(str(quote['price'])):.2f} | {quote['observed_at']} | "
                     f"{order.amount:.2f} | {cost:.2f} | 校验通过：非停牌/非ST/未触涨跌停 |")
    if not orders:
        lines.append("| 无换仓指令 | — | — | — | — | — | — | — | 保留持仓 |")
    lines += ["", "## 资金与边界",
              "- 佣金万3（最低 ¥" + str(limits.min_commission) + "）；卖出印花税万5；双向过户费 0.00001，逐项四舍五入到分。",
              f"- 已核实可用现金: ¥{Decimal(str(account['available_cash'])):.2f}；"
              f"买入所需（较高报价 + 费用）: ¥{validation.buy_required:.2f}；"
              f"卖出费用需补现金: ¥{validation.sell_fee_shortfall:.2f}；"
              f"剩余现金: ¥{Decimal(str(account['available_cash'])) - validation.buy_required - validation.sell_fee_shortfall:.2f}",
              f"- 卖出保守净额（较低报价扣费）: ¥{validation.sell_net:.2f}（未计入可用现金）",
              f"- 单笔上限: ¥{limits.max_order_amount:.2f}；单日买卖合计上限: ¥{limits.max_daily_amount:.2f}；"
              f"已成交: ¥{Decimal(str(account['daily_traded_amount'])):.2f}；本单保守买卖金额: ¥{validation.gross_amount:.2f}",
              "- 仅供人工核对后在券商 App 手动执行；本工具不连接券商、不下单。",
              "- 本文件为同一换仓日的完整计划，重复运行不表示新增指令；同一审计账本一天只允许确认一个输入快照。",
              "- 报价会变化；人工执行前再次确认价格、状态、资金和可卖数量。", ""]
    return "\n".join(lines).encode("utf-8")


def _write_once(path: Path, content: bytes) -> str:
    """原子创建；已有同内容可复用，已有不同快照拒绝覆盖。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() == content:
            return "unchanged"
        raise OrderSheetError("输出路径已有不同内容，拒绝覆盖")
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".order-sheet-", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != content:
                raise OrderSheetError("并发输出冲突：已有不同操作单，拒绝覆盖") from None
            return "unchanged"
        return "written"
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _append_audit(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as stream:
        stream.write(_json_bytes(record) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())


@contextmanager
def _audit_lock(audit: Path):
    """同一账本串行化日限额检查/预留/写出；崩溃遗留锁须人工核查。"""
    lock = _safe_path(audit.with_name(audit.name + ".lock"))
    lock.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise OrderSheetError("审计账本已锁定（并发或上次异常退出），请先核查现有操作单") from None
    try:
        os.close(descriptor)
        yield
    finally:
        lock.unlink()


def _check_daily_plan(audit: Path, rebalance_date: date, snapshot_hash: str) -> None:
    if not audit.exists():
        return
    for line in audit.read_bytes().splitlines():
        if not line.strip():
            continue
        entry = _load_json(line, "审计账本")
        if (entry.get("rebalance_date") == str(rebalance_date)
                and entry.get("status") in {"reserved", "written", "unchanged"}
                and entry.get("input_snapshot_hash") != snapshot_hash):
            raise OrderSheetError("同日已有不同快照的已确认/预留操作单，拒绝分次生成绕过单日上限")


def _independent_paths(output: Path, audit: Path, input_paths: dict[str, Path], events: Path | None) -> None:
    lock = _safe_path(audit.with_name(audit.name + ".lock"))
    targets = [output, audit, lock]
    for index, target in enumerate(targets):
        for other in targets[index + 1:] + list(input_paths.values()):
            if target == other or (target.exists() and other.exists() and target.samefile(other)):
                raise OrderSheetError("操作单、审计日志/锁和输入文件的路径必须彼此独立")
        if events is not None and events.is_dir() and target.is_relative_to(events):
            raise OrderSheetError("禁止向解禁输入目录写操作单、审计日志或锁")


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--panel", type=Path, default=ROOT / "replay_data" / "live_panel.parquet")
    ap.add_argument("--events", type=Path, help="真实解禁 parquet 或分片目录（必填）")
    ap.add_argument("--account", type=Path, help="当日真实现金/持仓 JSON（必填）")
    ap.add_argument("--quotes", type=Path, help="新鲜真实行情 JSON，含时点/状态/涨跌停价（必填）")
    ap.add_argument("--capital", default="10000", help="冻结 slot 分配资金，默认 10000")
    ap.add_argument("--edge", default="cold_lowvol_top5_hold")
    ap.add_argument("--rebalance-date", help="YYYY-MM-DD；默认当前中国市场日期")
    ap.add_argument("--max-order-amount", default="5000")
    ap.add_argument("--max-daily-amount", default="10000", help="含账户当日已成交金额及本次买卖合计")
    ap.add_argument("--max-quote-age-seconds", type=int, default=300)
    ap.add_argument("--min-commission", default="5")
    ap.add_argument("--output", type=Path, help="默认 simulation_data/order_sheets/换仓日.md")
    ap.add_argument("--audit-log", type=Path, default=ROOT / "simulation_data" / "order_sheets" / "audit.jsonl")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="仅预览（默认）；仍追加审计")
    mode.add_argument("--confirm", action="store_true", help="校验全部通过后写出人工操作单；不会下单")
    return ap


def main(argv: list[str] | None = None, *, now: datetime | None = None) -> int:
    args = parser().parse_args(argv)
    fixed_now = now  # 仅供离线单测注入；生产每个关键关口重新读时钟。
    now = now or datetime.now(MARKET_TZ)
    record: dict[str, Any] = {"time": now.isoformat(), "input_snapshot_hash": None,
                              "output_path": None, "mode": "confirm" if args.confirm else "dry-run",
                              "validation": {"passed": False, "errors": [], "checks": []},
                              "status": "rejected"}
    audit = None
    written_output = None
    try:
        candidate_audit = _safe_path(args.audit_log)
        rebalance_date = date.fromisoformat(args.rebalance_date) if args.rebalance_date else now.astimezone(MARKET_TZ).date()
        record["rebalance_date"] = str(rebalance_date)
        output = _safe_path(args.output or ROOT / "simulation_data" / "order_sheets" / f"{rebalance_date}.md")
        record["output_path"] = str(output)
        required_paths = {"panel": args.panel, "events": args.events, "account": args.account, "quotes": args.quotes}
        paths = {key: _safe_path(path) for key, path in required_paths.items() if path is not None}
        event_paths = []
        if "events" in paths:
            event_paths = sorted(paths["events"].glob("*.parquet")) if paths["events"].is_dir() else [paths["events"]]
        input_paths = {**{key: path for key, path in paths.items() if key != "events"},
                       **{f"events/{i}": _safe_path(path) for i, path in enumerate(event_paths)}}
        _independent_paths(output, candidate_audit, input_paths, paths.get("events"))
        audit = candidate_audit  # 所有路径检查先于任何读取/日志写入，包括输入缺失时。
        capital = _decimal(args.capital, "capital", positive=True)
        limits = Limits(_decimal(args.max_order_amount, "单笔上限", positive=True),
                        _decimal(args.max_daily_amount, "单日上限", positive=True),
                        args.max_quote_age_seconds, min_commission=_decimal(args.min_commission, "最低佣金"))
        if not 1 <= limits.max_quote_age_seconds <= 300:
            raise OrderSheetError("行情新鲜度必须在 1~300 秒之间，不能扩大为陈旧缓存")
        parameters = {"rebalance_date": str(rebalance_date), "capital": str(capital), "edge": args.edge,
                      "limits": asdict(limits), "format_version": 1}
        sources: dict[str, str] = {}
        record["input_snapshot_hash"] = hashlib.sha256(_json_bytes({"parameters": parameters, "sources": sources})).hexdigest()
        missing = [f"--{key}" for key, path in required_paths.items() if path is None]
        if missing:
            raise OrderSheetError("缺少真实输入 " + ", ".join(missing) + "；零模拟，拒绝生成")
        if not event_paths:
            raise OrderSheetError("解禁分片目录为空；零模拟，拒绝生成")
        blobs = {}
        for label, path in input_paths.items():
            blobs[label] = path.read_bytes()
            sources[label] = hashlib.sha256(blobs[label]).hexdigest()
            record["input_snapshot_hash"] = hashlib.sha256(_json_bytes({"parameters": parameters, "sources": sources})).hexdigest()
        account = _load_json(blobs["account"], "账户")
        quote_snapshot = _load_json(blobs["quotes"], "行情")
        cash, positions = _account(account, args.edge, rebalance_date, now)
        traded = _decimal(account.get("daily_traded_amount"), "账户 daily_traded_amount（真实当日已成交买卖金额）")
        panel = pd.read_parquet(io.BytesIO(blobs["panel"]))
        events = _events([blob for label, blob in blobs.items() if label.startswith("events/")])
        orders, context = select_orders(panel, events, positions, float(capital), rebalance_date)
        validation = validate_orders(orders, quote_snapshot, cash, positions, rebalance_date,
                                     fixed_now or datetime.now(MARKET_TZ), limits, traded)
        record["validation"] = {"passed": validation.passed, "errors": validation.errors, "checks": validation.checks}
        if not validation.passed:
            raise OrderSheetError("\n".join(validation.errors))
        content = render_sheet(orders, context, account, quote_snapshot, validation,
                               record["input_snapshot_hash"], rebalance_date, limits)
        with _audit_lock(audit):
            _check_daily_plan(audit, rebalance_date, record["input_snapshot_hash"])
            if args.confirm and output.exists() and output.read_bytes() != content:
                raise OrderSheetError("输出路径已有不同内容，拒绝覆盖")
            # 面板计算/账本读取可能较慢；临写出再检查时段与行情，不能沿用启动时间。
            validation = validate_orders(orders, quote_snapshot, cash, positions, rebalance_date,
                                         fixed_now or datetime.now(MARKET_TZ), limits, traded)
            record["validation"] = {"passed": validation.passed, "errors": validation.errors, "checks": validation.checks}
            if not validation.passed:
                raise OrderSheetError("\n".join(validation.errors))
            # 写入预留必须先成功；崩溃/写出失败保守保留当日快照，不自动释放额度。
            record["status"] = "reserved" if args.confirm else "previewed"
            _append_audit(audit, record)
            if args.confirm:
                validation = validate_orders(orders, quote_snapshot, cash, positions, rebalance_date,
                                             fixed_now or datetime.now(MARKET_TZ), limits, traded)
                if not validation.passed:
                    record["validation"] = {"passed": False, "errors": validation.errors, "checks": validation.checks}
                    raise OrderSheetError("\n".join(validation.errors))
                record["status"] = _write_once(output, content)
                written_output = output
                _append_audit(audit, record)
        print(content.decode("utf-8"), end="")
        print(f"\n{'已写出/复用' if args.confirm else 'dry-run：未写出操作单'} -> {output}")
        return 0
    except Exception as exc:  # fail closed: 禁止补占位数据或输出部分操作单。
        record["status"] = "rejected"
        record["error"] = str(exc)
        record["validation"]["passed"] = False
        if not record["validation"]["errors"]:
            record["validation"]["errors"] = [str(exc)]
        if written_output is not None:
            print(f"操作单文件已写出/复用，但完成审计失败：{written_output}；勿执行，请核查预留记录。原因: {exc}", file=sys.stderr)
        else:
            print(f"拒绝生成操作单: {exc}", file=sys.stderr)
        if audit is not None:
            try:
                with _audit_lock(audit):
                    _append_audit(audit, record)
            except (OSError, OrderSheetError) as audit_exc:
                print(f"审计写入失败: {type(audit_exc).__name__}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
