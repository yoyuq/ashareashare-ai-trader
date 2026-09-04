"""分钟级短线采集器 — 自建分钟/涨停池真历史 (历史数据源缺失, 今日起自建, 零模拟).

每次调用 (Windows 任务计划每 10 分钟, 交易日 09:30-15:00):
  1. 从昨日日线面板 (live_panel) 构造当日 watchlist: 昨日主板涨停 ∪ 昨日主板换手top10
     ∪ 冷落低波前瞻持仓; 全程点内数据, 无未来函数。
  2. 逐票抓腾讯分时 (web.ifzq.gtimg.cn, 免代理, 返回当日全天 1 分钟线, 幂等)。
     日期守卫: 接口返回日期 != 今天 → 不写 (节假日/盘前天然 no-op)。
  3. upsert 到 replay_data/minute_live/YYYY-MM-DD.parquet。
  4. ≥14:55 时额外抓全市场快照 → 市场广度 (涨跌停家数/上涨占比) 追加
     simulation_data/sentiment_live.jsonl (冰点信号数据基础)。
  5. 交易信号判定写入 simulation_data/forward_validation/shortline_journal.jsonl:
     10:00 后判入场 (前30分钟涨幅>2% 且量≥昨日全天10%), 14:55 后判卖出并记账
     (65bp 往返)。规则冻结于 prereg_shortline_intraday.md, 不得修改。

失败策略: 单票失败记日志跳过 (缺口=缺口, 不模拟); 全部失败 exit 2。
用法: python scripts/minute_collector.py [--date YYYY-MM-DD]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

MINUTE_DIR = ROOT / "replay_data" / "minute_live"
BOOK_DIR = ROOT / "replay_data" / "book_live"
JOURNAL_FP = ROOT / "simulation_data" / "forward_validation" / "shortline_journal.jsonl"
SENTIMENT_FP = ROOT / "simulation_data" / "sentiment_live.jsonl"
PANEL_FP = ROOT / "replay_data" / "live_panel.parquet"
REGISTRY_FP = ROOT / "simulation_data" / "forward_validation" / "registry.json"

ENTRY_MIN = "10:00"          # 入场判定时点
EXIT_MIN = "14:55"           # 尾盘卖出时点
GAIN_TH = 0.02               # 前30分钟累计涨幅阈值 (冻结)
VOL_FRAC_TH = 0.10           # 前30分钟量/昨日全天量阈值 (冻结)
COST_RT = 0.0065             # 往返成本 65bp (全项目统一口径)
# 指数分钟线采集 (数据管道增强, 非冻结规则; 不入 watchlist 不参与判定)。
# 用途: 判定日检验 A 股日内时序动量 (r1/r7 → 尾盘半小时, Zhang & Zhu 2018)。
INDEX_CODES = ["sh.000001", "sz.399006"]

_wl_cache: dict = {}

# 强制直连 (bat/环境变量代理残留会使腾讯分时整体失败)
_SESSION = requests.Session()
_SESSION.trust_env = False


def _force_utf8() -> None:
    for name in ("stdout", "stderr"):
        s = getattr(sys, name, None)
        if hasattr(s, "reconfigure"):
            try:
                s.reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass


def _is_mainboard(sym: str) -> bool:
    return sym.startswith("sh.60") or sym.startswith("sz.00")


def last_two_days(today: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    """最近两个 < today 的交易日面板 (d0=最近日, d1=其前一日, 供 prev_close)。"""
    df = pd.read_parquet(PANEL_FP)
    df["date"] = pd.to_datetime(df["date"])
    days = sorted(d for d in df["date"].unique() if d.strftime("%Y-%m-%d") < today)
    if len(days) < 2:
        raise RuntimeError("live_panel 不足两日, 无法点内构造 watchlist")
    d0, d1 = days[-1], days[-2]
    return df[df["date"] == d0], df[df["date"] == d1], d0


def build_watchlist(today: str) -> list[str]:
    """最近交易日主板涨停 (vs 前一日收盘) ∪ 主板换手top10 ∪ 冷落低波持仓。"""
    if today in _wl_cache:
        return _wl_cache[today]
    d0, d1, actual_day = last_two_days(today)
    if (datetime.now().date() - actual_day.date()).days >= 4:
        raise RuntimeError(f"live_panel 基期陈旧 ({actual_day.date()}), 拒绝点内构造 — 先跑 fetch_live_panel.py")
    y = d0[d0["isST"].astype(str) == "0"].copy()
    prev_close = d1.set_index("symbol")["close"]
    y["prev_close"] = y["symbol"].map(prev_close)
    y = y.dropna(subset=["prev_close"])
    y = y[y["symbol"].map(_is_mainboard)]
    if y.empty:
        raise RuntimeError(f"watchlist 构建为空 (基期 {actual_day.date()})")
    limit_up = y[y["close"] >= y["prev_close"] * 1.095]["symbol"].tolist()
    top_turn = y.sort_values("turn", ascending=False).head(10)["symbol"].tolist()
    hold: list[str] = []
    try:
        reg = json.loads(REGISTRY_FP.read_text(encoding="utf-8"))
        bet = reg.get("bets", {}).get("cold_lowvol_top5_hold", {})
        # registry 不存篮子 (tracker 每日重算); 仅当未来显式存入 latest_basket 时并入
        hold = list(bet.get("latest_basket", []))
    except Exception:
        pass
    wl = sorted(set(limit_up) | set(top_turn) | set(hold))
    _wl_cache[today] = wl
    return wl


def fetch_minute(sym: str) -> tuple[str, list[tuple]] | None:
    """腾讯分时: 返回 (date, [(HH:MM, price, cum_volume, cum_amount), ...])。"""
    code = sym.replace("sh.", "sh").replace("sz.", "sz")
    url = f"https://web.ifzq.gtimg.cn/appstock/app/minute/query?code={code}"
    r = _SESSION.get(url, timeout=10)
    r.raise_for_status()
    j = r.json()
    d = j["data"][code]["data"]
    rows = []
    for line in d["data"]:
        parts = line.split()
        # HH:MM price cum_vol(手) cum_amount(元)
        rows.append((parts[0], float(parts[1]), float(parts[2]), float(parts[3])))
    return d.get("date"), rows


def upsert_minute(today: str, sym: str, rows: list[tuple]) -> int:
    MINUTE_DIR.mkdir(parents=True, exist_ok=True)
    fp = MINUTE_DIR / f"{today}.parquet"
    new = pd.DataFrame(rows, columns=["time", "price", "cum_volume", "cum_amount"])
    new.insert(0, "symbol", sym)
    if fp.exists():
        old = pd.read_parquet(fp)
        old = old[old["symbol"] != sym]
        out = pd.concat([old, new], ignore_index=True)
    else:
        out = new
    out.to_parquet(fp, index=False)
    return len(new)


def capture_breadth(today: str, now: datetime) -> None:
    """≥14:55 全市场广度快照 (涨跌停近似/上涨占比), 每交易日仅一条 (同日去重)。"""
    if SENTIMENT_FP.exists():
        for line in SENTIMENT_FP.read_text(encoding="utf-8").splitlines():
            if line.strip() and json.loads(line).get("date") == today:
                return  # 已有当日快照, 不重复追加
    from refresh_market_cache import fetch_tencent, bootstrap_symbols
    quotes = fetch_tencent(bootstrap_symbols())
    if not quotes or len(quotes) < 3000:
        raise RuntimeError(f"全市场快照不足: {len(quotes) if quotes else 0}")
    # 分母含平盘股 (0.0), 仅剔缺失; up_ratio 分母=可判定家数
    chg = [q["pct_change"] for q in quotes.values()
           if q.get("pct_change") is not None]
    up = sum(1 for c in chg if c > 0)
    down = sum(1 for c in chg if c < 0)
    limit_up = sum(1 for c in chg if c >= 9.5)
    limit_dn = sum(1 for c in chg if c <= -9.5)
    rec = {"date": today, "n": len(chg), "up": up, "down": down,
           "limit_up": limit_up, "limit_dn": limit_dn,
           "up_ratio": round(up / len(chg), 4),
           "captured_at": now.strftime("%H:%M:%S")}
    SENTIMENT_FP.parent.mkdir(parents=True, exist_ok=True)
    with SENTIMENT_FP.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def judge_trades(today: str, now: datetime, wl: list[str]) -> None:
    """预注册交易判定: 10:00 入场 / 14:55 卖出记账。"""
    fp = MINUTE_DIR / f"{today}.parquet"
    if not fp.exists():
        return
    m = pd.read_parquet(fp)
    jrn = [json.loads(l) for l in JOURNAL_FP.read_text(encoding="utf-8").splitlines()
           if l.strip()] if JOURNAL_FP.exists() else []
    done_keys = {(r["date"], r["symbol"], r["event"]) for r in jrn}

    # 昨日 close / amount 一次读入 (点内, 无未来函数)
    df = pd.read_parquet(PANEL_FP)
    df["date"] = pd.to_datetime(df["date"])
    days = sorted(d for d in df["date"].unique() if d.strftime("%Y-%m-%d") < today)
    if not days:
        return
    yd = df[df["date"] == days[-1]].set_index("symbol")

    for sym in wl:
        sub = m[m["symbol"] == sym].sort_values("time")
        if sub.empty or sym not in yd.index:
            continue
        pc = float(yd.loc[sym, "close"]) if pd.notna(yd.loc[sym, "close"]) else None
        pv = float(yd.loc[sym, "amount"]) if pd.notna(yd.loc[sym, "amount"]) else None
        if not pc or pc <= 0:
            continue
        e = sub[sub["time"] <= ENTRY_MIN]
        x = sub[sub["time"] >= EXIT_MIN]
        events_today = {k[2] for k in done_keys if k[0] == today and k[1] == sym}
        # 入场判定
        if "in" not in events_today and now.strftime("%H:%M") >= ENTRY_MIN and not e.empty:
            p10 = e["price"].iloc[-1]
            gain30 = p10 / pc - 1
            vol_frac = (e["cum_amount"].iloc[-1] / pv) if pv else None
            trig = gain30 > GAIN_TH and vol_frac is not None and vol_frac >= VOL_FRAC_TH
            rec = {"date": today, "symbol": sym, "event": "in",
                   "price": p10, "gain30": round(gain30, 4),
                   "vol_frac": None if vol_frac is None else round(vol_frac, 4),
                   "triggered": bool(trig)}
            append_journal(rec)
            jrn.append(rec)  # 同步内存, 尾窗同轮 in→out 判定可见 (防 15:05 收窗丢笔)
            events_today.add("in")
        # 卖出判定 (仅对已触发的入场)
        ins = [r for r in jrn if r["date"] == today and r["symbol"] == sym
               and r["event"] == "in" and r["triggered"]]
        if (ins and "out" not in events_today
                and now.strftime("%H:%M") >= EXIT_MIN and not x.empty):
            p_in = ins[0]["price"]
            p_out = x["price"].iloc[-1]
            net = (p_out / p_in) * (1 - COST_RT) - 1  # 往返成本一次计 (全项目口径)
            append_journal({"date": today, "symbol": sym, "event": "out",
                            "price": p_out, "net_ret": round(net, 4)})


def capture_orderbook(today: str, now: datetime, wl: list[str]) -> None:
    """每次运行抓 watchlist 五档快照 (qt.gtimg 批量) → book_live/YYYY-MM-DD.parquet。

    自建盘口历史 (封单/竞价数据是历史数据源最大空白): 09:25-09:30 的运行 = 竞价
    快照; 封板判定 price>=limit_up 时 seal_vol=买一量 (封单), 否则 0。
    """
    if not wl:
        return
    import refresh_market_cache as R
    sess = requests.Session()
    sess.trust_env = False
    codes = [s.replace("sh.", "sh").replace("sz.", "sz") for s in wl]
    r = sess.get(f"https://qt.gtimg.cn/q={','.join(codes)}", timeout=10,
                 headers={"User-Agent": "Mozilla/5.0", "Referer": "https://finance.qq.com/"})
    r.encoding = "gbk"
    rows = []
    for line in r.text.strip().split("\n"):
        if "=" not in line or "~" not in line:
            continue
        try:
            f = line.split("=", 1)[1].strip('"').split("~")
            if len(f) < 49:
                continue
            code = f[2]
            sym = ("sh." if code.startswith("6") else "sz.") + code
            price, limit_up = float(f[3] or 0), float(f[47] or 0)
            rows.append({
                "ts": now.strftime("%H:%M:%S"), "symbol": sym,
                "price": price, "pct": float(f[32] or 0),
                "bid1_p": float(f[9] or 0), "bid1_vol": float(f[10] or 0),
                "ask1_p": float(f[19] or 0), "ask1_vol": float(f[20] or 0),
                "high": float(f[33] or 0), "low": float(f[34] or 0),
                "limit_up": limit_up, "limit_dn": float(f[48] or 0),
                "sealed_up": price > 0 and limit_up > 0 and price >= limit_up,
                "seal_vol": float(f[10] or 0) if (price > 0 and limit_up > 0
                                                  and price >= limit_up) else 0.0,
            })
        except (ValueError, IndexError):
            continue
    if not rows:
        raise RuntimeError("盘口快照解析为空")
    bdir = ROOT / "replay_data" / "book_live"
    bdir.mkdir(parents=True, exist_ok=True)
    fp = bdir / f"{today}.parquet"
    new = pd.DataFrame(rows)
    if fp.exists():
        out = pd.concat([pd.read_parquet(fp), new], ignore_index=True)
    else:
        out = new
    out.to_parquet(fp, index=False)


def append_journal(rec: dict) -> None:
    JOURNAL_FP.parent.mkdir(parents=True, exist_ok=True)
    rec["ts"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with JOURNAL_FP.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main() -> int:
    _force_utf8()
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=None)
    args = ap.parse_args()
    now = datetime.now()
    today = args.date or now.strftime("%Y-%m-%d")
    if now.weekday() >= 5 and args.date is None:
        print("周末 no-op")
        return 0
    hm = now.strftime("%H:%M")
    if not (args.date or ("09:25" <= hm <= "15:10")):
        print(f"非交易时段 {hm} no-op")
        return 0

    wl = build_watchlist(today)
    ok, fail = 0, 0
    for sym in wl:
        try:
            d, rows = fetch_minute(sym)
            if d != today.replace("-", ""):
                fail += 1  # 接口日期非今日 (盘前/节假日): 不写
                continue
            upsert_minute(today, sym, rows)
            ok += 1
        except Exception as e:
            fail += 1
            print(f"{sym} 抓取失败: {e}")
        time.sleep(0.15)
    if ok == 0 and fail == len(wl) and wl:
        print("全部失败")
        return 2

    # 指数分钟线 (管道增强; 失败仅记日志, 不影响个股采集与判定)
    for idx in INDEX_CODES:
        try:
            d, rows = fetch_minute(idx)
            if d == today.replace("-", ""):
                upsert_minute(today, idx, rows)
                print(f"指数 {idx} 分时已存 ({len(rows)} 行)")
        except Exception as e:
            print(f"指数 {idx} 抓取失败: {e}")

    if not args.date and hm >= "14:55":
        try:
            capture_breadth(today, now)
            print("广度快照已写")
        except Exception as e:
            print(f"广度快照失败: {e}")
    try:
        capture_orderbook(today, now, wl)
    except Exception as e:
        print(f"盘口快照失败: {e}")
    if not args.date and hm >= "10:00":
        judge_trades(today, now, wl)
    print(f"watchlist={len(wl)} 成功={ok} 跳过/失败={fail} @ {hm}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
