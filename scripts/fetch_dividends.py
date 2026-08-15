"""抓取真实分红数据 (Baostock, 免代理) — 用于"全收益"基准 (阶段1 #107 基准纪律)。

理论背景: replay 日线是不复权 (adjustflag=3), 除权除息日价格下跳但没把现金红利加回,
净值曲线的"价格收益"系统性低估真实总回报约 2~3%/年; 上证综指也是价格指数(不含分红),
"跑赢上证"因此虚高。要诚实回答"跑赢市场总回报", 需要把分红加回。

本脚本抓 universe (各窗口 top-800 流动性可投资) 的每股税前现金红利 + 除权除息日,
供 `analysis/benchmark.py` 构造总收益基准。全程真实数据, 缺数据跳过不兜底、不模拟。

可断点续传: 每 300 只写 checkpoint (replay_data/dividends_checkpoint.parquet), 重跑自动跳过已完成。
进度日志: reports/dividend_fetch.log (追加)。

用法: python scripts/fetch_dividends.py
输出: replay_data/dividends.parquet (code / dividOperateDate / dividCashPsBeforeTax)
"""
from __future__ import annotations

import socket
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
OUT = ROOT / "replay_data" / "dividends.parquet"
CHECKPOINT = ROOT / "replay_data" / "dividends_checkpoint.parquet"
LOGFILE = ROOT / "reports" / "dividend_fetch.log"

YEARS = [str(y) for y in range(2015, 2027)]  # 窗口覆盖 2015-2026


def _log(msg: str):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    LOGFILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LOGFILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def _universe_symbols(root: Path) -> list[str]:
    """各窗口 top-800 流动性可投资 universe 的并集 (与 A/B 脚本 _liquid_universe 同口径)。"""
    syms: set[str] = set()
    for f in sorted((root / "replay_data").glob("daily_*.parquet")):
        df = pd.read_parquet(f)
        if "symbol" not in df.columns:
            continue
        df = df[df.symbol != "sh.000001"].copy()
        df["isST"] = df["isST"].astype(str)
        inv = df[(df.isST == "0") & (df.is_trade == 1) & (df.peTTM > 0) & (df.pbMRQ > 0)]
        med = inv.groupby("symbol")["amount"].median().sort_values(ascending=False)
        syms.update(med.head(800).index.tolist())
    return sorted(syms)


def _query_stock(bs, code: str) -> list[dict]:
    """单只股票 2015-2026 全分红记录 (除权除息日 + 每股税前现金红利)。

    网络异常向上抛 (由 main 决定重连), 不再静默吞掉 —— 否则 socket 超时后连接死掉,
    后续年份全部空转 20s×N 才返回, 拖垮整体进度。
    """
    rows: list[dict] = []
    for yr in YEARS:
        rs = bs.query_dividend_data(code=code, year=yr, yearType="report")
        while rs.error_code == "0" and rs.next():
            r = rs.get_row_data()
            d = dict(zip(rs.fields, r))
            op = d.get("dividOperateDate", "")
            cash = d.get("dividCashPsBeforeTax", "")
            if not op or not cash or cash in ("", "0", "0.0", "0.000000"):
                continue
            try:
                c = float(cash)
            except (ValueError, TypeError):
                continue
            if c <= 0:
                continue
            rows.append({"code": code, "dividOperateDate": op, "dividCashPsBeforeTax": c})
    return rows


def main() -> int:
    import baostock as bs

    symbols = _universe_symbols(ROOT)
    _log(f"universe 并集 {len(symbols)} 只")

    # 断点续传
    done: set[str] = set()
    rows: list[dict] = []
    if CHECKPOINT.exists():
        cp = pd.read_parquet(CHECKPOINT)
        done = set(cp["code"].unique())
        rows = cp.to_dict("records")
        _log(f"续传: 已完成 {len(done)} 只 / {len(rows)} 条分红")

    todo = [s for s in symbols if s not in done]
    _log(f"待抓取 {len(todo)} 只")

    socket.setdefaulttimeout(20)  # baostock 原始 socket 无超时, 网络挂起会永久阻塞 → 20s 兜底
    bs.login()
    t0 = time.time()
    try:
        for i, code in enumerate(todo, 1):
            try:
                rows.extend(_query_stock(bs, code))
            except Exception as e:
                _log(f"{code} 查询异常 {type(e).__name__} → 重连重试")
                try:
                    bs.logout()
                except Exception:
                    pass
                time.sleep(1)
                bs.login()
                try:
                    rows.extend(_query_stock(bs, code))
                except Exception:
                    _log(f"{code} 重试仍失败, 跳过")
            if i % 100 == 0:
                pd.DataFrame(rows).to_parquet(CHECKPOINT, index=False)
                el = time.time() - t0
                _log(f"[{i}/{len(todo)}] {code} 累计分红 {len(rows)} 条 | 耗时 {el:.0f}s | "
                     f"预计剩 {el / i * (len(todo) - i) / 60:.0f}min")
    finally:
        try:
            bs.logout()
        except Exception:
            pass

    if not rows:
        _log("未抓到任何分红数据!")
        return 1

    df = pd.DataFrame(rows)
    df["dividOperateDate"] = pd.to_datetime(df["dividOperateDate"])
    df = df.drop_duplicates(subset=["code", "dividOperateDate"]).sort_values(["code", "dividOperateDate"])
    df.to_parquet(OUT, index=False)
    CHECKPOINT.unlink(missing_ok=True)
    _log(f"完成: {df['code'].nunique()} 只股票 | {len(df)} 条分红 | "
         f"{df['dividOperateDate'].min().date()} -> {df['dividOperateDate'].max().date()} → {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
