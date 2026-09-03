"""前瞻跟踪 (转债版) — 跟踪 registry 里的 cb_lowprem_top10_hold, 并攒积全市场转债快照.

每次运行:
  1. 腾讯实时批量拉取篮子+universe 全部转债当前价 (免代理), 用 harness compute_status 算
     前瞻收益 vs 冻结转债等权 vs 上证, 应用预注册判据, 追加 tracking 快照。
  2. (可选 --with-prem) 东财 point-in-time 逐债拉转股溢价率, 与价格快照一起存
     replay_data/cb_forward/ — 攒数月后可对真实未来数据忠实重放周轮动 (默认关闭, ~310 请求)。

诚实边界:
  - 当前数据日取腾讯行情时间戳; <= 入场日时跳过 (不追加 day-0 空快照)。
  - 停牌/退市券价缺失 → equalweight_return 内 ffill (与回测口径一致)。
  - 溢价率快照失败即报错退出 (零模拟), 不影响纯价格跟踪 (重跑即可)。

用法:
  python scripts/forward_track_cb.py                # 价格跟踪 (快, 每周可跑)
  python scripts/forward_track_cb.py --with-prem    # 同时攒溢价率快照 (供周轮动重放)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from analysis.forward_validation import (  # noqa: E402
    append_tracking,
    compute_status,
    load_registry,
)

FWD_DIR = ROOT / "replay_data" / "cb_forward"
CB_EDGE_IDS = {"cb_lowprem_top10_hold"}


def _tencent_quotes(codes: list[str]) -> tuple[dict[str, float], dict[str, str]]:
    """批量腾讯实时: {6位code: price}, {6位code: 行情日期YYYY-MM-DD}。失败抛错 (零模拟)。"""
    out_px: dict[str, float] = {}
    out_date: dict[str, str] = {}
    for i in range(0, len(codes), 60):
        chunk = codes[i:i + 60]
        syms = [("sh" if c.startswith("11") else "sz") + c for c in chunk]
        r = requests.get("https://qt.gtimg.cn/q=" + ",".join(syms), timeout=10,
                         headers={"User-Agent": "Mozilla/5.0", "Referer": "https://finance.qq.com/"})
        r.raise_for_status()
        r.encoding = "gbk"
        for line in r.text.strip().split("\n"):
            if "=" not in line or "~" not in line:
                continue
            code = line.split("=")[0].split("_")[-1]
            f = line.split("=", 1)[1].strip('"').split("~")
            if len(f) > 30 and f[3]:
                out_px[code] = float(f[3])
                ts = f[30]  # yyyymmddhhmmss
                out_date[code] = f"{ts[:4]}-{ts[4:6]}-{ts[6:8]}" if len(ts) >= 8 else ""
        time.sleep(0.3)
    if not out_px:
        raise RuntimeError("腾讯未返回任何转债行情")
    return out_px, out_date


def fetch_index(symbol: str = "sh000001") -> float:
    r = requests.get(f"https://qt.gtimg.cn/q={symbol}", timeout=8,
                     headers={"User-Agent": "Mozilla/5.0", "Referer": "https://finance.qq.com/"})
    r.encoding = "gbk"
    for line in r.text.strip().split("\n"):
        if "=" not in line or "~" not in line:
            continue
        f = line.split("=", 1)[1].strip('"').split("~")
        if len(f) > 3 and f[3]:
            return float(f[3])
    raise RuntimeError(f"腾讯未返回 {symbol} 指数价")


def snapshot_prem(date_str: str, codes: list[str]) -> pd.DataFrame:
    """拉当日全市场转股溢价率快照。

    原东财逐债 valuation 接口已 404 (2026-08 起) → 改 akshare bond_zh_cov
    (东财 kzz 列表页, 一次请求含全部活跃券现行溢价率/转股价/正股价)。
    已退市/停牌券列表中无真实溢价 → 跳过并计数 (零模拟: 不造数, 停牌券本就不可交易)。
    """
    import akshare as ak

    rows = []
    df = ak.bond_zh_cov()
    prem_map = {}
    for _, r in df.iterrows():
        code = str(r.get("债券代码", "")).strip()
        px = pd.to_numeric(r.get("债现价"), errors="coerce")
        prem = pd.to_numeric(r.get("转股溢价率"), errors="coerce")
        # 债现价=100 且溢价率 NaN → 未上市/失效; 两者都有才算真实快照
        if code and pd.notna(px) and pd.notna(prem):
            prem_map[code] = float(prem)
    skipped = [c for c in codes if c not in prem_map]
    if not prem_map:
        raise RuntimeError("bond_zh_cov 未返回任何有效溢价率 (零模拟, 中止快照)")
    if skipped:
        print(f"  prem 快照跳过 {len(skipped)} 只无溢价券 (退市/未上市/停牌): "
              f"{skipped[:5]}{'...' if len(skipped) > 5 else ''}")
    rows = [{"symbol": c, "date": date_str, "转股溢价率": prem_map[c]}
            for c in codes if c in prem_map]
    return pd.DataFrame(rows)


def main() -> int:
    with_prem = "--with-prem" in sys.argv
    reg = load_registry()
    bets = {k: v for k, v in reg.get("bets", {}).items() if k in CB_EDGE_IDS}
    if not bets:
        print("注册表无转债 bet, 先跑 scripts/forward_register_cb_double_low.py")
        return 1

    for edge_id, bet in bets.items():
        entry = bet["entry"]
        all_codes = sorted(set(entry["basket_symbols"]) | set(entry["universe_symbols"]))
        px, px_date = _tencent_quotes(all_codes)
        # 当前数据日 = 多数券的行情日 (停牌券取个别日期不影响判定)
        cur_date = pd.Series(list(px_date.values())).mode().iloc[0]
        if cur_date <= str(bet["entry_date"]):
            print(f"[{edge_id}] 无前瞻新数据 (行情日 {cur_date} <= 入场日 {bet['entry_date']}), 跳过。")
            continue
        tracking = bet.setdefault("tracking", [])
        if tracking and tracking[-1].get("current_date") == cur_date:
            print(f"[{edge_id}] {cur_date} 已跟踪过, 跳过 (无新交易日)。")
            continue

        sh_close = fetch_index("sh000001")
        status = compute_status(bet, px, px, sh_close, cur_date)
        append_tracking(edge_id, status)
        print("=" * 84)
        print(f"前瞻跟踪: {edge_id}  ({bet['entry_date']} → {cur_date})")
        print(f"  篮子净 {status['basket_return_pct']:+.2f}% | 等权 {status['universe_return_pct']:+.2f}% | "
              f"上证 {status['sh_index_return_pct']:+.2f}%")
        print(f"  选券alpha {status['selection_alpha_pp']:+.2f}pp (失败阈值 "
              f"{bet['criterion']['failure_threshold_pp']}pp) | edge_failing: {status['edge_failing']} | "
              f"horizon_reached: {status['horizon_reached']}")

        # 快照攒积 (价格全体; 溢价率可选)
        FWD_DIR.mkdir(parents=True, exist_ok=True)
        px_fp = FWD_DIR / f"prices_{cur_date.replace('-', '')}.parquet"
        if not px_fp.exists():
            pd.DataFrame({"symbol": list(px.keys()), "close": list(px.values()),
                          "date": cur_date}).to_parquet(px_fp)
            print(f"  价格快照已存: {px_fp.name} ({len(px)} 只)")
        if with_prem:
            prem_fp = FWD_DIR / f"prem_{cur_date.replace('-', '')}.parquet"
            if not prem_fp.exists():
                df = snapshot_prem(cur_date, sorted(px.keys()))
                df.to_parquet(prem_fp)
                print(f"  溢价率快照已存: {prem_fp.name} ({len(df)} 只)")

    print("\n跟踪完成。判据注册后只读 (禁止事后调参追赢)。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
