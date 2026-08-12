"""v5.6 多窗口 A/B 验证编排 — 进化内核在多市场环境是否真实占优.

方法 (隔离单变量·抗噪声):
  - 每窗口跑两边: 进化侧(--evolution) vs 基线侧(去掉 --evolution), 其余参数完全一致
    (同 data-file / 同 end / 同 days / 同滑点ON / 同 adv-mode). 唯一差异 = 进化记忆注入.
  - 关键 2020 窗口每边跑 runs 次取中位数 (抗 LLM 噪声); 其余窗口 1 次 + 标注噪声区间.
  - 通过判据: 进化在【多数窗口】的收益或夏普 >= 基线, 才算多环境占优; 单窗口胜出不算.

用法:
  python scripts/run_multiwindow_ab.py                    # 跑所有已就绪窗口
  python scripts/run_multiwindow_ab.py --windows 2020     # 只跑 2020
  python scripts/run_multiwindow_ab.py --adv-mode independent  # 对抗独立调用下重验
输出:
  reports/multiwindow_validation.md
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

REPLAY = "scripts/historical_replay.py"
REPLAY_DIR = Path("replay_data")
REPORT_DIR = Path("reports")

# ═══════════════════════════════════════════════════════════════════
# 窗口定义 — 数据到位后据 sh.000001 指数实际走势标 regime (不拍脑袋).
# 每个窗口: label / data_file / start~end(含收盘日) / regime / runs(去噪中位数样本数)
# ═══════════════════════════════════════════════════════════════════
WINDOWS = [
    {
        "label": "2020牛转崩",
        "data_file": "replay_data/daily_2020-06-01_2021-02-28.parquet",
        "start": "2020-12-02", "end": "2021-02-26",
        "regime": "牛→崩", "runs": 5,   # 关键窗口, 5次取中位数(降噪, 噪声分析证实n=3不足)
    },
    {
        "label": "2018熊",
        "data_file": "replay_data/daily_2018-01-01_2018-12-31.parquet",
        "start": "2018-06-01", "end": "2018-08-17",
        "regime": "熊", "runs": 1,
    },
    {
        "label": "2019牛",
        # 数据起点提前到 2018-10 (覆盖窗口 02-11 往前 60 日使 pct_60d 可算; 原 2019-01 起点导致前37天被新股过滤)
        "data_file": "replay_data/daily_2018-10-01_2019-12-31.parquet",
        "start": "2019-02-11", "end": "2019-04-30",
        "regime": "牛", "runs": 5,   # 关键反例窗口, 5次取中位数确认是否稳定跑输(与2020同强度)
    },
    {
        "label": "2024震荡",
        "data_file": "replay_data/daily_2024-01-01_2024-12-31.parquet",
        "start": "2024-05-06", "end": "2024-07-31",
        "regime": "震荡", "runs": 1,
    },
]

SIDES = [
    {"key": "evo", "name": "进化", "evolution": True},
    {"key": "base", "name": "基线", "evolution": False},
]


def _window_days(data_file: str, start: str, end: str) -> int:
    """从数据文件统计 [start,end] 内的交易日数 (供 --days 精确对齐窗口)."""
    import pandas as pd
    from datetime import date
    df = pd.read_parquet(data_file)
    d0, d1 = date.fromisoformat(start), date.fromisoformat(end)
    dates = pd.to_datetime(df["date"]).dt.date
    in_range = dates[(dates >= d0) & (dates <= d1)]
    return int(in_range.nunique()) if len(in_range) else 0


def _run_once(window: dict, side: dict, adv_mode: str, run_i: int, universe: list = None) -> dict:
    """跑一次回放, 返回指标. tag 唯一化避免覆盖."""
    label = window["label"]
    days = _window_days(window["data_file"], window["start"], window["end"])
    _sub = "_csi" if universe else ""  # 子集(CSI300)tag 后缀, 与全市场报告隔离
    _m_tag = os.environ.get("REPLAY_LLM_MODEL", "")
    _m_suffix = f"_{_m_tag}" if _m_tag else ""  # 第二模型复验(单模型局限缓解)时隔离报告
    tag = f"mw_{label}_{side['key']}_r{run_i}_{adv_mode}{_m_suffix}{_sub}"
    cmd = [
        sys.executable, REPLAY,
        "--flash-diag", "--diag-top-n", "20", "--diagnostic",
        "--data-file", window["data_file"],
        "--end", window["end"],
        "--days", str(days),
        "--tag", tag,
        "--adv-mode", adv_mode,
    ]
    if universe:
        cmd += ["--universe", ",".join(universe)]
    if side["evolution"]:
        cmd.append("--evolution")
    print(f"\n▶ [{label}/{side['name']}/run{run_i}] {cmd}", flush=True)
    _env = dict(os.environ)
    _env["PYTHONIOENCODING"] = "utf-8"
    _env["PYTHONUTF8"] = "1"
    # 不捕获, 让回放 stdout 直接流入 A/B 日志 (实时可见进度)
    result = subprocess.run(cmd, env=_env)
    if result.returncode != 0:
        print(f"  ✗ 回放失败 rc={result.returncode}")
        return None
    # 读取报告
    report = REPLAY_DIR / f"replay_report_{tag}.json"
    if not report.exists():
        print(f"  ✗ 报告缺失 {report}")
        return None
    data = json.loads(report.read_text(encoding="utf-8"))
    eq = data.get("equity_curve", [])
    return _metrics(eq, report=data)


def _metrics(eq_curve: list, report: dict = None) -> dict:
    """从权益曲线算收益/回撤/夏普/胜率.

    收益用**初始资金**作基线 (与 replay 报告 total_return_pct 及已验证 +3.34pp 方法论一致),
    而非首日权益 — 首日换仓会引入 day-1 蝴蝶效应噪声.
    """
    if not eq_curve:
        return {}
    import math
    peak = float(eq_curve[0]["total"])
    max_dd = 0.0
    rets = []
    for i, d in enumerate(eq_curve):
        v = float(d["total"])
        if v > peak:
            peak = v
        dd = (v - peak) / peak * 100
        if dd < max_dd:
            max_dd = dd
        if i > 0:
            prev = float(eq_curve[i - 1]["total"])
            if prev > 0:
                rets.append((v - prev) / prev)
    # 收益基线 = 初始资金 (缓启动: 首日已建仓, 首日权益 < 资金, 用资金作基线最干净)
    base = float(report.get("initial_capital") or 100000.0)
    total_ret = (float(eq_curve[-1]["total"]) - base) / base * 100
    sharpe = 0.0
    win_rate = 0.0
    if rets:
        mean_r = sum(rets) / len(rets)
        std_r = math.sqrt(sum((x - mean_r) ** 2 for x in rets) / len(rets))
        if std_r > 0:
            sharpe = mean_r / std_r * math.sqrt(250)
        win_rate = sum(1 for x in rets if x > 0) / len(rets)
    return {
        "total_return": total_ret,
        "max_drawdown": max_dd,
        "sharpe": sharpe,
        "win_rate": win_rate,
        "num_days": len(eq_curve),
        "final_equity": report.get("final_equity") if report else None,
    }


def _median_over_runs(runs: list[dict]) -> dict:
    """同一 (window, side) 多次运行取中位数."""
    valid = [r for r in runs if r and r.get("num_days")]
    if not valid:
        return {}
    out = {}
    for k in ("total_return", "max_drawdown", "sharpe", "win_rate"):
        vals = [r[k] for r in valid]
        out[k] = statistics.median(vals)
    out["n_runs"] = len(valid)
    out["all_runs"] = {k: [round(r[k], 3) for r in valid]
                       for k in ("total_return", "sharpe")}
    out["num_days"] = valid[0]["num_days"]
    return out


def _force_utf8():
    """Windows GBK 控制台无法打印 ▶ 等字符 — 强制 UTF-8 输出."""
    import os
    for name in ("stdout", "stderr"):
        s = getattr(sys, name, None)
        if hasattr(s, "reconfigure"):
            try:
                s.reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass
    os.environ["PYTHONIOENCODING"] = "utf-8"
    os.environ["PYTHONUTF8"] = "1"


def main():
    _force_utf8()
    ap = argparse.ArgumentParser(description="多窗口进化 A/B 验证")
    ap.add_argument("--windows", type=str, default=None,
                    help="逗号分隔的窗口label子串, 缺省跑全部")
    ap.add_argument("--adv-mode", type=str, default="same",
                    choices=["same", "independent", "off"])
    ap.add_argument("--skip-existing", action="store_true",
                    help="已存在报告则跳过该 run")
    ap.add_argument("--universe", type=str, default=None,
                    help="股票池子集文件(每行一个symbol), 用于幸存者偏差稳健性(如 replay_data/csi300_universe.txt)")
    ap.add_argument("--report-suffix", type=str, default="",
                    help="报告文件名后缀, 隔离本次运行不覆盖主报告 (如 --report-suffix noise)")
    ap.add_argument("--max-runs", type=int, default=None,
                    help="限制每窗口每侧 run 数 (复验/冒烟用, 缺省用窗口定义 runs)")
    args = ap.parse_args()

    universe = None
    if args.universe:
        uf = Path(args.universe)
        if not uf.exists():
            print(f"✗ universe 文件不存在: {uf}")
            return 1
        universe = [ln.strip() for ln in uf.read_text(encoding="utf-8").splitlines() if ln.strip()]
        print(f"使用子集股票池: {len(universe)} 只 ({uf.name})")

    windows = WINDOWS
    if args.windows:
        subs = [w.strip() for w in args.windows.split(",") if w.strip()]
        windows = [w for w in WINDOWS if any(s in w["label"] for s in subs)]

    results = []  # {label, regime, evo:{...}, base:{...}, evo_wins:{ret,sharpe}}
    for w in windows:
        data_file = Path(w["data_file"])
        if not data_file.exists():
            print(f"\n⏭ 跳过 {w['label']}: 数据未就绪 {data_file}")
            continue
        print(f"\n{'='*66}\n  窗口: {w['label']} ({w['regime']})\n  {w['data_file']}\n{'='*66}")
        row = {"label": w["label"], "regime": w["regime"], "window": f"{w['start']}~{w['end']}"}
        _n_runs = min(w["runs"], args.max_runs) if args.max_runs else w["runs"]
        for side in SIDES:
            runs = []
            for i in range(_n_runs):
                _m_tag = os.environ.get("REPLAY_LLM_MODEL", "")
                _m_suffix = f"_{_m_tag}" if _m_tag else ""
                tag = f"mw_{w['label']}_{side['key']}_r{i}_{args.adv_mode}{_m_suffix}{'_csi' if universe else ''}"
                if args.skip_existing and (REPLAY_DIR / f"replay_report_{tag}.json").exists():
                    print(f"  [缓存] {w['label']}/{side['name']}/run{i}")
                    report = json.loads((REPLAY_DIR / f"replay_report_{tag}.json").read_text(encoding="utf-8"))
                    runs.append(_metrics(report.get("equity_curve", []), report=report))
                    continue
                m = _run_once(w, side, args.adv_mode, i, universe=universe)
                runs.append(m)
            row[side["key"]] = _median_over_runs(runs)
        evo, base = row.get("evo", {}), row.get("base", {})
        if evo and base:
            row["_ret_diff"] = evo.get("total_return", 0) - base.get("total_return", 0)
            row["_sharpe_diff"] = evo.get("sharpe", 0) - base.get("sharpe", 0)
            row["_win"] = (evo.get("total_return", 0) >= base.get("total_return", 0),
                           evo.get("sharpe", 0) >= base.get("sharpe", 0))
        results.append(row)

    _write_report(results, universe=universe, suffix=args.report_suffix)
    _print_summary(results)


def _win_counts(results):
    ret_wins = sum(1 for r in results if r.get("_ret_diff", 0) >= 0)
    sharpe_wins = sum(1 for r in results if r.get("_sharpe_diff", 0) >= 0)
    total = len(results)
    return ret_wins, sharpe_wins, total


def _print_summary(results):
    print(f"\n{'='*66}\n  多窗口 A/B 汇总\n{'='*66}")
    for r in results:
        evo, base = r.get("evo", {}), r.get("base", {})
        if not evo or not base:
            print(f"\n  {r['label']}: 数据不完整")
            continue
        print(f"\n  {r['label']} ({r['regime']}) {r['window']}")
        print(f"    {'':6}{'进化':>10}{'基线':>10}{'Δ':>10}")
        for k, lab in [("total_return", "收益%"), ("max_drawdown", "回撤%"),
                       ("sharpe", "夏普"), ("win_rate", "胜率")]:
            ev, bs = evo.get(k, 0), base.get(k, 0)
            print(f"    {lab:<6}{ev:>+10.2f}{bs:>+10.2f}{ev-bs:>+10.2f}")
    ret_wins, sharpe_wins, total = _win_counts(results)
    print(f"\n  进化在 {total} 个窗口中的胜出场次: 收益 {ret_wins} / {total}, 夏普 {sharpe_wins} / {total}")
    if total >= 2 and ret_wins > total / 2:
        print("  ✅ 进化在多数窗口收益占优 —— 多环境占优成立")
    elif total < 2:
        print("  ⚠️ 仅 1 个窗口, 样本不足 —— 只能算初步, 不能宣称多环境占优")
    else:
        print("  ⚠️ 进化未在多数窗口收益占优 —— 多环境占优不成立")


def _write_report(results, universe=None, suffix=""):
    REPORT_DIR.mkdir(exist_ok=True)
    is_csi = bool(universe)
    ret_wins, sharpe_wins, total = _win_counts(results)
    lines = [
        "# 多窗口进化内核 A/B 验证 (v5.6)",
        "",
        f"生成日期: 2026-08-10 | 方法: 隔离单变量 (唯差异=进化记忆注入) | 滑点ON | 对抗模式: same"
        + (" | **股票池: CSI300大市值子集(300只, 幸存者偏差稳健性参考)**" if is_csi else ""),
        "",
        "## 结论",
        "",
        f"- 进化在 **{total}** 个窗口中收益占优 **{ret_wins}**, 夏普占优 **{sharpe_wins}**。",
    ]
    if total >= 2 and ret_wins > total / 2:
        lines.append("- **✅ 进化在多数窗口收益占优 — 多环境占优成立。**")
    elif total < 2:
        lines.append("- **⚠️ 仅单窗口, 样本不足 — 只能算初步验证, 不能宣称多环境占优。**")
    else:
        lines.append("- **⚠️ 进化未在多数窗口占优 — 多环境占优不成立 (如实报告)。**")
    lines += [
        "",
        "## 逐窗口对比 (进化 vs 基线)",
        "",
        "| 窗口 | regime | 收益% | 回撤% | 夏普 | 胜率 | 收益Δ | 夏普Δ | 进化占优 |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        evo, base = r.get("evo", {}), r.get("base", {})
        if not evo or not base:
            lines.append(f"| {r['label']} | {r['regime']} | 数据不完整 | | | | | | |")
            continue
        win = "收益+夏普" if r.get("_win") == (True, True) else ("收益" if r["_win"][0] else "—")
        lines.append(
            f"| {r['label']} | {r['regime']} | "
            f"{evo['total_return']:+.2f} / {base['total_return']:+.2f} | "
            f"{evo['max_drawdown']:.2f} / {base['max_drawdown']:.2f} | "
            f"{evo['sharpe']:.2f} / {base['sharpe']:.2f} | "
            f"{evo['win_rate']:.1%} / {base['win_rate']:.1%} | "
            f"{r['_ret_diff']:+.2f} | {r['_sharpe_diff']:+.2f} | {win} |"
        )
    lines += [
        "",
        "## 噪声与局限 (如实报告)",
        "",
        "- LLM 诊断随机, 单次 ±1-2%. 2020 每边 3 次取中位数, 其余 1 次 — 判据取多数窗口, 不用单次胜负。",
        "- 幸存者偏差: 用当前股票池回放历史, 排除已退市股 → 结果偏乐观。",
        "- 单模型 (deepseek) 全链路, 是真实局限。",
        "- 纸盘摩擦为校准近似, 非券商实测; 但在两边一致, 不偏向任何一侧。",
    ]
    out = REPORT_DIR / (
        (f"multiwindow_validation_csi300{suffix}.md" if is_csi
         else f"multiwindow_validation{suffix}.md"))
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n报告已写入: {out}")


if __name__ == "__main__":
    main()