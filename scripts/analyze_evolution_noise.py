"""进化记忆净效应噪声分析 — 预注册判据, 现有 runs 数据.

验证目标: 判定"进化记忆注入"是否有真实(非 LLM 噪声)的收益效应, 而非单对差异.

方法 (判据在运行前写死, 不因结果调整):
1. 噪声带宽: 每窗口每侧用 N 次独立运行的中位收益, 两侧 spread 的 p50 = 单次运行噪声稳健估计.
2. 主判据: 净化效应 |evo中位 - base中位|  >  噪声带宽 才算"有真实信号", 否则"证据不足不宣称".
3. 配对差分: 用"前段无注入期"标定每对 (evo_i, base_i) 的噪声, 配对差分出净进化记忆效应,
   跨 runs 求均值±std, 检验 0 是否在置信区间内.

用法:
  python scripts/analyze_evolution_noise.py [--windows 2020,2019]
  python scripts/analyze_evolution_noise.py --model glm-4-flash   # 第二模型复验(单模型局限缓解)
"""
import argparse
import json
import math
import statistics
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

REPLAY_DIR = Path("replay_data")

# 第二模型复验: --model 设置时仅分析该模型后缀的 runs (deepseek 报告无后缀).
MODEL_SUFFIX = ""

# 窗口定义: label, runs 数(自动检测实际存在的 run 文件)
WINDOWS = [
    {"label": "2020牛转崩", "runs": None},
    {"label": "2019牛", "runs": None},
    {"label": "2018熊", "runs": None},
    {"label": "2024震荡", "runs": None},
]

# t 分布临界值 (95%, 双侧), n=自由度+1
_T_CRIT = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
           6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}


def detect_runs(label, side="evo"):
    """自动检测该窗口 side 实际存在的 run 文件数."""
    i = 0
    while (REPLAY_DIR / f"replay_report_mw_{label}_{side}_r{i}_same{MODEL_SUFFIX}.json").exists():
        i += 1
    return i


def t_crit(n):
    return _T_CRIT.get(n, 2.0)

# 各窗口 memory 文件 glob 前缀 (用于找最早经验日期)
MEMORY_GLOB = {"2020牛转崩": "memory_mw_2020牛转崩_evo_r0_same.json",
               "2019牛": "memory_mw_2019牛_evo_r0_same.json"}


def load_equity(tag_side, label, r):
    f = REPLAY_DIR / f"replay_report_mw_{label}_{tag_side}_r{r}_same{MODEL_SUFFIX}.json"
    d = json.load(open(f, encoding="utf-8"))
    return d.get("equity_curve", [])


def rets_series(eq):
    """返回逐日累积收益% 列表 (基线=首日total)."""
    base = float(eq[0]["total"])
    return [(float(x["total"]) - base) / base * 100 for x in eq], [x["date"] for x in eq]


def memory_first_date(label):
    name = MEMORY_GLOB[label]
    if MODEL_SUFFIX:
        # 构造带模型后缀的 memory 文件 (如 ..._same_glm-4-flash.json)
        name = name.replace("_same.json", f"_same{MODEL_SUFFIX}.json")
    f = REPLAY_DIR / name
    if not f.exists():
        return None
    d = json.load(open(f, encoding="utf-8"))
    dates = sorted(i.get("date") for i in d.get("items", []))
    return dates[0] if dates else None


def main():
    global MODEL_SUFFIX
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", type=str, default=None)
    ap.add_argument("--model", type=str, default=None,
                    help="第二模型复验: 仅分析该模型后缀的 runs (如 glm-4-flash)")
    args = ap.parse_args()
    if args.model:
        MODEL_SUFFIX = f"_{args.model}"
    subs = [w.strip() for w in args.windows.split(",")] if args.windows else None

    for w in WINDOWS:
        label = w["label"]
        if subs and not any(s in label for s in subs):
            continue
        runs = detect_runs(label)
        print(f"\n{'='*66}\n窗口: {label} (每侧 {runs} 次运行)\n{'='*66}")

        # 收集每侧每次运行的累积收益 (全程)
        evo_ret, base_ret = [], []
        for r in range(runs):
            e = load_equity("evo", label, r)
            b = load_equity("base", label, r)
            if not e or not b:
                print(f"  缺 run{r}, 跳过")
                continue
            er, _ = rets_series(e)
            br, _ = rets_series(b)
            evo_ret.append(er[-1])
            base_ret.append(br[-1])

        if len(evo_ret) < 2 or len(base_ret) < 2:
            print(f"  数据不完整({len(evo_ret)} evo / {len(base_ret)} base), 需>=2次才能估噪声")
            continue

        em, bm = statistics.median(evo_ret), statistics.median(base_ret)
        e_spread = max(evo_ret) - min(evo_ret)
        b_spread = max(base_ret) - min(base_ret)
        noise_bw = statistics.median([e_spread, b_spread])  # 噪声带宽: 两侧 spread 中位
        effect = abs(em - bm)  # 净化效应

        print(f"  evo 各次收益: {[round(x,2) for x in evo_ret]} | 中位 {em:+.2f} | spread {e_spread:.2f}")
        print(f"  base各次收益: {[round(x,2) for x in base_ret]} | 中位 {bm:+.2f} | spread {b_spread:.2f}")
        print(f"  净化效应 |evo-base| = {effect:.2f} | 噪声带宽 = {noise_bw:.2f}")
        verdict = ("✅ 净化效应 > 噪声带宽 → 有真实信号" if effect > noise_bw
                   else "⚠️ 净化效应 ≤ 噪声带宽 → 证据不足, 不宣称进化效应")
        print(f"  判定: {verdict}")

        # 配对差分净效应 (仅当有 >=2 对且能分段)
        if runs >= 2 and label in MEMORY_GLOB:
            fidx = memory_first_date(label)
            nets = []
            pair_notes = []
            for r in range(runs):
                e = load_equity("evo", label, r)
                b = load_equity("base", label, r)
                er, ed = rets_series(e)
                br, bd = rets_series(b)
                # 前段(无注入)索引: 首个 >= 最早经验日期的位置
                idx = next((i for i, d in enumerate(ed) if d >= (fidx or "9999")), len(ed))
                def seg_rets(series, i0, i1):
                    if i1 <= i0:
                        return 0.0
                    return series[i1 - 1] - series[i0]
                pre_diff = seg_rets(er, 0, idx) - seg_rets(br, 0, idx)   # 前段 evo-base
                full_diff = er[-1] - br[-1]                              # 全程 evo-base
                net = full_diff - pre_diff                                # 净效应 = 后段增量
                nets.append(net)
                pair_notes.append(f"    r{r}: 前段差 {pre_diff:+.2f} | 全程差 {full_diff:+.2f} | 净效应 {net:+.2f}")
            mean = statistics.mean(nets)
            if len(nets) >= 2:
                sd = statistics.stdev(nets)
                se = sd / math.sqrt(len(nets))
                t = t_crit(len(nets))
                lo, hi = mean - t * se, mean + t * se
                sig = "显著非零" if not (lo <= 0 <= hi) else "未显著(0在区间内)"
                print(f"  配对差分净效应(记忆注入): 均值 {mean:+.2f} ± {sd:.2f} (95%CI [{lo:+.2f},{hi:+.2f}]) → {sig}")
                for n in pair_notes:
                    print(n)


if __name__ == "__main__":
    main()