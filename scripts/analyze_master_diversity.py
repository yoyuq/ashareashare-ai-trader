"""大师相关性与多元化诊断 — 拆穿"假多元化"。

背景: 5位大师(利弗莫尔/巴菲特/索罗斯/达利欧/缠中说禅)由同一个 LLM 调用,
每交易日只选 1 位主导大师, 产出唯一的 risk_level/position_multiplier/phase。
这带来两个可量化的质疑:

1. 大师是否真的多元化?  如果 90% 的天数都是同一位大师主导, 所谓"5大师"只是噱头。
   度量: 主导大师分布熵(H), 以及"实际活跃大师数" (有效大师数 = 2^H)。

2. 切换大师后输出是否真的不同?  如果换不换大师, risk_level 都差不多,
   那大师输出是冗余的(不管选谁结果都一样), 对抗/交叉验证没有信息量。
   度量: 同 market_phase 下, 不同大师的 risk_level 组间方差 / 组内方差 (F-style)。

用法:
    python -m scripts.analyze_master_diversity
    python -m scripts.analyze_master_diversity --journals replay_data/journal_diag_v42_full.jsonl
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

# 大师标准名 (日志里可能有乱码/别名, 统一映射)
MASTER_ALIASES = {
    "利弗莫尔": "利弗莫尔", "趋势投机之王": "利弗莫尔",
    "巴菲特": "巴菲特", "价值投资之父": "巴菲特",
    "索罗斯": "索罗斯", "反身性大师": "索罗斯",
    "达利欧": "达利欧", "全天候与经济机器": "达利欧",
    "缠中说禅": "缠中说禅", "缠宗师": "缠中说禅",
}


def _norm_master(name: str) -> str:
    if not name:
        return "(未知)"
    return MASTER_ALIASES.get(name, name)


def _load_journals(paths: list[Path]) -> list[dict]:
    recs = []
    for p in paths:
        if not p.exists():
            print(f"  ! 跳过缺失文件: {p}")
            continue
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    recs.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return recs


def _shannon_entropy(counts: Counter) -> float:
    total = sum(counts.values())
    if total == 0:
        return 0.0
    h = 0.0
    for c in counts.values():
        p = c / total
        if p > 0:
            h -= p * math.log2(p)
    return h


def analyze(recs: list[dict]) -> dict:
    # 1. 主导大师分布
    dominant = Counter(_norm_master(r.get("dominant_master")) for r in recs)
    # 2. risk_level 分布 (整体 + 按大师)
    risk_by_master: dict[str, list[int]] = defaultdict(list)
    phase_by_master: dict[str, Counter] = defaultdict(Counter)
    risk_by_phase_master: dict[tuple, list[int]] = defaultdict(list)
    for r in recs:
        m = _norm_master(r.get("dominant_master"))
        rl = r.get("risk_level")
        ph = r.get("market_phase")
        if isinstance(rl, (int, float)):
            risk_by_master[m].append(int(rl))
            if ph:
                risk_by_phase_master[(ph, m)].append(int(rl))
        if ph:
            phase_by_master[m][ph] += 1

    total = sum(dominant.values())  # 总天数 (不是大师个数!)
    h = _shannon_entropy(dominant)
    eff_masters = 2 ** h if h > 0 else 0.0

    # 3. 同 phase 下大师间 risk_level 分歧度 (组间方差占比如 JS 风格)
    #    - 对每个有≥2个大师样本的 phase, 算不同大师 risk 均值间的极差
    phase_divergence = {}
    for ph, sub in risk_by_phase_master.items():
        pass
    # 按 phase 分组统计
    phase_master_means: dict[str, dict[str, float]] = defaultdict(dict)
    phase_count: dict[str, int] = defaultdict(int)
    for (ph, m), rls in risk_by_phase_master.items():
        phase_master_means[ph][m] = sum(rls) / len(rls)
        phase_count[ph] += len(rls)
    # 每个 phase 内, 各大师均值之间的最大极差 (风险等级满分距)
    for ph, means in phase_master_means.items():
        if len(means) >= 2:
            vals = list(means.values())
            phase_divergence[ph] = round(max(vals) - min(vals), 2)

    # 4. 大师相异性: 同 phase 下, 大师 risk_level 的标准差 (跨大师)
    master_risk_std = {}
    for ph, means in phase_master_means.items():
        if len(means) >= 2:
            vals = list(means.values())
            _mu = sum(vals) / len(vals)
            _var = sum((v - _mu) ** 2 for v in vals) / len(vals)
            master_risk_std[ph] = round(math.sqrt(_var), 2)

    # 5. v5.2 对抗票统计 (新日志才有 adversarial 字段)
    adv_stats = {"n": 0, "diverged": 0, "applied": Counter()}
    for r in recs:
        adv = r.get("adversarial_divergence")
        if adv is None:
            continue
        adv_stats["n"] += 1
        if int(adv) >= 2:
            adv_stats["diverged"] += 1
            ap = r.get("adversarial_applied")
            if ap:
                adv_stats["applied"][ap] += 1
    adv_stats["applied"] = dict(adv_stats["applied"])

    return {
        "total_days": total,
        "dominant_distribution": {k: round(v * 100 / total, 1) for k, v in dominant.items()},
        "entropy_bits": round(h, 2),
        "effective_masters": round(eff_masters, 2),
        "risk_by_master": {
            k: {"mean": round(sum(v) / len(v), 2), "n": len(v)}
            for k, v in risk_by_master.items()
        },
        "phase_master_risk_divergence": phase_divergence,
        "phase_master_risk_std": master_risk_std,
        "adversarial_stats": adv_stats,
    }


def _verdict(res: dict) -> tuple[str, list[str]]:
    rows = []
    eff = res["effective_masters"]
    total = res["total_days"]
    if total == 0:
        return "无数据", ["没有可分析的天数"]
    # 有效大师数
    if eff < 1.5:
        verdict = "❌ 假多元化坐实: 实际有效大师数<1.5, 绝大多数天数同一位大师主导"
        rows.append(f"有效大师数仅 {eff} (名义5位, 分布熵 {res['entropy_bits']} 比特)")
    elif eff < 2.5:
        verdict = "⚠️ 多元化不足: 有效大师数 1.5~2.5, 大师选择偏科"
        rows.append(f"有效大师数 {eff}, 分布熵 {res['entropy_bits']} 比特")
    else:
        verdict = "✅ 多元化尚可: 有效大师数>=2.5"
        rows.append(f"有效大师数 {eff}, 分布熵 {res['entropy_bits']} 比特")

    # 同phase下大师risk分歧 (冗余度)
    stds = list(res["phase_master_risk_std"].values())
    if stds:
        avg_std = sum(stds) / len(stds)
        if avg_std < 0.3:
            rows.append(f"⚠️ 大师输出冗余: 同阶段下不同大师 risk_level 均值标准差仅 {avg_std:.2f} (换大师几乎不改变输出)")
        elif avg_std < 0.7:
            rows.append(f"大师输出有一定分歧: 同阶段 risk 均值标准差 {avg_std:.2f}")
        else:
            rows.append(f"✅ 大师输出分歧明显: 同阶段 risk 均值标准差 {avg_std:.2f}")

    # 对抗票
    adv = res["adversarial_stats"]
    if adv["n"] > 0:
        _div_pct = adv["diverged"] / adv["n"] * 100
        rows.append(
            f"对抗票: {adv['n']}天中 {adv['diverged']}天分歧(≥2级, {_div_pct:.0f}%) → "
            f"{adv['applied'].get('conservative',0)}次保守收 / {adv['applied'].get('dampen',0)}次降置信"
        )

    # 主导大师最大占比
    dom = res["dominant_distribution"]
    if dom:
        top_name, top_pct = max(dom.items(), key=lambda kv: kv[1])
        if top_pct > 60:
            rows.append(f"⚠️ 单大师依赖: {top_name} 主导了 {top_pct}% 的天数")
        else:
            rows.append(f"最常用大师 {top_name} 占 {top_pct}%")

    return verdict, rows


def main():
    parser = argparse.ArgumentParser(description="大师相关性与多元化诊断")
    default_journals = [str(p) for p in Path("replay_data").glob("journal_diag*.jsonl")]
    parser.add_argument("--journals", nargs="*", default=default_journals,
                        help="决策日志路径, 默认 replay_data/journal_diag*.jsonl")
    args = parser.parse_args()

    recs = _load_journals([Path(p) for p in args.journals])
    print(f"加载 {len(recs)} 条决策记录")
    if not recs:
        print("无数据, 退出")
        return

    res = analyze(recs)
    verdict, rows = _verdict(res)

    print(f"\n{'='*60}")
    print(f"判定: {verdict}")
    for r in rows:
        print(f"  • {r}")
    print(f"{'='*60}")

    print(f"\n── 主导大师分布 (共{res['total_days']}天) ──")
    for name, pct in sorted(res["dominant_distribution"].items(), key=lambda kv: -kv[1]):
        bar = "█" * int(pct / 2)
        print(f"  {name:<8} {pct:5.1f}%  {bar}")
    print(f"  分布熵 {res['entropy_bits']} 比特 | 有效大师数 {res['effective_masters']}")

    print(f"\n── 各大师产出的 risk_level ──")
    for name, st in sorted(res["risk_by_master"].items(), key=lambda kv: -kv[1]["n"]):
        print(f"  {name:<8} 均值risk {st['mean']:>4}  (样本 {st['n']})")

    print(f"\n── 同市场阶段下, 换大师后 risk_level 是否变化 (分歧度) ──")
    div = res["phase_master_risk_divergence"]
    std = res["phase_master_risk_std"]
    if div:
        for ph in sorted(div):
            print(f"  {ph:<14} 大师间risk极差 {div[ph]:>4}  标准差 {std.get(ph, 0):.2f}")
    else:
        print("  (没有同一阶段被≥2位大师主导的样本, 无法测分歧 — 这本身是过度集中的信号)")


if __name__ == "__main__":
    main()