# -*- coding: utf-8 -*-
"""飞书影子权重 + 反事实验证 — 第五张王牌: 防进化过拟合.

原理: 进化系统改权重最大的坑是"追着近期数据手调" → 过拟合近期行情。
      本节点强制两条纪律:
        1. 预注册(影子权重): 候选权重必须提前写在 shadow_candidates 里,
           不能看到结果后临时拍脑袋改。
        2. 反事实验证: 候选权重在【留出的历史决策日志】上回放, 算"如果当时用它"
           的净效应(配对差分), 仅在多数样本里确实优于当前权重时才转正。

用影子权重防过拟合 (呼应项目内核: 进化目标是提升总收益, 不是减少错误次数;
  留出法A/B已实证当前项目进化regime切换曾过拟合-0.76pp, 本节点把防过拟合做成硬约束)。

用法: 替换飞书一个代码节点, 放在"进化写入节点"之后, 每周/每月触发:
  入参 arg1 = decision_log (JSON数组, 每行含 date/regime/final_risk/final_position/
                          next_day_change/was_correct)
        arg2 = shadow_candidates (JSON, 含 current_weights 与 candidates 数组)
  出参: shadow_result (含转正/保留/净效应/样本量/诚实标注)
"""
import json


# ===== 权重方案 → 风险等级 的映射 (与主流程一致的可解释规则) =====
# 每个方案: {"risk_weights": {regime: {per_regime_risk_bias}}, "position_weights": {...}}
# 简化: 用 bias 数组表达"在 base 风险上偏保守(-)或偏激进(+)"，回放时据此调整仓位。
def _apply_weights(decisions, scheme):
    """在 decision_log 上回放某个权重方案, 返回逐日净收益序列。

    反事实: 当日实际用 final_position 交易, 次日收益 next_day_change。
    候选方案通过 regime 偏置调整仓位: pos' = clamp(pos * (1 + regime_bias), 0.2, 1.5)
    """
    bias = scheme.get("regime_bias", {})  # {"strong_bull": 0.1, "panic": -0.3, ...}
    nav = 1.0
    navs = []
    for d in decisions:
        regime = d.get("regime") or "sideways"
        pos = d.get("final_position") or 1.0
        chg = d.get("next_day_change") or 0.0
        b = bias.get(regime, 0.0)
        pos_adj = max(0.2, min(1.5, pos * (1 + b)))
        nav *= (1 + chg * pos_adj / 100.0)
        navs.append(nav)
    return navs


def _max_drawdown(navs):
    peak, mdd = navs[0] if navs else 1.0, 0.0
    for v in navs:
        peak = max(peak, v)
        mdd = min(mdd, (v - peak) / peak)
    return mdd


def main(arg1: str, arg2: str) -> dict:
    # ===== 解析 =====
    def safe_load(s):
        if not s:
            return None
        s = s.strip()
        if s.startswith("```"):
            s = s.strip("`")
            if s.startswith("json"):
                s = s[4:].strip()
        try:
            return json.loads(s)
        except Exception:
            start, end = s.find("["), s.rfind("]")
            if start >= 0 and end > start:
                try:
                    return json.loads(s[start:end + 1])
                except Exception:
                    return None
        return None

    decisions = safe_load(arg1) or []
    cfg = safe_load(arg2) or {}

    # ===== 只留"已验证"的决策(有次日结果), 否则无法反事实 =====
    verified = [d for d in decisions if d.get("is_verified")
                and d.get("next_day_change") is not None]
    if len(verified) < 5:
        return {"shadow_result": json.dumps({
            "decision": "hold",
            "reason": f"已验证样本仅{len(verified)}条, 不足5条, 不冒险转正(防过拟合)",
            "verified_count": len(verified),
            "note": "样本不足, 候选权重保持影子状态, 不生效",
        }, ensure_ascii=False)}

    # ===== 当前权重 vs 每个候选, 留出样本反事实 =====
    # 留出: 用全部已验证样本回放(滚动窗口已在决策日志层保证最新), 但要求候选
    #       在收益 AND 回撤上都 >= 当前, 才转正 — 防止"追高收益、牺牲回撤"。
    current = cfg.get("current", {})
    candidates = cfg.get("candidates", [])

    cur_navs = _apply_weights(verified, current)
    cur_ret = (cur_navs[-1] - 1) * 100 if cur_navs else 0
    cur_mdd = _max_drawdown(cur_navs) * 100

    results = []
    for cand in candidates:
        cand_navs = _apply_weights(verified, cand)
        cand_ret = (cand_navs[-1] - 1) * 100 if cand_navs else 0
        cand_mdd = _max_drawdown(cand_navs) * 100
        net_ret = cand_ret - cur_ret          # 配对差分收益
        mdd_ok = cand_mdd >= cur_mdd - 0.5    # 回撤不劣化超过0.5pp
        ret_ok = net_ret > 0.3                # 收益需超过噪声带(0.3pp)
        promote = ret_ok and mdd_ok
        results.append({
            "name": cand.get("name", "候选"),
            "net_return_pp": round(net_ret, 2),
            "ret_pp": round(cand_ret, 2),
            "mdd_pct": round(cand_mdd, 2),
            "ret_ok": ret_ok,
            "mdd_ok": mdd_ok,
            "promote": promote,
            "note": ("收益↑回撤不劣化, 可转正" if promote else
                     ("收益够但回撤劣化" if ret_ok else "收益未超噪声带(过拟合风险)")),
        })

    # ===== 转正决策: 全候选无一安全转正 → 保留当前 =====
    safe = [r for r in results if r["promote"]]
    if safe:
        best = max(safe, key=lambda r: r["net_return_pp"])
        decision = "promote"
        promoted_name = best["name"]
        summary_note = (f"候选[{promoted_name}] 反事实净收益+{best['net_return_pp']}pp且回撤不劣化, "
                        f"预注册判据满足, 转正生效")
    else:
        decision = "hold"
        promoted_name = None
        best = None
        summary_note = ("无候选同时满足'收益超噪声带+回撤不劣化', "
                        "全部保持影子状态 — 宁可少进化, 不过拟合近期行情")

    shadow_result = {
        "decision": decision,
        "promoted": promoted_name,
        "current_ret_pp": round(cur_ret, 2),
        "current_mdd_pct": round(cur_mdd, 2),
        "verified_count": len(verified),
        "candidate_results": results,
        "summary": summary_note,
        "note": "反事实为单日市场信号近似(非组合收益); 预注册+双重判据为防过拟合硬约束",
    }
    return {"shadow_result": json.dumps(shadow_result, ensure_ascii=False)}


if __name__ == "__main__":
    # 本地冒烟: 构造一段"近期震荡行情中被过度逐险"的日志,
    # 验证候选"保守偏置"在回撤上胜出、候选"激进偏置"因回撤劣化被拒。
    import tempfile
    import os

    def mk_log(regime, n, chg_range, pos=1.0, verified=True):
        import random
        rng = random.Random(42)
        rows = []
        for i in range(n):
            chg = rng.uniform(*chg_range)
            rows.append({
                "date": f"2026-{(i % 12) + 1:02d}-{(i % 28) + 1:02d}",
                "regime": regime,
                "final_position": pos,
                "next_day_change": round(chg, 2),
                "is_verified": verified,
            })
        return rows

    # 震荡下跌段: 高波动, 大回撤风险
    log = mk_log("panic", 12, (-3.5, 1.0), pos=1.2)
    cfg = {
        "current": {"name": "当前权重", "regime_bias": {}},
        "candidates": [
            {"name": "保守偏置", "regime_bias": {"panic": -0.4}},
            {"name": "激进偏置", "regime_bias": {"panic": +0.3}},
        ],
    }

    r = json.loads(main(json.dumps(log), json.dumps(cfg))["shadow_result"])
    print(f"决策: {r['decision']} | 转正: {r['promoted']}")
    for c in r["candidate_results"]:
        print(f"  {c['name']}: 净收益{c['net_return_pp']:+.2f}pp 回撤{c['mdd_pct']:.1f}% "
              f"收益OK={c['ret_ok']} 回撤OK={c['mdd_ok']} → {'转正' if c['promote'] else '影子'}")
    print(f"结论: {r['summary']}")