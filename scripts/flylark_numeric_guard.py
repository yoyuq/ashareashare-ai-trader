# -*- coding: utf-8 -*-
"""飞书 Code-as-Reasoning 数值护栏 — 第四张王牌: 防LLM编造数字.

原理: 所有关键数值由代码节点计算(可信数值池), LLM只写叙事。本节点扫描LLM文本中的
每个数字, 对照数值池, 找出"代码没算过、纯属LLM幻觉"的数字 → 报告里如实标注审计结果。

用法: 替换飞书一个代码节点, 放在报告节点之前或之后:
  入参 arg1 = 待审文本(裁判长 view_summary + 大师 key_reasons... 的拼接)
        arg2 = 数值池来源(可传 market_snapshot + summary 的JSON拼接)
  出参: audit_result (含 passed / suspicious名单 / 可信数字池规模)
"""
import json
import re


def main(arg1: str, arg2: str) -> dict:
    # ===== 从 arg2 提取可信数值池 (递归收集所有 int/float) =====
    def _collect(v, pool):
        if isinstance(v, bool):
            return
        if isinstance(v, (int, float)):
            pool.add(float(v))
        elif isinstance(v, dict):
            for x in v.values():
                _collect(x, pool)
        elif isinstance(v, (list, tuple)):
            for x in v:
                _collect(x, pool)

    pool = set()
    for chunk in (arg2 or "").split("||"):
        try:
            _collect(json.loads(chunk), pool)
        except Exception:
            pass

    # ===== 扫描 arg1 文本中的数字 =====
    text = arg1 or ""
    # 匹配 数字(含小数/负号/千分位), 排除日期/年份
    nums = re.findall(r'-?\d+(?:\.\d+)?', text)

    def is_trusted(f):
        # 容差: 绝对0.01 或 相对5% (容忍四舍五入/约数)
        for p in pool:
            if abs(f - p) <= max(0.01, abs(p) * 0.05):
                return True
            # 百分比归一化: 池里 0.85 (小数比例) ↔ 文本 85(%)
            # 容差基准用换算后的值, 不是原始 p (否则池里大数会误放行一切)
            if 1 <= f <= 100 and abs(f - p * 100) <= max(0.01, abs(p * 100) * 0.05):
                return True
            if abs(p) >= 1 and abs(f - p / 100) <= max(0.01, abs(p / 100) * 0.05):
                return True
        return False

    suspicious = []
    for n in nums:
        try:
            f = float(n)
        except ValueError:
            continue
        # 年份/日期(1700~2200整数)视为上下文, 不审
        if abs(f) >= 1700 and abs(f) <= 2200 and f == int(f):
            continue
        if not is_trusted(f):
            suspicious.append(n)

    # 去重保序
    seen = set()
    uniq = [n for n in suspicious if not (n in seen or seen.add(n))]

    passed = len(uniq) == 0
    result = {
        "passed": passed,
        "pool_count": len(pool),
        "text_digits": len(nums),
        "suspicious": uniq[:20],
        "suspicious_count": len(uniq),
        "audit_note": ("✅ 文本所有数字均在代码计算的数值池内(Code-as-Reasoning生效)"
                       if passed else
                       f"⚠️ 发现 {len(uniq)} 个代码未计算的数字, LLM可能编造: {uniq[:10]}"),
    }
    return {"audit_result": json.dumps(result, ensure_ascii=False)}


if __name__ == "__main__":
    # 本地冒烟
    pool_src = json.dumps({"avg_change_pct": 1.5, "up_ratio": 0.85, "limit_up": 120,
                           "median_pe": 42.3, "consensus_score": 0.75}) + "||" + \
               json.dumps({"risk": 3, "position": 1.2})

    honest = ("市场上涨1.5%，上涨占比85%，涨停120家，共识度0.75，建议仓位1.2，风险3级")
    hallu = ("市场上涨1.5%，但历史牛市平均涨9.8%，我预测未来涨17%，回撤控制到2.3%")

    for label, txt in (("诚实文本", honest), ("含幻觉文本", hallu)):
        r = json.loads(main(txt, pool_src)["audit_result"])
        print(f"{label}: passed={r['passed']} 可疑={r['suspicious']} | {r['audit_note']}")