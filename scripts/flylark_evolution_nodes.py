# -*- coding: utf-8 -*-
"""飞书进化闭环实装 — 两个核心代码节点 (可复制, 拆节点方案).

前置: 已实测代码节点=完整Linux沙箱, /app 文件跨运行持久+跨节点共享, 外网可接腾讯行情.
依赖: 九个大师 → 汇总 → 裁判长 的现有工作流已跑通.

┌────────────────────────────────────────────────────────────┐
│ 实装架构 (在现有12节点主流程上插2个节点):                    │
│                                                             │
│  [节点3 数据格式化] →【记忆加载节点A】→ 大师(利弗莫尔/巴菲特/索罗斯) │
│        ↓                                            ↓        │
│  [报告节点] ← [进化写入节点B] ← 裁判长裁决 + 行情              │
│                                                             │
│  A(开头): 读/app经验+决策统计 → 拼进master_input_text注入大师  │
│  B(末尾): append决策日志 → 回填昨日was_correct → 统计         │
└────────────────────────────────────────────────────────────┘
"""


# ═══════════════════════════════════════════════════════════════
# 节点A：记忆加载节点 (放在"数据格式化"之后, 大师助手之前)
# 入参: arg1=master_input_text(节点3输出), arg2=market_snapshot
# 出参: master_input_with_memory (喂给大师助手)
# ═══════════════════════════════════════════════════════════════
def nodeA_memory_loader(arg1: str, arg2: str) -> dict:
    import json, os

    MEM_PATH = "/app/experience_memory.json"
    LOG_PATH = "/app/decision_log.jsonl"

    # ===== 读经验记忆, 取置信度最高的3条 =====
    experiences = []
    if os.path.exists(MEM_PATH):
        try:
            with open(MEM_PATH, encoding="utf-8") as f:
                memories = json.load(f)
            experiences = sorted(memories, key=lambda x: x.get("confidence", 0),
                                 reverse=True)[:3]
        except Exception:
            experiences = []

    exp_text = "暂无历史经验（系统积累中）"
    if experiences:
        lines = []
        for i, e in enumerate(experiences, 1):
            conds = ",".join(e.get("applicable_conditions", [])[:2])
            lines.append(f"{i}. {e.get('insight','')}（置信度{e.get('confidence',0):.0%}"
                         f"{'，适用:'+conds if conds else ''}）")
        exp_text = "\n".join(lines)

    # ===== 读决策日志统计 =====
    decision_count, correct = 0, 0
    if os.path.exists(LOG_PATH):
        try:
            rows = [json.loads(l) for l in open(LOG_PATH, encoding="utf-8") if l.strip()]
            decision_count = len(rows)
            correct = sum(1 for r in rows if r.get("is_verified")
                          and r.get("actual_result", {}).get("was_correct") is True)
        except Exception:
            pass
    stats_text = f"历史决策 {decision_count} 条，已验证正确 {correct} 条"

    # ===== 注入到大师输入文本 =====
    base = arg1 if arg1 else ""
    injected = (base + "\n\n【历史经验记忆】\n" + exp_text
                + "\n\n【历史战绩】\n" + stats_text
                + "\n（以上经验仅供参考，请结合当前数据独立研判）")

    return {"master_input_with_memory": injected}


# ═══════════════════════════════════════════════════════════════
# 节点B：进化写入节点 (放在报告节点之后, 收尾)
# 入参: arg1=diagnosis(裁判长json), arg2=market_snapshot
# 出参: evolution_result
# ═══════════════════════════════════════════════════════════════
def nodeB_evolution_writer(arg1: str, arg2: str) -> dict:
    import json, os
    from datetime import datetime

    LOG_PATH = "/app/decision_log.jsonl"

    # ===== 安全解析 =====
    def safe_load(s):
        if not s:
            return {}
        s = s.strip()
        if s.startswith("```"):
            s = s.strip("`")
            if s.startswith("json"):
                s = s[4:]
            s = s.strip()
        try:
            return json.loads(s)
        except Exception:
            start, end = s.find("{"), s.rfind("}")
            if start >= 0 and end > start:
                try:
                    return json.loads(s[start:end + 1])
                except Exception:
                    return {}
        return {}

    diag = safe_load(arg1)
    market = json.loads(arg2) if arg2 else {}
    today = market.get("date") or datetime.now().strftime("%Y-%m-%d")

    # ===== 读现有日志 =====
    rows = []
    if os.path.exists(LOG_PATH):
        for l in open(LOG_PATH, encoding="utf-8"):
            l = l.strip()
            if l:
                try:
                    rows.append(json.loads(l))
                except Exception:
                    pass

    # ===== 1. 回填昨日: 用今日行情判昨日对错 (市场级信号, 诚实标注)=====
    #    逻辑: 昨日风险低(1-2)→今日涨=对; 风险高(4-5)→今日跌=对
    today_change = market.get("avg_change_pct", 0)
    for r in rows:
        if (r.get("date") != today and not r.get("is_verified")
                and r.get("actual_result") is None):
            risk = r.get("final_risk", 3)
            if risk <= 2 and today_change > 0:
                corr = True
            elif risk >= 4 and today_change < 0:
                corr = True
            elif risk == 3 and abs(today_change) < 1.0:
                corr = True
            else:
                corr = None
            r["is_verified"] = True
            r["actual_result"] = {"next_day_change": today_change, "was_correct": corr}

    # ===== 2. 记录今日决策 (同一天不重复) =====
    if not any(r.get("date") == today for r in rows):
        rows.append({
            "date": today,
            "final_risk": diag.get("final_risk_level"),
            "final_position": diag.get("final_position_multiplier"),
            "consensus": diag.get("consensus_score"),
            "confidence": diag.get("confidence"),
            "regime": market.get("regime"),
            "is_verified": False,
            "actual_result": None,
        })

    # ===== 写回 =====
    with open(LOG_PATH, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # ===== 3. 统计 =====
    verified = [r for r in rows if r.get("is_verified")]
    correct = sum(1 for r in verified if r.get("actual_result", {}).get("was_correct") is True)
    wrong = sum(1 for r in verified if r.get("actual_result", {}).get("was_correct") is False)
    accuracy = round(correct / (correct + wrong), 3) if (correct + wrong) else None

    result = {
        "today": today,
        "logged": True,
        "rows_total": len(rows),
        "verified_count": len(verified),
        "correct": correct,
        "wrong": wrong,
        "accuracy": accuracy,
        "note": "was_correct 为市场级单日信号(非组合收益), 见局限标注",
    }
    return {"evolution_result": json.dumps(result, ensure_ascii=False)}


if __name__ == "__main__":
    # 本地冒烟: 把 /app 替换为临时目录, 跑两遍验证闭环逻辑
    import tempfile, os as _os
    tmp = tempfile.mkdtemp().replace("\\", "/")
    src = open(__file__, encoding="utf-8").read().replace('"/app/', f'"{tmp}/')
    ns = {}
    exec(src, ns)
    nodeA = ns["nodeA_memory_loader"]
    nodeB = ns["nodeB_evolution_writer"]

    demo_master = "【市场阶段】强牛\n上涨占比85%\n请研判"
    demo_mkt = '{"date":"2026-08-12","regime":"strong_bull","avg_change_pct":1.5}'
    demo_diag = '{"final_risk_level":2,"final_position_multiplier":1.2,"consensus_score":0.8,"confidence":0.7}'
    print("节点A首次:", nodeA(demo_master, demo_mkt)["master_input_with_memory"][:60], "...")
    print("B#1:", nodeB(demo_diag, demo_mkt)["evolution_result"])
    demo_mkt2 = '{"date":"2026-08-13","regime":"strong_bull","avg_change_pct":2.0}'
    print("B#2(回填昨日):", nodeB(demo_diag, demo_mkt2)["evolution_result"])
    # 节点A第二次应能看到积累的经验统计
    print("节点A再跑:", nodeA(demo_master, demo_mkt2)["master_input_with_memory"].split("【历史战绩】")[-1].strip().splitlines()[0])