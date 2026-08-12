# -*- coding: utf-8 -*-
"""飞书进化成长看板 — 第二张王牌: 把"自我进化"做成可亲见的成长轨迹.

依赖: 已交付的 flylark_evolution_nodes.py (记忆加载A + 进化写入B).
前置: /app 文件跨运行持久+跨节点共享 (已实测).

新增 4 个可复制节点 (在 记忆加载A + 进化写入B 基础上补齐闭环):
  C  经验入库      : 接进化导师LLM输出 → 写经验库 + 更新进化状态
  D  成长统计      : 读决策日志+经验+进化状态 → 聚合逐日成长数据
  E  成长看板HTML  : 把成长数据渲染成可视化看板 (评委亲见"系统在学")
  M  进化导师助手  : 系统提示词 (每7天触发的LLM, 从已验证决策提炼经验)

编排:
  [B进化写入] ──→ (每7天) [M进化导师助手] ──→ [C经验入库]
                          [D成长统计] ──→ [E成长看板HTML] ──→ 报告节点
所有状态落 /app: decision_log.jsonl / experience_memory.json / evolution_state.json
"""

# ═══════════════════════════════════════════════════════════════
# M. 进化导师助手系统提示词 (粘到飞书"助手节点", 每7天触发一次)
# 入参 = 近期已验证决策样本 + 当前经验库; 出参 = 严格JSON
# ═══════════════════════════════════════════════════════════════
EVOLUTION_MENTOR_PROMPT = """你是【进化导师】——帮助系统自我改进的反思专家。

## 任务
从近期已验证决策中提炼可复用经验, 让系统越来越聪明。

## 核心原则
1. 反事实验证: 每条经验都要回答"如果当时用了, 结果会更好吗?"
2. 概率思维: 没有100%正确的经验, 只有特定条件下大概率有效的
3. 适用/失效条件: 每条经验必须明确在什么情况有效、什么情况失效
4. 置信度衰减: 越久远经验置信度越低 (市场在变)
5. 失败比成功更重要: 从错误中能学到更多

## 输入格式
【近期已验证决策】<决策样本 JSON>
【当前经验库】<经验库 JSON>
【当前市场环境】<regime 等>

## 输出 (严格JSON, 不要额外文字)
{
  "overall_assessment": "对近期决策的整体评价(一句话)",
  "key_insights": [
    {
      "insight": "核心洞察(一句话)",
      "applicable_conditions": ["适用条件1", "适用条件2"],
      "failure_conditions": ["失效条件1", "失效条件2"],
      "confidence": 0.0~1.0
    }
  ],
  "lessons_to_forget": ["应淘汰的旧经验ID或描述"],
  "supplementary_principles": [{"text": "纳入裁判长考量的补充原则", "confidence": 0.7}],
  "weight_adjustments": {"regime": {"trend": x, "value": y, "macro": z}},
    "可选: 派系权重调整(0.5~2.0)。没有把握就留空对象 {}"
}
"""


# ═══════════════════════════════════════════════════════════════
# C. 经验入库节点: 接进化导师LLM输出 → 写经验库 + 更新进化状态
# 入参: arg1=进化导师JSON输出, arg2=market_snapshot(含date)
# 出参: ingest_result
# ═══════════════════════════════════════════════════════════════
def nodeC_experience_ingest(arg1: str, arg2: str) -> dict:
    import json, os
    from datetime import datetime

    MEM_PATH = "/app/experience_memory.json"
    EVO_PATH = "/app/evolution_state.json"

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

    mentor = safe_load(arg1)
    market = json.loads(arg2) if arg2 else {}
    today = market.get("date") or datetime.now().strftime("%Y-%m-%d")

    # ===== 读经验库 =====
    memory = []
    if os.path.exists(MEM_PATH):
        try:
            with open(MEM_PATH, encoding="utf-8") as f:
                memory = json.load(f)
        except Exception:
            memory = []
    known_ids = {m.get("id") for m in memory}

    # ===== 新增/更新经验 =====
    added = updated = 0
    for ins in mentor.get("key_insights") or []:
        text = ins.get("insight") or ""
        if not text:
            continue
        # 去重: 相同insight视为同一条
        found = next((m for m in memory if m.get("insight") == text), None)
        if found:
            found["evidence_count"] = int(found.get("evidence_count", 1)) + 1
            found["confidence"] = min(0.95, float(found.get("confidence", 0.5)) + 0.05)
            found["last_verified_date"] = today
            updated += 1
        else:
            memory.append({
                "id": f"ins_{len(memory)+1:04d}",
                "insight": text,
                "applicable_conditions": ins.get("applicable_conditions", []),
                "failure_conditions": ins.get("failure_conditions", []),
                "confidence": float(ins.get("confidence", 0.5)),
                "evidence_count": 1,
                "created_date": today,
                "last_verified_date": today,
            })
            added += 1

    # ===== 淘汰失效经验 =====
    forget = set(mentor.get("lessons_to_forget") or [])
    before = len(memory)
    memory = [m for m in memory if m.get("insight") not in forget and m.get("id") not in forget]
    forgotten = before - len(memory)
    memory = memory[-100:]  # 上限100条

    # ===== 写经验库 =====
    with open(MEM_PATH, "w", encoding="utf-8") as f:
        json.dump(memory, f, ensure_ascii=False, indent=2)

    # ===== 更新进化状态 =====
    evo = {"evolution_count": 0, "last_evolution_date": None, "last_accuracy": None,
           "weight_adjustments": {}, "supplementary_principles": [], "history": []}
    if os.path.exists(EVO_PATH):
        try:
            with open(EVO_PATH, encoding="utf-8") as f:
                evo.update(json.load(f))
        except Exception:
            pass

    # 补充原则
    principles = []
    for p in mentor.get("supplementary_principles") or []:
        if isinstance(p, dict) and p.get("text"):
            principles.append({"text": p["text"],
                               "confidence": float(p.get("confidence", 0.6)),
                               "applied_from": today})
    if principles:
        known = {p["text"] for p in evo["supplementary_principles"]}
        evo["supplementary_principles"] = [p for p in principles if p["text"] not in known] \
            + evo["supplementary_principles"]
        evo["supplementary_principles"] = evo["supplementary_principles"][:20]

    # 权重调整 (规范化, 只保留合法值)
    for regime, schools in (mentor.get("weight_adjustments") or {}).items():
        if isinstance(schools, dict):
            norm = {}
            for school, mult in schools.items():
                try:
                    m = float(mult)
                except (TypeError, ValueError):
                    continue
                if school in ("trend", "value", "macro") and 0.5 <= m <= 2.0:
                    norm[school] = m
            if norm:
                evo["weight_adjustments"][regime] = norm

    evo["evolution_count"] = int(evo.get("evolution_count", 0)) + 1
    evo["last_evolution_date"] = today
    evo["history"].append({
        "date": today,
        "summary": mentor.get("overall_assessment", ""),
        "principles": [p["text"] for p in principles],
        "weight_adjustments": mentor.get("weight_adjustments") or {},
    })
    evo["history"] = evo["history"][-20:]

    with open(EVO_PATH, "w", encoding="utf-8") as f:
        json.dump(evo, f, ensure_ascii=False, indent=2)

    result = {
        "today": today,
        "insights_added": added,
        "insights_updated": updated,
        "forgotten": forgotten,
        "memory_total": len(memory),
        "evolution_count": evo["evolution_count"],
        "principles_total": len(evo["supplementary_principles"]),
        "weight_adjustments": evo["weight_adjustments"],
    }
    return {"ingest_result": json.dumps(result, ensure_ascii=False)}


# ═══════════════════════════════════════════════════════════════
# D. 成长统计节点: 读三份状态 → 聚合逐日成长数据 (喂给看板E)
# 入参: arg1=任意(忽略); 出参: growth_data
# ═══════════════════════════════════════════════════════════════
def nodeD_growth_stats(arg1: str) -> dict:
    import json, os

    LOG_PATH = "/app/decision_log.jsonl"
    MEM_PATH = "/app/experience_memory.json"
    EVO_PATH = "/app/evolution_state.json"

    # ===== 决策日志 =====
    rows = []
    if os.path.exists(LOG_PATH):
        for l in open(LOG_PATH, encoding="utf-8"):
            l = l.strip()
            if l:
                try:
                    rows.append(json.loads(l))
                except Exception:
                    pass
    rows.sort(key=lambda x: x.get("date", ""))

    # ===== 逐日成长曲线 =====
    day_curve = []          # [{date, decisions, correct, verified, acc, risk}]
    cum_decisions = cum_verified = cum_correct = 0
    for r in rows:
        date = r.get("date", "")
        cum_decisions += 1
        if r.get("is_verified"):
            cum_verified += 1
            if r.get("actual_result", {}).get("was_correct") is True:
                cum_correct += 1
        acc = round(cum_correct / cum_verified, 3) if cum_verified else None
        day_curve.append({
            "date": date,
            "decisions": cum_decisions,
            "verified": cum_verified,
            "correct": cum_correct,
            "acc": acc,
            "risk": r.get("final_risk"),
        })

    # ===== 经验库 =====
    memory = []
    if os.path.exists(MEM_PATH):
        try:
            with open(MEM_PATH, encoding="utf-8") as f:
                memory = json.load(f)
        except Exception:
            memory = []

    # ===== 进化状态 =====
    evo = {"evolution_count": 0, "supplementary_principles": [], "history": []}
    if os.path.exists(EVO_PATH):
        try:
            with open(EVO_PATH, encoding="utf-8") as f:
                evo.update(json.load(f))
        except Exception:
            pass

    return {"growth_data": json.dumps({
        "days": len(day_curve),
        "curve": day_curve[-30:],            # 最近30天曲线
        "memory_total": len(memory),
        "memory": memory[:10],
        "evolution_count": evo.get("evolution_count", 0),
        "last_accuracy": evo.get("last_accuracy"),
        "principles": evo.get("supplementary_principles", [])[:10],
        "evolution_history": evo.get("history", [])[-10:],
    }, ensure_ascii=False)}


# ═══════════════════════════════════════════════════════════════
# E. 成长看板HTML节点: 把成长数据渲染成可视化看板 (深色主题, 纯CSS/SVG)
# 入参: arg1=growth_data(节点D输出); 出参: report_html
# ═══════════════════════════════════════════════════════════════
def nodeE_growth_dashboard_html(arg1: str) -> dict:
    import json

    g = json.loads(arg1) if arg1 else {}
    curve = g.get("curve", [])
    memory = g.get("memory", [])
    principles = g.get("principles", [])
    evo_hist = g.get("evolution_history", [])

    days = g.get("days", 0)
    mem_total = g.get("memory_total", 0)
    evo_count = g.get("evolution_count", 0)
    last_acc = g.get("last_accuracy")

    # ===== 最近正确率 =====
    accs = [c["acc"] for c in curve if c.get("acc") is not None]
    latest_acc = accs[-1] if accs else None

    # ===== SVG 折线: 正确率 / 决策数 / 经验数 (归一化条形) =====
    n = len(curve)
    W, H = 720, 180
    def _poly(seq, norm):
        if not seq:
            return ""
        pts = []
        for i, v in enumerate(seq):
            x = i * W / max(n - 1, 1)
            y = H - (v / norm) * (H - 20) - 10
            pts.append(f"{x:.0f},{y:.0f}")
        return " ".join(pts)
    acc_seq = [c["acc"] * 100 if c.get("acc") is not None else None for c in curve]
    dec_seq = [c["decisions"] for c in curve]
    acc_filled = [v for v in acc_seq if v is not None]
    acc_norm = max(acc_filled) if acc_filled and max(acc_filled) > 0 else 100

    def _poly2(seq, norm):
        pts = []
        for i, v in enumerate(seq):
            if v is None:
                continue
            x = i * W / max(n - 1, 1)
            y = H - (v / norm) * (H - 20) - 10
            pts.append(f"{x:.0f},{y:.0f}")
        return " ".join(pts)

    # ===== 风险演变色块 =====
    RISK_COLOR = {1: "#3fb950", 2: "#56d364", 3: "#d29922", 4: "#f0883e", 5: "#f85149"}
    risk_blocks = ""
    for c in curve:
        r = c.get("risk")
        col = RISK_COLOR.get(r, "#30363d")
        risk_blocks += f'<div style="flex:1;height:14px;background:{col};opacity:.85;"></div>'
    risk_row = f'<div style="display:flex;gap:1px;margin-top:6px;">{risk_blocks}</div>' if curve else ""

    # ===== 进化时间线 =====
    evo_lines = ""
    for h in evo_hist:
        evo_lines += (f'<div style="padding:8px 0;border-bottom:1px solid #21262d;">'
                      f'<div style="color:#a371f7;font-size:12px;">🧬 进化 #{h.get("date","")}</div>'
                      f'<div style="font-size:13px;color:#c9d1d9;margin-top:2px;">{h.get("summary","")}</div>'
                      f'</div>')
    evo_section = (f'<div class="card"><div class="card-title">🧬 进化历史</div>{evo_lines or "<div style=color:#8b949e;font-size:13px;>暂无进化记录</div>"}</div>'
                   if evo_hist else "")

    # ===== 补充原则 =====
    prin_lines = "".join(
        f'<li style="padding:4px 0;font-size:13px;color:#c9d1d9;">· {p.get("text","")}'
        f' <span style="color:#8b949e;font-size:11px;">(置信度{p.get("confidence",0):.0%})</span></li>'
        for p in principles)
    prin_section = (f'<div class="card"><div class="card-title">📜 进化沉淀原则</div>'
                    f'<ul style="list-style:none;padding:0;margin:0;">{prin_lines}</ul></div>'
                    if principles else "")

    # ===== 经验精选 =====
    mem_lines = ""
    for m in memory:
        conds = "，".join(m.get("applicable_conditions", [])[:2])
        mem_lines += (f'<div style="padding:8px 0;border-bottom:1px solid #21262d;">'
                      f'<div style="font-size:13px;color:#c9d1d9;">💡 {m.get("insight","")}</div>'
                      f'<div style="font-size:11px;color:#8b949e;margin-top:2px;">'
                      f'置信度{m.get("confidence",0):.0%} · 已验证{m.get("evidence_count",0)}次'
                      f'{" · 适用:"+conds if conds else ""}</div></div>')
    mem_section = (f'<div class="card"><div class="card-title">🧠 经验库精选</div>{mem_lines or "<div style=color:#8b949e;font-size:13px;>暂无经验，系统正在学习中</div>"}</div>'
                   if memory else "")

    html = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="UTF-8">
<title>进化成长看板</title><style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:-apple-system,"Segoe UI","PingFang SC","Microsoft YaHei",sans-serif;background:#0d1117;color:#c9d1d9;padding:20px;line-height:1.6}}
.container{{max-width:1100px;margin:0 auto}}
.header{{background:linear-gradient(135deg,#161b22,#1f6feb22);border:1px solid #30363d;border-radius:12px;padding:28px;text-align:center;margin-bottom:20px}}
.header h1{{font-size:26px;color:#f0f6fc}}
.header .sub{{color:#8b949e;font-size:13px;margin-top:6px}}
.metrics{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin-bottom:20px}}
.metric{{background:#161b22;border:1px solid #30363d;border-radius:10px;padding:16px;text-align:center}}
.metric .v{{font-size:26px;font-weight:700;color:#58a6ff}}
.metric .l{{font-size:12px;color:#8b949e;margin-top:4px}}
.card{{background:#161b22;border:1px solid #30363d;border-radius:10px;padding:18px;margin-bottom:16px}}
.card-title{{font-size:15px;color:#f0f6fc;font-weight:600;margin-bottom:12px;padding-bottom:8px;border-bottom:1px solid #21262d}}
.chart-title{{font-size:12px;color:#8b949e;margin-top:8px}}
</style></head><body><div class="container">
<div class="header">
<h1>🧬 系统进化成长看板</h1>
<div class="sub">运行 {days} 天 · 累计进化 {evo_count} 次 · 经验库 {mem_total} 条</div>
</div>
<div class="metrics">
<div class="metric"><div class="v">{days}</div><div class="l">运行天数</div></div>
<div class="metric"><div class="v">{latest_acc * 100 if latest_acc is not None else "—":.0f}%</div><div class="l">最新决策正确率</div></div>
<div class="metric"><div class="v">{evo_count}</div><div class="l">进化代数</div></div>
<div class="metric"><div class="v">{mem_total}</div><div class="l">经验条目</div></div>
</div>
<div class="card">
<div class="card-title">📈 决策正确率成长曲线</div>
<svg width="100%" viewBox="0 0 {W} {H}" preserveAspectRatio="none">
<line x1="0" y1="{H-10}" x2="{W}" y2="{H-10}" stroke="#30363d" stroke-width="1"/>
<polyline points="{_poly2(acc_seq, acc_norm)}" fill="none" stroke="#3fb950" stroke-width="2"/>
</svg>
<div class="chart-title">纵轴=逐日累积正确率(%) · 横轴=运行天数</div>
</div>
<div class="card">
<div class="card-title">🚦 风险等级演变（色块=每日 final_risk）</div>
{risk_row}
<div class="chart-title"><span style="color:#3fb950">■</span>低估1-2 <span style="color:#d29922">■</span>正常3 <span style="color:#f0883e">■</span>偏高4 <span style="color:#f85149">■</span>泡沫5</div>
</div>
{evo_section}
{prin_section}
{mem_section}
<div style="text-align:center;color:#484f58;font-size:11px;padding:16px">⚠️ 正确率为市场级信号(非组合收益)，仅供演示与研究</div>
</div></body></html>"""

    return {"report_html": html}


if __name__ == "__main__":
    # 本地冒烟: 造一份模拟成长数据, 验证 D→E 渲染
    import json as _json
    import tempfile, os as _os
    tmp = tempfile.mkdtemp().replace("\\", "/")
    src = open(__file__, encoding="utf-8").read().replace('"/app/', f'"{tmp}/')
    ns = {}
    exec(src, ns)
    nodeA, nodeB = None, None
    # 用 flylark_evolution_nodes 里的 B 造几天决策日志
    import importlib.util
    spec = importlib.util.spec_from_file_location("e", "flylark_evolution_nodes.py")
    e = importlib.util.module_from_spec(spec); spec.loader.exec_module(e)
    e_src = open("flylark_evolution_nodes.py", encoding="utf-8").read().replace('"/app/', f'"{tmp}/')
    ens = {}; exec(e_src, ens)
    nodeB = ens["nodeB_evolution_writer"]
    # 造 5 天决策
    for i, (d, chg, risk) in enumerate([
            ("2026-08-08", 1.5, 2), ("2026-08-09", -1.2, 4),
            ("2026-08-10", 0.8, 3), ("2026-08-11", -2.0, 4), ("2026-08-12", 1.0, 2)]):
        mkt = f'{{"date":"{d}","regime":"mixed","avg_change_pct":{chg}}}'
        diag = f'{{"final_risk_level":{risk},"final_position_multiplier":1.0,"consensus_score":0.6,"confidence":0.6}}'
        nodeB(diag, mkt)
    # 模拟一次进化入库
    mentor_out = _json.dumps({
        "overall_assessment": "震荡市判断较准，极端行情需更谨慎",
        "key_insights": [{"insight": "跌停家数>200时恐慌可能超预期，勿急于抄底",
                          "applicable_conditions": ["恐慌下跌", "流动性收紧"],
                          "failure_conditions": ["政策强力救市"], "confidence": 0.7}],
        "supplementary_principles": [{"text": "极端行情下自动降低置信度", "confidence": 0.75}],
        "weight_adjustments": {"panic": {"trend": 0.8, "value": 1.3, "macro": 1.1}},
    })
    nodeC = ns["nodeC_experience_ingest"]
    print("入库:", nodeC(mentor_out, '{"date":"2026-08-12"}')["ingest_result"])
    nodeD = ns["nodeD_growth_stats"]
    gd = nodeD("")["growth_data"]
    g = _json.loads(gd)
    print("成长统计: days=%d mem=%d evo=%d days_in_curve=%d" % (g["days"], g["memory_total"], g["evolution_count"], len(g["curve"])))
    nodeE = ns["nodeE_growth_dashboard_html"]
    html = nodeE(gd)["report_html"]
    print("看板HTML长度:", len(html), "| 含正确率:", "决策正确率成长曲线" in html, "| 含原则:", "进化沉淀原则" in html)