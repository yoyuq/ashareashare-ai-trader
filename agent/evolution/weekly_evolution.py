"""周期性进化总结 — 智慧层。

每 N 天汇总所有复盘记录，让 LLM 做一次"深度反思"：
1. 回顾这段时间的所有成功/失败
2. 提炼长期有效的原则（可以更新到核心信念里）
3. 识别自己的系统性偏见（比如"总是低估风险"）
4. 输出"进化后的核心原则"，下次诊断时作为系统提示词的一部分

这是自我进化的最高层 — 不是记住一个个案例，而是提炼出规律。
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


SYSTEM_PROMPT = """你是一位交易系统的进化导师。

你的工作是：回顾过去一段时间的所有诊断决策和复盘记录，
从中提炼出长期有效的原则，识别系统性的偏见，
最终输出"进化后的核心指导原则"，让系统在下一周期表现更好。

## ⚠️ 最重要的平衡原则
进化的目标是**提升总收益**，不是**减少错误次数**。

很多系统越进化越保守，因为：
- 亏损是"看得见的痛" → 印象深刻 → 快速学会规避
- 踏空是"看不见的遗憾" → 印象模糊 → 不容易学到教训

但在交易中，**踏空的代价和被套的代价一样真实**。
你必须确保进化是双向的：既要减少"不该上的时候乱上"，也要减少"该上的时候不敢上"。

如果发现系统有保守型偏见（该上没上的错误 > 不该上却上的错误），
必须主动输出"更积极进攻"的原则，而不是只会加"小心风险"的原则。

## 你的分析框架

### 1. 总体表现评估
- 这段时间总收益是多少？趋势是向上还是向下？
- 保守型错误（该上没上）有几次？激进型错误（不该上却上）有几次？
- 哪类行情做得好，哪类做得差？
- 是"赚得不够多"还是"亏得不够少"是主要问题？

### 2. 成功模式提炼
- 反复出现的正确判断有什么共性？
- 哪位大师在什么场景下最可靠？
- 哪些信号/指标最有预测力？
- 进攻正确的时候多不多？（市场涨时你也看多）

### 3. 失败模式识别
- 反复犯的错误是什么？（系统性偏见）
- **保守型错误 vs 激进型错误的比例是多少？**
- 如果保守型错误更多，说明系统太胆小，需要增加进攻性原则
- 如果激进型错误更多，说明系统太鲁莽，需要增加防守性原则
- 错误是大师选错了，还是仓位算错了，还是风险评估错了？

### 4. 核心原则更新
根据上面的分析，给出 **5-8 条核心原则**。
每条原则要具体、可操作，不能是鸡汤。
原则格式:
- ✅ 应该做的 / ❌ 不应该做的 + 具体说明

**关键约束 1: 原则中至少要有 1-2 条是"进攻型"原则（告诉系统什么时候该大胆上），
不能全是"小心风险""谨慎行事"的防守型原则。**

**关键约束 2: ⚠️ 大师选择锚定原则**
不要轻易建议"换大师"。五位大师的框架已经很完善了，问题通常不是"大师选错了"，
而是"在这位大师的框架下，仓位/风险等级算得不够准"。

你的核心原则应该聚焦在：
- 什么情况下仓位系数应该更大/更小
- 什么情况下风险等级应该加1级/减1级
- 什么信号最值得关注，什么信号可以忽略

而不是：
- "XX大师不好用，应该换成YY大师"
- "这种场景应该选XX大师而不是YY"

大师选择是诊断官的事，你的工作是优化诊断官的判断力，不是替它选大师。

只有当某一位大师在同一场景下反复出错（≥3次），才可以在 master_usage_tips 里提醒。

## 输出格式 (严格JSON)
{
  "period_summary": "一句话总结这段时间的表现",
  "strengths": ["强项1", "强项2"],
  "weaknesses": ["弱项1", "弱项2"],
  "bias_type": "conservative / aggressive / balanced （系统整体偏向保守/激进/均衡）",
  "biases_identified": ["识别到的系统性偏见1", "偏见2"],
  "core_principles": [
    {"id": 1, "principle": "原则内容", "rationale": "为什么这条原则重要", "confidence": 0.5~0.9, "type": "offensive / defensive / neutral"},
    ...
  ],
  "master_usage_tips": [
    {"master": "大师名", "best_for": "最适合的场景", "avoid_for": "要避免的场景"},
    ...
  ],
  "next_period_focus": "下个周期最该改进的一个方面"
}
"""


@dataclass
class EvolutionSnapshot:
    """一个周期的进化总结。"""
    period_start: str
    period_end: str
    num_decisions: int
    summary: dict
    principles: list = field(default_factory=list)
    master_tips: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "period_start": self.period_start,
            "period_end": self.period_end,
            "num_decisions": self.num_decisions,
            "summary": self.summary,
            "principles": self.principles,
            "master_tips": self.master_tips,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "EvolutionSnapshot":
        return cls(
            period_start=d["period_start"],
            period_end=d["period_end"],
            num_decisions=d["num_decisions"],
            summary=d["summary"],
            principles=d.get("principles", []),
            master_tips=d.get("master_tips", []),
        )


class EvolutionManager:
    """周期性进化管理器。"""

    def __init__(self, path: str | Path, period_days: int = 7, rolling_window: int = 20):
        """
        Args:
            period_days: 每隔多少天做一次进化总结 (v4.2: 7天，原10天)
            rolling_window: 进化总结只看最近多少天的决策 (v4.2: 20天滚动窗口)
        """
        self.path = Path(path)
        self.period_days = period_days
        self.rolling_window = rolling_window
        self.snapshots: list[EvolutionSnapshot] = []
        self._load()

    def _load(self):
        if not self.path.exists():
            return
        with open(self.path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.snapshots = [EvolutionSnapshot.from_dict(d) for d in data.get("snapshots", [])]

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(
                {"snapshots": [s.to_dict() for s in self.snapshots]},
                f, ensure_ascii=False, indent=2
            )

    def should_evolve(self, current_date: str, decision_count: int) -> bool:
        """判断是否到了进化周期。"""
        if not self.snapshots:
            return decision_count >= self.period_days
        last_end = self.snapshots[-1].period_end
        # 距离上次进化已经过了 period_days 个决策日
        return decision_count - len(self.snapshots) * self.period_days >= self.period_days

    async def evolve(self, decisions: list, memory_items: list) -> Optional[EvolutionSnapshot]:
        """执行一次进化总结。

        Args:
            decisions: 本周期内的所有决策记录（含复盘）
            memory_items: 本周期内积累的经验条目

        Returns:
            EvolutionSnapshot，失败返回 None
        """
        if len(decisions) < 3:
            return None

        # v4.2: 滚动窗口 — 只看最近 rolling_window 天的决策，防止远古经验拖累
        if len(decisions) > self.rolling_window:
            decisions = decisions[-self.rolling_window:]

        try:
            from openai import AsyncOpenAI

            client = AsyncOpenAI(
                api_key=os.getenv("DEEPSEEK_API_KEY"),
                base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
                timeout=60.0,
            )

            # 构建决策摘要（精简，不给全部原文）
            decision_summary = []
            for d in decisions:
                review = d.review or {}
                decision_summary.append(
                    f"[{d.date}] 阶段:{d.market_phase} 大师:{d.dominant_master} "
                    f"风险:{d.risk_level}/5 仓位:×{d.position_multiplier:.2f} "
                    f"→ 复盘:{review.get('verdict', '未复盘')} "
                    f"偏差:{review.get('risk_level_deviation', 0)}"
                )

            # 构建经验摘要
            exp_summary = []
            for item in memory_items[:20]:  # 最多20条
                exp_summary.append(
                    f"- [{item.verdict}] {item.lesson_title} | {item.master_used}"
                )

            user_msg = (
                f"【分析周期】{decisions[0].date} ~ {decisions[-1].date}\n"
                f"共 {len(decisions)} 个交易日\n\n"
                f"【决策历史摘要】\n" + "\n".join(decision_summary) + "\n\n"
                f"【积累的经验教训】\n" + "\n".join(exp_summary) + "\n\n"
                f"请进行深度反思，输出进化总结。"
            )

            resp = await client.chat.completions.create(
                model=os.getenv("DEEPSEEK_FLASH_MODEL", "deepseek-v4-flash"),
                messages=[{"role": "system", "content": SYSTEM_PROMPT},
                          {"role": "user", "content": user_msg}],
                temperature=0.4,
                max_tokens=1500,
                extra_body={"thinking": {"type": "disabled"}},
            )

            content = (resp.choices[0].message.content or "").strip()
            if content.startswith("```"):
                content = content.strip("`")
                if content.lower().startswith("json"):
                    content = content[4:].strip()

            result = None
            try:
                result = json.loads(content)
            except json.JSONDecodeError:
                s, e = content.find("{"), content.rfind("}")
                if s >= 0 and e > s:
                    try:
                        result = json.loads(content[s:e+1])
                    except json.JSONDecodeError:
                        pass

            if result is None:
                return None

            snapshot = EvolutionSnapshot(
                period_start=decisions[0].date,
                period_end=decisions[-1].date,
                num_decisions=len(decisions),
                summary={
                    "period_summary": result.get("period_summary", ""),
                    "strengths": result.get("strengths", []),
                    "weaknesses": result.get("weaknesses", []),
                    "biases_identified": result.get("biases_identified", []),
                    "next_period_focus": result.get("next_period_focus", ""),
                },
                principles=result.get("core_principles", []),
                master_tips=result.get("master_usage_tips", []),
            )

            self.snapshots.append(snapshot)
            self.save()
            return snapshot

        except Exception as e:
            print(f"[进化总结] 失败: {e}")
            return None

    def get_latest_principles_text(self) -> str:
        """获取最新一周期的核心原则，格式化成提示词片段。"""
        if not self.snapshots:
            return ""
        latest = self.snapshots[-1]
        if not latest.principles:
            return ""

        lines = ["## 你自己总结的核心原则（从历史经验中提炼）"]
        for p in latest.principles:
            pid = p.get("id", "")
            text = p.get("principle", "")
            conf = p.get("confidence", 0)
            rationale = p.get("rationale", "")
            lines.append(f"- 原则{pid}: {text} (置信度 {conf:.0%})")
            if rationale:
                lines.append(f"  → {rationale}")
        lines.append("")
        return "\n".join(lines)

    def get_master_tips_text(self) -> str:
        """获取大师使用技巧，格式化成提示词片段。"""
        if not self.snapshots:
            return ""
        latest = self.snapshots[-1]
        if not latest.master_tips:
            return ""

        lines = ["## 大师使用心得（你自己总结的）"]
        for t in latest.master_tips:
            master = t.get("master", "")
            best = t.get("best_for", "")
            avoid = t.get("avoid_for", "")
            lines.append(f"- {master}: 最适合{best}；避免在{avoid}时使用")
        lines.append("")
        return "\n".join(lines)

    def get_summary_text(self) -> str:
        """获取最近一期进化的自我总结, 格式化成提示词片段.

        v5 落地: summary 之前算完即弃, 现在回注到诊断 prompt, 让 LLM 看到自己
        上个周期的强项/待改进/应避免的偏见/下周期重点.
        """
        if not self.snapshots:
            return ""
        latest = self.snapshots[-1]
        s = latest.summary or {}
        lines = ["## 你上个周期的自我总结（来自进化复盘）"]
        if s.get("period_summary"):
            lines.append(f"- 总结: {s['period_summary']}")
        if s.get("strengths"):
            lines.append(f"- 强项: {'、'.join(s['strengths'])}")
        if s.get("weaknesses"):
            lines.append(f"- 待改进: {'、'.join(s['weaknesses'])}")
        if s.get("biases_identified"):
            lines.append(f"- 应避免的偏见: {'、'.join(s['biases_identified'])}")
        if s.get("next_period_focus"):
            lines.append(f"- 下周期重点: {s['next_period_focus']}")
        lines.append("")
        return "\n".join(lines)
