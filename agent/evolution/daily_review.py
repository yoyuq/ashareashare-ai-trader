"""每日复盘 — 反思层。

T+1 日回看 T 日的诊断决策，对照实际走势，做自我批评。
输出结构化的复盘结论，写入决策记录，同时提取经验条目送入经验记忆库。
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Optional

from .decision_journal import DecisionRecord, DecisionJournal


@dataclass
class ExperienceItem:
    """一条经验教训（可被记忆库存储和检索）。"""
    id: str                          # 唯一标识（日期+场景类型）
    date: str                        # 发生日期
    scenario_type: str               # 场景类型（trend_up / bubble_late / range / ...）
    verdict: str                     # "correct" / "wrong" / "partial"
    lesson_title: str                # 一句话总结
    lesson_detail: str               # 详细说明（当时情况 + 为什么对/错 + 以后怎么做）
    master_used: str                 # 当时选的主导大师
    risk_level_given: int            # 给的风险等级
    actual_outcome: str              # 实际走势描述
    confidence: float = 0.5          # 经验置信度（0-1，复盘时越肯定分越高）
    tags: list[str] = field(default_factory=list)  # 标签，用于检索

    def to_dict(self) -> dict:
        return {
            "id": self.id, "date": self.date,
            "scenario_type": self.scenario_type, "verdict": self.verdict,
            "lesson_title": self.lesson_title, "lesson_detail": self.lesson_detail,
            "master_used": self.master_used, "risk_level_given": self.risk_level_given,
            "actual_outcome": self.actual_outcome,
            "confidence": self.confidence, "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ExperienceItem":
        return cls(**d)


SYSTEM_PROMPT = """你是一位严格的交易复盘教练。

你的工作是：回看昨天的市场诊断决策，对照今天的实际走势，给出客观、犀利、一针见血的复盘。
不要留情面，错了就是错了，对了也要说清楚为什么对。

## ⚠️ 最重要的原则：机会成本 = 实际亏损
大多数交易员只从"亏了钱"中学习，但"该赚的没赚到"也是错误，而且代价同样高昂。

- 市场大涨但你给了5级风险（空仓踏空）= 错误，和市场大跌你给了1级风险（满仓被套）一样严重
- 该进攻的时候保守 = 错失机会 = 机会成本 = 亏损
- 该防守的时候激进 = 实际亏损 = 亏损
- 两种错误的权重完全相等，没有哪个更可原谅

## 复盘原则
1. **结果导向** — 用实际走势检验判断，而不是看逻辑自洽
2. **双向错误都要抓** — 风险给高了（踏空）和给低了（被套）都是错，同等对待
3. **归因到具体因素** — 不要说"判断错了"，要说"错在低估了XX因素/忽略了YY信号"
4. **提炼可操作的教训** — 每条复盘必须给出"下次遇到类似情况应该怎么做"
5. **区分"运气"和"能力"** — 如果结论对了但理由是错的，也要指出来
6. **诚实面对不确定性** — 市场本来就有随机性，该承认的要承认

## 你要回答的问题
1. 昨天判断的市场阶段（market_phase）对吗？实际走势是什么？
2. 风险等级给高了还是给低了？偏差多大？（正=给高了=踏空风险；负=给低了=被套风险）
3. 选的主导大师合适吗？如果换一位会不会更好？
4. 哪些信号当时看对了？哪些看错了/漏掉了？
5. 最重要的一条教训是什么？
6. 这是"保守型错误"（该上没上）还是"激进型错误"（不该上却上了）？

## 输出格式 (严格JSON)
{
  "verdict": "correct / wrong / partial 三选一",
  "error_type": "conservative / aggressive / none （保守型错误/激进型错误/没有错误）",
  "actual_outcome": "实际走势一句话描述（如：市场继续上涨2%，普涨格局）",
  "correct_factors": ["当时判断正确的因素1", "因素2"],
  "missed_factors": ["当时漏掉或误判的因素1", "因素2"],
  "master_assessment": "选的大师是否合适（合适/不合适，原因）",
  "risk_level_deviation": -2 ~ +2 的整数（给高了为正=踏空，给低了为负=被套，0=准确）",
  "lesson_title": "一句话教训（不超过30字）",
  "lesson_detail": "详细教训：当时情况 + 为什么错/对 + 下次怎么做（100字以内）",
  "confidence": 0.3~0.9的浮点数（你对这个教训的信心，经验越清晰越可重复越高）
}
"""


async def review_decision(
    rec: DecisionRecord,
    next_day_stats: dict,
    market_move_pct: float,
) -> Optional[dict]:
    """对一条决策记录做复盘。

    Args:
        rec: 昨天的决策记录
        next_day_stats: 次日的市场统计（上涨占比、涨跌停、成交额等）
        market_move_pct: 次日市场涨跌幅（%），正数=涨，负数=跌

    Returns:
        复盘字典，失败返回 None
    """
    try:
        from openai import AsyncOpenAI

        client = AsyncOpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
            timeout=30.0,
        )

        # 构建当时的诊断描述
        diag_txt = (
            f"日期: {rec.date}\n"
            f"当时判断市场阶段: {rec.market_phase}\n"
            f"主导大师: {rec.dominant_master}（副: {rec.secondary_master}）\n"
            f"风险等级: {rec.risk_level}/5\n"
            f"仓位系数: ×{rec.position_multiplier:.2f}\n"
            f"诊断理由: {rec.diagnosis}\n"
            f"当时关注的风险: {', '.join(rec.key_risks) if rec.key_risks else '无'}\n"
        )

        # 当时的市场环境
        snap = rec.market_snapshot
        env_txt = (
            f"当时市场环境:\n"
            f"  上涨占比: {snap.get('up_ratio', 'N/A')}\n"
            f"  涨停/跌停: {snap.get('limit_up', 'N/A')} / {snap.get('limit_down', 'N/A')}\n"
            f"  中位PE: {snap.get('med_pe', 'N/A')}\n"
            f"  成交额: {snap.get('total_amt_yi', snap.get('total_amt', 'N/A'))}亿\n"
            f"  市场状态: {rec.regime}\n"
            f"  拥挤度: {rec.crowding_signal} ({rec.crowding_score:.2f})\n"
        )

        # 次日实际结果
        outcome_txt = (
            f"次日实际走势:\n"
            f"  市场涨跌: {market_move_pct:+.2f}%\n"
            f"  上涨占比: {next_day_stats.get('up_ratio', 'N/A')}\n"
            f"  涨停/跌停: {next_day_stats.get('limit_up', 'N/A')} / {next_day_stats.get('limit_down', 'N/A')}\n"
            f"  成交额: {next_day_stats.get('total_amt_yi', next_day_stats.get('total_amt', 'N/A'))}亿\n"
        )

        user_msg = (
            f"【昨日诊断】\n{diag_txt}\n"
            f"{env_txt}\n"
            f"{outcome_txt}\n"
            f"请对照实际走势，给出复盘结论。"
        )

        resp = await client.chat.completions.create(
            model=os.getenv("DEEPSEEK_FLASH_MODEL", "deepseek-v4-flash"),
            messages=[{"role": "system", "content": SYSTEM_PROMPT},
                      {"role": "user", "content": user_msg}],
            temperature=0.3,
            max_tokens=600,
            extra_body={"thinking": {"type": "disabled"}},
        )

        content = (resp.choices[0].message.content or "").strip()
        if content.startswith("```"):
            content = content.strip("`")
            if content.lower().startswith("json"):
                content = content[4:].strip()

        # 解析JSON
        review = None
        try:
            review = json.loads(content)
        except json.JSONDecodeError:
            s, e = content.find("{"), content.rfind("}")
            if s >= 0 and e > s:
                try:
                    review = json.loads(content[s:e+1])
                except json.JSONDecodeError:
                    pass

        if review is None:
            return None

        # 提取经验条目
        return review

    except Exception as e:
        print(f"[复盘] 失败: {e}")
        return None


def extract_experience(rec: DecisionRecord, review: dict) -> Optional[ExperienceItem]:
    """从复盘结果中提取经验条目。"""
    try:
        verdict = review.get("verdict", "partial")
        lesson_title = review.get("lesson_title", "")
        lesson_detail = review.get("lesson_detail", "")
        if not lesson_title or not lesson_detail:
            return None

        # v5.1 元认知偏差校准: 读取复盘 error_type, 落 conservative/aggressive tag.
        # 此前 error_type 被 LLM 产出但从不消费, 导致 experience_memory.get_metacognition_summary
        # 搜不到 `保守/激进` tag → 永远报"攻守平衡", 系统不知道自己偏保守还是偏激进.
        # 现在把复盘自己的判断回流成可量化的自我认知（自我进化最核心的反馈线）.
        bias_tags = []
        _et = str(review.get("error_type", "")).lower()
        if "conservative" in _et or "保守" in _et:
            bias_tags.append("conservative")
        elif "aggressive" in _et or "激进" in _et:
            bias_tags.append("aggressive")

        return ExperienceItem(
            id=f"{rec.date}_{rec.market_phase}_{verdict}",
            date=rec.date,
            scenario_type=rec.market_phase,
            verdict=verdict,
            lesson_title=lesson_title,
            lesson_detail=lesson_detail,
            master_used=rec.dominant_master,
            risk_level_given=rec.risk_level,
            actual_outcome=review.get("actual_outcome", ""),
            confidence=float(review.get("confidence", 0.5)),
            tags=[rec.dominant_master, rec.market_phase, verdict] + bias_tags +
                 (["missed_" + f.replace(" ", "_") for f in review.get("missed_factors", [])[:2]]
                  if isinstance(review.get("missed_factors"), list) else []),
        )
    except Exception:
        return None
