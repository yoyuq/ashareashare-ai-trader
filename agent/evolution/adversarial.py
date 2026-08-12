"""v5.6 独立对抗票 — 打破"假多元化".

现状: 5 位大师实为同一 deepseek 模型, 且对抗票(adversarial_risk_level)与主导
判断产自**同一次 LLM completion** (role-play "再从相反大师视角给一个分数").
这本质是模型自己跟自己辩论 — 冗余度实测 0.28~0.36, 大量时候主导与对抗趋同.

本模块提供**独立的第二次 LLM 调用**: 只问"与主导大师相反的对抗大师"对同一份
市场快照的独立风险评分. 与主导调用**不同样本、不同提示词**, 杜绝耦合.

A/B 用途 (--adv-mode):
  - off        : 不调对抗票 → 闸门永不触发 (闸门价值 A/B 的基线)
  - same       : 现状单 completion role-play (基线)
  - independent: 独立二次调用 (本模块, 新)
"""

from __future__ import annotations

import json
import os
from typing import Optional

from openai import AsyncOpenAI

# 对抗大师提示词 — 刻意与主导诊断官"5位大师"基座不同:
# 只塑造"与主导相反"的单一审慎视角, 独立读同一份数据给 1-5 风险分.
# 不注入进化记忆, 保持对抗票纯净(不双重复用主导的立场).
_ADVERSARIAL_SYSTEM_PROMPT = """你是一位独立的A股【对抗审查官】。

你的工作是**故意站在主导判断的对面**审视同一份市场数据 — 找主导可能忽略的风险。
你不必"正确"，你负责"唱反调"：如果主导说要进攻，你就找一切该防守的理由；如果主导说要防守，你也应独立评估是否真的该防守。

## 你的审慎清单 (逐条过, 不要凭感觉)
- 广度背离: 指数/中位是否在涨, 但上涨家数占比在收窄？(顶背离信号)
- 量能: 是放量滞涨(筹码派发)还是放量上涨(增量入场)？
- 极端情绪: 涨停家数是否异常多(情绪过热), 跌停家数是否异常多(恐慌)?
- 估值: 中位PE/PB是否处于历史高位(无安全边际)?
- 拥挤度: 市场是否已过度拥挤(极端活跃占比高)?
- 趋势: 中期(60日)上涨占比是否在衰竭(趋势末端)?

## 输出 (严格JSON, 不要多余文字)
{
  "adversarial_risk_level": 1~5的整数,
  "adversarial_reason": "一句话, 你看到了什么主导可能忽略的风险"
}

## 风险等级含义 (与主导诊断官同一把尺)
- 1=极度看多(该满仓进攻)  2=偏多  3=中性  4=谨慎  5=极度防御(该清仓避险)

## 记住
- 你独立读数据, 不是主导的复读机. 即使主导看多, 你也只按你看到的风险给分.
- 但也不要为了唱反调而乱给极值 — 数据确实健康时, 给4/5是过度防御, 会误伤收益.
- 诚实: 数据告诉你看不出风险, 就如实给中性3.
"""


async def adversarial_risk(
    client: Optional[AsyncOpenAI],
    snapshot_txt: str,
    dominant_master: str,
    regime: str,
    macro_txt: Optional[str] = None,
    model: Optional[str] = None,
) -> Optional[int]:
    """独立的第二次 LLM 调用 — 对抗大师对同一份市场快照的独立风险评分 (1-5).

    与主导调用不同样本/不同提示词, 消除 role-play 耦合. 失败返回 None (不拦截).
    """
    if client is None:
        return None
    try:
        user_msg = (
            f"主导大师判定当前为「{regime}」阶段, 主导大师是「{dominant_master}」。\n"
            f"以下是同一份市场数据快照, 请独立审慎评估:\n\n{snapshot_txt}"
        )
        if macro_txt:
            user_msg += f"\n\n【宏观背景】\n{macro_txt}"
        user_msg += "\n\n给出你的独立 adversarial_risk_level (1-5整数)。"

        resp = await client.chat.completions.create(
            model=model or os.getenv("DEEPSEEK_FLASH_MODEL", "deepseek-v4-flash"),
            messages=[
                {"role": "system", "content": _ADVERSARIAL_SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.4,
            max_tokens=200,
            extra_body={"thinking": {"type": "disabled"}},
        )
        content = (resp.choices[0].message.content or "").strip()
        if content.startswith("```"):
            content = content.strip("`")
            if content.lower().startswith("json"):
                content = content[4:].strip()
        data = None
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            s, e = content.find("{"), content.rfind("}")
            if s >= 0 and e > s:
                try:
                    data = json.loads(content[s:e + 1])
                except json.JSONDecodeError:
                    pass
        if data is None:
            return None
        raw = data.get("adversarial_risk_level")
        if raw is None:
            return None
        return max(1, min(5, int(raw)))
    except Exception:
        return None


def make_async_client() -> AsyncOpenAI:
    """构造与主导诊断官一致的 AsyncOpenAI 客户端 (独立调用复用同一连接)."""
    return AsyncOpenAI(
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
        timeout=45.0,
    )