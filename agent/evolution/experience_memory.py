"""经验记忆库 — 知识层。

累积从复盘中学到的经验教训，支持：
- 存储/检索结构化经验条目
- 按场景类型检索（给今天的情况找最相关的历史经验）
- 记忆衰减（越老的经验权重越低）
- 去重（相似的经验合并保留最强的）
- 动态注入提示词（在诊断前给 LLM 看相关经验）

这是自我进化系统的"长期记忆"。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from .daily_review import ExperienceItem


class ExperienceMemory:
    """经验记忆库。

    存储格式: JSON 文件，内含经验条目列表。
    设计目标：轻量、可解释、不依赖外部数据库。
    （经验条目数量级是"几百条"，不需要向量数据库）
    """

    def __init__(self, path: str | Path, max_items: int = 50):  # v4.2: 100→50，配合加速衰减
        self.path = Path(path)
        self.max_items = max_items
        self.items: list[ExperienceItem] = []
        self._load()

    def _load(self):
        if not self.path.exists():
            return
        with open(self.path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.items = [ExperienceItem.from_dict(d) for d in data.get("items", [])]

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(
                {"items": [it.to_dict() for it in self.items]},
                f, ensure_ascii=False, indent=2
            )

    def add(self, item: ExperienceItem) -> bool:
        """添加一条经验。如果太相似就合并（更新置信度），不重复存。

        Returns:
            True if added as new, False if merged
        """
        # 去重：同一场景 + 相似标题 → 合并
        for existing in self.items:
            if (existing.scenario_type == item.scenario_type
                    and self._similar(existing.lesson_title, item.lesson_title)):
                # 合并：取较新的描述，置信度取加权平均
                existing.lesson_detail = item.lesson_detail
                existing.confidence = max(existing.confidence, item.confidence)
                existing.date = max(existing.date, item.date)
                # 标签去重合并
                for t in item.tags:
                    if t not in existing.tags:
                        existing.tags.append(t)
                self.save()
                return False

        self.items.append(item)
        # 超限就删掉价值最低的 (v5.1 与 retrieve 共用同一打分 _score_item, 消除"注入top-k最先被淘汰"分叉)
        if len(self.items) > self.max_items:
            self.items.sort(key=lambda x: _score_item(x), reverse=True)
            self.items = self.items[:self.max_items]
        self.save()
        return True

    @staticmethod
    def _similar(a: str, b: str) -> float:
        """简单相似度：计算字符级Jaccard。>0.5 认为相似。"""
        sa, sb = set(a), set(b)
        if not sa or not sb:
            return 0.0
        return len(sa & sb) / len(sa | sb)

    def retrieve(
        self,
        scenario_type: Optional[str] = None,
        master: Optional[str] = None,
        verdict: Optional[str] = None,
        top_k: int = 5,
        current_date: Optional[str] = None,
        regime: Optional[str] = None,
    ) -> list[ExperienceItem]:
        """检索相关经验。

        按匹配度 + 衰减后的权重排序，返回 top_k。
        regime: v5.2 当前市场状态 (strong_bull/weak_bear/range_bound...), 用于跨界惩罚 —
                熊市/震荡时不把牛市学到的激进经验排在前面, 防止"regime 过拟合" (留出法 A/B 验证暴露).
        """
        scored = list(self.items)
        # v5.1 与淘汰共用 _score_item, 场景/大师/裁决匹配加成 + 置信度 + 高置信/cf_verified + 时间衰减
        scored.sort(
            key=lambda item: _score_item(item, current_date, scenario_type, master, verdict, regime),
            reverse=True,
        )
        return scored[:top_k]

    def format_for_prompt(
        self,
        scenario_type: Optional[str] = None,
        master: Optional[str] = None,
        current_date: Optional[str] = None,
        top_k: int = 5,
        regime: Optional[str] = None,
    ) -> str:
        """把最相关的经验格式化成提示词片段。"""
        items = self.retrieve(
            scenario_type=scenario_type, master=master,
            current_date=current_date, top_k=top_k, regime=regime,
        )
        if not items:
            return ""

        lines = ["## 你的历史经验（从过去的复盘中学到的）"]
        for i, it in enumerate(items, 1):
            verdict_cn = {"correct": "✅ 正确经验", "wrong": "❌ 错误教训", "partial": "⚠️ 部分正确"}.get(it.verdict, it.verdict)
            lines.append(
                f"{i}. [{it.date}] {verdict_cn} | {it.master_used} | {it.lesson_title}\n"
                f"   → {it.lesson_detail}"
            )
        lines.append("")  # 空行分隔
        return "\n".join(lines)

    def stats(self) -> dict:
        """统计信息。"""
        return {
            "total": len(self.items),
            "by_verdict": {
                v: sum(1 for i in self.items if i.verdict == v)
                for v in ["correct", "wrong", "partial"]
            },
            "by_master": {},
            "by_scenario": {},
        }

    def get_metacognition_summary(self, current_date: str, lookback_days: int = 15) -> str:
        """v4.2: 元认知摘要 — 告诉 LLM 它自己近期的表现如何.

        人类交易员会根据近期状态调整策略（连续错几次就会更谨慎），
        LLM 也应该有这种自我认知。这是元认知校准回路的核心。
        """
        if not self.items:
            return ""

        # 筛选近期经验
        try:
            from datetime import datetime
            cur = datetime.strptime(current_date, "%Y-%m-%d")
            recent = [
                it for it in self.items
                if (cur - datetime.strptime(it.date, "%Y-%m-%d")).days <= lookback_days
            ]
        except (ValueError, TypeError):
            recent = self.items[-10:]  # fallback: 最近10条

        if not recent:
            return ""

        total = len(recent)
        correct = sum(1 for it in recent if it.verdict == "correct")
        wrong = sum(1 for it in recent if it.verdict == "wrong")
        partial = sum(1 for it in recent if it.verdict == "partial")

        # 统计错误类型（从标签里找）
        conservative_err = sum(1 for it in recent if any("conservative" in t or "踏空" in t or "保守" in t for t in it.tags))
        aggressive_err = sum(1 for it in recent if any("aggressive" in t or "被套" in t or "激进" in t for t in it.tags))

        # 平均置信度
        avg_conf = sum(it.confidence for it in recent) / total

        # 判断偏差方向
        if conservative_err > aggressive_err + 1:
            bias = "你近期偏保守，踏空错误多于被套错误，可以适当提高进攻性。"
        elif aggressive_err > conservative_err + 1:
            bias = "你近期偏激进，被套错误多于踏空错误，应该更加谨慎。"
        else:
            bias = "你近期攻守平衡，继续保持。"

        accuracy = correct / total if total > 0 else 0

        lines = [
            f"## 你的近期自我评估（元认知校准）",
            f"- 近{lookback_days}天共 {total} 次有效复盘",
            f"- 正确: {correct}次 ({accuracy:.0%})  |  错误: {wrong}次  |  部分正确: {partial}次",
            f"- 保守型错误(踏空): {conservative_err}次  |  激进型错误(被套): {aggressive_err}次",
            f"- 平均自信度: {avg_conf:.0%}",
            f"- 校准建议: {bias}",
            "",
        ]
        return "\n".join(lines)


def _score_item(
    item: ExperienceItem,
    current_date: Optional[str] = None,
    scenario: Optional[str] = None,
    master: Optional[str] = None,
    verdict: Optional[str] = None,
    regime: Optional[str] = None,
) -> float:
    """统一打分 — retrieve(注入展示) 与 add(淘汰) 共用, 消除"注入top-k最先被淘汰"分叉.

    组成: 置信度 + 高置信/cf_verified 加成 + 查询匹配(scenario/master/verdict)
          + regime 方向对齐 + 时间衰减.
    regime: v5.2 跨界惩罚 — 熊市/震荡时不把牛市(趋势上行)学到的激进经验排在前面.
            留出法 A/B 证实: 进化在牛市学到"激进"经验, regime 切到下跌时仍被注入,
            导致 2 月调整里反而更激进. 这里按"经验所在阶段方向"与"当前regime方向"是否一致加减分.
    时间衰减: 半衰期25天, 最低保留30%权重; 参考日期默认取当前 (淘汰时), 检索时传 current_date.
    """
    score = item.confidence
    # 反事实验证通过的高置信经验优先 (v5 落地)
    if "high_confidence" in item.tags:
        score += 0.4
    elif "cf_verified" in item.tags:
        score += 0.2
    # 查询匹配加成 (检索时用; 淘汰时这些为 None 不加成)
    if scenario and item.scenario_type == scenario:
        score += 2.0
    if master and item.master_used == master:
        score += 1.0
    if verdict and item.verdict == verdict:
        score += 0.5
    # v5.2 regime 方向对齐 — 跨界惩罚 (检索时用; 淘汰时 regime=None 不加成)
    if regime:
        _adj = _regime_alignment(regime, item.scenario_type)
        if _adj < 0:
            score += _adj  # 跨界(方向相反) → 惩罚
        elif _adj > 0:
            score += _adj  # 同向 → 轻微加成
    # 时间衰减 (越新权重越高, 半衰期25天, 最低保留30%)
    _ref = current_date or datetime.now().isoformat()[:10]
    try:
        days = (datetime.strptime(_ref, "%Y-%m-%d")
                - datetime.strptime(item.date, "%Y-%m-%d")).days
        decay = 0.5 ** (days / 25.0)
        score *= (0.3 + 0.7 * decay)
    except (ValueError, TypeError):
        pass
    return score


# v5.2 方向映射 — 用于 regime 对齐的跨界惩罚
_SCENARIO_DIR = {
    "trend_up": "bullish", "turning": "bullish",
    "trend_down": "bearish", "bubble_late": "bearish", "panic_bottom": "bearish",
    "range": "neutral",
}
_REGIME_DIR = {
    "strong_bull": "bullish", "weak_bull": "bullish",
    "strong_bear": "bearish", "weak_bear": "bearish",
    "range_bound": "neutral",
}


def _regime_alignment(regime: str, scenario_type: str) -> float:
    """regime 与经验所在阶段的方向对齐调整.

    熊市/震荡时, 牛市学到的激进经验(trend_up)应降权 → 返回负分惩罚;
    熊市时防守经验(trend_down/bubble_late/panic)应优先 → 返回正分.
    牛市时相反. 中性不惩罚.
    """
    _r = _REGIME_DIR.get(regime)
    _s = _SCENARIO_DIR.get(scenario_type)
    if not _r or not _s or _s == "neutral":
        return 0.0
    if _r == _s:
        return 0.8   # 同向: 轻微加成
    return -3.0      # 反向(跨界): 强惩罚, 防止注入过时激进/过时防守经验


# 兼容层: 保留 ExperienceItem.weighted_score (v5.1 起委托给统一打分, 淘汰路径已改用 _score_item)
def _weighted_score(self) -> float:
    return _score_item(self)


ExperienceItem.weighted_score = _weighted_score  # monkey-patch
