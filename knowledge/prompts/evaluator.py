"""
PromptEvaluator — 提示词效果评估器 (v3.0-competition)

对同一输入运行多个prompt变体,从多个维度评估输出质量:
  - 格式合规: 是否按照要求的格式输出
  - 数值准确性: 引用的数字是否可验证
  - 逻辑一致性: 分析过程是否有逻辑漏洞
  - 风险提示完整性: 是否包含必要的风险提示

用于:
  - 提示词A/B测试
  - 提示词版本迭代效果验证
  - 比赛展示 (提示词工程自评)
"""

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class EvalResult:
    """单次评估结果"""
    prompt_name: str
    prompt_version: str
    test_input: str
    output_text: str
    scores: Dict[str, float] = field(default_factory=dict)
    total_score: float = 0.0
    issues: List[str] = field(default_factory=list)
    strengths: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "prompt_name": self.prompt_name,
            "prompt_version": self.prompt_version,
            "test_input_preview": self.test_input[:100],
            "output_preview": self.output_text[:200],
            "scores": self.scores,
            "total_score": self.total_score,
            "issues": self.issues,
            "strengths": self.strengths,
        }


@dataclass
class ComparisonReport:
    """A/B对比报告"""
    baseline: EvalResult
    variants: List[EvalResult] = field(default_factory=list)
    winner: Optional[str] = None
    recommendations: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "baseline": self.baseline.to_dict(),
            "variants": [v.to_dict() for v in self.variants],
            "winner": self.winner,
            "recommendations": self.recommendations,
        }


class PromptEvaluator:
    """
    提示词效果评估器

    评估维度 (每项0-10分,满分40分):
      1. 格式合规 (structure): 输出是否遵循指定格式
      2. 数值准确性 (accuracy): 数字是否可验证、是否来自计算
      3. 逻辑一致性 (logic): 分析推理是否有逻辑漏洞
      4. 风险提示 (risk_warning): 是否包含必要的风险提示
    """

    def __init__(self, name: str = "PromptEvaluator"):
        self.name = name
        self.dimensions = {
            "structure": {
                "label": "格式合规",
                "weight": 1.0,
                "description": "输出是否遵循指定格式,是否结构化",
            },
            "accuracy": {
                "label": "数值准确性",
                "weight": 1.0,
                "description": "引用数据是否可验证,是否注明数据来源",
            },
            "logic": {
                "label": "逻辑一致性",
                "weight": 1.0,
                "description": "分析推理是否有逻辑漏洞,结论是否有依据",
            },
            "risk_warning": {
                "label": "风险提示完整性",
                "weight": 1.0,
                "description": "是否包含必要的风险提示和免责声明",
            },
        }

    def evaluate(self, prompt_name: str, prompt_version: str,
                 test_input: str, output_text: str) -> EvalResult:
        """
        评估单个输出

        Args:
            prompt_name: 提示词名称
            prompt_version: 提示词版本
            test_input: 测试输入
            output_text: LLM输出文本

        Returns:
            EvalResult 评估结果
        """
        result = EvalResult(
            prompt_name=prompt_name,
            prompt_version=prompt_version,
            test_input=test_input,
            output_text=output_text,
        )

        # 1. 格式合规 (0-10)
        result.scores["structure"] = self._score_structure(output_text)
        # 2. 数值准确性 (0-10)
        result.scores["accuracy"] = self._score_accuracy(output_text)
        # 3. 逻辑一致性 (0-10)
        result.scores["logic"] = self._score_logic(output_text)
        # 4. 风险提示 (0-10)
        result.scores["risk_warning"] = self._score_risk_warning(output_text)

        result.total_score = sum(result.scores.values())
        result.total_score = min(40.0, result.total_score)

        # 收集问题和优点
        result.issues = self._collect_issues(result.scores, output_text)
        result.strengths = self._collect_strengths(result.scores, output_text)

        return result

    def compare(self, baseline: EvalResult,
                variants: List[EvalResult]) -> ComparisonReport:
        """
        对比多个提示词变体的效果

        Args:
            baseline: 基准提示词结果
            variants: 变体提示词结果列表

        Returns:
            ComparisonReport 对比报告
        """
        report = ComparisonReport(baseline=baseline, variants=variants)
        all_results = [baseline] + variants
        all_results.sort(key=lambda r: r.total_score, reverse=True)
        report.winner = all_results[0].prompt_name

        # 生成建议
        if len(variants) > 0:
            best_variant = variants[0]
            for v in variants:
                if v.total_score > best_variant.total_score:
                    best_variant = v

            if best_variant.total_score > baseline.total_score:
                report.recommendations.append(
                    f"推荐使用 {best_variant.prompt_name} v{best_variant.prompt_version} "
                    f"(得分 {best_variant.total_score:.1f} vs 基准 {baseline.total_score:.1f})"
                )
            else:
                report.recommendations.append(
                    f"当前基准版本为最佳 (得分 {baseline.total_score:.1f})"
                )

        # 维度建议
        for dim_key, dim_info in self.dimensions.items():
            baseline_dim = baseline.scores.get(dim_key, 0)
            best_variant_dim = max(
                (v.scores.get(dim_key, 0) for v in variants), default=baseline_dim
            )
            if best_variant_dim > baseline_dim + 2:
                report.recommendations.append(
                    f"维度 '{dim_info['label']}' 有显著提升空间 "
                    f"(当前 {baseline_dim:.0f}, 最佳 {best_variant_dim:.0f})"
                )

        return report

    # ═══════════════════════════════════════════════════════════════
    # 评分方法
    # ═══════════════════════════════════════════════════════════════

    def _score_structure(self, text: str) -> float:
        """评估格式合规性 (0-10)"""
        if not text:
            return 0
        score = 5.0
        # 有标题/段落结构
        if "###" in text or "##" in text:
            score += 2.0
        # 有编号列表
        if re.search(r'\d+\.\s', text):
            score += 1.0
        # 长度合理 (>100字)
        if len(text) > 100:
            score += 1.0
        # 包含明确的结论
        conclusion_keywords = ["综合", "建议", "结论", "推荐", "总体"]
        if any(kw in text for kw in conclusion_keywords):
            score += 1.0
        return min(10.0, score)

    def _score_accuracy(self, text: str) -> float:
        """评估数值准确性 (0-10)"""
        if not text:
            return 0
        score = 3.0
        # 引用了具体数字
        numbers = re.findall(r'\d+\.?\d*', text)
        if len(numbers) >= 3:
            score += 2.0
        if len(numbers) >= 8:
            score += 1.0
        # 包含数据来源
        source_keywords = ["数据", "指标", "计算", "代码", "来源", "统计"]
        if any(kw in text for kw in source_keywords):
            score += 2.0
        # 不会出现模糊表述的加分
        vague_patterns = ["大概", "可能", "估计", "猜测"]
        vague_count = sum(1 for p in vague_patterns if p in text)
        if vague_count <= 1:
            score += 1.0
        if vague_count == 0:
            score += 1.0
        return min(10.0, score)

    def _score_logic(self, text: str) -> float:
        """评估逻辑一致性 (0-10)"""
        if not text:
            return 0
        score = 3.0
        # 有因果推理
        causality = ["因为", "所以", "因此", "由于", "导致", "从而"]
        if any(kw in text for kw in causality):
            score += 2.0
        # 有对比分析
        compare = ["但是", "然而", "相比", "对比", "而", "反之"]
        if any(kw in text for kw in compare):
            score += 2.0
        # 有多维度分析
        dimension_marks = ["技术面", "基本面", "资金面", "市场", "风险"]
        dim_count = sum(1 for d in dimension_marks if d in text)
        score += min(3.0, dim_count * 1.0)
        return min(10.0, score)

    def _score_risk_warning(self, text: str) -> float:
        """评估风险提示完整性 (0-10)"""
        if not text:
            return 0
        score = 2.0
        # 包含风险关键词
        risk_kws = ["风险", "止损", "回撤", "止损价", "注意"]
        matched = sum(1 for kw in risk_kws if kw in text)
        score += min(5.0, matched * 1.0)
        # 包含免责声明
        disclaimers = ["不构成", "历史数据不代表", "仅供参考", "投资建议", "风险提示"]
        if any(d in text for d in disclaimers):
            score += 3.0
        return min(10.0, score)

    # ═══════════════════════════════════════════════════════════════
    # 问题与优点收集
    # ═══════════════════════════════════════════════════════════════

    def _collect_issues(self, scores: dict, text: str) -> List[str]:
        """收集评估中发现的问题"""
        issues = []
        if scores.get("structure", 0) < 5:
            issues.append("输出格式不够结构化,缺少标题和段落划分")
        if scores.get("accuracy", 0) < 5:
            issues.append("数值引用不足,缺少数据支撑")
        if scores.get("logic", 0) < 5:
            issues.append("逻辑推理不够清晰,缺少因果分析")
        if scores.get("risk_warning", 0) < 5:
            issues.append("缺少必要的风险提示和免责声明")
        return issues

    def _collect_strengths(self, scores: dict, text: str) -> List[str]:
        """收集输出中的优点"""
        strengths = []
        if scores.get("structure", 0) >= 8:
            strengths.append("输出格式规范,结构化程度高")
        if scores.get("accuracy", 0) >= 8:
            strengths.append("数据引用充分,有明确的数值支撑")
        if scores.get("logic", 0) >= 8:
            strengths.append("逻辑推理严密,多维度交叉验证")
        if scores.get("risk_warning", 0) >= 8:
            strengths.append("风险提示完整,包含免责声明")
        return strengths


# ═══════════════════════════════════════════════════════════════
# 便捷函数
# ═══════════════════════════════════════════════════════════════

def quick_evaluate(prompt_name: str, version: str,
                   test_input: str, output: str) -> dict:
    """快速评估 — 返回dict"""
    evaluator = PromptEvaluator()
    result = evaluator.evaluate(prompt_name, version, test_input, output)
    return result.to_dict()
