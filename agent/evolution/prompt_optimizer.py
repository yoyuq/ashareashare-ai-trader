"""OPRO 式提示词自动优化器。

用历史决策日志作为验证集，让 LLM 自动迭代优化诊断官的系统提示词。

思路来自 Google DeepMind 的 OPRO (Large Language Models as Optimizers, 2023)：
- 把提示词当作可优化的参数
- 用 LLM 当"优化器"，根据历史表现生成新的提示词变体
- 在验证集上评估每个变体，选最好的进入下一代

对于交易诊断场景，验证集天然存在：
- 历史上每天都有诊断记录
- 第二天走势就是"标准答案"（可以反推出"最优风险等级"）
- 评估指标：风险等级偏差、大师选择方向正确性

这是一个"离线优化"工具——用历史数据一次性跑完 N 轮，
产出优化后的提示词，再人工审核后部署。
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# v5.6 P1-6: 统一 LLM 调用走 ModelRouter (记账/扣预算/峰谷计价),
# 消除自进化模块直连 AsyncOpenAI 绕过成本核算的问题
_shared_router = None


def _get_router():
    """懒加载共享 ModelRouter (进化优化是长流程, 复用单例避免重复读配置/建连接)"""
    global _shared_router
    if _shared_router is None:
        from models.router import ModelRouter
        _shared_router = ModelRouter()
    return _shared_router


@dataclass
class EvaluationResult:
    """一轮提示词的评估结果。"""
    prompt_name: str
    avg_risk_deviation: float       # 风险等级平均偏差（绝对值，越小越好）
    direction_accuracy: float       # 方向准确率（高估/低估的方向对不对，越大越好）
    master_accuracy: float          # 大师选择准确率（0-1）
    format_compliance: float        # 输出格式合规率（0-1）
    overall_score: float            # 综合得分（越大越好）
    sample_count: int

    def to_dict(self) -> dict:
        return {
            "prompt_name": self.prompt_name,
            "avg_risk_deviation": self.avg_risk_deviation,
            "direction_accuracy": self.direction_accuracy,
            "master_accuracy": self.master_accuracy,
            "format_compliance": self.format_compliance,
            "overall_score": self.overall_score,
            "sample_count": self.sample_count,
        }


@dataclass
class OptimizationHistory:
    """优化历史记录。"""
    rounds: list[dict] = field(default_factory=list)
    best_prompt: str = ""
    best_score: float = -999.0
    best_round: int = 0
    # v5.2 P1 样本外诚实评估 (留出集, 优化期间未用于选优)
    out_of_sample_score: Optional[float] = None
    out_of_sample_avg_dev: Optional[float] = None
    out_of_sample_dir_acc: Optional[float] = None
    out_of_sample_detail: Optional[dict] = None

    def save(self, path: str | Path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({
                "rounds": self.rounds,
                "best_prompt": self.best_prompt,
                "best_score": self.best_score,
                "best_round": self.best_round,
                "out_of_sample_score": self.out_of_sample_score,
                "out_of_sample_avg_dev": self.out_of_sample_avg_dev,
                "out_of_sample_dir_acc": self.out_of_sample_dir_acc,
                "out_of_sample_detail": self.out_of_sample_detail,
            }, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "OptimizationHistory":
        if not Path(path).exists():
            return cls()
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        hist = cls()
        hist.rounds = data.get("rounds", [])
        hist.best_prompt = data.get("best_prompt", "")
        hist.best_score = data.get("best_score", -999.0)
        hist.best_round = data.get("best_round", 0)
        hist.out_of_sample_score = data.get("out_of_sample_score")
        hist.out_of_sample_avg_dev = data.get("out_of_sample_avg_dev")
        hist.out_of_sample_dir_acc = data.get("out_of_sample_dir_acc")
        hist.out_of_sample_detail = data.get("out_of_sample_detail")
        return hist


OPTIMIZER_SYSTEM_PROMPT = """你是一位提示词优化专家。

你的任务是：根据历史上诊断官的表现，迭代优化它的系统提示词。
你会看到过去几轮提示词的表现（好的、坏的），你的工作是生成新的提示词变体。

## 优化目标
- 主要目标: 降低风险等级判断偏差（让风险等级更准确）
- 次要目标: 提高大师选择的准确性、提升整体决策质量
- 约束: 输出格式必须是严格的 JSON（合规率要高）

## 工作方法
1. 仔细分析历史上表现最好的提示词有什么共同点
2. 分析表现最差的提示词有什么问题
3. 生成 3 个新的提示词变体，每个变体尝试不同的改进方向
4. 每个变体只改 1-2 个关键点，不要全部重写（小步迭代更稳定）

## 改进方向参考
- 调整推理框架（让诊断官按什么步骤思考）
- 调整风险等级的定义和边界
- 调整大师的描述和选择标准
- 增加/修改检查清单
- 调整输出格式的约束
- 尝试不同的角色设定

## 输出格式
直接输出 3 个提示词变体，每个用 === 分隔线分开:

=== PROMPT_A ===
{第一个变体的完整系统提示词}

=== PROMPT_B ===
{第二个变体的完整系统提示词}

=== PROMPT_C ===
{第三个变体的完整系统提示词}

每个变体都必须是完整的、可以直接使用的系统提示词。
不要只写修改点，要写完整的提示词全文。
"""


class PromptOptimizer:
    """OPRO 风格的提示词优化器。"""

    def __init__(
        self,
        initial_prompt: str,
        eval_samples: list,       # 评估用的样本列表
        output_dir: str | Path,
        max_rounds: int = 10,
        variants_per_round: int = 3,
        holdout_ratio: float = 0.2,   # v5.2 留出比例: 样本外诚实评估, 防优化过拟合
    ):
        self.initial_prompt = initial_prompt
        self.eval_samples = eval_samples
        self.output_dir = Path(output_dir)
        self.max_rounds = max_rounds
        self.variants_per_round = variants_per_round
        self.history = OptimizationHistory.load(self.output_dir / "optimization_history.json")

        # v5.2 P1 OPRO train/eval 分离: 把样本拆成"选择集"(选优) 与"留出集"(诚实评估).
        # 若优化器每次都只用同一批样本选优, 会过拟合到这些样本; 留出集在优化结束后才评估,
        # 得到的是接近真实部署的样本外得分, 防止"优化出的 prompt 在实盘未必最优".
        rng = random.Random(42)  # 固定种子, 每次拆分一致
        _idx = list(range(len(eval_samples)))
        rng.shuffle(_idx)
        _n_holdout = max(1, int(len(eval_samples) * holdout_ratio))
        self.holdout_samples = [eval_samples[i] for i in _idx[: _n_holdout]]
        self.selection_samples = [eval_samples[i] for i in _idx[_n_holdout:]]

    async def optimize(self) -> OptimizationHistory:
        """运行完整的优化流程。"""
        # 第0轮: 评估初始提示词
        if not self.history.rounds:
            print(f"[第0轮] 评估初始提示词...")
            base_result = await self._evaluate_prompt(
                "initial", self.initial_prompt
            )
            self.history.rounds.append({
                "round": 0,
                "variants": [{"name": "initial", "score": base_result.overall_score,
                              "result": base_result.to_dict()}],
                "best_this_round": "initial",
                "best_score": base_result.overall_score,
            })
            self.history.best_prompt = self.initial_prompt
            self.history.best_score = base_result.overall_score
            self.history.best_round = 0
            self._save()
            print(f"  初始得分: {base_result.overall_score:.3f}")

        # 迭代优化
        start_round = len(self.history.rounds)
        for r in range(start_round, self.max_rounds):
            print(f"\n[第{r}轮] 生成变体并评估...")

            # 1. 生成变体
            variants = await self._generate_variants(r)
            if not variants:
                print("  生成变体失败，停止优化")
                break

            # 2. 评估每个变体
            variant_results = []
            for name, prompt in variants.items():
                result = await self._evaluate_prompt(name, prompt)
                variant_results.append((name, prompt, result))
                print(f"  {name}: 得分 {result.overall_score:.3f} "
                      f"(偏差 {result.avg_risk_deviation:.2f}, "
                      f"方向准 {result.direction_accuracy:.1%})")

            # 3. 选最好的
            variant_results.sort(key=lambda x: x[2].overall_score, reverse=True)
            best_name, best_prompt, best_result = variant_results[0]

            # 和历史最佳比较
            if best_result.overall_score > self.history.best_score:
                self.history.best_prompt = best_prompt
                self.history.best_score = best_result.overall_score
                self.history.best_round = r
                print(f"  → 新的历史最佳! 得分 {best_result.overall_score:.3f}")
            else:
                print(f"  → 未超过历史最佳 ({self.history.best_score:.3f})")

            # 记录
            self.history.rounds.append({
                "round": r,
                "variants": [
                    {"name": n, "score": res.overall_score, "result": res.to_dict(),
                     "prompt_preview": p[:200]}
                    for n, p, res in variant_results
                ],
                "best_this_round": best_name,
                "best_score": best_result.overall_score,
            })
            self._save()

        print(f"\n优化完成! 最佳得分: {self.history.best_score:.3f} (第{self.history.best_round}轮)")

        # v5.2 P1 样本外诚实评估: 用留出集(优化期间从未用于选优)评估最终最佳提示词
        if self.holdout_samples:
            print(f"\n[样本外评估] 在 {len(self.holdout_samples)} 条留出样本上评估最终提示词...")
            _o = await self._evaluate_prompt("best_holdout", self.history.best_prompt, use_holdout=True)
            self.history.out_of_sample_score = _o.overall_score
            self.history.out_of_sample_detail = _o.to_dict()
            self.history.out_of_sample_avg_dev = _o.avg_risk_deviation
            self.history.out_of_sample_dir_acc = _o.direction_accuracy
            self._save()
            print(f"  样本外得分: {_o.overall_score:.3f} "
                  f"(偏差 {_o.avg_risk_deviation:.2f}, 方向准 {_o.direction_accuracy:.1%})")
            print(f"  对比选择集最佳: {self.history.best_score:.3f} "
                  f"→ 差距 {self.history.best_score - _o.overall_score:+.3f} "
                  f"({'过拟合' if _o.overall_score < self.history.best_score - 0.05 else '泛化尚可'})")

        return self.history

    async def _generate_variants(self, round_num: int) -> dict[str, str]:
        """让 LLM 根据历史表现生成新的提示词变体。"""
        try:
            # 构建历史表现摘要
            history_text = self._build_history_summary()

            user_msg = (
                f"【优化轮次】第 {round_num} 轮\n\n"
                f"【历史表现】\n{history_text}\n\n"
                f"【当前最佳提示词】\n{self.history.best_prompt[:3000]}...\n\n"
                f"请生成 3 个新的提示词变体。"
            )

            # v5.6 P1-6: 统一走 ModelRouter (记账/扣预算/峰谷), 不再直连 AsyncOpenAI
            result = await _get_router().route(
                messages=[{"role": "system", "content": OPTIMIZER_SYSTEM_PROMPT},
                          {"role": "user", "content": user_msg}],
                task_type="strategy_optimize",
                temperature=0.7,  # 温度稍高，增加多样性
                max_tokens=6000,
            )

            content = (result.response or "").strip()
            return self._parse_variants(content, round_num)

        except Exception as e:
            print(f"  生成变体失败: {e}")
            return {}

    def _build_history_summary(self) -> str:
        """构建历史表现摘要（给优化器LLM看）。"""
        lines = []
        # 只显示最近5轮
        recent = self.history.rounds[-5:]
        for rnd in recent:
            lines.append(f"第{rnd['round']}轮:")
            for v in rnd["variants"][:3]:  # 只显示前3名
                lines.append(
                    f"  {v['name']}: 得分 {v['score']:.3f} "
                    f"(偏差 {v['result']['avg_risk_deviation']:.2f}, "
                    f"方向准 {v['result']['direction_accuracy']:.1%})"
                )
            lines.append(f"  本轮最佳: {rnd['best_this_round']} ({rnd['best_score']:.3f})")
            lines.append("")
        lines.append(f"历史最佳得分: {self.history.best_score:.3f} (第{self.history.best_round}轮)")
        return "\n".join(lines)

    def _parse_variants(self, content: str, round_num: int) -> dict[str, str]:
        """从 LLM 输出中解析出 3 个提示词变体。"""
        variants = {}
        import re
        pattern = r"=== PROMPT_([A-Z]) ===\s*\n(.*?)(?=\n=== PROMPT_|$)"
        matches = re.findall(pattern, content, re.DOTALL)
        for letter, body in matches:
            name = f"r{round_num}_{letter.lower()}"
            variants[name] = body.strip()
        return variants

    async def _evaluate_prompt(self, name: str, prompt: str,
                               use_holdout: bool = False) -> EvaluationResult:
        """在验证集上评估一个提示词。

        注意: 这是简化版评估——直接调用 LLM 做诊断，然后和"标准答案"比较。
        标准答案用次日走势反推（涨了说明当时应该更激进，跌了说明应该更保守）。

        use_holdout=True 时用留出集(真实样本外), 否则用选择集(用于选优).
        """
        try:
            # v5.6 P1-6: 统一走 ModelRouter (记账/扣预算/峰谷), 不再直连 AsyncOpenAI
            router = _get_router()

            deviations = []
            direction_correct = 0
            master_correct = 0
            format_ok = 0
            _samples = self.holdout_samples if use_holdout else self.selection_samples
            total = len(_samples)

            for sample in _samples:
                try:
                    result = await router.route(
                        messages=[{"role": "system", "content": prompt},
                                  {"role": "user", "content": sample["user_msg"]}],
                        task_type="risk_assessment",
                        temperature=0.3,
                        max_tokens=400,
                    )
                    content = (result.response or "").strip()
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
                        continue  # 格式错误，跳过

                    format_ok += 1
                    risk = int(result.get("risk_level", 3))
                    optimal = sample["optimal_risk"]

                    # 风险等级偏差
                    dev = abs(risk - optimal)
                    deviations.append(dev)

                    # 方向准确率
                    actual_dir = sample["direction"]  # "up" / "down" / "flat"
                    if actual_dir == "up":
                        # 市场涨了，如果给的风险比中性低（更激进），方向对
                        correct = risk <= 3
                    elif actual_dir == "down":
                        # 市场跌了，如果给的风险比中性高（更保守），方向对
                        correct = risk >= 3
                    else:
                        correct = risk == 3  # 震荡市给中性
                    if correct:
                        direction_correct += 1

                    # 大师选择（简化：只要选了趋势大师在涨市就算对）
                    master = result.get("dominant_master", "")
                    if actual_dir == "up" and master in ["利弗莫尔", "索罗斯"]:
                        master_correct += 1
                    elif actual_dir == "down" and master in ["巴菲特", "达利欧"]:
                        master_correct += 1
                    elif actual_dir == "flat" and master in ["达利欧", "缠中说禅"]:
                        master_correct += 1

                except Exception:
                    continue

            # 计算综合得分
            avg_dev = sum(deviations) / len(deviations) if deviations else 99
            dir_acc = direction_correct / total if total > 0 else 0
            mst_acc = master_correct / total if total > 0 else 0
            fmt_rate = format_ok / total if total > 0 else 0

            # 综合得分: 方向准确率（60%）+ 格式合规（30%）+ 偏差小（10%）
            # 偏差越小越好，所以用倒数
            dev_score = 1.0 / (avg_dev + 0.5)  # 0偏差=2.0, 1偏差=0.67
            overall = dir_acc * 0.6 + fmt_rate * 0.3 + min(dev_score / 2.0, 1.0) * 0.1

            return EvaluationResult(
                prompt_name=name,
                avg_risk_deviation=avg_dev,
                direction_accuracy=dir_acc,
                master_accuracy=mst_acc,
                format_compliance=fmt_rate,
                overall_score=overall,
                sample_count=total,
            )

        except Exception as e:
            print(f"  评估失败 {name}: {e}")
            return EvaluationResult(
                prompt_name=name, avg_risk_deviation=99,
                direction_accuracy=0, master_accuracy=0,
                format_compliance=0, overall_score=-999,
                sample_count=0,
            )

    def _save(self):
        self.history.save(self.output_dir / "optimization_history.json")
        # 同时保存最佳提示词
        with open(self.output_dir / "best_prompt.txt", "w", encoding="utf-8") as f:
            f.write(self.history.best_prompt)
