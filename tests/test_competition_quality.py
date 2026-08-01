"""
竞赛对话质量自动化评估 (v3.0-competition)

对每个测试问题,调用 CompetitionAgent.chat(),
用 LLM-Judge 评估回复质量并输出评分报告。
"""

import asyncio
import json
import time
from pathlib import Path
from typing import Any, Dict, List

import pytest
from loguru import logger

# 全文件依赖真实 LLM 调用 (CompetitionAgent.chat + 评分), 标记 network, 不纳入默认套件
pytestmark = pytest.mark.network


# 加载测试问题集
def load_questions() -> List[Dict]:
    questions_path = Path(__file__).parent / "competition_questions.json"
    with open(questions_path, encoding="utf-8") as f:
        data = json.load(f)
    return data["questions"]


# 评分标准
SCORING_RUBRIC = {
    "format_compliance": {
        "label": "格式合规",
        "max": 3,
        "criteria": [
            "回复长度 > 50字 (1分)",
            "有明确的段落/标题结构 (1分)",
            "有编号列表或结构化输出 (1分)",
        ],
    },
    "data_accuracy": {
        "label": "数值准确性",
        "max": 2,
        "criteria": [
            "引用了具体数值 (1分)",
            "数值来源明确可验证 (1分)",
        ],
    },
    "logic_consistency": {
        "label": "逻辑一致性",
        "max": 2,
        "criteria": [
            "分析有因果推理 (1分)",
            "结论有数据支撑 (1分)",
        ],
    },
    "risk_warning": {
        "label": "风险提示完整性",
        "max": 3,
        "criteria": [
            "提及风险因素 (1分)",
            "包含止损/仓位建议 (1分)",
            "包含免责声明 (1分)",
        ],
    },
}


def evaluate_reply(reply: str, expected_points: List[str]) -> Dict[str, Any]:
    """
    评估单条回复质量

    Args:
        reply: AI回复文本
        expected_points: 期望回答包含的要点

    Returns:
        评分详情 dict
    """
    scores = {}
    details = {}

    if not reply:
        return {"total_score": 0, "scores": {}, "details": {"error": "回复为空"}}

    # 1. 格式合规 (0-3)
    fmt_score = 0.0
    if len(reply) > 50:
        fmt_score += 1.0
    if "###" in reply or "##" in reply:
        fmt_score += 1.0
    if any(c.isdigit() for c in reply[:5]) or "1." in reply:
        fmt_score += 1.0
    scores["format_compliance"] = fmt_score
    details["format_compliance"] = f"回复长度{len(reply)}字, 段落{'有' if '#' in reply else '无'}, 列表{'有' if any('1.' in reply for _ in [1]) else '无'}"

    # 2. 数值准确性 (0-2)
    import re
    numbers = re.findall(r'\d+\.?\d*%?', reply)
    acc_score = min(2.0, len(numbers) / 3)
    scores["data_accuracy"] = acc_score
    details["data_accuracy"] = f"引用数值{len(numbers)}个"

    # 3. 逻辑一致性 (0-2)
    logic_keywords = ["因为", "所以", "因此", "由于", "基于", "综合"]
    logic_score = min(2.0, sum(1 for kw in logic_keywords if kw in reply) / 2)
    scores["logic_consistency"] = logic_score
    details["logic_consistency"] = f"因果推理关键词: {sum(1 for kw in logic_keywords if kw in reply)}个"

    # 4. 风险提示 (0-3)
    risk_score = 0.0
    risk_kws = ["风险", "止损", "回撤", "仓位"]
    risk_score += min(2.0, sum(1 for kw in risk_kws if kw in reply) / 2)
    disclaimers = ["不构成", "历史数据不代表", "仅供参考", "投资建议", "风险提示"]
    if any(d in reply for d in disclaimers):
        risk_score += 1.0
    scores["risk_warning"] = risk_score
    details["risk_warning"] = f"风险相关词: {sum(1 for kw in risk_kws if kw in reply)}个, 免责声明: {'有' if any(d in reply for d in disclaimers) else '无'}"

    # 期望要点覆盖
    covered = sum(1 for point in expected_points if any(kw in reply for kw in _point_keywords(point)))
    details["expected_coverage"] = f"覆盖期望要点: {covered}/{len(expected_points)}"

    total = sum(scores.values())
    return {
        "total_score": round(total, 1),
        "max_possible": 10.0,
        "scores": {k: round(v, 1) for k, v in scores.items()},
        "details": details,
    }


def _point_keywords(point: str) -> List[str]:
    """根据期望要点返回关键词列表"""
    kw_map = {
        "市场体制判断": ["牛", "熊", "震荡", "体制", "市场状态"],
        "指数数据": ["指数", "上证", "深证", "创业板", "点位"],
        "成交量": ["成交量", "量", "换手", "成交额"],
        "涨跌比": ["涨跌比", "涨", "跌", "涨停", "跌停"],
        "趋势分析": ["趋势", "均线", "排列", "多头", "空头"],
        "动量指标": ["RSI", "MACD", "KDJ", "动量"],
        "支撑阻力": ["支撑", "阻力", "压力", "关口"],
        "综合评分": ["评分", "综合", "总分"],
        "RSI数值": ["RSI", "超买", "超卖"],
        "背离分析": ["背离", "顶背离", "底背离"],
        "北向资金数据": ["北向", "资金", "净流入", "净流出"],
        "流向分析": ["流入", "流出", "趋势"],
        "领涨板块": ["板块", "领涨", "涨幅"],
        "回测结果": ["回测", "胜率", "历史"],
        "风控触发": ["风控", "触发", "止损", "减仓"],
        "行业集中度": ["集中", "行业", "分散", "板块集中"],
        "T+1定义": ["T+1", "次日", "当天"],
        "MACD原理": ["MACD", "DIF", "DEA", "柱状"],
    }
    return kw_map.get(point, [point])


class TestCompetitionQuality:
    """
    竞赛对话质量测试套件

    使用方法:
        pytest tests/test_competition_quality.py -v

    注意: 这些测试需要 LLM 可用 (DeepSeek API 或 Ollama)
    """

    @pytest.fixture(autouse=True)
    async def setup_agent(self):
        """初始化 CompetitionAgent"""
        try:
            from agent.competition_agent import CompetitionAgent
            self.agent = CompetitionAgent()
            await self.agent._ensure_initialized()
        except Exception as e:
            pytest.skip(f"Agent初始化失败: {e}")

    @pytest.mark.asyncio
    @pytest.mark.parametrize("question", load_questions())
    async def test_question_quality(self, question):
        """逐题测试对话质量"""
        q_id = question["id"]
        q_text = question["question"]
        expected = question["expected_points"]
        min_score = question.get("min_score", 4)

        try:
            reply = await self.agent.chat(q_text, session_id=f"test_{q_id}")
            result = evaluate_reply(reply, expected)

            logger.info(f"[{q_id}] 得分: {result['total_score']}/10 "
                        f"(最低要求: {min_score})")
            logger.info(f"    问题: {q_text[:60]}...")
            logger.info(f"    回复预览: {reply[:100]}...")
            logger.info(f"    详情: {json.dumps(result['details'], ensure_ascii=False)}")

            assert result["total_score"] >= min_score, (
                f"[{q_id}] 质量不达标: {result['total_score']}/10 < {min_score}\n"
                f"回复: {reply[:200]}"
            )
        except Exception as e:
            logger.warning(f"[{q_id}] 测试出错: {e}")
            # API不可用时跳过,不fail
            if "API" in str(e) or "connect" in str(e).lower() or "timeout" in str(e).lower():
                pytest.skip(f"API不可用: {e}")
            else:
                raise


def run_quality_report():
    """
    运行质量评估并生成报告 (无需pytest,独立运行)

    用法:
        python tests/test_competition_quality.py
    """
    async def _run():
        from agent.competition_agent import CompetitionAgent

        print("=" * 60)
        print("AI+金融量化分析智能体 — 对话质量评估报告")
        print("=" * 60)

        agent = CompetitionAgent()
        print("初始化智能体...")
        await agent._ensure_initialized()

        questions = load_questions()
        results = []
        total_score = 0.0

        for q in questions:
            q_id = q["id"]
            print(f"\n[{q_id}] 测试: {q['question'][:60]}...")
            t0 = time.time()
            try:
                reply = await agent.chat(q["question"], session_id=f"quality_{q_id}")
                elapsed = time.time() - t0
                result = evaluate_reply(reply, q["expected_points"])
                result["id"] = q_id
                result["category"] = q["category"]
                result["elapsed_s"] = round(elapsed, 1)
                result["reply_preview"] = reply[:150]
                results.append(result)
                total_score += result["total_score"]
                status = "✅" if result["total_score"] >= q.get("min_score", 4) else "⚠️"
                print(f"  {status} 得分: {result['total_score']}/10 ({elapsed:.1f}s)")
            except Exception as e:
                print(f"  ❌ 错误: {e}")
                results.append({"id": q_id, "error": str(e), "total_score": 0})

        # 汇总
        avg_score = total_score / len(questions) if questions else 0
        print("\n" + "=" * 60)
        print("评估汇总")
        print("=" * 60)
        print(f"总题数: {len(questions)}")
        print(f"完成数: {sum(1 for r in results if 'error' not in r)}")
        print(f"平均得分: {avg_score:.1f}/10")
        print(f"总耗时: {sum(r.get('elapsed_s', 0) for r in results):.0f}s")

        # 按类别汇总
        print("\n按类别得分:")
        categories = {}
        for r in results:
            cat = r.get("category", "unknown")
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(r["total_score"])
        for cat, scores in sorted(categories.items()):
            avg = sum(scores) / len(scores) if scores else 0
            print(f"  {cat}: {avg:.1f}/10 ({len(scores)}题)")

        return {"average_score": avg_score, "results": results}

    return asyncio.run(_run())


if __name__ == "__main__":
    run_quality_report()
