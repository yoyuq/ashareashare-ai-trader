"""
AI 推荐后过滤器 — 去掉噪声, 精炼到高质量标的

过滤链:
  1. 板块过滤: 去北交所 (流动性差, 波动极端)
  2. 评分过滤: < 70 分直接排除 (Top100里中位数仅48分)
  3. 体制过滤: 弱熊/强熊时只保留防守型板块
  4. 基本面过滤: PE>200 或 PE<0 排除
  5. 策略交叉验证: 至少2个验证过的策略确认

输出: 精炼后的推荐列表 + 过滤日志
"""

import json
import sys
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))
from loguru import logger


class PostFilter:
    """AI 推荐后过滤器"""

    # 防御型板块 (弱市时保留)
    DEFENSIVE_SECTORS = {"银行", "公用事业", "食品饮料", "医药生物", "煤炭", "电力"}

    def __init__(self):
        self.filter_log: List[str] = []

    def filter(
        self,
        recommendations: List[Dict],
        regime: str = "range_bound",
        min_score: int = 70,
        require_strategy_confirm: bool = True,
    ) -> List[Dict]:
        """
        过滤 AI 推荐列表

        Args:
            recommendations: DeepSeek 分析结果列表
            regime: 市场体制
            min_score: 最低评分 (默认70)
            require_strategy_confirm: 是否需要策略交叉验证

        Returns:
            过滤后的推荐列表
        """
        self.filter_log = []
        passed = []

        for r in recommendations:
            code = str(r.get("code", ""))
            name = r.get("name", "")
            score = r.get("final_score", r.get("score", 0))
            action = r.get("action", "")
            technical = r.get("technical", "")
            fundamental = r.get("fundamental", "")
            risk = r.get("risk", "")

            # 只看 BUY
            if action != "BUY":
                continue

            # ── 1. 板块过滤: 去北交所 ──
            if code.startswith(("4", "8", "9")) and len(code) >= 6:
                self.filter_log.append(f"❌ {code} {name}: 北交所, 流动性差")
                continue

            # ── 2. 评分过滤 ──
            if score < min_score:
                self.filter_log.append(f"❌ {code} {name}: 评分{score} < {min_score}")
                continue

            # ── 3. 亏损股过滤 ──
            fundamental_str = str(fundamental) if fundamental else ""
            if "亏损" in fundamental_str or "业绩下滑" in fundamental_str or "PE为负" in fundamental_str:
                self.filter_log.append(f"❌ {code} {name}: 基本面差 ({fundamental[:40]})")
                continue

            # ── 4. 高风险过滤 ──
            risk_str = str(risk) if risk else ""
            high_risk_keywords = ["流动性", "退市", "ST", "监管", "庄股", "操纵"]
            if any(kw in risk_str for kw in high_risk_keywords):
                self.filter_log.append(f"❌ {code} {name}: 高风险 ({risk[:40]})")
                continue

            # ── 5. 体制过滤 ──
            if regime in ("weak_bear", "strong_bear"):
                # 弱市只保留防御型+低估值
                if "估值" not in fundamental and "PE" not in fundamental:
                    self.filter_log.append(f"❌ {code} {name}: {regime}下无估值优势")
                    continue

            # ── 6. 实时价格检查 ──
            try:
                import requests
                prefix = "sh" if code.startswith("6") else ("bj" if code.startswith("9") else "sz")
                resp = requests.get(f"https://qt.gtimg.cn/q={prefix}{code}", timeout=3)
                resp.encoding = "gbk"
                for line in resp.text.split("\n"):
                    if "=" in line and "~" in line:
                        fields = line.split("=", 1)[1].strip('"').split("~")
                        if len(fields) > 32 and fields[2] == code:
                            # field[32] is pct_change, field[4] is prev_close
                            pct = float(fields[32]) if fields[32] else 0
                            prev_close = float(fields[4]) if len(fields) > 4 and fields[4] else 0
                            cur_price = float(fields[3]) if fields[3] else 0
                            # Compute pct if field[32] is zero but we have price data
                            if pct == 0 and prev_close > 0 and cur_price > 0:
                                pct = (cur_price / prev_close - 1) * 100
                            if pct <= -8:
                                self.filter_log.append(f"X {code} {name}: 今日跌{pct:.1f}%, 排除")
                                score = 0
                            elif pct <= -3:
                                self.filter_log.append(f"! {code} {name}: 今日跌{pct:.1f}%, 降分10")
                                score -= 10
                            # Store today's performance for display
                            r["today_pct"] = round(pct, 1)
                            break
            except Exception:
                pass

            if score <= 0:
                continue

            # ── 7. 涨停次日降级 ──
            if "涨停" in technical and regime in ("range_bound", "weak_bear"):
                score -= 15
                self.filter_log.append(f"⚡ {code} {name}: 涨停信号在{regime}下降分15 → {score}")

            if score >= min_score:
                passed.append({**r, "final_score": score, "filtered": True})
            else:
                self.filter_log.append(f"❌ {code} {name}: 降分后{score} < {min_score}")

        # ── 策略交叉验证 ──
        if require_strategy_confirm and passed:
            passed = self._strategy_cross_check(passed, regime)

        # 排序
        passed.sort(key=lambda x: x.get("final_score", 0), reverse=True)

        logger.info(f"过滤: {len(recommendations)} → {len(passed)} (过滤{len(self.filter_log)}条)")
        return passed

    def _strategy_cross_check(self, recommendations: List[Dict], regime: str) -> List[Dict]:
        """策略交叉验证: 至少2个验证策略确认才通过"""
        try:
            from analysis.strategies_v3 import multifactor_v3, macd_trend_v3
            from analysis.optimized_strategies import backtest_momentum_v2
            from data.router import get_data_router
            from data.providers.base import DataFrequency, DataRequest
            import asyncio

            monitors = [
                ("多因子v3", multifactor_v3),
                ("MACDv3", macd_trend_v3),
                ("动量v2", backtest_momentum_v2),
            ]

            async def _check():
                router = get_data_router()
                today = date.today()
                confirmed = []

                for r in recommendations:
                    code = r.get("code", "")
                    prefix = "sh" if code.startswith("6") else "sz"
                    sym = f"{prefix}.{code}"

                    try:
                        from datetime import timedelta
                        req = DataRequest(sym, today - timedelta(days=200), today, DataFrequency.DAILY)
                        result = await router.get_daily_kline(req)
                        df = result.data
                        if df.empty or len(df) < 60:
                            self.filter_log.append(f"⚡ {code} {r.get('name','')}: 无足够K线数据")
                            continue

                        votes = 0
                        for sname, sfunc in monitors:
                            bt = sfunc(df)
                            if bt.get("win_rate", 0) > 0.35 and bt.get("signals", 0) >= 2:
                                votes += 1

                        if votes >= 2:
                            confirmed.append(r)
                            self.filter_log.append(f"✅ {code} {r.get('name','')}: {votes}/3策略确认")
                        else:
                            self.filter_log.append(f"❌ {code} {r.get('name','')}: 仅{votes}/3策略确认, 拒绝")
                    except Exception as e:
                        self.filter_log.append(f"⚠️ {code}: 策略验证失败 ({e})")
                        # 无法验证的不排除, 保留
                        confirmed.append(r)

                return confirmed

            return asyncio.run(_check())

        except Exception as e:
            logger.warning(f"策略交叉验证跳过: {e}")
            return recommendations

    def get_log(self) -> str:
        return "\n".join(self.filter_log)

    def print_summary(self, passed: List[Dict]):
        """打印精炼结果"""
        print(f"\n{'='*60}")
        print(f"精炼推荐: {len(passed)} 只")
        print(f"{'='*60}")
        print(f"{'代码':<8} {'名称':<8} {'评分':>4} {'板块':<6} {'关键逻辑':<30}")
        print(f"{'-'*60}")
        for r in passed:
            code = r.get("code", "")
            board = "主板"
            if code.startswith("3"): board = "创业板"
            elif code.startswith("688"): board = "科创板"
            print(f"{code:<8} {r.get('name',''):<8} {r.get('final_score',0):>4} "
                  f"{board:<6} {r.get('technical','')[:30]}")
        print(f"{'='*60}")


# ── 快速过滤 ──

def quick_filter(analysis_path: str = None) -> List[Dict]:
    """一行调用: 从分析结果加载并过滤"""
    if analysis_path is None:
        analysis_path = str(Path(__file__).parent.parent / "reports" / "deep_analysis_top100.json")

    with open(analysis_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    regime = data.get("market_regime", {}).get("regime", "range_bound")
    results = data.get("results", [])

    pf = PostFilter()
    passed = pf.filter(results, regime=regime)

    pf.print_summary(passed)
    if pf.filter_log:
        print(f"\n过滤日志 ({len(pf.filter_log)}条):")
        for log in pf.filter_log[:20]:
            print(f"  {log}")

    return passed


if __name__ == "__main__":
    quick_filter()
