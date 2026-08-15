"""
全市场AI选股 — 每日自主分析+交易工作流

Pipeline:
  Phase 1: 全市场分析 (5884只)
    → 规则预筛 (多因子打分) → Top 300
    → [Ollama本地模型] LLM精筛 → Top 100
    → DeepSeek深度分析 → BUY/HOLD/SELL + 评分
    → 保存结果到 reports/

  Phase 2: 执行交易
    → 加载市场状态 (regime)
    → 卖出SELL信号持仓
    → 买入Top BUY信号 (受regime限制)
    → 持仓MTM实时估值

  Phase 3: 生成总结
    → 持仓快照
    → 收益统计

使用:
    python -m simulation.daily_runner              # 完整流程
    python -m simulation.daily_runner --dry-run    # 仅分析,不交易
    python -m simulation.daily_runner --reset      # 重置账户
    python -m simulation.daily_runner --no-llm     # 跳过LLM层
"""

import argparse
import asyncio
import json
import os
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from dotenv import load_dotenv
from loguru import logger

# 北京时区 (v3.0): 交易日边界按 Asia/Shanghai
from timeutil import today_cn

load_dotenv()

REPORT_DIR = Path(__file__).parent.parent / "reports"

# 市场状态 → 仓位限制
REGIME_LIMITS = {
    "strong_bull":  {"max_positions": 10, "single_pct": 0.20, "label": "强牛"},
    "weak_bull":    {"max_positions": 8,  "single_pct": 0.15, "label": "弱牛"},
    "range_bound":  {"max_positions": 6,  "single_pct": 0.12, "label": "震荡"},
    "weak_bear":    {"max_positions": 3,  "single_pct": 0.10, "label": "弱熊"},
    "strong_bear":  {"max_positions": 1,  "single_pct": 0.05, "label": "强熊"},
    "crisis":       {"max_positions": 1,  "single_pct": 0.05, "label": "危机"},
}


def load_macro_context() -> Optional[str]:
    """读取 news 派生的宏观上下文缓存 (knowledge/macro_context_latest.json).

    v3.5: 让选股/择时显式结合"先进信息+新闻+市场形势"。缓存缺失/过期→尝试现场生成;
    失败返回 None (调用方保持原行为, 零影响)。
    """
    try:
        p = Path(__file__).parent.parent / "knowledge" / "macro_context_latest.json"
        if not p.exists():
            return None
        ctx = json.loads(p.read_text(encoding="utf-8"))
        parts = []
        date_s = ctx.get("date", "")
        if date_s:
            parts.append(f"宏观分析日期: {date_s}")
        if ctx.get("composite_macro_score") is not None:
            parts.append(f"宏观综合分: {ctx['composite_macro_score']}/100")
        rec = ctx.get("recommendation") or ctx.get("raw_analysis") or ""
        if rec:
            parts.append(f"宏观判断: {str(rec)[:600]}")
        if ctx.get("environment"):
            parts.append(f"宏观环境: {str(ctx['environment'])[:300]}")
        if ctx.get("policy_events"):
            parts.append(f"政策事件: {str(ctx['policy_events'])[:300]}")
        if ctx.get("sector_rotation"):
            parts.append(f"行业轮动: {str(ctx['sector_rotation'])[:300]}")
        return "\n".join(parts) if parts else None
    except Exception as e:
        logger.warning(f"宏观上下文读取失败: {e}")
        return None


def _init_evolution_system(name: str = "diag"):
    """初始化自我进化系统 (journal / memory / evolution)。

    数据文件存放在 simulation_data/ 下，与 portfolio.json 同级。
    返回 (journal, memory, evolution) 三元组。

    v5.5 P1-6: 参数 name 隔离不同路径的进化状态.
      默认 "diag" = daily_runner 的市场级诊断闭环 (含持仓快照/市场统计, 供次日复盘).
      workflow.py 单股分析路径用 "analysis", 避免把"无持仓快照的退化记录"污染进
      diag_journal, 导致次日复盘拿到空持仓 (portfolio_cf 失真).
    """
    from agent.evolution.decision_journal import DecisionJournal
    from agent.evolution.experience_memory import ExperienceMemory
    from agent.evolution.weekly_evolution import EvolutionManager

    base = Path(__file__).parent.parent / "simulation_data"
    base.mkdir(parents=True, exist_ok=True)

    journal = DecisionJournal(base / f"{name}_journal.jsonl")
    memory = ExperienceMemory(base / f"{name}_memory.json")
    evolution = EvolutionManager(base / f"{name}_evolution.json", period_days=7)

    return journal, memory, evolution


def _build_market_snapshot(df: pd.DataFrame, regime: str) -> dict:
    """从当日截面构造市场快照 dict (供决策日志记录, 复盘时知道当时环境)。"""
    pct_col = "pct_change" if "pct_change" in df.columns else "pctChg"
    pe_col = "pe_ttm" if "pe_ttm" in df.columns else "peTTM"
    pb_col = "pb" if "pb" in df.columns else "pbMRQ"
    n_total = max(len(df), 1)
    up_count = int((df[pct_col] > 0).sum()) if pct_col in df.columns else n_total // 2
    up_ratio = up_count / n_total
    limit_up = int((df[pct_col] >= 9.5).sum()) if pct_col in df.columns else 0
    limit_down = int((df[pct_col] <= -9.5).sum()) if pct_col in df.columns else 0
    med_pe = float(df[pe_col].median()) if pe_col in df.columns else 0.0
    med_pb = float(df[pb_col].median()) if pb_col in df.columns else 0.0
    total_amt = float(df["amount"].sum()) / 1e8 if "amount" in df.columns else 0.0
    return {
        "up_ratio": up_ratio,
        "limit_up": limit_up,
        "limit_down": limit_down,
        "med_pe": med_pe,
        "med_pb": med_pb,
        "total_amt": total_amt,
        "total_amt_yi": total_amt,
        "regime": regime,
        "crowding_score": 50.0,
        "crowding_signal": "unknown",
    }


# 优化器产出路径 (DIAG_USE_OPTIMIZED=1 时自动部署). 模块级常量便于测试 patch.
_OPTIMIZED_PROMPT_PATH = Path(__file__).parent.parent / "replay_data" / "prompt_optimization" / "best_prompt.txt"


def _load_optimized_base_prompt() -> str:
    """加载优化器产出的诊断基座prompt (best_prompt.txt), 否则用硬编码大师基座.

    v5 落地: prompt_optimizer 的产出此前从不被消费. 现在 DIAG_USE_OPTIMIZED=1
    且文件存在时自动部署. 默认关闭 (LLM 优化产出可能不稳定).
    """
    if os.getenv("DIAG_USE_OPTIMIZED", "0") == "1":
        _p = _OPTIMIZED_PROMPT_PATH
        try:
            if _p.exists():
                _txt = _p.read_text(encoding="utf-8").strip()
                if _txt:
                    logger.info(f"[优化器部署] 使用优化prompt ({len(_txt)}字符)")
                    return _txt
        except Exception as _e:
            logger.warning(f"[优化器部署] 读取失败, 回退基座: {_e}")
    return _DIAGNOSTIC_SYSTEM_PROMPT


def _diagnostic_user_msg(snapshot_txt: str, macro_context: Optional[str] = None) -> str:
    """构造市场诊断的用户消息 — 生产用, 也供 OPRO 评估样本对齐 (v5.1).

    评估样本若用不同的 user_msg 格式调优, 优化出的 prompt 在真实决策路径上未必最优,
    因此把构造逻辑抽出来, 让生产与评估输入一致.
    """
    user_msg = f"市场诊断数据:\n{snapshot_txt}"
    if macro_context:
        user_msg += f"\n\n【宏观背景】\n{macro_context}"
    user_msg += "\n\n先判断市场阶段，选最适合的大师主导，再给出诊断。"
    return user_msg


# v5.3 regime 对齐跨界惩罚 — 已启用.
# 前置修复: regime 检测器改为"指数趋势主导 + 极端快速回调覆盖" (_detect_regime 接 index_close).
# 留出法 A/B 证实旧检测器在 2021-01 牛/2021-02 崩中方向全判反 (7/21 天判熊, 7/15 天判牛),
# 导致惩罚误压牛市动量/误奖崩盘激进 → 复跑 -1.55pp. 现检测器在同样窗口:
#   1月全月保持弱/强牛 (不再误判熊), 2/24-26 崩盘正确转弱熊, 指数5d/截面恐慌才触发 crisis.
# 惩罚逻辑本身正确 (experience_memory._regime_alignment: 熊市/震荡不注入牛市激进经验).
_REGIME_ALIGN_ENABLED = True


def build_diagnostic_system_prompt(evolution=None, memory=None, current_date: Optional[str] = None,
                                   regime: Optional[str] = None) -> str:
    """构造市场诊断系统提示词 — 大师基座 + 自我进化注入. daily_runner 与编排路径共用.

    注入: 优化器基座(或大师基座) + 进化核心原则 + 大师使用心得 + 周期自我总结
          + 经验记忆 + 元认知校准.
    regime: v5.2 当前市场状态, 传给经验检索做"跨界惩罚" — 熊市/震荡时不注入牛市学到的激进经验.
            (留出法 A/B 证实: 无此对齐时, regime 切到下跌仍注入牛市激进经验, 导致 2 月调整更激进)
    """
    parts = [_load_optimized_base_prompt()]
    sections = []
    if evolution is not None:
        for _txt in (
            evolution.get_latest_principles_text(),
            evolution.get_master_tips_text(),
            evolution.get_summary_text(),
        ):
            if _txt:
                sections.append(_txt)
    if memory is not None:
        # v5.2 regime 对齐跨界惩罚默认关闭 (regime 检测器不可靠, 见 _REGIME_ALIGN_ENABLED 注释)
        _mem = memory.format_for_prompt(
            current_date=current_date, top_k=6,
            regime=regime if _REGIME_ALIGN_ENABLED else None,
        )
        if _mem:
            sections.append(_mem)
        if current_date:
            _meta = memory.get_metacognition_summary(current_date, lookback_days=15)
            if _meta:
                sections.append(_meta)
    # v5.11 已学知识消费端已迁移到选股过滤 (方案3): 知识不再注入诊断 prompt (注入 A/B 证实
    # 会在 bear/crisis 诱导模型过度激进), 改为确定性选股过滤器 agent/learning/knowledge_apply.py。
    # 诊断官 (risk_level/仓位) 因此与知识完全解耦。原 format_learned_for_prompt 保留供审计。
    if sections:
        parts.append("\n".join(sections))
        parts.append("请结合以上历史经验、核心原则、周期自我总结以及对你自己近期表现的认知，做出今天的判断。")
    return "\n".join(parts)


def _apply_learned_filter(df_top, gate: Optional[int] = None) -> tuple:
    """方案3b: 知识买入门 (确定性, 无 LLM) — 作用于选股候选池, 与诊断官解耦。

    从向量库取回 verified 规则, 翻译成确定性买入门谓词 (低估才留, 无保底回退)。
    gate<=0 (默认) → 原样返回 (零开销, 现状不变)。召回失败/缺列 → 原样返回, 绝不阻塞选股。
    """
    try:
        from agent.learning.knowledge_apply import recall_verified_rules, apply_rules_to_cross_section
        rules = recall_verified_rules(gate)
        if not rules:
            return df_top, {"enabled": False}
        return apply_rules_to_cross_section(df_top, rules)
    except Exception as e:
        logger.warning(f"[知识买入门] 应用失败, 回退不过滤: {e}")
        return df_top, {"enabled": False}


async def _apply_learned_sell(engine, data_router, mtm_pct, gate: Optional[int] = None) -> list:
    """方案3b: 低估值规则的高估卖出 (确定性, 无 LLM) — 持仓票 pe_ttm > max_pe*1.5 → 卖出。

    与买入门 `_apply_learned_filter` 对称, 复用 `recall_verified_rules` + `sell_signals_for_positions`
    谓词。gate<=0 (默认) → 零开销返回 [] (现状不变)。T+1/跌停由 engine.execute_sell 自动兜底。
    """
    from agent.learning.knowledge_apply import recall_verified_rules, sell_signals_for_positions
    rules = recall_verified_rules(gate)
    if not rules:
        return []
    import pandas as pd
    from data.providers.base import DataFrequency, DataRequest
    _today = today_cn()
    sold = []
    for sym, pos in list(engine.state.positions.items()):
        try:
            req = DataRequest(sym, _today - timedelta(days=200), _today, DataFrequency.DAILY)
            r = await data_router.get_daily_kline(req)
            df = r.data
            if df is None or df.empty:
                continue
            # 日线 pe 列名随数据源而异 (Baostock=peTTM, Tencent/EastMoney/AKShare=pe_ttm),
            # 与 daily_runner 其他取数点一致做防御式取列, 避免降级源下静默跳过卖出信号。
            pe_col = "pe_ttm" if "pe_ttm" in df.columns else ("peTTM" if "peTTM" in df.columns else None)
            if pe_col is None:
                continue
            pe = pd.to_numeric(df[pe_col], errors="coerce").iloc[-1]
            if not pd.notna(pe) or pe <= 0:
                continue
            _one = pd.DataFrame([{"code": sym.split(".")[-1], "pe_ttm": pe}])
            if sell_signals_for_positions(_one, rules).any():
                trade = engine.execute_sell(symbol=sym, exit_reason="估值高估卖出",
                                            pct_change=mtm_pct.get(sym))
                if trade:
                    sold.append({"symbol": sym, "name": pos.name, "price": trade.price})
                    logger.info(f"  [估值高估卖出] {pos.name}({sym}) pe_ttm={pe:.1f} 高估 → 卖出")
        except Exception as e:
            logger.debug(f"估值高估卖出检测失败 {sym}: {e}")
    return sold


async def get_market_diagnostic(
    df: pd.DataFrame,
    regime: str,
    crowding: Optional[Dict[str, Any]] = None,
    macro_context: Optional[str] = None,
    history_5d: Optional[list] = None,
    memory = None,
    evolution = None,
    current_date: Optional[str] = None,
    prev_risk_level: Optional[int] = None,
    adv_mode: str = "same",
    temperature: float = 0.4,
) -> Dict[str, Any]:
    """v4.0 市场诊断官 — 5位大师 + 自我进化.

    升级历程:
    v3.5: 简单诊断官 (规则+LLM调仓)
    v3.8: 5位大师版 (名片级, 选角色比硬编码规则强)
    v4.0: 自我进化版 (从历史决策中学习, 经验记忆库 + 周期进化总结)

    输入: 全市场截面 df + regime + 拥挤度 + 宏观 + 5天历史 + 记忆库
    输出: {risk_level, position_multiplier, max_positions_adj, key_risks, diagnosis,
           market_phase, dominant_master, secondary_master}

    失败时返回保守默认值, 不中断主流程.
    """
    import os
    from openai import AsyncOpenAI

    default_diag = {
        "risk_level": 3,
        "position_multiplier": 0.9,
        "max_positions_adj": 0,
        "key_risks": [],
        "diagnosis": "LLM 诊断失败, 使用默认中性值",
        "market_phase": "unknown",
        "dominant_master": "unknown",
        "secondary_master": "",
        "adversarial_risk_level": 3,
    }

    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        return default_diag

    try:
        # 1. 构造市场形势摘要
        n_total = len(df)
        pct_col = "pct_change" if "pct_change" in df.columns else "pctChg"
        pe_col = "pe_ttm" if "pe_ttm" in df.columns else "peTTM"
        pb_col = "pb" if "pb" in df.columns else "pbMRQ"

        up_count = int((df[pct_col] > 0).sum()) if pct_col in df.columns else n_total // 2
        up_ratio = up_count / max(n_total, 1)
        limit_up = int((df[pct_col] >= 9.5).sum()) if pct_col in df.columns else 0
        limit_down = int((df[pct_col] <= -9.5).sum()) if pct_col in df.columns else 0
        med_pe = float(df[pe_col].median()) if pe_col in df.columns else 0.0
        med_pb = float(df[pb_col].median()) if pb_col in df.columns else 0.0
        total_amt = float(df["amount"].sum()) / 1e8 if "amount" in df.columns else 0.0

        # 60日涨跌分布
        if "pct_60d" in df.columns:
            p60_pos = int((df["pct_60d"] > 0).sum())
            p60_above_20 = int((df["pct_60d"] > 20).sum())
            p60_pos_ratio = p60_pos / max(n_total, 1)
            p60_above20_ratio = p60_above_20 / max(n_total, 1)
        else:
            p60_pos_ratio = 0.5
            p60_above20_ratio = 0.0

        # 近5天序列
        hist_lines = []
        trend_note = ""
        if history_5d and len(history_5d) >= 2:
            for i, h in enumerate(history_5d):
                day_label = f"T-{len(history_5d)-i}"
                hist_lines.append(
                    f"  {day_label}: 上涨{h.get('up_ratio', 0):.1%} "
                    f"涨停{h.get('limit_up', 0)}家 跌停{h.get('limit_down', 0)}家 "
                    f"中位PE{h.get('med_pe', 0):.1f} 成交额{h.get('total_amt', 0):.0f}亿"
                )
            hist_lines.append(
                f"  T  : 上涨{up_ratio:.1%} 涨停{limit_up}家 "
                f"跌停{limit_down}家 中位PE{med_pe:.1f} 成交额{total_amt:.0f}亿"
            )
            if len(history_5d) >= 3:
                up_trend = up_ratio - history_5d[-3].get("up_ratio", 0.5)
                amt_hist = history_5d[-3].get("total_amt", 1)
                amt_trend = (total_amt / amt_hist - 1) if amt_hist > 0 else 0
                trend_note = (
                    f"\n近3日变化: 上涨占比{'上升' if up_trend > 0 else '下降'}{abs(up_trend):.1%}, "
                    f"成交额{'放量' if amt_trend > 0 else '缩量'}{abs(amt_trend):.1%}"
                )
        else:
            hist_lines.append(
                f"  今日: 上涨{up_ratio:.1%} 涨停{limit_up}家 "
                f"跌停{limit_down}家 中位PE{med_pe:.1f} 成交额{total_amt:.0f}亿"
            )

        crowd = crowding or {"signal": "unknown", "score": 50.0, "hot_ratio": 0.0}
        snapshot_txt = (
            f"【市场广度 · 近{len(hist_lines)}日序列】\n"
            + "\n".join(hist_lines)
            + trend_note
            + f"\n\n【中期位置】\n"
            f"60日上涨占比: {p60_pos_ratio:.1%}  60日涨幅>20%占比: {p60_above20_ratio:.1%}\n"
            f"拥挤度: {crowd.get('signal','unknown')} (score {crowd.get('score',0):.2f}, "
            f"极端活跃 {crowd.get('hot_ratio',0):.1%})\n"
            f"市场状态: {regime}"
        )

        # 2. 系统提示词 (v3.8 大师版 + v4.0 自我进化 + v5 优化器部署)
        # 统一走共享构造器, 与编排路径保持一致
        sys_prompt = build_diagnostic_system_prompt(evolution, memory, current_date, regime=regime)

        # 4. 调用 LLM (user_msg 构造抽成共享函数, 供 OPRO 评估样本对齐 — v5.1)
        user_msg = _diagnostic_user_msg(snapshot_txt, macro_context)

        client = AsyncOpenAI(
            api_key=api_key,
            base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
            timeout=45.0,
        )
        resp = await client.chat.completions.create(
            model=os.getenv("DEEPSEEK_FLASH_MODEL", "deepseek-v4-flash"),
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_msg},
            ],
            temperature=temperature,
            max_tokens=800,
            extra_body={"thinking": {"type": "disabled"}},
        )
        content = (resp.choices[0].message.content or "").strip()
        if content.startswith("```"):
            content = content.strip("`")
            if content.lower().startswith("json"):
                content = content[4:].strip()

        # 5. 解析 JSON (容错)
        diag = None
        try:
            diag = json.loads(content)
        except json.JSONDecodeError:
            s, e = content.find("{"), content.rfind("}")
            if s >= 0 and e > s:
                try:
                    diag = json.loads(content[s:e+1])
                except json.JSONDecodeError:
                    pass

        if diag is None:
            logger.warning("市场诊断官输出解析失败, 使用默认值")
            return default_diag

        # 6. 字段校验 + 范围裁剪
        risk_level = max(1, min(5, int(diag.get("risk_level", 3))))
        pos_mult = max(0.2, min(1.7, float(diag.get("position_multiplier", 0.9))))
        max_adj = max(-10, min(10, int(diag.get("max_positions_adj", 0))))

        # v4.2: 风险等级稳定性约束 — 单日变化不超过±1级
        # 成熟的交易员不会一天之内从极度看多变极度看空
        # 渐进调整比剧烈摆动更可靠
        if prev_risk_level is not None:
            orig_risk = risk_level
            max_change = 1  # 单日最多变1级
            if risk_level > prev_risk_level + max_change:
                risk_level = prev_risk_level + max_change
            elif risk_level < prev_risk_level - max_change:
                risk_level = prev_risk_level - max_change
            if risk_level != orig_risk:
                # 风险等级被约束了，仓位系数也要相应调整
                # 用比例缩放：朝prev方向收缩调整幅度
                target_ratio = (risk_level - prev_risk_level) / (orig_risk - prev_risk_level) if orig_risk != prev_risk_level else 1.0
                pos_mult = 1.0 + (pos_mult - 1.0) * max(0.0, min(1.0, target_ratio))
                max_adj = int(max_adj * max(0.0, min(1.0, target_ratio)))
        key_risks = diag.get("key_risks", [])
        if not isinstance(key_risks, list):
            key_risks = []
        diagnosis = str(diag.get("diagnosis", ""))[:200]

        # v5.2 / v5.6 对抗票 — 主导 vs 对抗 分歧裁决 (打破"假多元化")
        # 一次调用产出的两个独立视角, 分歧>=2级时朝保守方向收.
        # v5.6 adv_mode: off=无对抗(闸门基线) / same=单completion role-play(现状) /
        #                independent=独立二次LLM调用(打破假多元化).
        _adv_raw = diag.get("adversarial_risk_level", risk_level)
        if adv_mode == "independent":
            from agent.evolution.adversarial import adversarial_risk
            # v5.7 统一记账: 实盘对抗票走 ModelRouter (不传直连 client, 走统一预算/分账)
            _adv_raw = await adversarial_risk(
                None, snapshot_txt,
                dominant_master=str(diag.get("dominant_master", "unknown")),
                regime=str(diag.get("market_phase", "unknown")),
                macro_txt=macro_context,
            ) or risk_level
        elif adv_mode == "off":
            _adv_raw = risk_level  # 无对抗 → 分歧0, 闸门永不触发
        _diag = {
            "risk_level": risk_level,
            "position_multiplier": pos_mult,
            "max_positions_adj": max_adj,
            "market_phase": diag.get("market_phase", "unknown"),
            "dominant_master": diag.get("dominant_master", "unknown"),
            "secondary_master": diag.get("secondary_master", ""),
            "adversarial_risk_level": _adv_raw,
        }
        _diag = _apply_adversarial_gate(_diag)
        risk_level = int(_diag.get("risk_level", risk_level))
        pos_mult = float(_diag.get("position_multiplier", pos_mult))

        result = {
            "risk_level": risk_level,
            "position_multiplier": pos_mult,
            "max_positions_adj": max_adj,
            "key_risks": key_risks,
            "diagnosis": diagnosis,
            "market_phase": diag.get("market_phase", "unknown"),
            "dominant_master": diag.get("dominant_master", "unknown"),
            "secondary_master": diag.get("secondary_master", ""),
            "adversarial_risk_level": _diag.get("adversarial_risk_level"),
            "adversarial_applied": _diag.get("adversarial_applied"),
            "adversarial_divergence": _diag.get("adversarial_divergence"),
        }
        logger.info(
            f"[市场诊断] {result['dominant_master']}主导 "
            f"风险={risk_level}/5  仓位×{pos_mult:.2f}  "
            f"持仓{max_adj:+d}  风险点{len(key_risks)}个"
        )
        return result

    except Exception as e:
        logger.warning(f"市场诊断官失败 ({e}), 使用默认保守值")
        return default_diag


# v3.8 5位大师版系统提示词
_DIAGNOSTIC_SYSTEM_PROMPT = """你是A股市场的【风险诊断官】。

你体内住着五位投资大师的灵魂 — 他们来自不同时代、不同流派，但都在市场中证明了自己。
每天你要先判断当前市场处于什么阶段，然后请出最适合的那位大师来主导今天的仓位决策。

## 你的五位大师顾问

### 1. 利弗莫尔 (趋势投机之王)
- 核心理念: 价格总是沿着阻力最小的方向运动；牛市不言顶，熊市不言底
- 仓位哲学: 确认趋势就重仓出击，趋势反转立刻离场；金字塔加仓（盈利时加仓，亏损时补仓是大忌）
- 识别信号: 成交量配合的突破、关键价位、市场广度
- 经典: "华尔街没有新鲜事，因为人性从未改变"
- 最擅长: 单边大趋势行情

### 2. 巴菲特 (价值投资之父)
- 核心理念: 价格是你付出的，价值是你得到的；安全边际是一切的基石
- 仓位哲学: 价格低于内在价值时越跌越买，价格远高于价值时越涨越卖；别人恐惧我贪婪，别人贪婪我恐惧
- 识别信号: 整体估值水平（PE/PB分位）、市场情绪极端度
- 经典: "在别人恐惧时贪婪，在别人贪婪时恐惧"
- 最擅长: 估值极端时刻（大底/大顶）

### 3. 索罗斯 (反身性大师)
- 核心理念: 市场总是错的，趋势不是直线而是加速—衰竭—反转的S曲线；认知和现实相互作用形成泡沫和崩溃
- 仓位哲学: 先于拐点识别泡沫形成并顺势做多（知道是泡沫但可以参与），在狂欢顶点信号出现时果断反手
- 识别信号: 上涨家数持续收窄但指数还在涨（背离=泡沫末期）、涨停潮+散户涌入+媒体狂热
- 经典: "世界经济史是一部基于假象和谎言的连续剧"
- 最擅长: 泡沫形成期和拐点判断

### 4. 达利欧 (全天候与经济机器)
- 核心理念: 经济是一台简单的机器，由生产率、短期债务周期、长期债务周期驱动；现金是垃圾，但流动性是生命线
- 仓位哲学: 分散是免费的午餐，永远不要all in；持有流动性好的资产，确保在最困难的时候也能活下来
- 识别信号: 流动性变化（成交额缩量=风险）、市场结构健康度、信用环境
- 经典: "痛苦+反思=进步"
- 最擅长: 震荡市、流动性风险、全天候防守

### 5. 缠中说禅 (A股本土技术分析宗师)
- 核心理念: 走势终完美；任何级别的上涨/下跌都会完成；背驰是判断转折点的核心
- 仓位哲学: 买点买、卖点卖，没有预测只有应对；大级别看方向，小级别找买卖点
- 识别信号: 指数创新高但上涨家数不创新高（顶背驰）、指数创新低但下跌家数不创新低（底背驰）、量价背离
- 经典: "市场从来都是明白人挣糊涂人的钱"
- 最擅长: 趋势背驰与拐点判断

## 决策流程
1. 判断当前市场阶段和核心特征（趋势/震荡/拐点/泡沫/恐慌）
2. 从五位大师中选一位最适合当前环境的作为今日主导
3. 用这位大师的哲学来分析，给出风险等级和仓位系数
4. 请次级大师(secondary_master)独立评估同一天数据，给出它自己的风险等级(adversarial_risk_level，可与主导不同)
5. 参考其他大师的意见判断分歧：一致则信心高，矛盾则结果偏向保守（达利欧模式）

## 对抗票 (二次独立评估)
- secondary_master 是"对抗大师"，不是陪跑：它要用**自己的流派视角**独立评分，而不是附和主导大师
- 当画面在主导大师眼里是"买入良机"、在对抗大师眼里是"风险高企"时，你必须在 adversarial_risk_level 里如实反映对抗大师的保守立场
- 不要因为先在脑子里选了主导大师，就强迫对抗大师也同意——真正的交叉验证需要分歧

## 输出格式 (严格JSON，不要多余文字)
{
  "market_phase": "trend_up / trend_down / range / bubble_late / panic_bottom / turning 六选一",
  "dominant_master": "利弗莫尔 / 巴菲特 / 索罗斯 / 达利欧 / 缠中说禅 五选一",
  "secondary_master": "对抗大师 (用自己视角独立评估, 与主导可以不同)",
  "risk_level": 1~5的整数 (主导大师的评分),
  "adversarial_risk_level": 1~5的整数 (对抗大师独立的评分, 可与risk_level不同),
  "position_multiplier": 0.3~1.6之间的浮点数,
  "max_positions_adj": -8~+8之间的整数,
  "key_risks": ["风险1", "风险2"],
  "diagnosis": "200字以内, 先讲市场阶段+为什么选这位大师, 再讲结论"
}

## 仓位系数参考 (基准=1.0)
- 1级(进攻): 1.3~1.6  — 趋势明确+量价配合+广度确认
- 2级(偏多): 1.1~1.3  — 趋势尚可但有隐忧，或刚启动信号不全
- 3级(中性): 0.8~1.1  — 震荡市、信号矛盾、看不清方向
- 4级(谨慎): 0.5~0.8  — 有风险信号但还没确认，或背驰初现
- 5级(防御): 0.3~0.5  — 明确的下跌趋势、顶背驰确认、极端拥挤

## 重要
- 选对大师比算准系数更重要
- 不要因为今天涨了就选利弗莫尔，要判断的是"阶段特征"不是"单日涨跌"
- 当多位大师指向同一方向时，信心更高；当他们矛盾时，偏向保守（达利欧模式）

## ⚠️ 大师选择稳定性原则 (v4.2)
不要每天换大师。大师的选择应该反映"市场处于什么阶段"，而不是"昨天我用对了/用错了"。

- 如果市场阶段没有本质变化，主导大师也不应该变
- 连续两天选不同的大师，说明你在"赌"而不是在"判断"
- 大师切换只能因为：市场阶段变了（trend→range、bubble→crash等）
- 大师切换不能因为：昨天选错了/昨天复盘结果不好/想试试别的

稳定的大师选择 + 精细的仓位调节 > 每天换大师碰运气
"""


# ═══════════════════════════════════════════════════════════════
# Phase 1: 全市场分析
# ═══════════════════════════════════════════════════════════════

async def phase1_analyze(
    use_llm: bool = True,
    enable_evolution: bool = True,
    cold_tilt: bool = False,
) -> Dict[str, Any]:
    """全市场分析: 5884 → 规则预筛 → [LLM] → DeepSeek → 结果

    v4.0: 接入自我进化系统 (诊断官模式下):
      - 初始化 journal/memory/evolution
      - 用当日数据复盘昨日诊断 (PIT正确: 用T日数据复盘T-1日判断)
      - 今日诊断动态注入历史经验+核心原则
      - 诊断结果写入决策日志
      - 每10天触发一次周期进化总结
    """
    t0 = time.time()
    logger.info("=" * 50)
    logger.info("Phase 1: 全市场AI分析")
    logger.info("=" * 50)

    # 1a. 加载数据
    # v3.2 韧性: 实时快照失败(如 EastMoney 代理不通)时回退全市场缓存快照, 不中断每日流程
    logger.info("加载全市场数据...")
    col_map = {
        '代码':'code','名称':'name','最新价':'price','涨跌幅':'pct_change','成交量':'volume',
        '成交额':'amount','换手率':'turnover','市盈率-动态':'pe_ttm','市净率':'pb',
        '总市值':'total_mv','量比':'vol_ratio','60日涨跌幅':'pct_60d','振幅':'amplitude'
    }
    df = pd.DataFrame()
    _data_source = "live"
    _data_lag_days = 0
    try:
        import akshare as ak
        _raw = ak.stock_zh_a_spot_em()
        if _raw is not None and not _raw.empty:
            df = _raw
    except Exception as e:
        logger.warning(f"实时快照失败 ({e}), 尝试回退缓存")
    if df.empty:
        _cache = Path(__file__).parent.parent / "simulation_data" / "full_market_cache.json"
        if _cache.exists():
            try:
                _d = json.loads(_cache.read_text(encoding="utf-8"))
                df = pd.DataFrame(_d.get("data", []))
                _snap_date = str(_d.get("date", ""))
                _data_source = "cache"
                logger.warning(f"使用全市场缓存快照 {_snap_date} ({len(df)} 只)")
                # v5.4 数据新鲜度 (纸面实盘链路): 缓存滞后自然日 → 供 run_full_day 判定是否降级只读
                try:
                    from datetime import date as _dcls
                    _data_lag_days = (_dcls.today() - date.fromisoformat(_snap_date)).days
                    if _data_lag_days > 3:
                        logger.warning(f"⚠ 数据滞后 {_data_lag_days} 天 (快照 {_snap_date}) — 建议在交易时段/代理可用时运行, "
                                       f"决策基于过期行情, 将降级为只读不真交易")
                except Exception:
                    _data_lag_days = 0
            except Exception as e:
                logger.error(f"缓存快照加载失败: {e}")
    if df.empty:
        raise RuntimeError("全市场数据不可用: 实时快照与缓存均失败")
    df = df.rename(columns={k:v for k,v in col_map.items() if k in df.columns})
    # 数值列兜底转换 (缓存快照字段已是英文名)
    for _c in ("pct_change", "price", "volume", "amount", "turnover", "pe_ttm", "pb", "total_mv"):
        if _c in df.columns:
            df[_c] = pd.to_numeric(df[_c], errors="coerce")
    logger.info(f"全市场: {len(df)} 只")

    # v5.3: 尽力获取上证指数序列用于指数趋势 regime (失败回退单日广度)
    _idx_close = await _load_index_close()
    if _idx_close is not None:
        logger.info(f"指数趋势 regime 启用 (上证收盘 {len(_idx_close)} 日)")

    # 1b. 体制自适应多因子初筛 (v3.1 PreScreener)
    # 替代旧版 ad-hoc 打分, 使用6维度 x 体制自适应权重
    top_n = 300 if use_llm else 100

    try:
        from analysis.pre_screener import PreScreener

        # 检测当前市场体制
        regime_info = _detect_regime(df, index_close=_idx_close)
        regime = regime_info.get("regime", "range_bound")
        # v3.3 市场结构: 抱团动量 → 初筛用强牛权重 + 放宽估值帽 (让龙头进前300), 修复广度误判
        _structure = None
        try:
            from analysis.market_structure import market_structure, screening_regime
            _structure = market_structure(df)
            screen_regime = screening_regime(_structure, regime)
            if screen_regime != regime:
                logger.info(f"市场结构 {_structure} → 初筛 regime 用 {screen_regime} (原 {regime})")
        except Exception as _e:
            logger.warning(f"市场结构识别失败: {_e}")
            screen_regime = regime
        logger.info(f"市场体制: {regime} ({regime_info.get('label', '')}) | 结构→初筛 {screen_regime}")

        screener = PreScreener()
        result = screener.screen(df, regime=screen_regime, top_n=top_n, structure=_structure,
                                 cold_tilt=cold_tilt)
        if cold_tilt:
            logger.info(
                f"冷落模式 (cold_tilt) 启用: 选股池 = 低换手(被冷落) bottom-{top_n} tilt "
                f"— 跑赢指数构造 (5/5 窗口实证, 见 reports/improved_portfolio_ab.md)"
            )
        df_top = result.df
        # v5.11 方案3: 知识选股过滤 (确定性, gate<=0 时零开销原样返回; 与诊断官解耦)
        df_top, _krep = _apply_learned_filter(df_top)
        if _krep.get("removed"):
            logger.info(f"知识选股过滤: {_krep['before']} → {_krep['after']} (剔除 {_krep['removed']} 只)")

        logger.info(
            f"PreScreener: {result.total_in} -> L0:{result.filter_stats['after_hard_filter']} "
            f"-> L1:{result.filter_stats['after_quality_filter']} -> Top{result.total_out}"
        )
        logger.info(
            f"  Score range: [{result.score_distribution['min']:.0f}-{result.score_distribution['max']:.0f}] "
            f" median={result.score_distribution['median']:.0f}"
        )
        logger.info(
            f"  Weights: momentum={result.weights.get('momentum',0):.0%} "
            f"value={result.weights.get('value',0):.0%} "
            f"quality={result.weights.get('quality',0):.0%} "
            f"volatility={result.weights.get('volatility',0):.0%} "
            f"sentiment={result.weights.get('sentiment',0):.0%} "
            f"size={result.weights.get('size',0):.0%}"
        )

    except ImportError as e:
        logger.warning(f"PreScreener not available ({e}), using fallback scoring")
        # Fallback: 旧版 ad-hoc 打分
        df = df.dropna(subset=['code','name','price','pct_change'])
        df = df[df['amount'] > df['amount'].median() * 0.1]
        df = df[df['turnover'] >= 0.1]
        df = df[df['price'] > 2.0]
        df = df[df['pct_change'] > -5.0]
        logger.info(f"L1流动性过滤: {len(df)} 只")

        score = pd.Series(50.0, index=df.index)
        if 'pct_change' in df.columns: score += df['pct_change'].clip(-5, 10) * 3
        if 'pct_60d' in df.columns: score += df['pct_60d'].clip(-20, 50) * 0.5
        if 'turnover' in df.columns: score += df['turnover'].clip(0, 20) * 1.5
        if 'vol_ratio' in df.columns: score += (df['vol_ratio'].clip(0.3, 5) - 1) * 5
        if 'pe_ttm' in df.columns:
            pe_score = 10 - abs(df['pe_ttm'].clip(0, 100) - 20) / 8
            score += pe_score.clip(-10, 10)
        if 'total_mv' in df.columns: score += df['total_mv'].rank(pct=True) * 5
        df['rule_score'] = score.clip(0, 100)
        df_top = df.nlargest(top_n, 'rule_score')

    # v3.3: 市场语境提前 — regime 检测移到初筛前, 让 Flash 初筛也接收市场信息
    try:
        _regime_info = _detect_regime(df, index_close=_idx_close)
        _regime = _regime_info.get("regime", "range_bound")
    except Exception:
        _regime = "range_bound"

    # 1c. DeepSeek Flash 精筛 (替代 Ollama) — v3.3 注入市场语境 (进攻/防御倾向)
    if use_llm:
        try:
            df_top = await _flash_screen(df_top, top_k=100, regime=screen_regime)
            logger.info(f"Flash精筛后: {len(df_top)} 只 (regime={_regime})")
        except Exception as e:
            logger.warning(f"LLM精筛跳过 (Ollama不可用): {e}")

    # 1d. DeepSeek深度分析
    # v3.2: 注入 regime 门控 + 情绪上下文 (动量按体制打折, 估值/防御优先) — 主循环生效
    # v3.3: regime 自适应 thinking — 牛市开深度思考(A/B:+5.83 vs +4.17), 其他关(危机非think更好)
    try:
        regime_ctx = build_market_ctx(df, _regime)
        thinking_mode = regime_uses_thinking(_regime)
        logger.info(f"DeepSeek深度分析 thinking={'开' if thinking_mode else '关'} (regime={_regime})")
    except Exception:
        regime_ctx = None
        thinking_mode = False
    # v3.3 持仓纳入: 持仓不在当日 Top100 也纳入深析 (避免持仓盲区 — 否则持仓
    # 只靠止损/止盈触发, LLM 的 SELL 信号永远看不到它; 与 historical_replay 一致)
    df_deep = df_top.head(100).copy()
    try:
        from simulation.portfolio import PortfolioManager
        _held = set(PortfolioManager().state.positions.keys())
        if _held:
            _held_codes = {_s.split(".")[-1] for _s in _held}
            _top_codes = set(df_deep["code"].astype(str))
            _missing = [c for c in _held_codes if c not in _top_codes]
            if _missing:
                _held_rows = df[df["code"].astype(str).isin(_missing)]
                if not _held_rows.empty:
                    df_deep = pd.concat([df_deep, _held_rows], ignore_index=True)
                    logger.info(f"  持仓纳入深析: {len(_held_rows)} 只不在Top100")
    except Exception as e:
        logger.warning(f"持仓纳入失败(继续): {e}")
    deep_results = await _deepseek_analyze(df_deep, thinking=thinking_mode, regime_ctx=regime_ctx,
                                           macro_context=load_macro_context())
    logger.info(f"DeepSeek分析完成: {len(deep_results)} 只")

    # 1e. 市场诊断官 (v4.0 — 5位大师 + 自我进化)
    # 独立于选股，只输出风险等级和仓位调节建议。诊断模式下用它来调节规则选股。
    market_diag = None
    _journal = None
    _memory = None
    _evolution = None
    today = today_cn().isoformat()

    if enable_evolution:
        try:
            _journal, _memory, _evolution = _init_evolution_system()
            logger.info(f"[进化系统] 已加载: 日志{len(_journal)}条  经验{len(_memory.items)}条  "
                        f"进化周期{len(_evolution.snapshots)}个")
        except Exception as e:
            logger.warning(f"[进化系统] 初始化失败: {e}")

        # 1e-1. 复盘昨日诊断 (PIT正确: 用今日数据回看昨日)
        if _journal is not None:
            _unreviewed = _journal.unreviewed(before=today)
            if _unreviewed:
                # v5.1 PIT 修复: 只复盘最近一条未复盘记录(昨日), 用今日数据作它的次日结果.
                # 多天停市 back-filled 时, 旧记录的"真实次日"是更早的交易日, 不是今天;
                # 用今天一刀切评判几天前的记录是 PIT 违规 (与 historical_replay 的 _unreviewed[-1] 一致).
                _n_old = len(_unreviewed) - 1
                _unreviewed = _unreviewed[-1:]
                logger.info(f"[进化系统] 复盘昨日决策 1 条 (其余{_n_old}条积压待各自真实次日)...")
                # v5.7 进化健康追踪: 每日活动计数 (反事实验证/审计/拖累回流), 收尾写 evolution_health.json
                _cf_verified_count = 0
                _cf_total = 0
                _audit_count = 0
                _drag_count = 0
                _today_stats = _build_market_snapshot(df, _regime)
                # v5.1 市场涨跌用中位数(鲁棒于单只异动股), 避免 cross-sectional mean 被个别涨停/跌停扭曲反事实
                _today_pct = float(df['pct_change'].median()) if 'pct_change' in df.columns else 0.0
                # v5.5 P1-4: 组合级反事实闭环 — 持久化拖累票历史, 反复被验证的票 → 写经验回流
                _pcf_hist_path = Path(__file__).parent.parent / "simulation_data" / "portfolio_cf_history.json"
                _pcf_history = {}
                if _pcf_hist_path.exists():
                    try:
                        _pcf_history = json.loads(_pcf_hist_path.read_text(encoding="utf-8"))
                    except Exception:
                        _pcf_history = {}
                _pcf_verified_today = []
                for _rec in _unreviewed:
                    try:
                        from agent.evolution.daily_review import review_decision, extract_experience
                        _review = await review_decision(_rec, _today_stats, _today_pct)
                        if _review:
                            # v5.4 组合级反事实: 用昨日持仓快照 + 今日个股涨跌, 验证"移除拖累票"改善
                            try:
                                from agent.evolution.portfolio_counterfactual import (
                                    portfolio_level_counterfactual, accumulate_drag_history,
                                )
                                _sret = {}
                                if "pct_change" in df.columns and "symbol" in df.columns:
                                    _sub = df[["symbol", "pct_change"]].dropna(subset=["pct_change"])
                                    _sret = dict(zip(_sub["symbol"], pd.to_numeric(_sub["pct_change"], errors="coerce")))
                                _pcf = portfolio_level_counterfactual(
                                    _rec.positions_snapshot, _sret, date=_rec.date)
                                if _pcf is not None:
                                    _review["portfolio_cf"] = _pcf.to_dict()
                                    # v5.5 闭环: 收集 verified 拖累票 → 累积进历史
                                    if _pcf.verified and _pcf.worst_stock is not None:
                                        _pcf_verified_today.append(_pcf)
                            except Exception as _pce:
                                logger.warning(f"组合级反事实失败: {_pce}")
                            _journal.update_review(_rec.date, _review)
                            _exp = extract_experience(_rec, _review)
                            if _exp and _memory is not None:
                                # v5.2 P1: 第二审计者 — 用实际结果复核 LLM 自报偏差, 打破自证循环
                                _audit_note = "审计N/A"
                                try:
                                    from agent.evolution.counterfactual import audit_review_bias
                                    _audit = audit_review_bias(_review, _today_pct, _rec.risk_level)
                                    _audit_count += 1
                                    if _audit.get("overrides"):
                                        # 审计覆盖 LLM 自报偏差: 换掉 LLM 下的 bias tag, 打上审计偏差
                                        _exp.tags = [t for t in _exp.tags
                                                     if t not in ("conservative", "aggressive")]
                                        if _audit["audit_bias"] in ("conservative", "aggressive"):
                                            _exp.tags.append(_audit["audit_bias"])
                                        _exp.tags.append("audited")
                                        _audit_note = f"审计覆盖→{_audit['audit_bias']}"
                                        logger.warning(f"[审计] {_rec.date} {_audit['note']}")
                                    elif _audit.get("agrees") is True:
                                        _audit_note = f"审计确认{_audit['audit_bias']}"
                                    else:
                                        _audit_note = _audit.get("note", "审计N/A")
                                except Exception:
                                    pass
                                # v4.1: 反事实验证
                                _cf_note = "反事实N/A"
                                try:
                                    from agent.evolution.counterfactual import verify_counterfactual
                                    _cf = verify_counterfactual(_exp, _today_pct)
                                    if _cf is not None:
                                        _cf_total += 1
                                        if _cf.passed:
                                            _cf_verified_count += 1
                                            _exp.confidence = min(0.95, _exp.confidence + 0.15)
                                            _exp.tags.append("cf_verified")
                                            # v5: 反事实通过且高置信 → 升级为 high_confidence, 在 prompt 中优先展示
                                            if _exp.confidence >= 0.85 and "high_confidence" not in _exp.tags:
                                                _exp.tags.append("high_confidence")
                                            _cf_note = f"反事实OK"
                                        else:
                                            _exp.confidence = max(0.2, _exp.confidence - 0.1)
                                            _exp.tags.append("cf_failed")
                                            _cf_note = f"反事实FAIL"
                                except Exception:
                                    pass
                                _memory.add(_exp)
                                _memory.save()
                            logger.info(
                                f"  复盘 {_rec.date}: {_review.get('verdict','?')}  "
                                f"偏差={_review.get('risk_level_deviation',0):+d}  "
                                f"类型={_review.get('error_type','?')}  "
                                f"[{_cf_note}]"
                            )
                    except Exception as _re:
                        logger.debug(f"  复盘 {_rec.date} 失败: {_re}")

                # v5.5 P1-4 闭环: 累积今日 verified 拖累票 → 持久化 → 反复被标记的写经验回流
                if _pcf_verified_today:
                    try:
                        from agent.evolution.portfolio_counterfactual import (
                            accumulate_drag_history, drag_experiences,
                        )
                        _pcf_history = accumulate_drag_history(
                            _pcf_history, _pcf_verified_today, today)
                        # drag_experiences 会更新 last_emitted_date (限频), 故先取条目再持久化,
                        # 让 cooldown 状态跨天生效 (否则每日刷屏回流)
                        _drag_items = drag_experiences(_pcf_history, today)
                        _drag_count = len(_drag_items)
                        _pcf_hist_path.parent.mkdir(parents=True, exist_ok=True)
                        _pcf_hist_path.write_text(
                            json.dumps(_pcf_history, ensure_ascii=False, indent=2),
                            encoding="utf-8")
                        if _drag_items and _memory is not None:
                            for _di in _drag_items:
                                _added = _memory.add(_di)
                            _memory.save()
                            logger.warning(
                                f"[组合反事实闭环] {len(_drag_items)} 只票被反复验证为拖累, 已写入经验回流"
                                f" → LLM 将对其降权/谨慎")
                    except Exception as _pcfe:
                        logger.warning(f"[组合反事实闭环] 失败: {_pcfe}")

                # v5.7 进化健康追踪: 收尾落盘 evolution_health.json (记忆健康 + regime 对齐 + 每日活动)
                try:
                    from agent.evolution.health_tracker import record_health
                    record_health(
                        _memory, today, _regime,
                        path=Path(__file__).parent.parent / "simulation_data" / "evolution_health.json",
                        cf_verified_count=_cf_verified_count,
                        cf_total=_cf_total,
                        audit_count=_audit_count,
                        drag_count=_drag_count,
                    )
                except Exception as _he:
                    logger.debug(f"[进化健康] 快照失败: {_he}")

    # 1e-2. 今日市场诊断
    try:
        _mc = load_macro_context()
        # v4.2: 读取昨日风险等级作为锚点（稳定性约束）
        _prev_risk = None
        if _journal is not None and len(_journal) > 0:
            # 找最近一条记录
            _sorted_dates = sorted(_journal._cache.keys())
            if _sorted_dates:
                _prev_rec = _journal._cache[_sorted_dates[-1]]
                _prev_risk = _prev_rec.risk_level

        # v5.5 P1-5: 实盘诊断喂真实拥挤度 (与回放同一 market_crowding, 回退 cross-sectional 换手)
        _crowd_live = None
        try:
            from analysis.crowding import market_crowding
            _crowd_live = market_crowding(df)
            if _crowd_live.get("signal") not in (None, "cool") or _crowd_live.get("score", 50) >= 40:
                logger.info(f"[拥挤度] {_crowd_live.get('signal')} (score {_crowd_live.get('score',50):.0f}, hot {_crowd_live.get('hot_ratio',0):.2%})")
        except Exception as _cr:
            logger.warning(f"拥挤度计算失败: {_cr}")

        market_diag = await get_market_diagnostic(
            df, _regime,
            crowding=_crowd_live,
            macro_context=_mc,
            memory=_memory,
            evolution=_evolution,
            current_date=today,
            prev_risk_level=_prev_risk,
        )
    except Exception as e:
        logger.warning(f"市场诊断官调用失败: {e}")

    # 1e-3. 写入决策日志 + 触发周期进化
    if enable_evolution and _journal is not None and market_diag is not None:
        try:
            from agent.evolution.decision_journal import DecisionRecord
            _snap = _build_market_snapshot(df, _regime)
            # v5.4 组合级反事实: 记录当日持仓快照 (PIT 正确, 供次日复盘算个股级贡献)
            _pf_snap = []
            _pf_pos_val = 0.0
            try:
                _ih = PortfolioManager().state
                for _sym, _p in _ih.positions.items():
                    _val = float(_p.market_value)
                    _pf_pos_val += _val
                    _pf_snap.append({
                        "symbol": _sym, "name": _p.name,
                        "qty": int(_p.quantity),
                        "price": round(float(_p.current_price or _p.avg_cost), 3),
                        "value": round(_val, 2), "weight": 0.0,
                    })
                _ih_total = float(_ih.cash) + _pf_pos_val
            except Exception:
                _ih_total = _pf_pos_val
            for _s in _pf_snap:
                _s["weight"] = round(_s["value"] / _ih_total, 4) if _ih_total > 0 else 0.0
            _rec = DecisionRecord(
                date=today,
                market_phase=market_diag.get("market_phase", "unknown"),
                dominant_master=market_diag.get("dominant_master", "unknown"),
                secondary_master=market_diag.get("secondary_master", ""),
                risk_level=int(market_diag.get("risk_level", 3)),
                position_multiplier=float(market_diag.get("position_multiplier", 1.0)),
                max_positions_adj=int(market_diag.get("max_positions_adj", 0)),
                key_risks=list(market_diag.get("key_risks", [])),
                diagnosis=str(market_diag.get("diagnosis", "")),
                adversarial_risk_level=market_diag.get("adversarial_risk_level"),
                adversarial_applied=market_diag.get("adversarial_applied"),
                adversarial_divergence=market_diag.get("adversarial_divergence"),
                market_snapshot=_snap,
                regime=_regime,
                crowding_score=_snap.get("crowding_score", 50.0),
                crowding_signal=_snap.get("crowding_signal", "unknown"),
                positions_snapshot=_pf_snap,
                total_value=_ih_total,
            )
            _journal.record(_rec)
            logger.info(f"[进化系统] 决策日志已写入: {_rec.dominant_master} / {_rec.market_phase}")

            # 检查是否触发周期进化总结 (每10条决策一次)
            if _evolution is not None and len(_journal) >= 3:
                _dec_count = len(_journal)
                if _evolution.should_evolve(today, _dec_count):
                    logger.info(f"[进化系统] 触发周期进化总结 (已积累{_dec_count}条决策)...")
                    try:
                        from agent.evolution.daily_review import ExperienceItem
                        _all_decisions = _journal.load_range(
                            (today_cn() - timedelta(days=60)).isoformat(), today
                        )
                        # 收集本周期的经验（从 memory 里取最近的）
                        _recent_items = []
                        if _memory is not None:
                            # 直接从 memory items 取最近 20 条
                            _all_items = list(_memory.items)
                            _recent_items = sorted(_all_items, key=lambda x: x.date, reverse=True)[:20]
                        _snapshot = await _evolution.evolve(_all_decisions, _recent_items)
                        if _snapshot:
                            logger.info(
                                f"[进化系统] 进化完成: {_snapshot.summary.get('period_summary','')[:60]}"
                            )
                            logger.info(
                                f"  偏见: {_snapshot.summary.get('biases_identified', [])}"
                            )
                            logger.info(f"  核心原则: {len(_snapshot.principles)} 条")
                    except Exception as _ee:
                        logger.warning(f"[进化系统] 周期进化失败: {_ee}")
        except Exception as e:
            logger.warning(f"[进化系统] 日志写入失败: {e}")

    # 1f. 保存结果
    today = today_cn().isoformat()
    out = {
        "date": today,
        "total_screened": len(df),
        "rule_filtered_count": len(df_top),
        "deep_analyzed_count": len(deep_results),
        "market_regime": _detect_regime(df, index_close=_idx_close),
        "results": deep_results,
        "market_diagnostic": market_diag,   # v3.5 市场诊断结果
        # v3.5 规则排序后的 Top 股 (诊断模式下直接用, 转 records 便于 JSON 序列化)
        "df_top": df_top.to_dict("records") if hasattr(df_top, "to_dict") else df_top,
        "elapsed_seconds": round(time.time() - t0, 1),
        # v5.5 数据新鲜度: 供 run_full_day 判定是否降级只读 (缓存滞后>3天不真交易)
        "data_source": _data_source,
        "data_lag_days": int(_data_lag_days),
    }
    outpath = REPORT_DIR / f"deep_analysis_top100.json"
    outpath.parent.mkdir(parents=True, exist_ok=True)
    with open(outpath, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    logger.info(f"结果已保存: {outpath}")

    return out


async def _flash_screen(df: pd.DataFrame, top_k: int = 100,
                        blind: bool = False,
                        regime: str = "range_bound",
                        offensive: bool = False) -> pd.DataFrame:
    """DeepSeek V4-Flash 精筛 (替代 Ollama, 更快更强)

    v3.3: 注入市场语境 — 决定初筛选股倾向 (进攻/防御/均衡), 避免 100 只候选永远防御。
      bull  → 优先强势/动量/放量突破 (进攻)
      bear  → 优先低估值/防御 (防守)
      range → 均衡
      offensive=True → 强制进攻 (无视 regime, 用于抱团/窄幅动量牛等广度失真场景)
    """
    from openai import AsyncOpenAI
    import os

    client = AsyncOpenAI(
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
        timeout=60.0,
    )
    model = "deepseek-v4-flash"

    batch_size = 25
    all_scores = {}

    for i in range(0, len(df), batch_size):
        batch = df.iloc[i:i+batch_size]
        lines = []
        for idx, (_, r) in enumerate(batch.iterrows()):
            mv = r.get('total_mv', 0) / 1e8
            if blind:
                lines.append(
                    f"股票{idx+1} price={r['price']:.2f} chg={r['pct_change']:+.1f}% "
                    f"PE={r.get('pe_ttm',0):.0f} PB={r.get('pb',0):.1f} MV={mv:.0f}亿 "
                    f"换手={r.get('turnover',0):.1f}% 量比={r.get('vol_ratio',0):.1f}"
                )
            else:
                lines.append(
                    f"{r['code']} {r['name']} price={r['price']:.2f} chg={r['pct_change']:+.1f}% "
                    f"PE={r.get('pe_ttm',0):.0f} PB={r.get('pb',0):.1f} MV={mv:.0f}亿 "
                    f"换手={r.get('turnover',0):.1f}% 量比={r.get('vol_ratio',0):.1f}"
                )

        # v3.3 市场语境选股指令: 避免初筛永远防御
        if offensive:
            sel_guide = ("当前为抱团/动量市: 强势股/龙头持续走强, 普通股票普跌。"
                         "优先选择动量最强、相对强度最高、放量突破的龙头/强势股; "
                         "回避低估值但下跌的防御股。")
        elif regime in ("strong_bull", "weak_bull"):
            sel_guide = ("当前市场偏强(牛市), 优先选择动量强、相对强度高、放量突破的"
                         "强势股/龙头; 不要过分偏好低估值防御股。")
        elif regime in ("weak_bear", "strong_bear", "crisis"):
            sel_guide = ("当前市场偏弱(熊/危机), 优先选择低估值、高股息、防御性强的股票; "
                         "回避高波动与追高。")
        else:
            sel_guide = "当前市场震荡, 均衡考虑动量与估值, 回避追高。"
        prompt = (
            f"评估以下{len(lines)}只A股的短期潜力(0-100分)。"
            f"{sel_guide}"
            f"考虑: 动量趋势、估值合理性、成交量活跃度、市值规模。"
            f"只返回JSON数组[{{\"code\":\"..\",\"score\":0}}]按score降序:\n" + "\n".join(lines)
        )
        try:
            resp = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0, max_tokens=2000,
                # v3.1 修复: 禁用 thinking — 否则批处理输出全进 reasoning_content,
                # content 为空, JSON 解析失败 → 静默返回空结果 (40天回放 0 交易根因)
                extra_body={"thinking": {"type": "disabled"}},
            )
            content = resp.choices[0].message.content.strip()
            if content.startswith("```"): content = content.split("\n",1)[1].rsplit("```",1)[0]
            items = json.loads(content)
            if isinstance(items, list):
                for item in items:
                    all_scores[item.get("code","")] = item.get("score", 50)
        except Exception as e:
            logger.warning(f"Flash batch#{i}: {e}")

    if all_scores:
        df = df.copy()
        # v3.1.1: LLM JSON 的 score 可能为字符串 ("85"), 统一转 float
        # 否则 combined 列混型, nlargest 排序 float<=str 崩溃
        def _sc(v):
            try:
                return float(v)
            except (TypeError, ValueError):
                return 50.0
        if blind:
            df["llm_score"] = [_sc(all_scores.get(f"股票{idx+1}", 50))
                               for idx in range(len(df))]
        else:
            df["llm_score"] = df["code"].map(lambda c: _sc(all_scores.get(str(c), 50)))
        # v3.1 修复: PreScreener 输出 composite_score (非旧版 rule_score), 兼容两者
        base_score = pd.to_numeric(
            df["rule_score"] if "rule_score" in df.columns else df["composite_score"],
            errors="coerce").fillna(50.0)
        df["combined"] = base_score * 0.3 + df["llm_score"] * 0.7  # Flash权重更高
        df = df.nlargest(top_k, "combined")
    return df


async def _deepseek_analyze(df_top: pd.DataFrame, thinking: bool = False,
                            blind: bool = False,
                            macro_context: Optional[str] = None,
                            variant: str = "baseline",
                            regime_ctx: Optional[str] = None) -> List[Dict]:
    """
    DeepSeek深度分析 Top100

    Args:
        thinking: 是否启用 thinking 模式. 开 = 更深推理但更慢更贵 (max_tokens 16000,
                  批处理 10 只, 让 reasoning_content 与 content 都有空间);
                  关 = 快速 JSON (max_tokens 4000, 批处理 20).
                  初筛 (_flash_screen) 始终禁用 thinking 保持速度.
        blind: 盲测 — 隐藏公司名称/代码, 只给量化数据 (2026 最佳实践: 防 LLM
               用训练记忆里的品牌先验, 如"贵州茅台"知名股天然高分). 结果按
               位置映射回 code/name.
        macro_context: 宏观/政策/国际形势上下文 (来自 macro_context.py 缓存),
                      注入系统提示词, 让选股显式结合宏观环境.
    """
    from openai import AsyncOpenAI
    import os

    client = AsyncOpenAI(
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
        timeout=240.0,  # v3.1: thinking 批次可到 48-110s+, 120s 会超时截断只剩首批
    )

    # v3.1: thinking 开时仍用 batch 20 — 小批次 (10) 会让模型思考更深,
    # reasoning 吃光 16000 预算导致 content 为空. batch 20 反而经济.
    batch_size = 20
    results = []
    # 每批尝试序列: thinking(若请求) -> 非thinking回退 (保证 100% 覆盖)
    attempts = [(True, 16000, None)] if thinking else []
    attempts.append((False, 4000, {"thinking": {"type": "disabled"}}))
    # v3.3 修复: 去掉 min(100) 硬上限 — 持仓纳入后 df_top 为 Top100+持仓(~110),
    # 若只处理前 100, 追加的持仓会被丢弃 (持仓盲区补丁失效). 现在处理全部.
    for batch_i in range(0, len(df_top), batch_size):
        batch = df_top.iloc[batch_i:batch_i+batch_size]
        lines = []
        for idx, (_, r) in enumerate(batch.iterrows()):
            mv = r.get('total_mv',0)/1e8
            board = "主板"
            code_str = str(r['code'])
            if code_str.startswith('3'): board = "创业板"
            elif code_str.startswith('688'): board = "科创板"
            if blind:
                # 盲测: 隐藏名称/代码, 只给量化数据 + 中性编号 (结果按位置映射)
                label = f"股票{idx+1}"
                line = (
                    f"{label} [{board}] price={r['price']:.2f} chg={r['pct_change']:+7.1f}% "
                    f"PE={r.get('pe_ttm',0):.0f} PB={r.get('pb',0):.1f} MV={mv:.0f}亿 "
                    f"换手={r.get('turnover',0):.1f}% 60日涨跌={r.get('pct_60d',0):+.1f}% "
                    f"因子分={r.get('composite_score',r.get('rule_score',50)):.0f}"
                )
            else:
                line = (
                    f"{code_str} {r['name']} [{board}] price={r['price']:.2f} chg={r['pct_change']:+7.1f}% "
                    f"PE={r.get('pe_ttm',0):.0f} PB={r.get('pb',0):.1f} MV={mv:.0f}亿 "
                    f"换手={r.get('turnover',0):.1f}% 60日涨跌={r.get('pct_60d',0):+.1f}% "
                    f"因子分={r.get('composite_score',r.get('rule_score',50)):.0f}"
                )
            if variant == "v32":
                # v3.2 因子: 相对自身20日历史的估值分位 + 风险调整动量 + 反转
                line += (f" 估值ep={r.get('ep',0):.2f} bp={r.get('bp',0):.2f}"
                         f" PE20d分位={r.get('pe_pct_20d',0.5):.0%} PB20d分位={r.get('pb_pct_20d',0.5):.0%}"
                         f" 动Sharpe={r.get('sharpe_20',0):.2f} 昨反={r.get('reversal_1d',0):+.2f}")
            lines.append(line)

        system_prompt = (
            "你是资深A股分析师。对每只股票从技术面、基本面、风险三个维度评估。"
            "评分标准: 85+强烈看多, 70-84偏多, 55-69中性, 40-54偏空, <40看空。"
            "特别注意: 北交所(8/9/4开头)/ST/亏损股给低分; 主板蓝筹/业绩确定给高分。"
            "只返回JSON数组, 不要markdown包裹。"
            + ("只依据提供的量化数据评分, 不要依赖任何对公司身份的先验知识。" if blind else "")
            # v3.1.2: 宏观/政策/国际上下文注入 — 让选股显式结合宏观环境
            + (f"\n【当前宏观背景】\n{macro_context}" if macro_context else "")
            # v3.2: 市场状态/情绪/操作原则 — 按 regime 门控动量、强调相对估值
            + (f"\n【市场状态/操作原则】\n{regime_ctx}" if regime_ctx else "")
        )
        prompt = (
            f"分析以下{len(lines)}只A股,给出最终评分(0-100)和操作(BUY/HOLD/SELL)。"
            f"每只股票需包含: final_score, action, conviction(0-1), technical(技术面简述), fundamental(基本面简述), risk(风险简述)。\n"
            + "\n".join(lines) + "\n"
            "返回JSON:[{\"code\":\"\",\"name\":\"\",\"final_score\":0,\"action\":\"BUY\",\"conviction\":0.5,\"technical\":\"\",\"fundamental\":\"\",\"risk\":\"\"}]"
        )

        # v3.1 修复: attempts 块必须在 for batch_i 循环内 (此前误缩进在循环外,
        # 导致整批只调用 1 次 LLM 返回 20, 而非 100)
        parsed_this_batch = False
        # v3.1.1: 重试+退避 — 网络抖动/限流会整日失败 (实测 days 3-12 top=0),
        # 加 3 次重试 (1/2/4s 退避) 让瞬时故障自动恢复
        import time as _time
        for retry in range(3):
            for t_on, mt, eb in attempts:
                try:
                    resp = await client.chat.completions.create(
                        model=os.getenv("DEEPSEEK_FLASH_MODEL", "deepseek-v4-flash"),
                        messages=[{"role":"system","content": system_prompt}, {"role":"user","content": prompt}],
                        temperature=0.3, max_tokens=mt, extra_body=eb,
                    )
                    content = (resp.choices[0].message.content or "").strip()
                    if content.startswith("```"): content = content.split("\n",1)[1].rsplit("```",1)[0]
                    items = json.loads(content)
                    if isinstance(items, list) and items:
                        for idx, item in enumerate(items):
                            if not item.get("code") and idx < len(batch):
                                item["code"] = str(batch.iloc[idx]["code"])
                                item["name"] = str(batch.iloc[idx]["name"])
                        results.extend(items)
                        logger.debug(f"DeepSeek batch#{batch_i} thinking={t_on}: {len(items)} analyzed")
                        parsed_this_batch = True
                        break  # 该批成功, 跳出 attempts
                except Exception as e:
                    logger.warning(f"DeepSeek batch#{batch_i} thinking={t_on} retry={retry}: {e}")
                    continue  # 尝试下一个 (非thinking回退)
            if parsed_this_batch:
                break  # 成功, 跳出重试
            if retry < 2:
                _time.sleep(2 ** retry)  # 退避 1/2s
        if not parsed_this_batch:
            logger.warning(f"DeepSeek batch#{batch_i}: 所有重试失败, 跳过")

    # v3.1.1: LLM JSON 数值可能为字符串, 排序前安全转 float
    def _safe_score(x):
        try:
            return float(x.get("final_score", x.get("score", 0)))
        except (TypeError, ValueError):
            return 0.0
    results.sort(key=_safe_score, reverse=True)
    return results


_REGIME_LABELS = {
    "crisis": "危机", "strong_bear": "强熊", "weak_bear": "弱熊",
    "range_bound": "震荡", "weak_bull": "弱牛", "strong_bull": "强牛",
}


async def _load_index_close() -> Optional[pd.Series]:
    """尽力获取上证指数 PIT 收盘序列 (用于指数趋势 regime).

    失败/无数据返回 None → _detect_regime 回退单日广度 (不回归).
    """
    try:
        from data.providers.base import DataFrequency, DataRequest
        from data import get_data_router
        router = get_data_router()
        req = DataRequest(
            "sh.000001",
            date.today() - timedelta(days=90),
            date.today(), DataFrequency.DAILY, adjust="qfq",
        )
        res = await asyncio.wait_for(router.get_daily_kline(req), timeout=20)
        df = res.data
        if df is None or df.empty or "close" not in df.columns:
            return None
        close = pd.to_numeric(df["close"], errors="coerce").dropna()
        if len(close) < 26:
            return None
        if "date" in df.columns:
            _d = pd.to_datetime(df["date"])
            close = close.copy()
            close.index = _d
        return close.sort_index()
    except Exception as e:
        logger.debug(f"指数序列获取失败 (regime 回退广度): {e}")
        return None


def _index_benchmark_ret(index_close, analysis_date: str, review_date: str) -> float:
    """同期基准收益 (v5.5, P0-2): 上证指数从 analysis_date 到 review_date 的累计涨跌幅.

    取两个日期各自最后一个 ≤ 该日的收盘计算。解决 decision_log 的 benchmark 恒为 0
    (alpha 被度量成"是否正收益"而非"是否跑赢大盘")。无数据/异常 → 回退 0.0 (不回归旧行为).
    """
    if index_close is None or len(index_close) == 0:
        return 0.0
    try:
        s = index_close
        if not isinstance(s.index, pd.DatetimeIndex):
            return 0.0
        a_series = s[s.index <= pd.Timestamp(analysis_date)]
        r_series = s[s.index <= pd.Timestamp(review_date)]
        if a_series.empty or r_series.empty:
            return 0.0
        a_close = float(pd.to_numeric(a_series.iloc[-1], errors="coerce"))
        r_close = float(pd.to_numeric(r_series.iloc[-1], errors="coerce"))
        if not a_close or a_close <= 0:
            return 0.0
        return (r_close / a_close) - 1.0
    except Exception:
        return 0.0


def _detect_regime(df: pd.DataFrame, index_close: Optional[pd.Series] = None) -> Dict:
    """市场状态检测.

    优先用指数趋势 (上证/沪深300 收盘序列, PIT: 只含 ≤T 的收盘) 判断体制:
      单日截面均值在抱团牛/普跌期会逐日跳变 (leader涨但平均跌 → 误判熊),
      且无趋势成分。指数动量 (25d) + 极端快速回调覆盖 更稳:
        - 牛市保持多头 (不误判), 回调日只降档不翻熊
        - 崩盘早期 (指数5d/截面恐慌) 快速转弱
    无指数数据时回退单日广度 (旧逻辑, 不回归)。

    Args:
        df: 全市场截面 (pct_change 列)
        index_close: 截至今日的上证指数收盘序列 (PIT, 升序). 可选.
    """
    up_pct = (df['pct_change'] > 0).mean() if 'pct_change' in df.columns else 0.5
    avg_pct = df['pct_change'].mean() if 'pct_change' in df.columns else 0
    extreme_down = (df['pct_change'] < -5).mean() if 'pct_change' in df.columns else 0
    extreme_up = (df['pct_change'] > 5).mean() if 'pct_change' in df.columns else 0

    # ── 指数趋势优先 (v5.3: 修复 regime 检测器逐日跳变/抱团牛误判熊) ──
    if index_close is not None and len(index_close) >= 26:
        ic = pd.to_numeric(index_close, errors="coerce").dropna()
        if len(ic) >= 26:
            m5 = (ic.iloc[-1] / ic.iloc[-6] - 1) * 100 if len(ic) >= 6 else 0.0
            m25 = (ic.iloc[-1] / ic.iloc[-26] - 1) * 100
            # 极端覆盖: 截面恐慌 OR 指数5d崩
            if extreme_down > 0.1 or m5 < -6:
                regime = "crisis"
            elif avg_pct < -1.5 and m25 < -1:
                regime = "strong_bear"
            elif m25 < -2:
                regime = "strong_bear" if m5 < -1 else "weak_bear"
            elif m25 < 1.5:
                regime = "weak_bear" if m5 < 0 else "range_bound"
            elif m25 < 4:
                regime = "range_bound" if m5 < 0.5 else "weak_bull"
            elif m25 < 6:
                regime = "weak_bull" if m5 < 1 else "strong_bull"
            else:
                regime = "strong_bull"
            return {
                "regime": regime, "label": _REGIME_LABELS.get(regime, regime),
                "avg_pct": round(float(avg_pct), 2), "up_ratio": round(float(up_pct), 2),
                "index_m5": round(float(m5), 2), "index_m25": round(float(m25), 2),
            }

    # ── 回退: 单日广度 (旧版) ──
    if extreme_down > 0.1: regime = "crisis"
    elif avg_pct < -1.5: regime = "strong_bear"
    elif avg_pct < -0.3: regime = "weak_bear"
    elif avg_pct < 0.3: regime = "range_bound"
    elif avg_pct < 1.5 and extreme_up < 0.05: regime = "weak_bull"
    elif extreme_up > 0.05: regime = "strong_bull"
    else: regime = "strong_bull"

    return {"regime": regime, "label": _REGIME_LABELS.get(regime, regime),
            "avg_pct": round(float(avg_pct), 2), "up_ratio": round(float(up_pct), 2)}


def build_market_ctx(df: pd.DataFrame, regime: str) -> str:
    """从当日截面构建"市场状态/情绪/操作原则"上下文 (v3.2, 注入 _deepseek_analyze)。

    让 AI 判断按 regime 门控: 牛市信动量, 熊市/震荡把动量打折、优先相对低估+防御。
    无需历史数据 (单日快照即可算), 可直接用于 daily_runner 主循环。
    """
    n = max(len(df), 1)
    up = float((df["pct_change"] > 0).mean()) if "pct_change" in df.columns else 0.5
    avg = float(df["pct_change"].mean()) if "pct_change" in df.columns else 0.0
    lu = int((df["pct_change"] >= 9.5).sum()) if "pct_change" in df.columns else 0
    ld = int((df["pct_change"] <= -9.5).sum()) if "pct_change" in df.columns else 0
    sent = float(np.clip(
        up * 100 * 0.5 + np.clip(avg / 3, -1, 1) * 50 * 0.3
        + (lu / n) * 200 * 0.2 - (ld / n) * 200 * 0.2, 0, 100))
    label = ("极度恐慌" if sent <= 20 else "恐慌" if sent <= 40 else
             "中性" if sent <= 60 else "贪婪" if sent <= 80 else "极度贪婪")
    if regime in ("strong_bull", "weak_bull"):
        gate = "牛市: 动量/趋势信号可信, 强势优先, 估值次之; 可积极参与。"
    elif regime == "range_bound":
        gate = "震荡市: 动量信号打折, 低估值+超跌反弹优先, 追高谨慎。"
    else:
        gate = "熊市/危机: 动量按反向信号处理(勿追涨), 低估值(相对历史PE/PB)与防御优先, 严控仓位。"
    return (f"今日市场状态: {regime}; 情绪温度 {sent:.0f}/100 ({label}); "
            f"上涨广度 {up:.0%}, 平均涨跌 {avg:+.2f}%, 涨停 {lu}家/跌停 {ld}家。"
            f"操作原则: {gate}")


def regime_uses_thinking(regime: str) -> bool:
    """按 regime 决定深度思考 (v3.3): 多 regime A/B 显示 thinking 牛市好
    (+5.83 vs +4.17, 胜率80%)、危机差 (-5.19 vs -6.57/-5.80) → 牛市开, 其他关。
    这是有 A/B 实证的改动, 保留。regime_take_profit_mult 已回退移除 (A/B 证明净负)。"""
    return regime in ("strong_bull", "weak_bull")


# ═══════════════════════════════════════════════════════════════
# Phase 2: 执行交易
# ═══════════════════════════════════════════════════════════════

def _tencent_quote(symbol: str) -> Tuple[float, Optional[float]]:
    """从腾讯实时行情获取 (现价, 涨跌幅%). 字段3=现价, 字段4=昨收。

    v3.0: 统一行情入口, 供买入/卖出/MTM 复用并计算涨跌停封板标志。
    """
    import requests
    try:
        tc = symbol.replace("sh.", "sh").replace("sz.", "sz").replace("bj.", "bj")
        resp = requests.get(f"https://qt.gtimg.cn/q={tc}", timeout=5,
            headers={"User-Agent": "Mozilla/5.0", "Referer": "https://finance.qq.com/"})
        resp.encoding = "gbk"
        for line in resp.text.split("\n"):
            if "=" in line and "~" in line:
                fields = line.split("=", 1)[1].strip('"').split("~")
                if len(fields) > 4:
                    price = float(fields[3]) if fields[3] else 0.0
                    prev_close = float(fields[4]) if fields[4] else 0.0
                    pct = (price / prev_close - 1) * 100 if prev_close else None
                    return price, pct
    except Exception:
        pass
    return 0.0, None


def _apply_risk_gate(diag: dict, limits: dict) -> dict:
    """v3.5 风险硬性闸门 — 大师风险读数 → 强制约束仓位 (只收不放).

    risk_level >=4 时强制压缩持仓上限/单票, 驱动收敛逻辑真正减仓.
    单向收紧: 只取更严, 不覆写已算好的扩张型数值.
    """
    try:
        _rl = int(diag.get("risk_level", 3))
    except (TypeError, ValueError):
        _rl = 3
    if _rl >= 4:
        _cap = 4 if _rl == 4 else 2
        limits["max_positions"] = min(limits["max_positions"], _cap)
        limits["single_pct"] = min(limits["single_pct"], 0.10 if _rl == 4 else 0.06)
        logger.warning(
            f"[风险硬闸门] risk_level={_rl}/5 → 强制 最多{limits['max_positions']}只/单只{limits['single_pct']:.0%}"
        )
    return limits


def _apply_adversarial_gate(diag: dict) -> dict:
    """v5.2 对抗票 — 主导大师 vs 对抗大师 分歧裁决 (打破"假多元化").

    同一次LLM调用产出两个独立视角: 主导大师的 risk_level 与 secondary 对抗大师的
    adversarial_risk_level. 当分歧度>=2级时执行"危机→防守"不对称:
    - 对抗更保守(数字更大) → 主导风险+1级, 仓位向1.0收敛 (宁可错过, 不可满仓)
    - 对抗更激进(数字更小) → 不追激进, 仓位向1.0收敛降置信 (分歧=不确定)
    分歧<2级不干预 (历史冗余度0.28~0.36说明多数情况大师一致, 无需扰动).
    只收紧/降置信, 不放大 — 单向防守, 低误伤.
    """
    try:
        _dom = int(diag.get("risk_level", 3))
        _adv = int(diag.get("adversarial_risk_level", _dom))
    except (TypeError, ValueError):
        return diag
    _d = abs(_dom - _adv)
    if _d < 2:
        return diag  # 分歧小, 不干预

    _orig_mult = float(diag.get("position_multiplier", 1.0))
    if _adv > _dom:
        # 对抗更保守 → 主导风险+1级 (朝保守收)
        diag["risk_level"] = min(5, _dom + 1)
        diag["adversarial_applied"] = "conservative"
    else:
        # 对抗更激进 → 不追, 只降置信
        diag["adversarial_applied"] = "dampen"
    # 仓位向1.0收敛一半 (分歧=不确定, 减半偏离)
    diag["position_multiplier"] = 1.0 + (_orig_mult - 1.0) * 0.5
    diag["adversarial_divergence"] = _d
    diag["adversarial_risk_level"] = _adv
    logger.warning(
        f"[对抗票] 主导{_dom} vs 对抗{_adv} 分歧{_d}级 → "
        f"{diag['adversarial_applied']}, 风险{diag['risk_level']} 仓位×{diag['position_multiplier']:.2f}"
    )
    return diag


def _phase_conviction_mult(phase: str) -> float:
    """v5 market_phase 信念乘数 — 大师判定的市场阶段读写入过滤买入.

    bubble_late/trend_down 收紧, trend_up 放开. 与 regime 确信度乘数相乘.
    """
    return {
        "trend_up": 1.0, "range": 0.8, "bubble_late": 0.5,
        "panic_bottom": 0.7, "turning": 0.7, "trend_down": 0.4,
    }.get(phase, 0.8)


async def _below_ma20(sym: str, data_router) -> Optional[Tuple[float, float]]:
    """现价 < 自身MA20 则返回 (现价, MA20), 否则 None (个股走弱检测).

    防御减仓与目标仓位收敛共用. 数据不足/异常时返回 None (保守不卖).
    """
    from data.providers.base import DataFrequency, DataRequest
    try:
        req = DataRequest(sym, today_cn() - timedelta(days=60), today_cn(), DataFrequency.DAILY)
        r = await data_router.get_daily_kline(req)
        df = r.data
        if df is None or df.empty or "close" not in df.columns or len(df) < 25:
            return None
        cl = pd.to_numeric(df["close"], errors="coerce").dropna()
        px = float(cl.iloc[-1])
        ma20 = float(cl.rolling(20).mean().iloc[-1])
        return (px, ma20) if px < ma20 else None
    except Exception:
        return None


def _limit_pct_for_code(code: str) -> float:
    """按代码返回板块涨跌幅限制(%) — 主板±10%, 创业板/科创板±20%, 北交所±30%"""
    if code.startswith("30") or code.startswith("68"):
        return 20.0
    if code.startswith("8") or code.startswith("4") or code.startswith("920"):
        return 30.0
    return 10.0


def _tencent_prefix(code: str) -> str:
    """v5.5 P0-3: 板块 → 腾讯行情前缀. 北交所正确判据是 8/4/920 开头 (原 startswith('9') 取错行情)."""
    if code.startswith("6"):
        return "sh"
    if code.startswith("8") or code.startswith("4") or code.startswith("920"):
        return "bj"
    return "sz"


def _effective_dry_run_for_lag(analysis, base_dry_run: bool, skip_analyze: bool, _logger=None) -> bool:
    """v5.5 数据脱节拦截: 缓存滞后>3天 → 强制 dry-run (只读, 不真交易).

    防"Phase1 用过期缓存定仓位, Phase2 用实时价成交"的脱节。
    数据新鲜(lag<=3天)或已显式 --dry-run 时, 维持原值。
    """
    if base_dry_run:
        return True
    if skip_analyze or not isinstance(analysis, dict):
        return False
    lag = int(analysis.get("data_lag_days", 0) or 0)
    if lag > 3:
        if _logger is not None:
            _logger.warning(
                f"⚠ 数据滞后 {lag} 天 → Phase2 降级为 dry-run (只分析不交易), "
                f"避免基于过期行情真交易。数据新鲜时(交易时段/代理可用)自动恢复正常交易。"
            )
        return True
    return False


async def phase2_execute(dry_run: bool = False, diagnostic_mode: bool = False) -> Dict[str, Any]:
    """基于分析结果执行交易

    Args:
        dry_run: 仅分析不交易
        diagnostic_mode: v3.5 诊断模式 — LLM 不直接选股,只做市场风险诊断.
            选股交给规则系统 (PreScreener rule_score 排序), LLM 只调节仓位系数和持仓上限.
            理念: LLM 是监督者不是交易员.
    """
    logger.info("=" * 50)
    logger.info(f"Phase 2: 执行交易 {'(诊断模式)' if diagnostic_mode else ''}")
    logger.info("=" * 50)

    from simulation.portfolio import PortfolioManager
    from simulation.paper_trader import PaperTradingEngine
    import requests

    manager = PortfolioManager()
    engine = PaperTradingEngine(manager)
    state = engine.state
    today = today_cn().isoformat()

    # 加载分析结果
    analysis_path = REPORT_DIR / "deep_analysis_top100.json"
    if not analysis_path.exists():
        logger.error("无分析结果! 请先运行 Phase 1")
        return {"status": "no_analysis"}

    with open(analysis_path, "r", encoding="utf-8") as f:
        analysis = json.load(f)
    results = analysis.get("results", [])

    # ── v3.5 诊断模式: 用规则排序替代 LLM 选股 ──
    if diagnostic_mode:
        # 规则 Top 股全部作为 BUY 候选, 按 rule_score 降序
        df_top = analysis.get("df_top", [])
        if isinstance(df_top, list):
            buys = df_top[:50]  # 取前50只作为候选
        else:
            # 从 records 读 (df.to_dict('records') 格式)
            buys = []
        # 统一格式: code/name/final_score/action/conviction
        _buys = []
        for r in buys:
            _buys.append({
                "code": str(r.get("code", "")),
                "name": r.get("name", ""),
                "final_score": float(r.get("rule_score", r.get("composite_score", 50))),
                "action": "BUY",
                "conviction": min(0.9, 0.5 + float(r.get("rule_score", r.get("composite_score", 50))) / 200),
                "technical": "规则选股",
                "fundamental": "",
                "risk": "",
            })
        buys = _buys
        sells = []  # 诊断模式下不卖 (卖出靠止损/止盈 + 策略退出信号)
        logger.info(f"诊断模式: 规则Top{len(buys)}只 → 全部BUY候选 (LLM只调节仓位)")
    else:
        buys = [r for r in results if r.get("action") == "BUY"]
        sells = [r for r in results if r.get("action") == "SELL"]
        logger.info(f"分析结果: {len(results)}只 → BUY {len(buys)} | SELL {len(sells)}")

    # 加载市场状态
    regime_info = analysis.get("market_regime", {})
    regime = regime_info.get("regime", "range_bound")
    limits = dict(REGIME_LIMITS.get(regime, REGIME_LIMITS["range_bound"]))  # copy 一份

    # ── v3.5 市场诊断 → 仓位调节 (默认路径也生效, 不再依赖 diagnostic_mode) ──
    # 之前这段被 `if diagnostic_mode:` 门控, 而 run_full_day 从不传该 flag → 死代码.
    # 现在只要 market_diagnostic 存在就应用到仓位尺寸 (选股仍由 LLM, 见上方分支).
    _diag = analysis.get("market_diagnostic") or {}
    if _diag and os.getenv("DIAG_SIZE", "1") == "1":
        _pos_mult = float(_diag.get("position_multiplier", 1.0))
        _max_adj = int(_diag.get("max_positions_adj", 0))
        # 调节单只仓位比例
        limits["single_pct"] = max(0.02, min(0.25, limits["single_pct"] * _pos_mult))
        # 调节持仓上限
        limits["max_positions"] = max(1, min(30, limits["max_positions"] + _max_adj))
        limits["label"] = f"{limits.get('label', regime)}+diag(L{_diag.get('risk_level',3)}/5×{_pos_mult:.2f})"
        logger.info(
            f"[诊断调节] 风险等级={_diag.get('risk_level',3)}/5  "
            f"仓位×{_pos_mult:.2f}  持仓{_max_adj:+d}  "
            f"→ 单只{limits['single_pct']:.1%} / 最多{limits['max_positions']}只"
        )
        for r in _diag.get("key_risks", [])[:3]:
            logger.info(f"  ⚠ {r}")

    # ── v3.5 风险硬性闸门: 大师风险读数 → 强制约束仓位 (只收不放) ──
    # risk_level 此前仅作标签. 现在直接作为硬闸门: >=4 强制压缩持仓上限/单票,
    # 与下方 timing risk_off 叠加时取更严. 驱动已实现的收敛逻辑真正减仓.
    if _diag:
        _apply_risk_gate(_diag, limits)

    # v3.3 市场择时 Overlay (双动量): 指数<MA20 时强制防御 (少仓/现金) — 避开熊市
    # 回测: 指数>MA20 持市场 +34.8%/回撤2.6% vs 买入持有 -5.0%/回撤27.2%
    _timing_sig = None
    try:
        from analysis.market_timing import format_timing, market_ma_signal
        _timing_sig = await market_ma_signal()
        if _timing_sig:
            logger.info(f"市场择时: {format_timing(_timing_sig)}")
            if _timing_sig["signal"] == "risk_off":
                logger.warning(
                    f"市场择时 risk_off (上证{_timing_sig['index_close']} < MA{_timing_sig['window']} "
                    f"{_timing_sig['ma']}) → 强制防御仓位 (最大2只/单只5%)")
                limits = {"max_positions": 2, "single_pct": 0.05, "label": f"{regime}+risk_off"}
    except Exception as e:
        logger.warning(f"市场择时信号失败(保持regime上限): {e}")
    logger.info(f"市场: {regime} → {limits['label']} | 最大{limits['max_positions']}只 | 单只{limits['single_pct']:.0%}")

    if dry_run:
        logger.info("[DRY RUN] 仅分析, 不执行交易")
        return {"status": "dry_run", "buys": len(buys), "sells": len(sells)}

    # ── v3.5 指数择时进攻化: risk_on 买入 sh.000001 指数书吃满牛市, risk_off 清仓回现金.
    # 受 OFFENSIVE_INDEX_PCT 门控 (env, 0=关; 如 0.5=risk_on 时半仓持指数). 这是唯一
    # 有回测证据 (+22%~+35%) 跑赢普涨牛的机制, 由择时层驱动而非选股层.
    _offensive_index_pct = float(os.getenv("OFFENSIVE_INDEX_PCT", "0.0") or 0.0)
    if _offensive_index_pct > 0 and _timing_sig:
        try:
            _idx_lv = float(_timing_sig.get("index_close", 0) or 0)
            if _timing_sig["signal"] == "risk_on" and state.index_book_units <= 0 and _idx_lv > 0:
                _amt = state.cash * _offensive_index_pct
                if state.buy_index_book(_amt, _idx_lv, today):
                    logger.info(f"指数择时进攻: risk_on 买入指数书 ¥{_amt:,.0f} @上证{_idx_lv:.0f}")
            elif _timing_sig["signal"] == "risk_off" and state.index_book_units > 0:
                _proc = state.sell_index_book(_idx_lv, today)
                logger.info(f"指数择时进攻: risk_off 清仓指数书 ¥{_proc:,.0f}")
        except Exception as e:
            logger.warning(f"指数择时进攻失败: {e}")

    # ── v3.0 风控接线: 组合回撤断路器/危机模式 (此前从未在真实交易路径生效) ──
    risk_state = None
    halt_buys = False
    risk_mult = 1.0
    try:
        from analysis.risk_controls import PortfolioRiskManager
        risk_mgr = PortfolioRiskManager(initial_capital=state.initial_capital)
        snap_returns = [s.daily_return_pct for s in state.daily_snapshots if s.daily_return_pct]
        daily_returns = pd.Series(snap_returns, dtype=float)
        risk_state = risk_mgr.update(
            current_capital=state.total_value,
            daily_returns=daily_returns,
            market_breadth=float(regime_info.get("up_ratio", 0.5)),
        )
        for w in risk_state.warning_messages:
            logger.warning(f"[风控] {w}")
        if risk_state.circuit_breaker_active or risk_state.crisis_mode:
            halt_buys = True
            logger.warning(
                f"[风控] 回撤断路器/危机模式激活 (DD={risk_state.drawdown_pct:.1f}%), "
                f"暂停全部新买入, 仅执行卖出/减仓"
            )
        else:
            risk_mult = risk_state.risk_multiplier
    except Exception as e:
        logger.warning(f"[风控] 风控检查异常, 跳过: {e}")

    # 2a. 卖出SELL信号的持仓
    sell_codes = {r.get("code","") for r in sells}
    sold = []
    for sym, pos in list(state.positions.items()):
        code_short = sym.replace("sh.","").replace("sz.","").replace("bj.","")
        if code_short in sell_codes:
            # v3.0: 传入涨跌幅, 跌停封板时拒绝卖出
            _px, _pct = _tencent_quote(sym)
            # v5.6 P0-5: SELL 侧校验 t1_pending (当日买入不可卖) — 与 execute_sell
            # 硬阻断互补, 落 journal 可审计 (warning 级, 不阻断主流程)
            try:
                from agent.sub_agents.validator import DecisionValidator
                _sv = DecisionValidator()
                await _sv.run(_sv._start_context(
                    task_id=f"daily_runner_sell_{today}",
                    trading_params={},
                    stock_recommendations={sym: {"action": "SELL"}},
                    market_data={},
                    portfolio={"positions": {
                        sym: {"buy_date": getattr(pos, "buy_date", "")}
                    }},
                ))
            except Exception:
                pass
            trade = engine.execute_sell(
                symbol=sym, exit_reason="AI卖出信号",
                pct_change=_pct,
            )
            if trade:
                sold.append({"symbol": sym, "name": pos.name, "price": trade.price})
                logger.info(f"  SELL {pos.name}({sym}) @{trade.price:.2f} pnl={trade.pnl:+.2f}")

    if sold:
        manager.save()
        logger.info(f"已卖出 {len(sold)} 只")

    # ── v5.6 回撤断路器真实减仓 (此前仅 halt_buys, 从不减存量) ──
    # 触发 -8% → 每仓减 50%, -15%/危机 → 清仓。T+1 当日买入与跌停封板由 execute_sell 拒绝。
    if risk_state is not None and (risk_state.circuit_breaker_active or risk_state.crisis_mode):
        _target_mult = risk_state.risk_multiplier
        _liquidate = _target_mult <= 0.05
        _reduce = (not _liquidate) and _target_mult <= 0.5
        _forced = []
        for sym, pos in list(state.positions.items()):
            if pos.quantity <= 0:
                continue
            _px, _pct = _tencent_quote(sym)
            if _liquidate:
                qty, reason = None, "回撤断路器清仓"
            elif _reduce:
                qty, reason = max(100, (pos.quantity // 100 // 2) * 100), "回撤断路器减仓50%"
            else:
                continue
            trade = engine.execute_sell(symbol=sym, quantity=qty, exit_reason=reason, pct_change=_pct)
            if trade:
                _forced.append({"symbol": sym, "name": pos.name, "qty": trade.quantity, "price": trade.price})
                logger.info(f"  [风控] {reason} {pos.name}({sym}) x{trade.quantity} @{trade.price:.2f} pnl={trade.pnl:+.2f}")
        if _forced:
            manager.save()
            logger.warning(
                f"[风控] 回撤断路器已减仓 {len(_forced)} 只持仓 "
                f"(DD={risk_state.drawdown_pct:.1f}%, mult={_target_mult:.2f})"
            )

    # 2b. 买入Top BUY信号
    executed = []
    for rec in buys:
        if halt_buys:
            logger.info("[风控] 回撤断路器/危机模式激活, 停止买入")
            break
        if len(state.positions) >= limits["max_positions"]:
            logger.info(f"已达最大持仓 {limits['max_positions']} 只,停止买入")
            break

        code = rec.get("code", "")
        name = rec.get("name", "")
        score = rec.get("final_score", rec.get("score", 0))
        conv = rec.get("conviction", 0.5)
        technical = rec.get("technical", "")
        fundamental = rec.get("fundamental", "")
        risk = rec.get("risk", "")

        if not code:
            continue

        # 获取实时价格 + 涨跌幅 (v3.0: 一次请求同时取现价与昨收, 用于封板/缺口判断)
        # v3.1-deerflow: 修正北交所前缀 — bj 代码以 8/4 开头 (旧逻辑 startswith('9') 有误)
        prefix = "sh" if code.startswith("6") else ("bj" if code.startswith(("8", "4")) else "sz")
        sym_full = f"{prefix}.{code}"
        price, pct_change = _tencent_quote(sym_full)

        if price <= 0:
            continue

        # ── 体制 + 市场阶段修正确信度 ──
        # v5: 新增 market_phase 信念乘数 — 大师判定的市场阶段读数参与过滤买入
        # (bubble_late/trend_down 收紧, trend_up 放开). 与 regime 乘数相乘.
        regime_conv_mult = {
            "strong_bull": 1.0, "weak_bull": 0.9,
            "range_bound": 0.6, "weak_bear": 0.4,
            "strong_bear": 0.2, "crisis": 0.0,
        }
        _phase_mult = _phase_conviction_mult(_diag.get("market_phase", "unknown"))
        original_conv = conv
        conv = original_conv * regime_conv_mult.get(regime, 0.6) * _phase_mult
        if conv < 0.3:
            logger.info(f"  跳过 {name}({code}): 体制修正后确信度{conv:.0%} < 30% "
                        f"(原始{original_conv:.0%} × {regime_conv_mult.get(regime,0.6)} × phase{_phase_mult:.2f})")
            continue

        # ── 开盘确认 (防涨停次日闷杀/追高) — v3.0 复用已取行情, 无需二次请求 ──
        if pct_change is not None:
            # 低开超 3% 跳过 (涨停次日闷杀典型模式)
            if pct_change < -3:
                logger.info(f"  跳过 {name}({code}): 低开{pct_change:.1f}%, 疑似涨停次日闷杀")
                continue
            # 震荡市/弱市高开超 5% 也跳过 (追高风险)
            if pct_change > 5 and regime in ("range_bound", "weak_bear", "strong_bear"):
                logger.info(f"  跳过 {name}({code}): {regime}高开{pct_change:.1f}%, 追高风险")
                continue

        # 动态止损止盈 (v3.3 回退: 止盈A/B证明 regime 自适应净负 — 回到固定 +12%
        # v3.3 ATR近似止损: 高波动板块 (创业板/科创板20%限) 正常波动大, 止损放宽到2倍,
        # 主板(10%)保持基础 — 避免高波动股被正常噪声误杀)
        if score >= 80: sl_pct, tp_pct = 0.05, 0.12
        elif score >= 60: sl_pct, tp_pct = 0.07, 0.10
        else: sl_pct, tp_pct = 0.10, 0.08

        # v3.0: 板块感知的涨停封板判定, 传入 execute_buy 使其真正生效
        limit_pct = _limit_pct_for_code(code)
        sealed_limit_up = bool(pct_change is not None and pct_change >= limit_pct - 0.2)
        sl_pct = min(0.15, sl_pct * (limit_pct / 10.0))  # 高波动板块止损×limit/10

        enhanced = {
            "conviction": conv, "score": score/10, "composite_score": score,
            "key_reasons": [technical, fundamental], "risks": [risk],
            "verdict_summary": f"DeepSeek: {rec.get('action','')}({conv:.0%}) s={score}",
            "stop_loss": round(price*(1-sl_pct),2),
            "take_profit": round(price*(1+tp_pct),2),
        }

        # ── v3.1-deerflow: DecisionValidator 执行前硬约束校验 ──
        # 用与 execute_buy 相同的仓位/止损止盈参数校验, 拒绝则跳过并落 journal
        # v5.6 P0-5: 补传 market_data + portfolio, 让 limit_unbuyable/lot_too_small
        # 等硬约束真正生效 (此前 market_data={} 且无 portfolio → 只剩参数校验)。
        try:
            from agent.sub_agents.validator import DecisionValidator
            _pos_pct = min(limits["single_pct"] * risk_mult, 0.20)
            _vd = DecisionValidator()
            # 构造 2 行收盘价 (昨收=price/(1+pct), 今收=price), 供涨跌停可行性校验
            _prev_close = price / (1 + pct_change / 100) if pct_change not in (None, -100.0) else price
            _market_df = pd.DataFrame({"close": [round(_prev_close, 3), round(price, 3)]})
            _portfolio = {
                "capital": state.total_value,
                "positions": {
                    s: {"quantity": p.quantity, "buy_date": getattr(p, "buy_date", "")}
                    for s, p in state.positions.items()
                },
            }
            _v_res = await _vd.run(_vd._start_context(
                task_id=f"daily_runner_{today}",
                trading_params={sym_full: {
                    "entry_price": price,
                    "stop_loss": enhanced["stop_loss"],
                    "take_profit": enhanced["take_profit"],
                    "position_pct": round(_pos_pct, 3),
                }},
                stock_recommendations={sym_full: {"action": "BUY", "conviction": conv}},
                market_data={sym_full: _market_df},
                portfolio=_portfolio,
            ))
            _vrec = (
                _v_res.data.get("validation_results", {}).get(sym_full, {})
                if _v_res.success else {"valid": True}
            )
            if not _vrec.get("valid", True):
                _codes = [v["code"] for v in _vrec.get("violations", [])]
                logger.warning(f"  [验证器] 跳过 {name}({code}): {_codes}")
                continue
        except Exception as _ve:
            logger.debug(f"[验证器] 校验跳过: {_ve}")

        trade = engine.execute_buy(
            symbol=sym_full, name=name, price=price,
            recommendation=enhanced,
            max_position_pct=limits["single_pct"] * risk_mult,
            max_positions=limits["max_positions"],
            pct_change=pct_change,
            sealed_limit_up=sealed_limit_up,
        )

        if trade:
            executed.append({"name": name, "code": code, "qty": trade.quantity, "price": price, "score": score})
            logger.info(f"  BUY {name}({code}) {trade.quantity}股 @{price:.2f} score={score}")

    manager.save()
    state = engine.state

    # 2c. 持仓MTM (实时估值)
    mtm_prices = {}
    mtm_pct = {}
    for sym in state.positions:
        _px, _pct = _tencent_quote(sym)
        if _px > 0:
            mtm_prices[sym] = _px
            mtm_pct[sym] = _pct

    if mtm_prices:
        engine.mark_to_market(mtm_prices)

        # ── 策略共识退出检测 ──
        strategy_votes = {}
        try:
            from analysis.strategies_v3 import multifactor_v3, macd_trend_v3
            from analysis.optimized_strategies import backtest_momentum_v2
            from data.router import get_data_router
            from data.providers.base import DataFrequency, DataRequest

            monitors = {
                "多因子v3": multifactor_v3,
                "MACDv3": macd_trend_v3,
                "动量v2": backtest_momentum_v2,
            }

            data_router = get_data_router()
            for sym, pos in state.positions.items():
                try:
                    req = DataRequest(sym, today_cn() - timedelta(days=200), today_cn(), DataFrequency.DAILY)
                    r = await data_router.get_daily_kline(req)
                    df = r.data
                    if df.empty or len(df) < 30:
                        continue

                    exit_votes = 0
                    votes_detail = []
                    for sname, sfunc in monitors.items():
                        bt = sfunc(df, symbol=sym)
                        n = bt.get("signals", 0)
                        wr = bt.get("win_rate", 0)
                        # 如果策略在最近数据上无买入信号且有持仓 → 建议退出
                        if n == 0 or wr < 0.3:
                            exit_votes += 1
                            votes_detail.append(f"{sname}:无信号/低胜率{wr:.0%}")
                    if exit_votes >= 2:
                        strategy_votes[sym] = exit_votes
                        logger.info(f"  [策略投票] {pos.name}({sym}) {exit_votes}/3 建议退出: {', '.join(votes_detail)}")
                except Exception as e:
                    logger.debug(f"  策略检测失败 {sym}: {e}")
        except Exception as e:
            logger.warning(f"策略共识检测跳过: {e}")

        # 检查增强退出条件
        exit_triggers = engine.check_exit_conditions(
            mtm_prices, sell_signals={}, strategy_votes=strategy_votes,
        )
        for trigger in exit_triggers:
            if trigger.get("should_sell"):
                sym = trigger["symbol"]
                trade = engine.execute_sell(
                    symbol=sym, exit_reason=trigger["reason"],
                    pct_change=mtm_pct.get(sym),
                )
                if trade:
                    logger.info(f"  [退出] {trigger['name']}({sym}) {trigger['reason']}: {trigger['detail']}")

        # v5.11 方案3b: 低估值规则高估卖出 (确定性, 与诊断官解耦; gate<=0 零开销)
        _learned_sold = await _apply_learned_sell(engine, data_router, mtm_pct)
        if _learned_sold:
            sold.extend(_learned_sold)

        # v3.3 防御减仓 (risk_off) — 股票感知版: 只卖"自身走弱"的 (现价<自身MA20),
        # 保留强势股. 用户反馈: 指数弱 ≠ 个股弱 (结构性行情强股可持有),
        # 不能按浮亏盲卖强股 (实测: 上次 8→2 误卖 3 只自身在MA20上的盈利股).
        # v5.1 大师阶段防御: market_phase=trend_down/bubble_late 时, 即使 timing 未 off、
        # risk_level<4, 也触发减仓弱股 — 让大师对"泡沫末期/下行"的警告真正驱动降低敞口.
        _phase_defensive = _diag.get("market_phase") in ("trend_down", "bubble_late")
        if (_timing_sig and _timing_sig["signal"] == "risk_off") or _phase_defensive:
            _sold_weak = 0
            for sym, pos in list(state.positions.items()):
                _weak = await _below_ma20(sym, data_router)
                if _weak is None:  # 个股自身走弱 → 防跌卖出
                    continue
                _px_now, _ma20 = _weak
                _px, _pct = _tencent_quote(sym)
                trade = engine.execute_sell(
                    symbol=sym, exit_reason="risk_off+个股走弱", pct_change=_pct)
                if trade:
                    sold.append({"symbol": sym, "name": pos.name, "price": trade.price})
                    _sold_weak += 1
                    logger.info(f"  [防御减仓] {pos.name}({sym}) 现价{_px_now:.1f}<自身MA20 {_ma20:.1f} → 卖出 pnl={trade.pnl:+.2f}")
            if _sold_weak:
                logger.warning(f"风险关闭+个股走弱: 减仓 {_sold_weak} 只 (强势股保留)")

        # ── v3.5 主动目标仓位收敛: 持仓数>目标时, 只卖"自身走弱"(现价<自身MA20)股, 强势股保留 ──
        # 此前系统在超标时仅"停止新买", 从不主动减到目标. 现在按市场诊断后的目标持仓数收敛.
        _over = len(state.positions) - limits["max_positions"]
        if _over > 0 and os.getenv("POSITION_CONVERGENCE", "1") == "1":
            logger.warning(f"[收敛] 持仓{len(state.positions)} > 目标{limits['max_positions']} (超{_over}只)")
            _converged = 0
            for sym, pos in list(state.positions.items()):
                if len(state.positions) <= limits["max_positions"]:
                    break
                if pos.buy_date == today:  # T+1, 今日买入不可卖
                    continue
                _weak = await _below_ma20(sym, data_router)
                if _weak is None:
                    continue
                _px_now, _ma20 = _weak
                _px, _pct = _tencent_quote(sym)
                trade = engine.execute_sell(
                    symbol=sym, exit_reason="目标仓位收敛+个股走弱", pct_change=_pct)
                if trade:
                    sold.append({"symbol": sym, "name": pos.name, "price": trade.price})
                    _converged += 1
                    logger.info(f"  [收敛减仓] {pos.name}({sym}) 现价{_px_now:.1f}<自身MA20 {_ma20:.1f} → 卖出 pnl={trade.pnl:+.2f}")
            if _converged:
                logger.warning(f"目标仓位收敛: 减仓 {_converged} 只 (仅自身走弱股)")

        manager.save()

    return {
        "status": "ok",
        "sold": len(sold),
        "bought": len(executed),
        "positions": len(state.positions),
        "cash": round(state.cash, 2),
        "total_value": round(state.total_value, 2),
        "total_return_pct": round(state.total_return_pct, 2),
        "executed": executed,
    }


# ═══════════════════════════════════════════════════════════════
# Phase 3: 总结
# ═══════════════════════════════════════════════════════════════

async def phase3_summary() -> Dict[str, Any]:
    """生成持仓总结 + v3.1 决策结果闭环记录"""
    from simulation.portfolio import PortfolioManager
    from simulation.paper_trader import PaperTradingEngine

    manager = PortfolioManager()
    engine = PaperTradingEngine(manager)
    summary = engine.get_summary()

    # 添加每日快照
    state = engine.state
    today = today_cn().isoformat()
    existing = [s for s in state.daily_snapshots if s.date == today]
    if not existing:
        from simulation.portfolio import DailySnapshot
        # 计算相对于上一快照的日盈亏 (而非累计值)
        prev_value = state.initial_capital
        if state.daily_snapshots:
            prev_value = state.daily_snapshots[-1].total_value
        daily_pnl = state.total_value - prev_value
        daily_return_pct = (daily_pnl / prev_value * 100) if prev_value > 0 else 0.0
        state.daily_snapshots.append(DailySnapshot(
            date=today,
            total_value=round(state.total_value, 2),
            cash=round(state.cash, 2),
            position_value=round(state.position_value, 2),
            daily_pnl=round(daily_pnl, 2),
            daily_return_pct=round(daily_return_pct, 2),
            cumulative_return_pct=round(state.total_return_pct, 2),
            positions_count=state.position_count,
        ))
        # 保持快照按日期有序
        state.daily_snapshots.sort(key=lambda s: s.date)
        manager.save()

    # ── v3.1: 决策结果闭环 — 回顾5天前的决策,记录实际盈亏 ──
    try:
        from agent.orchestration.decision_log import DecisionLogger
        dl = DecisionLogger()

        # 获取需要回顾的旧决策 (5天前)
        unreviewed = dl.get_unreviewed()
        if unreviewed:
            logger.info(f"[决策回顾] 发现{len(unreviewed)}条待回顾的决策")

            # v5.5 P0-2: 基准用真实上证指数同期收益 (替代恒 0). 取一次序列, 逐条算对应该决策持有期.
            try:
                _bench_idx = await _load_index_close()
            except Exception:
                _bench_idx = None
            if _bench_idx is not None and len(_bench_idx) > 0:
                logger.info("[决策回顾] benchmark 接入上证指数同期收益")
            else:
                logger.warning("[决策回顾] 基准指数不可用, benchmark 回退 0.0 (不影响交易)")

            for record in unreviewed:
                normalized = record.symbol
                if "." not in normalized:
                    # Try to match with positions
                    for sym, pos in state.positions.items():
                        short = sym.replace("sh.", "").replace("sz.", "")
                        if short == record.symbol:
                            normalized = sym
                            break

                # 计算相对于市场的收益 (简化:使用持仓中的盈亏)
                for sym, pos in state.positions.items():
                    short = sym.replace("sh.", "").replace("sz.", "")
                    if short == record.symbol or sym == record.symbol:
                        realized_return = (pos.current_price / pos.avg_cost - 1) if pos.avg_cost > 0 else 0
                        # v5.5 P0-2: 真实大盘基准 (决策日→回顾日), 让 alpha 反映"是否跑赢大盘"
                        benchmark_return = _index_benchmark_ret(
                            _bench_idx, record.analysis_date, today
                        )
                        dl.log_outcome(
                            log_id=record.log_id,
                            realized_return=round(realized_return, 4),
                            benchmark_return=round(benchmark_return, 4),
                            review_date=today,
                            notes=f"自动回顾: {record.final_signal}@{record.confidence:.0%}置信度 → 持仓收益{realized_return:+.2%} vs 上证{benchmark_return:+.2%}",
                        )
                        break

            logger.info(f"[决策回顾] 完成{len(unreviewed)}条回顾")
        else:
            logger.debug(f"[决策回顾] 无待回顾决策")

        # 打印决策统计
        stats = dl.get_stats(days=30)
        if stats.get("total", 0) > 0:
            acc = stats.get("accuracy")
            if acc is not None:
                logger.info(f"[决策统计] 近30天{stats['total']}次决策, 已回顾{stats['reviewed']}次, "
                           f"准确率{acc:.0%}")

    except Exception as e:
        logger.debug(f"[决策回顾] 跳过: {e}")

    logger.info(f"持仓总结: {summary['position_count']}只 | 总资产 RMB{summary['total_value']:,.2f} | 收益 {summary['total_return_pct']:+.2f}%")
    return summary


# ═══════════════════════════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════════════════════════

async def run_full_day(
    no_llm: bool = False,
    dry_run: bool = False,
    reset: bool = False,
    skip_analyze: bool = False,
    enable_evolution: bool = True,
    cold_tilt: bool = False,
) -> dict:
    """全市场AI选股 — 每日自主工作流"""
    today = today_cn().isoformat()
    logger.info(f"========== 全市场AI选股 Workflow: {today} ==========")

    if reset:
        from simulation.portfolio import PortfolioManager
        PortfolioManager().reset()
        logger.info("账户已重置: RMB 100,000")

    # Phase 1: 全市场分析
    if not skip_analyze:
        analysis = await phase1_analyze(
            use_llm=not no_llm,
            enable_evolution=enable_evolution,
            cold_tilt=cold_tilt,
        )
        logger.info(f"Phase 1 完成: {analysis['deep_analyzed_count']}只分析 | 耗时{analysis.get('elapsed_seconds',0)}s")
    else:
        logger.info("Phase 1 跳过 (--skip-analyze)")

    # Phase 2: 执行交易
    # v5.5 数据脱节拦截: 基于过期缓存(lag>3天)的分析 → 降级只读, 不真交易, 防"旧行情定仓位/新价成交"脱节
    _effective_dry_run = _effective_dry_run_for_lag(analysis, dry_run, skip_analyze, logger)
    trade_result = await phase2_execute(dry_run=_effective_dry_run)
    logger.info(f"Phase 2 完成: 卖出{trade_result.get('sold',0)} | 买入{trade_result.get('bought',0)}")

    # Phase 3: 总结
    summary = await phase3_summary()
    logger.info(f"Phase 3 完成: {summary['position_count']}只持仓 | 总资产 RMB{summary['total_value']:,.2f}")

    # v3.1: 日终通知推送
    try:
        from notify import NotificationManager
        nm = NotificationManager()
        if nm.enabled:
            regime_label = "N/A"
            if not skip_analyze and isinstance(analysis, dict):
                regime_label = analysis.get("regime") or analysis.get("market_regime") or "N/A"
            nm.send_daily_summary(
                summary_text=f"日期: {today}\n市场: {regime_label}\n"
                           f"持仓: {summary['position_count']}只 | "
                           f"总资产: RMB{summary['total_value']:,.2f} | "
                           f"收益: {summary['total_return_pct']:+.2f}%\n"
                           f"交易: BUY {trade_result.get('bought',0)} | SELL {trade_result.get('sold',0)}"
            )
            logger.info("日终通知已推送")
    except Exception as e:
        logger.debug(f"通知推送跳过: {e}")

    # ── 记录 AI Track Record ──
    track_path = REPORT_DIR / "ai_track_record.json"
    try:
        track = []
        if track_path.exists():
            with open(track_path, "r", encoding="utf-8") as f:
                track = json.load(f)

        # 记录当天的 BUY 推荐
        # v3.0: 安全获取 regime (修复此前未定义 NameError; skip_analyze 路径无 analysis 变量)
        track_regime = "unknown"
        if not skip_analyze:
            try:
                track_regime = (
                    analysis.get("regime")
                    or analysis.get("market_regime", {}).get("regime", "unknown")
                )
            except Exception:
                track_regime = "unknown"

        today_entry = {
            "date": today,
            "regime": track_regime,
            "buys": [],
        }
        if not skip_analyze:
            analysis_path = REPORT_DIR / "deep_analysis_top100.json"
            if analysis_path.exists():
                with open(analysis_path, "r", encoding="utf-8") as f:
                    analysis = json.load(f)
                for r in analysis.get("results", []):
                    if r.get("action") == "BUY":
                        code = r.get("code", "")
                        # 获取当时价格 (v3.0: 复用统一行情 helper, 修复未导入 requests 的 F821)
                        try:
                            prefix = _tencent_prefix(code)
                            price, _pct = _tencent_quote(f"{prefix}.{code}")
                        except Exception:
                            price = 0

                        today_entry["buys"].append({
                            "code": code, "name": r.get("name", ""),
                            "score": r.get("final_score", 0),
                            "conviction": r.get("conviction", 0),
                            "price_at_rec": price,
                            "technical": r.get("technical", "")[:80],
                        })

        track.append(today_entry)

        # 标记过往推荐的表现 (回看 1/3/5 天)
        for old_entry in track[:-1]:
            if old_entry.get("reviewed"):
                continue
            old_date = old_entry.get("date", "")
            days_ago = (today_cn() - date.fromisoformat(old_date)).days
            if days_ago < 1:
                continue

            for b in old_entry.get("buys", []):
                try:
                    code = b["code"]
                    prefix = _tencent_prefix(code)
                    # v3.0: 复用统一行情 helper (修复此前未导入 requests 的 F821 NameError)
                    cur_price, _pct = _tencent_quote(f"{prefix}.{code}")
                    rec_price = b.get("price_at_rec", 0)
                    if rec_price > 0 and cur_price > 0:
                        b[f"return_d{days_ago}"] = round((cur_price / rec_price - 1) * 100, 2)
                except Exception:
                    pass
            if days_ago >= 5:
                old_entry["reviewed"] = True

        with open(track_path, "w", encoding="utf-8") as f:
            json.dump(track, f, ensure_ascii=False, indent=2)
        logger.info(f"Track record 已保存: {track_path} ({len(track)}天)")
    except Exception as e:
        logger.debug(f"Track record 保存跳过: {e}")

    logger.info(f"========== Workflow 完成 ==========")

    # v5.7 进化成本专项分账: 从共享 router 汇总进化模块成本 (task_type 前缀 evolution_/adversarial)
    try:
        from models.router import get_shared_router
        _cs = get_shared_router().cost_summary()
        _evo_tasks = {k: v for k, v in _cs["by_task_type"].items()
                      if k.startswith("evolution_") or k == "adversarial"}
        if _evo_tasks:
            _evo_cost = sum(v["cost"] for v in _evo_tasks.values())
            _evo_calls = sum(v["count"] for v in _evo_tasks.values())
            _detail = ", ".join(f"{k}=¥{v['cost']:.4f}" for k, v in sorted(_evo_tasks.items()))
            logger.info(f"[进化成本] 进化模块 {_evo_calls} 次调用 共 ¥{_evo_cost:.4f} ({_detail})")
    except Exception as _ce:
        logger.debug(f"[进化成本] 分账失败: {_ce}")

    return {"date": today, "analysis": "ok", "trade": trade_result, "summary": summary}


async def main():
    parser = argparse.ArgumentParser(description="全市场AI选股 — 每日自主工作流")
    parser.add_argument("--dry-run", action="store_true", help="仅分析,不交易")
    parser.add_argument("--no-llm", action="store_true", help="跳过Ollama LLM层")
    parser.add_argument("--reset", action="store_true", help="重置账户")
    parser.add_argument("--skip-analyze", action="store_true", help="跳过分析(使用已有结果)")
    parser.add_argument("--no-evolution", action="store_true",
                        help="禁用自我进化系统 (默认开启)")
    parser.add_argument("--cold-tilt", action="store_true",
                        help="冷落模式: 选股池=低换手(被冷落)bottom-N tilt (跑赢指数构造)")
    args = parser.parse_args()

    result = await run_full_day(
        no_llm=args.no_llm,
        dry_run=args.dry_run,
        reset=args.reset,
        skip_analyze=args.skip_analyze,
        enable_evolution=not args.no_evolution,
        cold_tilt=args.cold_tilt,
    )

    s = result["summary"]
    print(f"\n{'='*60}")
    print(f"全市场AI选股 Workflow 完成: {result['date']}")
    print(f"  总资产: RMB {s.get('total_value',0):,.2f}")
    print(f"  收益率: {s.get('total_return_pct',0):+.2f}%")
    print(f"  持仓: {s.get('position_count',0)} 只")
    print(f"  交易: 买{result['trade'].get('bought',0)} 卖{result['trade'].get('sold',0)}")
    print(f"{'='*60}")


if __name__ == "__main__":
    asyncio.run(main())
