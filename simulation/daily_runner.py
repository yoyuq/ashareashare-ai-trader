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


# ═══════════════════════════════════════════════════════════════
# Phase 1: 全市场分析
# ═══════════════════════════════════════════════════════════════

async def phase1_analyze(use_llm: bool = True) -> Dict[str, Any]:
    """全市场分析: 5884 → 规则预筛 → [LLM] → DeepSeek → 结果"""
    t0 = time.time()
    logger.info("=" * 50)
    logger.info("Phase 1: 全市场AI分析")
    logger.info("=" * 50)

    # 1a. 加载数据
    logger.info("加载全市场数据...")
    import akshare as ak
    df = ak.stock_zh_a_spot_em()
    col_map = {
        '代码':'code','名称':'name','最新价':'price','涨跌幅':'pct_change','成交量':'volume',
        '成交额':'amount','换手率':'turnover','市盈率-动态':'pe_ttm','市净率':'pb',
        '总市值':'total_mv','量比':'vol_ratio','60日涨跌幅':'pct_60d','振幅':'amplitude'
    }
    df = df.rename(columns={k:v for k,v in col_map.items() if k in df.columns})
    logger.info(f"全市场: {len(df)} 只")

    # 1b. 体制自适应多因子初筛 (v3.1 PreScreener)
    # 替代旧版 ad-hoc 打分, 使用6维度 x 体制自适应权重
    top_n = 300 if use_llm else 100

    try:
        from analysis.pre_screener import PreScreener

        # 检测当前市场体制
        regime_info = _detect_regime(df)
        regime = regime_info.get("regime", "range_bound")
        logger.info(f"市场体制: {regime} ({regime_info.get('label', '')})")

        screener = PreScreener()
        result = screener.screen(df, regime=regime, top_n=top_n)
        df_top = result.df

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

    # 1c. DeepSeek Flash 精筛 (替代 Ollama)
    if use_llm:
        try:
            df_top = await _flash_screen(df_top, top_k=100)
            logger.info(f"Flash精筛后: {len(df_top)} 只")
        except Exception as e:
            logger.warning(f"LLM精筛跳过 (Ollama不可用): {e}")

    # 1d. DeepSeek深度分析
    deep_results = await _deepseek_analyze(df_top.head(100))
    logger.info(f"DeepSeek分析完成: {len(deep_results)} 只")

    # 1e. 保存结果
    today = today_cn().isoformat()
    out = {
        "date": today,
        "total_screened": len(df),
        "rule_filtered_count": len(df_top),
        "deep_analyzed_count": len(deep_results),
        "market_regime": _detect_regime(df),
        "results": deep_results,
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    outpath = REPORT_DIR / f"deep_analysis_top100.json"
    outpath.parent.mkdir(parents=True, exist_ok=True)
    with open(outpath, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    logger.info(f"结果已保存: {outpath}")

    return out


async def _flash_screen(df: pd.DataFrame, top_k: int = 100) -> pd.DataFrame:
    """DeepSeek V4-Flash 精筛 (替代 Ollama, 更快更强)"""
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
        for _, r in batch.iterrows():
            mv = r.get('total_mv', 0) / 1e8
            lines.append(
                f"{r['code']} {r['name']} price={r['price']:.2f} chg={r['pct_change']:+.1f}% "
                f"PE={r.get('pe_ttm',0):.0f} PB={r.get('pb',0):.1f} MV={mv:.0f}亿 "
                f"换手={r.get('turnover',0):.1f}% 量比={r.get('vol_ratio',0):.1f}"
            )

        prompt = (
            f"评估以下{len(lines)}只A股的短期潜力(0-100分)。"
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
        df["llm_score"] = df["code"].map(lambda c: all_scores.get(str(c), 50))
        # v3.1 修复: PreScreener 输出 composite_score (非旧版 rule_score), 兼容两者
        base_score = (df["rule_score"] if "rule_score" in df.columns
                      else df["composite_score"])
        df["combined"] = base_score * 0.3 + df["llm_score"] * 0.7  # Flash权重更高
        df = df.nlargest(top_k, "combined")
    return df


async def _deepseek_analyze(df_top: pd.DataFrame) -> List[Dict]:
    """DeepSeek深度分析 Top100"""
    from openai import AsyncOpenAI
    import os

    client = AsyncOpenAI(
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
        timeout=120.0,
    )

    results = []
    for batch_i in range(0, min(100, len(df_top)), 20):
        batch = df_top.iloc[batch_i:batch_i+20]
        lines = []
        for _, r in batch.iterrows():
            mv = r.get('total_mv',0)/1e8
            board = "主板"
            code_str = str(r['code'])
            if code_str.startswith('3'): board = "创业板"
            elif code_str.startswith('688'): board = "科创板"
            lines.append(
                f"{code_str} {r['name']} [{board}] price={r['price']:.2f} chg={r['pct_change']:+7.1f}% "
                f"PE={r.get('pe_ttm',0):.0f} PB={r.get('pb',0):.1f} MV={mv:.0f}亿 "
                f"换手={r.get('turnover',0):.1f}% 60日涨跌={r.get('pct_60d',0):+.1f}% "
                f"因子分={r.get('composite_score',r.get('rule_score',50)):.0f}"
            )

        system_prompt = (
        "你是资深A股分析师。对每只股票从技术面、基本面、风险三个维度评估。"
        "评分标准: 85+强烈看多, 70-84偏多, 55-69中性, 40-54偏空, <40看空。"
        "特别注意: 北交所(8/9/4开头)/ST/亏损股给低分; 主板蓝筹/业绩确定给高分。"
        "只返回JSON数组, 不要markdown包裹。"
    )
    prompt = (
        f"分析以下{len(lines)}只A股,给出最终评分(0-100)和操作(BUY/HOLD/SELL)。"
        f"每只股票需包含: final_score, action, conviction(0-1), technical(技术面简述), fundamental(基本面简述), risk(风险简述)。\n"
        + "\n".join(lines) + "\n"
        "返回JSON:[{\"code\":\"\",\"name\":\"\",\"final_score\":0,\"action\":\"BUY\",\"conviction\":0.5,\"technical\":\"\",\"fundamental\":\"\",\"risk\":\"\"}]"
    )

    try:
        resp = await client.chat.completions.create(
            model=os.getenv("DEEPSEEK_FLASH_MODEL", "deepseek-v4-flash"),
            messages=[{"role":"system","content": system_prompt}, {"role":"user","content": prompt}],
            temperature=0.3, max_tokens=4000,
            # v3.1 修复: 禁用 thinking (同 _flash_screen)
            extra_body={"thinking": {"type": "disabled"}},
        )
        content = resp.choices[0].message.content.strip()
        if content.startswith("```"): content = content.split("\n",1)[1].rsplit("```",1)[0]
        items = json.loads(content)
        if isinstance(items, list):
            for idx, item in enumerate(items):
                if not item.get("code") and idx < len(batch):
                    item["code"] = str(batch.iloc[idx]["code"])
                    item["name"] = str(batch.iloc[idx]["name"])
            results.extend(items)
            logger.debug(f"DeepSeek batch#{batch_i}: {len(items)} analyzed")
    except Exception as e:
        logger.warning(f"DeepSeek batch#{batch_i}: {e}")

    results.sort(key=lambda x: x.get("final_score", x.get("score", 0)), reverse=True)
    return results


def _detect_regime(df: pd.DataFrame) -> Dict:
    """市场状态检测 (基于涨跌分布)"""
    up_pct = (df['pct_change'] > 0).mean() if 'pct_change' in df.columns else 0.5
    avg_pct = df['pct_change'].mean() if 'pct_change' in df.columns else 0
    extreme_down = (df['pct_change'] < -5).mean() if 'pct_change' in df.columns else 0
    extreme_up = (df['pct_change'] > 5).mean() if 'pct_change' in df.columns else 0

    if extreme_down > 0.1: regime, label = "crisis", "危机"
    elif avg_pct < -1.5: regime, label = "strong_bear", "强熊"
    elif avg_pct < -0.3: regime, label = "weak_bear", "弱熊"
    elif avg_pct < 0.3: regime, label = "range_bound", "震荡"
    elif avg_pct < 1.5 and extreme_up < 0.05: regime, label = "weak_bull", "弱牛"
    elif extreme_up > 0.05: regime, label = "strong_bull", "强牛"
    else: regime, label = "strong_bull", "强牛"

    return {"regime": regime, "label": label, "avg_pct": round(float(avg_pct), 2), "up_ratio": round(float(up_pct), 2)}


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


def _limit_pct_for_code(code: str) -> float:
    """按代码返回板块涨跌幅限制(%) — 主板±10%, 创业板/科创板±20%, 北交所±30%"""
    if code.startswith("30") or code.startswith("68"):
        return 20.0
    if code.startswith("8") or code.startswith("4"):
        return 30.0
    return 10.0


async def phase2_execute(dry_run: bool = False) -> Dict[str, Any]:
    """基于分析结果执行交易"""
    logger.info("=" * 50)
    logger.info("Phase 2: 执行交易")
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

    buys = [r for r in results if r.get("action") == "BUY"]
    sells = [r for r in results if r.get("action") == "SELL"]
    logger.info(f"分析结果: {len(results)}只 → BUY {len(buys)} | SELL {len(sells)}")

    # 加载市场状态
    regime_info = analysis.get("market_regime", {})
    regime = regime_info.get("regime", "range_bound")
    limits = REGIME_LIMITS.get(regime, REGIME_LIMITS["range_bound"])
    logger.info(f"市场: {regime} → {limits['label']} | 最大{limits['max_positions']}只 | 单只{limits['single_pct']:.0%}")

    if dry_run:
        logger.info("[DRY RUN] 仅分析, 不执行交易")
        return {"status": "dry_run", "buys": len(buys), "sells": len(sells)}

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

        # ── 体制修正确信度 ──
        regime_conv_mult = {
            "strong_bull": 1.0, "weak_bull": 0.9,
            "range_bound": 0.6, "weak_bear": 0.4,
            "strong_bear": 0.2, "crisis": 0.0,
        }
        original_conv = conv
        conv = original_conv * regime_conv_mult.get(regime, 0.6)
        if conv < 0.3:
            logger.info(f"  跳过 {name}({code}): 体制修正后确信度{conv:.0%} < 30% "
                        f"(原始{original_conv:.0%} × {regime_conv_mult.get(regime,0.6)})")
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

        # 动态止损止盈
        if score >= 80: sl_pct, tp_pct = 0.05, 0.12
        elif score >= 60: sl_pct, tp_pct = 0.07, 0.10
        else: sl_pct, tp_pct = 0.10, 0.08

        # v3.0: 板块感知的涨停封板判定, 传入 execute_buy 使其真正生效
        limit_pct = _limit_pct_for_code(code)
        sealed_limit_up = bool(pct_change is not None and pct_change >= limit_pct - 0.2)

        enhanced = {
            "conviction": conv, "score": score/10, "composite_score": score,
            "key_reasons": [technical, fundamental], "risks": [risk],
            "verdict_summary": f"DeepSeek: {rec.get('action','')}({conv:.0%}) s={score}",
            "stop_loss": round(price*(1-sl_pct),2),
            "take_profit": round(price*(1+tp_pct),2),
        }

        # ── v3.1-deerflow: DecisionValidator 执行前硬约束校验 ──
        # 用与 execute_buy 相同的仓位/止损止盈参数校验, 拒绝则跳过并落 journal
        try:
            from agent.sub_agents.validator import DecisionValidator
            _pos_pct = min(limits["single_pct"] * risk_mult, 0.20)
            _vd = DecisionValidator()
            _v_res = await _vd.run(_vd._start_context(
                task_id=f"daily_runner_{today}",
                trading_params={sym_full: {
                    "entry_price": price,
                    "stop_loss": enhanced["stop_loss"],
                    "take_profit": enhanced["take_profit"],
                    "position_pct": round(_pos_pct, 3),
                }},
                stock_recommendations={sym_full: {"action": "BUY", "conviction": conv}},
                market_data={},  # 涨跌停可行性已用实时 pct_change + sealed_limit_up 判断
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
                        bt = sfunc(df)
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

def phase3_summary() -> Dict[str, Any]:
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
                        benchmark_return = 0.0  # TODO: 对接 CSI300 同期数据
                        dl.log_outcome(
                            log_id=record.log_id,
                            realized_return=round(realized_return, 4),
                            benchmark_return=benchmark_return,
                            review_date=today,
                            notes=f"自动回顾: {record.final_signal}@{record.confidence:.0%}置信度 → 持仓收益{realized_return:+.2%}",
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
        analysis = await phase1_analyze(use_llm=not no_llm)
        logger.info(f"Phase 1 完成: {analysis['deep_analyzed_count']}只分析 | 耗时{analysis.get('elapsed_seconds',0)}s")
    else:
        logger.info("Phase 1 跳过 (--skip-analyze)")

    # Phase 2: 执行交易
    trade_result = await phase2_execute(dry_run=dry_run)
    logger.info(f"Phase 2 完成: 卖出{trade_result.get('sold',0)} | 买入{trade_result.get('bought',0)}")

    # Phase 3: 总结
    summary = phase3_summary()
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
                            prefix = "sh" if code.startswith("6") else ("bj" if code.startswith("9") else "sz")
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
                    prefix = "sh" if code.startswith("6") else ("bj" if code.startswith("9") else "sz")
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

    return {"date": today, "analysis": "ok", "trade": trade_result, "summary": summary}


async def main():
    parser = argparse.ArgumentParser(description="全市场AI选股 — 每日自主工作流")
    parser.add_argument("--dry-run", action="store_true", help="仅分析,不交易")
    parser.add_argument("--no-llm", action="store_true", help="跳过Ollama LLM层")
    parser.add_argument("--reset", action="store_true", help="重置账户")
    parser.add_argument("--skip-analyze", action="store_true", help="跳过分析(使用已有结果)")
    args = parser.parse_args()

    result = await run_full_day(
        no_llm=args.no_llm,
        dry_run=args.dry_run,
        reset=args.reset,
        skip_analyze=args.skip_analyze,
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
