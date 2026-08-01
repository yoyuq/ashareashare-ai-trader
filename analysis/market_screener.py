"""
全市场批量预筛引擎 — 免费模型并行处理 5800→100

使用免费大模型 (智谱GLM-4.7-Flash / 百度ERNIE-4.5-Flash) 对全A股
进行快速预筛选, 选出 Top 100 候选, 再交给 DeepSeek 做深度分析。

架构:
  5800 stocks → split into batches(20/batch) → 12x parallel GLM-4.7-Flash
  → collect scores + reasons → merge & rank → top 100 → DeepSeek analysis

模型池:
  - 智谱 GLM-4.7-Flash: 免费不限量, 128K ctx, 30 QPS (主力)
  - 百度 ERNIE-4.5-Flash: 免费不限量, 50 QPS (备用)
  - DeepSeek: 已有付费API (深度分析)

用法:
  python -m analysis.market_screener          # 全量扫描
  python -m analysis.market_screener --top 50  # 仅取前50
"""

import asyncio
import json
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from dotenv import load_dotenv
from loguru import logger

load_dotenv()

# ═══════════════════════════════════════════════════════════════
# 免费模型池配置
# ═══════════════════════════════════════════════════════════════

@dataclass
class ModelConfig:
    """模型配置"""
    name: str
    provider: str
    base_url: str
    model_id: str
    api_key: str = ""
    max_qps: int = 10       # 最大并发 (保守估计)
    max_context: int = 128_000
    temperature: float = 0.0  # 预筛用最低温, 追求一致性
    timeout: float = 20.0     # 单次请求超时
    enabled: bool = True
    extra_body: dict = field(default_factory=dict)  # 额外参数 (如禁用推理)

# 免费模型池 (按优先级排序)
# 支持多Key同模型: 用不同name区分
MODEL_POOL: List[ModelConfig] = [
    ModelConfig(
        name="GLM-4.7-Flash #1",
        provider="智谱AI",
        base_url="https://open.bigmodel.cn/api/paas/v4/",
        model_id="glm-4.7-flash",
        api_key=os.getenv("ZHIPU_API_KEY", ""),
        max_qps=5,
        max_context=128_000,
        extra_body={"thinking": {"type": "disabled"}},
    ),
    ModelConfig(
        name="GLM-4.7-Flash #2",
        provider="智谱AI",
        base_url="https://open.bigmodel.cn/api/paas/v4/",
        model_id="glm-4.7-flash",
        api_key=os.getenv("ZHIPU_API_KEY_2", ""),
        max_qps=5,
        max_context=128_000,
        extra_body={"thinking": {"type": "disabled"}},
    ),
    ModelConfig(
        name="ERNIE-4.5-Flash",
        provider="百度千帆",
        base_url="https://qianfan.baidubce.com/v2/",
        model_id="ernie-4.5-flash",
        api_key=os.getenv("BAIDU_QIANFAN_API_KEY", ""),
        max_qps=15,
        max_context=8_000,
    ),
    ModelConfig(
        name="Qwen-Flash",
        provider="硅基流动",
        base_url="https://api.siliconflow.cn/v1/",
        model_id="Qwen/Qwen2.5-7B-Instruct",
        api_key=os.getenv("SILICONFLOW_API_KEY", ""),
        max_qps=10,
        max_context=32_000,
    ),
]

# ═══════════════════════════════════════════════════════════════
# 预筛 prompt
# ═══════════════════════════════════════════════════════════════

SCREENING_SYSTEM_PROMPT = """你是A股量化分析师。根据股票的技术数据,快速评估每只股票的短期潜力。

评分标准(0-100):
- 涨跌幅(pct_change): 强势上涨+25分,温和上涨+15分,平盘+5分,下跌0分
- 成交量确认: 量比>1.5且上涨+20分,缩量上涨+10分
- 估值合理性: PE适中(10-40)+15分,极端-5分
- 市值优势: 中大盘+10分
- 趋势延续: 60日涨跌正值+15分,年初至今正+15分

返回JSON数组,每只股票一个对象,按score降序排列:
[{"code":"000001","name":"平安银行","score":85,"reason":"放量突破+量比2.1+PE合理"}]

只返回JSON,不要其他文字。最多输出20条。"""

# 每批次股票数量 (GLM-4.7-Flash 128K ctx, 50股约3K tokens输入+2K输出)
BATCH_SIZE = 50

# 最终输出数量
DEFAULT_TOP_N = 100

# 最大并发请求数 (双Key双并发)
MAX_CONCURRENT = 2
# 批次间延迟秒数 (双Key各1并发,无需延迟)
BATCH_DELAY = 0.5


# ═══════════════════════════════════════════════════════════════
# 批量预筛引擎
# ═══════════════════════════════════════════════════════════════

def rule_based_prefilter(df: pd.DataFrame, top_n: int = 500) -> pd.DataFrame:
    """
    规则预筛: 用量化指标快速过滤 5800 → ~500 只

    三层漏斗:
      1. 流动性过滤: 剔除成交额过低、换手率过低的僵尸股
      2. 动量过滤: 剔除暴跌股、长期阴跌股
      3. 综合打分: 多因子加权排名

    返回: 过滤+排序后的 DataFrame
    """
    df = df.copy()
    initial = len(df)

    # Layer 1: 流动性过滤
    if "amount" in df.columns:
        med_amt = df["amount"].median()
        df = df[df["amount"] > max(med_amt * 0.1, 1e6)]  # 保留成交额在前90%的
    if "turnover" in df.columns:
        df = df[df["turnover"] >= 0.1]  # 换手率>=0.1%, 剔除僵尸股
    if "price" in df.columns:
        df = df[df["price"] > 2.0]  # 价格>2元, 剔除准退市股
    after_l1 = len(df)

    # Layer 2: 动量过滤
    if "pct_change" in df.columns:
        df = df[df["pct_change"] > -5.0]  # 剔除当日跌超5%的
    if "pct_60d" in df.columns:
        df = df[df["pct_60d"] > -25.0]  # 剔除60日跌超25%的
    if "pe_ttm" in df.columns:
        df = df[(df["pe_ttm"] > 0) | (df["pe_ttm"].isna())]  # PE为正或NaN(金融/周期)
    after_l2 = len(df)

    # Layer 3: 多因子打分
    score = pd.Series(50.0, index=df.index)  # 基础分50

    if "pct_change" in df.columns:
        score += df["pct_change"].clip(-5, 10) * 3  # 当日动量 (最多+30, -15)
    if "pct_60d" in df.columns:
        score += df["pct_60d"].clip(-20, 50) * 0.5  # 中期趋势 (最多+25, -10)
    if "turnover" in df.columns:
        score += df["turnover"].clip(0, 20) * 1.5  # 活跃度加分 (最多+30)
    if "vol_ratio" in df.columns:
        score += (df["vol_ratio"].clip(0.3, 5) - 1) * 5  # 量比偏离 (最多+20)
    # PE合理加分: 10-40区间最好
    if "pe_ttm" in df.columns:
        pe_score = 10 - abs(df["pe_ttm"].clip(0, 100) - 20) / 8
        score += pe_score.clip(-10, 10)
    # 市值加分: 大盘蓝筹+5, 中盘+3
    if "total_mv" in df.columns:
        mv_rank = df["total_mv"].rank(pct=True)
        score += mv_rank * 5  # 大市值最多+5

    df["rule_score"] = score.clip(0, 100)

    # 取 Top N
    df = df.nlargest(top_n, "rule_score")
    after_l3 = len(df)

    logger.info(f"规则预筛: {initial} -> L1流动性{after_l1} -> L2动量{after_l2} -> L3打分Top{after_l3}")
    return df


class BatchScreener:
    """全市场批量预筛引擎 — 规则预筛 + LLM精筛 + DeepSeek深度分析"""

    def __init__(self, models: Optional[List[ModelConfig]] = None, use_rules: bool = True):
        self.models = [m for m in (models or MODEL_POOL) if m.enabled and m.api_key]
        self.use_rules = use_rules
        if not self.models:
            logger.warning("没有可用的免费模型配置,将使用本地Ollama降级")
        self._semaphore = asyncio.Semaphore(MAX_CONCURRENT)
        self._results: List[Dict] = []
        self._stats = {"total": 0, "processed": 0, "errors": 0, "elapsed": 0.0}

    def _build_batch_prompt(self, batch: pd.DataFrame) -> str:
        """为一组股票构建分析 prompt"""
        lines = []
        # 表头
        lines.append("code name price pct_change volume amount turnover pe_ttm pb total_mv amplitude pct_60d")
        for _, row in batch.iterrows():
            code = str(row.get("code", ""))
            name = str(row.get("name", ""))
            price = f"{row.get('price', 0):.2f}"
            pct = f"{row.get('pct_change', 0):+.2f}"
            vol = f"{row.get('volume', 0):.0f}"
            amt = f"{row.get('amount', 0):.0f}"
            turnover = f"{row.get('turnover', 0):.2f}"
            pe = f"{row.get('pe_ttm', 0):.1f}"
            pb = f"{row.get('pb', 0):.2f}"
            mv = f"{row.get('total_mv', 0):.0f}"
            amp = f"{row.get('amplitude', 0):.2f}"
            pct_60d = f"{row.get('pct_60d', 0):+.2f}"
            lines.append(f"{code} {name} {price} {pct} {vol} {amt} {turnover} {pe} {pb} {mv} {amp} {pct_60d}")
        return "\n".join(lines)

    async def _call_model(
        self, model_cfg: ModelConfig, batch: pd.DataFrame, batch_idx: int
    ) -> List[Dict]:
        """调用单个模型处理一批股票 (带重试)"""
        async with self._semaphore:
            # 批次间延迟, 避免触发限流
            if batch_idx > 0:
                await asyncio.sleep(BATCH_DELAY)

            last_error = None
            for attempt in range(3):  # 最多重试3次
                try:
                    from openai import AsyncOpenAI

                    client = AsyncOpenAI(
                        api_key=model_cfg.api_key,
                        base_url=model_cfg.base_url,
                        timeout=model_cfg.timeout,
                    )

                    user_prompt = self._build_batch_prompt(batch)
                    create_kwargs = dict(
                        model=model_cfg.model_id,
                        messages=[
                            {"role": "user", "content": f"快速评估以下每只A股的投资潜力(0-100分),只看技术面动量。返回JSON数组[{{\"code\":\"\",\"score\":0,\"reason\":\"15字内\"}}]按score降序:\n{user_prompt}"},
                        ],
                        temperature=model_cfg.temperature,
                        max_tokens=4000,
                    )
                    if model_cfg.extra_body:
                        create_kwargs["extra_body"] = model_cfg.extra_body
                    response = await client.chat.completions.create(**create_kwargs)

                    content = response.choices[0].message.content.strip()
                    data = json.loads(content)
                    items = data if isinstance(data, list) else data.get("stocks", data.get("results", []))
                    logger.debug(f"[{model_cfg.name}] batch#{batch_idx}: {len(items)} results")
                    return items

                except Exception as e:
                    last_error = e
                    msg = str(e)
                    if "429" in msg or "1302" in msg or "1305" in msg:
                        wait = (attempt + 1) * 5
                        logger.warning(f"[{model_cfg.name}] batch#{batch_idx} rate limited, retry in {wait}s (attempt {attempt+1}/3)")
                        await asyncio.sleep(wait)
                        continue
                    break

            logger.warning(f"[{model_cfg.name}] batch#{batch_idx} failed after 3 retries: {last_error}")
            self._stats["errors"] += 1
            return []

    async def screen(
        self,
        df_market: pd.DataFrame,
        top_n: int = DEFAULT_TOP_N,
        progress_callback=None,
    ) -> Tuple[List[Dict], Dict]:
        """
        执行全市场预筛

        Args:
            df_market: 全市场数据 (from get_full_market)
            top_n: 最终返回数量
            progress_callback: 进度回调 async fn(batch_idx, total_batches, stats)

        Returns:
            (top_stocks, stats)
        """
        t0 = time.time()
        self._stats = {"total": len(df_market), "processed": 0, "errors": 0, "elapsed": 0.0}

        # 过滤掉无效数据
        required_cols = ["code", "name", "price", "pct_change"]
        df = df_market.dropna(subset=[c for c in required_cols if c in df_market.columns])
        logger.info(f"预筛输入: {len(df)} 只股票 (过滤后)")

        # Step 1: 规则预筛 — 5884 → ~500 只 (免费LLM额度不足以处理全量)
        if self.use_rules:
            df = rule_based_prefilter(df, top_n=max(300, top_n * 3))
            logger.info(f"规则预筛后: {len(df)} 只")

        # Step 2: LLM精筛 — 分批交给GLM评分
        batches = []
        for i in range(0, len(df), BATCH_SIZE):
            batch = df.iloc[i : i + BATCH_SIZE]
            batches.append(batch)

        total_batches = len(batches)
        logger.info(f"LLM精筛: {total_batches} 批次, 每批 {BATCH_SIZE} 只, "
                    f"并发数 {MAX_CONCURRENT}")

        # 轮询模型池: 每个batch分配到下一个可用模型
        available_models = [m for m in self.models if m.enabled and m.api_key]
        if not available_models:
            logger.error("无可用模型! 请配置 ZHIPU_API_KEY 或 BAIDU_QIANFAN_API_KEY")
            return [], self._stats

        model_idx = 0
        model_load = {m.name: 0 for m in available_models}

        # 使用信号量控制并发
        sem = asyncio.Semaphore(MAX_CONCURRENT)

        async def process_one(batch: pd.DataFrame, idx: int, model: ModelConfig):
            async with sem:
                if progress_callback:
                    await progress_callback(idx, total_batches, self._stats)
                result = await self._call_model(model, batch, idx)
                self._stats["processed"] += len(batch)
                return result

        # 创建所有任务
        tasks = []
        for i, batch in enumerate(batches):
            model = available_models[model_idx % len(available_models)]
            model_load[model.name] += 1
            model_idx += 1
            tasks.append(process_one(batch, i, model))

        # 并行执行
        logger.info(f"启动 {len(tasks)} 个任务, "
                    f"模型分配: {model_load}")
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 汇总结果
        all_scores = []
        for r in results:
            if isinstance(r, list):
                all_scores.extend(r)
            elif isinstance(r, Exception):
                logger.warning(f"任务异常: {r}")

        # 去重 + 排序
        seen = set()
        unique_scores = []
        for item in all_scores:
            code = item.get("code", "")
            if code and code not in seen:
                seen.add(code)
                # 标准化 score
                score = item.get("score", 0)
                if isinstance(score, str):
                    try:
                        score = float(score)
                    except ValueError:
                        score = 50
                item["score"] = int(min(100, max(0, score)))
                unique_scores.append(item)

        unique_scores.sort(key=lambda x: x.get("score", 0), reverse=True)
        top_stocks = unique_scores[:top_n]

        self._stats["elapsed"] = round(time.time() - t0, 1)
        logger.info(f"预筛完成: {len(all_scores)}条 → 去重{len(unique_scores)} → "
                    f"Top{len(top_stocks)} | 耗时{self._stats['elapsed']}s | "
                    f"错误{self._stats['errors']}次")

        return top_stocks, self._stats

    async def screen_with_deepseek(
        self,
        df_market: pd.DataFrame,
        top_n: int = DEFAULT_TOP_N,
        progress_callback=None,
    ) -> List[Dict]:
        """
        预筛 + DeepSeek 深度分析

        Pipeline:
          1. 免费模型批量预筛 → Top N
          2. DeepSeek 对 Top N 做深度分析 (含基本面/技术面/消息面)
        """
        # Step 1: 预筛
        top_stocks, stats = await self.screen(df_market, top_n, progress_callback)
        if not top_stocks:
            return []

        logger.info(f"预筛完成, 启动 DeepSeek 深度分析 Top {len(top_stocks)} ...")

        # Step 2: DeepSeek 深度分析 (分组并行)
        deep_batch_size = 10  # DeepSeek每批10只, 保证分析质量
        deep_results = []

        deep_sem = asyncio.Semaphore(3)  # DeepSeek API限流更严格

        async def deep_analyze(batch_stocks: List[Dict], batch_idx: int):
            async with deep_sem:
                try:
                    from openai import AsyncOpenAI

                    client = AsyncOpenAI(
                        api_key=os.getenv("DEEPSEEK_API_KEY", ""),
                        base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
                        timeout=60.0,
                    )

                    stocks_text = "\n".join(
                        f"{s['code']} {s.get('name','')} 预筛分{s.get('score',0)} {s.get('reason','')}"
                        for s in batch_stocks
                    )

                    prompt = f"""对以下预筛出的A股进行深度分析, 结合技术面和基本面给出最终评分(0-100)和操作建议(BUY/HOLD/SELL)。

每只股票返回:
- score: 最终评分
- action: BUY/HOLD/SELL
- conviction: 确信度(0-1)
- technical: 技术面一句话
- fundamental: 基本面一句话
- risk: 主要风险

候选股票:
{stocks_text}

返回JSON数组, 按score降序。"""

                    response = await client.chat.completions.create(
                        model=os.getenv("DEEPSEEK_FLASH_MODEL", "deepseek-v4-flash"),
                        messages=[
                            {"role": "system", "content": "你是资深A股分析师。返回JSON格式。"},
                            {"role": "user", "content": prompt},
                        ],
                        temperature=0.3,
                        max_tokens=4000,
                    )

                    content = response.choices[0].message.content.strip()
                    data = json.loads(content)
                    items = data if isinstance(data, list) else data.get("stocks", data.get("results", []))
                    logger.debug(f"DeepSeek batch#{batch_idx}: {len(items)} analyzed")
                    return items

                except Exception as e:
                    logger.warning(f"DeepSeek batch#{batch_idx} error: {e}")
                    return []

        # 分批 + 并行
        deep_tasks = []
        for i in range(0, len(top_stocks), deep_batch_size):
            batch = top_stocks[i : i + deep_batch_size]
            deep_tasks.append(deep_analyze(batch, i))

        deep_all = await asyncio.gather(*deep_tasks, return_exceptions=True)

        for r in deep_all:
            if isinstance(r, list):
                deep_results.extend(r)

        deep_results.sort(key=lambda x: x.get("score", 0), reverse=True)
        logger.info(f"DeepSeek深度分析完成: {len(deep_results)}只")

        return deep_results


# ═══════════════════════════════════════════════════════════════
# 命令行入口
# ═══════════════════════════════════════════════════════════════

async def main():
    import argparse
    # Fix Windows GBK encoding
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

    parser = argparse.ArgumentParser(description="全市场批量预筛")
    parser.add_argument("--top", type=int, default=DEFAULT_TOP_N, help="返回数量")
    parser.add_argument("--deep", action="store_true", help="启用DeepSeek深度分析")
    parser.add_argument("--dry-run", action="store_true", help="仅检查配置,不实际调用")
    args = parser.parse_args()

    # 检查模型配置
    print("=" * 60)
    print("模型池状态:")
    for m in MODEL_POOL:
        has_key = bool(m.api_key)
        status = "[OK]" if has_key else "[MISSING KEY]"
        print(f"  [{m.provider}] {m.name} ({m.model_id}) - {status}")
        if has_key:
            print(f"     QPS~{m.max_qps} | ctx={m.max_context//1000}K | {m.base_url}")

    if args.dry_run:
        return

    # 获取数据
    print("\n加载全市场数据...")
    try:
        import akshare as ak
        df = ak.stock_zh_a_spot_em()
        # 标准化列名
        col_map = {
            "代码": "code", "名称": "name", "最新价": "price", "涨跌幅": "pct_change",
            "涨跌额": "change_amt", "成交量": "volume", "成交额": "amount",
            "振幅": "amplitude", "最高": "high", "最低": "low", "今开": "open",
            "昨收": "prev_close", "量比": "vol_ratio", "换手率": "turnover",
            "市盈率-动态": "pe_ttm", "市净率": "pb", "总市值": "total_mv",
            "流通市值": "float_mv", "60日涨跌幅": "pct_60d",
            "年初至今涨跌幅": "pct_ytd",
        }
        df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
        print(f"获取到 {len(df)} 只股票")
    except Exception as e:
        print(f"数据获取失败: {e}")
        return

    # 执行预筛
    screener = BatchScreener()

    async def show_progress(idx, total, stats):
        pct = (idx + 1) / total * 100
        print(f"\r进度: {idx+1}/{total} ({pct:.0f}%) | "
              f"已处理{stats['processed']}只 | 错误{stats['errors']}", end="")

    if args.deep:
        results = await screener.screen_with_deepseek(df, args.top, show_progress)
    else:
        results, stats = await screener.screen(df, args.top, show_progress)
        print(f"\n耗时: {stats['elapsed']}s | 错误: {stats['errors']}")

    # 输出
    print(f"\n{'='*60}")
    print(f"Top {len(results)} 结果:")
    print(f"{'排名':<5} {'代码':<10} {'名称':<10} {'评分':<6} {'操作':<6} {'原因'}")
    print("-" * 60)
    for i, r in enumerate(results[:30]):
        code = r.get("code", "")
        name = r.get("name", "")
        score = r.get("score", 0)
        action = r.get("action", "")
        reason = r.get("reason", "")[:40]
        print(f"{i+1:<5} {code:<10} {name:<10} {score:<6} {action:<6} {reason}")

    if len(results) > 30:
        print(f"... 还有 {len(results)-30} 只")


if __name__ == "__main__":
    asyncio.run(main())
