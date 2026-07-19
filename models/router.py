"""
模型路由器 — 三层漏斗混合调度 (v2.0+)

Tier 1: Ollama Qwen3-4B    — 本地免费, 处理60%日常任务
Tier 2: DeepSeek V4-Flash  — ¥1-2/M, 处理30%中等任务
Tier 3: DeepSeek V4-Pro    — ¥3-6/M, 处理10%复杂任务

核心特性:
  - 任务复杂度自动评估→自动路由
  - 高峰时段(9:00-12:00, 14:00-18:00)自动降级
  - 完全降级链: Pro→Flash→Local
  - 日预算控制(默认¥1/天)
  - 成本实时追踪
"""

import asyncio
import os
from dataclasses import dataclass, field
from datetime import datetime, time
from enum import Enum
from typing import Any, Dict, List, Optional

from loguru import logger


class ModelTier(Enum):
    """模型层级"""
    LOCAL = "local"      # Ollama 本地
    FLASH = "flash"      # DeepSeek V4-Flash
    PRO = "pro"          # DeepSeek V4-Pro


@dataclass
class ModelConfig:
    """单个模型配置"""
    tier: ModelTier
    provider: str
    model_name: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    max_context: int = 32768
    max_output: int = 4096
    temperature: float = 0.1
    request_timeout: int = 60
    # 定价 (¥/M tokens)
    input_price: float = 0.0
    output_price: float = 0.0

    @property
    def is_local(self) -> bool:
        return self.tier == ModelTier.LOCAL


@dataclass
class RouteResult:
    """路由结果"""
    tier: ModelTier
    model_name: str
    response: str
    input_tokens: int
    output_tokens: int
    cost: float
    latency_ms: float
    used_fallback: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


class ModelRouter:
    """
    三层模型路由器

    使用方式:
        router = ModelRouter(config)
        result = await router.route(
            task_type="technical_analysis",
            messages=[{"role": "user", "content": "分析MACD金叉信号"}],
        )
    """

    # 任务复杂度 → 默认路由层级
    TASK_ROUTING = {
        # === Tier 1: LOCAL (Ollama Qwen3-4B) ===
        "indicator_read": ModelTier.LOCAL,
        "kline_describe": ModelTier.LOCAL,
        "news_summary": ModelTier.LOCAL,
        "text_classify": ModelTier.LOCAL,
        "simple_qa": ModelTier.LOCAL,
        "data_format": ModelTier.LOCAL,

        # === Tier 2: FLASH (DeepSeek V4-Flash) ===
        "technical_analysis": ModelTier.FLASH,
        "strategy_match": ModelTier.FLASH,
        "signal_verify": ModelTier.FLASH,
        "multi_factor_analysis": ModelTier.FLASH,
        "backtest_interpret": ModelTier.FLASH,
        "regime_analysis": ModelTier.FLASH,

        # === Tier 3: PRO (DeepSeek V4-Pro) ===
        "daily_synthesis": ModelTier.PRO,
        "adversarial_debate": ModelTier.PRO,
        "strategy_optimize": ModelTier.PRO,
        "market_outlook": ModelTier.PRO,
        "risk_assessment": ModelTier.PRO,
        "portfolio_advice": ModelTier.PRO,
    }

    # 高峰时段 (北京时间)
    PEAK_WINDOWS = [
        (time(9, 0), time(12, 0)),
        (time(14, 0), time(18, 0)),
    ]

    def __init__(
        self,
        daily_budget: float = 1.0,
        monthly_budget: float = 15.0,
    ):
        self.daily_budget = daily_budget
        self.monthly_budget = monthly_budget
        self._daily_cost = 0.0
        self._monthly_cost = 0.0
        self._daily_reset_date = datetime.now().date()

        # 快速健康标记
        self._ollama_available: Optional[bool] = None  # None=未检测, True/False

        # 初始化各层客户端
        self._ollama_client = None
        self._deepseek_client = None
        self._init_clients()

        self._call_log: List[RouteResult] = []

    async def _detect_ollama(self) -> bool:
        """快速检测Ollama是否可用(仅检测一次)"""
        if self._ollama_available is not None:
            return self._ollama_available

        try:
            from ollama import AsyncClient
            client = AsyncClient(
                host=os.getenv("OLLAMA_HOST", "http://localhost:11434")
            )
            # 快速ping (1秒超时)
            import asyncio
            await asyncio.wait_for(client.list(), timeout=1.0)
            self._ollama_client = client
            self._ollama_available = True
            logger.info("Ollama可用")
            return True
        except Exception:
            self._ollama_available = False
            logger.info("Ollama不可用,本地层跳过(所有任务将由DeepSeek处理)")
            return False

    def _init_clients(self):
        """初始化LLM客户端"""
        self._ollama_client = None
        self._ollama_available = None  # 延迟检测

        try:
            from openai import AsyncOpenAI
            self._deepseek_client = AsyncOpenAI(
                api_key=os.getenv("DEEPSEEK_API_KEY", ""),
                base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
            )
            logger.info("DeepSeek客户端初始化完成")
        except Exception as e:
            logger.warning(f"DeepSeek初始化失败: {e}")

        try:
            from openai import AsyncOpenAI
            self._deepseek_client = AsyncOpenAI(
                api_key=os.getenv("DEEPSEEK_API_KEY", ""),
                base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
            )
            logger.info("DeepSeek客户端初始化完成")
        except Exception as e:
            logger.warning(f"DeepSeek初始化失败: {e}")

    # ═══════════════════════════════════════════════════════════════
    # 主路由入口
    # ═══════════════════════════════════════════════════════════════

    async def route(
        self,
        messages: List[Dict[str, str]],
        task_type: str = "simple_qa",
        force_tier: Optional[ModelTier] = None,
        max_retries: int = 2,
    ) -> RouteResult:
        """
        智能路由到最合适的模型

        Args:
            messages: 对话消息列表
            task_type: 任务类型(决定路由层级)
            force_tier: 强制指定层级(跳过自动路由)
            max_retries: 最大重试次数

        Returns:
            RouteResult
        """
        # 1. 确定目标层级
        if force_tier:
            tier = force_tier
        else:
            tier = self._determine_tier(task_type)

        # 2. 高峰降级
        if self._is_peak_hour() and tier != ModelTier.LOCAL:
            # 高峰时降一级
            if tier == ModelTier.PRO:
                tier = ModelTier.FLASH
                logger.debug("高峰降级: PRO→FLASH")
            elif tier == ModelTier.FLASH:
                tier = ModelTier.LOCAL
                logger.debug("高峰降级: FLASH→LOCAL")

        # 3. 预算检查
        if tier != ModelTier.LOCAL and self._daily_cost >= self.daily_budget * 0.9:
            logger.warning(
                f"日预算已用{self._daily_cost:.2f}/{self._daily_budget:.2f}元,"
                "强制使用本地模型"
            )
            tier = ModelTier.LOCAL

        # 4. 执行(带降级重试)
        return await self._execute_with_fallback(tier, messages, max_retries)

    # ═══════════════════════════════════════════════════════════════
    # 执行与降级
    # ═══════════════════════════════════════════════════════════════

    async def _execute_with_fallback(
        self,
        tier: ModelTier,
        messages: List[Dict[str, str]],
        max_retries: int,
    ) -> RouteResult:
        """执行请求,失败时沿降级链回退"""
        errors = []

        # 快速检测Ollama状态(仅首次)
        ollama_ok = await self._detect_ollama()

        # 降级链: 如果Ollama不可用,从链中移除LOCAL
        full_chain = {
            ModelTier.PRO: [ModelTier.PRO, ModelTier.FLASH, ModelTier.LOCAL],
            ModelTier.FLASH: [ModelTier.FLASH, ModelTier.LOCAL],
            ModelTier.LOCAL: [ModelTier.LOCAL],
        }
        chain = full_chain.get(tier, [ModelTier.LOCAL])

        # 如果Ollama不可用,移除LOCAL (跳过无意义的3次重试)
        if not ollama_ok:
            chain = [t for t in chain if t != ModelTier.LOCAL]
            if not chain:
                chain = [ModelTier.FLASH]  # 至少有一个

        for attempt_tier in chain:
            last_error = None
            for retry in range(max_retries + 1):
                try:
                    start = datetime.now()
                    response, usage = await self._call_model(attempt_tier, messages)
                    latency = (datetime.now() - start).total_seconds() * 1000

                    cost = self._calculate_cost(attempt_tier, usage)
                    self._track_cost(cost)

                    result = RouteResult(
                        tier=attempt_tier,
                        model_name=self._get_model_name(attempt_tier),
                        response=response,
                        input_tokens=usage.get("input_tokens", 0),
                        output_tokens=usage.get("output_tokens", 0),
                        cost=cost,
                        latency_ms=round(latency, 0),
                        used_fallback=(attempt_tier != tier),
                        metadata={"task_tier": tier.value},
                    )

                    self._call_log.append(result)
                    return result

                except Exception as e:
                    last_error = e
                    logger.warning(
                        f"{attempt_tier.value} 调用失败(重试{retry+1}/{max_retries}): {e}"
                    )
                    if retry < max_retries:
                        await asyncio.sleep(1.0 * (retry + 1))  # 指数退避

            errors.append(f"{attempt_tier.value}: {last_error}")

        raise RuntimeError(f"所有模型层级均失败: {'; '.join(errors)}")

    async def _call_model(
        self,
        tier: ModelTier,
        messages: List[Dict[str, str]],
    ) -> tuple[str, dict]:
        """调用具体模型"""
        if tier == ModelTier.LOCAL:
            return await self._call_ollama(messages)
        else:
            return await self._call_deepseek(tier, messages)

    async def _call_ollama(self, messages: List[Dict[str, str]]) -> tuple[str, dict]:
        """调用Ollama本地模型"""
        if self._ollama_client is None:
            raise RuntimeError("Ollama客户端未初始化")

        model = os.getenv("OLLAMA_MODEL", "qwen3:4b")

        response = await self._ollama_client.chat(
            model=model,
            messages=messages,
            options={
                "temperature": 0.1,
                "num_predict": 2048,
            },
        )

        content = response["message"]["content"]
        usage = {
            "input_tokens": response.get("prompt_eval_count", 0),
            "output_tokens": response.get("eval_count", 0),
        }
        return content, usage

    async def _call_deepseek(
        self,
        tier: ModelTier,
        messages: List[Dict[str, str]],
    ) -> tuple[str, dict]:
        """调用DeepSeek API"""
        if self._deepseek_client is None:
            raise RuntimeError("DeepSeek客户端未初始化")

        model_map = {
            ModelTier.FLASH: "deepseek-chat",      # V4-Flash
            ModelTier.PRO: "deepseek-reasoner",     # V4-Pro
        }
        model = model_map.get(tier, "deepseek-chat")

        response = await self._deepseek_client.chat.completions.create(
            model=model,
            messages=messages,  # type: ignore
            temperature=0.3 if tier == ModelTier.FLASH else 0.6,
            max_tokens=16384 if tier == ModelTier.FLASH else 32768,
        )

        content = response.choices[0].message.content or ""
        usage = {
            "input_tokens": response.usage.prompt_tokens if response.usage else 0,
            "output_tokens": response.usage.completion_tokens if response.usage else 0,
        }
        return content, usage

    # ═══════════════════════════════════════════════════════════════
    # 路由逻辑
    # ═══════════════════════════════════════════════════════════════

    def _determine_tier(self, task_type: str) -> ModelTier:
        """根据任务类型确定路由层级"""
        return self.TASK_ROUTING.get(task_type, ModelTier.LOCAL)

    def _is_peak_hour(self) -> bool:
        """检查是否在北京时间高峰时段"""
        now = datetime.now()
        current_time = now.time()

        for start, end in self.PEAK_WINDOWS:
            if start <= current_time <= end:
                return True
        return False

    # ═══════════════════════════════════════════════════════════════
    # 成本追踪
    # ═══════════════════════════════════════════════════════════════

    def _calculate_cost(self, tier: ModelTier, usage: dict) -> float:
        """计算本次调用成本(元)"""
        if tier == ModelTier.LOCAL:
            return 0.0

        # DeepSeek V4 定价 (2026年7月, ¥/百万tokens)
        pricing = {
            ModelTier.FLASH: (1.0, 2.0),   # (input, output)
            ModelTier.PRO: (3.0, 6.0),
        }
        input_price, output_price = pricing.get(tier, (0, 0))

        input_tokens = usage.get("input_tokens", 0) / 1_000_000
        output_tokens = usage.get("output_tokens", 0) / 1_000_000
        cost = input_tokens * input_price + output_tokens * output_price

        # 高峰×2
        if self._is_peak_hour():
            cost *= 2.0

        return cost

    def _track_cost(self, cost: float):
        """追踪累计成本"""
        today = datetime.now().date()
        if today != self._daily_reset_date:
            self._daily_cost = 0.0
            self._daily_reset_date = today

        self._daily_cost += cost
        self._monthly_cost += cost

    # ═══════════════════════════════════════════════════════════════
    # 监控接口
    # ═══════════════════════════════════════════════════════════════

    @property
    def daily_cost(self) -> float:
        return round(self._daily_cost, 4)

    @property
    def monthly_cost(self) -> float:
        return round(self._monthly_cost, 4)

    @property
    def budget_remaining(self) -> float:
        return round(self.daily_budget - self._daily_cost, 4)

    def cost_summary(self) -> Dict[str, Any]:
        """成本摘要"""
        tier_stats = {}
        for r in self._call_log:
            tier = r.tier.value
            if tier not in tier_stats:
                tier_stats[tier] = {"count": 0, "cost": 0, "tokens": 0}
            tier_stats[tier]["count"] += 1
            tier_stats[tier]["cost"] += r.cost
            tier_stats[tier]["tokens"] += r.input_tokens + r.output_tokens

        return {
            "daily_cost": self.daily_cost,
            "monthly_cost": self.monthly_cost,
            "daily_budget": self.daily_budget,
            "budget_remaining": self.budget_remaining,
            "total_calls": len(self._call_log),
            "by_tier": tier_stats,
        }

    def _get_model_name(self, tier: ModelTier) -> str:
        names = {
            ModelTier.LOCAL: os.getenv("OLLAMA_MODEL", "qwen3:4b"),
            ModelTier.FLASH: "deepseek-chat",
            ModelTier.PRO: "deepseek-reasoner",
        }
        return names.get(tier, "unknown")
