"""
模型路由 — 单模型统一调度 (v3.0)

v3.0 (2026-08): 全量迁移 DeepSeek V4-Flash。
删除 Ollama 本地层与 V4-Pro 层, 所有 LLM 调用统一走 deepseek-v4-flash。

核心特性:
  - 全任务统一路由 → deepseek-v4-flash
  - 日/月预算追踪 (默认¥1/天)
  - 成本实时计算 (含 DeepSeek 峰谷定价高峰×2)
  - 指数退避重试
"""

import asyncio
import os
from dataclasses import dataclass, field
from datetime import datetime, time
from enum import Enum
from typing import Any, Dict, List, Optional

from loguru import logger

# 北京时区 (v3.0): 高峰定价/日预算按 A 股交易日边界
from timeutil import now_cn, today_cn

# 模型名 — 与 models/__init__.py 保持同步, 经环境变量可覆盖
DEEPSEEK_FLASH_MODEL = os.getenv("DEEPSEEK_FLASH_MODEL", "deepseek-v4-flash")
# 兼容别名 (v2 曾有 PRO 层; v3 统一 flash, 保留导出避免破坏既有 import)
DEEPSEEK_PRO_MODEL = os.getenv("DEEPSEEK_PRO_MODEL", DEEPSEEK_FLASH_MODEL)

# 预算状态落盘路径 (v3.1): 进程重启不丢失成本累计
_BUDGET_STATE_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "simulation_data", "budget_state.json",
)


class ModelTier(Enum):
    """模型层级 — v3.0 起统一单层 (全部 FLASH)"""
    FLASH = "flash"


@dataclass
class ModelConfig:
    """单个模型配置 (v3.0: 统一 flash)"""
    tier: ModelTier = ModelTier.FLASH
    provider: str = "deepseek"
    model_name: str = DEEPSEEK_FLASH_MODEL
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    max_context: int = 131072
    max_output: int = 16384
    temperature: float = 0.3
    request_timeout: int = 90
    # 定价 (¥/M tokens)
    input_price: float = 1.0
    output_price: float = 2.0

    @property
    def is_local(self) -> bool:
        return False


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


class BudgetExhaustedError(RuntimeError):
    """日预算耗尽 — BUDGET_HARD_CUT=1 模式下 route() 拒绝付费调用时抛出"""


class ModelRouter:
    """
    单模型路由器 (v3.0: 全部调用统一走 DeepSeek V4-Flash)

    使用方式:
        router = ModelRouter(config)
        result = await router.route(
            task_type="technical_analysis",
            messages=[{"role": "user", "content": "分析MACD金叉信号"}],
        )

    预算控制 (v3.0):
      - 默认软提醒: 预算达 90% 时日志告警, 不阻断 (兼容现有工作流)
      - BUDGET_HARD_CUT=1: 预算耗尽时抛 BudgetExhaustedError (硬切断, 强制止付)
    """

    # 任务类型 → 统一路由到 FLASH
    # (v2 曾按复杂度分 LOCAL/PRO/FLASH 三层, 2026-08 全量迁移到 V4-Flash)
    TASK_ROUTING = {
        # === 常规/快速任务 ===
        "indicator_read": ModelTier.FLASH,
        "kline_describe": ModelTier.FLASH,
        "news_summary": ModelTier.FLASH,
        "text_classify": ModelTier.FLASH,
        "simple_qa": ModelTier.FLASH,
        "data_format": ModelTier.FLASH,
        "market_scan_prefilter": ModelTier.FLASH,
        # === 分析/评估 ===
        "technical_analysis": ModelTier.FLASH,
        "fundamental_analysis": ModelTier.FLASH,
        "strategy_match": ModelTier.FLASH,
        "signal_verify": ModelTier.FLASH,
        "multi_factor_analysis": ModelTier.FLASH,
        "backtest_interpret": ModelTier.FLASH,
        "regime_analysis": ModelTier.FLASH,
        "macro_event_analysis": ModelTier.FLASH,
        # === 决策/辩论 ===
        "daily_synthesis": ModelTier.FLASH,
        "adversarial_debate": ModelTier.FLASH,
        "bull_bear_research": ModelTier.FLASH,
        "judge_verdict": ModelTier.FLASH,
        "strategy_optimize": ModelTier.FLASH,
        "market_outlook": ModelTier.FLASH,
        "risk_assessment": ModelTier.FLASH,
        "portfolio_advice": ModelTier.FLASH,
    }

    # 高峰时段 (北京时间) — 对应 DeepSeek 峰谷定价(高峰×2)
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
        self._daily_reset_date = today_cn()
        self._monthly_reset_month = today_cn().strftime("%Y-%m")

        self._deepseek_client = None
        self._init_clients()
        # v3.0: 预算硬切断开关 (BUDGET_HARD_CUT=1 → 耗尽时拒绝付费调用)
        self._hard_cut = os.getenv("BUDGET_HARD_CUT", "").lower() in ("1", "true", "yes")

        self._call_log: List[RouteResult] = []
        # v3.1: 预算状态落盘 (重启不丢失)
        self._load_budget_state()

    def _init_clients(self):
        """初始化 DeepSeek 客户端"""
        try:
            from openai import AsyncOpenAI
            self._deepseek_client = AsyncOpenAI(
                api_key=os.getenv("DEEPSEEK_API_KEY", ""),
                base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
                timeout=float(os.getenv("DEEPSEEK_TIMEOUT", "90")),
                max_retries=int(os.getenv("DEEPSEEK_MAX_RETRIES", "3")),
            )
            logger.info("DeepSeek客户端就绪")
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
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> RouteResult:
        """
        统一调用 deepseek-v4-flash

        Args:
            messages: 对话消息列表
            task_type: 任务类型 (v3.0 起仅用于统计, 不再影响模型选择)
            force_tier: 兼容保留 (仅接受 ModelTier.FLASH)
            max_retries: 最大重试次数
            temperature: 采样温度 (None 用默认 0.3)
            max_tokens: 最大输出 token (None 用默认 16384)

        Returns:
            RouteResult
        """
        self._check_budget_hard_cut()
        tier = self._resolve_tier(task_type, force_tier)
        return await self._execute_with_fallback(
            tier, messages, max_retries,
            temperature=temperature, max_tokens=max_tokens,
        )

    async def route_with_tools(
        self,
        messages: List[Dict[str, str]],
        tools: List[Dict] = None,
        task_type: str = "simple_qa",
        force_tier: Optional[ModelTier] = None,
        max_retries: int = 2,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> RouteResult:
        """
        带工具调用的路由 — ChatAgent 专用

        v3.0 与 route() 一致, 统一走 flash (工具调用能力完整)。
        """
        tier = self._resolve_tier(task_type, force_tier)
        return await self._execute_with_fallback(
            tier, messages, max_retries, tools=tools,
            temperature=temperature, max_tokens=max_tokens,
        )

    def _check_budget_hard_cut(self):
        """BUDGET_HARD_CUT 模式下预算耗尽 → 抛 BudgetExhaustedError 强制止付"""
        if self._hard_cut and self._daily_cost >= self.daily_budget:
            raise BudgetExhaustedError(
                f"日预算已耗尽 ({self._daily_cost:.2f}/{self.daily_budget:.2f}元), "
                "BUDGET_HARD_CUT=1 已启用, 拒绝付费调用"
            )

    def _resolve_tier(self, task_type: str, force_tier: Optional[ModelTier] = None) -> ModelTier:
        """解析目标层级 — v3.0 统一为 FLASH"""
        if force_tier is not None:
            return force_tier
        tier = self.TASK_ROUTING.get(task_type, ModelTier.FLASH)

        # 预算软提醒 (硬切断由成本监控层负责, 见 /api/v1/cost 与 workflow)
        if self._daily_cost >= self.daily_budget * 0.9:
            logger.warning(
                f"日预算已用{self._daily_cost:.2f}/{self.daily_budget:.2f}元,"
                "建议关注成本"
            )
        return tier

    # ═══════════════════════════════════════════════════════════════
    # 执行与重试
    # ═══════════════════════════════════════════════════════════════

    async def _execute_with_fallback(
        self,
        tier: ModelTier,
        messages: List[Dict[str, str]],
        max_retries: int,
        tools: List[Dict] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> RouteResult:
        """执行请求,失败时按指数退避重试"""
        last_error = None
        for retry in range(max_retries + 1):
            try:
                start = datetime.now()
                response, usage = await self._call_model(
                    messages, tools=tools,
                    temperature=temperature, max_tokens=max_tokens,
                )
                latency = (datetime.now() - start).total_seconds() * 1000

                cost = self._calculate_cost(usage)
                self._track_cost(cost)

                result = RouteResult(
                    tier=ModelTier.FLASH,
                    model_name=DEEPSEEK_FLASH_MODEL,
                    response=response,
                    input_tokens=usage.get("input_tokens", 0),
                    output_tokens=usage.get("output_tokens", 0),
                    cost=cost,
                    latency_ms=round(latency, 0),
                    used_fallback=False,
                    metadata={"task_tier": tier.value},
                )

                self._call_log.append(result)
                return result

            except Exception as e:
                last_error = e
                logger.warning(
                    f"flash 调用失败(重试{retry+1}/{max_retries}): {e}"
                )
                if retry < max_retries:
                    await asyncio.sleep(1.0 * (retry + 1))  # 指数退避

        raise RuntimeError(f"所有模型调用均失败: {last_error}")

    async def _call_model(
        self,
        messages: List[Dict[str, str]],
        tools: List[Dict] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> tuple[str, dict]:
        """调用 DeepSeek V4-Flash"""
        if self._deepseek_client is None:
            raise RuntimeError("DeepSeek客户端未初始化")

        kwargs = {
            "model": DEEPSEEK_FLASH_MODEL,
            "messages": messages,  # type: ignore
            "temperature": temperature if temperature is not None else 0.3,
            "max_tokens": max_tokens if max_tokens is not None else 16384,
        }

        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
            # v3.0: 工具调用关闭思考模式 — 规避 DeepSeek 多轮工具调用必须回传
            # reasoning_content 否则 400 的问题, 且工具执行是确定性操作无需思维链
            # (降低延迟与成本)。来源: api-docs.deepseek.com/guides/thinking_mode/
            kwargs["extra_body"] = {"thinking": {"type": "disabled"}}

        response = await self._deepseek_client.chat.completions.create(**kwargs)

        choice = response.choices[0]
        content = choice.message.content or ""
        if not content:
            # v3.0: V4 思考模式(max_tokens 不足/思考占满) 时 content 可能为空,
            # 回退到 reasoning_content 作为响应文本
            content = getattr(choice.message, "reasoning_content", "") or ""

        # 工具调用: 将 tool_calls 序列化到 content 中传递
        if choice.message.tool_calls:
            import json
            tool_calls_data = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in choice.message.tool_calls
            ]
            content = json.dumps({
                "_tool_calls": tool_calls_data,
                "content": content,
            }, ensure_ascii=False)

        usage = {
            "input_tokens": response.usage.prompt_tokens if response.usage else 0,
            "output_tokens": response.usage.completion_tokens if response.usage else 0,
            # v3.0: 捕获缓存命中 tokens, 按 1/50 计价
            "cache_hit_tokens": (
                getattr(response.usage, "prompt_cache_hit_tokens", 0)
                if response.usage else 0
            ),
        }
        return content, usage

    # ═══════════════════════════════════════════════════════════════
    # 成本追踪
    # ═══════════════════════════════════════════════════════════════

    def _is_peak_hour(self) -> bool:
        """检查是否在北京时间高峰时段 (DeepSeek 峰谷定价: 高峰×2)"""
        if os.getenv("SKIP_PEAK_HOUR", "").lower() in ("1", "true", "yes"):
            return False
        now = now_cn()  # v3.0: 高峰窗口按北京时间判定
        current_time = now.time()
        for start, end in self.PEAK_WINDOWS:
            if start <= current_time <= end:
                return True
        return False

    def _calculate_cost(self, usage: dict) -> float:
        """计算本次调用成本(元) — DeepSeek V4-Flash 定价 (2026-07, ¥/M tokens)。

        缓存命中输入价 ¥0.02/M (为未命中的 1/50); 高峰时段×2 (峰谷定价, 工作日
        9:00-12:00 / 14:00-18:00 北京时间)。
        注意: 2026-08-06 DeepSeek 预告新一轮全面涨价, 幅度未公布 — 生效后需在此
        同步更新 input_price/output_price (与 config/model_config.yaml 保持一致)。
        """
        input_price, output_price = 1.0, 2.0
        cache_price = 0.02
        total_in = usage.get("input_tokens", 0) / 1_000_000
        cache_in = usage.get("cache_hit_tokens", 0) / 1_000_000
        fresh_in = max(0.0, total_in - cache_in)
        output_tokens = usage.get("output_tokens", 0) / 1_000_000
        cost = fresh_in * input_price + cache_in * cache_price + output_tokens * output_price

        # 高峰×2 (对应 DeepSeek 峰谷定价)
        if self._is_peak_hour():
            cost *= 2.0

        return cost

    def _track_cost(self, cost: float):
        """追踪累计成本 (v3.1: 月预算软提醒 + 状态落盘)"""
        today = today_cn()  # v3.0: 按北京日期跨日重置
        if today != self._daily_reset_date:
            self._daily_cost = 0.0
            self._daily_reset_date = today

        month = today.strftime("%Y-%m")
        if month != self._monthly_reset_month:
            self._monthly_cost = 0.0
            self._monthly_reset_month = month

        self._daily_cost += cost
        self._monthly_cost += cost

        # 月预算软提醒 (硬切断仅针对日预算, 见 _check_budget_hard_cut)
        if self._monthly_cost >= self.monthly_budget:
            logger.warning(
                f"月预算已用{self._monthly_cost:.2f}/{self.monthly_budget:.2f}元,"
                "建议关注成本"
            )

        self._save_budget_state()

    def _load_budget_state(self):
        """加载落盘的成本状态 (跨进程恢复)"""
        import json
        try:
            with open(_BUDGET_STATE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            if data.get("daily_reset_date") == today_cn():
                self._daily_cost = float(data.get("daily_cost", 0.0))
            self._monthly_cost = float(data.get("monthly_cost", 0.0))
            # 月成本仅在当月有效
            if data.get("monthly_reset_month") != self._monthly_reset_month:
                self._monthly_cost = 0.0
        except (FileNotFoundError, ValueError, OSError):
            pass

    def _save_budget_state(self):
        """落盘成本状态"""
        import json
        try:
            os.makedirs(os.path.dirname(_BUDGET_STATE_FILE), exist_ok=True)
            with open(_BUDGET_STATE_FILE, "w", encoding="utf-8") as f:
                json.dump({
                    "daily_cost": self._daily_cost,
                    "monthly_cost": self._monthly_cost,
                    "daily_reset_date": self._daily_reset_date,
                    "monthly_reset_month": self._monthly_reset_month,
                }, f, ensure_ascii=False)
        except OSError as e:
            logger.warning(f"预算状态落盘失败: {e}")

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
