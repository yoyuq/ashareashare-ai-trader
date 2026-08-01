"""
Config-Driven Strategy Executor — 配置驱动策略执行 (v3.0-competition, 2026 Best Practice)

基于 DolphinDB x 北大 2026 最佳实践:
  "AI生成JSON策略配置 → 标准化模板执行 → 成功率84%→99%"

核心理念:
  - 策略逻辑由JSON配置文件定义,而非硬编码
  - 执行引擎统一处理: 数据获取/信号生成/回测/风控
  - 新增策略只需添加JSON配置,无需修改代码

与 strategy registry (registry.yaml) 的关系:
  - registry.yaml: 策略元数据 (名称/类别/参数/适用体制)
  - strategy_executor.py: 策略执行引擎 (读取配置 → 标准化执行)
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger


@dataclass
class StrategyConfig:
    """策略配置 (JSON-serializable, AI可生成)"""
    id: str
    name: str
    category: str                           # trend_following | mean_reversion | momentum | ...
    description: str

    # 信号生成规则
    entry_conditions: List[Dict[str, Any]] = field(default_factory=list)
    exit_conditions: List[Dict[str, Any]] = field(default_factory=list)

    # 参数
    params: Dict[str, Any] = field(default_factory=dict)

    # 风控
    risk_controls: Dict[str, Any] = field(default_factory=dict)

    # 元数据
    market_regimes: List[str] = field(default_factory=list)
    capacity_limit: int = 10_000_000       # 资金容量 (元)
    timeframe: str = "daily"
    version: str = "1.0.0"

    def to_dict(self) -> dict:
        return {
            "id": self.id, "name": self.name, "category": self.category,
            "description": self.description,
            "entry_conditions": self.entry_conditions,
            "exit_conditions": self.exit_conditions,
            "params": self.params,
            "risk_controls": self.risk_controls,
            "market_regimes": self.market_regimes,
            "capacity_limit": self.capacity_limit,
            "timeframe": self.timeframe, "version": self.version,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "StrategyConfig":
        return cls(
            id=d["id"], name=d["name"], category=d.get("category", ""),
            description=d.get("description", ""),
            entry_conditions=d.get("entry_conditions", []),
            exit_conditions=d.get("exit_conditions", []),
            params=d.get("params", {}),
            risk_controls=d.get("risk_controls", {}),
            market_regimes=d.get("market_regimes", []),
            capacity_limit=d.get("capacity_limit", 10_000_000),
            timeframe=d.get("timeframe", "daily"),
            version=d.get("version", "1.0.0"),
        )


class StrategyExecutor:
    """
    配置驱动策略执行引擎

    用法:
        # 1. 从YAML/JSON加载策略配置
        config = StrategyConfig.from_dict(yaml_config)

        # 2. AI也可以生成策略配置
        config = StrategyConfig.from_dict(llm_generated_json)

        # 3. 标准化执行
        executor = StrategyExecutor(data_router)
        result = await executor.execute(config, symbol="sh.600519")

    优势:
      - AI可以生成和优化策略配置 (JSON格式,结构化)
      - 执行引擎统一处理所有策略 (代码复用,减少bug)
      - 新增策略零代码修改
    """

    def __init__(self, data_router=None):
        self.router = data_router

    async def execute(
        self,
        config: StrategyConfig,
        symbol: str,
        start_date: str = "2024-01-01",
        end_date: Optional[str] = None,
        initial_capital: float = 100000,
    ) -> Dict[str, Any]:
        """
        Execute strategy with real backtest engine (v3.1).

        Flow:
          1. Validate config
          2. Fetch historical data
          3. Run event-driven backtest
          4. Compute performance metrics
          5. Return structured results
        """
        # 1. Validate config
        validation = self.validate_config(config)
        if not validation["valid"]:
            return {"error": "Config validation failed", "issues": validation["issues"]}

        # 2. Fetch data
        data = None
        if self.router:
            try:
                from data.providers.base import DataFrequency, DataRequest
                from datetime import date as dt_date
                end = dt_date.fromisoformat(end_date) if end_date else dt_date.today()
                start = dt_date.fromisoformat(start_date)
                req = DataRequest(
                    symbol=symbol,
                    start_date=start,
                    end_date=end,
                    frequency=DataFrequency.DAILY,
                )
                result = await self.router.get_daily_kline(req)
                if result and not result.data.empty:
                    data = result.data
            except Exception as e:
                logger.warning(f"Data fetch failed for {symbol}: {e}")

        if data is None:
            return {
                "strategy_id": config.id,
                "strategy_name": config.name,
                "symbol": symbol,
                "error": "Data unavailable",
                "config_valid": True,
            }

        # 3. Run event-driven backtest
        try:
            from backtest.engine import BacktestConfig, EventDrivenBacktestEngine
            from datetime import date as dt_date

            # Build strategy function from config
            def config_strategy(today, bars, broker):
                self._execute_config_strategy(today, bars, broker, config)

            bt_config = BacktestConfig(
                initial_capital=initial_capital,
                symbol=symbol,
                start_date=start_date,
                end_date=end_date or dt_date.today().isoformat(),
            )
            engine = EventDrivenBacktestEngine(bt_config)
            engine.load_data(symbol, data)
            result = engine.run(config_strategy)

            return {
                "strategy_id": config.id,
                "strategy_name": config.name,
                "symbol": symbol,
                "config_valid": True,
                "total_return_pct": round(result.total_return, 2),
                "annual_return_pct": round(result.annual_return, 2),
                "sharpe_ratio": round(result.sharpe_ratio, 3),
                "max_drawdown_pct": round(result.max_drawdown, 2),
                "win_rate": round(result.win_rate, 3),
                "profit_factor": round(result.profit_factor, 2),
                "total_trades": result.total_trades,
                "metrics_raw": result.to_dict() if hasattr(result, 'to_dict') else {},
                "message": "Backtest completed",
            }
        except ImportError as e:
            return {
                "strategy_id": config.id,
                "strategy_name": config.name,
                "symbol": symbol,
                "config_valid": True,
                "message": f"Backtest engine not available: {e}",
            }
        except Exception as e:
            logger.warning(f"Backtest failed for {config.id} on {symbol}: {e}")
            return {
                "strategy_id": config.id,
                "strategy_name": config.name,
                "symbol": symbol,
                "config_valid": True,
                "error": str(e),
                "message": "Backtest execution failed",
            }

    def _execute_config_strategy(self, today, bars, broker, config: StrategyConfig):
        """
        v3.1: Execute config-driven strategy within event-driven backtest engine.

        Called by EventDrivenBacktestEngine.run() for each bar.
        Translates declarative entry/exit conditions into actual orders.
        """
        symbol = list(bars.keys())[0] if bars else None
        if symbol is None:
            return

        bar = bars[symbol]
        pos = broker.get_position(symbol)

        # Check risk controls
        rc = config.risk_controls
        stop_loss_pct = rc.get("stop_loss_pct", 0.05) if rc else 0.05
        take_profit_pct = rc.get("take_profit_pct", 0.15) if rc else 0.15

        # Exit: stop loss / take profit
        if pos and pos.quantity > 0:
            pnl_pct = (bar.close - pos.avg_cost) / pos.avg_cost
            if pnl_pct <= -stop_loss_pct:
                broker.sell(symbol, bar.close, pos.quantity, reason="stop_loss")
                return
            if pnl_pct >= take_profit_pct:
                broker.sell(symbol, bar.close, pos.quantity, reason="take_profit")
                return

        # Entry: evaluate conditions
        if pos is None or pos.quantity == 0:
            if self._evaluate_entry(config.entry_conditions, bar, bars):
                quantity = broker.calculate_position_size(
                    symbol, bar.close, config.risk_controls
                )
                if quantity > 0:
                    broker.buy(symbol, bar.close, quantity)

        # Exit: evaluate exit conditions (beyond SL/TP)
        if pos and pos.quantity > 0:
            if self._evaluate_exit(config.exit_conditions, bar, bars, pos):
                broker.sell(symbol, bar.close, pos.quantity, reason="signal_exit")

    def _evaluate_entry(self, conditions: list, bar, bars: dict) -> bool:
        """Evaluate entry conditions against current bar"""
        if not conditions:
            return False
        for cond in conditions:
            ctype = cond.get("type", "")
            if ctype == "ma_cross":
                if not self._check_ma_cross(bar, bars, cond.get("params", {})):
                    return False
            elif ctype == "rsi_threshold":
                rsi_val = getattr(bar, 'rsi_14', 50)
                threshold = cond.get("params", {}).get("value", 30)
                operator = cond.get("params", {}).get("operator", "<")
                if operator == "<" and rsi_val > threshold:
                    return False
                if operator == ">" and rsi_val < threshold:
                    return False
            elif ctype == "volume_surge":
                vol_ratio = getattr(bar, 'vol_ratio_5', 1.0)
                if vol_ratio < cond.get("params", {}).get("min_ratio", 1.5):
                    return False
            # All conditions must pass (AND logic)
        return True

    def _evaluate_exit(self, conditions: list, bar, bars: dict, pos) -> bool:
        """Evaluate exit conditions"""
        if not conditions:
            return False
        for cond in conditions:
            ctype = cond.get("type", "")
            if ctype == "ma_cross_reverse":
                if not self._check_ma_cross_reverse(bar, bars, cond.get("params", {})):
                    return False
                return True
            elif ctype == "rsi_overbought":
                if getattr(bar, 'rsi_14', 50) > cond.get("params", {}).get("value", 70):
                    return True
        return False

    @staticmethod
    def _check_ma_cross(bar, bars, params) -> bool:
        """Check if fast MA crossed above slow MA"""
        fast = params.get("fast", 5)
        slow = params.get("slow", 20)
        ma_fast = getattr(bar, f'ma_{fast}', 0)
        ma_slow = getattr(bar, f'ma_{slow}', 0)
        return ma_fast > 0 and ma_slow > 0 and ma_fast > ma_slow

    @staticmethod
    def _check_ma_cross_reverse(bar, bars, params) -> bool:
        """Check if fast MA crossed below slow MA"""
        fast = params.get("fast", 5)
        slow = params.get("slow", 20)
        ma_fast = getattr(bar, f'ma_{fast}', 0)
        ma_slow = getattr(bar, f'ma_{slow}', 0)
        return ma_fast > 0 and ma_slow > 0 and ma_fast < ma_slow

    def validate_config(self, config: StrategyConfig) -> Dict[str, Any]:
        """
        验证策略配置完整性

        检查项:
          - 必填字段
          - 入场条件至少1个
          - 风控参数合理性
          - 适用体制非空
        """
        issues = []

        if not config.id:
            issues.append("缺少策略ID")
        if not config.name:
            issues.append("缺少策略名称")
        if not config.entry_conditions:
            issues.append("至少需要1个入场条件")
        if not config.market_regimes:
            issues.append("未指定适用市场体制")

        # 风控检查
        rc = config.risk_controls
        if rc:
            if rc.get("stop_loss_pct", 0) > 0.15:
                issues.append("止损比例>15%过高,建议5%-10%")
            if rc.get("position_limit", 0) > 0.5:
                issues.append("单票仓位上限>50%过高,建议≤25%")

        return {"valid": len(issues) == 0, "issues": issues}

    @staticmethod
    def load_configs_from_registry(registry_path: str = "knowledge/strategies/registry.yaml") -> List[StrategyConfig]:
        """从策略注册表加载所有策略配置"""
        import yaml
        path = Path(registry_path)
        if not path.exists():
            return []

        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        configs = []
        for s in data.get("strategies", []):
            config = StrategyConfig(
                id=s["id"], name=s["name"],
                category=s.get("category", ""),
                description=s.get("description", ""),
                params=s.get("params", {}),
                market_regimes=s.get("market_regimes", []),
                capacity_limit=s.get("capacity_limit", 10_000_000),
                timeframe=s.get("timeframes", ["daily"])[0] if s.get("timeframes") else "daily",
                version=s.get("version", "1.0.0"),
                # 从参数自动生成入场/出场条件
                entry_conditions=[{"type": "signal_match", "strategy_id": s["id"]}],
                exit_conditions=[
                    {"type": "stop_loss", "pct": s.get("params", {}).get("stop_loss_pct", 0.05)},
                    {"type": "signal_reverse", "strategy_id": s["id"]},
                ],
                risk_controls={
                    "stop_loss_pct": s.get("params", {}).get("stop_loss_pct", 0.05),
                    "position_limit": 0.25,
                    "max_holding_days": 20,
                },
            )
            configs.append(config)

        return configs

    @staticmethod
    def config_to_json_schema() -> dict:
        """
        返回策略配置的 JSON Schema

        用于LLM生成策略配置时的约束 —
        LLM按照此Schema生成JSON,执行引擎直接执行
        """
        return {
            "type": "object",
            "required": ["id", "name", "category", "entry_conditions", "market_regimes"],
            "properties": {
                "id": {"type": "string", "description": "策略唯一标识"},
                "name": {"type": "string", "description": "策略名称"},
                "category": {
                    "type": "string",
                    "enum": ["trend_following", "mean_reversion", "momentum",
                             "multi_factor", "ashare_special", "flow_following"],
                },
                "description": {"type": "string"},
                "entry_conditions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "type": {"type": "string", "enum": ["ma_cross", "rsi_level",
                                     "bb_touch", "volume_surge", "signal_match"]},
                            "params": {"type": "object"},
                        },
                    },
                },
                "exit_conditions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "type": {"type": "string", "enum": ["stop_loss",
                                     "take_profit", "trailing_stop", "time_stop",
                                     "signal_reverse"]},
                            "pct": {"type": "number"},
                        },
                    },
                },
                "params": {"type": "object"},
                "risk_controls": {
                    "type": "object",
                    "properties": {
                        "stop_loss_pct": {"type": "number", "minimum": 0.01, "maximum": 0.15},
                        "position_limit": {"type": "number", "maximum": 0.5},
                        "max_holding_days": {"type": "integer", "maximum": 60},
                    },
                },
                "market_regimes": {
                    "type": "array",
                    "items": {"type": "string", "enum": ["strong_bull", "weak_bull",
                             "range_bound", "weak_bear", "strong_bear", "crisis"]},
                },
                "capacity_limit": {"type": "integer", "minimum": 0},
            },
        }
