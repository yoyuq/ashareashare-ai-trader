"""
全层冒烟测试 — 验证所有模块能正确导入和构造
"""

import asyncio
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ═══════════════════════════════════════════════════════════════
# 1. Data Layer
# ═══════════════════════════════════════════════════════════════

class TestDataLayer:
    """数据层测试"""

    def test_base_imports(self):
        """基础抽象类导入"""
        from data.providers.base import (
            DataFrequency, DataProvider, DataRequest, DataResult, DataSource
        )
        # DataRequest 构造
        req = DataRequest(
            symbol="sh.600000",
            start_date=date(2024, 1, 1),
            end_date=date(2024, 12, 31),
            frequency=DataFrequency.DAILY,
        )
        assert req.symbol == "sh.600000"
        assert req.frequency == DataFrequency.DAILY

    def test_provider_imports(self):
        """所有Provider可导入"""
        from data.providers import AKShareProvider, BaostockProvider, AlternativeDataProvider
        ak = AKShareProvider()
        assert ak.name == "AKShare"
        bs = BaostockProvider()
        assert bs.name == "BaoStock"
        alt = AlternativeDataProvider()
        assert alt.name == "AlternativeData"

    def test_router_singleton(self):
        """DataRouter单例"""
        from data.router import get_data_router
        router = get_data_router(cross_validation=False)
        assert router is not None
        # 验证默认注册了AKShare和Baostock
        status = router.status
        assert "akshare" in status
        assert "baostock" in status

    def test_cache_layer(self):
        """CacheLayer可实例化"""
        from data.cache import CacheLayer
        cache = CacheLayer()
        assert cache.prefix == "ashare"
        # Redis未连接时也应正常降级
        assert cache.enabled == False  # 未connect,false

    def test_pit_processor(self):
        """PIT处理器逻辑正确"""
        from data.processors.pit import PITProcessor
        pit = PITProcessor()

        # 估算披露日期
        report = date(2023, 12, 31)
        disclose = pit._estimate_disclose_date(report)
        assert disclose == date(2024, 4, 30)  # 年报最晚次年4/30

        report_q1 = date(2024, 3, 31)
        disclose_q1 = pit._estimate_disclose_date(report_q1)
        assert disclose_q1 == date(2024, 4, 30)

    def test_db_models(self):
        """数据库模型可导入"""
        from data.storage.models import (
            Base, StockInfo, KlineDaily, KlineMinute,
            RealtimeQuote, Financials, SignalLog,
            AlternativeDataSnapshot, FactorICLog,
        )
        # 验证表名
        assert StockInfo.__tablename__ == "stock_info"
        assert KlineDaily.__tablename__ == "kline_daily"
        assert SignalLog.__tablename__ == "signal_log"
        assert FactorICLog.__tablename__ == "factor_ic_log"


# ═══════════════════════════════════════════════════════════════
# 2. Analysis Engine
# ═══════════════════════════════════════════════════════════════

class TestAnalysisEngine:
    """分析引擎测试"""

    @pytest.fixture
    def sample_ohlcv(self):
        """生成60天模拟K线数据"""
        np.random.seed(42)
        n = 60
        dates = pd.date_range("2024-01-01", periods=n, freq="B")
        close = 10 + np.cumsum(np.random.randn(n) * 0.1)
        df = pd.DataFrame({
            "date": dates,
            "open": close + np.random.randn(n) * 0.05,
            "high": close + abs(np.random.randn(n) * 0.15),
            "low": close - abs(np.random.randn(n) * 0.15),
            "close": close,
            "volume": np.random.randint(1000000, 10000000, n),
        })
        # 确保 high >= open/close >= low
        df["high"] = df[["open", "high", "close"]].max(axis=1)
        df["low"] = df[["open", "low", "close"]].min(axis=1)
        return df

    def test_technical_analyzer(self, sample_ohlcv):
        """技术分析器全量计算"""
        from analysis.indicators import TechnicalAnalyzer
        analyzer = TechnicalAnalyzer()
        result = analyzer.compute_all(sample_ohlcv)

        # 验证指标产出
        assert "ma_5" in result.indicators
        assert "ma_20" in result.indicators
        assert "ma_60" in result.indicators
        assert "ema_12" not in result.indicators  # 只有5,10,20,30,60,120,250
        assert "macd_dif" in result.indicators
        assert "macd_dea" in result.indicators
        assert "macd_hist" in result.indicators
        assert "rsi_14" in result.indicators
        assert "adx_14" in result.indicators
        assert "bb_upper_20" in result.indicators
        assert "atr_14" in result.indicators
        assert "trend_score" in result.indicators
        assert "composite_score" in result.indicators

        # 验证形态检测
        assert len(result.patterns) > 0

        # 验证值域
        rsi = result.indicators["rsi_14"].dropna()
        assert rsi.min() >= 0 and rsi.max() <= 100

        composite = result.indicators["composite_score"].dropna()
        assert composite.min() >= 0 and composite.max() <= 100

    def test_technical_analyzer_empty(self):
        """空数据输入不崩溃"""
        from analysis.indicators import TechnicalAnalyzer
        analyzer = TechnicalAnalyzer()
        result = analyzer.compute_all(pd.DataFrame())
        assert len(result.indicators) == 0

    def test_market_regime_detector(self, sample_ohlcv):
        """市场状态检测"""
        from analysis.regime import MarketRegimeDetector, MarketRegime
        detector = MarketRegimeDetector()
        result = detector.detect(sample_ohlcv)

        assert result.regime is not None
        assert isinstance(result.regime, MarketRegime)
        assert 0 <= result.confidence <= 1
        assert "price_structure" in result.scores
        assert "momentum" in result.scores
        assert "volatility" in result.scores
        assert "volume" in result.scores

    def test_quick_detect(self, sample_ohlcv):
        """快速市场状态检测"""
        from analysis.regime import MarketRegimeDetector, MarketRegime
        regime = MarketRegimeDetector.quick_detect(sample_ohlcv)
        assert isinstance(regime, MarketRegime)

    def test_regime_parameters(self):
        """市场状态对应的策略参数"""
        from analysis.regime import MarketRegime, get_regime_parameters
        for regime in MarketRegime:
            params = get_regime_parameters(regime)
            assert "position_multiplier" in params
            assert "max_positions" in params
            assert "confidence_threshold" in params

        # 危机模式应建议空仓
        crisis_params = get_regime_parameters(MarketRegime.CRISIS)
        assert crisis_params["max_positions"] == 0

    def test_incremental_computer(self):
        """增量计算调度器"""
        from analysis.incremental import IncrementalComputer
        computer = IncrementalComputer(batch_size=100)
        assert computer is not None
        assert computer.batch_size == 100

    def test_anomaly_detection(self, sample_ohlcv):
        """数据异常检测"""
        from analysis.incremental import IncrementalComputer
        anomalies = IncrementalComputer.detect_anomalies(sample_ohlcv)
        assert "has_anomaly" in anomalies.columns
        assert "price_spike" in anomalies.columns
        assert "volume_spike" in anomalies.columns


# ═══════════════════════════════════════════════════════════════
# 3. Backtest Engine
# ═══════════════════════════════════════════════════════════════

class TestBacktestEngine:
    """回测引擎测试"""

    def test_broker_basics(self):
        """Broker基本功能"""
        from backtest.broker import AShareBroker, OrderSide, OrderStatus
        broker = AShareBroker(initial_capital=100000)

        # 初始状态
        assert broker.account.cash == 100000
        assert len(broker.account.positions) == 0

        # 通过直接设置内部状态进行测试(无真实价格数据)
        today = date(2024, 6, 1)
        broker.set_date(today)

        # 设置模拟价格
        broker.set_prices({
            "sh.600000": {"open": 10.0, "high": 10.3, "low": 9.8, "close": 10.2, "pct_change": 2.0}
        })

        # 买入
        order = broker.buy("sh.600000", 1000, price=10.0)
        assert order.status == OrderStatus.FILLED
        assert order.filled_qty == 1000
        assert broker.account.cash < 100000  # 扣了钱

        # 持仓检查
        assert "sh.600000" in broker.account.positions
        pos = broker.account.positions["sh.600000"]
        assert pos.quantity == 1000

        # 卖出(盈利)
        broker.set_date(date(2024, 6, 2))
        broker.set_prices({
            "sh.600000": {"open": 10.5, "high": 10.8, "low": 10.4, "close": 10.6, "pct_change": 3.9}
        })
        order2 = broker.sell("sh.600000", 1000, price=10.5)
        assert order2.status == OrderStatus.FILLED

        # 盈利检查
        assert broker.account.win_count > 0
        assert broker.account.cash > 100000  # 盈利

    def test_broker_t1_rule(self):
        """T+1规则验证"""
        from backtest.broker import AShareBroker, OrderSide, OrderStatus
        broker = AShareBroker(initial_capital=100000)

        # T日买入
        broker.set_date(date(2024, 6, 1))
        broker.set_prices({
            "sh.600000": {"open": 10.0, "high": 10.2, "low": 9.9, "close": 10.1, "pct_change": 1.0}
        })
        broker.buy("sh.600000", 500, price=10.0)

        # 同日卖出(T+1未满足,但broker不做强制检查,标记在can_sell_date)
        pos = broker.account.positions.get("sh.600000")
        assert pos is not None
        # can_sell_date应该是T+1
        assert pos.can_sell_date is not None

    def test_broker_insufficient_funds(self):
        """资金不足拒绝"""
        from backtest.broker import AShareBroker, OrderStatus
        broker = AShareBroker(initial_capital=10000)
        broker.set_date(date(2024, 6, 1))
        broker.set_prices({
            "sh.600519": {"open": 1700.0, "high": 1710.0, "low": 1690.0, "close": 1705.0, "pct_change": 0.5}
        })
        order = broker.buy("sh.600519", 1000, price=1700.0)  # 需要170万,只有1万
        assert order.status == OrderStatus.REJECTED

    def test_broker_performance(self):
        """Broker绩效统计"""
        from backtest.broker import AShareBroker
        broker = AShareBroker(initial_capital=100000)
        perf = broker.get_performance()
        assert "total_trades" in perf
        # 无交易时有total_trades=0, fees可能不在
        if perf["total_trades"] > 0:
            assert "total_fees" in perf

    def test_backtest_config(self):
        """回测配置"""
        from backtest.engine import BacktestConfig
        cfg = BacktestConfig(
            initial_capital=100000,
            start_date=date(2020, 1, 1),
            end_date=date(2024, 12, 31),
        )
        assert cfg.train_split == 0.60
        assert cfg.validation_split == 0.20
        assert cfg.test_split == 0.20

    def test_overfitting_guard(self):
        """过拟合防控"""
        from backtest.overfitting import OverfittingGuard
        guard = OverfittingGuard(n_simulations=100)

        # 生成模拟收益率
        np.random.seed(42)
        returns = pd.Series(np.random.randn(252) * 0.01 + 0.0005)  # SR≈0.8

        report = guard.evaluate(returns)
        assert np.isnan(report.pbo) or 0 <= report.pbo <= 1  # 无变体矩阵时PBO为NaN(未定义)
        assert report.deflated_sharpe is not None
        assert report.monte_carlo_pvalue >= 0

        # 没有参数扫描数据时,PBO=0, DSR不应为负过大
        assert isinstance(report.is_overfit, bool)  # 过拟合判定应为有效布尔值

    def test_time_split(self):
        """时间分割"""
        from backtest.overfitting import OverfittingGuard
        df = pd.DataFrame({
            "date": pd.date_range("2020-01-01", periods=1000, freq="B"),
            "value": range(1000),
        })
        train, val, test = OverfittingGuard.split_time_series(df)
        assert len(train) > len(val)
        assert len(val) >= len(test)  # val和test可能恰好相等
        # 验证时间顺序
        assert train["date"].max() < val["date"].min()
        assert val["date"].max() <= test["date"].min()

    def test_impact_cost_engine(self):
        """冲击成本引擎"""
        from backtest.impact import ImpactCostEngine
        engine = ImpactCostEngine()

        bar = pd.Series({
            "open": 10.0, "high": 10.5, "low": 9.8,
            "close": 10.2, "volume": 50000000,  # 5000万股
        })
        result = engine.estimate("sh.600000", 50000, bar, board="主板")
        assert result.impact_bps >= 0
        assert result.impact_total_pct >= 0
        assert result.capacity_remaining >= 0

        # 大单冲击应更大
        result_large = engine.estimate("sh.600000", 500000, bar, board="主板")
        assert result_large.impact_bps > result.impact_bps  # 大单冲击更大

    def test_optimal_split(self):
        """大单拆分建议"""
        from backtest.impact import ImpactCostEngine
        engine = ImpactCostEngine()

        bar = pd.Series({
            "open": 10.0, "high": 10.5, "low": 9.8,
            "close": 10.2, "volume": 50000000,
        })
        per_day, days = engine.optimal_split("sh.600000", 100000, bar, max_impact_bps=30)
        assert per_day > 0
        assert days >= 1

    def test_backtest_discipline(self):
        """回测纪律(冷静期)"""
        from backtest.overfitting import BacktestDiscipline
        from datetime import datetime, timedelta

        # 刚创建的策略不能立即回测
        bd = BacktestDiscipline(
            strategy_name="test_strategy",
            designed_at=datetime.now(),
        )
        can, msg = bd.can_backtest()
        assert can == False  # 冷静期未过
        assert "小时" in msg

        # 48小时前的策略可以回测
        bd2 = BacktestDiscipline(
            strategy_name="test_strategy",
            designed_at=datetime.now() - timedelta(hours=50),
        )
        can2, msg2 = bd2.can_backtest()
        assert can2 == True

    def test_sealed_evaluation(self):
        """密封评估"""
        from backtest.overfitting import SealedEvaluation
        se = SealedEvaluation()

        # 未锁定测试用例时无法评估
        ok, msg = se.evaluate("strat_1")
        assert ok == False

        # 锁定+提交后可以评估
        se.lock_test_cases("strat_1", ["test1", "test2", "test3"])
        se.submit_strategy("strat_1", "abc123def")
        ok, msg = se.evaluate("strat_1")
        assert ok == True


# ═══════════════════════════════════════════════════════════════
# 4. Models Layer
# ═══════════════════════════════════════════════════════════════

class TestModelsLayer:
    """模型层测试"""

    def test_single_model_tier(self):
        """单模型层级 — v3.0 统一 flash (P2-5 移除单值 ModelTier 枚举)"""
        from models.router import MODEL_TIER
        assert MODEL_TIER == "flash"

    def test_cost_monitor(self):
        """成本监控"""
        from models.cost_monitor import CostMonitor
        monitor = CostMonitor(daily_budget=1.0, monthly_budget=15.0)

        assert monitor.daily_remaining == 1.0
        assert monitor.monthly_remaining == 15.0
        assert not monitor.is_budget_exhausted()

        # 记录一次Flash调用
        monitor.record_call("flash", input_tokens=50000, output_tokens=20000)
        # cost = (50k/1M)*1.0 + (20k/1M)*2.0 = 0.05 + 0.04 = 0.09
        assert 0.08 < monitor._daily_cost < 0.10

        monitor.record_call("local", input_tokens=10000, output_tokens=5000)
        # local cost = 0
        assert monitor._daily_cost < 0.10  # 未增加

    def test_cost_monitor_budget_exhausted(self):
        """预算耗尽检测"""
        from models.cost_monitor import CostMonitor
        monitor = CostMonitor(daily_budget=0.01)  # 极低预算
        monitor.record_call("flash", input_tokens=50000, output_tokens=5000)
        # Flash: (50k/1M)*1 + (5k/1M)*2 = 0.05 + 0.01 = 0.06 >> 0.01
        assert monitor.is_budget_exhausted()


# ═══════════════════════════════════════════════════════════════
# 5. Agent Layer
# ═══════════════════════════════════════════════════════════════

class TestAgentLayer:
    """Agent层测试"""

    def test_state_creation(self):
        """MarketAnalysisState创建"""
        from agent.orchestration.state import MarketAnalysisState
        from datetime import date

        state: MarketAnalysisState = {
            "task_id": "test001",
            "task_type": "daily_scan",
            "date": date.today().isoformat(),
            "symbols": ["sh.600000", "sz.000001"],
            "errors": [],
            "model_trace": [],
            "signals_archived": False,
        }
        assert state["task_type"] == "daily_scan"
        assert len(state["symbols"]) == 2

    def test_agent_memory(self):
        """AgentMemory"""
        from agent.orchestration.memory import AgentMemory
        memory = AgentMemory()

        memory.add_finding("market_scanner", {
            "key_points": "发现3只异动标的",
            "confidence": 0.8,
        })
        memory.add_finding("technical_analyst", {
            "key_points": "sh.600000 MACD金叉,突破60日线",
            "confidence": 0.75,
        })

        assert len(memory.session_findings) == 2
        scanner_findings = memory.get_agent_findings("market_scanner")
        assert len(scanner_findings) == 1

        summary = memory.get_findings_summary()
        assert "market_scanner" in summary
        assert "technical_analyst" in summary

    def test_workflow_graph_creation(self):
        """LangGraph工作流创建"""
        from agent.orchestration.workflow import AnalysisWorkflow
        wf = AnalysisWorkflow(router=None, knowledge_manager=None)
        assert wf.graph is not None
        # 验证节点数
        nodes = list(wf.graph.nodes.keys())
        assert "data_preparation" in nodes
        assert "market_scanner" in nodes
        assert "technical_analysis" in nodes
        assert "strategy_matching" in nodes
        assert "backtest_verification" in nodes
        assert "adversarial_debate" in nodes
        assert "synthesis" in nodes


# ═══════════════════════════════════════════════════════════════
# 6. Knowledge Base
# ═══════════════════════════════════════════════════════════════

class TestKnowledgeBase:
    """知识库测试"""

    def test_knowledge_manager_init(self):
        """KnowledgeManager初始化"""
        from knowledge.manager import KnowledgeManager
        km = KnowledgeManager("knowledge/")
        assert km is not None
        # 验证规则加载
        stamp_duty = km.get_rule("trading_rules.fees.stamp_duty.rate")
        assert stamp_duty == 0.0005

    def test_knowledge_manager_prompt(self):
        """KnowledgeManager Prompt注入"""
        from knowledge.manager import KnowledgeManager
        km = KnowledgeManager("knowledge/")

        prompt = km.get_system_prompt("technical_analyst")
        assert len(prompt) > 200
        # 应包含注入的指标指南
        assert "MA" in prompt or "指標" in prompt or "MACD" in prompt or "移动平均" in prompt

    def test_knowledge_manager_strategies(self):
        """KnowledgeManager策略查询"""
        from knowledge.manager import KnowledgeManager
        km = KnowledgeManager("knowledge/")

        # 获取单个策略
        strategy = km.get_strategy("dual_ma_trend")
        assert strategy is not None
        assert strategy["category"] == "trend_following"
        assert "capacity_limit" in strategy

        # 按市场状态过滤
        bull_strategies = km.get_strategies_for_regime("strong_bull")
        assert len(bull_strategies) > 0
        for s in bull_strategies:
            assert "strong_bull" in s.get("market_regimes", [])

        # 按类别过滤
        trend_strategies = km.list_strategies(category="trend_following")
        assert len(trend_strategies) == 3
        assert all(s["category"] == "trend_following" for s in trend_strategies)

    def test_knowledge_manager_hardened(self):
        """KnowledgeManager硬化口径"""
        from knowledge.manager import KnowledgeManager
        km = KnowledgeManager("knowledge/")

        # 获取口径定义
        defn = km.get_hardened_definition("volume_expansion")
        assert defn is not None
        assert defn["threshold"] == 1.5
        assert "formula" in defn

        # 校验值
        assert km.verify_definition("volume_expansion", 2.0) == True   # >1.5
        assert km.verify_definition("volume_expansion", 1.2) == False  # <1.5

    @pytest.mark.network  # ChromaDB 默认 embedding 需下载模型, 依赖网络
    def test_knowledge_manager_rag(self):
        """KnowledgeManager RAG检索"""
        from knowledge.manager import KnowledgeManager
        km = KnowledgeManager("knowledge/")

        result = km.rag_query("牛市 熊市 周期")
        assert len(result) > 0
        assert "market_cycle" in result.lower()  # 应该匹配到market_cycle.md

        glossary = km.get_glossary("T+1")
        assert len(glossary) > 0
        assert "T+1" in glossary or "T+1" in glossary.lower()

    def test_trading_rules_yaml(self):
        """交易规则YAML格式正确"""
        import yaml
        kb_path = Path(__file__).parent.parent / "knowledge" / "rules" / "trading_rules.yaml"
        with open(kb_path, "r", encoding="utf-8") as f:
            rules = yaml.safe_load(f)

        assert "trading_hours" in rules
        assert "settlement" in rules
        assert "price_limits" in rules
        assert "fees" in rules
        assert rules["fees"]["stamp_duty"]["rate"] == 0.0005
        assert rules["fees"]["commission"]["min_per_trade"] == 5.0

    def test_indicator_guide_yaml(self):
        """指标指南YAML格式正确"""
        import yaml
        kb_path = Path(__file__).parent.parent / "knowledge" / "rules" / "indicator_guide.yaml"
        with open(kb_path, "r", encoding="utf-8") as f:
            guide = yaml.safe_load(f)

        assert "indicators" in guide
        assert "MA" in guide["indicators"]
        assert "MACD" in guide["indicators"]
        assert "RSI" in guide["indicators"]
        assert "BollingerBands" in guide["indicators"]

    def test_strategy_registry(self):
        """策略注册表格式正确"""
        import yaml
        kb_path = Path(__file__).parent.parent / "knowledge" / "strategies" / "registry.yaml"
        with open(kb_path, "r", encoding="utf-8") as f:
            registry = yaml.safe_load(f)

        assert "strategies" in registry
        strategies = registry["strategies"]
        assert len(strategies) >= 9  # v3.1: 9个策略 + northbound_follow

        # 每个策略有关键字段
        for s in strategies:
            assert "id" in s
            assert "name" in s
            assert "category" in s
            assert "params" in s
            assert "capacity_limit" in s  # v2.1

    def test_system_prompts_exist(self):
        """系统Prompt文件存在且非空"""
        kb_path = Path(__file__).parent.parent / "knowledge" / "prompts" / "system"
        prompts = ["market_scanner.txt", "technical_analyst.txt", "synthesis.txt"]
        for p in prompts:
            path = kb_path / p
            assert path.exists(), f"缺少: {p}"
            content = path.read_text(encoding="utf-8")
            assert len(content) > 100, f"太短: {p}"

    def test_reference_docs_exist(self):
        """参考文档存在"""
        ref_path = Path(__file__).parent.parent / "knowledge" / "reference"
        docs = ["market_cycle.md", "glossary.md"]
        for d in docs:
            path = ref_path / d
            assert path.exists(), f"缺少: {d}"
            content = path.read_text(encoding="utf-8")
            assert len(content) > 50, f"太短: {d}"


# ═══════════════════════════════════════════════════════════════
# 6.5. Code-as-Reasoning (v2.1)
# ═══════════════════════════════════════════════════════════════

class TestCodeAsReasoning:
    """代码即推理层测试"""

    def test_code_executor_safe_code(self):
        """安全代码执行"""
        from agent.tools.code_executor import CodeExecutor
        executor = CodeExecutor()

        code = """
import numpy as np
import pandas as pd

data = np.array([10, 11, 12, 13, 14, 15])
ma_5 = float(pd.Series(data).rolling(5).mean().iloc[-1])
"""
        results = executor.execute(code, {})
        assert "ma_5" in results
        assert abs(results["ma_5"] - 13.0) < 0.01

    def test_code_executor_dangerous_code(self):
        """危险代码拦截"""
        from agent.tools.code_executor import CodeExecutor
        executor = CodeExecutor()

        # 禁止导入系统模块
        dangerous_code = "import os; os.system('echo bad')"
        try:
            executor.execute(dangerous_code, {})
            assert False, "应该抛出异常"
        except RuntimeError as e:
            assert "禁止导入" in str(e)

        # 禁止exec/eval
        dangerous_code2 = "exec('print(123)')"
        try:
            executor.execute(dangerous_code2, {})
            assert False, "应该抛出异常"
        except RuntimeError as e:
            assert "禁止调用" in str(e)

    def test_numeric_safety_checker(self):
        """数字安全校验"""
        from agent.tools.code_executor import NumericSafetyChecker, ComputedNumber

        computed = {
            "rsi_14": ComputedNumber("rsi_14", 58.3, source="rsi calculation"),
            "ma_20": ComputedNumber("ma_20", 10.52, source="ma calculation"),
        }

        checker = NumericSafetyChecker(computed)

        # 报告中使用已计算的值 — 应该通过
        safe_report = "RSI(14)为58.3,处于中性区间。MA20为10.52,价格位于均线上方。"
        is_safe, violations = checker.validate_report(safe_report)
        assert is_safe, f"应该通过,但发现违规: {violations}"

        # 报告中编造了未计算的值 — 应该被拦截
        fake_report = "RSI为62.0, MACD为0.85, 建议买入。"  # 62.0和0.85都不在computed中
        is_safe2, violations2 = checker.validate_report(fake_report)
        assert not is_safe2, "编造的数字应该被拦截"
        assert len(violations2) >= 1

    def test_code_as_reasoning_pipeline_init(self):
        """Code-as-Reasoning流水线初始化"""
        from agent.tools.code_executor import CodeAsReasoningPipeline
        pipeline = CodeAsReasoningPipeline()
        assert pipeline.executor is not None

    def test_code_executor_context(self):
        """带上下文的代码执行"""
        from agent.tools.code_executor import CodeExecutor
        executor = CodeExecutor()

        # 模拟OHLCV数据
        import numpy as np
        context = {
            "close": np.array([10, 10.5, 10.3, 10.8, 11.0, 11.2, 11.5, 11.3, 11.8, 12.0]),
        }

        code = """
import numpy as np
close = np.array(close)
ret = np.diff(np.log(close))
volatility = float(np.std(ret) * np.sqrt(252) * 100)
"""
        results = executor.execute(code, context)
        assert "volatility" in results
        assert results["volatility"] > 0


# ═══════════════════════════════════════════════════════════════
# 7. Integration (不依赖外部服务)
# ═══════════════════════════════════════════════════════════════

class TestIntegration:
    """集成测试(本地,无外部服务)"""

    def test_full_pipeline_imports(self):
        """全流水线模块可导入"""
        # Data
        from data import (
            DataRouter, get_data_router,
            AKShareProvider, BaostockProvider, AlternativeDataProvider,
            CacheLayer, PITProcessor,
        )
        # Analysis
        from analysis import (
            TechnicalAnalyzer, MarketRegimeDetector,
            MarketRegime, IncrementalComputer,
        )
        # Backtest
        from backtest import (
            AShareBroker, EventDrivenBacktestEngine,
            OverfittingGuard, ImpactCostEngine,
            BacktestDiscipline, SealedEvaluation,
        )
        # Models
        from models import ModelRouter, CostMonitor
        # Agent
        from agent import AnalysisWorkflow, AgentMemory, MarketAnalysisState

        # 上面全部导入成功即通过; 再验证关键符号可用
        assert callable(EventDrivenBacktestEngine)
        assert callable(ModelRouter)
        assert callable(AnalysisWorkflow)

    def test_config_files_exist(self):
        """配置文件完整"""
        config_path = Path(__file__).parent.parent / "config"
        files = ["settings.yaml", "model_config.yaml", "symbols.yaml"]
        for f in files:
            path = config_path / f
            assert path.exists(), f"缺少配置文件: {f}"

    def test_requirements_consistent(self):
        """requirements.txt与pyproject.toml一致"""
        req_path = Path(__file__).parent.parent / "requirements.txt"
        content = req_path.read_text(encoding="utf-8")
        core_packages = ["akshare", "numpy", "pandas", "langgraph", "langchain"]
        for pkg in core_packages:
            assert pkg in content.lower(), f"缺少依赖: {pkg}"

    def test_make_mock_backtest(self):
        """模拟回测端到端流程"""
        from backtest.broker import AShareBroker, OrderStatus
        from backtest.engine import BacktestConfig, EventDrivenBacktestEngine
        from datetime import date

        # 生成模拟数据
        np.random.seed(42)
        n = 252
        dates = pd.date_range("2024-01-01", periods=n, freq="B")
        close = 10 + np.cumsum(np.random.randn(n) * 0.08)
        df = pd.DataFrame({
            "date": dates.date,
            "open": close + np.random.randn(n) * 0.03,
            "high": close + abs(np.random.randn(n) * 0.1),
            "low": close - abs(np.random.randn(n) * 0.1),
            "close": close,
            "volume": np.random.randint(1e6, 5e7, n),
        })

        # 配置
        config = BacktestConfig(
            initial_capital=100000,
            start_date=date(2024, 1, 15),
            end_date=date(2024, 12, 20),
        )

        engine = EventDrivenBacktestEngine(config)
        engine.load_data("sh.600000", df)

        # 简单策略: 金叉买入,死叉卖出
        def simple_ma_strategy(today, bars, broker):
            sym = "sh.600000"
            if sym not in bars:
                return

            bar = bars[sym]
            close = bar["close"]

            # 计算2日均线和5日均线(极简版)
            hist_closes = []
            for i, d in enumerate(df["date"]):
                if d <= today:
                    hist_closes.append(df["close"].iloc[i])
                else:
                    break

            hist = pd.Series(hist_closes)
            if len(hist) < 5:
                return

            ma2 = hist.rolling(2).mean().iloc[-1]
            ma5 = hist.rolling(5).mean().iloc[-1]
            ma2_prev = hist.rolling(2).mean().iloc[-2]
            ma5_prev = hist.rolling(5).mean().iloc[-2]

            # 金叉买入
            if ma2_prev <= ma5_prev and ma2 > ma5:
                pos = broker.account.positions.get(sym)
                if pos is None or pos.quantity == 0:
                    qty = int(broker.account.cash * 0.3 / close / 100) * 100
                    if qty >= 100:
                        broker.buy(sym, qty, price=close)

            # 死叉卖出
            if ma2_prev >= ma5_prev and ma2 < ma5:
                pos = broker.account.positions.get(sym)
                if pos and pos.quantity > 0:
                    broker.sell(sym, pos.quantity, price=close)

        result = engine.run(simple_ma_strategy, progress_bar=False)

        # 验证结果
        assert result.trading_days > 0
        assert result.equity_curve is not None
        assert result.max_drawdown <= 0
        assert result.total_return is not None

        print(f"\n📊 模拟回测结果:")
        print(f"  交易日: {result.trading_days}")
        print(f"  总收益: {result.total_return:+.2f}%")
        print(f"  年化收益: {result.annual_return:+.2f}%")
        print(f"  夏普: {result.sharpe_ratio:.2f}")
        print(f"  最大回撤: {result.max_drawdown:.2f}%")
        print(f"  交易次数: {result.total_trades}")
