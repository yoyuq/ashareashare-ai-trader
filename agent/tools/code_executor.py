"""
🆕 v2.1 代码即推理架构 — Code-as-Reasoning Pipeline

核心原则:
  Numbers are CODE-computed. Narratives are LLM-assisted.

4阶段执行:
  Phase 1: LLM生成分析计划(不生成数字,只生成计算指令)
  Phase 2: Python代码执行所有数值计算
  Phase 3: 结构化数据注入LLM上下文(数字已锁定)
  Phase 4: LLM生成最终报告(附数据溯源表)

数字安全保证:
  - NumericSafetyChecker 验证报告中每个数字都能在computed_data中找到
  - 任何"凭空出现"的数字触发拦截
"""

import ast
import hashlib
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from loguru import logger


@dataclass
class ComputedNumber:
    """计算出的数值 — 带溯源信息"""
    name: str                    # 变量名
    value: float                 # 数值
    unit: str = ""               # 单位 (%, 元, 倍, ...)
    source: str = ""             # 来源: "ma_20 calculation", "rsi calculation"
    formula: str = ""            # 计算公式
    computed_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def fingerprint(self) -> str:
        """生成唯一指纹 (value + source)"""
        raw = f"{self.name}:{self.value:.4f}:{self.source}"
        return hashlib.sha256(raw.encode()).hexdigest()[:12]


@dataclass
class CodePlan:
    """LLM生成的计算计划"""
    tasks: List[Dict[str, Any]] = field(default_factory=list)
    raw_plan: str = ""


@dataclass
class ProvenanceReport:
    """数据溯源报告"""
    computed_numbers: Dict[str, ComputedNumber] = field(default_factory=dict)
    execution_log: List[str] = field(default_factory=list)
    num_safe: bool = True
    violations: List[str] = field(default_factory=list)


def _restricted_import(name, *args, **kwargs):
    """受限导入: 仅允许白名单顶层模块 numpy/pandas (及其子模块), 其余一律拒绝。

    沙箱代码可能写 `import numpy as np`, 因此保留 __import__ 但收窄为白名单。
    即便借助 fromlist 拿到子模块对象, AST 审计层已拦截 dunder 与危险属性访问
    (如 .os / .popen / .__dict__), 无法据此逃逸。
    """
    root = str(name).split(".")[0]
    if root not in ("numpy", "pandas"):
        raise ImportError(f"禁止导入: {name}。沙箱仅允许 numpy / pandas")
    return __import__(name, *args, **kwargs)


class CodeExecutor:
    """
    Python代码执行引擎 (受控沙箱)

    只有白名单中的函数可用:
      - numpy: 数学运算
      - pandas: 数据处理
      - ta/pandas_ta: 技术指标
      - 内置: abs, min, max, round, sum, len

    禁止:
      - 文件I/O (open, os, pathlib)
      - 网络请求 (requests, urllib, akshare)
      - 系统调用 (subprocess, os.system)
      - 导入任意模块
    """

    # 安全白名单 — __import__ 收窄为仅 numpy/pandas 的受限版本;
    # 绝不放入 getattr/eval/globals/vars 等可用于内省逃逸的内置。
    SAFE_BUILTINS = {
        "abs": abs, "min": min, "max": max, "round": round,
        "sum": sum, "len": len, "range": range, "enumerate": enumerate,
        "zip": zip, "sorted": sorted, "filter": filter, "map": map,
        "int": int, "float": float, "str": str, "bool": bool,
        "list": list, "dict": dict, "tuple": tuple, "set": set,
        "True": True, "False": False, "None": None,
        "print": print, "__import__": _restricted_import,
    }

    # 属性访问黑名单叶子名 — 拦截经白名单模块内部对象的逃逸链
    # (如 pd.io.common.os.popen / np.lib.os.system)
    _DANGEROUS_ATTRS = {
        "os", "sys", "subprocess", "shutil", "socket", "signal", "ctypes",
        "popen", "system", "exec", "eval", "compile", "execfile",
        "globals", "vars", "open", "input", "breakpoint",
        "load", "loads", "__import__",
    }

    ALLOWED_MODULES = {"numpy": "np", "pandas": "pd"}

    def __init__(self):
        self._execution_count = 0

    def execute(
        self,
        code: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        在受控沙箱中执行Python代码

        Args:
            code: Python代码字符串
            context: 预置变量 (如 ohlcv_df, indicators)

        Returns:
            {变量名: 计算结果值}

        Raises:
            RuntimeError: 代码包含不安全操作
        """
        self._execution_count += 1

        # 检查安全性
        self._audit_code(code)

        # 准备执行环境
        safe_globals = {"__builtins__": self.SAFE_BUILTINS}
        safe_globals.update({
            alias: __import__(mod)
            for mod, alias in self.ALLOWED_MODULES.items()
        })

        safe_locals = {}
        if context:
            safe_locals.update(context)

        # 执行
        try:
            exec(code, safe_globals, safe_locals)

            # 提取计算结果(排除内置变量和DataFrame)
            results = {}
            for k, v in safe_locals.items():
                if k.startswith("_"):
                    continue
                if isinstance(v, pd.DataFrame):
                    continue
                if isinstance(v, (int, float, np.floating, np.integer)):
                    results[k] = float(v)
                elif isinstance(v, (list, tuple)) and len(v) <= 100:
                    try:
                        results[k] = [float(x) for x in v]
                    except (TypeError, ValueError):
                        results[k] = str(v)
                elif isinstance(v, str) and len(v) < 1000:
                    results[k] = v

            return results

        except Exception as e:
            logger.error(f"代码执行失败 (exec #{self._execution_count}): {e}")
            raise

    def _audit_code(self, code: str):
        """审计代码安全性"""
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            raise RuntimeError(f"代码语法错误: {e}")

        for node in ast.walk(tree):
            # 禁止 dunder 属性访问 — 拦截对象内省逃逸
            # (().__class__.__bases__[0].__subclasses__() 等全部依赖 dunder 属性)
            if isinstance(node, ast.Attribute):
                if node.attr.startswith("__") and node.attr.endswith("__"):
                    raise RuntimeError(f"禁止访问特殊属性: {node.attr}")
                # 拦截经白名单模块内部对象到达的危险叶子 (pd.io.common.os 等)
                if node.attr in self._DANGEROUS_ATTRS:
                    raise RuntimeError(f"禁止访问危险属性: {node.attr}")

            # 禁止import
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                if isinstance(node, ast.ImportFrom):
                    module = node.module or ""
                else:
                    module = node.names[0].name if node.names else ""

                # 允许白名单
                if module.split(".")[0] not in self.ALLOWED_MODULES:
                    raise RuntimeError(
                        f"禁止导入: {module}。只允许: {list(self.ALLOWED_MODULES.keys())}"
                    )

            # 禁止函数调用中的危险操作
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Attribute):
                    if isinstance(node.func.value, ast.Name):
                        name = node.func.value.id
                        if name in ("os", "subprocess", "sys", "shutil"):
                            raise RuntimeError(f"禁止调用系统模块: {name}")

            # 禁止exec/eval
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id in ("exec", "eval", "compile", "__import__"):
                        raise RuntimeError(f"禁止调用: {node.func.id}")


class NumericSafetyChecker:
    """
    数字安全校验器

    验证LLM生成的报告中的每个数字都可以在computed_data中找到对应值。
    如果发现"凭空出现"的数字 → 拦截并标记。
    """

    # 中文数字正则: 匹配 "12.5%", "3.2倍", "¥150.00", "0.85" 等
    # 排除: 指标参数 (如 RSI(14) 里的 14), 日期/年份, 序号
    NUMBER_PATTERN = re.compile(
        r'(?<![a-zA-Z\(\d])([-+]?\d+\.?\d*)\s*(%|倍|元|亿|万|点|bps)?(?![a-zA-Z\)])'
    )

    # 已知指标参数上下文: RSI(14), ATR(14), MA(20), EMA(12), MACD(12,26,9)
    # 这些括号中的数字是指标参数，不是计算输出。
    # v5.6 P1-9: 支持多参数 (MACD(12,26,9)) — 此前只匹配单参数 \d+, 导致
    # "MACD(12,26,9)" 里的 "26" 被误判为未溯源数字。
    INDICATOR_PARAM_PATTERN = re.compile(
        r'(RSI|ATR|MA|EMA|SMA|MACD|BB|KDJ|CCI|ADX|OBV)\s*\(\s*[\d,\s]+\s*\)'
    )

    # v5.6 P1-9: 按量级绝对容差 — 小数字用绝对容差下限, 大数字按相对比例缩放,
    # 避免此前单一相对容差对量级小的数字过严 (浮点误差占比被放大)。
    ABS_TOLERANCE = 0.05    # 绝对容差下限
    REL_TOLERANCE = 0.005   # 相对容差 (量级大的数字按比例)

    def __init__(self, computed_numbers: Dict[str, ComputedNumber]):
        self._computed = computed_numbers
        # 建立数值索引: {(value_key): ComputedNumber}
        self._value_index: Dict[float, ComputedNumber] = {}
        for cn in computed_numbers.values():
            self._value_index[round(cn.value, 4)] = cn

    def validate_report(self, report_text: str) -> Tuple[bool, List[str]]:
        """
        验证报告中的所有数字是否有溯源

        Args:
            report_text: LLM生成的报告文本

        Returns:
            (is_safe, violations)
        """
        violations = []

        # 先找出所有指标参数上下文的位置，标记其中的数字为"豁免"
        exempt_positions = set()
        for m in self.INDICATOR_PARAM_PATTERN.finditer(report_text):
            # 括号内的数字是指标参数,豁免检查
            exempt_positions.update(range(m.start(), m.end()))

        for match in self.NUMBER_PATTERN.finditer(report_text):
            try:
                value = float(match.group(1))
                unit = match.group(2) if match.lastindex and match.lastindex >= 2 else ""

                # 跳过豁免位置(指标参数中的数字)
                if match.start() in exempt_positions or any(
                    pos in exempt_positions
                    for pos in range(match.start(), match.end())
                ):
                    continue

                # 跳过太小的数字(可能是序号、年份等)
                if abs(value) < 0.001:
                    continue

                # 跳过整百的年份(如2024, 2026)
                if value > 1900 and value < 2100 and abs(value - round(value)) < 0.001:
                    continue

                # 跳过纯小数且无单位的小数字(< 5,可能是序号)
                if abs(value) < 5 and not unit:
                    continue

                # 检查是否有匹配的计算值
                found = self._find_matching_value(value)

                if not found:
                    violations.append(
                        f"未溯源数字: {value}{unit} — "
                        f"此数字不在computed_data中,可能被LLM编造"
                    )

            except ValueError:
                continue

        is_safe = len(violations) == 0
        if not is_safe:
            logger.warning(
                f"数字安全校验失败: {len(violations)}个未溯源数字"
            )
            for v in violations:
                logger.warning(f"  {v}")

        return is_safe, violations

    def _find_matching_value(self, value: float) -> bool:
        """在计算值中查找匹配 (v5.6 P1-9: 按量级绝对容差)"""
        # 精确匹配
        rounded = round(value, 4)
        if rounded in self._value_index:
            return True

        # 按量级容差匹配: tol = 绝对下限 + 相对比例 (适配大/小量级)
        for computed_val in self._value_index:
            tol = self.ABS_TOLERANCE + self.REL_TOLERANCE * abs(computed_val)
            if abs(value - computed_val) <= tol:
                return True

        return False


class CodeAsReasoningPipeline:
    """
    🆕 v2.1 代码即推理流水线

    使用方式:
        pipeline = CodeAsReasoningPipeline(executor, safety_checker)
        result = await pipeline.run(
            plan="计算sh.600000的MACD和RSI",
            context={"ohlcv_df": kline_data},
        )
    """

    def __init__(self):
        self.executor = CodeExecutor()
        self.execution_history: List[ProvenanceReport] = []

    async def run(
        self,
        plan: str,
        context: Dict[str, Any],
        router=None,  # ModelRouter (用于LLM调用)
    ) -> ProvenanceReport:
        """
        执行代码即推理流水线

        Phase 1: LLM → 计算计划
        Phase 2: Python → 执行计算
        Phase 3: 数值注入 → 锁定
        Phase 4: LLM → 叙事报告

        Args:
            plan: 分析任务描述
            context: 数据上下文 (ohlcv_df, indicators, etc.)
            router: 模型路由器

        Returns:
            ProvenanceReport (含所有计算值+安全校验结果)
        """
        report = ProvenanceReport()

        # ===== Phase 1: 生成计算代码 =====
        if router:
            code = await self._generate_code(plan, context, router)
        else:
            code = self._generate_code_local(plan, context)

        report.execution_log.append(f"Phase 1: 代码生成完成 ({len(code)} chars)")

        # ===== Phase 2: 执行计算 =====
        try:
            results = self.executor.execute(code, context)
            report.execution_log.append(
                f"Phase 2: 计算完成, {len(results)}个结果"
            )

            # 注册为ComputedNumber
            for k, v in results.items():
                if isinstance(v, (int, float)):
                    report.computed_numbers[k] = ComputedNumber(
                        name=k,
                        value=v,
                        formula=f"code_exec #{self.executor._execution_count}",
                    )

        except Exception as e:
            report.execution_log.append(f"Phase 2: 执行失败 - {e}")
            report.num_safe = False
            return report

        report.execution_log.append(
            f"Phase 2: {len(report.computed_numbers)}个数值已锁定"
        )

        # ===== Phase 3: 数字锁定 (注入上下文) =====
        # 这一步确保后续LLM只看到"已计算"的数字,不会自己编造

        # ===== Phase 4: 生成报告 (可选,需要router) =====
        if router:
            narrative = await self._generate_narrative(
                plan, report.computed_numbers, context, router
            )
            report.execution_log.append(f"Phase 4: 叙事报告生成完成")

            # 安全校验 (v5.6 P1-9: 失败 → 显式改写重生成一次, 而非仅告警)
            checker = NumericSafetyChecker(report.computed_numbers)
            is_safe, violations = checker.validate_report(narrative)
            if not is_safe:
                report.execution_log.append(
                    f"安全校验: ❌ {len(violations)}个违规, 触发改写重生成"
                )
                narrative = await self._regenerate_narrative(
                    plan, report.computed_numbers, context, router, violations
                )
                is_safe, violations = checker.validate_report(narrative)
            report.num_safe = is_safe
            report.violations = violations
            report.execution_log.append(
                f"安全校验: {'✅ 通过' if is_safe else f'❌ {len(violations)}个违规'}"
            )

        self.execution_history.append(report)
        return report

    async def _generate_code(
        self,
        plan: str,
        context: Dict[str, Any],
        router,
    ) -> str:
        """Phase 1: LLM生成计算代码"""
        context_summary = {
            k: f"<{type(v).__name__}: {getattr(v, 'shape', 'N/A')}>"
            for k, v in context.items()
        }

        prompt = (
            "你是一个Python量化计算引擎。根据以下任务描述生成计算代码。\n\n"
            f"任务: {plan}\n\n"
            f"可用数据: {context_summary}\n\n"
            "规则:\n"
            "1. 只能使用numpy(np)和pandas(pd)\n"
            "2. 计算结果存入变量(如 rsi_14 = ...)\n"
            "3. 不要用import语句(已预置)\n"
            "4. 不要用print/文件I/O/网络请求\n"
            "5. 只输出代码,不要解释\n"
        )

        try:
            result = await router.route(
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": "请生成计算代码:"},
                ],
                task_type="indicator_read",
            )
            # 提取代码块
            code = result.response
            if "```python" in code:
                code = code.split("```python")[1].split("```")[0]
            elif "```" in code:
                code = code.split("```")[1].split("```")[0]
            return code.strip()
        except Exception:
            return self._generate_code_local(plan, context)

    def _generate_code_local(
        self, plan: str, context: Dict[str, Any]
    ) -> str:
        """本地代码生成 (无LLM时的fallback)"""
        code_lines = []
        if "ohlcv_df" in context:
            code_lines.append("df = ohlcv_df")
            code_lines.append("close = df['close'].values")
            code_lines.append("high = df['high'].values")
            code_lines.append("low = df['low'].values")
            code_lines.append("volume = df['volume'].values")

        # 根据plan中的关键词判断要计算什么
        plan_lower = plan.lower()
        if "macd" in plan_lower:
            code_lines.extend([
                "ema12 = pd.Series(close).ewm(span=12, adjust=False).mean().values[-1]",
                "ema26 = pd.Series(close).ewm(span=26, adjust=False).mean().values[-1]",
                "macd_dif = float(ema12 - ema26)",
                "macd_signal = float(pd.Series(close).ewm(span=12, adjust=False).mean().iloc[-1] - pd.Series(close).ewm(span=26, adjust=False).mean().iloc[-1])",
            ])
        if "rsi" in plan_lower:
            code_lines.extend([
                "delta = np.diff(close)",
                "gain = np.where(delta > 0, delta, 0)",
                "loss = np.where(delta < 0, -delta, 0)",
                "avg_gain = float(pd.Series(gain).ewm(span=14, adjust=False).mean().iloc[-1])",
                "avg_loss = float(pd.Series(loss).ewm(span=14, adjust=False).mean().iloc[-1])",
                "rs = avg_gain / max(avg_loss, 1e-10)",
                "rsi_14 = float(100 - (100 / (1 + rs)))",
            ])
        if "ma" in plan_lower or "均线" in plan:
            code_lines.extend([
                "ma_5 = float(pd.Series(close).rolling(5).mean().iloc[-1])",
                "ma_20 = float(pd.Series(close).rolling(20).mean().iloc[-1])",
                "ma_60 = float(pd.Series(close).rolling(60).mean().iloc[-1])",
            ])
        if "volatility" in plan_lower or "波动" in plan_lower:
            code_lines.extend([
                "returns = np.diff(np.log(close))",
                "hv_20 = float(np.std(returns[-20:]) * np.sqrt(252) * 100)",
                "atr = float(pd.DataFrame({'h': high, 'l': low, 'c': close}).apply(lambda r: max(r['h']-r['l'], abs(r['h']-r['c']), abs(r['l']-r['c'])), axis=1).tail(14).mean())",
                "atr_pct = float(atr / close[-1] * 100)",
            ])

        if not code_lines[1:]:  # 没有匹配到具体指标
            code_lines.extend([
                "close_price = float(close[-1])",
                "volume_today = float(volume[-1])",
                "vol_ma5 = float(pd.Series(volume).rolling(5).mean().iloc[-1])",
                "vol_ratio = float(volume_today / max(vol_ma5, 1))",
            ])

        return "\n".join(code_lines)

    async def _generate_narrative(
        self,
        plan: str,
        computed: Dict[str, ComputedNumber],
        context: Dict[str, Any],
        router,
    ) -> str:
        """Phase 4: LLM生成叙事报告"""
        numbers_summary = {
            k: {"value": v.value, "unit": v.unit, "source": v.source}
            for k, v in computed.items()
        }

        prompt = (
            f"基于以下代码计算的指标生成简要技术分析:\n\n"
            f"任务: {plan}\n\n"
            f"计算指标(代码执行,非LLM生成):\n{numbers_summary}\n\n"
            "要求:\n"
            "1. 引用数字时必须标注来源(如'根据代码计算的RSI(14)=xxx')\n"
            "2. 不要编造任何不在上述列表中的数字\n"
            "3. 如果有不确定的地方,标注'需要进一步数据验证'\n"
        )

        try:
            result = await router.route(
                messages=[
                    {"role": "system", "content": "你是量化技术分析报告生成器。"},
                    {"role": "user", "content": prompt},
                ],
                task_type="indicator_read",
            )
            return result.response
        except Exception as e:
            return f"叙事生成失败: {e}\n\n计算数据: {numbers_summary}"

    async def _regenerate_narrative(
        self,
        plan: str,
        computed: Dict[str, ComputedNumber],
        context: Dict[str, Any],
        router,
        violations: List[str],
    ) -> str:
        """v5.6 P1-9: 校验失败后显式改写 — 把违规数字清单回喂, 要求删除/改述"""
        numbers_summary = {
            k: {"value": v.value, "unit": v.unit, "source": v.source}
            for k, v in computed.items()
        }

        prompt = (
            f"你上一版技术分析报告包含无法溯源的数字(疑似编造), 请改写:\n\n"
            f"任务: {plan}\n\n"
            f"合法计算指标(只能引用这些):\n{numbers_summary}\n\n"
            f"被标记的未溯源数字:\n{'; '.join(violations[:10])}\n\n"
            "改写要求:\n"
            "1. 删除或修正所有未溯源数字, 只保留能对应到上述指标的数字\n"
            "2. 引用数字时必须标注来源(如'根据代码计算的RSI(14)=xxx')\n"
            "3. 不要编造任何不在指标列表中的数字\n"
        )

        try:
            result = await router.route(
                messages=[
                    {"role": "system", "content": "你是量化技术分析报告生成器, 负责修正编造数字。"},
                    {"role": "user", "content": prompt},
                ],
                task_type="indicator_read",
            )
            return result.response
        except Exception as e:
            return f"改写失败: {e}\n\n计算数据: {numbers_summary}"
