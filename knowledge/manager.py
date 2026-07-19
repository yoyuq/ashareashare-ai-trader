"""
KnowledgeManager — 知识库管理器 (v2.0)

三种检索方式:
  1. 直接注入: Prompt中的{knowledge_file} → 读文件拼接
  2. 向量检索: ChromaDB.similarity_search() → Top-K片段
  3. 结构化查询: YAML解析 → 按key精确取值

使用方式:
    km = KnowledgeManager("knowledge/")
    prompt = km.get_system_prompt("technical_analyst")  # 含自动注入
    rules = km.get_rule("trading_rules.fees.stamp_duty")
    strategy = km.get_strategy("dual_ma_trend")
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from loguru import logger


class KnowledgeManager:
    """知识库管理器 (v2.0)"""

    def __init__(self, root: str = "knowledge/"):
        self.root = Path(root)
        self._rules_cache: Dict[str, Dict] = {}
        self._prompt_cache: Dict[str, str] = {}
        self._strategy_cache: Dict[str, Dict] = {}

        # 验证目录结构
        self._validate_structure()

    # ═══════════════════════════════════════════════════════════════
    # 1. Prompt知识 (直接注入)
    # ═══════════════════════════════════════════════════════════════

    def get_system_prompt(self, agent_name: str) -> str:
        """
        获取Agent系统Prompt,自动注入关联规则知识

        支持的占位符:
          {indicator_guide} → knowledge/rules/indicator_guide.yaml
          {trading_rules}   → knowledge/rules/trading_rules.yaml
          {strategy_list}   → 策略注册表摘要
          {pattern_handbook} → K线形态手册

        Args:
            agent_name: Agent名称(不含.txt), e.g. 'technical_analyst'

        Returns:
            完整系统Prompt(含注入的知识文件内容)
        """
        # 缓存检查
        if agent_name in self._prompt_cache:
            return self._prompt_cache[agent_name]

        prompt_path = self.root / "prompts" / "system" / f"{agent_name}.txt"

        if not prompt_path.exists():
            logger.warning(f"Prompt文件不存在: {prompt_path}, 返回空")
            return f"你是{agent_name} Agent。(Prompt文件缺失)"

        prompt = prompt_path.read_text(encoding="utf-8")

        # ---- 自动注入 ----
        # {indicator_guide}
        if "{indicator_guide}" in prompt:
            guide = self._load_yaml("rules/indicator_guide.yaml")
            if guide:
                prompt = prompt.replace(
                    "{indicator_guide}",
                    json.dumps(guide, ensure_ascii=False, indent=2)[:8000]
                )

        # {trading_rules}
        if "{trading_rules}" in prompt:
            rules = self._load_yaml("rules/trading_rules.yaml")
            if rules:
                prompt = prompt.replace(
                    "{trading_rules}",
                    json.dumps(rules, ensure_ascii=False, indent=2)[:5000]
                )

        # {strategy_list}
        if "{strategy_list}" in prompt:
            strategies = self._load_yaml("strategies/registry.yaml")
            if strategies:
                # 只注入策略摘要(名称+描述,不含参数)
                summary = [
                    {"id": s["id"], "name": s["name"], "category": s["category"],
                     "description": s.get("description", ""),
                     "regimes": s.get("market_regimes", [])}
                    for s in strategies.get("strategies", [])
                ]
                prompt = prompt.replace(
                    "{strategy_list}",
                    json.dumps(summary, ensure_ascii=False, indent=2)
                )

        # {pattern_handbook}
        if "{pattern_handbook}" in prompt:
            guide = self._load_yaml("rules/indicator_guide.yaml")
            if guide and "indicators" in guide:
                patterns = guide["indicators"].get("candlestick_patterns", {})
                prompt = prompt.replace(
                    "{pattern_handbook}",
                    json.dumps(patterns, ensure_ascii=False, indent=2)
                )

        # 缓存
        self._prompt_cache[agent_name] = prompt
        return prompt

    def get_task_prompt(self, task_name: str) -> Optional[str]:
        """获取任务级Prompt"""
        path = self.root / "prompts" / "tasks" / f"{task_name}.txt"
        if path.exists():
            return path.read_text(encoding="utf-8")
        return None

    def get_few_shot_examples(self, scenario: str) -> Optional[List[Dict]]:
        """获取Few-shot示例"""
        path = self.root / "prompts" / "few_shots" / f"{scenario}.json"
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        return None

    # ═══════════════════════════════════════════════════════════════
    # 2. 规则知识 (结构化查询)
    # ═══════════════════════════════════════════════════════════════

    def get_rule(self, path: str) -> Any:
        """
        按路径查询规则值

        Args:
            path: 点分隔路径, e.g. 'trading_rules.fees.stamp_duty.rate'

        Returns:
            规则值或 None
        """
        parts = path.split(".")
        filename = f"rules/{parts[0]}.yaml"

        data = self._load_yaml(filename)
        if data is None:
            return None

        # 沿路径遍历
        current = data
        for key in parts[1:]:
            if isinstance(current, dict):
                current = current.get(key)
            else:
                return None
            if current is None:
                return None

        return current

    def get_all_rules(self, rule_set: str) -> Optional[Dict]:
        """获取整个规则集"""
        return self._load_yaml(f"rules/{rule_set}.yaml")

    def get_hardened_definition(self, concept: str) -> Optional[Dict]:
        """
        🆕 v2.1 获取硬化口径定义

        优先从 hardened_definitions.yaml 获取,
        其次从 trading_rules.yaml 的 trading_signals 部分获取

        e.g. concept='volume_expansion' → {formula: ..., threshold: 1.5}
        """
        # 主来源: hardened_definitions.yaml
        hardened = self._load_yaml("hardened_definitions.yaml")
        if hardened:
            defs = hardened.get("definitions", {})
            result = defs.get(concept)
            if result:
                return result

        # 备来源: trading_rules.yaml
        rules = self._load_yaml("rules/trading_rules.yaml")
        if rules:
            signals = rules.get("trading_signals", {})
            return signals.get(concept)

        return None

    def verify_definition(self, concept: str, value: Any) -> bool:
        """
        🆕 v2.1 校验一个计算值是否符合硬化口径

        例如: concept='volume_expansion', value=1.2
        → 检查 1.2 > threshold(1.5)? → False (不达标)
        """
        defn = self.get_hardened_definition(concept)
        if defn is None:
            logger.warning(f"口径'{concept}'未定义,跳过校验")
            return True  # 未定义时不阻塞

        threshold = defn.get("threshold")
        if threshold is None:
            return True

        return value >= threshold

    # ═══════════════════════════════════════════════════════════════
    # 3. 策略知识
    # ═══════════════════════════════════════════════════════════════

    def get_strategy(self, strategy_id: str) -> Optional[Dict]:
        """获取单个策略完整配置"""
        if strategy_id in self._strategy_cache:
            return self._strategy_cache[strategy_id]

        registry = self._load_yaml("strategies/registry.yaml")
        if registry is None:
            return None

        for s in registry.get("strategies", []):
            if s["id"] == strategy_id:
                self._strategy_cache[strategy_id] = s
                return s

        return None

    def list_strategies(self, category: Optional[str] = None) -> List[Dict]:
        """列出策略(可按类别过滤)"""
        registry = self._load_yaml("strategies/registry.yaml")
        if registry is None:
            return []

        strategies = registry.get("strategies", [])
        if category:
            strategies = [s for s in strategies if s.get("category") == category]

        return strategies

    def get_strategies_for_regime(self, regime: str) -> List[Dict]:
        """获取适用于当前市场状态的策略列表"""
        return [
            s for s in self.list_strategies()
            if regime in s.get("market_regimes", [])
        ]

    def get_strategy_knowledge(self, strategy_id: str) -> Dict[str, Any]:
        """
        获取策略完整知识包: 配置 + 适用场景 + 风险提示
        """
        strategy = self.get_strategy(strategy_id)
        if strategy is None:
            return {}

        return {
            "id": strategy["id"],
            "name": strategy["name"],
            "category": strategy["category"],
            "description": strategy.get("description", ""),
            "params": strategy.get("params", {}),
            "market_regimes": strategy.get("market_regimes", []),
            "capacity_limit": strategy.get("capacity_limit", 0),
            "note": strategy.get("note", ""),
            # 风险提示
            "risks": self._get_strategy_risks(strategy),
        }

    def _get_strategy_risks(self, strategy: Dict) -> List[str]:
        """获取策略风险提示"""
        risks = []
        category = strategy.get("category", "")

        if category == "trend_following":
            risks = [
                "震荡市中可能频繁止损",
                "趋势反转时回撤较大",
                "需配合市场状态识别使用",
            ]
        elif category == "mean_reversion":
            risks = [
                "强趋势市可能持续亏损",
                "需设置严格止损",
                "流动性差时滑点成本高",
            ]
        elif category == "ashare_special":
            risks = [
                "高风险策略,需严格止损",
                "次日竞价流动性不足可能导致无法成交",
                "炸板风险(封板后被打开)",
            ]
        elif category == "multi_factor":
            risks = [
                "因子失效风险(需监控IC衰减)",
                "调仓频率高于趋势策略,交易成本更高",
                "模型复杂度高,需定期再平衡权重",
            ]

        return risks

    # ═══════════════════════════════════════════════════════════════
    # 4. 向量检索 (ChromaDB — 初始化时延迟加载)
    # ═══════════════════════════════════════════════════════════════

    def search_similar_klines(
        self,
        vector: List[float],
        top_k: int = 30,
    ) -> List[Dict]:
        """
        向量检索历史相似K线形态

        注意: 需要ChromaDB已初始化并有数据
        """
        try:
            import chromadb
            client = chromadb.PersistentClient(
                path=str(self.root / "vector_store" / "chroma")
            )
            collection = client.get_or_create_collection("kline_patterns")
            results = collection.query(
                query_embeddings=[vector],
                n_results=top_k,
            )
            return results
        except Exception as e:
            logger.debug(f"向量检索不可用: {e}")
            return []

    def rag_query(self, query: str, top_k: int = 5) -> str:
        """
        从参考文档中RAG检索相关知识片段

        Args:
            query: 查询文本
            top_k: 返回片段数

        Returns:
            拼接后的相关文本
        """
        # 简化版: 基于关键词匹配
        reference_dir = self.root / "reference"
        if not reference_dir.exists():
            return ""

        snippets = []
        for md_file in reference_dir.glob("*.md"):
            content = md_file.read_text(encoding="utf-8")
            # 简单关键词匹配
            query_words = set(query.lower().split())
            content_lower = content.lower()
            score = sum(1 for w in query_words if w in content_lower)

            if score > 0:
                # 提取相关段落(简化: 前500字符)
                snippets.append({
                    "source": md_file.stem,
                    "relevance": score,
                    "content": content[:500],
                })

        snippets.sort(key=lambda x: x["relevance"], reverse=True)
        top = snippets[:top_k]

        if not top:
            return ""

        result = "相关知识库参考:\n\n"
        for i, s in enumerate(top):
            result += f"### {s['source']} (相关度: {s['relevance']})\n{s['content']}\n\n"

        return result

    # ═══════════════════════════════════════════════════════════════
    # 5. 参考知识
    # ═══════════════════════════════════════════════════════════════

    def get_reference(self, name: str) -> Optional[str]:
        """读取参考文档全文"""
        path = self.root / "reference" / f"{name}.md"
        if path.exists():
            return path.read_text(encoding="utf-8")
        return None

    def get_glossary(self, term: Optional[str] = None) -> str:
        """查询术语表"""
        glossary = self.get_reference("glossary")
        if glossary is None:
            return ""
        if term:
            # 简单搜索
            lines = glossary.split("\n")
            in_section = False
            result = []
            for line in lines:
                if line.startswith("## "):
                    in_section = False
                if f"**{term}**" in line or line.startswith(f"- **{term}"):
                    in_section = True
                    result.append(line)
                elif in_section and line.startswith("- **"):
                    in_section = False

            if result:
                return "\n".join(result)
            return f"术语'{term}'未在术语表中找到"
        return glossary

    # ═══════════════════════════════════════════════════════════════
    # 内部工具
    # ═══════════════════════════════════════════════════════════════

    def _load_yaml(self, rel_path: str) -> Optional[Dict]:
        """加载并缓存YAML文件"""
        if rel_path in self._rules_cache:
            return self._rules_cache[rel_path]

        full_path = self.root / rel_path
        if not full_path.exists():
            logger.warning(f"知识文件不存在: {full_path}")
            return None

        try:
            with open(full_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            self._rules_cache[rel_path] = data
            return data
        except Exception as e:
            logger.error(f"加载知识文件失败 {rel_path}: {e}")
            return None

    def _validate_structure(self) -> bool:
        """验证知识库目录结构"""
        required_dirs = [
            "prompts/system",
            "rules",
            "strategies",
            "reference",
        ]
        missing = []
        for d in required_dirs:
            if not (self.root / d).exists():
                missing.append(d)

        if missing:
            logger.warning(f"知识库目录不完整,缺少: {missing}")
            return False
        return True

    def reload(self):
        """重载所有缓存"""
        self._rules_cache.clear()
        self._prompt_cache.clear()
        self._strategy_cache.clear()
        logger.info("知识库缓存已刷新")
