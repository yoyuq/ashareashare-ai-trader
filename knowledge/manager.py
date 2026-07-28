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
        self._chroma_client = None   # v2.2: 延迟初始化

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
    # 4. 向量检索 (ChromaDB — 延迟加载 + 可初始化)
    # ═══════════════════════════════════════════════════════════════

    @property
    def chroma_available(self) -> bool:
        """ChromaDB是否可用(含数据)"""
        try:
            if self._chroma_client is None:
                import chromadb
                store_path = self.root / "vector_store" / "chroma"
                store_path.mkdir(parents=True, exist_ok=True)
                self._chroma_client = chromadb.PersistentClient(
                    path=str(store_path)
                )
            # 检查是否有collection
            collections = self._chroma_client.list_collections()
            return len(collections) > 0
        except Exception:
            return False

    def initialize_vectordb(
        self,
        collection_name: str = "kline_patterns",
        embedding_fn=None,
    ) -> bool:
        """
        v2.9 初始化向量数据库 (含示例数据播种)

        Args:
            collection_name: collection名称
            embedding_fn: 嵌入函数(默认使用ChromaDB内置)

        Returns:
            是否初始化成功
        """
        try:
            import chromadb
            store_path = self.root / "vector_store" / "chroma"
            store_path.mkdir(parents=True, exist_ok=True)

            self._chroma_client = chromadb.PersistentClient(
                path=str(store_path)
            )

            # 检查collection是否已存在
            existing = [c.name for c in self._chroma_client.list_collections()]
            if collection_name in existing:
                logger.info(f"ChromaDB collection '{collection_name}' 已存在 ({self._chroma_client.get_collection(collection_name).count()} 条)")
                return True

            # 创建新collection
            collection = self._chroma_client.create_collection(
                name=collection_name,
                metadata={"description": "K线形态向量索引 (v2.9)"},
            )

            # 🆕 v2.9: 播种示例形态数据 (真实形态的归一化向量)
            self._seed_pattern_data(collection)

            logger.info(f"ChromaDB初始化成功: {store_path} / {collection_name}")
            return True
        except Exception as e:
            logger.warning(f"ChromaDB初始化失败: {e}")
            return False

    def _seed_pattern_data(self, collection) -> None:
        """
        🆕 v2.9 播种基础K线形态数据

        使用归一化的K线特征向量, 覆盖8种核心形态:
          - doji (十字星)
          - hammer (锤子线)
          - shooting_star (射击之星)
          - bullish_engulfing (看涨吞没)
          - bearish_engulfing (看跌吞没)
          - morning_star (晨星)
          - evening_star (暮星)
          - three_soldiers (红三兵)
        """
        import numpy as np

        seed_patterns = {
            "doji": "十字星 — 开盘≈收盘,上下影线长度接近。趋势反转信号,高位为黄昏十字星,低位为早晨十字星。",
            "hammer": "锤子线 — 长下影线(≥2倍实体),小实体在顶部。出现在下跌趋势末端为反转看涨信号。",
            "shooting_star": "射击之星 — 长上影线(≥2倍实体),小实体在底部。出现在上涨趋势末端为反转看跌信号。",
            "bullish_engulfing": "看涨吞没 — 阳线实体完全包住前一阴线实体。出现在下跌趋势中为强烈反转看涨信号。",
            "bearish_engulfing": "看跌吞没 — 阴线实体完全包住前一阳线实体。出现在上涨趋势中为强烈反转看跌信号。",
            "morning_star": "晨星 — 三日形态: 阴线→十字星→阳线,且第三日收盘超过首日。底部反转看涨信号。",
            "evening_star": "暮星 — 三日形态: 阳线→十字星→阴线,且第三日收盘低于首日。顶部反转看跌信号。",
            "three_soldiers": "红三兵 — 连续三根阳线,每根收盘高于前根。出现在盘整后为看涨持续信号。",
        }

        # 每个形态的特征向量 [body_ratio, upper_shadow_ratio, lower_shadow_ratio, body_direction, day2_body, day3_body]
        # day2/day3 仅用于三日形态, 单日形态设为0
        seed_vectors = {
            "doji": [0.0, 0.45, 0.45, 0.0, 0.0, 0.0],
            "hammer": [0.15, 0.1, 0.75, 1.0, 0.0, 0.0],
            "shooting_star": [0.15, 0.75, 0.1, -1.0, 0.0, 0.0],
            "bullish_engulfing": [0.8, 0.1, 0.1, 1.0, -0.3, 0.0],
            "bearish_engulfing": [0.8, 0.1, 0.1, -1.0, 0.3, 0.0],
            "morning_star": [0.0, 0.1, 0.1, 1.0, -0.5, 0.6],
            "evening_star": [0.0, 0.1, 0.1, -1.0, 0.5, -0.6],
            "three_soldiers": [0.6, 0.1, 0.1, 1.0, 0.3, 0.3],
        }

        # 归一化向量
        ids = []
        embeddings = []
        documents = []
        metadatas = []

        for i, (name, desc) in enumerate(seed_patterns.items()):
            vec = seed_patterns.get(name, [0.0] * 6)
            vec_arr = np.array(vec, dtype=np.float32)
            norm = np.linalg.norm(vec_arr)
            if norm > 0:
                vec_arr = vec_arr / norm

            ids.append(f"seed_{i}_{name}")
            embeddings.append(vec_arr.tolist())
            documents.append(desc)
            metadatas.append({
                "pattern": name,
                "category": "candlestick",
                "source": "seed_data_v2.9",
            })

        try:
            collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
            )
            logger.info(f"[ChromaDB] 播种 {len(ids)} 条K线形态数据")
        except Exception as e:
            logger.debug(f"[ChromaDB] 播种跳过: {e}")

    def search_similar_klines(
        self,
        vector: List[float],
        top_k: int = 30,
    ) -> List[Dict]:
        """
        向量检索历史相似K线形态

        Args:
            vector: 查询向量
            top_k: 返回数量

        Returns:
            相似K线列表, 含metadata
        """
        if not self.chroma_available:
            logger.debug("ChromaDB不可用或无数据, 跳过向量检索")
            return []

        try:
            collection = self._chroma_client.get_or_create_collection(
                "kline_patterns"
            )
            results = collection.query(
                query_embeddings=[vector],
                n_results=top_k,
                include=["metadatas", "distances"],
            )
            return results
        except Exception as e:
            logger.debug(f"向量检索失败: {e}")
            return []

    def rag_query(self, query: str, top_k: int = 5) -> str:
        """
        从参考文档中检索相关知识片段 (v2.2 改进: 优先向量检索)

        检索策略:
        1. 优先: ChromaDB向量检索(如果可用且有数据)
        2. 降级: 关键词匹配(零依赖fallback)

        Args:
            query: 查询文本
            top_k: 返回片段数

        Returns:
            拼接后的相关文本
        """
        # 尝试向量检索
        if self.chroma_available:
            try:
                collection = self._chroma_client.get_collection(
                    "knowledge_base"
                )
                results = collection.query(
                    query_texts=[query],
                    n_results=top_k,
                    include=["documents", "metadatas"],
                )
                if results and results.get("documents") and results["documents"][0]:
                    docs = results["documents"][0]
                    metas = results.get("metadatas", [[]])[0]
                    parts = ["相关知识库参考 (向量检索):\n"]
                    for i, (doc, meta) in enumerate(zip(docs, metas)):
                        source = meta.get("source", "unknown") if meta else "unknown"
                        parts.append(f"### {source}\n{doc[:800]}\n")
                    return "\n".join(parts)
            except Exception as e:
                logger.debug(f"向量检索降级到关键词匹配: {e}")

        # Fallback: 关键词匹配
        reference_dir = self.root / "reference"
        if not reference_dir.exists():
            return ""

        snippets = []
        for md_file in reference_dir.glob("*.md"):
            content = md_file.read_text(encoding="utf-8")
            query_words = set(query.lower().split())
            content_lower = content.lower()
            score = sum(1 for w in query_words if w in content_lower)

            if score > 0:
                # 提取相关段落: 找到第一个匹配词的位置, 取周围文本
                best_pos = 0
                best_word = ""
                for w in query_words:
                    pos = content_lower.find(w)
                    if pos != -1 and (best_pos == 0 or score > 0):
                        best_pos = pos
                        best_word = w

                # 取匹配位置前后各500字符
                start = max(0, best_pos - 300)
                end = min(len(content), best_pos + 700)
                excerpt = content[start:end]
                if start > 0:
                    excerpt = "..." + excerpt[excerpt.find("\n"):] if "\n" in excerpt[3:] else "..." + excerpt[3:]
                if end < len(content):
                    excerpt = excerpt[:excerpt.rfind("\n")] + "..."

                snippets.append({
                    "source": md_file.stem,
                    "relevance": score,
                    "content": excerpt[:800],
                })

        snippets.sort(key=lambda x: x["relevance"], reverse=True)
        top = snippets[:top_k]

        if not top:
            return ""

        result = "相关知识库参考 (关键词匹配):\n\n"
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
