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

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from loguru import logger


def _stable_hash_embed(text: str, dim: int = 256) -> List[float]:
    """无依赖确定性文本嵌入 (v3.3 加权 n-gram): 拉丁词 + 中文 3/2/1-gram 加权哈希 → L2归一化。

    md5 保证跨进程稳定 (Python 内置 hash() 对 str 有随机种子, 不可用于持久化向量)。
    加权: 中文 3-gram (+3, 最能抓金融术语如"止损/止盈/低估值"), 2-gram (+2),
    单字 (+1), 拉丁词 (+2)。比旧版单字+2-gram 同权更能区分语义相近片段。
    """
    vec = [0.0] * dim
    s = text.lower()
    for tok in re.findall(r"[a-z0-9_]{2,}", s):
        h = int.from_bytes(hashlib.md5(tok.encode()).digest()[:4], "big") % dim
        vec[h] += 2.0
    for seg in re.findall(r"[一-鿿]+", s):
        for i in range(len(seg) - 2):
            h = int.from_bytes(hashlib.md5(f"t:{seg[i:i+3]}".encode()).digest()[:4], "big") % dim
            vec[h] += 3.0
        for i in range(len(seg) - 1):
            h = int.from_bytes(hashlib.md5(f"b:{seg[i:i+2]}".encode()).digest()[:4], "big") % dim
            vec[h] += 2.0
        for ch in seg:
            h = int.from_bytes(hashlib.md5(f"c:{ch}".encode()).digest()[:4], "big") % dim
            vec[h] += 1.0
    norm = math.sqrt(sum(x * x for x in vec))
    return [x / norm for x in vec] if norm else vec


# 参考文档 → regime 打标关键词 (用于 regime 感知检索)
REGIME_TAG_KEYWORDS = {"牛": "bull", "熊": "bear", "危机": "crisis", "震荡": "range",
                       "通用": "general", "通用原则": "general"}


def _chunk_markdown(text: str, max_len: int = 800) -> List[tuple]:
    """v3.3 按 `##` 节切分参考文档, 保留节标题作前缀, 大节再按段落切。

    返回 [(header, chunk_text)]。比旧版纯 `\n\n` 段落切分更能保留上下文结构,
    且方便按节打 regime 标签。
    """
    lines = text.split("\n")
    sections = []  # [(header, [lines])]
    cur_header, cur = "", []
    for ln in lines:
        if ln.startswith("## "):
            if cur:
                sections.append((cur_header, "\n".join(cur)))
            cur_header = ln[3:].strip()
            cur = [ln]
        else:
            cur.append(ln)
    if cur:
        sections.append((cur_header, "\n".join(cur)))

    out = []
    for header, body in sections:
        prefix = f"## {header}\n" if header else ""
        if len(body) <= max_len:
            if len(body) > 30:
                out.append((header, prefix + body))
            continue
        for para in re.split(r"\n\n+", body):
            para = para.strip()
            if len(para) > 30:
                out.append((header, prefix + para[:max_len]))
    return out


def _regime_from_header(header: str) -> Optional[str]:
    """从节标题推断 regime (用于检索过滤)。"""
    if not header:
        return None
    for kw, tag in REGIME_TAG_KEYWORDS.items():
        if kw in header:
            return tag
    return None


# K线形态描述 (中英双语) — 文本检索 collection 播种用
KLINE_PATTERN_TEXTS = {
    "doji": "十字星 — 开盘≈收盘,上下影线长度接近。趋势反转信号,高位为黄昏十字星,低位为早晨十字星。",
    "hammer": "锤子线 — 长下影线(≥2倍实体),小实体在顶部。出现在下跌趋势末端为反转看涨信号。",
    "shooting_star": "射击之星 — 长上影线(≥2倍实体),小实体在底部。出现在上涨趋势末端为反转看跌信号。",
    "bullish_engulfing": "看涨吞没 — 阳线实体完全包住前一阴线实体。出现在下跌趋势中为强烈反转看涨信号。",
    "bearish_engulfing": "看跌吞没 — 阴线实体完全包住前一阳线实体。出现在上涨趋势中为强烈反转看跌信号。",
    "morning_star": "晨星 — 三日形态: 阴线→十字星→阳线,且第三日收盘超过首日。底部反转看涨信号。",
    "evening_star": "暮星 — 三日形态: 阳线→十字星→阴线,且第三日收盘低于首日。顶部反转看跌信号。",
    "three_soldiers": "红三兵 — 连续三根阳线,每根收盘高于前根。出现在盘整后为看涨持续信号。",
}


def _cn_tokens(text: str) -> List[str]:
    """
    中英文检索 token 提取 (v3.1-deerflow RAG 降级修复)

    中文没有空格分词, 旧的 `query.split()` 会把整句当做一个 token 导致检索永远为空。
    这里:
      - 拉丁词/数字: 按单词提取
      - 中文: 提取 2/3 字 n-gram (均线/金叉/止损等金融术语多为 2-4 字)

    Returns:
        去重后的 token 列表
    """
    if not text:
        return []
    tokens = set()
    for w in re.findall(r"[a-z][a-z0-9_]*", text.lower()):
        tokens.add(w)
    for chunk in re.findall(r"[一-鿿]+", text):
        if len(chunk) < 2:
            continue
        for size in (2, 3):
            for i in range(len(chunk) - size + 1):
                tokens.add(chunk[i:i + size])
    return list(tokens)


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
                guide_json = json.dumps(guide, ensure_ascii=False, indent=2)
                if len(guide_json) > 10000:
                    logger.warning(f"indicator_guide 过大({len(guide_json)}字符), 截断至10000字符")
                prompt = prompt.replace(
                    "{indicator_guide}",
                    guide_json[:10000] + ("...(已截断)" if len(guide_json) > 10000 else "")
                )

        # {trading_rules}
        if "{trading_rules}" in prompt:
            rules = self._load_yaml("rules/trading_rules.yaml")
            if rules:
                rules_json = json.dumps(rules, ensure_ascii=False, indent=2)
                if len(rules_json) > 8000:
                    logger.warning(f"trading_rules 过大({len(rules_json)}字符), 截断至8000字符")
                prompt = prompt.replace(
                    "{trading_rules}",
                    rules_json[:8000] + ("...(已截断)" if len(rules_json) > 8000 else "")
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

    def get_few_shot_examples(self, scenario: str, normalized: bool = True) -> Optional[List[Dict]]:
        """
        获取Few-shot示例

        Args:
            scenario: 场景名(不含.json)
            normalized: 是否统一格式为标准 examples 数组

        Returns:
            标准化后的示例列表, 或 None
        """
        path = self.root / "prompts" / "few_shots" / f"{scenario}.json"
        if not path.exists():
            return None

        raw = json.loads(path.read_text(encoding="utf-8"))
        if normalized:
            return self._normalize_few_shot(raw)
        return raw if isinstance(raw, list) else [raw]

    def _normalize_few_shot(self, raw: Any) -> List[Dict]:
        """
        统一 Few-shot 格式 → 标准 examples 数组

        标准格式: [{scenario, description, examples: [{id, query, analysis, decision}]}]

        支持的输入格式:
          - examples 数组 (panic_sell, sector_rotation, signal_review)
          - assistant_response.decision_tree (limit_up_analysis, breakout_chase)
          - 直接对象
        """
        # 已是标准格式
        if isinstance(raw, dict) and "examples" in raw:
            return raw["examples"] if isinstance(raw["examples"], list) else [raw["examples"]]

        # decision_tree 格式 → 转为 examples
        if isinstance(raw, dict) and "assistant_response" in raw:
            ar = raw["assistant_response"]
            example = {
                "id": raw.get("scenario", "unknown"),
                "query": ar.get("structure", {}).get("current_status", raw.get("description", "")),
                "analysis": {"key_principles": ar.get("key_principles", []),
                            "decision_tree": ar.get("decision_tree", {})},
                "decision": {"action": "VARIES", "reasoning": ar.get("key_principles", []),
                            "risk_warning": "见决策树各分支"},
            }
            return [example]

        # 直接对象
        if isinstance(raw, dict):
            return [raw]

        # 数组
        if isinstance(raw, list):
            return raw

        return []

    # ═══════════════════════════════════════════════════════════════
    # 1.5 提示词版本管理 (v3.0-competition)
    # ═══════════════════════════════════════════════════════════════

    def get_prompt_version(self, agent_name: str) -> Optional[Dict[str, str]]:
        """
        获取提示词的版本信息 (从 YAML frontmatter 解析)

        Args:
            agent_name: Agent名称

        Returns:
            dict with keys: version, date, author, changes
            如果文件不存在或无版本信息,返回 None
        """
        path = self.root / "prompts" / "system" / f"{agent_name}.txt"
        if not path.exists():
            return None

        content = path.read_text(encoding="utf-8")
        frontmatter = self._parse_prompt_frontmatter(content)
        return frontmatter if frontmatter else None

    def _parse_prompt_frontmatter(self, content: str) -> Dict[str, str]:
        """解析提示词文件的 YAML frontmatter"""
        if not content.startswith("---"):
            return {}

        try:
            end = content.index("---", 3)
            fm_text = content[3:end].strip()
            result = {}
            for line in fm_text.split("\n"):
                line = line.strip()
                if ":" in line:
                    key, _, value = line.partition(":")
                    result[key.strip()] = value.strip().strip('"')
            return result
        except (ValueError, IndexError):
            return {}

    def list_all_prompts(self) -> List[Dict]:
        """列出所有系统提示词及其版本信息"""
        prompts = []
        prompt_dir = self.root / "prompts" / "system"
        if not prompt_dir.exists():
            return prompts

        for f in sorted(prompt_dir.glob("*.txt")):
            version_info = self.get_prompt_version(f.stem)
            prompts.append({
                "name": f.stem,
                "file": f.name,
                "size_chars": f.stat().st_size,
                "version": version_info.get("version", "unknown") if version_info else "unknown",
                "date": version_info.get("date", "") if version_info else "",
                "author": version_info.get("author", "") if version_info else "",
            })

        return prompts

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

    def pit_verify(self, data_date: str, analysis_date: str = None) -> Dict[str, Any]:
        """
        PIT (Point-in-Time) 数据时点校验

        防止 look-ahead bias: 确认分析所用数据的日期不晚于分析执行日期。

        Args:
            data_date: 数据报告日期 (如 '2026Q1', '2026-03-31')
            analysis_date: 分析执行日期 (默认今天)

        Returns:
            {pit_ok: bool, gap_days: int, warning: str}
        """
        from datetime import date as dt_date, timedelta

        if analysis_date is None:
            analysis_date = dt_date.today().isoformat()

        try:
            # 处理季度格式
            if "Q" in data_date:
                year = int(data_date[:4])
                quarter = int(data_date[5]) if len(data_date) > 5 else int(data_date.split("Q")[1])
                month = quarter * 3
                data_dt = dt_date(year, month, 1) + timedelta(days=90)
            else:
                data_dt = dt_date.fromisoformat(data_date[:10])

            analysis_dt = dt_date.fromisoformat(analysis_date[:10])
            gap = (analysis_dt - data_dt).days

            if gap < -7:
                return {"pit_ok": False, "gap_days": gap,
                        "warning": f"数据日期({data_date})晚于分析日期({analysis_date}), 存在未来信息泄露风险"}
            elif gap > 180:
                return {"pit_ok": True, "gap_days": gap,
                        "warning": f"数据过旧({gap}天前), 建议更新"}
            else:
                return {"pit_ok": True, "gap_days": gap, "warning": ""}
        except Exception as e:
            return {"pit_ok": False, "gap_days": 0, "warning": f"日期解析失败: {e}"}

    def get_pit_report(self, data_sources: Dict[str, str]) -> Dict[str, Any]:
        """
        批量 PIT 校验

        Args:
            data_sources: {来源名: 数据日期}, e.g. {"财报": "2026Q1", "行情": "2026-07-30"}

        Returns:
            {all_ok: bool, checks: [{source, pit_ok, warning}], summary: str}
        """
        checks = []
        for source, ddate in data_sources.items():
            r = self.pit_verify(ddate)
            r["source"] = source
            checks.append(r)

        all_ok = all(c["pit_ok"] for c in checks)
        issues = [c for c in checks if not c["pit_ok"] or c["warning"]]
        summary = "PIT校验通过" if all_ok else f"{len(issues)}个数据源存在问题"

        return {"all_ok": all_ok, "checks": checks, "summary": summary}

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
        """
        ChromaDB是否可用(含数据)

        v3.0-competition: 首次访问时自动初始化ChromaDB并播种示例数据
        """
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
            if len(collections) == 0:
                # v3.0: 自动播种示例数据
                logger.info("ChromaDB未初始化,自动播种K线形态数据...")
                self.initialize_vectordb()
                return len(self._chroma_client.list_collections()) > 0
            return True
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
                col = self._chroma_client.get_collection(collection_name)
                count = col.count()
                logger.info(f"ChromaDB collection '{collection_name}' exists ({count} patterns)")
                # v3.1: 如果为空, 重新播种
                if count == 0:
                    self._seed_pattern_data(col)
                return True

            # 创建新collection (v3.1: manual float vectors, no model download needed)
            # v3.1-deerflow: hnsw:space=cosine — 用余弦距离, 否则默认 L2 让相似度失真
            collection = self._chroma_client.create_collection(
                name=collection_name,
                metadata={"description": "K-line pattern vector index (v3.1)",
                          "hnsw:space": "cosine"},
            )

            # v3.1: 播种K线形态数据 (manual float embeddings)
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
            # v3.1-deerflow 修复: 取 seed_vectors 特征向量 (旧 bug 取 seed_patterns
            # 描述文本 → np.array(字符串) 归一化报错被静默吞掉 → 集合 0 条)
            vec = seed_vectors.get(name, [0.0] * 6)
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
            # v3.1: Use manual float embeddings (no model download required)
            collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
            )
            logger.info(f"[ChromaDB] Seeded {len(ids)} K-line patterns (manual vectors)")
        except Exception as e:
            logger.debug(f"[ChromaDB] Seed skipped: {e}")

    def _ensure_kline_text_collection(self) -> Optional[Any]:
        """v3.0: 文本形态检索 collection (哈希向量) — 真实余弦相似度, 供文本查询"""
        try:
            name = "kline_text"
            existing = [c.name for c in self._chroma_client.list_collections()]
            if name in existing:
                return self._chroma_client.get_collection(name)
            col = self._chroma_client.create_collection(
                name=name,
                metadata={"description": "K-line pattern TEXT vector index (hash_v1)",
                          "hnsw:space": "cosine"},  # v3.1-deerflow: 余弦距离
            )
            ids, docs, embeds, metas = [], [], [], []
            for i, (pname, desc) in enumerate(KLINE_PATTERN_TEXTS.items()):
                ids.append(f"txt_{i}_{pname}")
                docs.append(desc)
                embeds.append(_stable_hash_embed(desc))
                metas.append({"pattern": pname, "category": "candlestick_text", "source": "seed_v3.0"})
            col.add(ids=ids, embeddings=embeds, documents=docs, metadatas=metas)
            logger.info(f"[ChromaDB] Seeded kline_text ({len(ids)} patterns, hash vectors)")
            return col
        except Exception as e:
            logger.warning(f"[ChromaDB] kline_text 初始化失败: {e}")
            return None

    def _ensure_knowledge_base_collection(self) -> Optional[Any]:
        """v3.3: 参考文档 RAG collection — 按节分块 + regime 打标 + 加权哈希向量

        v2 集合名 (knowledge_base_v2): 分块/嵌入策略升级, 旧集合不兼容新查询向量。
        """
        try:
            if self._chroma_client is None:
                import chromadb
                store_path = self.root / "vector_store" / "chroma"
                store_path.mkdir(parents=True, exist_ok=True)
                self._chroma_client = chromadb.PersistentClient(path=str(store_path))
            name = "knowledge_base_v2"
            existing = [c.name for c in self._chroma_client.list_collections()]
            if name in existing:
                col = self._chroma_client.get_collection(name)
                if col.count() == 0:
                    self._seed_knowledge_base_v2(col)
                return col
            col = self._chroma_client.create_collection(
                name=name,
                metadata={"description": "Reference docs RAG index v3.3 (section+regime, hash_v2)",
                          "hnsw:space": "cosine"},  # 余弦距离
            )
            self._seed_knowledge_base_v2(col)
            return col
        except Exception as e:
            logger.warning(f"[ChromaDB] knowledge_base_v2 初始化失败: {e}")
            return None

    def _seed_knowledge_base_v2(self, col) -> None:
        """v3.3: 按节分块 + regime 打标 播种参考文档 RAG。"""
        reference_dir = self.root / "reference"
        ids, docs, embeds, metas = [], [], [], []
        if reference_dir.exists():
            for md_file in sorted(reference_dir.glob("*.md")):
                text = md_file.read_text(encoding="utf-8")
                chunks = _chunk_markdown(text)
                for j, (header, chunk) in enumerate(chunks):
                    clip = chunk[:800]
                    ids.append(f"ref_{md_file.stem}_{j}")
                    docs.append(clip)
                    embeds.append(_stable_hash_embed(clip))
                    metas.append({
                        "source": md_file.name, "chunk": j,
                        "section": header,
                        "regime": _regime_from_header(header) or "general",
                        "doc_type": "regime_playbook" if "regime_playbook" in md_file.name else "reference",
                    })
        if ids:
            col.add(ids=ids, embeddings=embeds, documents=docs, metadatas=metas)
            logger.info(f"[ChromaDB] Seeded knowledge_base_v2 ({len(ids)} chunks, section+regime)")

    def index_document(self, md_path: Path, doc_type: str = "reference") -> bool:
        """v3.3: 把新生成的 md (如 trade_lessons.md) 按节分块索引进 knowledge_base_v2。

        连续优化闭环用: 每次跑完回放/实盘后生成的经验文档, 重新索引供 AI 检索。
        已存在的同名 chunk 先删除再重建 (idempotent)。
        """
        try:
            path = Path(md_path)
            if not path.exists():
                logger.warning(f"索引文档不存在: {path}")
                return False
            col = self._ensure_knowledge_base_collection()
            if col is None:
                return False
            text = path.read_text(encoding="utf-8")
            chunks = _chunk_markdown(text)
            ids, docs, embeds, metas = [], [], [], []
            for j, (header, chunk) in enumerate(chunks):
                clip = chunk[:800]
                ids.append(f"gen_{path.stem}_{j}")
                docs.append(clip)
                embeds.append(_stable_hash_embed(clip))
                metas.append({
                    "source": path.name, "chunk": j,
                    "section": header,
                    "regime": _regime_from_header(header) or "general",
                    "doc_type": doc_type,
                })
            if ids:
                # 先删旧 (idempotent)
                try:
                    col.delete(ids=[f"gen_{path.stem}_{j}" for j in range(len(chunks))])
                except Exception:
                    pass
                col.add(ids=ids, embeddings=embeds, documents=docs, metadatas=metas)
                logger.info(f"[ChromaDB] 索引文档 {path.name} ({len(ids)} chunks)")
            return True
        except Exception as e:
            logger.warning(f"[ChromaDB] 索引文档失败: {e}")
            return False

    def search_similar_klines(
        self,
        query: Any = None,
        top_k: int = 30,
    ) -> List[Dict]:
        """
        Retrieve similar K-line patterns from ChromaDB.

        v3.1: Accepts either text query (str) or float vector (List[float]).

        Args:
            query: Text description OR float vector for similarity search
            top_k: Number of results to return

        Returns:
            List of matching patterns with metadata
        """
        if not self.chroma_available:
            logger.debug("ChromaDB unavailable, skipping vector search")
            return []

        if query is None:
            return []

        try:
            # v3.0: 文本查询 → kline_text 集合真实余弦相似度 (哈希嵌入);
            #       向量查询 → kline_patterns 结构向量集合 (indicators 形态搜索用)
            if isinstance(query, str):
                text_col = self._ensure_kline_text_collection()
                if text_col is None:
                    return self._keyword_match_klines(query, top_k)
                results = text_col.query(
                    query_embeddings=[_stable_hash_embed(query)],
                    n_results=top_k,
                    include=["metadatas", "distances", "documents"],  # v3.1-deerflow: 取回中文描述
                )
            else:
                collection = self._chroma_client.get_collection("kline_patterns")
                results = collection.query(
                    query_embeddings=[query],
                    n_results=top_k,
                    include=["metadatas", "distances"],
                )

            # Format results
            formatted = []
            if results and results.get("ids") and results["ids"][0]:
                for i, doc_id in enumerate(results["ids"][0]):
                    meta = results.get("metadatas", [[]])[0]
                    dist = results.get("distances", [[]])[0]
                    formatted.append({
                        "id": doc_id,
                        "pattern": meta[i].get("pattern", "unknown") if i < len(meta) else "unknown",
                        # cosine 距离∈[0,2], 1-dist 即余弦相似度, 裁剪到 [0,1]
                        "similarity": max(0.0, min(1.0, round(1 - dist[i], 3))) if i < len(dist) else 0,
                        "description": results.get("documents", [[""]])[0][i] if results.get("documents") else "",
                    })
            return formatted
        except Exception as e:
            logger.debug(f"向量检索失败: {e}")
            return []

    def _keyword_match_klines(self, query: str, top_k: int = 5) -> List[Dict]:
        """v3.1: Fallback keyword matching for K-line pattern search.

        v3.1-deerflow 修复: 支持中文查询 — 用 KLINE_PATTERN_TEXTS 的中文描述 +
        中英文 n-gram 打分 (旧实现只有英文 patterns_map + 空格分词, 中文查询永远匹配不上)。
        """
        query_tokens = _cn_tokens(query)
        results = []
        for name, desc in KLINE_PATTERN_TEXTS.items():
            desc_lower = desc.lower()
            score = sum(1 for t in query_tokens if t in desc_lower)
            # 英文名匹配 (doji/hammer 等)
            if name.replace("_", " ") in query.lower():
                score += 2
            if score > 0:
                results.append({
                    "id": f"kw_{name}",
                    "pattern": name,
                    "similarity": round(min(0.5 + score * 0.08, 0.95), 3),
                    "description": desc,
                })
        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:top_k]

    def rag_query(self, query: str, top_k: int = 5, regime: Optional[str] = None) -> str:
        """
        从参考文档中检索相关知识片段 (v3.3: regime 感知检索)

        检索策略:
        1. 优先: ChromaDB向量检索 (knowledge_base_v2, 加权哈希向量, 余弦相似度)
           - 指定 regime 时按 metadata.regime 过滤 (bull/bear/crisis/range/general)
        2. 降级: 关键词匹配(零依赖fallback)

        Args:
            query: 查询文本
            top_k: 返回片段数
            regime: 可选市场状态过滤 (bull/bear/crisis/range) — 让检索匹配当前形势

        Returns:
            拼接后的相关文本
        """
        # 尝试向量检索 (v3.3: knowledge_base_v2, 加权哈希 + regime 过滤)
        if self.chroma_available:
            try:
                text_col = self._ensure_knowledge_base_collection()
                if text_col is not None and text_col.count() > 0:
                    q_kwargs = dict(
                        query_embeddings=[_stable_hash_embed(query)],
                        n_results=top_k * 3,  # 过滤后可能不足 top_k
                        include=["documents", "metadatas", "distances"],
                    )
                    if regime:
                        q_kwargs["where"] = {"regime": regime}
                    results = text_col.query(**q_kwargs)
                    if results and results.get("documents") and results["documents"][0]:
                        docs = results["documents"][0]
                        metas = results.get("metadatas", [[]])[0]
                        dists = results.get("distances", [[]])[0]
                        # 过滤后取前 top_k
                        picked = []
                        for i, (doc, meta) in enumerate(zip(docs, metas)):
                            if len(picked) >= top_k:
                                break
                            if not doc or not doc.strip():
                                continue
                            picked.append((doc, meta, dists[i] if i < len(dists) else 1.0))
                        if picked:
                            parts = [f"相关知识库参考 (向量检索, regime={regime or 'all'}):\n"]
                            for doc, meta, dist in picked:
                                source = meta.get("source", "unknown") if meta else "unknown"
                                section = meta.get("section", "") if meta else ""
                                sim = max(0.0, min(1.0, round(1 - dist, 3)))
                                header = f" [{section}]" if section else ""
                                parts.append(f"### {source}{header} (相似度{sim:.0%})\n{doc[:800]}\n")
                            return "\n".join(parts)
            except Exception as e:
                logger.debug(f"向量检索降级到关键词匹配: {e}")

        # Fallback: 关键词匹配 (v3.1-deerflow: 中文 n-gram 打分, 旧的空格分词对中文失效)
        reference_dir = self.root / "reference"
        if not reference_dir.exists():
            return ""

        snippets = []
        query_words = _cn_tokens(query)
        for md_file in reference_dir.glob("*.md"):
            content = md_file.read_text(encoding="utf-8")
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

    def get_regime_playbook(self, regime: str) -> str:
        """v3.3: 确定性返回对应 regime 的作战手册节 (免向量检索, 快)。

        手册: knowledge/reference/regime_playbook.md — 每节 header 含 regime 标签。
        """
        playbook = self.get_reference("regime_playbook")
        if not playbook:
            return ""
        sections = {}
        cur = None
        for line in playbook.split("\n"):
            if line.startswith("## "):
                cur = line[3:].strip()
                sections[cur] = [line]
            elif cur:
                sections[cur].append(line)
        tag = {"strong_bull": "强牛市", "weak_bull": "弱牛市", "range_bound": "震荡",
               "weak_bear": "弱熊市", "strong_bear": "强熊市", "crisis": "危机"}.get(regime, "")
        if not tag:
            return ""
        for header, lines in sections.items():
            if tag in header:
                return "\n".join(lines)[:2000]
        return ""

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
