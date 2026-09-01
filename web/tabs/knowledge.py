"""Tab: 📚 知识库管理 — 三层知识体系浏览 + 向量检索测试"""
import json

import pandas as pd
import streamlit as st

from web.render import render_dataframe


def render():
    st.subheader("📚 知识库管理")
    st.caption("三层知识体系: YAML规则 + Markdown文档 + ChromaDB向量检索")

    try:
        from knowledge.manager import KnowledgeManager
        km = KnowledgeManager()
    except Exception as e:
        st.error(f"知识库初始化失败: {e}")
        km = None

    if km:
        # ── 知识库概览 ──
        kb_root = km.root
        prompt_files = list((kb_root / "prompts" / "system").glob("*.txt"))
        task_files = list((kb_root / "prompts" / "tasks").glob("*.txt"))
        few_shot_files = list((kb_root / "prompts" / "few_shots").glob("*.json"))
        rule_files = list((kb_root / "rules").glob("*.yaml"))
        ref_files = list((kb_root / "reference").glob("*.md"))
        strategy_files = list((kb_root / "strategies").glob("*.yaml"))

        total = len(prompt_files) + len(task_files) + len(few_shot_files) + len(rule_files) + len(ref_files) + len(strategy_files)

        # Metrics row
        mc1, mc2, mc3, mc4, mc5 = st.columns(5)
        mc1.metric("系统提示词", len(prompt_files))
        mc2.metric("任务提示词", len(task_files))
        mc3.metric("Few-shot", len(few_shot_files))
        mc4.metric("规则文件", len(rule_files))
        mc5.metric("总文件数", total)

        # ChromaDB status
        chroma_ok = km.chroma_available
        st.markdown(f"""
        <div style="background-color:var(--ds-bg2,#161b22);border:1px solid var(--ds-border,#30363d);border-radius:8px;padding:12px;margin:8px 0">
            <span style="color:var(--ds-text2,#8b949e)">ChromaDB 向量库: </span>
            <span style="color:{'#3fb950' if chroma_ok else '#f0883e'}">● {'已初始化' if chroma_ok else '未初始化 (首次查询时自动播种)'}</span>
        </div>
        """, unsafe_allow_html=True)

        # ── Tabs for different sections ──
        kbt1, kbt2, kbt3, kbt4 = st.tabs(["📋 提示词", "📖 规则与策略", "📝 参考文档", "🔍 检索测试"])

        with kbt1:
            st.subheader("系统提示词")
            for f in sorted(prompt_files):
                content = f.read_text(encoding="utf-8")
                version = "unknown"
                if content.startswith("---"):
                    try:
                        end = content.index("---", 3)
                        fm = content[3:end].strip()
                        for line in fm.split("\n"):
                            if line.startswith("version:"):
                                version = line.split(":", 1)[1].strip()
                    except ValueError:
                        pass
                with st.expander(f"{f.stem}  v{version}  ({len(content):,} 字)"):
                    st.code(content[:2000], language="markdown")

            if task_files:
                st.subheader("任务提示词")
                for f in sorted(task_files):
                    content = f.read_text(encoding="utf-8")
                    with st.expander(f"{f.stem}  ({len(content):,} 字)"):
                        st.code(content[:2000], language="markdown")

        with kbt2:
            st.subheader("规则文件")
            for f in sorted(rule_files):
                content = f.read_text(encoding="utf-8")
                with st.expander(f"{f.name}  ({len(content):,} 字)"):
                    st.code(content[:3000], language="yaml")

            st.subheader("策略注册表")
            for f in sorted(strategy_files):
                content = f.read_text(encoding="utf-8")
                with st.expander(f"{f.name}  ({len(content):,} 字)"):
                    st.code(content[:3000], language="yaml")

            # 策略统计
            try:
                strategies = km.list_strategies()
                if strategies:
                    strat_data = []
                    for s in strategies:
                        strat_data.append({
                            "策略名称": s.get("name", ""),
                            "类别": s.get("category", ""),
                            "适用体制": ", ".join(s.get("market_regimes", [])),
                            "容量上限": f"{s.get('capacity_limit', 0) / 1e6:.0f}M",
                        })
                    render_dataframe(pd.DataFrame(strat_data), height=400)
            except Exception:
                pass

        with kbt3:
            st.subheader("参考文档")
            for f in sorted(ref_files):
                content = f.read_text(encoding="utf-8")
                # Count sections
                sections = content.count("## ")
                with st.expander(f"{f.stem}  ({len(content):,} 字, {sections} 章节)"):
                    st.markdown(content[:5000])

            st.subheader("Few-shot 样例")
            for f in sorted(few_shot_files):
                data = json.loads(f.read_text(encoding="utf-8"))
                if isinstance(data, list):
                    examples = data
                    scenario = f.stem
                else:
                    examples = data.get("examples", [])
                    scenario = data.get("scenario", f.stem)
                with st.expander(f"{scenario}  ({len(examples)} 示例)"):
                    st.json(data)

        with kbt4:
            st.subheader("🔍 知识库检索测试")
            st.caption("测试 ChromaDB 向量检索和关键词匹配")

            query = st.text_input("检索查询", placeholder="例如: 锤子线形态、T+1交易规则、MACD金叉...",
                                  key="kb_search_query")
            if query:
                with st.spinner("检索中..."):
                    try:
                        result = km.rag_query(query)
                        st.success(f"检索完成 — 返回 {len(str(result)):,} 字")
                        st.markdown(f"```\n{str(result)[:3000]}\n```")
                    except Exception as e:
                        st.warning(f"检索出错: {e}")

            st.divider()
            st.subheader("📊 提示词版本统计")
            try:
                all_prompts = km.list_all_prompts()
                if all_prompts:
                    vdata = []
                    for p in all_prompts:
                        vdata.append({
                            "名称": p["name"],
                            "版本": p["version"],
                            "更新日期": p["date"],
                            "大小": f"{p['size_chars']:,}字",
                        })
                    render_dataframe(pd.DataFrame(vdata), height=400)
            except Exception as e:
                st.warning(f"无法获取版本信息: {e}")

    else:
        st.info("知识库管理器未初始化,请检查 knowledge/ 目录结构")
