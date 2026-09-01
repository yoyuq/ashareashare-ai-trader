"""Tab: 📋 全市场行情 — 全市场实时行情 + AI预筛 + 板块热力图"""
from pathlib import Path

import pandas as pd
import streamlit as st

from web.data import _guess_sector, _run_async, get_full_market
from web.render import render_dataframe

ROOT = Path(__file__).parent.parent.parent


def render():
    st.subheader("全市场实时行情")

    # 先加载数据
    df_market = get_full_market()

    # --- AI Pre-Screening ---
    ai_col1, ai_col2, ai_col3 = st.columns([2, 1, 1])
    with ai_col1:
        st.caption("🚀 用免费大模型(智谱GLM-4.7-Flash)并行预筛全市场, 选出Top100再交给DeepSeek深度分析")
    with ai_col2:
        top_n = st.number_input("筛选数量", 20, 200, 100, step=20, key="screen_top_n")
    with ai_col3:
        do_screen = st.button("🚀 AI预筛", type="primary", use_container_width=True,
                              help="将全市场数据分批发送给免费模型并行分析, 选出最优标的")

    if do_screen:
        if df_market is None:
            st.warning("无法获取全市场数据 — 闭市时段数据源可能不可用。请在工作日 9:30-15:00 重试，或先运行 `python -m simulation.daily_runner` 生成分析报告。")
        else:
            try:
                from analysis.market_screener import BatchScreener, rule_based_prefilter
                from web.progress_tracker import ProgressTracker

                screener = BatchScreener()
                available = [m.name for m in screener.models]
                if not available:
                    st.error("❌ 没有可用的免费模型! 请在 .env 中配置 ZHIPU_API_KEY")
                else:
                    # ── 阶段1: 规则预筛 ──
                    tracker = ProgressTracker(
                        pipeline_name=f"全市场AI预筛 → Top{top_n}",
                        total_stocks=len(df_market),
                    )
                    tracker_placeholder = st.empty()

                    # Stage 1: 数据加载 (already done)
                    tracker.update(stage=0, log_msg=f"数据加载完成: {len(df_market):,} 只")
                    tracker.set_stage_total(0, 1)
                    tracker_placeholder.markdown(tracker.render_html(), unsafe_allow_html=True)

                    # Stage 2: 规则预筛
                    tracker.update(stage=1, log_msg="规则预筛启动...")
                    tracker_placeholder.markdown(tracker.render_html(), unsafe_allow_html=True)

                    df_filtered = rule_based_prefilter(df_market, top_n=max(300, top_n * 3))
                    tracker.set_stage_total(1, len(df_filtered))
                    tracker.update(stage=1, stocks_done=len(df_filtered),
                                   log_msg=f"规则预筛完成: {len(df_market):,} → {len(df_filtered):,} 只")
                    tracker_placeholder.markdown(tracker.render_html(), unsafe_allow_html=True)

                    # Stage 3: LLM精筛
                    batch_size = 50
                    total_batches = (len(df_filtered) + batch_size - 1) // batch_size
                    tracker.total_batches = total_batches
                    tracker.total_stocks = len(df_filtered)
                    tracker.set_stage_total(2, total_batches)
                    tracker.update(stage=2, log_msg=f"LLM精筛: {total_batches} 批 × {batch_size} 只, {len(available)} 模型并行")
                    tracker_placeholder.markdown(tracker.render_html(), unsafe_allow_html=True)

                    async def progress_cb(idx, total, stats):
                        model_name = available[idx % len(available)] if available else ""
                        tracker.update(
                            batch_idx=idx + 1,
                            stocks_done=min(stats['processed'], len(df_filtered)),
                            errors=stats['errors'],
                            stage=2,
                            model=model_name,
                            log_msg=f"批次{idx+1}/{total} | 已分析{stats['processed']}只 | 错误{stats['errors']}",
                        )
                        tracker.total_batches = total
                        tracker_placeholder.markdown(tracker.render_html(), unsafe_allow_html=True)

                    async def run_screen():
                        return await screener.screen(df_market, top_n=top_n, progress_callback=progress_cb)

                    top_result, stats = _run_async(run_screen())

                    # Stage 4: 完成
                    if top_result:
                        tracker.update(stage=3, stocks_done=len(df_filtered),
                                       log_msg=f"预筛完成! {len(top_result)} 只候选, 耗时{stats['elapsed']}s")
                        tracker.set_stage_total(3, len(top_result))
                        tracker_placeholder.markdown(tracker.render_html(), unsafe_allow_html=True)

                        st.success(f"✅ 预筛完成! 从 {stats['total']:,} 只中选出 Top {len(top_result)} | "
                                   f"耗时 {stats['elapsed']}s | 模型: {', '.join(available)}")

                        df_result = pd.DataFrame(top_result)
                        render_dataframe(df_result, height=500)

                        # ── DeepSeek深度分析 ──
                        st.divider()
                        dc1, dc2 = st.columns([2, 1])
                        with dc1:
                            st.caption("将预筛结果交给 DeepSeek 做深度分析 (技术面+基本面+风险)")
                        with dc2:
                            if st.button("🔬 DeepSeek深度分析", type="secondary", use_container_width=True):
                                # DeepSeek progress tracker
                                deep_tracker = ProgressTracker(
                                    pipeline_name="DeepSeek深度分析",
                                    total_stocks=len(top_result),
                                )
                                deep_tracker.set_stage_total(0, 1)
                                deep_tracker.update(stage=0, log_msg=f"准备分析 {len(top_result)} 只候选股")
                                deep_placeholder = st.empty()

                                deep_batches = (len(top_result) + 19) // 20
                                deep_tracker.total_batches = deep_batches
                                deep_tracker.set_stage_total(1, deep_batches)
                                deep_tracker.update(stage=1, model="DeepSeek-Chat",
                                                    log_msg=f"启动 {deep_batches} 批深度分析")

                                async def deep_progress_cb(idx, total, stats):
                                    deep_tracker.update(
                                        batch_idx=idx + 1, stocks_done=min((idx + 1) * 20, len(top_result)),
                                        errors=stats.get('errors', 0), stage=1,
                                        log_msg=f"深度分析 {idx+1}/{total} | {min((idx+1)*20, len(top_result))}只完成",
                                    )
                                    deep_tracker.total_batches = total
                                    deep_placeholder.markdown(deep_tracker.render_html(), unsafe_allow_html=True)

                                async def run_deep():
                                    return await screener.screen_with_deepseek(df_market, top_n=top_n)

                                deep_results = _run_async(run_deep())
                                if deep_results:
                                    deep_tracker.update(stage=2, stocks_done=len(deep_results),
                                                        log_msg=f"深度分析完成: {len(deep_results)} 只")
                                    deep_placeholder.markdown(deep_tracker.render_html(), unsafe_allow_html=True)

                                    st.success(f"深度分析完成: {len(deep_results)} 只")
                                    df_deep = pd.DataFrame(deep_results)
                                    render_dataframe(df_deep, height=500)
                    else:
                        tracker.update(log_msg="预筛未返回结果!", errors=1)
                        tracker_placeholder.markdown(tracker.render_html(), unsafe_allow_html=True)
                        st.warning("预筛未返回结果, 请检查模型配置和网络")
            except Exception as e:
                st.error(f"预筛失败: {e}")

    if df_market is None:
        st.warning("无法获取全市场数据。可能原因：网络未连接、闭市时段数据源暂时不可用。请在工作日 9:30-15:00 之间查看实时行情。")

        # 引导用户使用缓存
        cache_path_check = ROOT / "simulation_data" / "full_market_cache.json"
        if cache_path_check.exists():
            st.caption("💡 提示: 点击下方「刷新全部」清除缓存后重新加载可能会恢复数据。")
            if st.button("🔄 强制刷新数据", key="force_refresh_market"):
                st.cache_data.clear()
                st.rerun()
    else:
        # ── 数据来源横幅 (v5.6 P1-15: 显著展示时间戳 + 陈旧标记) ──
        ms = st.session_state.get("_market_source", {})
        src = ms.get("source", "unknown")
        data_date = ms.get("date", "N/A")
        count = ms.get("count", len(df_market))
        lag = ms.get("lag_days", -1)

        # 陈旧数据优先显著警示 (滞后 > 1 自然日即提示, 无论来源)
        if lag > 1:
            st.warning(f"⚠️ 数据已滞后 {lag} 天 (快照 {data_date}) — 当前展示为历史行情, "
                       f"非实时数据。建议交易时段 (9:30-15:00) 查看, 或运行 `python scripts/refresh_market_cache.py` 刷新。")
        if src == "live":
            st.success(f"✅ 实时行情 — {count:,} 只 | 数据时间: {data_date}")
        elif src == "cached":
            st.info(f"⏸️ 市场已收盘 — 显示最近缓存数据 ({data_date}) | {count:,} 只 | 非实时行情, 仅供参考")
        elif src == "report":
            st.info(f"⏸️ 市场已收盘 — 显示最近分析数据 ({data_date}) | {count:,} 只 | 仅含当日被分析的标的, 非全市场行情")
        else:
            st.caption(f"数据来源: {src} | {count:,} 只 | {data_date}")

        total_stocks = len(df_market)
        # --- Top Bar Metrics ---
        m1, m2, m3, m4, m5, m6 = st.columns(6)
        up_count = len(df_market[df_market["pct_change"] > 0]) if "pct_change" in df_market.columns else 0
        down_count = len(df_market[df_market["pct_change"] < 0]) if "pct_change" in df_market.columns else 0
        flat_count = total_stocks - up_count - down_count
        m1.metric("📊 总计", f"{total_stocks:,}只")
        m2.metric("🟢 上涨", f"{up_count}", f"{up_count / total_stocks * 100:.0f}%")
        m3.metric("🔴 下跌", f"{down_count}", f"{down_count / total_stocks * 100:.0f}%")
        m4.metric("⚪ 平盘", f"{flat_count}")
        if "amount" in df_market.columns:
            total_amt = df_market["amount"].sum() / 1e8
            m5.metric("💰 成交额", f"{total_amt:,.0f}亿")
        avg_pct = df_market["pct_change"].mean() if "pct_change" in df_market.columns else 0
        m6.metric("📈 均涨跌", f"{avg_pct:+.2f}%")

        # --- Filters ---
        st.divider()
        f1, f2, f3, f4, f5 = st.columns(5)
        with f1:
            search = st.text_input("🔍 搜索代码/名称", placeholder="eg: 茅台 or 600519")
        with f2:
            price_range = st.selectbox("💰 价格", ["全部", "0-10元", "10-30元", "30-100元", "100-500元", "500元以上"])
        with f3:
            mv_range = st.selectbox("🏢 市值", ["全部", "🟢大盘", "🟡中盘", "🟠小盘", "🔴微盘"])
        with f4:
            sort_by = st.selectbox("📊 排序", ["涨跌幅↓", "涨跌幅↑", "最新价↓", "最新价↑", "成交额↓", "市盈率↓", "换手率↓"])
        with f5:
            exchange_filter = st.selectbox("🏛 交易所", ["全部", "SH(沪市)", "SZ(深市)", "BJ(北交所)"])

        # --- Apply Filters ---
        df_filtered = df_market.copy()

        if search:
            search_lower = search.lower().strip()
            df_filtered = df_filtered[
                df_filtered["_search_key"].str.contains(search_lower, na=False)
            ]

        if price_range != "全部":
            lo, hi = 0, 999999
            if price_range == "0-10元":
                hi = 10
            elif price_range == "10-30元":
                lo, hi = 10, 30
            elif price_range == "30-100元":
                lo, hi = 30, 100
            elif price_range == "100-500元":
                lo, hi = 100, 500
            elif price_range == "500元以上":
                lo = 500
            df_filtered = df_filtered[(df_filtered["price"] >= lo) & (df_filtered["price"] < hi)]

        if mv_range != "全部" and "mv_tier" in df_filtered.columns:
            df_filtered = df_filtered[df_filtered["mv_tier"] == mv_range]

        if exchange_filter != "全部":
            ex = "SH" if "沪" in exchange_filter else ("BJ" if "北" in exchange_filter else "SZ")
            df_filtered = df_filtered[df_filtered["exchange"] == ex]

        # --- Sort ---
        sort_col_map = {
            "涨跌幅↓": ("pct_change", False),
            "涨跌幅↑": ("pct_change", True),
            "最新价↓": ("price", False),
            "最新价↑": ("price", True),
            "成交额↓": ("amount", False),
            "市盈率↓": ("pe_ttm", False),
            "换手率↓": ("turnover", False),
        }
        sort_col, sort_asc = sort_col_map.get(sort_by, ("pct_change", False))
        if sort_col in df_filtered.columns:
            df_filtered = df_filtered.sort_values(sort_col, ascending=sort_asc, na_position="last")

        st.caption(f"显示 {len(df_filtered):,} 只 (全市场 {total_stocks:,} 只)")

        # --- Display Table (HTML渲染, 绕过st.dataframe暗色主题CSS冲突) ---
        display_cols = ["code", "name", "price", "pct_change", "volume", "amount",
                        "turnover", "pe_ttm", "pb", "total_mv", "amplitude", "mv_tier"]
        avail_cols = [c for c in display_cols if c in df_filtered.columns]
        df_show = df_filtered[avail_cols].head(500)
        col_cn = {
            "code": "代码", "name": "名称", "price": "现价", "pct_change": "涨跌幅",
            "volume": "成交量", "amount": "成交额", "turnover": "换手率",
            "pe_ttm": "市盈率", "pb": "市净率", "total_mv": "总市值",
            "amplitude": "振幅", "mv_tier": "市值档",
        }
        fmts = {
            "price": "¥%.2f", "pct_change": "%+.2f%%", "volume": "%.0f",
            "amount": "¥%.0f", "turnover": "%.2f%%", "pe_ttm": "%.1f",
            "pb": "%.2f", "total_mv": "¥%.0f", "amplitude": "%.2f%%",
        }
        render_dataframe(df_show, height=600, col_rename=col_cn, formatters=fmts)

        # --- Top Movers ---
        st.divider()
        l1, l2 = st.columns(2)
        with l1:
            st.subheader("🚀 涨幅榜 TOP10")
            if "pct_change" in df_market.columns:
                top_up = (df_market.nlargest(10, "pct_change")
                          [["code", "name", "price", "pct_change", "amount", "turnover"]]
                          if all(c in df_market.columns for c in ["code", "name", "price", "pct_change", "amount", "turnover"])
                          else df_market.nlargest(10, "pct_change")[["code", "name", "price", "pct_change"]])
                render_dataframe(top_up, max_rows=10, height=350,
                                 col_rename={"code": "代码", "name": "名称", "price": "现价",
                                             "pct_change": "涨跌幅", "amount": "成交额", "turnover": "换手率"},
                                 formatters={"price": "¥%.2f", "pct_change": "%+.2f%%",
                                             "amount": "¥%.0f", "turnover": "%.2f%%"})
        with l2:
            st.subheader("📉 跌幅榜 TOP10")
            if "pct_change" in df_market.columns:
                top_down = (df_market.nsmallest(10, "pct_change")
                            [["code", "name", "price", "pct_change", "amount", "turnover"]]
                            if all(c in df_market.columns for c in ["code", "name", "price", "pct_change", "amount", "turnover"])
                            else df_market.nsmallest(10, "pct_change")[["code", "name", "price", "pct_change"]])
                render_dataframe(top_down, max_rows=10, height=350,
                                 col_rename={"code": "代码", "name": "名称", "price": "现价",
                                             "pct_change": "涨跌幅", "amount": "成交额", "turnover": "换手率"},
                                 formatters={"price": "¥%.2f", "pct_change": "%+.2f%%",
                                             "amount": "¥%.0f", "turnover": "%.2f%%"})

        # --- Sector Heatmap ---
        st.divider()
        st.subheader("板块热力图")
        df_m = df_market.copy()
        df_m["sector"] = df_m["code"].apply(lambda x: _guess_sector(str(x)))
        sector_stats = df_m.groupby("sector").agg(
            数量=("code", "count"),
            均涨幅=("pct_change", "mean"),
            总成交额亿=("amount", "sum"),
            平均PE=("pe_ttm", "mean"),
        ).reset_index()
        sector_stats["总成交额亿"] = sector_stats["总成交额亿"] / 1e8
        sector_stats = sector_stats.sort_values("均涨幅", ascending=False)
        sector_stats["趋势"] = sector_stats["均涨幅"].apply(
            lambda x: "🟢" if x > 1 else ("🟡" if x > -1 else "🔴")
        )

        scols = st.columns(min(len(sector_stats), 4))
        for i, (_, row) in enumerate(sector_stats.iterrows()):
            col = scols[i % 4]
            with col:
                bg = "#1a2f1a" if row["均涨幅"] > 1 else ("#2d1f1a" if row["均涨幅"] < -1 else "#1a2332")
                st.markdown(f"""
                <div style="background:{bg};border:1px solid var(--ds-border,#30363d);border-radius:8px;padding:10px 14px;margin:3px 0">
                    <div style="font-size:14px;color:var(--ds-text-h,#f0f6fc)">{row['趋势']} <b>{row['sector']}</b></div>
                    <div style="font-size:22px;color:{'#3fb950' if row['均涨幅'] > 0 else '#f85149'}">{row['均涨幅']:+.1f}%</div>
                    <div style="font-size:11px;color:var(--ds-text2,#8b949e)">{row['数量']}只 | 成交{row['总成交额亿']:.0f}亿 | PE{row['平均PE']:.0f}</div>
                </div>
                """, unsafe_allow_html=True)
