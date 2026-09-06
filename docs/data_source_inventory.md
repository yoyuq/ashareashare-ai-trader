# 数据源能力盘点（真实网络探测）

探测时间：2026-09-06T14:16:57.571180+00:00 至 2026-09-06T14:23:36.250329+00:00；Python 3.12.10；AKShare 1.18.64；Baostock 0.9.3。

本表只记录数据能力，不作策略判断。历史起点全部来自本次返回数据，不引用文档声称的起始日。起点是固定样本中的最早值，不等于全市场最早；覆盖率只代表这 10 只跨沪深主板、创业板、科创板样本，不外推至北交所、退市股或全市场。

固定样本：600519、600036、601318、600030、600276、000001、000858、000333、300750、688981。宏观没有股票维度，覆盖记为不适用，不把重复请求冒充 10 只覆盖。

每个 HTTP 样本先用指定本地代理，失败后清除全部代理设置直连一次；Baostock 是 TCP 直连。调用失败/空数据标不可用；进程超时且未留下证据的样本标未完成。稳定性仅为本次调用的成功率/耗时，未压测频率上限；本次源内间隔 0.15 秒、3 个独立进程，单 HTTP 请求 6 秒。

东财端点直连指绕过 AKShare 封装直接请求 API；网络仍可能经代理，以 attempts.mode 为准，不能据此宣称无需代理。按原始日期字段升序、pageSize=500 分页；本次最多取 2 页以验证历史起点和股票过滤，server_count/server_pages/truncated 留在证据中。截断数据的 last 仅为返回页末日，不代表数据库最新日。财报最早报告期可能早于上市，报告期不等于当时已公开。带公告/更新时间字段只能做日期对齐，未证明版本历史完整，均不标完整 PIT。

重跑（不加载项目配置或密钥）：

```powershell
$env:PYTHON_DOTENV_DISABLED='1'
.venv\Scripts\python.exe scripts/probe_source_inventory.py
# 单独复测不会覆盖正式全量矩阵：
.venv\Scripts\python.exe scripts/probe_source_inventory.py --only holder balance news --output docs/data_source_inventory_partial.md
```

完整机器证据：[JSON](data_source_inventory_evidence.json)。--codes 必须提供至少 10 个不同六位 A 股代码；--pages 控制东财历史页数，--source-timeout 控制单源截止时间。脚本拒绝写入 reports/ 和密钥文件。

## 实测矩阵

| 类别 / 接口或端点 | 数据内容 | 历史起点（实测） | 频率粒度 | PIT 质量（实测字段） | 覆盖度（抽样实测） | 调用稳定性 | 已知坑 |
|---|---|---|---|---|---|---|---|
| 行情 / `ak.stock_zh_a_hist` ([证据](#evidence-daily_em)) | 东财不复权 OHLCV/成交额/换手 | 不可用，无法实测 | 日 | 不可用，未验证 | 0/10 返回非空 | 均失败；尝试成功 0/20；最大 8.31s；ConnectionError: ('Connection aborted.', RemoteDisconnected('Remote end closed connection witho | 从 1990-01-01 请求；不复权避免混入今天的复权因子；价格仍可能校正 |
| 行情 / `ak.stock_zh_a_hist_tx` ([证据](#evidence-daily_tx)) | 腾讯不复权 OHLCV | 样本最早 1991-04-03 | 日 | 未返回公告/发布时间字段 | 10/10 返回非空 | proxy；尝试成功 10/10；最大 35.69s | 从 1990-01-01 请求；AKShare 按年多次请求，耗时较长 |
| 行情 / `ak.stock_zh_a_hist_min_em` ([证据](#evidence-minute_em)) | 东财 5 分钟 K 线 | 不可用，无法实测 | 5 分钟 | 不可用，未验证 | 0/10 返回非空 | 均失败；尝试成功 0/20；最大 5.31s；ConnectionError: ('Connection aborted.', RemoteDisconnected('Remote end closed connection witho | 请求从 1990 年开始；只能量到服务器保留窗口，不能视为上市以来分钟历史 |
| 行情 / `qt.gtimg.cn/q=…` ([证据](#evidence-quote_tx)) | 腾讯实时价/昨收/成交量/PE/PB | 仅快照；返回日 2026-09-04 | 实时快照 | 未返回公告/发布时间字段 | 10/10 返回非空 | proxy；尝试成功 10/10；最大 1.55s | 仅当前快照；周末可能返回上个交易日，时间戳不是采集时刻 |
| 行情 / `bs.query_history_k_data_plus` ([证据](#evidence-daily_bs)) | Baostock 不复权日线/PE/PB/停牌/ST | 样本最早 1991-04-03 | 日 | 未返回公告/发布时间字段 | 10/10 返回非空 | tcp_direct；尝试成功 10/10；最大 34.41s | 从 1990-01-01 请求；TCP 不走 HTTP 代理；用 next/get_row_data，避免旧 get_data 的 DataFrame.append 不兼容；空串不是零 |
| 基本面 / `ak.stock_individual_info_em` ([证据](#evidence-profile_em)) | 总股本/流通股/市值/行业/上市日期 | 不可用，无法实测 | 当前快照 | 不可用，未验证 | 0/10 返回非空 | 均失败；尝试成功 0/20；最大 3.97s；ConnectionError: ('Connection aborted.', RemoteDisconnected('Remote end closed connection witho | 上市日期是属性，不是可回溯历史起点；覆盖成功不代表所有属性完整 |
| 基本面 / `RPT_HOLDERNUM_DET` ([证据](#evidence-holder)) | 股东户数及变动 | 样本最早 2013-03-01 | 不定期/报告期 | 有 HOLD_NOTICE_DATE；版本 PIT 未证实 | 10/10 返回非空 | proxy；尝试成功 10/10；最大 5.19s | HOLD_NOTICE_DATE 可对齐公告；历史记录可能被修订 |
| 基本面 / `RPT_LICO_FN_CPD` ([证据](#evidence-financial)) | 业绩报表/营收/利润/EPS | 样本最早 1989-12-31 | 季/年 | 有 EITIME/NOTICE_DATE/UPDATE_DATE；版本 PIT 未证实 | 10/10 返回非空 | direct/proxy；尝试成功 10/11；最大 14.16s；ConnectionError: HTTPSConnectionPool(host='datacenter-web.eastmoney.com', port=443): Read timed | 直连保留 NOTICE_DATE/REPORTDATE；公告日不等于所有数值的首发版本 |
| 基本面 / `RPT_DMSK_FN_BALANCE` ([证据](#evidence-balance)) | 资产负债表主要项目 | 样本最早 1989-12-31 | 季/年 | 有 NOTICE_DATE；版本 PIT 未证实 | 10/10 返回非空 | direct/proxy；尝试成功 10/11；最大 6.44s；ReadTimeout: HTTPSConnectionPool(host='datacenter-web.eastmoney.com', port=443): Read timed out | 直连不套行业过滤；银行与非金融企业字段适用性不同 |
| 基本面 / `RPT_DMSK_FN_INCOME` ([证据](#evidence-income)) | 利润表主要项目 | 样本最早 1989-12-31 | 季/年 | 有 NOTICE_DATE；版本 PIT 未证实 | 10/10 返回非空 | proxy；尝试成功 10/10；最大 2.47s | 累计口径不是单季口径；历史值可能追溯调整 |
| 基本面 / `RPT_DMSK_FN_CASHFLOW` ([证据](#evidence-cashflow)) | 现金流量表主要项目 | 样本最早 1998-06-30 | 季/年 | 有 NOTICE_DATE；版本 PIT 未证实 | 10/10 返回非空 | proxy；尝试成功 10/10；最大 5.94s | 累计口径；当前可见版本不构成完整 PIT |
| 基本面 / `bs.query_profit_data` ([证据](#evidence-profit_bs)) | Baostock 盈利能力指标 | 样本最早 2007-03-31 | 季 | 有 pubDate；版本 PIT 未证实 | 10/10 返回非空 | tcp_direct；尝试成功 10/10；最大 13.14s | 从 2007Q1 逐季找首个非空季；只探最早非空季，不能把查询下界当服务起点 |
| 资金流 / `ak.stock_individual_fund_flow` ([证据](#evidence-flow_em)) | 主力/超大单/大中小单净流入 | 样本最早 2026-03-16 | 日 | 未返回公告/发布时间字段 | 1/10 返回非空 | proxy；尝试成功 1/19；最大 7.05s；ConnectionError: ('Connection aborted.', RemoteDisconnected('Remote end closed connection witho | 返回滚动保留窗口；资金分类为供应商算法口径，无公告日 |
| 资金流 / `RPT_MUTUAL_HOLDSTOCKNDATE_STA` ([证据](#evidence-north)) | 北向个股持股/增减/持股市值 | 样本最早 2017-03-16 | 交易日 | 未返回公告/发布时间字段 | 10/10 返回非空 | direct/proxy；尝试成功 10/11；最大 8.27s；ReadTimeout: HTTPSConnectionPool(host='datacenter-web.eastmoney.com', port=443): Read timed out | 披露政策变化后序列可能停止更新；只记录实际返回的保留历史 |
| 事件 / `RPT_PUBLIC_BS_APPOIN` ([证据](#evidence-appointment)) | 定期报告预约/变更/实际披露日 | 样本最早 2006-12-31 | 事件 | 有 ACTUAL_PUBLISH_DATE/EITIME/FIRST_APPOINT_DATE；版本 PIT 未证实 | 10/10 返回非空 | proxy；尝试成功 10/10；最大 4.70s | 预约修改字段不含每次修改获知时间，不能重建每个历史时点的预约快照 |
| 事件 / `RPT_PUBLIC_OP_NEWPREDICT` ([证据](#evidence-forecast)) | 业绩预告区间/类型/原因 | 样本最早 2003-12-31 | 事件 | 有 NOTICE_DATE；版本 PIT 未证实 | 10/10 返回非空 | direct/proxy；尝试成功 10/11；最大 6.62s；ReadTimeout: HTTPSConnectionPool(host='datacenter-web.eastmoney.com', port=443): Read timed out | 同一报告期多指标/多版本；必须以公告日对齐且检查修订 |
| 事件 / `RPT_DAILYBILLBOARD_DETAILSNEW` ([证据](#evidence-lhb)) | 龙虎榜公开交易明细 | 样本最早 2006-11-09 | 交易日事件 | 未返回公告/发布时间字段 | 10/10 返回非空 | proxy；尝试成功 10/10；最大 8.97s | 非每只股票每天都有事件；D1/D5 等事后涨跌幅不可当时点特征 |
| 事件 / `RPT_BLOCKTRADE_STA` ([证据](#evidence-blocktrade)) | 大宗交易每日统计 | 样本最早 2003-04-10 | 交易日事件 | 未返回公告/发布时间字段 | 10/10 返回非空 | direct/proxy；尝试成功 10/12；最大 10.36s；ReadTimeout: HTTPSConnectionPool(host='datacenter-web.eastmoney.com', port=443): Read timed out | 事件覆盖率不是全市场覆盖率；含事后收益列，使用时需排除 |
| 舆情 / `ak.stock_hot_rank_detail_em` ([证据](#evidence-hotrank)) | 东方财富历史人气/粉丝比例 | 样本最早 2025-09-06 | 日 | 未返回公告/发布时间字段 | 10/10 返回非空 | proxy；尝试成功 10/10；最大 4.45s | 排名与粉丝为两次 HTTP 请求；按位置拼接，需注意日期对齐；无首发版本时间 |
| 舆情 / `RPT_DMSK_TS_STOCKNEW` ([证据](#evidence-comment)) | 千股千评/关注/排名/综合得分 | 仅快照；返回日 2026-09-04 | 日快照 | 未返回公告/发布时间字段 | 10/10 返回非空 | proxy；尝试成功 10/10；最大 4.17s | 只探到当前快照；不能从日期列推断存在日度历史 |
| 舆情 / `ak.stock_news_em` ([证据](#evidence-news)) | 按股票代码搜索财经新闻 | 不可用，无法实测 | 新闻事件 | 不可用，未验证 | 0/10 返回非空 | 均失败；尝试成功 0/20；最大 2.36s；ArrowInvalid: Invalid regular expression: invalid escape sequence: \u | 当前封装 pageSize=10，相关性排序；实测最早只是返回页最早，非完整历史；关键词命中不等于实体准确关联 |
| 宏观 / `RPT_ECONOMY_GDP` ([证据](#evidence-gdp)) | 中国 GDP/产业增加值 | 样本最早 2006-03-01 | 季 | 未返回公告/发布时间字段 | 宏观序列 1/1，82 期；个股 N/A | proxy；尝试成功 1/1；最大 1.73s | 宏观修订序列，不等于历史首发版本；非个股数据 |
| 宏观 / `RPT_ECONOMY_CPI` ([证据](#evidence-cpi)) | 全国/城乡 CPI | 样本最早 2008-01-01 | 月 | 未返回公告/发布时间字段 | 宏观序列 1/1，223 期；个股 N/A | proxy；尝试成功 1/1；最大 2.59s | 统计期不等于发布日期；非个股数据 |
| 宏观 / `RPT_ECONOMY_PMI` ([证据](#evidence-pmi)) | 制造业/非制造业 PMI | 样本最早 2008-01-01 | 月 | 未返回公告/发布时间字段 | 宏观序列 1/1，224 期；个股 N/A | proxy；尝试成功 1/1；最大 1.70s | 统计期不等于发布日期；非个股数据 |
| 宏观 / `RPT_ECONOMY_CURRENCY_SUPPLY` ([证据](#evidence-money)) | M0/M1/M2 货币供应量 | 样本最早 2008-01-01 | 月 | 未返回公告/发布时间字段 | 宏观序列 1/1，223 期；个股 N/A | proxy；尝试成功 1/1；最大 1.52s | 定义变更/修订不可由当前序列还原；非个股数据 |

## 同底层重复与保留依据

下列 AKShare 封装与对应直连使用相同 reportName，矩阵只保留直连一行。实测比较代表性的逐股股东户数、全市场业绩报表、宏观 GDP 封装（结果如下）；其余同族按原始字段保留和受控分页选择直连，未声称逐个证明所有环境中最稳。腾讯、东财、Baostock 日线来自不同服务；巨潮 stock_report_disclosure 也不是东财 stock_yysj_em 的同底层别名。

对照两侧返回范围不同（封装业绩报表取去年年报全市场、直连取每只样本历史），下列开销为本次累计耗时，不是严格同负载性能测试。

| 封装代表 | 封装本次结果 | 端点本次结果 | 直连保留依据 |
|---|---|---|---|
| `ak.stock_zh_a_gdhs_detail_em` | 调用返回 10/10；累计 45.89s | 非空 10/10；累计 27.84s | 本次覆盖相同；端点保留原始日期/公告列、排序及分页证据；未声称稳定性显著更优 |
| `ak.stock_yjbb_em` | 调用返回 1/1；固定股票覆盖 10/10；累计 73.39s | 非空 10/10；累计 34.77s | 本次覆盖相同；端点保留原始日期/公告列、排序及分页证据；未声称稳定性显著更优 |
| `ak.macro_china_gdp` | 调用返回 1/1；累计 1.75s | 非空 1/1；累计 1.73s | 本次覆盖相同；端点保留原始日期/公告列、排序及分页证据；未声称稳定性显著更优 |

- `RPT_HOLDERNUM_DET` ← `ak.stock_zh_a_gdhs_detail_em`
- `RPT_LICO_FN_CPD` ← `ak.stock_yjbb_em`
- `RPT_DMSK_FN_BALANCE` ← `ak.stock_zcfz_em`
- `RPT_DMSK_FN_INCOME` ← `ak.stock_lrb_em`
- `RPT_DMSK_FN_CASHFLOW` ← `ak.stock_xjll_em`
- `RPT_MUTUAL_HOLDSTOCKNDATE_STA` ← `ak.stock_hsgt_individual_em`
- `RPT_PUBLIC_BS_APPOIN` ← `ak.stock_yysj_em`
- `RPT_PUBLIC_OP_NEWPREDICT` ← `ak.stock_yjyg_em`
- `RPT_DAILYBILLBOARD_DETAILSNEW` ← `ak.stock_lhb_detail_em`
- `RPT_BLOCKTRADE_STA` ← `ak.stock_dzjy_mrtj`
- `RPT_DMSK_TS_STOCKNEW` ← `ak.stock_comment_em`
- `RPT_ECONOMY_GDP` ← `ak.macro_china_gdp`
- `RPT_ECONOMY_CPI` ← `ak.macro_china_cpi`
- `RPT_ECONOMY_PMI` ← `ak.macro_china_pmi`
- `RPT_ECONOMY_CURRENCY_SUPPLY` ← `ak.macro_china_money_supply`

新增直连验证范围是除已知股东户数和预约披露外的矩阵 reportName；是否实际可用见各行，失败端点不计已验证可用。

## AKShare 公开接口枚举（符号级索引）

这是已安装版本导出名的可重跑枚举，按功能关键字分组，混合市场接口可能包含非 A 股。排除明确 HK/US/B 股命名。它用于界定后续扩展范围：只有实测矩阵及同底层映射具有本次能力证据，其余全部**未测**，不宣称可用或具有历史覆盖。

### 行情（160 个导出符号）

`stock_a_all_pb`, `stock_a_below_net_asset_statistics`, `stock_a_code_to_symbol`, `stock_a_congestion_lg`, `stock_a_gxl_lg`, `stock_a_high_low_statistics`, `stock_a_ttm_lyr`, `stock_account_statistics_em`, `stock_allotment_cninfo`, `stock_analyst_detail_em`, `stock_analyst_rank_em`, `stock_balance_sheet_by_report_em`, `stock_balance_sheet_by_yearly_em`, `stock_bid_ask_em`, `stock_bj_a_spot_em`, `stock_board_change_em`, `stock_board_concept_cons_em`, `stock_board_concept_hist_em`, `stock_board_concept_hist_min_em`, `stock_board_concept_index_ths`, `stock_board_concept_info_ths`, `stock_board_concept_name_em`, `stock_board_concept_name_ths`, `stock_board_concept_spot_em`, `stock_board_concept_summary_ths`, `stock_buffett_index_lg`, `stock_cash_flow_sheet_by_quarterly_em`, `stock_cash_flow_sheet_by_report_em`, `stock_cash_flow_sheet_by_yearly_em`, `stock_cg_equity_mortgage_cninfo`, `stock_cg_guarantee_cninfo`, `stock_cg_lawsuit_cninfo`, `stock_changes_em`, `stock_classify_sina`, `stock_concept_cons_futu`, `stock_cy_a_spot_em`, `stock_cyq_em`, `stock_dxsyl_em`, `stock_ebs_lg`, `stock_esg_hz_sina`, `stock_esg_msci_sina`, `stock_esg_rate_sina`, `stock_esg_rft_sina`, `stock_esg_zd_sina`, `stock_fhps_detail_em`, `stock_fhps_detail_ths`, `stock_fhps_em`, `stock_gddh_em`, `stock_individual_basic_info_xq`, `stock_individual_spot_xq`, `stock_info_a_code_name`, `stock_info_bj_name_code`, `stock_info_change_name`, `stock_info_cjzc_em`, `stock_info_global_cls`, `stock_info_global_em`, `stock_info_global_futu`, `stock_info_global_sina`, `stock_info_global_ths`, `stock_info_sh_name_code`, `stock_info_sz_change_name`, `stock_info_sz_name_code`, `stock_inner_trade_xq`, `stock_institute_recommend`, `stock_institute_recommend_detail`, `stock_intraday_em`, `stock_intraday_sina`, `stock_irm_ans_cninfo`, `stock_irm_cninfo`, `stock_jgdy_detail_em`, `stock_jgdy_tj_em`, `stock_kc_a_spot_em`, `stock_lh_yyb_capital`, `stock_lh_yyb_control`, `stock_lh_yyb_most`, `stock_management_change_ths`, `stock_market_activity_legu`, `stock_pg_em`, `stock_price_js`, `stock_profit_forecast_em`, `stock_profit_forecast_ths`, `stock_profit_sheet_by_quarterly_em`, `stock_profit_sheet_by_report_em`, `stock_profit_sheet_by_yearly_em`, `stock_qbzf_em`, `stock_qsjy_em`, `stock_rank_cxd_ths`, `stock_rank_cxfl_ths`, `stock_rank_cxg_ths`, `stock_rank_cxsl_ths`, `stock_rank_forecast_cninfo`, `stock_rank_ljqd_ths`, `stock_rank_ljqs_ths`, `stock_rank_lxsz_ths`, `stock_rank_lxxd_ths`, `stock_rank_xstp_ths`, `stock_rank_xxtp_ths`, `stock_rank_xzjp_ths`, `stock_sector_detail`, `stock_sector_spot`, `stock_sgt_reference_exchange_rate_sse`, `stock_sgt_reference_exchange_rate_szse`, `stock_sgt_settlement_exchange_rate_sse`, `stock_sgt_settlement_exchange_rate_szse`, `stock_sh_a_spot_em`, `stock_share_change_cninfo`, `stock_sse_deal_daily`, `stock_sse_summary`, `stock_sy_em`, `stock_sy_hy_em`, `stock_sy_jz_em`, `stock_sy_yq_em`, `stock_sz_a_spot_em`, `stock_szse_area_summary`, `stock_szse_sector_summary`, `stock_szse_summary`, `stock_tfp_em`, `stock_value_em`, `stock_xgsglb_em`, `stock_xgsr_ths`, `stock_yysj_em`, `stock_yzxdr_em`, `stock_zh_a_cdr_daily`, `stock_zh_a_daily`, `stock_zh_a_gbjg_em`, `stock_zh_a_hist`, `stock_zh_a_hist_min_em`, `stock_zh_a_hist_pre_min_em`, `stock_zh_a_hist_tx`, `stock_zh_a_minute`, `stock_zh_a_new`, `stock_zh_a_spot`, `stock_zh_a_spot_em`, `stock_zh_a_st_em`, `stock_zh_a_tick_tx_js`, `stock_zh_ab_comparison_em`, `stock_zh_ah_daily`, `stock_zh_ah_name`, `stock_zh_ah_spot`, `stock_zh_ah_spot_em`, `stock_zh_dupont_comparison_em`, `stock_zh_growth_comparison_em`, `stock_zh_index_daily`, `stock_zh_index_daily_em`, `stock_zh_index_daily_tx`, `stock_zh_index_hist_csindex`, `stock_zh_index_spot_em`, `stock_zh_index_spot_sina`, `stock_zh_index_value_csindex`, `stock_zh_kcb_daily`, `stock_zh_kcb_report_em`, `stock_zh_kcb_spot`, `stock_zh_scale_comparison_em`, `stock_zh_vote_baidu`, `stock_zt_pool_dtgc_em`, `stock_zt_pool_em`, `stock_zt_pool_previous_em`, `stock_zt_pool_strong_em`, `stock_zt_pool_zbgc_em`, `stock_zyjs_ths`

### 基本面（67 个导出符号）

`stock_board_industry_cons_em`, `stock_board_industry_hist_em`, `stock_board_industry_hist_min_em`, `stock_board_industry_index_ths`, `stock_board_industry_info_ths`, `stock_board_industry_name_em`, `stock_board_industry_name_ths`, `stock_board_industry_spot_em`, `stock_board_industry_summary_ths`, `stock_circulate_stock_holder`, `stock_financial_abstract`, `stock_financial_abstract_ths`, `stock_financial_analysis_indicator`, `stock_financial_analysis_indicator_em`, `stock_financial_benefit_ths`, `stock_financial_cash_ths`, `stock_financial_debt_ths`, `stock_financial_report_sina`, `stock_fund_stock_holder`, `stock_gdfx_free_holding_analyse_em`, `stock_gdfx_free_holding_change_em`, `stock_gdfx_free_holding_detail_em`, `stock_gdfx_free_holding_statistics_em`, `stock_gdfx_free_holding_teamwork_em`, `stock_gdfx_free_top_10_em`, `stock_gdfx_holding_analyse_em`, `stock_gdfx_holding_change_em`, `stock_gdfx_holding_detail_em`, `stock_gdfx_holding_statistics_em`, `stock_gdfx_holding_teamwork_em`, `stock_gdfx_top_10_em`, `stock_hold_change_cninfo`, `stock_hold_control_cninfo`, `stock_hold_management_detail_cninfo`, `stock_hold_management_detail_em`, `stock_hold_management_person_em`, `stock_hold_num_cninfo`, `stock_index_pb_lg`, `stock_index_pe_lg`, `stock_individual_info_em`, `stock_industry_category_cninfo`, `stock_industry_change_cninfo`, `stock_industry_clf_hist_sw`, `stock_industry_pe_ratio_cninfo`, `stock_institute_hold`, `stock_institute_hold_detail`, `stock_lrb_em`, `stock_main_stock_holder`, `stock_market_pb_lg`, `stock_market_pe_lg`, `stock_profile_cninfo`, `stock_report_fund_hold`, `stock_report_fund_hold_detail`, `stock_share_hold_change_bse`, `stock_share_hold_change_sse`, `stock_share_hold_change_szse`, `stock_shareholder_change_ths`, `stock_sy_profile_em`, `stock_xjll_em`, `stock_yjbb_em`, `stock_zcfz_bj_em`, `stock_zcfz_em`, `stock_zh_a_gdhs`, `stock_zh_a_gdhs_detail_em`, `stock_zh_valuation_baidu`, `stock_zh_valuation_comparison_em`, `stock_zygc_em`

### 资金流（29 个导出符号）

`stock_concept_fund_flow_hist`, `stock_fund_flow_big_deal`, `stock_fund_flow_concept`, `stock_fund_flow_individual`, `stock_fund_flow_industry`, `stock_hsgt_board_rank_em`, `stock_hsgt_fund_flow_summary_em`, `stock_hsgt_fund_min_em`, `stock_hsgt_hist_em`, `stock_hsgt_hold_stock_em`, `stock_hsgt_individual_detail_em`, `stock_hsgt_individual_em`, `stock_hsgt_institution_statistics_em`, `stock_hsgt_stock_statistics_em`, `stock_individual_fund_flow`, `stock_individual_fund_flow_rank`, `stock_lhb_ggtj_sina`, `stock_main_fund_flow`, `stock_margin_account_info`, `stock_margin_detail_sse`, `stock_margin_detail_szse`, `stock_margin_ratio_pa`, `stock_margin_sse`, `stock_margin_szse`, `stock_margin_underlying_info_szse`, `stock_market_fund_flow`, `stock_sector_fund_flow_hist`, `stock_sector_fund_flow_rank`, `stock_sector_fund_flow_summary`

### 事件（77 个导出符号）

`stock_add_stock`, `stock_balance_sheet_by_report_delisted_em`, `stock_cash_flow_sheet_by_report_delisted_em`, `stock_dividend_cninfo`, `stock_dzjy_hygtj`, `stock_dzjy_hyyybtj`, `stock_dzjy_mrmx`, `stock_dzjy_mrtj`, `stock_dzjy_sctj`, `stock_dzjy_yybph`, `stock_financial_abstract_new_ths`, `stock_financial_benefit_new_ths`, `stock_financial_cash_new_ths`, `stock_financial_debt_new_ths`, `stock_ggcg_em`, `stock_gpzy_distribute_statistics_bank_em`, `stock_gpzy_distribute_statistics_company_em`, `stock_gpzy_individual_pledge_ratio_detail_em`, `stock_gpzy_industry_data_em`, `stock_gpzy_pledge_ratio_detail_em`, `stock_gpzy_pledge_ratio_em`, `stock_gpzy_profile_em`, `stock_gsrl_gsdt_em`, `stock_history_dividend`, `stock_history_dividend_detail`, `stock_individual_notice_report`, `stock_info_sh_delist`, `stock_info_sz_delist`, `stock_ipo_benefit_ths`, `stock_ipo_declare_em`, `stock_ipo_info`, `stock_ipo_review_em`, `stock_ipo_summary_cninfo`, `stock_ipo_ths`, `stock_ipo_tutor_em`, `stock_lhb_detail_daily_sina`, `stock_lhb_detail_em`, `stock_lhb_hyyyb_em`, `stock_lhb_jgmmtj_em`, `stock_lhb_jgmx_sina`, `stock_lhb_jgstatistic_em`, `stock_lhb_jgzz_sina`, `stock_lhb_stock_detail_date_em`, `stock_lhb_stock_detail_em`, `stock_lhb_stock_statistic_em`, `stock_lhb_traderstatistic_em`, `stock_lhb_yyb_detail_em`, `stock_lhb_yybph_em`, `stock_lhb_yytj_sina`, `stock_new_a_spot_em`, `stock_new_gh_cninfo`, `stock_new_ipo_cninfo`, `stock_notice_report`, `stock_profit_sheet_by_report_delisted_em`, `stock_register_all_em`, `stock_register_bj`, `stock_register_cyb`, `stock_register_db`, `stock_register_kcb`, `stock_register_sh`, `stock_register_sz`, `stock_report_disclosure`, `stock_repurchase_em`, `stock_restricted_release_detail_em`, `stock_restricted_release_queue_em`, `stock_restricted_release_queue_sina`, `stock_restricted_release_stockholder_em`, `stock_restricted_release_summary_em`, `stock_staq_net_stop`, `stock_yjkb_em`, `stock_yjyg_em`, `stock_zdhtmx_em`, `stock_zh_a_disclosure_relation_cninfo`, `stock_zh_a_disclosure_report_cninfo`, `stock_zh_a_new_em`, `stock_zh_a_stop_em`, `stock_zt_pool_sub_new_em`

### 舆情（22 个导出符号）

`stock_comment_detail_scrd_desire_em`, `stock_comment_detail_scrd_focus_em`, `stock_comment_detail_zhpj_lspf_em`, `stock_comment_detail_zlkp_jgcyd_em`, `stock_comment_em`, `stock_hot_deal_xq`, `stock_hot_follow_xq`, `stock_hot_keyword_em`, `stock_hot_rank_detail_em`, `stock_hot_rank_detail_realtime_em`, `stock_hot_rank_em`, `stock_hot_rank_latest_em`, `stock_hot_rank_relate_em`, `stock_hot_search_baidu`, `stock_hot_tweet_xq`, `stock_hot_up_em`, `stock_js_weibo_nlp_time`, `stock_js_weibo_report`, `stock_news_em`, `stock_news_main_cx`, `stock_research_report_em`, `stock_sns_sseinfo`

### 宏观（75 个导出符号）

`macro_china_agricultural_index`, `macro_china_agricultural_product`, `macro_china_au_report`, `macro_china_bank_financing`, `macro_china_bdti_index`, `macro_china_bond_public`, `macro_china_bsi_index`, `macro_china_central_bank_balance`, `macro_china_commodity_price_index`, `macro_china_construction_index`, `macro_china_construction_price_index`, `macro_china_consumer_goods_retail`, `macro_china_cpi`, `macro_china_cpi_monthly`, `macro_china_cpi_yearly`, `macro_china_cx_pmi_yearly`, `macro_china_cx_services_pmi_yearly`, `macro_china_czsr`, `macro_china_daily_energy`, `macro_china_energy_index`, `macro_china_enterprise_boom_index`, `macro_china_exports_yoy`, `macro_china_fdi`, `macro_china_foreign_exchange_gold`, `macro_china_freight_index`, `macro_china_fx_gold`, `macro_china_fx_reserves_yearly`, `macro_china_gdp`, `macro_china_gdp_yearly`, `macro_china_gdzctz`, `macro_china_gyzjz`, `macro_china_hgjck`, `macro_china_imports_yoy`, `macro_china_industrial_production_yoy`, `macro_china_insurance`, `macro_china_insurance_income`, `macro_china_international_tourism_fx`, `macro_china_lpi_index`, `macro_china_lpr`, `macro_china_m2_yearly`, `macro_china_market_margin_sh`, `macro_china_market_margin_sz`, `macro_china_mobile_number`, `macro_china_money_supply`, `macro_china_national_tax_receipts`, `macro_china_nbs_nation`, `macro_china_nbs_region`, `macro_china_new_financial_credit`, `macro_china_new_house_price`, `macro_china_non_man_pmi`, `macro_china_passenger_load_factor`, `macro_china_pmi`, `macro_china_pmi_yearly`, `macro_china_postal_telecommunicational`, `macro_china_ppi`, `macro_china_ppi_yearly`, `macro_china_qyspjg`, `macro_china_real_estate`, `macro_china_reserve_requirement_ratio`, `macro_china_retail_price_index`, `macro_china_rmb`, `macro_china_shibor_all`, `macro_china_shrzgm`, `macro_china_society_electricity`, `macro_china_society_traffic_volume`, `macro_china_stock_market_cap`, `macro_china_supply_of_money`, `macro_china_swap_rate`, `macro_china_trade_balance`, `macro_china_urban_unemployment`, `macro_china_vegetable_basket`, `macro_china_wbck`, `macro_china_whxd`, `macro_china_xfzxx`, `macro_china_yw_electronic_index`

## 证据摘要

status=ok 表示有实际非空响应，字段非空数见 pit_fields；status=unavailable 见 attempts.error。payload_sha256 为本次规范化响应校验和，只保留工程证据，不收录新闻正文或财务原文。

<a id="evidence-daily_em"></a>
### ak.stock_zh_a_hist

| 样本 | 状态 | 返回行数 | 返回最早 / 最晚 | 传输尝试 |
|---|---|---|---|---|
| 600519 | unavailable | 0 | — / — | proxy=FAIL 5.437s; direct=FAIL 0.422s |
| 600036 | unavailable | 0 | — / — | proxy=FAIL 3.343s; direct=FAIL 0.407s |
| 601318 | unavailable | 0 | — / — | proxy=FAIL 3.937s; direct=FAIL 0.407s |
| 600030 | unavailable | 0 | — / — | proxy=FAIL 5.281s; direct=FAIL 0.313s |
| 600276 | unavailable | 0 | — / — | proxy=FAIL 7.235s; direct=FAIL 0.484s |
| 000001 | unavailable | 0 | — / — | proxy=FAIL 3.531s; direct=FAIL 0.407s |
| 000858 | unavailable | 0 | — / — | proxy=FAIL 3.359s; direct=FAIL 0.391s |
| 000333 | unavailable | 0 | — / — | proxy=FAIL 8.312s; direct=FAIL 0.453s |
| 300750 | unavailable | 0 | — / — | proxy=FAIL 3.953s; direct=FAIL 0.406s |
| 688981 | unavailable | 0 | — / — | proxy=FAIL 3.782s; direct=FAIL 0.468s |

<a id="evidence-daily_tx"></a>
### ak.stock_zh_a_hist_tx

| 样本 | 状态 | 返回行数 | 返回最早 / 最晚 | 传输尝试 |
|---|---|---|---|---|
| 600519 | ok | 5998 | 2001-08-27T00:00:00 / 2026-09-04T00:00:00 | proxy=ok 25.578s |
| 600036 | ok | 5855 | 2002-04-09T00:00:00 / 2026-09-04T00:00:00 | proxy=ok 26.297s |
| 601318 | ok | 4680 | 2007-03-01T00:00:00 / 2026-09-04T00:00:00 | proxy=ok 22.235s |
| 600030 | ok | 5649 | 2003-01-06T00:00:00 / 2026-09-04T00:00:00 | proxy=ok 23.797s |
| 600276 | ok | 6233 | 2000-10-18T00:00:00 / 2026-09-04T00:00:00 | proxy=ok 25.156s |
| 000001 | ok | 8484 | 1991-04-03T00:00:00 / 2026-09-04T00:00:00 | proxy=ok 35.688s |
| 000858 | ok | 6742 | 1998-04-27T00:00:00 / 2026-09-04T00:00:00 | proxy=ok 27.563s |
| 000333 | ok | 3089 | 2013-09-18T00:00:00 / 2026-09-04T00:00:00 | proxy=ok 13.812s |
| 300750 | ok | 2001 | 2018-06-11T00:00:00 / 2026-09-04T00:00:00 | proxy=ok 9.578s |
| 688981 | ok | 1485 | 2020-07-16T00:00:00 / 2026-09-04T00:00:00 | proxy=ok 7.016s |

<a id="evidence-minute_em"></a>
### ak.stock_zh_a_hist_min_em

| 样本 | 状态 | 返回行数 | 返回最早 / 最晚 | 传输尝试 |
|---|---|---|---|---|
| 600519 | unavailable | 0 | — / — | proxy=FAIL 4.64s; direct=FAIL 0.485s |
| 600036 | unavailable | 0 | — / — | proxy=FAIL 3.328s; direct=FAIL 0.422s |
| 601318 | unavailable | 0 | — / — | proxy=FAIL 3.641s; direct=FAIL 0.406s |
| 600030 | unavailable | 0 | — / — | proxy=FAIL 3.89s; direct=FAIL 0.36s |
| 600276 | unavailable | 0 | — / — | proxy=FAIL 5.312s; direct=FAIL 0.375s |
| 000001 | unavailable | 0 | — / — | proxy=FAIL 3.359s; direct=FAIL 0.375s |
| 000858 | unavailable | 0 | — / — | proxy=FAIL 4.359s; direct=FAIL 0.359s |
| 000333 | unavailable | 0 | — / — | proxy=FAIL 3.781s; direct=FAIL 0.406s |
| 300750 | unavailable | 0 | — / — | proxy=FAIL 4.969s; direct=FAIL 0.406s |
| 688981 | unavailable | 0 | — / — | proxy=FAIL 3.437s; direct=FAIL 0.406s |

<a id="evidence-quote_tx"></a>
### qt.gtimg.cn/q=…

| 样本 | 状态 | 返回行数 | 返回最早 / 最晚 | 传输尝试 |
|---|---|---|---|---|
| 600519 | ok | 1 | 2026-09-04T16:14:33 / 2026-09-04T16:14:33 | proxy=ok 1.546s |
| 600036 | ok | 1 | 2026-09-04T16:14:53 / 2026-09-04T16:14:53 | proxy=ok 0.968s |
| 601318 | ok | 1 | 2026-09-04T16:14:50 / 2026-09-04T16:14:50 | proxy=ok 1.281s |
| 600030 | ok | 1 | 2026-09-04T16:14:48 / 2026-09-04T16:14:48 | proxy=ok 0.984s |
| 600276 | ok | 1 | 2026-09-04T16:14:50 / 2026-09-04T16:14:50 | proxy=ok 0.812s |
| 000001 | ok | 1 | 2026-09-04T16:15:00 / 2026-09-04T16:15:00 | proxy=ok 0.828s |
| 000858 | ok | 1 | 2026-09-04T16:14:57 / 2026-09-04T16:14:57 | proxy=ok 0.968s |
| 000333 | ok | 1 | 2026-09-04T16:14:36 / 2026-09-04T16:14:36 | proxy=ok 0.812s |
| 300750 | ok | 1 | 2026-09-04T16:14:45 / 2026-09-04T16:14:45 | proxy=ok 0.781s |
| 688981 | ok | 1 | 2026-09-04T16:14:58 / 2026-09-04T16:14:58 | proxy=ok 0.922s |

<a id="evidence-daily_bs"></a>
### bs.query_history_k_data_plus

| 样本 | 状态 | 返回行数 | 返回最早 / 最晚 | 传输尝试 |
|---|---|---|---|---|
| 600519 | ok | 6072 | 2001-08-27T00:00:00 / 2026-09-04T00:00:00 | tcp_direct=ok 14.578s |
| 600036 | ok | 5929 | 2002-04-09T00:00:00 / 2026-09-04T00:00:00 | tcp_direct=ok 11.109s |
| 601318 | ok | 2000 | 2007-03-01T00:00:00 / 2015-05-20T00:00:00 | tcp_direct=ok 26.25s |
| 600030 | ok | 5747 | 2003-01-06T00:00:00 / 2026-09-04T00:00:00 | tcp_direct=ok 8.469s |
| 600276 | ok | 6279 | 2000-10-18T00:00:00 / 2026-09-04T00:00:00 | tcp_direct=ok 24.734s |
| 000001 | ok | 8647 | 1991-04-03T00:00:00 / 2026-09-04T00:00:00 | tcp_direct=ok 19.484s |
| 000858 | ok | 4000 | 1998-04-27T00:00:00 / 2014-11-03T00:00:00 | tcp_direct=ok 34.406s |
| 000333 | ok | 3151 | 2013-09-18T00:00:00 / 2026-09-04T00:00:00 | tcp_direct=ok 7.875s |
| 300750 | ok | 2001 | 2018-06-11T00:00:00 / 2026-09-04T00:00:00 | tcp_direct=ok 3.468s |
| 688981 | ok | 1491 | 2020-07-16T00:00:00 / 2026-09-04T00:00:00 | tcp_direct=ok 4.172s |

<a id="evidence-profile_em"></a>
### ak.stock_individual_info_em

| 样本 | 状态 | 返回行数 | 返回最早 / 最晚 | 传输尝试 |
|---|---|---|---|---|
| 600519 | unavailable | 0 | — / — | proxy=FAIL 3.969s; direct=FAIL 0.469s |
| 600036 | unavailable | 0 | — / — | proxy=FAIL 1.812s; direct=FAIL 0.391s |
| 601318 | unavailable | 0 | — / — | proxy=FAIL 1.703s; direct=FAIL 0.406s |
| 600030 | unavailable | 0 | — / — | proxy=FAIL 2.719s; direct=FAIL 0.406s |
| 600276 | unavailable | 0 | — / — | proxy=FAIL 2.109s; direct=FAIL 0.407s |
| 000001 | unavailable | 0 | — / — | proxy=FAIL 1.89s; direct=FAIL 0.407s |
| 000858 | unavailable | 0 | — / — | proxy=FAIL 3.734s; direct=FAIL 0.406s |
| 000333 | unavailable | 0 | — / — | proxy=FAIL 1.906s; direct=FAIL 0.484s |
| 300750 | unavailable | 0 | — / — | proxy=FAIL 2.0s; direct=FAIL 0.407s |
| 688981 | unavailable | 0 | — / — | proxy=FAIL 1.797s; direct=FAIL 0.406s |

<a id="evidence-holder"></a>
### RPT_HOLDERNUM_DET

| 样本 | 状态 | 返回行数 | 返回最早 / 最晚 | 传输尝试 |
|---|---|---|---|---|
| 600519 | ok | 63 | 2013-03-22T00:00:00 / 2026-06-30T00:00:00 | proxy=ok 2.047s |
| 600036 | ok | 51 | 2013-03-22T00:00:00 / 2026-06-30T00:00:00 | proxy=ok 4.953s |
| 601318 | ok | 68 | 2013-03-08T00:00:00 / 2026-06-30T00:00:00 | proxy=ok 1.516s |
| 600030 | ok | 67 | 2013-03-21T00:00:00 / 2026-06-30T00:00:00 | proxy=ok 4.781s |
| 600276 | ok | 62 | 2013-03-26T00:00:00 / 2026-06-30T00:00:00 | proxy=ok 4.735s |
| 000001 | ok | 68 | 2013-03-01T00:00:00 / 2026-06-30T00:00:00 | proxy=ok 1.125s |
| 000858 | ok | 85 | 2013-03-26T00:00:00 / 2026-08-10T00:00:00 | proxy=ok 0.937s |
| 000333 | ok | 60 | 2013-09-18T00:00:00 / 2026-02-28T00:00:00 | proxy=ok 5.188s |
| 300750 | ok | 35 | 2018-06-11T00:00:00 / 2026-06-30T00:00:00 | proxy=ok 1.156s |
| 688981 | ok | 28 | 2020-07-16T00:00:00 / 2026-06-30T00:00:00 | proxy=ok 1.406s |

<a id="evidence-financial"></a>
### RPT_LICO_FN_CPD

| 样本 | 状态 | 返回行数 | 返回最早 / 最晚 | 传输尝试 |
|---|---|---|---|---|
| 600519 | ok | 103 | 1998-12-31T00:00:00 / 2026-06-30T00:00:00 | proxy=ok 2.438s |
| 600036 | ok | 102 | 1999-12-31T00:00:00 / 2026-06-30T00:00:00 | proxy=FAIL 14.156s; direct=ok 0.453s |
| 601318 | ok | 84 | 2003-12-31T00:00:00 / 2026-06-30T00:00:00 | proxy=ok 1.671s |
| 600030 | ok | 100 | 1999-12-31T00:00:00 / 2026-06-30T00:00:00 | proxy=ok 2.609s |
| 600276 | ok | 105 | 1997-12-31T00:00:00 / 2026-06-30T00:00:00 | proxy=ok 2.437s |
| 000001 | ok | 122 | 1989-12-31T00:00:00 / 2026-06-30T00:00:00 | proxy=ok 2.282s |
| 000858 | ok | 111 | 1995-12-31T00:00:00 / 2026-06-30T00:00:00 | proxy=ok 1.375s |
| 000333 | ok | 79 | 2004-12-31T00:00:00 / 2026-06-30T00:00:00 | proxy=ok 4.985s |
| 300750 | ok | 41 | 2014-12-31T00:00:00 / 2026-06-30T00:00:00 | proxy=ok 0.953s |
| 688981 | ok | 40 | 2012-12-31T00:00:00 / 2026-06-30T00:00:00 | proxy=ok 1.406s |

<a id="evidence-balance"></a>
### RPT_DMSK_FN_BALANCE

| 样本 | 状态 | 返回行数 | 返回最早 / 最晚 | 传输尝试 |
|---|---|---|---|---|
| 600519 | ok | 103 | 1998-12-31T00:00:00 / 2026-06-30T00:00:00 | proxy=ok 2.344s |
| 600036 | ok | 101 | 1999-12-31T00:00:00 / 2026-06-30T00:00:00 | proxy=ok 1.109s |
| 601318 | ok | 83 | 2003-12-31T00:00:00 / 2026-06-30T00:00:00 | proxy=ok 1.234s |
| 600030 | ok | 99 | 1999-12-31T00:00:00 / 2026-06-30T00:00:00 | proxy=ok 1.688s |
| 600276 | ok | 105 | 1997-12-31T00:00:00 / 2026-06-30T00:00:00 | proxy=ok 2.234s |
| 000001 | ok | 119 | 1989-12-31T00:00:00 / 2026-06-30T00:00:00 | proxy=FAIL 6.437s; direct=ok 0.547s |
| 000858 | ok | 111 | 1995-12-31T00:00:00 / 2026-06-30T00:00:00 | proxy=ok 2.547s |
| 000333 | ok | 79 | 2004-12-31T00:00:00 / 2026-06-30T00:00:00 | proxy=ok 1.453s |
| 300750 | ok | 39 | 2014-12-31T00:00:00 / 2026-06-30T00:00:00 | proxy=ok 2.0s |
| 688981 | ok | 40 | 2011-12-31T00:00:00 / 2026-06-30T00:00:00 | proxy=ok 1.5s |

<a id="evidence-income"></a>
### RPT_DMSK_FN_INCOME

| 样本 | 状态 | 返回行数 | 返回最早 / 最晚 | 传输尝试 |
|---|---|---|---|---|
| 600519 | ok | 103 | 1998-12-31T00:00:00 / 2026-06-30T00:00:00 | proxy=ok 1.781s |
| 600036 | ok | 102 | 1999-12-31T00:00:00 / 2026-06-30T00:00:00 | proxy=ok 1.906s |
| 601318 | ok | 84 | 2003-12-31T00:00:00 / 2026-06-30T00:00:00 | proxy=ok 1.203s |
| 600030 | ok | 100 | 1999-12-31T00:00:00 / 2026-06-30T00:00:00 | proxy=ok 1.688s |
| 600276 | ok | 105 | 1997-12-31T00:00:00 / 2026-06-30T00:00:00 | proxy=ok 1.391s |
| 000001 | ok | 122 | 1989-12-31T00:00:00 / 2026-06-30T00:00:00 | proxy=ok 1.219s |
| 000858 | ok | 111 | 1995-12-31T00:00:00 / 2026-06-30T00:00:00 | proxy=ok 2.468s |
| 000333 | ok | 79 | 2004-12-31T00:00:00 / 2026-06-30T00:00:00 | proxy=ok 1.218s |
| 300750 | ok | 41 | 2014-12-31T00:00:00 / 2026-06-30T00:00:00 | proxy=ok 1.359s |
| 688981 | ok | 40 | 2012-12-31T00:00:00 / 2026-06-30T00:00:00 | proxy=ok 1.687s |

<a id="evidence-cashflow"></a>
### RPT_DMSK_FN_CASHFLOW

| 样本 | 状态 | 返回行数 | 返回最早 / 最晚 | 传输尝试 |
|---|---|---|---|---|
| 600519 | ok | 99 | 2000-12-31T00:00:00 / 2026-06-30T00:00:00 | proxy=ok 2.469s |
| 600036 | ok | 98 | 2000-12-31T00:00:00 / 2026-06-30T00:00:00 | proxy=ok 5.937s |
| 601318 | ok | 84 | 2003-12-31T00:00:00 / 2026-06-30T00:00:00 | proxy=ok 1.328s |
| 600030 | ok | 99 | 1999-12-31T00:00:00 / 2026-06-30T00:00:00 | proxy=ok 2.641s |
| 600276 | ok | 101 | 1999-12-31T00:00:00 / 2026-06-30T00:00:00 | proxy=ok 1.469s |
| 000001 | ok | 104 | 1998-06-30T00:00:00 / 2026-06-30T00:00:00 | proxy=ok 1.172s |
| 000858 | ok | 103 | 1998-12-31T00:00:00 / 2026-06-30T00:00:00 | proxy=ok 1.391s |
| 000333 | ok | 79 | 2004-12-31T00:00:00 / 2026-06-30T00:00:00 | proxy=ok 2.515s |
| 300750 | ok | 41 | 2014-12-31T00:00:00 / 2026-06-30T00:00:00 | proxy=ok 2.016s |
| 688981 | ok | 40 | 2012-12-31T00:00:00 / 2026-06-30T00:00:00 | proxy=ok 2.343s |

<a id="evidence-profit_bs"></a>
### bs.query_profit_data

| 样本 | 状态 | 返回行数 | 返回最早 / 最晚 | 传输尝试 |
|---|---|---|---|---|
| 600519 | ok | 1 | 2007-03-31T00:00:00 / 2007-03-31T00:00:00 | tcp_direct=ok 0.796s |
| 600036 | ok | 1 | 2007-03-31T00:00:00 / 2007-03-31T00:00:00 | tcp_direct=ok 0.297s |
| 601318 | ok | 1 | 2007-03-31T00:00:00 / 2007-03-31T00:00:00 | tcp_direct=ok 0.266s |
| 600030 | ok | 1 | 2007-03-31T00:00:00 / 2007-03-31T00:00:00 | tcp_direct=ok 0.219s |
| 600276 | ok | 1 | 2007-03-31T00:00:00 / 2007-03-31T00:00:00 | tcp_direct=ok 0.25s |
| 000001 | ok | 1 | 2007-03-31T00:00:00 / 2007-03-31T00:00:00 | tcp_direct=ok 0.562s |
| 000858 | ok | 1 | 2007-03-31T00:00:00 / 2007-03-31T00:00:00 | tcp_direct=ok 0.25s |
| 000333 | ok | 1 | 2013-09-30T00:00:00 / 2013-09-30T00:00:00 | tcp_direct=ok 3.062s |
| 300750 | ok | 1 | 2018-03-31T00:00:00 / 2018-03-31T00:00:00 | tcp_direct=ok 13.14s |
| 688981 | ok | 1 | 2020-03-31T00:00:00 / 2020-03-31T00:00:00 | tcp_direct=ok 8.969s |

<a id="evidence-flow_em"></a>
### ak.stock_individual_fund_flow

| 样本 | 状态 | 返回行数 | 返回最早 / 最晚 | 传输尝试 |
|---|---|---|---|---|
| 600519 | unavailable | 0 | — / — | proxy=FAIL 4.172s; direct=FAIL 0.328s |
| 600036 | unavailable | 0 | — / — | proxy=FAIL 4.328s; direct=FAIL 0.406s |
| 601318 | unavailable | 0 | — / — | proxy=FAIL 4.156s; direct=FAIL 0.344s |
| 600030 | unavailable | 0 | — / — | proxy=FAIL 4.047s; direct=FAIL 0.375s |
| 600276 | unavailable | 0 | — / — | proxy=FAIL 3.547s; direct=FAIL 0.344s |
| 000001 | unavailable | 0 | — / — | proxy=FAIL 3.609s; direct=FAIL 0.406s |
| 000858 | unavailable | 0 | — / — | proxy=FAIL 7.047s; direct=FAIL 0.469s |
| 000333 | unavailable | 0 | — / — | proxy=FAIL 3.266s; direct=FAIL 0.375s |
| 300750 | unavailable | 0 | — / — | proxy=FAIL 3.375s; direct=FAIL 0.453s |
| 688981 | ok | 120 | 2026-03-16T00:00:00 / 2026-09-04T00:00:00 | proxy=ok 1.515s |

<a id="evidence-north"></a>
### RPT_MUTUAL_HOLDSTOCKNDATE_STA

| 样本 | 状态 | 返回行数 | 返回最早 / 最晚 | 传输尝试 |
|---|---|---|---|---|
| 600519 | ok | 1000 | 2017-03-16T00:00:00 / 2021-07-09T00:00:00 | proxy=ok 7.703s |
| 600036 | ok | 1000 | 2017-03-16T00:00:00 / 2021-07-09T00:00:00 | proxy=ok 8.265s |
| 601318 | ok | 1000 | 2017-03-16T00:00:00 / 2021-07-09T00:00:00 | proxy=FAIL 6.422s; direct=ok 0.984s |
| 600030 | ok | 1000 | 2017-03-16T00:00:00 / 2021-07-09T00:00:00 | proxy=ok 4.516s |
| 600276 | ok | 1000 | 2017-03-16T00:00:00 / 2021-07-09T00:00:00 | proxy=ok 4.969s |
| 000001 | ok | 1000 | 2017-03-16T00:00:00 / 2021-07-09T00:00:00 | proxy=ok 3.844s |
| 000858 | ok | 1000 | 2017-03-16T00:00:00 / 2021-07-09T00:00:00 | proxy=ok 3.032s |
| 000333 | ok | 1000 | 2017-03-16T00:00:00 / 2021-08-09T00:00:00 | proxy=ok 5.359s |
| 300750 | ok | 1000 | 2019-01-02T00:00:00 / 2023-04-26T00:00:00 | proxy=ok 4.453s |
| 688981 | ok | 281 | 2023-01-12T00:00:00 / 2024-08-16T00:00:00 | proxy=ok 0.875s |

<a id="evidence-appointment"></a>
### RPT_PUBLIC_BS_APPOIN

| 样本 | 状态 | 返回行数 | 返回最早 / 最晚 | 传输尝试 |
|---|---|---|---|---|
| 600519 | ok | 79 | 2006-12-31T00:00:00 / 2026-06-30T00:00:00 | proxy=ok 1.781s |
| 600036 | ok | 79 | 2006-12-31T00:00:00 / 2026-06-30T00:00:00 | proxy=ok 0.86s |
| 601318 | ok | 79 | 2006-12-31T00:00:00 / 2026-06-30T00:00:00 | proxy=ok 0.844s |
| 600030 | ok | 79 | 2006-12-31T00:00:00 / 2026-06-30T00:00:00 | proxy=ok 1.078s |
| 600276 | ok | 79 | 2006-12-31T00:00:00 / 2026-06-30T00:00:00 | proxy=ok 1.625s |
| 000001 | ok | 79 | 2006-12-31T00:00:00 / 2026-06-30T00:00:00 | proxy=ok 1.953s |
| 000858 | ok | 79 | 2006-12-31T00:00:00 / 2026-06-30T00:00:00 | proxy=ok 3.375s |
| 000333 | ok | 52 | 2013-09-30T00:00:00 / 2026-06-30T00:00:00 | proxy=ok 2.062s |
| 300750 | ok | 33 | 2018-06-30T00:00:00 / 2026-06-30T00:00:00 | proxy=ok 4.703s |
| 688981 | ok | 25 | 2020-06-30T00:00:00 / 2026-06-30T00:00:00 | proxy=ok 1.032s |

<a id="evidence-forecast"></a>
### RPT_PUBLIC_OP_NEWPREDICT

| 样本 | 状态 | 返回行数 | 返回最早 / 最晚 | 传输尝试 |
|---|---|---|---|---|
| 600519 | ok | 33 | 2003-12-31T00:00:00 / 2024-12-31T00:00:00 | proxy=ok 1.641s |
| 600036 | ok | 12 | 2004-06-30T00:00:00 / 2010-09-30T00:00:00 | proxy=ok 3.64s |
| 601318 | ok | 7 | 2006-12-31T00:00:00 / 2015-06-30T00:00:00 | proxy=ok 1.906s |
| 600030 | ok | 17 | 2003-12-31T00:00:00 / 2026-06-30T00:00:00 | proxy=ok 1.063s |
| 600276 | ok | 3 | 2007-09-30T00:00:00 / 2009-12-31T00:00:00 | proxy=ok 0.875s |
| 000001 | ok | 17 | 2006-06-30T00:00:00 / 2015-12-31T00:00:00 | proxy=ok 0.766s |
| 000858 | ok | 21 | 2006-12-31T00:00:00 / 2026-06-30T00:00:00 | proxy=ok 0.813s |
| 000333 | ok | 11 | 2013-12-31T00:00:00 / 2018-12-31T00:00:00 | proxy=FAIL 6.625s; direct=ok 0.344s |
| 300750 | ok | 26 | 2018-06-30T00:00:00 / 2024-12-31T00:00:00 | proxy=ok 1.313s |
| 688981 | ok | 1 | 2022-12-31T00:00:00 / 2022-12-31T00:00:00 | proxy=ok 0.765s |

<a id="evidence-lhb"></a>
### RPT_DAILYBILLBOARD_DETAILSNEW

| 样本 | 状态 | 返回行数 | 返回最早 / 最晚 | 传输尝试 |
|---|---|---|---|---|
| 600519 | ok | 2 | 2007-10-24T00:00:00 / 2013-01-28T00:00:00 | proxy=ok 1.453s |
| 600036 | ok | 3 | 2008-10-24T00:00:00 / 2015-08-27T00:00:00 | proxy=ok 0.953s |
| 601318 | ok | 7 | 2007-03-01T00:00:00 / 2014-02-26T00:00:00 | proxy=ok 1.047s |
| 600030 | ok | 11 | 2007-09-04T00:00:00 / 2015-01-20T00:00:00 | proxy=ok 8.969s |
| 600276 | ok | 8 | 2006-11-09T00:00:00 / 2023-07-31T00:00:00 | proxy=ok 0.765s |
| 000001 | ok | 29 | 2006-11-21T00:00:00 / 2024-02-21T00:00:00 | proxy=ok 1.328s |
| 000858 | ok | 8 | 2007-04-17T00:00:00 / 2026-01-29T00:00:00 | proxy=ok 1.437s |
| 000333 | ok | 5 | 2013-09-18T00:00:00 / 2021-02-18T00:00:00 | proxy=ok 1.078s |
| 300750 | ok | 17 | 2018-06-11T00:00:00 / 2020-07-07T00:00:00 | proxy=ok 2.313s |
| 688981 | ok | 4 | 2024-10-09T00:00:00 / 2026-05-25T00:00:00 | proxy=ok 1.922s |

<a id="evidence-blocktrade"></a>
### RPT_BLOCKTRADE_STA

| 样本 | 状态 | 返回行数 | 返回最早 / 最晚 | 传输尝试 |
|---|---|---|---|---|
| 600519 | ok | 1000 | 2005-12-21T00:00:00 / 2023-08-09T00:00:00 | proxy=ok 8.094s |
| 600036 | ok | 569 | 2003-04-10T00:00:00 / 2026-08-17T00:00:00 | proxy=ok 2.297s |
| 601318 | ok | 746 | 2008-03-04T00:00:00 / 2026-09-02T00:00:00 | proxy=ok 3.985s |
| 600030 | ok | 310 | 2007-07-18T00:00:00 / 2026-09-02T00:00:00 | proxy=ok 1.437s |
| 600276 | ok | 274 | 2008-11-26T00:00:00 / 2026-07-31T00:00:00 | proxy=ok 2.828s |
| 000001 | ok | 227 | 2008-09-26T00:00:00 / 2026-07-09T00:00:00 | proxy=FAIL 10.359s; direct=ok 0.469s |
| 000858 | ok | 376 | 2007-10-23T00:00:00 / 2026-07-13T00:00:00 | proxy=ok 4.859s |
| 000333 | ok | 703 | 2014-04-08T00:00:00 / 2026-08-31T00:00:00 | proxy=FAIL 6.578s; direct=ok 0.844s |
| 300750 | ok | 607 | 2018-08-08T00:00:00 / 2026-09-03T00:00:00 | proxy=ok 5.094s |
| 688981 | ok | 187 | 2021-04-07T00:00:00 / 2026-07-13T00:00:00 | proxy=ok 1.672s |

<a id="evidence-hotrank"></a>
### ak.stock_hot_rank_detail_em

| 样本 | 状态 | 返回行数 | 返回最早 / 最晚 | 传输尝试 |
|---|---|---|---|---|
| 600519 | ok | 366 | 2025-09-06T00:00:00 / 2026-09-06T00:00:00 | proxy=ok 4.328s |
| 600036 | ok | 366 | 2025-09-06T00:00:00 / 2026-09-06T00:00:00 | proxy=ok 2.594s |
| 601318 | ok | 366 | 2025-09-06T00:00:00 / 2026-09-06T00:00:00 | proxy=ok 4.453s |
| 600030 | ok | 366 | 2025-09-06T00:00:00 / 2026-09-06T00:00:00 | proxy=ok 2.406s |
| 600276 | ok | 366 | 2025-09-06T00:00:00 / 2026-09-06T00:00:00 | proxy=ok 3.235s |
| 000001 | ok | 366 | 2025-09-06T00:00:00 / 2026-09-06T00:00:00 | proxy=ok 2.813s |
| 000858 | ok | 366 | 2025-09-06T00:00:00 / 2026-09-06T00:00:00 | proxy=ok 2.64s |
| 000333 | ok | 366 | 2025-09-06T00:00:00 / 2026-09-06T00:00:00 | proxy=ok 3.843s |
| 300750 | ok | 366 | 2025-09-06T00:00:00 / 2026-09-06T00:00:00 | proxy=ok 3.0s |
| 688981 | ok | 366 | 2025-09-06T00:00:00 / 2026-09-06T00:00:00 | proxy=ok 3.844s |

<a id="evidence-comment"></a>
### RPT_DMSK_TS_STOCKNEW

| 样本 | 状态 | 返回行数 | 返回最早 / 最晚 | 传输尝试 |
|---|---|---|---|---|
| 600519 | ok | 1 | 2026-09-04T00:00:00 / 2026-09-04T00:00:00 | proxy=ok 2.297s |
| 600036 | ok | 1 | 2026-09-04T00:00:00 / 2026-09-04T00:00:00 | proxy=ok 1.781s |
| 601318 | ok | 1 | 2026-09-04T00:00:00 / 2026-09-04T00:00:00 | proxy=ok 0.843s |
| 600030 | ok | 1 | 2026-09-04T00:00:00 / 2026-09-04T00:00:00 | proxy=ok 0.984s |
| 600276 | ok | 1 | 2026-09-04T00:00:00 / 2026-09-04T00:00:00 | proxy=ok 4.172s |
| 000001 | ok | 1 | 2026-09-04T00:00:00 / 2026-09-04T00:00:00 | proxy=ok 1.172s |
| 000858 | ok | 1 | 2026-09-04T00:00:00 / 2026-09-04T00:00:00 | proxy=ok 0.782s |
| 000333 | ok | 1 | 2026-09-04T00:00:00 / 2026-09-04T00:00:00 | proxy=ok 1.156s |
| 300750 | ok | 1 | 2026-09-04T00:00:00 / 2026-09-04T00:00:00 | proxy=ok 1.078s |
| 688981 | ok | 1 | 2026-09-04T00:00:00 / 2026-09-04T00:00:00 | proxy=ok 1.172s |

<a id="evidence-news"></a>
### ak.stock_news_em

| 样本 | 状态 | 返回行数 | 返回最早 / 最晚 | 传输尝试 |
|---|---|---|---|---|
| 600519 | unavailable | 0 | — / — | proxy=FAIL 2.266s; direct=FAIL 0.125s |
| 600036 | unavailable | 0 | — / — | proxy=FAIL 1.453s; direct=FAIL 0.11s |
| 601318 | unavailable | 0 | — / — | proxy=FAIL 1.391s; direct=FAIL 0.125s |
| 600030 | unavailable | 0 | — / — | proxy=FAIL 1.203s; direct=FAIL 0.094s |
| 600276 | unavailable | 0 | — / — | proxy=FAIL 1.672s; direct=FAIL 0.172s |
| 000001 | unavailable | 0 | — / — | proxy=FAIL 1.281s; direct=FAIL 0.11s |
| 000858 | unavailable | 0 | — / — | proxy=FAIL 2.36s; direct=FAIL 0.187s |
| 000333 | unavailable | 0 | — / — | proxy=FAIL 1.359s; direct=FAIL 0.157s |
| 300750 | unavailable | 0 | — / — | proxy=FAIL 1.172s; direct=FAIL 0.125s |
| 688981 | unavailable | 0 | — / — | proxy=FAIL 1.328s; direct=FAIL 0.125s |

<a id="evidence-gdp"></a>
### RPT_ECONOMY_GDP

| 样本 | 状态 | 返回行数 | 返回最早 / 最晚 | 传输尝试 |
|---|---|---|---|---|
| macro | ok | 82 | 2006-03-01T00:00:00 / 2026-06-01T00:00:00 | proxy=ok 1.734s |

<a id="evidence-cpi"></a>
### RPT_ECONOMY_CPI

| 样本 | 状态 | 返回行数 | 返回最早 / 最晚 | 传输尝试 |
|---|---|---|---|---|
| macro | ok | 223 | 2008-01-01T00:00:00 / 2026-07-01T00:00:00 | proxy=ok 2.594s |

<a id="evidence-pmi"></a>
### RPT_ECONOMY_PMI

| 样本 | 状态 | 返回行数 | 返回最早 / 最晚 | 传输尝试 |
|---|---|---|---|---|
| macro | ok | 224 | 2008-01-01T00:00:00 / 2026-08-01T00:00:00 | proxy=ok 1.703s |

<a id="evidence-money"></a>
### RPT_ECONOMY_CURRENCY_SUPPLY

| 样本 | 状态 | 返回行数 | 返回最早 / 最晚 | 传输尝试 |
|---|---|---|---|---|
| macro | ok | 223 | 2008-01-01T00:00:00 / 2026-07-01T00:00:00 | proxy=ok 1.516s |

接口语义参考（历史与覆盖数字均来自上面的实测）：[AKShare 股票文档](https://akshare.akfamily.xyz/data/stock/stock.html)、[AKShare 宏观文档](https://akshare.akfamily.xyz/data/macro/macro.html)、[Baostock 官方文档](http://www.baostock.com/baostock/index.php/Python_API文档)。同底层映射依据安装包对应函数源码中的 reportName。
