# 前瞻纸面验证闭环 (#108): 把「回测验证过的 edge」送进未来真实行情

> 阶段2 交付。回答 roadmap §3.1「前瞻验证闭环(最该补, 最容易)」: 冷落 beta 回测 11/11,
> 但**从未被未来真实行情验证过** —— 回测是 in-sample, 纸面前瞻才是 out-of-sample。
> 本 harness = 「任何验证过的 edge 都能自动纸面跟踪」的机器, 第一个试点 = 冷落 beta。

## 一句话结论

**回测和「跑赢」之间缺的那一步补上了。** 现在冷落 beta 有了一个**冻结入场 + 预注册判据**的前瞻 bet,
未来每个交易日由 `scripts/forward_track.py` 用真实行情跟踪它 vs 匹配 universe vs 上证综指,
一旦滚动偏离跌破预注册阈值即标记 `edge_failing`。判据**今天 (2026-08-15) 写死, 之后只读** —— 满足
「禁止事后调参追赢」的铁律。

## 为什么回测不够 (问题)

冷落 beta 的「跑赢上证 11/11」是**用过去 11 年数据回测出来的** ([[cold-tilt-all-regimes]])。
回测有两处固有盲区:

1. **in-sample**: 参数 (K=100 / UNIVERSE_N=800 / 再平衡周期) 是在**同一批历史数据**上选的,
   对未来无保证。冷落 beta 回测「持有 vs 匹配 universe 8/11」, 但从未被明天开始的行情检验过。
2. **幸存者偏差**: 回测用当前 replay parquet 的股票池, 退市股缺失 ([[cold-tilt-all-regimes]] 已诚实标注)。

前瞻验证补的正是这块: **今天冻结名单, 明天用真实行情看它是否兑现**。

## 设计

| 组件 | 落点 | 职责 |
|------|------|------|
| 核心 harness | `analysis/forward_validation.py` | 等权篮子收益 (停牌 ffill)、交易日计数、预注册判据应用、注册表读写 (9 单测) |
| 注册 | `scripts/forward_register_cold_tilt.py` | 冻结入场快照 + 写死判据 |
| 跟踪 | `scripts/forward_track.py` | 未来每交易日算前瞻收益 + 滚动偏离 + 失效标记 |
| 注册表 | `simulation_data/forward_validation/registry.json` | 判据已注册的证据 (随仓保存) |

### 预注册判据 (2026-08-15 写死, 之后只读)

| 字段 | 值 | 含义 |
|------|-----|------|
| horizon | 60 交易日 (~3 个月) | 主判据在此落定 |
| primary_benchmark | matched universe (top-800 等权) | 诚实基准 = 自家可投池子 (不是错配的上证) |
| primary_success | 篮子净收益 ≥ universe 收益 | 选股 alpha 非负 (回测「持有 8/11」的 claims 在前瞻窗兑现) |
| secondary_success | 篮子净收益 ≥ 上证收益 | 「跑赢上证」名义目标 (回测 11/11) |
| failure_threshold | −10pp | 篮子累计落后 universe ≥10pp → `edge_failing` (参考回测最差 −14.1pp, 只标「明显坏掉」) |

判据的**成功/失败布尔断言是确定性的** (净收益正负), 无 LLM、无事后调节空间。

### 口径 (与回测严格对齐)

- **篮子**: bottom-100 低换手等权, 从 top-800 流动性 universe (非ST/可交易/pe>0/pb>0) 选, **冻结入场名单
  持有** (「持有」= 回测里 vs universe 最强的 8/11 口径)。
- **成本**: 31bp 一次性全额往返上界 (佣金万3+印花0.05%卖+滑点10bp), 只扣篮子, 与
  `run_cold_tilt_rebalance.py`「持有」口径一致 ([[transaction-costs-unified]])。
- **停牌/退市**: ffill (价格冻结贡献 0 收益), 与 `matched_universe_curve` 一致。

## 今天注册的 bet (真实数据)

- **edge_id**: `cold_tilt_bottom100_hold`
- **入场日**: 2026-08-14 (最近交易日, 腾讯实时快照)
- **篮子**: 100 只 (换手中位数 0.64%, 范围 0.06%~1.17%) — 以 mega-cap 银行/石油为主 (工商/农业/中国银行、
  中国石油、中国人寿), 印证「冷落 = 低换手大票」的防御型 beta 本质。
- **匹配 universe**: 800 只 (top-800 by 成交额)。
- **上证综指**: 3927.18。
- 全部 100 篮子 + 800 universe 代码在实时快照中可解析 (100/100, 800/800), 跟踪链路无缺票。

## 用法

```bash
python scripts/refresh_market_cache.py          # 每日先刷新快照到最近交易日
python scripts/forward_register_cold_tilt.py    # (一次性) 注册/重注册 bet
python scripts/forward_track.py                 # 每日跟踪: 算前瞻收益 + 失效标记
```

跟踪幂等: 快照日 ≤ 入场日 (无前瞻新数据) 或已跟踪过则跳过, 不追加空快照。

## 纪律检查

- [x] 判据**先注册后运行**, `criterion` 字段跟踪时只读 (`append_tracking` 只改 `tracking` 数组, 单测断言)。
- [x] 全程真实数据 (腾讯实时快照 + qt.gtimg.cn 上证综指), 指数抓取失败即报错, 不兜底不模拟。
- [x] 单测 `tests/unit/test_forward_validation.py` 9 个全过; 全套 `pytest tests/unit/` 547 过。
- [x] ruff E/F/W 全过 (含新文件)。
- [x] 未调参追赢: K=100/UNIVERSE_N=800/31bp/持有 与回测同口径, 判据一次性写死。

## 诚实边界 (预先声明, 避免事后找补)

1. **价格收益口径** (`total_return: false`): full_market_cache 无分红。基准纪律 #107 已证分红在篮子 vs
   universe 之间大致抵消 (价格/全收益两口径结论同为 8/11), 故价格收益是公平近似; 若将来要精确, 需逐日
   抓 Baostock 分红加回。
2. **这是 beta 不是 alpha**: 冷落 beta 跑赢上证的大头是小盘 universe beta + 冷落溢价 ([[benchmark_discipline]])。
   它**不是「跑赢牛市」的答案** —— 回测里 3 个牛市窗口它跑输自家池子 (2019 −14.1pp / 2021 −6.8pp /
   2025-26 −2.9pp)。前瞻判据的 primary_success 只在「选股 alpha 非负」上设门槛, 不承诺牛市跑赢。
3. **持有口径**: 冻结名单不换仓; 月频/季频再平衡的「实际操作」尚未前瞻验证 (回测里持有已是 vs universe 最强口径)。
4. **失效标记是滞后信号**: −10pp 阈值是「明显坏掉」才报警, 不阻止小偏离; 它喂给 #109 的失效监控, 而非自动止损。

## 对「跑赢市场/跑赢牛市」目标的含义

前瞻闭环补上了「验证可信」的最后一块: 从此每个宣称跑赢的 edge 都要先回测、再**前瞻冻结**, 用未来行情
作证。冷落 beta 是第一个试点 —— 若 60 交易日后 primary_success=True (选股 alpha 非负), 则「持有 vs
universe 8/11」的 claims 首次得到 out-of-sample 支撑; 若 False, 则诚实记录该 edge 在前瞻窗失效。
无论结果如何, 都是「禁止事后调参」纪律下的可复现证据, 而非又一轮 in-sample 自我满足。

---

## 失效监控部署 (#109)

前瞻闭环的计算层已在 #108 落地 (滚动偏离 `selection_alpha_pp` + `edge_failing` 阈值), #109 把它**接进
每日自动化**, 让 edge 在真实未来行情上被持续监控、失效即报警。

| 组件 | 落点 | 状态 |
|------|------|------|
| 每日监控批处理 | `scripts/run_forward_track.bat` (刷新快照 → forward_track, ASCII-only 避免 cmd 编码错位) | ✅ 端到端验证过 (5107/5107 只 75s, exit 0) |
| 任务计划注册 | `scripts/register_forward_track.ps1` (每天 15:20, 与 daily_runner 隔离) | ✅ 已注册 (AITraderForwardTrack, Daily 15:20) |
| 失效标记 | `forward_track.py` 末尾输出可 grep 的 `[EDGE_FAILING]` 行 | ✅ |
| 日志 | `simulation_data/forward_validation.log` | ✅ |

**注册命令** (需用户手动执行或授权; 注册是一次性系统持久化, 不在无授权下自动执行):

```powershell
powershell -ExecutionPolicy Bypass -File scripts\register_forward_track.ps1
```

- 触发: 每交易日 15:20 (收盘后、daily_runner 15:05 之后), 用当天收盘价跟踪。
- 与 daily_runner 隔离 (阶段4「实验探针不混进核心交易路径」); 腾讯行情免代理, bat 内不设 HTTP_PROXY。
- 停用: `powershell -ExecutionPolicy Bypass -File scripts\register_forward_track.ps1 -Unregister`。
- 首次真实跟踪将发生在下一个交易日 (周一), 之后每日追加 `tracking` 快照并判断 `edge_failing`。

