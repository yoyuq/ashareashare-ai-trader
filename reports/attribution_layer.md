# Alpha/Beta 因子归因层 (阶段1 #106)

**日期**: 2026-08-15
**状态**: 已完成
**代码**: `analysis/attribution.py` + `tests/unit/test_attribution.py` (11 测试全绿)

## 为什么需要这一层

Roadmap 阶段1 核心缺口「**Alpha 和 beta 没分解**」。项目反复出现「把 beta 当 alpha」的误判:

- 上证综指 (sh.000001) 是**大盘市值加权**指数。任何小盘/低换手/等权 tilt 天然带小盘 beta,
  跑赢上证的量里混着「小盘溢价 (系统性 beta)」和「选股技能 (真 alpha)」。
- 单因子模型 `r_p = α + β·r_mkt + ε` 只认市场 beta, 小盘 beta 会漏进 α 残差 → 被误读成「选股 alpha」。

本层补两套分解, 让每个 A/B 自动报「多少是 beta、多少是 alpha」。

## 方法 (全程零模拟, 只用真实日收益回归)

### 单因子 (Jensen 分解)

```
r_p,t = α + β·r_b,t + ε          β = Cov(r_p, r_b)/Var(r_b),  α = mean(r_p) − β·mean(r_b)
总收益恒等式:  strategy_return = β × benchmark_return + alpha_contribution
```

### 双因子 (市场 + 小盘 SMB)

```
r_p = α + β_mkt·r_mkt + β_smb·(r_ew − r_mkt) + ε      SMB = 等权 universe − 市值加权市场
总收益恒等式:  strategy_return = β_mkt×市场收益 + β_smb×SMB收益 + alpha_contribution
```

- `r_mkt` = 上证综指 (市值加权), `r_ew` = 策略所属 universe 的**等权**日收益。
- SMB 因子把小盘溢价单独拆出; 剩下的 α 才是「扣掉市场+小盘 beta 后」的**真实选股 alpha**。
- OLS 含截距 (设计矩阵加常数列), 截距 α 不再被斜率吸收 (已修 bug)。

## 结果

### 1. 冷落/低换手 tilt 的「跑赢」到底是不是小盘 beta? —— 不是, 4/5 窗口有真 alpha

`run_improved_portfolio_ab.py` (低换手 bottom-100 vs 上证, 双因子归因):

| 窗口 | Δret vs 上证 | β_mkt | β_smb | 市场贡献 | 小盘贡献 | **真α贡献** |
|------|------|------|------|------|------|------|
| 2018熊 | +1.70pp | 0.88 | 0.55 | -22.5% | -4.3% | **+3.0%** |
| 2019牛 | +8.51pp | 1.01 | 0.46 | +23.9% | +10.6% | **-2.3%** |
| 2020牛转崩 | +24.80pp | 1.06 | 0.08 | +21.6% | +1.2% | **+22.3%** |
| 2024震荡 | +7.03pp | 1.01 | -0.14 | +13.3% | -0.6% | **+7.5%** |
| 2025-26现期 | +16.48pp | 0.97 | -0.10 | -2.5% | -1.9% | **+18.3%** |

**结论 (诚实)**: 冷落/低换手 tilt 跑赢上证**不是纯小盘 beta**。扣掉市场+小盘后, 4/5 窗口仍有显著正真α
(+3.0 / +22.3 / +7.5 / +18.3%)。唯一的负真α是 **2019牛 (-2.3%)**: 那年「跑赢 +8.51pp」里 +10.6pp
是小盘 beta, 选股本身反而拖累 -2.3% —— 与记忆 [[evolution-2019-niu-contrarian]]「牛市系统性弱点」完全一致。

### 2. 单因子会把小盘 beta 误报成 alpha —— 双因子纠偏的实锤

`run_fullcycle_ab.py` 等权小盘 vs 上证 (单因子):

| 窗口 | 等权Δvs上证 | 单因子「alpha贡献」 | 双因子真α (各 tilt) |
|------|------|------|------|
| 2019牛 | +23.22pp | **+16.5%** | low_turn 真α **-6.6%** |
| 2025-26现期 | +18.85pp | **+21.0%** | low_turn 真α **+14.9%** |

2019 牛那行是决定性证据: 单因子把等权跑赢上证 +23.22pp 里的 +16.5pp 报成「alpha」, 但双因子一看,
等权的超额几乎全是小盘 beta (β_smb 为正), 真正的低换手选股 tilt 真α反而是 **-6.6%**。这正是 roadmap
警告的「把 beta 当 alpha」, 单因子永远看不出来, 双因子一次揭穿。

### 3. 各因子 tilt 的诚实真α (扣掉市场+小盘后, `run_fullcycle_ab.py`)

| 因子 | 2018熊 | 2019牛 | 2020牛转崩 | 2024震荡 | 2025-26现期 |
|------|------|------|------|------|------|
| low_turn 低换手 | +5.1% | **-6.6%** | +20.1% | +19.8% | +14.9% |
| low_pe 低估值 | -5.3% | +3.7% | +10.9% | +24.4% | +18.1% |
| low_vol 低波 | — | +2.1% | +19.1% | +16.7% | — |

规律: 低换手/低波/低估值 tilt 的 β_smb 大多为**负** (它们不是小盘 beta, 反而逆小盘), 却仍有正真α →
「冷落溢价」是真实信号, 不靠小盘 beta 灌水。

## 落地

- `analysis/attribution.py`: `alpha_beta_attribution` (单因子) / `two_factor_attribution` (双因子) /
  `attribute_equity_curves` / `two_factor_attribute_equity_curves` (净值曲线包装) / `load_index_series`。
- 已接入 3 个 A/B 脚本的自动归因输出: `run_cold_tilt_rebalance.py` (单因子)、
  `run_improved_portfolio_ab.py` / `run_fullcycle_ab.py` (单+双因子)。
- `tests/unit/test_attribution.py`: 11 测试 (完美 beta/纯 alpha/日期对齐/数据不足/NaN/净值等价/
  超额分解/双因子还原/双因子净值包装等价/缺文件), 全绿。
- 数据不足 (<3 对齐点) 返回全 NaN 不抛异常 (软增强, 不阻塞 A/B)。

## 判读纪律

- **alpha_contribution ≈ 0** → 跑赢几乎全是 beta (小盘/高波动 tilt), 不是选股 alpha。
- **alpha_contribution 显著为正** → 有真实 alpha (扣掉 beta 后仍跑赢)。
- 报告「跑赢指数」时必须同时报 β_mkt/β_smb 与真α, 避免再把 beta 当 alpha。
