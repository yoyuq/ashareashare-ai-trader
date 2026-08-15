# 多模型对抗 — 预注册 A/B 协议

> 状态: **脚手架已落地, 默认关闭**。本文档是"多模型对抗是否真有用"的**预注册判据**——在跑 A/B **之前**先写下通过/不通过标准, 防止事后追赢。

## 背景

进化系统的对抗票 (`agent/evolution/adversarial.py`) 已用**独立二次 LLM 调用**打破"同一次 completion 里 role-play 假对抗"。但对抗票与主导诊断官仍是**同一个模型** (`deepseek-v4-flash`)。

- 根因未除尽: 对抗审查官与主导共享同一套模型权重/偏好, 冗余度实测 0.28~0.36, 大量时候主导与对抗趋同。
- 多模型对抗 = 让对抗票用**第二模型**(另一个权重/另一个厂), 看它能否 catch 主导模型看不见的盲区。

## 开关 (默认关, 不改变现状)

| 环境变量 | 默认 | 说明 |
|---------|------|------|
| `ADVERSARIAL_LLM_MODEL` | 未设 | 设了才启用多模型对抗 |
| `ADVERSARIAL_LLM_BASE_URL` | `https://api.deepseek.com/v1` | OpenAI 兼容 base |
| `ADVERSARIAL_LLM_API_KEY` | `DEEPSEEK_API_KEY` | 第二模型的 key |

```bash
# 对抗票用 pro, 主导仍 flash (只需现有 DEEPSEEK_API_KEY)
ADVERSARIAL_LLM_MODEL=deepseek-v4-pro python scripts/historical_replay.py --adv-mode independent ...

# 对抗票用第二家模型 (需 ZHIPU_API_KEY)
ADVERSARIAL_LLM_MODEL=glm-4-flash \
ADVERSARIAL_LLM_BASE_URL=https://open.bigmodel.cn/api/paas/v4 \
ADVERSARIAL_LLM_API_KEY=$ZHIPU_API_KEY \
python scripts/historical_replay.py --adv-mode independent ...
```

未设 `ADVERSARIAL_LLM_MODEL` 时, 行为与现状完全一致 (对抗票与主导同模型)。启动后日志出现 `[对抗票] 多模型对抗启用: 独立模型 <model>` 即确认开关生效——**跑 A/B 前必须先 grep 到这一行**, 否则静默落回同模型会污染结果。

## 预注册判据 (跑之前先定)

A/B 对比:**对抗票=第二模型** vs **对抗票=主导同模型 (基线)**, 其余完全一致 (`--adv-mode independent` 双端)。

| # | 判据 | 通过标准 |
|---|------|---------|
| **1. 主判据(降险)** | 空头/牛转崩段配对收益差 | 均值 **> 0** 且 **≥4/5 seeds 为正** (多模型对抗在降险段不劣于基线并改善) |
| **2. 零假设(判模型依赖)** | 空头段配对差分 t 检验 | 若 **p>0.05 (不显著)** → 判"效应依赖模型能力", **不默认启用** |
| **3. 护栏(防降敏噪声)** | 2019 趋势牛段 | 不得**显著跑输** (避免把"牛市过度谨慎"噪声误当价值) |

## 方法论 (复用既有 matched-pair A/B)

与 [[evolution-multiwindow-validation]] / [[evolution-noise-baseline]] 同一套:

- 每臂 **5 次种子** (随机顺序抖动), 取**配对差分** (同种子两侧收益相减), 不做跨种子均值混比较。
- 窗口沿用 4 窗口 (2019 牛 / 2020 牛转崩 / 2021 / 2023) + 全量/CSI300 双口径。
- 结论按段拆 (空头段 vs 牛市段), 不看单一全窗口净收益。

## 现状依据 (为什么"预注册"而非直接默认)

- 同模型对抗的**真实价值已定位在空头/牛转崩段降险** (2020 配对净效应 +2.02pp, 5/5 全正)。
- 换第二模型复验进化净效应时, GLM **+0.85pp 未显著** → 效应**依赖模型能力**, 不是"换个模型必然更好"。
- 故多模型对抗**可能有用也可能没用**, 必须用上面的判据先注册、再判定, 不做"默认开"的拍脑袋。

## 与已落地改动的关系

本脚手架是**纯增量、默认关**, 不影响:
- v5.7 统一记账 (实盘对抗票仍默认走 ModelRouter 分账)。
- 回放换模型复验 (`REPLAY_LLM_*` 仍控制诊断/复盘/进化总结的整条链路)。
- 两者可组合: `REPLAY_LLM_MODEL` 管主导, `ADVERSARIAL_LLM_MODEL` 管对抗票, 优先级为对抗票独立开关 > 回放 client > 实盘 router。
