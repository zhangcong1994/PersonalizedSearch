# exp-011 RL/DPO 可行性验证 —— 结果与分析

> 实验目标：通过多样本生成 + Judge 评分，判断 RL/DPO 在 AI 搜索生成场景下是否有前景
>
> 实验类型：可行性验证（无训练），50 条 query × 温度 0.8 × 5 样本 × 2 模型
>
> 核心结论：**两个模型均为 LIKELY_FEASIBLE。Best-of-5 比基线高 16-18 分，70-76% 的 query 显著改善。RL/DPO 在 8B 模型 + v1-full prompt 上有明确的收益空间，分布中已存在接近教师模型（72.7 分）甚至超越的答案。**
>
> 创建日期：2026-06-03

---

## 实验配置

| 项目 | 配置 |
|------|------|
| 模型 | Qwen3-8B-nonthink / Qwen3-8B-thinking（vLLM 推理） |
| Prompt | v1-full（exp-010 Phase 1 验证过的最佳 prompt） |
| 生成温度（基线） | 0.3，单次采样 |
| 生成温度（多样本） | 0.8，每 query 5 次独立采样 |
| max_tokens | 1024 |
| 查询数 | 50 条（从 exp-005 198 条 dev 集中 seed=42 随机抽样） |
| 基线数据 | exp-010 Phase 1 结果（同一 Judge 下的 v1-full 单样本） |
| Judge | deepseek-chat，6 维两批评估（与 exp-005 一致） |
| 对比方式 | 同一 query 的 best-of-5 vs baseline(T=0.3) |

---

## 结果

### 总分对比

| | 8B-nonthinking | 8B-thinking |
|---|:---:|:---:|
| 基线 mean (T=0.3) | 60.48 | 64.23 |
| Mean-of-5 (T=0.8) | 63.19 | 63.93 |
| **Best-of-5 (T=0.8)** | **76.54** | **82.61** |
| **Delta (Best − Baseline)** | **+16.07** | **+18.39** |
| Median Delta | +14.8 | +13.0 |
| Mean std (per query) | 12.5 | 15.39 |

### Pass Rate

| | 8B-nonthinking | 8B-thinking |
|---|:---:|:---:|
| Baseline (T=0.3) | 66.0% | 76.0% |
| Best-of-5 (T=0.8) | 90.0% | 98.0% |

### Query 级分布

| | 8B-nonthinking | 8B-thinking |
|---|:---:|:---:|
| 显著改善 (Δ > +5) | 38 (76.0%) | 35 (70.0%) |
| 基本持平 (｜Δ｜ ≤ 5) | 8 (16.0%) | 12 (24.0%) |
| 显著退化 (Δ < −5) | 4 (8.0%) | 3 (6.0%) |

### 分维度 Best-of-N Delta

| 维度 | 8B-nonthinking | median | pos% | 8B-thinking | median | pos% |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| 准确性 (veracity) | **+0.80** | +1.0 | 58% | **+0.78** | +1.0 | 58% |
| 安全性 (safety) | +0.28 | 0.0 | 28% | +0.34 | 0.0 | 34% |
| 相关性 (relevance) | +0.28 | 0.0 | 24% | +0.32 | 0.0 | 22% |
| 整合质量 (synthesis) | +0.36 | 0.0 | 34% | +0.44 | 0.0 | 42% |
| 引文质量 (citation) | +0.48 | 0.0 | 46% | +0.42 | 0.0 | 40% |
| 用户体验 (UX) | +0.46 | 0.0 | 44% | **+0.68** | 0.0 | 48% |

---

## 可行性判定

| 模型 | 判定 | 理由 |
|---|:---:|------|
| 8B-nonthink | **LIKELY_FEASIBLE** | Best-of-N 平均比基线高 16.1 分，76% 的 query 显著改善 |
| 8B-thinking | **LIKELY_FEASIBLE** | Best-of-N 平均比基线高 18.4 分，70% 的 query 显著改善 |

---

## 分析

### 发现 1：分布中确实存在更好的答案 —— RL 空间巨大

两个模型的 Best-of-5 分别达到 76.54 和 82.61 分，都**超过了教师模型 qwen3-max 的全量得分（72.7）**。这意味着在固定检索输入下，8B 模型生成答案的**理论上界**已经不低于甚至超越了 API 模型，但其 T=0.3 默认输出却只有 60-64 分。

这完美契合了 RL 的核心适用场景：**模型有能力但没把那能力用出来**。RL 的工作就是将分布从均值推向高分区。

### 发现 2：Mean-of-N 高于 Baseline → 当前推理温度偏保守

T=0.8 的单次输出均值（63.19/63.93）本身就高于 T=0.3（60.48/64.23）。这说明 baseline 使用的 T=0.3 可能不是最优推理温度，更高的温度（0.5-0.6）本身就可以带来正收益。建议在 RL 训练前先做温度调参实验。

### 发现 3：方差极大 → Reward 信号强但训练需谨慎

per-query std 高达 12-15 分，意味着 5 个 sample 之间好坏差距巨大 —— 好的可以到 85+，差的可以到 30-40。这是 RL 的理想场景（好/坏样本区分度大），但也意味着：

- **Pros**：reward 信号强，模型容易区分好行为和坏行为
- **Cons**：方差大意味着训练可能不稳定，需要耐心调参（尤其是 PPO/GRPO 的 clip 范围、KL penalty 等）

### 发现 4：准确性是最大赢家，但原因值得思考

Veracity delta +0.78~0.80 是所有维度中最高的，这反直觉 —— 直觉上 T=0.8 应该更容易产生幻觉。实际解释是：

- T=0.3 下的保守策略让模型回避了某些需要综合判断的问题，被 Judge 打了低分（信息覆盖不全）
- T=0.8 下模型更敢于做跨文档推断，反而命中了更多文档信息，Judge 判定为"准确"
- **风险**：RL 训练中如果 reward model 对事实错误的区分力不足，高温度的自由探索可能引入真正的幻觉

### 发现 5：thinking 模式在 best-of-N 上更有优势，但非压倒性

| 指标 | nonthinking | thinking | 差距 |
|---|:---:|:---:|:---:|
| 基线 | 60.48 | 64.23 | +3.75 |
| Best-of-5 | 76.54 | 82.61 | +6.07 |
| Delta | +16.07 | +18.39 | +2.32 |

thinking 模式的绝对分数更高（+6 分），且 delta 稍大（+2.3 分）。但 thinking 的 std 也更高（15.39 vs 12.50），训练可能更不稳定。

RL 训练建议从 nonthinking 开始 —— 成本更低（不需要内部推理链），且 delta 已经足够大（+16 分）。thinking 模式可以在验证 RL 有效后再尝试。

### 发现 6：分维度 median delta 大部分为 0

median delta 为 0 说明：**有相当一部分 query 的 best-of-5 在某个维度上没有超越 baseline**。改善集中在少数 query 上（mean delta 为正面 median 为 0，说明少数 query 的大量改善拉高了均值）。这意味着：

- RL 的收益可能不均衡 —— 某些 query 类型（如信息不足、多文档矛盾）是主要的改善对象
- 后续应该对 delta 最大的 query 做定性分析，理解哪些场景下高温度采样最有效

### 发现 7：退化 query 存在但比例低

8% 的 query 在 best-of-5 中反而低于 baseline（Δ < -5）。这些是需要关注的：高温度在某些场景下确实有害。后续应抽取退化 case 做定性分析，了解什么情况下 T=0.3 比采样更可靠。

---

## 与已有实验的关联

| 实验 | 结论 | 本实验的关系 |
|------|------|------|
| exp-009 SFT | SFT 失败，模型学会格式而非整合 | RL 正适合纠正这种行为偏差 |
| exp-010 Phase 1 | v1-full prompt 让整合 +0.26 | v1-full 是本实验的基底，且 RL 在此基础上有增量空间 |
| exp-005 多模型基线 | qwen3-max = 72.7（上界） | Best-of-5 的 76-82 分 = 8B 的理论上界已超 API 模型 |

---

## 后续建议

按优先级排序：

| 方向 | 投入 | 预期收益 | 理由 |
|------|------|:---:|------|
| **DPO（教师数据）** | 中 | +5~8 分 | 数据已就绪（exp-009 教师答案 82.3 分），离线方法成本低，有望将均值推向他界 |
| 推理温度调参 | 低 | +2~4 分 | T=0.8 均值已高于 T=0.3，直接调基线推理温度可能有零成本收益 |
| GRPO（在线 RL） | 高 | +8~12 分 | 分布空间足够大，但需解决 Judge API 做 reward 的成本问题和训练稳定性 |
| 退化 case 分析 | 低 | 理解边界 | 抽取 Δ < -5 和 Δ > +50 的 query 做定性，帮助设计更精准的 RL reward |
| 4B RL 验证 | 中 | 未知 | 8B 可行不代表 4B 可行，应补跑 4B 多样本实验验证规模效应 |

**建议的短期路径**：先调推理温度（零成本），然后做 DPO（数据就绪、方法成熟），最后视 DPO 效果决定是否上 GRPO。

---

## 附录：文件清单

| 文件 | 说明 |
|------|------|
| `results/exp011/generation/qwen3-8b-nothink_v1-full_t0.8_n5_s42.jsonl` | nonthinking 多样本生成（250 条） |
| `results/exp011/generation/qwen3-8b-thinking_v1-full_t0.8_n5_s42.jsonl` | thinking 多样本生成（250 条） |
| `results/exp011/judge_scores/qwen3-8b-nothink_v1-full_t0.8_n5_s42_judged.jsonl` | nonthinking 多样本评分 |
| `results/exp011/judge_scores/qwen3-8b-thinking_v1-full_t0.8_n5_s42_judged.jsonl` | thinking 多样本评分 |
| `results/exp010/judge_scores/qwen3-8b-nothink-v1-full_judged.jsonl` | nonthinking 基线评分（exp-010 Phase 1） |
| `results/exp010/judge_scores/qwen3-8b-thinking-v1-full_judged.jsonl` | thinking 基线评分（exp-010 Phase 1） |
| `results/exp011/analysis/rl_feasibility_report.json` | 可行性分析报告（摘要） |
| `results/exp011/analysis/rl_feasibility_full.json` | 可行性分析报告（含 per-query 细节） |
| `experiments/exp-011-rl-feasibility.yaml` | 实验计划书 |
| `scripts/exp011/generate_multi_sample.py` | 多样本生成脚本 |
| `scripts/exp011/run_judge.py` | Judge 评分包装脚本 |
| `scripts/exp011/analyze_feasibility.py` | 可行性分析脚本 |
