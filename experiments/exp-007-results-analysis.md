# exp-007 实验结果与分析

> 实验目标：在 T2Ranking 训练集上微调 M3E-base，验证对比学习（MNRL）对中文 passage 检索的召回提升
>
> 实验类型：全训练流程——数据准备 → 模型微调 → 向量索引构建 → 检索评估
>
> 核心指标：Recall@10（主指标）、Recall@50、MRR

---

## 实验配置

| 项目 | 配置 |
|------|------|
| 基座模型 | moka-ai/m3e-base（768 维，~110M 参数） |
| 训练框架 | sentence-transformers |
| 训练数据 | T2Ranking `qrels.retrieval.train.tsv`（744K query-positive 对，去重 + 过滤后） |
| Loss 函数 | MultipleNegativesRankingLoss（in-batch negatives） |
| Epochs | 3 |
| Learning Rate | 2e-5 |
| Warmup Ratio | 0.1 |
| Scheduler | warmupcosine |
| Optimizer | AdamW (weight_decay=0.01, max_grad_norm=1.0) |
| Batch Size | 64 |
| FP16 | 开启 |
| 训练硬件 | NVIDIA GeForce RTX 5090 32GB |
| Query 输入 | 原始 query（Phase 1 不包含改写/HyDE） |

---

## 数据集

| 项目 | 数值 |
|------|------|
| 训练集 | T2Ranking train 全量（258K query） |
| 评估集 | T2Ranking dev（约 24K query） |
| 评估 query 数 | 2000 条（前 2000 条有 qrels 的 query） |
| 检索库 | T2Ranking collection（全量 2.3M passages） |
| 检索方式 | ChromaDB 向量索引（cosine similarity，Top-50） |
| 评估 qrels | `qrels.retrieval.dev.tsv` + `qrels.dev.tsv`（graded NDCG） |

---

## 一、Phase 1 结果：M3E-base Pretrained vs Fine-tuned

### 指标总表

| Metric | Pretrained | Fine-tuned | Delta | 相对提升 |
|--------|-----------|------------|-------|---------|
| **Recall@10** | 0.3896 | **0.4539** | +0.0643 | **+16.5%** |
| Recall@20 | 0.5074 | 0.5784 | +0.0710 | +14.0% |
| Recall@50 | 0.6278 | 0.7003 | +0.0725 | +11.5% |
| MRR | 0.4556 | 0.4976 | +0.0421 | +9.2% |
| NDCG@10 | 0.3405 | 0.3903 | +0.0498 | +14.6% |
| NDCG@10_graded | 0.3566 | 0.4114 | +0.0548 | +15.4% |
| Hit@10 | 0.7350 | 0.7960 | +0.0610 | +8.3% |
| Hit@50 | 0.8420 | 0.8940 | +0.0520 | +6.2% |

### 结论

**1. 全指标正向提升，微调有效**

所有 8 个指标全部上涨，无一回退。三项核心指标的提升均具有实际意义：
- Recall@10: **0.3896 → 0.4539（+16.5%）**——近一半目标 query 在前 10 条就能找到答案
- Recall@50: **0.6278 → 0.7003（+11.5%）**——召回上限首次突破 70%
- MRR: **0.4556 → 0.4976（+9.2%）**——首个正确答案的平均排位从 ~2.2 提升到 ~2.0

**2. Recall 提升幅度随 K 递减，符合预期**

| K | 提升 |
|---|------|
| Recall@10 | +16.5% |
| Recall@20 | +14.0% |
| Recall@50 | +11.5% |

小 K 的提升更大，说明微调主要改善了「最相关 passage 的排位」——模型学会了把正确答案推到更前面，而不是仅仅提升相关性排序的整体质量。这恰好是 MRR 提升的直接原因。

**3. Graded NDCG 同步提升，说明微调不止是二元相关性的改善**

NDCG@10_graded 从 0.3566 → 0.4114（+15.4%），与 NDCG@10（binary）的 +14.6% 基本持平。说明微调后的模型不仅在「区分相关/不相关」上进步，在「区分高度相关/部分相关」上也有同步改善。这证明了 MNRL 的对比学习范式在二元标注下训练，也能隐式学到分级相关性。

---

## 二、Phase 3.1 结果：CachedMNRL 继续训练（⚠️ 退化）

### 实验配置

| 项目 | Phase 1 | Phase 3.1 |
|------|---------|-----------|
| 起点模型 | moka-ai/m3e-base（原始权重） | Phase 1 微调后模型 |
| Loss 函数 | MultipleNegativesRankingLoss | CachedMultipleNegativesRankingLoss |
| mini_batch_size | — | 32 |
| Epochs | 3 | 2 |
| 训练数据 | 744K query-positive 对 | 相同（同一份 JSONL） |
| 其他超参 | lr=2e-5, batch=64, warmup=0.1 | 相同 |

### 指标总表

| Metric | Phase 1 | Phase 3.1 | Delta | 相对变化 |
|--------|---------|-----------|-------|---------|
| **Recall@10** | **0.4539** | 0.4470 | -0.0069 | **-1.5%** |
| Recall@20 | 0.5784 | 0.5743 | -0.0041 | -0.7% |
| Recall@50 | 0.7003 | 0.6917 | -0.0086 | -1.2% |
| MRR | 0.4976 | 0.4897 | -0.0079 | -1.6% |
| NDCG@10 | 0.3903 | 0.3831 | -0.0072 | -1.8% |
| NDCG@10_graded | 0.4114 | 0.4036 | -0.0078 | -1.9% |
| Hit@10 | 0.7960 | 0.7915 | -0.0045 | -0.6% |
| Hit@50 | 0.8940 | 0.8900 | -0.0040 | -0.4% |

### 结论：全指标退化，CachedMNRL 无效

**Phase 3.1 所有 8 个指标均低于 Phase 1**，且退化幅度均匀（-0.004 ~ -0.009），这是一个清晰的过拟合信号。

**根因分析：**

1. **模型已收敛，同一份数据再训练无意义**：Phase 1 的 3 epochs × 744K 对已经让模型充分吸收了训练数据中的信号。Phase 3.1 用同一份 JSONL 继续训练 2 个 epoch，模型只是在不断降低已见过的 (query, positive) 对的 loss，验证集泛化反而变差。

2. **CachedMNRL 的收益条件不成立**：CachedMNRL 的核心优势是"更多负样本"——通过缓存最近几个 mini-batch 的 passage embedding，将负样本量从 batch_size 级别放大到 mini_batch_size × batch_size 级别。但对于一个已经收敛的模型，这些缓存的负样本早已被轻易区分，提供不了新的梯度信号。

3. **均匀退化 = 经典过拟合**：如果 CachedMNRL 只是引入了噪声（例如缓存的 stale embedding 导致梯度方向错误），通常会出现部分指标升、部分指标降。全指标均匀下降的特征强烈指向：模型在训练集上继续优化，在验证集上泛化能力变差。

**行动项：**
- ❌ Phase 3.1 废弃，Phase 1 模型仍是当前 best model
- ✅ Phase 3.2 从 Phase 1 模型开始，仅训 1~2 epoch，引入 hard negatives 作为唯一新信息源
- ✅ 后续如需重新引入 CachedMNRL，应在 Phase 2 引入新数据分布（改写 query / HyDE）时使用

---

## 三、与预期目标的差距分析

| 指标 | 实验计划目标 | 实际结果 | 差距 |
|------|-------------|---------|------|
| Recall@10 | 0.50 ~ 0.55 | 0.4539 | -0.05 ~ -0.10 |
| Recall@50 | 0.72 ~ 0.78 | 0.7003 | -0.02 ~ -0.08 |

**实际 Recall@10=0.4539，低于预期下限 0.50。**

可能原因分析：

1. **训练数据量不足**：744K 训练对仅来自二元 qrels（label>0），未使用 T2Ranking 官方的 1.6M 训练对（包含 graded label）。官方 Recall@50=0.67 使用 batch=128、epoch=20 训练，我们的 batch=64、epoch=3 在训练数据量和训练步数上均更少。

2. **仅使用 in-batch negatives**：MNRL 的负样本量受 batch_size=64 限制，每个 query 仅 63 个负样本。Phase 3.1 尝试了 CachedMNRL（负样本放大 5-10 倍），但因训练数据不变、模型已收敛而过拟合退化。

3. **未使用 instruction 前缀**：实验计划中提到了"为这个句子生成表示以用于检索相关文章："前缀，本次未加入。BGE/M3E 家族对 instruction 敏感，加入后可能提升 1-3%。

---

## 三、后续计划

### 短期优化（Phase 1 补充）

| 方向 | 预期收益 | 难度 |
|------|---------|------|
| 增加 instruction 前缀重新训练 | +1~3% Recall | 低 |
| 使用全量 graded qrels 训练数据 | +3~5% Recall | 中 |
| 增大 batch_size=128（显存允许的话） | +1~2% Recall | 低 |

### Phase 2：多输入分布对齐

目标：引入改写 query 和 HyDE 假答案共同训练，消除离线/在线分布偏移。
- 改写 query 从 24K dev query 中生成（DeepSeek API）
- HyDE 假答案从改写后的 query 生成（DeepSeek API）
- 预期改写 query 检索的 Recall 显著提升

### Phase 3：高级训练策略

#### Phase 3.1: ~~CachedMNRL~~ → **已废弃（过拟合退化 -1.5%）**

#### Phase 3.2: Dynamic Dense Hard Negative Mining + TripletLoss + LoRA

从 Phase 1 best model 起步（Recall@10=0.4539），每 epoch 用最新模型做全量 2.3M passages Dense 检索挖掘 hard negatives，配合 TripletLoss（margin 0.05/0.03） + MNRL 双 Loss，LoRA 防灾难性遗忘。

**训练配置：**

| 项目 | 配置 |
|------|------|
| 基座模型 | Phase 1 best model (models/m3e-base-t2ranking-phase1, Recall@10=0.4539) |
| Hard Negative 来源 | 每 epoch 用当前模型编码 2.3M passages → FAISS IndexFlatIP → 取 top-5 非正样本 |
| Loss | TripletLoss(COSINE_DISTANCE, margin) + MultipleNegativesRankingLoss |
| Epochs | 3（ep0 仅挖掘，ep1-3 训练） |
| Margin | ep1=0.05, ep2/3=0.03 |
| LoRA | r=16, alpha=32, target_modules=["query","key","value","dense"], ~1.2% 参数 |
| Learning Rate | 1e-5 |
| Batch Size | 32 |
| Warmup | 0.1 (warmupcosine) |
| FP16 | 开启 |
| 训练硬件 | NVIDIA GeForce RTX 5090 32GB |

<details>
<summary><b>📊 Phase 3.2 Epoch 1 结果（点击展开）</b></summary>

**对比 Pretrained (m3e-base)：**

| Metric | Pretrained | Phase 3.2 ep1 | Δ 绝对 | Δ 相对 |
|--------|-----------|---------------|--------|--------|
| **Recall@10** | 0.3896 | **0.5064** | +0.1168 | **+30.0%** |
| Recall@20 | 0.5074 | 0.6359 | +0.1285 | +25.3% |
| **Recall@50** | 0.6278 | **0.7452** | +0.1174 | **+18.7%** |
| **MRR** | 0.4556 | **0.5549** | +0.0993 | **+21.8%** |
| NDCG@10 | 0.3405 | 0.4396 | +0.0991 | +29.1% |
| Hit@10 | 0.7350 | 0.8435 | +0.1085 | +14.8% |
| Hit@50 | 0.8420 | 0.9280 | +0.0860 | +10.2% |
| NDCG@10_graded | 0.3566 | 0.4617 | +0.1051 | +29.5% |

**对比 Phase 1 (best model)：**

| 指标 | Phase 1 | Phase 3.2 ep1 | Δ |
|------|---------|---------------|-----|
| Recall@10 | 0.4539 | 0.5064 | **+11.6%** |
| Recall@50 | 0.7003 | 0.7452 | **+6.4%** |
| MRR | 0.4976 | 0.5549 | **+11.5%** |

</details>

<details>
<summary><b>📊 Phase 3.2 Epoch 2 结果（点击展开）</b></summary>

**对比 Pretrained (m3e-base)：**

| Metric | Pretrained | Phase 3.2 ep2 | Δ 绝对 | Δ 相对 |
|--------|-----------|---------------|--------|--------|
| **Recall@10** | 0.3896 | **0.4956** | +0.1060 | +27.2% |
| Recall@20 | 0.5074 | 0.6246 | +0.1172 | +23.1% |
| **Recall@50** | 0.6278 | **0.7363** | +0.1085 | +17.3% |
| **MRR** | 0.4556 | **0.5371** | +0.0815 | +17.9% |
| NDCG@10 | 0.3405 | 0.4279 | +0.0874 | +25.7% |
| Hit@10 | 0.7350 | 0.8355 | +0.1005 | +13.7% |
| Hit@50 | 0.8420 | 0.9240 | +0.0820 | +9.7% |
| NDCG@10_graded | 0.3566 | 0.4504 | +0.0938 | +26.3% |

</details>

<details>
<summary><b>📊 Phase 3.2 Epoch 3 结果（点击展开）</b></summary>

**对比 Pretrained (m3e-base)：**

| Metric | Pretrained | Phase 3.2 ep3 | Δ 绝对 | Δ 相对 |
|--------|-----------|---------------|--------|--------|
| **Recall@10** | 0.3896 | **0.5013** | +0.1117 | +28.7% |
| Recall@20 | 0.5074 | 0.6353 | +0.1279 | +25.2% |
| **Recall@50** | 0.6278 | **0.7406** | +0.1128 | +18.0% |
| **MRR** | 0.4556 | **0.5461** | +0.0905 | +19.9% |
| NDCG@10 | 0.3405 | 0.4340 | +0.0935 | +27.5% |
| Hit@10 | 0.7350 | 0.8405 | +0.1055 | +14.4% |
| Hit@50 | 0.8420 | 0.9255 | +0.0835 | +9.9% |
| NDCG@10_graded | 0.3566 | 0.4562 | +0.0996 | +27.9% |

</details>

**跨 Epoch 对比（核心指标）：**

| 指标 | Pretrained | Phase 1 | ep1 | ep2 | ep3 |
|------|-----------|---------|-----|-----|-----|
| **Recall@10** | 0.3896 | 0.4539 | **0.5064** ★ | 0.4956 | 0.5013 |
| **Recall@50** | 0.6278 | 0.7003 | **0.7452** ★ | 0.7363 | 0.7406 |
| **MRR** | 0.4556 | 0.4976 | **0.5549** ★ | 0.5371 | 0.5461 |

| 指标 | ep1→ep2 Δ | ep2→ep3 Δ | ep1→ep3 Δ |
|------|-----------|-----------|-----------|
| Recall@10 | -0.0108 (-2.1%) | +0.0057 (+1.1%) | -0.0051 (-1.0%) |
| Recall@50 | -0.0089 (-1.2%) | +0.0043 (+0.6%) | -0.0046 (-0.6%) |
| MRR | -0.0178 (-3.2%) | +0.0090 (+1.7%) | -0.0088 (-1.6%) |

> ★ = 当前最佳 checkpoint (ep1)

**关键发现：**
1. **ep1 仍然是不可撼动的最佳 checkpoint**——Recall@10=0.5064，唯一突破 0.50 的 epoch
2. **ep3 相比 ep2 微幅回升**（Recall@10 +0.0057, MRR +0.0090），说明 margin=0.03 的第三轮训练没有继续恶化——margin 0.05→0.03 的降低影响很小
3. 三 epoch 的 Recall@10 差异极窄（0.4956~0.5064，带宽仅 0.0108），说明模型在 ep1 后已达到收敛平台——后续 hard negative 重挖掘 + 重训练的边际收益几乎为零
4. **结论：1 轮 Dynamic Hard Negative Mining 足够**——多轮重挖掘不带来增益，ep1 即为 Phase 3.2 最终交付模型

#### Phase 3.3: Graded Label (CosineSimilarityLoss + MNRL multi-task) → 待评估

---

## 四、文件索引

| 文件 | 说明 |
|------|------|
| `scripts/exp007/prepare_training_data.py` | 训练数据准备（TSV → JSONL） |
| `scripts/exp007/train_embedding_phase1.py` | Phase 1 训练脚本 |
| `scripts/exp007/train_embedding_phase3_1.py` | Phase 3.1 训练脚本（CachedMNRL，已废弃） |
| `scripts/exp007/train_embedding_phase3_2.py` | Phase 3.2 训练脚本（Dynamic Hard Negatives + TripletLoss） |
| `scripts/exp007/build_faiss_index.py` | FAISS 向量索引构建脚本 |
| `scripts/exp007/evaluate_embedding.py` | 检索效果评估脚本 |
| `scripts/exp007/build_index.py` | ChromaDB 向量索引构建脚本（旧版） |
| `models/m3e-base-t2ranking-phase1/` | Phase 1 微调后的模型权重 |
| `models/m3e-base-t2ranking-phase3-1/` | Phase 3.1 微调后的模型权重（已过拟合，废弃） |
| `models/m3e-base-t2ranking-phase3-2/ep1/merged/` | Phase 3.2 Epoch 1 模型（LoRA merged） |
| `models/m3e-base-t2ranking-phase3-2/ep2/merged/` | Phase 3.2 Epoch 2 模型（LoRA merged） |
| `models/m3e-base-t2ranking-phase3-2/ep3/merged/` | Phase 3.2 Epoch 3 模型（LoRA merged） |
| `data/vector_db/t2ranking/m3e-base/` | Pretrained M3E 向量索引 |
| `data/vector_db/t2ranking/m3e-base-t2ranking-phase1/` | Fine-tuned M3E Phase 1 向量索引 |
| `data/vector_db/t2ranking/m3e-base-t2ranking-phase3-1/` | Fine-tuned M3E Phase 3.1 向量索引 |

---

*实验日期: 2026-05-28 ~ 2026-05-30 | 评估脚本: `scripts/exp007/evaluate_embedding.py`*
