# exp-007 Phase 1 实验结果与分析

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

## 二、与预期目标的差距分析

| 指标 | 实验计划目标 | 实际结果 | 差距 |
|------|-------------|---------|------|
| Recall@10 | 0.50 ~ 0.55 | 0.4539 | -0.05 ~ -0.10 |
| Recall@50 | 0.72 ~ 0.78 | 0.7003 | -0.02 ~ -0.08 |

**实际 Recall@10=0.4539，低于预期下限 0.50。**

可能原因分析：

1. **训练数据量不足**：744K 训练对仅来自二元 qrels（label>0），未使用 T2Ranking 官方的 1.6M 训练对（包含 graded label）。官方 Recall@50=0.67 使用 batch=128、epoch=20 训练，我们的 batch=64、epoch=3 在训练数据量和训练步数上均更少。

2. **仅使用 in-batch negatives**：MNRL 的负样本量受 batch_size=64 限制，每个 query 仅 63 个负样本。CachedMNRL（Phase 3.1）将负样本量放大 5-10 倍，预期可进一步提升 1-2%。

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

如果 Phase 1 + 2 的 Recall@10 仍低于 0.50，启动 Phase 3：
- 3.1: CachedMNRL（负样本量放大 5-10 倍）→ 预期 +1~2%
- 3.2: BM25 + Dense Dynamic Hard Negative Mining → 预期 +1~3%
- 3.3: Graded Label (CosineSimilarityLoss + MNRL multi-task) → 预期 +1~2%

---

## 四、文件索引

| 文件 | 说明 |
|------|------|
| `scripts/exp007/prepare_training_data.py` | 训练数据准备（TSV → JSONL） |
| `scripts/exp007/train_embedding_phase1.py` | Phase 1 训练脚本 |
| `scripts/exp007/evaluate_embedding.py` | 检索效果评估脚本 |
| `scripts/build_t2ranking_index.py` | 向量索引构建脚本（已适配微调模型） |
| `models/m3e-base-t2ranking-phase1/` | Phase 1 微调后的模型权重 |
| `data/vector_db/t2ranking/m3e-base/` | Pretrained M3E 向量索引 |
| `data/vector_db/t2ranking/m3e-base-t2ranking-phase1/` | Fine-tuned M3E 向量索引 |

---

*实验日期: 2026-05-28 | 评估脚本: `scripts/exp007/evaluate_embedding.py`*
