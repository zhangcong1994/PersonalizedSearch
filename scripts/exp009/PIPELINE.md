# Exp-009 管道说明

每个阶段的脚本、参数、输入、输出一览。

状态标记: [x] 已完成 / [.] 待编写 / [ ] 未开始

---

## 路径约定

> **所有数据/模型路径均为 `{DATA_ROOT}/` 下的相对路径。**

`DATA_ROOT` 由 `src/utils/config.py` 读取环境变量 `PERSONALIZEDSEARCH_DATA_ROOT` 决定：

| 环境 | `PERSONALIZEDSEARCH_DATA_ROOT` | `DATA_ROOT` 实际值 | 说明 |
|------|:--:|------|------|
| 本机 Windows | 未设置 | 项目根目录（如 `E:\Users\czhang\trae_projects\PersonalizedSearch`） | 回退到项目根 |
| 服务器 Linux | 已设置（如 `/root/autodl-fs`） | 数据盘挂载点 | 避免占用系统盘 |

子目录由 `config.yaml` 定义：

```
{DATA_ROOT}/
├── data/
│   ├── raw/t2ranking/          ← RAW_DATA_DIR: 原始 TSV 数据 (queries/qrels/collection)
│   ├── processed/              ← 中间产物 JSONL（本文档中的 exp009_*.jsonl 均在此）
│   └── vector_db/t2ranking/    ← VECTOR_DB_DIR: FAISS 索引 + pids.json
└── models/                     ← 本地模型缓存 + 微调输出
```

**脚本行为**：所有 exp009 脚本通过 `from src.utils.config import DATA_ROOT` 获取根路径，相对路径参数自动 prepend DATA_ROOT 后再读写。

下文所有路径标注 `{DATA_ROOT}/` 前缀以提示两端的差异。

---

## 阶段零：检索难度统计（前置分析）

统计 T2Ranking train split 中每个 query 的 num_positive/max_grade/avg_grade 分布，为分层阈值提供数据支撑。

| 项目 | 内容 |
|------|------|
| 脚本 | [x] `scripts/exp009/sample_training_queries.py --mode stats` |
| 输入 | `{DATA_ROOT}/data/raw/t2ranking/queries.train.tsv` (258K queries) |
| | `{DATA_ROOT}/data/raw/t2ranking/qrels.retrieval.train.tsv` (744K pairs) |
| | `{DATA_ROOT}/data/raw/t2ranking/qrels.train.tsv` (graded labels) |
| 输出 | 终端：直方图 + 交叉表 + 建议阈值 |
| | 可选: `--output {DATA_ROOT}/data/processed/exp009_retrieval_difficulty.json` (全量明细) |

```powershell
python scripts/exp009/sample_training_queries.py --mode stats
```

**分层阈值（2026-05-31 数据驱动确认）**：

| 层 | 定义 | 自然占比 | 采样配比 |
|:--:|------|:--:|:--:|
| T1 富信息 | num_positive >= 3 | 47.8% | 35% (1750) |
| T2 中等 | num_positive = 1-2 | 29.9% | 25% (1250) |
| T3 贫信息 | num_positive = 0 | 22.3% | 40% (2000) |

`max_grade` 不纳入分层条件（num_positive > 0 时无区分度，全部 >= 2）。

---

## 阶段一：分层抽样

从 258K 训练集 query 中按 T1/T2/T3 分层无放回抽取 5000 条。

| 项目 | 内容 |
|------|------|
| 脚本 | [x] `scripts/exp009/sample_training_queries.py --mode sample` |
| 输入 | 同阶段零（queries.train.tsv + qrels.retrieval.train.tsv + qrels.train.tsv） |
| 输出 | `{DATA_ROOT}/data/processed/exp009_sampled_queries.jsonl` |

**输出格式**（每行一条）：
```json
{"qid": "50000", "query": "如何减肥最快", "stratum": "T1", "num_positive": 5, "max_grade": 3, "avg_grade": 2.8}
```

```powershell
python scripts/exp009/sample_training_queries.py --mode sample --total 5000 --seed 42
```

参数: `--total 5000` `--seed 42` `--output <path>`

---

## 阶段二：检索管道

> **编排方式**：Shell 脚本串联 6 个独立子步骤，每步检测中间文件自动跳过（断点续跑）。

| 平台 | 脚本 | 用法 |
|------|------|------|
| Windows | `scripts/exp009/run_retrieval_pipeline.ps1` | `.\scripts\exp009\run_retrieval_pipeline.ps1` |
| Linux | `scripts/exp009/run_retrieval_pipeline.sh` | `bash scripts/exp009/run_retrieval_pipeline.sh` |

公用参数: `-Device cuda` / `-SkipAugment` / `-SkipIndex` / `-RebuildIndex`

```
exp009_sampled_queries.jsonl (5000 queries)
    │
    ├──[Step 1: run_query_augment.py]──→ exp009_rewritten_queries.jsonl
    │                                    exp009_hyde_answers.jsonl
    │
    ├──[Step 2: build_faiss_index.py]──→ data/vector_db/t2ranking/{model}/  (FAISS IndexFlatIP)
    │
    ├──[Step 3: dense_retrieve.py]─────→ exp009_dense_B0.jsonl  (原查询 dense)
    │                                    exp009_dense_P2.jsonl  (改写 dense)
    │                                    exp009_dense_H2.jsonl  (HyDE dense)
    │
    ├──[Step 4: bm25_retrieve.py]──────→ exp009_bm25_B0.jsonl  (BM25)
    │
    ├──[Step 5: rrf_fuse.py]───────────→ exp009_rrf_fused.jsonl  (4路 RRF → top-50)
    │
    └──[Step 6: rerank.py]─────────────→ exp009_reranked_top10.jsonl  (bge-reranker-v2-m3 → top-10)
```

### Step 2.1: 查询增强（改写 + HyDE）

| 项目 | 内容 |
|------|------|
| 脚本 | [x] `scripts/exp009/run_query_augment.py` |
| 输入 | `{DATA_ROOT}/data/processed/exp009_sampled_queries.jsonl` |
| 输出 | `{DATA_ROOT}/data/processed/exp009_rewritten_queries.jsonl` |
| | `{DATA_ROOT}/data/processed/exp009_hyde_answers.jsonl` |
| 模型 | Qwen3-4B (本地 vLLM, OpenAI compatible) |
| 降级 | DeepSeek API（vLLM 部署失败时） |
| 费用 | ¥0（本地推理） / ~¥4（降级 API） |

**输出格式**：
```json
// rewritten: {"qid": "50000", "original": "如何减肥最快", "rewritten": "快速减肥方法 科学减重饮食运动计划"}
// hyde:     {"qid": "50000", "original": "如何减肥最快", "hyde": "减肥的核心是热量赤字..."}
```

```powershell
# 单独运行（shell 脚本会自动调）
python scripts/exp009/run_query_augment.py \
    --input data/processed/exp009_sampled_queries.jsonl \
    --output-rw data/processed/exp009_rewritten_queries.jsonl \
    --output-hy data/processed/exp009_hyde_answers.jsonl \
    --llm-url http://localhost:8000/v1 \
    --batch-size 32
```

参数: `--input` `--output-rw` `--output-hy` `--llm-url` `--batch-size 32` `--task {rewrite,hyde,both}`

### Step 2.2: FAISS 索引构建

| 项目 | 内容 |
|------|------|
| 脚本 | [.] 复用 `scripts/exp007/build_faiss_index.py` |
| 输入 | `{DATA_ROOT}/data/raw/t2ranking/collection.tsv` (2.3M passages) |
| 输出 | `{DATA_ROOT}/data/vector_db/t2ranking/{model_short}/index.faiss` + `pids.json` |
| 模型 | M3E-base exp-007 Phase 3.2 微调版 |
| 设备 | RTX 4090, ~10-15 分钟 |

```powershell
python scripts/exp007/build_faiss_index.py \
    --model models/m3e-base-t2ranking-phase3-2/ep1/merged \
    --device cuda \
    --offline
```

参数: `--model` `--device cuda` `--offline` `--rebuild` `--encode-batch-size 256`

### Step 2.3: Dense 检索（3 路）

| 项目 | 内容 |
|------|------|
| 脚本 | [x] `scripts/exp009/dense_retrieve.py` |
| 输入 | `{DATA_ROOT}/data/processed/exp009_sampled_queries.jsonl` + rewritten + hyde |
| | FAISS index (Step 2.2, `{DATA_ROOT}/data/vector_db/t2ranking/{model}/`) |
| 输出 | `{DATA_ROOT}/data/processed/exp009_dense_B0.jsonl` (原查询 → top-50) |
| | `{DATA_ROOT}/data/processed/exp009_dense_P2.jsonl` (改写 → top-50) |
| | `{DATA_ROOT}/data/processed/exp009_dense_H2.jsonl` (HyDE → top-50) |
| 方法 | FAISS IndexFlatIP, 内积相似度搜索 |

**输出格式**（每行一条 query 的完整检索结果）：
```json
{"qid": "50000", "query": "...", "results": [{"pid": "...", "text": "...", "score": 0.95, "rank": 1}, ...]}
```

```powershell
python scripts/exp009/dense_retrieve.py \
    --model models/m3e-base-t2ranking-phase3-2/ep1/merged \
    --device cuda \
    --input-queries data/processed/exp009_sampled_queries.jsonl \
    --input-rewritten data/processed/exp009_rewritten_queries.jsonl \
    --input-hyde data/processed/exp009_hyde_answers.jsonl \
    --output-b0 data/processed/exp009_dense_B0.jsonl \
    --output-p2 data/processed/exp009_dense_P2.jsonl \
    --output-h2 data/processed/exp009_dense_H2.jsonl \
    --top-k 50
```

参数: `--model` `--device` `--input-queries` `--input-rewritten` `--input-hyde` `--output-b0` `--output-p2` `--output-h2` `--top-k 50`

### Step 2.4: BM25 检索（1 路）

| 项目 | 内容 |
|------|------|
| 脚本 | [x] `scripts/exp009/bm25_retrieve.py` |
| 输入 | `{DATA_ROOT}/data/processed/exp009_sampled_queries.jsonl` + `{DATA_ROOT}/data/raw/t2ranking/collection.tsv` |
| 索引 | `{DATA_ROOT}/data/bm25s_index/t2ranking/` (exp003 构建的 BM25s 分片索引, 2.3M documents) |
| 输出 | `{DATA_ROOT}/data/processed/exp009_bm25_B0.jsonl` |
| 方法 | BM25 (robertson, k1=1.5, b=0.75) 词匹配 |
| 复用 | `src/retrieval/bm25s_store.ShardedBM25S` (分片加载) + `src/retrieval/bm25_store.tokenize_query` (jieba 分词+停用词) |

```powershell
python scripts/exp009/bm25_retrieve.py \
    --input-queries data/processed/exp009_sampled_queries.jsonl \
    --output data/processed/exp009_bm25_B0.jsonl \
    --top-k 50
```

参数: `--input-queries` `--output` `--top-k 50` `--bm25s-dir` (默认 `{DATA_ROOT}/data/bm25s_index/t2ranking/`)

### Step 2.5: RRF 融合

| 项目 | 内容 |
|------|------|
| 脚本 | [x] `scripts/exp009/rrf_fuse.py` |
| 输入 | 4 路检索结果: `dense_B0` + `dense_P2` + `dense_H2` + `bm25_B0` |
| 输出 | `{DATA_ROOT}/data/processed/exp009_rrf_fused.jsonl` |
| 方法 | Reciprocal Rank Fusion (k=60, per-route K=50, 输出 top-50) |
| 复用 | exp003 `rrf_fuse()` 函数 |

```powershell
python scripts/exp009/rrf_fuse.py \
    --route-files \
        data/processed/exp009_dense_B0.jsonl \
        data/processed/exp009_dense_P2.jsonl \
        data/processed/exp009_dense_H2.jsonl \
        data/processed/exp009_bm25_B0.jsonl \
    --per-route-k 50 \
    --rrf-k 60 \
    --output-top-k 50 \
    --output data/processed/exp009_rrf_fused.jsonl
```

参数: `--route-files` (4个) `--per-route-k 50` `--rrf-k 60` `--output-top-k 50` `--output`

### Step 2.6: Reranker 精排

| 项目 | 内容 |
|------|------|
| 脚本 | [x] `scripts/exp009/rerank.py` |
| 输入 | `{DATA_ROOT}/data/processed/exp009_rrf_fused.jsonl` (top-50 per query) |
| 输出 | `{DATA_ROOT}/data/processed/exp009_reranked_top10.jsonl` |
| 模型 | bge-reranker-v2-m3 (Cross-Encoder) |
| 设备 | RTX 4090, ~30 分钟 |

```powershell
python scripts/exp009/rerank.py \
    --input data/processed/exp009_rrf_fused.jsonl \
    --model BAAI/bge-reranker-v2-m3 \
    --device cuda \
    --top-k 10 \
    --output data/processed/exp009_reranked_top10.jsonl
```

参数: `--input` `--model BAAI/bge-reranker-v2-m3` `--device cuda` `--top-k 10` `--output`

---

## 阶段三：教师生成

用 qwen3-max 为每条 query 生成高质量参考答案。

| 项目 | 内容 |
|------|------|
| 脚本 | [ ] `scripts/exp009/generate_teacher_answers.py` |
| 输入 | `{DATA_ROOT}/data/processed/exp009_reranked_top10.jsonl` (query + top-10 passages) |
| 输出 | `{DATA_ROOT}/data/processed/exp009_teacher_answers.jsonl` |
| Prompt | 复用 `src/generation/prompts.py`（SYSTEM_PROMPT + FEW_SHOT，同 exp-005） |
| 模型 | qwen3-max, temperature=0.3, thinking=OFF, max_tokens=1024 |
| 费用 | 5000 × ¥0.003 ≈ ¥15 |

**输出格式**：
```json
{"qid": "50000", "query": "...", "passages": [...], "teacher_answer": "...", "model": "qwen3-max", "temperature": 0.3}
```

---

## 阶段四：质量过滤

筛除教师生成的脏数据（空答案、重复、非中文、意外拒答、无引用、准确度不达标）。

| 项目 | 内容 |
|------|------|
| 脚本 | [ ] `scripts/exp009/filter_and_bucket.py` (过滤部分) |
| 输入 | `{DATA_ROOT}/data/processed/exp009_teacher_answers.jsonl` (~5000 条) |
| 输出 | `{DATA_ROOT}/data/processed/exp009_filtered_answers.jsonl` (~3000 条, ~60% 留存) |
| 规则过滤 | 空/超长/重复/非中文/意外拒答/无引用标记 |
| Judge 过滤 | deepseek-chat 单维度准确性评分，< 3 → 丢弃 |
| 费用 | ~4000 × ¥0.001 ≈ ¥4 |

---

## 阶段五：检索质量分桶

按 search_coverage 和 best_grade 将过滤后数据归入 A/B/C 三个桶。

| 项目 | 内容 |
|------|------|
| 脚本 | [ ] `scripts/exp009/filter_and_bucket.py` (分桶部分) |
| 输入 | `{DATA_ROOT}/data/processed/exp009_filtered_answers.jsonl` |
| 输出 | 同文件（新增 `bucket` 字段）, 桶 A(~56%) / B(~24%) / C(~20%) |

| 桶 | 条件 | 目标类别 |
|:--:|------|------|
| A | coverage ≥ 0.5 且 best_grade ≥ 2 | 标准 / 引文强调 |
| B | 0.2 ≤ coverage < 0.5 或 best_grade == 1 | 部分标准 / 噪声注入 |
| C | coverage < 0.2 或 top-10 全部无标注 | 拒答 / 噪声 / 矛盾 |

**注意**：桶 C ≠ T3 层。T3 是 query 层面的先验（num_positive），桶 C 是检索结果的后验（coverage）。T3 的 query 如果检索意外成功 → 进桶 A。

---

## 阶段六：类别构造

> **5 类 SFT 数据**：标准搜索问答(55%) + 引文强调(12%) + 信息不足/拒答(17%) + 噪声干扰(8%) + 矛盾处理(8%)

| 项目 | 内容 |
|------|------|
| 脚本 | [ ] `scripts/exp009/construct_categories.py` |
| 输入 | `{DATA_ROOT}/data/processed/exp009_filtered_answers.jsonl` (含 bucket 字段) |
| 输出 | `{DATA_ROOT}/data/processed/exp009_sft_train.jsonl` (~2720 条) |
| | `{DATA_ROOT}/data/processed/exp009_sft_val.jsonl` (桶 A 标准数据 200 条) |
| 额外 API 调用 | 引文强调重生成 320 条 + 噪声重生成 220 条 + 矛盾检测 2000 条 + 矛盾重生成 220 条 |
| 额外费用 | ~¥4.4 |

**输出格式**（训练用）：
```json
{
  "qid": "50000",
  "query": "如何减肥最快",
  "category": "standard",
  "system_prompt": "你是AI搜索助手...",
  "input_passages": [{"rank": 1, "pid": "...", "text": "...", "source": "dense_B0"}],
  "output_answer": "根据资料... [来源: 1]",
  "teacher_model": "qwen3-max",
  "teacher_temperature": 0.3,
  "retrieval_bucket": "A",
  "search_coverage": 0.75
}
```

| 类别 | 占比 | 条数 | 数据来源 | 构造方式 |
|------|:--:|:--:|------|------|
| 标准搜索问答 | 55% | ~1500 | 桶 A 随机采样 | 不修改，直接使用教师原始生成 |
| 引文强调 | 12% | ~320 | 桶 A (不重叠) | Prompt 加引文强调指令 → 重生成 |
| 信息不足/拒答 | 17% | ~460 | 桶 C | 仅保留教师正确拒答的样本 |
| 噪声干扰 | 8% | ~220 | 桶 B + 人工噪声 | 注入无关 passage → 重生成 |
| 矛盾处理 | 8% | ~220 | 桶 A+B 含矛盾 | deepseek-chat 检测矛盾 → 重生成 |

---

## 阶段七：QLoRA SFT 训练

| 项目 | 内容 |
|------|------|
| 脚本 | [ ] `scripts/exp009/train_sft.py` |
| 输入 | `{DATA_ROOT}/data/processed/exp009_sft_train.jsonl` (~2720 条) |
| | `{DATA_ROOT}/data/processed/exp009_sft_val.jsonl` (200 条, 仅监控 loss) |
| 输出 | `{DATA_ROOT}/models/qwen3-4b-t2ranking-sft/` (QLoRA adapter + merged) |
| 基座模型 | Qwen/Qwen3-4B |
| 方法 | QLoRA (4-bit NF4, r=16, alpha=32, dropout=0.05) |
| 超参 | epochs=3, lr=2e-4 cosine, batch=16, max_seq=6144 |
| 显存 | ~8-10GB (RTX 4090) |
| 时间 | ~2-3 小时 |

```powershell
python scripts/exp009/train_sft.py \
    --train-data data/processed/exp009_sft_train.jsonl \
    --val-data data/processed/exp009_sft_val.jsonl \
    --base-model Qwen/Qwen3-4B \
    --output-dir models/qwen3-4b-t2ranking-sft \
    --epochs 3 \
    --lr 2e-4
```

---

## 阶段八：评估

| 项目 | 内容 |
|------|------|
| 脚本 | [ ] 复用 `scripts/exp005/evaluate_exp005.py`（或新建 eval wrapper） |
| 测试集 | exp-005 200 条 dev 集（queries.dev.tsv → exp-004 reranker top-10） |
| Judge | deepseek-chat, 6 维评分（准确性+安全性+相关性+整合质量+引文质量+用户体验） |
| 基线 | Qwen3-4B-nonthink 60.2 分 → qwen3-max 72.7 分（差距 12.5 分） |

---

## 文件清单总览

| 阶段 | 文件 | 状态 | 类型 |
|:--:|------|:--:|------|
| 零 | `sample_training_queries.py` | x | stats + sample |
| 一 | `sample_training_queries.py (--mode sample)` | x | 同上 |
| 二 | `run_retrieval_pipeline.ps1` | x | Shell (Win) |
| 二 | `run_retrieval_pipeline.sh` | x | Shell (Linux) |
| 2.1 | `run_query_augment.py` | x | Python |
| 2.2 | `../exp007/build_faiss_index.py` | 复用 | Python |
| 2.3 | `dense_retrieve.py` | x | Python |
| 2.4 | `bm25_retrieve.py` | x | Python |
| 2.5 | `rrf_fuse.py` | x | Python |
| 2.6 | `rerank.py` | x | Python |
| 三 | `generate_teacher_answers.py` | | Python |
| 四+五 | `filter_and_bucket.py` | | Python |
| 六 | `construct_categories.py` | | Python |
| 七 | `train_sft.py` | | Python |
| 八 | 复用 exp005 evaluate | | Python |

### 中间数据文件（全部在 `{DATA_ROOT}/data/processed/`）

| 文件 | 产生于 | 行数 | 格式 |
|------|:--:|:--:|------|
| `exp009_sampled_queries.jsonl` | 阶段一 | 5000 | qid/query/stratum/num_positive |
| `exp009_rewritten_queries.jsonl` | 2.1 | 5000 | qid/original/rewritten |
| `exp009_hyde_answers.jsonl` | 2.1 | 5000 | qid/original/hyde |
| `exp009_dense_B0.jsonl` | 2.3 | 5000 | qid/query/results[50] |
| `exp009_dense_P2.jsonl` | 2.3 | 5000 | qid/query/results[50] |
| `exp009_dense_H2.jsonl` | 2.3 | 5000 | qid/query/results[50] |
| `exp009_bm25_B0.jsonl` | 2.4 | 5000 | qid/query/results[50] |
| `exp009_rrf_fused.jsonl` | 2.5 | 5000 | qid/query/results[50] |
| `exp009_reranked_top10.jsonl` | 2.6 | 5000 | qid/query/results[10] |
| `exp009_teacher_answers.jsonl` | 阶段三 | ~5000 | qid/query/passages/teacher_answer |
| `exp009_filtered_answers.jsonl` | 阶段四+五 | ~3000 | 同上 + bucket + coverage |
| `exp009_sft_train.jsonl` | 阶段六 | ~2720 | SFT 训练格式 |
| `exp009_sft_val.jsonl` | 阶段六 | 200 | SFT 训练格式 |

**另外**：

| 路径 | 产生于 | 说明 |
|------|:--:|------|
| `{DATA_ROOT}/data/vector_db/t2ranking/{model}/` | 2.2 | FAISS 索引 |
| `{DATA_ROOT}/models/qwen3-4b-t2ranking-sft/` | 阶段七 | QLoRA adapter |
