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

公用参数: `-Device cuda` / `-SkipAugment` / `-SkipIndex` / `-RebuildIndex` / `-Backend {vllm,deepseek}`

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
| 模型 | `--backend vllm`（默认）: Qwen3-4B (本地 vLLM, OpenAI compatible) |
| | `--backend deepseek`: DeepSeek API（当 vLLM 部署失败时） |
| 费用 | ¥0（本地推理） / ~¥4（DeepSeek API） |

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
    --backend vllm \
    --llm-url http://localhost:8000/v1 \
    --batch-size 32
```

参数: `--input` `--output-rw` `--output-hy` `--backend {vllm,deepseek}` `--llm-url` `--batch-size 32` `--task {rewrite,hyde,both}`

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

用 qwen3-max Batch API 为每条 query 生成高质量参考答案。

| 项目 | 内容 |
|------|------|
| 脚本 | [x] `scripts/exp009/generate_teacher_answers.py` |
| 输入 | `{DATA_ROOT}/data/processed/exp009_reranked_top10.jsonl` (query + top-10 passages) |
| 输出 | `{DATA_ROOT}/data/processed/exp009_teacher_answers.jsonl` (4,990 entries) |
| | `{DATA_ROOT}/data/processed/exp009_batch_state.json` (batch job state for resume) |
| Prompt | 复用 `src/generation/prompts.py`（SYSTEM_PROMPT + FEW_SHOT，同 exp-005） |
| 模型 | qwen3-max, 阿里云百炼 Batch API（OpenAI 兼容, 费用 50% off） |
| 参数 | temperature=0.3, enable_thinking=False, max_tokens=1024 |
| 费用 | 5000 × ~¥0.0015 ≈ ¥7.5（Batch 半价, 实际按输入输出 token 计费） |
| 结果 | 5000 submitted → 4990 success, 10 条被百炼内容安全审查拦截（data_inspection_failed, 0.2%） |

**三步流程**: `test` → `submit` → `align`

```powershell
# Step 1: 冒烟测试（免费, 用 batch-test-model 验证 5 条）
python scripts/exp009/generate_teacher_answers.py test

# Step 2: 提交 batch job（上传 JSONL, 提交任务, 轮询等待完成）
python scripts/exp009/generate_teacher_answers.py submit

# Step 3: 对齐（下载 batch 结果, 按 custom_id 对齐到原始输入）
python scripts/exp009/generate_teacher_answers.py align
```

**关键实现细节**:
- 使用 `DASHSCOPE_API_KEY`, base_url `https://dashscope.aliyuncs.com/compatible-mode/v1`
- `test` 子命令先试 `batch-test-model` 全链路免费测试, 失败回退到在线 API 测试（¥0.015）
- `submit` 子命令在上传 batch 文件之前, 额外调用一次在线 API（纯 smoke test, ¥0.003）确保 API 完全可用后再提交
- `align` 默认只对齐成功的行，遇到 failed/expired/cancelled 时跳过并打 warning
- 状态文件 `exp009_batch_state.json` 支持断点续跑

**输出格式**：
```json
{"qid": "50000", "query": "...", "passages": [...], "teacher_answer": "...", "model": "qwen3-max", "temperature": 0.3, "batch_job_id": "batch_xxx", "custom_id": "...", "batch_status": "completed"}
```

---

## 阶段四+五：规则过滤 + 检索质量分桶 + (可选) 幻觉检测

两个阶段合并为一个脚本 `filter_and_bucket.py`，按顺序执行：规则过滤 → 检索质量分桶 → (可选幻觉检测)。

| 项目 | 内容 |
|------|------|
| 脚本 | [x] `scripts/exp009/filter_and_bucket.py` |
| 输入 | `{DATA_ROOT}/data/processed/exp009_teacher_answers.jsonl` (4,990 entries) |
| 输出 | `{DATA_ROOT}/data/processed/exp009_filtered_bucketed.jsonl` (过滤后 + bucket + hallu 标注) |
| | `{DATA_ROOT}/data/processed/exp009_discarded.jsonl` (被丢弃的条目) |
| 辅助脚本 | [x] `scripts/exp009/test_hallu_detection.py`（幻觉检测试水, 50 条） |
| | [x] `scripts/exp009/check_synthesis_quality.py`（整合质量抽样, 30 条） |

```powershell
# 仅规则过滤 + 分桶（免费, 瞬时完成）
python scripts/exp009/filter_and_bucket.py

# 规则过滤 + 分桶 + 幻觉检测（deepseek-chat, ~¥5, 8并发 ~20分钟）
python scripts/exp009/filter_and_bucket.py --hallu-check --hallu-workers 8
```

### 规则过滤（3 条规则）

| 规则 | 逻辑 | 丢弃数 |
|------|------|:---:|
| 过短 | `len(answer) < 20` | 0 |
| 意外拒答 | 含拒答关键词 + `len(answer) <= 400` + top-10 有 relevant passage → 教师该答却没答 | 23 |
| 无引用 | `"[来源"` 不在 answer 中 | 33 |
| **合计** | | **56 / 4990 (1.1%)** |

**拒答关键词列表（19 个）**：`"无法确定", "无法回答", "没有提供", "无法提供", "资料中未", "资料中没有", "没有提及", "未提及", "无相关信息", "没有相关信息", "没能找到", "没有找到", "未找到", "参考资料中未", "未涉及", "没有涉及", "无法判断", "无法确认", "没有直接提供"`

**设计要点**:
- `REFUSAL_MAX_LEN = 400`：拒答关键词 + 长答案（>400 chars）通常是"我找到了这些信息, 但你问的 XXX 资料中没有", 不算拒答
- 需要 qrels 交叉检验：只有在"检索确实有相关文档但教师却说没有"时才丢弃（意外拒答）
- 如果检索确实无相关文档, 教师拒答是正确答案 → 保留（用于训练拒答能力）

### 幻觉检测（可选, `--hallu-check`）

当 `--hallu-check` 启用时, 在规则过滤之后对保留的答案做幻觉检测。使用 `ThreadPoolExecutor` 并发调用 API（`--hallu-workers` 控制并发数，默认 8）。

| 项目 | 内容 |
|------|------|
| 模型 | deepseek-chat（**不用** deepseek-reasoner：幻觉检测是结构化比对任务，不需要深度推理） |
| 参数 | temperature=0.0, max_tokens=256 |
| 方式 | 8 并发 ThreadPoolExecutor, ~3.9 q/s |
| 输出 | PASS/FAIL + 一句话理由 |
| 时间 | ~21 分钟 |
| 费用 | ~¥5 |

**Prompt 8 条规则总结**：逐条检查事实陈述 → 看资料是否支撑 → PASS。合理推断、诚实拒答、表述偏差不扣分。张冠李戴（资料说 A 却安到 B 上）必 FAIL。

**试水结果（50 条, 2 轮 prompt 迭代）**：
- Round 1（基础 prompt）: PASS 74%
- Round 2（增加规则 5-8：合理推断/诚实拒答/表述偏差/张冠李戴）: **PASS 82%**
- 估计真实幻觉率约 10%

**全量结果（4934 条）**：PASS 3589 (72.7%), FAIL 1345 (27.3%)。FAIL 率高于试水的 18%，说明全量数据中幻觉问题比抽样更严重。

### 整合质量抽样（已做, 不上线）

| 项目 | 内容 |
|------|------|
| 脚本 | `scripts/exp009/check_synthesis_quality.py` |
| 样本 | 30 条（分桶等比例: 10A+10B+10C） |
| 评分 | deepseek-chat, 单维度 1-4 分 |

```powershell
python scripts/exp009/check_synthesis_quality.py --n 30
```

**结果**：均分 2.90 / 低分 1-2 占 26.7%。结论：部分 2 分条目是 query 类型导致的假阳性（如"锤头线对上影线有要求吗", 来源高度一致不需要整合）, 整合质量不是致命问题, **不做全量过滤**。

### 检索质量分桶

| 桶 | 条件 | 条数 | 占比 |
|:--:|------|:---:|:---:|
| A | coverage ≥ 0.5 且 best_grade ≥ 2 | 1,698 | 47.3% |
| B | 0.2 ≤ coverage < 0.5 或 best_grade == 1 | 679 | 18.9% |
| C | coverage < 0.2 或 top-10 全部无 qrels 标注 | 1,212 | 33.8% |
| 丢弃 | 规则过滤 + 幻觉检测 | 1,401 | 28.1% |

其中丢弃明细：
- 意外拒答 (accidental_refusal): 23 条
- 无引用 (no_citation): 33 条
- 幻觉 (hallu_fail): 1345 条

指标定义：
- `search_coverage` = \|top-10 pids ∩ retrieval_qrels\| / min(10, \|retrieval_qrels\|)
- `best_grade` = max(graded_qrels[pid] for pid in top-10 if pid in graded_qrels)

**与计划的偏离**：桶 A（45%）低于预期（56%）, 桶 C（35%）高于预期（20%）, 反映 M3E-base 检索模型 + bge-reranker-v2-m3 精排的整体质量上限。

**注意**：桶 C ≠ T3 层。T3 是 query 层面的先验（num_positive）, 桶 C 是检索结果的后验。T3 的 query 如果检索意外成功 → 进桶 A。

---

## 阶段六：类别构造（4 个子阶段）

> **目标**：构造 5 种训练场景，教会学生条件化的行为映射——输入什么检索质量，输出什么风格的答案。

```
exp009_filtered_bucketed.jsonl (3589条, 已带 bucket + hallu)
    │
    ├─[6.1] 筛选分类 ──→  标准搜索问答 (不改动)
    │                    信息不足/拒答 (不改动)
    │
    ├─[6.2] 矛盾检测 ──→  找出"教师答案中处理了矛盾"的样本
    │
    ├─[6.3] 改写生成 ──→  引文强调 (重生成)
    │                    噪声干扰 (重生成)
    │                    矛盾处理 (重生成)
    │
    └─[6.4] 组装切分 ──→  exp009_sft_train.jsonl (~2720条)
                         exp009_sft_val.jsonl (200条)
```

| 子阶段 | 操作类型 | API | 费用 | 时间 |
|:--:|------|------|:---:|:---:|
| 6.1 | 🗂️ 分类+采样 + ✅ 打分筛选 | 无 | ¥0 | 瞬时 |
| 6.2 | ✅ 打分筛选 (矛盾检测) | deepseek-chat | ~¥2 | ~5 min (并发) |
| 6.3 | 改写生成 | deepseek-chat (8 并发) | ~¥0.7 | <10 min |
| 6.4 | 🔧 组装切分 | 无 | ¥0 | 瞬时 |

---

### 6.1 — 筛选分类（无 API 调用）

从已有数据中挑出**不需要改写**的两类：

| 类别 | 来源 | 条数 | 怎么选 |
|------|------|:---:|------|
| 标准搜索问答 | 桶 A (hallu=PASS) | ~1200 | 随机采样，不改动 |
| 信息不足/拒答 | 桶 C | ~230 | 筛出"教师正确拒答"的：含拒答关键词 + 检索确实无相关文档 |
| 验证集 | 桶 A (hallu=PASS) | 200 | 从标准类中预留，只监控 loss |

核心逻辑：
- 标准类从桶 A 先拿（检索质量最好），保证训练数据质量底线
- 桶 A 挑剩的部分留给 6.3 做引文强调改写（不重叠）
- 拒答类：只保留"该拒答的"——资料确实没有，教师说没有 → 正确行为。教师不该拒答却拒答的已在阶段四被"意外拒答"规则筛掉

---

### 6.2 — 矛盾检测（deepseek-chat）

在桶 A+B 中检测哪些教师答案**包含了对多文档矛盾的识别和处理**。

| 项目 | 内容 |
|------|------|
| 输入 | 桶 A + 桶 B 的教师答案（除去 6.1 已选为标准类的 ~1500 条） |
| 模型 | deepseek-chat, temperature=0.0 |
| 输出 | `{"has_contradiction": true/false, "contradiction_type": "数值矛盾/观点对立/信息冲突/无"}` |
| 目标 | 找到 ~220 条含矛盾的样本 → 作为 6.3 矛盾处理的输入 |

判断依据：回答是否对比了不同来源的差异、指出了信息冲突、或尝试解释了分歧原因。

---

### 6.3 — 改写生成（deepseek-chat 并发，✅ 完成）

三类改写，都是对**同一条 query + passages** 用不同 prompt 调用 deepseek-chat 重新生成。

| 类别 | 来源 | 条数 | 改了什么 | Prompt 差异 |
|------|------|:---:|------|------|
| 引文强调 | 桶 A（6.1 挑剩的） | ~300 | **不改 passages**，只改 system prompt | 要求"每个关键陈述必须标注来源编号" |
| 噪声干扰 | 桶 B | ~220 | **注入 2-3 条无关 passage**，混入 top-10 | 要求"忽略无关信息，只基于相关部分回答" |
| 矛盾处理 | 桶 A+B（6.2 筛出的） | ~220 | **不改 passages**，只改 system prompt | 要求"对比各方观点，分析差异原因并给出综合判断" |

改用 deepseek-chat + 8 并发（原计划 qwen3-max Batch API，但阿里云 Batch 排队 >40 分钟不可接受），深度改写温度用 0.3，参数：max_tokens=1024。

> **实际耗时**：7.6 min, 738/738 完成（1 条失败），~¥0.7。

---

### 6.4 — 组装切分（无 API 调用）

把 5 类数据拼到一起，输出标准 SFT 训练格式：

```
标准 ~1200 + 引文 ~300 + 拒答 ~230 + 噪声 ~220 + 矛盾 ~220 = ~2170 条 → train
从标准类中预留 200 条 → val（只监控 loss，不参与评估）
```

**输出格式**（训练用）：
```json
{
  "qid": "50000",
  "query": "如何减肥最快",
  "passages": [{"pid": "...", "text": "...", "rank": 1}],
  "answer": "根据资料... [来源: 1,2]",
  "category": "standard",
  "bucket": "A",
  "teacher_model": "qwen3-max"
}
```

| 字段 | 说明 |
|------|------|
| `passages` | top-10 检索结果（噪声类包含注入的无关 passage） |
| `answer` | 教师原始答案（标准/拒答）或 6.3 重生成的答案（引文/噪声/矛盾） |
| `category` | 5 种类别之一：standard / citation_emphasis / refusal / noise / contradiction |
| `bucket` | A/B/C，分桶来源 |

---

### 5 类训练数据的设计意图

| 类别 | 占比(约) | 条数(约) | 教给学生的行为 |
|------|:---:|:---:|------|
| 标准搜索问答 | ~55% | ~1200 | 检索结果好 → 正常引用回答 |
| 引文强调 | ~14% | ~300 | 检索结果好 + 被要求严格引用 → 精细标注来源 |
| 信息不足/拒答 | ~11% | ~230 | 检索结果差 → 诚实说"没有找到信息" |
| 噪声干扰 | ~10% | ~220 | 检索混入无关文档 → 学会忽略它们，不编造 |
| 矛盾处理 | ~10% | ~220 | 检索结果互相矛盾 → 对比分析各方观点 |

> **与计划偏离**：由于幻觉检测筛掉了 1345 条（27.3%），标准类和拒答类的可用数据量低于预期。标准类从 1500 调至 1200，拒答类从 460 调至 234（只剩这么多"正确拒答"的样本）。实际训练数据总量 ~2170 条（vs 计划 2720）。

**操作方式**：

```powershell
python scripts/exp009/construct_categories.py select                     # 6.1 筛选标准+拒答
python scripts/exp009/construct_categories.py detect-contradictions      # 6.2 矛盾检测
python scripts/exp009/construct_categories.py rewrite --workers 8        # 6.3 deepseek-chat 改写
python scripts/exp009/construct_categories.py assemble                   # 6.4 组装切分
```

---

## 阶段七：QLoRA SFT 训练

| 项目 | 内容 |
|------|------|
| 脚本 | [x] `scripts/exp009/train_sft.py` |
| 输入 | `{DATA_ROOT}/data/processed/exp009_sft_train.jsonl` (2172 条) |
| | `{DATA_ROOT}/data/processed/exp009_sft_val.jsonl` (200 条, 仅监控 loss) |
| 输出 | `{DATA_ROOT}/models/qwen3-4b-t2ranking-sft/` (QLoRA adapter + merged) |
| 基座模型 | Qwen/Qwen3-4B |
| 方法 | QLoRA (4-bit NF4, r=16, alpha=32, dropout=0.05) |
| 超参 | epochs=3, lr=2e-4 cosine, batch=16, max_seq=6144 |
| 显存 | ~8-10GB (RTX 4090 24GB) |
| 时间 | ~2-3 小时 |

### 推荐云 GPU 配置

| 平台 | GPU | 显存 | 单价 | 预计费用 |
|------|------|:---:|------|:---:|
| AutoDL | RTX 4090 | 24GB | ~¥1.8/h | ¥4-6 (2-3h) |
| AutoDL | RTX 3090 | 24GB | ~¥1.2/h | ¥4-5 (3-4h) |
| AutoDL | RTX 4070 Ti Super | 16GB | ~¥0.9/h | ¥3-4 (3-4h) |

> 推荐 **RTX 4090**：fp16 推理 + QLoRA 训练完全够用，单价比 3090 贵 50% 但速度快 ~30%，总费用差不多。4070 Ti Super 16GB 够用但显存紧凑。

### 服务器部署流程

```bash
# 1. AutoDL 选配：PyTorch 2.5+ / Python 3.11 / CUDA 12.4+ / 系统盘 50GB

# 2. 上传训练数据（从本机）
scp data/processed/exp009_sft_train.jsonl your_server:/root/autodl-tmp/data/processed/
scp data/processed/exp009_sft_val.jsonl   your_server:/root/autodl-tmp/data/processed/

# 3. 服务器上一键安装 + 下载模型
bash scripts/exp009/server_setup.sh

# 4. 开始训练（模型自动从 HF 下载，或指定本地路径提速）
python scripts/exp009/train_sft.py \
    --train data/processed/exp009_sft_train.jsonl \
    --val data/processed/exp009_sft_val.jsonl \
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
| 三 | `generate_teacher_answers.py` | x | Python |
| 四+五 | `filter_and_bucket.py` | x | Python |
| 四+五 | `test_hallu_detection.py` | x | Python (pilot) |
| 四+五 | `check_synthesis_quality.py` | x | Python (analysis) |
| 六 | `construct_categories.py` | x | Python |
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
| `exp009_batch_state.json` | 阶段三 | - | batch job state (resume) |
| `exp009_teacher_answers.jsonl` | 阶段三 | 4990 | qid/query/passages/teacher_answer |
| `exp009_hallu_test_results.jsonl` | 阶段四 pilot | 50 | qid/verdict/reason |
| `exp009_filtered_bucketed.jsonl` | 阶段四+五 | 3589 | 同上 + bucket + coverage + hallu |
| `exp009_discarded.jsonl` | 阶段四+五 | 1401 | qid/query/discard_reason |
| `exp009_sft_train.jsonl` | 阶段六 | ~2720 | SFT 训练格式 |
| `exp009_sft_val.jsonl` | 阶段六 | 200 | SFT 训练格式 |

**另外**：

| 路径 | 产生于 | 说明 |
|------|:--:|------|
| `{DATA_ROOT}/data/vector_db/t2ranking/{model}/` | 2.2 | FAISS 索引 |
| `{DATA_ROOT}/models/qwen3-4b-t2ranking-sft/` | 阶段七 | QLoRA adapter |
