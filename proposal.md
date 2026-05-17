# 个性化AI搜索系统 - 项目设计书

## 1. 项目概述

### 1.1 项目定位

本项目采用**渐进式迭代策略**构建AI检索生成系统：

- **V0阶段**：优先跑通常规RAG完整流程（文档分片→向量化→向量检索→LLM生成），使用维基百科数据集学习核心技术栈
- **V1+阶段**：在此基础上扩展个性化搜索、多轮对话等高级功能

完整覆盖 **意图理解（查询重写）→ 搜索排序 → LLM生成推荐理由** 三大核心模块，从最简可行版本演进至具备多轮对话、偏好对齐与自动评估的前沿系统。

### 1.2 核心用户场景

用户在网页搜索场景中进行自然语言查询，系统结合其历史搜索偏好返回个性化结果，并生成可解释的推荐理由。

### 1.3 技术亮点

- **意图理解**：基于大模型微调的查询重写，将模糊查询转化为明确搜索词
- **个性化排序**：融合用户长期兴趣与当前查询的多路召回与重排序
- **LLM生成**：基于搜索结果和用户画像生成个性化推荐理由

***

## 2. 数据集选择

### 2.1 主数据集：维基百科（RAG基础流程）

| 属性        | 说明                                                                |
| --------- | ----------------------------------------------------------------- |
| **来源**    | Wikipedia dump 或 Hugging Face 预构建数据集                              |
| **推荐数据集** | `wikipedia` (Hugging Face Datasets) - 英文维基百科                      |
| **替代选择**  | `Cohere/wikipedia-22-12-en-embeddings` (已预嵌入)                     |
| **规模**    | 约600万篇文章，可按需采样（如取前10万篇）                                           |
| **存储大小**  | 原始dump约20GB，采样子集约1-2GB                                            |
| **核心字段**  | `id`、`url`、`title`、`text`（完整文章内容）                                 |
| **获取方式**  | Hugging Face Datasets: `load_dataset("wikipedia", "20220301.en")` |
| **适用模块**  | 文档分片、向量化、向量检索（V0阶段核心）                                             |

**数据集使用策略**：

- **V0开发阶段**：使用维基百科抽样（自抽3万篇）跑通RAG完整流程，重点练习文档切片、嵌入生成、向量检索。抽样数据量小、迭代快，适合日常开发调试
- **V0评估阶段**：引入 **T2Ranking** 作为主力标准化评估基准——30万+真实中文查询 + 4级细粒度相关性标注 + 意图类别标签，可直接用于计算 Recall@k、MRR、NDCG 等检索指标；同时以 **MIRACL** 作为辅助参考（Wikipedia 检索对标 + WSDM 2023 leaderboard）
- **V1+阶段**：可引入AOL4PS等用户行为数据集扩展个性化功能

### 2.1.5 评估数据集

#### 2.1.5.1 主力评估集：T2Ranking ⭐

| 属性 | 说明 |
| --- | --- |
| **来源** | SIGIR 2023 Resource Paper（清华大学 + 腾讯） |
| **语料** | 互联网真实段落，>200 万 unique passages，来自多搜索引擎 |
| **查询集** | **30 万+ 条中文查询**，来自搜狗真实搜索日志 |
| **标注** | **4 级细粒度相关性**（Level 0~3），专家标注 + active learning |
| **Query Types** | ✅ 附带意图类别标注 |
| **文档元数据** | ✅ 附带 XML 源文件信息 |
| **假负例缓解** | ✅ 多商业搜索引擎召回 + 完整标注 |
| **数据划分** | train / dev / test |
| **许可证** | Apache 2.0 |
| **获取方式** | GitHub: `https://github.com/THUIR/T2Ranking` |
| **论文** | Xie et al., "T2Ranking: A large-scale Chinese Benchmark for Passage Ranking", SIGIR 2023 |
| **适用模块** | 检索评估（Recall@k, MRR, NDCG@k）+ 意图识别评估 + 查询重写评估 + 重排序训练 |

**T2Ranking 核心优势**（相比 MIRACL 等数据集）：

| 维度 | T2Ranking 优势 | 对本项目的价值 |
| --- | --- | --- |
| **数据规模** | 30 万+ 查询 vs MIRACL ~1,500 | 统计显著性极高，评估结论更可靠 |
| **查询来源** | 搜狗真实搜索日志（短查询、模糊查询、口语化） | 更贴近真实用户场景，适合评估查询重写 |
| **相关性标注** | 4 级细粒度（0~3）vs MIRACL 二元 | 可计算 nDCG，排序质量评估更精确 |
| **Query Types** | 附带意图分类标签 | 唯一支持意图识别模块评估的数据集 |
| **假负例处理** | 多搜索引擎召回 + 完整标注 | 评估准确性显著优于单源标注 |

**T2Ranking 使用策略**：
- 用其语料构建独立评估索引（~200 万 passages，与 Wikipedia 分开管理）
- Dev split 作为 V0 检索评估主力查询集
- 利用 query types 评估意图分类器准确率
- 利用原始短查询 vs 标准问法评估查询重写效果
- 后续重排序模型训练直接使用其 train/dev/test 划分（见 V1.5.3 节）

**T2Ranking 下载方式**：

| 途径 | 说明 |
| --- | --- |
| **Git LFS 克隆** | `git lfs install && git clone https://huggingface.co/datasets/THUIR/T2Ranking` |
| **Web 单文件下载** | `https://huggingface.co/datasets/THUIR/T2Ranking/resolve/main/data/<文件名>` |
| **镜像下载** | `https://hf-mirror.com/datasets/THUIR/T2Ranking/resolve/main/data/<文件名>` |

**V0 评估阶段仅需以下 4 个文件**（约 3.5 GB，无需下载训练负样本）：

| 文件 | 记录数 | 实际大小 | 用途 |
| --- | --- | --- | --- |
| `collection.tsv` | 2,303,643 | ~3.5 GB | 语料库（pid + passage text），建评估索引 |
| `queries.dev.tsv` | 24,832 | ~0.9 MB | 开发查询（V0 检索评估主力） |
| `qrels.retrieval.dev.tsv` | 118,933 | ~1.4 MB | 检索评估 qrels |
| `queries.test.tsv` | 24,832 | ~0.9 MB | 测试查询（预留 V1 最终评估，避免数据泄露） |

> 两个训练负样本文件（`train.bm25.tsv` 3.65GB + `train.mined.tsv` 5.58GB）仅在 V1 重排序训练阶段需要，V0 评估无需下载。

**向量库增量构建策略**：

T2Ranking 语料（~230 万 passages）通过 `scripts/build_t2ranking_index.py` 分批构建向量索引：

| 属性 | 说明 |
| --- | --- |
| **构建脚本** | `scripts/build_t2ranking_index.py` |
| **批次大小** | 1,000 passages/batch → ~2,304 批 |
| **单批耗时（CPU）** | ~37s（embed + write） |
| **预估总耗时（CPU）** | ~23.5 小时 |
| **ChromaDB 写入方式** | `add_documents()` 增量追加 |
| **HTML 清洗** | 正则去除 `<br>`、`<img>` 等标签（63.4% 段落含 HTML） |
| **分片策略** | 不切片 — T2Ranking passage 即检索单元，保持 pid 与 qrels 一致 |
| **长度过滤** | min 10 字符（去噪音），max 2000 字符（截断，4.1% 段落受影响） |

**状态文件体系**（位于 `data/raw/t2ranking/`）：

| 文件 | 作用 | 关键字段 |
| --- | --- | --- |
| `state.json` | 断点续跑 | `last_processed_line`, `total_stored`, `batches_completed` |
| `index_info.json` | 在线检索元信息 | `collection_name`, `embedding_model`, `embedding_dim`, `total_passages` |
| `build_log.jsonl` | 诊断回溯 | 每批：`batch`, `stored`, `skipped`, `embed_time_s`, `batch_time_s` |

**使用方式**：
```bash
python scripts/build_t2ranking_index.py                    # 续跑（自动从 state.json 恢复）
python scripts/build_t2ranking_index.py --rebuild          # 重建（清除旧索引和状态）
python scripts/build_t2ranking_index.py --dry-run          # 查看计划，不执行
python scripts/build_t2ranking_index.py --max-batches 5    # 限制最多跑 5 批（测试用）
python scripts/build_t2ranking_index.py --device cpu       # CPU 模式（默认，兼容性最好）
```

#### 2.1.5.2 辅助参考集：MIRACL

| 属性 | 说明 |
| --- | --- |
| **来源** | MIRACL 2023 多语言 IR 基准（TACL 2023，Waterloo + 华为） |
| **语料** | 中文维基百科段落（4,934,368 passages，1,246,389 articles） |
| **查询集** | dev ~600 + test-B ~871 ≈ **~1,500 条中文查询** |
| **标注** | 二元相关性（相关/不相关），native speaker 人工标注 + heuristic 验证 |
| **Query Types** | ❌ 不提供 |
| **官方 leaderboard** | ✅ WSDM 2023 Cup 公开排行榜 |
| **许可证** | CC-BY 4.0 |
| **获取方式** | Hugging Face Datasets: `load_dataset("miracl/miracl", "zh")` |
| **适用模块** | Wikipedia 检索表现验证 + 学术界对标 |

**MIRACL 定位**：
- 辅助参考集：当需要验证"纯 Wikipedia 知识检索"能力时使用
- 学术界对标：MIRACL 有 WSDM 2023 Cup 公开 leaderboard，可用于与其他团队的检索系统横向对比
- 语料与开发集同源：MIRACL 语料就是中文 Wikipedia，评估结果可直接反映检索器在 Wikipedia 上的效果
- 备选方案：如 T2Ranking 下载/处理遇到困难，MIRACL 可快速替代作为检索评估集

**三轨数据集分工**：

| 数据集 | 用途 | 规模 | 索引频率 |
| --- | --- | --- | --- |
| **维基百科抽样**（自抽 3 万篇） | V0 开发调试、交互式查询、流程验证 | 271k chunks | 日常使用 |
| **T2Ranking** ⭐ | 主力评估基准：检索+排序+意图识别+查询重写 | 300K+ queries, 2M+ passages | 一次性构建，仅用于评估 |
| **MIRACL** | 辅助参考：Wikipedia 检索表现 + 公开 leaderboard 对标 | ~1,500 queries, 4.9M passages | 一次性构建，仅用于评估

### 2.2 意图识别/查询重写数据集

| 数据集                  | 用途      | 特点               | 获取方式                                                     |
| -------------------- | ------- | ---------------- | -------------------------------------------------------- |
| **TopiOCQA**         | 对话式查询重写 | 开放域对话问答，含多轮对话历史  | Hugging Face/Download                                    |
| **Restoration-200K** | 会话查询重写  | 手动标注的CQR数据集      | 公开下载                                                     |
| **RECAP**            | 意图重写评估  | 针对Agent规划的意图理解基准 | [OpenReview](https://openreview.net/forum?id=UelTYgX3YN) |
| **SynRewrite**       | 合成查询重写  | GPT-4o生成的高质量重写样本 | 论文附带数据                                                   |

### 2.3 LLM生成微调数据集

| 数据集               | 用途   | 特点           | 获取方式         |
| ----------------- | ---- | ------------ | ------------ |
| **Alpaca**        | 指令微调 | 52K高质量指令-回答对 | GitHub       |
| **Dolly**         | 指令微调 | 15K指令-回答对    | Databricks   |
| **UltraChat**     | 多轮对话 | 140万多轮对话样本   | Hugging Face |
| **Self-Instruct** | 指令生成 | 自动生成多样化指令    | GitHub       |

***

## 3. 系统架构与数据流

```
[用户输入(查询+对话历史)] 
       │
       ▼
  ┌──────────────────┐
  │ 意图理解模块      │  ← 解析查询意图，改写/细化查询，识别个性化信号
  │ (Query Rewriting)│
  └────┬─────────────┘
       │ 结构化意图 + 用户画像
       ▼
  ┌──────────────────────┐
  │ 搜索排序模块          │  ← 多路召回(语义+个性化) → 特征排序 → Top-K结果
  │ (Retrieval & Ranking)│
  └──────┬───────────────┘
       │ 有序结果列表
       ▼
  ┌──────────────────┐
  │ LLM生成模块       │  ← 基于结果和画像生成个性化总结与推荐理由
  │ (Generation)     │
  └────┬─────────────┘
       │
       ▼
[最终回复展示给用户]
```

**横向支撑层**：用户画像存储、对话状态管理、评估与日志系统贯穿其中。

***

## 4. 技术选型与环境

| 分类         | 技术                                              | 说明                               |
| ---------- | ----------------------------------------------- | -------------------------------- |
| **开发语言**   | Python 3.10+                                    | 主流AI开发语言                         |
| **深度学习框架** | PyTorch 2.x, Transformers (HuggingFace), TRL    | 支持LLM微调与DPO                      |
| **搜索与检索**  | LangChain / LlamaIndex                          | 切片、检索链                           |
| **向量数据库**  | ChromaDB / FAISS                                | 高效向量检索                           |
| **排序模型**   | LightGBM / LambdaMART                           | 学习排序算法                           |
| **LLM集成**  | OpenAI API / DeepSeek API / Qwen-7B-Chat (vLLM) | **混合方案**：开发阶段用API（效率高），最终部署用本地模型 |
| **前端演示**   | Gradio                                          | 快速构建Web界面                        |
| **评估工具**   | pytrec\_eval, scikit-learn, LLM-as-a-judge      | 多维度评估                            |
| **环境**     | AutoDL GPU（RTX 4090 24G）                        | 按需租用                             |

***

## 5. 版本迭代路线图

### V0 – 基础RAG流程版 (目标：1周)

**核心目标**：快速跑通"文档分片→向量化→向量检索→LLM生成"的完整RAG链路，重点学习核心技术栈。

| 任务       | 描述                                                |
| -------- | ------------------------------------------------- |
| 环境搭建     | 安装依赖，初始化 LangChain/LlamaIndex 项目骨架                |
| 数据获取     | 从Hugging Face加载维基百科数据集，采样适量文档（如1万-10万篇）           |
| 文档索引     | 将维基百科文档切片（chunk size=256, overlap=20），生成嵌入存入向量数据库 |
| 基础检索     | 实现向量相似度检索，返回Top-10文档片段                            |
| LLM生成    | 调用API或本地模型，设计Prompt模板基于检索结果生成回答                   |
| Gradio界面 | 输入框输出框，展示检索片段和AI回复                                |

#### V0.1 – RAG检索质量评估实验（内嵌于V0）

**目标**：在跑通基础流程后，立即建立评估基准线，通过单组件消融实验理解每个环节对最终效果的影响。

##### 实验1：分片策略消融

| 实验组     | chunk\_size | chunk\_overlap | 预期效果                 | 观察指标             |
| ------- | ----------- | -------------- | -------------------- | ---------------- |
| 小分片     | 128         | 20             | 检索更精准，但丢失上下文         | Recall\@5 可能降低   |
| 中分片（基线） | 500         | 50             | 当前基线配置               | 基线分数             |
| 大分片     | 1000        | 100            | 上下文更完整，但噪音更多         | Recall\@5 vs 精确率 |
| 语义分片    | 500         | 50             | 在 `。！？` 处强制断句，比较字符分片 | 可读性 + Recall     |

##### 实验2：嵌入模型对比

| 模型                                | 维度   | 中文支持  | 本地/API     |
| --------------------------------- | ---- | ----- | ---------- |
| `BAAI/bge-small-zh-v1.5`          | 512  | ★★★★  | 本地免费       |
| `BAAI/bge-large-zh-v1.5`          | 1024 | ★★★★★ | 本地免费（更大更慢） |
| `moka-ai/m3e-base`                | 768  | ★★★★  | 本地免费       |
| `text-embedding-3-small` (OpenAI) | 1536 | ★★★   | API按量付费    |

##### 实验3：检索策略对比

| 策略                           | 说明               | 适用场景       |
| ---------------------------- | ---------------- | ---------- |
| `similarity`                 | 余弦相似度检索（当前默认）    | 通用、稳定.     |
| `mmr`                        | 最大边际相关：平衡相关性与多样性 | 需要覆盖多方面信息时 |
| `similarity_score_threshold` | 设置最低相似度阈值过滤噪音    | 语料质量参差不齐时  |

##### 评估查询集设计

**主力方案：T2Ranking 标准化查询集 ⭐**

直接使用 T2Ranking 提供的 `dev` split（数万条中文查询），附带官方专家标注的 4 级细粒度 qrels。查询来自搜狗真实搜索日志，覆盖短查询、模糊查询、口语化表达等真实场景，同时附带 query types 可用于意图识别评估。学术权威性高（SIGIR 2023）。

**辅助参考方案：MIRACL 标准化查询集**

使用 MIRACL 的 `dev` split（~600 条中文查询），附带官方 native speaker 标注的二元 qrels。语料同为中文 Wikipedia，评估结果可直接在 WSDM 2023 Cup 公开 leaderboard 上与其他团队横向对比。

**备选方案：LLM Pooling 标注**

若上述数据集的语料与自抽文档重叠度不足，启动 LLM 自动标注流程：

```
Step 1: 多系统召回构造 Pool
  对每条查询，用 B0(BM25) + B1(Dense) + B2(Hybrid) 各取 Top-50
  三路结果并集去重 → Pool (约 80-150 篇/查询)

Step 2: LLM 批量打分
  对 Pool 内每篇文档，调用 DeepSeek API 评估相关性：
  "查询：{query} | 文档：{title + first 300 chars}"
  → 输出 3(直接相关)/2(部分相关)/1(弱相关)/0(不相关)

Step 3: 保存为 qrels
  将 LLM 打分结果按 TREC qrels 格式存储，后续评估脚本统一读取
```

- **数量**：30-50 条中文查询（与 MIRACL 互补覆盖）
- **类型配比**：

  - 事实型（"爱因斯坦什么时候出生？"）—— 测精确匹配
  - 概念型（"什么是量子纠缠？"）—— 测语义理解
  - 开放型（"人工智能的未来发展趋势"）—— 测覆盖面
  - 对比型（"儒家和道家的核心区别"）—— 测多文档关联

- **标注**：每条查询标注 3-5 篇相关文档标题，用于计算 Recall\@k, MRR

**Pooling 标注注意事项**：
- Pool 外的文档默认标为"不相关"（TREC Pooling 假设）
- LLM 对"弱相关 vs 不相关"边界判断不如人工精确，建议最终以二元相关（相关=≥2, 不相关=≤1）计算指标
- 一次性标注后保存为 `data/evaluation/qrels.json`，后续所有实验复用

##### 核心评估指标（检索层面）

| 指标                             | 公式/含义             | 关注点            |
| ------------------------------ | ----------------- | -------------- |
| **Recall\@k**                  | Top-K结果中包含相关文档的比例 | 是否"找到了"——召回能力  |
| **MRR** (Mean Reciprocal Rank) | 首个相关文档排名的倒数均值     | 是否"排在前面"——排序质量 |
| **NDCG\@k**                    | 归一化折损累计增益         | 综合考虑排名和相关性等级   |
| **Hit Rate\@k**                | 至少命中一个相关的查询比例     | 粗粒度成功率         |

#### V0.2 – 检索基线递进设计

**设计理念**：在 V0 阶段即建立清晰的多层检索基线体系，使每一步改进都有量化的对比基准。整个项目采用"关键词 → 语义 → 混合 → 个性化"递进式进化路径，确保每个阶段的增益都有数据支撑。

##### 基线层级总览

```
Baseline 0: Pure BM25 (纯关键词检索)
    │  技术：BM25 倒排索引，无需模型
    │  定位：传统 IR 基准，验证语义检索的必要性
    ↓  + 向量嵌入 (embedding)
    
Baseline 1: Pure Dense (纯向量检索) ✅ 当前 V0 实现
    │  技术：bge-small-zh-v1.5 + ChromaDB 向量相似度检索
    │  定位：V0 纯语义检索基线，验证嵌入模型与分片策略的有效性
    ↓  + BM25 + RRF 融合
    
Baseline 2: Hybrid Search (混合检索) ⭐ V0 系统基线
    │  技术：BM25 + Dense 双路召回，EnsembleRetriever + RRF 融合
    │  定位：V0 阶段最终系统基线，融合关键词精确匹配与语义理解优势
    ↓  + 用户画像 + 个性化召回
    
Baseline 3: Personalized Hybrid (个性化混合检索) 🎯 V1 目标
    │  技术：在 Hybrid 基础上加入用户兴趣向量，多路召回 + LightGBM/LambdaMART 重排序
    │  定位：V1 核心系统，实现个性化搜索
```

##### 各基线详细说明

| 基线                           | 检索方式                     | 核心组件                                   | 预期优势                         | 预期不足                     | 实现阶段        |
| ---------------------------- | ------------------------ | -------------------------------------- | ---------------------------- | ------------------------ | ----------- |
| **B0 — Pure BM25**           | 纯关键词倒排索引                 | `BM25Retriever` (langchain\_community) | 精确匹配（实体名、数字、代号）；零模型成本        | 无法理解语义、同义词；对开放型问题检索效果差   | V0          |
| **B1 — Pure Dense**          | 纯向量语义检索                  | `bge-small-zh-v1.5` + ChromaDB         | 语义理解、跨语言/同义词；概念型/开放型问题效果好    | 对专有名词精确匹配弱；罕见实体检索不稳定     | V0（已实现）     |
| **B2 — Hybrid**              | BM25 + Dense 双路召回，RRF 融合 | `EnsembleRetriever` + RRF              | **互补优势**：精确匹配 + 语义理解；工业界事实标准 | 需要保留原始文档建 BM25 索引；检索延迟翻倍 | **V0 系统基线** |
| **B3 — Personalized Hybrid** | B2 + 用户画像向量 + 重排序        | 用户兴趣向量 + LambdaMART                    | 个性化排序，结果贴合用户偏好               | 需要用户行为数据（AOL4PS）；冷启动问题   | V1          |

##### 融合策略选型

| 融合方法                             | 原理                                             | 优点                        | 缺点               | 选用阶段      |
| -------------------------------- | ---------------------------------------------- | ------------------------- | ---------------- | --------- |
| **RRF (Reciprocal Rank Fusion)** | 对每条结果的排名取倒数求和：`score(d) = Σ 1/(k + rank_i(d))` | 无需调参，数学上稳定，LangChain 原生支持 | 丢失原始分数信息         | **V0 默认** |
| 加权分数融合                           | 分别归一化 BM25 和向量相似度分数，加权求和                       | 可精细调节 BM25/向量的权重比例        | 分数归一化麻烦，需要调 α 参数 | V0 备选     |
| 多路召回 + LTR 重排序                   | 两路召回各取 Top-N，用 LightGBM/LambdaMART 学习融合权重      | 效果最优，可利用更多特征              | 需要标注训练数据         | V1        |

##### 基线对比评估矩阵

使用统一的 T2Ranking 查询集（dev split）配合 T2Ranking 官方 qrels，一次性跑通四组基线。同时在自建 30k Wikipedia 索引上使用 MIRACL 做辅助参考评估（验证纯 Wikipedia 检索能力，可与学术界对标）：

| 实验组    | 检索器                 | Recall\@5 | MRR   | NDCG\@5 | 说明          |
| ------ | ------------------- | --------- | ----- | ------- | ----------- |
| B0     | BM25 only           | —         | —     | —       | 证明纯关键词不够    |
| B1     | Dense only (当前)     | —         | —     | —       | 纯向量检索（当前实现） |
| **B2** | **Hybrid (RRF)**    | **—**     | **—** | **—**   | **新系统基线**   |
| B3     | Personalized Hybrid | —         | —     | —       | V1 研发完成后填入  |

**评估数据源分工**：
- **T2Ranking**（主力）：检索能力 + 排序质量 + 意图识别 + 查询重写 全链路评估
- **MIRACL**（辅助参考）：纯 Wikipedia 检索表现 + WSDM 2023 Cup leaderboard 对标

**预期增益验证**：

- **语义增益**：B1 vs B0，验证向量嵌入相比纯关键词的语义理解提升（预期概念型/开放型查询 Recall 显著提升）
- **混合增益**：B2 vs B1，验证关键词+向量的互补效果（预期精确匹配型查询 Recall 提升 5-15%）
- **个性化增益**：B3 vs B2，验证个性化排序的增量价值（V1 阶段）

### V1 – 个性化与系统化评估版 (目标：2-3周，核心版本)

**核心目标**：注入推荐能力，构建完整的离线评估体系。

#### 5.1 用户画像构建

- 从AOL4PS训练集中提取用户高点击文档列表
- 计算这些文档向量的平均嵌入，作为用户的"长期兴趣向量"
- 提取最近交互的K个文档作为"短期兴趣向量"

#### 5.2 多路召回融合

- **语义召回**：查询嵌入直接检索
- **个性化召回**：使用 `user_embedding * 0.7 + query_embedding * 0.3` 构造混合向量检索
- **融合策略**：并集去重

#### 5.3 重排序（精排）

- 特征工程：语义相似度、用户兴趣余弦相似度、文档流行度、类型匹配度等
- 排序模型：**LambdaMART**（经典学习排序算法，效果优异）
- 重排序后截取 Top-5 送给 LLM
- **训练数据划分**：直接使用 T2Ranking 官方 `train/dev/test` 划分，无需手动切分
  - T2Ranking 的 4 级细粒度标注天然适合训练重排序模型
  - 重排序模型的训练和超参选择基于 T2Ranking `train` + `dev`
  - 最终检索评估指标只在 T2Ranking `test` 上计算
  - 辅助方案：如需评估 Wikipedia 场景的重排序效果，可用 MIRACL 的 `train/dev/test-B` 划分

#### 5.4 意图理解微调

- **阶段策略**：先实现查询重写功能，后续迭代再添加意图分类
- 使用 TopiOCQA + Restoration-200K 数据集
- 微调 `distilbert-base-uncased` 或 `bert-base-chinese`
- 意图类别（后续扩展）：指定主题、按类型搜索、模糊推荐、个性化闲聊

**增强查询重写方案（V1.5扩展，暂不开发）**：

- **设计思路**：让LLM先尝试回答用户查询，再基于回答内容生成更精准的搜索关键词
- **执行流程**：
  1. 用户查询 → LLM生成初步回答
  2. 从回答中提取核心概念、实体、关键词
  3. 将提取的关键词与原始查询融合，生成增强后的搜索词
- **预期收益**：增加查询与文档的语义相关性，提升召回精度
- **应用场景**：模糊查询、开放性问题、需要深度理解的查询

#### 5.5 系统化评估框架

| 评估方式           | 指标/方法                                           | 目的      |
| -------------- | ----------------------------------------------- | ------- |
| 离线指标           | MRR\@10, NDCG\@10, **Personalized Hit Rate\@k** | 量化检索效果  |
| 基线对比           | 纯语义搜索 vs 热门推荐 vs 个性化系统                          | 验证个性化收益 |
| LLM-as-a-judge | 相关性、个性化、流畅性、事实一致性（1-5分）                         | 生成质量评估  |
| 模拟在线           | Gradio界面加入👍/👎按钮，收集用户反馈                        | 真实用户验证  |

**个性化增益指标定义（Personalized Gain\@k）**：

- 计算公式：`PG@k = (个性化系统HR@k - 纯语义系统HR@k) / 纯语义系统HR@k × 100%`
- 含义：衡量个性化系统相比纯语义搜索在用户偏好文档上的提升比例
- HR\@k：用户高点击/偏好文档在top-K结果中的命中率

### V2 – 多轮对话与高级优化版 (目标：3-4周，打造亮点)

**核心目标**：实现多轮对话交互，添加创新评估手段。

#### 5.6 多轮对话搜索

- 升级意图理解，送入最近3轮对话历史
- 检索时考虑上下文中的实体
- LLM生成时根据对话历史进行连贯回复

#### 5.7 "AI评审团"创意评估

- 抽取典型用户画像（如"科技爱好者"、"新闻关注者"）
- 将用户偏好摘要注入LLM，创建用户模拟代理
- 设计评估集（20个查询），让代理对结果打分评价
- 聚合评分，自动化评估个性化程度和多样性

#### 5.8 偏好对齐 (DPO)

- 收集偏好对（好回复 vs 坏回复）构建偏好数据集
- 使用 TRL 库的 DPOTrainer 进行对齐训练
- 评估对齐前后 LLM-as-a-judge 分数变化

***

## 6. 项目文件结构

```
personalized-ai-search/
├── data/                    # 数据集与预处理脚本
│   ├── raw/                 # 原始数据集
│   │   ├── wikipedia/       # 维基百科数据集（V0阶段）
│   │   ├── t2ranking/       # T2Ranking 中文检索评估数据集 ⭐ 主力
│   │   ├── miracL/          # MIRACL 中文检索评估数据集（辅助参考）
│   │   ├── aol4ps/          # AOL4PS数据集（V1+阶段预留）
│   │   ├── query_rewrite/   # 查询重写数据集
│   │   │   ├── topiocqa/    # TopiOCQA数据集
│   │   │   └── restoration/ # Restoration-200K数据集
│   │   └── llm_finetune/    # LLM微调数据集
│   └── processed/           # 预处理后的数据
│       ├── documents/       # 处理后的文档
│       ├── queries/         # 处理后的查询
│       └── user_profiles/   # 用户画像数据
├── evaluation/              # 评估数据（与代码分离，一次性生成）
│   ├── t2ranking_queries.json # T2Ranking 查询集（adapter格式）⭐ 主力
│   ├── t2ranking_qrels.json   # T2Ranking 官方 4 级 qrels
│   ├── miracl_queries.json  # MIRACL 查询集（adapter格式，辅助参考）
│   ├── miracl_qrels.json    # MIRACL 官方 qrels
│   ├── custom_queries.json  # 自建查询集（LLM Pooling备选）
│   └── custom_qrels.json    # LLM Pooling标注结果
├── src/
│   ├── data/                # 数据处理模块
│   │   ├── downloader.py    # 数据集下载脚本
│   │   ├── processor.py     # 数据预处理（清洗、标准化）
│   │   ├── analyzer.py      # 数据探索性分析（EDA）
│   │   └── io.py            # 数据读写工具
│   ├── indexing/            # 文档切片、向量库构建
│   │   ├── chunker.py       # 文档切片器
│   │   ├── embedder.py      # 嵌入模型封装
│   │   └── vector_db.py     # 向量数据库操作
│   ├── intent/              # 意图识别与查询重写模块
│   │   ├── __init__.py
│   │   ├── base.py          # 基础抽象类定义
│   │   ├── api_client.py    # API调用客户端（OpenAI/DeepSeek等）
│   │   ├── local_model.py   # 本地模型推理
│   │   ├── prompt_manager.py# Prompt模板管理（可配置）
│   │   ├── intent_classifier.py # 意图分类器（后续扩展）
│   │   ├── query_rewriter.py    # 查询重写器
│   │   ├── trainer.py       # 模型微调脚本
│   │   └── config.py        # 意图模块配置
│   ├── retrieval/           # 多路召回、重排序
│   │   ├── __init__.py
│   │   ├── recall.py
│   │   └── ranker.py
│   ├── generation/          # LLM生成模块
│   │   ├── __init__.py
│   │   ├── generator.py
│   │   └── prompts.py
│   ├── evaluation/          # 指标计算、裁判评分、模拟用户
│   │   ├── __init__.py
│   │   ├── metrics.py
│   │   └── judge.py
│   └── utils/               # 通用工具函数
│       ├── __init__.py
│       └── logging.py
├── models/                  # 微调后的模型权重
│   ├── intent/              # 意图识别模型
│   └── ranking/             # 排序模型
├── app.py                   # Gradio 主界面
├── config.yaml              # 全局配置项
├── requirements.txt
└── README.md                # 项目说明与迭代日志
```

***

### 6.1 数据处理模块 - 类设计

#### 6.1.1 数据集下载器 (`downloader.py`)

| 类名                    | 方法                                   | 输入参数                                | 输出      | 功能说明                       |
| --------------------- | ------------------------------------ | ----------------------------------- | ------- | -------------------------- |
| `WikipediaDownloader` | `__init__(save_dir)`                 | save\_dir: str                      | -       | 初始化下载器，指定保存目录              |
| `WikipediaDownloader` | `download(num_samples=None)`         | num\_samples: Optional\[int]        | Dataset | 从Hugging Face加载维基百科数据集，可采样 |
| `WikipediaDownloader` | `save_to_disk(dataset, output_path)` | dataset: Dataset, output\_path: str | bool    | 保存数据集到磁盘                   |
| `WikipediaDownloader` | `load_from_disk(input_path)`         | input\_path: str                    | Dataset | 从磁盘加载数据集                   |

**配置示例**（config.yaml）：

```yaml
data:
  download:
    wikipedia:
      name: "wikipedia"
      version: "20220301.en"
      save_dir: "./data/raw/wikipedia/"
      num_samples: 100000  # 采样10万篇，可调整
```

#### 6.1.2 数据预处理 (`processor.py`)

| 类名              | 方法                                       | 输入参数                                  | 输出           | 功能说明             |
| --------------- | ---------------------------------------- | ------------------------------------- | ------------ | ---------------- |
| `DataProcessor` | `__init__(config)`                       | config: dict                          | -            | 初始化处理器           |
| `DataProcessor` | `load_raw_data(file_path)`               | file\_path: str                       | pd.DataFrame | 加载原始数据           |
| `DataProcessor` | `clean_text(text)`                       | text: str                             | str          | 清洗文本（去HTML、特殊字符） |
| `DataProcessor` | `normalize_text(text)`                   | text: str                             | str          | 文本标准化（大小写、数字处理）  |
| `DataProcessor` | `filter_short_documents(min_length)`     | min\_length: int                      | pd.DataFrame | 过滤过短文档           |
| `DataProcessor` | `remove_duplicates()`                    | -                                     | pd.DataFrame | 去重               |
| `DataProcessor` | `save_processed_data(data, output_path)` | data: pd.DataFrame, output\_path: str | bool         | 保存处理后数据          |

#### 6.1.3 数据探索性分析 (`analyzer.py`)

| 类名             | 方法                             | 输入参数               | 输出   | 功能说明              |
| -------------- | ------------------------------ | ------------------ | ---- | ----------------- |
| `DataAnalyzer` | `__init__(data)`               | data: pd.DataFrame | -    | 初始化分析器            |
| `DataAnalyzer` | `get_basic_stats()`            | -                  | dict | 基本统计（文档数、用户数、查询数） |
| `DataAnalyzer` | `analyze_document_length()`    | -                  | dict | 文档长度分布            |
| `DataAnalyzer` | `analyze_query_distribution()` | -                  | dict | 查询频率分布            |
| `DataAnalyzer` | `analyze_user_behavior()`      | -                  | dict | 用户行为分析（点击次数、会话长度） |
| `DataAnalyzer` | `generate_report(output_path)` | output\_path: str  | -    | 生成分析报告            |

#### 6.1.4 数据读写工具 (`io.py`)

| 类名       | 方法                               | 输入参数                                | 输出           | 功能说明        |
| -------- | -------------------------------- | ----------------------------------- | ------------ | ----------- |
| `DataIO` | `read_csv(file_path)`            | file\_path: str                     | pd.DataFrame | 读取CSV文件     |
| `DataIO` | `write_csv(data, file_path)`     | data: pd.DataFrame, file\_path: str | bool         | 写入CSV文件     |
| `DataIO` | `read_json(file_path)`           | file\_path: str                     | dict/list    | 读取JSON文件    |
| `DataIO` | `write_json(data, file_path)`    | data: dict/list, file\_path: str    | bool         | 写入JSON文件    |
| `DataIO` | `read_parquet(file_path)`        | file\_path: str                     | pd.DataFrame | 读取Parquet文件 |
| `DataIO` | `write_parquet(data, file_path)` | data: pd.DataFrame, file\_path: str | bool         | 写入Parquet文件 |

***

### 6.2 索引模块 - 类设计

#### 6.2.1 文档切片器 (`chunker.py`)

| 类名                | 方法                                     | 输入参数                           | 输出                | 功能说明     |
| ----------------- | -------------------------------------- | ------------------------------ | ----------------- | -------- |
| `DocumentChunker` | `__init__(chunk_size=256, overlap=20)` | chunk\_size: int, overlap: int | -                 | 初始化切片器   |
| `DocumentChunker` | `chunk_document(text)`                 | text: str                      | List\[str]        | 切分单篇文档   |
| `DocumentChunker` | `chunk_documents(documents)`           | documents: List\[str]          | List\[List\[str]] | 批量切分文档   |
| `DocumentChunker` | `get_chunk_stats(chunks)`              | chunks: List\[str]             | dict              | 统计切片长度分布 |

#### 6.2.2 嵌入模型封装 (`embedder.py`)

| 类名         | 方法                                        | 输入参数              | 输出         | 功能说明    |
| ---------- | ----------------------------------------- | ----------------- | ---------- | ------- |
| `Embedder` | `__init__(model_name="all-MiniLM-L6-v2")` | model\_name: str  | -          | 初始化嵌入模型 |
| `Embedder` | `encode(texts)`                           | texts: List\[str] | np.ndarray | 生成文本嵌入  |
| `Embedder` | `encode_single(text)`                     | text: str         | np.ndarray | 生成单文本嵌入 |
| `Embedder` | `get_embedding_dimension()`               | -                 | int        | 获取嵌入维度  |
| `Embedder` | `save_model(path)`                        | path: str         | -          | 保存模型    |
| `Embedder` | `load_model(path)`                        | path: str         | -          | 加载模型    |

**支持的嵌入模型**：

- sentence-transformers: `all-MiniLM-L6-v2`, `all-mpnet-base-v2`, `all-distilroberta-v1`
- 预留扩展: OpenAI embeddings, BGE models

#### 6.2.3 向量数据库操作 (`vector_db.py`)

| 类名         | 方法                                            | 输入参数                                                              | 输出          | 功能说明     |
| ---------- | --------------------------------------------- | ----------------------------------------------------------------- | ----------- | -------- |
| `VectorDB` | `__init__(db_path, embedding_dim)`            | db\_path: str, embedding\_dim: int                                | -           | 初始化向量数据库 |
| `VectorDB` | `add_documents(chunks, embeddings, metadata)` | chunks: List\[str], embeddings: np.ndarray, metadata: List\[dict] | bool        | 添加文档     |
| `VectorDB` | `search(query_embedding, top_k=10)`           | query\_embedding: np.ndarray, top\_k: int                         | List\[dict] | 向量检索     |
| `VectorDB` | `delete_document(doc_id)`                     | doc\_id: str                                                      | bool        | 删除文档     |
| `VectorDB` | `update_document(doc_id, chunk, embedding)`   | doc\_id: str, chunk: str, embedding: np.ndarray                   | bool        | 更新文档     |
| `VectorDB` | `persist()`                                   | -                                                                 | bool        | 持久化到磁盘   |

**配置示例**（config.yaml）：

```yaml
indexing:
  chunk_size: 256
  chunk_overlap: 20
  embedding_model: "all-MiniLM-L6-v2"
  vector_db:
    type: "chromadb"  # 可选: chromadb, faiss
    path: "./data/vector_db/"
    top_k: 10
```

***

### 6.3 意图识别/查询重写模块 - 类设计

#### 6.3.1 基础抽象类 (`base.py`)

| 类名                   | 方法                                                                | 功能说明               |
| -------------------- | ----------------------------------------------------------------- | ------------------ |
| `BaseLLMClient`      | `generate(prompt: str) -> str`                                    | 抽象方法，子类实现LLM调用     |
| `BaseQueryRewriter`  | `rewrite(query: str, context: Optional[List[str]] = None) -> str` | 抽象方法，子类实现查询重写      |
| `BasePromptTemplate` | `format(**kwargs) -> str`                                         | 抽象方法，子类实现Prompt格式化 |

#### 6.3.2 API客户端 (`api_client.py`)

| 类名                 | 方法                                          | 输入参数                                    | 输出              |
| ------------------ | ------------------------------------------- | --------------------------------------- | --------------- |
| `OpenAIClient`     | `generate(prompt, model="gpt-3.5-turbo")`   | prompt: str, model: str                 | response: str   |
| `DeepSeekClient`   | `generate(prompt, model="deepseek-chat")`   | prompt: str, model: str                 | response: str   |
| `APIClientFactory` | `create(client_type: str) -> BaseLLMClient` | client\_type: str ("openai"/"deepseek") | BaseLLMClient实例 |

**配置示例**（config.yaml）：

```yaml
intent:
  api_client:
    type: "openai"  # 可选: openai, deepseek
    api_key: "${OPENAI_API_KEY}"
    model: "gpt-3.5-turbo"
    max_tokens: 512
    temperature: 0.7
```

#### 6.3.3 Prompt模板管理 (`prompt_manager.py`)

| 类名              | 方法                                  | 功能说明       |
| --------------- | ----------------------------------- | ---------- |
| `PromptManager` | `register_template(name, template)` | 注册Prompt模板 |
| `PromptManager` | `get_template(name)`                | 获取指定模板     |
| `PromptManager` | `format_template(name, **kwargs)`   | 格式化模板      |

**内置Prompt模板示例**：

| 模板名称                         | 用途       | 模板内容                                            |
| ---------------------------- | -------- | ----------------------------------------------- |
| `query_rewrite_basic`        | 基础查询重写   | "请将用户查询改写成更明确的搜索词：{query}"                      |
| `query_rewrite_context`      | 带上下文查询重写 | "基于对话历史：{context}，将当前查询重写为独立搜索词：{query}"        |
| `query_rewrite_personalized` | 个性化查询重写  | "用户偏好：{user\_profile}\n查询：{query}\n请重写为更精准的搜索词" |

#### 6.3.4 查询重写器 (`query_rewriter.py`)

| 类名              | 方法                                                | 输入参数                                                                      | 输出                             |
| --------------- | ------------------------------------------------- | ------------------------------------------------------------------------- | ------------------------------ |
| `QueryRewriter` | `__init__(llm_client, prompt_manager)`            | llm\_client: BaseLLMClient, prompt\_manager: PromptManager                | -                              |
| `QueryRewriter` | `rewrite(query, context=None, user_profile=None)` | query: str, context: Optional\[List\[str]], user\_profile: Optional\[str] | rewritten\_query: str          |
| `QueryRewriter` | `batch_rewrite(queries, contexts=None)`           | queries: List\[str], contexts: Optional\[List\[List\[str]]]               | rewritten\_queries: List\[str] |

**输入输出示例**：

```python
# 输入
query = "推荐一些好看的"
context = ["用户问：最近有什么新电影？", "助手答：最近有《奥本海默》《芭比》等热门影片"]
user_profile = "用户喜欢科幻和悬疑类型的电影"

# 输出
rewritten_query = "推荐科幻和悬疑类型的好看电影"
```

#### 6.3.5 配置管理 (`config.py`)

| 类名             | 方法                             | 功能说明         |
| -------------- | ------------------------------ | ------------ |
| `IntentConfig` | `load(config_path)`            | 从YAML文件加载配置  |
| `IntentConfig` | `get_api_client_config()`      | 获取API客户端配置   |
| `IntentConfig` | `get_prompt_template_config()` | 获取Prompt模板配置 |

***

## 7. RAG 组件优化路线图

> 每项优化建议包含：可调节参数 → 评估指标 → 预期效果。目的是建立"假设→实验→验证"的优化闭环。

### 7.1 分片策略优化

| 优化维度  | 待探索参数                          | 评估方法             | 关键洞察                                       |
| ----- | ------------------------------ | ---------------- | ------------------------------------------ |
| 分片大小  | 128 / 256 / 500 / 1000 / 2000  | Recall\@5, 平均分片数 | 太小丢失上下文，太大引入噪音；中文500-800字符通常最优             |
| 重叠量   | 0 / 50 / 100 / chunk\_size×20% | 边界信息完整性          | 重叠可防止关键句被腰斩，但过多会冗余                         |
| 分割粒度  | 字符级 vs 句号级 vs 段落级              | 分片可读性、检索精确率      | `RecursiveCharacterTextSplitter` 优先在语义边界切分 |
| 元数据策略 | 是否保留标题、章节名、位置序号                | LLM回答引用准确率       | 元数据不参与向量化但影响生成质量——LLM需要溯源信息                |

### 7.2 嵌入模型对比

| 模型                  | 维度   | 中文效果  | 速度 | 内存      | 适用阶段      |
| ------------------- | ---- | ----- | -- | ------- | --------- |
| `bge-small-zh-v1.5` | 512  | ★★★★  | 快  | \~100MB | **V0 默认** |
| `bge-large-zh-v1.5` | 1024 | ★★★★★ | 慢  | \~1.3GB | V0调优对比    |
| `m3e-base`          | 768  | ★★★★  | 中  | \~400MB | V1备选      |
| `bge-m3`            | 1024 | ★★★★★ | 慢  | \~2GB   | V2多语言场景   |

### 7.3 检索策略优化

| 优化方向 | 方法                               | 关键参数                          | 适用场景                     |
| ---- | -------------------------------- | ----------------------------- | ------------------------ |
| 多样性  | MMR (Maximum Marginal Relevance) | `fetch_k=20, lambda_mult=0.5` | 开放型问题、需要多角度信息            |
| 精度   | 相似度阈值过滤                          | `score_threshold=0.5`         | 事实型查询，减少无关噪音             |
| 混合检索 | 向量检索 + BM25 关键词检索                | 融合权重 α                        | 实体名、专有名词查询（BM25对精确匹配更敏感） |
| 重排序  | Cross-Encoder 对 Top-N 精排         | N=20→5                        | 在 V1 中结合个性化信号            |

### 7.4 Prompt 工程优化

| 优化方向        | 可调节项                       | 评估方法                |
| ----------- | -------------------------- | ------------------- |
| 指令清晰度       | System Prompt 的角色定义、输出格式约束 | LLM-as-a-judge 评分   |
| 参考资料的呈现方式   | 编号列表 vs 段落 vs 表格           | 引用准确率（LLM 是否正确标注来源） |
| Few-shot 示例 | 提供 1-2 个理想回答样例             | 输出格式一致性             |
| 压缩检索结果      | 对检索片段二次摘要再送入 LLM           | 回答完整性 vs Token 消耗   |

### 7.5 优化实验记录模板

每次实验建议记录以下内容，形成可对比的优化日志：

```
实验ID: EXP-001
变更项: chunk_size 256 → 500
数据集: zhwiki 30,000 篇
评估查询数: 40
结果:
  Recall@5:  0.72 → 0.78 (+8.3%)
  MRR:       0.45 → 0.51 (+13.3%)
  LLM评分:   3.8 → 4.1 (1-5分)
结论: chunk_size=500 显著优于 256，建议作为新基线
```

### 7.6 优化优先级建议

| 优先级     | 优化项                           | 理由                    |
| ------- | ----------------------------- | --------------------- |
| P0（立即做） | 评估查询集 + 基线对比实验（B0/B1/B2）      | 建立评估基准线，量化每一步改进收益     |
| P1（V0内） | 混合检索（BM25 + Dense + RRF）      | 低代码量高收益，建立系统基线 B2     |
| P2（V0内） | 分片大小 + 嵌入模型对比                 | 对检索质量影响大，需与 B2 基线配合调优 |
| P3（V1）  | MMR 检索 + Prompt 优化            | 提升生成质量                |
| P4（V1）  | 重排序（Cross-Encoder / LightGBM） | 需要标注数据，V1 个性化阶段       |

***

## 8. 评估体系总结

| 评估阶段 | 指标/方法 | 数据集 | 目的 |
| --- | --- | --- | --- |
| **V0 检索评估（主力）** | Recall@k, MRR, NDCG@k, Hit Rate@k | T2Ranking dev split + 官方 qrels | 全链路检索+排序质量评估（4 级标注） |
| **V0 检索评估（辅助）** | Recall@k, MRR | MIRACL dev split + 官方 qrels | Wikipedia 检索对标 + WSDM 2023 leaderboard |
| **V0 意图识别评估** | 意图分类准确率 | T2Ranking query types | 评估意图分类器效果 |
| **V0 查询重写评估** | 检索增强率（重写前后 Recall 对比） | T2Ranking 短查询子集 | 评估查询重写对检索的增益 |
| **V0 检索评估（备选）** | 同上 | 自建查询集 + LLM Pooling 标注 | 上述数据集不适配时的补充方案 |
| **V0 生成评估** | LLM评审（相关性、流畅性、事实性）、人工体验 | DeepSeek API | 验证 RAG 端到端生成质量 |
| **V1 离线评估** | MRR@10, NDCG@10, PHR@k | T2Ranking test split | 量化个性化检索性能（避免数据泄露） |
| **基线对比** | 纯语义 vs 混合 vs 个性化 | T2Ranking 统一查询集 | 验证每一步的增量收益 |
| **LLM评审** | 相关性、个性化、流畅性、事实性 | LLM-as-a-judge | 生成质量评估 |
| **模拟在线** | 👍/👎按钮、点击日志 | Gradio 用户交互 | 用户反馈收集 |
| **AI评审团** | 多维度评分聚合 | 用户模拟代理 | 自动化评估 |

**评估数据流**：

```
T2Ranking (300K+ queries + 4级 qrels + query types) ⭐ 主力
    │
    ├─ Train → 重排序模型训练
    ├─ Dev   → 超参选择 + V0 检索评估
    ├─ Test  → 最终评估（仅使用一次）
    └─ Query Types → 意图识别评估 + 查询重写评估

MIRACL (~1,500 queries + 二元 qrels) 辅助参考
    │
    ├─ Wikipedia 检索对标
    └─ WSDM 2023 Cup leaderboard 横向对比

自建 LLM Pooling 标注 (备选)
    └─ 30-50 queries + qrels (当两套现成数据不适配时启用)
```

***

## 9. 执行建议与简历呈现

### 9.1 Git管理

- 从V0开始用Git管理，打好有意义的Tag（如 `v0-baseline`, `v1-personalized`）
- 记录每次提交，体现项目迭代过程

### 9.2 关键词植入

| 版本 | 关键词                                 |
| -- | ----------------------------------- |
| V0 | BM25、向量检索、混合检索、RRF融合、LLM调用、Gradio、T2Ranking  |
| V1 | 多路召回、LambdaMART、NDCG、LLM-as-a-judge、意图识别、查询重写 |
| V2 | 多轮对话、DPO、偏好对齐、AI评审团                 |

### 9.3 预期产出

- **V0**：可访问的Web Demo，基础搜索回复
- **V1**：完整个性化搜索系统 + 离线评估报告
- **V2**：具备多轮对话和偏好对齐的完整系统

***

## 10. 参考文献

1. Guo, Q., Chen, W., & Wan, H. (2021). AOL4PS: A Large-scale Data Set for Personalized Search. *Data Intelligence*.
2. Zheng, J. Y., et al. (2025). Can Synthetic Query Rewrites Capture User Intent Better than Humans? *arXiv*.
3. Liu, H., et al. (2021). Conversational Query Rewriting with Self-Supervised Learning. *arXiv*.
4. Chu, X., et al. (2026). Accurate and Efficient Personalized Query Rewriting in Baidu Search. *WWW*.

