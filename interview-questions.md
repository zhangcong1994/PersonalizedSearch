# 面试题库 —— PersonalizedSearch 项目

> 本文件汇总了该项目在面试中可能被问到的各类问题，
> 按主题分为 BM25/关键词检索、Dense 语义检索、RAG 系统设计、查询重写、实验评估、LLM/工程实践六大板块。

---

## 一、BM25 / 关键词检索

### Q1: BM25 的原理是什么？

BM25（Best Matching 25）是 TF-IDF 的改进版，属于概率检索模型。核心公式：

\[
\text{BM25}(q, d) = \sum_{t \in q} \text{IDF}(t) \cdot \frac{tf(t, d) \cdot (k_1 + 1)}{tf(t, d) + k_1 \cdot (1 - b + b \cdot \frac{|d|}{\text{avgdl}})}
\]

三个关键改进相比 TF-IDF：

- **TF 饱和**：`k_1`（通常取 1.2~2.0）让词频增长非线性饱和，避免一个词出现 100 次得分就是出现 1 次的 100 倍。
- **文档长度归一化**：`b`（通常取 0.75）控制长度惩罚强度。长文档天然含有更多单词，`b` 参数抑制这种偏差。
- **IDF**：\(\text{IDF}(t) = \ln\left(\frac{N - df(t) + 0.5}{df(t) + 0.5} + 1\right)\)，比经典 IDF 更平滑，避免稀有词权重爆炸。

**本项目实现**：使用 `rank_bm25.BM25Okapi` + jieba 中文分词 + 停用词过滤构建 BM25 索引。

> 面试追问：如果面试官让你在白板上手写 BM25 伪代码，你可以从 "倒排索引查找 → 每条文档累加 BM25 分 → 堆排序取 Top-K" 的流程来描述。

---

### Q2: BM25 相比 Dense Retrieval（向量检索）的优劣势？

| 维度 | BM25 | Dense Retrieval |
|------|------|-----------------|
| **匹配方式** | 精确词袋匹配（term-level） | 语义向量相似度（semantic-level） |
| **同义词/改写** | 不敏感，需要查询重写弥补 | 天然支持语义泛化 |
| **稀有词/专有名词** | 精确匹配极强 | 可能被 OOV 或欠训练影响 |
| **长尾查询** | 依赖 term overlap | 依赖嵌入模型泛化能力 |
| **索引速度** | 极快（只需分词+倒排） | 慢（需 GPU 推理嵌入，本项目 230 万 passages 约 23 小时 CPU） |
| **存储成本** | 倒排索引 + 原始文本 | 向量存储（768 维 × 230 万 ≈ 7GB 浮点） |
| **查询延迟** | ~35ms（本项目实测） | ~14ms（本项目实测，ChromaDB HNSW） |
| **可解释性** | 强（可精确指出命中哪些 term） | 弱（"相似度 0.87" 难以解释） |

**本项目关键发现**：在 T2Ranking 全量 230 万池上，BM25 的 Recall@10 接近 0（因为中文互联网段落同义词/改写极丰富），Dense 能达到 0.45。但 BM25 对改写友好——每一次 Prompt 工程注入的新 term 都是独立的正向匹配信号，不存在 Dense 模型的"向量偏移"问题。

---

### Q3: BM25 可以分布式构建索引吗？

可以，三种方式：

- **文档分片（Document Partitioning）**：将语料库按 pid 哈希分到不同机器，每台机器各自构建完整的 BM25 索引（含自己的 IDF 统计）。查询时广播到所有分片，合并结果。**优点**：实现简单，每个分片独立完整。**缺点**：IDF 是局部的，跨分片不一致。
- **全局统计 + 局部倒排（Global IDF + Local Posting）**：先做一轮 MapReduce 统计全局 DF/IDF，然后各节点各自构建倒排列表 + 用全局 IDF 算分。本项目用 `rank_bm25` 内置的 `corpus_size` 和全局 DF 即可做到。
- **基于 Elasticsearch**：ES 内置 BM25（5.x 后默认相似度算法），自动分片 + 分布式 IDF + 协调节点合并。这是工业界的工程首选。

**本项目当前状态**：单机 pickle 序列化（`build_bm25_index.py`），但架构上已经封装了 `bm25_store.build()` 和 `bm25_store.load()` 接口，后续可替换为分布式后端。

---

### Q4: BM25 可以构建分布式服务吗？

可以，典型方案：

- **轻量级**：FastAPI + 单机加载 pickle 索引（本项目当前可做到），QPS 数百级别，适合评估/V0 演示。
- **中等规模**：Elasticsearch 集群（自动分片 + 副本 + REST API），开箱即用的 BM25 + 分布式检索。
- **大规模**：基于 Lucene/ES 的 BM25 检索服务 + Redis 缓存热门查询结果，QPS 万级别。

**本项目当前**：评估脚本中 BM25 和 Dense 均在同一进程内调用，未独立部署为服务。

---

### Q5: 有哪些文本预处理方法？本项目用了哪些？

**通用文本预处理 pipeline**：

| 步骤 | 方法 | 目的 |
|------|------|------|
| HTML 清洗 | 正则 `<[^>]*>` + `html.unescape` | 去除标签（本项目 63.4% 段落含 HTML） |
| URL 去除 | 正则 `https?://\S+` | 去除链接噪声 |
| 控制字符清理 | 正则 `[\x00-\x1f\x7f]` | 去除不可见字符 |
| 换行/空白规范化 | `replace("\n"," ")` + `\s+ → " "` | 统一空格 |
| 中文分词 | jieba 精确模式 (`lcut`) | 将连续中文文本切分为词序列 |
| 停用词过滤 | jieba 内置停用词表 + 自定义补充 | 去除 "的/了/是/在/我/有/和/就/不/..." |
| 短词过滤 | `len(w) > 1` | 过滤单字（通常信息量低） |
| 长度截断 | 超过 2000 字符截断 | 避免单 passage 过长（本项目中 4.1% 段落受影响） |
| 最小长度过滤 | 短于 10 字符丢弃 | 去除空/噪音段落 |

**本项目代码位置**：
- HTML 清洗 + 文本规范化：`src/evaluation/data_loader.py` 中的 `clean_text()`
- 中文分词 + 停用词：`src/retrieval/bm25_store.py` 中的 `_tokenize()`
- 文档分片：`src/indexing/chunker.py` 中的 `DocumentChunker`（Wikipedia 场景用，T2Ranking 部分片）

**可能追问**：
- 为什么用 jieba 而不是其他分词器？（轻量、中文社区最成熟、本项目 BM25 场景不需要 subword tokenization）
- 停用词过滤在 BM25 中的作用？（IDF 已经压低高频词权重，但显式过滤减少索引大小和计算量）
- 为什么长度截断取 2000？（T2Ranking passage 中位数 368 字符，截断只影响尾部 4.1% 长段落，取舍平衡）

---

## 二、Dense Retrieval / 语义检索

### Q6: Dense Retrieval（双塔模型）的原理是什么？和 Cross-Encoder 的区别？

**双塔模型（Bi-Encoder）**：
- Query 和 Document 分别过独立的编码器（通常是共享参数的同一个模型），各自产出向量。
- 检索时用近似最近邻（ANN）索引（如 HNSW/IVF）做向量相似度搜索。
- **优点**：文档向量可预先计算并索引，检索极快。
- **缺点**：Query-Document 交互只在点积/余弦这一步，信息交互晚且弱。

**Cross-Encoder**：
- Query 和 Document 拼接后一起输入模型（如 BERT），做全交互注意力。
- **优点**：精度远高于 Bi-Encoder。
- **缺点**：每对 (query, doc) 都要过一遍完整模型，无法预先索引，只能用于重排序（Reranker）。

**本项目实践**：V0 阶段使用 Bi-Encoder（bge-small-zh-v1.5/m3e-base），V1 阶段计划引入 Cross-Encoder 做重排序。

---

### Q7: 为什么选择 ChromaDB？和 FAISS/Milvus 的对比？

| 维度 | ChromaDB | FAISS | Milvus |
|------|----------|-------|--------|
| 定位 | 轻量嵌入式向量库 | Meta 开源向量检索库 | 云原生分布式向量数据库 |
| 部署复杂度 | `pip install` 即可 | 需自行封装服务 | 需 Docker/K8s 部署 |
| 索引算法 | HNSW（默认） | HNSW/IVF/PQ/OPQ 等全系列 | HNSW/IVF/DiskANN 等 |
| 持久化 | 内置（SQLite + Parquet） | 需自行实现 | 内置（对象存储+元数据） |
| 过滤/元数据 | 支持 metadata filter | 需自行实现 | 支持标量过滤 |
| 适用阶段 | V0 开发/原型 | 单机高性能 | 生产集群 |

**本项目选择理由**：V0 阶段优先简单、Python 原生、开箱即用。ChromaDB 的 `PersistentClient` 自动处理持久化，减少工程负担。V1 阶段若遇到性能瓶颈可迁移至 Milvus。

---

### Q8: 嵌入模型选型的考量因素？

- **中文支持**：bge 系列和 m3e 都是专门的中文模型。
- **维度**：512 (bge-small) vs 768 (m3e) vs 1024 (bge-large)，更高维度表达力更强但存储和检索成本也更高。
- **训练数据污染**：本项目的关键教训——bge 系列的 C-MTP labeled 训练数据包含了 T2Ranking 的 query-passage 对，导致模型在评估集上被"预训练过"，改写后的 OOD 查询无法公平评估。**面试中被问到"如何检测和避免训练数据泄露"时，这是一个很好的实例。**
- **速度/成本权衡**：bge-small ~100MB 最快，bge-large ~1.3GB 效果最好但慢 3 倍。

**本项目实验结果**：bge-small Recall@10=0.455 > m3e Recall@10=0.389（差距 14%），m3e 在中文互联网段落上显著弱于 bge-small。

---

### Q9: Hybrid Search 融合策略有哪些？RRF 的原理？

**常见融合策略**：

| 策略 | 原理 | 优点 | 缺点 |
|------|------|------|------|
| **RRF** | 按倒数排名加权：score = Σ 1/(k+rank) | 无需分数归一化，参数少 | 丢弃了原始分数信息 |
| **加权分数融合** | 各分数归一化后加权求和 | 保留原始分数信息 | 需要调 α，对分数分布敏感 |
| **Round-robin** | 依次从各路取结果 | 极简，无参数 | 完全无视分数 |
| **学习融合** | 用 LambdaMART/XGBoost 学各路特征组合 | 效果最优 | 需要训练数据 |

**RRF 公式**（本项目 `src/retrieval/multi_query.py` 实现）：

\[
\text{RRF}(d) = \sum_{r \in R} \frac{1}{k + \text{rank}_r(d)}
\]

其中 `k=60` 是平滑常数（默认值来自 Cormack et al., SIGIR 2009）。

**本项目使用场景**：
- E2b Multi-Query 融合：多路子查询各自检索后 RRF 合并
- E2c HyDE+Query 融合：原查询检索 + 假答案检索两路 RRF

---

## 三、RAG 系统架构与设计

### Q10: 描述你的 RAG 系统整体架构

```
[用户输入] 
    → 意图理解/查询重写（LLM Prompt 工程 / 关键词分类）
    → 搜索排序（BM25 + Dense 双路召回 → RRF 融合 → Top-K）
    → LLM 生成（DeepSeek API + typed prompt templates + 检索结果上下文）
    → [最终回复]
```

**Pipeline 特点**：
- 渐进式基线递进：BM25 (B0) → Dense (B1) → Hybrid (B2) → Personalized (B3)
- 查询重写后置（检索之前），而非前置修改 query
- 评估与生成分离：检索评估独立于 LLM 生成，节省 API 费用

---

### Q11: 如何处理长文档？分片策略如何选？

**分片参数权衡**：

| chunk_size | 优点 | 缺点 |
|------------|------|------|
| 小（128-256） | 检索更精确 | 丢失上下文，Recall 降低 |
| 中（500） | 上下文-精度平衡 | 一条 chunk 可能不完整 |
| 大（1000+） | 上下文完整 | 噪音多，精确率降低 |

**本项目两种策略**：
- **Wikipedia 场景**：`chunk_size=500, overlap=50`，固定长度分片 + 语义断句。
- **T2Ranking 场景**：不分片，passage 即检索单元。原因：qrels 以 passage 粒度标注，分片会破坏 qid→pid 映射。

---

### Q12: 为什么不直接用 LangChain 默认的 RAG chain？

LangChain 的 `RetrievalQA` chain 做了很多默认假设（如默认 prompt 模板、默认文档拼接方式），不适合本项目特点：

- 本项目有 **typed prompt**（factual/concept/comparison/open_discussion 四种 System Prompt + few-shot 示例）。
- 本项目评估指标是 **纯检索层面的 Recall@k/MRR/NDCG**，不需要跑完生成再评估。
- 检索器需要同时支持 BM25 和 Dense 双路 + RRF 融合，LangChain 的 `EnsembleRetriever` 对此支持有限。

**所以本项目只用 LangChain 的 `BaseLanguageModel` 接口和 `PromptTemplate`，检索和评估逻辑全部自行实现。**

---

## 四、查询重写 / Prompt 工程

### Q13: 为什么需要查询重写？不重写直接检索有什么问题？

**核心问题**：短查询（5-15 字）与长 passage（中位数 368 字）之间的 **语义鸿沟**。

- 用户用口语词（"摁错"），passage 用正式术语（"输入错误"）→ **术语不匹配**。
- 查询省略了隐含上下文（"手机银行转账限额怎么改" 缺少银行名称/场景）→ **缺少隐含上下文**。
- 单条查询只锚定一个语义方向 → **语义覆盖不足**。

**本项目量化证据**：bge-small 全量池 Recall@10 仅 0.455，超过一半的相关 passage 第一轮找不出来。

---

### Q14: 你的 Prompt 工程做了哪些消融实验？

详见 `experiments/exp-002-query-rewriting.yaml`，核心消融维度：

| 维度 | 变体 | 假设 |
|------|------|------|
| 规则密度 | B1（4 条规则）→ P1（8-10 条 + 硬约束） | 更细的规则减少 LLM 偏离 |
| 示例来源 | B1（通用 7 例）→ P2（T2Ranking 领域 10 例）→ P3（对比好/坏） | 领域 few-shot > 通用；对比教学学到边界 |
| 输出格式 | B1-P4（自然语句）→ P5（关键词串） | 关键词串密度高但 BM25 IDF 也压低通用词 |
| 类型感知 | P4（按 query type 分发不同 prompt） | 统一 vs 差异化策略 |

**面试中可能的追问**："消融实验为什么要这样设计？" → 回答：单变量控制，每次只改变一个维度，确保能归因增益来源。

---

### Q15: Multi-Query RRF 融合是什么？为什么比单条查询好？

**做法**：1 条用户查询 → LLM 生成 N 条子查询 → N 路分别检索 → RRF 合并。

**三条子查询的设计逻辑**（E2b-M1）：
- 原查询（保留精确 term）
- 术语规范化版（口语→正式术语）
- 上下文补全版（补充隐含场景）

**为什么更好**：单条查询的向量只能锚定一个语义方向。多条查询覆盖不同侧面和抽象层次，RRF 合并后能互补覆盖 passage 的不同区域。

**本项目还引入了 Step-Back（Google DeepMind 2023）**：将具体查询抽象化为高层问题（"XX 是哪年成立" → "XX 的历史背景和创立过程"），进一步扩大召回范围。

---

### Q16: HyDE（Hypothetical Document Embedding）的原理？为什么 BM25 也能用 HyDE？

**HyDE 原始论文（Gao et al., 2022）**：
Query → LLM 生成假答案 → 用假答案的向量做 Dense 检索。
核心假设：假答案的向量比短查询的向量更接近真实相关文档的向量分布。

**BM25 版 HyDE（本项目 E2c）**：
Query → LLM 生成 100-150 字假答案 → jieba 分词 → BM25 term 匹配。
原理不同但理念相同：假答案的术语丰富度和信息密度远高于短查询，预期能通过词袋匹配命中更多相关 passage。

**Dense vs BM25 HyDE 的关键区别**：
- Dense：靠假答案的语义向量匹配。
- BM25：靠假答案的 term 直接命中。没有向量偏移问题，每个 term 都是独立的正向匹配信号。

---

### Q17: PRF（伪相关反馈）的原理？

**完整流程**（`src/retrieval/prf.py`）：

```
Step 1: 原始查询 → 第一轮检索 → Top-K 结果（假设 top-20 相关）
Step 2: 对 top-20 结果做 TF-IDF 词提取 → 选出 N 个最高权重的扩展词
Step 3: 原始查询 + 扩展词 → 第二轮检索 → 最终 Top-K 结果
```

**关键实现细节**：
- 扩展词选择：过滤已在原查询中的词，按 `df × idf`（文档频率 × 逆文档频率）排序。
- 可选加权模式：`weighted=True` 时用完整 TF-IDF 分数（而非仅计数）。
- **PRF 最大优势**：不需要 LLM API 调用，完全本地计算！成本为零。

---

## 五、实验评估

### Q18: 你的评估体系是怎么设计的？

**四层递进策略**：

| 阶段 | 抽样量 | 目的 |
|------|--------|------|
| Smoke Test | 100 条 | 验证全链路无报错 |
| Ablation | 2000 条 | 对比实验组差异（多方案排序） |
| Full Report | 22812 条 | 产出论文级最终数字 |
| Stratified | 全量 | 按 query type 分层分析 |

**核心指标**：
- **Recall@k**：召回能力（"找到了吗？"）
- **MRR**：排序质量（"排在前面了吗？"）
- **NDCG@k**：考虑标注等级的排序质量
- **Precision@k**：检索效率（"返回的有多少相关？"）

---

### Q19: 为什么选 T2Ranking 而不是 MIRACL 或自建评估集？

| 维度 | T2Ranking | MIRACL |
|------|-----------|--------|
| 查询数量 | 30 万+ | ~1,500 |
| 标注粒度 | 4 级（0~3） | 二元（相关/不相关） |
| 查询来源 | 搜狗真实搜索日志 | 人工构造 |
| Query Types | ✅ 附带意图标签 | ❌ |
| 统计可靠性 | 极高 | 一般 |
| 学术权威性 | SIGIR 2023 | TACL 2023 + WSDM Cup |

---

### Q20: 训练数据污染（Data Contamination）在你这项目中怎么体现的？

**关键决策**：从 exp-002 中排除了 BGE 系列模型。

**原因**：BGE 的 C-MTP labeled 训练数据包含 T2Ranking 的 query-passage 对。这意味着 BGE 模型在训练时已经"见过"T2Ranking 评估集的 query 分布。如果用 BGE 做 Dense 检索，检索指标会被人为抬高。更严重的是——LLM 改写后的 query 对 BGE 来说是分布外（OOD）输入，改写增益无法公正评估。

**面试亮点**：这展示了你能识别并处理数据泄露问题的能力。面试官可能追问"在你的项目中怎么发现和排除的"，答案是：查阅了 BGE 训练数据来源（C-MTP labeled），发现其构建过程中使用了 T2Ranking，确认了污染后切换为 m3e-base。

---

### Q21: 你的实验设计中有哪些"科学方法"的体现？

- **单变量消融（Ablation）**：每次只改变一个因素（如 E2a：规则密度/示例来源/输出格式/类型感知各一个消融）。
- **对照组（Control）**：E2a-B0（no rewrite）是所有改写实验的统一对照组。
- **干净的变量隔离**：E2b-M2 vs M1 唯一变量是多了 1 条 step-back 查询，其他完全相同。
- **效应量量化**：不仅看"好不好"，还量化"好多少"（Recall 差距、MRR 差距）。
- **成本预估**：每组分 API 成本预先计算，避免实验跑一半预算不够。

---

## 六、LLM 集成 / 工程实践

### Q22: 为什么 V0 阶段选择 DeepSeek API 而不是本地部署模型？

- **成本极低**：单条查询重写约 ¥0.00036，全量 22812 条仅 ¥30 左右。
- **迭代速度快**：Prompt 调优只需改文本，无需等 GPU 推理。
- **V0 阶段重点**：验证 Prompt 工程/HyDE/PRF 的上限，而非模型微调。
- **V1 计划**：确认最优策略后，再部署本地 Qwen2.5-7B-Instruct 做查询重写。

---

### Q23: 多进程分词为什么要特殊处理？

**`bm25_store.py` 中的多进程分词设计**：

```python
def _init_worker(stopwords_set):
    global _worker_stopwords
    _worker_stopwords = stopwords_set
    jieba.lcut("")  # 在每个 worker 中初始化 jieba

def _tokenize_worker(text):
    global _worker_stopwords
    tokens = jieba.lcut(text)
    # ...
```

**设计原因**：
- 每个子进程需要独立的 jieba 实例（jieba 内部有 C 扩展和全局状态，不能跨进程共享）。
- 停用词集合通过 `initializer` + `initargs` 传入（比每次序列化传递更高效）。
- 限制 worker 上限为 8（`max_safe_jobs`），防止多进程同时加载 jieba 导致内存压力。
- `maxtasksperchild` 防止长时间运行后内存泄漏。

---

### Q24: 分词后为什么过滤 `len(w) <= 1` 的 token？

- 中文单字词（"的"、"是"、"在"等）单独看信息量极低。
- BM25 的 IDF 虽然会压低它们，但显式过滤能显著减少索引大小和检索计算量。
- Jieba 分词后仍会产生大量单字 token（标点、语气词等）。

---

### Q25: 断点续跑（Resume）是如何实现的？

**T2Ranking 索引构建**（`build_t2ranking_index.py`）：

`state.json` 记录：
```json
{
  "last_processed_line": 150000,
  "total_stored": 149873,
  "batches_completed": 150
}
```

- 每批次完成后更新 state.json。
- 重新运行时自动从 `last_processed_line` 续跑。
- `--rebuild` 参数清除旧索引和状态文件从头开始。
- 配合 `build_log.jsonl` 做诊断回溯。

---

### Q26: 为什么 T2Ranking 评估中不分片（chunk）？

- qrels 标注粒度是 passage 级别（qid→pid 映射）。
- 分片后一个 passage 变成多个 chunk，无法直接对应 qrels 中的 pid。
- 必然需要做 chunk→pid 的回溯映射，增加了工程复杂度。
- T2Ranking passage 中位数 368 字符，自身长度已接近常见的 chunk_size（500），分片收益有限。

---

### Q27: 这个项目中你遇到的最大的技术难题是什么？怎么解决的？

**开放式问题，候选回答方向**：

1. **训练数据污染**：发现 BGE 模型已"见过" T2Ranking 数据，果断排除，切换 m3e-base。（展示科研诚信）
2. **BM25 在中文段落完全失效**：Recall@10 接近 0 → 深入分析发现中文互联网段落术语密度高、同义词丰富，BM25 词袋匹配天然不足 → 转而重点投入查询重写弥补 gap。（展示分析能力）
3. **多进程 jieba 死锁/内存泄漏**：通过 `maxtasksperchild` + worker 上限 + `jieba.lcut("")` 初始化解决。（展示工程能力）

---

## 七、开放性问题（面试官可能从简历发散问）

### Q28: 如果有无限资源，你会怎么改进这个系统？

- Dense 检索模型升级为 ColBERT 或 late-interaction 模型（效果接近 Cross-Encoder 但可索引）。
- 部署完整的 Elasticsearch + Milvus 混合检索服务，支持分布式。
- 引入真正的个性化模块：基于用户历史行为（AOL4PS 数据集）训练双塔模型。
- 重排序从 RRF 升级为 LambdaMART/LightGBM，学习多路召回特征的融合权重。
- 本地部署 7B 模型做查询重写和 LLM 生成，减少 API 依赖。

---

### Q29: 你这个系统如果要上线，最大的风险是什么？

- **查询重写的 LLM 幻觉**：生成不存在的术语、偏离原意的改写 → 需要人工评估改写质量（Semantic Similarity、BLEU 等）。
- **BM25 检索的冷启动**：如果 T2Ranking 语料换成完全不同领域的数据，BM25 的分词和停用词可能需要重新调优。
- **chromadb 的持久化不是为生产设计的**：高并发下性能会急剧下降，需要迁移到 Milvus 或自建 FAISS 服务。
- **DeepSeek API 依赖**：API 不稳定或涨价会直接影响整个 pipeline。

---

### Q30: 你怎么看待 LLM-as-a-Judge 做评估？项目中有用到吗？

**本项目未使用 LLM-as-a-Judge 做检索评估**（用 T2Ranking 人工标注的 qrels 更可靠）。但在 proposal.md 中规划了 Pooling 标注方案作为备选：

> 对 Pool 内每篇文档，调用 DeepSeek API 评估相关性 → 输出 3(直接相关)/2(部分相关)/1(弱相关)/0(不相关)

LLM-as-a-Judge 的优势是灵活和低成本，劣势是"弱相关 vs 不相关"边界判断不如人工精确，本项目规划中建议降级为二元相关。

---

## 八、"被问到概率极高"的 Top 5 题

如果你只想准备 5 道题，以下是面试官最可能从你简历上这个项目顺藤摸瓜问到的：

1. **BM25 原理 + 为什么不直接用 BM25**（引到 Dense/Hybrid 的必要性）
2. **RAG 系统整体架构图**（考察系统设计能力）
3. **查询重写怎么做，效果如何验证**（考察 Prompt 工程和实验设计）
4. **T2Ranking 是什么，为什么选它**（考察数据 sense 和评估意识）
5. **训练数据污染是怎么发现和处理的**（考察科研严谨性，非常加分）
