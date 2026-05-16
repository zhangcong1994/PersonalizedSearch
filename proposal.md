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

---

## 2. 数据集选择

### 2.1 主数据集：维基百科（RAG基础流程）

| 属性 | 说明 |
| --- | --- |
| **来源** | Wikipedia dump 或 Hugging Face 预构建数据集 |
| **推荐数据集** | `wikipedia` (Hugging Face Datasets) - 英文维基百科 |
| **替代选择** | `Cohere/wikipedia-22-12-en-embeddings` (已预嵌入) |
| **规模** | 约600万篇文章，可按需采样（如取前10万篇） |
| **存储大小** | 原始dump约20GB，采样子集约1-2GB |
| **核心字段** | `id`、`url`、`title`、`text`（完整文章内容） |
| **获取方式** | Hugging Face Datasets: `load_dataset("wikipedia", "20220301.en")` |
| **适用模块** | 文档分片、向量化、向量检索（V0阶段核心） |

**数据集使用策略**：
- **V0阶段**：使用维基百科跑通RAG完整流程，重点练习文档切片、嵌入生成、向量检索
- **V1+阶段**：可引入AOL4PS等用户行为数据集扩展个性化功能

### 2.2 意图识别/查询重写数据集

| 数据集 | 用途 | 特点 | 获取方式 |
| --- | --- | --- | --- |
| **TopiOCQA** | 对话式查询重写 | 开放域对话问答，含多轮对话历史 | Hugging Face/Download |
| **Restoration-200K** | 会话查询重写 | 手动标注的CQR数据集 | 公开下载 |
| **RECAP** | 意图重写评估 | 针对Agent规划的意图理解基准 | [OpenReview](https://openreview.net/forum?id=UelTYgX3YN) |
| **SynRewrite** | 合成查询重写 | GPT-4o生成的高质量重写样本 | 论文附带数据 |

### 2.3 LLM生成微调数据集

| 数据集 | 用途 | 特点 | 获取方式 |
| --- | --- | --- | --- |
| **Alpaca** | 指令微调 | 52K高质量指令-回答对 | GitHub |
| **Dolly** | 指令微调 | 15K指令-回答对 | Databricks |
| **UltraChat** | 多轮对话 | 140万多轮对话样本 | Hugging Face |
| **Self-Instruct** | 指令生成 | 自动生成多样化指令 | GitHub |

---

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

---

## 4. 技术选型与环境

| 分类 | 技术 | 说明 |
| --- | --- | --- |
| **开发语言** | Python 3.10+ | 主流AI开发语言 |
| **深度学习框架** | PyTorch 2.x, Transformers (HuggingFace), TRL | 支持LLM微调与DPO |
| **搜索与检索** | LangChain / LlamaIndex | 切片、检索链 |
| **向量数据库** | ChromaDB / FAISS | 高效向量检索 |
| **排序模型** | LightGBM / LambdaMART | 学习排序算法 |
| **LLM集成** | OpenAI API / DeepSeek API / Qwen-7B-Chat (vLLM) | **混合方案**：开发阶段用API（效率高），最终部署用本地模型 |
| **前端演示** | Gradio | 快速构建Web界面 |
| **评估工具** | pytrec_eval, scikit-learn, LLM-as-a-judge | 多维度评估 |
| **环境** | AutoDL GPU（RTX 4090 24G） | 按需租用 |

---

## 5. 版本迭代路线图

### V0 – 基础RAG流程版 (目标：1周)

**核心目标**：快速跑通"文档分片→向量化→向量检索→LLM生成"的完整RAG链路，重点学习核心技术栈。

| 任务 | 描述 |
| --- | --- |
| 环境搭建 | 安装依赖，初始化 LangChain/LlamaIndex 项目骨架 |
| 数据获取 | 从Hugging Face加载维基百科数据集，采样适量文档（如1万-10万篇） |
| 文档索引 | 将维基百科文档切片（chunk size=256, overlap=20），生成嵌入存入向量数据库 |
| 基础检索 | 实现向量相似度检索，返回Top-10文档片段 |
| LLM生成 | 调用API或本地模型，设计Prompt模板基于检索结果生成回答 |
| Gradio界面 | 输入框输出框，展示检索片段和AI回复 |

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
| 评估方式 | 指标/方法 | 目的 |
| --- | --- | --- |
| 离线指标 | MRR@10, NDCG@10, **Personalized Hit Rate@k** | 量化检索效果 |
| 基线对比 | 纯语义搜索 vs 热门推荐 vs 个性化系统 | 验证个性化收益 |
| LLM-as-a-judge | 相关性、个性化、流畅性、事实一致性（1-5分） | 生成质量评估 |
| 模拟在线 | Gradio界面加入👍/👎按钮，收集用户反馈 | 真实用户验证 |

**个性化增益指标定义（Personalized Gain@k）**：
- 计算公式：`PG@k = (个性化系统HR@k - 纯语义系统HR@k) / 纯语义系统HR@k × 100%`
- 含义：衡量个性化系统相比纯语义搜索在用户偏好文档上的提升比例
- HR@k：用户高点击/偏好文档在top-K结果中的命中率

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

---

## 6. 项目文件结构

```
personalized-ai-search/
├── data/                    # 数据集与预处理脚本
│   ├── raw/                 # 原始数据集
│   │   ├── wikipedia/       # 维基百科数据集（V0阶段）
│   │   ├── aol4ps/          # AOL4PS数据集（V1+阶段预留）
│   │   ├── query_rewrite/   # 查询重写数据集
│   │   │   ├── topiocqa/    # TopiOCQA数据集
│   │   │   └── restoration/ # Restoration-200K数据集
│   │   └── llm_finetune/    # LLM微调数据集
│   └── processed/           # 预处理后的数据
│       ├── documents/       # 处理后的文档
│       ├── queries/         # 处理后的查询
│       └── user_profiles/   # 用户画像数据
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

---

### 6.1 数据处理模块 - 类设计

#### 6.1.1 数据集下载器 (`downloader.py`)

| 类名 | 方法 | 输入参数 | 输出 | 功能说明 |
| --- | --- | --- | --- | --- |
| `WikipediaDownloader` | `__init__(save_dir)` | save_dir: str | - | 初始化下载器，指定保存目录 |
| `WikipediaDownloader` | `download(num_samples=None)` | num_samples: Optional[int] | Dataset | 从Hugging Face加载维基百科数据集，可采样 |
| `WikipediaDownloader` | `save_to_disk(dataset, output_path)` | dataset: Dataset, output_path: str | bool | 保存数据集到磁盘 |
| `WikipediaDownloader` | `load_from_disk(input_path)` | input_path: str | Dataset | 从磁盘加载数据集 |

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

| 类名 | 方法 | 输入参数 | 输出 | 功能说明 |
| --- | --- | --- | --- | --- |
| `DataProcessor` | `__init__(config)` | config: dict | - | 初始化处理器 |
| `DataProcessor` | `load_raw_data(file_path)` | file_path: str | pd.DataFrame | 加载原始数据 |
| `DataProcessor` | `clean_text(text)` | text: str | str | 清洗文本（去HTML、特殊字符） |
| `DataProcessor` | `normalize_text(text)` | text: str | str | 文本标准化（大小写、数字处理） |
| `DataProcessor` | `filter_short_documents(min_length)` | min_length: int | pd.DataFrame | 过滤过短文档 |
| `DataProcessor` | `remove_duplicates()` | - | pd.DataFrame | 去重 |
| `DataProcessor` | `save_processed_data(data, output_path)` | data: pd.DataFrame, output_path: str | bool | 保存处理后数据 |

#### 6.1.3 数据探索性分析 (`analyzer.py`)

| 类名 | 方法 | 输入参数 | 输出 | 功能说明 |
| --- | --- | --- | --- | --- |
| `DataAnalyzer` | `__init__(data)` | data: pd.DataFrame | - | 初始化分析器 |
| `DataAnalyzer` | `get_basic_stats()` | - | dict | 基本统计（文档数、用户数、查询数） |
| `DataAnalyzer` | `analyze_document_length()` | - | dict | 文档长度分布 |
| `DataAnalyzer` | `analyze_query_distribution()` | - | dict | 查询频率分布 |
| `DataAnalyzer` | `analyze_user_behavior()` | - | dict | 用户行为分析（点击次数、会话长度） |
| `DataAnalyzer` | `generate_report(output_path)` | output_path: str | - | 生成分析报告 |

#### 6.1.4 数据读写工具 (`io.py`)

| 类名 | 方法 | 输入参数 | 输出 | 功能说明 |
| --- | --- | --- | --- | --- |
| `DataIO` | `read_csv(file_path)` | file_path: str | pd.DataFrame | 读取CSV文件 |
| `DataIO` | `write_csv(data, file_path)` | data: pd.DataFrame, file_path: str | bool | 写入CSV文件 |
| `DataIO` | `read_json(file_path)` | file_path: str | dict/list | 读取JSON文件 |
| `DataIO` | `write_json(data, file_path)` | data: dict/list, file_path: str | bool | 写入JSON文件 |
| `DataIO` | `read_parquet(file_path)` | file_path: str | pd.DataFrame | 读取Parquet文件 |
| `DataIO` | `write_parquet(data, file_path)` | data: pd.DataFrame, file_path: str | bool | 写入Parquet文件 |

---

### 6.2 索引模块 - 类设计

#### 6.2.1 文档切片器 (`chunker.py`)

| 类名 | 方法 | 输入参数 | 输出 | 功能说明 |
| --- | --- | --- | --- | --- |
| `DocumentChunker` | `__init__(chunk_size=256, overlap=20)` | chunk_size: int, overlap: int | - | 初始化切片器 |
| `DocumentChunker` | `chunk_document(text)` | text: str | List[str] | 切分单篇文档 |
| `DocumentChunker` | `chunk_documents(documents)` | documents: List[str] | List[List[str]] | 批量切分文档 |
| `DocumentChunker` | `get_chunk_stats(chunks)` | chunks: List[str] | dict | 统计切片长度分布 |

#### 6.2.2 嵌入模型封装 (`embedder.py`)

| 类名 | 方法 | 输入参数 | 输出 | 功能说明 |
| --- | --- | --- | --- | --- |
| `Embedder` | `__init__(model_name="all-MiniLM-L6-v2")` | model_name: str | - | 初始化嵌入模型 |
| `Embedder` | `encode(texts)` | texts: List[str] | np.ndarray | 生成文本嵌入 |
| `Embedder` | `encode_single(text)` | text: str | np.ndarray | 生成单文本嵌入 |
| `Embedder` | `get_embedding_dimension()` | - | int | 获取嵌入维度 |
| `Embedder` | `save_model(path)` | path: str | - | 保存模型 |
| `Embedder` | `load_model(path)` | path: str | - | 加载模型 |

**支持的嵌入模型**：
- sentence-transformers: `all-MiniLM-L6-v2`, `all-mpnet-base-v2`, `all-distilroberta-v1`
- 预留扩展: OpenAI embeddings, BGE models

#### 6.2.3 向量数据库操作 (`vector_db.py`)

| 类名 | 方法 | 输入参数 | 输出 | 功能说明 |
| --- | --- | --- | --- | --- |
| `VectorDB` | `__init__(db_path, embedding_dim)` | db_path: str, embedding_dim: int | - | 初始化向量数据库 |
| `VectorDB` | `add_documents(chunks, embeddings, metadata)` | chunks: List[str], embeddings: np.ndarray, metadata: List[dict] | bool | 添加文档 |
| `VectorDB` | `search(query_embedding, top_k=10)` | query_embedding: np.ndarray, top_k: int | List[dict] | 向量检索 |
| `VectorDB` | `delete_document(doc_id)` | doc_id: str | bool | 删除文档 |
| `VectorDB` | `update_document(doc_id, chunk, embedding)` | doc_id: str, chunk: str, embedding: np.ndarray | bool | 更新文档 |
| `VectorDB` | `persist()` | - | bool | 持久化到磁盘 |

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

---

### 6.3 意图识别/查询重写模块 - 类设计

#### 6.3.1 基础抽象类 (`base.py`)

| 类名 | 方法 | 功能说明 |
| --- | --- | --- |
| `BaseLLMClient` | `generate(prompt: str) -> str` | 抽象方法，子类实现LLM调用 |
| `BaseQueryRewriter` | `rewrite(query: str, context: Optional[List[str]] = None) -> str` | 抽象方法，子类实现查询重写 |
| `BasePromptTemplate` | `format(**kwargs) -> str` | 抽象方法，子类实现Prompt格式化 |

#### 6.3.2 API客户端 (`api_client.py`)

| 类名 | 方法 | 输入参数 | 输出 |
| --- | --- | --- | --- |
| `OpenAIClient` | `generate(prompt, model="gpt-3.5-turbo")` | prompt: str, model: str | response: str |
| `DeepSeekClient` | `generate(prompt, model="deepseek-chat")` | prompt: str, model: str | response: str |
| `APIClientFactory` | `create(client_type: str) -> BaseLLMClient` | client_type: str ("openai"/"deepseek") | BaseLLMClient实例 |

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

| 类名 | 方法 | 功能说明 |
| --- | --- | --- |
| `PromptManager` | `register_template(name, template)` | 注册Prompt模板 |
| `PromptManager` | `get_template(name)` | 获取指定模板 |
| `PromptManager` | `format_template(name, **kwargs)` | 格式化模板 |

**内置Prompt模板示例**：

| 模板名称 | 用途 | 模板内容 |
| --- | --- | --- |
| `query_rewrite_basic` | 基础查询重写 | "请将用户查询改写成更明确的搜索词：{query}" |
| `query_rewrite_context` | 带上下文查询重写 | "基于对话历史：{context}，将当前查询重写为独立搜索词：{query}" |
| `query_rewrite_personalized` | 个性化查询重写 | "用户偏好：{user_profile}\n查询：{query}\n请重写为更精准的搜索词" |

#### 6.3.4 查询重写器 (`query_rewriter.py`)

| 类名 | 方法 | 输入参数 | 输出 |
| --- | --- | --- | --- |
| `QueryRewriter` | `__init__(llm_client, prompt_manager)` | llm_client: BaseLLMClient, prompt_manager: PromptManager | - |
| `QueryRewriter` | `rewrite(query, context=None, user_profile=None)` | query: str, context: Optional[List[str]], user_profile: Optional[str] | rewritten_query: str |
| `QueryRewriter` | `batch_rewrite(queries, contexts=None)` | queries: List[str], contexts: Optional[List[List[str]]] | rewritten_queries: List[str] |

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

| 类名 | 方法 | 功能说明 |
| --- | --- | --- |
| `IntentConfig` | `load(config_path)` | 从YAML文件加载配置 |
| `IntentConfig` | `get_api_client_config()` | 获取API客户端配置 |
| `IntentConfig` | `get_prompt_template_config()` | 获取Prompt模板配置 |

---

## 7. 评估体系总结

| 评估阶段 | 指标/方法 | 目的 | 
| --- | --- | --- |
| **V0评估** | LLM评审（相关性、流畅性、事实性）、人工体验 | 验证RAG基础流程可用性 |
| **V1+离线评估** | MRR@10, NDCG@10, PHR@k | 量化检索性能 |
| **基线对比** | 纯语义 vs 热门 vs 个性化 | 验证个性化收益 |
| **LLM评审** | 相关性、个性化、流畅性、事实性 | 生成质量评估 |
| **模拟在线** | 👍/👎按钮、点击日志 | 用户反馈收集 |
| **AI评审团** | 多维度评分聚合 | 自动化评估 |

---

## 8. 执行建议与简历呈现

### 8.1 Git管理
- 从V0开始用Git管理，打好有意义的Tag（如 `v0-baseline`, `v1-personalized`）
- 记录每次提交，体现项目迭代过程

### 8.2 关键词植入
| 版本 | 关键词 |
| --- | --- |
| V0 | 向量检索、LLM调用、Gradio |
| V1 | 多路召回、LambdaMART、NDCG、LLM-as-a-judge |
| V2 | 多轮对话、DPO、偏好对齐、AI评审团 |

### 8.3 预期产出
- **V0**：可访问的Web Demo，基础搜索回复
- **V1**：完整个性化搜索系统 + 离线评估报告
- **V2**：具备多轮对话和偏好对齐的完整系统

---

## 9. 参考文献

1. Guo, Q., Chen, W., & Wan, H. (2021). AOL4PS: A Large-scale Data Set for Personalized Search. *Data Intelligence*.
2. Zheng, J. Y., et al. (2025). Can Synthetic Query Rewrites Capture User Intent Better than Humans? *arXiv*.
3. Liu, H., et al. (2021). Conversational Query Rewriting with Self-Supervised Learning. *arXiv*.
4. Chu, X., et al. (2026). Accurate and Efficient Personalized Query Rewriting in Baidu Search. *WWW*.
