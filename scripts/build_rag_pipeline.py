"""
LangChain 标准 RAG 流程脚本

完整的"文档加载 → 分片(Chunking) → 向量化(Embedding) → 向量检索 → LLM生成"链路。

用法：
    # 首次运行：构建索引
    python scripts/build_rag_pipeline.py --build_index

    # 已有索引：直接查询
    python scripts/build_rag_pipeline.py

    # 重新构建索引
    python scripts/build_rag_pipeline.py --build_index --rebuild

前提：
    1. 先运行 parse_wikipedia.py 生成 data/processed/wikipedia_articles.jsonl
    2. 在 .env 中配置 DEEPSEEK_API_KEY
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════
# Step 1: 文档加载
# ════════════════════════════════════════════════════════

def load_documents_from_jsonl(jsonl_path: str):
    """从 JSONL 文件加载文档，转为 LangChain Document 对象"""
    from langchain_core.documents import Document

    documents = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            article = json.loads(line.strip())
            doc = Document(
                page_content=article["text"],
                metadata={
                    "source": article["title"],
                    "article_id": article.get("id", ""),
                },
            )
            documents.append(doc)

    logger.info(f"从 JSONL 加载了 {len(documents):,} 篇文档")
    return documents


# ════════════════════════════════════════════════════════
# Step 2: 文档分片 (Chunking)
# ════════════════════════════════════════════════════════

def split_documents(documents, chunk_size=500, chunk_overlap=50):
    """
    使用 LangChain 的 RecursiveCharacterTextSplitter 分片。
    中文友好的分隔符列表：优先按段落、句子、逗号分割。
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=[
            "\n\n",
            "\n",
            "。",
            "！",
            "？",
            "；",
            "，",
            " ",
            "",
        ],
        length_function=len,
        is_separator_regex=False,
    )

    chunks = text_splitter.split_documents(documents)
    logger.info(f"分片完成: {len(documents):,} 篇文档 → {len(chunks):,} 个分片")
    logger.info(f"  平均每个文档 {len(chunks) / max(len(documents), 1):.1f} 个分片")
    return chunks


# ════════════════════════════════════════════════════════
# Step 3: 向量化 + 存储 (Embedding + Vector Store)
# ════════════════════════════════════════════════════════

def get_embedding_model(model_name: str = "BAAI/bge-small-zh-v1.5", device: str = "auto"):
    """获取中文友好的嵌入模型，自动检测 GPU，优先使用本地模型"""
    import torch
    from langchain_huggingface import HuggingFaceEmbeddings

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    local_path = os.path.join(
        os.path.dirname(__file__), "..", "models", "bge-small-zh-v1.5"
    )
    if os.path.isdir(local_path):
        model_name = os.path.abspath(local_path)
        logger.info(f"加载嵌入模型: 本地路径 (device={device})")
    else:
        logger.info(f"加载嵌入模型: {model_name} (device={device})")

    model_kwargs = {"device": device}
    encode_kwargs = {"normalize_embeddings": True, "batch_size": 128}

    if device == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logger.info(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")

    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
    )
    return embeddings


def build_vector_store(chunks, embeddings, persist_dir: str = "./data/vector_db", rebuild: bool = False):
    """构建或加载 Chroma 向量数据库"""
    from langchain_chroma import Chroma

    collection_name = "zhwiki_articles"

    if rebuild and os.path.exists(persist_dir):
        import shutil
        shutil.rmtree(os.path.join(persist_dir, collection_name), ignore_errors=True)
        logger.info("已清除旧的向量数据库")

    os.makedirs(persist_dir, exist_ok=True)

    logger.info(f"构建向量数据库... (共 {len(chunks):,} 个分片)")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir,
        collection_name=collection_name,
    )

    logger.info(f"向量数据库已保存到: {persist_dir}")
    return vectorstore


def load_vector_store(embeddings, persist_dir: str = "./data/vector_db"):
    """加载已有的向量数据库"""
    from langchain_chroma import Chroma

    collection_name = "zhwiki_articles"
    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=persist_dir,
    )
    doc_count = vectorstore._collection.count()
    logger.info(f"已加载向量数据库: {doc_count:,} 个分片")
    return vectorstore


# ════════════════════════════════════════════════════════
# Step 4: LLM 客户端
# ════════════════════════════════════════════════════════

def get_llm():
    """获取 DeepSeek LLM（通过 OpenAI 兼容接口）"""
    from langchain_openai import ChatOpenAI

    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError(
            "请设置环境变量 DEEPSEEK_API_KEY，或在 .env 文件中配置。\n"
            "例如: DEEPSEEK_API_KEY=sk-xxxx"
        )

    llm = ChatOpenAI(
        model="deepseek-chat",
        api_key=api_key,
        base_url="https://api.deepseek.com/v1",
        temperature=0.3,
        max_tokens=1024,
    )
    logger.info(f"LLM 客户端已创建: deepseek-chat")
    return llm


# ════════════════════════════════════════════════════════
# Step 5: RAG 生成模块
# ════════════════════════════════════════════════════════

def build_retriever(vectorstore, top_k: int = 5):
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": top_k},
    )
    logger.info(f"检索器已创建 (Top-K={top_k})")
    return retriever


# ════════════════════════════════════════════════════════
# Step 6: 检索 + 生成演示
# ════════════════════════════════════════════════════════

def run_query(generator, retriever, question: str, show_docs: bool = True):
    """执行一次 RAG 查询并展示结果"""
    from src.generation.generator import Generator

    print("\n" + "=" * 60)
    print(f"  用户查询: {question}")
    print("=" * 60)

    docs = retriever.invoke(question)
    print(f"\n  [检索到 {len(docs)} 个相关片段]")

    if show_docs:
        for i, doc in enumerate(docs):
            title = doc.metadata.get("source", "未知")
            snippet = doc.page_content[:120].replace("\n", " ")
            print(f"    {i+1}. [{title}] {snippet}...")

    print("\n  [AI 回答]")
    try:
        result = generator.generate(question, docs)
        print(f"  {result['answer']}")
        print(f"\n  [查询类型: {result['query_type']}]")
    except Exception as e:
        print(f"  错误: {e}")

    print("-" * 60)


def interactive_mode(generator, retriever):
    """交互式查询模式"""
    print("\n" + "=" * 60)
    print("  个性化AI搜索系统 - RAG 交互模式")
    print("  输入 'quit' 或 'exit' 退出")
    print("=" * 60)

    demo_queries = [
        "人工智能的历史是什么？",
        "量子计算机的基本原理",
        "中国的四大发明有哪些？",
    ]
    print("\n  试试输入以下问题，或自己提问：")
    for q in demo_queries:
        print(f"    · {q}")
    print()

    while True:
        try:
            question = input("  请输入问题: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  再见！")
            break

        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            print("  再见！")
            break

        run_query(generator, retriever, question)


# ════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════

def main():
    project_root = os.path.join(os.path.dirname(__file__), "..")

    parser = argparse.ArgumentParser(description="LangChain RAG 流程")
    parser.add_argument(
        "--build_index", action="store_true",
        help="构建向量索引（首次运行或数据更新后需要）"
    )
    parser.add_argument(
        "--rebuild", action="store_true",
        help="重建索引（清除旧数据）"
    )
    parser.add_argument(
        "--jsonl_path", type=str,
        default=os.path.join(project_root, "data", "processed", "wikipedia_articles.jsonl"),
        help="JSONL 文档路径"
    )
    parser.add_argument(
        "--vector_db_dir", type=str,
        default=os.path.join(project_root, "data", "vector_db"),
        help="向量数据库目录"
    )
    parser.add_argument(
        "--chunk_size", type=int, default=500,
        help="分片大小（字符数）"
    )
    parser.add_argument(
        "--chunk_overlap", type=int, default=50,
        help="分片重叠大小"
    )
    parser.add_argument(
        "--embedding_model", type=str, default="BAAI/bge-small-zh-v1.5",
        help="嵌入模型名称"
    )
    parser.add_argument(
        "--top_k", type=int, default=5,
        help="检索返回数量"
    )
    parser.add_argument(
        "--query", type=str, default=None,
        help="直接查询（非交互模式）"
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        choices=["auto", "cpu", "cuda"],
        help="设备选择 (默认: auto)"
    )
    args = parser.parse_args()

    # 1. 加载嵌入模型（索引和检索都需要）
    embeddings = get_embedding_model(args.embedding_model, device=args.device)

    # 2. 构建或加载向量数据库
    should_build = args.build_index or args.rebuild
    vectorstore_exists = os.path.exists(
        os.path.join(args.vector_db_dir, "zhwiki_articles")
    )

    if should_build or not vectorstore_exists:
        if not os.path.exists(args.jsonl_path):
            logger.error(f"找不到 JSONL 文件: {args.jsonl_path}")
            logger.error("请先运行: python scripts/parse_wikipedia.py")
            sys.exit(1)

        documents = load_documents_from_jsonl(args.jsonl_path)
        chunks = split_documents(documents, args.chunk_size, args.chunk_overlap)
        vectorstore = build_vector_store(chunks, embeddings, args.vector_db_dir, rebuild=args.rebuild)
    else:
        vectorstore = load_vector_store(embeddings, args.vector_db_dir)

    # 3. 创建 LLM
    llm = get_llm()

    # 4. 创建生成器 + 检索器
    from src.generation import Generator
    generator = Generator(llm=llm)
    retriever = build_retriever(vectorstore, top_k=args.top_k)

    # 5. 查询
    if args.query:
        run_query(generator, retriever, args.query)
    else:
        interactive_mode(generator, retriever)


if __name__ == "__main__":
    main()
