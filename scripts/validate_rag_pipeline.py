import os
import sys
import argparse
import logging

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

from src.utils.config import PROJECT_ROOT, MODEL_CACHE_DIR, VECTOR_DB_DIR

TEST_QUERIES = [
    "蜂巢取快递验证码摁错怎么办",
    "生产过后怎么还有一层肚子",
    "考研英语一和英语二有什么区别",
    "比特币和以太坊哪个更值得投资",
    "西红柿炒鸡蛋的正确做法是什么",
    "为什么晚上睡觉会磨牙",
]


def get_embedding_model(model_name: str = "BAAI/bge-small-zh-v1.5", device: str = "cpu"):
    from langchain_huggingface import HuggingFaceEmbeddings

    local_path = MODEL_CACHE_DIR / "bge-small-zh-v1.5"
    if local_path.is_dir():
        model_name = str(local_path.resolve())

    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True, "batch_size": 128},
    )


def get_vectorstore(embeddings, persist_dir: str, collection_name: str):
    from langchain_chroma import Chroma

    vs = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=persist_dir,
    )
    count = vs._collection.count()
    logger.info(f"Vector store loaded: {count:,} documents in '{collection_name}'")
    return vs, count


def get_llm():
    from langchain_openai import ChatOpenAI

    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY not set")

    return ChatOpenAI(
        model="deepseek-chat",
        api_key=api_key,
        base_url="https://api.deepseek.com/v1",
        temperature=0.3,
        max_tokens=1024,
    )


def run_query(question: str, retriever, generator, show_docs: bool = True):
    from src.generation.generator import Generator

    docs = retriever.invoke(question)

    if show_docs:
        print()
        print("─" * 60)
        print(f"  Retrieved {len(docs)} passages:")
        print("─" * 60)
        for i, doc in enumerate(docs):
            pid = doc.metadata.get("pid", "?")
            snippet = doc.page_content[:200].replace("\n", " ")
            print(f"  [{i+1}] pid={pid} | {snippet}...")
        print("─" * 60)

    result = generator.generate(question, docs)

    print()
    print("=" * 60)
    print(f"  Query: {question}")
    print(f"  Type:  {result['query_type']}")
    print("=" * 60)
    print(f"\n{result['answer']}\n")

    return result


def main():
    parser = argparse.ArgumentParser(description="T2Ranking RAG full pipeline validation")
    parser.add_argument("--query", type=str, help="Single query to test")
    parser.add_argument("--all", action="store_true", help="Run all test queries")
    parser.add_argument("--top-k", type=int, default=5, help="Top-K retrieval results")
    parser.add_argument("--no-docs", action="store_true", help="Hide retrieved docs")
    parser.add_argument("--device", default="cpu", help="Device for embedding model")
    parser.add_argument(
        "--vector-db", default=str(VECTOR_DB_DIR / "t2ranking" / "bge-small-zh-v1.5"), help="Vector DB directory"
    )
    parser.add_argument(
        "--collection", default="t2ranking_passages", help="Collection name"
    )
    args = parser.parse_args()

    print()
    print("=" * 60)
    print("  T2Ranking RAG Pipeline Validation")
    print("=" * 60)

    embeddings = get_embedding_model(device=args.device)
    vs, count = get_vectorstore(
        embeddings,
        persist_dir=args.vector_db,
        collection_name=args.collection,
    )

    if count == 0:
        print("ERROR: Vector store is empty. Build the index first.")
        return 1

    retriever = vs.as_retriever(
        search_type="similarity",
        search_kwargs={"k": args.top_k},
    )

    llm = get_llm()
    logger.info("LLM client ready: deepseek-chat")

    from src.generation.generator import Generator
    generator = Generator(llm=llm)

    show_docs = not args.no_docs

    if args.query:
        result = run_query(args.query, retriever, generator, show_docs=show_docs)
        return 0

    if args.all:
        for q in TEST_QUERIES:
            try:
                run_query(q, retriever, generator, show_docs=show_docs)
            except Exception as e:
                print(f"  ERROR on '{q}': {e}")
        return 0

    print()
    print("Available modes:")
    print("  --query 'your question'   Single query test")
    print("  --all                     Run all preset test queries")
    print("  --top-k 5                 Number of passages to retrieve")
    print("  --no-docs                 Hide retrieved passages")
    print()
    print("Example: python scripts/validate_rag_pipeline.py --all")

    return 0


if __name__ == "__main__":
    sys.exit(main())
