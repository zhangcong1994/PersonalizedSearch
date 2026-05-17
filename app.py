import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import gradio as gr
import logging
import numpy as np

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

from src.utils.config import VECTOR_DB_DIR, MODEL_CACHE_DIR

COLLECTION_NAME = "t2ranking_passages"


class SentenceTransformerEmbeddings:
    def __init__(self, model_path: str, device: str = "cpu"):
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(
            model_path,
            device=device,
            local_files_only=True,
        )
        self._dim = self._model.get_sentence_embedding_dimension()
        logger.info(f"Embedding model loaded: dim={self._dim}")

    def embed_documents(self, texts):
        embeddings = self._model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embeddings.tolist() if isinstance(embeddings, np.ndarray) else embeddings

    def embed_query(self, text):
        embedding = self._model.encode(
            [text],
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embedding[0].tolist()


class RAGPipeline:
    def __init__(self):
        self.embeddings = None
        self.vectorstore = None
        self.llm = None
        self.generator = None
        self.retriever = None
        self._loaded = False
        self._loading = False
        self._count = 0
        self._error = None

    def load(self, device: str = "cpu"):
        if self._loaded or self._loading:
            return self._count

        self._loading = True
        try:
            from langchain_openai import ChatOpenAI
            from langchain_chroma import Chroma
            from src.generation.generator import Generator

            local_path = MODEL_CACHE_DIR / "bge-small-zh-v1.5"
            if not local_path.is_dir():
                raise RuntimeError(
                    f"Model not found at {local_path}. Run download_bge_model.py first."
                )

            logger.info(f"Loading embedding model from: {local_path}")
            self.embeddings = SentenceTransformerEmbeddings(
                str(local_path.resolve()), device=device
            )

            self.vectorstore = Chroma(
                collection_name=COLLECTION_NAME,
                embedding_function=self.embeddings,
                persist_directory=str(VECTOR_DB_DIR / "t2ranking" / "bge-small-zh-v1.5"),
            )

            self._count = self.vectorstore._collection.count()
            logger.info(f"Vector store loaded: {self._count:,} documents")

            self.retriever = self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 5},
            )

            api_key = os.getenv("DEEPSEEK_API_KEY")
            if not api_key:
                raise RuntimeError("DEEPSEEK_API_KEY not set in .env")

            self.llm = ChatOpenAI(
                model="deepseek-chat",
                api_key=api_key,
                base_url="https://api.deepseek.com/v1",
                temperature=0.3,
                max_tokens=1024,
            )
            self.generator = Generator(llm=self.llm)
            self._loaded = True
            return self._count

        except Exception as e:
            self._error = str(e)
            logger.error(f"Failed to load pipeline: {e}")
            raise
        finally:
            self._loading = False

    @property
    def is_loaded(self):
        return self._loaded

    @property
    def count(self):
        return self._count

    @property
    def error(self):
        return self._error

    def query(self, question: str):
        if not self._loaded:
            return {
                "answer": "Pipeline not loaded. Please reload the page.",
                "query_type": "error",
                "passages": "",
            }

        docs = self.retriever.invoke(question)
        result = self.generator.generate(question, docs)

        passages_md = ""
        for i, doc in enumerate(docs):
            pid = doc.metadata.get("pid", "?")
            text = doc.page_content[:300].replace("\n", " ")
            passages_md += f"**[{i+1}] pid={pid}**\n> {text}...\n\n"

        return {
            "answer": result["answer"],
            "query_type": result["query_type"],
            "passages": passages_md,
        }


def create_app():
    pipeline = RAGPipeline()

    with gr.Blocks(title="T2Ranking RAG Search") as app:
        gr.Markdown(
            "# T2Ranking RAG Pipeline\n"
            "T2Ranking 中文段落检索 + DeepSeek 生成，全链路验证。"
        )

        status = gr.Textbox(
            label="Status",
            value="Loading pipeline...",
            interactive=False,
        )

        with gr.Row():
            query_input = gr.Textbox(
                label="Query",
                placeholder="Enter your question in Chinese...",
                scale=4,
                lines=2,
            )
            search_btn = gr.Button("Search", variant="primary", scale=1)

        with gr.Row():
            query_type = gr.Textbox(label="Query Type", interactive=False, scale=1)

        with gr.Row():
            answer_output = gr.Textbox(
                label="Answer",
                interactive=False,
                lines=12,
                max_lines=20,
                elem_classes="answer-box",
            )

        with gr.Accordion("Retrieved Passages", open=False):
            passages_output = gr.Markdown()

        def do_search(query):
            if not query.strip():
                return "", "", ""
            try:
                result = pipeline.query(query)
                return result["answer"], result["query_type"], result["passages"]
            except Exception as e:
                return f"Error: {e}", "error", ""

        search_btn.click(
            fn=do_search,
            inputs=[query_input],
            outputs=[answer_output, query_type, passages_output],
        )
        query_input.submit(
            fn=do_search,
            inputs=[query_input],
            outputs=[answer_output, query_type, passages_output],
        )

        def on_load():
            try:
                count = pipeline.load()
                return f"Ready: {count:,} passages in '{COLLECTION_NAME}'"
            except Exception as e:
                return f"Error loading: {e}"

        app.load(fn=on_load, outputs=[status])

    return app


if __name__ == "__main__":
    app = create_app()
    app.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        theme=gr.themes.Soft(),
        css="""
        .answer-box textarea { font-size: 16px !important; line-height: 1.7 !important; }
        """,
    )
