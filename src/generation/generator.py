import logging
from typing import List, Iterator, Optional

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from .prompts import PromptManager, get_default_prompts

logger = logging.getLogger(__name__)


class Generator:
    def __init__(
        self,
        llm: BaseChatModel,
        prompt_manager: Optional[PromptManager] = None,
        max_context_chars: int = 3000,
    ):
        self.llm = llm
        self.prompt_manager = prompt_manager or get_default_prompts()
        self.max_context_chars = max_context_chars

    def _trim_context(self, docs: List[Document]) -> List[Document]:
        total = 0
        trimmed = []
        for doc in docs:
            char_len = len(doc.page_content)
            if total + char_len > self.max_context_chars:
                remaining = self.max_context_chars - total
                if remaining > 200:
                    doc.page_content = doc.page_content[:remaining] + "..."
                    trimmed.append(doc)
                break
            trimmed.append(doc)
            total += char_len
        return trimmed

    def _build_messages(self, question: str, docs: List[Document]):
        query_type = self.prompt_manager.classify_query_type(question)
        logger.info(f"查询类型: {query_type} | 问题: {question[:50]}...")

        system_prompt = self.prompt_manager.get_system_prompt(query_type)

        docs = self._trim_context(docs)
        context = self.prompt_manager.format_context(docs)
        user_prompt = self.prompt_manager.build_user_prompt(question, context)

        messages = [
            ("system", system_prompt),
            ("human", user_prompt),
        ]

        logger.info(
            f"上下文: {len(docs)} 个分片, {len(context)} 字符"
        )
        return messages, query_type

    def generate(
        self,
        question: str,
        docs: List[Document],
    ) -> dict:
        messages, query_type = self._build_messages(question, docs)

        prompt = ChatPromptTemplate.from_messages(messages)
        chain = prompt | self.llm | StrOutputParser()

        answer = chain.invoke({"question": question})

        return {
            "question": question,
            "query_type": query_type,
            "answer": answer,
            "context_docs": docs,
        }

    def generate_stream(
        self,
        question: str,
        docs: List[Document],
    ) -> Iterator[dict]:
        messages, query_type = self._build_messages(question, docs)

        prompt = ChatPromptTemplate.from_messages(messages)
        chain = prompt | self.llm | StrOutputParser()

        accumulated = []
        for chunk in chain.stream({"question": question}):
            accumulated.append(chunk)
            yield {
                "question": question,
                "query_type": query_type,
                "answer": "".join(accumulated),
                "delta": chunk,
                "context_docs": docs,
                "finished": False,
            }

        yield {
            "question": question,
            "query_type": query_type,
            "answer": "".join(accumulated),
            "delta": "",
            "context_docs": docs,
            "finished": True,
        }
