import os
from app.models.rag_config import RAGConfig, QueryTranslationMethod
from app.models.retrieval.vector_stores import VectorStoreManager
from app.models.retrieval.retriever import Retriever
from app.models.generation.self_rag import SelfRAG
from app.models.utils.helpers import format_answer

from langchain_groq import ChatGroq
from app.core.config import settings
class MLKnowledgeRAG:
    """
    Top-level facade. Wires all components together.

    Usage:
        rag = MLKnowledgeRAG(groq_api_key="sk-...")
        rag.ingest("./data")
        result = rag.query("How does backpropagation work?")
        print(result["answer"])
    """

    def __init__(
        self,
        groq_api_key: str | None = None,
        pinecone_api_key: str | None = None,
        cfg: RAGConfig | None = None,
    ):
        self._cfg = cfg or RAGConfig()
        self._cfg.groq_api_key = groq_api_key or settings.GROQ_API_KEY
        self._cfg.pinecone_api_key = pinecone_api_key or settings.PINECONE_API_KEY
        self._llm = ChatGroq(
            model=self._cfg.llm.model_name,
            temperature=self._cfg.llm.temperature,
            max_tokens=self._cfg.llm.max_tokens,
            api_key=self._cfg.groq_api_key,
        )

        self._vsm = VectorStoreManager(self._cfg)
        self._retriever = Retriever(self._cfg, self._vsm, self._llm)
        self._self_rag = SelfRAG(self._cfg, self._retriever, self._llm)

    def query(
        self,
        question: str,
        method: QueryTranslationMethod = QueryTranslationMethod.AUTO,
    ) -> dict:
        """Ask a question; returns answer, sources, and grading metadata."""
        result = self._self_rag.answer(question, method=method)
        answer = format_answer(result)
        return answer
