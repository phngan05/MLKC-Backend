from abc import ABC, abstractmethod
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from app.models.rag_config import RAGConfig, QueryTranslationMethod
from app.models.utils.helpers import get_logger, reciprocal_rank_fusion, deduplicate_docs

logger = get_logger(__name__)


# ======================================================================== #
# Base class
# ======================================================================== #

class BaseQueryTranslator(ABC):
    def __init__(self, llm: ChatGroq):
        self._llm = llm

    @abstractmethod
    async def translate(self, query: str) -> list[str]:
        """Return a list of (possibly transformed) queries."""


# ======================================================================== #
# No translation — pass-through
# ======================================================================== #

class NoTranslation(BaseQueryTranslator):
    async def translate(self, query: str) -> list[str]:
        return [query]


# ======================================================================== #
# RAG-Fusion
# ======================================================================== #

class RAGFusionTranslator(BaseQueryTranslator):
    """Generate N alternative queries, retrieve for each, merge via RRF."""

    _PROMPT = ChatPromptTemplate.from_template(
        """You are an AI assistant helping with machine learning questions.
        Generate {n} different, diverse search queries for the question below.
        Output ONLY the queries, one per line, with no numbering or extra text.
        
        Question: {question}"""
    )

    def __init__(self, llm: ChatGroq, n: int = 4):
        super().__init__(llm)
        self._n = n
        self._chain = self._PROMPT | self._llm

    async def translate(self, query: str) -> list[str]:
        response = await self._chain.ainvoke({"question": query, "n": self._n})
        queries = [q.strip() for q in response.content.strip().split("\n") if q.strip()]
        logger.debug("RAG-Fusion queries: %s", queries)
        return queries[:self._n]


# ======================================================================== #
# Step-Back
# ======================================================================== #

class StepbackTranslator(BaseQueryTranslator):
    """
    Rewrite query into a higher-level 'step-back' question to
    retrieve broader context, then use both for retrieval.
    """

    _PROMPT = ChatPromptTemplate.from_template(
        """You are an expert in machine learning.
        Given the specific question below, generate a more general, higher-level question that captures the underlying concept.\n"
        Output ONLY the general question, no extra text.
        
        Specific question: {question}"""
    )

    def __init__(self, llm: ChatGroq):
        super().__init__(llm)
        self._chain = self._PROMPT | self._llm

    async def translate(self, query: str) -> list[str]:
        response = await self._chain.ainvoke({"question": query})
        stepback_q = response.content.strip()
        logger.debug("Stepback query: %s", stepback_q)
        # Return both so caller retrieves with each
        return [query, stepback_q]


# ======================================================================== #
# HyDE — Hypothetical Document Embeddings
# ======================================================================== #

class HyDETranslator(BaseQueryTranslator):
    """
    Generate a hypothetical answer, embed it, and use it
    as the retrieval query (better semantic alignment).
    """

    _PROMPT = ChatPromptTemplate.from_template(
        """You are a machine learning textbook author.
        Write a concise, factual paragraph (3-5 sentences) that would ideally answer the question below. Do not say 'I don't know'.
        
        Question: {question}"""
    )

    def __init__(self, llm: ChatGroq):
        super().__init__(llm)
        self._chain = self._PROMPT | self._llm

    async def translate(self, query: str) -> list[str]:
        response = await self._chain.ainvoke({"question": query})
        hypothetical_doc = response.content.strip()
        logger.debug("HyDE hypothetical doc (first 120 chars): %s...", hypothetical_doc[:120])
        # Use the hypothetical doc as the embedding query
        return [hypothetical_doc]


# ======================================================================== #
# Auto-selector
# ======================================================================== #

class QueryTranslationMethod_Classifier:
    """
    Use an LLM to pick the best translation strategy for a given question,
    when QueryTranslationMethod.AUTO is requested.
    """

    _PROMPT = ChatPromptTemplate.from_template(
        """Classify the following machine learning question into ONE of these "
        retrieval strategies:
        - rag_fusion: question is ambiguous or could benefit from multiple angles
        - stepback: question is very specific and may need broader context first
        - hyde: question asks for explanation/definition where generating a hypothetical answer would help
        - none: question is clear and direct
        
        Output ONLY the strategy name.
        
        Question: {question}"""
    )

    def __init__(self, llm: ChatGroq):
        self._chain = self._PROMPT | llm

    async def classify(self, query: str) -> QueryTranslationMethod:
        response = await self._chain.ainvoke({"question": query})
        raw = response.content.strip().lower()
        try:
            method = QueryTranslationMethod(raw)
            logger.info("Auto-selected query translation: %s", method)
            return method
        except ValueError:
            logger.warning("Unrecognised method '%s', defaulting to NONE", raw)
            return QueryTranslationMethod.NONE


# ======================================================================== #
# Factory
# ======================================================================== #

class QueryTranslatorFactory:
    @staticmethod
    def create(
        method: QueryTranslationMethod,
        llm: ChatGroq,
        n_fusion: int = 4,
    ) -> BaseQueryTranslator:
        mapping = {
            QueryTranslationMethod.RAG_FUSION: lambda: RAGFusionTranslator(llm, n=n_fusion),
            QueryTranslationMethod.STEPBACK:   lambda: StepbackTranslator(llm),
            QueryTranslationMethod.HYDE:       lambda: HyDETranslator(llm),
            QueryTranslationMethod.NONE:       lambda: NoTranslation(llm),
        }
        if method not in mapping:
            raise ValueError(f"Unsupported method: {method}")
        return mapping[method]()
