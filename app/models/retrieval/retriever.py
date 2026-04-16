from langchain_groq import ChatGroq
from langchain_core.documents import Document

from app.models.retrieval.vector_stores import VectorStoreManager
from app.models.retrieval.query_translator import (
    QueryTranslatorFactory,
    QueryTranslationMethod_Classifier,
    BaseQueryTranslator,
)
from app.models.retrieval.router import QueryRouter
from app.models.rag_config import RAGConfig, QueryTranslationMethod
from app.models.utils.helpers import (
    get_logger, deduplicate_docs, reciprocal_rank_fusion
)

logger = get_logger(__name__)


class Retriever:
    """
    Orchestrates: query translation → query routing → vector search → fusion.
    """

    def __init__(self, cfg: RAGConfig, vsm: VectorStoreManager, llm: ChatGroq):
        self._cfg = cfg
        self._vsm = vsm
        self._llm = llm
        self._router = QueryRouter(llm)
        self._auto_classifier = QueryTranslationMethod_Classifier(llm)

    def retrieve(
        self,
        query: str,
        method: QueryTranslationMethod = QueryTranslationMethod.AUTO,
    ) -> list[Document]:
        """
        Full retrieval pipeline.

        1. Resolve translation method (auto-classify if needed).
        2. Translate query → one or more sub-queries.
        3. Route each sub-query to the correct vector store(s).
        4. Merge and deduplicate results.
        """
        # Step 1 — resolve method
        resolved_method = (
            self._auto_classifier.classify(query)
            if method == QueryTranslationMethod.AUTO
            else method
        )

        # Step 2 — translate
        translator = QueryTranslatorFactory.create(
            resolved_method, self._llm, self._cfg.rag_fusion_queries
        )
        sub_queries = translator.translate(query)

        # Step 3 — route & search
        store_keys = self._router.route(query)
        all_result_lists: list[list[Document]] = []

        for sub_q in sub_queries:
            for key in store_keys:
                results = self._vsm.similarity_search(
                    key, sub_q, k=self._cfg.retrieval_k
                )
                if results:
                    all_result_lists.append(results)

        if not all_result_lists:
            logger.warning("No documents retrieved for query: %s", query)
            return []

        # Step 4 — fuse (RRF handles both single and multiple lists)
        merged = reciprocal_rank_fusion(all_result_lists)
        final = deduplicate_docs(merged)[: self._cfg.retrieval_k * 2]
        logger.info("Retrieved %d final docs", len(final))
        return final
