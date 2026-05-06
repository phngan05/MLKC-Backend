import asyncio
from langchain_groq import ChatGroq
from langchain_core.documents import Document

from app.models.retrieval.vector_stores import VectorStoreManager
from app.models.retrieval.query_translator import (
    QueryTranslatorFactory,
    QueryTranslationMethod_Classifier,
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
    Optimized for high-speed concurrent execution.
    """

    def __init__(self, cfg: RAGConfig, vsm: VectorStoreManager, llm: ChatGroq):
        self._cfg = cfg
        self._vsm = vsm
        self._llm = llm
        self._router = QueryRouter(llm)
        self._auto_classifier = QueryTranslationMethod_Classifier(llm)

    async def retrieve(
        self,
        query: str,
        method: QueryTranslationMethod = QueryTranslationMethod.AUTO,
    ) -> list[Document]:
        """
        Full retrieval pipeline using async parallel processing.
        """
        
        # --- STEP 1: Resolve Translation Method ---
        # We need this first to know which translator to create
        resolved_method = method
        if method == QueryTranslationMethod.AUTO:
            resolved_method = await self._auto_classifier.classify(query)

        # Create the appropriate translator
        translator = QueryTranslatorFactory.create(
            resolved_method, self._llm, self._cfg.rag_fusion_queries
        )

        # --- STEP 2: Parallel Translation & Routing ---
        # Optimization: Translate queries and Decide DB routes at the same time
        logger.info(f"Processing retrieval for method: {resolved_method.value}")
        
        translation_task = translator.translate(query)
        routing_task = self._router.route(query)

        # Wait for both LLM tasks to complete simultaneously
        sub_queries, store_keys = await asyncio.gather(translation_task, routing_task)

        # --- STEP 3: Concurrent Vector Search ---
        # Optimization: Create search tasks for every sub-query across every selected store
        search_tasks = []
        for sub_q in sub_queries:
            for key in store_keys:
                search_tasks.append(self._execute_search(key, sub_q))

        if not search_tasks:
            logger.warning("No search tasks generated for query: %s", query)
            return []

        # Execute ALL vector searches in parallel (e.g., 4 sub-queries * 2 stores = 8 searches at once)
        all_result_lists = await asyncio.gather(*search_tasks)

        # Filter out empty results
        valid_results = [res for res in all_result_lists if res]

        if not valid_results:
            logger.warning("No documents retrieved after search for query: %s", query)
            return []

        # --- STEP 4: Fusion & Deduplication ---
        # Reciprocal Rank Fusion (RRF) combines results from multiple sources
        merged = reciprocal_rank_fusion(valid_results)
        
        # Deduplicate and limit to K*2 for better context coverage before final grading
        final = deduplicate_docs(merged)[: self._cfg.retrieval_k * 2]
        
        logger.info(f"Retrieved {len(final)} final docs using {len(search_tasks)} parallel searches")
        return final

    async def _execute_search(self, store_key: str, sub_query: str) -> list[Document]:
        """
        Helper to run the vector store similarity search in a non-blocking way.
        """
        try:
            # Pinecone similarity_search is typically network-bound (IO).
            # If the library doesn't provide a native 'asimilarity_search', 
            # we run it in an executor to prevent blocking the event loop.
            loop = asyncio.get_running_loop()
            
            # Using run_in_executor to handle synchronous VectorStore calls
            results = await loop.run_in_executor(
                None, 
                self._vsm.similarity_search, 
                store_key, 
                sub_query, 
                self._cfg.retrieval_k
            )
            return results
        except Exception as e:
            logger.error(f"Error searching in store '{store_key}': {e}")
            return []