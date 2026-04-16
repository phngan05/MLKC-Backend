from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from langchain_core.documents import Document
from app.models.rag_config import RAGConfig
from app.models.utils.helpers import get_logger

logger = get_logger(__name__)

_STORE_KEYS = ("text", "code")


class VectorStoreManager:
    """
    Manages two separate Chroma collections:
      - "text"  → theory, definitions, formulas
      - "code"  → algorithms, implementations
    """

    def __init__(self, cfg: RAGConfig):
        self._cfg = cfg
        self._embedding = HuggingFaceEmbeddings(
            model=cfg.embedding.model_name,
        )
        self.pc = Pinecone(api_key = cfg.pinecone_api_key)
        self._stores: dict[str, PineconeVectorStore] = {
            "text": self._build_store(cfg.vector_store.text_index_name),
            "code": self._build_store(cfg.vector_store.code_index_name),
        }

    # ------------------------------------------------------------------ #
    # Public interface
    # ------------------------------------------------------------------ #

    def add_documents(self, store_key: str, docs: list[Document]) -> None:
        self._validate_key(store_key)
        self._stores[store_key].add_documents(docs)
        logger.info("Added %d docs to '%s' vector store", len(docs), store_key)

    def similarity_search(
        self, store_key: str, query: str, k: int = 5
    ) -> list[Document]:
        self._validate_key(store_key)
        return self._stores[store_key].similarity_search(query, k=k)

    def as_retriever(self, store_key: str, k: int = 5):
        self._validate_key(store_key)
        return self._stores[store_key].as_retriever(
            search_kwargs={"k": k}
        )

    def get_store(self, store_key: str) -> PineconeVectorStore:
        self._validate_key(store_key)
        return self._stores[store_key]

    # ------------------------------------------------------------------ #
    # Private helpers
    # ------------------------------------------------------------------ #

    def _build_store(self, index_name: str) -> PineconeVectorStore:
        if not self.pc.has_index(index_name):
            self.pc.create_index(
                name = index_name,
                dimension = self._cfg.vector_store.dimension,
                metric = "cosine",
                spec=ServerlessSpec(
                    cloud = "aws",
                    region = "us-east-1"
                )
            )
        
        index = self.pc.Index(index_name)
        return PineconeVectorStore(
            embedding = self._embedding,
            index = index
        )

    def _validate_key(self, key: str) -> None:
        if key not in _STORE_KEYS:
            raise ValueError(f"Unknown store key '{key}'. Choose from {_STORE_KEYS}.")
