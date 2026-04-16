from app.models.ml_knowledge_rag import MLKnowledgeRAG

class RAGManager:
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = MLKnowledgeRAG()
        return cls._instance
