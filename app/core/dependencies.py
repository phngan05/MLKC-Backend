from app.core.rag import RAGManager
   
def get_rag_service():
    return RAGManager.get_instance()