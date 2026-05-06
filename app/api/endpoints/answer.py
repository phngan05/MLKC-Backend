from fastapi import APIRouter, HTTPException, Depends
from app.core.dependencies import get_rag_service
from app.models.ml_knowledge_rag import MLKnowledgeRAG
from app.schemas.question import QuestionRequest

router = APIRouter()
    
@router.post("")
async def answer(data: QuestionRequest, rag: MLKnowledgeRAG = Depends(get_rag_service)):
    try:
        result = await rag.query(data.question)
        return {"question": data.question, "answer" : result}
    except Exception as e:
        print(f"DEBUG ERROR: {e}")
        raise HTTPException(status_code=500, detail=str(e))