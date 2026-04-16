from fastapi import APIRouter
from app.api.endpoints import answer

api_router = APIRouter()

api_router.include_router(answer.router, prefix="/answer", tags=["answer"])
