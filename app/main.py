import os
import asyncio
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings
from app.core.rag import RAGManager
from app.api.api import api_router

app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url=f"{settings.API_V1_STR}/openapi.json"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    """Pre-initialize RAG system on startup to avoid cold start latency on first request."""
    try:
        rag = RAGManager.get_instance()
        # Warmup with a simple query to initialize LLM connections
        await rag.query("Hello")
    except Exception as e:
        print(f"Warmup error: {e}")

# Include routers
app.include_router(api_router, prefix=settings.API_V1_STR)

@app.get("/")
def root():
    return {"message": "Welcome to Machine Learning Chatbot"}
