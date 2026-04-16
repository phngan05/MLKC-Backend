from pydantic_settings import BaseSettings
from typing import List, Optional
import os
from dotenv import load_dotenv

class Settings(BaseSettings):
    # Application
    PROJECT_NAME: str = "Machine Learning Chatbot" 
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    
    # API
    API_V1_STR: str = "/api"  
    
    # GROQ API KEY
    GROQ_API_KEY: str
    
    # PINECONE API KEY
    PINECONE_API_KEY: str
    
    # CORS
    BACKEND_CORS_ORIGINS: List[str] = ["*"]
    
    class Config:
        env_file = ".env"
        env_file_encoding = 'utf-8'

settings = Settings()

