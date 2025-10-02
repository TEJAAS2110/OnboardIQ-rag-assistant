from pydantic_settings import BaseSettings
from typing import Optional
import os

class Settings(BaseSettings):
    # OpenAI
    OPENAI_API_KEY: str
    
    # Models
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    LLM_MODEL: str = "gpt-4o-mini"
    
    # ChromaDB
    CHROMA_PERSIST_DIR: str = "./chroma_db"
    COLLECTION_NAME: str = "onboardiq_knowledge"
    
    # Chunking
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    
    # Retrieval
    TOP_K_SEMANTIC: int = 20
    TOP_K_BM25: int = 20
    TOP_K_RERANK: int = 10
    FINAL_TOP_K: int = 5
    
    # Generation
    TEMPERATURE: float = 0.3
    MAX_TOKENS: int = 1000
    
    class Config:
        env_file = ".env"
        extra = "allow"

_settings = None

def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
