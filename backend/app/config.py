from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    # OpenAI
    OPENAI_API_KEY: str
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    LLM_MODEL: str = "gpt-4o-mini"
    
    # ChromaDB
    CHROMA_PERSIST_DIR: str = "./chroma_db"
    COLLECTION_NAME: str = "company_documents"
    
    # Document Processing
    UPLOAD_DIR: str = "./uploads"
    MAX_FILE_SIZE: int = 10485760
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50
    
    # Retrieval
    TOP_K_SEMANTIC: int = 20
    TOP_K_BM25: int = 20
    TOP_K_RERANK: int = 10
    FINAL_TOP_K: int = 5
    
    # Generation
    MAX_TOKENS: int = 1000
    TEMPERATURE: float = 0.3
    
    class Config:
        env_file = ".env"

@lru_cache()
def get_settings():
    return Settings()