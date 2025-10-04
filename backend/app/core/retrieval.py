from typing import List, Dict, Any
from openai import OpenAI
import numpy as np
from app.config import get_settings
from app.core.ingestion import IngestionPipeline

settings = get_settings()

class HybridRetriever:
    
    def __init__(self, ingestion_pipeline: IngestionPipeline):
        self.pipeline = ingestion_pipeline
        self.openai_client = OpenAI(api_key=settings.OPENAI_API_KEY)
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        try:
            response = self.openai_client.embeddings.create(
                model=settings.EMBEDDING_MODEL,
                input=query
            )
            query_embedding = response.data[0].embedding
            
            results = self.pipeline.vector_store.search(query_embedding, top_k)
            
            return [{
                'text': r['text'],
                'metadata': r['metadata'],
                'score': r['score']
            } for r in results]
        
        except Exception as e:
            print(f"Retrieval error: {e}")
            return []
