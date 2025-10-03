from openai import OpenAI
from typing import List, Dict, Any
import uuid
import numpy as np
from app.config import get_settings
from app.utils.document_processor import DocumentProcessor
from app.utils.chunking import SmartChunker

settings = get_settings()

class SimpleVectorStore:
    """Simple in-memory vector store - no ChromaDB"""
    def __init__(self):
        self.chunks = []
        self.embeddings = []
        self.metadata = []
        self.ids = []
    
    def add(self, texts, embeddings, metadatas):
        for text, embedding, metadata in zip(texts, embeddings, metadatas):
            chunk_id = str(uuid.uuid4())
            self.ids.append(chunk_id)
            self.chunks.append(text)
            self.embeddings.append(embedding)
            self.metadata.append(metadata)
    
    def search(self, query_embedding, top_k=5):
        if not self.embeddings:
            return []
        
        # Cosine similarity
        query_norm = np.linalg.norm(query_embedding)
        scores = []
        for emb in self.embeddings:
            emb_norm = np.linalg.norm(emb)
            similarity = np.dot(query_embedding, emb) / (query_norm * emb_norm)
            scores.append(similarity)
        
        # Get top K
        top_indices = np.argsort(scores)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            results.append({
                'text': self.chunks[idx],
                'metadata': self.metadata[idx],
                'score': scores[idx]
            })
        
        return results
    
    def count(self):
        return len(self.chunks)

class IngestionPipeline:
    def __init__(self):
        self.vector_store = SimpleVectorStore()
        self.openai_client = OpenAI(api_key=settings.OPENAI_API_KEY)
        self.doc_processor = DocumentProcessor()
        self.chunker = SmartChunker(
            chunk_size=settings.CHUNK_SIZE,
            overlap=settings.CHUNK_OVERLAP
        )
        print(f"Initialized SimpleVectorStore")
    
    def ingest_document(self, file_path: str) -> Dict[str, Any]:
        try:
            print(f"Processing: {file_path}")
            doc_data = self.doc_processor.process_file(file_path)
            
            if not doc_data.get('content'):
                return {"success": False, "error": "No content extracted"}
            
            print(f"Chunking document...")
            chunks = self.chunker.chunk_document(
                doc_data['content'],
                doc_data['metadata']
            )
            print(f"Created {len(chunks)} chunks")
            
            print(f"Generating embeddings...")
            texts = [c['text'] for c in chunks]
            embeddings = self._generate_embeddings(texts)
            
            print(f"Storing in vector store...")
            metadatas = []
            for chunk in chunks:
                metadatas.append({
                    **chunk['metadata'],
                    'file_name': doc_data['file_name'],
                    'file_type': doc_data['file_type'],
                })
            
            self.vector_store.add(texts, embeddings, metadatas)
            
            return {
                "success": True,
                "file_name": doc_data['file_name'],
                "chunks_created": len(chunks),
                "total_chars": len(doc_data['content'])
            }
        
        except Exception as e:
            print(f"Error ingesting document: {e}")
            import traceback
            traceback.print_exc()
            return {"success": False, "error": str(e)}
    
    def _generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        batch_size = 100
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            response = self.openai_client.embeddings.create(
                model=settings.EMBEDDING_MODEL,
                input=batch
            )
            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)
        
        return all_embeddings
    
    def get_stats(self) -> Dict[str, Any]:
        unique_files = set()
        for meta in self.vector_store.metadata:
            if 'file_name' in meta:
                unique_files.add(meta['file_name'])
        
        return {
            "total_chunks": self.vector_store.count(),
            "unique_documents": len(unique_files),
            "documents": list(unique_files)
        }
