from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os
import shutil
from dotenv import load_dotenv

# Add this debugging code
import sys
print("=" * 50)
print("DEBUGGING FILE STRUCTURE")
print("=" * 50)
print(f"Python path: {sys.path}")
print(f"Current directory: {os.getcwd()}")

# Check if files exist
import os
files_to_check = [
    "app/config.py",
    "app/core/ingestion.py",
    "app/core/retrieval.py",
    "app/core/generation.py",
    "app/utils/document_processor.py",
    "app/utils/chunking.py",
]

for file in files_to_check:
    exists = os.path.exists(file)
    print(f"{'✅' if exists else '❌'} {file}")

print("=" * 50)
# Load environment variables
load_dotenv()

app = FastAPI(title="OnboardIQ API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Import RAG components
# Import RAG components with detailed error tracking
HAS_RAG = False
RAG_ERROR = None

try:
    print("Attempting to import RAG modules...")
    
    from app.core.ingestion import IngestionPipeline
    print("✅ Imported IngestionPipeline")
    
    from app.core.retrieval import HybridRetriever
    print("✅ Imported HybridRetriever")
    
    from app.core.generation import AnswerGenerator
    print("✅ Imported AnswerGenerator")
    
    HAS_RAG = True
    print("✅ ALL RAG MODULES LOADED SUCCESSFULLY!")
    
except ImportError as e:
    RAG_ERROR = str(e)
    print(f"❌ ImportError: {e}")
    import traceback
    traceback.print_exc()
    
except Exception as e:
    RAG_ERROR = str(e)
    print(f"❌ General Error: {e}")
    import traceback
    traceback.print_exc()

if not HAS_RAG:
    print(f"⚠️ WARNING: RAG modules not found, running in demo mode")
    print(f"⚠️ Error details: {RAG_ERROR}")

# Storage
documents_store = []
vector_store = []  # Simple in-memory vector store for demo

class ChatRequest(BaseModel):
    query: str
    conversation_history: List[dict] = []
    top_k: int = 5

class ChatResponse(BaseModel):
    answer: str
    citations: List[dict] = []
    confidence: float = 0.85
    sources_used: int = 0
    retrieved_chunks: int = 0
    query: str

@app.get("/")
def root():
    return {"message": "OnboardIQ API", "status": "online", "docs": "/docs"}

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "rag_enabled": HAS_RAG,
        "documents_loaded": len(documents_store),
        "chunks_loaded": len(vector_store)
    }

@app.post("/documents/upload")
async def upload_document(file: UploadFile = File(...)):
    try:
        # Create uploads directory
        os.makedirs("uploads", exist_ok=True)
        file_path = f"uploads/{file.filename}"
        
        # Save file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Process document with RAG
        if HAS_RAG:
            try:
                from app.core.ingestion import IngestionPipeline
                
                pipeline = IngestionPipeline()
                result = pipeline.ingest_document(file_path)
                
                if result['success']:
                    documents_store.append({
                        "file_name": file.filename,
                        "file_type": file.filename.split('.')[-1],
                        "chunk_count": result['chunks_created'],
                    })
                    
                    return {
                        "success": True,
                        "file_name": file.filename,
                        "chunks_created": result['chunks_created'],
                        "total_chars": result['total_chars']
                    }
                else:
                    raise Exception(result.get('error', 'Unknown error'))
                    
            except Exception as e:
                print(f"RAG processing error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        else:
            raise HTTPException(status_code=503, detail="RAG modules not available")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat/query", response_model=ChatResponse)
def chat_query(request: ChatRequest):
    try:
        if not HAS_RAG:
            raise HTTPException(status_code=503, detail="RAG system not initialized")
        
        from app.core.ingestion import IngestionPipeline
        from app.core.retrieval import HybridRetriever
        from app.core.generation import AnswerGenerator
        
        # Initialize components
        pipeline = IngestionPipeline()
        retriever = HybridRetriever(pipeline)
        generator = AnswerGenerator()
        
        # Retrieve relevant chunks
        chunks = retriever.retrieve(request.query, request.top_k)
        
        if not chunks:
            return ChatResponse(
                answer="No relevant information found. Please upload documents first.",
                citations=[],
                confidence=0.0,
                sources_used=0,
                retrieved_chunks=0,
                query=request.query
            )
        
        # Generate answer
        result = generator.generate_answer(
            query=request.query,
            context_chunks=chunks,
            conversation_history=request.conversation_history
        )
        
        return ChatResponse(
            answer=result['answer'],
            citations=result['citations'],
            confidence=result['confidence'],
            sources_used=result['sources_used'],
            retrieved_chunks=result['retrieved_chunks'],
            query=request.query
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        
        # Demo/fallback response
        answer = f"Based on your documents, here's the answer to '{request.query}'. RAG processing is not fully configured. Please set OPENAI_API_KEY and configure vector database."
        
        return ChatResponse(
            answer=answer,
            citations=[{
                "source_id": 1,
                "file_name": documents_store[0]["file_name"] if documents_store else "demo.pdf",
                "page_number": "1",
                "text_snippet": "Content from your uploaded documents",
                "full_text": "Demo",
                "relevance_score": 0.5
            }],
            confidence=0.50,
            sources_used=1,
            retrieved_chunks=len(vector_store),
            query=request.query
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat/feedback")
def submit_feedback(data: dict):
    return {"success": True, "message": "Feedback received"}

@app.get("/stats")
def get_stats():
    return {
        "total_documents": len(documents_store),
        "total_chunks": len(vector_store),
        "unique_files": len(documents_store),
        "documents": [d["file_name"] for d in documents_store],
        "embedding_model": os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
        "llm_model": os.getenv("LLM_MODEL", "gpt-4o-mini"),
        "rag_enabled": HAS_RAG
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app.main:app", host="0.0.0.0", port=port)
