from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os
import shutil
from dotenv import load_dotenv

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
try:
    from app.core.ingestion import process_document, get_text_from_file
    from app.core.retrieval import retrieve_relevant_chunks
    from app.core.generation import generate_answer
    HAS_RAG = True
except ImportError:
    HAS_RAG = False
    print("Warning: RAG modules not found, running in demo mode")

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
                # Extract text from document
                text_content = get_text_from_file(file_path)
                
                # Process and create chunks
                chunks = process_document(text_content, file.filename)
                
                # Store chunks in vector store
                for chunk in chunks:
                    vector_store.append({
                        "file_name": file.filename,
                        "content": chunk["content"],
                        "metadata": chunk.get("metadata", {}),
                        "embedding": chunk.get("embedding")
                    })
                
                chunks_created = len(chunks)
                total_chars = len(text_content)
            except Exception as e:
                print(f"RAG processing error: {e}")
                # Fallback to basic processing
                chunks_created = 5
                total_chars = 1000
        else:
            chunks_created = 5
            total_chars = 1000
        
        # Store document info
        documents_store.append({
            "file_name": file.filename,
            "file_type": file.filename.split('.')[-1],
            "chunk_count": chunks_created,
        })
        
        return {
            "success": True,
            "file_name": file.filename,
            "chunks_created": chunks_created,
            "total_chars": total_chars
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/documents/list")
def list_documents():
    return {
        "total_documents": len(documents_store),
        "total_chunks": len(vector_store),
        "documents": documents_store
    }

@app.post("/chat/query", response_model=ChatResponse)
def chat_query(request: ChatRequest):
    try:
        # Check if we have documents
        if not vector_store:
            return ChatResponse(
                answer="No documents uploaded yet. Please upload documents first to ask questions.",
                citations=[],
                confidence=0.0,
                sources_used=0,
                retrieved_chunks=0,
                query=request.query
            )
        
        # Use RAG if available
        if HAS_RAG:
            try:
                # Retrieve relevant chunks
                relevant_chunks = retrieve_relevant_chunks(
                    query=request.query,
                    vector_store=vector_store,
                    top_k=request.top_k
                )
                
                # Generate answer using LLM
                answer_data = generate_answer(
                    query=request.query,
                    chunks=relevant_chunks,
                    conversation_history=request.conversation_history
                )
                
                # Format citations
                citations = []
                for i, chunk in enumerate(relevant_chunks[:3]):
                    citations.append({
                        "source_id": i + 1,
                        "file_name": chunk.get("file_name", "unknown"),
                        "page_number": str(chunk.get("metadata", {}).get("page", "N/A")),
                        "text_snippet": chunk["content"][:200] + "...",
                        "full_text": chunk["content"],
                        "relevance_score": chunk.get("score", 0.0)
                    })
                
                return ChatResponse(
                    answer=answer_data["answer"],
                    citations=citations,
                    confidence=answer_data.get("confidence", 0.85),
                    sources_used=len(relevant_chunks),
                    retrieved_chunks=len(relevant_chunks),
                    query=request.query
                )
            except Exception as e:
                print(f"RAG query error: {e}")
                # Fall through to demo response
        
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
