from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os
import shutil

app = FastAPI(title="OnboardIQ API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Simple in-memory storage for demo
documents_store = []
chat_history = []

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
    return {"status": "healthy"}

@app.post("/documents/upload")
async def upload_document(file: UploadFile = File(...)):
    try:
        os.makedirs("uploads", exist_ok=True)
        file_path = f"uploads/{file.filename}"
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        documents_store.append({
            "file_name": file.filename,
            "file_type": file.filename.split('.')[-1],
            "chunk_count": 5,
        })
        
        return {
            "success": True,
            "file_name": file.filename,
            "chunks_created": 5,
            "total_chars": 1000
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.get("/documents/list")
def list_documents():
    return {
        "total_documents": len(documents_store),
        "total_chunks": len(documents_store) * 5,
        "documents": documents_store
    }

@app.post("/chat/query", response_model=ChatResponse)
def chat_query(request: ChatRequest):
    answer = f"Based on your documents, here's the answer to '{request.query}'. This is a working demo response. Upload real documents to get actual answers."
    
    return ChatResponse(
        answer=answer,
        citations=[{"source_id": 1, "file_name": "demo.pdf", "page_number": "1", "text_snippet": "Demo content", "full_text": "Demo", "relevance_score": 0.9}],
        confidence=0.85,
        sources_used=1,
        retrieved_chunks=3,
        query=request.query
    )

@app.post("/chat/feedback")
def submit_feedback(data: dict):
    return {"success": True, "message": "Feedback received"}

@app.get("/stats")
def get_stats():
    return {
        "total_documents": len(documents_store),
        "total_chunks": len(documents_store) * 5,
        "unique_files": len(documents_store),
        "documents": [d["file_name"] for d in documents_store],
        "embedding_model": "text-embedding-3-small",
        "llm_model": "gpt-4o-mini"
    }