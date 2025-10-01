from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime

# ============ CHAT MODELS ============

class ChatMessage(BaseModel):
    role: str = Field(..., description="Role: 'user' or 'assistant'")
    content: str = Field(..., description="Message content")
    timestamp: Optional[str] = Field(default=None)

class Citation(BaseModel):
    source_id: int
    file_name: str
    page_number: str
    text_snippet: str
    full_text: str
    relevance_score: float

class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)
    conversation_history: Optional[List[ChatMessage]] = Field(default=[])
    top_k: Optional[int] = Field(default=5, ge=1, le=20)

class ChatResponse(BaseModel):
    answer: str
    citations: List[Citation]
    confidence: float
    sources_used: int
    retrieved_chunks: int
    query: str
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())

# ============ DOCUMENT MODELS ============

class DocumentMetadata(BaseModel):
    file_name: str
    file_size: int
    created_at: str
    modified_at: str
    extension: str

class IngestionResult(BaseModel):
    success: bool
    file_name: Optional[str] = None
    chunks_created: Optional[int] = None
    total_chars: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

class DocumentInfo(BaseModel):
    file_name: str
    file_type: str
    upload_date: str
    chunk_count: int
    file_size: Optional[int] = None

class DocumentListResponse(BaseModel):
    total_documents: int
    total_chunks: int
    documents: List[DocumentInfo]

# ============ SUMMARY MODELS ============

class SummaryRequest(BaseModel):
    file_name: str

class SummaryResponse(BaseModel):
    file_name: str
    summary: str
    success: bool
    error: Optional[str] = None

# ============ FEEDBACK MODELS ============

class FeedbackRequest(BaseModel):
    query: str
    answer: str
    rating: str = Field(..., pattern="^(positive|negative)$")
    comment: Optional[str] = None
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())

class FeedbackResponse(BaseModel):
    success: bool
    message: str

# ============ STATS MODELS ============

class SystemStats(BaseModel):
    total_documents: int
    total_chunks: int
    unique_files: int
    documents: List[str]
    embedding_model: str
    llm_model: str

class HealthCheck(BaseModel):
    status: str
    timestamp: str
    database_connected: bool
    openai_configured: bool