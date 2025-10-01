from fastapi import FastAPI, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from app.config import get_settings
from app.models.schemas import SystemStats, HealthCheck
from app.core.ingestion import IngestionPipeline
from app.core.retrieval import HybridRetriever
from app.core.generation import AnswerGenerator
from app.api import documents, chat
from datetime import datetime
import os

# Get settings
settings = get_settings()

# Create FastAPI app
app = FastAPI(
    title="RAG Onboarding Assistant API",
    description="AI Knowledge Assistant for Employee Onboarding",
    version="1.0.0"
)

# CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize core components on startup
@app.on_event("startup")
async def startup_event():
    """Initialize RAG components"""
    print("\n" + "="*60)
    print("🚀 Starting RAG Onboarding Assistant")
    print("="*60)
    
    # Create necessary directories
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    os.makedirs(settings.CHROMA_PERSIST_DIR, exist_ok=True)
    
    # Initialize ingestion pipeline
    print("\n📚 Initializing Ingestion Pipeline...")
    app.state.ingestion_pipeline = IngestionPipeline()
    
    # Initialize retriever
    print("\n🔍 Initializing Hybrid Retriever...")
    app.state.retriever = HybridRetriever(app.state.ingestion_pipeline)
    
    # Initialize generator
    print("\n💬 Initializing Answer Generator...")
    app.state.generator = AnswerGenerator()
    
    # Inject dependencies into routers
    documents.set_dependencies(
        app.state.ingestion_pipeline,
        app.state.retriever,
        app.state.generator
    )
    chat.set_dependencies(
        app.state.retriever,
        app.state.generator
    )
    
    print("\n✅ All systems ready!")
    print("="*60 + "\n")

# Include routers
app.include_router(documents.router)
app.include_router(chat.router)

# Root endpoint
@app.get("/")
async def root():
    """API root"""
    return {
        "message": "RAG Onboarding Assistant API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

# Health check
@app.get("/health", response_model=HealthCheck)
async def health_check():
    """System health check"""
    try:
        # Check ChromaDB connection
        db_connected = app.state.ingestion_pipeline.collection.count() >= 0
        
        # Check OpenAI API key
        openai_configured = bool(settings.OPENAI_API_KEY)
        
        return HealthCheck(
            status="healthy" if (db_connected and openai_configured) else "degraded",
            timestamp=datetime.now().isoformat(),
            database_connected=db_connected,
            openai_configured=openai_configured
        )
    except Exception as e:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "status": "unhealthy",
                "timestamp": datetime.now().isoformat(),
                "error": str(e)
            }
        )

# System statistics
@app.get("/stats", response_model=SystemStats)
async def get_system_stats():
    """Get system statistics"""
    try:
        stats = app.state.ingestion_pipeline.get_stats()
        
        return SystemStats(
            total_documents=stats['unique_documents'],
            total_chunks=stats['total_chunks'],
            unique_files=stats['unique_documents'],
            documents=stats['documents'],
            embedding_model=settings.EMBEDDING_MODEL,
            llm_model=settings.LLM_MODEL
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error fetching stats: {str(e)}"
        )

# Run with: uvicorn app.main:app --reload --port 8000