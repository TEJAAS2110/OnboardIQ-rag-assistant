from fastapi import APIRouter, UploadFile, File, HTTPException, status
from fastapi.responses import JSONResponse
from typing import List
import os
import shutil
from pathlib import Path
from app.models.schemas import (
    IngestionResult, DocumentListResponse, DocumentInfo,
    SummaryRequest, SummaryResponse
)
from app.config import get_settings
from datetime import datetime

settings = get_settings()
router = APIRouter(prefix="/documents", tags=["documents"])

# Global variables (will be injected by main.py)
ingestion_pipeline = None
retriever = None
generator = None

def set_dependencies(pipeline, ret, gen):
    global ingestion_pipeline, retriever, generator
    ingestion_pipeline = pipeline
    retriever = ret
    generator = gen

@router.post("/upload", response_model=IngestionResult, status_code=status.HTTP_201_CREATED)
async def upload_document(file: UploadFile = File(...)):
    """
    Upload and ingest a document into the knowledge base
    """
    # Validate file type
    file_ext = Path(file.filename).suffix.lower()
    allowed_extensions = ['.pdf', '.docx', '.txt', '.md', '.html']
    
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type: {file_ext}. Allowed: {', '.join(allowed_extensions)}"
        )
    
    # Check file size
    file.file.seek(0, 2)  # Seek to end
    file_size = file.file.tell()
    file.file.seek(0)  # Reset
    
    if file_size > settings.MAX_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large. Max size: {settings.MAX_FILE_SIZE / 1024 / 1024}MB"
        )
    
    # Save file
    upload_dir = Path(settings.UPLOAD_DIR)
    upload_dir.mkdir(exist_ok=True)
    
    file_path = upload_dir / file.filename
    
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        print(f"📁 Saved file: {file_path}")
        
        # Ingest document
        result = ingestion_pipeline.ingest_document(str(file_path))
        
        # Refresh BM25 index for hybrid search
        if result['success']:
            retriever.refresh_bm25_index()
            print("🔄 BM25 index refreshed")
        
        return IngestionResult(**result)
    
    except Exception as e:
        # Clean up file if ingestion fails
        if file_path.exists():
            file_path.unlink()
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing document: {str(e)}"
        )

@router.get("/list", response_model=DocumentListResponse)
async def list_documents():
    """
    Get list of all ingested documents
    """
    try:
        stats = ingestion_pipeline.get_stats()
        
        # Get detailed info for each document
        documents = []
        
        if stats['total_chunks'] > 0:
            results = ingestion_pipeline.collection.get(limit=stats['total_chunks'])
            
            # Group by file_name
            file_chunks = {}
            for i, metadata in enumerate(results['metadatas']):
                file_name = metadata.get('file_name', 'Unknown')
                if file_name not in file_chunks:
                    file_chunks[file_name] = {
                        'file_type': metadata.get('file_type', 'unknown'),
                        'created_at': metadata.get('created_at', ''),
                        'chunks': 0,
                        'file_size': metadata.get('file_size', 0)
                    }
                file_chunks[file_name]['chunks'] += 1
            
            # Format as DocumentInfo list
            for file_name, info in file_chunks.items():
                documents.append(DocumentInfo(
                    file_name=file_name,
                    file_type=info['file_type'],
                    upload_date=info['created_at'] or datetime.now().isoformat(),
                    chunk_count=info['chunks'],
                    file_size=int(info['file_size']) if info['file_size'] else None
                ))
        
        return DocumentListResponse(
            total_documents=len(documents),
            total_chunks=stats['total_chunks'],
            documents=documents
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error listing documents: {str(e)}"
        )

@router.delete("/{file_name}")
async def delete_document(file_name: str):
    """
    Delete a document from the knowledge base
    """
    try:
        success = ingestion_pipeline.delete_document(file_name)
        
        if success:
            # Refresh BM25 index
            retriever.refresh_bm25_index()
            
            # Delete physical file if exists
            file_path = Path(settings.UPLOAD_DIR) / file_name
            if file_path.exists():
                file_path.unlink()
            
            return {"success": True, "message": f"Deleted {file_name}"}
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document not found: {file_name}"
            )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error deleting document: {str(e)}"
        )

@router.post("/summarize", response_model=SummaryResponse)
async def summarize_document(request: SummaryRequest):
    """
    Generate summary for a specific document (bonus feature)
    """
    try:
        # Get all chunks for this document
        results = ingestion_pipeline.collection.get(
            where={"file_name": request.file_name}
        )
        
        if not results['documents']:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document not found: {request.file_name}"
            )
        
        # Combine chunks
        full_text = "\n\n".join(results['documents'])
        
        # Generate summary
        summary = generator.generate_summary(full_text, request.file_name)
        
        return SummaryResponse(
            file_name=request.file_name,
            summary=summary,
            success=True
        )
    
    except HTTPException:
        raise
    except Exception as e:
        return SummaryResponse(
            file_name=request.file_name,
            summary="",
            success=False,
            error=str(e)
        )