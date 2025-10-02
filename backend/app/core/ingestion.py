import chromadb
from openai import OpenAI
from typing import List, Dict, Any
import uuid
from app.config import get_settings
from app.utils.document_processor import DocumentProcessor
from app.utils.chunking import SmartChunker

settings = get_settings()

class IngestionPipeline:
    """
    Handles document ingestion: process → chunk → embed → store
    """
    
    def __init__(self):
        # Initialize ChromaDB - FIXED VERSION
        self.chroma_client = chromadb.PersistentClient(
            path=settings.CHROMA_PERSIST_DIR
        )
        
        # Get or create collection
        self.collection = self.chroma_client.get_or_create_collection(
            name=settings.COLLECTION_NAME,
            metadata={"description": "Company knowledge base"}
        )
        
        # Initialize OpenAI for embeddings
        self.openai_client = OpenAI(
            api_key=settings.OPENAI_API_KEY,
            timeout=60.0,
            max_retries=3
        )
        
        # Initialize processors
        self.doc_processor = DocumentProcessor()
        self.chunker = SmartChunker(
            chunk_size=settings.CHUNK_SIZE,
            overlap=settings.CHUNK_OVERLAP
        )
        
        print(f"✅ Initialized ChromaDB collection: {settings.COLLECTION_NAME}")
        print(f"   Current document count: {self.collection.count()}")
    
    def ingest_document(self, file_path: str) -> Dict[str, Any]:
        """
        Main ingestion method: process single document end-to-end
        """
        try:
            # Step 1: Process document
            print(f"📄 Processing: {file_path}")
            doc_data = self.doc_processor.process_file(file_path)
            
            if not doc_data['content']:
                return {
                    "success": False,
                    "error": "No content extracted from document"
                }
            
            # Step 2: Chunk document
            print(f"✂️  Chunking document...")
            chunks = self.chunker.chunk_document(
                doc_data['content'],
                doc_data['metadata']
            )
            print(f"   Created {len(chunks)} chunks")
            
            # Step 3: Generate embeddings
            print(f"🔢 Generating embeddings...")
            embeddings = self._generate_embeddings([c['text'] for c in chunks])
            
            # Step 4: Store in ChromaDB
            print(f"💾 Storing in vector database...")
            self._store_chunks(chunks, embeddings, doc_data)
            
            return {
                "success": True,
                "file_name": doc_data['file_name'],
                "chunks_created": len(chunks),
                "total_chars": len(doc_data['content']),
                "metadata": doc_data['metadata']
            }
        
        except Exception as e:
            print(f"❌ Error ingesting document: {e}")
            import traceback
            traceback.print_exc()
            return {
                "success": False,
                "error": str(e)
            }
    
    def _generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using OpenAI API"""
        try:
            # Batch process for efficiency (OpenAI allows up to 2048 texts)
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
        
        except Exception as e:
            print(f"Error generating embeddings: {e}")
            raise
    
    def _store_chunks(self, chunks: List[Dict], embeddings: List[List[float]], doc_data: Dict):
        """Store chunks with embeddings in ChromaDB"""
        ids = []
        documents = []
        metadatas = []
        
        for chunk, embedding in zip(chunks, embeddings):
            chunk_id = str(uuid.uuid4())
            
            ids.append(chunk_id)
            documents.append(chunk['text'])
            
            # Combine document and chunk metadata
            combined_metadata = {
                **chunk['metadata'],
                'file_name': doc_data['file_name'],
                'file_type': doc_data['file_type'],
                'file_path': doc_data['file_path'],
            }
            
            # ChromaDB requires string values for metadata
            metadatas.append({
                k: str(v) if v is not None else ""
                for k, v in combined_metadata.items()
            })
        
        # Add to collection
        self.collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas
        )
        
        print(f"✅ Stored {len(ids)} chunks in ChromaDB")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get ingestion statistics"""
        count = self.collection.count()
        
        # Get unique documents
        if count > 0:
            results = self.collection.get(limit=count)
            unique_files = set(m.get('file_name', '') for m in results['metadatas'])
        else:
            unique_files = set()
        
        return {
            "total_chunks": count,
            "unique_documents": len(unique_files),
            "documents": list(unique_files)
        }
    
    def delete_document(self, file_name: str) -> bool:
        """Delete all chunks from a specific document"""
        try:
            # Get all IDs for this document
            results = self.collection.get(
                where={"file_name": file_name}
            )
            
            if results['ids']:
                self.collection.delete(ids=results['ids'])
                print(f"🗑️  Deleted {len(results['ids'])} chunks from {file_name}")
                return True
            
            return False
        
        except Exception as e:
            print(f"Error deleting document: {e}")
            return False
