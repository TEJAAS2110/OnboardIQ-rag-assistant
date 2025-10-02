from typing import List, Dict, Any, Tuple
from openai import OpenAI
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
import numpy as np
from app.config import get_settings
from app.core.ingestion import IngestionPipeline

settings = get_settings()

class HybridRetriever:
    """
    Advanced retrieval combining:
    1. Semantic search (ChromaDB + OpenAI embeddings)
    2. Keyword search (BM25)
    3. Cross-encoder re-ranking
    """
    
    def __init__(self, ingestion_pipeline: IngestionPipeline):
        self.pipeline = ingestion_pipeline
        self.collection = ingestion_pipeline.collection
        
        # Initialize OpenAI - FIXED VERSION
        self.openai_client = OpenAI(
            api_key=settings.OPENAI_API_KEY,
            timeout=60.0,
            max_retries=3
        )
        
        # Load re-ranker model
        print("🔄 Loading cross-encoder re-ranker...")
        self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        print("✅ Re-ranker loaded")
        
        # Initialize BM25 index
        self._initialize_bm25()
    
    def _initialize_bm25(self):
        """Build BM25 index from all documents"""
        print("🔍 Building BM25 keyword index...")
        
        # Get all documents from ChromaDB
        count = self.collection.count()
        
        if count == 0:
            self.bm25 = None
            self.bm25_docs = []
            self.bm25_ids = []
            print("⚠️  No documents in database - BM25 index empty")
            return
        
        results = self.collection.get(limit=count)
        
        self.bm25_docs = results['documents']
        self.bm25_ids = results['ids']
        
        # Tokenize for BM25
        tokenized_docs = [doc.lower().split() for doc in self.bm25_docs]
        self.bm25 = BM25Okapi(tokenized_docs)
        
        print(f"✅ BM25 index built with {len(self.bm25_docs)} documents")
    
    def retrieve(self, query: str, top_k: int = None) -> List[Dict[str, Any]]:
        """
        Main retrieval method using hybrid approach
        """
        if top_k is None:
            top_k = settings.FINAL_TOP_K
        
        print(f"\n🔍 Retrieving for query: '{query[:50]}...'")
        
        # Step 1: Semantic search
        semantic_results = self._semantic_search(query, settings.TOP_K_SEMANTIC)
        print(f"   Semantic search: {len(semantic_results)} results")
        
        # Step 2: Keyword search (BM25)
        bm25_results = self._bm25_search(query, settings.TOP_K_BM25)
        print(f"   BM25 search: {len(bm25_results)} results")
        
        # Step 3: Fusion (Reciprocal Rank Fusion)
        fused_results = self._reciprocal_rank_fusion(
            semantic_results,
            bm25_results
        )
        print(f"   Fused results: {len(fused_results)}")
        
        # Step 4: Re-rank top candidates
        reranked_results = self._rerank(
            query,
            fused_results[:settings.TOP_K_RERANK]
        )
        print(f"   Re-ranked: {len(reranked_results)}")
        
        # Return top-k
        return reranked_results[:top_k]
    
    def _semantic_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Semantic search using embeddings"""
        try:
            # Generate query embedding
            response = self.openai_client.embeddings.create(
                model=settings.EMBEDDING_MODEL,
                input=query
            )
            query_embedding = response.data[0].embedding
            
            # Query ChromaDB
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=min(top_k, self.collection.count())
            )
            
            # Format results
            formatted_results = []
            for i in range(len(results['ids'][0])):
                formatted_results.append({
                    'id': results['ids'][0][i],
                    'text': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'distance': results['distances'][0][i],
                    'score': 1 / (1 + results['distances'][0][i]),
                    'source': 'semantic'
                })
            
            return formatted_results
        except Exception as e:
            print(f"Semantic search error: {e}")
            return []
    
    def _bm25_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Keyword search using BM25"""
        if self.bm25 is None or len(self.bm25_docs) == 0:
            return []
        
        try:
            # Tokenize query
            tokenized_query = query.lower().split()
            
            # Get BM25 scores
            scores = self.bm25.get_scores(tokenized_query)
            
            # Get top-k indices
            top_indices = np.argsort(scores)[::-1][:top_k]
            
            # Format results
            formatted_results = []
            for idx in top_indices:
                if scores[idx] > 0:
                    # Get metadata from ChromaDB
                    doc_id = self.bm25_ids[idx]
                    chroma_result = self.collection.get(ids=[doc_id])
                    
                    formatted_results.append({
                        'id': doc_id,
                        'text': self.bm25_docs[idx],
                        'metadata': chroma_result['metadatas'][0],
                        'score': float(scores[idx]),
                        'source': 'bm25'
                    })
            
            return formatted_results
        except Exception as e:
            print(f"BM25 search error: {e}")
            return []
    
    def _reciprocal_rank_fusion(
        self,
        semantic_results: List[Dict],
        bm25_results: List[Dict],
        k: int = 60
    ) -> List[Dict[str, Any]]:
        """
        Reciprocal Rank Fusion: combines rankings from multiple sources
        """
        rrf_scores = {}
        doc_map = {}
        
        # Add semantic results
        for rank, result in enumerate(semantic_results, 1):
            doc_id = result['id']
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + (1 / (k + rank))
            if doc_id not in doc_map:
                doc_map[doc_id] = result
        
        # Add BM25 results
        for rank, result in enumerate(bm25_results, 1):
            doc_id = result['id']
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + (1 / (k + rank))
            if doc_id not in doc_map:
                doc_map[doc_id] = result
        
        # Sort by RRF score
        sorted_ids = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Format results
        fused_results = []
        for doc_id, rrf_score in sorted_ids:
            result = doc_map[doc_id].copy()
            result['rrf_score'] = rrf_score
            result['source'] = 'fused'
            fused_results.append(result)
        
        return fused_results
    
    def _rerank(self, query: str, candidates: List[Dict]) -> List[Dict[str, Any]]:
        """Re-rank using cross-encoder"""
        if not candidates:
            return []
        
        try:
            # Prepare query-document pairs
            pairs = [[query, candidate['text']] for candidate in candidates]
            
            # Get cross-encoder scores
            rerank_scores = self.reranker.predict(pairs)
            
            # Add scores to candidates
            for candidate, score in zip(candidates, rerank_scores):
                candidate['rerank_score'] = float(score)
                candidate['final_score'] = float(score)
            
            # Sort by rerank score
            reranked = sorted(candidates, key=lambda x: x['rerank_score'], reverse=True)
            
            return reranked
        except Exception as e:
            print(f"Reranking error: {e}")
            return candidates
    
    def refresh_bm25_index(self):
        """Rebuild BM25 index"""
        self._initialize_bm25()
