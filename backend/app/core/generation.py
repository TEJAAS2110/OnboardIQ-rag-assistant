from typing import List, Dict, Any, Optional
from openai import OpenAI
import re
from app.config import get_settings

settings = get_settings()

class AnswerGenerator:
    """
    Generates answers with proper citations and confidence scoring
    """
    
    def __init__(self):
        # Initialize OpenAI - FIXED VERSION
        self.openai_client = OpenAI(
            api_key=settings.OPENAI_API_KEY,
            timeout=60.0,
            max_retries=3
        )
    
    def generate_answer(
        self,
        query: str,
        context_chunks: List[Dict[str, Any]],
        conversation_history: Optional[List[Dict]] = None
    ) -> Dict[str, Any]:
        """
        Generate answer with inline citations
        """
        if not context_chunks:
            return {
                "answer": "I don't have enough information to answer that question. Please try rephrasing or ensure relevant documents have been uploaded.",
                "citations": [],
                "confidence": 0.0,
                "sources_used": 0
            }
        
        # Build context with source IDs
        context_text = self._format_context_with_ids(context_chunks)
        
        # Build prompt
        prompt = self._build_prompt(query, context_text, conversation_history)
        
        # Generate answer
        print("🤖 Generating answer with GPT-4o-mini...")
        
        try:
            response = self.openai_client.chat.completions.create(
                model=settings.LLM_MODEL,
                messages=prompt,
                temperature=settings.TEMPERATURE,
                max_tokens=settings.MAX_TOKENS
            )
            
            answer_text = response.choices[0].message.content
            
            # Extract and map citations
            citations = self._extract_citations(answer_text, context_chunks)
            
            # Calculate confidence
            confidence = self._calculate_confidence(context_chunks, answer_text)
            
            return {
                "answer": answer_text,
                "citations": citations,
                "confidence": confidence,
                "sources_used": len(set(c['file_name'] for c in citations)),
                "retrieved_chunks": len(context_chunks)
            }
        
        except Exception as e:
            print(f"❌ Error generating answer: {e}")
            import traceback
            traceback.print_exc()
            return {
                "answer": f"I encountered an error generating the answer: {str(e)}",
                "citations": [],
                "confidence": 0.0,
                "sources_used": 0
            }
    
    def _format_context_with_ids(self, chunks: List[Dict]) -> str:
        """Format context with source IDs for citation"""
        formatted_parts = []
        
        for i, chunk in enumerate(chunks, 1):
            metadata = chunk.get('metadata', {})
            file_name = metadata.get('file_name', 'Unknown')
            page_num = metadata.get('page_number', 'N/A')
            
            formatted_parts.append(
                f"[Source {i}] (File: {file_name}, Page: {page_num})\n{chunk.get('text', '')}\n"
            )
        
        return "\n---\n".join(formatted_parts)
    
    def _build_prompt(
        self,
        query: str,
        context: str,
        history: Optional[List[Dict]] = None
    ) -> List[Dict[str, str]]:
        """Build conversation prompt"""
        
        system_prompt = """You are a knowledgeable AI assistant helping employees find information from company documents.

Your task:
1. Answer questions accurately using ONLY the provided context
2. Cite your sources using [Source X] notation for every claim
3. If information is not in the context, say so clearly
4. Be conversational and helpful, but precise
5. For multi-part questions, address each part clearly

Citation rules:
- Use [Source X] immediately after each claim
- Multiple sources: [Source 1, 2]
- Be specific about what each source supports

Example:
"According to company policy [Source 1], employees are entitled to 15 days of paid leave annually. For contractors [Source 2], the allocation is different."
"""
        
        messages = [{"role": "system", "content": system_prompt}]
        
        # Add conversation history if provided
        if history:
            messages.extend(history[-6:])
        
        # Add current query with context
        user_message = f"""Context from company documents:

{context}

---

Question: {query}

Please provide a clear, well-cited answer based on the context above."""
        
        messages.append({"role": "user", "content": user_message})
        
        return messages
    
    def _extract_citations(
        self,
        answer: str,
        context_chunks: List[Dict]
    ) -> List[Dict[str, Any]]:
        """Extract citation references and map to actual sources"""
        
        citation_pattern = r'\[Source (\d+(?:,\s*\d+)*)\]'
        matches = re.finditer(citation_pattern, answer)
        
        citations = []
        seen_sources = set()
        
        for match in matches:
            source_ids = [int(x.strip()) for x in match.group(1).split(',')]
            
            for source_id in source_ids:
                if source_id <= len(context_chunks) and source_id not in seen_sources:
                    chunk = context_chunks[source_id - 1]
                    metadata = chunk.get('metadata', {})
                    
                    citations.append({
                        'source_id': source_id,
                        'file_name': metadata.get('file_name', 'Unknown'),
                        'page_number': metadata.get('page_number', 'N/A'),
                        'text_snippet': chunk.get('text', '')[:200] + "...",
                        'full_text': chunk.get('text', ''),
                        'relevance_score': chunk.get('final_score', 0.0)
                    })
                    
                    seen_sources.add(source_id)
        
        return citations
    
    def _calculate_confidence(self, chunks: List[Dict], answer: str) -> float:
        """Calculate confidence score"""
        if not chunks:
            return 0.0
        
        # Average of top 3 retrieval scores
        top_scores = [c.get('final_score', 0) for c in chunks[:3]]
        avg_score = sum(top_scores) / len(top_scores) if top_scores else 0
        
        # Normalize (cross-encoder scores typically -10 to +10)
        normalized_score = (avg_score + 10) / 20
        normalized_score = max(0.0, min(1.0, normalized_score))
        
        # Citation bonus
        has_citations = '[Source' in answer
        citation_bonus = 0.1 if has_citations else 0.0
        
        confidence = min(1.0, normalized_score + citation_bonus)
        
        return round(confidence, 2)
    
    def generate_summary(self, document_text: str, file_name: str) -> str:
        """Generate document summary"""
        
        prompt = f"""Provide a concise summary of this document in 3-4 sentences. 
Focus on the key information and main topics covered.

Document: {file_name}

Content:
{document_text[:3000]}

Summary:"""
        
        try:
            response = self.openai_client.chat.completions.create(
                model=settings.LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=200
            )
            
            return response.choices[0].message.content
        
        except Exception as e:
            return f"Error generating summary: {str(e)}"
