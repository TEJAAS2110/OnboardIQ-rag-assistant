from fastapi import APIRouter, HTTPException, status
from app.models.schemas import (
    ChatRequest, ChatResponse, Citation,
    FeedbackRequest, FeedbackResponse
)
import json
from pathlib import Path

router = APIRouter(prefix="/chat", tags=["chat"])

# Global variables (will be injected by main.py)
retriever = None
generator = None

def set_dependencies(ret, gen):
    global retriever, generator
    retriever = ret
    generator = gen

@router.post("/query", response_model=ChatResponse)
async def chat_query(request: ChatRequest):
    """
    Process a chat query with RAG pipeline
    """
    try:
        print(f"\n{'='*60}")
        print(f"📨 New Query: {request.query}")
        print(f"{'='*60}")
        
        # Step 1: Retrieve relevant chunks
        retrieved_chunks = retriever.retrieve(request.query, top_k=request.top_k)
        
        if not retrieved_chunks:
            return ChatResponse(
                answer="I couldn't find relevant information to answer your question. Please ensure documents have been uploaded or try rephrasing your question.",
                citations=[],
                confidence=0.0,
                sources_used=0,
                retrieved_chunks=0,
                query=request.query
            )
        
        # Step 2: Generate answer with citations
        # Convert conversation history to proper format
        history = None
        if request.conversation_history:
            history = [
                {"role": msg.role, "content": msg.content}
                for msg in request.conversation_history
            ]
        
        result = generator.generate_answer(
            request.query,
            retrieved_chunks,
            history
        )
        
        # Step 3: Format citations
        citations = [Citation(**cite) for cite in result['citations']]
        
        print(f"\n✅ Response generated:")
        print(f"   Confidence: {result['confidence']}")
        print(f"   Sources: {result['sources_used']}")
        print(f"   Citations: {len(citations)}")
        
        return ChatResponse(
            answer=result['answer'],
            citations=citations,
            confidence=result['confidence'],
            sources_used=result['sources_used'],
            retrieved_chunks=result['retrieved_chunks'],
            query=request.query
        )
    
    except Exception as e:
        print(f"❌ Error in chat query: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing query: {str(e)}"
        )

@router.post("/feedback", response_model=FeedbackResponse)
async def submit_feedback(feedback: FeedbackRequest):
    """
    Submit user feedback on answers (bonus feature for improvement)
    """
    try:
        # Save feedback to JSON file for analysis
        feedback_dir = Path("feedback")
        feedback_dir.mkdir(exist_ok=True)
        
        feedback_file = feedback_dir / "feedback_log.jsonl"
        
        with open(feedback_file, "a") as f:
            f.write(json.dumps(feedback.dict()) + "\n")
        
        print(f"💬 Feedback received: {feedback.rating} for query: {feedback.query[:50]}...")
        
        return FeedbackResponse(
            success=True,
            message="Thank you for your feedback!"
        )
    
    except Exception as e:
        return FeedbackResponse(
            success=False,
            message=f"Error saving feedback: {str(e)}"
        )

@router.get("/feedback/stats")
async def get_feedback_stats():
    """
    Get feedback statistics (for evaluation dashboard)
    """
    try:
        feedback_file = Path("feedback/feedback_log.jsonl")
        
        if not feedback_file.exists():
            return {
                "total_feedback": 0,
                "positive": 0,
                "negative": 0,
                "positive_rate": 0.0
            }
        
        positive = 0
        negative = 0
        total = 0
        
        with open(feedback_file, "r") as f:
            for line in f:
                feedback = json.loads(line)
                total += 1
                if feedback['rating'] == 'positive':
                    positive += 1
                else:
                    negative += 1
        
        return {
            "total_feedback": total,
            "positive": positive,
            "negative": negative,
            "positive_rate": round(positive / total * 100, 1) if total > 0 else 0.0
        }
    
    except Exception as e:
        return {
            "error": str(e),
            "total_feedback": 0
        }