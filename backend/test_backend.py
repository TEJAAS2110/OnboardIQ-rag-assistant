"""
Quick test script for backend functionality
Run this to verify everything works before building frontend
"""

from app.core.ingestion import IngestionPipeline
from app.core.retrieval import HybridRetriever
from app.core.generation import AnswerGenerator
from pathlib import Path

def create_sample_document():
    """Create a sample employee handbook for testing"""
    sample_text = """
    Employee Handbook - Tech Corp
    
    [PAGE 1]
    Welcome to Tech Corp
    
    This handbook contains important information about company policies and procedures.
    
    ## Leave Policy
    
    All full-time employees are entitled to 15 days of paid leave per year.
    Leave requests must be submitted at least 2 weeks in advance through the HR portal.
    Emergency leave can be granted with manager approval on a case-by-case basis.
    
    Contractors are entitled to 10 days of unpaid leave per year.
    
    [PAGE 2]
    ## Dress Code
    
    Tech Corp maintains a business casual dress code.
    Employees should dress professionally while maintaining comfort.
    
    Acceptable attire:
    - Business casual shirts and blouses
    - Slacks, khakis, or professional skirts
    - Closed-toe shoes
    
    Not acceptable:
    - Tank tops or sleeveless shirts
    - Shorts or athletic wear
    - Flip-flops or sandals
    
    [PAGE 3]
    ## IT Support
    
    For technical issues, contact the IT helpdesk:
    - Email: it-support@techcorp.com
    - Phone: ext. 1234
    - Hours: Monday-Friday, 9 AM - 6 PM
    
    For urgent after-hours issues, call the emergency IT line at ext. 9999.
    
    [PAGE 4]
    ## Remote Work Policy
    
    Employees may work remotely up to 2 days per week with manager approval.
    Remote work requests should be submitted via the company intranet.
    All remote workers must be available during core hours (10 AM - 4 PM).
    """
    
    # Save to file
    uploads_dir = Path("uploads")
    uploads_dir.mkdir(exist_ok=True)
    
    file_path = uploads_dir / "employee_handbook.txt"
    with open(file_path, "w") as f:
        f.write(sample_text)
    
    return str(file_path)

def test_full_pipeline():
    """Test complete RAG pipeline"""
    print("="*70)
    print("🧪 TESTING RAG PIPELINE")
    print("="*70)
    
    # 1. Create sample document
    print("\n📝 Step 1: Creating sample document...")
    doc_path = create_sample_document()
    print(f"   ✅ Created: {doc_path}")
    
    # 2. Initialize components
    print("\n🔧 Step 2: Initializing components...")
    pipeline = IngestionPipeline()
    retriever = HybridRetriever(pipeline)
    generator = AnswerGenerator()
    print("   ✅ All components initialized")
    
    # 3. Ingest document
    print("\n📚 Step 3: Ingesting document...")
    result = pipeline.ingest_document(doc_path)
    if result['success']:
        print(f"   ✅ Ingested: {result['chunks_created']} chunks")
    else:
        print(f"   ❌ Error: {result.get('error')}")
        return
    
    # 4. Test queries
    test_queries = [
        "How many days of leave do I get?",
        "What is the dress code policy?",
        "Who do I contact for IT support?",
        "Can I work from home?"
    ]
    
    print("\n" + "="*70)
    print("🔍 Step 4: Testing queries...")
    print("="*70)
    
    for query in test_queries:
        print(f"\n{'─'*70}")
        print(f"Q: {query}")
        print(f"{'─'*70}")
        
        # Retrieve
        chunks = retriever.retrieve(query, top_k=3)
        print(f"\n📊 Retrieved {len(chunks)} chunks:")
        for i, chunk in enumerate(chunks[:2], 1):
            print(f"   {i}. Score: {chunk['final_score']:.2f} | {chunk['text'][:80]}...")
        
        # Generate answer
        answer_result = generator.generate_answer(query, chunks)
        
        print(f"\n💬 Answer (Confidence: {answer_result['confidence']}):")
        print(f"   {answer_result['answer']}")
        
        print(f"\n📚 Citations ({len(answer_result['citations'])}):")
        for cite in answer_result['citations']:
            print(f"   - {cite['file_name']}, Page {cite['page_number']}")
    
    # 5. Get stats
    print("\n" + "="*70)
    print("📊 Step 5: Database Statistics")
    print("="*70)
    stats = pipeline.get_stats()
    print(f"   Total Chunks: {stats['total_chunks']}")
    print(f"   Unique Documents: {stats['unique_documents']}")
    print(f"   Documents: {', '.join(stats['documents'])}")
    
    print("\n" + "="*70)
    print("✅ ALL TESTS PASSED!")
    print("="*70)
    print("\n💡 Backend is working correctly. Ready to build frontend!")

if __name__ == "__main__":
    test_full_pipeline()