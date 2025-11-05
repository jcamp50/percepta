"""
Test RAG query directly with seeded channel to verify it works
"""

import asyncio
import sys
import codecs
from pathlib import Path

if sys.platform == "win32":
    sys.stdout = codecs.getwriter("utf-8")(sys.stdout.buffer, "strict")
    sys.stderr = codecs.getwriter("utf-8")(sys.stderr.buffer, "strict")

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from py.reason.rag import RAGService
from py.memory.vector_store import VectorStore
from py.utils.embeddings import embed_text

async def test_rag_direct():
    """Test RAG service directly with seeded channel."""
    print("=" * 80)
    print("Direct RAG Query Test")
    print("=" * 80)
    
    channel_id = "testchannel123"
    
    # Test query
    question = "what game is Mystic Realm?"
    print(f"\nQuestion: {question}")
    print(f"Channel ID: {channel_id}")
    
    try:
        rag_service = RAGService()
        
        # Test direct query
        result = await rag_service.answer(
            channel_id=channel_id,
            question=question,
            top_k=5,
        )
        
        print(f"\nAnswer: {result.get('answer', 'No answer')}")
        print(f"Chunks found: {len(result.get('chunks', []))}")
        print(f"Citations: {len(result.get('citations', []))}")
        
        if result.get('chunks'):
            print("\nTop chunks:")
            for i, chunk in enumerate(result.get('chunks', [])[:3], 1):
                print(f"  {i}. Score: {chunk.get('score', 0):.4f}")
                print(f"     Text: {chunk.get('text', '')[:100]}...")
        
        # Check if answer references Mystic Realm
        answer = result.get('answer', '').lower()
        if 'mystic realm' in answer or 'puzzle' in answer or 'adventure' in answer:
            print("\n[OK] Answer appears to reference transcript content!")
        elif "don't have enough context" in answer.lower():
            print("\n[!] Answer says 'no context' - may be embedding/search issue")
        else:
            print(f"\n[?] Answer: {result.get('answer', '')[:200]}")
            
    except Exception as e:
        print(f"\n[X] Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_rag_direct())

