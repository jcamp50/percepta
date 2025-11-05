"""Test script for enhanced prompts with various question types."""
import json
import time
import requests
import sys


def test_question(channel_id: str, question: str, question_type: str):
    """Test a single question and return the response."""
    try:
        response = requests.post(
            "http://localhost:8000/rag/answer",
            json={"channel": channel_id, "question": question},
            headers={"Content-Type": "application/json"},
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()
        
        return {
            "question": question,
            "type": question_type,
            "answer": data.get("answer", ""),
            "chunks_count": len(data.get("chunks", [])),
            "context_count": len(data.get("context", [])),
            "has_citations": len(data.get("citations", [])) > 0,
            "chunks": [
                {
                    "text": c.get("text", "")[:80],
                    "timestamp": c.get("timestamp", ""),
                }
                for c in data.get("chunks", [])[:3]
            ],
        }
    except Exception as e:
        return {
            "question": question,
            "type": question_type,
            "answer": f"ERROR: {str(e)}",
            "error": True,
        }


def main():
    channel_id = "672238954"  # plaqueboymax
    
    test_questions = [
        # Original JCB-22 questions
        ("What game are they playing?", "METADATA"),
        ("Who raided?", "EVENT"),
        ("When did the stream start?", "EVENT"),
        ("What did they say about the boss?", "TRANSCRIPT"),
        ("Is there a prediction active?", "EVENT"),
        
        # Additional metadata questions
        ("How many viewers are watching?", "METADATA"),
        ("What's the stream title?", "METADATA"),
        ("Are they live right now?", "METADATA"),
        
        # Additional event questions
        ("Has anyone raided recently?", "EVENT"),
        ("Any subscriptions today?", "EVENT"),
        
        # Additional transcript questions
        ("What are they talking about?", "TRANSCRIPT"),
        ("What's the current topic of discussion?", "TRANSCRIPT"),
        ("What game are they discussing?", "TRANSCRIPT"),
        
        # Inference questions
        ("What's happening right now?", "INFERENCE"),
        ("Are they winning?", "INFERENCE"),
        ("How's the stream going?", "INFERENCE"),
        ("What's the mood of the stream?", "INFERENCE"),
        
        # Edge cases
        ("What's their favorite color?", "NO_INFO"),
        ("What's the weather like?", "NO_INFO"),
    ]
    
    print("=" * 80)
    print("Testing Enhanced Prompts")
    print("=" * 80)
    print(f"Channel: {channel_id}\n")
    
    results = []
    for question, qtype in test_questions:
        print(f"\n[{qtype}] {question}")
        print("-" * 80)
        result = test_question(channel_id, question, qtype)
        results.append(result)
        
        print(f"Answer: {result['answer']}")
        if result.get("chunks_count"):
            print(f"Chunks: {result['chunks_count']}, Context entries: {result['context_count']}")
            if result["has_citations"]:
                print("[OK] Includes citations")
        
        time.sleep(0.5)  # Rate limiting
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    by_type = {}
    for r in results:
        qtype = r["type"]
        if qtype not in by_type:
            by_type[qtype] = []
        by_type[qtype].append(r)
    
    for qtype, questions in by_type.items():
        print(f"\n{qtype} Questions ({len(questions)}):")
        for q in questions:
            has_answer = not q.get("error") and "don't have enough" not in q["answer"].lower()
            status = "[OK]" if has_answer else "[NO]"
            print(f"  {status} {q['question']}")
            answer_preview = q['answer'][:100].replace('\n', ' ')
            print(f"     -> {answer_preview}...")
    
    # Save results
    with open("docs/enhanced_prompts_test_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str, ensure_ascii=False)
    
    print(f"\n\nFull results saved to: docs/enhanced_prompts_test_results.json")


if __name__ == "__main__":
    main()
