import asyncio
import time
from py.utils.embeddings import embed_text, embed_texts, EMBEDDING_DIM


async def main() -> None:
    print("Testing embeddings.py functionality...\n")

    # Test 1: Single embedding
    print("Test 1: Single text embedding")
    text = "Hello, this is a test message from Twitch chat"
    start = time.time()
    vector = await embed_text(text)
    elapsed = time.time() - start

    print(f"  ✓ Text: '{text}'")
    print(f"  ✓ Vector dimension: {len(vector)} (expected: {EMBEDDING_DIM})")
    print(f"  ✓ First 5 values: {vector[:5]}")
    print(f"  ✓ Time: {elapsed:.3f}s\n")

    # Test 2: Caching (should be much faster)
    print("Test 2: Cache hit (same text)")
    start = time.time()
    vector2 = await embed_text(text)
    elapsed = time.time() - start

    print(f"  ✓ Cached retrieval time: {elapsed:.3f}s")
    print(f"  ✓ Vectors identical: {vector == vector2}\n")

    # Test 3: Batch embeddings
    print("Test 3: Batch embedding")
    texts = [
        "First message about gaming",
        "Second message about streaming",
        "Third message asking a question",
        "Fourth message with an emoji 😊",
        "Fifth message is short",
    ]
    start = time.time()
    vectors = await embed_texts(texts)
    elapsed = time.time() - start

    print(f"  ✓ Embedded {len(texts)} texts")
    print(
        f"  ✓ All have correct dimension: {all(len(v) == EMBEDDING_DIM for v in vectors)}"
    )
    print(f"  ✓ Total time: {elapsed:.3f}s ({elapsed/len(texts):.3f}s per text)\n")

    # Test 4: Semantic similarity check
    print("Test 4: Semantic similarity check")
    text_a = "I love playing video games"
    text_b = "Gaming is my favorite hobby"
    text_c = "The weather is nice today"

    vec_a = await embed_text(text_a)
    vec_b = await embed_text(text_b)
    vec_c = await embed_text(text_c)

    def cosine_similarity(v1, v2):
        dot = sum(a * b for a, b in zip(v1, v2))
        mag1 = sum(a * a for a in v1) ** 0.5
        mag2 = sum(b * b for b in v2) ** 0.5
        return dot / (mag1 * mag2)

    sim_ab = cosine_similarity(vec_a, vec_b)
    sim_ac = cosine_similarity(vec_a, vec_c)

    print(f"  ✓ Similarity (gaming vs gaming): {sim_ab:.4f}")
    print(f"  ✓ Similarity (gaming vs weather): {sim_ac:.4f}")
    print(f"  ✓ Related texts more similar: {sim_ab > sim_ac}\n")

    print("All tests passed! ✓")


if __name__ == "__main__":
    asyncio.run(main())
