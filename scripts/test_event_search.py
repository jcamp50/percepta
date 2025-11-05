"""Test script to verify event search and combined retrieval."""

import asyncio
from py.memory.vector_store import VectorStore
from py.utils.embeddings import embed_text
from py.reason.retriever import Retriever, RetrievalParams


async def main():
    vs = VectorStore()
    retriever = Retriever(vector_store=vs)

    # Test query
    query = "Who raided?"
    query_embedding = await embed_text(query)

    params = RetrievalParams(
        channel_id="672238954",
        limit=10,
        half_life_minutes=60,
        prefilter_limit=50,
    )

    print("Testing combined retrieval...")
    results = await retriever.retrieve_combined(
        query_embedding=query_embedding,
        params=params,
    )

    print(f"\nTotal results: {len(results)}")
    print("\nResults:")
    for i, r in enumerate(results[:5], 1):
        source = (
            "EVENT" if r.id == "76d42cd3-bb5e-4752-9fbd-ed024ab9bd2a" else "TRANSCRIPT"
        )
        print(f"{i}. [{source}] {r.text[:80]}... (score: {r.score:.4f})")

    # Test event search separately
    print("\n\nTesting event search separately...")
    event_results = await retriever.from_events(
        query_embedding=query_embedding,
        params=params,
    )
    print(f"Event results: {len(event_results)}")
    for r in event_results:
        print(f"  - {r.text} (score: {r.score:.4f})")


if __name__ == "__main__":
    asyncio.run(main())
