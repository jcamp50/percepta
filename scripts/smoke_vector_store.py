import asyncio
from datetime import datetime, timedelta
from typing import List

from py.memory.vector_store import VectorStore


def zero_vector(dim: int = 1536) -> List[float]:
    return [1.0] + [0.0] * (dim - 1)


async def main() -> None:
    store = VectorStore()

    now = datetime.utcnow()
    start = now - timedelta(seconds=15)
    end = now

    embedding = zero_vector()

    transcript_id = await store.insert_transcript(
        channel_id="test_channel",
        text_value="hello world",
        start_time=start,
        end_time=end,
        embedding=embedding,
    )
    print({"inserted_id": transcript_id})

    results = await store.search_transcripts(
        query_embedding=embedding,
        limit=3,
        half_life_minutes=60,
        channel_id="test_channel",
        prefilter_limit=50,
    )
    print({"top_results": results})

    # Optional cleanup example (commented out by default):
    # deleted = await store.delete_old_transcripts(older_than_minutes=1, channel_id="test_channel")
    # print({"deleted": deleted})


if __name__ == "__main__":
    asyncio.run(main())
