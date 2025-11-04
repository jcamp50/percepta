"""Test script to insert a test event for testing combined retrieval."""

import asyncio
from datetime import datetime, timezone
from py.memory.vector_store import VectorStore
from py.utils.embeddings import embed_text


async def main():
    vs = VectorStore()

    # Insert a test raid event
    summary = "xQc raided with 150 viewers"
    embedding = await embed_text(summary)

    event_id = await vs.insert_event(
        channel_id="672238954",
        event_type="channel.raid",
        timestamp=datetime.now(timezone.utc),
        summary=summary,
        payload_json={"test": True},
        embedding=embedding,
    )

    print(f"Test event inserted: {event_id}")
    print(f"Summary: {summary}")


if __name__ == "__main__":
    asyncio.run(main())
