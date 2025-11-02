import asyncio
import uuid
from sqlalchemy import text
from py.database.connection import engine


def zero_vector_literal(dimensions: int = 1536) -> str:
    # First element is 1.0, rest are 0
    vec_values = ["1.0"] + ["0.0"] * (dimensions - 1)
    return "[" + ",".join(vec_values) + "]"


async def main() -> None:
    vec = zero_vector_literal()
    new_id = str(uuid.uuid4())

    async with engine.begin() as conn:
        # Ensure extension exists (harmless if already present)
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))

        # Insert a test transcript row
        await conn.execute(
            text(
                """
                INSERT INTO transcripts (id, channel_id, started_at, ended_at, text, embedding)
                VALUES (:id, :channel_id, NOW() - INTERVAL '15 seconds', NOW(), :text, (:vec)::vector)
                """
            ),
            {
                "id": new_id,
                "channel_id": "test_channel",
                "text": "hello world",
                "vec": vec,
            },
        )

        # Query top-1 by cosine distance against the same vector
        result = await conn.execute(
            text(
                """
                SELECT id
                FROM transcripts
                ORDER BY embedding <=> (:vec)::vector
                LIMIT 1
                """
            ),
            {"vec": vec},
        )

        top = result.scalar_one_or_none()
        print({"inserted_id": new_id, "top_match": top})


if __name__ == "__main__":
    asyncio.run(main())
