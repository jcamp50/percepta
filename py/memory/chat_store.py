"""
Chat Store

Manages storage and retrieval of chat messages with embeddings.
Follows the same pattern as VectorStore and VideoStore for consistency.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import Integer, String, bindparam, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from py.database.connection import SessionLocal
from py.database.models import ChatMessage
from py.utils.logging import get_logger

logger = get_logger(__name__, category="chat")


def _vector_literal(values: List[float]) -> str:
    """Convert vector to PostgreSQL literal format."""
    if not values:
        return "[]"
    # Format with reasonable precision; pgvector parses standard floats
    return "[" + ",".join(f"{v:.8f}" for v in values) + "]"


class ChatStore:
    def __init__(
        self,
        session_factory: Optional[async_sessionmaker[AsyncSession]] = None,
    ) -> None:
        """Initialize ChatStore.

        Args:
            session_factory: Database session factory (defaults to SessionLocal)
        """
        self.session_factory: async_sessionmaker[AsyncSession] = (
            session_factory or SessionLocal
        )

    async def insert_message(
        self,
        channel_id: str,
        username: str,
        message: str,
        sent_at: datetime,
        embedding: List[float],
    ) -> str:
        """Insert a chat message into the database.

        Args:
            channel_id: Broadcaster channel ID
            username: Username who sent the message
            message: Message text content
            sent_at: Timestamp when message was sent
            embedding: Message embedding vector (1536 dimensions)

        Returns:
            Message ID as string
        """
        new_id = uuid.uuid4()

        async with self.session_factory() as session:
            entity = ChatMessage(
                id=new_id,
                channel_id=channel_id,
                username=username,
                message=message,
                sent_at=sent_at,
                embedding=embedding,
            )
            session.add(entity)
            await session.commit()

        logger.debug(
            f"Inserted chat message: {new_id} from {username} in channel {channel_id} at {sent_at}"
        )

        return str(new_id)

    async def search_messages(
        self,
        query_embedding: List[float],
        limit: int = 5,
        half_life_minutes: int = 60,
        channel_id: Optional[str] = None,
        prefilter_limit: int = 200,
    ) -> List[Dict[str, Any]]:
        """Search chat messages using vector similarity with time-decay scoring.

        Args:
            query_embedding: Query vector embedding (1536 dimensions)
            limit: Maximum number of results to return
            half_life_minutes: Half-life for time decay scoring (default: 60 minutes)
            channel_id: Optional channel ID to filter by
            prefilter_limit: Maximum candidates to consider before time decay

        Returns:
            List of message dictionaries with scores
        """
        if prefilter_limit < limit:
            prefilter_limit = max(limit, 1)

        vec_str = _vector_literal(query_embedding)
        half_life_seconds = half_life_minutes * 60

        cosine_order = "embedding <=> (:vec)::vector"
        sql = f"""
            WITH pre AS (
            SELECT id, channel_id, username, message, sent_at,
                    {cosine_order} AS dist
            FROM chat_messages
            WHERE (:channel_id IS NULL OR channel_id = :channel_id)
            ORDER BY {cosine_order}
            LIMIT :k
            )
            SELECT id, channel_id, username, message, sent_at, dist,
                dist / POWER(2, EXTRACT(EPOCH FROM (NOW() - sent_at)) / :half_life_seconds) AS score
            FROM pre
            ORDER BY score ASC
            LIMIT :limit
            """

        async with self.session_factory() as session:
            stmt = text(sql).bindparams(
                bindparam("vec", type_=String()),
                bindparam("channel_id", type_=String()),
                bindparam("k", type_=Integer()),
                bindparam("half_life_seconds", type_=Integer()),
                bindparam("limit", type_=Integer()),
            )
            result = await session.execute(
                stmt,
                {
                    "vec": vec_str,
                    "channel_id": channel_id,
                    "k": prefilter_limit,
                    "half_life_seconds": half_life_seconds,
                    "limit": limit,
                },
            )
            rows = result.mappings().all()

        output: List[Dict[str, Any]] = []
        for row in rows:
            output.append(
                {
                    "id": str(row["id"]),
                    "channel_id": row["channel_id"],
                    "username": row["username"],
                    "message": row["message"],
                    "sent_at": row["sent_at"],
                    "cosine_distance": float(row["dist"]),
                    "score": float(row["score"]),
                }
            )
        return output

    async def get_messages_by_ids(self, message_ids: List[str]) -> List[Dict[str, Any]]:
        """Get chat messages by their IDs.

        Args:
            message_ids: List of chat message UUIDs as strings

        Returns:
            List of chat message dicts with id, username, message, sent_at
        """
        if not message_ids:
            return []

        # Use IN clause - convert string UUIDs to UUID objects for PostgreSQL
        # Build parameterized query with proper UUID casting
        placeholders = ",".join([f":id{i}" for i in range(len(message_ids))])
        sql = f"""
        SELECT id, channel_id, username, message, sent_at
        FROM chat_messages
        WHERE id IN ({placeholders})
        ORDER BY sent_at
        """

        # Build parameter dict with UUID conversion
        params = {f"id{i}": uuid.UUID(msg_id) for i, msg_id in enumerate(message_ids)}

        async with self.session_factory() as session:
            result = await session.execute(text(sql), params)
            rows = result.mappings().all()

            return [
                {
                    "id": str(row["id"]),
                    "channel_id": row["channel_id"],
                    "username": row["username"],
                    "message": row["message"],
                    "sent_at": row["sent_at"],
                }
                for row in rows
            ]

    async def count_recent_messages(
        self,
        channel_id: str,
        window_seconds: int = 30,
    ) -> int:
        """Count messages sent within a window ending at NOW()."""
        sql = """
        SELECT COUNT(*) AS message_count
        FROM chat_messages
        WHERE channel_id = :channel_id
          AND sent_at >= (NOW() - (:window_seconds * INTERVAL '1 second'))
        """

        async with self.session_factory() as session:
            result = await session.execute(
                text(sql),
                {"channel_id": channel_id, "window_seconds": window_seconds},
            )
            row = result.mappings().first()
            return int(row["message_count"]) if row else 0

    async def get_range(
        self,
        channel_id: str,
        start_time: datetime,
        end_time: datetime,
    ) -> List[Dict[str, Any]]:
        """Return chat messages within a time range.

        Args:
            channel_id: Broadcaster channel ID
            start_time: Start of time range (inclusive)
            end_time: End of time range (inclusive)

        Returns:
            List of chat message dicts with id, username, message, sent_at
        """
        sql = """
        SELECT id, channel_id, username, message, sent_at
        FROM chat_messages
        WHERE channel_id = :channel_id
          AND sent_at >= :start_time
          AND sent_at < :end_time
        ORDER BY sent_at ASC
        """

        async with self.session_factory() as session:
            result = await session.execute(
                text(sql),
                {
                    "channel_id": channel_id,
                    "start_time": start_time,
                    "end_time": end_time,
                },
            )
            rows = result.mappings().all()

        return [
            {
                "id": str(row["id"]),
                "channel_id": row["channel_id"],
                "username": row["username"],
                "message": row["message"],
                "sent_at": row["sent_at"],
            }
            for row in rows
        ]
