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

