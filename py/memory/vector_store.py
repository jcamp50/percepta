from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import text, String, Integer, bindparam
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from py.database.connection import SessionLocal
from py.database.models import Transcript, ChannelSnapshot, Event


def _vector_literal(values: List[float]) -> str:
    if not values:
        return "[]"
    # Format with reasonable precision; pgvector parses standard floats
    return "[" + ",".join(f"{v:.8f}" for v in values) + "]"


class VectorStore:
    def __init__(
        self, session_factory: Optional[async_sessionmaker[AsyncSession]] = None
    ) -> None:
        self.session_factory: async_sessionmaker[AsyncSession] = (
            session_factory or SessionLocal
        )

    async def insert_transcript(
        self,
        channel_id: str,
        text_value: str,
        start_time: datetime,
        end_time: datetime,
        embedding: List[float],
    ) -> str:
        new_id = uuid.uuid4()
        async with self.session_factory() as session:
            entity = Transcript(
                id=new_id,
                channel_id=channel_id,
                started_at=start_time,
                ended_at=end_time,
                text=text_value,
                embedding=embedding,
            )
            session.add(entity)
            await session.commit()
        return str(new_id)

    async def search_transcripts(
        self,
        query_embedding: List[float],
        limit: int = 5,
        half_life_minutes: int = 60,
        channel_id: Optional[str] = None,
        prefilter_limit: int = 200,
    ) -> List[Dict[str, Any]]:
        if prefilter_limit < limit:
            prefilter_limit = max(limit, 1)

        vec_str = _vector_literal(query_embedding)
        half_life_seconds = half_life_minutes * 60

        cosine_order = "embedding <=> (:vec)::vector"
        sql = f"""
            WITH pre AS (
            SELECT id, channel_id, text, started_at, ended_at,
                    {cosine_order} AS dist
            FROM transcripts
            WHERE (:channel_id IS NULL OR channel_id = :channel_id)
            ORDER BY {cosine_order}
            LIMIT :k
            )
            SELECT id, channel_id, text, started_at, ended_at, dist,
                dist / POWER(2, EXTRACT(EPOCH FROM (NOW() - ended_at)) / :half_life_seconds) AS score
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
                    "text": row["text"],
                    "started_at": row["started_at"],
                    "ended_at": row["ended_at"],
                    "cosine_distance": float(row["dist"]),
                    "score": float(row["score"]),
                }
            )
        return output

    async def delete_old_transcripts(
        self, older_than_minutes: int, channel_id: Optional[str] = None
    ) -> int:
        sql = """
        DELETE FROM transcripts
        WHERE ended_at < NOW() - make_interval(mins => :minutes)
        AND (:channel_id IS NULL OR channel_id = :channel_id)
        """
        async with self.session_factory() as session:
            result = await session.execute(
                text(sql), {"minutes": older_than_minutes, "channel_id": channel_id}
            )
            await session.commit()
            # rowcount can be -1 with some drivers; coerce to int >= 0
            count = result.rowcount if result.rowcount is not None else 0
        return int(count)

    async def insert_channel_snapshot(
        self,
        channel_id: str,
        title: Optional[str] = None,
        game_id: Optional[str] = None,
        game_name: Optional[str] = None,
        tags: Optional[List[str]] = None,
        viewer_count: Optional[int] = None,
        payload_json: Optional[Dict[str, Any]] = None,
        embedding: Optional[List[float]] = None,
    ) -> str:
        """
        Insert a channel snapshot into the database.

        Args:
            channel_id: Broadcaster user ID
            title: Stream title
            game_id: Game/category ID
            game_name: Game/category name
            tags: List of stream tags
            viewer_count: Current viewer count (if live)
            payload_json: Full API response for debugging
            embedding: Vector embedding (optional, for JCB-21)

        Returns:
            Snapshot ID as string
        """
        new_id = uuid.uuid4()
        async with self.session_factory() as session:
            entity = ChannelSnapshot(
                id=new_id,
                channel_id=channel_id,
                ts=datetime.now(),
                title=title,
                game_id=game_id,
                game_name=game_name,
                tags=tags,
                viewer_count=viewer_count,
                payload_json=payload_json,
                embedding=embedding,
            )
            session.add(entity)
            await session.commit()
        return str(new_id)

    async def get_latest_channel_snapshot(
        self, channel_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get the most recent channel snapshot for a channel.

        Args:
            channel_id: Broadcaster user ID

        Returns:
            Dictionary with snapshot data, or None if not found
        """
        sql = """
        SELECT id, channel_id, ts, title, game_id, game_name, tags, 
               viewer_count, payload_json
        FROM channel_snapshots
        WHERE channel_id = :channel_id
        ORDER BY ts DESC
        LIMIT 1
        """
        async with self.session_factory() as session:
            result = await session.execute(text(sql), {"channel_id": channel_id})
            row = result.mappings().first()

            if row:
                return {
                    "id": str(row["id"]),
                    "channel_id": row["channel_id"],
                    "ts": row["ts"],
                    "title": row["title"],
                    "game_id": row["game_id"],
                    "game_name": row["game_name"],
                    "tags": row["tags"],
                    "viewer_count": row["viewer_count"],
                    "payload_json": row["payload_json"],
                }
            return None

    async def insert_event(
        self,
        channel_id: str,
        event_type: str,
        timestamp: datetime,
        summary: str,
        payload_json: Optional[Dict[str, Any]] = None,
        embedding: Optional[List[float]] = None,
    ) -> str:
        """
        Insert an event into the database.

        Args:
            channel_id: Broadcaster user ID
            event_type: Type of event (e.g., "stream.online")
            timestamp: Event timestamp
            summary: Human-readable summary
            payload_json: Raw event payload for debugging
            embedding: Vector embedding (optional)

        Returns:
            Event ID as string
        """
        new_id = uuid.uuid4()
        async with self.session_factory() as session:
            entity = Event(
                id=new_id,
                channel_id=channel_id,
                ts=timestamp,
                type=event_type,
                summary=summary,
                payload_json=payload_json,
                embedding=embedding,
            )
            session.add(entity)
            await session.commit()
        return str(new_id)

    async def search_events(
        self,
        query_embedding: List[float],
        limit: int = 5,
        half_life_minutes: int = 60,
        channel_id: Optional[str] = None,
        prefilter_limit: int = 200,
    ) -> List[Dict[str, Any]]:
        """
        Search events using vector similarity with time-decay scoring.

        Args:
            query_embedding: Query vector embedding
            limit: Maximum number of results to return
            half_life_minutes: Half-life for time decay scoring
            channel_id: Optional channel ID to filter by
            prefilter_limit: Maximum candidates to consider before time decay

        Returns:
            List of event dictionaries with scores
        """
        if prefilter_limit < limit:
            prefilter_limit = max(limit, 1)

        vec_str = _vector_literal(query_embedding)
        half_life_seconds = half_life_minutes * 60

        # Only search events that have embeddings (non-null)
        cosine_order = "embedding <=> (:vec)::vector"
        sql = f"""
            WITH pre AS (
            SELECT id, channel_id, ts, type, summary,
                    {cosine_order} AS dist
            FROM events
            WHERE embedding IS NOT NULL
            AND (:channel_id IS NULL OR channel_id = :channel_id)
            ORDER BY {cosine_order}
            LIMIT :k
            )
            SELECT id, channel_id, ts, type, summary, dist,
                dist / POWER(2, EXTRACT(EPOCH FROM (NOW() - ts)) / :half_life_seconds) AS score
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
                    "ts": row["ts"],
                    "type": row["type"],
                    "summary": row["summary"],
                    "cosine_distance": float(row["dist"]),
                    "score": float(row["score"]),
                }
            )
        return output

    async def get_transcript_by_id(self, transcript_id: str) -> Optional[Dict[str, Any]]:
        """Get a transcript by its ID.
        
        Args:
            transcript_id: Transcript UUID as string
            
        Returns:
            Transcript dict with id, channel_id, text, started_at, ended_at, or None if not found
        """
        sql = """
        SELECT id, channel_id, started_at, ended_at, text
        FROM transcripts
        WHERE id = :transcript_id
        """
        
        async with self.session_factory() as session:
            result = await session.execute(
                text(sql), {"transcript_id": transcript_id}
            )
            row = result.mappings().first()
            
            if row:
                return {
                    "id": str(row["id"]),
                    "channel_id": row["channel_id"],
                    "started_at": row["started_at"],
                    "ended_at": row["ended_at"],
                    "text": row["text"],
                }
            return None

    async def get_recent_transcript_text(
        self,
        channel_id: str,
        window_seconds: int = 30,
    ) -> Optional[str]:
        """Return the most recent transcript text within a time window."""
        sql = """
        SELECT text
        FROM transcripts
        WHERE channel_id = :channel_id
          AND ended_at >= (NOW() - (:window_seconds * INTERVAL '1 second'))
        ORDER BY ended_at DESC
        LIMIT 1
        """

        async with self.session_factory() as session:
            result = await session.execute(
                text(sql),
                {"channel_id": channel_id, "window_seconds": window_seconds},
            )
            row = result.mappings().first()
            if row:
                return row["text"]
            return None