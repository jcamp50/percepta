"""
Video Store

Manages storage and retrieval of video frames with embeddings.
Follows the same pattern as VectorStore for consistency.
"""

from __future__ import annotations

import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from sqlalchemy import Integer, String, bindparam, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from py.database.connection import SessionLocal
from py.database.models import VideoFrame
from py.utils.logging import get_logger
from py.utils.video_embeddings import generate_clip_embedding

logger = get_logger(__name__, category="video")


def _vector_literal(values: List[float]) -> str:
    """Convert vector to PostgreSQL literal format."""
    if not values:
        return "[]"
    # Format with reasonable precision; pgvector parses standard floats
    return "[" + ",".join(f"{v:.8f}" for v in values) + "]"


class VideoStore:
    def __init__(
        self,
        session_factory: Optional[async_sessionmaker[AsyncSession]] = None,
        frames_dir: str = "frames",
    ) -> None:
        """Initialize VideoStore.
        
        Args:
            session_factory: Database session factory (defaults to SessionLocal)
            frames_dir: Directory to store video frame images (default: "frames")
        """
        self.session_factory: async_sessionmaker[AsyncSession] = (
            session_factory or SessionLocal
        )
        self.frames_dir = frames_dir
        
        # Ensure frames directory exists
        Path(self.frames_dir).mkdir(parents=True, exist_ok=True)

    def _get_frame_path(self, channel_id: str, frame_id: str) -> str:
        """Get filesystem path for a video frame.
        
        Args:
            channel_id: Broadcaster channel ID
            frame_id: Frame UUID
            
        Returns:
            Relative path to frame image file
        """
        # Create channel-specific subdirectory
        channel_dir = os.path.join(self.frames_dir, channel_id)
        Path(channel_dir).mkdir(parents=True, exist_ok=True)
        
        # Use frame_id as filename
        return os.path.join(channel_dir, f"{frame_id}.jpg")

    async def insert_frame(
        self,
        channel_id: str,
        image_path: str,
        captured_at: datetime,
        embedding: Optional[List[float]] = None,
    ) -> str:
        """Insert a video frame into the database.
        
        Args:
            channel_id: Broadcaster channel ID
            image_path: Path to the image file (will be moved to frames directory)
            captured_at: Timestamp when frame was captured
            embedding: Optional pre-computed embedding (will be generated if not provided)
            
        Returns:
            Frame ID as string
        """
        new_id = uuid.uuid4()
        
        # Generate embedding if not provided
        if embedding is None:
            # Validate image file exists
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image file not found: {image_path}")
            
            embedding = await generate_clip_embedding(image_path)
        
        # Determine destination path
        dest_path = self._get_frame_path(channel_id, str(new_id))
        
        # Move/copy image to frames directory
        import shutil
        shutil.move(image_path, dest_path)
        
        # Store in database
        async with self.session_factory() as session:
            entity = VideoFrame(
                id=new_id,
                channel_id=channel_id,
                captured_at=captured_at,
                image_path=dest_path,
                embedding=embedding,
            )
            session.add(entity)
            await session.commit()
        
        logger.debug(
            f"Inserted video frame: {new_id} for channel {channel_id} at {captured_at}"
        )
        
        return str(new_id)

    async def search_frames(
        self,
        query_embedding: List[float],
        limit: int = 5,
        half_life_minutes: int = 60,
        channel_id: Optional[str] = None,
        prefilter_limit: int = 200,
    ) -> List[Dict[str, Any]]:
        """Search video frames using vector similarity with time-decay scoring.
        
        Args:
            query_embedding: Query vector embedding (1536 dimensions)
            limit: Maximum number of results to return
            half_life_minutes: Half-life for time decay scoring (default: 60 minutes)
            channel_id: Optional channel ID to filter by
            prefilter_limit: Maximum candidates to consider before time decay
            
        Returns:
            List of frame dictionaries with scores
        """
        if prefilter_limit < limit:
            prefilter_limit = max(limit, 1)

        vec_str = _vector_literal(query_embedding)
        half_life_seconds = half_life_minutes * 60

        cosine_order = "embedding <=> (:vec)::vector"
        sql = f"""
            WITH pre AS (
            SELECT id, channel_id, captured_at, image_path,
                    {cosine_order} AS dist
            FROM video_frames
            WHERE (:channel_id IS NULL OR channel_id = :channel_id)
            ORDER BY {cosine_order}
            LIMIT :k
            )
            SELECT id, channel_id, captured_at, image_path, dist,
                dist / POWER(2, EXTRACT(EPOCH FROM (NOW() - captured_at)) / :half_life_seconds) AS score
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
                    "captured_at": row["captured_at"],
                    "image_path": row["image_path"],
                    "cosine_distance": float(row["dist"]),
                    "score": float(row["score"]),
                }
            )
        return output

