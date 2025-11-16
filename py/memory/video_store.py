"""
Video Store

Manages storage and retrieval of video frames with embeddings.
Follows the same pattern as VectorStore for consistency.
"""

from __future__ import annotations

import asyncio
import time
import os
import uuid
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image
from sqlalchemy import Integer, String, bindparam, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from py.database.connection import SessionLocal
from py.database.models import VideoFrame, Transcript, ChatMessage, ChannelSnapshot
from py.config import settings
from py.utils.logging import get_logger
from py.utils.video_embeddings import generate_clip_embedding, create_grounded_embedding
from py.utils.video_descriptions import generate_frame_description
from py.utils.embeddings import embed_text

logger = get_logger(__name__, category="video")
description_logger = get_logger(__name__, category="video_description")

DEFAULT_KEYWORDS: Tuple[str, ...] = (
    "boss",
    "clutch",
    "hype",
    "raid",
    "gg",
    "crazy",
    "omg",
    "insane",
)


def _vector_literal(values: List[float]) -> str:
    """Convert vector to PostgreSQL literal format."""
    if not values:
        return "[]"
    # Format with reasonable precision; pgvector parses standard floats
    return "[" + ",".join(f"{v:.8f}" for v in values) + "]"


def _json_to_text_for_embedding(json_dict: Dict[str, Any]) -> str:
    """Convert JSON description dict to text representation for embeddings.
    
    This preserves semantic markers (field names) while creating a readable
    text representation suitable for embedding generation.
    
    Args:
        json_dict: JSON dict with video frame description structure
        
    Returns:
        Text representation of the JSON structure
    """
    if not json_dict:
        return ""
    
    parts: List[str] = []
    
    # High-level summary
    if high_level := json_dict.get("high_level_summary"):
        parts.append(f"Summary: {high_level}")
    
    # Participants
    if participants := json_dict.get("participants"):
        if streamer := participants.get("streamer"):
            appearance = streamer.get("appearance", "")
            posture = streamer.get("posture", "")
            expression = streamer.get("expression", "")
            if appearance or posture or expression:
                parts.append(f"Streamer: {appearance}")
                if posture:
                    parts.append(f"Posture: {posture}")
                if expression:
                    parts.append(f"Expression: {expression}")
    
    # Primary scene
    if primary_scene := json_dict.get("primary_scene"):
        activity = primary_scene.get("activity", "")
        location = primary_scene.get("location", "")
        actions = primary_scene.get("actions", [])
        if activity:
            parts.append(f"Activity: {activity}")
        if location:
            parts.append(f"Location: {location}")
        if actions:
            parts.append(f"Actions: {', '.join(actions)}")
    
    # On-screen media
    if on_screen_media := json_dict.get("on_screen_media"):
        if isinstance(on_screen_media, list) and on_screen_media:
            media_descriptions = []
            for media in on_screen_media:
                if isinstance(media, dict):
                    media_type = media.get("type", "")
                    title = media.get("title", "")
                    desc = media.get("description", "")
                    if media_type or title or desc:
                        media_descriptions.append(f"{media_type}: {title} - {desc}")
            if media_descriptions:
                parts.append(f"On-screen media: {'; '.join(media_descriptions)}")
    
    # Environment
    if environment := json_dict.get("environment"):
        physical_space = environment.get("physical_space", "")
        lighting = environment.get("lighting", "")
        props = environment.get("props", [])
        if physical_space:
            parts.append(f"Environment: {physical_space}")
        if lighting:
            parts.append(f"Lighting: {lighting}")
        if props:
            parts.append(f"Props: {', '.join(props)}")
    
    # Overlays
    if overlays := json_dict.get("overlays"):
        overlay_parts = []
        if alerts := overlays.get("alerts"):
            overlay_parts.append(f"Alerts: {', '.join(alerts)}")
        if captions := overlays.get("captions"):
            overlay_parts.append(f"Captions: {', '.join(captions)}")
        if notifications := overlays.get("notifications"):
            overlay_parts.append(f"Notifications: {', '.join(notifications)}")
        if gameplay_hud := overlays.get("gameplay_hud"):
            hud_parts = []
            for key, value in gameplay_hud.items():
                if value:
                    hud_parts.append(f"{key}: {value}")
            if hud_parts:
                overlay_parts.append(f"HUD: {', '.join(hud_parts)}")
        if overlay_parts:
            parts.append(f"Overlays: {'; '.join(overlay_parts)}")
    
    # Overall impression
    if impression := json_dict.get("overall_impression"):
        parts.append(f"Impression: {impression}")
    
    return " | ".join(parts)


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
        self._description_locks: Dict[str, asyncio.Lock] = {}

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

    async def _find_aligned_transcript(
        self,
        channel_id: str,
        captured_at: datetime,
        window_seconds: int = 5,
    ) -> Optional[Dict[str, Any]]:
        """Find transcript aligned to a video frame timestamp.

        Args:
            channel_id: Broadcaster channel ID
            captured_at: Timestamp when frame was captured
            window_seconds: Time window for alignment (±seconds)

        Returns:
            Transcript dict with id, text, started_at, ended_at, or None if not found
        """
        sql = """
        SELECT id, channel_id, started_at, ended_at, text
        FROM transcripts
        WHERE channel_id = :channel_id
        AND (
            -- Frame falls within transcript time range
            (started_at <= :captured_at AND ended_at >= :captured_at)
            OR
            -- Transcript midpoint is within window
            ABS(EXTRACT(EPOCH FROM (
                (started_at + (ended_at - started_at) / 2) - :captured_at
            ))) <= :window_seconds
        )
        ORDER BY
            -- Prefer transcripts that contain the timestamp
            CASE WHEN started_at <= :captured_at AND ended_at >= :captured_at THEN 0 ELSE 1 END,
            -- Then by closest midpoint
            ABS(EXTRACT(EPOCH FROM (
                (started_at + (ended_at - started_at) / 2) - :captured_at
            )))
        LIMIT 1
        """

        async with self.session_factory() as session:
            result = await session.execute(
                text(sql),
                {
                    "channel_id": channel_id,
                    "captured_at": captured_at,
                    "window_seconds": window_seconds,
                },
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

    async def _find_aligned_chat(
        self,
        channel_id: str,
        captured_at: datetime,
        window_seconds: int = 5,
    ) -> List[Dict[str, Any]]:
        """Find chat messages aligned to a video frame timestamp.

        Args:
            channel_id: Broadcaster channel ID
            captured_at: Timestamp when frame was captured
            window_seconds: Time window for alignment (±seconds)

        Returns:
            List of chat message dicts with id, username, message, sent_at
        """
        sql = """
        SELECT id, channel_id, username, message, sent_at
        FROM chat_messages
        WHERE channel_id = :channel_id
        AND ABS(EXTRACT(EPOCH FROM (sent_at - :captured_at))) <= :window_seconds
        ORDER BY ABS(EXTRACT(EPOCH FROM (sent_at - :captured_at)))
        LIMIT 50
        """

        async with self.session_factory() as session:
            result = await session.execute(
                text(sql),
                {
                    "channel_id": channel_id,
                    "captured_at": captured_at,
                    "window_seconds": window_seconds,
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

    async def _get_metadata_at_time(
        self,
        channel_id: str,
        captured_at: datetime,
    ) -> Optional[Dict[str, Any]]:
        """Get channel metadata snapshot closest to (but not after) a timestamp.

        Args:
            channel_id: Broadcaster channel ID
            captured_at: Timestamp when frame was captured

        Returns:
            Metadata dict with snapshot data, or None if not found
        """
        sql = """
        SELECT id, channel_id, ts, title, game_id, game_name, tags, 
               viewer_count, payload_json
        FROM channel_snapshots
        WHERE channel_id = :channel_id
        AND ts <= :captured_at
        ORDER BY ts DESC
        LIMIT 1
        """

        async with self.session_factory() as session:
            result = await session.execute(
                text(sql),
                {
                    "channel_id": channel_id,
                    "captured_at": captured_at,
                },
            )
            row = result.mappings().first()

            if row:
                return {
                    "id": str(row["id"]),
                    "channel_id": row["channel_id"],
                    "ts": (
                        row["ts"].isoformat() if row["ts"] else None
                    ),  # Convert datetime to ISO string
                    "title": row["title"],
                    "game_id": row["game_id"],
                    "game_name": row["game_name"],
                    "tags": row["tags"],
                    "viewer_count": row["viewer_count"],
                    "payload_json": row["payload_json"],
                }
            return None

    async def insert_frame(
        self,
        channel_id: str,
        image_path: str,
        captured_at: datetime,
        embedding: Optional[List[float]] = None,
        *,
        is_interesting: Optional[bool] = None,
        recent_summary: Optional[str] = None,
    ) -> str:
        """Insert a video frame into the database with temporally-aligned context.

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

        # Compute perceptual hash for similarity detection
        frame_hash = None
        try:
            frame_hash = self._compute_perceptual_hash(dest_path)
        except Exception as exc:
            logger.warning(
                f"Failed to compute perceptual hash for frame {new_id}: {exc}",
                exc_info=True,
            )

        # Find temporally-aligned context (±5 seconds)
        transcript = None
        chat: List[Dict[str, Any]] = []
        transcript_id = None
        aligned_chat_ids = None
        metadata_snapshot = None

        try:
            transcript = await self._find_aligned_transcript(
                channel_id, captured_at, window_seconds=5
            )
            if transcript:
                transcript_id = uuid.UUID(transcript["id"])
        except Exception as exc:
            logger.warning(
                f"Failed to find aligned transcript for frame {new_id}: {exc}",
                exc_info=True,
            )

        try:
            chat = await self._find_aligned_chat(
                channel_id, captured_at, window_seconds=5
            )
            if chat:
                # Extract chat message IDs as strings (UUIDs)
                aligned_chat_ids = [c["id"] for c in chat]
        except Exception as exc:
            logger.warning(
                f"Failed to find aligned chat for frame {new_id}: {exc}",
                exc_info=True,
            )

        try:
            metadata = await self._get_metadata_at_time(channel_id, captured_at)
            if metadata:
                # Store full metadata snapshot as JSONB
                metadata_snapshot = metadata
        except Exception as exc:
            logger.warning(
                f"Failed to get metadata snapshot for frame {new_id}: {exc}",
                exc_info=True,
            )

        # Determine whether we can reuse a cached description
        description_json = None
        description = None
        description_source = None
        similar_frame_id: Optional[str] = None

        if frame_hash:
            try:
                similar_frame = await self._find_similar_frame(
                    channel_id=channel_id,
                    frame_hash=frame_hash,
                    captured_at=captured_at,
                    window_seconds=settings.video_frame_hash_window_seconds,
                    max_distance=settings.video_frame_hash_max_distance,
                    require_description=True,
                )
                if similar_frame:
                    # Try to get JSON description first, fall back to text
                    similar_frame_id = similar_frame["id"]
                    async with self.session_factory() as session:
                        similar_frame_obj = await session.get(VideoFrame, uuid.UUID(similar_frame_id))
                        if similar_frame_obj and similar_frame_obj.description_json:
                            description_json = similar_frame_obj.description_json
                            description = _json_to_text_for_embedding(description_json)
                        elif similar_frame.get("description"):
                            # Old format - just text, convert to JSON-like structure
                            description = similar_frame["description"]
                            # For cached descriptions, we'll store as text only
                            # The JSON will be regenerated if needed
                    description_source = "cache"
                    description_logger.info(
                        "Frame %s reused description from frame %s (hash distance <= %s)",
                        new_id,
                        similar_frame_id,
                        settings.video_frame_hash_max_distance,
                    )
            except Exception as exc:
                logger.warning(
                    f"Failed to find similar frame for {new_id}: {exc}", exc_info=True
                )

        # Decide if we need to generate a new description immediately
        interesting_flag = bool(is_interesting) or self._determine_interesting_frame(
            chat, transcript
        )

        if description_json is None:
            if interesting_flag:
                try:
                    previous_description = await self._get_previous_frame_description(
                        channel_id, captured_at
                    )
                    description_logger.info(
                        "Generating immediate description for frame %s (interesting=%s, chat=%s)",
                        new_id,
                        interesting_flag,
                        len(chat) if chat else 0,
                    )
                    description_json = await self._generate_description(
                        image_path=dest_path,
                        channel_id=channel_id,
                        captured_at=captured_at,
                        previous_frame_description=previous_description,
                        recent_summary=recent_summary,
                        transcript=transcript,
                        chat=chat,
                        metadata=metadata_snapshot,
                    )
                    # Generate text representation from JSON
                    description = _json_to_text_for_embedding(description_json)
                    description_source = "immediate"
                except Exception as exc:
                    logger.warning(
                        f"Failed to generate immediate description for frame {new_id}: {exc}",
                        exc_info=True,
                    )
                    description_logger.warning(
                        "Immediate description failed for frame %s: %s (deferring)",
                        new_id,
                        exc,
                    )
                    description_json = None
                    description = None
                    description_source = "lazy"
                else:
                    description_source = "lazy"
                    description_logger.info(
                        "Queued frame %s for lazy description (chat=%s, interesting=%s)",
                        new_id,
                        len(chat) if chat else 0,
                        interesting_flag,
                    )

        # Generate grounded embedding if description is available
        grounded_embedding = None
        if description and description.strip():
            try:
                grounded_embedding = await create_grounded_embedding(
                    clip_embedding=embedding,
                    description_text=description,
                )
                logger.debug(
                    f"Generated grounded embedding for frame {new_id}"
                )
            except Exception as exc:
                logger.warning(
                    f"Failed to generate grounded embedding for frame {new_id}: {exc}",
                    exc_info=True,
                )
                # Continue without grounded embedding - frame will still be stored

        # Store in database with context references
        async with self.session_factory() as session:
            entity = VideoFrame(
                id=new_id,
                channel_id=channel_id,
                captured_at=captured_at,
                image_path=dest_path,
                embedding=embedding,
                grounded_embedding=grounded_embedding,
                description=description,
                description_json=description_json,
                description_source=description_source,
                frame_hash=frame_hash,
                transcript_id=transcript_id,
                aligned_chat_ids=aligned_chat_ids,
                metadata_snapshot=metadata_snapshot,
            )
            session.add(entity)
            await session.commit()

        if description_source == "lazy":
            self._schedule_background_description_generation(str(new_id))

        description_logger.info(
            "Stored frame %s (channel=%s, description_source=%s, reused_from=%s)",
            new_id,
            channel_id,
            description_source,
            similar_frame_id,
        )

        logger.debug(
            f"Inserted video frame: {new_id} for channel {channel_id} at {captured_at} "
            f"(transcript: {transcript_id is not None}, "
            f"chat: {len(aligned_chat_ids) if aligned_chat_ids else 0} messages, "
            f"metadata: {metadata_snapshot is not None}, "
            f"description_source: {description_source})"
        )

        chat_count = len(chat) if chat else 0
        transcript_text = transcript["text"] if transcript else None

        return {
            "frame_id": str(new_id),
            "description_source": description_source,
            "description": description,
            "chat_count": chat_count,
            "transcript_text": transcript_text,
            "was_interesting": interesting_flag,
            "reused_description": description_source == "cache",
        }

    async def search_frames(
        self,
        query_embedding: List[float],
        limit: int = 5,
        half_life_minutes: int = 60,
        channel_id: Optional[str] = None,
        prefilter_limit: int = 200,
        use_grounded: bool = True,
    ) -> List[Dict[str, Any]]:
        """Search video frames using vector similarity with time-decay scoring.

        Args:
            query_embedding: Query vector embedding (1536 dimensions)
            limit: Maximum number of results to return
            half_life_minutes: Half-life for time decay scoring (default: 60 minutes)
            channel_id: Optional channel ID to filter by
            prefilter_limit: Maximum candidates to consider before time decay
            use_grounded: If True, use grounded embeddings (preferring grounded_embedding, falling back to embedding).
                         If False, use pure CLIP embeddings only.

        Returns:
            List of frame dictionaries with scores
        """
        if prefilter_limit < limit:
            prefilter_limit = max(limit, 1)

        vec_str = _vector_literal(query_embedding)
        half_life_seconds = half_life_minutes * 60

        # Choose embedding column based on use_grounded flag
        # If use_grounded is True, prefer grounded_embedding but fall back to embedding if not available
        if use_grounded:
            # Use COALESCE to prefer grounded_embedding, fall back to embedding
            cosine_order = "COALESCE(grounded_embedding, embedding) <=> (:vec)::vector"
            embedding_select = "COALESCE(grounded_embedding, embedding) <=> (:vec)::vector AS dist"
        else:
            # Use pure CLIP embedding
            cosine_order = "embedding <=> (:vec)::vector"
            embedding_select = "embedding <=> (:vec)::vector AS dist"

        # Build WHERE clause for embedding existence check
        if use_grounded:
            embedding_where = "AND (grounded_embedding IS NOT NULL OR embedding IS NOT NULL)"
        else:
            embedding_where = "AND embedding IS NOT NULL"

        sql = f"""
            WITH pre AS (
            SELECT id, channel_id, captured_at, image_path, transcript_id, aligned_chat_ids, metadata_snapshot,
                    description, description_source, frame_hash,
                    {embedding_select}
            FROM video_frames
            WHERE (:channel_id IS NULL OR channel_id = :channel_id)
            {embedding_where}
            ORDER BY {cosine_order}
            LIMIT :k
            )
            SELECT id, channel_id, captured_at, image_path, transcript_id, aligned_chat_ids, metadata_snapshot,
                   description, description_source, frame_hash, dist,
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
                    "description": row["description"],
                    "description_source": row["description_source"],
                    "frame_hash": row["frame_hash"],
                    "transcript_id": (
                        str(row["transcript_id"]) if row["transcript_id"] else None
                    ),
                    "aligned_chat_ids": (
                        row["aligned_chat_ids"] if row["aligned_chat_ids"] else None
                    ),
                    "metadata_snapshot": row["metadata_snapshot"],
                    "cosine_distance": float(row["dist"]),
                    "score": float(row["score"]),
                }
            )
        return output

    def _compute_perceptual_hash(
        self, image_path: str, hash_size: int = 16
    ) -> Optional[str]:
        """Compute a simple average hash (aHash) for perceptual similarity."""
        try:
            with Image.open(image_path) as img:
                img = img.convert("L")
                resample = getattr(Image, "Resampling", Image).LANCZOS
                img = img.resize((hash_size, hash_size), resample=resample)  # type: ignore[arg-type]
                pixels = list(img.getdata())
                avg = sum(pixels) / len(pixels)
                bits = ["1" if pixel >= avg else "0" for pixel in pixels]
                bitstring = "".join(bits)
                hash_int = int(bitstring, 2)
                hex_length = (hash_size * hash_size) // 4
                return f"{hash_int:0{hex_length}x}"
        except Exception as exc:
            logger.warning(f"Failed to compute perceptual hash for {image_path}: {exc}")
            return None

    def _hamming_distance(self, hash_a: str, hash_b: str) -> int:
        """Compute hamming distance between two hex hashes."""
        if hash_a is None or hash_b is None:
            return 1_000_000
        try:
            int_a = int(hash_a, 16)
            int_b = int(hash_b, 16)
            return (int_a ^ int_b).bit_count()
        except ValueError:
            return 1_000_000

    async def _find_similar_frame(
        self,
        *,
        channel_id: str,
        frame_hash: Optional[str],
        captured_at: datetime,
        window_seconds: int = 60,
        max_distance: int = 8,
        require_description: bool = True,
    ) -> Optional[Dict[str, Any]]:
        """Find a previously stored frame with similar perceptual hash."""
        if not frame_hash:
            return None

        sql = """
            SELECT id, frame_hash, description, description_source, captured_at
        FROM video_frames
        WHERE channel_id = :channel_id
          AND frame_hash IS NOT NULL
          AND captured_at >= :window_start
          AND captured_at <= :captured_at
        ORDER BY captured_at DESC
        LIMIT 200
        """

        window_start = captured_at - timedelta(seconds=window_seconds)

        async with self.session_factory() as session:
            result = await session.execute(
                text(sql),
                {
                    "channel_id": channel_id,
                    "captured_at": captured_at,
                    "window_start": window_start,
                },
            )
            rows = result.mappings().all()

        best_match: Optional[Dict[str, Any]] = None
        best_distance = max_distance + 1

        for row in rows:
            existing_hash = row["frame_hash"]
            distance = self._hamming_distance(frame_hash, existing_hash)
            if distance <= max_distance and distance < best_distance:
                if require_description and not row["description"]:
                    continue
                best_distance = distance
                best_match = {
                    "id": str(row["id"]),
                    "description": row["description"],
                    "description_source": row["description_source"],
                    "captured_at": row["captured_at"],
                }

        return best_match

    def _keyword_list(self) -> List[str]:
        if settings.video_capture_keyword_list:
            return [
                token.strip().lower()
                for token in settings.video_capture_keyword_list.split(",")
                if token.strip()
            ]
        return list(DEFAULT_KEYWORDS)

    def _determine_interesting_frame(
        self,
        chat_messages: Optional[List[Dict[str, Any]]],
        transcript: Optional[Dict[str, Any]],
    ) -> bool:
        """Rudimentary heuristics to decide if a frame is interesting."""
        chat_volume_threshold = max(
            1, settings.video_capture_interesting_chat_threshold
        )
        if chat_messages and len(chat_messages) >= chat_volume_threshold:
            return True

        if transcript and transcript.get("text"):
            text_lower = transcript["text"].lower()
            if any(keyword in text_lower for keyword in self._keyword_list()):
                return True

        return False

    async def generate_description_for_frame(self, frame_id: str) -> Optional[Dict[str, Any]]:
        """Generate (or regenerate) description for a stored frame.
        
        Returns:
            JSON dict with video frame description structure, or None if failed
        """
        lock = self._description_locks.get(frame_id)
        if lock is None:
            lock = asyncio.Lock()
            self._description_locks[frame_id] = lock

        try:
            async with lock:
                frame_uuid = uuid.UUID(frame_id)

                async with self.session_factory() as session:
                    frame = await session.get(VideoFrame, frame_uuid)
                    if frame is None:
                        logger.warning(
                            f"Tried to generate description for missing frame {frame_id}"
                        )
                        return None

                    # If description JSON already exists and was not marked lazy, return it
                    if frame.description_json and frame.description_source != "lazy":
                        return frame.description_json

                    transcript = None
                    if frame.transcript_id:
                        transcript_obj = await session.get(
                            Transcript, frame.transcript_id
                        )
                        if transcript_obj:
                            transcript = {
                                "id": str(transcript_obj.id),
                                "text": transcript_obj.text,
                                "started_at": transcript_obj.started_at,
                                "ended_at": transcript_obj.ended_at,
                            }

                    chat_messages: List[Dict[str, Any]] = []
                    if frame.aligned_chat_ids:
                        result = await session.execute(
                            text(
                                """
                                SELECT id, username, message, sent_at
                                FROM chat_messages
                                WHERE id = ANY(:ids)
                                """
                            ),
                            {"ids": frame.aligned_chat_ids},
                        )
                        chat_messages = [
                            {
                                "id": str(row["id"]),
                                "username": row["username"],
                                "message": row["message"],
                                "sent_at": row["sent_at"],
                            }
                            for row in result.mappings().all()
                        ]

                    metadata_snapshot = frame.metadata_snapshot
                    previous_description = await self._get_previous_frame_description(
                        frame.channel_id, frame.captured_at, exclude_frame_id=frame_uuid
                    )

                    try:
                        start_time = time.perf_counter()
                        description_json = await self._generate_description(
                            image_path=frame.image_path,
                            channel_id=frame.channel_id,
                            captured_at=frame.captured_at,
                            previous_frame_description=previous_description,
                            transcript=transcript,
                            chat=chat_messages,
                            metadata=metadata_snapshot,
                        )
                        duration_ms = (time.perf_counter() - start_time) * 1000
                        
                        # Generate text representation from JSON for description field
                        description_text = _json_to_text_for_embedding(description_json)
                        
                        # Generate grounded embedding now that description is available
                        grounded_embedding = None
                        if description_text and description_text.strip():
                            try:
                                grounded_embedding = await create_grounded_embedding(
                                    clip_embedding=frame.embedding,
                                    description_text=description_text,
                                )
                                logger.debug(
                                    f"Generated grounded embedding for frame {frame_id} after description"
                                )
                            except Exception as exc:
                                logger.warning(
                                    f"Failed to generate grounded embedding for frame {frame_id}: {exc}",
                                    exc_info=True,
                                )
                                # Continue without grounded embedding
                        
                        # Update frame with JSON description, text representation, and grounded embedding
                        frame.description_json = description_json
                        frame.description = description_text
                        frame.description_source = "generated"
                        if grounded_embedding:
                            frame.grounded_embedding = grounded_embedding
                        
                        await session.commit()
                        
                        description_logger.info(
                            "Generated description for frame %s in %.2fms",
                            frame_id,
                            duration_ms,
                        )
                        return description_json
                    except Exception as exc:
                        logger.warning(
                            f"Failed to generate description for frame {frame_id}: {exc}",
                            exc_info=True,
                        )
                        description_logger.error(
                            "Failed description generation for frame %s: %s",
                            frame_id,
                            exc,
                        )
                        return None

                    frame.description = description
                    frame.description_source = (
                        "lazy_generated"
                        if frame.description_source == "lazy"
                        else frame.description_source or "generated"
                    )

                    await session.commit()
                    logger.debug(
                        "Generated description for frame %s in %.1f ms (source=%s)",
                        frame_id,
                        duration_ms,
                        frame.description_source,
                    )
                    description_logger.info(
                        "Generated description for frame %s via %s (%.1f ms)",
                        frame_id,
                        frame.description_source,
                        duration_ms,
                    )
                    return description
        finally:
            self._cleanup_description_lock(frame_id)

    async def _get_previous_frame_description(
        self,
        channel_id: str,
        captured_at: datetime,
        exclude_frame_id: Optional[uuid.UUID] = None,
    ) -> Optional[str]:
        sql = """
        SELECT description
        FROM video_frames
        WHERE channel_id = :channel_id
          AND description IS NOT NULL
          AND captured_at < :captured_at
          {exclude_clause}
        ORDER BY captured_at DESC
        LIMIT 1
        """
        exclude_clause = ""
        params: Dict[str, Any] = {
            "channel_id": channel_id,
            "captured_at": captured_at,
        }
        if exclude_frame_id:
            exclude_clause = "AND id <> :exclude_id"
            params["exclude_id"] = exclude_frame_id
        sql = sql.format(exclude_clause=exclude_clause)

        async with self.session_factory() as session:
            result = await session.execute(text(sql), params)
            row = result.mappings().first()
            if row:
                return row["description"]
            return None

    async def _generate_description(
        self,
        *,
        image_path: str,
        channel_id: str,
        captured_at: datetime,
        previous_frame_description: Optional[str] = None,
        recent_summary: Optional[str] = None,
        transcript: Optional[Dict[str, Any]] = None,
        chat: Optional[List[Dict[str, Any]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Generate JSON description for a video frame.
        
        Returns:
            JSON dict with video frame description structure
        """
        # Convert previous_frame_description from text to JSON if needed
        # (for now, we'll pass it as-is since it might be from old format)
        return await generate_frame_description(
            image_path=image_path,
            channel_id=channel_id,
            captured_at=captured_at,
            previous_frame_description=previous_frame_description,
            recent_summary=recent_summary,
            transcript=transcript,
            chat=chat,
            metadata=metadata,
        )

    def _schedule_background_description_generation(self, frame_id: str) -> None:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            logger.warning(
                "No running event loop to schedule description generation; skipping."
            )
            return

        description_logger.info(
            "Scheduling background description generation for frame %s",
            frame_id,
        )
        loop.create_task(self._background_description_generation(frame_id))

    async def _background_description_generation(self, frame_id: str) -> None:
        try:
            await self.generate_description_for_frame(frame_id)
        except Exception as exc:
            logger.warning(
                f"Background description generation failed for frame {frame_id}: {exc}",
                exc_info=True,
            )
            description_logger.error(
                "Background description generation failed for frame %s: %s",
                frame_id,
                exc,
            )
        else:
            description_logger.info(
                "Background description generation completed for frame %s",
                frame_id,
            )

    def _cleanup_description_lock(self, frame_id: str) -> None:
        lock = self._description_locks.get(frame_id)
        if lock and not lock.locked():
            self._description_locks.pop(frame_id, None)

        if len(self._description_locks) > 5000:
            # Drop oldest unlocked locks to prevent unbounded growth
            for key in list(self._description_locks.keys()):
                if key == frame_id:
                    continue
                lock_ref = self._description_locks.get(key)
                if lock_ref and not lock_ref.locked():
                    self._description_locks.pop(key, None)
                if len(self._description_locks) <= 4000:
                    break

    async def get_lazy_frames_in_range(
        self,
        channel_id: str,
        start_time: datetime,
        end_time: datetime,
    ) -> List[str]:
        """Return frame IDs that still need descriptions within a time range."""
        sql = """
        SELECT id
        FROM video_frames
        WHERE channel_id = :channel_id
          AND captured_at >= :start_time
          AND captured_at <= :end_time
          AND (description IS NULL OR description_source = 'lazy')
        ORDER BY captured_at ASC
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

        return [str(row["id"]) for row in rows]

    async def get_range(
        self,
        channel_id: str,
        start_time: datetime,
        end_time: datetime,
    ) -> List[Dict[str, Any]]:
        """Return frames (with descriptions if available) within a time range."""
        sql = """
        SELECT id, captured_at, image_path, description, description_json, description_source
        FROM video_frames
        WHERE channel_id = :channel_id
          AND captured_at >= :start_time
          AND captured_at < :end_time
        ORDER BY captured_at ASC
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
                "captured_at": row["captured_at"],
                "image_path": row["image_path"],
                "description": row["description"],
                "description_json": row.get("description_json"),  # Include JSON if available
                "description_source": row["description_source"],
            }
            for row in rows
        ]
