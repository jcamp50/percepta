from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import List, Optional

from py.memory.vector_store import VectorStore
from py.memory.video_store import VideoStore
from py.memory.chat_store import ChatStore
from .interfaces import SearchResult


@dataclass
class RetrievalParams:
    channel_id: Optional[str]
    limit: int
    half_life_minutes: int
    prefilter_limit: int


class Retriever:
    """Abstraction over multiple stores. Currently transcripts, events, video frames, and chat messages."""

    def __init__(
        self,
        *,
        vector_store: Optional[VectorStore] = None,
        video_store: Optional[VideoStore] = None,
        chat_store: Optional[ChatStore] = None,
    ) -> None:
        self.vector_store = vector_store or VectorStore()
        self.video_store = video_store
        self.chat_store = chat_store

    async def from_transcripts(
        self, *, query_embedding: List[float], params: RetrievalParams
    ) -> List[SearchResult]:
        rows = await self.vector_store.search_transcripts(
            query_embedding=query_embedding,
            limit=params.limit,
            half_life_minutes=params.half_life_minutes,
            channel_id=params.channel_id,
            prefilter_limit=params.prefilter_limit,
        )
        return [
            SearchResult(
                id=r["id"],
                channel_id=r["channel_id"],
                text=r["text"],
                started_at=r["started_at"],
                ended_at=r["ended_at"],
                cosine_distance=float(r["cosine_distance"]),
                score=float(r["score"]),
            )
            for r in rows
        ]

    async def from_events(
        self, *, query_embedding: List[float], params: RetrievalParams
    ) -> List[SearchResult]:
        """
        Retrieve events using vector similarity with time-decay scoring.

        Events use the same timestamp for started_at and ended_at.
        """
        rows = await self.vector_store.search_events(
            query_embedding=query_embedding,
            limit=params.limit,
            half_life_minutes=params.half_life_minutes,
            channel_id=params.channel_id,
            prefilter_limit=params.prefilter_limit,
        )
        return [
            SearchResult(
                id=r["id"],
                channel_id=r["channel_id"],
                text=r["summary"],  # Use summary as text
                started_at=r["ts"],  # Use event timestamp for both
                ended_at=r["ts"],  # Events are point-in-time
                cosine_distance=float(r["cosine_distance"]),
                score=float(r["score"]),
            )
            for r in rows
        ]

    async def retrieve_video_with_context(
        self, *, query_embedding: List[float], params: RetrievalParams
    ) -> List[SearchResult]:
        """
        Retrieve video frames with enriched context (transcripts, chat, metadata).
        
        Enriches video frame results with temporally-aligned context from transcripts,
        chat messages, and metadata snapshots.
        """
        if self.video_store is None:
            return []

        rows = await self.video_store.search_frames(
            query_embedding=query_embedding,
            limit=params.limit,
            half_life_minutes=params.half_life_minutes,
            channel_id=params.channel_id,
            prefilter_limit=params.prefilter_limit,
        )
        
        enriched_results = []
        for r in rows:
            # Build enriched text representation
            description = None
            if hasattr(r, "get"):
                description = r.get("description")
            else:
                description = r["description"] if "description" in r.keys() else None

            # Prefer visual description (JCB-41), fall back to image path
            if description:
                text_parts = [f"[Video Frame] {description}"]
            else:
                text_parts = [f"[Video Frame] {r['image_path']}"]
            
            # Add transcript text if available
            transcript_text = None
            if r.get("transcript_id") and self.vector_store:
                try:
                    transcript = await self.vector_store.get_transcript_by_id(r["transcript_id"])
                    if transcript:
                        transcript_text = transcript["text"]
                        text_parts.append(f"Transcript: {transcript_text}")
                except Exception:
                    # Log but don't fail if transcript retrieval fails
                    pass
            
            # Add chat messages if available
            chat_messages = []
            if r.get("aligned_chat_ids") and self.chat_store:
                try:
                    chat_messages = await self.chat_store.get_messages_by_ids(r["aligned_chat_ids"])
                    if chat_messages:
                        chat_text = " | ".join([f"{c['username']}: {c['message']}" for c in chat_messages])
                        text_parts.append(f"Chat: {chat_text}")
                except Exception:
                    # Log but don't fail if chat retrieval fails
                    pass
            
            # Add metadata if available
            metadata = r.get("metadata_snapshot")
            if metadata:
                metadata_parts = []
                if metadata.get("game_name"):
                    metadata_parts.append(f"Game: {metadata['game_name']}")
                if metadata.get("title"):
                    metadata_parts.append(f"Title: {metadata['title']}")
                if metadata_parts:
                    text_parts.append(f"Metadata: {' | '.join(metadata_parts)}")
            
            enriched_text = " | ".join(text_parts)
            
            enriched_results.append(
                SearchResult(
                    id=r["id"],
                    channel_id=r["channel_id"],
                    text=enriched_text,
                    started_at=r["captured_at"],
                    ended_at=r["captured_at"],
                    cosine_distance=float(r["cosine_distance"]),
                    score=float(r["score"]),
                )
            )
        
        return enriched_results

    async def from_video_frames(
        self, *, query_embedding: List[float], params: RetrievalParams, include_context: bool = True
    ) -> List[SearchResult]:
        """
        Retrieve video frames using vector similarity with time-decay scoring.

        Video frames use the same timestamp for started_at and ended_at (point-in-time).
        
        Args:
            include_context: If True, enrich results with aligned transcripts, chat, and metadata
        """
        if self.video_store is None:
            return []

        if include_context:
            return await self.retrieve_video_with_context(
                query_embedding=query_embedding, params=params
            )
        
        # Fallback to basic retrieval without context enrichment
        rows = await self.video_store.search_frames(
            query_embedding=query_embedding,
            limit=params.limit,
            half_life_minutes=params.half_life_minutes,
            channel_id=params.channel_id,
            prefilter_limit=params.prefilter_limit,
        )
        return [
            SearchResult(
                id=r["id"],
                channel_id=r["channel_id"],
                text=f"[Video Frame] {r['image_path']}",  # Use image path as text representation
                started_at=r["captured_at"],  # Use captured_at for both
                ended_at=r["captured_at"],  # Frames are point-in-time
                cosine_distance=float(r["cosine_distance"]),
                score=float(r["score"]),
            )
            for r in rows
        ]

    async def from_chat_messages(
        self, *, query_embedding: List[float], params: RetrievalParams
    ) -> List[SearchResult]:
        """
        Retrieve chat messages using vector similarity with time-decay scoring.

        Chat messages use the same timestamp for started_at and ended_at (point-in-time).
        """
        if self.chat_store is None:
            return []

        rows = await self.chat_store.search_messages(
            query_embedding=query_embedding,
            limit=params.limit,
            half_life_minutes=params.half_life_minutes,
            channel_id=params.channel_id,
            prefilter_limit=params.prefilter_limit,
        )
        return [
            SearchResult(
                id=r["id"],
                channel_id=r["channel_id"],
                text=f"[Chat] {r['username']}: {r['message']}",  # Include username in text representation
                started_at=r["sent_at"],  # Use sent_at for both
                ended_at=r["sent_at"],  # Messages are point-in-time
                cosine_distance=float(r["cosine_distance"]),
                score=float(r["score"]),
            )
            for r in rows
        ]

    async def retrieve_combined(
        self, *, query_embedding: List[float], params: RetrievalParams
    ) -> List[SearchResult]:
        """
        Retrieve from transcripts, events, video frames, and chat messages in parallel, then merge results.

        Results are ranked by time-decay adjusted score across all sources.
        """
        # Build list of coroutines to run in parallel
        coroutines = [
            self.from_transcripts(query_embedding=query_embedding, params=params),
            self.from_events(query_embedding=query_embedding, params=params),
        ]

        # Add video frames if video_store is available
        if self.video_store is not None:
            coroutines.append(
                self.from_video_frames(query_embedding=query_embedding, params=params)
            )

        # Add chat messages if chat_store is available
        if self.chat_store is not None:
            coroutines.append(
                self.from_chat_messages(query_embedding=query_embedding, params=params)
            )

        # Query all sources in parallel
        results = await asyncio.gather(*coroutines)

        # Combine results from all sources
        combined = []
        for result_list in results:
            combined.extend(result_list)

        # Sort by score (lower is better, time-decay adjusted)
        combined.sort(key=lambda r: r.score)

        # Return top results up to limit
        return combined[: params.limit]

    async def retrieve(
        self, *, query_embedding: List[float], params: RetrievalParams
    ) -> List[SearchResult]:
        """
        Retrieve from transcripts and events combined.
        """
        return await self.retrieve_combined(
            query_embedding=query_embedding, params=params
        )
