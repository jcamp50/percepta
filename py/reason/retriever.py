from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import List, Optional

from py.memory.vector_store import VectorStore
from py.memory.video_store import VideoStore
from .interfaces import SearchResult


@dataclass
class RetrievalParams:
    channel_id: Optional[str]
    limit: int
    half_life_minutes: int
    prefilter_limit: int


class Retriever:
    """Abstraction over multiple stores. Currently transcripts, events, and video frames."""

    def __init__(
        self,
        *,
        vector_store: Optional[VectorStore] = None,
        video_store: Optional[VideoStore] = None,
    ) -> None:
        self.vector_store = vector_store or VectorStore()
        self.video_store = video_store

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

    async def from_video_frames(
        self, *, query_embedding: List[float], params: RetrievalParams
    ) -> List[SearchResult]:
        """
        Retrieve video frames using vector similarity with time-decay scoring.

        Video frames use the same timestamp for started_at and ended_at (point-in-time).
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

    async def retrieve_combined(
        self, *, query_embedding: List[float], params: RetrievalParams
    ) -> List[SearchResult]:
        """
        Retrieve from transcripts, events, and video frames in parallel, then merge results.

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
