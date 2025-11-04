from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import List, Optional

from py.memory.vector_store import VectorStore
from .interfaces import SearchResult


@dataclass
class RetrievalParams:
    channel_id: Optional[str]
    limit: int
    half_life_minutes: int
    prefilter_limit: int


class Retriever:
    """Abstraction over multiple stores. Currently transcripts and events."""

    def __init__(self, *, vector_store: Optional[VectorStore] = None) -> None:
        self.vector_store = vector_store or VectorStore()

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

    async def retrieve_combined(
        self, *, query_embedding: List[float], params: RetrievalParams
    ) -> List[SearchResult]:
        """
        Retrieve from both transcripts and events in parallel, then merge results.

        Results are ranked by time-decay adjusted score across both sources.
        """
        # Query both sources in parallel
        transcript_results, event_results = await asyncio.gather(
            self.from_transcripts(query_embedding=query_embedding, params=params),
            self.from_events(query_embedding=query_embedding, params=params),
        )

        # Combine results
        combined = list(transcript_results) + list(event_results)

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
