from __future__ import annotations

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
    """Abstraction over multiple stores. Currently transcripts only."""

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

    async def retrieve(
        self, *, query_embedding: List[float], params: RetrievalParams
    ) -> List[SearchResult]:
        # Future: fanout to events/summaries, then merge + rank
        return await self.from_transcripts(
            query_embedding=query_embedding, params=params
        )


