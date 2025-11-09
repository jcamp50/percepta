from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Awaitable, Dict, List, Optional, Tuple

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


MODALITY_WEIGHTS: Dict[str, float] = {
    "transcript": 1.0,
    "event": 1.05,
    "video": 0.95,
    "chat": 1.1,
}

TIME_ALIGNMENT_WINDOW_SECONDS = 5

logger = logging.getLogger(__name__)


@dataclass
class _WeightedResult:
    result: SearchResult
    modality: str
    adjusted_score: float

    @property
    def started_at(self) -> Optional[datetime]:
        return self.result.started_at  # type: ignore[return-value]

    @property
    def ended_at(self) -> Optional[datetime]:
        return self.result.ended_at  # type: ignore[return-value]

    @property
    def cosine_distance(self) -> float:
        return self.result.cosine_distance


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
                text=f"[Transcript] {r['text']}",
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
                text=f"[Event] {r['summary']}",  # Use summary as text
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
            description = r.get("description")
            if not description:
                try:
                    description = await self.video_store.generate_description_for_frame(
                        r["id"]
                    )
                except Exception as exc:  # pragma: no cover - defensive logging
                    logger.warning(
                        "Failed on-demand description for frame %s: %s",
                        r.get("id"),
                        exc,
                    )
                    description = None

            # Prefer visual description (JCB-41), fall back to image path
            if description:
                text_parts = [f"[Video Frame] {description}"]
            else:
                text_parts = [f"[Video Frame] {r['image_path']}"]

            # Add transcript text if available
            transcript_text = None
            if r.get("transcript_id") and self.vector_store:
                try:
                    transcript = await self.vector_store.get_transcript_by_id(
                        r["transcript_id"]
                    )
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
                    chat_messages = await self.chat_store.get_messages_by_ids(
                        r["aligned_chat_ids"]
                    )
                    if chat_messages:
                        chat_text = " | ".join(
                            [f"{c['username']}: {c['message']}" for c in chat_messages]
                        )
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
        self,
        *,
        query_embedding: List[float],
        params: RetrievalParams,
        include_context: bool = True,
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

    def _make_weighted(self, result: SearchResult, modality: str) -> _WeightedResult:
        weight = MODALITY_WEIGHTS.get(modality, 1.0)
        adjusted = result.score * weight
        return _WeightedResult(
            result=result, modality=modality, adjusted_score=adjusted
        )

    @staticmethod
    def _result_midpoint(result: SearchResult) -> Optional[datetime]:
        started_at = result.started_at
        ended_at = result.ended_at

        if isinstance(started_at, datetime) and isinstance(ended_at, datetime):
            try:
                delta = ended_at - started_at
            except TypeError:
                return started_at

            if isinstance(delta, timedelta):
                return started_at + delta / 2
        if isinstance(started_at, datetime):
            return started_at
        if isinstance(ended_at, datetime):
            return ended_at
        return None

    def _time_difference_seconds(
        self, lhs: _WeightedResult, rhs: _WeightedResult
    ) -> Optional[float]:
        left_midpoint = self._result_midpoint(lhs.result)
        right_midpoint = self._result_midpoint(rhs.result)

        if left_midpoint is None or right_midpoint is None:
            return None
        return abs((left_midpoint - right_midpoint).total_seconds())

    def _alignment_anchor(self, weighted: _WeightedResult) -> datetime:
        midpoint = self._result_midpoint(weighted.result)
        if isinstance(midpoint, datetime):
            return midpoint

        started_at = weighted.result.started_at
        if isinstance(started_at, datetime):
            return started_at

        ended_at = weighted.result.ended_at
        if isinstance(ended_at, datetime):
            return ended_at

        return datetime.min

    def _align_by_time(
        self,
        results: List[_WeightedResult],
        window_seconds: int = TIME_ALIGNMENT_WINDOW_SECONDS,
    ) -> List[List[_WeightedResult]]:
        if not results:
            return []

        sorted_results = sorted(results, key=self._alignment_anchor)
        groups: List[List[_WeightedResult]] = []

        for item in sorted_results:
            placed = False
            for group in groups:
                for existing in group:
                    diff = self._time_difference_seconds(existing, item)
                    if diff is None:
                        continue
                    if diff <= window_seconds:
                        group.append(item)
                        placed = True
                        break
                if placed:
                    break
            if not placed:
                groups.append([item])

        return groups

    def _fuse_modalities(self, group: List[_WeightedResult]) -> SearchResult:
        if not group:
            raise ValueError("Cannot fuse an empty result group")

        sorted_group = sorted(group, key=lambda item: item.adjusted_score)
        texts: List[str] = []
        seen = set()
        for item in sorted_group:
            text = item.result.text.strip()
            if text and text not in seen:
                texts.append(text)
                seen.add(text)

        fused_text = " | ".join(texts) if texts else sorted_group[0].result.text

        channel_id = sorted_group[0].result.channel_id
        fused_id = "+".join(sorted({item.result.id for item in sorted_group}))

        start_candidates = [
            item.result.started_at
            for item in sorted_group
            if isinstance(item.result.started_at, datetime)
        ]
        end_candidates = [
            item.result.ended_at
            for item in sorted_group
            if isinstance(item.result.ended_at, datetime)
        ]

        started_at: object = (
            min(start_candidates)
            if start_candidates
            else sorted_group[0].result.started_at
        )
        ended_at: object = (
            max(end_candidates) if end_candidates else sorted_group[0].result.ended_at
        )

        cosine_distance = min(item.cosine_distance for item in sorted_group)
        score = min(item.adjusted_score for item in sorted_group)

        return SearchResult(
            id=fused_id,
            channel_id=channel_id,
            text=fused_text,
            started_at=started_at,
            ended_at=ended_at,
            cosine_distance=cosine_distance,
            score=score,
        )

    def _merge_and_rank(
        self,
        *,
        transcripts: List[SearchResult],
        events: List[SearchResult],
        videos: List[SearchResult],
        chats: List[SearchResult],
        limit: int,
        window_seconds: int = TIME_ALIGNMENT_WINDOW_SECONDS,
    ) -> List[SearchResult]:
        weighted: List[_WeightedResult] = []
        for result in transcripts:
            weighted.append(self._make_weighted(result, "transcript"))
        for result in events:
            weighted.append(self._make_weighted(result, "event"))
        for result in videos:
            weighted.append(self._make_weighted(result, "video"))
        for result in chats:
            weighted.append(self._make_weighted(result, "chat"))

        if not weighted:
            return []

        groups = self._align_by_time(weighted, window_seconds=window_seconds)
        fused = [self._fuse_modalities(group) for group in groups]
        fused.sort(key=lambda item: item.score)

        if limit <= 0:
            return fused

        return fused[:limit]

    async def retrieve_combined(
        self, *, query_embedding: List[float], params: RetrievalParams
    ) -> List[SearchResult]:
        """
        Retrieve from transcripts, events, video frames, and chat messages in parallel, then merge results.

        Results are ranked by time-decay adjusted score across all sources.
        """
        # Build list of coroutines to run in parallel
        coroutines: List[Tuple[str, Awaitable[List[SearchResult]]]] = []

        coroutines.append(
            (
                "transcripts",
                self.from_transcripts(query_embedding=query_embedding, params=params),
            )
        )
        coroutines.append(
            ("events", self.from_events(query_embedding=query_embedding, params=params))
        )

        if self.video_store is not None:
            coroutines.append(
                (
                    "videos",
                    self.from_video_frames(
                        query_embedding=query_embedding, params=params
                    ),
                )
            )

        if self.chat_store is not None:
            coroutines.append(
                (
                    "chats",
                    self.from_chat_messages(
                        query_embedding=query_embedding, params=params
                    ),
                )
            )

        if not coroutines:
            return []

        labels = [label for label, _ in coroutines]
        tasks = [coro for _, coro in coroutines]
        results = await asyncio.gather(*tasks)
        results_map = dict(zip(labels, results))

        return self._merge_and_rank(
            transcripts=results_map.get("transcripts", []),
            events=results_map.get("events", []),
            videos=results_map.get("videos", []),
            chats=results_map.get("chats", []),
            limit=params.limit,
        )

    async def retrieve(
        self, *, query_embedding: List[float], params: RetrievalParams
    ) -> List[SearchResult]:
        """
        Retrieve from transcripts and events combined.
        """
        return await self.retrieve_combined(
            query_embedding=query_embedding, params=params
        )
