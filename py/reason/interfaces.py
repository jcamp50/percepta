from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Protocol, Sequence


class EmbeddingTool(Protocol):
    async def __call__(self, text: str) -> List[float]:
        ...


@dataclass
class SearchResult:
    id: str
    channel_id: str
    text: str
    started_at: object
    ended_at: object
    cosine_distance: float
    score: float


class VectorSearchTool(Protocol):
    async def __call__(
        self,
        *,
        query_embedding: List[float],
        channel_id: Optional[str],
        limit: int,
        half_life_minutes: int,
        prefilter_limit: int,
    ) -> List[SearchResult]:
        ...


class ContextBuilder(Protocol):
    def __call__(self, lines: Sequence[str], budget_chars: int) -> Sequence[str]:
        ...


class LLMTool(Protocol):
    async def __call__(self, messages: List[dict]) -> str:
        ...


Compressor = Callable[[Sequence[str], int], Sequence[str]]
Critic = Callable[[str], bool]


