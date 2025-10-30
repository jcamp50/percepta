from __future__ import annotations

import asyncio
import hashlib
import os
import random
from collections import OrderedDict
from typing import List, Optional, Tuple

from openai import OpenAI
from openai import APIError, RateLimitError, APITimeoutError, APIConnectionError

from py.config import settings


DEFAULT_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536


class _LRU:
    def __init__(self, max_entries: int = 5000) -> None:
        self.max = max_entries
        self.data: OrderedDict[str, List[float]] = OrderedDict()

    def get(self, key: str) -> Optional[List[float]]:
        value = self.data.pop(key, None)
        if value is not None:
            # re-insert to mark as most-recently used
            self.data[key] = value
        return value

    def set(self, key: str, value: List[float]) -> None:
        if key in self.data:
            self.data.pop(key)
        self.data[key] = value
        if len(self.data) > self.max:
            # evict least-recently used
            self.data.popitem(last=False)


def _cache_key(model: str, text: str) -> str:
    return hashlib.sha256((model + "\n" + text).encode("utf-8")).hexdigest()


_client = OpenAI(api_key=settings.openai_api_key)
_cache = _LRU()


async def _retry_embed(inputs: List[str], model: str) -> List[List[float]]:
    delay_seconds = 0.5
    max_attempts = 4
    for attempt in range(max_attempts):
        try:
            # The OpenAI client is synchronous; run in a thread to avoid blocking.
            response = await asyncio.to_thread(
                _client.embeddings.create, model=model, input=inputs
            )
            vectors = [d.embedding for d in response.data]

            # Validate dimensions
            for vec in vectors:
                if len(vec) != EMBEDDING_DIM:
                    raise ValueError(
                        f"Unexpected embedding dimension: {len(vec)} != {EMBEDDING_DIM}"
                    )

            # Optional token usage + cost logging
            usage = getattr(response, "usage", None)
            if usage is not None:
                per_1k = float(os.getenv("OPENAI_EMBEDDING_COST_PER_1K", "0") or 0)
                total_tokens = getattr(usage, "total_tokens", None)
                if per_1k and total_tokens is not None:
                    estimated_cost = (total_tokens / 1000.0) * per_1k
                    print(
                        {
                            "embedding_tokens": total_tokens,
                            "estimated_cost_usd": round(estimated_cost, 6),
                        }
                    )

            return vectors

        except (RateLimitError, APITimeoutError, APIConnectionError, APIError):
            if attempt == max_attempts - 1:
                raise
            # Exponential backoff with jitter
            await asyncio.sleep(delay_seconds + random.random() * 0.25)
            delay_seconds *= 2


async def embed_text(text: str, *, model: str = DEFAULT_MODEL) -> List[float]:
    """Embed a single piece of text, with LRU caching and retries."""
    key = _cache_key(model, text)
    cached = _cache.get(key)
    if cached is not None:
        return cached

    [vector] = await _retry_embed([text], model)
    _cache.set(key, vector)
    return vector


async def embed_texts(
    texts: List[str], *, model: str = DEFAULT_MODEL, batch_size: int = 128
) -> List[List[float]]:
    """Embed multiple texts. Preserves order and caches results.

    Only cache misses are sent to the API; hits are filled immediately.
    """
    # Pre-fill from cache and collect misses
    results: List[Optional[List[float]]] = [None] * len(texts)
    misses: List[Tuple[int, str]] = []

    for index, text in enumerate(texts):
        key = _cache_key(model, text)
        cached = _cache.get(key)
        if cached is None:
            misses.append((index, text))
        else:
            results[index] = cached

    # Process misses in batches
    for start in range(0, len(misses), batch_size):
        chunk = misses[start : start + batch_size]
        if not chunk:
            continue
        idxs = [i for i, _ in chunk]
        inputs = [t for _, t in chunk]
        vectors = await _retry_embed(inputs, model)
        for i, t, v in zip(idxs, inputs, vectors):
            _cache.set(_cache_key(model, t), v)
            results[i] = v

    # All results should be filled now
    return [r for r in results if r is not None]
