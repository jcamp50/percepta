from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from typing import List, Optional, Sequence

from py.config import settings
from py.memory.chat_store import ChatStore
from py.memory.vector_store import VectorStore
from py.memory.video_store import VideoStore
from py.utils.logging import get_logger

logger = get_logger(__name__, category="summarizer")


class Summarizer:
    """Placeholder summarizer with lazy frame backfill support.

    The actual summarization logic (LLM calls, propagation, etc.) will be
    implemented in JCB-35. For now we ensure that all frames in the target
    window have descriptions prior to summarization.
    """

    def __init__(
        self,
        *,
        video_store: VideoStore,
        vector_store: VectorStore,
        chat_store: ChatStore,
        completion_model: Optional[str] = None,
    ) -> None:
        self.video_store = video_store
        self.vector_store = vector_store
        self.chat_store = chat_store
        self.completion_model = completion_model or settings.rag_completion_model

    async def summarize_segment(
        self,
        *,
        channel_id: str,
        start_time: datetime,
        end_time: datetime,
        previous_summaries: Optional[Sequence[str]] = None,
    ) -> str:
        """Ensure frames within the window have descriptions, then (eventually) summarize.

        Returns an empty string for now – actual summarization will be completed in JCB-35.
        """
        # 1. Generate descriptions for lazy frames first
        lazy_frames = await self.video_store.get_lazy_frames_in_range(
            channel_id=channel_id,
            start_time=start_time,
            end_time=end_time,
        )

        if lazy_frames:
            logger.debug(
                "Backfilling %d lazy frame descriptions for channel %s between %s and %s",
                len(lazy_frames),
                channel_id,
                start_time.isoformat(),
                end_time.isoformat(),
            )
            await asyncio.gather(
                *[
                    self.video_store.generate_description_for_frame(frame_id)
                    for frame_id in lazy_frames
                ]
            )

        # 2. (Future) Retrieve data & build context – not implemented yet
        logger.debug(
            "Summarizer placeholder called for channel %s between %s and %s (summary not generated yet)",
            channel_id,
            start_time.isoformat(),
            end_time.isoformat(),
        )
        return ""
