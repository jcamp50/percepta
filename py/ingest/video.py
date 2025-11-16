from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, List

from py.memory.chat_store import ChatStore
from py.memory.vector_store import VectorStore
from py.config import settings
from py.utils.logging import get_logger

logger = get_logger(__name__, category="video")


DEFAULT_KEYWORDS: Sequence[str] = (
    "boss",
    "clutch",
    "hype",
    "insane",
    "crazy",
    "raid",
    "pog",
    "omg",
)


@dataclass
class CaptureDecision:
    next_interval_seconds: int
    recent_chat_count: int
    keyword_trigger: bool

    def to_dict(self) -> Dict[str, object]:
        return {
            "next_interval_seconds": self.next_interval_seconds,
            "recent_chat_count": self.recent_chat_count,
            "keyword_trigger": self.keyword_trigger,
        }


def _keyword_list() -> List[str]:
    if settings.video_capture_keyword_list:
        return [
            token.strip().lower()
            for token in settings.video_capture_keyword_list.split(",")
            if token.strip()
        ]
    return list(DEFAULT_KEYWORDS)


async def determine_capture_interval(
    *,
    channel_id: str,
    captured_at,
    chat_store: ChatStore,
    vector_store: VectorStore,
    baseline_interval: Optional[int] = None,
    active_interval: Optional[int] = None,
    chat_threshold: Optional[int] = None,
    keyword_triggers: Optional[Sequence[str]] = None,
    recent_chat_count: Optional[int] = None,
    transcript_text: Optional[str] = None,
    was_interesting: bool = False,
) -> Dict[str, object]:
    """Decide how soon the next frame should be captured."""
    baseline_interval = (
        max(2, baseline_interval)
        if baseline_interval is not None
        else max(2, settings.video_capture_baseline_interval)
    )
    active_interval = (
        max(2, active_interval)
        if active_interval is not None
        else max(2, settings.video_capture_active_interval)
    )
    chat_threshold = (
        chat_threshold
        if chat_threshold is not None
        else settings.video_capture_chat_threshold
    )
    keyword_list = list(keyword_triggers) if keyword_triggers is not None else _keyword_list()

    chat_count = recent_chat_count
    if chat_count is None:
        try:
            chat_count = await chat_store.count_recent_messages(
                channel_id=channel_id,
                window_seconds=30,
            )
        except Exception as exc:
            logger.warning(
                "Failed to count recent chat messages for %s: %s",
                channel_id,
                exc,
            )
            chat_count = 0

    keyword_hit = False
    text_to_check = transcript_text
    if text_to_check is None:
        try:
            text_to_check = await vector_store.get_recent_transcript_text(
                channel_id=channel_id,
                window_seconds=30,
            )
        except Exception as exc:
            logger.warning(
                "Failed to fetch recent transcript for %s: %s", channel_id, exc
            )

    if text_to_check:
        lowered = text_to_check.lower()
        keyword_hit = any(keyword in lowered for keyword in keyword_list)

    high_activity = chat_count >= chat_threshold or keyword_hit or was_interesting
    interval = active_interval if high_activity else baseline_interval

    decision = CaptureDecision(
        next_interval_seconds=interval,
        recent_chat_count=chat_count,
        keyword_trigger=keyword_hit,
    )

    logger.debug(
        "Capture decision for channel %s at %s: %s",
        channel_id,
        captured_at.isoformat(),
        decision,
    )
    return decision.to_dict()

