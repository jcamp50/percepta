"""
Ingest layer: Twitch EventSub, transcription, and metadata polling
"""

from .transcription import TranscriptionService, TranscriptionResult, TranscriptionSegment
from .twitch import EventSubWebSocketClient

__all__ = [
    "TranscriptionService",
    "TranscriptionResult",
    "TranscriptionSegment",
    "EventSubWebSocketClient",
]