"""
Ingest layer: Twitch EventSub, transcription, and metadata polling
"""

from .transcription import TranscriptionService, TranscriptionResult, TranscriptionSegment

__all__ = ["TranscriptionService", "TranscriptionResult", "TranscriptionSegment"]