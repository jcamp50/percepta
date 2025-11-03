"""
Transcription Service

Handles speech-to-text transcription using faster-whisper.
Provides async transcription with word-level timestamps.
"""

from __future__ import annotations

import asyncio
import io
import logging
import os
import tempfile
import time
from typing import Optional, List, Dict, Any

try:
    from faster_whisper import WhisperModel
except ImportError:
    WhisperModel = None

from py.config import settings

logger = logging.getLogger(__name__)


class TranscriptionSegment:
    """Represents a transcribed segment with timestamps."""

    def __init__(
        self,
        start: float,
        end: float,
        text: str,
        words: Optional[List[Dict[str, Any]]] = None,
    ):
        self.start = start
        self.end = end
        self.text = text
        self.words = words or []

    def to_dict(self) -> Dict[str, Any]:
        return {
            "start": self.start,
            "end": self.end,
            "text": self.text,
            "words": self.words,
        }


class TranscriptionResult:
    """Result of a transcription operation."""

    def __init__(
        self,
        transcript: str,
        segments: List[TranscriptionSegment],
        language: str,
        duration: float,
        model: str,
        processing_time_ms: int,
    ):
        self.transcript = transcript
        self.segments = segments
        self.language = language
        self.duration = duration
        self.model = model
        self.processing_time_ms = processing_time_ms

    def to_dict(self) -> Dict[str, Any]:
        return {
            "transcript": self.transcript,
            "segments": [seg.to_dict() for seg in self.segments],
            "language": self.language,
            "duration": self.duration,
            "model": self.model,
            "processing_time_ms": self.processing_time_ms,
        }


class TranscriptionService:
    """Service for transcribing audio using faster-whisper."""

    _instance: Optional["TranscriptionService"] = None
    _model: Optional[Any] = None

    def __init__(
        self,
        model_name: Optional[str] = None,
        language: Optional[str] = None,
        device: Optional[str] = None,
        compute_type: Optional[str] = None,
        use_gpu: Optional[bool] = None,
    ):
        if WhisperModel is None:
            raise ImportError(
                "faster-whisper is not installed. Install with: pip install faster-whisper"
            )

        self.model_name = model_name or getattr(settings, "whisper_model", "base")
        self.language = language or getattr(settings, "whisper_language", "en")
        self.use_gpu = (
            use_gpu if use_gpu is not None else getattr(settings, "use_gpu", False)
        )

        # Auto-detect device if not specified
        self.device = device or self._get_device()

        # Auto-select compute type if not specified
        self.compute_type = compute_type or self._get_compute_type()

        self._model_loaded = False

    @classmethod
    def get_instance(cls) -> "TranscriptionService":
        """Get singleton instance of transcription service."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _get_device(self) -> str:
        """Auto-detect available device (CPU or GPU)."""
        if self.use_gpu:
            try:
                import torch

                if torch.cuda.is_available():
                    return "cuda"
            except ImportError:
                pass
        return "cpu"

    def _get_compute_type(self) -> str:
        """Auto-select optimal compute type based on device."""
        if self.device == "cuda":
            return "float16"  # Faster on GPU
        # For CPU, use int8 if available, otherwise default
        return "int8"  # Faster on CPU

    def load_model(self) -> None:
        """Load Whisper model (lazy loading on first use)."""
        if self._model_loaded and TranscriptionService._model is not None:
            return

        logger.info(
            f"Loading Whisper model: {self.model_name} "
            f"(device={self.device}, compute_type={self.compute_type})"
        )

        try:
            TranscriptionService._model = WhisperModel(
                self.model_name,
                device=self.device,
                compute_type=self.compute_type,
            )
            self._model_loaded = True
            logger.info("Whisper model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            raise

    def transcribe(
        self, audio_bytes: bytes, language: Optional[str] = None
    ) -> TranscriptionResult:
        """
        Transcribe audio from bytes.

        Args:
            audio_bytes: Raw audio file bytes (WAV format expected)
            language: Language code (e.g., 'en'). None for auto-detect.

        Returns:
            TranscriptionResult with transcript, segments, and metadata
        """
        if not self._model_loaded:
            self.load_model()

        start_time = time.monotonic()

        try:
            # faster-whisper works best with file paths
            # Create temporary file for transcription
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                tmp_file.write(audio_bytes)
                tmp_file_path = tmp_file.name

            try:
                # Run transcription synchronously (will be called from async context)
                # VAD filter: Use False or adjust threshold to be less aggressive
                # If all audio is being filtered, set vad_filter=False or use vad_parameters
                segments_generator, info = TranscriptionService._model.transcribe(
                    tmp_file_path,
                    language=language or self.language,
                    beam_size=5,
                    vad_filter=False,  # Disabled: was too aggressive, filtering all audio
                    # Alternative: use vad_parameters for less aggressive filtering
                    # vad_filter=True,
                    # vad_parameters=dict(min_silence_duration_ms=1000, threshold=0.5),
                    word_timestamps=True,
                )
            finally:
                # Clean up temp file
                if os.path.exists(tmp_file_path):
                    os.unlink(tmp_file_path)

            # Convert generator to list
            segments_list = list(segments_generator)

            # Extract segments with word-level timestamps
            transcription_segments = []
            full_text_parts = []

            for segment in segments_list:
                # Extract words if available
                words = []
                if hasattr(segment, "words") and segment.words:
                    words = [
                        {
                            "word": word.word,
                            "start": word.start,
                            "end": word.end,
                        }
                        for word in segment.words
                    ]

                transcription_segments.append(
                    TranscriptionSegment(
                        start=segment.start,
                        end=segment.end,
                        text=segment.text,
                        words=words,
                    )
                )
                full_text_parts.append(segment.text)

            full_text = " ".join(full_text_parts).strip()

            processing_time_ms = int((time.monotonic() - start_time) * 1000)

            # Get duration from audio info or estimate from segments
            duration = (
                info.duration
                if hasattr(info, "duration")
                else segments_list[-1].end if segments_list else 0.0
            )

            detected_language = (
                info.language if hasattr(info, "language") else self.language
            )

            result = TranscriptionResult(
                transcript=full_text,
                segments=transcription_segments,
                language=detected_language,
                duration=duration,
                model=self.model_name,
                processing_time_ms=processing_time_ms,
            )

            logger.info(
                f"Transcribed {duration:.2f}s audio in {processing_time_ms}ms "
                f"(language={detected_language}, words={len(full_text.split())})"
            )

            return result

        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            raise

    async def transcribe_async(
        self, audio_bytes: bytes, language: Optional[str] = None
    ) -> TranscriptionResult:
        """Async wrapper for transcribe method."""
        return await asyncio.to_thread(self.transcribe, audio_bytes, language)

    def transcribe_file(
        self, audio_file_path: str, language: Optional[str] = None
    ) -> TranscriptionResult:
        """
        Transcribe audio from file path.

        Args:
            audio_file_path: Path to audio file
            language: Language code. None for auto-detect.

        Returns:
            TranscriptionResult
        """
        with open(audio_file_path, "rb") as f:
            audio_bytes = f.read()
        return self.transcribe(audio_bytes, language)
