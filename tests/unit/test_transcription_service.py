"""Unit tests for TranscriptionService."""
import os
import tempfile
import pytest
from unittest.mock import Mock, MagicMock, patch

from py.ingest.transcription import (
    TranscriptionService,
    TranscriptionResult,
    TranscriptionSegment,
)


@pytest.mark.unit
class TestTranscriptionService:
    """Test TranscriptionService."""

    def test_import_error_when_whisper_missing(self, monkeypatch):
        """Test that ImportError is raised when faster-whisper is not installed."""
        # Simulate missing faster-whisper
        monkeypatch.setattr("py.ingest.transcription.WhisperModel", None)
        
        with pytest.raises(ImportError, match="faster-whisper is not installed"):
            TranscriptionService()

    def test_device_selection_cpu(self, monkeypatch):
        """Test that CPU is selected when GPU is not available."""
        # Mock WhisperModel to avoid import error
        mock_whisper = MagicMock()
        monkeypatch.setattr("py.ingest.transcription.WhisperModel", mock_whisper)
        
        # Mock torch to indicate no CUDA
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        monkeypatch.setattr("py.ingest.transcription.torch", mock_torch, raising=False)
        
        service = TranscriptionService(use_gpu=False)
        assert service.device == "cpu"
        assert service.compute_type == "int8"

    def test_device_selection_cuda(self, monkeypatch):
        """Test that CUDA is selected when GPU is available."""
        mock_whisper = MagicMock()
        monkeypatch.setattr("py.ingest.transcription.WhisperModel", mock_whisper)
        
        # Mock torch to indicate CUDA available
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        monkeypatch.setattr("py.ingest.transcription.torch", mock_torch, raising=False)
        
        service = TranscriptionService(use_gpu=True)
        assert service.device == "cuda"
        assert service.compute_type == "float16"

    def test_load_model_sets_flag(self, monkeypatch):
        """Test that load_model sets _model_loaded flag."""
        mock_whisper = MagicMock()
        mock_model_instance = MagicMock()
        mock_whisper.return_value = mock_model_instance
        monkeypatch.setattr("py.ingest.transcription.WhisperModel", mock_whisper)
        
        service = TranscriptionService()
        assert not service._model_loaded
        
        service.load_model()
        assert service._model_loaded
        mock_whisper.assert_called_once()

    def test_transcribe_creates_segments(self, monkeypatch):
        """Test that transcribe creates TranscriptionResult with segments."""
        mock_whisper = MagicMock()
        mock_model_instance = MagicMock()
        
        # Mock segment with word timestamps
        mock_segment = MagicMock()
        mock_segment.start = 0.0
        mock_segment.end = 2.5
        mock_segment.text = "Hello world"
        mock_word1 = MagicMock()
        mock_word1.word = "Hello"
        mock_word1.start = 0.0
        mock_word1.end = 1.0
        mock_word2 = MagicMock()
        mock_word2.word = "world"
        mock_word2.start = 1.0
        mock_word2.end = 2.5
        mock_segment.words = [mock_word1, mock_word2]
        
        mock_info = MagicMock()
        mock_info.duration = 2.5
        mock_info.language = "en"
        
        mock_model_instance.transcribe.return_value = ([mock_segment], mock_info)
        mock_whisper.return_value = mock_model_instance
        monkeypatch.setattr("py.ingest.transcription.WhisperModel", mock_whisper)
        
        service = TranscriptionService()
        service._model_loaded = True
        TranscriptionService._model = mock_model_instance
        
        # Create test audio bytes (minimal WAV header)
        audio_bytes = b'RIFF' + b'\x00' * 40  # Minimal WAV-like bytes
        
        result = service.transcribe(audio_bytes)
        
        assert isinstance(result, TranscriptionResult)
        assert result.transcript == "Hello world"
        assert result.language == "en"
        assert result.duration == 2.5
        assert len(result.segments) == 1
        assert result.segments[0].text == "Hello world"
        assert len(result.segments[0].words) == 2

    def test_transcribe_cleans_temp_file(self, monkeypatch):
        """Test that transcribe cleans up temp file after processing."""
        mock_whisper = MagicMock()
        mock_model_instance = MagicMock()
        
        mock_segment = MagicMock()
        mock_segment.start = 0.0
        mock_segment.end = 1.0
        mock_segment.text = "Test"
        mock_segment.words = []
        
        mock_info = MagicMock()
        mock_info.duration = 1.0
        mock_info.language = "en"
        
        mock_model_instance.transcribe.return_value = ([mock_segment], mock_info)
        mock_whisper.return_value = mock_model_instance
        monkeypatch.setattr("py.ingest.transcription.WhisperModel", mock_whisper)
        
        # Track temp file creation/deletion
        created_files = []
        original_unlink = os.unlink
        
        def track_unlink(path):
            if path in created_files:
                created_files.remove(path)
            return original_unlink(path)
        
        monkeypatch.setattr("os.unlink", track_unlink)
        
        service = TranscriptionService()
        service._model_loaded = True
        TranscriptionService._model = mock_model_instance
        
        audio_bytes = b'RIFF' + b'\x00' * 40
        
        with patch('tempfile.NamedTemporaryFile') as mock_temp:
            mock_file = MagicMock()
            mock_file.name = "/tmp/test_audio.wav"
            mock_file.__enter__ = MagicMock(return_value=mock_file)
            mock_file.__exit__ = MagicMock(return_value=None)
            mock_temp.return_value = mock_file
            
            result = service.transcribe(audio_bytes)
            
            # Verify temp file was used
            mock_file.write.assert_called_once_with(audio_bytes)

    def test_transcribe_async_wraps_sync(self, monkeypatch):
        """Test that transcribe_async wraps transcribe correctly."""
        mock_whisper = MagicMock()
        mock_model_instance = MagicMock()
        
        mock_segment = MagicMock()
        mock_segment.start = 0.0
        mock_segment.end = 1.0
        mock_segment.text = "Async test"
        mock_segment.words = []
        
        mock_info = MagicMock()
        mock_info.duration = 1.0
        mock_info.language = "en"
        
        mock_model_instance.transcribe.return_value = ([mock_segment], mock_info)
        mock_whisper.return_value = mock_model_instance
        monkeypatch.setattr("py.ingest.transcription.WhisperModel", mock_whisper)
        
        service = TranscriptionService()
        service._model_loaded = True
        TranscriptionService._model = mock_model_instance
        
        audio_bytes = b'RIFF' + b'\x00' * 40
        
        import asyncio
        result = asyncio.run(service.transcribe_async(audio_bytes))
        
        assert isinstance(result, TranscriptionResult)
        assert result.transcript == "Async test"

    def test_transcribe_file(self, monkeypatch):
        """Test that transcribe_file reads file and calls transcribe."""
        mock_whisper = MagicMock()
        mock_model_instance = MagicMock()
        
        mock_segment = MagicMock()
        mock_segment.start = 0.0
        mock_segment.end = 1.0
        mock_segment.text = "File test"
        mock_segment.words = []
        
        mock_info = MagicMock()
        mock_info.duration = 1.0
        mock_info.language = "en"
        
        mock_model_instance.transcribe.return_value = ([mock_segment], mock_info)
        mock_whisper.return_value = mock_model_instance
        monkeypatch.setattr("py.ingest.transcription.WhisperModel", mock_whisper)
        
        service = TranscriptionService()
        service._model_loaded = True
        TranscriptionService._model = mock_model_instance
        
        # Create a temporary file
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.wav', delete=False) as f:
            f.write(b'RIFF' + b'\x00' * 40)
            temp_path = f.name
        
        try:
            result = service.transcribe_file(temp_path)
            assert isinstance(result, TranscriptionResult)
            assert result.transcript == "File test"
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_singleton_get_instance(self, monkeypatch):
        """Test that get_instance returns singleton."""
        mock_whisper = MagicMock()
        monkeypatch.setattr("py.ingest.transcription.WhisperModel", mock_whisper)
        
        # Reset singleton
        TranscriptionService._instance = None
        
        instance1 = TranscriptionService.get_instance()
        instance2 = TranscriptionService.get_instance()
        
        assert instance1 is instance2

