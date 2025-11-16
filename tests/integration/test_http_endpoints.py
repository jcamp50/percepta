"""
Integration tests for FastAPI HTTP endpoints.

Tests key endpoints with mocked heavy dependencies (transcription, video store, etc.)
but real HTTP layer and database.
"""
import pytest
from datetime import datetime, timezone
from httpx import AsyncClient
from unittest.mock import AsyncMock, MagicMock, patch

from py.main import app


@pytest.mark.integration
@pytest.mark.asyncio
async def test_health_endpoint():
    """Test the /health endpoint."""
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "percepta-python"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_get_broadcaster_id_endpoint():
    """Test the /api/get-broadcaster-id endpoint."""
    # Mock the Twitch API call where it's imported in py.main
    with patch("py.main.get_broadcaster_id_from_channel_name") as mock_get_id:
        # Make it an async mock
        async def mock_get_id_async(*args, **kwargs):
            return "123456789"
        mock_get_id.side_effect = mock_get_id_async
        
        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.get("/api/get-broadcaster-id?channel_name=testchannel")
            
            assert response.status_code == 200
            data = response.json()
            assert data["broadcaster_id"] == "123456789"
            assert data["channel_name"] == "testchannel"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_chat_message_endpoint(db_session):
    """Test the /chat/message endpoint."""
    # Mock chat_store to avoid DB operations in this test
    with patch("py.main.chat_store") as mock_chat_store:
        mock_chat_store.insert_message = AsyncMock()
        mock_chat_store.is_available = True
        
        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.post(
                "/chat/message",
                json={
                    "channel": "#testchannel",
                    "username": "testuser",
                    "message": "Hello, world!",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
            )
            
            assert response.status_code == 200
            data = response.json()
            assert data["received"] is True
            assert "message_id" in data


@pytest.mark.integration
@pytest.mark.asyncio
async def test_chat_send_endpoint():
    """Test the /chat/send endpoint."""
    async with AsyncClient(app=app, base_url="http://test") as client:
        # First, queue a message via /chat/message
        with patch("py.main.chat_store"):
            await client.post(
                "/chat/message",
                json={
                    "channel": "#testchannel",
                    "username": "testuser",
                    "message": "@percepta ping",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
            )
        
        # Then retrieve it via /chat/send
        response = await client.post(
            "/chat/send",
            json={"channel": "#testchannel"},
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "messages" in data
        # May be empty if rate limited, but should be valid response


@pytest.mark.integration
@pytest.mark.asyncio
async def test_transcribe_endpoint_mocked():
    """Test the /transcribe endpoint with mocked transcription service."""
    # Create a minimal WAV file content (WAV header + some data)
    wav_header = b"RIFF" + b"\x00" * 4 + b"WAVE" + b"fmt " + b"\x10\x00\x00\x00" + b"\x01\x00" + b"\x01\x00" + b"\x44\xac\x00\x00" + b"\x88\x58\x01\x00" + b"\x02\x00" + b"\x10\x00" + b"data" + b"\x00" * 1000
    
    # Mock transcription service
    mock_result = MagicMock()
    mock_result.transcript = "Test transcription"
    mock_result.segments = []
    mock_result.language = "en"
    mock_result.duration = 1.0
    mock_result.model = "base"
    mock_result.processing_time_ms = 100
    
    with patch("py.main.transcription_service") as mock_transcription:
        mock_transcription.transcribe_async = AsyncMock(return_value=mock_result)
        mock_transcription.is_available = True
        
        # Mock vector_store to avoid DB operations
        with patch("py.main.vector_store") as mock_vector_store:
            mock_vector_store.insert_transcript = AsyncMock(return_value="test-id")
            mock_vector_store.is_available = True
            
            async with AsyncClient(app=app, base_url="http://test") as client:
                response = await client.post(
                    "/transcribe",
                    files={"audio_file": ("test.wav", wav_header, "audio/wav")},
                    data={
                        "channel_id": "123456789",
                        "started_at": datetime.now(timezone.utc).isoformat(),
                        "ended_at": datetime.now(timezone.utc).isoformat(),
                    },
                )
                
                assert response.status_code == 200
                data = response.json()
                assert data["transcript"] == "Test transcription"
                assert data["language"] == "en"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_video_frame_endpoint_mocked():
    """Test the /api/video-frame endpoint with mocked video store."""
    # Create a minimal JPEG file content
    jpeg_content = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00H\x00H\x00\x00\xff\xdb\x00C\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\t\t\x08\n\x0c\x14\r\x0c\x0b\x0b\x0e\x11\x0e\x0f\x11\x17\x1a\x16\x14\x18\x19\x17\x1a\x1f\x1e\x1b\x1b\x1e\x1f!\x1f\x1f\x1f\x1f\x1f\x1f\x1f\x1f\x1f\x1f\x1f\x1f\x1f\x1f\x1f\x1f\x1f\x1f\x1f\x1f\x1f\x1f\x1f\x1f\x1f\x1f\x1f\x1f\x1f\x1f\x1f\x1f\x1f\x1f\x1f\x1f\x1f\x1f\x1f\x1f\x1f\x1f\x1f\x1f\x1f\x1f\x1f\x1f\x1f\x1f\xff\xc0\x00\x11\x08\x00\x01\x00\x01\x01\x01\x11\x00\x02\x11\x01\x03\x11\x01\xff\xc4\x00\x14\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x08\xff\xc4\x00\x14\x10\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xff\xda\x00\x08\x01\x01\x00\x00?\x00\xff\xd9"
    
    # Mock video_store
    with patch("py.main.video_store") as mock_video_store:
        mock_video_store.insert_frame = AsyncMock(
            return_value={
                "frame_id": "test-frame-id",
                "chat_count": 0,
                "transcript_text": None,
                "was_interesting": False,
            }
        )
        mock_video_store.is_available = True
        
        # Mock chat_store and vector_store for determine_capture_interval
        with patch("py.main.chat_store"), patch("py.main.vector_store"):
            async with AsyncClient(app=app, base_url="http://test") as client:
                response = await client.post(
                    "/api/video-frame",
                    files={"image_file": ("test.jpg", jpeg_content, "image/jpeg")},
                    data={
                        "channel_id": "123456789",
                        "captured_at": datetime.now(timezone.utc).isoformat(),
                    },
                )
                
                assert response.status_code == 200
                data = response.json()
                assert data["status"] == "success"
                assert "frame_id" in data


@pytest.mark.integration
@pytest.mark.asyncio
async def test_transcribe_endpoint_empty_file():
    """Test /transcribe endpoint with empty file (should return 400)."""
    with patch("py.main.transcription_service") as mock_transcription:
        mock_transcription.is_available = True
        
        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.post(
                "/transcribe",
                files={"audio_file": ("empty.wav", b"", "audio/wav")},
                data={
                    "channel_id": "123456789",
                    "started_at": datetime.now(timezone.utc).isoformat(),
                    "ended_at": datetime.now(timezone.utc).isoformat(),
                },
            )
            
            assert response.status_code == 400
            assert "empty" in response.json()["detail"].lower()

