"""Smoke tests for ORM models."""
import uuid
import pytest
from datetime import datetime, timezone

from py.database.models import (
    Transcript,
    Event,
    ChannelSnapshot,
    VideoFrame,
    ChatMessage,
    Summary,
)


@pytest.mark.unit
class TestModelsSmoke:
    """Smoke tests to verify models can be instantiated."""

    def test_transcript_model(self):
        """Test Transcript model instantiation."""
        transcript = Transcript(
            channel_id="123456",
            started_at=datetime.now(timezone.utc),
            ended_at=datetime.now(timezone.utc),
            text="Test transcript text",
            embedding=[0.1] * 1536,
        )
        
        assert transcript.channel_id == "123456"
        assert transcript.text == "Test transcript text"
        assert len(transcript.embedding) == 1536
        assert isinstance(transcript.id, uuid.UUID)

    def test_event_model(self):
        """Test Event model instantiation."""
        event = Event(
            channel_id="123456",
            ts=datetime.now(timezone.utc),
            type="stream.online",
            summary="Stream went online",
            payload_json={"test": "data"},
            embedding=[0.2] * 1536,
        )
        
        assert event.channel_id == "123456"
        assert event.type == "stream.online"
        assert event.summary == "Stream went online"
        assert event.payload_json == {"test": "data"}
        assert len(event.embedding) == 1536
        assert isinstance(event.id, uuid.UUID)

    def test_event_model_optional_fields(self):
        """Test Event model with optional fields as None."""
        event = Event(
            channel_id="123456",
            ts=datetime.now(timezone.utc),
            type="stream.online",
        )
        
        assert event.summary is None
        assert event.payload_json is None
        assert event.embedding is None

    def test_channel_snapshot_model(self):
        """Test ChannelSnapshot model instantiation."""
        snapshot = ChannelSnapshot(
            channel_id="123456",
            ts=datetime.now(timezone.utc),
            title="Test Stream",
            game_id="12345",
            game_name="Test Game",
            tags=["tag1", "tag2"],
            viewer_count=100,
            payload_json={"test": "data"},
            embedding=[0.3] * 1536,
        )
        
        assert snapshot.channel_id == "123456"
        assert snapshot.title == "Test Stream"
        assert snapshot.game_name == "Test Game"
        assert snapshot.tags == ["tag1", "tag2"]
        assert snapshot.viewer_count == 100
        assert len(snapshot.embedding) == 1536
        assert isinstance(snapshot.id, uuid.UUID)

    def test_channel_snapshot_optional_fields(self):
        """Test ChannelSnapshot with optional fields."""
        snapshot = ChannelSnapshot(
            channel_id="123456",
            ts=datetime.now(timezone.utc),
        )
        
        assert snapshot.title is None
        assert snapshot.game_id is None
        assert snapshot.tags is None
        assert snapshot.viewer_count is None
        assert snapshot.embedding is None

    def test_video_frame_model(self):
        """Test VideoFrame model instantiation."""
        frame = VideoFrame(
            channel_id="123456",
            captured_at=datetime.now(timezone.utc),
            image_path="/path/to/frame.jpg",
            embedding=[0.4] * 1536,
            grounded_embedding=[0.5] * 1536,
            description="A test frame",
            description_json={"source": "test"},
            description_source="test_source",
            frame_hash="abc123",
            aligned_chat_ids=["msg1", "msg2"],
            metadata_snapshot={"viewers": 100},
        )
        
        assert frame.channel_id == "123456"
        assert frame.image_path == "/path/to/frame.jpg"
        assert len(frame.embedding) == 1536
        assert len(frame.grounded_embedding) == 1536
        assert frame.description == "A test frame"
        assert frame.frame_hash == "abc123"
        assert frame.aligned_chat_ids == ["msg1", "msg2"]
        assert isinstance(frame.id, uuid.UUID)

    def test_video_frame_optional_fields(self):
        """Test VideoFrame with optional fields."""
        frame = VideoFrame(
            channel_id="123456",
            captured_at=datetime.now(timezone.utc),
            image_path="/path/to/frame.jpg",
            embedding=[0.4] * 1536,
        )
        
        assert frame.grounded_embedding is None
        assert frame.description is None
        assert frame.frame_hash is None
        assert frame.transcript_id is None
        assert frame.aligned_chat_ids is None

    def test_chat_message_model(self):
        """Test ChatMessage model instantiation."""
        message = ChatMessage(
            channel_id="123456",
            username="test_user",
            message="Hello world",
            sent_at=datetime.now(timezone.utc),
            embedding=[0.6] * 1536,
        )
        
        assert message.channel_id == "123456"
        assert message.username == "test_user"
        assert message.message == "Hello world"
        assert len(message.embedding) == 1536
        assert isinstance(message.id, uuid.UUID)

    def test_summary_model(self):
        """Test Summary model instantiation."""
        summary = Summary(
            channel_id="123456",
            start_time=datetime.now(timezone.utc),
            end_time=datetime.now(timezone.utc),
            summary_text="This is a summary",
            summary_json={"key": "value"},
            embedding=[0.7] * 1536,
            segment_number=1,
        )
        
        assert summary.channel_id == "123456"
        assert summary.summary_text == "This is a summary"
        assert summary.segment_number == 1
        assert len(summary.embedding) == 1536
        assert isinstance(summary.id, uuid.UUID)

    def test_summary_optional_fields(self):
        """Test Summary with optional fields."""
        summary = Summary(
            channel_id="123456",
            start_time=datetime.now(timezone.utc),
            end_time=datetime.now(timezone.utc),
            summary_text="Summary text",
            embedding=[0.7] * 1536,
            segment_number=1,
        )
        
        assert summary.summary_json is None

