"""Unit tests for capture decision logic."""
import asyncio
import pytest
from datetime import datetime, timezone

from py.ingest.video import determine_capture_interval


@pytest.mark.unit
@pytest.mark.asyncio
class TestDetermineCaptureInterval:
    """Test determine_capture_interval function."""

    async def test_high_activity_chat_threshold(self):
        """Test that high chat count triggers active interval."""
        class DummyChatStore:
            async def count_recent_messages(self, channel_id, window_seconds):
                return 30  # Above threshold (default 25)
        
        class DummyVectorStore:
            async def get_recent_transcript_text(self, channel_id, window_seconds):
                return "Just chatting"
        
        decision = await determine_capture_interval(
            channel_id="123",
            captured_at=datetime.now(timezone.utc),
            chat_store=DummyChatStore(),
            vector_store=DummyVectorStore(),
        )
        
        assert decision["next_interval_seconds"] == 5  # active_interval
        assert decision["recent_chat_count"] == 30
        assert decision["keyword_trigger"] is False

    async def test_keyword_trigger(self):
        """Test that keyword in transcript triggers active interval."""
        class DummyChatStore:
            async def count_recent_messages(self, channel_id, window_seconds):
                return 5  # Below threshold
        
        class DummyVectorStore:
            async def get_recent_transcript_text(self, channel_id, window_seconds):
                return "Big boss fight happening"  # Contains "boss"
        
        decision = await determine_capture_interval(
            channel_id="123",
            captured_at=datetime.now(timezone.utc),
            chat_store=DummyChatStore(),
            vector_store=DummyVectorStore(),
        )
        
        assert decision["next_interval_seconds"] == 5  # active_interval
        assert decision["keyword_trigger"] is True

    async def test_was_interesting_short_circuit(self):
        """Test that was_interesting=True triggers active interval."""
        class DummyChatStore:
            async def count_recent_messages(self, channel_id, window_seconds):
                return 2  # Low activity
        
        class DummyVectorStore:
            async def get_recent_transcript_text(self, channel_id, window_seconds):
                return "Just chatting"  # No keywords
        
        decision = await determine_capture_interval(
            channel_id="123",
            captured_at=datetime.now(timezone.utc),
            chat_store=DummyChatStore(),
            vector_store=DummyVectorStore(),
            was_interesting=True,
        )
        
        assert decision["next_interval_seconds"] == 5  # active_interval

    async def test_low_activity_baseline_interval(self):
        """Test that low activity uses baseline interval."""
        class DummyChatStore:
            async def count_recent_messages(self, channel_id, window_seconds):
                return 2  # Low activity
        
        class DummyVectorStore:
            async def get_recent_transcript_text(self, channel_id, window_seconds):
                return "Just chatting"  # No keywords
        
        decision = await determine_capture_interval(
            channel_id="123",
            captured_at=datetime.now(timezone.utc),
            chat_store=DummyChatStore(),
            vector_store=DummyVectorStore(),
            was_interesting=False,
        )
        
        assert decision["next_interval_seconds"] == 10  # baseline_interval
        assert decision["keyword_trigger"] is False

    async def test_custom_thresholds(self):
        """Test that custom thresholds override defaults."""
        class DummyChatStore:
            async def count_recent_messages(self, channel_id, window_seconds):
                return 15
        
        class DummyVectorStore:
            async def get_recent_transcript_text(self, channel_id, window_seconds):
                return "Just chatting"
        
        decision = await determine_capture_interval(
            channel_id="123",
            captured_at=datetime.now(timezone.utc),
            chat_store=DummyChatStore(),
            vector_store=DummyVectorStore(),
            baseline_interval=15,
            active_interval=3,
            chat_threshold=10,  # 15 > 10, should trigger active
        )
        
        assert decision["next_interval_seconds"] == 3  # custom active_interval

    async def test_keyword_list_override(self):
        """Test that custom keyword list is used."""
        class DummyChatStore:
            async def count_recent_messages(self, channel_id, window_seconds):
                return 2
        
        class DummyVectorStore:
            async def get_recent_transcript_text(self, channel_id, window_seconds):
                return "epic moment"  # Not in default keywords
        
        # With default keywords, "epic" wouldn't trigger
        decision_default = await determine_capture_interval(
            channel_id="123",
            captured_at=datetime.now(timezone.utc),
            chat_store=DummyChatStore(),
            vector_store=DummyVectorStore(),
        )
        assert decision_default["keyword_trigger"] is False
        
        # With custom keyword list including "epic"
        decision_custom = await determine_capture_interval(
            channel_id="123",
            captured_at=datetime.now(timezone.utc),
            chat_store=DummyChatStore(),
            vector_store=DummyVectorStore(),
            keyword_triggers=["epic", "moment"],
        )
        assert decision_custom["keyword_trigger"] is True
        assert decision_custom["next_interval_seconds"] == 5

    async def test_chat_store_failure_fallback(self):
        """Test that chat store failure falls back gracefully."""
        class FailingChatStore:
            async def count_recent_messages(self, channel_id, window_seconds):
                raise Exception("Chat store error")
        
        class DummyVectorStore:
            async def get_recent_transcript_text(self, channel_id, window_seconds):
                return "Just chatting"
        
        # Should not raise, should use fallback count (0)
        decision = await determine_capture_interval(
            channel_id="123",
            captured_at=datetime.now(timezone.utc),
            chat_store=FailingChatStore(),
            vector_store=DummyVectorStore(),
        )
        
        assert decision["recent_chat_count"] == 0
        assert decision["next_interval_seconds"] == 10  # baseline (low activity)

    async def test_vector_store_failure_fallback(self):
        """Test that vector store failure falls back gracefully."""
        class DummyChatStore:
            async def count_recent_messages(self, channel_id, window_seconds):
                return 2
        
        class FailingVectorStore:
            async def get_recent_transcript_text(self, channel_id, window_seconds):
                raise Exception("Vector store error")
        
        # Should not raise, should use None for transcript_text
        decision = await determine_capture_interval(
            channel_id="123",
            captured_at=datetime.now(timezone.utc),
            chat_store=DummyChatStore(),
            vector_store=FailingVectorStore(),
        )
        
        assert decision["keyword_trigger"] is False
        assert decision["next_interval_seconds"] == 10  # baseline

    async def test_provided_recent_chat_count(self):
        """Test that provided recent_chat_count is used instead of querying."""
        class DummyChatStore:
            async def count_recent_messages(self, channel_id, window_seconds):
                return 999  # Should not be called
        
        class DummyVectorStore:
            async def get_recent_transcript_text(self, channel_id, window_seconds):
                return "Just chatting"
        
        decision = await determine_capture_interval(
            channel_id="123",
            captured_at=datetime.now(timezone.utc),
            chat_store=DummyChatStore(),
            vector_store=DummyVectorStore(),
            recent_chat_count=30,  # Provided directly
        )
        
        assert decision["recent_chat_count"] == 30
        assert decision["next_interval_seconds"] == 5  # active due to high count

    async def test_provided_transcript_text(self):
        """Test that provided transcript_text is used instead of querying."""
        class DummyChatStore:
            async def count_recent_messages(self, channel_id, window_seconds):
                return 2
        
        class DummyVectorStore:
            async def get_recent_transcript_text(self, channel_id, window_seconds):
                return "should not be used"
        
        decision = await determine_capture_interval(
            channel_id="123",
            captured_at=datetime.now(timezone.utc),
            chat_store=DummyChatStore(),
            vector_store=DummyVectorStore(),
            transcript_text="boss fight",  # Provided directly
        )
        
        assert decision["keyword_trigger"] is True
        assert decision["next_interval_seconds"] == 5

    async def test_minimum_interval_enforcement(self):
        """Test that intervals are enforced to be at least 2 seconds."""
        class DummyChatStore:
            async def count_recent_messages(self, channel_id, window_seconds):
                return 2
        
        class DummyVectorStore:
            async def get_recent_transcript_text(self, channel_id, window_seconds):
                return "Just chatting"
        
        # Try to set baseline to 1 (should be clamped to 2)
        decision = await determine_capture_interval(
            channel_id="123",
            captured_at=datetime.now(timezone.utc),
            chat_store=DummyChatStore(),
            vector_store=DummyVectorStore(),
            baseline_interval=1,  # Too low
        )
        
        assert decision["next_interval_seconds"] >= 2

