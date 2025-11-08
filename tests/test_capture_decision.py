import asyncio
import unittest
from datetime import datetime, timezone

from py.ingest.video import determine_capture_interval


class DummyChatStore:
    def __init__(self, count):
        self.count = count

    async def count_recent_messages(self, channel_id: str, window_seconds: int = 30) -> int:
        return self.count


class DummyVectorStore:
    def __init__(self, text: str):
        self.text = text

    async def get_recent_transcript_text(self, channel_id: str, window_seconds: int = 30):
        return self.text


class DetermineCaptureIntervalTests(unittest.IsolatedAsyncioTestCase):
    async def test_high_activity(self):
        decision = await determine_capture_interval(
            channel_id="123",
            captured_at=datetime.now(timezone.utc),
            chat_store=DummyChatStore(count=40),
            vector_store=DummyVectorStore(text="Big boss fight happening"),
            recent_chat_count=40,
            transcript_text="Big boss fight happening",
            was_interesting=False,
        )
        self.assertEqual(decision["next_interval_seconds"], 5)
        self.assertEqual(decision["recent_chat_count"], 40)
        self.assertTrue(decision["keyword_trigger"])

    async def test_low_activity(self):
        decision = await determine_capture_interval(
            channel_id="123",
            captured_at=datetime.now(timezone.utc),
            chat_store=DummyChatStore(count=2),
            vector_store=DummyVectorStore(text="Just chatting about settings"),
            recent_chat_count=2,
            transcript_text="Just chatting about settings",
            was_interesting=False,
        )
        self.assertEqual(decision["next_interval_seconds"], 10)
        self.assertFalse(decision["keyword_trigger"])


if __name__ == "__main__":
    unittest.main()

