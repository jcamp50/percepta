import unittest
from datetime import datetime, timedelta

from py.reason.interfaces import SearchResult
from py.reason.retriever import Retriever, RetrievalParams


class FakeVectorStore:
    def __init__(self, base_time: datetime):
        self.base_time = base_time

    async def search_transcripts(
        self, *, query_embedding, limit, half_life_minutes, channel_id, prefilter_limit
    ):
        return [
            {
                "id": "t1",
                "channel_id": channel_id,
                "text": "Boss is low!",
                "started_at": self.base_time,
                "ended_at": self.base_time + timedelta(seconds=2),
                "cosine_distance": 0.1,
                "score": 0.1,
            }
        ]

    async def search_events(
        self, *, query_embedding, limit, half_life_minutes, channel_id, prefilter_limit
    ):
        return [
            {
                "id": "e1",
                "channel_id": channel_id,
                "summary": "Streamer triggered ultimate.",
                "ts": self.base_time + timedelta(seconds=1),
                "cosine_distance": 0.2,
                "score": 0.2,
            }
        ]

    async def get_transcript_by_id(self, transcript_id: str):
        if transcript_id == "t1":
            return {
                "id": transcript_id,
                "text": "Boss is low!",
                "started_at": self.base_time,
                "ended_at": self.base_time + timedelta(seconds=2),
            }
        return None


class FakeVideoStore:
    def __init__(self, base_time: datetime):
        self.base_time = base_time
        self.generated_ids = []

    async def search_frames(
        self, query_embedding, limit, half_life_minutes, channel_id, prefilter_limit
    ):
        return [
            {
                "id": "v1",
                "channel_id": channel_id,
                "captured_at": self.base_time + timedelta(seconds=1),
                "image_path": "frames/channel/v1.jpg",
                "description": None,
                "description_source": "lazy",
                "frame_hash": None,
                "transcript_id": "t1",
                "aligned_chat_ids": ["c1"],
                "metadata_snapshot": {"game_name": "Test Game", "title": "Boss Fight"},
                "cosine_distance": 0.3,
                "score": 0.3,
            }
        ]

    async def generate_description_for_frame(self, frame_id: str):
        self.generated_ids.append(frame_id)
        return "Generated fallback description"


class FakeChatStore:
    def __init__(self, base_time: datetime):
        self.base_time = base_time
        self.requested_ids = []

    async def search_messages(
        self, query_embedding, limit, half_life_minutes, channel_id, prefilter_limit
    ):
        return [
            {
                "id": "c2",
                "channel_id": channel_id,
                "username": "viewer2",
                "message": "Let's finish this!",
                "sent_at": self.base_time + timedelta(seconds=2),
                "cosine_distance": 0.4,
                "score": 0.4,
            }
        ]

    async def get_messages_by_ids(self, message_ids):
        self.requested_ids.append(tuple(message_ids))
        return [
            {
                "id": "c1",
                "channel_id": "channel-1",
                "username": "viewer1",
                "message": "Huge crit!",
                "sent_at": self.base_time + timedelta(seconds=1),
            }
        ]


class RetrieverFusionTests(unittest.IsolatedAsyncioTestCase):
    async def test_retrieve_merges_modalities_and_generates_video_description(self):
        base_time = datetime.utcnow()
        vector_store = FakeVectorStore(base_time)
        video_store = FakeVideoStore(base_time)
        chat_store = FakeChatStore(base_time)

        retriever = Retriever(
            vector_store=vector_store, video_store=video_store, chat_store=chat_store
        )

        params = RetrievalParams(
            channel_id="channel-1", limit=5, half_life_minutes=60, prefilter_limit=10
        )

        results = await retriever.retrieve(query_embedding=[0.0], params=params)

        self.assertEqual(len(results), 1)
        fused = results[0]
        self.assertIn("[Transcript] Boss is low!", fused.text)
        self.assertIn("[Video Frame] Generated fallback description", fused.text)
        self.assertIn("Chat:", fused.text)
        self.assertIn("Metadata: Game: Test Game", fused.text)
        self.assertTrue(video_store.generated_ids)
        self.assertIn("c1", chat_store.requested_ids[0])


class RetrieverAlignmentTests(unittest.TestCase):
    def test_merge_and_rank_respects_time_window(self):
        retriever = Retriever(vector_store=object())
        base_time = datetime.utcnow()
        transcripts = [
            SearchResult(
                id="t1",
                channel_id="channel-1",
                text="[Transcript] First event",
                started_at=base_time,
                ended_at=base_time + timedelta(seconds=2),
                cosine_distance=0.1,
                score=0.1,
            ),
            SearchResult(
                id="t2",
                channel_id="channel-1",
                text="[Transcript] Far later event",
                started_at=base_time + timedelta(seconds=12),
                ended_at=base_time + timedelta(seconds=14),
                cosine_distance=0.2,
                score=0.2,
            ),
        ]

        fused = retriever._merge_and_rank(
            transcripts=transcripts,
            events=[],
            videos=[],
            chats=[],
            limit=5,
        )

        self.assertEqual(len(fused), 2)
        self.assertTrue(fused[0].text.startswith("[Transcript]"))
        self.assertTrue(fused[1].text.startswith("[Transcript]"))


if __name__ == "__main__":
    unittest.main()

