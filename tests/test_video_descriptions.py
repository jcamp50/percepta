import asyncio
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from py.utils import video_descriptions as vd
from py.utils.video_descriptions import _build_description_prompt


class PromptBuilderTests(unittest.TestCase):
    def test_build_description_prompt_includes_context(self):
        prompt = _build_description_prompt(
            previous_frame_description="Prev frame text",
            recent_summary="Recent summary text about the boss fight",
            transcript={"text": "The boss is almost down!"},
            chat=[
                {"username": "viewer1", "message": "let's go!"},
                {"username": "viewer2", "message": "huge damage"},
            ],
            metadata={"game_name": "Action Game", "title": "Raid Night"},
        )

        self.assertIn("Participants & Appearance", prompt)
        self.assertIn("Primary Scene / Activity", prompt)
        self.assertIn("Overlay & Interface Elements", prompt)
        self.assertIn("Chat discussion", prompt)
        self.assertIn("Game: Action Game", prompt)

    def test_build_description_prompt_handles_empty_context(self):
        prompt = _build_description_prompt()
        self.assertIn("CRITICAL REQUIREMENTS", prompt)
        self.assertNotIn("CONTEXT:", prompt)


class GenerateDescriptionRetryTests(unittest.IsolatedAsyncioTestCase):
    async def test_generate_frame_description_retries_on_rate_limit(self):
        sleep_calls = []

        class DummyRateLimitError(Exception):
            pass

        call_counter = {"count": 0}

        class FakeCompletions:
            def create(self, **kwargs):
                call_counter["count"] += 1
                if call_counter["count"] < 3:
                    raise DummyRateLimitError("rate limit")
                return SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            message=SimpleNamespace(content="Recovered description")
                        )
                    ]
                )

        class FakeChat:
            def __init__(self):
                self.completions = FakeCompletions()

        class FakeClient:
            def __init__(self):
                self.chat = FakeChat()

        image_dir = tempfile.TemporaryDirectory()
        image_path = Path(image_dir.name) / "frame.jpg"
        image_path.write_bytes(b"\xff\xd8\xff")

        async def fake_sleep(delay):
            sleep_calls.append(delay)

        async def fake_to_thread(func, *a, **kw):
            return func(*a, **kw)

        with mock.patch.object(
            vd, "RateLimitError", DummyRateLimitError
        ), mock.patch.object(vd.asyncio, "sleep", new=fake_sleep), mock.patch.object(
            vd.asyncio, "to_thread", new=fake_to_thread
        ):
            result = await vd.generate_frame_description(
                image_path=str(image_path),
                channel_id="test-channel",
                captured_at=datetime.utcnow(),
                openai_client=FakeClient(),
            )

        image_dir.cleanup()

        self.assertEqual(result, "Recovered description")
        self.assertEqual(call_counter["count"], 3)
        self.assertEqual(len(sleep_calls), 2)


if __name__ == "__main__":
    unittest.main()
