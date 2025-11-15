"""Tests for grounded embedding fusion functionality (JCB-37)."""

import unittest
from unittest import mock

import numpy as np

from py.utils.video_embeddings import create_grounded_embedding, TARGET_EMBEDDING_DIM


class GroundedEmbeddingTests(unittest.IsolatedAsyncioTestCase):
    """Test grounded embedding creation."""

    async def test_create_grounded_embedding_fuses_correctly(self):
        """Test that grounded embedding correctly fuses CLIP and text embeddings."""
        # Create mock embeddings (normalized unit vectors)
        clip_embedding = np.random.randn(TARGET_EMBEDDING_DIM).tolist()
        clip_array = np.array(clip_embedding)
        clip_normalized = clip_array / np.linalg.norm(clip_array)
        clip_embedding = clip_normalized.tolist()

        # Mock text embedding (different from CLIP)
        text_embedding = np.random.randn(TARGET_EMBEDDING_DIM).tolist()
        text_array = np.array(text_embedding)
        text_normalized = text_array / np.linalg.norm(text_array)
        text_embedding = text_normalized.tolist()

        description_text = "A boss fight scene with intense action"

        # Mock embed_text to return our test text embedding
        with mock.patch("py.utils.video_embeddings.embed_text") as mock_embed:
            mock_embed.return_value = text_embedding

            result = await create_grounded_embedding(
                clip_embedding=clip_embedding,
                description_text=description_text,
            )

            # Verify embed_text was called with description
            mock_embed.assert_called_once_with(description_text)

            # Verify result is correct dimension
            self.assertEqual(len(result), TARGET_EMBEDDING_DIM)

            # Verify result is normalized (unit vector)
            result_norm = np.linalg.norm(result)
            self.assertAlmostEqual(result_norm, 1.0, places=5)

            # Verify result is different from both inputs (fusion occurred)
            clip_diff = np.linalg.norm(np.array(result) - np.array(clip_embedding))
            text_diff = np.linalg.norm(np.array(result) - np.array(text_embedding))
            self.assertGreater(clip_diff, 0.01)  # Should be different
            self.assertGreater(text_diff, 0.01)  # Should be different

            # Verify weighted fusion: result should be closer to CLIP (70% weight)
            clip_distance = np.linalg.norm(np.array(result) - np.array(clip_embedding))
            text_distance = np.linalg.norm(np.array(result) - np.array(text_embedding))
            self.assertLess(
                clip_distance, text_distance, "Result should be closer to CLIP embedding"
            )

    async def test_create_grounded_embedding_requires_description(self):
        """Test that empty description raises ValueError."""
        clip_embedding = [0.1] * TARGET_EMBEDDING_DIM

        with self.assertRaises(ValueError):
            await create_grounded_embedding(
                clip_embedding=clip_embedding, description_text=""
            )

        with self.assertRaises(ValueError):
            await create_grounded_embedding(
                clip_embedding=clip_embedding, description_text="   "
            )

    async def test_create_grounded_embedding_validates_dimensions(self):
        """Test that incorrect embedding dimensions raise ValueError."""
        wrong_dim_clip = [0.1] * 512  # Wrong dimension
        description_text = "Test description"

        with self.assertRaises(ValueError):
            await create_grounded_embedding(
                clip_embedding=wrong_dim_clip, description_text=description_text
            )

