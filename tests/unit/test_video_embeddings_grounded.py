"""Unit tests for grounded embedding creation."""
import numpy as np
import pytest

from py.utils.video_embeddings import (
    create_grounded_embedding,
    TARGET_EMBEDDING_DIM,
)


@pytest.mark.unit
@pytest.mark.asyncio
class TestCreateGroundedEmbedding:
    """Test create_grounded_embedding function."""

    async def test_grounded_fusion_dimension(self, monkeypatch):
        """Test that grounded embedding has correct dimension."""
        # Mock embed_text to return deterministic vector
        async def fake_embed_text(text):
            return [0.5] * TARGET_EMBEDDING_DIM
        
        monkeypatch.setattr(
            "py.utils.video_embeddings.embed_text",
            fake_embed_text
        )
        
        clip_embedding = [1.0] * TARGET_EMBEDDING_DIM
        result = await create_grounded_embedding(clip_embedding, "test description")
        
        assert len(result) == TARGET_EMBEDDING_DIM

    async def test_grounded_fusion_normalization(self, monkeypatch):
        """Test that grounded embedding is normalized."""
        async def fake_embed_text(text):
            return [0.3] * TARGET_EMBEDDING_DIM
        
        monkeypatch.setattr(
            "py.utils.video_embeddings.embed_text",
            fake_embed_text
        )
        
        clip_embedding = [0.7] * TARGET_EMBEDDING_DIM
        result = await create_grounded_embedding(clip_embedding, "test description")
        
        norm = np.linalg.norm(result)
        assert abs(norm - 1.0) < 1e-5, f"Expected unit vector, got norm={norm}"

    async def test_grounded_fusion_weights(self, monkeypatch):
        """Test that fusion uses correct weights (70% CLIP, 30% text)."""
        # Use distinct vectors to verify weights
        clip_vec = np.array([1.0] * TARGET_EMBEDDING_DIM, dtype=np.float32)
        text_vec = np.array([0.0] * TARGET_EMBEDDING_DIM, dtype=np.float32)
        
        # Normalize clip vector
        clip_norm = clip_vec / np.linalg.norm(clip_vec)
        
        async def fake_embed_text(text):
            return text_vec.tolist()
        
        monkeypatch.setattr(
            "py.utils.video_embeddings.embed_text",
            fake_embed_text
        )
        
        result = await create_grounded_embedding(clip_vec.tolist(), "test")
        result_norm = np.linalg.norm(result)
        
        # Result should be normalized
        assert abs(result_norm - 1.0) < 1e-5
        
        # With text_vec = 0, result should be exactly clip_norm (after normalization)
        # Since we're fusing 0.7 * clip_norm + 0.3 * 0 = 0.7 * clip_norm
        # Then normalized, it should equal clip_norm
        np.testing.assert_allclose(result, clip_norm, rtol=1e-5)

    async def test_empty_description_raises(self):
        """Test that empty description raises ValueError."""
        clip_embedding = [1.0] * TARGET_EMBEDDING_DIM
        
        with pytest.raises(ValueError, match="description_text is required"):
            await create_grounded_embedding(clip_embedding, "")
        
        with pytest.raises(ValueError, match="description_text is required"):
            await create_grounded_embedding(clip_embedding, "   ")

    async def test_invalid_clip_dimension_raises(self, monkeypatch):
        """Test that invalid CLIP embedding dimension raises ValueError."""
        async def fake_embed_text(text):
            return [0.0] * TARGET_EMBEDDING_DIM
        
        monkeypatch.setattr(
            "py.utils.video_embeddings.embed_text",
            fake_embed_text
        )
        
        invalid_clip = [1.0] * 512  # Wrong dimension
        
        with pytest.raises(ValueError, match="Expected CLIP embedding dimension"):
            await create_grounded_embedding(invalid_clip, "test description")

    async def test_invalid_text_dimension_raises(self, monkeypatch):
        """Test that invalid text embedding dimension raises ValueError."""
        async def fake_embed_text(text):
            return [0.0] * 512  # Wrong dimension
        
        monkeypatch.setattr(
            "py.utils.video_embeddings.embed_text",
            fake_embed_text
        )
        
        clip_embedding = [1.0] * TARGET_EMBEDDING_DIM
        
        with pytest.raises(ValueError, match="Expected text embedding dimension"):
            await create_grounded_embedding(clip_embedding, "test description")

    async def test_fusion_combines_both_embeddings(self, monkeypatch):
        """Test that fusion actually combines both embeddings."""
        # Use orthogonal vectors to verify both contribute
        clip_vec = np.array([1.0] + [0.0] * (TARGET_EMBEDDING_DIM - 1), dtype=np.float32)
        text_vec = np.array([0.0] * (TARGET_EMBEDDING_DIM - 1) + [1.0], dtype=np.float32)
        
        # Normalize
        clip_norm = clip_vec / np.linalg.norm(clip_vec)
        text_norm = text_vec / np.linalg.norm(text_vec)
        
        async def fake_embed_text(text):
            return text_norm.tolist()
        
        monkeypatch.setattr(
            "py.utils.video_embeddings.embed_text",
            fake_embed_text
        )
        
        result = await create_grounded_embedding(clip_norm.tolist(), "test")
        
        # Result should have contributions from both
        # 0.7 * clip_norm + 0.3 * text_norm, then normalized
        expected = 0.7 * clip_norm + 0.3 * text_norm
        expected = expected / np.linalg.norm(expected)
        
        np.testing.assert_allclose(result, expected, rtol=1e-5)

