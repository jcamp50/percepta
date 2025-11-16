"""Unit tests for CLIP embedding projection."""
import numpy as np
import pytest

from py.utils.video_embeddings import (
    project_clip_to_1536,
    CLIP_EMBEDDING_DIM,
    TARGET_EMBEDDING_DIM,
)


@pytest.mark.unit
class TestProjectClipTo1536:
    """Test project_clip_to_1536 function."""

    def test_projection_dimension(self):
        """Test that projection produces correct dimension."""
        vec = np.ones(CLIP_EMBEDDING_DIM, dtype=np.float32).tolist()
        result = project_clip_to_1536(vec)
        assert len(result) == TARGET_EMBEDDING_DIM

    def test_projection_normalization(self):
        """Test that projected vector is normalized (unit vector)."""
        vec = np.random.rand(CLIP_EMBEDDING_DIM).astype(np.float32).tolist()
        result = project_clip_to_1536(vec)
        norm = np.linalg.norm(result)
        assert abs(norm - 1.0) < 1e-5, f"Expected unit vector, got norm={norm}"

    def test_projection_preserves_similarity(self):
        """Test that projection preserves cosine similarity properties."""
        vec1 = np.random.rand(CLIP_EMBEDDING_DIM).astype(np.float32).tolist()
        vec2 = np.random.rand(CLIP_EMBEDDING_DIM).astype(np.float32).tolist()
        
        proj1 = project_clip_to_1536(vec1)
        proj2 = project_clip_to_1536(vec2)
        
        # Original cosine similarity
        orig_sim = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        # Projected cosine similarity (both are normalized)
        proj_sim = np.dot(proj1, proj2)
        
        # Should be approximately equal (within floating point precision)
        assert abs(orig_sim - proj_sim) < 1e-3

    def test_invalid_dimension_raises(self):
        """Test that invalid input dimension raises ValueError."""
        with pytest.raises(ValueError, match="Expected CLIP embedding dimension"):
            project_clip_to_1536([1.0] * 256)  # Wrong dimension

    def test_zero_vector_handling(self):
        """Test that zero vector is handled gracefully."""
        vec = [0.0] * CLIP_EMBEDDING_DIM
        result = project_clip_to_1536(vec)
        assert len(result) == TARGET_EMBEDDING_DIM
        # Should still be normalized (will be uniform vector)
        norm = np.linalg.norm(result)
        assert abs(norm - 1.0) < 1e-5

    def test_repetition_pattern(self):
        """Test that projection uses repetition pattern (512 * 3 = 1536)."""
        # Create a vector with distinct values
        vec = [float(i) for i in range(CLIP_EMBEDDING_DIM)]
        result = project_clip_to_1536(vec)
        
        # Check that first 512 values match original (before normalization)
        # After normalization, values will be scaled, but pattern should repeat
        assert len(result) == TARGET_EMBEDDING_DIM
        # Verify it's a repetition by checking first 512 vs next 512 (scaled)
        first_chunk = np.array(result[:CLIP_EMBEDDING_DIM])
        second_chunk = np.array(result[CLIP_EMBEDDING_DIM:2*CLIP_EMBEDDING_DIM])
        # After normalization, chunks should be proportional
        if np.linalg.norm(first_chunk) > 0:
            ratio = np.linalg.norm(second_chunk) / np.linalg.norm(first_chunk)
            assert abs(ratio - 1.0) < 1e-5  # Should be same magnitude after normalization

