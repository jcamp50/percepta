"""Test utilities and helpers."""
import numpy as np


# Constants for test embeddings
TEST_EMBEDDING_DIM = 1536
TEST_CLIP_DIM = 512


def create_test_embedding(dim=TEST_EMBEDDING_DIM, seed=None):
    """Create a deterministic test embedding vector.
    
    Args:
        dim: Dimension of the embedding (default: 1536)
        seed: Random seed for reproducibility (default: None)
    
    Returns:
        List of floats representing an embedding vector
    """
    if seed is not None:
        np.random.seed(seed)
    vec = np.random.rand(dim).astype(np.float32)
    # Normalize to unit vector
    vec = vec / np.linalg.norm(vec)
    return vec.tolist()


def create_test_clip_embedding(seed=None):
    """Create a deterministic CLIP embedding vector (512 dim).
    
    Args:
        seed: Random seed for reproducibility (default: None)
    
    Returns:
        List of floats representing a CLIP embedding vector
    """
    return create_test_embedding(dim=TEST_CLIP_DIM, seed=seed)

