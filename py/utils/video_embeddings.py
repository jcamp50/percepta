"""
Video Frame Embedding Utility

Generates CLIP embeddings for video frames and projects them to 1536 dimensions
to match OpenAI text embedding dimensions for unified vector search.
"""

from __future__ import annotations

import asyncio
import os
from typing import List, Optional

import numpy as np
from PIL import Image

from py.utils.logging import get_logger
from py.utils.embeddings import embed_text

logger = get_logger(__name__, category="video")

# CLIP model will be loaded lazily
_clip_model = None
_clip_processor = None
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
CLIP_EMBEDDING_DIM = 512
TARGET_EMBEDDING_DIM = 1536


def _load_clip_model():
    """Load CLIP model and processor (lazy loading, cached)."""
    global _clip_model, _clip_processor
    
    if _clip_model is not None and _clip_processor is not None:
        return _clip_model, _clip_processor
    
    try:
        from transformers import CLIPModel, CLIPProcessor
        
        logger.info(f"Loading CLIP model: {CLIP_MODEL_NAME}")
        _clip_model = CLIPModel.from_pretrained(CLIP_MODEL_NAME)
        _clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
        
        # Set model to evaluation mode
        _clip_model.eval()
        
        logger.info(f"CLIP model loaded successfully: {CLIP_MODEL_NAME}")
        return _clip_model, _clip_processor
    except ImportError as e:
        logger.error(
            f"CLIP dependencies not installed. Install with: pip install transformers torch torchvision"
        )
        raise ImportError(
            "CLIP model requires transformers, torch, and torchvision. "
            "Install with: pip install transformers torch torchvision"
        ) from e
    except Exception as e:
        logger.error(f"Failed to load CLIP model: {e}")
        raise


def project_clip_to_1536(clip_512: List[float]) -> List[float]:
    """Project 512-dim CLIP embedding to 1536-dim via repetition + normalization.
    
    This preserves cosine similarity while matching OpenAI embedding dimension.
    
    Args:
        clip_512: CLIP embedding vector (512 dimensions)
    
    Returns:
        Projected embedding vector (1536 dimensions)
    """
    clip_array = np.array(clip_512, dtype=np.float32)
    
    # Validate input dimension
    if len(clip_array) != CLIP_EMBEDDING_DIM:
        raise ValueError(
            f"Expected CLIP embedding dimension {CLIP_EMBEDDING_DIM}, got {len(clip_array)}"
        )
    
    # Repeat pattern to reach 1536 dimensions: 512 * 3 = 1536
    projected = np.tile(clip_array, 3)
    
    # Normalize to maintain unit vector (preserves cosine similarity)
    norm = np.linalg.norm(projected)
    if norm > 0:
        projected = projected / norm
    else:
        # Handle edge case where all values are zero: return a uniform unit vector
        projected = np.ones(TARGET_EMBEDDING_DIM, dtype=np.float32) / np.sqrt(TARGET_EMBEDDING_DIM)
    
    # Validate output dimension
    if len(projected) != TARGET_EMBEDDING_DIM:
        raise ValueError(
            f"Projection failed: expected {TARGET_EMBEDDING_DIM} dimensions, got {len(projected)}"
        )
    
    return projected.tolist()


async def generate_clip_embedding(image_path: str) -> List[float]:
    """Generate CLIP embedding for an image and project to 1536 dimensions.
    
    Args:
        image_path: Path to the image file (JPEG, PNG, etc.)
    
    Returns:
        Embedding vector (1536 dimensions) compatible with OpenAI text embeddings
    
    Raises:
        FileNotFoundError: If image file doesn't exist
        ValueError: If image cannot be processed
        ImportError: If CLIP dependencies are not installed
    """
    # Validate image file exists
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    # Load CLIP model (lazy loading, cached)
    model, processor = _load_clip_model()
    
    try:
        # Load and preprocess image in thread (CLIP is CPU-bound)
        def _process_image():
            # Load image
            image = Image.open(image_path).convert("RGB")
            
            # Preprocess image for CLIP
            inputs = processor(images=image, return_tensors="pt")
            
            # Generate embedding (512-dim)
            import torch
            with torch.no_grad():
                image_features = model.get_image_features(**inputs)
                # Normalize to unit vector
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                # Convert to numpy and extract single vector
                embedding_512 = image_features[0].cpu().numpy().tolist()
            
            return embedding_512
        
        # Run CLIP inference in thread pool to avoid blocking
        embedding_512 = await asyncio.to_thread(_process_image)
        
        # Validate CLIP embedding dimension
        if len(embedding_512) != CLIP_EMBEDDING_DIM:
            raise ValueError(
                f"CLIP model returned unexpected dimension: {len(embedding_512)} != {CLIP_EMBEDDING_DIM}"
            )
        
        # Project to 1536 dimensions
        embedding_1536 = project_clip_to_1536(embedding_512)
        
        logger.debug(
            f"Generated CLIP embedding for {image_path}: "
            f"{CLIP_EMBEDDING_DIM} -> {TARGET_EMBEDDING_DIM} dimensions"
        )
        
        return embedding_1536
        
    except Exception as e:
        logger.error(f"Failed to generate CLIP embedding for {image_path}: {e}")
        raise


async def create_grounded_embedding(
    clip_embedding: List[float], description_text: str
) -> List[float]:
    """Create a grounded embedding by fusing CLIP visual embedding with description text embedding.
    
    Uses weighted fusion: 70% CLIP embedding + 30% description text embedding.
    Both embeddings are normalized before fusion to maintain cosine similarity properties.
    
    Args:
        clip_embedding: CLIP embedding vector (1536 dimensions, already projected)
        description_text: Text description of the video frame (required)
    
    Returns:
        Grounded embedding vector (1536 dimensions) combining visual and text context
    
    Raises:
        ValueError: If embeddings have incorrect dimensions
    """
    if not description_text or not description_text.strip():
        raise ValueError("description_text is required for grounded embedding")
    
    # Validate CLIP embedding dimension
    clip_array = np.array(clip_embedding, dtype=np.float32)
    if len(clip_array) != TARGET_EMBEDDING_DIM:
        raise ValueError(
            f"Expected CLIP embedding dimension {TARGET_EMBEDDING_DIM}, got {len(clip_array)}"
        )
    
    # Generate text embedding from description
    text_embedding = await embed_text(description_text)
    text_array = np.array(text_embedding, dtype=np.float32)
    
    # Validate text embedding dimension
    if len(text_array) != TARGET_EMBEDDING_DIM:
        raise ValueError(
            f"Expected text embedding dimension {TARGET_EMBEDDING_DIM}, got {len(text_array)}"
        )
    
    # Normalize both embeddings to unit vectors (preserves cosine similarity)
    clip_norm = np.linalg.norm(clip_array)
    if clip_norm > 0:
        clip_normalized = clip_array / clip_norm
    else:
        clip_normalized = clip_array / np.sqrt(TARGET_EMBEDDING_DIM)
    
    text_norm = np.linalg.norm(text_array)
    if text_norm > 0:
        text_normalized = text_array / text_norm
    else:
        text_normalized = text_array / np.sqrt(TARGET_EMBEDDING_DIM)
    
    # Weighted fusion: 70% CLIP + 30% text
    CLIP_WEIGHT = 0.7
    TEXT_WEIGHT = 0.3
    
    fused = CLIP_WEIGHT * clip_normalized + TEXT_WEIGHT * text_normalized
    
    # Normalize the fused result to maintain unit vector
    fused_norm = np.linalg.norm(fused)
    if fused_norm > 0:
        fused_normalized = fused / fused_norm
    else:
        fused_normalized = fused / np.sqrt(TARGET_EMBEDDING_DIM)
    
    logger.debug(
        f"Created grounded embedding: CLIP ({CLIP_WEIGHT*100:.0f}%) + Text ({TEXT_WEIGHT*100:.0f}%)"
    )
    
    return fused_normalized.tolist()

