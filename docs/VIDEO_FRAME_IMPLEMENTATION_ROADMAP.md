# Video Frame Implementation Roadmap

## Current State (JCB-31 - MVP)

**Status**: ✅ Implemented (Basic/Temporary)

**Current Implementation**:
- Video frames stored with CLIP embeddings (512-dim → projected to 1536-dim)
- Basic text representation: `[Video Frame] {image_path}` (e.g., `[Video Frame] frames\233300375\abc123.jpg`)
- Simple retrieval via vector similarity
- Included in multi-source retrieval but with minimal context

**Limitations**:
- Text representation is just a file path (not useful for LLM)
- No temporal alignment with transcripts/chat
- No context enrichment
- Pure CLIP embeddings without grounding

**Why This Is Temporary**:
This is the MVP implementation to get video frames working. The file path representation is a placeholder until proper context enrichment is implemented.

---

## Future Enhancements (Post-MVP)

### JCB-33: Temporal Alignment (Post-MVP 1.3)
**Priority**: High  
**Status**: Backlog

**What It Adds**:
- Link video frames to temporally-aligned transcripts (±5 seconds)
- Link video frames to temporally-aligned chat messages (±5 seconds)
- Link video frames to metadata snapshots
- Enrich video frame retrieval with aligned context

**Database Changes**:
```python
class VideoFrame(Base):
    # ... existing fields ...
    transcript_id: Mapped[Optional[uuid.UUID]]  # Link to aligned transcript
    aligned_chat_ids: Mapped[Optional[List[uuid.UUID]]]  # Chat IDs
    metadata_snapshot: Mapped[Optional[dict]]  # Channel metadata
```

**Retrieval Enhancement**:
- When retrieving video frames, also fetch aligned transcript text
- Include aligned chat messages
- Include metadata snapshot
- Provide richer context to LLM

**Impact**: Video frames will have meaningful text context (transcript + chat) instead of just file paths.

---

### JCB-37: Joint Embedding Fusion (Post-MVP 2.3)
**Priority**: Medium  
**Status**: Backlog

**What It Adds**:
- Create "grounded embeddings" that combine CLIP visual + text context
- Store both pure CLIP embeddings and grounded embeddings
- Better semantic alignment for text queries
- Improved retrieval accuracy

**Implementation**:
```python
# Combine CLIP (512) + text context (1536) → grounded (1536)
grounded_embedding = weighted_average(
    clip_embedding,      # 70% weight
    text_context_embedding,  # 30% weight
)
```

**Database Changes**:
```python
class VideoFrame(Base):
    # ... existing fields ...
    embedding: Mapped[List[float]]  # Pure CLIP (current)
    grounded_embedding: Mapped[Optional[List[float]]]  # NEW: Grounded embedding
```

**Search Strategy**:
- Use grounded embeddings for text queries (better semantic alignment)
- Use pure CLIP for visual queries (better visual similarity)
- Or search both and merge results

**Impact**: Video frames will be more findable via text queries (e.g., "What game are they playing?" will find relevant frames even if the visual similarity isn't perfect).

---

### JCB-34: Multi-Source Retrieval Enhancement (Post-MVP 1.4)
**Priority**: High  
**Status**: Backlog

**What It Adds**:
- Enhanced merging and ranking of results from all sources
- Temporal alignment across modalities
- Multi-modal fusion with proper weighting
- Deduplication of temporally-aligned results

**Impact**: Better integration of video frames with transcripts, chat, and metadata in retrieval results.

---

### JCB-35-40: Advanced Enhancements

- **JCB-35**: Memory-Propagated Summarization (long-term context)
- **JCB-36**: Streaming KV Cache (faster retrieval)
- **JCB-38**: Sticky Memories (important events)
- **JCB-39**: Advanced Context Engineering Framework
- **JCB-40**: Metadata Embeddings Enhancement

---

## Roadmap Summary

| Issue | Phase | Enhancement | Impact on Video Frames |
|-------|-------|-------------|----------------------|
| **JCB-31** | MVP | Basic CLIP embeddings | ✅ Current implementation |
| **JCB-33** | Post-MVP 1.3 | Temporal alignment | Adds transcript/chat context to frames |
| **JCB-37** | Post-MVP 2.3 | Joint embedding fusion | Creates richer embeddings (visual + text) |
| **JCB-34** | Post-MVP 1.4 | Multi-source retrieval | Better integration with other sources |

---

## Confirmation: Current Implementation Is Temporary

✅ **Confirmed**: The current basic implementation (file path as text representation) is temporary.

**Evidence**:
1. **JCB-33** explicitly plans to add temporal alignment, which will provide transcript/chat text context for video frames
2. **JCB-37** plans to create grounded embeddings that combine visual + text context
3. **JCB-34** plans to enhance multi-source retrieval with better context fusion
4. The `CONTEXT_LAYER_EXPANSION.md` document outlines two approaches:
   - **Approach 1** (JCB-33): Temporal alignment - link frames to context
   - **Approach 2** (JCB-37): Joint embedding fusion - fuse visual + text embeddings

**Future State**:
- Video frames will have rich text context (transcript + chat + metadata)
- Video frames will have grounded embeddings (visual + text combined)
- Video frames will be better integrated into multi-source retrieval
- LLM will receive meaningful context about video frames, not just file paths

---

## Current Code References

**Current Text Representation** (Temporary):
- `py/reason/retriever.py` line 105: `text=f"[Video Frame] {r['image_path']}"`

**Future Enhancement Points**:
- `py/memory/video_store.py` - Will add temporal alignment logic
- `py/utils/video_embeddings.py` - Will add grounding/fusion functions
- `py/reason/retriever.py` - Will enhance retrieval with context enrichment

---

## Conclusion

The current elementary implementation (using file paths as text) is **intentionally temporary** and part of the MVP. The roadmap clearly shows plans for:

1. **Temporal alignment** (JCB-33) - Will provide transcript/chat context
2. **Joint embedding fusion** (JCB-37) - Will create richer embeddings
3. **Enhanced retrieval** (JCB-34) - Will better integrate video frames

These enhancements will transform video frames from simple file path references into rich, context-aware entries that provide meaningful information to the LLM for RAG queries.

