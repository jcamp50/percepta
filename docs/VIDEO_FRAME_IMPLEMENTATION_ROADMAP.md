# Video Frame Implementation Roadmap

## Current State - Phase 6 Complete ✅

**Status**: ✅ **FULLY IMPLEMENTED** - Video Understanding + MVP Memory Integration Complete

**Current Implementation**:
- ✅ **JCB-31**: Video frames stored with CLIP embeddings (512-dim → projected to 1536-dim)
- ✅ **JCB-33**: Temporal alignment with transcripts, chat messages, and metadata snapshots
- ✅ **JCB-41**: Visual description generation pipeline with hybrid lazy/cached strategy
- ✅ **JCB-42**: Adaptive frame capture (5-10s intervals) with interesting frame detection
- ✅ **JCB-37**: Grounded embeddings (70% CLIP + 30% description) for improved text query performance
- ✅ **JCB-35**: Memory-propagated summarization (2-minute segments) for long-term context
- ✅ **JCB-32**: Chat message embeddings with temporal alignment

**Rich Context Representation**:
- Video frames include rich JSON descriptions (activities, UI elements, scene changes, etc.)
- Temporal alignment links frames to transcripts, chat messages, and metadata
- Grounded embeddings combine visual and text context for better semantic search
- Context enrichment provides aligned transcript/chat/metadata at retrieval time

---

## Future Enhancements (Post-MVP)

### JCB-33: Temporal Alignment ✅ **COMPLETE**
**Priority**: High  
**Status**: ✅ **DONE**

**What It Adds**:
- ✅ Link video frames to temporally-aligned transcripts (±5 seconds)
- ✅ Link video frames to temporally-aligned chat messages (±5 seconds)
- ✅ Link video frames to metadata snapshots
- ✅ Enrich video frame retrieval with aligned context

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

### JCB-37: Joint Embedding Fusion ✅ **COMPLETE**
**Priority**: Medium  
**Status**: ✅ **DONE**

**What It Adds**:
- ✅ Create "grounded embeddings" that combine CLIP visual + text context
- ✅ Store both pure CLIP embeddings and grounded embeddings
- ✅ Better semantic alignment for text queries
- ✅ Improved retrieval accuracy

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

| Issue | Phase | Enhancement | Impact on Video Frames | Status |
|-------|-------|-------------|----------------------|--------|
| **JCB-31** | MVP | Basic CLIP embeddings | Pure CLIP embeddings (512→1536) | ✅ **DONE** |
| **JCB-32** | MVP | Chat message embeddings | Chat messages with embeddings | ✅ **DONE** |
| **JCB-33** | Post-MVP 1.3 | Temporal alignment | Adds transcript/chat/metadata context | ✅ **DONE** |
| **JCB-41** | MVP 1.6 | Visual description generation | Rich JSON descriptions for frames | ✅ **DONE** |
| **JCB-42** | MVP 1.6 | Adaptive frame capture | 5-10s intervals, interesting detection | ✅ **DONE** |
| **JCB-37** | Post-MVP 2.3 | Joint embedding fusion | Grounded embeddings (70% CLIP + 30% text) | ✅ **DONE** |
| **JCB-35** | Post-MVP 2.1 | Memory-propagated summarization | Long-term context via 2-min summaries | ✅ **DONE** |
| **JCB-34** | Post-MVP 1.4 | Multi-source retrieval | Better integration with other sources | Backlog |

---

## Implementation Status: Phase 6 Complete ✅

✅ **COMPLETE**: Video understanding and MVP memory integration fully implemented.

**Completed Features**:
1. ✅ **JCB-33**: Temporal alignment implemented - video frames linked to transcripts, chat, and metadata
2. ✅ **JCB-37**: Joint embedding fusion implemented - grounded embeddings (70% CLIP + 30% description)
3. ✅ **JCB-41**: Visual description generation - rich JSON descriptions with hybrid lazy/cached strategy
4. ✅ **JCB-42**: Adaptive frame capture - 5-10s intervals with interesting frame detection
5. ✅ **JCB-35**: Memory-propagated summarization - 2-minute segment summaries for long-term context
6. ✅ **JCB-32**: Chat message embeddings - chat messages stored with embeddings and temporal alignment

**Current State**:
- ✅ Video frames have rich text context (descriptions + transcript + chat + metadata)
- ✅ Video frames have grounded embeddings (visual + text combined)
- ✅ Video frames integrated into multi-source retrieval with context enrichment
- ✅ LLM receives meaningful context about video frames via enriched JSON descriptions

---

## Implementation References

**Current Implementation**:
- `py/memory/video_store.py` - Temporal alignment, description generation, grounded embeddings
- `py/utils/video_embeddings.py` - CLIP embeddings + `create_grounded_embedding()` fusion
- `py/utils/video_descriptions.py` - Visual description generation with GPT-4o-mini Vision API
- `py/reason/retriever.py` - Context-enriched video frame retrieval (`retrieve_video_with_context()`)
- `py/memory/summarizer.py` - Memory-propagated summarization (JCB-35)
- `py/memory/chat_store.py` - Chat message embeddings and storage (JCB-32)

---

## Conclusion

✅ **Phase 6 Complete**: Video understanding and MVP memory integration are fully implemented.

**Completed Enhancements**:
1. ✅ **Temporal alignment** (JCB-33) - Provides transcript/chat/metadata context
2. ✅ **Visual descriptions** (JCB-41) - Rich JSON descriptions for all frames
3. ✅ **Joint embedding fusion** (JCB-37) - Grounded embeddings for better text query performance
4. ✅ **Adaptive capture** (JCB-42) - Cost-efficient 5-10s intervals with interesting frame detection
5. ✅ **Memory propagation** (JCB-35) - Long-term context via 2-minute segment summaries
6. ✅ **Chat embeddings** (JCB-32) - Chat messages with embeddings and temporal alignment

Video frames are now rich, context-aware entries that provide comprehensive information to the LLM for RAG queries, combining visual understanding, temporal context, and long-term memory.

