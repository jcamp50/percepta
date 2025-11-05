# Context Layer Expansion Guide

**Quick Reference**: Implementation guide for expanding Percepta's memory/context layer based on research findings.

---

## Overview

This guide provides actionable steps to enhance Percepta's context layer from current audio-only implementation to a comprehensive multimodal system supporting live audio, video screenshots, chat, and metadata.

**Current State**: Audio transcripts with time-biased retrieval  
**Target State**: Multimodal context with hierarchical memory, compression, and efficient retrieval

---

## Quick Wins (MVP 1.5)

### 1. Video Frame Embeddings

**Priority**: High  
**Effort**: Medium  
**Impact**: High

#### Implementation Steps

1. **Capture Screenshots** (Already planned)
   ```python
   # In node/audio.js or new node/video.js
   # Capture screenshot every 2 seconds
   # Send to Python /api/video-frame endpoint
   ```

2. **Create Video Frames Table**
   ```sql
   -- Add to py/database/models.py
   class VideoFrame(Base):
       __tablename__ = "video_frames"
       id: Mapped[uuid.UUID] = mapped_column(primary_key=True)
       channel_id: Mapped[str]
       captured_at: Mapped[datetime]
       image_path: Mapped[str]  # or base64, or storage URL
       embedding: Mapped[List[float]] = mapped_column(Vector(1536))
       transcription_id: Mapped[Optional[uuid.UUID]]  # Link to transcript
   ```

3. **Generate Embeddings**
   ```python
   # Option 1: Local CLIP model (recommended for cost)
   from transformers import CLIPProcessor, CLIPModel
   
   # Option 2: OpenAI CLIP API (if available)
   # Use OpenAI vision API or wait for CLIP endpoint
   
   # py/utils/video_embeddings.py
   async def embed_video_frame(image_path: str) -> List[float]:
       # Use CLIP to generate embeddings
       pass
   ```

4. **Store in Vector DB**
   ```python
   # py/memory/video_store.py
   class VideoStore:
       async def insert_frame(
           self,
           channel_id: str,
           image_path: str,
           captured_at: datetime,
           embedding: List[float],
           transcription_id: Optional[uuid.UUID] = None,
       ) -> str:
           # Insert into video_frames table
           pass
   ```

#### Integration Points

- **Ingest**: `py/ingest/video.py` - Receive screenshots, generate embeddings
- **Memory**: `py/memory/video_store.py` - Store and retrieve frames
- **Reason**: `py/reason/retriever.py` - Add video source to retrieval

---

### 1.5. Grounding CLIP Embeddings with Context

**Priority**: High  
**Effort**: Medium  
**Impact**: High

#### Why Ground CLIP Embeddings?

Raw CLIP embeddings capture visual information but lack context from:
- **Audio transcripts**: What's being said when the frame was captured
- **Chat messages**: Viewer reactions and discussions
- **Metadata**: Game, title, stream state

Grounding enriches embeddings with this context, enabling:
- Better semantic search (text queries find relevant images)
- Richer context understanding
- Improved retrieval accuracy

#### Approach 1: Temporal Alignment (Simplest - MVP)

**Concept**: Link video frames to temporally-aligned context without modifying embeddings.

**Implementation**:

```python
# py/memory/video_store.py
class VideoStore:
    async def insert_frame(
        self,
        channel_id: str,
        image_path: str,
        captured_at: datetime,
        clip_embedding: List[float],
    ) -> str:
        # Find aligned context (±5 seconds)
        transcript = await self._find_aligned_transcript(
            channel_id, captured_at, window_seconds=5
        )
        chat = await self._find_aligned_chat(
            channel_id, captured_at, window_seconds=5
        )
        metadata = await self._get_metadata_at_time(captured_at)
        
        # Store with context references
        frame = VideoFrame(
            channel_id=channel_id,
            captured_at=captured_at,
            image_path=image_path,
            embedding=clip_embedding,  # Pure CLIP embedding
            transcript_id=transcript.id if transcript else None,
            aligned_chat_ids=[c.id for c in chat],  # JSONB array
            metadata_snapshot=metadata,  # JSONB field
        )
        
        return await session.commit()
```

**Database Schema**:

```python
# py/database/models.py
class VideoFrame(Base):
    __tablename__ = "video_frames"
    id: Mapped[uuid.UUID] = mapped_column(primary_key=True)
    channel_id: Mapped[str]
    captured_at: Mapped[datetime]
    image_path: Mapped[str]
    embedding: Mapped[List[float]] = mapped_column(Vector(1536))  # Pure CLIP
    transcript_id: Mapped[Optional[uuid.UUID]]  # Link to aligned transcript
    aligned_chat_ids: Mapped[Optional[List[uuid.UUID]]] = mapped_column(ARRAY(String))  # Chat IDs
    metadata_snapshot: Mapped[Optional[dict]] = mapped_column(JSONB)  # Channel metadata
```

**Retrieval with Context**:

```python
# py/reason/retriever.py
async def retrieve_video_with_context(
    self,
    query_embedding: List[float],
    params: RetrievalParams,
) -> List[SearchResult]:
    # Search video frames
    frames = await self.video_store.search_frames(
        query_embedding=query_embedding,
        limit=params.limit,
        channel_id=params.channel_id,
    )
    
    # Enrich with aligned context
    for frame in frames:
        if frame.transcript_id:
            frame.transcript_text = await self.transcript_store.get(frame.transcript_id)
        if frame.aligned_chat_ids:
            frame.chat_messages = await self.chat_store.get_many(frame.aligned_chat_ids)
        if frame.metadata_snapshot:
            frame.metadata = frame.metadata_snapshot
    
    return frames
```

**Benefits**:
- ✅ Simple to implement
- ✅ Preserves original CLIP embedding
- ✅ Context available at retrieval time
- ✅ No embedding modification needed

---

#### Approach 2: Joint Embedding Fusion (Advanced - Post-MVP)

**Concept**: Combine CLIP embedding with text embeddings from aligned context into unified representation.

**Implementation**:

```python
# py/utils/video_embeddings.py
import numpy as np

async def create_grounded_embedding(
    clip_embedding: List[float],
    transcript: Optional[Transcript],
    chat: List[ChatMessage],
    metadata: Optional[ChannelSnapshot],
) -> List[float]:
    """
    Create a grounded embedding by fusing CLIP visual embedding with text context.
    
    Args:
        clip_embedding: Pure CLIP embedding from image
        transcript: Aligned transcript (if available)
        chat: Aligned chat messages (if available)
        metadata: Channel metadata snapshot (if available)
    
    Returns:
        Grounded embedding combining visual and context
    """
    if not (transcript or chat or metadata):
        return clip_embedding  # No context, return pure CLIP
    
    # Build context text
    context_parts = []
    if transcript:
        context_parts.append(transcript.text[:200])  # Limit length
    if chat:
        context_parts.extend([c.message for c in chat[:3]])  # Top 3 messages
    if metadata:
        context_parts.append(
            f"Game: {metadata.game_name} | Title: {metadata.title}"
        )
    
    if not context_parts:
        return clip_embedding
    
    context_text = " | ".join(context_parts)
    text_embedding = await embed_text(context_text)
    
    # Fuse embeddings (weighted average)
    # Visual 70%, Context 30% - adjust based on experimentation
    grounded = (
        0.7 * np.array(clip_embedding) +
        0.3 * np.array(text_embedding)
    )
    
    return grounded.tolist()
```

**Database Schema Update**:

```python
# py/database/models.py
class VideoFrame(Base):
    # ... existing fields ...
    clip_embedding: Mapped[List[float]] = mapped_column(Vector(1536))  # Pure CLIP
    grounded_embedding: Mapped[Optional[List[float]]] = mapped_column(Vector(1536))  # + Context
    
    # Index both embeddings
    __table_args__ = (
        Index("idx_video_clip_embedding", "clip_embedding", postgresql_using="ivfflat"),
        Index("idx_video_grounded_embedding", "grounded_embedding", postgresql_using="ivfflat"),
    )
```

**Storage**:

```python
# py/memory/video_store.py
async def insert_grounded_frame(
    self,
    channel_id: str,
    image_path: str,
    captured_at: datetime,
    use_grounding: bool = True,
) -> str:
    # 1. Generate CLIP embedding
    clip_embedding = await generate_clip_embedding(image_path)
    
    # 2. Find aligned context
    transcript = await self._find_aligned_transcript(channel_id, captured_at, window=5)
    chat = await self._find_aligned_chat(channel_id, captured_at, window=5)
    metadata = await self._get_metadata_at_time(captured_at)
    
    # 3. Create grounded embedding if enabled
    grounded_embedding = None
    if use_grounding and (transcript or chat or metadata):
        grounded_embedding = await create_grounded_embedding(
            clip_embedding, transcript, chat, metadata
        )
    
    # 4. Store both embeddings
    frame = VideoFrame(
        channel_id=channel_id,
        captured_at=captured_at,
        image_path=image_path,
        clip_embedding=clip_embedding,
        grounded_embedding=grounded_embedding,
        transcript_id=transcript.id if transcript else None,
        aligned_chat_ids=[c.id for c in chat],
        metadata_snapshot=metadata,
    )
    
    return await session.commit()
```

**Search Strategy**:

```python
# py/reason/retriever.py
async def search_video_frames(
    self,
    query_embedding: List[float],
    channel_id: str,
    use_grounded: bool = True,
    limit: int = 5,
) -> List[VideoFrame]:
    """
    Search video frames with option to use grounded or pure CLIP embeddings.
    
    - use_grounded=True: Better for text queries (e.g., "boss fight")
    - use_grounded=False: Better for visual queries (e.g., "red UI element")
    """
    if use_grounded:
        # Search grounded embeddings (better semantic alignment with text)
        frames = await self.video_store.search_frames(
            query_embedding=query_embedding,
            use_grounded=True,
            channel_id=channel_id,
            limit=limit,
        )
    else:
        # Search pure CLIP (better for visual similarity)
        frames = await self.video_store.search_frames(
            query_embedding=query_embedding,
            use_grounded=False,
            channel_id=channel_id,
            limit=limit,
        )
    
    return frames
```

**Benefits**:
- ✅ Single embedding represents both visual and context
- ✅ Better semantic alignment for text queries
- ✅ Can search by text and find relevant images
- ✅ Maintains visual specificity with pure CLIP option

**Challenges**:
- ⚠️ Need to decide fusion weights (70/30 is starting point)
- ⚠️ May lose some visual specificity
- ⚠️ Requires careful normalization

---

#### Approach 3: Cross-Modal Attention (Advanced - Future)

**Concept**: Use attention mechanism to dynamically weight relevant context when encoding frames.

**Implementation** (Future Enhancement):

```python
# py/memory/attention_encoder.py
from transformers import MultiHeadAttention

class ContextualVideoEncoder:
    """
    Advanced encoder using attention to weight context relevance.
    """
    def __init__(self):
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.attention = MultiHeadAttention(embed_dim=512, num_heads=8)
    
    async def encode_with_attention(
        self,
        image: Image,
        context_window: List[Context],  # Transcripts, chat, metadata
    ) -> List[float]:
        """
        Encode image with attention-weighted context.
        """
        # 1. Base CLIP embedding
        visual_emb = self.clip_model.encode_image(image)
        
        # 2. Encode context
        context_embs = [
            self.clip_model.encode_text(c.text) 
            for c in context_window
        ]
        
        # 3. Attention: which context is most relevant to this frame?
        attention_weights = self.attention(
            query=visual_emb.unsqueeze(0),
            keys=torch.stack(context_embs),
            values=torch.stack(context_embs),
        )
        
        # 4. Weighted context embedding
        attended_context = torch.sum(
            attention_weights * torch.stack(context_embs),
            dim=1
        )
        
        # 5. Combine
        grounded_emb = 0.7 * visual_emb + 0.3 * attended_context
        
        return grounded_emb.tolist()
```

**Benefits**:
- ✅ Dynamically focuses on relevant context
- ✅ Can handle variable context windows
- ✅ More sophisticated than simple averaging

**Challenges**:
- ❌ Requires training attention mechanism
- ❌ More complex implementation
- ❌ Higher computational cost

---

#### Approach 4: Continual Refinement (Advanced - Future)

**Concept**: Update embeddings as more context arrives over time.

**Implementation** (Future Enhancement):

```python
# py/memory/continual_grounding.py
class ContinualGrounding:
    """
    Refine embeddings as more context accumulates.
    """
    def __init__(self):
        self.frame_cache = {}  # Store frames with initial embeddings
    
    async def refine_embedding(
        self,
        frame_id: str,
        new_context: Context,
    ):
        """
        Refine embedding when new context arrives.
        """
        # Get original embedding
        original_emb = self.frame_cache[frame_id]['embedding']
        
        # Get accumulated context since frame was captured
        all_context = (
            self.frame_cache[frame_id]['context'] + [new_context]
        )
        
        # Re-compute grounded embedding with more context
        refined_emb = await self.fuse_embeddings(
            visual=original_emb,
            context=all_context,
        )
        
        # Update in database
        await video_store.update_embedding(frame_id, refined_emb)
```

**Benefits**:
- ✅ Embeddings improve as context accumulates
- ✅ Can handle delayed context (chat after frame)

**Challenges**:
- ❌ Requires re-computation
- ❌ Need to decide when to refine
- ❌ Database updates more frequent

---

#### Recommended Implementation Strategy

**Phase 1: Temporal Alignment (MVP 1.5)**
- Start with approach #1 (temporal linking)
- Simple to implement
- Preserves original CLIP embeddings
- Provides context at retrieval time

**Phase 2: Joint Embedding (Post-MVP)**
- Add approach #2 (joint embedding fusion)
- Better semantic alignment
- Single embedding for both visual and context
- More sophisticated retrieval

**Phase 3: Advanced Techniques (Future)**
- Consider attention mechanisms
- Implement continual refinement
- Experiment with different fusion weights

---

#### Implementation Checklist

- [ ] **Temporal Alignment**
  - [ ] Add `transcript_id` field to `VideoFrame` model
  - [ ] Add `aligned_chat_ids` array field
  - [ ] Add `metadata_snapshot` JSONB field
  - [ ] Implement `_find_aligned_transcript()` method
  - [ ] Implement `_find_aligned_chat()` method
  - [ ] Update retrieval to include context

- [ ] **Joint Embedding Fusion**
  - [ ] Create `create_grounded_embedding()` function
  - [ ] Add `grounded_embedding` field to model
  - [ ] Implement fusion logic (weighted average)
  - [ ] Add index for grounded embeddings
  - [ ] Update search to support both embeddings
  - [ ] Experiment with fusion weights

- [ ] **Testing**
  - [ ] Test temporal alignment accuracy
  - [ ] Test grounded embedding quality
  - [ ] Compare retrieval performance (grounded vs pure CLIP)
  - [ ] Measure latency impact

---

#### Benefits of Grounding

1. **Better Semantic Search**: Text queries like "boss fight" can match frames even if visual alone is ambiguous
2. **Richer Context**: Frames carry aligned audio/chat context automatically
3. **Improved Retrieval**: Grounded embeddings bridge visual and text queries
4. **Continual Improvement**: Embeddings can be refined as more context arrives

---

### 2. Chat Message Embeddings

**Priority**: High  
**Effort**: Low  
**Impact**: High

#### Implementation Steps

1. **Create Chat Messages Table**
   ```sql
   -- Add to py/database/models.py
   class ChatMessage(Base):
       __tablename__ = "chat_messages"
       id: Mapped[uuid.UUID] = mapped_column(primary_key=True)
       channel_id: Mapped[str]
       username: Mapped[str]
       message: Mapped[str]
       sent_at: Mapped[datetime]
       embedding: Mapped[List[float]] = mapped_column(Vector(1536))
   ```

2. **Store Chat Messages**
   ```python
   # Modify py/main.py receive_message endpoint
   async def receive_message(message: ChatMessage):
       # ... existing code ...
       
       # Generate embedding
       embedding = await embed_text(message.message)
       
       # Store in vector DB
       await chat_store.insert_message(
           channel_id=message.channel,
           username=message.username,
           message=message.message,
           sent_at=datetime.now(),
           embedding=embedding,
       )
   ```

3. **Add Chat Store**
   ```python
   # py/memory/chat_store.py
   class ChatStore:
       async def insert_message(...):
           pass
       
       async def search_messages(
           self,
           query_embedding: List[float],
           channel_id: str,
           limit: int = 5,
           half_life_minutes: int = 60,
       ) -> List[Dict]:
           # Similar to VectorStore.search_transcripts
           pass
   ```

#### Integration Points

- **Ingest**: `py/main.py` - Store messages when received
- **Memory**: `py/memory/chat_store.py` - Chat storage and retrieval
- **Reason**: `py/reason/retriever.py` - Add chat source to retrieval

---

### 3. Enhanced Multi-Source Retrieval

**Priority**: Medium  
**Effort**: Medium  
**Impact**: High

#### Implementation Steps

1. **Extend Retriever Interface**
   ```python
   # py/reason/retriever.py
   class Retriever:
       def __init__(
           self,
           vector_store: VectorStore,
           video_store: VideoStore,
           chat_store: ChatStore,
       ):
           self.vector_store = vector_store
           self.video_store = video_store
           self.chat_store = chat_store
       
       async def retrieve(
           self,
           query_embedding: List[float],
           params: RetrievalParams,
       ) -> List[SearchResult]:
           # Retrieve from all sources
           transcripts = await self.from_transcripts(...)
           videos = await self.from_video_frames(...)
           chats = await self.from_chat_messages(...)
           
           # Merge and rank
           return self._merge_and_rank(transcripts, videos, chats)
   ```

2. **Temporal Alignment**
   ```python
   # Align results by timestamp for coherent context
   def _align_by_time(
       self,
       results: List[SearchResult],
       time_window: int = 5,  # seconds
   ) -> List[SearchResult]:
       # Group results within time_window
       # Sort by relevance within each group
       pass
   ```

3. **Multi-Modal Fusion**
   ```python
   # Combine different modalities for better context
   def _fuse_modalities(
       self,
       transcripts: List[SearchResult],
       videos: List[SearchResult],
       chats: List[SearchResult],
   ) -> List[SearchResult]:
       # Weight different sources
       # Merge temporally aligned results
       pass
   ```

---

## Medium-Term Enhancements (Post-MVP)

### 4. Memory-Propagated Summarization

**Priority**: High  
**Effort**: High  
**Impact**: High

#### Implementation Steps

1. **Segment-Based Processing**
   ```python
   # py/memory/summarizer.py
   class Summarizer:
       async def summarize_segment(
           self,
           channel_id: str,
           start_time: datetime,
           end_time: datetime,
           previous_summaries: List[str],
       ) -> str:
           # Retrieve all data in segment
           transcripts = await vector_store.get_range(...)
           videos = await video_store.get_range(...)
           chats = await chat_store.get_range(...)
           
           # Build context with previous summaries
           context = self._build_context(
               transcripts, videos, chats, previous_summaries
           )
           
           # Generate summary via LLM
           summary = await self._generate_summary(context)
           return summary
   ```

2. **Memory Propagation**
   ```python
   # Propagate summaries forward
   segment_summaries = []
   for segment in segments:
       summary = await summarizer.summarize_segment(
           segment,
           previous_summaries=segment_summaries,
       )
       segment_summaries.append(summary)
       
       # Store summary with embedding
       await vector_store.insert_summary(summary, embedding)
   ```

3. **Adaptive Memory Selection**
   ```python
   # Select most relevant summaries for query
   def select_memories(
       self,
       query: str,
       all_summaries: List[Summary],
       max_tokens: int = 2000,
   ) -> List[Summary]:
       # Semantic search
       # Select top N that fit in token budget
       # Prioritize recent + relevant
       pass
   ```

---

### 5. Streaming KV Cache

**Priority**: Medium  
**Effort**: Medium  
**Impact**: Medium

#### Implementation Steps

1. **KV Cache Structure**
   ```python
   # py/memory/kv_cache.py
   from collections import OrderedDict
   
   class StreamingKVCache:
       def __init__(self, max_size: int = 100):
           self.cache: OrderedDict = OrderedDict()
           self.max_size = max_size
       
       def add(self, key: str, value: Dict):
           # Add to cache
           # If full, compress oldest entry
           if len(self.cache) >= self.max_size:
               oldest_key = next(iter(self.cache))
               compressed = self._compress(oldest_key)
               self.cache[oldest_key] = compressed
           self.cache[key] = value
       
       def get(self, key: str) -> Optional[Dict]:
           return self.cache.get(key)
       
       def _compress(self, key: str) -> Dict:
           # Compress to summary
           pass
   ```

2. **Hybrid Retrieval**
   ```python
   # py/reason/hybrid_retriever.py
   class HybridRetriever:
       def __init__(self, kv_cache, vector_store):
           self.kv_cache = kv_cache
           self.vector_store = vector_store
       
       async def retrieve(self, query: str, time_range: int = 2):
           # Recent data: KV cache (fast)
           recent = self.kv_cache.get_recent(time_range)
           
           # Older data: Vector store (semantic search)
           older = await self.vector_store.search(...)
           
           return self._merge(recent, older)
   ```

---

### 6. Continuous-Time Memory Consolidation

**Priority**: Medium  
**Effort**: High  
**Impact**: Medium

#### Implementation Steps

1. **Event Detection**
   ```python
   # py/memory/event_detector.py
   class EventDetector:
       def detect_important_events(
           self,
           channel_id: str,
           time_range: datetime,
       ) -> List[Event]:
           # Detect raids (EventSub)
           # Detect boss fights (high chat activity, keywords)
           # Detect high engagement (many @mentions)
           # Return list of important events
           pass
   ```

2. **Sticky Memory Formation**
   ```python
   # py/memory/sticky_memory.py
   class StickyMemory:
       def form_sticky_memory(
           self,
           event: Event,
           context: Dict,
       ) -> StickyMemory:
           # Allocate higher granularity
           # Store detailed context
           # Mark as persistent
           pass
       
       def update_sticky_memory(
           self,
           memory_id: str,
           new_context: Dict,
       ):
           # Evolve memory with new information
           pass
   ```

3. **Attention-Based Retrieval**
   ```python
   # Prioritize sticky memories in retrieval
   def retrieve_with_attention(
       self,
       query: str,
       sticky_memories: List[StickyMemory],
       regular_memories: List[Memory],
   ) -> List[Memory]:
       # Higher weight for sticky memories
       # Blend with regular memories
       pass
   ```

---

## Context Engineering Framework

### Four Pillars Implementation

#### 1. Write Context

**Current**: ✅ Audio transcripts → embeddings  
**Enhancement**: Add video frames, chat messages, metadata

```python
# py/memory/context_writer.py
class ContextWriter:
    async def write_audio_context(self, transcript: str) -> Embedding:
        pass
    
    async def write_video_context(self, frame: Image) -> Embedding:
        pass
    
    async def write_chat_context(self, message: str) -> Embedding:
        pass
    
    async def write_metadata_context(self, metadata: Dict) -> Embedding:
        pass
```

#### 2. Select Context

**Current**: ✅ Semantic search  
**Enhancement**: Temporal filtering, relevance scoring, multi-modal fusion

```python
# py/reason/context_selector.py
class ContextSelector:
    async def select_context(
        self,
        query: str,
        sources: List[Source],
        max_tokens: int,
    ) -> List[Context]:
        # Semantic search
        candidates = await self._semantic_search(query, sources)
        
        # Temporal filtering
        candidates = self._filter_by_time(candidates)
        
        # Relevance scoring
        candidates = self._score_relevance(candidates, query)
        
        # Multi-modal fusion
        candidates = self._fuse_modalities(candidates)
        
        # Select top N within token budget
        return self._select_within_budget(candidates, max_tokens)
    ```

#### 3. Compress Context

**Current**: ❌ Missing  
**Enhancement**: Summarization, abstraction, token budget management

```python
# py/reason/context_compressor.py
class ContextCompressor:
    async def compress(
        self,
        contexts: List[Context],
        max_tokens: int,
    ) -> List[CompressedContext]:
        # Summarize long contexts
        # Abstract redundant information
        # Maintain critical details
        # Fit within token budget
        pass
```

#### 4. Isolate Context

**Current**: ⚠️ Partial (separate tables)  
**Enhancement**: Clear boundaries, structured packaging

```python
# py/reason/context_isolator.py
class ContextIsolator:
    def isolate_by_modality(
        self,
        contexts: List[Context],
    ) -> Dict[str, List[Context]]:
        return {
            "audio": [c for c in contexts if c.modality == "audio"],
            "video": [c for c in contexts if c.modality == "video"],
            "chat": [c for c in contexts if c.modality == "chat"],
            "metadata": [c for c in contexts if c.modality == "metadata"],
        }
    
    def package_context(
        self,
        isolated: Dict[str, List[Context]],
    ) -> StructuredContext:
        # Clear structure for LLM
        # Prevent confusion between sources
        pass
```

---

## Implementation Checklist

### MVP 1.5 (Immediate)

- [ ] **Video Frame Storage**
  - [ ] Create `video_frames` table schema
  - [ ] Implement screenshot capture (Node.js)
  - [ ] Create video embeddings utility (CLIP)
  - [ ] Implement video store (storage + retrieval)
  - [ ] Integrate video into retrieval pipeline

- [ ] **Chat Message Storage**
  - [ ] Create `chat_messages` table schema
  - [ ] Store messages with embeddings
  - [ ] Implement chat store (storage + retrieval)
  - [ ] Integrate chat into retrieval pipeline

- [ ] **Multi-Source Retrieval**
  - [ ] Extend retriever to handle multiple sources
  - [ ] Implement temporal alignment
  - [ ] Implement multi-modal fusion
  - [ ] Update RAG service to use multi-source retrieval

### Post-MVP (Medium-Term)

- [ ] **Memory-Propagated Summarization**
  - [ ] Implement segment-based processing
  - [ ] Implement memory propagation
  - [ ] Implement adaptive memory selection
  - [ ] Integrate into retrieval pipeline

- [ ] **Streaming KV Cache**
  - [ ] Implement KV cache structure
  - [ ] Implement compression logic
  - [ ] Implement hybrid retrieval
  - [ ] Integrate into retrieval pipeline

- [ ] **Continuous-Time Memory Consolidation**
  - [ ] Implement event detection
  - [ ] Implement sticky memory formation
  - [ ] Implement attention-based retrieval
  - [ ] Integrate into retrieval pipeline

- [ ] **Context Engineering Framework**
  - [ ] Implement context writer enhancements
  - [ ] Implement context selector improvements
  - [ ] Implement context compressor
  - [ ] Implement context isolator improvements

---

## Testing Strategy

### Unit Tests

- Test each new component independently
- Mock external dependencies (CLIP, OpenAI API)
- Verify data structures and algorithms

### Integration Tests

- Test multi-source retrieval end-to-end
- Test memory propagation pipeline
- Test KV cache with vector store
- Test temporal alignment across modalities

### Performance Tests

- Measure latency impact of new features
- Measure memory usage growth
- Measure API cost increases
- Verify performance targets are met

### Quality Tests

- Evaluate context relevance (human evaluation)
- Evaluate temporal alignment accuracy
- Evaluate multi-modal fusion effectiveness
- Evaluate long-term recall accuracy

---

## Cost Considerations

### Video Embeddings

- **Local CLIP**: Free (GPU required)
- **OpenAI API**: ~$X per image (if available)
- **Recommendation**: Use local CLIP for MVP

### Chat Embeddings

- **OpenAI**: $0.02 per 1M tokens
- **Estimate**: ~100 messages/hour × 50 tokens = 5K tokens/hour
- **Cost**: ~$0.0001 per hour (negligible)

### Summarization

- **OpenAI**: $0.15 per 1M tokens (gpt-4o-mini)
- **Estimate**: ~30 summaries/hour × 500 tokens = 15K tokens/hour
- **Cost**: ~$0.002 per hour (very low)

### Total Cost Estimate

- **Current**: ~$0.01 per hour (audio embeddings + LLM)
- **With Video**: ~$0.01 per hour (local CLIP)
- **With Summarization**: ~$0.012 per hour
- **Total**: ~$0.12 per 10-hour stream

---

## Next Steps

1. **Review Research Document**: `docs/LONG_TERM_CONTEXT_RESEARCH.md`
2. **Prioritize Features**: Choose which enhancements to implement first
3. **Create Linear Issues**: Break down into tasks
4. **Start Implementation**: Begin with video frame embeddings (MVP 1.5)
5. **Iterate**: Test, measure, improve

---

**Document Version**: 1.0  
**Last Updated**: 2025-01-30  
**Related Documents**: `docs/LONG_TERM_CONTEXT_RESEARCH.md`, `docs/ARCHITECTURE.md`

