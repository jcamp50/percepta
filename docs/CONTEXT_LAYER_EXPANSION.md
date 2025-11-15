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

- [x] **Joint Embedding Fusion** ✅ **COMPLETE (JCB-37)**
  - [x] Create `create_grounded_embedding()` function
  - [x] Add `grounded_embedding` field to model
  - [x] Implement fusion logic (weighted average: 70% CLIP + 30% description)
  - [x] Add index for grounded embeddings
  - [x] Update search to support `use_grounded` flag (default: True)
  - [x] Search uses grounded embeddings by default for better text query performance

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

### 1.6. Visual Description Generation (MVP Solution)

**Priority**: High  
**Effort**: Medium  
**Impact**: High

#### The Visual Information Gap

**Problem Identified**: While CLIP embeddings enable semantic search for video frames, the LLM cannot "see" what's actually in the image. Current implementation only provides:
- File path (`frames/channel123/uuid.jpg`)
- Temporally-aligned transcript text
- Temporally-aligned chat messages
- Metadata snapshots

**Missing**: Explicit visual description of what's on screen (UI elements, gameplay state, visual details, on-screen text, etc.)

**Impact**: LLM cannot answer visual questions like "What's on screen right now?" or "What UI elements are visible?" without explicit visual descriptions.

#### MVP Solution: GPT-4o-mini Vision API with Hybrid Lazy/Cached Generation

**Selected Approach**: Generate high-detail visual descriptions using GPT-4o-mini Vision API with cost-optimized hybrid strategy.

**Key Decisions**:
1. **Model**: GPT-4o-mini Vision API (cheapest option with high quality)
2. **Capture Interval**: Adaptive 5-10 seconds (reduces cost by 60-80%)
3. **Generation Strategy**: Hybrid lazy + cached + immediate
4. **Temporal Continuity**: Include previous frame descriptions for visual continuity
5. **Summarization Requirement**: All lazy frames must have descriptions before 2-minute summarization jobs

#### Cost Analysis

**GPT-4o-mini Vision API Pricing**:
- Input: $0.150 per 1M tokens
- Output: $0.600 per 1M tokens
- Image tokenization: ~170 tokens per 720p frame
- Prompt tokens: ~100-200 tokens (with context)
- Description tokens: ~200-300 tokens (high detail)

**Cost Per Description**:
- Input: 270 tokens × $0.150/1M = $0.0000405
- Output: 250 tokens × $0.600/1M = $0.00015
- **Total**: ~$0.00019 per frame

**Cost Scenarios**:

| Interval | Frames/Hour | Cost/Hour | Cost/10hr Stream |
|----------|-------------|-----------|------------------|
| 2 seconds | 1,800 | $0.342 | $3.42 |
| 5 seconds | 720 | $0.137 | $1.37 |
| 10 seconds | 360 | $0.068 | $0.68 |
| Adaptive (5-10s) | ~540 | ~$0.103 | ~$1.03 |

**MVP Target**: Adaptive 5-10 second intervals = **~$1.00 per 10-hour stream**

#### Implementation Strategy

**Hybrid Generation Approach**:

```python
# py/memory/video_store.py
class VideoStore:
    async def insert_frame(
        self,
        channel_id: str,
        image_path: str,
        captured_at: datetime,
        embedding: Optional[List[float]] = None,
    ) -> str:
        """
        Insert frame with hybrid description generation strategy.
        
        Strategy:
        1. Check cache for similar frames (perceptual hash)
        2. Generate immediately for "interesting" frames (high activity)
        3. Mark others as "lazy" (generate on-demand or before summarization)
        """
        frame_hash = self._compute_perceptual_hash(image_path)
        
        # 1. Check cache for similar frame (within last 60 seconds)
        similar_frame = await self._find_similar_frame(
            channel_id, frame_hash, window_seconds=60
        )
        
        if similar_frame and similar_frame.description:
            # Reuse description (with minor timestamp update if needed)
            description = similar_frame.description
            description_source = "cache"
        else:
            # 2. Determine if frame is "interesting" (high activity)
            is_interesting = await self._is_interesting_frame(
                channel_id, captured_at
            )
            
            if is_interesting:
                # Generate immediately
                description = await self._generate_description(
                    image_path, channel_id, captured_at
                )
                description_source = "immediate"
            else:
                # Mark as lazy (will generate before summarization)
                description = None
                description_source = "lazy"
        
        # Store frame
        frame_id = await self._insert_frame_with_description(
            channel_id=channel_id,
            image_path=image_path,
            captured_at=captured_at,
            embedding=embedding,
            description=description,
            description_source=description_source,
            frame_hash=frame_hash,
        )
        
        # 3. Background job: Generate descriptions for lazy frames
        if description_source == "lazy":
            asyncio.create_task(
                self._background_description_generation(frame_id)
            )
        
        return frame_id
```

**Description Generation with Temporal Context**:

```python
# py/utils/video_descriptions.py
async def generate_frame_description(
    image_path: str,
    channel_id: str,
    captured_at: datetime,
    previous_frame_description: Optional[str] = None,
    recent_summary: Optional[str] = None,
    transcript: Optional[dict] = None,
    chat: Optional[List[dict]] = None,
    metadata: Optional[dict] = None,
) -> str:
    """
    Generate high-detail visual description using GPT-4o-mini Vision API.
    
    Includes temporal context for visual continuity and grounding.
    """
    from openai import OpenAI
    from base64 import b64encode
    
    client = OpenAI(api_key=settings.openai_api_key)
    
    # Load and encode image
    with open(image_path, "rb") as image_file:
        base64_image = b64encode(image_file.read()).decode("utf-8")
    
    # Build contextual prompt
    prompt = _build_description_prompt(
        previous_frame_description=previous_frame_description,
        recent_summary=recent_summary,
        transcript=transcript,
        chat=chat,
        metadata=metadata,
    )
    
    # Call Vision API
    response = await asyncio.to_thread(
        client.chat.completions.create,
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        },
                    },
                ],
            }
        ],
        max_tokens=500,  # Allow detailed descriptions
    )
    
    return response.choices[0].message.content


def _build_description_prompt(
    previous_frame_description: Optional[str] = None,
    recent_summary: Optional[str] = None,
    transcript: Optional[dict] = None,
    chat: Optional[List[dict]] = None,
    metadata: Optional[dict] = None,
) -> str:
    """
    Build structured prompt for high-detail visual descriptions.
    
    Enforces format similar to example:
    - High-level description
    - Detailed visual breakdown
    - On-screen interface elements
    - Overall impression
    """
    context_parts = []
    
    # Visual continuity
    if previous_frame_description:
        context_parts.append(
            f"Previous frame context: {previous_frame_description[:200]}"
        )
    
    # Recent stream summary
    if recent_summary:
        context_parts.append(f"Recent stream summary: {recent_summary[:300]}")
    
    # Temporal alignment
    if transcript:
        context_parts.append(
            f"Streamer is saying: \"{transcript['text'][:200]}\""
        )
    
    if chat:
        chat_lines = [
            f"{c['username']}: {c['message']}" for c in chat[:5]
        ]
        context_parts.append(f"Chat discussion: {' | '.join(chat_lines)}")
    
    if metadata:
        context_parts.append(
            f"Game: {metadata.get('game_name', 'Unknown')} | "
            f"Title: {metadata.get('title', 'Unknown')}"
        )
    
    # Main instruction with format enforcement
    prompt = f"""Describe this Twitch stream screenshot in extreme detail, following this exact format:

**High-Level Description:**
[1-2 sentences summarizing the overall scene]

**Detailed Visual Description:**

1. **Streamer's Webcam Feed (if visible):**
   [Describe streamer appearance, clothing, expression, background, lighting]

2. **Main Gameplay View:**
   - **Perspective:** [First-person/Third-person/Top-down/etc.]
   - **Player Character:** [Appearance, equipment, position, actions]
   - **Environment - Foreground:** [Immediate surroundings, objects, surfaces]
   - **Environment - Midground:** [Buildings, structures, terrain]
   - **Environment - Background:** [Distant elements, sky, horizon]
   - **Lighting:** [Time of day, lighting conditions]

3. **On-Screen Interface and Overlays:**
   - **[Element Name]:** [Exact text, values, position]
   [List ALL UI elements: health bars, ammo counts, minimaps, chat overlays, etc.]
   - **Text Elements:** [Extract ALL visible text with exact wording]
   - **Icons/Symbols:** [Describe all icons, indicators, status symbols]

**Overall Impression:**
[1-2 sentences about the scene's significance or atmosphere]

CRITICAL REQUIREMENTS:
- Extract ALL visible text exactly as shown (weapon names, ammo counts, usernames, etc.)
- Describe ALL UI elements in detail (health bars, compass, chat messages, etc.)
- Mention specific visual details (colors, positions, states)
- Include temporal context from the stream when relevant
- Be concise but comprehensive (aim for 300-500 words)

{f'CONTEXT: {chr(10).join(context_parts)}' if context_parts else ''}"""

    return prompt
```

**Adaptive Capture Interval**:

```python
# py/ingest/video.py
async def determine_capture_interval(
    channel_id: str,
    current_time: datetime,
) -> int:
    """
    Determine adaptive capture interval based on stream activity.
    
    Returns: Interval in seconds (5-10)
    """
    # Check recent activity indicators
    recent_chat_count = await chat_store.count_recent_messages(
        channel_id, window_seconds=30
    )
    recent_transcript_activity = await vector_store.check_recent_activity(
        channel_id, window_seconds=30
    )
    
    # High activity: capture more frequently
    if recent_chat_count > 50 or recent_transcript_activity:
        return 5  # seconds
    
    # Normal activity: standard interval
    return 10  # seconds
```

**Summarization Requirement: Pre-Generate Lazy Descriptions**:

```python
# py/memory/summarizer.py
async def summarize_segment(
    self,
    channel_id: str,
    start_time: datetime,
    end_time: datetime,
) -> str:
    """
    Generate summary for 2-minute segment.
    
    CRITICAL: All lazy frames in this segment must have descriptions
    before summarization runs.
    """
    # 1. Ensure all lazy frames have descriptions
    lazy_frames = await video_store.get_lazy_frames_in_range(
        channel_id, start_time, end_time
    )
    
    if lazy_frames:
        # Generate descriptions for all lazy frames
        await asyncio.gather(*[
            video_store.generate_description_for_frame(frame_id)
            for frame_id in lazy_frames
        ])
    
    # 2. Retrieve all data for segment
    transcripts = await vector_store.get_range(channel_id, start_time, end_time)
    video_frames = await video_store.get_range(channel_id, start_time, end_time)
    chat_messages = await chat_store.get_range(channel_id, start_time, end_time)
    
    # 3. Build context with video frame descriptions
    context_parts = []
    
    # Add transcripts
    for transcript in transcripts:
        context_parts.append(f"[{transcript['started_at']}] {transcript['text']}")
    
    # Add video frame descriptions (CRITICAL: descriptions now available)
    for frame in video_frames:
        if frame['description']:
            context_parts.append(
                f"[{frame['captured_at']}] [Video Frame] {frame['description']}"
            )
        else:
            # Fallback: use image path if description missing (shouldn't happen)
            context_parts.append(
                f"[{frame['captured_at']}] [Video Frame] {frame['image_path']}"
            )
    
    # Add chat messages
    for chat in chat_messages:
        context_parts.append(
            f"[{chat['sent_at']}] [Chat] {chat['username']}: {chat['message']}"
        )
    
    # 4. Generate summary with full context
    summary = await self._generate_summary(context_parts)
    
    return summary
```

#### Database Schema Update

```python
# py/database/models.py
class VideoFrame(Base):
    __tablename__ = "video_frames"
    
    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    channel_id: Mapped[str] = mapped_column(String(255), nullable=False)
    captured_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), nullable=False
    )
    image_path: Mapped[str] = mapped_column(Text, nullable=False)
    embedding: Mapped[List[float]] = mapped_column(Vector(1536), nullable=False)
    
    # Visual description (NEW)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    description_source: Mapped[Optional[str]] = mapped_column(
        String(50), nullable=True
    )  # "immediate", "lazy", "cache"
    frame_hash: Mapped[Optional[str]] = mapped_column(
        String(64), nullable=True
    )  # Perceptual hash for caching
    
    # Temporal alignment (existing)
    transcript_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        ForeignKey("transcripts.id"), nullable=True
    )
    aligned_chat_ids: Mapped[Optional[List[str]]] = mapped_column(
        ARRAY(String), nullable=True
    )
    metadata_snapshot: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    
    __table_args__ = (
        Index("idx_video_frames_channel_captured", "channel_id", "captured_at"),
        Index("idx_video_frames_hash", "frame_hash"),  # For cache lookups
        Index("idx_video_frames_lazy", "channel_id", "description_source"),  # For lazy generation
        Index(
            "idx_video_frames_embedding",
            "embedding",
            postgresql_using="ivfflat",
            postgresql_with={"lists": 100},
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )
```

#### Retrieval Integration

```python
# py/reason/retriever.py
async def retrieve_video_with_context(
    self, *, query_embedding: List[float], params: RetrievalParams
) -> List[SearchResult]:
    """
    Retrieve video frames with enriched context including visual descriptions.
    """
    rows = await self.video_store.search_frames(
        query_embedding=query_embedding,
        limit=params.limit,
        half_life_minutes=params.half_life_minutes,
        channel_id=params.channel_id,
        prefilter_limit=params.prefilter_limit,
    )
    
    enriched_results = []
    for r in rows:
        # Build enriched text representation
        # Use description if available, otherwise fallback to image_path
        if r.get("description"):
            text_parts = [f"[Video Frame] {r['description']}"]
        else:
            # Generate description on-demand if missing (lazy generation)
            description = await self.video_store.generate_description_for_frame(
                r["id"]
            )
            text_parts = [f"[Video Frame] {description}"]
        
        # Add aligned context (existing)
        if r.get("transcript_id") and self.vector_store:
            transcript = await self.vector_store.get_transcript_by_id(
                r["transcript_id"]
            )
            if transcript:
                text_parts.append(f"Transcript: {transcript['text']}")
        
        if r.get("aligned_chat_ids") and self.chat_store:
            chat_messages = await self.chat_store.get_messages_by_ids(
                r["aligned_chat_ids"]
            )
            if chat_messages:
                chat_text = " | ".join(
                    [f"{c['username']}: {c['message']}" for c in chat_messages]
                )
                text_parts.append(f"Chat: {chat_text}")
        
        if r.get("metadata_snapshot"):
            metadata = r["metadata_snapshot"]
            metadata_parts = []
            if metadata.get("game_name"):
                metadata_parts.append(f"Game: {metadata['game_name']}")
            if metadata.get("title"):
                metadata_parts.append(f"Title: {metadata['title']}")
            if metadata_parts:
                text_parts.append(f"Metadata: {' | '.join(metadata_parts)}")
        
        enriched_text = " | ".join(text_parts)
        
        enriched_results.append(
            SearchResult(
                id=r["id"],
                channel_id=r["channel_id"],
                text=enriched_text,
                started_at=r["captured_at"],
                ended_at=r["captured_at"],
                cosine_distance=float(r["cosine_distance"]),
                score=float(r["score"]),
            )
        )
    
    return enriched_results
```

#### Tradeoffs and Considerations

**Advantages**:
- ✅ **Solves Visual Gap**: LLM can now "see" what's on screen
- ✅ **Cost-Effective**: ~$1.00 per 10-hour stream with adaptive intervals
- ✅ **High Quality**: GPT-4o-mini provides detailed, structured descriptions
- ✅ **Temporal Continuity**: Previous frame descriptions enable visual flow understanding
- ✅ **Flexible**: Hybrid approach balances cost and latency

**Challenges**:
- ⚠️ **Latency**: Lazy generation adds ~1-3s latency on first retrieval
- ⚠️ **Summarization Dependency**: Must ensure lazy frames have descriptions before summarization
- ⚠️ **Cache Hit Rate**: Depends on stream content (gaming streams often have similar frames)
- ⚠️ **API Rate Limits**: Need to handle OpenAI API rate limits gracefully

**Mitigations**:
- **Background Jobs**: Pre-generate lazy descriptions in background before summarization
- **Retry Logic**: Implement retry with exponential backoff for API failures
- **Rate Limiting**: Queue description generation requests to respect API limits
- **Monitoring**: Track description generation success rate and latency

**Future Enhancements** (Post-MVP):
- **Local Models**: Consider LLaVA or Qwen-VL for cost reduction at scale
- **Description Embeddings**: Generate embeddings from descriptions for better text search
- **Joint Embedding Fusion**: ✅ **COMPLETE** - Combine CLIP + description embeddings (JCB-37)
  - Grounded embeddings (70% CLIP + 30% description) stored in `grounded_embedding` field
  - Search uses grounded embeddings by default for better text query performance
- **Adaptive Quality**: Use lower detail for similar frames, high detail for unique frames

#### Implementation Checklist

- [ ] **Database Schema**
  - [ ] Add `description` field to `VideoFrame` model
  - [ ] Add `description_source` field
  - [ ] Add `frame_hash` field for caching
  - [ ] Add indexes for cache lookups and lazy generation queries

- [ ] **Description Generation**
  - [ ] Create `py/utils/video_descriptions.py` with GPT-4o-mini integration
  - [ ] Implement `generate_frame_description()` with temporal context
  - [ ] Implement `_build_description_prompt()` with format enforcement
  - [ ] Add retry logic and error handling

- [ ] **Hybrid Strategy**
  - [ ] Implement perceptual hashing for frame similarity
  - [ ] Implement cache lookup logic
  - [ ] Implement "interesting frame" detection
  - [ ] Implement lazy generation background jobs

- [ ] **Adaptive Intervals**
  - [ ] Implement `determine_capture_interval()` logic
  - [ ] Update video capture to use adaptive intervals

- [ ] **Summarization Integration**
  - [ ] Update `summarize_segment()` to pre-generate lazy descriptions
  - [ ] Ensure all frames have descriptions before summarization
  - [ ] Include video frame descriptions in summary context

- [ ] **Retrieval Integration**
  - [ ] Update `retrieve_video_with_context()` to use descriptions
  - [ ] Implement on-demand generation fallback
  - [ ] Update prompt templates to include visual descriptions

- [ ] **Testing**
  - [ ] Test description generation quality
  - [ ] Test cache hit rates
  - [ ] Test lazy generation latency
  - [ ] Test summarization with descriptions
  - [ ] Measure cost per stream

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

### Video Frame Embeddings (CLIP)

- **Local CLIP**: Free (GPU required)
- **OpenAI API**: Not available (CLIP is local-only)
- **Recommendation**: Use local CLIP for MVP
- **Cost**: $0.00 (hardware cost only)

### Video Frame Descriptions (Vision API)

- **GPT-4o-mini Vision**: $0.150 per 1M input tokens, $0.600 per 1M output tokens
- **Per Frame**: ~$0.00019 (270 input + 250 output tokens)
- **With Adaptive Intervals (5-10s)**: ~540 frames/hour
- **Cost**: ~$0.103 per hour = **~$1.03 per 10-hour stream**
- **With Hybrid Strategy**: Additional 30-50% reduction via caching = **~$0.50-$0.70 per 10-hour stream**

### Chat Embeddings

- **OpenAI**: $0.02 per 1M tokens
- **Estimate**: ~100 messages/hour × 50 tokens = 5K tokens/hour
- **Cost**: ~$0.0001 per hour (negligible)

### Summarization

- **OpenAI**: $0.15 per 1M tokens (gpt-4o-mini)
- **Estimate**: ~30 summaries/hour × 500 tokens = 15K tokens/hour
- **Cost**: ~$0.002 per hour (very low)

### Total Cost Estimate (MVP)

- **Current (Audio Only)**: ~$0.01 per hour (audio embeddings + LLM)
- **With Video Frames (CLIP)**: ~$0.01 per hour (local CLIP, no API cost)
- **With Video Descriptions**: ~$0.103 per hour (GPT-4o-mini Vision API)
- **With Summarization**: ~$0.105 per hour
- **Total**: **~$1.05 per 10-hour stream** (with hybrid caching: **~$0.50-$0.70 per 10-hour stream**)

### Cost Optimization Strategies

1. **Adaptive Capture Intervals**: Reduces frames by 60-80% (5-10s vs 2s)
2. **Hybrid Lazy/Cached Generation**: 
   - Cache hits: 30% of frames (free)
   - Immediate generation: 20% of frames (high activity)
   - Lazy generation: 50% of frames (on-demand or before summarization)
3. **Perceptual Hashing**: Detect similar frames and reuse descriptions
4. **Background Jobs**: Pre-generate lazy descriptions before summarization (avoids latency)

### Cost Comparison

| Approach | Cost/Hour | Cost/10hr | Notes |
|----------|-----------|-----------|-------|
| GPT-4o (2s interval) | $57.24 | $572.40 | Too expensive for MVP |
| GPT-4o-mini (2s interval) | $0.342 | $3.42 | Baseline |
| GPT-4o-mini (10s interval) | $0.068 | $0.68 | Fixed interval |
| GPT-4o-mini (adaptive 5-10s) | $0.103 | $1.03 | Activity-based |
| GPT-4o-mini (hybrid lazy/cached) | $0.05-$0.07 | $0.50-$0.70 | **MVP Recommended** |
| Local LLaVA/Qwen-VL | $0.00* | $0.00* | *After hardware purchase ($1,500-$3,000) |

**MVP Recommendation**: GPT-4o-mini with adaptive intervals and hybrid lazy/cached generation = **~$0.50-$0.70 per 10-hour stream**

---

## Next Steps

1. **Review Research Document**: `docs/LONG_TERM_CONTEXT_RESEARCH.md`
2. **Prioritize Features**: Choose which enhancements to implement first
3. **Create Linear Issues**: Break down into tasks
4. **Start Implementation**: Begin with video frame embeddings (MVP 1.5)
5. **Iterate**: Test, measure, improve

---

**Document Version**: 2.0  
**Last Updated**: 2025-01-30  
**Related Documents**: `docs/LONG_TERM_CONTEXT_RESEARCH.md`, `docs/CONTEXT_RESEARCH_SUMMARY.md`, `docs/ARCHITECTURE.md`

**Major Updates (v2.0)**:
- Added Section 1.6: Visual Description Generation (MVP Solution)
- Documented GPT-4o-mini Vision API approach with hybrid lazy/cached generation
- Updated cost analysis with detailed breakdown
- Added implementation details for adaptive intervals and summarization integration
- Documented tradeoffs and future enhancements

