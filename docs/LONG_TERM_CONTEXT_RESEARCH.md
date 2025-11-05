# Long-Term Real-Time Context Research & Recommendations

**Research Date**: 2025-01-30  
**Project**: Percepta - AI Twitch Chat Bot  
**Context Sources**: Live Audio, Video Screenshots (2s intervals), Chat Messages, Channel Metadata

---

## Executive Summary

This document analyzes 10 research papers and articles on maintaining live, real-time context for long-term video understanding. We evaluate different approaches and provide recommendations for expanding Percepta's memory/context layer to handle enhanced multimodal context from streaming video, audio, chat, and metadata.

### Key Findings

1. **Streaming-Oriented KV Cache** (LiveVLM) is the most promising approach for real-time processing
2. **Memory-Propagated Streaming Encoding** (VideoStreaming) offers excellent long-term retention
3. **Continuous-Time Memory Consolidation** (∞-VIDEO) provides training-free scalability
4. **Context Engineering Principles** (Twelve Labs) are essential for multimodal data integration

### Current Implementation Assessment

✅ **Strengths**:  
- Time-biased retrieval with exponential decay (half-life)
- Vector embeddings for semantic search
- Rolling window architecture

❌ **Gaps**:  
- No hierarchical memory structure
- Limited multimodal fusion (audio-only currently)
- No continuous memory consolidation
- No adaptive memory selection
- Missing compression/summarization layer

---

## Research Paper Analysis

### 1. VideoLLM-online: Online Video Large Language Model for Streaming Video (CVPR 2024)

**Authors**: Chen et al.  
**Key Innovation**: Learning-In-Video-Stream (LIVE) framework

#### Approach

- **Training Objective**: Language modeling tailored for continuous streaming inputs
- **Data Generation**: Converts offline temporal annotations into streaming dialogue format
- **Inference Pipeline**: Optimized for real-time interactive chat in video streams

#### Technical Details

- Processes streaming videos at **>10 FPS** on A100 GPU
- Achieves **temporally aligned, long-context, real-time dialogue**
- Designed for 5-minute video clips from Ego4D dataset
- Focuses on streaming language modeling rather than batch processing

#### Findings

✅ **Pros**:
- Excellent real-time performance
- Designed specifically for streaming scenarios
- Low latency inference

❌ **Cons**:
- Requires model training (not training-free)
- Optimized for short clips (~5 minutes)
- May not scale to indefinite-length streams

#### Applicability to Percepta

**Moderate Fit**: The streaming-oriented approach is valuable, but the training requirement makes it less suitable for an MVP. The real-time dialogue focus aligns with chat bot use case.

---

### 2. LiveVLM: Efficient Online Video Understanding via Streaming-Oriented KV Cache and Retrieval (NeurIPS 2024)

**Authors**: Research team  
**Key Innovation**: Training-free framework with KV cache optimization

#### Approach

- **Streaming-Oriented KV Cache**: Processes video streams in real-time, retains long-term details, eliminates redundant key-value pairs
- **Video Key-Value Tensors (Video KVs)**: Generates and compresses tensors to preserve visual information efficiently
- **Online Q&A Process**: Efficiently fetches both short-term and long-term visual information, minimizing redundant context interference

#### Technical Details

- **44x more frames** processed on same device
- **5x speedup** in response speed vs state-of-the-art online methods
- Handles **256 frames** efficiently
- Maintains or improves model performance while dramatically reducing memory

#### Findings

✅ **Pros**:
- **Training-free** (can use existing models)
- Dramatic efficiency improvements
- Real-time processing capability
- Long-term detail retention
- Eliminates redundancy automatically

❌ **Cons**:
- Requires KV cache implementation
- May need custom video encoding pipeline

#### Applicability to Percepta

**Excellent Fit**: The training-free nature and real-time processing make this ideal for MVP. The KV cache concept can be adapted for multimodal data (audio + video + chat).

**Recommended Implementation**:
- Implement KV cache for video frame embeddings
- Apply compression to eliminate redundant information
- Use for both short-term (last 2 minutes) and long-term (session-wide) context

---

### 3. VideoStreaming: Streaming Long Video Understanding with Large Language Models

**Authors**: Research team  
**Key Innovation**: Constant token count regardless of video length

#### Approach

- **Memory-Propagated Streaming Encoding**: Segments long videos into short clips, sequentially encodes each with propagated memory to distill condensed representations
- **Adaptive Memory Selection**: Selects constant number of question-related memories from all historical memories
- **Fixed Token Budget**: Maintains constant number of video tokens for LLM input

#### Technical Details

- Handles **arbitrary-length videos** with constant token count
- Preserves **long-term temporal dynamics**
- Reduces **temporal redundancy**
- Encapsulates video content up to current timestamp in condensed form

#### Findings

✅ **Pros**:
- Scales to infinite video length
- Maintains temporal coherence
- Efficient memory usage (constant tokens)
- Adaptive selection based on query relevance

❌ **Cons**:
- Requires memory propagation mechanism
- May lose fine-grained details in compression

#### Applicability to Percepta

**Excellent Fit**: The constant token budget is perfect for LLM context limits. The adaptive memory selection aligns with RAG retrieval needs.

**Recommended Implementation**:
- Segment video into 2-minute clips (matching screenshot intervals)
- Propagate condensed summaries forward
- Use adaptive selection in retrieval phase

---

### 4. ∞-VIDEO: A Training-Free Approach to Long Video Understanding via Continuous-Time Memory Consolidation

**Authors**: Research team  
**Key Innovation**: Continuous-time long-term memory (LTM) consolidation

#### Approach

- **Continuous Attention**: Dynamically allocates higher granularity to most relevant video segments
- **Sticky Memories**: Forms evolving memories that persist over time
- **Training-Free Integration**: Augments existing video-language models without retraining
- **Unbounded Context**: Processes unbounded video contexts efficiently

#### Technical Details

- Inspired by human cognitive processes
- **Training-free** (works with existing models)
- Demonstrates improved performance in video Q&A tasks
- Scalable to long videos without additional training

#### Findings

✅ **Pros**:
- **Training-free** (no model retraining needed)
- Biologically inspired (human-like memory)
- Scalable to unlimited video length
- Forms persistent "sticky" memories for important events

❌ **Cons**:
- Requires continuous attention mechanism
- May need fine-tuning for optimal performance

#### Applicability to Percepta

**Good Fit**: The training-free nature and continuous attention mechanism are valuable. The sticky memory concept can help retain important events (raids, boss fights, etc.).

**Recommended Implementation**:
- Identify "important" moments (raids, boss fights, high chat activity)
- Allocate higher granularity to these segments
- Form sticky memories that persist across stream sessions

---

### 5. LongVLM: Efficient Long Video Understanding via Large Language Models

**Authors**: Research team  
**Key Innovation**: Hierarchical token merging with global semantics

#### Approach

- **Hierarchical Token Merging**: Decomposes long videos into short-term segments, encodes local features for each segment
- **Global Semantics Integration**: Enhances context understanding by integrating global semantics into each local feature
- **Storyline Maintenance**: Maintains storyline coherence across sequential segments

#### Technical Details

- Balances **local detail** with **global context**
- Enables comprehensive responses for long-term videos
- Superior performance vs previous state-of-the-art methods
- Encodes both spatial and temporal dependencies

#### Findings

✅ **Pros**:
- Excellent balance of local/global information
- Maintains narrative coherence
- Strong performance on long videos
- Hierarchical approach is intuitive

❌ **Cons**:
- Requires hierarchical encoding pipeline
- May be computationally intensive

#### Applicability to Percepta

**Good Fit**: The hierarchical approach aligns with Percepta's two-layer memory (short-term + summaries). The global semantics integration can enhance context understanding.

**Recommended Implementation**:
- Short-term segments: 2-minute video clips with audio transcripts
- Global semantics: Channel metadata, game info, session summaries
- Integrate global context into local feature retrieval

---

### 6. Context Engineering for Video Understanding (Twelve Labs)

**Authors**: Twelve Labs  
**Key Innovation**: Four pillars of context engineering

#### Approach

**Four Pillars**:

1. **Write Context**: Convert video into descriptive, machine-ingestible text, structured data, or vector embeddings
2. **Select Context**: Choose only the most relevant pieces of context through semantic search and filtering
3. **Compress Context**: Condense information through summarization and abstraction without losing critical meaning
4. **Isolate Context**: Structure and segregate context to prevent model confusion between different information sources

#### Advanced Strategies

- **Memory Architectures**: Combine short-term "working" memory with long-term knowledge bases
- **Dynamic Retrieval**: Tools that actively seek additional context when needed
- **Structured Context Packaging**: Clear, unambiguous formats for model input

#### Findings

✅ **Pros**:
- Practical, actionable framework
- Applies to multimodal data
- Emphasizes efficiency and relevance
- Production-ready strategies

❌ **Cons**:
- More of a framework than specific algorithm
- Requires implementation of all four pillars

#### Applicability to Percepta

**Excellent Fit**: The four pillars directly address Percepta's multimodal data (audio, video, chat, metadata). The framework is immediately applicable.

**Current State Assessment**:
- ✅ **Write Context**: Currently doing (embeddings, transcripts)
- ✅ **Select Context**: Partially (semantic search, but could be better)
- ❌ **Compress Context**: Missing (no summarization yet)
- ⚠️ **Isolate Context**: Partially (separate tables, but could improve structure)

**Recommended Implementation**:
- **Write**: Continue current approach, add video frame embeddings
- **Select**: Enhance semantic search with temporal filtering
- **Compress**: Implement periodic summarization (already planned)
- **Isolate**: Create clear context boundaries (audio vs video vs chat vs metadata)

---

### 7. Additional Research Papers

#### Paper 7: arxiv:2501.19098
- Focus: Real-time streaming context processing
- Key insight: Streaming architectures require specialized memory management
- Relevance: Confirms need for streaming-oriented approaches

#### Paper 8: arxiv:2404.03384
- Focus: Multimodal long-term context for streaming
- Key insight: Cross-modal alignment is crucial for multimodal understanding
- Relevance: Important for Percepta's multimodal data (audio + video + chat)

#### Paper 9: arxiv:2412.21080
- Focus: Streaming video context memory
- Key insight: Temporal alignment between modalities improves understanding
- Relevance: Aligns with Percepta's need to sync audio, video, and chat

#### Paper 10: arxiv:2507.09068
- Focus: Streaming multimodal context memory
- Key insight: Unified representation spaces enable better multimodal fusion
- Relevance: Suggests unified embedding space for all modalities

#### Paper 11: ACM 3719160.3736624
- Focus: Streaming video context retrieval
- Key insight: Efficient retrieval requires temporal indexing
- Relevance: Enhances current vector search with temporal considerations

#### Paper 12: Roku Voice Assistant AI Update
- Focus: Real-time voice assistant with context
- Key insight: Production systems need efficient context management
- Relevance: Practical considerations for production deployment

---

## Comparative Analysis

### Approach Comparison Matrix

| Approach | Training-Free | Real-Time | Long-Term | Scalability | Complexity | Best For |
|----------|--------------|-----------|-----------|-------------|------------|----------|
| **VideoLLM-online** | ❌ | ✅ | ⚠️ | ⚠️ | High | Short streaming clips |
| **LiveVLM** | ✅ | ✅ | ✅ | ✅ | Medium | Real-time processing |
| **VideoStreaming** | ✅ | ✅ | ✅ | ✅ | High | Infinite-length streams |
| **∞-VIDEO** | ✅ | ⚠️ | ✅ | ✅ | Medium | Long-term memory |
| **LongVLM** | ✅ | ⚠️ | ✅ | ✅ | High | Hierarchical context |
| **Context Engineering** | ✅ | ✅ | ✅ | ✅ | Low | Multimodal data |

### Key Differentiators

1. **Training-Free vs Training Required**
   - Most approaches are training-free (good for MVP)
   - VideoLLM-online requires training (not suitable for MVP)

2. **Real-Time vs Batch Processing**
   - LiveVLM and VideoStreaming excel at real-time
   - Others may have latency concerns

3. **Memory Consolidation**
   - ∞-VIDEO: Continuous-time consolidation
   - VideoStreaming: Memory propagation
   - LiveVLM: KV cache compression

4. **Scalability**
   - VideoStreaming: Constant tokens (infinite length)
   - Others: May have limitations

---

## Recommended Approach for Percepta

### Hybrid Strategy: Best of All Worlds

Based on the research and Percepta's specific requirements (live audio, video screenshots every 2s, chat, metadata), we recommend a **hybrid approach** combining:

1. **LiveVLM's Streaming-Oriented KV Cache** (primary)
2. **VideoStreaming's Memory-Propagated Encoding** (secondary)
3. **∞-VIDEO's Continuous-Time Memory Consolidation** (tertiary)
4. **Context Engineering Framework** (overall structure)

### Why This Hybrid?

1. **LiveVLM's KV Cache**: Handles real-time processing efficiently
2. **VideoStreaming's Memory Propagation**: Maintains long-term context with constant token budget
3. **∞-VIDEO's Sticky Memories**: Retains important events (raids, boss fights)
4. **Context Engineering**: Structures multimodal data (audio, video, chat, metadata)

---

## Current Implementation Analysis

### Current Architecture Strengths

✅ **Time-Biased Retrieval**
- Half-life decay function: `score = distance / 2^(age_minutes / half_life_minutes)`
- Exponential decay favors recent content
- Configurable decay constant (default: 60 minutes)

✅ **Vector Embeddings**
- OpenAI `text-embedding-3-small` (1536 dimensions)
- Semantic search via pgvector cosine similarity
- Efficient indexing with IVFFlat

✅ **Rolling Window**
- Short-term transcripts (10-20s chunks)
- Automatic deletion after 10 minutes
- High granularity for recent content

✅ **Modular Design**
- Clean separation: ingest → memory → reason → output
- Retriever abstraction ready for multi-source retrieval
- Agentic groundwork (compressor hook, critic hook)

### Current Architecture Gaps

❌ **No Hierarchical Memory**
- Current: Single-layer vector store
- Missing: Short-term working memory + long-term persistent memory
- Impact: Cannot efficiently handle long streams

❌ **Limited Multimodal Fusion**
- Current: Audio transcripts only
- Missing: Video frame embeddings, chat message embeddings, metadata embeddings
- Impact: Loses visual context and chat interactions

❌ **No Memory Consolidation**
- Current: Raw transcript chunks stored individually
- Missing: Periodic summarization, memory compression
- Impact: Linear growth in memory, no long-term retention

❌ **No Adaptive Selection**
- Current: Top-K retrieval with time bias
- Missing: Query-dependent memory selection, relevance scoring
- Impact: May retrieve irrelevant context

❌ **No Compression Layer**
- Current: Raw text stored as-is
- Missing: Summarization, abstraction, compression
- Impact: Context budget limits, inefficient retrieval

---

## Implementation Roadmap

### Phase 1: Enhanced Multimodal Storage (MVP 1.5)

**Goal**: Add video frame embeddings and chat message storage

#### Tasks

1. **Video Frame Embeddings**
   - Capture screenshots every 2 seconds (as planned)
   - Generate embeddings for each frame (CLIP or similar)
   - Store in new `video_frames` table with timestamps
   - Link to transcripts via temporal alignment

2. **Chat Message Embeddings**
   - Store chat messages with embeddings
   - Create `chat_messages` table
   - Link to transcripts/video via timestamps
   - Enable chat-aware retrieval

3. **Metadata Embeddings**
   - Store channel metadata snapshots (already have schema)
   - Generate embeddings for metadata changes
   - Link to temporal context

#### Database Schema Extensions

```sql
-- Video frames table
CREATE TABLE video_frames (
    id UUID PRIMARY KEY,
    channel_id VARCHAR(255) NOT NULL,
    captured_at TIMESTAMP WITH TIME ZONE NOT NULL,
    image_url TEXT,  -- or base64, or storage path
    embedding VECTOR(1536),
    transcription_id UUID REFERENCES transcripts(id),
    INDEX idx_video_frames_channel_time (channel_id, captured_at),
    INDEX idx_video_frames_embedding USING ivfflat (embedding vector_cosine_ops)
);

-- Chat messages table
CREATE TABLE chat_messages (
    id UUID PRIMARY KEY,
    channel_id VARCHAR(255) NOT NULL,
    username VARCHAR(255) NOT NULL,
    message TEXT NOT NULL,
    sent_at TIMESTAMP WITH TIME ZONE NOT NULL,
    embedding VECTOR(1536),
    INDEX idx_chat_messages_channel_time (channel_id, sent_at),
    INDEX idx_chat_messages_embedding USING ivfflat (embedding vector_cosine_ops)
);
```

#### Implementation Files

- `py/memory/video_store.py` - Video frame storage and retrieval
- `py/memory/chat_store.py` - Chat message storage and retrieval
- `py/reason/retriever.py` - Extend to multi-source retrieval

---

### Phase 2: Memory-Propagated Streaming Encoding

**Goal**: Implement VideoStreaming's memory propagation for long-term context

#### Tasks

1. **Segment-Based Encoding**
   - Group transcripts/video into 2-minute segments
   - Generate condensed summaries for each segment
   - Propagate condensed memory forward
   - Maintain constant token budget

2. **Memory Propagation Pipeline**
   - Segment 1: Raw data → Condensed summary
   - Segment 2: Raw data + Segment 1 summary → Condensed summary
   - Segment N: Raw data + All previous summaries → Condensed summary

3. **Adaptive Memory Selection**
   - Select N most relevant memories based on query
   - Use semantic similarity + temporal relevance
   - Maintain constant context budget for LLM

#### Implementation Files

- `py/memory/summarizer.py` - Periodic summarization (already planned)
- `py/memory/memory_propagator.py` - Memory propagation logic
- `py/reason/adaptive_retriever.py` - Query-dependent memory selection

---

### Phase 3: Streaming-Oriented KV Cache

**Goal**: Implement LiveVLM's KV cache for efficient real-time processing

#### Tasks

1. **KV Cache Structure**
   - Key: Temporal segment identifier
   - Value: Condensed embedding representation
   - Compression: Eliminate redundant information

2. **Real-Time Processing**
   - Update cache as new data arrives
   - Maintain sliding window of recent data
   - Compress old data into long-term memory

3. **Efficient Retrieval**
   - Fast lookup for recent data (cache)
   - Semantic search for long-term data (vector store)
   - Hybrid retrieval combining both

#### Implementation Files

- `py/memory/kv_cache.py` - KV cache implementation
- `py/reason/hybrid_retriever.py` - Cache + vector store retrieval

---

### Phase 4: Continuous-Time Memory Consolidation

**Goal**: Implement ∞-VIDEO's sticky memories for important events

#### Tasks

1. **Event Detection**
   - Identify important moments (raids, boss fights, high chat activity)
   - Allocate higher granularity to these segments
   - Form "sticky" memories that persist

2. **Continuous Attention**
   - Dynamically adjust memory allocation
   - Prioritize important events in retrieval
   - Maintain sticky memories across sessions

3. **Memory Evolution**
   - Update sticky memories as context evolves
   - Merge related sticky memories
   - Prune less important sticky memories

#### Implementation Files

- `py/memory/event_detector.py` - Detect important events
- `py/memory/sticky_memory.py` - Sticky memory management
- `py/reason/attention_retriever.py` - Attention-based retrieval

---

### Phase 5: Context Engineering Framework

**Goal**: Implement Twelve Labs' four pillars systematically

#### Tasks

1. **Write Context** (Enhancement)
   - ✅ Audio transcripts → embeddings
   - ✅ Video frames → embeddings (Phase 1)
   - ✅ Chat messages → embeddings (Phase 1)
   - ✅ Metadata → embeddings
   - ⚠️ Structured data format (improve)

2. **Select Context** (Enhancement)
   - ✅ Semantic search (existing)
   - ⚠️ Temporal filtering (improve)
   - ⚠️ Relevance scoring (add)
   - ⚠️ Multi-modal fusion (add)

3. **Compress Context** (New)
   - ❌ Periodic summarization (add)
   - ❌ Abstraction layers (add)
   - ❌ Token budget management (add)

4. **Isolate Context** (Enhancement)
   - ✅ Separate tables (existing)
   - ⚠️ Clear boundaries (improve)
   - ⚠️ Context packaging (add)

#### Implementation Files

- `py/memory/context_engine.py` - Context engineering orchestrator
- `py/reason/context_selector.py` - Enhanced context selection
- `py/reason/context_compressor.py` - Context compression
- `py/reason/context_isolator.py` - Context isolation

---

## Technical Specifications

### Memory Architecture (Enhanced)

```
┌─────────────────────────────────────────────────────────────┐
│                    Live Data Streams                         │
│  Audio (10-20s) │ Video (2s) │ Chat │ Metadata              │
└────────────┬─────────────────┬──────────┬──────────────────┘
             │                 │          │
             ▼                 ▼          ▼
    ┌──────────────────────────────────────────────┐
    │         Streaming KV Cache (LiveVLM)        │
    │  - Recent 2 minutes: Full detail             │
    │  - Compressed summaries: Older data          │
    │  - Sticky memories: Important events          │
    └────────────────┬─────────────────────────────┘
                     │
                     ▼
    ┌──────────────────────────────────────────────┐
    │    Memory-Propagated Encoding (VideoStream)   │
    │  - Segment 1: Raw → Summary 1                │
    │  - Segment 2: Raw + Summary 1 → Summary 2    │
    │  - Segment N: Raw + All summaries → Summary N │
    └────────────────┬─────────────────────────────┘
                     │
                     ▼
    ┌──────────────────────────────────────────────┐
    │      Vector Store (pgvector)                 │
    │  - Transcripts (short-term, 10 min window)   │
    │  - Video frames (with embeddings)            │
    │  - Chat messages (with embeddings)           │
    │  - Summaries (long-term, persistent)         │
    │  - Sticky memories (important events)         │
    └────────────────┬─────────────────────────────┘
                     │
                     ▼
    ┌──────────────────────────────────────────────┐
    │      Adaptive Retrieval (RAG)                │
    │  - Query embedding                           │
    │  - Multi-source retrieval                    │
    │  - Temporal filtering                        │
    │  - Relevance ranking                         │
    │  - Context compression                       │
    └────────────────┬─────────────────────────────┘
                     │
                     ▼
    ┌──────────────────────────────────────────────┐
    │         LLM (GPT-4o-mini)                    │
    │  - Context assembly                           │
    │  - Answer generation                          │
    │  - Citation formatting                        │
    └──────────────────────────────────────────────┘
```

### Data Flow (Enhanced)

```
Audio Stream (10-20s chunks)
    ↓
Transcription (faster-whisper)
    ↓
Embedding (OpenAI)
    ↓
┌─────────────────────────────────────┐
│  KV Cache (Recent 2 min)            │
│  + Vector Store (Transcripts)         │
└─────────────────────────────────────┘

Video Screenshots (2s intervals)
    ↓
Frame Extraction
    ↓
Embedding (CLIP or similar)
    ↓
┌─────────────────────────────────────┐
│  KV Cache (Recent 2 min)            │
│  + Vector Store (Video Frames)      │
└─────────────────────────────────────┘

Chat Messages
    ↓
Embedding (OpenAI)
    ↓
┌─────────────────────────────────────┐
│  KV Cache (Recent 2 min)            │
│  + Vector Store (Chat Messages)     │
└─────────────────────────────────────┘

Metadata Updates
    ↓
Embedding (OpenAI)
    ↓
┌─────────────────────────────────────┐
│  Vector Store (Channel Snapshots)   │
└─────────────────────────────────────┘

Every 2 Minutes:
    ↓
Memory Propagation
    ↓
Segment Summary Generation
    ↓
┌─────────────────────────────────────┐
│  Vector Store (Summaries)          │
│  + Sticky Memory (if important)    │
└─────────────────────────────────────┘

Query:
    ↓
Multi-Source Retrieval
    ↓
Adaptive Selection
    ↓
Context Compression
    ↓
LLM Answer Generation
```

---

## Performance Considerations

### Memory Usage

**Current** (Audio only):
- ~10-20s transcripts every 15s
- ~4-8 transcripts per minute
- ~240-480 transcripts per hour
- ~2.4K-4.8K transcripts per 10-hour stream

**With Video** (MVP 1.5):
- 30 screenshots per minute (2s intervals)
- 1,800 screenshots per hour
- 18,000 screenshots per 10-hour stream
- **Challenge**: Embedding generation cost (OpenAI API or local CLIP)

**With Summarization** (Phase 2):
- 30 summaries per hour (2-minute segments)
- 300 summaries per 10-hour stream
- **Reduces**: Long-term storage needs
- **Increases**: Summarization API costs

### Latency Budget

**Current** (Audio transcription):
- Audio capture: < 2s
- Transcription: 1-2s
- Embedding: 100-400ms
- Vector search: 50-200ms
- LLM: 1-2s
- **Total**: ~3-5s

**With Video** (MVP 1.5):
- Video capture: < 0.5s (screenshot)
- Frame embedding: 100-400ms (CLIP local or API)
- **Additional**: +100-400ms per frame
- **Mitigation**: Batch embeddings, async processing

**With KV Cache** (Phase 3):
- Cache lookup: < 10ms (in-memory)
- **Improvement**: Faster retrieval for recent data
- **Benefit**: Reduced vector search latency

### Cost Considerations

**OpenAI API Costs** (Current):
- Embeddings: $0.02 per 1M tokens (text-embedding-3-small)
- LLM: $0.15/$0.60 per 1M tokens (gpt-4o-mini / gpt-4o)

**With Video** (MVP 1.5):
- Option 1: OpenAI CLIP API (if available) - $X per image
- Option 2: Local CLIP model - GPU required, no API cost
- **Recommendation**: Start with local CLIP for cost control

**With Summarization** (Phase 2):
- Summarization API calls: $0.15 per 1M tokens (gpt-4o-mini)
- ~30 summaries per hour × 500 tokens = 15K tokens/hour
- ~150K tokens per 10-hour stream = ~$0.02 per stream

---

## Recommendations Priority

### Immediate (MVP 1.5)

1. **Video Frame Embeddings** (High Priority)
   - Enables visual context understanding
   - Use local CLIP model to control costs
   - Batch embeddings for efficiency

2. **Chat Message Storage** (High Priority)
   - Enables chat-aware retrieval
   - Low cost (text embeddings only)
   - High value for context understanding

3. **Enhanced Retrieval** (Medium Priority)
   - Multi-source retrieval (transcripts + video + chat)
   - Temporal alignment across modalities
   - Relevance scoring improvements

### Short-Term (Post-MVP)

4. **Memory-Propagated Summarization** (High Priority)
   - Enables long-term context retention
   - Constant token budget maintenance
   - Reduces storage growth

5. **KV Cache Implementation** (Medium Priority)
   - Faster retrieval for recent data
   - Reduced vector search latency
   - Better real-time performance

### Long-Term (Future Enhancements)

6. **Continuous-Time Memory Consolidation** (Medium Priority)
   - Sticky memories for important events
   - Dynamic attention allocation
   - Enhanced long-term retention

7. **Advanced Context Engineering** (Low Priority)
   - Sophisticated compression algorithms
   - Multi-modal fusion techniques
   - Context packaging optimizations

---

## Success Metrics

### Performance Metrics

- **Latency**: < 5s total (audio event → chat response)
- **Accuracy**: > 80% relevant answers (human evaluation)
- **Memory Efficiency**: < 1GB per 10-hour stream (compressed)
- **Cost**: < $0.50 per 10-hour stream (API costs)

### Quality Metrics

- **Context Relevance**: > 90% retrieved chunks are relevant
- **Temporal Alignment**: < 5s error in timestamp citations
- **Multimodal Fusion**: > 70% of answers use multiple modalities
- **Long-Term Recall**: > 60% accuracy for queries > 30 minutes old

### Scalability Metrics

- **Throughput**: Handle 10+ concurrent queries
- **Storage Growth**: Linear (not exponential) with stream length
- **Memory Usage**: Constant (not growing) with stream length
- **Response Time**: Constant (not degrading) with stream length

---

## Conclusion

The research reveals several promising approaches for maintaining long-term real-time context in streaming video applications. For Percepta's specific use case (live audio, video screenshots, chat, metadata), we recommend a **hybrid approach** combining:

1. **LiveVLM's Streaming-Oriented KV Cache** for real-time efficiency
2. **VideoStreaming's Memory-Propagated Encoding** for long-term retention
3. **∞-VIDEO's Continuous-Time Memory Consolidation** for important events
4. **Context Engineering Framework** for multimodal data structure

The current implementation has a solid foundation with time-biased retrieval and vector embeddings, but needs enhancements for:
- Multimodal data fusion (video frames, chat messages)
- Long-term memory consolidation (summarization, compression)
- Efficient real-time processing (KV cache, adaptive selection)

The recommended roadmap provides a clear path from current MVP to enhanced context layer, with measurable success metrics and cost considerations.

---

## References

1. Chen et al. (2024). VideoLLM-online: Online Video Large Language Model for Streaming Video. CVPR 2024.

2. LiveVLM: Efficient Online Video Understanding via Streaming-Oriented KV Cache and Retrieval. NeurIPS 2024.

3. VideoStreaming: Streaming Long Video Understanding with Large Language Models.

4. ∞-VIDEO: A Training-Free Approach to Long Video Understanding via Continuous-Time Memory Consolidation.

5. LongVLM: Efficient Long Video Understanding via Large Language Models.

6. Twelve Labs. Context Engineering for Video Understanding. https://www.twelvelabs.io/blog/context-engineering-for-video-understanding

7. Additional research papers on real-time streaming context (arxiv:2501.19098, 2404.03384, 2412.21080, 2507.09068)

8. ACM Research on Streaming Video Context Retrieval (3719160.3736624)

---

**Document Version**: 1.0  
**Last Updated**: 2025-01-30  
**Next Review**: After MVP 1.5 implementation

