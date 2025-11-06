# JCB-31 Implementation Notes & Research Findings

## Answers to Implementation Questions

### 1. Channel ID vs Broadcaster ID

**Question**: Shouldn't `channel_id` field be the numeric `broadcaster_id` that we changed to use across other tables?

**Answer**: Yes, you're correct! The codebase uses `channel_id` as the field name but stores broadcaster user ID (numeric string) values. This is consistent across all tables:

- `Transcript.channel_id` - stores broadcaster ID (see comments: "Broadcaster user ID")
- `Event.channel_id` - stores broadcaster ID
- `ChannelSnapshot.channel_id` - stores broadcaster ID

**Decision**: Use `channel_id` field name (for consistency with existing schema) but ensure it stores broadcaster user ID values, not channel names. This matches the existing pattern.

**Implementation**: The `VideoFrame` model should follow the same pattern - use `channel_id` field name but store broadcaster ID values.

---

### 2. Video Embeddings Utility - Dimension Mismatch Solution

**Problem**: CLIP models output 512-dim embeddings, but our database uses 1536-dim (OpenAI text-embedding-3-small).

**Research Findings**:

- CLIP models (openai/clip-vit-base-patch32): 512 dimensions
- OpenAI text-embedding-3-small: 1536 dimensions
- pgvector supports different dimensions per column, but we want unified search across modalities

**Recommended Approach for MVP (JCB-31)**:
**Linear Projection: 512 → 1536**

**Why this approach**:

1. **Maintains unified vector space**: All embeddings in same 1536-dim space for cross-modal search
2. **Preserves cosine similarity**: Normalization after projection maintains similarity relationships
3. **Simple and fast**: No training required, works immediately
4. **Future-compatible**: Can enhance with learned projection in JCB-37 (grounding)

**Implementation**:

```python
def project_clip_to_1536(clip_512: List[float]) -> List[float]:
    """Project 512-dim CLIP embedding to 1536-dim.

    Strategy: Repeat pattern (512 * 3 = 1536) + normalize
    This preserves cosine similarity while matching dimension.
    """
    clip_array = np.array(clip_512)
    # Repeat to reach 1536 dimensions
    projected = np.tile(clip_array, 3)
    # Normalize to maintain unit vector (preserves cosine similarity)
    norm = np.linalg.norm(projected)
    if norm > 0:
        projected = projected / norm
    return projected.tolist()
```

**Future Enhancement (JCB-37 - Grounding)**:

- Joint embedding fusion: Combine CLIP (512) + text context (1536) → grounded (1536)
- This will use proper learned fusion weights instead of simple projection

**Alternative Approaches Considered**:

1. **Separate 512-dim column**: Rejected - breaks unified search across transcripts/video/chat
2. **OpenAI image embedding API**: Not available for CLIP-equivalent embeddings
3. **Learned projection matrix**: Better accuracy but requires training - defer to JCB-37

---

### 3. Video Store - Blob Storage Requirements

**Question**: Does video store require another database like a blob store, or not necessary since we're not storing actual images?

**Answer**: **No separate blob store needed for MVP**. Store images on local filesystem.

**Rationale**:

- **Database stores**: Metadata + embeddings (efficient, small)
- **Filesystem stores**: Actual image files (~100-500KB each)
- **Storage pattern**: `frames/{broadcaster_id}/{timestamp}_{frame_id}.jpg`
- **Benefits**:
  - PostgreSQL not optimized for large binary blobs
  - Easy cleanup/deletion of old frames
  - Can migrate to S3/blob storage later if needed
  - Reduces database size and improves query performance

**Storage Requirements**:

- 18,000 frames per 10-hour stream
- ~100-500KB per frame = 1.8-9GB per 10-hour stream
- Filesystem handles this easily
- Can add cleanup policy (delete frames older than X days)

**Future Enhancement**: If storage becomes an issue, migrate to S3/Cloud Storage and store URLs in database.

---

### 4. Video Stream Capture - How to Get Video Stream

**Question**: How are we getting actual video stream? We use streamlink for audio, can we get video as well?

**Answer**: **Yes! Use Streamlink for video streams, same pattern as audio.**

**How it works**:

1. **Python Service**: Create `/api/get-video-stream-url` endpoint (similar to `/api/get-audio-stream-url`)

   - Uses Streamlink to get authenticated HLS URL
   - Instead of `streams.get("audio_only")`, use `streams.get("best")` or `streams.get("worst")`
   - Returns full video HLS URL

2. **Node.js Service**: Create `node/video.js` module
   - Similar structure to `audio.js`
   - Call Python endpoint to get video stream URL
   - Use `fluent-ffmpeg` to process video stream
   - Extract frames every 2 seconds using `-vf fps=0.5`

**FFmpeg Command Pattern**:

```javascript
// In node/video.js
ffmpeg(videoStreamUrl)
  .inputOptions(['-loglevel', 'error', '-reconnect', '1'])
  .outputOptions([
    '-vf',
    'fps=0.5', // Capture 1 frame every 2 seconds
    '-f',
    'image2', // Output as image sequence
    '-q:v',
    '2', // High quality JPEG
  ])
  .on('end', () => {
    // Frame captured, send to Python /api/video-frame
  });
```

**Streamlink Integration**:

```python
# In py/main.py - /api/get-video-stream-url
import streamlink

session = streamlink.Streamlink()
streams = session.streams(f"https://www.twitch.tv/{channel_id}")
video_stream = streams.get("best")  # or "worst" for lower bandwidth
return video_stream.url
```

**Key Differences from Audio**:

- Audio: Uses `streams.get("audio_only")` + `-vn` flag in FFmpeg
- Video: Uses `streams.get("best")` + `-vf fps=0.5` to extract frames

---

### 5. Full Roadmap Context (JCB-31 through JCB-40)

**Overview of Related Issues**:

- **JCB-31** (Current): Video frame embeddings - Pure CLIP embeddings
- **JCB-32**: Chat message embeddings - Will align temporally with video frames
- **JCB-33**: Enhanced multi-source retrieval - Combine video + transcripts + chat
- **JCB-37**: Grounding CLIP embeddings - Joint embedding fusion (CLIP + text context)
- **JCB-40**: Metadata embeddings - Game, title context

**Strategic Approach**:

1. **MVP (JCB-31)**: Start with pure CLIP embeddings, simple projection to 1536
2. **Prepare for JCB-32**: Add `transcript_id` field to VideoFrame (nullable) for future temporal alignment
3. **Post-MVP (JCB-37)**: Enhance with grounding - joint embedding fusion
4. **Future (JCB-40)**: Add metadata context to video frames

**Why This Approach**:

- **Incremental**: Build working MVP first, enhance later
- **Compatible**: MVP design supports future enhancements
- **Practical**: Simple projection works, can optimize later
- **Research-backed**: Documents recommend temporal alignment first (Approach 1), then grounding (Approach 2)

---

## Implementation Summary

### Key Decisions:

1. ✅ Use `channel_id` field name (stores broadcaster ID values)
2. ✅ Linear projection: CLIP 512 → 1536 for MVP
3. ✅ Filesystem storage for images (no blob store needed)
4. ✅ Use Streamlink for video streams (same pattern as audio)
5. ✅ Create `node/video.js` separate from `audio.js`

### Next Steps:

See updated implementation plan with all these decisions incorporated.
