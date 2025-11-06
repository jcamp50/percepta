# Log Analysis: Node Service Race Condition

## What Happened

### Timeline from Logs:

**Node Service:**

1. **07:42:46** - Service starts, both audio and video capture initialize
2. **07:42:46** - "Stream came online" appears **TWICE** (once for audio, once for video)
3. **07:42:48** - "Resolved channel xqc to broadcaster ID: 71092938" appears **TWICE**
4. **07:42:49** - "Stream URL not available yet" appears **TWICE**
5. **07:42:50** - "Resolved channel xqc to broadcaster ID: 71092938" appears again **TWICE**

**Python Service:**

1. **02:42:30** - Service starts, EventSub initialized
2. **02:42:31** - Metadata polling starts, confirms stream is live (12,305 viewers)
3. **02:42:32** - EventSub subscriptions being created (some succeed, one fails with 403)

## Root Cause Analysis

### 1. **Duplicate Logs (Not a Race Condition)**

The duplicate logs are **expected behavior**, not a race condition:

- Both `AudioCapture` and `VideoCapture` have separate event listeners
- Both listen to the same `StreamManager` events:
  - `streamOnline` event → Both log "Stream came online"
  - `streamUrl` event → Both log "Resolved channel" and try to get broadcaster ID
- This is **by design** - each service needs to know when the stream is available

### 2. **Actual Issue: Stream URL Not Available**

The real problem is:

- StreamManager detects stream is online
- It tries to fetch the stream URL from Python service
- The fetch is **failing or returning null**
- This causes "Stream URL not available yet" messages

### 3. **Why Stream URL Fetch Might Fail**

Looking at `stream.js` line 183-243 (`_fetchStreamUrl`):

- Calls `/api/get-video-stream-url` endpoint
- Uses Streamlink to get authenticated HLS URL
- If stream is live but Streamlink can't get URL, it returns null
- Possible reasons:
  1. Streamlink authentication issue
  2. Stream not fully ready yet (just came online)
  3. Python service Streamlink not configured correctly
  4. Network/timing issue

### 4. **The "Race Condition" Pattern**

There IS a timing issue, but it's more of a **sequence issue**:

1. StreamManager detects stream is online → emits `streamOnline`
2. AudioCapture and VideoCapture both receive `streamOnline` → log message
3. StreamManager tries to fetch stream URL (async)
4. If URL fetch is slow/fails, AudioCapture/VideoCapture see no URL yet
5. They log "Stream URL not available yet"
6. Later, when URL fetch completes, `streamUrl` event is emitted
7. Both services receive it → log "Resolved channel" again

## Code Flow

```
StreamManager.startMonitoring()
  ↓
_checkAndUpdateStream() detects stream is live
  ↓
Emits 'streamOnline' event
  ↓ (BOTH AudioCapture AND VideoCapture receive this)
AudioCapture: logs "Stream came online"
VideoCapture: logs "Stream came online"
  ↓
StreamManager.getStreamUrl() called
  ↓
_fetchStreamUrl() calls Python service
  ↓
If URL fetch fails/slow → returns null
  ↓
AudioCapture/VideoCapture: log "Stream URL not available yet"
  ↓
Later: URL fetch succeeds → emits 'streamUrl' event
  ↓ (BOTH receive this)
AudioCapture: logs "Resolved channel" + gets broadcaster ID
VideoCapture: logs "Resolved channel" + gets broadcaster ID
```

## Solutions

### 1. **Fix Stream URL Fetch**

The main issue is the stream URL not being available. Check:

- Python service `/api/get-video-stream-url` endpoint
- Streamlink configuration
- Streamlink authentication

### 2. **Reduce Log Noise (Optional)**

If duplicate logs are confusing, we could:

- Only log from StreamManager (not from both AudioCapture and VideoCapture)
- Use a single "Stream available" log instead of separate logs

### 3. **Better Error Handling**

- Log why stream URL fetch failed
- Add retry logic with exponential backoff
- Log Streamlink errors more clearly

## Conclusion

**Not a race condition** - the duplicate logs are expected (two services listening to same events).

**Real issue**: Stream URL fetch is failing or slow, causing capture to not start.

**Next steps**: Investigate why `/api/get-video-stream-url` is returning null/empty when stream is live.
