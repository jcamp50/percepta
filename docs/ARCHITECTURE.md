# Percepta - AI Twitch Chat Bot

> Real-time contextual Q&A for Twitch streams powered by RAG and live transcription

**Linear Project**: [Percepta - AI Twitch Chat Bot](https://linear.app/jcbuilds/project/percepta-ai-twitch-chat-bot-ead8d9ea69f6)

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Architecture](#architecture)
3. [Technical Stack](#technical-stack)
4. [Memory Architecture](#memory-architecture)
5. [Development Phases](#development-phases)
6. [Project Structure](#project-structure)
7. [Environment Setup](#environment-setup)
8. [Development Workflow](#development-workflow)
9. [Performance Targets](#performance-targets)
10. [Safety & Moderation](#safety--moderation)
11. [Linear Integration](#linear-integration)

---

## Project Overview

### What is Percepta?

Percepta is an AI-powered Twitch chat bot that provides real-time, contextual answers about what's happening in a live stream. It listens to chat, transcribes stream audio, tracks channel events, and uses RAG (Retrieval-Augmented Generation) to answer viewer questions with grounded, timestamped responses.

### Key Capabilities

- **Real-time Transcription**: Captures and transcribes stream audio in 10-20s chunks
- **Contextual Memory**: Maintains rolling 5-10 minute memory of transcripts, events, and metadata
- **Grounded Answers**: Responds to @mentions with answers backed by actual stream content
- **Multi-User Support**: Handles parallel conversations with multiple viewers
- **Event Awareness**: Tracks raids, subscriptions, polls, predictions, and stream state
- **Long-term Recall**: Periodic summarization enables recall beyond the rolling window

### Target Users

- **MVP**: Single channel monitoring (configurable via ENV)
- **Future**: Multi-channel support for any streamer who opts in

### Use Cases

- "What just happened?"
- "What game are they playing?"
- "Who raided?"
- "What did they say about the boss fight?"
- "When did the stream start?"

---

## Architecture

### High-Level System Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         TWITCH STREAM                            │
│                    (Audio + Chat + Events)                       │
└────────────┬────────────────────────┬───────────────────────────┘
             │                        │
             │                        │
    ┌────────▼────────┐      ┌────────▼────────┐
    │  Node Service   │      │  Twitch APIs    │
    │  - Audio Cap    │      │  - EventSub WS  │
    │  - IRC (tmi.js) │      │  - Helix API    │
    └────────┬────────┘      └────────┬────────┘
             │                        │
             │    HTTP/WebSocket      │
             │                        │
    ┌────────▼────────────────────────▼────────┐
    │         Python Backend (FastAPI)         │
    │  ┌──────────────────────────────────┐   │
    │  │  Ingest Layer                    │   │
    │  │  - Transcription (faster-whisper)│   │
    │  │  - Event Processing              │   │
    │  │  - Metadata Polling              │   │
    │  └──────────────┬───────────────────┘   │
    │                 │                        │
    │  ┌──────────────▼───────────────────┐   │
    │  │  Memory Layer                    │   │
    │  │  - Vector Store (pgvector)       │   │
    │  │  - Embeddings (OpenAI)           │   │
    │  │  - Summarizer (periodic)         │   │
    │  │  - Session Manager (Redis)       │   │
    │  └──────────────┬───────────────────┘   │
    │                 │                        │
    │  ┌──────────────▼───────────────────┐   │
    │  │  Reasoning Layer                 │   │
    │  │  - RAG Retrieval                 │   │
    │  │  - LLM (GPT-4o-mini)             │   │
    │  │  - Fallback Agent                │   │
    │  └──────────────┬───────────────────┘   │
    │                 │                        │
    │  ┌──────────────▼───────────────────┐   │
    │  │  Output Layer                    │   │
    │  │  - IRC Bridge                    │   │
    │  │  - Rate Limiting                 │   │
    │  │  - Response Formatting           │   │
    │  └──────────────────────────────────┘   │
    └─────────────────┬────────────────────────┘
                      │
                      ▼
              ┌───────────────┐
              │  Twitch Chat  │
              │   (Response)  │
              └───────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                      Storage Layer                               │
│  ┌──────────────────┐         ┌──────────────────┐             │
│  │  PostgreSQL      │         │     Redis        │             │
│  │  + pgvector      │         │  - Sessions      │             │
│  │  - Transcripts   │         │  - Rate Limits   │             │
│  │  - Events        │         │  - Hot Cache     │             │
│  │  - Summaries     │         │                  │             │
│  └──────────────────┘         └──────────────────┘             │
└─────────────────────────────────────────────────────────────────┘
```

### Component Breakdown

#### Node.js Services

- **Chat I/O**: `tmi.js` IRC client for reading/sending messages
- **Audio Capture**: Extracts audio from Twitch HLS stream via ffmpeg
- **Bridge**: Forwards messages and audio to Python backend

#### Python Backend (FastAPI)

- **Ingest**: Receives audio, transcribes, processes events, polls metadata
- **Memory**: Manages vector storage, embeddings, summarization, sessions
- **Reasoning**: RAG retrieval + LLM answer generation (`py/reason/rag.py`)
  - Embed viewer question (`py/utils/embeddings.embed_text`)
  - Retrieve recent transcripts via time-biased search (`VectorStore.search_transcripts`)
  - Build grounded prompt with timestamped citations
  - Call OpenAI `gpt-4o-mini` and return answer + citations via `POST /rag/answer`
- **Output**: Formats responses, enforces rate limits, queues messages

#### Databases

- **PostgreSQL + pgvector**: Vector embeddings for transcripts, events, summaries
- **Redis**: User sessions, rate limiting, hot cache

#### External APIs

- **Twitch EventSub**: Real-time channel events (raids, subs, polls)
- **Twitch Helix**: Channel metadata (title, game, tags)
- **Twitch IRC**: Chat read/write
- **OpenAI**: Embeddings (text-embedding-3-small) + LLM (GPT-4o-mini)

### Data Flow

```
Audio Stream → Node Capture → 15s Chunks → Python /transcribe
                                                    ↓
                                            faster-whisper
                                                    ↓
                                            Embed (OpenAI)
                                                    ↓
                                            pgvector INSERT
                                                    ↓
                                            [Rolling Window]

Chat Message → Node IRC → Python /chat/message → @mention check
                                                       ↓
                                                  RAG Query
                                                       ↓
                                    Search pgvector (transcripts + events + summaries)
                                                       ↓
                                                  LLM (GPT-4o-mini)
                                                       ↓
                                            Response + Timestamp
                                                       ↓
                                            Node IRC → Twitch Chat
```

---

## Technical Stack

### Python (Backend)

- **FastAPI**: Web server and API endpoints
- **LangGraph**: Agent orchestration (simplified for MVP: single RAG node)
- **SQLAlchemy**: ORM for PostgreSQL
- **faster-whisper**: Local speech-to-text
- **twitchAPI**: EventSub WebSocket + Helix API client
- **openai**: Embeddings and LLM
- **redis-py**: Session and cache management
- **pgvector**: Vector similarity search
- **pydantic**: Data validation and schemas

### Node.js (Chat & Audio)

- **tmi.js**: Twitch IRC client
- **ffmpeg / fluent-ffmpeg**: Audio extraction from HLS
- **axios**: HTTP client for Python backend communication

### Databases

- **PostgreSQL 15+**: Primary database with pgvector extension
- **Redis 7+**: Session state, rate limiting, hot cache

### External APIs

- **OpenAI API**:
  - `text-embedding-3-small` (1536 dimensions)
  - `gpt-4o-mini` (reasoning)
- **Twitch API**:
  - EventSub WebSocket (channel events)
  - Helix API (metadata)
  - IRC (chat I/O)

### Infrastructure

- **Development**: ngrok for local tunneling
- **Production**: Fly.io (planned post-MVP)
- **Containerization**: Docker Compose for local PostgreSQL + Redis

---

## Memory Architecture

### Two-Layer Memory System

#### Layer 1: Short-Term (Ephemeral)

**Rolling 5-10 Minute Window**

- **Purpose**: Real-time retrieval for "what just happened" queries
- **Storage**: Raw transcript chunks (10-20s each) with embeddings
- **Retention**: Automatically deleted after 10 minutes
- **Granularity**: High (word-level timestamps)
- **Use Case**: "What did they just say?" "What boss are they fighting?"

**Data Structure**:

```python
{
  "id": "uuid",
  "channel_id": "streamer_username",
  "started_at": "2025-10-26T01:15:00Z",
  "ended_at": "2025-10-26T01:15:15Z",
  "text": "Alright, let's try this boss again...",
  "embedding": [0.123, -0.456, ...],  # 1536 dims
  "speaker": "streamer"
}
```

#### Layer 2: Long-Term (Persistent Summaries)

**Periodic Summarization (Every 30s)**

- **Purpose**: Long-term context and historical recall
- **Storage**: Summarized documents with embeddings
- **Retention**: Persists across stream session
- **Granularity**: Low (high-level narrative)
- **Use Case**: "What happened 20 minutes ago?" "How many times did they fight the boss?"

**Summary Types**:

1. **Now Summary** (last 1-2 min): "Streamer is fighting boss, health at 50%"
2. **Recent Summary** (last 5 min): "Streamer attempted boss 3 times, died each time"
3. **Session Summary** (optional): "Stream started with intro, now in boss fight section"

**Data Structure**:

```python
{
  "id": "uuid",
  "channel_id": "streamer_username",
  "summary_type": "recent",  # now, recent, session
  "time_range_start": "2025-10-26T01:10:00Z",
  "time_range_end": "2025-10-26T01:15:00Z",
  "summary_text": "Streamer attempted boss fight three times...",
  "embedding": [0.789, -0.234, ...]
}
```

### Vector Search Strategy

**Time-Biased Retrieval**:

- Recent chunks weighted higher (exponential decay)
- Formula: `similarity_score * exp(-age_minutes / decay_constant)`
- Decay constant: 5 minutes (configurable)

**Hybrid Search**:

1. Search short-term transcripts (last 10 min)
2. Search long-term summaries (entire session)
3. Search events (raids, subs, polls)
4. Search metadata (current game, title)
5. Merge and rank by relevance + recency

### Agentic Groundwork (MVP-ready)

To prepare for a future agentic architecture without adding latency to MVP, we added:

- Tool boundaries: small interfaces around embedding, vector search, context building, and LLM calls (`py/reason/interfaces.py`).
- Retriever abstraction: a single entry point to route retrievals (`py/reason/retriever.py`), currently transcripts-only but ready to add events/summaries/metadata.
- Minimal RAG state: per-channel/user shape for session-aware tuning (`schemas/rag_state.py`).
- Telemetry: step timings and hit stats in `RAGService` for data-driven policies.
- Budgeting hook: optional compressor for context assembly (no-op by default).
- Guardrails: optional lightweight critic to ensure timestamp citations.

These changes keep the current single-pass RAG fast while allowing an easy evolution to a node/edge graph later.

### MVP 1.5 (Planned) – Agentic + Video Understanding

- Agentic graph with nodes: QueryRewrite → Multi-Retriever (transcripts/events/summaries/metadata) → Merger/Ranker → Compressor → Generator → Critic/Repair.
- Video understanding: clip embeddings aligned to transcripts; scene/shot detectors; cross-modal ranking.
- Policies: adaptive time windows, multi-hop retrieval, cost/time step limits, safety checks.

---

## Development Phases

### Phase 1: Chat I/O Only

**Goal**: Bot responds to messages in chat

**Linear Issues**: JCB-5 through JCB-9

**Tasks**:

1. [JCB-5] Project Setup & Environment Configuration
2. [JCB-6] Twitch Bot Account & OAuth Setup
3. [JCB-7] Node IRC Chat Service
4. [JCB-8] Python FastAPI Service Stub
5. [JCB-9] End-to-End Chat Flow (Ping/Pong)

**Success Criteria**: Bot joins channel and replies "pong" to "@bot ping"

---

### Phase 2: Vector Store + RAG

**Goal**: Bot answers from embedded knowledge

**Linear Issues**: JCB-10 through JCB-14

**Tasks**:

1. [JCB-10] PostgreSQL + pgvector Setup
2. [JCB-11] SQLAlchemy Models & Vector Store Interface
3. [JCB-12] OpenAI Embedding Utility
4. [JCB-13] RAG Retrieval Pipeline
5. [JCB-14] Manual Testing with Sample Data

**Success Criteria**: Manually insert transcript, bot answers questions with citations

---

### Phase 3: Transcription Pipeline

**Goal**: Real-time audio → text → vector DB

**Linear Issues**: JCB-15 through JCB-18

**Tasks**:

1. [JCB-15] Twitch Stream Audio Capture (Node)
2. [JCB-16] faster-whisper Integration
3. [JCB-17] Audio-to-Transcript Pipeline
4. [JCB-18] Live Stream Testing

**Success Criteria**: Transcripts appear in DB within 5s during live stream

---

### Phase 4: EventSub + Metadata

**Goal**: Channel context (title, game, events)

**Linear Issues**: JCB-19 through JCB-22

**Tasks**:

1. [JCB-19] EventSub WebSocket Client
2. [JCB-20] Channel Metadata Polling
3. [JCB-21] Event Storage & Embedding
4. [JCB-22] Enhanced Context Testing

**Success Criteria**: Bot knows current game and responds to raid events

---

### Phase 5: Summarization + Multi-User

**Goal**: Long-term memory and parallel conversations

**Linear Issues**: JCB-23 through JCB-27

**Tasks**:

1. [JCB-23] Redis Session Management
2. [JCB-24] Periodic Summarization Job
3. [JCB-25] Fallback Agent for Out-of-Context Questions
4. [JCB-26] Rate Limiting & Safety
5. [JCB-27] Multi-User Parallel Testing

**Success Criteria**: Bot maintains context across multiple users for 5+ minutes

---

### MVP Final: Integration & Demo

**Goal**: Complete end-to-end system

**Linear Issue**: JCB-28

**Tasks**:

- Full system integration test
- Deploy to ngrok for demo
- Test with real Twitch stream (30+ min)
- Documentation and demo video
- Performance optimization

**Success Criteria**: All phases complete, sub-5s latency, successful demo

---

## Project Structure

```
percepta/
├── README.md                        # Project overview
├── PROJECT.md                       # This file (comprehensive docs)
├── .env.example                     # Environment variable template
├── .gitignore                       # Git ignore rules
├── requirements.txt                 # Python dependencies
├── package.json                     # Node dependencies
├── docker-compose.yml               # PostgreSQL + Redis local dev
│
├── py/                              # Python "brain" service
│   ├── main.py                      # FastAPI entry point
│   ├── config.py                    # Settings management
│   │
│   ├── database/
│   │   ├── __init__.py
│   │   ├── models.py                # SQLAlchemy models
│   │   └── connection.py            # DB connection + pgvector
│   │
│   ├── ingest/                      # Context builders
│   │   ├── __init__.py
│   │   ├── twitch.py                # EventSub WebSocket + Helix
│   │   ├── transcription.py         # Audio → text (faster-whisper)
│   │   └── metadata.py              # Channel info polling
│   │
│   ├── memory/
│   │   ├── __init__.py
│   │   ├── vector_store.py          # pgvector interface
│   │   ├── summarizer.py            # Periodic summary generation
│   │   └── redis_session.py         # User session management
│   │
│   ├── reason/
│   │   ├── __init__.py
│   │   ├── rag.py                   # RAG retrieval + LLM
│   │   └── fallback.py              # Fallback agent
│   │
│   ├── output/
│   │   ├── __init__.py
│   │   └── irc_bridge.py            # Message queue to Node
│   │
│   └── utils/
│       ├── __init__.py
│       └── embeddings.py            # OpenAI embedding utility
│
├── node/                            # Node.js "chat I/O" service
│   ├── index.js                     # Main entry
│   ├── chat.js                      # tmi.js IRC client
│   └── audio.js                     # Stream audio capture + STT
│
├── schemas/                         # Shared data models
│   ├── events.py                    # Pydantic event schemas
│   └── messages.py                  # Chat message schemas
│
└── scripts/
    ├── setup_db.sql                 # pgvector extension setup
    ├── init_twitch_oauth.js         # OAuth token generator
    └── test_chat.py                 # Local testing script
```

---

## Environment Setup

### Required Credentials

1. **Twitch Bot Account**

   - Create a Twitch account for the bot
   - Register app at [Twitch Developer Console](https://dev.twitch.tv/console)
   - Get Client ID and Client Secret
   - Generate OAuth token with scopes: `chat:read`, `chat:edit`

2. **OpenAI API Key**

   - Sign up at [OpenAI Platform](https://platform.openai.com)
   - Create API key with access to embeddings and chat models

3. **ngrok Auth Token** (for local dev)
   - Sign up at [ngrok](https://ngrok.com)
   - Get auth token from dashboard

### Environment Variables

Create a `.env` file (copy from `.env.example`):

```env
# Twitch Bot Credentials
TWITCH_CLIENT_ID=your_client_id_here
TWITCH_CLIENT_SECRET=your_client_secret_here
TWITCH_BOT_TOKEN=oauth:your_user_access_token_here
TWITCH_BOT_NAME=perceptabot
TARGET_CHANNEL=target_streamer_username

# OpenAI
OPENAI_API_KEY=sk-...

# Database
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/percepta
REDIS_URL=redis://localhost:6379

# ngrok (development)
NGROK_AUTH_TOKEN=your_ngrok_token_here

# STT Configuration
WHISPER_MODEL=base                   # tiny, base, small, medium, large
USE_GPU=false                        # Set to true if CUDA available

# Memory Configuration
TRANSCRIPT_WINDOW_MINUTES=10         # Rolling window size
SUMMARY_INTERVAL_SECONDS=30          # How often to generate summaries

# Performance
MAX_CONCURRENT_REQUESTS=10
EMBEDDING_BATCH_SIZE=10

# Logging
LOG_LEVEL=INFO                       # DEBUG, INFO, WARNING, ERROR
```

### Docker Services

Start PostgreSQL and Redis:

```bash
docker-compose up -d
```

`docker-compose.yml`:

```yaml
version: '3.8'

services:
  postgres:
    image: ankane/pgvector:latest
    environment:
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: postgres
      POSTGRES_DB: percepta
    ports:
      - '5432:5432'
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./scripts/setup_db.sql:/docker-entrypoint-initdb.d/setup.sql

  redis:
    image: redis:7-alpine
    ports:
      - '6379:6379'
    volumes:
      - redis_data:/data

volumes:
  postgres_data:
  redis_data:
```

### Initial Setup

```bash
# Install Python dependencies
pip install -r requirements.txt

# Install Node dependencies
npm install

# Start databases
docker-compose up -d

# Initialize database schema
python scripts/setup_db.py

# Generate Twitch OAuth token
node scripts/init_twitch_oauth.js

# Start ngrok (in separate terminal)
ngrok http 8000
```

---

## Development Workflow

### Local Development

1. **Start Services**

   ```bash
   # Terminal 1: Databases
   docker-compose up

   # Terminal 2: Python Backend
   cd py
   uvicorn main:app --reload --port 8000

   # Terminal 3: Node Chat Service
   cd node
   node index.js

   # Terminal 4: ngrok (if needed)
   ngrok http 8000
   ```

2. **Development Cycle**
   - Make code changes
   - Test with local Twitch channel
   - Check logs for errors
   - Iterate

### Testing Strategy

#### Unit Tests

- Test individual components (embeddings, vector store, RAG)
- Mock external APIs (OpenAI, Twitch)
- Run with `pytest`

#### Integration Tests

- Test end-to-end flows with test data
- Use `scripts/test_chat.py` for manual testing
- Use `scripts/test_rag.py` to smoke-test RAG retrieval + answer generation
- Verify latency and accuracy

#### Live Testing

- Connect to real Twitch stream
- Monitor bot behavior in chat
- Measure performance metrics
- Gather user feedback

### Deployment to Fly.io (Post-MVP)

```bash
# Install Fly CLI
curl -L https://fly.io/install.sh | sh

# Login
fly auth login

# Launch app
fly launch

# Deploy
fly deploy

# Set secrets
fly secrets set TWITCH_CLIENT_ID=xxx
fly secrets set OPENAI_API_KEY=xxx
```

---

## Performance Targets

### Latency Budget (Total: ≤5s)

| Component             | Target    | Notes                       |
| --------------------- | --------- | --------------------------- |
| Audio capture lag     | < 2s      | Behind live stream          |
| STT (faster-whisper)  | 1-2s      | For 15s audio chunk         |
| Embedding generation  | 100-400ms | OpenAI API call             |
| Vector search         | 50-200ms  | pgvector similarity search  |
| LLM answer generation | 1-2s      | GPT-4o-mini streaming       |
| IRC send              | < 200ms   | Twitch IRC latency          |
| **Total**             | **< 5s**  | Audio event → chat response |

### Optimization Strategies

1. **Parallel Processing**: Transcribe while previous chunk is being embedded
2. **Batch Embeddings**: Embed multiple chunks in single API call
3. **Hot Cache**: Keep last 5 minutes in Redis for instant access
4. **Streaming LLM**: Start sending response as soon as first tokens arrive
5. **Connection Pooling**: Reuse DB and API connections

### Monitoring Metrics

- Average response latency
- 95th percentile latency
- Transcription accuracy
- Vector search relevance
- LLM answer quality
- Error rates
- API costs (OpenAI)

---

## Safety & Moderation

### Rate Limiting

**Per-User Limits**:

- 1 question per 10 seconds
- Cooldown for repeated identical questions: 60s

**Global Limits**:

- 20 messages per 30 seconds (Twitch default)
- Verified bot status can increase to 100 msg/30s

**Implementation**: Redis with sliding window counters

### Content Filtering

**Input Filtering**:

- Ignore messages with banned words
- Ignore messages from banned users
- Length limits (max 500 chars)

**Output Filtering**:

- No toxic/offensive language in responses
- No personal information (emails, addresses, etc.)
- No political/religious content
- Keep responses neutral and helpful

### Twitch TOS Compliance

- Only access channels that opt in (via ENV config)
- Respect Twitch API rate limits
- No scraping or unauthorized data collection
- Follow Twitch Developer Agreement
- Respect user privacy (no logging personal data)

### Error Handling

**Graceful Degradation**:

- No transcripts available → Use only events + metadata
- No RAG matches → Use fallback agent
- API errors → Return friendly error message
- Database down → Queue messages, retry later

**Fallback Responses**:

- "I don't have enough context to answer that right now."
- "I can only answer questions about the current stream."
- "Sorry, I'm having trouble connecting. Try again in a moment."

### Admin Commands

- `!bot pause` - Pause bot responses
- `!bot resume` - Resume bot responses
- `!bot status` - Show bot health metrics
- `!bot clear` - Clear user session

---

## Linear Integration

### Project Tracking

**Linear Project**: [Percepta - AI Twitch Chat Bot](https://linear.app/jcbuilds/project/percepta-ai-twitch-chat-bot-ead8d9ea69f6)

**Team**: Jcbuilds

### Issue Organization

**Labels**:

- `phase-1`, `phase-2`, `phase-3`, `phase-4`, `phase-5`
- `setup`, `backend`, `frontend`, `database`, `testing`
- `bug`, `enhancement`, `documentation`

**Priorities**:

- Urgent (P1): Blocking issues, critical bugs
- High (P2): Phase tasks, important features
- Normal (P3): Nice-to-haves, optimizations
- Low (P4): Future enhancements

### Workflow

1. **Planning**: Issues created in Backlog
2. **In Progress**: Move issue when starting work
3. **In Review**: Move when ready for testing
4. **Done**: Move when complete and tested

### Issue References

All 24 MVP issues are tracked in Linear:

- Phase 1: JCB-5 through JCB-9
- Phase 2: JCB-10 through JCB-14
- Phase 3: JCB-15 through JCB-18
- Phase 4: JCB-19 through JCB-22
- Phase 5: JCB-23 through JCB-27
- MVP Final: JCB-28

### Progress Tracking

Use Linear's built-in features:

- Project progress view
- Burndown charts
- Cycle tracking
- Time estimates

---

## Next Steps

### Immediate Actions

1. ✅ Linear project and issues created
2. ✅ PROJECT.md documentation complete
3. ⏭️ Start Phase 1: [JCB-5] Project Setup & Environment Configuration

### Getting Started

```bash
# Clone repository
git clone <repo-url>
cd percepta

# Review documentation
cat PROJECT.md

# Check Linear for current sprint
# https://linear.app/jcbuilds/project/percepta-ai-twitch-chat-bot-ead8d9ea69f6

# Begin Phase 1 implementation
# Start with JCB-5: Project Setup
```

### Resources

- **Twitch API Docs**: https://dev.twitch.tv/docs/api/
- **OpenAI API Docs**: https://platform.openai.com/docs/
- **pgvector Docs**: https://github.com/pgvector/pgvector
- **faster-whisper**: https://github.com/guillaumekln/faster-whisper
- **LangGraph**: https://langchain-ai.github.io/langgraph/

---

## Questions or Issues?

- **Linear Project**: [View all issues](https://linear.app/jcbuilds/project/percepta-ai-twitch-chat-bot-ead8d9ea69f6)
- **Documentation**: This file (PROJECT.md)
- **Architecture Questions**: Review [Architecture](#architecture) section

---

**Last Updated**: 2025-10-26  
**Status**: Planning Complete, Ready for Phase 1  
**Next Milestone**: Phase 1 Complete (Chat I/O working)
