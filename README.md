# Percepta - AI Twitch Chat Bot

> Real-time contextual Q&A for Twitch streams powered by RAG and live transcription

[![Phase 1](https://img.shields.io/badge/Phase%201-Complete-success)](https://linear.app/jcbuilds/project/percepta-ai-twitch-chat-bot-ead8d9ea69f6)
[![Phase 2](https://img.shields.io/badge/Phase%202-Complete-success)](https://linear.app/jcbuilds/project/percepta-ai-twitch-chat-bot-ead8d9ea69f6)
[![Phase 3](https://img.shields.io/badge/Phase%203-Complete-success)](https://linear.app/jcbuilds/project/percepta-ai-twitch-chat-bot-ead8d9ea69f6)
[![Phase 4](https://img.shields.io/badge/Phase%204-Complete-success)](https://linear.app/jcbuilds/project/percepta-ai-twitch-chat-bot-ead8d9ea69f6)
[![Phase 5](https://img.shields.io/badge/Phase%205-Complete-success)](https://linear.app/jcbuilds/project/percepta-ai-twitch-chat-bot-ead8d9ea69f6)
[![Phase 6](https://img.shields.io/badge/Phase%206-Complete-success)](https://linear.app/jcbuilds/project/percepta-ai-twitch-chat-bot-ead8d9ea69f6)
[![Linear](https://img.shields.io/badge/Linear-Project-blue)](https://linear.app/jcbuilds/project/percepta-ai-twitch-chat-bot-ead8d9ea69f6)

## What is Percepta?

Percepta is an AI-powered Twitch chat bot that provides real-time, contextual answers about what's happening in a live stream. It listens to chat, transcribes stream audio, captures video frames, tracks channel events, stores real-time event data, and uses RAG (Retrieval-Augmented Generation) to answer viewer questions with grounded, timestamped responses using multi-modal context (transcripts, video, chat, events, summaries, metadata).

**Current Status:** Phase 6 Complete ✅ - Video Understanding + MVP Memory Integration | Multi-modal AI assistant with visual understanding and long-term memory

## Quick Start (Phase 1)

### Prerequisites

- **Docker Desktop** - For PostgreSQL + Redis (future phases)
- **Python 3.10+** - Backend service
- **Node.js 18+** - Chat I/O service
- **Twitch Account** - For the bot

### Installation

```bash
# 1. Clone repository
git clone <repo-url>
cd percepta

# 2. Install dependencies
pip install -r requirements.txt
npm install

# 3. Set up environment variables
cp .env.example .env
# Edit .env with your Twitch credentials (see Setup Guide)

# 4. Generate Twitch OAuth token
node scripts/init_twitch_oauth.js
```

### Running the Bot

```bash
# Terminal 1: Start Python backend
uvicorn py.main:app --reload --port 8000

# Terminal 2: Start Node chat service
node node/index.js
```

The bot will connect to your configured Twitch channel. Test it by typing:

```
@yourbotname ping
```

The bot should respond: `@yourusername pong`

## Project Status

### ✅ Phase 1: Chat I/O (Complete)

- [x] Project setup & environment configuration
- [x] Twitch bot account & OAuth setup
- [x] Node IRC chat service (tmi.js)
- [x] Python FastAPI service stub
- [x] End-to-end chat flow (ping/pong)

### ✅ Phase 2: Vector Store + RAG (Complete)

- [x] PostgreSQL + pgvector setup
- [x] SQLAlchemy models & vector store interface
- [x] OpenAI embedding utility
- [x] RAG retrieval pipeline
- [x] Manual testing with sample data

### ✅ Phase 3: Transcription Pipeline (Complete)

- [x] Twitch stream audio capture (Streamlink integration)
- [x] Audio chunking with FFmpeg (15-second segments)
- [x] faster-whisper integration for speech-to-text
- [x] Audio-to-transcript pipeline (transcribe → embed → store)
- [x] Full pipeline testing with live streams
- [x] Windows file locking handling
- [x] Stream offline/online detection

### ✅ Phase 4: Enhanced Context (Complete)

- [x] Channel metadata polling (game, title, viewer count)
- [x] EventSub WebSocket integration (real-time events)
- [x] Event storage & embedding (raids, subscriptions, stream online/offline)
- [x] Combined retrieval (transcripts + events with unified ranking)
- [x] Enhanced prompts with few-shot examples
- [x] Metadata integration in RAG responses
- [x] Comprehensive testing (84% success rate across question types)

### ✅ Phase 5: Summarization + Multi-User (Complete)

- [x] Redis session management for parallel conversations
- [x] Periodic summarization job for long-term context
- [x] Fallback agent for out-of-context questions
- [x] Rate limiting & safety measures
- [x] Multi-user parallel testing

### ✅ Phase 6: Video Understanding + MVP Memory Integration (Complete)

- [x] Video frame embeddings with CLIP (512→1536 projection)
- [x] Chat message embeddings with temporal alignment
- [x] Temporal alignment linking frames to transcripts, chat, and metadata
- [x] Visual description generation pipeline (rich JSON descriptions)
- [x] Adaptive frame capture (5-10s intervals) with interesting frame detection
- [x] Grounded embeddings (70% CLIP + 30% description) for improved text queries
- [x] Memory-propagated summarization (2-minute segments) for long-term context
- [x] Multi-modal retrieval across all sources (transcripts, video, chat, events, summaries, metadata)

## Architecture

```
┌─────────────────────────────────────────────────────┐
│              TWITCH STREAM                           │
│    (Audio + Video + Chat + Events)                    │
└────────────┬────────────────────────┬────────────────┘
             │                        │
    ┌────────▼────────┐      ┌────────▼────────┐
    │  Node Service   │      │  Twitch APIs    │
    │  - IRC (tmi.js) │      │  - Helix API    │
    │  - Audio Capture│      │  - EventSub WS  │
    │    (FFmpeg)     │      │                  │
    │  - Video Capture│      │                  │
    │    (Screenshots)│      │                  │
    └────────┬────────┘      └────────┬────────┘
             │                        │
    ┌────────▼────────────────────────▼────────┐
    │    Python Backend (FastAPI)              │
    │  - Streamlink (audio URL extraction)     │
    │  - Transcription (faster-whisper)       │
    │  - Video Frame Processing (CLIP)         │
    │  - Visual Description Generation        │
    │  - Vector Store (pgvector)               │
    │  - Multi-Modal RAG Retrieval             │
    │  - Memory-Propagated Summarization      │
    │  - Response Generation                   │
    └─────────────────┬────────────────────────┘
                      │
              ┌───────▼────────┐
              │  Twitch Chat   │
              │   (Response)   │
              └────────────────┘
```

**Phase 1-6 Implementation:**

- **Chat Flow**: Node.js forwards ALL chat messages to Python `/chat/message`, stores with embeddings (JCB-32)
- **Audio Flow**:
  - Node.js requests authenticated audio-only stream URLs from Python `/api/get-audio-stream-url` (uses Streamlink)
  - Node.js captures audio with FFmpeg, chunks into 15-second segments
  - Node.js sends audio chunks to Python `/transcribe` endpoint
  - Python transcribes with faster-whisper, generates embeddings, stores in pgvector
- **Video Flow** (Phase 6):
  - Node.js captures video screenshots at adaptive 5-10s intervals (JCB-42)
  - Node.js sends frames to Python `/api/video-frame` endpoint
  - Python generates CLIP embeddings, visual descriptions (JCB-41), and grounded embeddings (JCB-37)
  - Python performs temporal alignment with transcripts, chat, and metadata (JCB-33)
- **Metadata Flow**: Python polls Twitch Helix API every 60 seconds for channel/stream metadata, stores snapshots
- **Event Flow**: Python connects to EventSub WebSocket, receives real-time events (raids, subscriptions, stream online/offline), generates summaries, stores with embeddings
- **Memory Flow** (Phase 6):
  - Python generates 2-minute segment summaries with memory propagation (JCB-35)
  - Summaries include visual context, chat highlights, and streamer commentary
- **Multi-Modal RAG Flow**: Python retrieves from all sources (transcripts, video frames, chat, events, summaries, metadata), enriches with temporal alignment, generates contextual answers with multi-modal context
- **Response Flow**: Node polls Python `/chat/send` every 500ms, sends responses to Twitch chat

## Technology Stack

**Python Backend:**

- FastAPI - Web framework
- Pydantic - Data validation
- Streamlink - Twitch audio-only stream URL extraction
- faster-whisper - Speech-to-text transcription
- CLIP (transformers) - Video frame embeddings
- pgvector - Vector similarity search
- OpenAI API - Text embeddings, GPT-4o-mini (reasoning), GPT-4o-mini Vision (descriptions)
- Redis - Session management

**Node.js Services:**

- tmi.js - Twitch IRC client
- axios - HTTP client
- fluent-ffmpeg - Audio capture and chunking
- Screenshot capture - Video frame capture (adaptive 5-10s intervals)

**Infrastructure:**

- PostgreSQL + pgvector - Vector database for embeddings
- Redis - Session state and rate limiting
- Docker Compose - Local development environment

## Documentation

- **[Architecture & Planning](docs/ARCHITECTURE.md)** - Comprehensive technical documentation
- **[Phase 6 Completion Summary](docs/PHASE_6_COMPLETE.md)** - Video understanding & memory integration
- **[Video Frame Implementation Roadmap](docs/VIDEO_FRAME_IMPLEMENTATION_ROADMAP.md)** - Video understanding details
- **[Context Layer Expansion](docs/CONTEXT_LAYER_EXPANSION.md)** - Multi-modal context implementation
- **[Detailed Setup Guide](docs/DETAILED_SETUP.md)** - Step-by-step installation
- **[Twitch Setup](docs/TWITCH_SETUP.md)** - Creating bot account
- **[OAuth Quickstart](docs/OAUTH_QUICKSTART.md)** - Generating tokens

## Development

### Project Structure

```
percepta/
├── py/                    # Python backend service
│   ├── main.py           # FastAPI app
│   ├── config.py         # Settings management
│   └── [future modules]  # database/, memory/, reason/
├── node/                 # Node.js chat I/O service
│   ├── index.js          # Entry point
│   ├── chat.js           # IRC client wrapper
│   └── utils/            # Logging utilities
├── schemas/              # Pydantic data models
│   └── messages.py       # Chat message schemas
├── scripts/              # Setup & testing scripts
└── docs/                 # Documentation
```

### Running Tests

```bash
# Test Twitch authentication
node scripts/test_twitch_auth.js

# Test Python API (with service running)
curl http://localhost:8000/health
```

### Environment Variables

See `.env.example` for all available configuration options. Key variables:

```env
# Twitch Bot Credentials
TWITCH_CLIENT_ID=your_client_id
TWITCH_CLIENT_SECRET=your_client_secret
TWITCH_BOT_TOKEN=oauth:your_token
TWITCH_BOT_NAME=your_bot_name
TARGET_CHANNEL=channel_to_join

# Python Service
PYTHON_SERVICE_URL=http://localhost:8000  # (optional, defaults to this)
LOG_LEVEL=INFO

# Audio Capture (Phase 3)
AUDIO_CHUNK_SECONDS=15      # Duration of each audio chunk in seconds
AUDIO_SAMPLE_RATE=16000     # Audio sample rate (Hz) - FFmpeg resamples from 48kHz
AUDIO_CHANNELS=1            # Audio channels (1 = mono, 2 = stereo)

# Whisper Transcription (Phase 3)
WHISPER_MODEL=base          # tiny, base, small, medium, large
WHISPER_LANGUAGE=en         # or empty for auto-detect
USE_GPU=false               # Set to true if CUDA available

# Channel Metadata Polling (Phase 4 - JCB-19)
METADATA_POLL_INTERVAL_SECONDS=60  # How often to poll for channel metadata

# EventSub WebSocket (Phase 4 - JCB-20)
EVENTSUB_ENABLED=true              # Enable EventSub WebSocket client
TARGET_CHANNEL=plaqueboymax        # Channel to subscribe to events for
TWITCH_CLIENT_ID=your_client_id    # Twitch API client ID
TWITCH_BOT_TOKEN=your_bot_token    # Bot OAuth token for EventSub

# Note: Streamlink is used for getting authenticated Twitch audio-only stream URLs
# No additional configuration needed - handled automatically by Python service
```

## Performance Targets

**Phase 1 Achieved:**

- Response latency: < 1 second
- Message forwarding: Non-blocking (fire-and-forget)
- Polling frequency: 500ms intervals

**Future Goals (Phase 3+):**

- Total latency (audio → response): < 5 seconds
- Transcription: 1-2s per 15s audio chunk
- LLM generation: 1-2s (GPT-4o-mini)

## Contributing

This is currently a personal learning project. Feedback and suggestions welcome!

## Linear Project

Track progress and issues: [Percepta - AI Twitch Chat Bot](https://linear.app/jcbuilds/project/percepta-ai-twitch-chat-bot-ead8d9ea69f6)

---

**Last Updated**: 2025-11-15  
**Current Phase**: Phase 6 Complete ✅ - Video Understanding + MVP Memory Integration  
**Completed Issues**: JCB-31, JCB-32, JCB-33, JCB-35, JCB-37, JCB-41, JCB-42  
**Next Milestone**: Post-MVP enhancements (JCB-34, JCB-36, JCB-38, JCB-39, JCB-40)
