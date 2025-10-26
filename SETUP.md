# Percepta Setup Guide

## ✅ Phase 1.1 Complete: Project Setup & Environment Configuration

All foundational project structure and configuration files have been created successfully.

### Directory Structure Created

```
percepta/
├── py/                     # Python backend service
│   ├── database/          # SQLAlchemy models & pgvector
│   ├── ingest/            # Transcription & event processing
│   ├── memory/            # Vector store & sessions
│   ├── reason/            # RAG & LLM
│   ├── output/            # IRC bridge
│   └── utils/             # Embeddings & helpers
├── node/                  # Node.js chat I/O service
├── schemas/               # Shared data models
└── scripts/               # Setup scripts
```

### Configuration Files Created

- ✅ `.env.example` - Environment variable template
- ✅ `requirements.txt` - Python dependencies (FastAPI, LangGraph, faster-whisper, etc.)
- ✅ `package.json` - Node dependencies (tmi.js, ffmpeg, axios)
- ✅ `docker-compose.yml` - PostgreSQL + Redis containers
- ✅ `.gitignore` - Git ignore patterns
- ✅ `scripts/setup_db.sql` - pgvector initialization script

## Next Steps: Setting Up the Environment

### Prerequisites

Before proceeding with Phase 1.2, you'll need to install:

1. **Docker Desktop** (for PostgreSQL + Redis)

   - Download: https://www.docker.com/products/docker-desktop
   - Required for `docker-compose.yml` to work

2. **Python 3.10+**

   - Verify: `python --version`

3. **Node.js 18+**
   - Verify: `node --version`

### Installation Steps

1. **Install Python dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

2. **Install Node dependencies:**

   ```bash
   npm install
   ```

3. **Start Docker services:**

   ```bash
   # After installing Docker Desktop
   docker compose up -d
   ```

4. **Create your environment file:**
   ```bash
   cp .env.example .env
   ```
   Then edit `.env` with your actual credentials.

### What's Next?

**Phase 1.3: JCB-7 - Node IRC Chat Service**

This involves:

- Building the Node.js IRC client with tmi.js
- Connecting to Twitch chat
- Reading messages from chat
- Sending responses back to chat

## Verification

To verify this phase is complete, confirm:

- [x] All directories exist with `__init__.py` files
- [x] Configuration files are present
- [x] `requirements.txt` contains all Python dependencies
- [x] `package.json` contains all Node dependencies
- [x] `docker-compose.yml` is configured for PostgreSQL + Redis
- [x] `.gitignore` covers Python, Node, and environment files

## Notes

- ✅ **Docker is installed and verified!** Docker version 28.5.1 with Compose v2.40.2
- ✅ **Database containers tested and working:**
  - PostgreSQL 15.4 with pgvector v0.5.1 extension ✓
  - Redis 7 Alpine ✓
- All environment variables are documented in `.env.example`.
- The project structure follows the architecture defined in `PROJECT.md`.

## Issues Completed

- ✅ **JCB-5**: Phase 1.1: Project Setup & Environment Configuration
- ✅ **JCB-6**: Phase 1.2: Twitch Bot Account & OAuth Setup
