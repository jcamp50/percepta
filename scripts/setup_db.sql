-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Verify extension is enabled
SELECT extname, extversion FROM pg_extension WHERE extname = 'vector';

-- Create initial schema (tables will be created by SQLAlchemy migrations later)
-- This script ensures the pgvector extension is available

-- Transcripts: short-term rolling window
CREATE TABLE IF NOT EXISTS transcripts (
  id UUID PRIMARY KEY,
  channel_id VARCHAR(255) NOT NULL,
  started_at TIMESTAMPTZ NOT NULL,
  ended_at TIMESTAMPTZ NOT NULL,
  text TEXT NOT NULL,
  embedding VECTOR(1536) NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_transcripts_channel_started
  ON transcripts (channel_id, started_at DESC);
CREATE INDEX IF NOT EXISTS idx_transcripts_embedding
  ON transcripts USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- Events: raids, subs, polls, etc.
CREATE TABLE IF NOT EXISTS events (
  id UUID PRIMARY KEY,
  channel_id VARCHAR(255) NOT NULL,
  ts TIMESTAMPTZ NOT NULL,
  type VARCHAR(100) NOT NULL,
  summary TEXT,
  payload_json JSONB,
  embedding VECTOR(1536)
);
CREATE INDEX IF NOT EXISTS idx_events_channel_ts
  ON events (channel_id, ts DESC);
CREATE INDEX IF NOT EXISTS idx_events_embedding
  ON events USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- Channel snapshots: title, game, tags, viewer count, etc.
CREATE TABLE IF NOT EXISTS channel_snapshots (
  id UUID PRIMARY KEY,
  channel_id VARCHAR(255) NOT NULL,
  ts TIMESTAMPTZ NOT NULL,
  title TEXT,
  game_id VARCHAR(50),
  game_name VARCHAR(255),
  tags TEXT[],
  viewer_count INT,
  payload_json JSONB,
  embedding VECTOR(1536)
);
CREATE INDEX IF NOT EXISTS idx_snapshots_channel_ts
  ON channel_snapshots (channel_id, ts DESC);
CREATE INDEX IF NOT EXISTS idx_snapshots_embedding
  ON channel_snapshots USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- Video frames: screenshots from Twitch streams
CREATE TABLE IF NOT EXISTS video_frames (
  id UUID PRIMARY KEY,
  channel_id VARCHAR(255) NOT NULL,
  captured_at TIMESTAMPTZ NOT NULL,
  image_path TEXT NOT NULL,
  embedding VECTOR(1536) NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_video_frames_channel_captured
  ON video_frames (channel_id, captured_at DESC);
CREATE INDEX IF NOT EXISTS idx_video_frames_embedding
  ON video_frames USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

