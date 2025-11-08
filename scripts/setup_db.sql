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

-- Chat messages: viewer messages from Twitch chat
CREATE TABLE IF NOT EXISTS chat_messages (
  id UUID PRIMARY KEY,
  channel_id VARCHAR(255) NOT NULL,
  username VARCHAR(255) NOT NULL,
  message TEXT NOT NULL,
  sent_at TIMESTAMPTZ NOT NULL,
  embedding VECTOR(1536) NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chat_messages_channel_sent
  ON chat_messages (channel_id, sent_at DESC);
CREATE INDEX IF NOT EXISTS idx_chat_messages_embedding
  ON chat_messages USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- Migration: Add temporal alignment fields to video_frames (JCB-33)
-- These columns link video frames to aligned transcripts, chat messages, and metadata
DO $$
BEGIN
  -- Add transcript_id foreign key column
  IF NOT EXISTS (
    SELECT 1 FROM information_schema.columns 
    WHERE table_name = 'video_frames' AND column_name = 'transcript_id'
  ) THEN
    ALTER TABLE video_frames 
    ADD COLUMN transcript_id UUID REFERENCES transcripts(id);
  END IF;

  -- Add aligned_chat_ids array column (VARCHAR[] to match SQLAlchemy model)
  IF NOT EXISTS (
    SELECT 1 FROM information_schema.columns 
    WHERE table_name = 'video_frames' AND column_name = 'aligned_chat_ids'
  ) THEN
    ALTER TABLE video_frames 
    ADD COLUMN aligned_chat_ids VARCHAR[];
  END IF;

  -- Fix existing column if it was created as UUID[] (migration fix)
  IF EXISTS (
    SELECT 1 FROM information_schema.columns 
    WHERE table_name = 'video_frames' 
    AND column_name = 'aligned_chat_ids'
    AND udt_name = '_uuid'
  ) THEN
    -- Change column type from UUID[] to VARCHAR[]
    ALTER TABLE video_frames 
    ALTER COLUMN aligned_chat_ids TYPE VARCHAR[] USING aligned_chat_ids::text[];
  END IF;

  -- Add metadata_snapshot JSONB column
  IF NOT EXISTS (
    SELECT 1 FROM information_schema.columns 
    WHERE table_name = 'video_frames' AND column_name = 'metadata_snapshot'
  ) THEN
    ALTER TABLE video_frames 
    ADD COLUMN metadata_snapshot JSONB;
  END IF;

  -- Add description column
  IF NOT EXISTS (
    SELECT 1 FROM information_schema.columns 
    WHERE table_name = 'video_frames' AND column_name = 'description'
  ) THEN
    ALTER TABLE video_frames
    ADD COLUMN description TEXT;
  END IF;

  -- Add description_source column
  IF NOT EXISTS (
    SELECT 1 FROM information_schema.columns 
    WHERE table_name = 'video_frames' AND column_name = 'description_source'
  ) THEN
    ALTER TABLE video_frames
    ADD COLUMN description_source VARCHAR(50);
  END IF;

  -- Add frame_hash column
  IF NOT EXISTS (
    SELECT 1 FROM information_schema.columns 
    WHERE table_name = 'video_frames' AND column_name = 'frame_hash'
  ) THEN
    ALTER TABLE video_frames
    ADD COLUMN frame_hash VARCHAR(64);
  END IF;

  -- Create indexes for new columns if they do not exist
  IF NOT EXISTS (
    SELECT 1 FROM pg_class WHERE relname = 'idx_video_frames_hash'
  ) THEN
    CREATE INDEX idx_video_frames_hash ON video_frames(frame_hash);
  END IF;

  IF NOT EXISTS (
    SELECT 1 FROM pg_class WHERE relname = 'idx_video_frames_description_source'
  ) THEN
    CREATE INDEX idx_video_frames_description_source 
      ON video_frames(channel_id, description_source);
  END IF;
END $$;

