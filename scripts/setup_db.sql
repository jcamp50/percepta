-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Verify extension is enabled
SELECT extname, extversion FROM pg_extension WHERE extname = 'vector';

-- Create initial schema (tables will be created by SQLAlchemy migrations later)
-- This script ensures the pgvector extension is available

