"""
Integration tests for pgvector database operations.

Tests vector insertion and cosine similarity search using real Postgres + pgvector.
"""
import pytest
from datetime import datetime, timezone
from sqlalchemy import select, text
from pgvector.sqlalchemy import Vector

from py.database.models import Transcript, Event


@pytest.mark.integration
@pytest.mark.asyncio
async def test_insert_transcript_with_vector(db_session, sample_embedding_1536, sample_channel_id, sample_timestamp):
    """Test inserting a Transcript with a 1536-dim vector."""
    transcript = Transcript(
        channel_id=sample_channel_id,
        started_at=sample_timestamp,
        ended_at=sample_timestamp,
        text="Test transcript text",
        embedding=sample_embedding_1536,
    )
    
    db_session.add(transcript)
    await db_session.commit()
    await db_session.refresh(transcript)
    
    assert transcript.id is not None
    assert transcript.channel_id == sample_channel_id
    assert transcript.text == "Test transcript text"
    assert len(transcript.embedding) == 1536


@pytest.mark.integration
@pytest.mark.asyncio
async def test_insert_event_with_vector(db_session, sample_embedding_1536, sample_channel_id, sample_timestamp):
    """Test inserting an Event with a 1536-dim vector."""
    event = Event(
        channel_id=sample_channel_id,
        ts=sample_timestamp,
        type="test_event",
        summary="Test event summary",
        embedding=sample_embedding_1536,
    )
    
    db_session.add(event)
    await db_session.commit()
    await db_session.refresh(event)
    
    assert event.id is not None
    assert event.channel_id == sample_channel_id
    assert event.type == "test_event"
    assert event.embedding is not None
    assert len(event.embedding) == 1536


@pytest.mark.integration
@pytest.mark.asyncio
async def test_cosine_similarity_search(db_session, sample_channel_id, sample_timestamp):
    """Test cosine similarity search using pgvector operators."""
    import numpy as np
    
    # Create two similar vectors and one different vector
    base_vec = np.random.randn(1536).astype(np.float32)
    base_vec = base_vec / np.linalg.norm(base_vec)
    
    # Similar vector (slight variation)
    similar_vec = base_vec + np.random.randn(1536).astype(np.float32) * 0.1
    similar_vec = similar_vec / np.linalg.norm(similar_vec)
    
    # Different vector
    different_vec = np.random.randn(1536).astype(np.float32)
    different_vec = different_vec / np.linalg.norm(different_vec)
    
    # Insert transcripts
    transcript1 = Transcript(
        channel_id=sample_channel_id,
        started_at=sample_timestamp,
        ended_at=sample_timestamp,
        text="Similar text",
        embedding=base_vec.tolist(),
    )
    transcript2 = Transcript(
        channel_id=sample_channel_id,
        started_at=sample_timestamp,
        ended_at=sample_timestamp,
        text="Also similar text",
        embedding=similar_vec.tolist(),
    )
    transcript3 = Transcript(
        channel_id=sample_channel_id,
        started_at=sample_timestamp,
        ended_at=sample_timestamp,
        text="Different text",
        embedding=different_vec.tolist(),
    )
    
    db_session.add_all([transcript1, transcript2, transcript3])
    await db_session.commit()
    
    # Refresh objects to ensure they have IDs
    await db_session.refresh(transcript1)
    await db_session.refresh(transcript2)
    await db_session.refresh(transcript3)
    
    # Search for similar vectors using cosine distance
    # Use raw SQL like the rest of the codebase (pgvector SQLAlchemy integration)
    from sqlalchemy import text
    import json
    
    query_vec_str = "[" + ",".join(f"{v:.8f}" for v in base_vec.tolist()) + "]"
    
    # Use pgvector cosine distance operator (<=>) via raw SQL
    sql = text("""
        SELECT id, channel_id, started_at, ended_at, text, embedding
        FROM transcripts
        WHERE channel_id = :channel_id
        ORDER BY embedding <=> (:vec)::vector
        LIMIT 2
    """)
    
    result = await db_session.execute(
        sql,
        {"channel_id": sample_channel_id, "vec": query_vec_str}
    )
    results = result.fetchall()
    
    # Should find at least 2 results (the ones we just inserted)
    assert len(results) >= 2, f"Expected at least 2 results, got {len(results)}"
    
    # Get the IDs from results
    result_ids = [row[0] for row in results]
    
    # First result should be transcript1 (identical vector)
    assert transcript1.id in result_ids
    # Second result should be transcript2 (similar vector)  
    assert transcript2.id in result_ids
    # transcript3 should not be in top 2
    assert transcript3.id not in result_ids[:2]


@pytest.mark.integration
@pytest.mark.asyncio
async def test_vector_dimension_validation(db_session, sample_channel_id, sample_timestamp):
    """Test that vectors must be exactly 1536 dimensions."""
    import numpy as np
    
    # Try to insert with wrong dimension (should fail at DB level)
    wrong_dim_vec = np.random.randn(512).astype(np.float32).tolist()
    
    transcript = Transcript(
        channel_id=sample_channel_id,
        started_at=sample_timestamp,
        ended_at=sample_timestamp,
        text="Test",
        embedding=wrong_dim_vec,
    )
    
    db_session.add(transcript)
    
    # Should raise an error when committing
    with pytest.raises(Exception):  # SQLAlchemy or Postgres error
        await db_session.commit()

