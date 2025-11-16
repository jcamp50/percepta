"""
Integration tests for pgvector schema and indexes.

Tests that pgvector extension is enabled and indexes exist.
"""
import pytest
from sqlalchemy import text, inspect


@pytest.mark.integration
@pytest.mark.asyncio
async def test_pgvector_extension_exists(test_engine):
    """Test that pgvector extension is enabled."""
    async with test_engine.begin() as conn:
        result = await conn.execute(
            text("SELECT extname, extversion FROM pg_extension WHERE extname = 'vector'")
        )
        row = result.first()
        
        assert row is not None, "pgvector extension not found"
        assert row.extname == "vector"
        assert row.extversion is not None


@pytest.mark.integration
@pytest.mark.asyncio
async def test_vector_indexes_exist(test_engine):
    """Test that vector indexes exist on embedding columns."""
    async with test_engine.begin() as conn:
        # Check for ivfflat indexes on embedding columns
        result = await conn.execute(
            text("""
                SELECT indexname, indexdef 
                FROM pg_indexes 
                WHERE schemaname = 'public' 
                AND indexdef LIKE '%ivfflat%'
                AND indexdef LIKE '%embedding%'
            """)
        )
        indexes = result.fetchall()
        
        # Should have at least one ivfflat index
        assert len(indexes) > 0, "No ivfflat indexes found on embedding columns"
        
        # Verify indexes exist for key tables
        index_names = [idx.indexname for idx in indexes]
        assert any("transcripts_embedding" in name for name in index_names) or \
               any("idx_transcripts_embedding" in name for name in index_names)
        assert any("events_embedding" in name for name in index_names) or \
               any("idx_events_embedding" in name for name in index_names)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_vector_cosine_ops_available(test_engine):
    """Test that vector cosine operators are available."""
    async with test_engine.begin() as conn:
        # Test cosine distance operator (<->)
        result = await conn.execute(
            text("SELECT '[1,2,3]'::vector <-> '[4,5,6]'::vector as distance")
        )
        distance = result.scalar()
        
        assert distance is not None
        assert isinstance(distance, float)
        assert distance >= 0  # Distance should be non-negative

