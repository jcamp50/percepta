"""
Integration test fixtures for database and HTTP client setup.
"""
import asyncio
import os
import pytest
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy import text

from py.database.models import Base
from py.database.connection import DATABASE_URL


# Override DATABASE_URL for integration tests
# Default to port 5432 (docker-compose) but allow override via env var
TEST_DATABASE_URL = os.getenv(
    "TEST_DATABASE_URL",
    "postgresql+asyncpg://postgres:postgres@localhost:5432/percepta_test"
)


def _check_database_available():
    """Check if test database is available."""
    try:
        import asyncpg
        # Try to connect synchronously (quick check)
        # Note: asyncpg doesn't have a sync connect, so we'll check in the async fixture
        return True
    except ImportError:
        return False


# Use function scope for async fixtures to avoid event loop issues
# Session-scoped async fixtures can cause problems with pytest-asyncio


@pytest.fixture(scope="function")
async def test_engine():
    """Create a test database engine.
    
    Skips tests if database is not available.
    """
    import asyncpg
    from urllib.parse import urlparse
    
    # Parse database URL to extract connection info
    parsed = urlparse(TEST_DATABASE_URL.replace("postgresql+asyncpg://", "postgresql://"))
    db_name = parsed.path.lstrip("/")
    host = parsed.hostname or "localhost"
    port = parsed.port or 5432
    user = parsed.username or "postgres"
    password = parsed.password or "postgres"
    
    # Check if database server is available by connecting to default 'postgres' database
    try:
        conn = await asyncio.wait_for(
            asyncpg.connect(
                host=host,
                port=port,
                user=user,
                password=password,
                database="postgres",  # Connect to default DB first
            ),
            timeout=2.0
        )
        await conn.close()
    except (asyncio.TimeoutError, ConnectionRefusedError, OSError, Exception) as e:
        pytest.skip(f"Test database server not available at {host}:{port}: {e}")
    
    # Try to create test database if it doesn't exist (only once per session)
    # Use a module-level flag to track if we've checked
    if not hasattr(test_engine, '_db_checked'):
        try:
            admin_conn = await asyncpg.connect(
                host=host,
                port=port,
                user=user,
                password=password,
                database="postgres",
            )
            # Check if test database exists
            db_exists = await admin_conn.fetchval(
                "SELECT 1 FROM pg_database WHERE datname = $1", db_name
            )
            if not db_exists:
                await admin_conn.execute(f'CREATE DATABASE "{db_name}"')
            await admin_conn.close()
            test_engine._db_checked = True
        except Exception as e:
            # If we can't create DB, try to continue anyway (might already exist)
            test_engine._db_checked = True
            pass
    
    # Create engine for test database
    engine = create_async_engine(
        TEST_DATABASE_URL,
        pool_pre_ping=True,
        echo=False,  # Set to True for SQL debugging
    )
    
    try:
        # Ensure pgvector extension exists
        async with engine.begin() as conn:
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        
        # Create all tables
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
    except Exception as e:
        await engine.dispose()
        pytest.skip(f"Failed to set up test database: {e}")
    
    yield engine
    
    # Don't drop tables between tests - keep them for faster test runs
    # Tables will be cleaned up manually or by dropping the test database
    # This allows tests to share data if needed
    await engine.dispose()


@pytest.fixture
async def db_session(test_engine: AsyncEngine):
    """Create a database session for a test."""
    async_session = async_sessionmaker(
        test_engine,
        expire_on_commit=False,
        class_=AsyncSession,
    )
    
    async with async_session() as session:
        yield session
        # Rollback any uncommitted changes
        await session.rollback()


@pytest.fixture
def sample_embedding_1536():
    """Generate a sample 1536-dimensional embedding vector."""
    import numpy as np
    # Generate a random normalized vector
    vec = np.random.randn(1536).astype(np.float32)
    vec = vec / np.linalg.norm(vec)
    return vec.tolist()


@pytest.fixture
def sample_channel_id():
    """Sample broadcaster ID for tests."""
    return "123456789"


@pytest.fixture
def sample_timestamp():
    """Sample timestamp for tests."""
    from datetime import datetime, timezone
    return datetime.now(timezone.utc)

