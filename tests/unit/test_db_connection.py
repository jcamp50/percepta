"""Unit tests for database connection."""
import os
import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from py.database.connection import get_session, DATABASE_URL, engine, SessionLocal


@pytest.mark.unit
@pytest.mark.asyncio
class TestDatabaseConnection:
    """Test database connection utilities."""

    async def test_get_session_yields_async_session(self):
        """Test that get_session yields an AsyncSession."""
        async for session in get_session():
            assert isinstance(session, AsyncSession)
            break  # Only need to check first iteration

    async def test_get_session_is_context_manager(self):
        """Test that get_session can be used as async context manager."""
        async for session in get_session():
            # Session should be usable
            assert hasattr(session, 'add')
            assert hasattr(session, 'commit')
            assert hasattr(session, 'rollback')
            break

    def test_database_url_env_override(self, monkeypatch):
        """Test that DATABASE_URL can be overridden via environment."""
        custom_url = "postgresql+asyncpg://custom:pass@localhost:5432/testdb"
        monkeypatch.setenv("DATABASE_URL", custom_url)
        
        # Re-import to pick up new env var
        import importlib
        import py.database.connection as conn_module
        importlib.reload(conn_module)
        
        # Note: In real usage, the module is loaded once, so this test
        # verifies the pattern works. For actual testing, we'd use conftest
        # to set env vars before import.
        assert conn_module.DATABASE_URL == custom_url
        
        # Restore
        monkeypatch.delenv("DATABASE_URL", raising=False)

    def test_engine_creation(self):
        """Test that engine is created with correct configuration."""
        assert engine is not None
        assert engine.pool is not None
        # Engine should be async
        assert hasattr(engine, 'connect')

    def test_session_local_factory(self):
        """Test that SessionLocal is a session factory."""
        assert SessionLocal is not None
        # Should be callable (factory)
        assert callable(SessionLocal)

