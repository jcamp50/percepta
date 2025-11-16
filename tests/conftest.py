import asyncio
import os
import pytest


@pytest.fixture(scope="session")
def event_loop():
    """Create an event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(autouse=True)
def settings_env(monkeypatch):
    """Set test environment variables."""
    # Use a test database or mocks during unit tests
    monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://postgres:postgres@localhost:5433/percepta_test")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-for-unit-tests")
    yield

