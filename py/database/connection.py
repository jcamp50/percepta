from typing import AsyncIterator
import os

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)


DEFAULT_DB_URL = "postgresql+asyncpg://postgres:postgres@localhost:5432/percepta"
DATABASE_URL = os.getenv("DATABASE_URL", DEFAULT_DB_URL)


engine: AsyncEngine = create_async_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    connect_args={
        "server_settings": {
            "timezone": "UTC"  # Set session timezone to UTC for consistent display
        }
    },
)

SessionLocal = async_sessionmaker(
    engine,
    expire_on_commit=False,
    class_=AsyncSession,
)


async def get_session() -> AsyncIterator[AsyncSession]:
    async with SessionLocal() as session:
        yield session
