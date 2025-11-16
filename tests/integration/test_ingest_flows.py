"""
Integration tests for ingest flows.

Tests EventHandler/VectorStore path with mocked Twitch/LLM APIs but real database.
"""
import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

from py.ingest.event_handler import EventHandler
from py.memory.vector_store import VectorStore
from py.schemas.events import StreamOnlineEvent, StreamOfflineEvent, ChannelRaidEvent


@pytest.mark.integration
@pytest.mark.asyncio
async def test_event_handler_stream_online(test_engine, sample_channel_id):
    """Test EventHandler processes stream.online event and stores in DB."""
    from sqlalchemy.ext.asyncio import async_sessionmaker, AsyncSession
    session_factory = async_sessionmaker(test_engine, expire_on_commit=False, class_=AsyncSession)
    vector_store = VectorStore(session_factory=session_factory)
    handler = EventHandler(vector_store)
    
    event = StreamOnlineEvent(
        broadcaster_user_id=sample_channel_id,
        broadcaster_user_login="testchannel",
        broadcaster_user_name="TestChannel",
        started_at=datetime.now(timezone.utc),
    )
    
    payload_json = {
        "subscription": {"type": "stream.online"},
        "event": {
            "broadcaster_user_id": sample_channel_id,
            "broadcaster_user_login": "testchannel",
            "broadcaster_user_name": "TestChannel",
            "started_at": event.started_at.isoformat(),
        },
    }
    
    # Mock embed_text since stream.online doesn't need embedding
    with patch("py.ingest.event_handler.embed_text"):
        event_id = await handler.handle_stream_online(event, payload_json)
    
    assert event_id is not None
    
    # Verify event was stored in DB
    from sqlalchemy import select
    from py.database.models import Event
    
    async with session_factory() as session:
        result = await session.execute(
            select(Event).where(Event.id == event_id)
        )
        stored_event = result.scalar_one_or_none()
    
    assert stored_event is not None
    assert stored_event.channel_id == sample_channel_id
    assert stored_event.type == "stream.online"
    assert stored_event.summary is not None
    assert "live" in stored_event.summary.lower()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_event_handler_stream_offline(test_engine, sample_channel_id):
    """Test EventHandler processes stream.offline event and stores in DB."""
    from sqlalchemy.ext.asyncio import async_sessionmaker, AsyncSession
    session_factory = async_sessionmaker(test_engine, expire_on_commit=False, class_=AsyncSession)
    vector_store = VectorStore(session_factory=session_factory)
    handler = EventHandler(vector_store)
    
    event = StreamOfflineEvent(
        broadcaster_user_id=sample_channel_id,
        broadcaster_user_login="testchannel",
        broadcaster_user_name="TestChannel",
    )
    
    payload_json = {
        "subscription": {"type": "stream.offline"},
        "event": {
            "broadcaster_user_id": sample_channel_id,
            "broadcaster_user_login": "testchannel",
            "broadcaster_user_name": "TestChannel",
        },
    }
    
    # Mock embed_text since stream.offline doesn't need embedding
    with patch("py.ingest.event_handler.embed_text"):
        event_id = await handler.handle_stream_offline(event, payload_json)
    
    assert event_id is not None
    
    # Verify event was stored
    from sqlalchemy import select
    from py.database.models import Event
    
    async with session_factory() as session:
        result = await session.execute(
            select(Event).where(Event.id == event_id)
        )
        stored_event = result.scalar_one_or_none()
    
    assert stored_event is not None
    assert stored_event.type == "stream.offline"
    assert stored_event.summary is not None
    assert "ended" in stored_event.summary.lower() or "offline" in stored_event.summary.lower()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_event_handler_channel_raid_with_embedding(test_engine, sample_channel_id, sample_embedding_1536):
    """Test EventHandler processes channel.raid event with embedding."""
    from sqlalchemy.ext.asyncio import async_sessionmaker, AsyncSession
    session_factory = async_sessionmaker(test_engine, expire_on_commit=False, class_=AsyncSession)
    vector_store = VectorStore(session_factory=session_factory)
    handler = EventHandler(vector_store)
    
    event = ChannelRaidEvent(
        from_broadcaster_user_id="987654321",
        from_broadcaster_user_login="raider",
        from_broadcaster_user_name="Raider",
        to_broadcaster_user_id=sample_channel_id,
        to_broadcaster_user_login="testchannel",
        to_broadcaster_user_name="TestChannel",
        viewers=100,
        started_at=datetime.now(timezone.utc),
    )
    
    payload_json = {
        "subscription": {"type": "channel.raid"},
        "event": {
            "from_broadcaster_user_id": "987654321",
            "from_broadcaster_user_login": "raider",
            "from_broadcaster_user_name": "Raider",
            "to_broadcaster_user_id": sample_channel_id,
            "to_broadcaster_user_login": "testchannel",
            "to_broadcaster_user_name": "TestChannel",
            "viewers": 100,
            "started_at": event.started_at.isoformat(),
        },
    }
    
    # Mock embed_text to return a test embedding
    with patch("py.ingest.event_handler.embed_text", new_callable=AsyncMock) as mock_embed:
        mock_embed.return_value = sample_embedding_1536
        
        event_id = await handler.handle_channel_raid(event, payload_json)
    
    assert event_id is not None
    mock_embed.assert_called_once()
    
    # Verify event was stored with embedding
    from sqlalchemy import select
    from py.database.models import Event
    
    async with session_factory() as session:
        result = await session.execute(
            select(Event).where(Event.id == event_id)
        )
        stored_event = result.scalar_one_or_none()
    
    assert stored_event is not None
    assert stored_event.type == "channel.raid"
    assert stored_event.embedding is not None
    assert len(stored_event.embedding) == 1536
    assert stored_event.summary is not None
    assert "raid" in stored_event.summary.lower()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_vector_store_insert_and_search(test_engine, sample_channel_id, sample_embedding_1536, sample_timestamp):
    """Test VectorStore insert_transcript and search_transcripts."""
    import numpy as np
    
    from sqlalchemy.ext.asyncio import async_sessionmaker, AsyncSession
    session_factory = async_sessionmaker(test_engine, expire_on_commit=False, class_=AsyncSession)
    vector_store = VectorStore(session_factory=session_factory)
    
    # Insert a transcript
    transcript_id = await vector_store.insert_transcript(
        channel_id=sample_channel_id,
        text_value="This is a test transcript about machine learning",
        start_time=sample_timestamp,
        end_time=sample_timestamp,
        embedding=sample_embedding_1536,
    )
    
    assert transcript_id is not None
    
    # Search for similar transcripts
    # Use a slightly modified version of the same embedding
    query_vec = np.array(sample_embedding_1536) + np.random.randn(1536).astype(np.float32) * 0.01
    query_vec = query_vec / np.linalg.norm(query_vec)
    
    results = await vector_store.search_transcripts(
        query_embedding=query_vec.tolist(),
        limit=5,
        channel_id=sample_channel_id,
    )
    
    assert len(results) > 0
    assert results[0]["id"] == transcript_id
    assert "machine learning" in results[0]["text"].lower()

