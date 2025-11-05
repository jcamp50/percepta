"""
Event Handler for Twitch EventSub Events

Processes incoming EventSub notifications, generates human-readable summaries,
and stores events with embeddings in the database.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List

from schemas.events import (
    StreamOnlineEvent,
    StreamOfflineEvent,
    ChannelRaidEvent,
    ChannelSubscribeEvent,
)
from py.memory.vector_store import VectorStore
from py.utils.embeddings import embed_text
from py.utils.logging import get_logger

logger = get_logger(__name__, category='stream_event_sub')


class EventHandler:
    """Handles processing and storage of Twitch EventSub events."""

    # Event types that should have embeddings generated
    EMBEDDING_ENABLED_TYPES = {
        "channel.raid",
        "channel.subscribe",
    }

    def __init__(self, vector_store: VectorStore):
        """
        Initialize event handler.

        Args:
            vector_store: VectorStore instance for storing events
        """
        self.vector_store = vector_store

    async def handle_stream_online(
        self, event: StreamOnlineEvent, payload_json: Dict[str, Any]
    ) -> Optional[str]:
        """
        Handle stream.online event.

        Args:
            event: Parsed StreamOnlineEvent
            payload_json: Raw event payload for storage

        Returns:
            Event ID if stored, None otherwise
        """
        # Format timestamp for summary
        timestamp_str = event.started_at.strftime("%I:%M %p").lstrip("0")

        # Generate summary
        summary = f"Stream went live at {timestamp_str}"

        # Stream online is low-information, no embedding
        return await self._store_event(
            channel_id=event.broadcaster_user_id,
            event_type="stream.online",
            timestamp=event.started_at,
            summary=summary,
            payload_json=payload_json,
            generate_embedding=False,
        )

    async def handle_stream_offline(
        self, event: StreamOfflineEvent, payload_json: Dict[str, Any]
    ) -> Optional[str]:
        """
        Handle stream.offline event.

        Args:
            event: Parsed StreamOfflineEvent
            payload_json: Raw event payload for storage

        Returns:
            Event ID if stored, None otherwise
        """
        # Use current time as timestamp
        timestamp = datetime.now(timezone.utc)
        timestamp_str = timestamp.strftime("%I:%M %p").lstrip("0")

        # Generate summary
        summary = f"Stream ended at {timestamp_str}"

        # Stream offline is low-information, no embedding
        return await self._store_event(
            channel_id=event.broadcaster_user_id,
            event_type="stream.offline",
            timestamp=timestamp,
            summary=summary,
            payload_json=payload_json,
            generate_embedding=False,
        )

    async def handle_channel_raid(
        self, event: ChannelRaidEvent, payload_json: Dict[str, Any]
    ) -> Optional[str]:
        """
        Handle channel.raid event.

        Args:
            event: Parsed ChannelRaidEvent
            payload_json: Raw event payload for storage

        Returns:
            Event ID if stored, None otherwise
        """
        # Format timestamp
        timestamp_str = event.started_at.strftime("%I:%M %p").lstrip("0")

        # Generate summary
        summary = (
            f"{event.from_broadcaster_user_name} raided with {event.viewers} viewers"
        )

        # Raids are meaningful, generate embedding
        return await self._store_event(
            channel_id=event.to_broadcaster_user_id,
            event_type="channel.raid",
            timestamp=event.started_at,
            summary=summary,
            payload_json=payload_json,
            generate_embedding=True,
        )

    async def handle_channel_subscribe(
        self, event: ChannelSubscribeEvent, payload_json: Dict[str, Any]
    ) -> Optional[str]:
        """
        Handle channel.subscribe event.

        Args:
            event: Parsed ChannelSubscribeEvent
            payload_json: Raw event payload for storage

        Returns:
            Event ID if stored, None otherwise
        """
        # Use current time as timestamp
        timestamp = datetime.now(timezone.utc)
        timestamp_str = timestamp.strftime("%I:%M %p").lstrip("0")

        # Generate summary
        gift_text = " (gift)" if event.is_gift else ""
        summary = f"{event.user_name} subscribed (Tier {event.tier}){gift_text}"

        # Subscriptions are meaningful, generate embedding
        return await self._store_event(
            channel_id=event.broadcaster_user_id,
            event_type="channel.subscribe",
            timestamp=timestamp,
            summary=summary,
            payload_json=payload_json,
            generate_embedding=True,
        )

    async def _store_event(
        self,
        channel_id: str,
        event_type: str,
        timestamp: datetime,
        summary: str,
        payload_json: Dict[str, Any],
        generate_embedding: bool,
    ) -> Optional[str]:
        """
        Store event in database with optional embedding.

        Args:
            channel_id: Broadcaster channel ID
            event_type: Type of event (e.g., "stream.online")
            timestamp: Event timestamp
            summary: Human-readable summary
            payload_json: Raw event payload
            generate_embedding: Whether to generate and store embedding

        Returns:
            Event ID if stored successfully, None otherwise
        """
        try:
            # Generate embedding if needed
            embedding: Optional[List[float]] = None
            if generate_embedding:
                try:
                    embedding = await embed_text(summary)
                    logger.debug(
                        f"Generated embedding for event {event_type}: "
                        f"channel={channel_id}, summary_length={len(summary)}"
                    )
                except Exception as e:
                    logger.error(
                        f"Failed to generate embedding for event {event_type}: {e}"
                    )
                    # Continue without embedding rather than failing

            # Store event
            event_id = await self.vector_store.insert_event(
                channel_id=channel_id,
                event_type=event_type,
                timestamp=timestamp,
                summary=summary,
                payload_json=payload_json,
                embedding=embedding,
            )

            logger.info(
                f"Stored event: type={event_type}, channel={channel_id}, "
                f"id={event_id}, summary='{summary}', has_embedding={embedding is not None}"
            )

            return event_id

        except Exception as e:
            logger.error(
                f"Failed to store event {event_type} for channel {channel_id}: {e}",
                exc_info=True,
            )
            return None
