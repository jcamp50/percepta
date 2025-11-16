"""
Channel Metadata Polling Service

Periodically polls Twitch Helix API to fetch channel and stream metadata
(title, game, viewer count, tags) and stores snapshots in the database.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Optional, Dict, Any
import httpx

from py.config import settings
from py.memory.vector_store import VectorStore
from py.schemas.metadata import ChannelMetadata, ChannelInfo, StreamInfo
from py.utils.logging import get_logger

logger = get_logger(__name__, category='stream_metadata')

# Helix API base URL
HELIX_API_BASE = "https://api.twitch.tv/helix"


class ChannelMetadataPoller:
    """Polls Twitch Helix API for channel metadata and stores snapshots."""

    def __init__(
        self,
        client_id: Optional[str] = None,
        access_token: Optional[str] = None,
        target_channel: Optional[str] = None,
        broadcaster_id: Optional[str] = None,
        vector_store: Optional[VectorStore] = None,
    ):
        """
        Initialize channel metadata poller.

        Args:
            client_id: Twitch Client ID (defaults to settings.twitch_client_id)
            access_token: Twitch user access token (defaults to settings.twitch_bot_token)
            target_channel: Channel name to poll (defaults to settings.target_channel)
            broadcaster_id: Broadcaster user ID (will be fetched if not provided)
            vector_store: VectorStore instance for database operations
        """
        self.client_id = client_id or settings.twitch_client_id
        self.access_token = access_token or settings.twitch_bot_token
        self.target_channel = target_channel or settings.target_channel
        self.broadcaster_id = broadcaster_id
        self.vector_store = vector_store

        # Remove 'oauth:' prefix if present
        if self.access_token and self.access_token.startswith("oauth:"):
            self.access_token = self.access_token[6:]

        # Polling state
        self.is_running = False
        self.poll_task: Optional[asyncio.Task] = None
        self.last_title: Optional[str] = None
        self.last_game_name: Optional[str] = None

        # HTTP client for Helix API calls
        if self.client_id and self.access_token:
            self.http_client = httpx.AsyncClient(
                timeout=30.0,
                headers={
                    "Client-ID": self.client_id,
                    "Authorization": f"Bearer {self.access_token}",
                },
            )
        else:
            self.http_client = None
            logger.warning("HTTP client not initialized - missing credentials")

    async def start(self) -> None:
        """Start the polling task."""
        if not self.http_client:
            raise ValueError("HTTP client not initialized - missing credentials")

        if not self.target_channel:
            raise ValueError("Target channel is required for metadata polling")

        if self.is_running:
            logger.warning("Metadata poller is already running")
            return

        # Get broadcaster ID if not provided
        if not self.broadcaster_id:
            self.broadcaster_id = await self._get_broadcaster_id(self.target_channel)
            if not self.broadcaster_id:
                raise ValueError(
                    f"Failed to get broadcaster ID for channel: {self.target_channel}"
                )

        self.is_running = True
        logger.info(
            f"Starting metadata polling for channel: {self.target_channel} "
            f"(broadcaster_id: {self.broadcaster_id})"
        )

        # Start polling task
        self.poll_task = asyncio.create_task(self._poll_loop())

    async def stop(self) -> None:
        """Stop the polling task."""
        self.is_running = False

        if self.poll_task:
            self.poll_task.cancel()
            try:
                await self.poll_task
            except asyncio.CancelledError:
                # Task cancellation is expected when stopping the poller; ignore this exception.
                pass
            self.poll_task = None

        # Close HTTP client
        if self.http_client:
            await self.http_client.aclose()

        logger.info("Metadata polling stopped")

    async def _poll_loop(self) -> None:
        """Main polling loop."""
        interval = settings.metadata_poll_interval_seconds

        try:
            # Initial poll immediately
            await self.poll_and_store()

            # Then poll at intervals
            while self.is_running:
                await asyncio.sleep(interval)
                if self.is_running:
                    await self.poll_and_store()

        except asyncio.CancelledError:
            logger.info("Metadata polling loop cancelled")
        except Exception as e:
            logger.error(f"Error in metadata polling loop: {e}")
            self.is_running = False

    async def poll_and_store(self) -> None:
        """Poll metadata and store snapshot."""
        try:
            metadata = await self.fetch_metadata()
            if metadata:
                await self._store_snapshot(metadata)
                self._log_changes(metadata)
        except Exception as e:
            logger.error(f"Error polling and storing metadata: {e}")

    async def fetch_metadata(self) -> Optional[ChannelMetadata]:
        """Fetch combined channel and stream metadata."""
        if not self.broadcaster_id:
            logger.error("Broadcaster ID not set")
            return None

        # Fetch channel info
        channel_info = await self.fetch_channel_info(self.broadcaster_id)
        if not channel_info:
            logger.warning("Failed to fetch channel info")
            return None

        # Fetch stream info (only returns data if live)
        stream_info = await self.fetch_stream_info(self.broadcaster_id)

        # Combine into metadata
        metadata = ChannelMetadata(
            channel_id=self.broadcaster_id,
            channel_name=channel_info.broadcaster_name,
            title=channel_info.title or stream_info.title if stream_info else channel_info.title,
            game_id=channel_info.game_id or (
                stream_info.game_id if stream_info else None
            ),
            game_name=channel_info.game_name
            or (stream_info.game_name if stream_info else None),
            tags=channel_info.tags or (stream_info.tags if stream_info else None),
            viewer_count=stream_info.viewer_count if stream_info else None,
            is_live=stream_info is not None,
            broadcaster_language=channel_info.broadcaster_language,
            stream_started_at=stream_info.started_at if stream_info else None,
        )

        return metadata

    async def fetch_channel_info(
        self, broadcaster_id: str
    ) -> Optional[ChannelInfo]:
        """Fetch channel information from Helix API."""
        if not self.http_client:
            return None

        url = f"{HELIX_API_BASE}/channels"
        params = {"broadcaster_id": broadcaster_id}

        try:
            response = await self.http_client.get(url, params=params)
            response.raise_for_status()

            data = response.json()
            channels = data.get("data", [])

            if not channels:
                logger.warning(f"No channel data found for broadcaster: {broadcaster_id}")
                return None

            channel_data = channels[0]
            return ChannelInfo(
                broadcaster_id=channel_data.get("broadcaster_id"),
                broadcaster_name=channel_data.get("broadcaster_name"),
                game_id=channel_data.get("game_id"),
                game_name=channel_data.get("game_name"),
                title=channel_data.get("title"),
                broadcaster_language=channel_data.get("broadcaster_language"),
                tags=channel_data.get("tags"),
            )

        except httpx.HTTPStatusError as e:
            logger.error(
                f"Failed to fetch channel info: {e.response.status_code} - {e.response.text}"
            )
            return None
        except Exception as e:
            logger.error(f"Error fetching channel info: {e}")
            return None

    async def fetch_stream_info(self, broadcaster_id: str) -> Optional[StreamInfo]:
        """Fetch stream information from Helix API (only if live)."""
        if not self.http_client:
            return None

        url = f"{HELIX_API_BASE}/streams"
        params = {"user_id": broadcaster_id}

        try:
            response = await self.http_client.get(url, params=params)
            response.raise_for_status()

            data = response.json()
            streams = data.get("data", [])

            if not streams:
                # Stream is offline
                return None

            stream_data = streams[0]
            started_at_str = stream_data.get("started_at")
            started_at = None
            if started_at_str:
                try:
                    started_at = datetime.fromisoformat(
                        started_at_str.replace("Z", "+00:00")
                    )
                except Exception as e:
                    logger.warning(f"Failed to parse started_at timestamp '{started_at_str}': {e}")

            return StreamInfo(
                id=stream_data.get("id"),
                user_id=stream_data.get("user_id"),
                user_name=stream_data.get("user_name"),
                game_id=stream_data.get("game_id"),
                game_name=stream_data.get("game_name"),
                title=stream_data.get("title"),
                viewer_count=stream_data.get("viewer_count"),
                tags=stream_data.get("tags"),
                type=stream_data.get("type", "live"),
                started_at=started_at,
            )

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                # Stream is offline (expected)
                return None
            logger.error(
                f"Failed to fetch stream info: {e.response.status_code} - {e.response.text}"
            )
            return None
        except Exception as e:
            logger.error(f"Error fetching stream info: {e}")
            return None

    async def _get_broadcaster_id(self, channel_name: str) -> Optional[str]:
        """Get broadcaster user ID from channel name."""
        if not self.http_client:
            return None

        url = f"{HELIX_API_BASE}/users"
        params = {"login": channel_name.lower()}

        try:
            response = await self.http_client.get(url, params=params)
            response.raise_for_status()

            data = response.json()
            users = data.get("data", [])

            if users:
                return users[0].get("id")
            else:
                logger.warning(f"Channel not found: {channel_name}")
                return None

        except Exception as e:
            logger.error(f"Failed to get broadcaster ID: {e}")
            return None

    async def _store_snapshot(self, metadata: ChannelMetadata) -> None:
        """Store metadata snapshot in database."""
        if not self.vector_store:
            logger.warning("Vector store not available, skipping snapshot storage")
            return

        try:
            # Prepare payload JSON for debugging
            payload_json: Dict[str, Any] = {
                "channel_id": metadata.channel_id,
                "channel_name": metadata.channel_name,
                "title": metadata.title,
                "game_id": metadata.game_id,
                "game_name": metadata.game_name,
                "tags": metadata.tags,
                "viewer_count": metadata.viewer_count,
                "is_live": metadata.is_live,
                "broadcaster_language": metadata.broadcaster_language,
                "stream_started_at": (
                    metadata.stream_started_at.isoformat()
                    if metadata.stream_started_at
                    else None
                ),
            }

            snapshot_id = await self.vector_store.insert_channel_snapshot(
                channel_id=metadata.channel_id,
                title=metadata.title,
                game_id=metadata.game_id,
                game_name=metadata.game_name,
                tags=metadata.tags,
                viewer_count=metadata.viewer_count,
                payload_json=payload_json,
                embedding=None,  # Will be added in JCB-21
            )

            logger.debug(f"Stored channel snapshot: {snapshot_id}")

        except Exception as e:
            logger.error(f"Error storing channel snapshot: {e}")

    def _log_changes(self, metadata: ChannelMetadata) -> None:
        """Log metadata changes (title, game, etc.)."""
        # Log title changes
        if metadata.title != self.last_title:
            if self.last_title is not None:
                logger.info(
                    f"Title changed: '{self.last_title}' → '{metadata.title}'"
                )
            self.last_title = metadata.title

        # Log game changes
        if metadata.game_name != self.last_game_name:
            if self.last_game_name is not None:
                logger.info(
                    f"Game changed: '{self.last_game_name}' → '{metadata.game_name}'"
                )
            self.last_game_name = metadata.game_name

        # Log status
        status = "live" if metadata.is_live else "offline"
        status_info = f"Status: {status}"
        if metadata.is_live and metadata.viewer_count is not None:
            status_info += f", Viewers: {metadata.viewer_count:,}"
        if metadata.game_name:
            status_info += f", Game: {metadata.game_name}"
        logger.info(f"Channel metadata polled - {status_info}")

