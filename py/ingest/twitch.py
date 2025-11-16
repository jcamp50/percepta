"""
Twitch EventSub WebSocket Client

Handles real-time Twitch EventSub WebSocket connections for receiving
stream events (online, offline, raids, subscriptions, etc.).
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Callable, Dict, Optional
import httpx
import websockets
from websockets.client import WebSocketClientProtocol

from py.config import settings
from py.schemas.events import (
    StreamOnlineEvent,
    StreamOfflineEvent,
    ChannelRaidEvent,
    ChannelSubscribeEvent,
    EventSubNotification,
    EventSubSession,
)
from py.ingest.event_handler import EventHandler
from py.utils.logging import get_logger

logger = get_logger(__name__, category='stream_event_sub')

# EventSub WebSocket URL
EVENTSUB_WS_URL = "wss://eventsub.wss.twitch.tv/ws"
# Helix API base URL for creating subscriptions
HELIX_API_BASE = "https://api.twitch.tv/helix"


class EventSubWebSocketClient:
    """EventSub WebSocket client for receiving real-time Twitch events."""

    def __init__(
        self,
        client_id: Optional[str] = None,
        access_token: Optional[str] = None,
        target_channel: Optional[str] = None,
    ):
        """
        Initialize EventSub WebSocket client.
        """
        self.client_id = client_id or settings.twitch_client_id
        self.access_token = access_token or settings.twitch_bot_token
        self.target_channel = target_channel or settings.target_channel

        # Debug logging to verify values are loaded
        if not self.client_id:
            logger.warning("TWITCH_CLIENT_ID not set - check environment variable")
        if not self.access_token:
            logger.warning("TWITCH_BOT_TOKEN not set - check environment variable")
        else:
            logger.debug(f"Token loaded (length: {len(self.access_token)})")

        # Remove 'oauth:' prefix if present (6 characters, not 7!)
        if self.access_token and self.access_token.startswith("oauth:"):
            self.access_token = self.access_token[6:]  # 'oauth:' is 6 chars
            logger.debug("Removed 'oauth:' prefix from token")

        # WebSocket connection
        self.ws: Optional[WebSocketClientProtocol] = None
        self.session: Optional[EventSubSession] = None
        self.is_connected = False
        self.is_running = False

        # Reconnection state
        self.reconnect_delay = settings.eventsub_reconnect_delay
        self.max_reconnect_delay = settings.eventsub_max_reconnect_delay
        self.reconnect_task: Optional[asyncio.Task] = None

        # Event handlers (callbacks)
        self.event_handlers: Dict[str, Callable] = {
            "stream.online": self._on_stream_online,
            "stream.offline": self._on_stream_offline,
            "channel.raid": self._on_channel_raid,
            "channel.subscribe": self._on_channel_subscribe,
        }

        # Subscriptions tracking
        self.subscriptions: Dict[str, dict] = {}

        # HTTP client for Helix API calls - initialize AFTER token processing
        # Only create if we have credentials
        if self.client_id and self.access_token:
            # Log token info (first 10 chars only for security)
            token_preview = (
                f"{self.access_token[:10]}..." if len(self.access_token) > 10 else "***"
            )
            logger.info(
                f"Initializing HTTP client - Client-ID: {self.client_id[:8]}..., "
                f"Token: {token_preview}"
            )
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

    async def connect(self) -> None:
        """Establish WebSocket connection to EventSub."""
        if not self.client_id or not self.access_token:
            raise ValueError("Twitch Client ID and access token are required")

        if not self.target_channel:
            raise ValueError("Target channel is required for EventSub subscriptions")

        # Validate token before connecting
        try:
            await self._validate_token()
        except Exception as e:
            logger.error(f"Token validation failed: {e}")
            logger.error(
                "Please verify your TWITCH_BOT_TOKEN is valid and not expired. "
                "Regenerate with: node scripts/init_twitch_oauth.js"
            )
            raise

        logger.info("Connecting to Twitch EventSub WebSocket...")
        self.is_running = True

        try:
            self.ws = await websockets.connect(
                EVENTSUB_WS_URL,
                ping_interval=20,  # Send ping every 20 seconds
                ping_timeout=10,  # Wait 10 seconds for pong
            )
            self.is_connected = True
            logger.info("Connected to EventSub WebSocket")

            # Start message handler
            asyncio.create_task(self._handle_messages())

        except Exception as e:
            logger.error(f"Failed to connect to EventSub WebSocket: {e}")
            self.is_connected = False
            if self.is_running:
                await self._schedule_reconnect()

    async def disconnect(self) -> None:
        """Close WebSocket connection."""
        self.is_running = False
        self.is_connected = False

        # Cancel reconnect task
        if self.reconnect_task:
            self.reconnect_task.cancel()
            try:
                await self.reconnect_task
            except asyncio.CancelledError:
                # Task cancellation is expected when disconnecting; ignore this exception.
                pass
            self.reconnect_task = None

        # Close WebSocket
        if self.ws:
            try:
                await self.ws.close()
            except Exception as e:
                logger.warning(f"Error closing WebSocket: {e}")
            self.ws = None

        # Close HTTP client
        if self.http_client:
            await self.http_client.aclose()

        logger.info("Disconnected from EventSub WebSocket")

    async def _handle_messages(self) -> None:
        """Handle incoming WebSocket messages."""
        if not self.ws:
            return

        try:
            async for message in self.ws:
                try:
                    data = json.loads(message)
                    message_type = data.get("metadata", {}).get("message_type")

                    if message_type == "session_welcome":
                        await self._handle_session_welcome(data)
                    elif message_type == "session_keepalive":
                        # Keepalive received, connection is alive
                        pass
                    elif message_type == "notification":
                        await self._handle_notification(data)
                    elif message_type == "session_reconnect":
                        await self._handle_session_reconnect(data)
                    elif message_type == "revocation":
                        await self._handle_revocation(data)
                    else:
                        logger.debug(f"Unknown message type: {message_type}")

                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse WebSocket message: {e}")
                except Exception as e:
                    logger.error(f"Error handling WebSocket message: {e}")

        except websockets.exceptions.ConnectionClosed:
            logger.warning("EventSub WebSocket connection closed")
            self.is_connected = False
            if self.is_running:
                await self._schedule_reconnect()
        except Exception as e:
            logger.error(f"Error in message handler: {e}")
            self.is_connected = False
            if self.is_running:
                await self._schedule_reconnect()

    async def _handle_session_welcome(self, data: dict) -> None:
        """Handle session_welcome message."""
        payload = data.get("payload", {})
        session = payload.get("session", {})
        session_id = session.get("id")
        keepalive_interval = session.get("keepalive_timeout_seconds")

        self.session = EventSubSession(
            id=session_id,
            status="connected",
            keepalive_interval_seconds=keepalive_interval,
            created_at=datetime.now(timezone.utc),
        )

        logger.info(f"EventSub session established: {session_id}")

        # Create subscriptions
        await self._create_subscriptions(session_id)

    async def _handle_notification(self, data: dict) -> None:
        """Handle notification message (event received)."""
        payload = data.get("payload", {})
        subscription = payload.get("subscription", {})
        event_data = payload.get("event", {})
        subscription_type = subscription.get("type")

        logger.info(f"Received EventSub notification: {subscription_type}")

        # Route to appropriate handler
        handler = self.event_handlers.get(subscription_type)
        if handler:
            try:
                await handler(event_data)
            except Exception as e:
                logger.error(f"Error in event handler for {subscription_type}: {e}")
        else:
            logger.warning(f"No handler for event type: {subscription_type}")

    async def _handle_session_reconnect(self, data: dict) -> None:
        """Handle session_reconnect message."""
        payload = data.get("payload", {})
        session = payload.get("session", {})
        reconnect_url = session.get("reconnect_url")

        logger.info("EventSub session reconnect requested")
        self.is_connected = False

        if reconnect_url:
            # Close current connection
            if self.ws:
                try:
                    await self.ws.close()
                except Exception as e:
                    logger.error(f"Error closing websocket during reconnect: {e}")

            # Reconnect to new URL
            try:
                self.ws = await websockets.connect(reconnect_url)
                self.is_connected = True
                logger.info("Reconnected to EventSub WebSocket")
                # Message handler will continue automatically
            except Exception as e:
                logger.error(f"Failed to reconnect: {e}")
                await self._schedule_reconnect()

    async def _handle_revocation(self, data: dict) -> None:
        """Handle revocation message (subscription revoked)."""
        payload = data.get("payload", {})
        subscription = payload.get("subscription", {})
        subscription_id = subscription.get("id")
        subscription_type = subscription.get("type")

        logger.warning(
            f"EventSub subscription revoked: {subscription_type} ({subscription_id})"
        )

        # Remove from tracking
        if subscription_id in self.subscriptions:
            del self.subscriptions[subscription_id]

    async def _schedule_reconnect(self) -> None:
        """Schedule reconnection with exponential backoff."""
        if self.reconnect_task and not self.reconnect_task.done():
            return  # Already scheduled

        delay = self.reconnect_delay
        logger.info(f"Scheduling EventSub reconnect in {delay} seconds...")

        async def reconnect():
            await asyncio.sleep(delay)
            if self.is_running and not self.is_connected:
                logger.info("Attempting EventSub reconnection...")
                await self.connect()
                # Reset delay on successful connection
                if self.is_connected:
                    self.reconnect_delay = settings.eventsub_reconnect_delay
                else:
                    # Exponential backoff
                    self.reconnect_delay = min(
                        self.reconnect_delay * 2, self.max_reconnect_delay
                    )

        self.reconnect_task = asyncio.create_task(reconnect())

    async def _create_subscriptions(self, session_id: str) -> None:
        """Create EventSub subscriptions for target channel."""
        if not self.target_channel:
            logger.warning("No target channel configured, skipping subscriptions")
            return

        # Get broadcaster user ID from channel name
        broadcaster_id = await self._get_broadcaster_id(self.target_channel)
        if not broadcaster_id:
            logger.error(
                f"Failed to get broadcaster ID for channel: {self.target_channel}"
            )
            return

        # Define subscriptions to create
        subscriptions = [
            {"type": "stream.online", "version": "1"},
            {"type": "stream.offline", "version": "1"},
            {"type": "channel.raid", "version": "1"},
            {"type": "channel.subscribe", "version": "1"},
        ]

        for sub_config in subscriptions:
            try:
                await self._create_subscription(
                    session_id,
                    broadcaster_id,
                    sub_config["type"],
                    sub_config["version"],
                )
                await asyncio.sleep(
                    0.5
                )  # Rate limit: avoid hitting Twitch API too fast
            except Exception as e:
                logger.error(
                    f"Failed to create subscription for {sub_config['type']}: {e}"
                )

    async def _create_subscription(
        self, session_id: str, broadcaster_id: str, event_type: str, version: str
    ) -> None:
        """Create a single EventSub subscription."""
        url = f"{HELIX_API_BASE}/eventsub/subscriptions"

        # Special handling for channel.raid - needs to_broadcaster_user_id
        # We want to receive raids TO our target channel
        if event_type == "channel.raid":
            condition = {"to_broadcaster_user_id": broadcaster_id}
        else:
            condition = {"broadcaster_user_id": broadcaster_id}

        payload = {
            "type": event_type,
            "version": version,
            "condition": condition,
            "transport": {"method": "websocket", "session_id": session_id},
        }

        try:
            response = await self.http_client.post(url, json=payload)
            response.raise_for_status()

            data = response.json()
            subscription_data = data.get("data", [{}])[0]
            subscription_id = subscription_data.get("id")

            if subscription_id:
                self.subscriptions[subscription_id] = subscription_data
                logger.info(
                    f"Created EventSub subscription: {event_type} ({subscription_id})"
                )
            else:
                logger.warning(f"Subscription created but no ID returned: {event_type}")

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 403:
                logger.warning(
                    f"Subscription {event_type} requires broadcaster permission "
                    "(or may not be available with current token). Skipping."
                )
            elif e.response.status_code == 401:
                logger.error(
                    f"Authentication failed for subscription {event_type}. "
                    "Check your access token."
                )
            else:
                logger.error(
                    f"Failed to create subscription {event_type}: "
                    f"{e.response.status_code} - {e.response.text}"
                )
        except Exception as e:
            logger.error(f"Error creating subscription {event_type}: {e}")

    async def _validate_token(self) -> None:
        """Validate the OAuth token with Twitch."""
        if not self.http_client:
            raise ValueError("HTTP client not initialized")

        try:
            response = await self.http_client.get(
                "https://id.twitch.tv/oauth2/validate"
            )
            response.raise_for_status()

            data = response.json()
            logger.info(
                f"Token validated - Client ID: {data.get('client_id')}, "
                f"User: {data.get('login')}"
            )
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 401:
                raise ValueError(
                    "Token is invalid or expired. Please regenerate your OAuth token."
                )
            raise

    async def _get_broadcaster_id(self, channel_name: str) -> Optional[str]:
        """Get broadcaster user ID from channel name."""
        if not self.http_client:
            logger.error("HTTP client not initialized - cannot get broadcaster ID")
            return None

        url = f"{HELIX_API_BASE}/users"
        params = {"login": channel_name.lower()}

        try:
            # Debug: Log request details
            logger.debug(
                f"Requesting broadcaster ID for channel: {channel_name}, "
                f"Client-ID header present: {bool(self.client_id)}, "
                f"Token present: {bool(self.access_token)}"
            )

            response = await self.http_client.get(url, params=params)

            # Log response details for debugging
            logger.debug(
                f"Helix API response: status={response.status_code}, "
                f"headers={dict(response.headers)}"
            )

            response.raise_for_status()

            data = response.json()
            users = data.get("data", [])

            if users:
                broadcaster_id = users[0].get("id")
                logger.info(
                    f"Found broadcaster ID for {channel_name}: {broadcaster_id}"
                )
                return broadcaster_id
            else:
                logger.warning(f"Channel not found: {channel_name}")
                return None

        except httpx.HTTPStatusError as e:
            # Detailed error logging for 401
            if e.response.status_code == 401:
                logger.error(
                    f"Authentication failed getting broadcaster ID. "
                    f"Status: {e.response.status_code}, "
                    f"Response: {e.response.text[:200]}"
                )
                # Check if token is being sent
                request_headers = dict(e.response.request.headers)
                auth_header = request_headers.get("authorization", "NOT SET")
                logger.error(
                    f"Authorization header present: {bool(auth_header)}, "
                    f"Value preview: {auth_header[:20] if auth_header != 'NOT SET' else 'N/A'}..."
                )
            else:
                logger.error(
                    f"HTTP error getting broadcaster ID: {e.response.status_code} - {e.response.text}"
                )
            return None
        except Exception as e:
            logger.error(f"Failed to get broadcaster ID: {e}")
            return None

    # Event handlers

    async def _on_stream_online(self, event_data: dict) -> None:
        """Handle stream.online event."""
        try:
            event = StreamOnlineEvent(**event_data)
            logger.info(
                f"Stream went online: {event.broadcaster_user_name} "
                f"at {event.started_at}"
            )
            # TODO: Forward to ingest pipeline (JCB-21)
        except Exception as e:
            logger.error(f"Error processing stream.online event: {e}")

    async def _on_stream_offline(self, event_data: dict) -> None:
        """Handle stream.offline event."""
        try:
            event = StreamOfflineEvent(**event_data)
            logger.info(f"Stream went offline: {event.broadcaster_user_name}")
            # TODO: Forward to ingest pipeline (JCB-21)
        except Exception as e:
            logger.error(f"Error processing stream.offline event: {e}")

    async def _on_channel_raid(self, event_data: dict) -> None:
        """Handle channel.raid event."""
        try:
            event = ChannelRaidEvent(**event_data)
            logger.info(
                f"Raid received: {event.from_broadcaster_user_name} → "
                f"{event.to_broadcaster_user_name} ({event.viewers} viewers)"
            )
            # TODO: Forward to ingest pipeline (JCB-21)
        except Exception as e:
            logger.error(f"Error processing channel.raid event: {e}")

    async def _on_channel_subscribe(self, event_data: dict) -> None:
        """Handle channel.subscribe event."""
        try:
            event = ChannelSubscribeEvent(**event_data)
            logger.info(
                f"Subscription: {event.user_name} subscribed to "
                f"{event.broadcaster_user_name} (Tier {event.tier})"
            )
            # TODO: Forward to ingest pipeline (JCB-21)
        except Exception as e:
            logger.error(f"Error processing channel.subscribe event: {e}")
