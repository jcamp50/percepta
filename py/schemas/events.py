"""
Event Sub WebSocket Event Schemas

Pydantic models for Twitch EventSub WebSocket events.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field


class StreamOnlineEvent(BaseModel):
	"""Event when a stream goes live."""

	broadcaster_user_id: str
	broadcaster_user_login: str
	broadcaster_user_name: str
	type: str = Field(default="live")  # Always "live" for stream.online
	started_at: datetime


class StreamOfflineEvent(BaseModel):
	"""Event when a stream ends."""

	broadcaster_user_id: str
	broadcaster_user_login: str
	broadcaster_user_name: str


class ChannelRaidEvent(BaseModel):
	"""Event when a channel receives a raid."""

	from_broadcaster_user_id: str
	from_broadcaster_user_login: str
	from_broadcaster_user_name: str
	to_broadcaster_user_id: str
	to_broadcaster_user_login: str
	to_broadcaster_user_name: str
	viewers: int
	started_at: datetime


class ChannelSubscribeEvent(BaseModel):
	"""Event when a user subscribes to a channel."""

	broadcaster_user_id: str
	broadcaster_user_login: str
	broadcaster_user_name: str
	user_id: str
	user_login: str
	user_name: str
	tier: str = Field(description="Subscription tier: '1000', '2000', or '3000'")
	is_gift: bool = False


class EventSubNotification(BaseModel):
	"""Generic EventSub notification wrapper."""

	subscription: dict
	event: dict
	event_type: str = Field(description="Type of event (stream.online, stream.offline, etc.)")


class EventSubSession(BaseModel):
	"""EventSub WebSocket session metadata."""

	id: str = Field(description="Session ID from session_welcome")
	status: str = Field(description="Session status: 'connected', 'disconnected', 'reconnecting'")
	keepalive_interval_seconds: Optional[int] = None
	reconnect_url: Optional[str] = None
	created_at: datetime


