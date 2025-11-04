"""
Channel Metadata Schemas

Pydantic models for channel and stream metadata from Twitch Helix API.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel


class ChannelInfo(BaseModel):
    """Channel information from Helix API."""

    broadcaster_id: str
    broadcaster_name: str
    game_id: Optional[str] = None
    game_name: Optional[str] = None
    title: Optional[str] = None
    broadcaster_language: Optional[str] = None
    tags: Optional[List[str]] = None


class StreamInfo(BaseModel):
    """Stream information from Helix API (only when live)."""

    id: str
    user_id: str
    user_name: str
    game_id: Optional[str] = None
    game_name: Optional[str] = None
    title: Optional[str] = None
    viewer_count: Optional[int] = None
    tags: Optional[List[str]] = None
    type: str  # "live"
    started_at: Optional[datetime] = None


class ChannelMetadata(BaseModel):
    """Combined channel and stream metadata."""

    channel_id: str
    channel_name: str
    title: Optional[str] = None
    game_id: Optional[str] = None
    game_name: Optional[str] = None
    tags: Optional[List[str]] = None
    viewer_count: Optional[int] = None
    is_live: bool = False
    broadcaster_language: Optional[str] = None
    stream_started_at: Optional[datetime] = None


class ChannelSnapshotResponse(BaseModel):
    """Response model for stored channel snapshot."""

    snapshot_id: str
    channel_id: str
    timestamp: datetime
    title: Optional[str] = None
    game_name: Optional[str] = None
    viewer_count: Optional[int] = None
    is_live: bool = False

