"""
Redis Session Management

Manages per-user conversation state in Redis with session storage,
retrieval, expiry, and rate limiting.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any

import redis.asyncio as redis

logger = logging.getLogger(__name__)


class RedisSessionManager:
    """Manages user session state in Redis."""

    def __init__(
        self,
        redis_url: str,
        session_expiry_minutes: int = 15,
        max_history: int = 5,
    ):
        """
        Initialize Redis session manager.

        Args:
            redis_url: Redis connection URL (e.g., "redis://localhost:6379/0")
            session_expiry_minutes: TTL for sessions in minutes
            max_history: Maximum number of Q&A pairs to store per session
        """
        self.redis_url = redis_url
        self.session_expiry_seconds = session_expiry_minutes * 60
        self.max_history = max_history
        self.redis_client: Optional[redis.Redis] = None
        self._connected = False

    async def connect(self) -> None:
        """Establish connection to Redis."""
        try:
            self.redis_client = redis.from_url(self.redis_url, decode_responses=True)
            # Test connection
            await self.redis_client.ping()
            self._connected = True
            logger.info("Connected to Redis")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            self._connected = False
            raise

    async def close(self) -> None:
        """Close Redis connection."""
        if self.redis_client:
            await self.redis_client.aclose()
            self._connected = False
            logger.info("Redis connection closed")

    def _get_session_key(self, user_id: str, channel: str) -> str:
        """Generate Redis key for session."""
        return f"session:{channel}:{user_id}"

    async def get_session(
        self, user_id: str, channel: str
    ) -> Dict[str, Any]:
        """
        Retrieve user session or create new one.

        Args:
            user_id: Username
            channel: Channel name

        Returns:
            Session dictionary with user state
        """
        if not self._connected or not self.redis_client:
            # Return empty session if Redis unavailable
            return self._create_empty_session(user_id, channel)

        key = self._get_session_key(user_id, channel)
        try:
            data = await self.redis_client.get(key)
            if data:
                session = json.loads(data)
                # Update last_seen timestamp
                session["last_seen"] = datetime.now(timezone.utc).isoformat()
                # Save updated timestamp (refresh TTL)
                await self._save_session(key, session)
                return session
            else:
                # Create new session
                session = self._create_empty_session(user_id, channel)
                await self._save_session(key, session)
                return session
        except Exception as e:
            logger.error(f"Error retrieving session for {user_id}: {e}")
            return self._create_empty_session(user_id, channel)

    def _create_empty_session(self, user_id: str, channel: str) -> Dict[str, Any]:
        """Create a new empty session dictionary."""
        now = datetime.now(timezone.utc).isoformat()
        return {
            "user_id": user_id,
            "channel": channel,
            "last_questions": [],
            "last_answers": [],
            "last_seen": now,
            "message_count": 0,
            "last_message_time": None,
        }

    async def _save_session(self, key: str, session: Dict[str, Any]) -> None:
        """Save session to Redis with TTL."""
        if not self._connected or not self.redis_client:
            return

        try:
            await self.redis_client.setex(
                key,
                self.session_expiry_seconds,
                json.dumps(session),
            )
        except Exception as e:
            logger.error(f"Error saving session {key}: {e}")

    async def update_session(
        self,
        user_id: str,
        channel: str,
        question: str,
        answer: str,
    ) -> None:
        """
        Add Q&A pair to session history.

        Args:
            user_id: Username
            channel: Channel name
            question: User's question
            answer: Bot's answer
        """
        if not self._connected or not self.redis_client:
            return

        session = await self.get_session(user_id, channel)
        now = datetime.now(timezone.utc).isoformat()

        # Add to history (FIFO queue, max N items)
        session["last_questions"].append(question)
        session["last_answers"].append(answer)
        session["message_count"] = session.get("message_count", 0) + 1
        session["last_seen"] = now
        session["last_message_time"] = now

        # Trim history to max size
        if len(session["last_questions"]) > self.max_history:
            session["last_questions"] = session["last_questions"][-self.max_history :]
        if len(session["last_answers"]) > self.max_history:
            session["last_answers"] = session["last_answers"][-self.max_history :]

        # Save updated session
        key = self._get_session_key(user_id, channel)
        await self._save_session(key, session)

    async def check_rate_limit(
        self, user_id: str, channel: str, limit_seconds: int
    ) -> bool:
        """
        Check if user is within rate limit.

        Args:
            user_id: Username
            channel: Channel name
            limit_seconds: Minimum seconds between messages

        Returns:
            True if user can send message, False if rate limited
        """
        if not self._connected or not self.redis_client:
            # Allow if Redis unavailable (graceful degradation)
            return True

        session = await self.get_session(user_id, channel)
        last_message_time = session.get("last_message_time")

        if not last_message_time:
            # No previous messages, allow
            return True

        try:
            last_time = datetime.fromisoformat(last_message_time.replace("Z", "+00:00"))
            now = datetime.now(timezone.utc)
            elapsed = (now - last_time).total_seconds()

            if elapsed < limit_seconds:
                logger.info(
                    f"Rate limit hit for {user_id} in {channel}: "
                    f"{elapsed:.1f}s < {limit_seconds}s"
                )
                return False

            return True
        except Exception as e:
            logger.error(f"Error checking rate limit for {user_id}: {e}")
            # Allow on error (graceful degradation)
            return True

    async def cleanup_expired_sessions(self) -> None:
        """
        Background task to clean up expired sessions.

        Note: Redis TTL handles expiry automatically, but this method
        can be used for additional cleanup if needed.
        """
        if not self._connected or not self.redis_client:
            return

        try:
            # Get all session keys
            keys = await self.redis_client.keys("session:*")
            expired_count = 0

            for key in keys:
                ttl = await self.redis_client.ttl(key)
                if ttl == -1:
                    # Key exists but has no TTL, set one
                    await self.redis_client.expire(key, self.session_expiry_seconds)
                elif ttl == -2:
                    # Key doesn't exist (shouldn't happen, but handle it)
                    expired_count += 1

            if expired_count > 0:
                logger.debug(f"Cleaned up {expired_count} expired sessions")
        except Exception as e:
            logger.error(f"Error during session cleanup: {e}")
