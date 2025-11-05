"""
Redis Session Management

Manages per-user conversation state in Redis with session storage,
retrieval, expiry, and rate limiting.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Tuple

import redis.asyncio as redis

logger = logging.getLogger(__name__)


class RedisSessionManager:
    """Manages user session state in Redis."""

    def __init__(
        self,
        redis_url: str,
        session_expiry_minutes: int = 15,
        max_history: int = 5,
        global_rate_limit_msgs: int = 20,
        global_rate_limit_window: int = 30,
        repeated_question_cooldown: int = 60,
    ):
        """
        Initialize Redis session manager.

        Args:
            redis_url: Redis connection URL (e.g., "redis://localhost:6379/0")
            session_expiry_minutes: TTL for sessions in minutes
            max_history: Maximum number of Q&A pairs to store per session
            global_rate_limit_msgs: Maximum messages allowed in time window
            global_rate_limit_window: Time window in seconds for global rate limit
            repeated_question_cooldown: Cooldown period in seconds for repeated questions
        """
        self.redis_url = redis_url
        self.session_expiry_seconds = session_expiry_minutes * 60
        self.max_history = max_history
        self.global_rate_limit_msgs = global_rate_limit_msgs
        self.global_rate_limit_window = global_rate_limit_window
        self.repeated_question_cooldown = repeated_question_cooldown
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

    def _get_global_rate_limit_key(self, channel: str) -> str:
        """Generate Redis key for global rate limit counter."""
        return f"global_rate_limit:{channel}"

    def _get_repeated_question_cooldown_key(self, user_id: str, channel: str) -> str:
        """Generate Redis key for repeated question cooldown."""
        return f"cooldown:{channel}:{user_id}"

    def _get_long_answer_key(self, answer_id: str) -> str:
        """Generate Redis key for stored long answer."""
        return f"long_answer:{answer_id}"

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

    async def check_global_rate_limit(self, channel: str) -> bool:
        """
        Check if global rate limit is exceeded.

        Args:
            channel: Channel name

        Returns:
            True if under limit, False if rate limit exceeded
        """
        if not self._connected or not self.redis_client:
            # Allow if Redis unavailable (graceful degradation)
            return True

        key = self._get_global_rate_limit_key(channel)
        try:
            current_count = await self.redis_client.get(key)
            if current_count is None:
                # No messages yet, allow
                return True

            count = int(current_count)
            if count >= self.global_rate_limit_msgs:
                logger.info(
                    f"Global rate limit exceeded for {channel}: "
                    f"{count}/{self.global_rate_limit_msgs} messages"
                )
                return False

            return True
        except Exception as e:
            logger.error(f"Error checking global rate limit for {channel}: {e}")
            # Allow on error (graceful degradation)
            return True

    async def update_global_rate_limit(self, channel: str) -> None:
        """
        Increment global message counter with TTL.

        Args:
            channel: Channel name
        """
        if not self._connected or not self.redis_client:
            return

        key = self._get_global_rate_limit_key(channel)
        try:
            # Increment counter and set TTL if key doesn't exist
            pipe = self.redis_client.pipeline()
            pipe.incr(key)
            pipe.expire(key, self.global_rate_limit_window)
            await pipe.execute()
        except Exception as e:
            logger.error(f"Error updating global rate limit for {channel}: {e}")

    def _normalize_question(self, question: str) -> str:
        """
        Normalize question for comparison (lowercase, remove punctuation).

        Args:
            question: Question text

        Returns:
            Normalized question string
        """
        # Convert to lowercase and remove extra whitespace
        normalized = question.lower().strip()
        # Remove common punctuation
        normalized = re.sub(r'[^\w\s]', '', normalized)
        # Remove extra whitespace
        normalized = re.sub(r'\s+', ' ', normalized)
        return normalized

    async def check_repeated_question_cooldown(
        self, user_id: str, channel: str, question: str
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if question is similar to recent questions and still in cooldown.

        Args:
            user_id: Username
            channel: Channel name
            question: New question to check

        Returns:
            Tuple of (is_allowed, cooldown_key)
            - is_allowed: True if question can be asked, False if in cooldown
            - cooldown_key: Redis key for cooldown if in cooldown, None otherwise
        """
        if not self._connected or not self.redis_client:
            # Allow if Redis unavailable (graceful degradation)
            return True, None

        # Get session to check recent questions
        session = await self.get_session(user_id, channel)
        recent_questions = session.get("last_questions", [])
        if not recent_questions:
            # No recent questions, allow
            return True, None

        # Normalize the new question
        normalized_question = self._normalize_question(question)

        # Check if question is similar to any recent question
        for recent_q in recent_questions:
            normalized_recent = self._normalize_question(recent_q)
            # Simple similarity check: if normalized questions are identical
            if normalized_question == normalized_recent:
                # Check if cooldown is still active
                cooldown_key = self._get_repeated_question_cooldown_key(user_id, channel)
                cooldown_active = await self.redis_client.exists(cooldown_key)
                if cooldown_active:
                    logger.info(
                        f"Repeated question cooldown active for {user_id} in {channel}"
                    )
                    return False, cooldown_key
                else:
                    # Cooldown expired, set new cooldown
                    await self.redis_client.setex(
                        cooldown_key, self.repeated_question_cooldown, "1"
                    )
                    return True, None

        # Question is not repeated, allow
        return True, None

    async def store_long_answer(
        self, answer_id: str, full_answer: str, ttl_hours: int
    ) -> None:
        """
        Store full answer in Redis for !more command.

        Args:
            answer_id: Unique identifier for the answer
            full_answer: Full answer text to store
            ttl_hours: Time to live in hours
        """
        if not self._connected or not self.redis_client:
            return

        key = self._get_long_answer_key(answer_id)
        try:
            await self.redis_client.setex(
                key, ttl_hours * 3600, json.dumps({"answer": full_answer})
            )
        except Exception as e:
            logger.error(f"Error storing long answer {answer_id}: {e}")

    def _get_answer_prefix_key(self, channel: str, user_id: str, prefix: str) -> str:
        """Generate Redis key for answer ID prefix mapping."""
        return f"answer_prefix:{channel}:{user_id}:{prefix}"

    async def store_answer_prefix_mapping(
        self, channel: str, user_id: str, prefix: str, full_answer_id: str, ttl_hours: int
    ) -> None:
        """
        Store mapping from answer prefix to full answer ID.

        Args:
            channel: Channel name
            user_id: Username
            prefix: Answer ID prefix (first 8 chars)
            full_answer_id: Full UUID answer ID
            ttl_hours: Time to live in hours
        """
        if not self._connected or not self.redis_client:
            return

        key = self._get_answer_prefix_key(channel, user_id, prefix)
        try:
            await self.redis_client.setex(
                key, ttl_hours * 3600, json.dumps({"answer_id": full_answer_id})
            )
        except Exception as e:
            logger.error(f"Error storing answer prefix mapping {prefix}: {e}")

    async def get_answer_id_from_prefix(
        self, channel: str, user_id: str, prefix: str
    ) -> Optional[str]:
        """
        Get full answer ID from prefix.

        Args:
            channel: Channel name
            user_id: Username
            prefix: Answer ID prefix

        Returns:
            Full answer ID if found, None otherwise
        """
        if not self._connected or not self.redis_client:
            return None

        key = self._get_answer_prefix_key(channel, user_id, prefix)
        try:
            data = await self.redis_client.get(key)
            if data:
                stored = json.loads(data)
                return stored.get("answer_id")
            return None
        except Exception as e:
            logger.error(f"Error retrieving answer prefix mapping {prefix}: {e}")
            return None

    async def get_long_answer(self, answer_id: str) -> Optional[str]:
        """
        Retrieve stored long answer.

        Args:
            answer_id: Unique identifier for the answer

        Returns:
            Full answer text if found, None otherwise
        """
        if not self._connected or not self.redis_client:
            return None

        key = self._get_long_answer_key(answer_id)
        try:
            data = await self.redis_client.get(key)
            if data:
                stored = json.loads(data)
                return stored.get("answer")
            return None
        except Exception as e:
            logger.error(f"Error retrieving long answer {answer_id}: {e}")
            return None

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
