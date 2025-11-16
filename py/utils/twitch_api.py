"""
Shared Twitch API utilities
"""

import asyncio
import time
from typing import Optional, Dict

import httpx
from py.config import settings
from py.utils.logging import get_logger

logger = get_logger(__name__, category="system")


_client: Optional[httpx.AsyncClient] = None
_client_lock = asyncio.Lock()
_broadcaster_cache: Dict[str, str] = {}
_failure_count = 0
_circuit_open_until: float = 0.0
MAX_FAILURES = 5
CIRCUIT_BREAK_SECONDS = 30


async def _get_client() -> httpx.AsyncClient:
    global _client
    if _client is None:
        async with _client_lock:
            if _client is None:
                timeout = httpx.Timeout(30.0, connect=10.0)
                _client = httpx.AsyncClient(timeout=timeout)
    return _client


async def _close_client() -> None:
    global _client
    if _client is not None:
        await _client.aclose()
        _client = None


async def get_broadcaster_id_from_channel_name(channel_name: str) -> Optional[str]:
    """
    Get broadcaster user ID from channel name using Twitch Helix API.

    This is a shared utility function that can be used by both the endpoint
    and the metadata poller to ensure consistent behavior.

    Args:
        channel_name: Twitch channel name (e.g., "xqc")

    Returns:
        Broadcaster ID as string, or None if not found
    """
    # Get credentials
    client_id = settings.twitch_client_id
    access_token = settings.twitch_bot_token or ""

    if not client_id or not access_token:
        logger.error("Twitch credentials not configured")
        return None

    # Remove 'oauth:' prefix if present
    if access_token.startswith("oauth:"):
        access_token = access_token[6:]

    global _failure_count, _circuit_open_until

    channel_key = channel_name.lower()
    if channel_key in _broadcaster_cache:
        return _broadcaster_cache[channel_key]

    now = time.monotonic()
    if now < _circuit_open_until:
        remaining = int(_circuit_open_until - now)
        logger.warning(
            "Skipping Twitch lookup for %s due to open circuit (%ss remaining)",
            channel_name,
            remaining,
        )
        return None

    client = await _get_client()

    url = "https://api.twitch.tv/helix/users"
    params = {"login": channel_key}
    headers = {
        "Client-ID": client_id,
        "Authorization": f"Bearer {access_token}",
    }

    backoff = 0.5
    for attempt in range(1, 4):
        try:
            response = await client.get(url, params=params, headers=headers)
            response.raise_for_status()

            data = response.json()
            users = data.get("data", [])

            if users:
                broadcaster_id = users[0].get("id")
                if broadcaster_id:
                    _broadcaster_cache[channel_key] = broadcaster_id
                logger.info(
                    "Found broadcaster ID for %s: %s (cache size=%s)",
                    channel_name,
                    broadcaster_id,
                    len(_broadcaster_cache),
                )
                _failure_count = 0
                _circuit_open_until = 0.0
                return broadcaster_id
            else:
                logger.warning(f"Channel not found: {channel_name}")
                return None

        except httpx.HTTPStatusError as exc:
            logger.error(
                "HTTP error getting broadcaster ID (%s): %s - %s",
                channel_name,
                exc.response.status_code,
                exc.response.text,
            )
            if exc.response.status_code in {429, 500, 502, 503, 504}:
                await asyncio.sleep(backoff)
                backoff *= 2
                continue
            return None
        except httpx.RequestError as exc:
            logger.warning(
                "Transient error getting broadcaster ID for %s (attempt %s/3): %s",
                channel_name,
                attempt,
                exc,
            )
            await asyncio.sleep(backoff)
            backoff *= 2
            continue
        except Exception as exc:
            logger.error(
                "Failed to get broadcaster ID for %s: %s",
                channel_name,
                exc,
                exc_info=True,
            )
            break

    _failure_count += 1
    if _failure_count >= MAX_FAILURES:
        _circuit_open_until = time.monotonic() + CIRCUIT_BREAK_SECONDS
        logger.error(
            "Opening Twitch lookup circuit breaker for %ss after %s consecutive failures",
            CIRCUIT_BREAK_SECONDS,
            _failure_count,
        )
        _failure_count = 0
    return None
