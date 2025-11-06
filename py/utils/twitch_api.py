"""
Shared Twitch API utilities
"""
from typing import Optional
import httpx
from py.config import settings
from py.utils.logging import get_logger

logger = get_logger(__name__, category="system")


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
    
    # Create HTTP client with headers (same pattern as metadata poller)
    async with httpx.AsyncClient(
        timeout=30.0,
        headers={
            "Client-ID": client_id,
            "Authorization": f"Bearer {access_token}",
        },
    ) as client:
        url = "https://api.twitch.tv/helix/users"
        params = {"login": channel_name.lower()}
        
        try:
            response = await client.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            users = data.get("data", [])
            
            if users:
                broadcaster_id = users[0].get("id")
                logger.info(f"Found broadcaster ID for {channel_name}: {broadcaster_id}")
                return broadcaster_id
            else:
                logger.warning(f"Channel not found: {channel_name}")
                return None
                
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error getting broadcaster ID: {e.response.status_code} - {e.response.text}")
            return None
        except Exception as e:
            logger.error(f"Failed to get broadcaster ID: {e}", exc_info=True)
            return None

