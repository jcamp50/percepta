"""
Get broadcaster ID for a channel name
"""

import asyncio
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from py.ingest.twitch import EventSubWebSocketClient
from py.config import settings

async def get_broadcaster_id(channel_name: str):
    """Get broadcaster ID for a channel name."""
    client = EventSubWebSocketClient(
        client_id=settings.twitch_client_id,
        access_token=settings.twitch_bot_token,
        target_channel=channel_name
    )
    broadcaster_id = await client._get_broadcaster_id(channel_name)
    if broadcaster_id:
        print(f"Channel: {channel_name}")
        print(f"Broadcaster ID: {broadcaster_id}")
        return broadcaster_id
    else:
        print(f"Could not find broadcaster ID for {channel_name}")
        return None

if __name__ == "__main__":
    channel = sys.argv[1] if len(sys.argv) > 1 else "awinewofi5783"
    result = asyncio.run(get_broadcaster_id(channel))
    if result:
        print(f"\nUse this broadcaster ID for seeding: {result}")

