"""
Test using the exact same method as metadata poller
"""
import asyncio
import httpx
import os
from dotenv import load_dotenv

load_dotenv()

async def test_metadata_poller_method():
    """Test using exact same pattern as metadata poller"""
    print("=" * 70)
    print("TESTING WITH METADATA POLLER EXACT PATTERN")
    print("=" * 70)
    
    # Get credentials exactly like metadata poller
    client_id = os.getenv("TWITCH_CLIENT_ID")
    access_token = os.getenv("TWITCH_BOT_TOKEN")
    
    # Remove 'oauth:' prefix if present (same as metadata poller)
    if access_token and access_token.startswith("oauth:"):
        access_token = access_token[6:]
    
    print(f"\nCredentials:")
    print(f"  Client-ID: {client_id[:15]}... (length: {len(client_id)})")
    print(f"  Token: {access_token[:15]}... (length: {len(access_token)})")
    
    # Create HTTP client exactly like metadata poller
    http_client = httpx.AsyncClient(
        timeout=30.0,
        headers={
            "Client-ID": client_id,
            "Authorization": f"Bearer {access_token}",
        },
    )
    
    # Make request exactly like metadata poller
    url = "https://api.twitch.tv/helix/users"
    params = {"login": "xqc"}
    
    print(f"\nMaking request:")
    print(f"  URL: {url}")
    print(f"  Params: {params}")
    print(f"  Headers: Client-ID={client_id[:15]}..., Authorization=Bearer {access_token[:15]}...")
    
    try:
        response = await http_client.get(url, params=params)
        response.raise_for_status()
        
        data = response.json()
        users = data.get("data", [])
        
        print(f"\nResponse:")
        print(f"  Status: {response.status_code}")
        print(f"  Data: {data}")
        
        if users:
            broadcaster_id = users[0].get("id")
            print(f"\n[OK] SUCCESS: broadcaster_id = {broadcaster_id}")
        else:
            print(f"\n[FAIL] Empty users array")
    except Exception as e:
        print(f"\n[ERROR] {type(e).__name__}: {e}")
    finally:
        await http_client.aclose()
    
    print("=" * 70)

if __name__ == "__main__":
    asyncio.run(test_metadata_poller_method())

