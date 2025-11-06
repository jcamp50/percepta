"""
Test script that requires Python service to be running
Tests endpoint and captures detailed information
"""
import asyncio
import httpx
import os
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

async def test_with_service():
    """Test endpoint with service running"""
    print("=" * 70)
    print("TESTING WITH LIVE PYTHON SERVICE")
    print("=" * 70)
    print("\nMake sure Python service is running on http://localhost:8000")
    print("Press Enter to continue...")
    input()
    
    # Test 1: Debug credentials endpoint
    print("\n1. Testing /api/debug-credentials endpoint...")
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get("http://localhost:8000/api/debug-credentials")
            if response.status_code == 200:
                creds = response.json()
                print(f"   [OK] Credentials loaded:")
                print(f"   - Client-ID present: {creds.get('client_id_present')}")
                print(f"   - Client-ID length: {creds.get('client_id_length')}")
                print(f"   - Client-ID prefix: {creds.get('client_id_prefix')}")
                print(f"   - Token present: {creds.get('bot_token_present')}")
                print(f"   - Token length: {creds.get('bot_token_length')}")
                print(f"   - Token prefix: {creds.get('bot_token_prefix')}")
                print(f"   - Has oauth: prefix: {creds.get('bot_token_has_oauth_prefix')}")
            else:
                print(f"   [FAIL] Status {response.status_code}: {response.text}")
    except Exception as e:
        print(f"   [ERROR] {e}")
    
    # Test 2: Get broadcaster ID endpoint
    print("\n2. Testing /api/get-broadcaster-id endpoint...")
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                "http://localhost:8000/api/get-broadcaster-id",
                params={"channel_name": "xqc"}
            )
            print(f"   Status: {response.status_code}")
            print(f"   Response: {response.text[:200]}")
            
            if response.status_code == 200:
                data = response.json()
                print(f"   [OK] broadcaster_id = {data.get('broadcaster_id')}")
            else:
                print(f"   [FAIL] Error response")
    except Exception as e:
        print(f"   [ERROR] {e}")
    
    # Test 3: Direct Twitch API for comparison
    print("\n3. Testing direct Twitch API (for comparison)...")
    client_id = os.getenv("TWITCH_CLIENT_ID")
    bot_token = os.getenv("TWITCH_BOT_TOKEN")
    if bot_token.startswith("oauth:"):
        bot_token = bot_token[6:]
    
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                "https://api.twitch.tv/helix/users",
                params={"login": "xqc"},
                headers={
                    "Client-ID": client_id,
                    "Authorization": f"Bearer {bot_token}",
                }
            )
            if response.status_code == 200:
                data = response.json()
                if data.get("data"):
                    print(f"   [OK] broadcaster_id = {data['data'][0]['id']}")
                else:
                    print(f"   [FAIL] Empty data array")
            else:
                print(f"   [FAIL] Status {response.status_code}")
    except Exception as e:
        print(f"   [ERROR] {e}")
    
    print("\n" + "=" * 70)

if __name__ == "__main__":
    asyncio.run(test_with_service())

