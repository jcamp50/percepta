"""
Test script for broadcaster ID endpoint
Tests the Python endpoint directly to diagnose issues
"""
import asyncio
import httpx
import sys
from datetime import datetime

TEST_LOG_FILE = "test_broadcaster_id.log"

def log(message):
    """Log to both console and file"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_message = f"[{timestamp}] {message}"
    # Replace Unicode characters for Windows console compatibility
    safe_message = log_message.replace("✓", "[OK]").replace("✗", "[FAIL]")
    print(safe_message)
    with open(TEST_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(log_message + "\n")

async def test_python_endpoint():
    """Test the Python /api/get-broadcaster-id endpoint"""
    log("=" * 60)
    log("TESTING PYTHON ENDPOINT: /api/get-broadcaster-id")
    log("=" * 60)
    
    url = "http://localhost:8000/api/get-broadcaster-id"
    params = {"channel_name": "xqc"}
    
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            log(f"Making request to: {url}?channel_name=xqc")
            response = await client.get(url, params=params)
            
            log(f"Response Status: {response.status_code}")
            log(f"Response Headers: {dict(response.headers)}")
            log(f"Response Body: {response.text}")
            
            if response.status_code == 200:
                data = response.json()
                log(f"[OK] SUCCESS: broadcaster_id = {data.get('broadcaster_id')}")
                return True
            else:
                log(f"[FAIL] FAILED: Status {response.status_code}")
                return False
                
    except httpx.ConnectError as e:
        log(f"[FAIL] CONNECTION ERROR: Cannot connect to Python service: {e}")
        log("  Make sure Python service is running on http://localhost:8000")
        return False
    except Exception as e:
        log(f"[FAIL] ERROR: {type(e).__name__}: {e}")
        return False

async def test_twitch_api_direct():
    """Test Twitch API directly (like Postman)"""
    log("=" * 60)
    log("TESTING TWITCH API DIRECTLY")
    log("=" * 60)
    
    # Read credentials from .env file
    import os
    from dotenv import load_dotenv
    load_dotenv()
    
    client_id = os.getenv("TWITCH_CLIENT_ID")
    bot_token = os.getenv("TWITCH_BOT_TOKEN")
    
    if not client_id or not bot_token:
        log("[FAIL] ERROR: TWITCH_CLIENT_ID or TWITCH_BOT_TOKEN not found in .env")
        return False
    
    # Remove oauth: prefix if present
    if bot_token.startswith("oauth:"):
        bot_token = bot_token[6:]
    
    log(f"Client-ID: {client_id[:10]}...")
    log(f"Token length: {len(bot_token)}")
    
    url = "https://api.twitch.tv/helix/users"
    params = {"login": "xqc"}
    headers = {
        "Client-ID": client_id,
        "Authorization": f"Bearer {bot_token}",
    }
    
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            log(f"Making request to: {url}?login=xqc")
            log(f"Headers: Client-ID={client_id[:10]}..., Authorization=Bearer {bot_token[:10]}...")
            
            response = await client.get(url, params=params, headers=headers)
            
            log(f"Response Status: {response.status_code}")
            log(f"Response Body: {response.text}")
            
            if response.status_code == 200:
                data = response.json()
                users = data.get("data", [])
                if users:
                    broadcaster_id = users[0].get("id")
                    log(f"[OK] SUCCESS: broadcaster_id = {broadcaster_id}")
                    return True
                else:
                    log(f"[FAIL] FAILED: Empty users array in response")
                    return False
            else:
                log(f"[FAIL] FAILED: Status {response.status_code}")
                return False
                
    except Exception as e:
        log(f"[FAIL] ERROR: {type(e).__name__}: {e}")
        return False

async def main():
    """Run all tests"""
    log("Starting broadcaster ID diagnostic tests...")
    log("")
    
    # Test 1: Direct Twitch API (should work like Postman)
    twitch_ok = await test_twitch_api_direct()
    log("")
    
    # Test 2: Python endpoint
    python_ok = await test_python_endpoint()
    log("")
    
    # Summary
    log("=" * 60)
    log("TEST SUMMARY")
    log("=" * 60)
    log(f"Twitch API Direct: {'[OK] PASS' if twitch_ok else '[FAIL] FAIL'}")
    log(f"Python Endpoint: {'[OK] PASS' if python_ok else '[FAIL] FAIL'}")
    
    if twitch_ok and not python_ok:
        log("")
        log("DIAGNOSIS: Twitch API works, but Python endpoint fails.")
        log("  This suggests an issue with the Python endpoint implementation.")
    elif not twitch_ok:
        log("")
        log("DIAGNOSIS: Twitch API fails. Check credentials.")
    elif twitch_ok and python_ok:
        log("")
        log("DIAGNOSIS: Both tests pass! Issue may be timing-related.")
    
    log("=" * 60)

if __name__ == "__main__":
    # Clear log file
    with open(TEST_LOG_FILE, "w", encoding="utf-8") as f:
        f.write("")
    
    asyncio.run(main())

