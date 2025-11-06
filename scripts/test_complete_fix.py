"""
Comprehensive test to verify broadcaster ID fix works end-to-end
"""
import asyncio
import httpx
import os
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

TEST_LOG_FILE = "test_complete_fix.log"

def log(message):
    """Log to both console and file"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_message = f"[{timestamp}] {message}"
    safe_message = log_message.replace("✓", "[OK]").replace("✗", "[FAIL]")
    print(safe_message)
    with open(TEST_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(log_message + "\n")

async def test_shared_utility():
    """Test the shared utility function directly"""
    log("=" * 70)
    log("TEST 1: Shared Utility Function")
    log("=" * 70)
    
    try:
        from py.utils.twitch_api import get_broadcaster_id_from_channel_name
        broadcaster_id = await get_broadcaster_id_from_channel_name("xqc")
        
        if broadcaster_id == "71092938":
            log("[OK] Shared utility function works correctly")
            log(f"    broadcaster_id = {broadcaster_id}")
            return True
        else:
            log(f"[FAIL] Unexpected broadcaster_id: {broadcaster_id}")
            return False
    except Exception as e:
        log(f"[FAIL] Error testing shared utility: {e}")
        return False

async def test_python_endpoint():
    """Test the Python endpoint"""
    log("")
    log("=" * 70)
    log("TEST 2: Python Endpoint")
    log("=" * 70)
    
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                "http://localhost:8000/api/get-broadcaster-id",
                params={"channel_name": "xqc"}
            )
            
            log(f"Response Status: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                broadcaster_id = data.get("broadcaster_id")
                if broadcaster_id == "71092938":
                    log("[OK] Python endpoint works correctly")
                    log(f"    broadcaster_id = {broadcaster_id}")
                    return True
                else:
                    log(f"[FAIL] Unexpected broadcaster_id: {broadcaster_id}")
                    return False
            else:
                log(f"[FAIL] Endpoint returned status {response.status_code}")
                log(f"    Response: {response.text[:200]}")
                return False
    except httpx.ConnectError:
        log("[FAIL] Cannot connect to Python service at http://localhost:8000")
        log("    Make sure the Python service is running")
        return False
    except Exception as e:
        log(f"[FAIL] Error testing endpoint: {e}")
        return False

async def test_debug_endpoint():
    """Test the debug credentials endpoint"""
    log("")
    log("=" * 70)
    log("TEST 3: Debug Credentials Endpoint")
    log("=" * 70)
    
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get("http://localhost:8000/api/debug-credentials")
            
            if response.status_code == 200:
                creds = response.json()
                log("[OK] Debug endpoint accessible")
                log(f"    Client-ID present: {creds.get('client_id_present')}")
                log(f"    Token present: {creds.get('bot_token_present')}")
                log(f"    Token length: {creds.get('bot_token_length')}")
                return True
            else:
                log(f"[FAIL] Debug endpoint returned status {response.status_code}")
                return False
    except Exception as e:
        log(f"[FAIL] Error testing debug endpoint: {e}")
        return False

async def test_direct_twitch_api():
    """Test direct Twitch API for comparison"""
    log("")
    log("=" * 70)
    log("TEST 4: Direct Twitch API (Baseline)")
    log("=" * 70)
    
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
                    broadcaster_id = data["data"][0]["id"]
                    if broadcaster_id == "71092938":
                        log("[OK] Direct Twitch API works (baseline)")
                        return True
                    else:
                        log(f"[FAIL] Unexpected broadcaster_id: {broadcaster_id}")
                        return False
                else:
                    log("[FAIL] Empty data array from Twitch API")
                    return False
            else:
                log(f"[FAIL] Twitch API returned status {response.status_code}")
                return False
    except Exception as e:
        log(f"[FAIL] Error testing direct Twitch API: {e}")
        return False

async def main():
    """Run all tests"""
    # Clear log file
    with open(TEST_LOG_FILE, "w", encoding="utf-8") as f:
        f.write("")
    
    log("Starting comprehensive broadcaster ID fix verification...")
    log("")
    
    results = {}
    
    # Test 1: Shared utility function
    results["shared_utility"] = await test_shared_utility()
    
    # Test 2: Python endpoint (requires service to be running)
    results["python_endpoint"] = await test_python_endpoint()
    
    # Test 3: Debug endpoint
    results["debug_endpoint"] = await test_debug_endpoint()
    
    # Test 4: Direct Twitch API
    results["direct_api"] = await test_direct_twitch_api()
    
    # Summary
    log("")
    log("=" * 70)
    log("TEST SUMMARY")
    log("=" * 70)
    log(f"Shared Utility Function: {'[OK] PASS' if results.get('shared_utility') else '[FAIL] FAIL'}")
    log(f"Python Endpoint: {'[OK] PASS' if results.get('python_endpoint') else '[FAIL] FAIL'}")
    log(f"Debug Endpoint: {'[OK] PASS' if results.get('debug_endpoint') else '[FAIL] FAIL'}")
    log(f"Direct Twitch API: {'[OK] PASS' if results.get('direct_api') else '[FAIL] FAIL'}")
    log("")
    
    all_passed = all(results.values())
    if all_passed:
        log("[OK] ALL TESTS PASSED! The fix is working correctly.")
    else:
        log("[FAIL] Some tests failed. Check the logs above for details.")
        if not results.get("python_endpoint"):
            log("")
            log("NOTE: If Python endpoint test failed, make sure:")
            log("  1. Python service is running on http://localhost:8000")
            log("  2. Service has been restarted to load the new code")
            log("  3. Check Python service logs for any errors")
    
    log("=" * 70)
    log(f"Full test log saved to: {TEST_LOG_FILE}")

if __name__ == "__main__":
    asyncio.run(main())

