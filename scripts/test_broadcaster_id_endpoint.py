"""
Test broadcaster ID endpoint to diagnose the issue
"""
import asyncio
import httpx
import sys

async def test_broadcaster_id():
    """Test the broadcaster ID endpoint"""
    base_url = "http://localhost:8000"
    channel = "clix"
    
    print("=" * 80)
    print("Testing Broadcaster ID Endpoint")
    print("=" * 80)
    print(f"Channel: {channel}")
    print(f"Endpoint: {base_url}/api/get-broadcaster-id")
    print()
    
    async with httpx.AsyncClient(timeout=10.0) as client:
        # Test 1: Health check
        print("[1] Testing /health endpoint...")
        try:
            resp = await client.get(f"{base_url}/health")
            print(f"    Status: {resp.status_code}")
            if resp.status_code == 200:
                data = resp.json()
                print(f"    Service: {data.get('service')}")
                print(f"    Status: {data.get('status')}")
                broadcaster_lookup = data.get('broadcaster_id_lookup', {})
                print(f"    Broadcaster ID Lookup Status: {broadcaster_lookup.get('status')}")
                print(f"    Test Channel: {broadcaster_lookup.get('test_channel')}")
                print(f"    Test Broadcaster ID: {broadcaster_lookup.get('test_broadcaster_id')}")
            print()
        except Exception as e:
            print(f"    ERROR: {e}")
            print()
        
        # Test 2: Get broadcaster ID
        print(f"[2] Testing /api/get-broadcaster-id?channel_name={channel}...")
        try:
            resp = await client.get(
                f"{base_url}/api/get-broadcaster-id",
                params={"channel_name": channel}
            )
            print(f"    Status: {resp.status_code}")
            print(f"    Response: {resp.text}")
            
            if resp.status_code == 200:
                data = resp.json()
                print(f"    ✅ SUCCESS!")
                print(f"    Channel: {data.get('channel_name')}")
                print(f"    Broadcaster ID: {data.get('broadcaster_id')}")
            elif resp.status_code == 404:
                print(f"    ❌ Channel not found")
            elif resp.status_code == 500:
                print(f"    ❌ Server error")
            elif resp.status_code == 503:
                print(f"    ❌ Service unavailable")
            print()
        except httpx.HTTPStatusError as e:
            print(f"    HTTP Error: {e.response.status_code}")
            print(f"    Response: {e.response.text}")
            print()
        except Exception as e:
            print(f"    ERROR: {e}")
            print()
        
        # Test 3: Test with xqc (should work)
        print(f"[3] Testing /api/get-broadcaster-id?channel_name=xqc (for comparison)...")
        try:
            resp = await client.get(
                f"{base_url}/api/get-broadcaster-id",
                params={"channel_name": "xqc"}
            )
            print(f"    Status: {resp.status_code}")
            if resp.status_code == 200:
                data = resp.json()
                print(f"    ✅ SUCCESS!")
                print(f"    Channel: {data.get('channel_name')}")
                print(f"    Broadcaster ID: {data.get('broadcaster_id')}")
            else:
                print(f"    Response: {resp.text}")
            print()
        except Exception as e:
            print(f"    ERROR: {e}")
            print()

if __name__ == "__main__":
    asyncio.run(test_broadcaster_id())

