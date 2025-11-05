"""
Manual test script to verify rate limiting and safety features
Uses direct API calls to test functionality
"""

import asyncio
import httpx
import json
from datetime import datetime

BASE_URL = "http://localhost:8000"
TEST_CHANNEL = "testchannel"

async def test_basic_flow():
    """Test basic message flow."""
    print("=" * 80)
    print("Testing Rate Limiting & Safety Features")
    print("=" * 80)
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        # Test 1: Check health
        print("\n1. Testing health endpoint...")
        try:
            resp = await client.get(f"{BASE_URL}/health")
            print(f"   Status: {resp.status_code}")
            print(f"   Response: {json.dumps(resp.json(), indent=2)}")
        except Exception as e:
            print(f"   ERROR: {e}")
            return
        
        # Test 2: Send a normal question
        print("\n2. Testing normal question...")
        msg1 = {
            "channel": TEST_CHANNEL,
            "username": "testuser1",
            "message": "@percepta what is the meaning of life?",
            "timestamp": datetime.now().isoformat()
        }
        resp = await client.post(f"{BASE_URL}/chat/message", json=msg1)
        print(f"   Status: {resp.status_code}")
        await asyncio.sleep(1)
        
        # Get queued messages
        send_resp = await client.post(f"{BASE_URL}/chat/send", json={"channel": TEST_CHANNEL})
        if send_resp.status_code == 200:
            data = send_resp.json()
            print(f"   Queued messages: {len(data.get('messages', []))}")
            if data.get('messages'):
                print(f"   First message: {data['messages'][0]['message'][:100]}...")
        
        # Test 3: Test rate limiting (send immediately again)
        print("\n3. Testing per-user rate limit (should be blocked)...")
        msg2 = {
            "channel": TEST_CHANNEL,
            "username": "testuser1",  # Same user
            "message": "@percepta what is the weather?",
            "timestamp": datetime.now().isoformat()
        }
        resp = await client.post(f"{BASE_URL}/chat/message", json=msg2)
        print(f"   Status: {resp.status_code}")
        await asyncio.sleep(1)
        
        send_resp = await client.post(f"{BASE_URL}/chat/send", json={"channel": TEST_CHANNEL})
        if send_resp.status_code == 200:
            data = send_resp.json()
            print(f"   Queued messages: {len(data.get('messages', []))} (should be 0 if rate limited)")
        
        # Test 4: Test PII filtering
        print("\n4. Testing PII filtering (should be blocked)...")
        msg3 = {
            "channel": TEST_CHANNEL,
            "username": "testuser2",
            "message": "@percepta my email is test@example.com",
            "timestamp": datetime.now().isoformat()
        }
        resp = await client.post(f"{BASE_URL}/chat/message", json=msg3)
        print(f"   Status: {resp.status_code}")
        await asyncio.sleep(1)
        
        send_resp = await client.post(f"{BASE_URL}/chat/send", json={"channel": TEST_CHANNEL})
        if send_resp.status_code == 200:
            data = send_resp.json()
            print(f"   Queued messages: {len(data.get('messages', []))} (should be 0 if filtered)")
        
        # Test 5: Test admin commands (if admin_users is configured)
        print("\n5. Testing admin commands...")
        print("   Note: Admin commands require ADMIN_USERS env var to be set")
        msg4 = {
            "channel": TEST_CHANNEL,
            "username": "admin_user",
            "message": "!status",
            "timestamp": datetime.now().isoformat()
        }
        resp = await client.post(f"{BASE_URL}/chat/message", json=msg4)
        print(f"   Status: {resp.status_code}")
        await asyncio.sleep(0.5)
        
        send_resp = await client.post(f"{BASE_URL}/chat/send", json={"channel": TEST_CHANNEL})
        if send_resp.status_code == 200:
            data = send_resp.json()
            print(f"   Queued messages: {len(data.get('messages', []))}")
            if data.get('messages'):
                print(f"   Admin response: {data['messages'][0]['message']}")
        
        # Test 6: Test repeated question cooldown
        print("\n6. Testing repeated question cooldown...")
        question = "@percepta what is Python?"
        msg5 = {
            "channel": TEST_CHANNEL,
            "username": "testuser3",
            "message": question,
            "timestamp": datetime.now().isoformat()
        }
        resp = await client.post(f"{BASE_URL}/chat/message", json=msg5)
        print(f"   First question - Status: {resp.status_code}")
        await asyncio.sleep(1)
        
        # Send same question again
        msg6 = {
            "channel": TEST_CHANNEL,
            "username": "testuser3",
            "message": question,
            "timestamp": datetime.now().isoformat()
        }
        resp = await client.post(f"{BASE_URL}/chat/message", json=msg6)
        print(f"   Repeated question - Status: {resp.status_code}")
        await asyncio.sleep(1)
        
        send_resp = await client.post(f"{BASE_URL}/chat/send", json={"channel": TEST_CHANNEL})
        if send_resp.status_code == 200:
            data = send_resp.json()
            print(f"   Queued messages: {len(data.get('messages', []))} (should be 1, second blocked)")
        
        print("\n" + "=" * 80)
        print("Testing complete!")
        print("=" * 80)

if __name__ == "__main__":
    asyncio.run(test_basic_flow())

