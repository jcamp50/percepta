"""
Quick verification script to check if service is running and test basic functionality
"""

import asyncio
import httpx
import sys

BASE_URL = "http://localhost:8000"
TEST_CHANNEL = "testchannel123"

async def verify_service():
    """Verify service is running and basic functionality works."""
    print("=" * 80)
    print("Service Verification")
    print("=" * 80)
    
    async with httpx.AsyncClient(timeout=10.0) as client:
        # Check health
        try:
            resp = await client.get(f"{BASE_URL}/health")
            if resp.status_code == 200:
                print("[OK] Service is running")
                print(f"    Response: {resp.json()}")
            else:
                print(f"[X] Service returned status {resp.status_code}")
                return False
        except Exception as e:
            print(f"[X] Service not available: {e}")
            print("\nPlease start the service:")
            print("  uvicorn py.main:app --reload --port 8000")
            return False
        
        # Test a simple question
        print("\nTesting basic question...")
        try:
            resp = await client.post(
                f"{BASE_URL}/chat/message",
                json={
                    "channel": TEST_CHANNEL,
                    "username": "test_user",
                    "message": "@percepta what game are they playing?",
                    "timestamp": "2025-11-05T20:00:00Z"
                },
                timeout=30.0
            )
            if resp.status_code == 200:
                print("[OK] Message sent successfully")
                
                # Wait and check for response
                await asyncio.sleep(5)
                send_resp = await client.post(
                    f"{BASE_URL}/chat/send",
                    json={"channel": TEST_CHANNEL},
                    timeout=10.0
                )
                if send_resp.status_code == 200:
                    data = send_resp.json()
                    messages = data.get("messages", [])
                    if messages:
                        print(f"[OK] Received {len(messages)} response(s)")
                        for msg in messages:
                            print(f"    Response: {msg.get('message', '')[:100]}...")
                    else:
                        print("[!] No response yet (may be processing)")
            else:
                print(f"[X] Failed to send message: {resp.status_code}")
        except Exception as e:
            print(f"[X] Error testing question: {e}")
    
    print("\n" + "=" * 80)
    print("Verification complete!")
    print("=" * 80)
    return True

if __name__ == "__main__":
    success = asyncio.run(verify_service())
    sys.exit(0 if success else 1)

