"""
Simple API test to verify responses with seeded channel
"""

import asyncio
import httpx
import json
import sys
import codecs
from datetime import datetime

if sys.platform == "win32":
    sys.stdout = codecs.getwriter("utf-8")(sys.stdout.buffer, "strict")
    sys.stderr = codecs.getwriter("utf-8")(sys.stderr.buffer, "strict")

BASE_URL = "http://localhost:8000"
TEST_CHANNEL = "testchannel123"
BOT_NAME = "percepta"

async def simple_test():
    """Simple test of API with seeded channel."""
    print("=" * 80)
    print("Simple API Test - Seeded Channel")
    print("=" * 80)
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        # Test 1: Question about Mystic Realm
        print("\n[Test 1] Question: What is Mystic Realm?")
        try:
            resp = await client.post(
                f"{BASE_URL}/chat/message",
                json={
                    "channel": TEST_CHANNEL,
                    "username": "test_user1",
                    "message": f"@{BOT_NAME} what is Mystic Realm?",
                    "timestamp": datetime.now().isoformat()
                }
            )
            print(f"Status: {resp.status_code}")
            
            # Wait and check response
            await asyncio.sleep(8)
            send_resp = await client.post(
                f"{BASE_URL}/chat/send",
                json={"channel": TEST_CHANNEL}
            )
            if send_resp.status_code == 200:
                data = send_resp.json()
                msgs = data.get("messages", [])
                if msgs:
                    answer = msgs[0].get("message", "")
                    print(f"Answer: {answer[:200]}...")
                    if "mystic realm" in answer.lower() or "puzzle" in answer.lower():
                        print("[OK] Answer references transcript content!")
                    elif "don't have enough context" in answer.lower():
                        print("[!] Got 'no context' response")
                    else:
                        print(f"[?] Answer: {answer}")
                else:
                    print("[!] No messages in queue")
        except Exception as e:
            print(f"[X] Error: {e}")
        
        # Test 2: Question about graphics card
        print("\n[Test 2] Question: What graphics card are they using?")
        await asyncio.sleep(2)
        try:
            resp = await client.post(
                f"{BASE_URL}/chat/message",
                json={
                    "channel": TEST_CHANNEL,
                    "username": "test_user2",
                    "message": f"@{BOT_NAME} what graphics card are they using?",
                    "timestamp": datetime.now().isoformat()
                }
            )
            print(f"Status: {resp.status_code}")
            
            await asyncio.sleep(8)
            send_resp = await client.post(
                f"{BASE_URL}/chat/send",
                json={"channel": TEST_CHANNEL}
            )
            if send_resp.status_code == 200:
                data = send_resp.json()
                msgs = data.get("messages", [])
                if msgs:
                    for msg in msgs:
                        if msg.get("reply_to") == "test_user2":
                            answer = msg.get("message", "")
                            print(f"Answer: {answer[:200]}...")
                            if "rtx" in answer.lower() or "4090" in answer.lower():
                                print("[OK] Answer references RTX 4090!")
                            elif "don't have enough context" in answer.lower():
                                print("[!] Got 'no context' response")
                            break
        except Exception as e:
            print(f"[X] Error: {e}")
        
        # Test 3: Follow-up question
        print("\n[Test 3] Follow-up: What type of game is it?")
        await asyncio.sleep(12)  # Wait for rate limit
        try:
            resp = await client.post(
                f"{BASE_URL}/chat/message",
                json={
                    "channel": TEST_CHANNEL,
                    "username": "test_user1",
                    "message": f"@{BOT_NAME} what type of game is it?",
                    "timestamp": datetime.now().isoformat()
                }
            )
            print(f"Status: {resp.status_code}")
            
            await asyncio.sleep(8)
            send_resp = await client.post(
                f"{BASE_URL}/chat/send",
                json={"channel": TEST_CHANNEL}
            )
            if send_resp.status_code == 200:
                data = send_resp.json()
                msgs = data.get("messages", [])
                if msgs:
                    for msg in msgs:
                        if msg.get("reply_to") == "test_user1":
                            answer = msg.get("message", "")
                            print(f"Answer: {answer[:200]}...")
                            if "puzzle" in answer.lower() or "adventure" in answer.lower() or "mystic realm" in answer.lower():
                                print("[OK] Follow-up uses context from previous answer!")
                            elif "don't have enough context" in answer.lower():
                                print("[!] Follow-up got 'no context' - session history may not be working")
                            break
        except Exception as e:
            print(f"[X] Error: {e}")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    asyncio.run(simple_test())

