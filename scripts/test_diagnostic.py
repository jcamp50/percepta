"""
Diagnostic script to check what's happening with responses
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
TEST_CHANNEL = "awinewofi5783"
BOT_NAME = "percepta"

async def diagnostic_test():
    """Diagnostic test to see what's happening."""
    print("=" * 80)
    print("DIAGNOSTIC TEST")
    print("=" * 80)
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        # Step 1: Check service
        print("\n[1] Checking service health...")
        try:
            resp = await client.get(f"{BASE_URL}/health", timeout=5.0)
            print(f"Health check: {resp.status_code} - {resp.json()}")
        except Exception as e:
            print(f"Health check failed: {e}")
        
        # Step 2: Send question
        print("\n[2] Sending question...")
        username = "diag_user"
        question = f"@{BOT_NAME} what is Mystic Realm?"
        print(f"Question: {question}")
        
        try:
            resp = await client.post(
                f"{BASE_URL}/chat/message",
                json={
                    "channel": TEST_CHANNEL,
                    "username": username,
                    "message": question,
                    "timestamp": datetime.now().isoformat()
                },
                timeout=30.0
            )
            print(f"Send response: {resp.status_code}")
            if resp.status_code == 200:
                print(f"Send data: {resp.json()}")
        except Exception as e:
            print(f"Send failed: {e}")
            return
        
        # Step 3: Poll for responses
        print("\n[3] Polling for responses (25 rounds, 2s each)...")
        all_messages = []
        seen_keys = set()
        
        for round_num in range(25):
            try:
                send_resp = await client.post(
                    f"{BASE_URL}/chat/send",
                    json={"channel": TEST_CHANNEL},
                    timeout=10.0
                )
                
                if send_resp.status_code == 200:
                    data = send_resp.json()
                    msgs = data.get("messages", [])
                    print(f"\nRound {round_num + 1}: Found {len(msgs)} messages")
                    
                    for msg in msgs:
                        msg_key = (msg.get("reply_to"), msg.get("message")[:50])
                        if msg_key not in seen_keys:
                            seen_keys.add(msg_key)
                            all_messages.append(msg)
                            print(f"  New message for {msg.get('reply_to')}: {msg.get('message')[:100]}...")
                    
                    # Check if we got our user's response
                    user_msgs = [m for m in msgs if m.get("reply_to") == username]
                    if user_msgs:
                        print(f"\n[SUCCESS] Found response for {username}!")
                        print(f"Message: {user_msgs[0].get('message')}")
                        print(f"Full message: {json.dumps(user_msgs[0], indent=2)}")
                        return
                else:
                    print(f"Round {round_num + 1}: Error {send_resp.status_code} - {send_resp.text[:200]}")
            except Exception as e:
                print(f"Round {round_num + 1}: Exception - {e}")
            
            await asyncio.sleep(2.0)
        
        print(f"\n[4] Final summary:")
        print(f"Total unique messages seen: {len(all_messages)}")
        print(f"Messages for {username}: {len([m for m in all_messages if m.get('reply_to') == username])}")
        
        if all_messages:
            print("\nAll messages seen:")
            for msg in all_messages:
                print(f"  {msg.get('reply_to')}: {msg.get('message')[:100]}...")

if __name__ == "__main__":
    asyncio.run(diagnostic_test())

