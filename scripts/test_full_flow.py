"""
Test to verify the full message flow with detailed logging
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

async def test_full_flow():
    """Test the full message flow with detailed checks."""
    print("=" * 80)
    print("FULL MESSAGE FLOW TEST")
    print("=" * 80)
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        username = "flow_test_user"
        question = f"@{BOT_NAME} what is Mystic Realm?"
        
        print(f"\n[1] Sending message to {BASE_URL}/chat/message")
        print(f"    Channel: {TEST_CHANNEL}")
        print(f"    Username: {username}")
        print(f"    Question: {question}")
        
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
            print(f"\n[2] Response status: {resp.status_code}")
            if resp.status_code == 200:
                data = resp.json()
                print(f"    Response data: {json.dumps(data, indent=2)}")
            else:
                print(f"    Error: {resp.text}")
                return
        except Exception as e:
            print(f"    Exception: {e}")
            return
        
        print(f"\n[3] Waiting 20 seconds for RAG processing...")
        await asyncio.sleep(20)
        
        print(f"\n[4] Checking message queue via {BASE_URL}/chat/send")
        try:
            send_resp = await client.post(
                f"{BASE_URL}/chat/send",
                json={"channel": TEST_CHANNEL},
                timeout=10.0
            )
            print(f"    Response status: {send_resp.status_code}")
            if send_resp.status_code == 200:
                data = send_resp.json()
                print(f"    Response data: {json.dumps(data, indent=2)}")
                msgs = data.get("messages", [])
                print(f"\n    Found {len(msgs)} messages")
                for i, msg in enumerate(msgs, 1):
                    print(f"\n    Message {i}:")
                    print(f"      Reply to: {msg.get('reply_to')}")
                    print(f"      Message: {msg.get('message')[:200]}...")
                    print(f"      Full: {json.dumps(msg, indent=6)}")
            else:
                print(f"    Error: {send_resp.text}")
        except Exception as e:
            print(f"    Exception: {e}")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    asyncio.run(test_full_flow())

