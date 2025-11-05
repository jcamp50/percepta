"""
Detailed test to verify follow-up questions work with session history
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

async def test_followup_detailed():
    """Detailed test of follow-up question functionality."""
    print("=" * 80)
    print("Follow-up Question Verification Test")
    print("=" * 80)
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        user = "followup_test_user"
        
        # First question
        print("\n[Step 1] Sending first question...")
        q1 = f"@{BOT_NAME} what game are they playing?"
        print(f"Question: {q1}")
        
        resp1 = await client.post(
            f"{BASE_URL}/chat/message",
            json={
                "channel": TEST_CHANNEL,
                "username": user,
                "message": q1,
                "timestamp": datetime.now().isoformat()
            }
        )
        print(f"Status: {resp1.status_code}")
        
        # Wait for response
        print("\n[Step 2] Waiting for response...")
        await asyncio.sleep(8)
        
        send_resp1 = await client.post(
            f"{BASE_URL}/chat/send",
            json={"channel": TEST_CHANNEL}
        )
        
        if send_resp1.status_code == 200:
            data1 = send_resp1.json()
            msgs1 = [m for m in data1.get("messages", []) if m.get("reply_to") == user]
            if msgs1:
                answer1 = msgs1[0].get("message")
                print(f"Answer 1: {answer1}")
                print(f"Length: {len(answer1)} chars")
            else:
                print("No response received yet")
                return
        
        # Wait for rate limit
        print("\n[Step 3] Waiting for rate limit (12s)...")
        await asyncio.sleep(12)
        
        # Second question (follow-up)
        print("\n[Step 4] Sending follow-up question...")
        q2 = f"@{BOT_NAME} tell me more about that game"
        print(f"Question: {q2}")
        
        resp2 = await client.post(
            f"{BASE_URL}/chat/message",
            json={
                "channel": TEST_CHANNEL,
                "username": user,
                "message": q2,
                "timestamp": datetime.now().isoformat()
            }
        )
        print(f"Status: {resp2.status_code}")
        
        # Wait for response
        print("\n[Step 5] Waiting for follow-up response...")
        await asyncio.sleep(8)
        
        send_resp2 = await client.post(
            f"{BASE_URL}/chat/send",
            json={"channel": TEST_CHANNEL}
        )
        
        if send_resp2.status_code == 200:
            data2 = send_resp2.json()
            msgs2 = [m for m in data2.get("messages", []) if m.get("reply_to") == user]
            if msgs2:
                answer2 = msgs2[0].get("message")
                print(f"Answer 2: {answer2}")
                print(f"Length: {len(answer2)} chars")
                
                # Check if answers are different (should be)
                if answer1 != answer2:
                    print("\n[OK] Follow-up answer is different from first answer")
                else:
                    print("\n[!] Follow-up answer is identical to first answer")
                
                # Check if answer seems contextually aware
                if len(answer2) > 50 and "don't have enough context" not in answer2.lower():
                    print("[OK] Follow-up answer appears contextually aware")
                else:
                    print("[!] Follow-up answer may not be using session history")
            else:
                print("No follow-up response received")
        
        # Check Redis session
        print("\n[Step 6] Checking Redis session...")
        try:
            import redis.asyncio as redis
            from py.config import settings
            
            redis_client = redis.from_url(
                f"redis://{settings.redis_host}:{settings.redis_port}/{settings.redis_db}",
                decode_responses=True
            )
            await redis_client.ping()
            
            session_key = f"session:{TEST_CHANNEL}:{user}"
            session_data = await redis_client.get(session_key)
            
            if session_data:
                import json
                session = json.loads(session_data)
                print(f"[OK] Session found in Redis")
                print(f"    Questions: {len(session.get('last_questions', []))}")
                print(f"    Answers: {len(session.get('last_answers', []))}")
                if session.get('last_questions'):
                    print(f"    Last question: {session['last_questions'][-1]}")
                if session.get('last_answers'):
                    print(f"    Last answer: {session['last_answers'][-1][:100]}...")
            else:
                print("[!] No session found in Redis")
            
            await redis_client.aclose()
        except Exception as e:
            print(f"[!] Error checking Redis: {e}")
    
    print("\n" + "=" * 80)
    print("Test Complete")
    print("=" * 80)

if __name__ == "__main__":
    asyncio.run(test_followup_detailed())

