"""
Comprehensive follow-up question test with detailed verification
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

async def comprehensive_followup_test():
    """Comprehensive test of follow-up questions with session history."""
    print("=" * 80)
    print("Comprehensive Follow-up Question Test")
    print("=" * 80)
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        user = "comprehensive_followup_user"
        
        # Step 1: First question about Mystic Realm
        print("\n[Step 1] First Question: What is Mystic Realm?")
        q1 = f"@{BOT_NAME} what is Mystic Realm?"
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
        
        # Wait and get response
        print("\n[Step 2] Waiting for response (10s)...")
        await asyncio.sleep(10)
        
        for poll in range(10):
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
                    break
            await asyncio.sleep(1)
        else:
            print("[X] No response received for first question")
            return
        
        # Check if answer mentions Mystic Realm
        if "mystic realm" in answer1.lower() or "puzzle" in answer1.lower():
            print("[OK] First answer references Mystic Realm!")
        else:
            print(f"[!] First answer doesn't mention Mystic Realm: {answer1[:100]}...")
        
        # Step 3: Wait for rate limit
        print("\n[Step 3] Waiting for rate limit (12s)...")
        await asyncio.sleep(12)
        
        # Step 4: Follow-up question
        print("\n[Step 4] Follow-up Question: What type of game is it?")
        q2 = f"@{BOT_NAME} what type of game is it?"
        print(f"Question: {q2}")
        print("Expected: Should reference 'puzzle adventure' from previous answer")
        
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
        
        # Wait and get follow-up response
        print("\n[Step 5] Waiting for follow-up response (10s)...")
        await asyncio.sleep(10)
        
        for poll in range(10):
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
                    break
            await asyncio.sleep(1)
        else:
            print("[X] No follow-up response received")
            return
        
        # Analyze follow-up answer
        answer2_lower = answer2.lower()
        print("\n[Step 6] Analyzing Follow-up Answer:")
        
        if "mystic realm" in answer2_lower or "puzzle" in answer2_lower or "adventure" in answer2_lower:
            print("[OK] Follow-up mentions game details from previous answer!")
            print("[OK] Session history appears to be working!")
        elif "don't have enough context" in answer2_lower:
            print("[X] Follow-up returned 'no context'")
            print("[!] Session history may not be included in prompt")
        else:
            print(f"[?] Follow-up answer: {answer2[:200]}...")
        
        # Step 7: Check Redis session
        print("\n[Step 7] Checking Redis Session:")
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
                session = json.loads(session_data)
                print(f"[OK] Session found in Redis")
                print(f"    Questions: {len(session.get('last_questions', []))}")
                print(f"    Answers: {len(session.get('last_answers', []))}")
                
                if session.get('last_questions'):
                    print(f"\n    Last Questions:")
                    for i, q in enumerate(session['last_questions'], 1):
                        print(f"      {i}. {q}")
                
                if session.get('last_answers'):
                    print(f"\n    Last Answers:")
                    for i, a in enumerate(session['last_answers'], 1):
                        print(f"      {i}. {a[:100]}...")
                
                # Check if first answer mentions Mystic Realm
                if session.get('last_answers'):
                    first_answer = session['last_answers'][0].lower()
                    if "mystic realm" in first_answer or "puzzle" in first_answer:
                        print("\n[OK] Session history contains relevant context!")
                    else:
                        print("\n[!] Session history may not contain relevant context")
            else:
                print("[X] No session found in Redis")
            
            await redis_client.aclose()
        except Exception as e:
            print(f"[!] Error checking Redis: {e}")
    
    print("\n" + "=" * 80)
    print("Test Complete")
    print("=" * 80)

if __name__ == "__main__":
    asyncio.run(comprehensive_followup_test())

