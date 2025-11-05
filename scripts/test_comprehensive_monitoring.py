"""
Comprehensive test with service stability monitoring and sustained load testing
"""

import asyncio
import httpx
import json
import sys
import codecs
import time
from datetime import datetime
from typing import List, Dict

if sys.platform == "win32":
    sys.stdout = codecs.getwriter("utf-8")(sys.stdout.buffer, "strict")
    sys.stderr = codecs.getwriter("utf-8")(sys.stderr.buffer, "strict")

BASE_URL = "http://localhost:8000"
TEST_CHANNEL = "awinewofi5783"
BOT_NAME = "percepta"

async def check_service_health(client: httpx.AsyncClient) -> Dict:
    """Check service health and return status."""
    try:
        resp = await client.get(f"{BASE_URL}/health", timeout=5.0)
        if resp.status_code == 200:
            return {"status": "healthy", "data": resp.json()}
        return {"status": "unhealthy", "code": resp.status_code}
    except Exception as e:
        return {"status": "error", "error": str(e)}

async def send_question(client: httpx.AsyncClient, username: str, question: str) -> Dict:
    """Send a question and return response info."""
    start_time = time.time()
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
        send_time = time.time() - start_time
        return {
            "success": resp.status_code == 200,
            "status_code": resp.status_code,
            "send_time": send_time,
            "error": None if resp.status_code == 200 else resp.text
        }
    except Exception as e:
        return {
            "success": False,
            "status_code": 0,
            "send_time": time.time() - start_time,
            "error": str(e)
        }

async def get_response(client: httpx.AsyncClient, username: str, max_wait: int = 25) -> Dict:
    """Poll for response with extended wait time."""
    start_time = time.time()
    seen_messages = set()
    
    for poll_round in range(max_wait):
        try:
            send_resp = await client.post(
                f"{BASE_URL}/chat/send",
                json={"channel": TEST_CHANNEL},
                timeout=10.0
            )
            if send_resp.status_code == 200:
                data = send_resp.json()
                msgs = data.get("messages", [])
                for msg in msgs:
                    if msg.get("reply_to") == username:
                        msg_text = msg.get("message", "")
                        msg_key = (username, msg_text[:50])
                        if msg_key not in seen_messages:
                            seen_messages.add(msg_key)
                            elapsed = time.time() - start_time
                            return {
                                "success": True,
                                "message": msg_text,
                                "elapsed": elapsed,
                                "poll_round": poll_round + 1
                            }
        except Exception as e:
            pass
        
        await asyncio.sleep(2.0)
    
    return {
        "success": False,
        "message": None,
        "elapsed": time.time() - start_time,
        "poll_round": max_wait
    }

async def comprehensive_followup_test():
    """Comprehensive follow-up test with service monitoring."""
    print("=" * 80)
    print("Comprehensive Follow-up Test with Service Monitoring")
    print("=" * 80)
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        # Step 1: Check service health
        print("\n[Step 1] Checking service health...")
        health = await check_service_health(client)
        print(f"Service status: {health.get('status')}")
        if health.get('status') != 'healthy':
            print(f"[!] Service may be reloading: {health}")
            print("Waiting 10 seconds for service to stabilize...")
            await asyncio.sleep(10)
            health = await check_service_health(client)
            print(f"Service status after wait: {health.get('status')}")
        
        user = "comprehensive_followup_user"
        
        # Step 2: First question
        print("\n[Step 2] Sending first question...")
        q1 = f"@{BOT_NAME} what is Mystic Realm?"
        print(f"Question: {q1}")
        
        send_result1 = await send_question(client, user, q1)
        print(f"Send result: {'Success' if send_result1['success'] else 'Failed'} ({send_result1['send_time']:.2f}s)")
        if not send_result1['success']:
            print(f"Error: {send_result1.get('error')}")
            return
        
        # Wait and get first response
        print("\n[Step 3] Waiting for first response (up to 25s)...")
        response1 = await get_response(client, user, max_wait=25)
        
        if not response1['success']:
            print(f"[X] No response received after {response1['elapsed']:.2f}s")
            return
        
        answer1 = response1['message']
        print(f"[OK] Received first response after {response1['elapsed']:.2f}s")
        print(f"Answer: {answer1[:200]}...")
        print(f"Length: {len(answer1)} chars")
        
        # Analyze first answer
        answer1_lower = answer1.lower()
        if "mystic realm" in answer1_lower or "puzzle" in answer1_lower:
            print("[OK] First answer references Mystic Realm!")
        else:
            print(f"[!] First answer doesn't mention Mystic Realm: {answer1[:100]}...")
        
        # Step 4: Wait for rate limit
        print("\n[Step 4] Waiting for rate limit (12s)...")
        await asyncio.sleep(12)
        
        # Step 5: Follow-up question
        print("\n[Step 5] Sending follow-up question...")
        q2 = f"@{BOT_NAME} what type of game is it?"
        print(f"Question: {q2}")
        print("Expected: Should reference 'puzzle adventure' from previous answer")
        
        send_result2 = await send_question(client, user, q2)
        print(f"Send result: {'Success' if send_result2['success'] else 'Failed'} ({send_result2['send_time']:.2f}s)")
        if not send_result2['success']:
            print(f"Error: {send_result2.get('error')}")
            return
        
        # Wait and get follow-up response
        print("\n[Step 6] Waiting for follow-up response (up to 25s)...")
        response2 = await get_response(client, user, max_wait=25)
        
        if not response2['success']:
            print(f"[X] No follow-up response after {response2['elapsed']:.2f}s")
            return
        
        answer2 = response2['message']
        print(f"[OK] Received follow-up response after {response2['elapsed']:.2f}s")
        print(f"Answer: {answer2[:200]}...")
        print(f"Length: {len(answer2)} chars")
        
        # Step 7: Analyze contextual awareness
        print("\n[Step 7] Analyzing contextual awareness...")
        answer2_lower = answer2.lower()
        answer1_lower = answer1.lower()
        
        contextual_indicators = []
        if "mystic realm" in answer2_lower:
            contextual_indicators.append("✓ Mentions 'Mystic Realm'")
        if "puzzle" in answer2_lower:
            contextual_indicators.append("✓ Mentions 'puzzle'")
        if "adventure" in answer2_lower:
            contextual_indicators.append("✓ Mentions 'adventure'")
        if "game" in answer2_lower and ("mystic" in answer1_lower or "puzzle" in answer1_lower):
            contextual_indicators.append("✓ References game type from previous answer")
        
        if "don't have enough context" in answer2_lower:
            print("[X] Follow-up returned 'no context'")
            print("[!] Session history may not be working correctly")
        elif contextual_indicators:
            print("[OK] Follow-up shows contextual awareness:")
            for indicator in contextual_indicators:
                print(f"    {indicator}")
        elif len(answer2) > 50:
            print(f"[?] Follow-up is substantive ({len(answer2)} chars) but may not explicitly reference previous answer")
            print(f"    Answer: {answer2[:150]}...")
        else:
            print(f"[!] Follow-up may not be contextually aware (length: {len(answer2)} chars)")
    
    print("\n" + "=" * 80)
    print("Test Complete")
    print("=" * 80)

async def sustained_load_test():
    """Test system under sustained load."""
    print("\n" + "=" * 80)
    print("Sustained Load Test")
    print("=" * 80)
    
    questions = [
        f"@{BOT_NAME} what is Mystic Realm?",
        f"@{BOT_NAME} what graphics card are they using?",
        f"@{BOT_NAME} what did they say about consistency for growing a channel?",
        f"@{BOT_NAME} how do you dodge the boss attack when he raises his left arm?",
        f"@{BOT_NAME} what microphone are they using?",
    ]
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        results = []
        start_time = time.time()
        
        print(f"\nSending {len(questions)} questions sequentially...")
        for i, question in enumerate(questions, 1):
            username = f"load_user_{i}"
            print(f"\n[{i}/{len(questions)}] {question}")
            
            # Check service health before each request
            health = await check_service_health(client)
            if health.get('status') != 'healthy':
                print(f"[!] Service unhealthy before request {i}: {health.get('status')}")
            
            # Send question
            send_result = await send_question(client, username, question)
            if not send_result['success']:
                print(f"[X] Failed to send: {send_result.get('error')}")
                results.append({"question": i, "success": False, "error": "send_failed"})
                continue
            
            # Get response
            response = await get_response(client, username, max_wait=25)
            if response['success']:
                elapsed = response['elapsed']
                answer_len = len(response['message'])
                print(f"[OK] Response received in {elapsed:.2f}s ({answer_len} chars)")
                results.append({
                    "question": i,
                    "success": True,
                    "elapsed": elapsed,
                    "answer_length": answer_len
                })
            else:
                print(f"[X] No response after {response['elapsed']:.2f}s")
                results.append({"question": i, "success": False, "error": "no_response"})
            
            # Rate limit delay
            if i < len(questions):
                await asyncio.sleep(12)
        
        total_time = time.time() - start_time
        successful = sum(1 for r in results if r.get('success'))
        
        print("\n" + "=" * 80)
        print("Load Test Results")
        print("=" * 80)
        print(f"Total questions: {len(questions)}")
        print(f"Successful: {successful}/{len(questions)}")
        print(f"Total time: {total_time:.2f}s")
        if successful > 0:
            avg_elapsed = sum(r['elapsed'] for r in results if r.get('success')) / successful
            print(f"Average response time: {avg_elapsed:.2f}s")
            print(f"Success rate: {successful/len(questions)*100:.1f}%")
        
        print("\nDetailed results:")
        for r in results:
            if r.get('success'):
                print(f"  Q{r['question']}: {r['elapsed']:.2f}s ({r['answer_length']} chars)")
            else:
                print(f"  Q{r['question']}: FAILED - {r.get('error', 'unknown')}")

async def main():
    """Run all comprehensive tests."""
    print("=" * 80)
    print("COMPREHENSIVE TEST SUITE")
    print("=" * 80)
    
    # Test 1: Follow-up with service monitoring
    await comprehensive_followup_test()
    
    # Test 2: Sustained load
    await sustained_load_test()
    
    print("\n" + "=" * 80)
    print("All Tests Complete")
    print("=" * 80)

if __name__ == "__main__":
    asyncio.run(main())

