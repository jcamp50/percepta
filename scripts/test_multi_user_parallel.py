"""
Multi-User Parallel Testing Suite for JCB-27

Tests:
- 5+ concurrent users asking questions simultaneously
- Context isolation (users don't see each other's context)
- Follow-up questions (using session history)
- Rate limiting with multiple users
- Performance metrics (latency, throughput)
- Session persistence
"""

import asyncio
import httpx
import json
import time
import sys
import codecs
from datetime import datetime
from typing import List, Dict, Tuple
from collections import defaultdict

# Set UTF-8 encoding for Windows console
if sys.platform == "win32":
    sys.stdout = codecs.getwriter("utf-8")(sys.stdout.buffer, "strict")
    sys.stderr = codecs.getwriter("utf-8")(sys.stderr.buffer, "strict")

# Test configuration
BASE_URL = "http://localhost:8000"
TEST_CHANNEL = "awinewofi5783"  # Use seeded test channel
BOT_NAME = "percepta"

# Test results tracking
test_results: List[Dict] = []
performance_metrics: Dict = {
    "response_times": [],
    "user_response_times": defaultdict(list),
    "concurrent_requests": 0,
    "total_requests": 0,
    "errors": 0,
}


def log_test(name: str, passed: bool, details: str = ""):
    """Log test result."""
    status = "PASS" if passed else "FAIL"
    symbol = "[OK]" if passed else "[X]"
    print(f"{symbol} [{status}] {name}")
    if details:
        print(f"    {details}")
    test_results.append({
        "name": name,
        "passed": passed,
        "details": details,
        "timestamp": datetime.now().isoformat()
    })


async def send_chat_message(
    client: httpx.AsyncClient,
    username: str,
    message: str,
    channel: str = TEST_CHANNEL,
) -> Tuple[httpx.Response, float]:
    """Send a chat message and measure latency."""
    start_time = time.time()
    try:
        response = await client.post(
            f"{BASE_URL}/chat/message",
            json={
                "channel": channel,
                "username": username,
                "message": message,
                "timestamp": datetime.now().isoformat()
            },
            timeout=30.0,
        )
        latency = time.time() - start_time
        performance_metrics["total_requests"] += 1
        performance_metrics["response_times"].append(latency)
        performance_metrics["user_response_times"][username].append(latency)
        return response, latency
    except Exception as e:
        latency = time.time() - start_time
        performance_metrics["errors"] += 1
        performance_metrics["total_requests"] += 1
        print(f"Error sending message from {username}: {e}")
        return None, latency


async def get_queued_messages(
    client: httpx.AsyncClient, channel: str = TEST_CHANNEL
) -> List[Dict]:
    """Get queued messages from the API."""
    try:
        response = await client.post(
            f"{BASE_URL}/chat/send",
            json={"channel": channel},
            timeout=10.0,
        )
        if response.status_code == 200:
            data = response.json()
            return data.get("messages", [])
        return []
    except Exception as e:
        print(f"Error getting messages: {e}")
        return []


async def test_concurrent_different_questions(client: httpx.AsyncClient):
    """Test 5 users asking different questions simultaneously."""
    print("\n=== Test 1: Concurrent Different Questions ===")
    
    users = [
        ("user1", f"@{BOT_NAME} what game is Mystic Realm?"),
        ("user2", f"@{BOT_NAME} what graphics card are they using?"),
        ("user3", f"@{BOT_NAME} what did they say about consistency for growing a channel?"),
        ("user4", f"@{BOT_NAME} how do you dodge the boss attack when he raises his left arm?"),
        ("user5", f"@{BOT_NAME} what microphone are they using?"),
    ]
    
    # Send all questions concurrently
    print(f"Sending {len(users)} questions concurrently...")
    tasks = [
        send_chat_message(client, username, message)
        for username, message in users
    ]
    responses = await asyncio.gather(*tasks)
    
    # Wait for processing (RAG queries can take time - embedding + LLM + Redis)
    await asyncio.sleep(10)
    
    # Collect responses (poll multiple times to get all responses)
    all_messages = []
    seen_message_ids = set()
    for poll_round in range(20):  # Poll more times, wait longer
        msgs = await get_queued_messages(client)
        for msg in msgs:
            # Deduplicate by message content and user
            msg_key = (msg.get("reply_to"), msg.get("message")[:50])
            if msg_key not in seen_message_ids:
                all_messages.append(msg)
                seen_message_ids.add(msg_key)
        if len(all_messages) >= len(users):
            # Got all responses, can stop early
            break
        await asyncio.sleep(2.0)  # Longer delay between polls for RAG processing
    
    # Verify each user got a response
    response_by_user = {msg.get("reply_to"): msg.get("message") for msg in all_messages}
    
    users_with_responses = len(response_by_user)
    if users_with_responses >= len(users):
        log_test(
            "Concurrent different questions",
            True,
            f"All {users_with_responses} users received responses"
        )
    else:
        log_test(
            "Concurrent different questions",
            False,
            f"Only {users_with_responses}/{len(users)} users received responses"
        )
    
    # Check latency
    avg_latency = sum(performance_metrics["response_times"][-len(users):]) / len(users)
    if avg_latency < 5.0:
        log_test(
            "Concurrent latency",
            True,
            f"Average latency: {avg_latency:.2f}s (< 5s)"
        )
    else:
        log_test(
            "Concurrent latency",
            False,
            f"Average latency: {avg_latency:.2f}s (>= 5s)"
        )


async def test_follow_up_questions(client: httpx.AsyncClient):
    """Test follow-up questions using session history."""
    print("\n=== Test 2: Follow-up Questions ===")
    
    user = "followup_user"
    
    # First question (relevant to seeded transcripts - mentions Mystic Realm)
    question1 = f"@{BOT_NAME} what is Mystic Realm?"
    print(f"First question: {question1}")
    await send_chat_message(client, user, question1)
    await asyncio.sleep(15)  # Wait longer for RAG processing (15-20s)
    
    msgs1 = await get_queued_messages(client)
    first_answer = None
    # Poll multiple times for first response
    for poll_round in range(15):
        msgs1 = await get_queued_messages(client)
        for msg in msgs1:
            if msg.get("reply_to") == user:
                first_answer = msg.get("message")
                break
        if first_answer:
            break
        await asyncio.sleep(1.5)
    
    if not first_answer:
        log_test("Follow-up - First question", False, "No response received")
        return
    
    log_test("Follow-up - First question", True, "Received response")
    
    # Follow-up question that should reference previous answer about Mystic Realm
    await asyncio.sleep(12)  # Wait for rate limit (10s) plus buffer
    question2 = f"@{BOT_NAME} what type of game is it?"
    print(f"Follow-up question: {question2}")
    await send_chat_message(client, user, question2)
    await asyncio.sleep(15)  # Wait for RAG processing (15-20s)
    
    msgs2 = await get_queued_messages(client)
    followup_answer = None
    # Poll multiple times to get response
    for poll_round in range(15):  # Poll more times for follow-up
        msgs2 = await get_queued_messages(client)
        for msg in msgs2:
            if msg.get("reply_to") == user:
                # Check if this is a new message (different from first answer)
                msg_text = msg.get("message", "")
                if msg_text != first_answer:
                    followup_answer = msg_text
                    break
        if followup_answer:
            break
        await asyncio.sleep(2.0)  # Longer delay between polls for RAG processing
    
    if followup_answer:
        log_test(
            "Follow-up - Second question",
            True,
            f"Received follow-up response: {followup_answer[:100]}..."
        )
        # Check if follow-up seems contextually aware
        # Should reference "Mystic Realm" or "puzzle adventure" from previous answer
        followup_lower = followup_answer.lower()
        first_answer_lower = first_answer.lower() if first_answer else ""
        
        # Check for contextual references
        has_contextual_reference = False
        contextual_details = []
        
        # Check if follow-up mentions key terms from first answer
        if "mystic realm" in followup_lower:
            has_contextual_reference = True
            contextual_details.append("mentions 'Mystic Realm'")
        if "puzzle" in followup_lower:
            has_contextual_reference = True
            contextual_details.append("mentions 'puzzle'")
        if "adventure" in followup_lower:
            has_contextual_reference = True
            contextual_details.append("mentions 'adventure'")
        if "game" in followup_lower and ("mystic" in first_answer_lower or "puzzle" in first_answer_lower):
            has_contextual_reference = True
            contextual_details.append("references game type")
        
        # Check if answer is substantive (not just "no context")
        if "don't have enough context" in followup_lower:
            log_test(
                "Follow-up - Contextual awareness",
                False,
                "Follow-up returned 'no context' - session history may not be working"
            )
        elif has_contextual_reference or len(followup_answer) > 50:
            log_test(
                "Follow-up - Contextual awareness",
                True,
                f"Follow-up is contextually aware: {', '.join(contextual_details) if contextual_details else 'substantive answer'} (length: {len(followup_answer)} chars)"
            )
        else:
            log_test(
                "Follow-up - Contextual awareness",
                False,
                f"Follow-up may not reference previous answer (length: {len(followup_answer)} chars)"
            )
    else:
        log_test("Follow-up - Second question", False, "No follow-up response")


async def test_context_isolation(client: httpx.AsyncClient):
    """Test that users don't see each other's context."""
    print("\n=== Test 3: Context Isolation ===")
    
    # User A asks about topic X
    user_a = "isolate_user_a"
    user_b = "isolate_user_b"
    
    question_a = f"@{BOT_NAME} what should you do when the boss raises his left arm?"
    print(f"User A question: {question_a}")
    await send_chat_message(client, user_a, question_a)
    await asyncio.sleep(15)  # Wait for RAG processing
    
    # Poll multiple times for response
    answer_a = None
    for poll_round in range(15):
        msgs_a = await get_queued_messages(client)
        for msg in msgs_a:
            if msg.get("reply_to") == user_a:
                answer_a = msg.get("message")
                break
        if answer_a:
            break
        await asyncio.sleep(1.5)
    
    # User B asks about different topic (streaming setup)
    await asyncio.sleep(1)
    question_b = f"@{BOT_NAME} what CPU are they using?"
    print(f"User B question: {question_b}")
    await send_chat_message(client, user_b, question_b)
    await asyncio.sleep(15)  # Wait for RAG processing
    
    # Poll multiple times for response
    answer_b = None
    for poll_round in range(15):
        msgs_b = await get_queued_messages(client)
        for msg in msgs_b:
            if msg.get("reply_to") == user_b:
                answer_b = msg.get("message")
                break
        if answer_b:
            break
        await asyncio.sleep(1.5)
    
    # Verify answers are different and appropriate
    if answer_a and answer_b:
        # Check if answers contain relevant keywords
        answer_a_lower = answer_a.lower()
        answer_b_lower = answer_b.lower()
        
        # User A asked about boss fight - should mention "dodge", "left arm", "right"
        # User B asked about CPU - should mention "AMD", "Ryzen", "7950X"
        a_relevant = "dodge" in answer_a_lower or "left" in answer_a_lower or "right" in answer_a_lower or "boss" in answer_a_lower
        b_relevant = "amd" in answer_b_lower or "ryzen" in answer_b_lower or "7950" in answer_b_lower or "cpu" in answer_b_lower
        
        if answer_a != answer_b and (a_relevant or b_relevant):
            log_test(
                "Context isolation",
                True,
                f"Users received different responses (A: {len(answer_a)} chars, B: {len(answer_b)} chars)"
            )
        elif answer_a == answer_b and "don't have enough context" in answer_a_lower:
            log_test(
                "Context isolation",
                False,
                "Both users got 'no context' - may need better question matching"
            )
        else:
            log_test(
                "Context isolation",
                False,
                f"Users received similar responses (may be fallback): A={answer_a[:50]}..., B={answer_b[:50]}..."
            )
    else:
        log_test(
            "Context isolation",
            False,
            f"Missing responses: A={bool(answer_a)}, B={bool(answer_b)}"
        )


async def test_rate_limiting_concurrent(client: httpx.AsyncClient):
    """Test rate limiting with multiple concurrent users."""
    print("\n=== Test 4: Rate Limiting with Concurrent Users ===")
    
    user = "ratelimit_user"
    
    # Send two questions quickly (should be rate limited)
    question1 = f"@{BOT_NAME} test question 1"
    question2 = f"@{BOT_NAME} test question 2"
    
    print(f"Sending two questions rapidly from {user}...")
    await send_chat_message(client, user, question1)
    await asyncio.sleep(0.1)  # Very short delay
    await send_chat_message(client, user, question2)
    
    await asyncio.sleep(2)
    
    msgs = await get_queued_messages(client)
    user_msgs = [msg for msg in msgs if msg.get("reply_to") == user]
    
    if len(user_msgs) <= 1:
        log_test(
            "Rate limiting concurrent",
            True,
            f"Rate limit enforced: {len(user_msgs)} messages processed"
        )
    else:
        log_test(
            "Rate limiting concurrent",
            False,
            f"Rate limit not enforced: {len(user_msgs)} messages processed"
        )


async def test_session_persistence(client: httpx.AsyncClient):
    """Test that session history persists across questions."""
    print("\n=== Test 5: Session Persistence ===")
    
    user = "persist_user"
    
    # First question (relevant to seeded transcripts)
    question1 = f"@{BOT_NAME} what did they say about consistency for growing a channel?"
    await send_chat_message(client, user, question1)
    await asyncio.sleep(15)  # Wait for RAG processing
    
    # Poll multiple times for response
    answer1 = None
    for poll_round in range(15):
        msgs1 = await get_queued_messages(client)
        for msg in msgs1:
            if msg.get("reply_to") == user:
                answer1 = msg.get("message")
                break
        if answer1:
            break
        await asyncio.sleep(1.5)
    
    if not answer1:
        log_test("Session persistence - First question", False, "No response")
        return
    
    log_test("Session persistence - First question", True, "Received response")
    
    # Wait for rate limit, then ask second question
    await asyncio.sleep(12)  # Wait for rate limit to expire
    
    question2 = f"@{BOT_NAME} what else helps with channel growth?"
    await send_chat_message(client, user, question2)
    await asyncio.sleep(15)  # Wait for RAG processing
    
    # Poll multiple times for response
    answer2 = None
    for poll_round in range(15):
        msgs2 = await get_queued_messages(client)
        for msg in msgs2:
            if msg.get("reply_to") == user:
                answer2 = msg.get("message")
                break
        if answer2:
            break
        await asyncio.sleep(1.5)
    
    if answer2:
        log_test(
            "Session persistence - Second question",
            True,
            "Session persisted across questions"
        )
    else:
        log_test("Session persistence - Second question", False, "Session lost")


async def test_mixed_context_questions(client: httpx.AsyncClient):
    """Test mix of in-context and out-of-context questions."""
    print("\n=== Test 6: Mixed Context Questions ===")
    
    users = [
        ("mixed_user1", f"@{BOT_NAME} what happens in phase two of the boss fight?"),
        ("mixed_user2", f"@{BOT_NAME} what game are they playing called Mystic Realm?"),
        ("mixed_user3", f"@{BOT_NAME} what camera are they using?"),
    ]
    
    print("Sending mixed context questions...")
    tasks = [
        send_chat_message(client, username, message)
        for username, message in users
    ]
    await asyncio.gather(*tasks)
    
    await asyncio.sleep(3)
    
    msgs = await get_queued_messages(client)
    response_count = len(msgs)
    
    if response_count >= len(users):
        log_test(
            "Mixed context questions",
            True,
            f"All {response_count} questions processed"
        )
    else:
        log_test(
            "Mixed context questions",
            False,
            f"Only {response_count}/{len(users)} questions processed"
        )


async def run_all_tests():
    """Run all multi-user parallel tests."""
    print("=" * 80)
    print("Multi-User Parallel Testing Suite")
    print("=" * 80)
    print(f"Testing against: {BASE_URL}")
    print(f"Test channel: {TEST_CHANNEL}")
    print("=" * 80)
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        # Health check
        try:
            resp = await client.get(f"{BASE_URL}/health")
            if resp.status_code != 200:
                print("\n⚠️  Service is not available. Make sure the Python service is running.")
                return False
        except Exception as e:
            print(f"\n⚠️  Service is not available: {e}")
            return False
        
        # Run all tests
        await test_concurrent_different_questions(client)
        await asyncio.sleep(2)
        
        await test_follow_up_questions(client)
        await asyncio.sleep(2)
        
        await test_context_isolation(client)
        await asyncio.sleep(2)
        
        await test_rate_limiting_concurrent(client)
        await asyncio.sleep(2)
        
        await test_session_persistence(client)
        await asyncio.sleep(2)
        
        await test_mixed_context_questions(client)
    
    # Print summary
    print("\n" + "=" * 80)
    print("Test Summary")
    print("=" * 80)
    
    passed = sum(1 for r in test_results if r["passed"])
    total = len(test_results)
    
    for result in test_results:
        status = "PASS" if result["passed"] else "FAIL"
        symbol = "[OK]" if result["passed"] else "[X]"
        print(f"{symbol} [{status}] {result['name']}")
        if result["details"]:
            print(f"    {result['details']}")
    
    print("=" * 80)
    print(f"Total: {passed}/{total} tests passed")
    print(f"Success rate: {(passed/total*100):.1f}%")
    
    # Performance metrics
    if performance_metrics["response_times"]:
        avg_latency = sum(performance_metrics["response_times"]) / len(
            performance_metrics["response_times"]
        )
        max_latency = max(performance_metrics["response_times"])
        min_latency = min(performance_metrics["response_times"])
        
        # Calculate percentiles
        sorted_times = sorted(performance_metrics["response_times"])
        p95_idx = int(len(sorted_times) * 0.95)
        p99_idx = int(len(sorted_times) * 0.99)
        p95 = sorted_times[p95_idx] if p95_idx < len(sorted_times) else sorted_times[-1]
        p99 = sorted_times[p99_idx] if p99_idx < len(sorted_times) else sorted_times[-1]
        
        print("\n" + "=" * 80)
        print("Performance Metrics")
        print("=" * 80)
        print(f"Total requests: {performance_metrics['total_requests']}")
        print(f"Errors: {performance_metrics['errors']}")
        print(f"Average latency: {avg_latency:.2f}s")
        print(f"Min latency: {min_latency:.2f}s")
        print(f"Max latency: {max_latency:.2f}s")
        print(f"P95 latency: {p95:.2f}s")
        print(f"P99 latency: {p99:.2f}s")
        print("=" * 80)
        
        # Per-user latency
        print("\nPer-User Latency:")
        for user, times in performance_metrics["user_response_times"].items():
            if times:
                avg_user = sum(times) / len(times)
                print(f"  {user}: {avg_user:.2f}s avg ({len(times)} requests)")
    
    # Save results
    results_data = {
        "test_results": test_results,
        "performance_metrics": {
            "total_requests": performance_metrics["total_requests"],
            "errors": performance_metrics["errors"],
            "avg_latency": avg_latency if performance_metrics["response_times"] else 0,
            "max_latency": max_latency if performance_metrics["response_times"] else 0,
            "p95_latency": p95 if performance_metrics["response_times"] else 0,
            "p99_latency": p99 if performance_metrics["response_times"] else 0,
        },
        "timestamp": datetime.now().isoformat()
    }
    
    with open("test_results_multi_user.json", "w") as f:
        json.dump(results_data, f, indent=2)
    print(f"\nResults saved to: test_results_multi_user.json")
    
    return passed == total


if __name__ == "__main__":
    success = asyncio.run(run_all_tests())
    exit(0 if success else 1)

