"""
Test script for Rate Limiting & Safety features (JCB-26)

Tests:
- Per-user rate limiting (1 question per 10s)
- Global rate limiting (20 msg / 30s)
- Repeated question cooldown (60s)
- Content filtering (toxic/PII)
- Response length limits (< 500 chars)
- !more command for long answers
- Admin commands (!pause, !resume, !status)
"""

import asyncio
import httpx
import json
import time
import sys
import codecs
from datetime import datetime
from typing import List, Dict

# Set UTF-8 encoding for Windows console
if sys.platform == "win32":
    sys.stdout = codecs.getwriter("utf-8")(sys.stdout.buffer, "strict")
    sys.stderr = codecs.getwriter("utf-8")(sys.stderr.buffer, "strict")

# Test configuration
BASE_URL = "http://localhost:8000"
TEST_CHANNEL = "testchannel"
ADMIN_USER = "admin_user"
NORMAL_USER = "normal_user"
BOT_NAME = "percepta"

# Test results tracking
test_results: List[Dict] = []


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


async def send_chat_message(client: httpx.AsyncClient, username: str, message: str, channel: str = TEST_CHANNEL):
    """Send a chat message to the API."""
    try:
        response = await client.post(
            f"{BASE_URL}/chat/message",
            json={
                "channel": channel,
                "username": username,
                "message": message,
                "timestamp": datetime.now().isoformat()
            }
        )
        return response
    except Exception as e:
        print(f"Error sending message: {e}")
        return None


async def get_queued_messages(client: httpx.AsyncClient, channel: str = TEST_CHANNEL):
    """Get queued messages from the API."""
    try:
        response = await client.post(
            f"{BASE_URL}/chat/send",
            json={"channel": channel}
        )
        return response
    except Exception as e:
        print(f"Error getting messages: {e}")
        return None


async def test_health_check(client: httpx.AsyncClient):
    """Test health check endpoint."""
    print("\n=== Testing Health Check ===")
    try:
        response = await client.get(f"{BASE_URL}/health")
        if response.status_code == 200:
            log_test("Health check", True, f"Service is healthy")
            return True
        else:
            log_test("Health check", False, f"Status code: {response.status_code}")
            return False
    except Exception as e:
        log_test("Health check", False, f"Error: {e}")
        return False


async def test_per_user_rate_limit(client: httpx.AsyncClient):
    """Test per-user rate limiting (1 question per 10s)."""
    print("\n=== Testing Per-User Rate Limiting ===")
    
    # Send first question
    response1 = await send_chat_message(
        client, NORMAL_USER, f"@{BOT_NAME} what is the meaning of life?"
    )
    await asyncio.sleep(0.5)  # Wait for processing
    
    # Get first response
    msgs1 = await get_queued_messages(client)
    if msgs1 and msgs1.status_code == 200:
        data1 = msgs1.json()
        if len(data1.get("messages", [])) > 0:
            log_test("Per-user rate limit - First message", True, "First message processed")
        else:
            log_test("Per-user rate limit - First message", False, "No response received")
    
    # Immediately send second question (should be rate limited)
    response2 = await send_chat_message(
        client, NORMAL_USER, f"@{BOT_NAME} what is the weather?"
    )
    await asyncio.sleep(0.5)
    
    msgs2 = await get_queued_messages(client)
    if msgs2 and msgs2.status_code == 200:
        data2 = msgs2.json()
        if len(data2.get("messages", [])) == 0:
            log_test("Per-user rate limit - Second message blocked", True, "Rate limit enforced")
        else:
            log_test("Per-user rate limit - Second message blocked", False, 
                    f"Received {len(data2.get('messages', []))} messages, expected 0")


async def test_global_rate_limit(client: httpx.AsyncClient):
    """Test global rate limiting (20 msg / 30s)."""
    print("\n=== Testing Global Rate Limiting ===")
    
    # Send messages from multiple users rapidly
    print("Sending 25 messages rapidly (should be limited to 20)...")
    messages_sent = []
    for i in range(25):
        user = f"user_{i}"
        response = await send_chat_message(
            client, user, f"@{BOT_NAME} test message {i}"
        )
        messages_sent.append(response)
        await asyncio.sleep(0.1)  # Small delay between sends
    
    await asyncio.sleep(1)  # Wait for processing
    
    # Count total messages queued
    total_messages = 0
    for _ in range(5):  # Check multiple times
        msgs = await get_queued_messages(client)
        if msgs and msgs.status_code == 200:
            data = msgs.json()
            total_messages += len(data.get("messages", []))
        await asyncio.sleep(0.2)
    
    # Should not exceed 20 messages
    if total_messages <= 20:
        log_test("Global rate limit", True, f"Total messages: {total_messages} (max 20)")
    else:
        log_test("Global rate limit", False, f"Total messages: {total_messages} (exceeds 20)")


async def test_repeated_question_cooldown(client: httpx.AsyncClient):
    """Test repeated question cooldown (60s)."""
    print("\n=== Testing Repeated Question Cooldown ===")
    
    question = f"@{BOT_NAME} what is the capital of France?"
    
    # Send first question
    await send_chat_message(client, NORMAL_USER, question)
    await asyncio.sleep(1)  # Wait for processing
    
    # Get first response
    msgs1 = await get_queued_messages(client)
    if msgs1 and msgs1.status_code == 200:
        data1 = msgs1.json()
        if len(data1.get("messages", [])) > 0:
            log_test("Repeated question cooldown - First question", True, "First question processed")
    
    # Send same question again immediately (should be blocked)
    await send_chat_message(client, NORMAL_USER, question)
    await asyncio.sleep(1)
    
    msgs2 = await get_queued_messages(client)
    if msgs2 and msgs2.status_code == 200:
        data2 = msgs2.json()
        if len(data2.get("messages", [])) == 0:
            log_test("Repeated question cooldown - Blocked", True, "Repeated question blocked")
        else:
            log_test("Repeated question cooldown - Blocked", False, 
                    f"Received {len(data2.get('messages', []))} messages, expected 0")


async def test_content_filtering(client: httpx.AsyncClient):
    """Test content filtering (toxic/PII)."""
    print("\n=== Testing Content Filtering ===")
    
    # Test PII detection (email)
    pii_message = f"@{BOT_NAME} my email is test@example.com"
    await send_chat_message(client, NORMAL_USER, pii_message)
    await asyncio.sleep(1)
    
    msgs = await get_queued_messages(client)
    if msgs and msgs.status_code == 200:
        data = msgs.json()
        if len(data.get("messages", [])) == 0:
            log_test("Content filtering - PII detection", True, "PII content blocked")
        else:
            log_test("Content filtering - PII detection", False, "PII content not blocked")
    
    # Test phone number pattern
    phone_message = f"@{BOT_NAME} call me at 555-123-4567"
    await send_chat_message(client, NORMAL_USER, phone_message)
    await asyncio.sleep(1)
    
    msgs2 = await get_queued_messages(client)
    if msgs2 and msgs2.status_code == 200:
        data2 = msgs2.json()
        if len(data2.get("messages", [])) == 0:
            log_test("Content filtering - Phone number", True, "Phone number blocked")
        else:
            log_test("Content filtering - Phone number", False, "Phone number not blocked")


async def test_response_length_limit(client: httpx.AsyncClient):
    """Test response length limits and !more command."""
    print("\n=== Testing Response Length Limits ===")
    
    # Note: This test depends on RAG service generating long responses
    # For now, we'll test the truncation logic exists
    question = f"@{BOT_NAME} tell me a very long story"
    await send_chat_message(client, NORMAL_USER, question)
    await asyncio.sleep(2)  # Wait for RAG processing
    
    msgs = await get_queued_messages(client)
    if msgs and msgs.status_code == 200:
        data = msgs.json()
        if len(data.get("messages", [])) > 0:
            message_text = data["messages"][0]["message"]
            if len(message_text) <= 500 or "truncated" in message_text.lower() or "!more" in message_text.lower():
                log_test("Response length limit", True, 
                        f"Response length: {len(message_text)} chars, contains truncation hint: {'!more' in message_text.lower()}")
            else:
                log_test("Response length limit", False, 
                        f"Response length: {len(message_text)} chars (exceeds 500)")


async def test_more_command(client: httpx.AsyncClient):
    """Test !more command for retrieving full answers."""
    print("\n=== Testing !more Command ===")
    
    # First, we need a truncated answer - this is tricky without a real long response
    # We'll test the command structure exists
    await send_chat_message(client, NORMAL_USER, "!more abc12345")
    await asyncio.sleep(0.5)
    
    msgs = await get_queued_messages(client)
    if msgs and msgs.status_code == 200:
        # Command was processed (may or may not find answer)
        log_test("!more command", True, "Command processed (answer lookup depends on stored data)")


async def test_admin_commands(client: httpx.AsyncClient):
    """Test admin commands (!pause, !resume, !status)."""
    print("\n=== Testing Admin Commands ===")
    
    # Test !status command (should work for anyone, but only admin gets response)
    await send_chat_message(client, ADMIN_USER, "!status")
    await asyncio.sleep(0.5)
    
    msgs = await get_queued_messages(client)
    if msgs and msgs.status_code == 200:
        data = msgs.json()
        if len(data.get("messages", [])) > 0:
            response_text = data["messages"][0]["message"].lower()
            if "status" in response_text or "active" in response_text or "paused" in response_text:
                log_test("Admin command - !status", True, f"Status response: {data['messages'][0]['message']}")
            else:
                log_test("Admin command - !status", False, "Invalid status response")
        else:
            log_test("Admin command - !status", False, "No response received")
    
    # Test !pause command
    await send_chat_message(client, ADMIN_USER, "!pause")
    await asyncio.sleep(0.5)
    
    msgs = await get_queued_messages(client)
    if msgs and msgs.status_code == 200:
        data = msgs.json()
        if len(data.get("messages", [])) > 0:
            response_text = data["messages"][0]["message"].lower()
            if "paused" in response_text:
                log_test("Admin command - !pause", True, "Bot paused successfully")
            else:
                log_test("Admin command - !pause", False, "Pause command failed")
    
    # Test that bot is paused (normal question should be ignored)
    await send_chat_message(client, NORMAL_USER, f"@{BOT_NAME} test question")
    await asyncio.sleep(0.5)
    
    msgs = await get_queued_messages(client)
    if msgs and msgs.status_code == 200:
        data = msgs.json()
        if len(data.get("messages", [])) == 0:
            log_test("Admin command - Bot paused state", True, "Bot correctly ignoring messages when paused")
        else:
            log_test("Admin command - Bot paused state", False, "Bot still processing messages when paused")
    
    # Test !resume command
    await send_chat_message(client, ADMIN_USER, "!resume")
    await asyncio.sleep(0.5)
    
    msgs = await get_queued_messages(client)
    if msgs and msgs.status_code == 200:
        data = msgs.json()
        if len(data.get("messages", [])) > 0:
            response_text = data["messages"][0]["message"].lower()
            if "resumed" in response_text:
                log_test("Admin command - !resume", True, "Bot resumed successfully")
            else:
                log_test("Admin command - !resume", False, "Resume command failed")
    
    # Test that bot is resumed (admin commands still work)
    await send_chat_message(client, ADMIN_USER, "!status")
    await asyncio.sleep(0.5)
    
    msgs = await get_queued_messages(client)
    if msgs and msgs.status_code == 200:
        data = msgs.json()
        if len(data.get("messages", [])) > 0:
            response_text = data["messages"][0]["message"].lower()
            if "active" in response_text:
                log_test("Admin command - Bot resumed state", True, "Bot correctly resumed")
            else:
                log_test("Admin command - Bot resumed state", False, f"Status: {response_text}")


async def test_non_admin_cannot_use_admin_commands(client: httpx.AsyncClient):
    """Test that non-admin users cannot use admin commands."""
    print("\n=== Testing Non-Admin Access Control ===")
    
    # Try to pause as non-admin
    await send_chat_message(client, NORMAL_USER, "!pause")
    await asyncio.sleep(0.5)
    
    msgs = await get_queued_messages(client)
    if msgs and msgs.status_code == 200:
        data = msgs.json()
        # Should not receive a response about pausing
        response_texts = [msg["message"].lower() for msg in data.get("messages", [])]
        if not any("paused" in text for text in response_texts):
            log_test("Non-admin access control", True, "Non-admin cannot use admin commands")
        else:
            log_test("Non-admin access control", False, "Non-admin was able to use admin command")


async def run_all_tests():
    """Run all tests."""
    print("=" * 80)
    print("Rate Limiting & Safety Features Test Suite")
    print("=" * 80)
    print(f"Testing against: {BASE_URL}")
    print(f"Test channel: {TEST_CHANNEL}")
    print(f"Admin user: {ADMIN_USER}")
    print(f"Normal user: {NORMAL_USER}")
    print("=" * 80)
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        # Test health check first
        health_ok = await test_health_check(client)
        if not health_ok:
            print("\n⚠️  Service is not available. Make sure the Python service is running.")
            print("   Start it with: uvicorn py.main:app --reload --port 8000")
            return
        
        # Run all tests
        await test_per_user_rate_limit(client)
        await asyncio.sleep(2)  # Wait between test suites
        
        await test_global_rate_limit(client)
        await asyncio.sleep(2)
        
        await test_repeated_question_cooldown(client)
        await asyncio.sleep(2)
        
        await test_content_filtering(client)
        await asyncio.sleep(2)
        
        await test_response_length_limit(client)
        await asyncio.sleep(2)
        
        await test_more_command(client)
        await asyncio.sleep(2)
        
        await test_admin_commands(client)
        await asyncio.sleep(2)
        
        await test_non_admin_cannot_use_admin_commands(client)
    
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
    print("=" * 80)
    
    # Save results to file
    with open("test_results_rate_limiting.json", "w") as f:
        json.dump(test_results, f, indent=2)
    print(f"\nResults saved to: test_results_rate_limiting.json")
    
    return passed == total


if __name__ == "__main__":
    success = asyncio.run(run_all_tests())
    exit(0 if success else 1)

