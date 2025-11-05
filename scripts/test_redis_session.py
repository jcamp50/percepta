"""
Test script to verify Redis session management is working.

This script:
1. Connects to Redis using the same configuration as the application
2. Lists all session keys
3. Displays session data for each key
4. Shows Redis connection status and statistics
"""

import asyncio
import json
import sys
from pathlib import Path

# Set UTF-8 encoding for Windows console
if sys.platform == "win32":
    import codecs

    sys.stdout = codecs.getwriter("utf-8")(sys.stdout.buffer, "strict")
    sys.stderr = codecs.getwriter("utf-8")(sys.stderr.buffer, "strict")

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from py.config import settings
import redis.asyncio as redis


async def test_redis_sessions():
    """Test Redis connection and list all session keys."""
    redis_url = (
        f"redis://{settings.redis_host}:{settings.redis_port}/{settings.redis_db}"
    )

    print("=" * 80)
    print("Redis Session Management Test")
    print("=" * 80)
    print(f"Redis URL: {redis_url}")
    print(f"Redis Host: {settings.redis_host}")
    print(f"Redis Port: {settings.redis_port}")
    print(f"Redis DB: {settings.redis_db}")
    print()

    try:
        # Connect to Redis
        print("Connecting to Redis...")
        client = redis.from_url(redis_url, decode_responses=True)

        # Test connection
        pong = await client.ping()
        if pong:
            print("[OK] Redis connection successful!")
        else:
            print("[ERROR] Redis ping failed")
            return

        print()

        # Get all session keys
        print("Searching for session keys...")
        session_keys = await client.keys("session:*")

        if not session_keys:
            print("No session keys found in Redis.")
            print("\nThis could mean:")
            print("  - No users have sent messages yet")
            print("  - Sessions have expired (15 min TTL)")
            print("  - Redis was recently cleared")
        else:
            print(f"Found {len(session_keys)} session key(s):\n")

            for key in session_keys:
                print("-" * 80)
                print(f"Key: {key}")

                # Get session data
                data = await client.get(key)
                if data:
                    session = json.loads(data)
                    print(f"  User ID: {session.get('user_id', 'N/A')}")
                    print(f"  Channel: {session.get('channel', 'N/A')}")
                    print(f"  Last Seen: {session.get('last_seen', 'N/A')}")
                    print(f"  Message Count: {session.get('message_count', 0)}")
                    print(
                        f"  Last Message Time: {session.get('last_message_time', 'N/A')}"
                    )

                    # Show Q&A history
                    questions = session.get("last_questions", [])
                    answers = session.get("last_answers", [])

                    if questions or answers:
                        print(f"  Q&A History ({len(questions)} pairs):")
                        for i, (q, a) in enumerate(zip(questions, answers), 1):
                            print(f"    {i}. Q: {q[:60]}{'...' if len(q) > 60 else ''}")
                            print(f"       A: {a[:60]}{'...' if len(a) > 60 else ''}")
                    else:
                        print("  Q&A History: None")

                    # Get TTL
                    ttl = await client.ttl(key)
                    if ttl > 0:
                        minutes = ttl // 60
                        seconds = ttl % 60
                        print(f"  TTL: {minutes}m {seconds}s remaining")
                    elif ttl == -1:
                        print(f"  TTL: No expiry set")
                    else:
                        print(f"  TTL: Key expired")
                else:
                    print("  (No data found)")

                print()

        # Show Redis info
        print("-" * 80)
        print("Redis Server Info:")
        info = await client.info("server")
        print(f"  Redis Version: {info.get('redis_version', 'N/A')}")
        print(f"  Connected Clients: {info.get('connected_clients', 'N/A')}")

        info_db = await client.info("keyspace")
        if info_db:
            db_key = f"db{settings.redis_db}"
            if db_key in info_db:
                keyspace = info_db[db_key]
                print(f"  Keys in DB {settings.redis_db}: {keyspace.get('keys', 0)}")
                print(f"  Expires: {keyspace.get('expires', 0)}")
                print(f"  Avg TTL: {keyspace.get('avg_ttl', 0) / 1000:.2f}s")

        await client.aclose()
        print("\n[OK] Test completed successfully!")

    except redis.ConnectionError as e:
        print(f"[ERROR] Failed to connect to Redis: {e}")
        print("\nMake sure Redis is running:")
        print("  - Check if Redis server is started")
        print("  - Verify host and port are correct")
        print("  - Check firewall settings")
    except Exception as e:
        print(f"[ERROR] Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_redis_sessions())
