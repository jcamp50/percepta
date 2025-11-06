"""
Test Node service integration with Python endpoint
"""
import httpx
import asyncio

async def test_node_integration():
    """Test that the endpoint works as Node service would call it"""
    async with httpx.AsyncClient() as c:
        # Test the exact endpoint Node service calls
        response = await c.get(
            'http://localhost:8000/api/get-broadcaster-id',
            params={'channel_name': 'xqc'},
            timeout=5.0
        )
        
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            broadcaster_id = data.get('broadcaster_id')
            channel_name = data.get('channel_name')
            print(f"[OK] SUCCESS!")
            print(f"   Channel: {channel_name}")
            print(f"   Broadcaster ID: {broadcaster_id}")
            print(f"   Expected ID: 71092938")
            if broadcaster_id == "71092938":
                print(f"   [OK] Broadcaster ID matches expected value!")
                return True
            else:
                print(f"   [FAIL] Broadcaster ID mismatch!")
                return False
        else:
            print(f"[FAIL] FAILED: {response.status_code}")
            print(f"   Response: {response.text}")
            return False

if __name__ == "__main__":
    success = asyncio.run(test_node_integration())
    exit(0 if success else 1)

