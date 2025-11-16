"""
Health check script to verify broadcaster ID lookup is working
Can be used in monitoring, CI/CD, or startup validation
"""
import sys
import asyncio
import httpx
from py.utils.twitch_api import get_broadcaster_id_from_channel_name

async def check_broadcaster_id_lookup():
    """Test broadcaster ID lookup directly"""
    try:
        broadcaster_id = await get_broadcaster_id_from_channel_name("xqc")
        if broadcaster_id == "71092938":
            print("[OK] Direct broadcaster ID lookup working")
            return True
        else:
            print(f"[FAIL] Direct lookup returned wrong ID: {broadcaster_id}")
            return False
    except Exception as e:
        print(f"[FAIL] Direct lookup failed: {e}")
        return False

async def check_endpoint():
    """Test the HTTP endpoint"""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(
                'http://localhost:8000/api/get-broadcaster-id',
                params={'channel_name': 'xqc'}
            )
            if response.status_code == 200:
                data = response.json()
                if data.get('broadcaster_id') == "71092938":
                    print("[OK] HTTP endpoint working")
                    return True
                else:
                    print(f"[FAIL] HTTP endpoint returned wrong ID: {data.get('broadcaster_id')}")
                    return False
            else:
                print(f"[FAIL] HTTP endpoint returned status {response.status_code}: {response.text}")
                return False
    except httpx.ConnectError:
        print("[FAIL] Cannot connect to service on port 8000")
        return False
    except Exception as e:
        print(f"[FAIL] HTTP endpoint check failed: {e}")
        return False

async def main():
    """Run all health checks"""
    print("Running health checks...")
    print("=" * 50)
    
    direct_check = await check_broadcaster_id_lookup()
    endpoint_check = await check_endpoint()
    
    print("=" * 50)
    if direct_check and endpoint_check:
        print("[OK] All health checks passed!")
        sys.exit(0)
    else:
        print("[FAIL] Some health checks failed!")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())

