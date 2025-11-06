"""
Comprehensive diagnostic script for broadcaster ID issue
Tests both Python service and simulates Node service calls
"""
import asyncio
import httpx
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BASE_URL = "http://localhost:8000"
TEST_CHANNEL = "clix"

async def test_python_service():
    """Test Python service endpoints"""
    print("=" * 80)
    print("TESTING PYTHON SERVICE")
    print("=" * 80)
    
    async with httpx.AsyncClient(timeout=10.0) as client:
        # Test 1: Health check
        print("\n[1] Health Check Endpoint")
        print("-" * 80)
        try:
            resp = await client.get(f"{BASE_URL}/health")
            print(f"Status: {resp.status_code}")
            if resp.status_code == 200:
                data = resp.json()
                print(f"Service: {data.get('service')}")
                print(f"Status: {data.get('status')}")
                
                broadcaster_lookup = data.get('broadcaster_id_lookup', {})
                print(f"\nBroadcaster ID Lookup:")
                print(f"  Status: {broadcaster_lookup.get('status')}")
                print(f"  Test Channel: {broadcaster_lookup.get('test_channel')}")
                print(f"  Test Broadcaster ID: {broadcaster_lookup.get('test_broadcaster_id')}")
                if broadcaster_lookup.get('error'):
                    print(f"  Error: {broadcaster_lookup.get('error')}")
                
                if broadcaster_lookup.get('status') == 'working':
                    print("\n[OK] Health check shows broadcaster ID lookup is working")
                else:
                    print(f"\n[WARNING] Health check shows broadcaster ID lookup status: {broadcaster_lookup.get('status')}")
            else:
                print(f"[FAILED] Health check returned {resp.status_code}")
                print(f"Response: {resp.text}")
        except httpx.ConnectError:
            print("[ERROR] Cannot connect to Python service. Is it running?")
            print("Start it with: uvicorn py.main:app --reload --port 8000")
            return False
        except Exception as e:
            print(f"[ERROR] Health check failed: {e}")
            return False
        
        # Test 2: Broadcaster ID endpoint with clix
        print(f"\n[2] Broadcaster ID Endpoint (channel: {TEST_CHANNEL})")
        print("-" * 80)
        try:
            resp = await client.get(
                f"{BASE_URL}/api/get-broadcaster-id",
                params={"channel_name": TEST_CHANNEL},
                timeout=10.0
            )
            print(f"Status: {resp.status_code}")
            print(f"Response: {resp.text}")
            
            if resp.status_code == 200:
                data = resp.json()
                broadcaster_id = data.get('broadcaster_id')
                channel_name = data.get('channel_name')
                print(f"\n[SUCCESS] Got broadcaster ID!")
                print(f"  Channel: {channel_name}")
                print(f"  Broadcaster ID: {broadcaster_id}")
                
                if broadcaster_id == "233300375":
                    print(f"\n[OK] Broadcaster ID matches expected value for clix")
                    return True
                else:
                    print(f"\n[WARNING] Broadcaster ID doesn't match expected value")
                    print(f"  Expected: 233300375")
                    print(f"  Got: {broadcaster_id}")
                    return False
            elif resp.status_code == 404:
                print(f"\n[FAILED] Channel not found: {TEST_CHANNEL}")
                print("This suggests the Python service is using old code or wrong credentials")
                return False
            elif resp.status_code == 500:
                print(f"\n[FAILED] Server error")
                return False
            elif resp.status_code == 503:
                print(f"\n[FAILED] Service unavailable")
                return False
            else:
                print(f"\n[FAILED] Unexpected status code: {resp.status_code}")
                return False
        except Exception as e:
            print(f"[ERROR] Broadcaster ID endpoint failed: {e}")
            return False

async def simulate_node_service_call():
    """Simulate what Node service does"""
    print("\n" + "=" * 80)
    print("SIMULATING NODE SERVICE CALL")
    print("=" * 80)
    
    async with httpx.AsyncClient(timeout=5.0) as client:
        print(f"\nSimulating Node service call to get broadcaster ID for '{TEST_CHANNEL}'...")
        print(f"Endpoint: {BASE_URL}/api/get-broadcaster-id")
        print(f"Params: channel_name={TEST_CHANNEL}")
        
        try:
            resp = await client.get(
                f"{BASE_URL}/api/get-broadcaster-id",
                params={"channel_name": TEST_CHANNEL},
                timeout=5000  # 5 seconds like Node service
            )
            
            print(f"\nResponse Status: {resp.status_code}")
            print(f"Response Body: {resp.text}")
            
            if resp.status_code == 200:
                data = resp.json()
                broadcaster_id = data.get('broadcaster_id')
                print(f"\n[SUCCESS] Node service would get broadcaster ID: {broadcaster_id}")
                return True
            else:
                print(f"\n[FAILED] Node service would fail with status {resp.status_code}")
                print("This is why Node service shows 'Failed to get broadcaster ID'")
                return False
        except httpx.TimeoutException:
            print(f"\n[FAILED] Timeout - Node service would timeout")
            return False
        except httpx.ConnectError:
            print(f"\n[FAILED] Connection error - Python service not running")
            return False
        except Exception as e:
            print(f"\n[FAILED] Error: {e}")
            return False

async def main():
    """Run all diagnostics"""
    print("\n" + "=" * 80)
    print("BROADCASTER ID ISSUE DIAGNOSTIC")
    print("=" * 80)
    print(f"\nTesting with channel: {TEST_CHANNEL}")
    print(f"Python service URL: {BASE_URL}")
    print()
    
    # Test Python service
    python_ok = await test_python_service()
    
    if python_ok:
        # Simulate Node service call
        node_ok = await simulate_node_service_call()
        
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        if python_ok and node_ok:
            print("\n[SUCCESS] All tests passed!")
            print("Python service is working correctly.")
            print("Node service should be able to get broadcaster ID.")
        elif python_ok and not node_ok:
            print("\n[WARNING] Python service works, but Node simulation failed.")
            print("Check Node service configuration and network connectivity.")
        else:
            print("\n[FAILED] Python service is not working correctly.")
            print("Check:")
            print("  1. Is Python service running in venv?")
            print("  2. Is TARGET_CHANNEL set correctly in .env?")
            print("  3. Are Twitch credentials correct?")
            print("  4. Is there another Python service on port 8000?")
    else:
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        print("\n[FAILED] Python service is not responding correctly.")
        print("\nTroubleshooting steps:")
        print("1. Check if Python service is running:")
        print("   netstat -ano | findstr :8000")
        print("\n2. Verify which Python is running:")
        print("   .\\venv\\Scripts\\python.exe scripts\\verify_python_service.py")
        print("\n3. Kill any system Python processes on port 8000:")
        print("   taskkill /F /PID <PID>")
        print("\n4. Start Python service with venv:")
        print("   .\\scripts\\start_python_service.ps1")
        print("   OR")
        print("   .\\venv\\Scripts\\activate")
        print("   uvicorn py.main:app --reload --port 8000")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nDiagnostic interrupted by user")
    except Exception as e:
        print(f"\n\n[ERROR] Diagnostic failed: {e}")
        import traceback
        traceback.print_exc()

