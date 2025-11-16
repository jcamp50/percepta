"""
Verify which Python service is running on port 8000
"""
import socket
import subprocess
import sys
import os

def check_port(port=8000):
    """Check if port is in use and which process"""
    try:
        # Try to connect to the port
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = sock.connect_ex(('127.0.0.1', port))
        sock.close()
        
        if result == 0:
            print(f"[OK] Port {port} is in use")
            return True
        else:
            print(f"[NOT IN USE] Port {port} is NOT in use")
            return False
    except Exception as e:
        print(f"Error checking port: {e}")
        return False

def get_process_info(port=8000):
    """Get process information for port"""
    try:
        if sys.platform == "win32":
            # Windows: use netstat and tasklist
            result = subprocess.run(
                ['netstat', '-ano'],
                capture_output=True,
                text=True
            )
            
            lines = result.stdout.split('\n')
            pids = []
            for line in lines:
                if f':{port}' in line and 'LISTENING' in line:
                    parts = line.split()
                    if len(parts) > 0:
                        pid = parts[-1]
                        if pid.isdigit():
                            pids.append(pid)
            
            if pids:
                print(f"\nFound {len(pids)} process(es) on port {port}:")
                for pid in set(pids):
                    try:
                        result = subprocess.run(
                            ['tasklist', '/FI', f'PID eq {pid}', '/FO', 'CSV'],
                            capture_output=True,
                            text=True
                        )
                        lines = result.stdout.split('\n')
                        if len(lines) > 1:
                            # CSV format: "Image Name","PID","Session Name","Session#","Mem Usage"
                            parts = lines[1].split('","')
                            if len(parts) >= 2:
                                proc_name = parts[0].strip('"')
                                proc_path = parts[1] if len(parts) > 1 else "N/A"
                                
                                # Get full path
                                try:
                                    import psutil
                                    proc = psutil.Process(int(pid))
                                    proc_path = proc.exe()
                                except:
                                    proc_path = "N/A"
                                
                                print(f"  PID: {pid}")
                                print(f"  Name: {proc_name}")
                                print(f"  Path: {proc_path}")
                                
                                # Check if it's venv Python
                                if 'venv' in proc_path.lower():
                                    print(f"  [OK] Using VENV Python")
                                elif 'python' in proc_path.lower():
                                    print(f"  [WARNING] Using SYSTEM Python (not venv!)")
                                print()
                    except Exception as e:
                        print(f"  Error getting info for PID {pid}: {e}")
            else:
                print(f"No processes found on port {port}")
    except Exception as e:
        print(f"Error getting process info: {e}")

def test_endpoint():
    """Test the broadcaster ID endpoint"""
    try:
        import httpx
        import asyncio
        
        async def test():
            async with httpx.AsyncClient(timeout=5.0) as client:
                try:
                    # Test health endpoint
                    resp = await client.get("http://localhost:8000/health")
                    print(f"\n[OK] Health endpoint responded: {resp.status_code}")
                    data = resp.json()
                    print(f"   Service: {data.get('service')}")
                    broadcaster_lookup = data.get('broadcaster_id_lookup', {})
                    print(f"   Broadcaster ID Lookup Status: {broadcaster_lookup.get('status')}")
                    print(f"   Test Channel: {broadcaster_lookup.get('test_channel')}")
                    
                    # Test broadcaster ID endpoint
                    resp = await client.get(
                        "http://localhost:8000/api/get-broadcaster-id",
                        params={"channel_name": "clix"}
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        print(f"\n[OK] Broadcaster ID endpoint working!")
                        print(f"   Channel: {data.get('channel_name')}")
                        print(f"   Broadcaster ID: {data.get('broadcaster_id')}")
                    else:
                        print(f"\n[FAILED] Broadcaster ID endpoint failed: {resp.status_code}")
                        print(f"   Response: {resp.text}")
                except Exception as e:
                    print(f"\n[ERROR] Error testing endpoint: {e}")
        
        asyncio.run(test())
    except ImportError:
        print("\n[WARNING] httpx not available, skipping endpoint test")

if __name__ == "__main__":
    print("=" * 80)
    print("Python Service Verification")
    print("=" * 80)
    
    port = 8000
    print(f"\n[1] Checking port {port}...")
    is_in_use = check_port(port)
    
    if is_in_use:
        print(f"\n[2] Getting process information...")
        get_process_info(port)
        
        print(f"\n[3] Testing endpoint...")
        test_endpoint()
    else:
        print(f"\n[WARNING] No service running on port {port}")
        print("   Start the service with: .\\scripts\\start_python_service.ps1")
    
    print("\n" + "=" * 80)

