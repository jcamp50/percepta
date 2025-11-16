# Broadcaster ID Issue - Root Cause and Solution

## Problem
Node service was failing to get broadcaster ID for "clix" even though Python service logs showed it successfully resolved the broadcaster ID.

## Root Cause
**Two Python services were running:**
1. ✅ **Venv Python service** (correct) - Running in venv, has updated code, working correctly
2. ❌ **System Python service** (wrong) - Running `C:\Python312\python.exe`, had old code, responding to requests on port 8000

The Node service was connecting to the **system Python service** (old code) instead of the **venv Python service** (new code).

## Evidence
- Python service logs showed: `Found broadcaster ID for clix: 233300375` ✅
- But API endpoint returned: `{"detail":"Channel not found: clix"}` ❌
- Process on port 8000: PID 43304 using `C:\Python312\python.exe` (system Python, not venv)

## Solution

### 1. Kill the old system Python service
```powershell
# Find process on port 8000
netstat -ano | findstr :8000

# Kill the process (replace <PID> with actual PID)
taskkill /F /PID <PID>
```

### 2. Verify port is free
```powershell
netstat -ano | findstr :8000
# Should return nothing
```

### 3. Start Python service using venv
**Option A: Use the startup script (recommended)**
```powershell
.\scripts\start_python_service.ps1
```

**Option B: Manual start**
```powershell
.\venv\Scripts\activate
uvicorn py.main:app --reload --port 8000
```

### 4. Verify correct Python is running
```powershell
.\venv\Scripts\python.exe scripts\verify_python_service.py
```

This will show:
- Which process is on port 8000
- Whether it's using venv Python or system Python
- Test the broadcaster ID endpoint

### 5. Run comprehensive diagnostic
```powershell
.\venv\Scripts\python.exe scripts\diagnose_broadcaster_id_issue.py
```

This tests:
- Health endpoint
- Broadcaster ID endpoint with "clix"
- Simulates Node service calls

## Prevention

### Updated Startup Script
The `scripts/start_python_service.ps1` script now:
- ✅ Checks which Python is running on port 8000
- ✅ Warns if system Python is detected
- ✅ Automatically kills system Python processes
- ✅ Only starts venv Python

### Always Use Venv Python
**Never run:**
```powershell
python -m uvicorn py.main:app --reload --port 8000  # ❌ Uses system Python
```

**Always run:**
```powershell
.\venv\Scripts\python.exe -m uvicorn py.main:app --reload --port 8000  # ✅ Uses venv Python
```

Or use the startup script:
```powershell
.\scripts\start_python_service.ps1  # ✅ Automatically uses venv Python
```

## Testing

### Test 1: Verify Python Service
```powershell
.\venv\Scripts\python.exe scripts\diagnose_broadcaster_id_issue.py
```

Expected output:
```
[SUCCESS] All tests passed!
Python service is working correctly.
Node service should be able to get broadcaster ID.
```

### Test 2: Test Node Service
Start Node service and check logs:
```powershell
npm run start
```

Expected logs:
```
[INFO] Resolved channel clix to broadcaster ID: 233300375
```

NOT:
```
[WARN] Failed to get broadcaster ID for clix
```

### Test 3: Direct API Test
```powershell
# PowerShell
Invoke-WebRequest -Uri "http://localhost:8000/api/get-broadcaster-id?channel_name=clix" | Select-Object -ExpandProperty Content

# Should return:
# {"broadcaster_id":"233300375","channel_name":"clix"}
```

## Files Changed

1. **py/main.py**
   - Fixed hardcoded "xqc" references to use `settings.target_channel` from .env
   - Updated startup validation to use configured channel
   - Updated health check endpoint to use configured channel

2. **scripts/start_python_service.ps1**
   - Added detection of system Python vs venv Python
   - Automatically kills system Python processes
   - Warns when system Python is detected

3. **New diagnostic scripts:**
   - `scripts/verify_python_service.py` - Verify which Python is running
   - `scripts/diagnose_broadcaster_id_issue.py` - Comprehensive diagnostic
   - `scripts/test_broadcaster_id_endpoint.py` - Test broadcaster ID endpoint

## Next Steps

1. ✅ Kill old system Python service (DONE)
2. ⏳ Start Python service with venv
3. ⏳ Run diagnostic script to verify
4. ⏳ Start Node service and verify it connects correctly
5. ⏳ Check Node service logs for successful broadcaster ID resolution

