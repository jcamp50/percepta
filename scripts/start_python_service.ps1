# PowerShell script to safely start Python service
# Ensures only one instance runs and uses correct Python interpreter

param(
    [int]$Port = 8000,
    [string]$PythonPath = ".\venv\Scripts\python.exe"
)

Write-Host "Starting Python service on port $Port..." -ForegroundColor Cyan

# Check if port is already in use
$existingProcesses = netstat -ano | findstr ":$Port" | findstr "LISTENING"
if ($existingProcesses) {
    Write-Host "Port $Port is already in use. Checking processes..." -ForegroundColor Yellow
    
    $pids = $existingProcesses | ForEach-Object { 
        ($_ -split '\s+')[-1] 
    } | Select-Object -Unique
    
    foreach ($pid in $pids) {
        if ($pid -and $pid -ne "0") {
            try {
                $proc = Get-Process -Id $pid -ErrorAction Stop
                $procPath = $proc.Path
                Write-Host "  Found process ${pid}: ${procPath}" -ForegroundColor Yellow
                
                # Check if it's NOT the venv Python
                if ($procPath -notlike "*venv*" -and $procPath -like "*python.exe*") {
                    Write-Host "  WARNING: Process is using system Python, not venv!" -ForegroundColor Red
                    Write-Host "  Killing process $pid..." -ForegroundColor Yellow
                    Stop-Process -Id $pid -Force -ErrorAction Stop
                    Write-Host "  Killed process $pid" -ForegroundColor Green
                } elseif ($procPath -like "*venv*") {
                    Write-Host "  Process is using venv Python - OK" -ForegroundColor Green
                } else {
                    Write-Host "  Killing process $pid..." -ForegroundColor Yellow
                    Stop-Process -Id $pid -Force -ErrorAction Stop
                    Write-Host "  Killed process $pid" -ForegroundColor Green
                }
            } catch {
                Write-Host "  Failed to kill process ${pid}: ${_}" -ForegroundColor Red
            }
        }
    }
    
    Start-Sleep -Seconds 2
}

# Verify port is free
$stillInUse = netstat -ano | findstr ":$Port" | findstr "LISTENING"
if ($stillInUse) {
    Write-Host "ERROR: Port $Port is still in use after cleanup!" -ForegroundColor Red
    exit 1
}

# Verify Python interpreter exists
if (-not (Test-Path $PythonPath)) {
    Write-Host "ERROR: Python interpreter not found at $PythonPath" -ForegroundColor Red
    exit 1
}

# Start the service
Write-Host "Starting service with $PythonPath..." -ForegroundColor Cyan
$env:PYTHONPATH = "."
Start-Process -FilePath $PythonPath -ArgumentList "-m","uvicorn","py.main:app","--reload","--port","$Port" -WindowStyle Normal

Start-Sleep -Seconds 3

# Verify service started
$listening = netstat -ano | findstr ":$Port" | findstr "LISTENING"
if ($listening) {
    Write-Host "Service started successfully on port $Port!" -ForegroundColor Green
} else {
    Write-Host "WARNING: Service may not have started. Check logs." -ForegroundColor Yellow
}

