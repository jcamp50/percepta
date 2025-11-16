# PowerShell script to start Python service and test endpoint
Write-Host "Starting Python service..." -ForegroundColor Green
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$PWD'; .\venv\Scripts\Activate.ps1; uvicorn py.main:app --reload --port 8000" -WindowStyle Normal

Write-Host "Waiting 5 seconds for service to start..." -ForegroundColor Yellow
Start-Sleep -Seconds 5

Write-Host "Testing endpoint..." -ForegroundColor Green
.\venv\Scripts\python.exe test_broadcaster_id.py

Write-Host "`nPress Enter to stop services..." -ForegroundColor Yellow
Read-Host

