@echo off
title ChurnIQ Pro Launcher
echo ========================================================
echo  Starting ChurnIQ Customer Churn & Behaviour Platform
echo ========================================================

cd /d "%~dp0"

echo [1/2] Starting FastAPI Backend on http://127.0.0.1:8000...
start "" ".\.venv\Scripts\python.exe" -m uvicorn main:app --host 127.0.0.1 --port 8000

echo [2/2] Exposing secure public link with Cloudflare...
timeout /t 3 >nul
start "" ".\cloudflared.exe" tunnel --url http://127.0.0.1:8000

echo.
echo ========================================================
echo ChurnIQ Pro is running!
echo Local Dashboard: http://127.0.0.1:8000/dashboard
echo Local API Docs:  http://127.0.0.1:8000/docs
echo ========================================================
pause
