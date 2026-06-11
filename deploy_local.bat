@echo off
title DrowsiShield Local Deployment Launcher
echo ==========================================================
echo       DrowsiShield Driver Drowsiness Detection System
echo ==========================================================
echo.

:: Check Node and build frontend if needed
echo [1/3] Checking frontend build...
if not exist "frontend\dist\index.html" (
    echo [INFO] Production frontend build not found. Running build now...
    cd frontend
    call npm run build
    cd ..
) else (
    echo [SUCCESS] Pre-built frontend found in frontend/dist.
)
echo.

:: Check virtual environment
echo [2/3] Checking python environment...
if exist "venv\Scripts\activate.bat" (
    echo [INFO] Activating local virtual environment (venv)...
    call venv\Scripts\activate.bat
) else if exist "backend\venv\Scripts\activate.bat" (
    echo [INFO] Activating local virtual environment (backend\venv)...
    call backend\venv\Scripts\activate.bat
) else (
    echo [INFO] No local virtual environment found. Using system Python environment.
)
echo.

:: Running flask server
echo [3/3] Starting Flask Application server...
echo Access the application locally at: http://localhost:5000
echo Access on your local network using your PC's IP (e.g., http://192.168.x.x:5000)
echo.
echo Press Ctrl+C in this terminal window to stop the server.
echo.
python backend/server.py
pause
