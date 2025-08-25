@echo off
echo ========================================
echo TEKNOFEST 2025 - Frontend (Next.js)
echo ========================================
echo.

echo [1/3] Checking Node.js installation...
node --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Node.js is not installed!
    echo Please install Node.js from https://nodejs.org/
    pause
    exit /b 1
)
echo Node.js is installed.
echo.

echo [2/3] Installing dependencies...
call npm install
if %errorlevel% neq 0 (
    echo ERROR: Failed to install dependencies!
    pause
    exit /b 1
)
echo Dependencies installed successfully.
echo.

echo [3/3] Starting development server...
echo.
echo ========================================
echo Frontend will be available at:
echo http://localhost:3000
echo ========================================
echo.
echo Press Ctrl+C to stop the server
echo.

call npm run dev