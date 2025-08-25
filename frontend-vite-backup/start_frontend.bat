@echo off
cls
echo ============================================================
echo       TEKNOFEST 2025 - Frontend Dashboard Setup
echo ============================================================
echo.

cd /d C:\Users\husey\teknofest-2025-egitim-eylemci\frontend

echo [STEP 1] Checking Node.js installation...
node --version
if errorlevel 1 (
    echo ERROR: Node.js is not installed!
    echo Please install Node.js from https://nodejs.org/
    pause
    exit /b 1
)

echo.
echo [STEP 2] Installing dependencies...
echo This may take a few minutes...
echo.

call npm install

echo.
echo [STEP 3] Installing additional UI dependencies...
echo.

call npm install lucide-react

echo.
echo ============================================================
echo            Frontend setup completed!
echo ============================================================
echo.
echo [STEP 4] Starting development server...
echo.
echo The application will open in your browser at:
echo http://localhost:5173
echo.
echo Press Ctrl+C to stop the server
echo.

npm run dev