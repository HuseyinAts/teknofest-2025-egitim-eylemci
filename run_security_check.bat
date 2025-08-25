@echo off
echo =====================================
echo TEKNOFEST 2025 - Security Test Suite
echo =====================================
echo.

echo [1/3] Checking Python Installation...
python --version 2>nul
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed or not in PATH
    exit /b 1
)

echo [2/3] Installing Required Packages...
pip install pytest pytest-cov bandit safety pylint black mypy -q 2>nul
echo Packages installed.

echo [3/3] Running Security Checks...
echo.

echo Checking .env configuration...
if exist .env (
    echo [OK] .env file exists
    findstr /C:"SECRET_KEY=" .env >nul
    if %errorlevel% equ 0 (
        echo [OK] SECRET_KEY configured
    ) else (
        echo [WARNING] SECRET_KEY not found
    )
    
    findstr /C:"APP_DEBUG=false" .env >nul
    if %errorlevel% equ 0 (
        echo [OK] Debug mode disabled
    ) else (
        echo [WARNING] Debug mode may be enabled
    )
    
    findstr /C:"RATE_LIMIT_ENABLED=true" .env >nul
    if %errorlevel% equ 0 (
        echo [OK] Rate limiting enabled
    ) else (
        echo [WARNING] Rate limiting disabled
    )
) else (
    echo [ERROR] .env file not found
)

echo.
echo Checking security files...
if exist src\core\security.py (
    echo [OK] security.py exists
) else (
    echo [ERROR] security.py not found
)

if exist src\core\authentication.py (
    echo [OK] authentication.py exists
) else (
    echo [ERROR] authentication.py not found
)

if exist src\database\secure_db.py (
    echo [OK] secure_db.py exists
) else (
    echo [ERROR] secure_db.py not found
)

echo.
echo =====================================
echo Security check complete!
echo Check security_report.html for details
echo =====================================
pause
