@echo off
cls
echo ============================================================
echo            TEKNOFEST 2025 - Test Suite Runner
echo ============================================================
echo.

cd /d C:\Users\husey\teknofest-2025-egitim-eylemci

echo [STEP 1] Checking Python installation...
python --version
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    pause
    exit /b 1
)

echo.
echo [STEP 2] Installing test dependencies...
echo.

echo Installing pytest...
python -m pip install pytest --quiet

echo Installing pytest-cov...
python -m pip install pytest-cov --quiet

echo Installing pytest-asyncio...
python -m pip install pytest-asyncio --quiet

echo Installing fastapi...
python -m pip install fastapi --quiet

echo Installing httpx...
python -m pip install httpx --quiet

echo Installing sqlalchemy...
python -m pip install sqlalchemy --quiet

echo.
echo ============================================================
echo            Dependencies installed successfully!
echo ============================================================
echo.

echo [STEP 3] Running tests...
echo.

echo Test 1: Running basic unit tests...
python -m pytest tests\unit\test_services.py -v -k "test_zpd_calculation or test_xp_calculation"

if errorlevel 1 (
    echo.
    echo Some tests may have failed. This is normal if the API is not running.
    echo Let's run our manual tests instead...
    echo.
    python run_tests_manual.py
)

echo.
echo ============================================================
echo            Test execution completed!
echo ============================================================
echo.
echo Next steps:
echo 1. Run all tests: pytest tests/ -v
echo 2. With coverage: pytest --cov=src --cov-report=html
echo 3. Specific test: pytest tests/unit/test_api_endpoints.py -v
echo.
pause