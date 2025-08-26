@echo off
REM E2E Test Suite Runner Script for Windows
REM This script runs the complete E2E test suite with proper setup and teardown

echo ==========================================
echo Teknofest 2025 - E2E Test Suite Runner
echo ==========================================
echo.

REM Check if Python virtual environment exists
if not exist "venv" (
    echo Creating Python virtual environment...
    python -m venv venv
)

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Install dependencies
echo Installing E2E test dependencies...
pip install -r tests\e2e\requirements.txt

REM Install Playwright browsers
echo Installing Playwright browsers...
playwright install

REM Start Docker containers if needed
echo Starting Docker services...
docker-compose -f docker-compose.yml up -d postgres redis

REM Wait for services to be ready
echo Waiting for services to be ready...
timeout /t 10 /nobreak >nul

REM Run database migrations
echo Running database migrations...
python manage_db.py upgrade

REM Start backend server in background
echo Starting backend server...
start /B python src\app.py

REM Start frontend server in background
echo Starting frontend server...
cd frontend
start /B npm run dev
cd ..

REM Wait for servers to start
echo Waiting for servers to start...
timeout /t 15 /nobreak >nul

REM Run E2E tests based on argument
echo Running E2E tests...

if "%1"=="full" (
    REM Run all E2E tests with full reporting
    pytest tests\e2e\ ^
        -v ^
        --tb=short ^
        --html=tests\e2e\reports\e2e_report.html ^
        --self-contained-html ^
        --json-report --json-report-file=tests\e2e\reports\e2e_report.json ^
        --cov=src ^
        --cov-report=html:tests\e2e\reports\coverage ^
        --cov-report=term ^
        -n auto
) else if "%1"=="performance" (
    REM Run only performance tests
    pytest tests\e2e\test_performance_e2e.py ^
        -v ^
        --tb=short ^
        --html=tests\e2e\reports\performance_report.html
) else if "%1"=="integration" (
    REM Run only integration tests
    pytest tests\e2e\test_full_system_integration.py ^
        -v ^
        --tb=short ^
        --html=tests\e2e\reports\integration_report.html
) else if "%1"=="user-journey" (
    REM Run only user journey tests
    pytest tests\e2e\test_user_journey.py ^
        -v ^
        --tb=short ^
        --html=tests\e2e\reports\user_journey_report.html
) else if "%1"=="smoke" (
    REM Run smoke tests (quick essential tests)
    pytest tests\e2e\ ^
        -v ^
        -m "not slow" ^
        --tb=short ^
        --maxfail=1
) else (
    REM Default: Run standard E2E test suite
    pytest tests\e2e\ ^
        -v ^
        --tb=short ^
        --html=tests\e2e\reports\e2e_report.html ^
        --self-contained-html
)

REM Capture test exit code
set TEST_EXIT_CODE=%ERRORLEVEL%

REM Cleanup
echo Cleaning up...

REM Stop servers
taskkill /F /IM python.exe 2>nul
taskkill /F /IM node.exe 2>nul

REM Stop Docker containers if not keeping them running
if not "%2"=="keep-running" (
    docker-compose down
)

REM Report results
if %TEST_EXIT_CODE%==0 (
    echo.
    echo ✓ E2E tests passed successfully!
    echo Reports available at: tests\e2e\reports\
) else (
    echo.
    echo ✗ E2E tests failed!
    echo Check reports at: tests\e2e\reports\
)

exit /b %TEST_EXIT_CODE%