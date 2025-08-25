@echo off
echo ========================================
echo TEKNOFEST 2025 - Test Dependencies Setup
echo ========================================
echo.

cd /d C:\Users\husey\teknofest-2025-egitim-eylemci

echo [1/6] Installing pytest...
pip install pytest

echo [2/6] Installing pytest-cov...
pip install pytest-cov

echo [3/6] Installing pytest-asyncio...
pip install pytest-asyncio

echo [4/6] Installing fastapi...
pip install fastapi

echo [5/6] Installing httpx...
pip install httpx

echo [6/6] Installing sqlalchemy...
pip install sqlalchemy

echo.
echo ========================================
echo Dependencies installation completed!
echo ========================================
echo.

echo Running first test...
echo.
pytest tests\unit\test_api_endpoints.py::TestHealthEndpoint::test_health_check -v

echo.
echo ========================================
echo Press any key to see full test report...
pause

echo Running all unit tests with coverage...
pytest tests\unit -v --tb=short

echo.
echo ========================================
echo Test setup completed!
echo ========================================
pause