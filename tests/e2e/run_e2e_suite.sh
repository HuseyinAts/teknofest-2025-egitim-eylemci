#!/bin/bash

# E2E Test Suite Runner Script
# This script runs the complete E2E test suite with proper setup and teardown

set -e

echo "==========================================
Teknofest 2025 - E2E Test Suite Runner
==========================================
"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if Python virtual environment exists
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}Creating Python virtual environment...${NC}"
    python -m venv venv
fi

# Activate virtual environment
source venv/bin/activate || source venv/Scripts/activate

# Install dependencies
echo -e "${YELLOW}Installing E2E test dependencies...${NC}"
pip install -r tests/e2e/requirements.txt

# Install Playwright browsers
echo -e "${YELLOW}Installing Playwright browsers...${NC}"
playwright install

# Start Docker containers if needed
echo -e "${YELLOW}Starting Docker services...${NC}"
docker-compose -f docker-compose.yml up -d postgres redis

# Wait for services to be ready
echo -e "${YELLOW}Waiting for services to be ready...${NC}"
sleep 10

# Run database migrations
echo -e "${YELLOW}Running database migrations...${NC}"
python manage_db.py upgrade

# Start backend server in background
echo -e "${YELLOW}Starting backend server...${NC}"
python src/app.py &
BACKEND_PID=$!

# Start frontend server in background
echo -e "${YELLOW}Starting frontend server...${NC}"
cd frontend && npm run dev &
FRONTEND_PID=$!
cd ..

# Wait for servers to start
echo -e "${YELLOW}Waiting for servers to start...${NC}"
sleep 15

# Run E2E tests
echo -e "${GREEN}Running E2E tests...${NC}"

# Run tests with different options based on argument
if [ "$1" == "full" ]; then
    # Run all E2E tests with full reporting
    pytest tests/e2e/ \
        -v \
        --tb=short \
        --html=tests/e2e/reports/e2e_report.html \
        --self-contained-html \
        --json-report --json-report-file=tests/e2e/reports/e2e_report.json \
        --cov=src \
        --cov-report=html:tests/e2e/reports/coverage \
        --cov-report=term \
        -n auto
elif [ "$1" == "performance" ]; then
    # Run only performance tests
    pytest tests/e2e/test_performance_e2e.py \
        -v \
        --tb=short \
        --html=tests/e2e/reports/performance_report.html
elif [ "$1" == "integration" ]; then
    # Run only integration tests
    pytest tests/e2e/test_full_system_integration.py \
        -v \
        --tb=short \
        --html=tests/e2e/reports/integration_report.html
elif [ "$1" == "user-journey" ]; then
    # Run only user journey tests
    pytest tests/e2e/test_user_journey.py \
        -v \
        --tb=short \
        --html=tests/e2e/reports/user_journey_report.html
elif [ "$1" == "smoke" ]; then
    # Run smoke tests (quick essential tests)
    pytest tests/e2e/ \
        -v \
        -m "not slow" \
        --tb=short \
        --maxfail=1
else
    # Default: Run standard E2E test suite
    pytest tests/e2e/ \
        -v \
        --tb=short \
        --html=tests/e2e/reports/e2e_report.html \
        --self-contained-html
fi

# Capture test exit code
TEST_EXIT_CODE=$?

# Cleanup
echo -e "${YELLOW}Cleaning up...${NC}"

# Stop servers
kill $BACKEND_PID 2>/dev/null || true
kill $FRONTEND_PID 2>/dev/null || true

# Stop Docker containers if started by this script
if [ "$2" != "keep-running" ]; then
    docker-compose down
fi

# Report results
if [ $TEST_EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}✓ E2E tests passed successfully!${NC}"
    echo -e "Reports available at: tests/e2e/reports/"
else
    echo -e "${RED}✗ E2E tests failed!${NC}"
    echo -e "Check reports at: tests/e2e/reports/"
fi

exit $TEST_EXIT_CODE