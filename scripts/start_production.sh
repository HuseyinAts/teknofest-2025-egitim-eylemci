#!/bin/bash
# Production Startup Script for TEKNOFEST 2025 API
# Multi-Worker Deployment with Health Checks

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
APP_NAME="TEKNOFEST 2025 API"
APP_DIR="/opt/teknofest-api"
VENV_PATH="${APP_DIR}/venv"
LOG_DIR="/var/log/teknofest"
PID_FILE="/var/run/teknofest-api.pid"
HEALTH_CHECK_URL="http://localhost:8000/health"
MAX_WAIT_TIME=60

# Functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_dependencies() {
    log_info "Checking dependencies..."
    
    # Check Python
    if ! command -v python3 &> /dev/null; then
        log_error "Python3 is not installed"
        exit 1
    fi
    
    # Check PostgreSQL
    if ! pg_isready -h localhost -p 5432 &> /dev/null; then
        log_warn "PostgreSQL is not running. Starting..."
        systemctl start postgresql
        sleep 5
    fi
    
    # Check Redis
    if ! redis-cli ping &> /dev/null; then
        log_warn "Redis is not running. Starting..."
        systemctl start redis
        sleep 3
    fi
    
    log_info "All dependencies are ready"
}

setup_environment() {
    log_info "Setting up environment..."
    
    # Create necessary directories
    mkdir -p ${LOG_DIR}
    mkdir -p ${APP_DIR}/uploads
    mkdir -p /tmp/prometheus
    
    # Set permissions
    chown -R teknofest:teknofest ${LOG_DIR}
    chown -R teknofest:teknofest ${APP_DIR}
    
    # Activate virtual environment
    if [ ! -d "${VENV_PATH}" ]; then
        log_info "Creating virtual environment..."
        python3 -m venv ${VENV_PATH}
    fi
    
    source ${VENV_PATH}/bin/activate
    
    # Upgrade pip and install dependencies
    log_info "Installing/updating dependencies..."
    pip install --upgrade pip
    pip install -r ${APP_DIR}/requirements.txt
    pip install gunicorn psutil
    
    # Set production environment variables
    export APP_ENV=production
    export LOG_LEVEL=info
    export API_WORKERS=$(nproc)
    export PYTHONPATH=${APP_DIR}:${PYTHONPATH}
    
    log_info "Environment setup complete"
}

start_application() {
    log_info "Starting ${APP_NAME}..."
    
    # Check if already running
    if [ -f ${PID_FILE} ]; then
        OLD_PID=$(cat ${PID_FILE})
        if ps -p ${OLD_PID} > /dev/null 2>&1; then
            log_error "Application is already running with PID ${OLD_PID}"
            exit 1
        else
            log_warn "Removing stale PID file"
            rm -f ${PID_FILE}
        fi
    fi
    
    # Start Gunicorn with multiple workers
    cd ${APP_DIR}
    gunicorn src.app:app \
        --config gunicorn_config.py \
        --pid ${PID_FILE} \
        --daemon \
        --log-level info \
        --access-logfile ${LOG_DIR}/access.log \
        --error-logfile ${LOG_DIR}/error.log
    
    log_info "Waiting for application to start..."
    
    # Wait for application to be ready
    WAIT_COUNT=0
    while [ ${WAIT_COUNT} -lt ${MAX_WAIT_TIME} ]; do
        if curl -sf ${HEALTH_CHECK_URL} > /dev/null 2>&1; then
            log_info "Application started successfully!"
            break
        fi
        sleep 1
        WAIT_COUNT=$((WAIT_COUNT + 1))
    done
    
    if [ ${WAIT_COUNT} -eq ${MAX_WAIT_TIME} ]; then
        log_error "Application failed to start within ${MAX_WAIT_TIME} seconds"
        stop_application
        exit 1
    fi
    
    # Display status
    show_status
}

stop_application() {
    log_info "Stopping ${APP_NAME}..."
    
    if [ -f ${PID_FILE} ]; then
        PID=$(cat ${PID_FILE})
        if ps -p ${PID} > /dev/null 2>&1; then
            # Send SIGTERM for graceful shutdown
            kill -TERM ${PID}
            
            # Wait for process to stop
            WAIT_COUNT=0
            while [ ${WAIT_COUNT} -lt 30 ]; do
                if ! ps -p ${PID} > /dev/null 2>&1; then
                    log_info "Application stopped successfully"
                    rm -f ${PID_FILE}
                    return 0
                fi
                sleep 1
                WAIT_COUNT=$((WAIT_COUNT + 1))
            done
            
            # Force kill if still running
            log_warn "Force killing application..."
            kill -9 ${PID}
            rm -f ${PID_FILE}
        else
            log_warn "Application is not running (stale PID file)"
            rm -f ${PID_FILE}
        fi
    else
        log_warn "Application is not running (no PID file)"
    fi
}

reload_application() {
    log_info "Reloading ${APP_NAME}..."
    
    if [ -f ${PID_FILE} ]; then
        PID=$(cat ${PID_FILE})
        if ps -p ${PID} > /dev/null 2>&1; then
            # Send SIGHUP for graceful reload
            kill -HUP ${PID}
            log_info "Reload signal sent"
            sleep 3
            show_status
        else
            log_error "Application is not running"
            exit 1
        fi
    else
        log_error "Application is not running (no PID file)"
        exit 1
    fi
}

show_status() {
    echo -e "\n${GREEN}========================================${NC}"
    echo -e "${GREEN}  ${APP_NAME} Status${NC}"
    echo -e "${GREEN}========================================${NC}"
    
    if [ -f ${PID_FILE} ]; then
        PID=$(cat ${PID_FILE})
        if ps -p ${PID} > /dev/null 2>&1; then
            echo -e "Status: ${GREEN}Running${NC}"
            echo -e "PID: ${PID}"
            
            # Get worker count
            WORKER_COUNT=$(pgrep -P ${PID} | wc -l)
            echo -e "Workers: ${WORKER_COUNT}"
            
            # Check health endpoint
            if HEALTH_RESPONSE=$(curl -sf ${HEALTH_CHECK_URL} 2>/dev/null); then
                echo -e "Health: ${GREEN}Healthy${NC}"
            else
                echo -e "Health: ${YELLOW}Unhealthy${NC}"
            fi
            
            # Show resource usage
            if command -v ps &> /dev/null; then
                echo -e "\nResource Usage:"
                ps -p ${PID} -o pid,vsz,rss,pcpu,pmem,comm --no-headers
            fi
            
            # Show recent logs
            echo -e "\nRecent Error Logs:"
            tail -n 5 ${LOG_DIR}/error.log 2>/dev/null || echo "No error logs"
            
        else
            echo -e "Status: ${RED}Stopped${NC} (stale PID file)"
        fi
    else
        echo -e "Status: ${RED}Stopped${NC}"
    fi
    
    echo -e "${GREEN}========================================${NC}\n"
}

scale_workers() {
    local NEW_WORKERS=$1
    
    if [ -z "${NEW_WORKERS}" ]; then
        log_error "Please specify number of workers"
        exit 1
    fi
    
    log_info "Scaling to ${NEW_WORKERS} workers..."
    
    # Update environment variable
    export API_WORKERS=${NEW_WORKERS}
    
    # Reload application with new worker count
    reload_application
}

# Main script
case "$1" in
    start)
        check_dependencies
        setup_environment
        start_application
        ;;
    stop)
        stop_application
        ;;
    restart)
        stop_application
        sleep 2
        check_dependencies
        setup_environment
        start_application
        ;;
    reload)
        reload_application
        ;;
    status)
        show_status
        ;;
    scale)
        scale_workers $2
        ;;
    logs)
        tail -f ${LOG_DIR}/error.log ${LOG_DIR}/access.log
        ;;
    test)
        log_info "Running health check..."
        curl -v ${HEALTH_CHECK_URL}
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|reload|status|scale <num>|logs|test}"
        echo ""
        echo "Commands:"
        echo "  start    - Start the application"
        echo "  stop     - Stop the application"
        echo "  restart  - Restart the application"
        echo "  reload   - Reload workers without downtime"
        echo "  status   - Show application status"
        echo "  scale    - Scale worker count (e.g., scale 8)"
        echo "  logs     - Tail application logs"
        echo "  test     - Run health check"
        exit 1
        ;;
esac

exit 0