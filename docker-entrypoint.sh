#!/bin/bash
set -e

echo "Starting application validation..."

# Run configuration validation
python -m config.validators

if [ $? -ne 0 ]; then
    echo "Configuration validation failed. Exiting..."
    exit 1
fi

echo "Configuration validation passed!"

# Database migrations
if [ "$RUN_MIGRATIONS" = "true" ]; then
    echo "Running database migrations..."
    alembic upgrade head
fi

# Start the application based on the service
case "$SERVICE_TYPE" in
    "api")
        echo "Starting API server..."
        exec uvicorn src.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers ${WORKERS:-4}
        ;;
    "worker")
        echo "Starting Celery worker..."
        exec celery -A src.celery_app worker --loglevel=${LOG_LEVEL:-info}
        ;;
    "beat")
        echo "Starting Celery beat..."
        exec celery -A src.celery_app beat --loglevel=${LOG_LEVEL:-info}
        ;;
    *)
        echo "Unknown SERVICE_TYPE: $SERVICE_TYPE"
        exit 1
        ;;
esac