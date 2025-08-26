"""
Production Metrics Collection for Prometheus
TEKNOFEST 2025 - Application metrics and monitoring
"""

from typing import Optional
import time
import logging
from prometheus_client import Counter, Histogram, Gauge, Info, CollectorRegistry, generate_latest
from prometheus_client import CONTENT_TYPE_LATEST
from fastapi import Response
from functools import wraps

logger = logging.getLogger(__name__)

# Create a custom registry for application metrics
registry = CollectorRegistry()

# Application info
app_info = Info(
    'teknofest_app',
    'Application information',
    registry=registry
)

# Request metrics
request_count = Counter(
    'teknofest_http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status'],
    registry=registry
)

request_duration = Histogram(
    'teknofest_http_request_duration_seconds',
    'HTTP request duration in seconds',
    ['method', 'endpoint'],
    registry=registry
)

# AI Model metrics
model_inference_count = Counter(
    'teknofest_model_inference_total',
    'Total model inference requests',
    ['model', 'status'],
    registry=registry
)

model_inference_duration = Histogram(
    'teknofest_model_inference_duration_seconds',
    'Model inference duration in seconds',
    ['model'],
    buckets=(0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30),
    registry=registry
)

model_token_usage = Counter(
    'teknofest_model_tokens_total',
    'Total tokens processed',
    ['model', 'type'],  # type: input/output
    registry=registry
)

# Database metrics
db_query_count = Counter(
    'teknofest_db_queries_total',
    'Total database queries',
    ['operation', 'table', 'status'],
    registry=registry
)

db_query_duration = Histogram(
    'teknofest_db_query_duration_seconds',
    'Database query duration in seconds',
    ['operation', 'table'],
    registry=registry
)

db_connections = Gauge(
    'teknofest_db_connections_active',
    'Active database connections',
    ['pool'],
    registry=registry
)

# Cache metrics
cache_hits = Counter(
    'teknofest_cache_hits_total',
    'Total cache hits',
    ['cache', 'operation'],
    registry=registry
)

cache_misses = Counter(
    'teknofest_cache_misses_total',
    'Total cache misses',
    ['cache', 'operation'],
    registry=registry
)

cache_size = Gauge(
    'teknofest_cache_size_bytes',
    'Current cache size in bytes',
    ['cache'],
    registry=registry
)

# Authentication metrics
auth_attempts = Counter(
    'teknofest_auth_attempts_total',
    'Total authentication attempts',
    ['method', 'status'],
    registry=registry
)

active_sessions = Gauge(
    'teknofest_auth_sessions_active',
    'Active user sessions',
    registry=registry
)

# Business metrics
quiz_generated = Counter(
    'teknofest_quiz_generated_total',
    'Total quizzes generated',
    ['topic', 'difficulty'],
    registry=registry
)

learning_path_created = Counter(
    'teknofest_learning_path_created_total',
    'Total learning paths created',
    ['grade', 'subject'],
    registry=registry
)

user_registrations = Counter(
    'teknofest_user_registrations_total',
    'Total user registrations',
    ['role'],
    registry=registry
)

# System health metrics
health_check_status = Gauge(
    'teknofest_health_check_status',
    'Health check status (1=healthy, 0=unhealthy)',
    ['component'],
    registry=registry
)

# Rate limiting metrics
rate_limit_exceeded = Counter(
    'teknofest_rate_limit_exceeded_total',
    'Total rate limit exceeded events',
    ['endpoint', 'limit_type'],
    registry=registry
)

# Error metrics
error_count = Counter(
    'teknofest_errors_total',
    'Total errors',
    ['type', 'endpoint', 'severity'],
    registry=registry
)

# Background job metrics
background_job_count = Counter(
    'teknofest_background_jobs_total',
    'Total background jobs processed',
    ['job_type', 'status'],
    registry=registry
)

background_job_duration = Histogram(
    'teknofest_background_job_duration_seconds',
    'Background job duration in seconds',
    ['job_type'],
    registry=registry
)

# Queue metrics
queue_size = Gauge(
    'teknofest_queue_size',
    'Current queue size',
    ['queue_name'],
    registry=registry
)


def init_metrics(app_name: str, version: str, environment: str):
    """Initialize application metrics with basic info"""
    app_info.info({
        'app_name': app_name,
        'version': version,
        'environment': environment
    })
    logger.info(f"Metrics initialized for {app_name} v{version} ({environment})")


def track_request(method: str, endpoint: str, status_code: int, duration: float):
    """Track HTTP request metrics"""
    request_count.labels(method=method, endpoint=endpoint, status=str(status_code)).inc()
    request_duration.labels(method=method, endpoint=endpoint).observe(duration)


def track_model_inference(model: str, duration: float, success: bool = True, 
                         input_tokens: int = 0, output_tokens: int = 0):
    """Track model inference metrics"""
    status = "success" if success else "failure"
    model_inference_count.labels(model=model, status=status).inc()
    model_inference_duration.labels(model=model).observe(duration)
    
    if input_tokens > 0:
        model_token_usage.labels(model=model, type="input").inc(input_tokens)
    if output_tokens > 0:
        model_token_usage.labels(model=model, type="output").inc(output_tokens)


def track_db_query(operation: str, table: str, duration: float, success: bool = True):
    """Track database query metrics"""
    status = "success" if success else "failure"
    db_query_count.labels(operation=operation, table=table, status=status).inc()
    db_query_duration.labels(operation=operation, table=table).observe(duration)


def track_cache_access(cache_name: str, operation: str, hit: bool):
    """Track cache access metrics"""
    if hit:
        cache_hits.labels(cache=cache_name, operation=operation).inc()
    else:
        cache_misses.labels(cache=cache_name, operation=operation).inc()


def track_auth_attempt(method: str, success: bool):
    """Track authentication attempt"""
    status = "success" if success else "failure"
    auth_attempts.labels(method=method, status=status).inc()


def track_business_metric(metric_type: str, **labels):
    """Track business-specific metrics"""
    if metric_type == "quiz_generated":
        quiz_generated.labels(**labels).inc()
    elif metric_type == "learning_path_created":
        learning_path_created.labels(**labels).inc()
    elif metric_type == "user_registration":
        user_registrations.labels(**labels).inc()


def track_error(error_type: str, endpoint: str, severity: str = "error"):
    """Track application errors"""
    error_count.labels(type=error_type, endpoint=endpoint, severity=severity).inc()


def update_health_status(component: str, is_healthy: bool):
    """Update component health status"""
    health_check_status.labels(component=component).set(1 if is_healthy else 0)


def update_db_connections(pool_name: str, count: int):
    """Update database connection count"""
    db_connections.labels(pool=pool_name).set(count)


def update_cache_size(cache_name: str, size_bytes: int):
    """Update cache size"""
    cache_size.labels(cache=cache_name).set(size_bytes)


def update_active_sessions(count: int):
    """Update active session count"""
    active_sessions.set(count)


def update_queue_size(queue_name: str, size: int):
    """Update queue size"""
    queue_size.labels(queue_name=queue_name).set(size)


def track_rate_limit_exceeded(endpoint: str, limit_type: str = "api"):
    """Track rate limit exceeded events"""
    rate_limit_exceeded.labels(endpoint=endpoint, limit_type=limit_type).inc()


def track_background_job(job_type: str, duration: float, success: bool = True):
    """Track background job execution"""
    status = "success" if success else "failure"
    background_job_count.labels(job_type=job_type, status=status).inc()
    background_job_duration.labels(job_type=job_type).observe(duration)


def metrics_middleware(app):
    """Middleware to track request metrics"""
    @app.middleware("http")
    async def track_requests(request, call_next):
        start_time = time.time()
        
        response = await call_next(request)
        
        duration = time.time() - start_time
        track_request(
            method=request.method,
            endpoint=request.url.path,
            status_code=response.status_code,
            duration=duration
        )
        
        return response


def get_metrics() -> bytes:
    """Generate metrics in Prometheus format"""
    return generate_latest(registry)


async def metrics_endpoint() -> Response:
    """FastAPI endpoint for Prometheus metrics"""
    metrics_data = get_metrics()
    return Response(
        content=metrics_data,
        media_type=CONTENT_TYPE_LATEST
    )


# Decorator for tracking function execution time
def track_execution_time(metric_type: str, **default_labels):
    """Decorator to track function execution time"""
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                success = True
            except Exception as e:
                success = False
                raise
            finally:
                duration = time.time() - start_time
                
                if metric_type == "model":
                    track_model_inference(
                        model=default_labels.get('model', 'unknown'),
                        duration=duration,
                        success=success
                    )
                elif metric_type == "db":
                    track_db_query(
                        operation=default_labels.get('operation', 'unknown'),
                        table=default_labels.get('table', 'unknown'),
                        duration=duration,
                        success=success
                    )
                elif metric_type == "job":
                    track_background_job(
                        job_type=default_labels.get('job_type', 'unknown'),
                        duration=duration,
                        success=success
                    )
            
            return result
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                success = True
            except Exception as e:
                success = False
                raise
            finally:
                duration = time.time() - start_time
                
                if metric_type == "model":
                    track_model_inference(
                        model=default_labels.get('model', 'unknown'),
                        duration=duration,
                        success=success
                    )
                elif metric_type == "db":
                    track_db_query(
                        operation=default_labels.get('operation', 'unknown'),
                        table=default_labels.get('table', 'unknown'),
                        duration=duration,
                        success=success
                    )
                elif metric_type == "job":
                    track_background_job(
                        job_type=default_labels.get('job_type', 'unknown'),
                        duration=duration,
                        success=success
                    )
            
            return result
        
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator