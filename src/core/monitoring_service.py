"""
Comprehensive Monitoring Service with Sentry Integration
"""

import os
import logging
import time
from typing import Dict, Any, Optional, Callable
from functools import wraps
from contextlib import contextmanager
import json
from datetime import datetime

import sentry_sdk
from sentry_sdk.integrations.fastapi import FastApiIntegration
from sentry_sdk.integrations.sqlalchemy import SqlalchemyIntegration
from sentry_sdk.integrations.redis import RedisIntegration
from sentry_sdk.integrations.logging import LoggingIntegration

from opentelemetry import trace, metrics
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
from opentelemetry.instrumentation.redis import RedisInstrumentor

from prometheus_client import Counter, Histogram, Gauge, generate_latest
import redis

logger = logging.getLogger(__name__)


class MonitoringService:
    """Centralized monitoring service with Sentry and OpenTelemetry."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._load_config()
        self.is_initialized = False
        self.tracer = None
        self.meter = None
        self.redis_client = None
        
        # Prometheus metrics
        self.request_count = Counter(
            'app_requests_total',
            'Total number of requests',
            ['method', 'endpoint', 'status']
        )
        
        self.request_duration = Histogram(
            'app_request_duration_seconds',
            'Request duration in seconds',
            ['method', 'endpoint']
        )
        
        self.active_users = Gauge(
            'app_active_users',
            'Number of active users'
        )
        
        self.error_count = Counter(
            'app_errors_total',
            'Total number of errors',
            ['error_type', 'module']
        )
        
        self.ai_model_latency = Histogram(
            'ai_model_latency_seconds',
            'AI model inference latency',
            ['model_name', 'operation']
        )
        
        self.database_query_duration = Histogram(
            'database_query_duration_seconds',
            'Database query duration',
            ['query_type', 'table']
        )
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from environment variables."""
        return {
            'sentry': {
                'dsn': os.getenv('SENTRY_DSN'),
                'environment': os.getenv('ENVIRONMENT', 'development'),
                'traces_sample_rate': float(os.getenv('SENTRY_TRACES_SAMPLE_RATE', '0.1')),
                'profiles_sample_rate': float(os.getenv('SENTRY_PROFILES_SAMPLE_RATE', '0.1')),
                'release': os.getenv('APP_VERSION', 'unknown'),
                'server_name': os.getenv('SERVER_NAME', 'teknofest-api')
            },
            'otel': {
                'endpoint': os.getenv('OTEL_EXPORTER_OTLP_ENDPOINT', 'localhost:4317'),
                'service_name': os.getenv('OTEL_SERVICE_NAME', 'teknofest-education-platform'),
                'enabled': os.getenv('OTEL_ENABLED', 'true').lower() == 'true'
            },
            'redis': {
                'url': os.getenv('REDIS_URL', 'redis://localhost:6379/0')
            }
        }
    
    def initialize(self, app=None):
        """Initialize monitoring services."""
        if self.is_initialized:
            logger.warning("Monitoring service already initialized")
            return
        
        # Initialize Sentry
        if self.config['sentry']['dsn']:
            self._initialize_sentry()
        
        # Initialize OpenTelemetry
        if self.config['otel']['enabled']:
            self._initialize_opentelemetry()
        
        # Initialize Redis for metrics storage
        self._initialize_redis()
        
        # Instrument FastAPI if app provided
        if app:
            self._instrument_fastapi(app)
        
        self.is_initialized = True
        logger.info("Monitoring service initialized successfully")
    
    def _initialize_sentry(self):
        """Initialize Sentry SDK."""
        sentry_sdk.init(
            dsn=self.config['sentry']['dsn'],
            environment=self.config['sentry']['environment'],
            traces_sample_rate=self.config['sentry']['traces_sample_rate'],
            profiles_sample_rate=self.config['sentry']['profiles_sample_rate'],
            release=self.config['sentry']['release'],
            server_name=self.config['sentry']['server_name'],
            integrations=[
                FastApiIntegration(transaction_style="endpoint"),
                SqlalchemyIntegration(),
                RedisIntegration(),
                LoggingIntegration(
                    level=logging.INFO,
                    event_level=logging.ERROR
                )
            ],
            before_send=self._before_send_sentry,
            traces_sampler=self._traces_sampler
        )
        logger.info("Sentry initialized successfully")
    
    def _initialize_opentelemetry(self):
        """Initialize OpenTelemetry."""
        # Setup tracing
        trace.set_tracer_provider(TracerProvider())
        tracer_provider = trace.get_tracer_provider()
        
        otlp_exporter = OTLPSpanExporter(
            endpoint=self.config['otel']['endpoint'],
            insecure=True
        )
        
        span_processor = BatchSpanProcessor(otlp_exporter)
        tracer_provider.add_span_processor(span_processor)
        
        self.tracer = trace.get_tracer(
            self.config['otel']['service_name'],
            "1.0.0"
        )
        
        # Setup metrics
        metric_reader = PeriodicExportingMetricReader(
            exporter=OTLPMetricExporter(
                endpoint=self.config['otel']['endpoint'],
                insecure=True
            ),
            export_interval_millis=60000  # Export every minute
        )
        
        metrics.set_meter_provider(
            MeterProvider(metric_readers=[metric_reader])
        )
        
        self.meter = metrics.get_meter(
            self.config['otel']['service_name'],
            "1.0.0"
        )
        
        # Auto-instrument libraries
        SQLAlchemyInstrumentor().instrument()
        RedisInstrumentor().instrument()
        
        logger.info("OpenTelemetry initialized successfully")
    
    def _initialize_redis(self):
        """Initialize Redis client for metrics storage."""
        try:
            self.redis_client = redis.from_url(
                self.config['redis']['url'],
                decode_responses=True
            )
            self.redis_client.ping()
            logger.info("Redis client initialized for monitoring")
        except Exception as e:
            logger.error(f"Failed to initialize Redis: {e}")
            self.redis_client = None
    
    def _instrument_fastapi(self, app):
        """Instrument FastAPI application."""
        FastAPIInstrumentor.instrument_app(app)
        
        @app.middleware("http")
        async def monitoring_middleware(request, call_next):
            start_time = time.time()
            
            # Create span for request
            with self.create_span(f"{request.method} {request.url.path}") as span:
                span.set_attribute("http.method", request.method)
                span.set_attribute("http.url", str(request.url))
                span.set_attribute("http.scheme", request.url.scheme)
                
                # Process request
                response = await call_next(request)
                
                # Record metrics
                duration = time.time() - start_time
                self.request_count.labels(
                    method=request.method,
                    endpoint=request.url.path,
                    status=response.status_code
                ).inc()
                
                self.request_duration.labels(
                    method=request.method,
                    endpoint=request.url.path
                ).observe(duration)
                
                # Set span attributes
                span.set_attribute("http.status_code", response.status_code)
                
                # Record to Redis if available
                if self.redis_client:
                    self._record_request_metrics(
                        request.method,
                        request.url.path,
                        response.status_code,
                        duration
                    )
                
                return response
    
    def _before_send_sentry(self, event, hint):
        """Process event before sending to Sentry."""
        # Filter sensitive data
        if 'request' in event and 'data' in event['request']:
            data = event['request']['data']
            if isinstance(data, dict):
                # Remove sensitive fields
                sensitive_fields = ['password', 'token', 'secret', 'api_key']
                for field in sensitive_fields:
                    if field in data:
                        data[field] = '[REDACTED]'
        
        # Add custom context
        event['contexts']['app'] = {
            'environment': self.config['sentry']['environment'],
            'server_name': self.config['sentry']['server_name']
        }
        
        return event
    
    def _traces_sampler(self, sampling_context):
        """Custom sampling logic for Sentry traces."""
        # Always sample errors
        if sampling_context.get('error'):
            return 1.0
        
        # Sample based on transaction name
        transaction_name = sampling_context.get('transaction_context', {}).get('name', '')
        
        # High priority endpoints
        if any(endpoint in transaction_name for endpoint in ['/api/v1/quiz', '/api/v1/learning-path']):
            return 0.5
        
        # Health check endpoints - low sampling
        if 'health' in transaction_name or 'metrics' in transaction_name:
            return 0.01
        
        # Default sampling rate
        return self.config['sentry']['traces_sample_rate']
    
    @contextmanager
    def create_span(self, name: str, attributes: Optional[Dict[str, Any]] = None):
        """Create a new span for tracing."""
        if self.tracer:
            with self.tracer.start_as_current_span(name) as span:
                if attributes:
                    for key, value in attributes.items():
                        span.set_attribute(key, value)
                yield span
        else:
            yield None
    
    def record_error(self, error: Exception, context: Optional[Dict[str, Any]] = None):
        """Record an error to Sentry and metrics."""
        # Send to Sentry
        sentry_sdk.capture_exception(error)
        
        # Record in metrics
        error_type = type(error).__name__
        module = context.get('module', 'unknown') if context else 'unknown'
        
        self.error_count.labels(
            error_type=error_type,
            module=module
        ).inc()
        
        # Log error
        logger.error(f"Error recorded: {error_type} in {module}", exc_info=error, extra=context)
        
        # Store in Redis if available
        if self.redis_client:
            self._store_error_in_redis(error, context)
    
    def record_ai_metrics(self, model_name: str, operation: str, latency: float, 
                         tokens_used: Optional[int] = None, success: bool = True):
        """Record AI model metrics."""
        self.ai_model_latency.labels(
            model_name=model_name,
            operation=operation
        ).observe(latency)
        
        if self.redis_client:
            key = f"ai_metrics:{model_name}:{operation}:{datetime.now().strftime('%Y%m%d')}"
            self.redis_client.hincrby(key, 'request_count', 1)
            self.redis_client.hincrbyfloat(key, 'total_latency', latency)
            if tokens_used:
                self.redis_client.hincrby(key, 'total_tokens', tokens_used)
            if not success:
                self.redis_client.hincrby(key, 'error_count', 1)
            self.redis_client.expire(key, 86400 * 7)  # Keep for 7 days
    
    def record_database_metrics(self, query_type: str, table: str, duration: float):
        """Record database query metrics."""
        self.database_query_duration.labels(
            query_type=query_type,
            table=table
        ).observe(duration)
        
        if self.redis_client:
            key = f"db_metrics:{table}:{query_type}:{datetime.now().strftime('%Y%m%d%H')}"
            self.redis_client.hincrby(key, 'query_count', 1)
            self.redis_client.hincrbyfloat(key, 'total_duration', duration)
            self.redis_client.expire(key, 86400)  # Keep for 1 day
    
    def set_user_context(self, user_id: str, username: str, email: str, **extra):
        """Set user context for Sentry."""
        sentry_sdk.set_user({
            "id": user_id,
            "username": username,
            "email": email,
            **extra
        })
    
    def add_breadcrumb(self, message: str, category: str = 'custom', 
                      level: str = 'info', data: Optional[Dict] = None):
        """Add breadcrumb for Sentry."""
        sentry_sdk.add_breadcrumb(
            message=message,
            category=category,
            level=level,
            data=data or {}
        )
    
    def _record_request_metrics(self, method: str, path: str, status_code: int, duration: float):
        """Record request metrics to Redis."""
        try:
            # Hourly metrics
            hour_key = f"requests:{datetime.now().strftime('%Y%m%d%H')}"
            self.redis_client.hincrby(hour_key, f"{method}:{path}:{status_code}", 1)
            self.redis_client.expire(hour_key, 86400)  # Keep for 1 day
            
            # Response time percentiles
            latency_key = f"latency:{path}:{datetime.now().strftime('%Y%m%d')}"
            self.redis_client.zadd(latency_key, {str(time.time()): duration})
            self.redis_client.expire(latency_key, 86400 * 7)  # Keep for 7 days
        except Exception as e:
            logger.error(f"Failed to record request metrics: {e}")
    
    def _store_error_in_redis(self, error: Exception, context: Optional[Dict[str, Any]]):
        """Store error details in Redis."""
        try:
            error_data = {
                'timestamp': datetime.now().isoformat(),
                'error_type': type(error).__name__,
                'error_message': str(error),
                'context': context or {}
            }
            
            key = f"errors:{datetime.now().strftime('%Y%m%d')}"
            self.redis_client.lpush(key, json.dumps(error_data))
            self.redis_client.ltrim(key, 0, 999)  # Keep last 1000 errors
            self.redis_client.expire(key, 86400 * 7)  # Keep for 7 days
        except Exception as e:
            logger.error(f"Failed to store error in Redis: {e}")
    
    def get_metrics(self) -> bytes:
        """Get Prometheus metrics."""
        return generate_latest()
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get health status of monitoring services."""
        status = {
            'sentry': 'unknown',
            'opentelemetry': 'unknown',
            'redis': 'unknown',
            'initialized': self.is_initialized
        }
        
        # Check Sentry
        try:
            if self.config['sentry']['dsn']:
                # Sentry doesn't provide a direct health check
                status['sentry'] = 'configured'
        except Exception:
            status['sentry'] = 'error'
        
        # Check OpenTelemetry
        if self.tracer and self.meter:
            status['opentelemetry'] = 'active'
        
        # Check Redis
        if self.redis_client:
            try:
                self.redis_client.ping()
                status['redis'] = 'connected'
            except Exception:
                status['redis'] = 'disconnected'
        
        return status
    
    def shutdown(self):
        """Shutdown monitoring services."""
        if self.redis_client:
            self.redis_client.close()
        
        # Flush Sentry
        sentry_sdk.flush()
        
        self.is_initialized = False
        logger.info("Monitoring service shut down")


# Decorator for monitoring functions
def monitor_function(name: Optional[str] = None, record_args: bool = False):
    """Decorator to monitor function execution."""
    def decorator(func: Callable) -> Callable:
        func_name = name or f"{func.__module__}.{func.__name__}"
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            monitoring = MonitoringService()
            start_time = time.time()
            
            try:
                with monitoring.create_span(func_name) as span:
                    if record_args and span:
                        span.set_attribute("args", str(args))
                        span.set_attribute("kwargs", str(kwargs))
                    
                    result = func(*args, **kwargs)
                    
                    if span:
                        span.set_attribute("success", True)
                    
                    return result
            except Exception as e:
                monitoring.record_error(e, {'function': func_name})
                raise
            finally:
                duration = time.time() - start_time
                if monitoring.redis_client:
                    key = f"function_metrics:{func_name}:{datetime.now().strftime('%Y%m%d')}"
                    monitoring.redis_client.hincrby(key, 'call_count', 1)
                    monitoring.redis_client.hincrbyfloat(key, 'total_duration', duration)
                    monitoring.redis_client.expire(key, 86400)
        
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            monitoring = MonitoringService()
            start_time = time.time()
            
            try:
                with monitoring.create_span(func_name) as span:
                    if record_args and span:
                        span.set_attribute("args", str(args))
                        span.set_attribute("kwargs", str(kwargs))
                    
                    result = await func(*args, **kwargs)
                    
                    if span:
                        span.set_attribute("success", True)
                    
                    return result
            except Exception as e:
                monitoring.record_error(e, {'function': func_name})
                raise
            finally:
                duration = time.time() - start_time
                if monitoring.redis_client:
                    key = f"function_metrics:{func_name}:{datetime.now().strftime('%Y%m%d')}"
                    monitoring.redis_client.hincrby(key, 'call_count', 1)
                    monitoring.redis_client.hincrbyfloat(key, 'total_duration', duration)
                    monitoring.redis_client.expire(key, 86400)
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    
    return decorator


# Global monitoring instance
monitoring_service = MonitoringService()