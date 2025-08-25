"""
Production-Ready Error Tracking and Monitoring
TEKNOFEST 2025 - Eğitim Teknolojileri
"""

import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Callable
import threading
import json
from contextlib import contextmanager

from src.config import Settings, get_settings
from src.exceptions import BaseApplicationException

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of metrics"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


@dataclass
class Metric:
    """Metric data structure"""
    name: str
    type: MetricType
    value: float
    labels: Dict[str, str] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


class MetricsCollector:
    """Collects and aggregates application metrics"""
    
    def __init__(self, window_size: int = 300):  # 5 minutes default
        self.window_size = window_size
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.counters: Dict[str, float] = defaultdict(float)
        self.gauges: Dict[str, float] = {}
        self.histograms: Dict[str, List[float]] = defaultdict(list)
        self._lock = threading.Lock()
    
    def increment_counter(
        self,
        name: str,
        value: float = 1.0,
        labels: Optional[Dict[str, str]] = None
    ):
        """Increment a counter metric"""
        with self._lock:
            key = self._make_key(name, labels)
            self.counters[key] += value
            self._record_metric(
                Metric(name, MetricType.COUNTER, self.counters[key], labels or {})
            )
    
    def set_gauge(
        self,
        name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None
    ):
        """Set a gauge metric"""
        with self._lock:
            key = self._make_key(name, labels)
            self.gauges[key] = value
            self._record_metric(
                Metric(name, MetricType.GAUGE, value, labels or {})
            )
    
    def record_histogram(
        self,
        name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None
    ):
        """Record a histogram metric"""
        with self._lock:
            key = self._make_key(name, labels)
            self.histograms[key].append(value)
            # Keep only recent values
            if len(self.histograms[key]) > 1000:
                self.histograms[key] = self.histograms[key][-1000:]
            self._record_metric(
                Metric(name, MetricType.HISTOGRAM, value, labels or {})
            )
    
    def _make_key(self, name: str, labels: Optional[Dict[str, str]]) -> str:
        """Create a unique key for metric with labels"""
        if not labels:
            return name
        label_str = json.dumps(labels, sort_keys=True)
        return f"{name}:{label_str}"
    
    def _record_metric(self, metric: Metric):
        """Record a metric"""
        self.metrics[metric.name].append(metric)
        self._cleanup_old_metrics(metric.name)
    
    def _cleanup_old_metrics(self, name: str):
        """Remove metrics outside the window"""
        cutoff = time.time() - self.window_size
        while self.metrics[name] and self.metrics[name][0].timestamp < cutoff:
            self.metrics[name].popleft()
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get all current metrics"""
        with self._lock:
            return {
                "counters": dict(self.counters),
                "gauges": dict(self.gauges),
                "histograms": {
                    k: {
                        "count": len(v),
                        "sum": sum(v),
                        "avg": sum(v) / len(v) if v else 0,
                        "min": min(v) if v else 0,
                        "max": max(v) if v else 0
                    }
                    for k, v in self.histograms.items()
                }
            }
    
    def get_error_metrics(self) -> Dict[str, Any]:
        """Get error-specific metrics"""
        with self._lock:
            error_counters = {
                k: v for k, v in self.counters.items()
                if "error" in k.lower() or "exception" in k.lower()
            }
            
            return {
                "total_errors": sum(error_counters.values()),
                "error_breakdown": error_counters,
                "error_rate": self._calculate_error_rate()
            }
    
    def _calculate_error_rate(self) -> float:
        """Calculate current error rate"""
        total_requests = self.counters.get("http_requests_total", 0)
        total_errors = self.counters.get("http_errors_total", 0)
        
        if total_requests == 0:
            return 0.0
        
        return (total_errors / total_requests) * 100


# Global metrics collector
_metrics_collector = MetricsCollector()


def get_metrics_collector() -> MetricsCollector:
    """Get the global metrics collector"""
    return _metrics_collector


class ErrorTracker:
    """Tracks and aggregates error information"""
    
    def __init__(self, max_errors: int = 1000):
        self.max_errors = max_errors
        self.errors: deque = deque(maxlen=max_errors)
        self.error_counts: Dict[str, int] = defaultdict(int)
        self.error_by_endpoint: Dict[str, List[Dict]] = defaultdict(list)
        self._lock = threading.Lock()
    
    def track_error(
        self,
        error: Exception,
        request_path: Optional[str] = None,
        request_method: Optional[str] = None,
        user_id: Optional[str] = None,
        additional_context: Optional[Dict[str, Any]] = None
    ):
        """Track an error occurrence"""
        with self._lock:
            error_info = {
                "timestamp": datetime.utcnow().isoformat(),
                "error_type": type(error).__name__,
                "error_message": str(error),
                "request_path": request_path,
                "request_method": request_method,
                "user_id": user_id,
                "context": additional_context or {}
            }
            
            # Add error ID if it's an application exception
            if isinstance(error, BaseApplicationException):
                error_info["error_id"] = error.error_id
                error_info["error_code"] = error.error_code.value
                error_info["status_code"] = error.status_code
            
            self.errors.append(error_info)
            self.error_counts[error_info["error_type"]] += 1
            
            if request_path:
                endpoint_key = f"{request_method or 'UNKNOWN'} {request_path}"
                self.error_by_endpoint[endpoint_key].append(error_info)
                # Keep only recent errors per endpoint
                if len(self.error_by_endpoint[endpoint_key]) > 100:
                    self.error_by_endpoint[endpoint_key] = \
                        self.error_by_endpoint[endpoint_key][-100:]
            
            # Update metrics
            metrics = get_metrics_collector()
            metrics.increment_counter(
                "errors_total",
                labels={
                    "error_type": error_info["error_type"],
                    "path": request_path or "unknown"
                }
            )
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get error summary statistics"""
        with self._lock:
            recent_errors = list(self.errors)[-100:]  # Last 100 errors
            
            return {
                "total_errors": len(self.errors),
                "error_types": dict(self.error_counts),
                "top_endpoints": self._get_top_error_endpoints(),
                "recent_errors": recent_errors,
                "error_rate_by_time": self._calculate_error_rate_by_time()
            }
    
    def _get_top_error_endpoints(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get endpoints with most errors"""
        endpoint_counts = [
            {
                "endpoint": endpoint,
                "error_count": len(errors),
                "latest_error": errors[-1]["timestamp"] if errors else None
            }
            for endpoint, errors in self.error_by_endpoint.items()
        ]
        
        return sorted(
            endpoint_counts,
            key=lambda x: x["error_count"],
            reverse=True
        )[:limit]
    
    def _calculate_error_rate_by_time(self) -> Dict[str, int]:
        """Calculate error rate over time buckets"""
        now = datetime.utcnow()
        buckets = {
            "last_minute": 0,
            "last_5_minutes": 0,
            "last_15_minutes": 0,
            "last_hour": 0,
            "last_24_hours": 0
        }
        
        for error in self.errors:
            error_time = datetime.fromisoformat(error["timestamp"])
            time_diff = now - error_time
            
            if time_diff <= timedelta(minutes=1):
                buckets["last_minute"] += 1
            if time_diff <= timedelta(minutes=5):
                buckets["last_5_minutes"] += 1
            if time_diff <= timedelta(minutes=15):
                buckets["last_15_minutes"] += 1
            if time_diff <= timedelta(hours=1):
                buckets["last_hour"] += 1
            if time_diff <= timedelta(hours=24):
                buckets["last_24_hours"] += 1
        
        return buckets


# Global error tracker
_error_tracker = ErrorTracker()


def get_error_tracker() -> ErrorTracker:
    """Get the global error tracker"""
    return _error_tracker


class SentryIntegration:
    """Sentry error tracking integration"""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.enabled = False
        self.sentry_sdk = None
        
        if settings.sentry_dsn:
            try:
                import sentry_sdk
                from sentry_sdk.integrations.fastapi import FastApiIntegration
                from sentry_sdk.integrations.starlette import StarletteIntegration
                from sentry_sdk.integrations.logging import LoggingIntegration
                
                sentry_sdk.init(
                    dsn=settings.sentry_dsn,
                    environment=settings.sentry_environment,
                    traces_sample_rate=settings.sentry_traces_sample_rate,
                    integrations=[
                        FastApiIntegration(transaction_style="endpoint"),
                        StarletteIntegration(transaction_style="endpoint"),
                        LoggingIntegration(
                            level=logging.INFO,
                            event_level=logging.ERROR
                        )
                    ],
                    before_send=self._before_send,
                    attach_stacktrace=True,
                    send_default_pii=False  # Don't send PII
                )
                
                self.sentry_sdk = sentry_sdk
                self.enabled = True
                logger.info("Sentry integration enabled")
                
            except ImportError:
                logger.warning("Sentry SDK not installed, error tracking disabled")
    
    def _before_send(self, event: Dict[str, Any], hint: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process event before sending to Sentry"""
        # Filter out sensitive data
        if "request" in event:
            request = event["request"]
            # Remove sensitive headers
            if "headers" in request:
                sensitive_headers = ["authorization", "cookie", "x-api-key"]
                for header in sensitive_headers:
                    request["headers"].pop(header, None)
            
            # Remove sensitive data from body
            if "data" in request:
                self._sanitize_data(request["data"])
        
        # Add custom context
        if "exc_info" in hint:
            exc = hint["exc_info"][1]
            if isinstance(exc, BaseApplicationException):
                event["extra"]["error_id"] = exc.error_id
                event["extra"]["error_code"] = exc.error_code.value
                event["tags"]["error_code"] = exc.error_code.value
        
        return event
    
    def _sanitize_data(self, data: Any):
        """Remove sensitive data from request/response"""
        if isinstance(data, dict):
            sensitive_fields = ["password", "token", "secret", "api_key", "credit_card"]
            for key in list(data.keys()):
                if any(field in key.lower() for field in sensitive_fields):
                    data[key] = "[REDACTED]"
                elif isinstance(data[key], (dict, list)):
                    self._sanitize_data(data[key])
        elif isinstance(data, list):
            for item in data:
                if isinstance(item, (dict, list)):
                    self._sanitize_data(item)
    
    def capture_exception(
        self,
        error: Exception,
        context: Optional[Dict[str, Any]] = None
    ):
        """Manually capture an exception"""
        if self.enabled and self.sentry_sdk:
            with self.sentry_sdk.push_scope() as scope:
                if context:
                    for key, value in context.items():
                        scope.set_context(key, value)
                
                if isinstance(error, BaseApplicationException):
                    scope.set_tag("error_code", error.error_code.value)
                    scope.set_extra("error_id", error.error_id)
                
                self.sentry_sdk.capture_exception(error)
    
    def capture_message(
        self,
        message: str,
        level: str = "info",
        context: Optional[Dict[str, Any]] = None
    ):
        """Capture a message"""
        if self.enabled and self.sentry_sdk:
            with self.sentry_sdk.push_scope() as scope:
                if context:
                    for key, value in context.items():
                        scope.set_context(key, value)
                
                self.sentry_sdk.capture_message(message, level=level)


# Initialize Sentry integration
_sentry_integration: Optional[SentryIntegration] = None


def initialize_monitoring():
    """Initialize monitoring and error tracking"""
    global _sentry_integration
    
    settings = get_settings()
    
    # Initialize Sentry
    _sentry_integration = SentryIntegration(settings)
    
    # Initialize Prometheus metrics if enabled
    if settings.metrics_enabled:
        try:
            from prometheus_client import start_http_server, Counter, Histogram, Gauge
            
            # Start metrics server
            start_http_server(settings.metrics_port)
            logger.info(f"Prometheus metrics server started on port {settings.metrics_port}")
            
        except ImportError:
            logger.warning("Prometheus client not installed, metrics disabled")
    
    logger.info("Monitoring initialized")


def get_sentry() -> Optional[SentryIntegration]:
    """Get Sentry integration"""
    return _sentry_integration


@contextmanager
def monitor_operation(
    operation_name: str,
    track_duration: bool = True,
    track_errors: bool = True,
    custom_tags: Optional[Dict[str, str]] = None
):
    """Context manager for monitoring operations"""
    start_time = time.time()
    metrics = get_metrics_collector()
    error_tracker = get_error_tracker()
    
    # Increment operation counter
    metrics.increment_counter(
        f"operation_{operation_name}_total",
        labels=custom_tags
    )
    
    try:
        yield
        
        # Record success
        metrics.increment_counter(
            f"operation_{operation_name}_success",
            labels=custom_tags
        )
        
    except Exception as e:
        # Record failure
        metrics.increment_counter(
            f"operation_{operation_name}_failure",
            labels=custom_tags
        )
        
        if track_errors:
            error_tracker.track_error(
                e,
                additional_context={
                    "operation": operation_name,
                    "tags": custom_tags
                }
            )
        
        # Re-raise the exception
        raise
    
    finally:
        if track_duration:
            duration = time.time() - start_time
            metrics.record_histogram(
                f"operation_{operation_name}_duration_seconds",
                duration,
                labels=custom_tags
            )


def track_metric(metric_type: str, name: str):
    """Decorator for tracking metrics"""
    
    def decorator(func: Callable) -> Callable:
        from functools import wraps
        
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            with monitor_operation(
                operation_name=f"{func.__name__}",
                custom_tags={"type": metric_type}
            ):
                return await func(*args, **kwargs)
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            with monitor_operation(
                operation_name=f"{func.__name__}",
                custom_tags={"type": metric_type}
            ):
                return func(*args, **kwargs)
        
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator