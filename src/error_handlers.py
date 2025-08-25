"""
Production-Ready Error Handlers and Middleware
TEKNOFEST 2025 - Eğitim Teknolojileri

Comprehensive error handling system with:
- Circuit breaker pattern for external services
- Automatic retry mechanisms with exponential backoff
- Error aggregation and reporting
- Detailed error tracking and correlation
- Graceful degradation strategies
"""

import logging
import sys
import time
import asyncio
import json
from typing import Any, Callable, Dict, Optional, Union, List, Tuple
from functools import wraps
from collections import defaultdict, deque
from datetime import datetime, timedelta
import traceback
import hashlib

from fastapi import FastAPI, Request, Response, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from pydantic import ValidationError

from src.exceptions import (
    BaseApplicationException,
    ValidationException,
    AuthenticationException,
    AuthorizationException,
    ExternalServiceException,
    DatabaseException,
    RateLimitException,
    ErrorCode
)
from src.config import Settings, get_settings

logger = logging.getLogger(__name__)

# Error tracking metrics
class ErrorMetrics:
    """Track error metrics for monitoring and alerting"""
    
    def __init__(self, window_size: int = 100, time_window_minutes: int = 5):
        self.errors_by_code = defaultdict(int)
        self.errors_by_endpoint = defaultdict(int)
        self.recent_errors = deque(maxlen=window_size)
        self.time_window = timedelta(minutes=time_window_minutes)
        self.error_timestamps = deque()
        self.circuit_breakers = {}
    
    def record_error(
        self,
        error_code: str,
        endpoint: str,
        error_id: str,
        details: Dict[str, Any]
    ):
        """Record an error occurrence"""
        timestamp = datetime.utcnow()
        
        # Update counters
        self.errors_by_code[error_code] += 1
        self.errors_by_endpoint[endpoint] += 1
        
        # Store recent error
        error_record = {
            "error_id": error_id,
            "error_code": error_code,
            "endpoint": endpoint,
            "timestamp": timestamp.isoformat(),
            "details": details
        }
        self.recent_errors.append(error_record)
        self.error_timestamps.append(timestamp)
        
        # Clean old timestamps
        cutoff = timestamp - self.time_window
        while self.error_timestamps and self.error_timestamps[0] < cutoff:
            self.error_timestamps.popleft()
    
    def get_error_rate(self) -> float:
        """Calculate current error rate per minute"""
        if not self.error_timestamps:
            return 0.0
        
        now = datetime.utcnow()
        cutoff = now - self.time_window
        recent_count = sum(1 for ts in self.error_timestamps if ts > cutoff)
        
        return recent_count / self.time_window.total_seconds() * 60
    
    def get_top_errors(self, limit: int = 10) -> List[Tuple[str, int]]:
        """Get most frequent error codes"""
        return sorted(
            self.errors_by_code.items(),
            key=lambda x: x[1],
            reverse=True
        )[:limit]
    
    def should_circuit_break(self, service: str, threshold: int = 5) -> bool:
        """Check if circuit breaker should be activated"""
        if service not in self.circuit_breakers:
            self.circuit_breakers[service] = {
                "failures": 0,
                "last_failure": None,
                "state": "closed",  # closed, open, half-open
                "next_retry": None
            }
        
        breaker = self.circuit_breakers[service]
        now = datetime.utcnow()
        
        # Check if circuit is open and should retry
        if breaker["state"] == "open":
            if breaker["next_retry"] and now >= breaker["next_retry"]:
                breaker["state"] = "half-open"
                return False
            return True
        
        # Check if threshold exceeded
        if breaker["failures"] >= threshold:
            breaker["state"] = "open"
            breaker["next_retry"] = now + timedelta(seconds=30)
            return True
        
        return False
    
    def record_service_failure(self, service: str):
        """Record a service failure for circuit breaker"""
        if service not in self.circuit_breakers:
            self.circuit_breakers[service] = {
                "failures": 0,
                "last_failure": None,
                "state": "closed",
                "next_retry": None
            }
        
        breaker = self.circuit_breakers[service]
        breaker["failures"] += 1
        breaker["last_failure"] = datetime.utcnow()
    
    def record_service_success(self, service: str):
        """Record a service success"""
        if service in self.circuit_breakers:
            self.circuit_breakers[service]["failures"] = 0
            self.circuit_breakers[service]["state"] = "closed"

# Global error metrics instance
error_metrics = ErrorMetrics()


class RetryStrategy:
    """Retry strategy with exponential backoff"""
    
    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True
    ):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
    
    def get_delay(self, attempt: int) -> float:
        """Calculate delay for retry attempt"""
        delay = min(
            self.base_delay * (self.exponential_base ** attempt),
            self.max_delay
        )
        
        if self.jitter:
            import random
            delay *= (0.5 + random.random())
        
        return delay
    
    async def execute_with_retry(
        self,
        func: Callable,
        *args,
        retry_on: Tuple[type, ...] = (Exception,),
        **kwargs
    ):
        """Execute function with retry logic"""
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                if asyncio.iscoroutinefunction(func):
                    return await func(*args, **kwargs)
                else:
                    return func(*args, **kwargs)
            
            except retry_on as e:
                last_exception = e
                
                if attempt < self.max_retries:
                    delay = self.get_delay(attempt)
                    logger.warning(
                        f"Retry attempt {attempt + 1}/{self.max_retries} "
                        f"after {delay:.2f}s delay. Error: {str(e)}"
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error(
                        f"All {self.max_retries} retry attempts failed",
                        exc_info=True
                    )
        
        raise last_exception


class ErrorResponse:
    """Standardized error response format with enhanced features"""
    
    @staticmethod
    def create(
        error_code: str,
        message: str,
        status_code: int,
        error_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None
    ) -> JSONResponse:
        """Create a standardized error response"""
        
        # Generate error fingerprint for deduplication
        fingerprint = ErrorResponse._generate_fingerprint(error_code, message)
        
        content = {
            "success": False,
            "error": {
                "code": error_code,
                "message": message,
                "fingerprint": fingerprint,
                "timestamp": datetime.utcnow().isoformat()
            }
        }
        
        if error_id:
            content["error"]["error_id"] = error_id
        
        if details:
            content["error"]["details"] = details
        
        # Add support information for critical errors
        if status_code >= 500:
            content["error"]["support"] = {
                "message": "If this error persists, please contact support",
                "reference": error_id or fingerprint
            }
        
        return JSONResponse(
            status_code=status_code,
            content=content,
            headers=headers
        )
    
    @staticmethod
    def _generate_fingerprint(error_code: str, message: str) -> str:
        """Generate a fingerprint for error deduplication"""
        data = f"{error_code}:{message}"
        return hashlib.md5(data.encode()).hexdigest()[:8]


async def application_exception_handler(
    request: Request,
    exc: BaseApplicationException
) -> JSONResponse:
    """Handle application-specific exceptions with enhanced tracking"""
    
    settings = get_settings()
    
    # Record error metrics
    error_metrics.record_error(
        error_code=exc.error_code.value,
        endpoint=request.url.path,
        error_id=exc.error_id,
        details=exc.details
    )
    
    # Log the error with appropriate level
    if exc.status_code >= 500:
        logger.error(
            f"Server error: {exc}",
            extra={
                "error_id": exc.error_id,
                "error_code": exc.error_code,
                "path": request.url.path,
                "method": request.method,
                "client": request.client.host if request.client else None,
                "traceback": exc.traceback
            },
            exc_info=exc.cause
        )
    elif exc.status_code >= 400:
        logger.warning(
            f"Client error: {exc}",
            extra={
                "error_id": exc.error_id,
                "error_code": exc.error_code,
                "path": request.url.path,
                "method": request.method,
                "client": request.client.host if request.client else None
            }
        )
    
    # Prepare response headers
    headers = {}
    
    # Add retry-after header for rate limit exceptions
    if isinstance(exc, RateLimitException) and exc.details.get("retry_after"):
        headers["Retry-After"] = str(exc.details["retry_after"])
    
    # Add correlation ID header
    headers["X-Error-Id"] = exc.error_id
    
    # Check error rate for alerting
    error_rate = error_metrics.get_error_rate()
    if error_rate > 10:  # More than 10 errors per minute
        logger.critical(
            f"High error rate detected: {error_rate:.2f} errors/minute",
            extra={"top_errors": error_metrics.get_top_errors(5)}
        )
    
    # Include internal details only in development
    include_internal = settings.app_debug and not settings.is_production()
    
    response_data = exc.to_dict(include_internal=include_internal)
    
    # Add retry information for retryable errors
    if exc.status_code in [502, 503, 504]:
        response_data["error"]["retry"] = {
            "retryable": True,
            "suggested_delay": 5
        }
    
    return JSONResponse(
        status_code=exc.status_code,
        content=response_data,
        headers=headers
    )


async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError
) -> JSONResponse:
    """Handle Pydantic validation errors"""
    
    logger.warning(
        f"Validation error on {request.url.path}",
        extra={
            "path": request.url.path,
            "method": request.method,
            "errors": exc.errors()
        }
    )
    
    # Transform Pydantic errors to user-friendly format
    validation_errors = []
    for error in exc.errors():
        validation_errors.append({
            "field": ".".join(str(loc) for loc in error["loc"]),
            "message": error["msg"],
            "type": error["type"]
        })
    
    return ErrorResponse.create(
        error_code=ErrorCode.VALIDATION_ERROR,
        message="Request validation failed",
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        details={"validation_errors": validation_errors}
    )


async def http_exception_handler(
    request: Request,
    exc: Union[HTTPException, StarletteHTTPException]
) -> JSONResponse:
    """Handle FastAPI/Starlette HTTP exceptions"""
    
    # Map HTTP status codes to error codes
    status_to_error_code = {
        400: ErrorCode.VALIDATION_ERROR,
        401: ErrorCode.AUTHENTICATION_REQUIRED,
        403: ErrorCode.AUTHORIZATION_FAILED,
        404: ErrorCode.RESOURCE_NOT_FOUND,
        429: ErrorCode.RATE_LIMIT_EXCEEDED,
        500: ErrorCode.UNKNOWN_ERROR,
        502: ErrorCode.EXTERNAL_SERVICE_ERROR,
        503: ErrorCode.EXTERNAL_SERVICE_ERROR
    }
    
    error_code = status_to_error_code.get(exc.status_code, ErrorCode.UNKNOWN_ERROR)
    
    if exc.status_code >= 500:
        logger.error(
            f"HTTP {exc.status_code} on {request.url.path}: {exc.detail}",
            extra={
                "path": request.url.path,
                "method": request.method,
                "status_code": exc.status_code
            }
        )
    
    return ErrorResponse.create(
        error_code=error_code,
        message=str(exc.detail),
        status_code=exc.status_code,
        details=getattr(exc, "headers", None)
    )


async def unhandled_exception_handler(
    request: Request,
    exc: Exception
) -> JSONResponse:
    """Handle unexpected exceptions"""
    
    import uuid
    error_id = str(uuid.uuid4())
    
    # Log the full exception with traceback
    logger.critical(
        f"Unhandled exception on {request.url.path}",
        extra={
            "error_id": error_id,
            "path": request.url.path,
            "method": request.method,
            "exception_type": type(exc).__name__,
            "traceback": traceback.format_exc()
        },
        exc_info=True
    )
    
    settings = get_settings()
    
    # In production, return generic error message
    if settings.is_production():
        message = "An internal server error occurred"
        details = None
    else:
        message = str(exc)
        details = {
            "exception_type": type(exc).__name__,
            "traceback": traceback.format_exc().split('\n')
        }
    
    return ErrorResponse.create(
        error_code=ErrorCode.UNKNOWN_ERROR,
        message=message,
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        error_id=error_id,
        details=details
    )


class ErrorRecoveryManager:
    """Manage error recovery strategies"""
    
    def __init__(self):
        self.recovery_strategies = {}
        self.fallback_responses = {}
    
    def register_recovery(
        self,
        error_type: type,
        recovery_func: Callable
    ):
        """Register a recovery strategy for an error type"""
        self.recovery_strategies[error_type] = recovery_func
    
    def register_fallback(
        self,
        endpoint: str,
        fallback_data: Any
    ):
        """Register fallback response for an endpoint"""
        self.fallback_responses[endpoint] = fallback_data
    
    async def attempt_recovery(
        self,
        error: Exception,
        request: Request
    ) -> Optional[Any]:
        """Attempt to recover from an error"""
        # Try specific recovery strategy
        for error_type, recovery_func in self.recovery_strategies.items():
            if isinstance(error, error_type):
                try:
                    if asyncio.iscoroutinefunction(recovery_func):
                        return await recovery_func(error, request)
                    else:
                        return recovery_func(error, request)
                except Exception as recovery_error:
                    logger.error(
                        f"Recovery failed for {type(error).__name__}: {recovery_error}"
                    )
        
        # Try fallback response
        endpoint = request.url.path
        if endpoint in self.fallback_responses:
            logger.info(f"Using fallback response for {endpoint}")
            return self.fallback_responses[endpoint]
        
        return None

# Global recovery manager
error_recovery = ErrorRecoveryManager()


class ErrorHandlingMiddleware:
    """Enhanced middleware for comprehensive error handling"""
    
    def __init__(self, app: FastAPI):
        self.app = app
        self.settings = get_settings()
        self.retry_strategy = RetryStrategy()
    
    async def __call__(self, request: Request, call_next):
        # Add request ID to context
        import uuid
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        
        # Track request timing
        start_time = time.time()
        
        # Add request context for error tracking
        request.state.error_context = {
            "user_agent": request.headers.get("user-agent"),
            "ip_address": request.client.host if request.client else None,
            "method": request.method,
            "path": request.url.path
        }
        
        try:
            # Check circuit breaker for external service endpoints
            if "/api/external/" in request.url.path:
                service_name = request.url.path.split("/")[-1]
                if error_metrics.should_circuit_break(service_name):
                    logger.warning(f"Circuit breaker open for service: {service_name}")
                    raise ExternalServiceException(
                        service_name=service_name,
                        message="Service temporarily unavailable due to high error rate"
                    )
            
            response = await call_next(request)
            
            # Record successful external service calls
            if "/api/external/" in request.url.path and response.status_code < 400:
                service_name = request.url.path.split("/")[-1]
                error_metrics.record_service_success(service_name)
            
            # Add custom headers
            response.headers["X-Request-Id"] = request_id
            response.headers["X-Process-Time"] = str(time.time() - start_time)
            
            # Add error rate header for monitoring
            response.headers["X-Error-Rate"] = str(error_metrics.get_error_rate())
            
            # Log successful requests in debug mode
            if self.settings.app_debug:
                logger.debug(
                    f"{request.method} {request.url.path} - {response.status_code}",
                    extra={
                        "request_id": request_id,
                        "duration": time.time() - start_time,
                        "status_code": response.status_code
                    }
                )
            
            return response
            
        except Exception as exc:
            # Record service failure if applicable
            if "/api/external/" in request.url.path:
                service_name = request.url.path.split("/")[-1]
                error_metrics.record_service_failure(service_name)
            
            # Attempt error recovery
            recovery_result = await error_recovery.attempt_recovery(exc, request)
            if recovery_result is not None:
                logger.info(f"Error recovered for {request.url.path}")
                return JSONResponse(
                    content={"success": True, "data": recovery_result},
                    headers={
                        "X-Request-Id": request_id,
                        "X-Recovered": "true"
                    }
                )
            
            # Log the error
            logger.error(
                f"Error in middleware for {request.method} {request.url.path}",
                extra={
                    "request_id": request_id,
                    "duration": time.time() - start_time,
                    "exception": str(exc),
                    "context": request.state.error_context
                },
                exc_info=True
            )
            
            # Re-raise to be handled by exception handlers
            raise


def register_error_handlers(app: FastAPI):
    """Register all error handlers with the FastAPI app"""
    
    # Register custom exception handlers
    app.add_exception_handler(BaseApplicationException, application_exception_handler)
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(HTTPException, http_exception_handler)
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)
    app.add_exception_handler(Exception, unhandled_exception_handler)
    
    # Add error handling middleware
    app.add_middleware(ErrorHandlingMiddleware)
    
    # Register default recovery strategies
    error_recovery.register_recovery(
        ExternalServiceException,
        lambda e, r: {"message": "Service temporarily unavailable", "cached": True}
    )
    
    error_recovery.register_recovery(
        DatabaseException,
        lambda e, r: {"message": "Using cached data", "cached": True}
    )
    
    logger.info("Error handlers and recovery strategies registered successfully")


def error_handler(
    exception_type: type = Exception,
    status_code: int = 500,
    error_code: ErrorCode = ErrorCode.UNKNOWN_ERROR,
    message: Optional[str] = None
):
    """Decorator for handling exceptions in route handlers"""
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except exception_type as e:
                error_message = message or str(e)
                
                if isinstance(e, BaseApplicationException):
                    raise e
                
                raise BaseApplicationException(
                    message=error_message,
                    error_code=error_code,
                    status_code=status_code,
                    cause=e
                )
        return wrapper
    return decorator


class ErrorContext:
    """Context manager for error handling with cleanup"""
    
    def __init__(
        self,
        operation: str,
        cleanup: Optional[Callable] = None,
        error_code: ErrorCode = ErrorCode.UNKNOWN_ERROR,
        reraise: bool = True
    ):
        self.operation = operation
        self.cleanup = cleanup
        self.error_code = error_code
        self.reraise = reraise
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        logger.debug(f"Starting operation: {self.operation}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        
        if exc_type is None:
            logger.debug(
                f"Operation completed: {self.operation}",
                extra={"duration": duration}
            )
            return False
        
        # Log the error
        logger.error(
            f"Operation failed: {self.operation}",
            extra={
                "duration": duration,
                "exception_type": exc_type.__name__,
                "exception": str(exc_val)
            },
            exc_info=(exc_type, exc_val, exc_tb)
        )
        
        # Run cleanup if provided
        if self.cleanup:
            try:
                self.cleanup()
            except Exception as cleanup_error:
                logger.error(
                    f"Cleanup failed for {self.operation}: {cleanup_error}",
                    exc_info=True
                )
        
        # Wrap in application exception if needed
        if self.reraise and not isinstance(exc_val, BaseApplicationException):
            raise BaseApplicationException(
                message=f"Operation '{self.operation}' failed: {str(exc_val)}",
                error_code=self.error_code,
                cause=exc_val
            )
        
        return not self.reraise


def validate_request(schema: type):
    """Enhanced decorator for request validation with custom error handling"""
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(request: Request, *args, **kwargs):
            try:
                # Get request body with size limit
                body_bytes = await request.body()
                if len(body_bytes) > 10 * 1024 * 1024:  # 10MB limit
                    raise ValidationException(
                        message="Request body too large (max 10MB)"
                    )
                
                # Parse JSON
                try:
                    body = json.loads(body_bytes)
                except json.JSONDecodeError as e:
                    raise ValidationException(
                        message=f"Invalid JSON: {str(e)}",
                        field="body",
                        value=body_bytes[:100].decode('utf-8', errors='ignore')
                    )
                
                # Validate against schema
                validated_data = schema(**body)
                
                # Add validated data to kwargs
                kwargs['validated_data'] = validated_data
                
                return await func(request, *args, **kwargs)
                
            except ValidationError as e:
                raise ValidationException(
                    message="Request validation failed",
                    validation_errors=[
                        {
                            "field": ".".join(str(loc) for loc in error["loc"]),
                            "message": error["msg"],
                            "type": error["type"]
                        }
                        for error in e.errors()
                    ]
                )
            except ValidationException:
                raise
            except Exception as e:
                raise ValidationException(
                    message=f"Failed to process request: {str(e)}"
                )
        
        return wrapper
    return decorator


def with_error_handling(
    operation: str,
    error_code: ErrorCode = ErrorCode.UNKNOWN_ERROR,
    retry_strategy: Optional[RetryStrategy] = None
):
    """Decorator for automatic error handling and retry"""
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            retry = retry_strategy or RetryStrategy(max_retries=0)
            
            try:
                if asyncio.iscoroutinefunction(func):
                    return await retry.execute_with_retry(
                        func, *args, **kwargs
                    )
                else:
                    return retry.execute_with_retry(
                        func, *args, **kwargs
                    )
            except BaseApplicationException:
                raise
            except Exception as e:
                logger.error(
                    f"Operation '{operation}' failed: {str(e)}",
                    exc_info=True
                )
                raise BaseApplicationException(
                    message=f"Operation '{operation}' failed",
                    error_code=error_code,
                    cause=e
                )
        
        return wrapper
    return decorator


async def get_error_statistics() -> Dict[str, Any]:
    """Get current error statistics for monitoring"""
    return {
        "error_rate": error_metrics.get_error_rate(),
        "top_errors": dict(error_metrics.get_top_errors()),
        "recent_errors": list(error_metrics.recent_errors),
        "circuit_breakers": error_metrics.circuit_breakers,
        "timestamp": datetime.utcnow().isoformat()
    }