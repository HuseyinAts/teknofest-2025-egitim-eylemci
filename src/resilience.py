"""
Production-Ready Resilience Patterns
Circuit Breakers, Retry Mechanisms, and Fallbacks
TEKNOFEST 2025 - Eğitim Teknolojileri
"""

import asyncio
import logging
import random
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union
from collections import deque
import threading

from src.exceptions import (
    BaseApplicationException,
    ExternalServiceException,
    DatabaseException,
    ErrorCode
)

logger = logging.getLogger(__name__)

T = TypeVar('T')


class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"      # Circuit is broken, calls fail fast
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker"""
    failure_threshold: int = 5  # Failures before opening
    recovery_timeout: int = 60  # Seconds before trying half-open
    expected_exception: type = Exception  # Exception types to track
    success_threshold: int = 3  # Successes needed to close from half-open
    window_size: int = 60  # Time window for failure counting (seconds)


class CircuitBreakerMetrics:
    """Metrics tracking for circuit breaker"""
    
    def __init__(self, window_size: int = 60):
        self.window_size = window_size
        self.calls = deque()
        self.failures = deque()
        self.lock = threading.Lock()
    
    def record_call(self, success: bool):
        """Record a call result"""
        with self.lock:
            now = time.time()
            self.calls.append(now)
            if not success:
                self.failures.append(now)
            self._cleanup(now)
    
    def _cleanup(self, now: float):
        """Remove old entries outside the window"""
        cutoff = now - self.window_size
        while self.calls and self.calls[0] < cutoff:
            self.calls.popleft()
        while self.failures and self.failures[0] < cutoff:
            self.failures.popleft()
    
    def get_failure_rate(self) -> float:
        """Get current failure rate"""
        with self.lock:
            if not self.calls:
                return 0.0
            return len(self.failures) / len(self.calls)
    
    def get_failure_count(self) -> int:
        """Get current failure count"""
        with self.lock:
            self._cleanup(time.time())
            return len(self.failures)
    
    def reset(self):
        """Reset all metrics"""
        with self.lock:
            self.calls.clear()
            self.failures.clear()


class CircuitBreaker:
    """Circuit breaker implementation"""
    
    def __init__(
        self,
        name: str,
        config: Optional[CircuitBreakerConfig] = None
    ):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.state = CircuitState.CLOSED
        self.last_failure_time = None
        self.consecutive_successes = 0
        self.metrics = CircuitBreakerMetrics(self.config.window_size)
        self._lock = threading.Lock()
    
    def call(self, func: Callable[..., T], *args, **kwargs) -> T:
        """Execute function with circuit breaker protection"""
        if not self._can_execute():
            raise ExternalServiceException(
                service_name=self.name,
                message=f"Circuit breaker is OPEN for {self.name}",
                status_code=503
            )
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except self.config.expected_exception as e:
            self._on_failure()
            raise
    
    async def async_call(self, func: Callable[..., T], *args, **kwargs) -> T:
        """Execute async function with circuit breaker protection"""
        if not self._can_execute():
            raise ExternalServiceException(
                service_name=self.name,
                message=f"Circuit breaker is OPEN for {self.name}",
                status_code=503
            )
        
        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result
        except self.config.expected_exception as e:
            self._on_failure()
            raise
    
    def _can_execute(self) -> bool:
        """Check if call can be executed"""
        with self._lock:
            if self.state == CircuitState.CLOSED:
                return True
            
            if self.state == CircuitState.OPEN:
                # Check if recovery timeout has passed
                if self.last_failure_time:
                    time_since_failure = time.time() - self.last_failure_time
                    if time_since_failure > self.config.recovery_timeout:
                        logger.info(f"Circuit breaker {self.name} entering HALF_OPEN state")
                        self.state = CircuitState.HALF_OPEN
                        return True
                return False
            
            # HALF_OPEN state - allow call
            return True
    
    def _on_success(self):
        """Handle successful call"""
        with self._lock:
            self.metrics.record_call(success=True)
            
            if self.state == CircuitState.HALF_OPEN:
                self.consecutive_successes += 1
                if self.consecutive_successes >= self.config.success_threshold:
                    logger.info(f"Circuit breaker {self.name} transitioning to CLOSED")
                    self.state = CircuitState.CLOSED
                    self.consecutive_successes = 0
                    self.metrics.reset()
    
    def _on_failure(self):
        """Handle failed call"""
        with self._lock:
            self.metrics.record_call(success=False)
            self.last_failure_time = time.time()
            
            if self.state == CircuitState.HALF_OPEN:
                logger.warning(f"Circuit breaker {self.name} reopening from HALF_OPEN")
                self.state = CircuitState.OPEN
                self.consecutive_successes = 0
            elif self.state == CircuitState.CLOSED:
                failure_count = self.metrics.get_failure_count()
                if failure_count >= self.config.failure_threshold:
                    logger.warning(
                        f"Circuit breaker {self.name} opening after {failure_count} failures"
                    )
                    self.state = CircuitState.OPEN
    
    def get_state(self) -> Dict[str, Any]:
        """Get current circuit breaker state"""
        with self._lock:
            return {
                "name": self.name,
                "state": self.state.value,
                "failure_count": self.metrics.get_failure_count(),
                "failure_rate": self.metrics.get_failure_rate(),
                "last_failure_time": self.last_failure_time
            }
    
    def reset(self):
        """Manually reset circuit breaker"""
        with self._lock:
            self.state = CircuitState.CLOSED
            self.last_failure_time = None
            self.consecutive_successes = 0
            self.metrics.reset()
            logger.info(f"Circuit breaker {self.name} manually reset")


# Global circuit breaker registry
_circuit_breakers: Dict[str, CircuitBreaker] = {}


def get_circuit_breaker(
    name: str,
    config: Optional[CircuitBreakerConfig] = None
) -> CircuitBreaker:
    """Get or create a circuit breaker"""
    if name not in _circuit_breakers:
        _circuit_breakers[name] = CircuitBreaker(name, config)
    return _circuit_breakers[name]


@dataclass
class RetryConfig:
    """Configuration for retry mechanism"""
    max_attempts: int = 3
    initial_delay: float = 1.0  # Seconds
    max_delay: float = 60.0  # Seconds
    exponential_base: float = 2.0
    jitter: bool = True  # Add randomization to prevent thundering herd
    retry_on: tuple = (Exception,)  # Exception types to retry


class RetryStrategy:
    """Base class for retry strategies"""
    
    def get_delay(self, attempt: int, config: RetryConfig) -> float:
        """Calculate delay before next retry"""
        raise NotImplementedError


class ExponentialBackoffStrategy(RetryStrategy):
    """Exponential backoff with optional jitter"""
    
    def get_delay(self, attempt: int, config: RetryConfig) -> float:
        delay = min(
            config.initial_delay * (config.exponential_base ** (attempt - 1)),
            config.max_delay
        )
        
        if config.jitter:
            delay = delay * (0.5 + random.random())
        
        return delay


class LinearBackoffStrategy(RetryStrategy):
    """Linear backoff strategy"""
    
    def get_delay(self, attempt: int, config: RetryConfig) -> float:
        delay = min(
            config.initial_delay * attempt,
            config.max_delay
        )
        
        if config.jitter:
            delay = delay * (0.5 + random.random())
        
        return delay


class FixedDelayStrategy(RetryStrategy):
    """Fixed delay between retries"""
    
    def get_delay(self, attempt: int, config: RetryConfig) -> float:
        return config.initial_delay


def retry(
    config: Optional[RetryConfig] = None,
    strategy: Optional[RetryStrategy] = None,
    circuit_breaker: Optional[str] = None
):
    """Decorator for adding retry logic to functions"""
    
    config = config or RetryConfig()
    strategy = strategy or ExponentialBackoffStrategy()
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            last_exception = None
            cb = get_circuit_breaker(circuit_breaker) if circuit_breaker else None
            
            for attempt in range(1, config.max_attempts + 1):
                try:
                    if cb:
                        return await cb.async_call(func, *args, **kwargs)
                    else:
                        return await func(*args, **kwargs)
                
                except config.retry_on as e:
                    last_exception = e
                    
                    if attempt == config.max_attempts:
                        logger.error(
                            f"All {config.max_attempts} retry attempts failed for {func.__name__}",
                            exc_info=True
                        )
                        raise
                    
                    delay = strategy.get_delay(attempt, config)
                    logger.warning(
                        f"Retry attempt {attempt}/{config.max_attempts} for {func.__name__} "
                        f"after {delay:.2f}s delay. Error: {str(e)}"
                    )
                    
                    await asyncio.sleep(delay)
            
            raise last_exception
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            last_exception = None
            cb = get_circuit_breaker(circuit_breaker) if circuit_breaker else None
            
            for attempt in range(1, config.max_attempts + 1):
                try:
                    if cb:
                        return cb.call(func, *args, **kwargs)
                    else:
                        return func(*args, **kwargs)
                
                except config.retry_on as e:
                    last_exception = e
                    
                    if attempt == config.max_attempts:
                        logger.error(
                            f"All {config.max_attempts} retry attempts failed for {func.__name__}",
                            exc_info=True
                        )
                        raise
                    
                    delay = strategy.get_delay(attempt, config)
                    logger.warning(
                        f"Retry attempt {attempt}/{config.max_attempts} for {func.__name__} "
                        f"after {delay:.2f}s delay. Error: {str(e)}"
                    )
                    
                    time.sleep(delay)
            
            raise last_exception
        
        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


class Fallback:
    """Fallback mechanism for graceful degradation"""
    
    def __init__(
        self,
        fallback_value: Any = None,
        fallback_function: Optional[Callable] = None,
        cache_key: Optional[str] = None,
        log_fallback: bool = True
    ):
        self.fallback_value = fallback_value
        self.fallback_function = fallback_function
        self.cache_key = cache_key
        self.log_fallback = log_fallback
        self._cache: Dict[str, Any] = {}
    
    def __call__(self, func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            try:
                result = await func(*args, **kwargs)
                
                # Cache successful result if cache_key provided
                if self.cache_key:
                    self._cache[self.cache_key] = result
                
                return result
            
            except Exception as e:
                if self.log_fallback:
                    logger.warning(
                        f"Using fallback for {func.__name__}: {str(e)}",
                        exc_info=True
                    )
                
                # Try fallback function
                if self.fallback_function:
                    try:
                        return await self.fallback_function(*args, **kwargs)
                    except Exception as fallback_error:
                        logger.error(
                            f"Fallback function failed: {str(fallback_error)}",
                            exc_info=True
                        )
                
                # Try cached value
                if self.cache_key and self.cache_key in self._cache:
                    logger.info(f"Returning cached value for {self.cache_key}")
                    return self._cache[self.cache_key]
                
                # Return static fallback value
                return self.fallback_value
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            try:
                result = func(*args, **kwargs)
                
                # Cache successful result if cache_key provided
                if self.cache_key:
                    self._cache[self.cache_key] = result
                
                return result
            
            except Exception as e:
                if self.log_fallback:
                    logger.warning(
                        f"Using fallback for {func.__name__}: {str(e)}",
                        exc_info=True
                    )
                
                # Try fallback function
                if self.fallback_function:
                    try:
                        return self.fallback_function(*args, **kwargs)
                    except Exception as fallback_error:
                        logger.error(
                            f"Fallback function failed: {str(fallback_error)}",
                            exc_info=True
                        )
                
                # Try cached value
                if self.cache_key and self.cache_key in self._cache:
                    logger.info(f"Returning cached value for {self.cache_key}")
                    return self._cache[self.cache_key]
                
                # Return static fallback value
                return self.fallback_value
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper


class Timeout:
    """Timeout decorator for functions"""
    
    def __init__(self, seconds: float, error_message: Optional[str] = None):
        self.seconds = seconds
        self.error_message = error_message or f"Operation timed out after {seconds} seconds"
    
    def __call__(self, func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            try:
                return await asyncio.wait_for(
                    func(*args, **kwargs),
                    timeout=self.seconds
                )
            except asyncio.TimeoutError:
                raise ExternalServiceException(
                    service_name=func.__name__,
                    message=self.error_message,
                    status_code=504
                )
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            raise ValueError("Timeout decorator only supports async functions")


class BulkheadPool:
    """Bulkhead pattern for resource isolation"""
    
    def __init__(self, name: str, max_concurrent: int = 10, max_queue: int = 100):
        self.name = name
        self.max_concurrent = max_concurrent
        self.max_queue = max_queue
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.queue_size = 0
        self._lock = asyncio.Lock()
    
    async def execute(self, func: Callable, *args, **kwargs):
        """Execute function with bulkhead protection"""
        async with self._lock:
            if self.queue_size >= self.max_queue:
                raise ExternalServiceException(
                    service_name=self.name,
                    message=f"Bulkhead queue full for {self.name}",
                    status_code=503
                )
            self.queue_size += 1
        
        try:
            async with self.semaphore:
                return await func(*args, **kwargs)
        finally:
            async with self._lock:
                self.queue_size -= 1


# Global bulkhead registry
_bulkheads: Dict[str, BulkheadPool] = {}


def get_bulkhead(
    name: str,
    max_concurrent: int = 10,
    max_queue: int = 100
) -> BulkheadPool:
    """Get or create a bulkhead pool"""
    if name not in _bulkheads:
        _bulkheads[name] = BulkheadPool(name, max_concurrent, max_queue)
    return _bulkheads[name]


def resilient(
    retry_config: Optional[RetryConfig] = None,
    circuit_breaker_name: Optional[str] = None,
    fallback_value: Any = None,
    timeout_seconds: Optional[float] = None,
    bulkhead_name: Optional[str] = None
):
    """Combined resilience decorator"""
    
    def decorator(func: Callable) -> Callable:
        # Apply decorators in order
        wrapped = func
        
        if fallback_value is not None:
            wrapped = Fallback(fallback_value=fallback_value)(wrapped)
        
        if retry_config:
            wrapped = retry(
                config=retry_config,
                circuit_breaker=circuit_breaker_name
            )(wrapped)
        
        if timeout_seconds:
            wrapped = Timeout(seconds=timeout_seconds)(wrapped)
        
        if bulkhead_name:
            @wraps(wrapped)
            async def bulkhead_wrapper(*args, **kwargs):
                bulkhead = get_bulkhead(bulkhead_name)
                return await bulkhead.execute(wrapped, *args, **kwargs)
            return bulkhead_wrapper
        
        return wrapped
    
    return decorator