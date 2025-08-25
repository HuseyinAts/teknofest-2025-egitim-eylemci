"""
Comprehensive Error Handling Tests
TEKNOFEST 2025 - Eğitim Teknolojileri
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, patch, AsyncMock
from typing import Optional

from src.exceptions import (
    BaseApplicationException,
    ValidationException,
    ResourceNotFoundException,
    DuplicateResourceException,
    BusinessRuleViolationException,
    AuthenticationException,
    AuthorizationException,
    TokenException,
    ExternalServiceException,
    DatabaseException,
    RateLimitException,
    ModelException,
    ModelNotFoundException,
    ModelLoadingException,
    ModelInferenceException,
    EducationException,
    InvalidCurriculumException,
    StudentNotFoundException,
    AssessmentException,
    ErrorCode
)

from src.resilience import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitState,
    RetryConfig,
    ExponentialBackoffStrategy,
    LinearBackoffStrategy,
    FixedDelayStrategy,
    retry,
    Fallback,
    Timeout,
    BulkheadPool,
    resilient,
    get_circuit_breaker,
    get_bulkhead
)

from src.monitoring import (
    MetricsCollector,
    ErrorTracker,
    monitor_operation,
    track_metric,
    get_metrics_collector,
    get_error_tracker
)


class TestExceptions:
    """Test custom exception hierarchy"""
    
    def test_base_exception_creation(self):
        """Test base exception creation with all fields"""
        exc = BaseApplicationException(
            message="Test error",
            error_code=ErrorCode.UNKNOWN_ERROR,
            status_code=500,
            details={"key": "value"},
            internal_message="Internal details",
            cause=ValueError("Original error")
        )
        
        assert exc.message == "Test error"
        assert exc.error_code == ErrorCode.UNKNOWN_ERROR
        assert exc.status_code == 500
        assert exc.details == {"key": "value"}
        assert exc.internal_message == "Internal details"
        assert exc.cause is not None
        assert exc.error_id is not None
        assert exc.timestamp is not None
    
    def test_exception_to_dict(self):
        """Test exception serialization"""
        exc = BaseApplicationException(
            message="Test error",
            error_code=ErrorCode.VALIDATION_ERROR,
            status_code=400,
            details={"field": "username"}
        )
        
        result = exc.to_dict(include_internal=False)
        
        assert "error" in result
        assert result["error"]["code"] == ErrorCode.VALIDATION_ERROR.value
        assert result["error"]["message"] == "Test error"
        assert result["error"]["error_id"] == exc.error_id
        assert result["error"]["details"] == {"field": "username"}
        assert "internal_message" not in result["error"]
    
    def test_validation_exception(self):
        """Test validation exception"""
        exc = ValidationException(
            message="Invalid input",
            field="email",
            value="invalid-email",
            validation_errors=[{"field": "email", "message": "Invalid format"}]
        )
        
        assert exc.status_code == 400
        assert exc.error_code == ErrorCode.VALIDATION_ERROR
        assert exc.details["field"] == "email"
        assert "validation_errors" in exc.details
    
    def test_resource_not_found_exception(self):
        """Test resource not found exception"""
        exc = ResourceNotFoundException(
            resource_type="User",
            resource_id="123"
        )
        
        assert exc.status_code == 404
        assert exc.error_code == ErrorCode.RESOURCE_NOT_FOUND
        assert "123" in exc.message
        assert exc.details["resource_type"] == "User"
        assert exc.details["resource_id"] == "123"
    
    def test_duplicate_resource_exception(self):
        """Test duplicate resource exception"""
        exc = DuplicateResourceException(
            resource_type="Email",
            identifier="test@example.com"
        )
        
        assert exc.status_code == 409
        assert exc.error_code == ErrorCode.RESOURCE_ALREADY_EXISTS
        assert "test@example.com" in exc.message
    
    def test_authentication_exception(self):
        """Test authentication exception"""
        exc = AuthenticationException()
        
        assert exc.status_code == 401
        assert exc.error_code == ErrorCode.AUTHENTICATION_REQUIRED
    
    def test_authorization_exception(self):
        """Test authorization exception"""
        exc = AuthorizationException(
            required_permission="admin",
            user_permissions=["user", "viewer"]
        )
        
        assert exc.status_code == 403
        assert exc.error_code == ErrorCode.AUTHORIZATION_FAILED
        assert exc.details["required_permission"] == "admin"
    
    def test_token_exception(self):
        """Test token exception"""
        exc = TokenException(
            message="Token has expired",
            token_type="refresh",
            expired=True
        )
        
        assert exc.status_code == 401
        assert exc.error_code == ErrorCode.TOKEN_EXPIRED
        assert exc.details["token_type"] == "refresh"
        assert exc.details["expired"] is True
    
    def test_external_service_exception(self):
        """Test external service exception"""
        exc = ExternalServiceException(
            service_name="PaymentAPI",
            message="Service unavailable",
            status_code=503,
            response_body="Gateway timeout"
        )
        
        assert exc.status_code == 502
        assert exc.error_code == ErrorCode.EXTERNAL_SERVICE_ERROR
        assert exc.details["service"] == "PaymentAPI"
        assert exc.details["external_status_code"] == 503
    
    def test_database_exception(self):
        """Test database exception"""
        exc = DatabaseException(
            message="Connection failed",
            operation="INSERT",
            table="users"
        )
        
        assert exc.status_code == 503
        assert exc.error_code == ErrorCode.DATABASE_ERROR
        assert exc.details["operation"] == "INSERT"
        assert exc.details["table"] == "users"
    
    def test_rate_limit_exception(self):
        """Test rate limit exception"""
        exc = RateLimitException(
            limit=100,
            window=60,
            retry_after=30
        )
        
        assert exc.status_code == 429
        assert exc.error_code == ErrorCode.RATE_LIMIT_EXCEEDED
        assert exc.details["limit"] == 100
        assert exc.details["retry_after"] == 30
    
    def test_model_exceptions(self):
        """Test AI/ML model exceptions"""
        # Model not found
        exc1 = ModelNotFoundException("gpt-4")
        assert exc1.status_code == 503
        assert exc1.error_code == ErrorCode.MODEL_NOT_FOUND
        assert "gpt-4" in exc1.message
        
        # Model loading failed
        exc2 = ModelLoadingException(
            model_name="bert-base",
            reason="Out of memory"
        )
        assert exc2.error_code == ErrorCode.MODEL_LOADING_FAILED
        assert "Out of memory" in exc2.message
        
        # Model inference failed
        exc3 = ModelInferenceException(
            model_name="transformer",
            input_data={"text": "test"}
        )
        assert exc3.error_code == ErrorCode.MODEL_INFERENCE_FAILED
    
    def test_education_exceptions(self):
        """Test education domain exceptions"""
        # Invalid curriculum
        exc1 = InvalidCurriculumException(
            message="Invalid grade level",
            grade="13",
            subject="Math"
        )
        assert exc1.error_code == ErrorCode.INVALID_CURRICULUM
        assert exc1.details["grade"] == "13"
        
        # Student not found
        exc2 = StudentNotFoundException("STU-123")
        assert exc2.error_code == ErrorCode.STUDENT_NOT_FOUND
        assert "STU-123" in exc2.message
        
        # Assessment failed
        exc3 = AssessmentException(
            message="Assessment processing failed",
            assessment_type="quiz",
            reason="Invalid format"
        )
        assert exc3.error_code == ErrorCode.ASSESSMENT_FAILED
        assert exc3.details["assessment_type"] == "quiz"


class TestCircuitBreaker:
    """Test circuit breaker implementation"""
    
    def test_circuit_breaker_closed_state(self):
        """Test circuit breaker in closed state"""
        config = CircuitBreakerConfig(failure_threshold=3)
        cb = CircuitBreaker("test_service", config)
        
        assert cb.state == CircuitState.CLOSED
        
        # Successful calls should work
        result = cb.call(lambda: "success")
        assert result == "success"
    
    def test_circuit_breaker_opens_on_failures(self):
        """Test circuit breaker opens after threshold failures"""
        config = CircuitBreakerConfig(
            failure_threshold=3,
            expected_exception=ValueError
        )
        cb = CircuitBreaker("test_service", config)
        
        def failing_function():
            raise ValueError("Test error")
        
        # Fail threshold times
        for _ in range(3):
            with pytest.raises(ValueError):
                cb.call(failing_function)
        
        assert cb.state == CircuitState.OPEN
        
        # Further calls should fail fast
        with pytest.raises(ExternalServiceException) as exc_info:
            cb.call(lambda: "success")
        
        assert "Circuit breaker is OPEN" in str(exc_info.value)
    
    def test_circuit_breaker_half_open_transition(self):
        """Test circuit breaker transitions to half-open"""
        config = CircuitBreakerConfig(
            failure_threshold=2,
            recovery_timeout=0.1,  # 100ms for testing
            expected_exception=ValueError
        )
        cb = CircuitBreaker("test_service", config)
        
        # Open the circuit
        for _ in range(2):
            with pytest.raises(ValueError):
                cb.call(lambda: 1/0)
        
        assert cb.state == CircuitState.OPEN
        
        # Wait for recovery timeout
        time.sleep(0.2)
        
        # Should transition to half-open and allow call
        result = cb.call(lambda: "recovered")
        assert result == "recovered"
        
        # After success, should still be half-open (need more successes)
        assert cb.state == CircuitState.HALF_OPEN
    
    def test_circuit_breaker_closes_after_successes(self):
        """Test circuit breaker closes after successful calls in half-open"""
        config = CircuitBreakerConfig(
            failure_threshold=2,
            recovery_timeout=0.1,
            success_threshold=2,
            expected_exception=ValueError
        )
        cb = CircuitBreaker("test_service", config)
        
        # Open the circuit
        for _ in range(2):
            with pytest.raises(ValueError):
                cb.call(lambda: 1/0)
        
        # Wait and transition to half-open
        time.sleep(0.2)
        
        # Success calls to close circuit
        for _ in range(2):
            cb.call(lambda: "success")
        
        assert cb.state == CircuitState.CLOSED
    
    @pytest.mark.asyncio
    async def test_async_circuit_breaker(self):
        """Test async circuit breaker"""
        config = CircuitBreakerConfig(failure_threshold=2)
        cb = CircuitBreaker("async_service", config)
        
        async def async_success():
            await asyncio.sleep(0.01)
            return "async_result"
        
        result = await cb.async_call(async_success)
        assert result == "async_result"
    
    def test_circuit_breaker_metrics(self):
        """Test circuit breaker metrics"""
        config = CircuitBreakerConfig(failure_threshold=5)
        cb = CircuitBreaker("metrics_test", config)
        
        # Mix of successes and failures
        cb.call(lambda: "success")
        cb.call(lambda: "success")
        
        try:
            cb.call(lambda: 1/0)
        except:
            pass
        
        state = cb.get_state()
        assert state["name"] == "metrics_test"
        assert state["state"] == "closed"
        assert state["failure_count"] > 0


class TestRetryMechanism:
    """Test retry mechanisms"""
    
    @pytest.mark.asyncio
    async def test_retry_decorator_success(self):
        """Test retry decorator with eventual success"""
        call_count = 0
        
        @retry(RetryConfig(max_attempts=3, initial_delay=0.01))
        async def flaky_function():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("Temporary error")
            return "success"
        
        result = await flaky_function()
        assert result == "success"
        assert call_count == 3
    
    @pytest.mark.asyncio
    async def test_retry_decorator_failure(self):
        """Test retry decorator exhausting attempts"""
        call_count = 0
        
        @retry(RetryConfig(max_attempts=3, initial_delay=0.01))
        async def always_fails():
            nonlocal call_count
            call_count += 1
            raise ValueError("Permanent error")
        
        with pytest.raises(ValueError):
            await always_fails()
        
        assert call_count == 3
    
    def test_exponential_backoff_strategy(self):
        """Test exponential backoff calculation"""
        config = RetryConfig(
            initial_delay=1.0,
            exponential_base=2.0,
            max_delay=10.0,
            jitter=False
        )
        strategy = ExponentialBackoffStrategy()
        
        assert strategy.get_delay(1, config) == 1.0  # 1 * 2^0
        assert strategy.get_delay(2, config) == 2.0  # 1 * 2^1
        assert strategy.get_delay(3, config) == 4.0  # 1 * 2^2
        assert strategy.get_delay(4, config) == 8.0  # 1 * 2^3
        assert strategy.get_delay(5, config) == 10.0  # Capped at max_delay
    
    def test_linear_backoff_strategy(self):
        """Test linear backoff calculation"""
        config = RetryConfig(
            initial_delay=1.0,
            max_delay=5.0,
            jitter=False
        )
        strategy = LinearBackoffStrategy()
        
        assert strategy.get_delay(1, config) == 1.0
        assert strategy.get_delay(2, config) == 2.0
        assert strategy.get_delay(3, config) == 3.0
        assert strategy.get_delay(6, config) == 5.0  # Capped at max_delay
    
    def test_fixed_delay_strategy(self):
        """Test fixed delay strategy"""
        config = RetryConfig(initial_delay=2.0)
        strategy = FixedDelayStrategy()
        
        assert strategy.get_delay(1, config) == 2.0
        assert strategy.get_delay(5, config) == 2.0
        assert strategy.get_delay(10, config) == 2.0
    
    @pytest.mark.asyncio
    async def test_retry_with_circuit_breaker(self):
        """Test retry decorator with circuit breaker integration"""
        call_count = 0
        
        # Reset any existing circuit breaker
        cb = get_circuit_breaker("retry_test")
        cb.reset()
        
        @retry(
            RetryConfig(max_attempts=3, initial_delay=0.01),
            circuit_breaker="retry_test"
        )
        async def protected_function():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ValueError("First attempt fails")
            return "success"
        
        result = await protected_function()
        assert result == "success"
        assert call_count == 2


class TestFallback:
    """Test fallback mechanisms"""
    
    def test_fallback_with_static_value(self):
        """Test fallback with static value"""
        @Fallback(fallback_value="default", log_fallback=False)
        def failing_function():
            raise ValueError("Error")
        
        result = failing_function()
        assert result == "default"
    
    def test_fallback_with_function(self):
        """Test fallback with fallback function"""
        def fallback_func():
            return "fallback_result"
        
        @Fallback(fallback_function=fallback_func, log_fallback=False)
        def failing_function():
            raise ValueError("Error")
        
        result = failing_function()
        assert result == "fallback_result"
    
    def test_fallback_with_cache(self):
        """Test fallback with caching"""
        call_count = 0
        
        @Fallback(cache_key="test_cache", log_fallback=False)
        def sometimes_fails():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return "cached_value"
            raise ValueError("Error")
        
        # First call succeeds and caches
        result1 = sometimes_fails()
        assert result1 == "cached_value"
        
        # Second call fails but returns cached value
        result2 = sometimes_fails()
        assert result2 == "cached_value"
    
    @pytest.mark.asyncio
    async def test_async_fallback(self):
        """Test async fallback"""
        @Fallback(fallback_value="async_default", log_fallback=False)
        async def async_failing():
            await asyncio.sleep(0.01)
            raise ValueError("Async error")
        
        result = await async_failing()
        assert result == "async_default"


class TestMonitoring:
    """Test monitoring and metrics"""
    
    def test_metrics_collector_counters(self):
        """Test metrics counter collection"""
        collector = MetricsCollector()
        
        collector.increment_counter("test_counter", 1.0)
        collector.increment_counter("test_counter", 2.0)
        collector.increment_counter("test_counter", 3.0, {"env": "test"})
        
        metrics = collector.get_metrics()
        assert collector.counters["test_counter"] == 3.0
    
    def test_metrics_collector_gauges(self):
        """Test metrics gauge collection"""
        collector = MetricsCollector()
        
        collector.set_gauge("memory_usage", 75.5)
        collector.set_gauge("cpu_usage", 45.2)
        
        metrics = collector.get_metrics()
        assert collector.gauges["memory_usage"] == 75.5
        assert collector.gauges["cpu_usage"] == 45.2
    
    def test_metrics_collector_histograms(self):
        """Test metrics histogram collection"""
        collector = MetricsCollector()
        
        for value in [1.0, 2.0, 3.0, 4.0, 5.0]:
            collector.record_histogram("response_time", value)
        
        metrics = collector.get_metrics()
        histogram = metrics["histograms"]["response_time"]
        
        assert histogram["count"] == 5
        assert histogram["sum"] == 15.0
        assert histogram["avg"] == 3.0
        assert histogram["min"] == 1.0
        assert histogram["max"] == 5.0
    
    def test_error_tracker(self):
        """Test error tracking"""
        tracker = ErrorTracker()
        
        # Track different error types
        tracker.track_error(
            ValueError("Test error"),
            request_path="/api/test",
            request_method="POST",
            user_id="user123"
        )
        
        tracker.track_error(
            ValidationException("Invalid input", field="email"),
            request_path="/api/users",
            request_method="POST"
        )
        
        summary = tracker.get_error_summary()
        
        assert summary["total_errors"] >= 2
        assert "ValueError" in summary["error_types"]
        assert "ValidationException" in summary["error_types"]
    
    def test_monitor_operation_context(self):
        """Test monitor operation context manager"""
        collector = get_metrics_collector()
        
        with monitor_operation("test_operation", custom_tags={"env": "test"}):
            # Simulate some work
            time.sleep(0.01)
        
        # Check metrics were recorded
        metrics = collector.get_metrics()
        assert any("test_operation" in key for key in collector.counters.keys())
    
    def test_monitor_operation_with_error(self):
        """Test monitor operation with error"""
        collector = get_metrics_collector()
        error_tracker = get_error_tracker()
        
        with pytest.raises(ValueError):
            with monitor_operation("failing_operation"):
                raise ValueError("Test error")
        
        # Check failure was recorded
        metrics = collector.get_metrics()
        assert any("failing_operation_failure" in key for key in collector.counters.keys())


class TestBulkhead:
    """Test bulkhead pattern"""
    
    @pytest.mark.asyncio
    async def test_bulkhead_concurrent_limit(self):
        """Test bulkhead concurrent execution limit"""
        bulkhead = BulkheadPool("test", max_concurrent=2, max_queue=5)
        
        async def slow_task(duration):
            await asyncio.sleep(duration)
            return "done"
        
        # Run tasks concurrently
        tasks = [
            bulkhead.execute(slow_task, 0.1)
            for _ in range(2)
        ]
        
        results = await asyncio.gather(*tasks)
        assert all(r == "done" for r in results)
    
    @pytest.mark.asyncio
    async def test_bulkhead_queue_overflow(self):
        """Test bulkhead queue overflow"""
        bulkhead = BulkheadPool("overflow_test", max_concurrent=1, max_queue=1)
        
        async def slow_task():
            await asyncio.sleep(1.0)
            return "done"
        
        # Start one task to occupy the bulkhead
        task1 = asyncio.create_task(bulkhead.execute(slow_task))
        
        # Queue one more task
        task2 = asyncio.create_task(bulkhead.execute(slow_task))
        
        # This should fail as queue is full
        with pytest.raises(ExternalServiceException) as exc_info:
            await bulkhead.execute(slow_task)
        
        assert "queue full" in str(exc_info.value)
        
        # Cancel pending tasks
        task1.cancel()
        task2.cancel()


class TestResilientDecorator:
    """Test combined resilient decorator"""
    
    @pytest.mark.asyncio
    async def test_resilient_decorator_all_features(self):
        """Test resilient decorator with all features"""
        call_count = 0
        
        @resilient(
            retry_config=RetryConfig(max_attempts=2, initial_delay=0.01),
            fallback_value="fallback",
            timeout_seconds=1.0
        )
        async def complex_function():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ValueError("First attempt fails")
            await asyncio.sleep(0.01)
            return "success"
        
        result = await complex_function()
        assert result == "success"
        assert call_count == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])