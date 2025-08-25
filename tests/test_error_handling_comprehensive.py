"""
Comprehensive Error Handling Tests
TEKNOFEST 2025 - Eğitim Teknolojileri
"""

import pytest
import asyncio
import json
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timedelta
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from src.exceptions import (
    BaseApplicationException,
    ValidationException,
    ExternalServiceException,
    DatabaseException,
    RateLimitException,
    AuthenticationException,
    AuthorizationException,
    ModelException,
    ErrorCode,
    ErrorContext,
    create_error_from_exception,
    aggregate_exceptions
)
from src.error_handlers import (
    ErrorMetrics,
    RetryStrategy,
    ErrorResponse,
    ErrorRecoveryManager,
    ErrorHandlingMiddleware,
    application_exception_handler,
    validation_exception_handler,
    register_error_handlers,
    with_error_handling,
    get_error_statistics
)


class TestErrorMetrics:
    """Test error metrics tracking"""
    
    def test_error_recording(self):
        """Test recording error metrics"""
        metrics = ErrorMetrics()
        
        # Record some errors
        metrics.record_error(
            error_code="ERR_1000",
            endpoint="/api/test",
            error_id="test-123",
            details={"test": "data"}
        )
        
        assert metrics.errors_by_code["ERR_1000"] == 1
        assert metrics.errors_by_endpoint["/api/test"] == 1
        assert len(metrics.recent_errors) == 1
        assert metrics.recent_errors[0]["error_id"] == "test-123"
    
    def test_error_rate_calculation(self):
        """Test error rate calculation"""
        metrics = ErrorMetrics(time_window_minutes=1)
        
        # Record multiple errors
        for i in range(5):
            metrics.record_error(
                error_code="ERR_1000",
                endpoint="/api/test",
                error_id=f"test-{i}",
                details={}
            )
        
        # Error rate should be 5 per minute
        error_rate = metrics.get_error_rate()
        assert error_rate == 5.0
    
    def test_circuit_breaker(self):
        """Test circuit breaker functionality"""
        metrics = ErrorMetrics()
        
        # Record failures
        for _ in range(5):
            metrics.record_service_failure("test-service")
        
        # Circuit should be open
        assert metrics.should_circuit_break("test-service", threshold=5)
        assert metrics.circuit_breakers["test-service"]["state"] == "open"
        
        # Record success
        metrics.record_service_success("test-service")
        
        # Circuit should be closed
        assert not metrics.should_circuit_break("test-service")
        assert metrics.circuit_breakers["test-service"]["state"] == "closed"
    
    def test_top_errors(self):
        """Test getting top error codes"""
        metrics = ErrorMetrics()
        
        # Record different error types
        for _ in range(10):
            metrics.record_error("ERR_1000", "/api/test", "id1", {})
        for _ in range(5):
            metrics.record_error("ERR_2000", "/api/test", "id2", {})
        for _ in range(3):
            metrics.record_error("ERR_3000", "/api/test", "id3", {})
        
        top_errors = metrics.get_top_errors(2)
        assert len(top_errors) == 2
        assert top_errors[0] == ("ERR_1000", 10)
        assert top_errors[1] == ("ERR_2000", 5)


class TestRetryStrategy:
    """Test retry strategy implementation"""
    
    @pytest.mark.asyncio
    async def test_exponential_backoff(self):
        """Test exponential backoff calculation"""
        strategy = RetryStrategy(
            max_retries=3,
            base_delay=1.0,
            exponential_base=2.0,
            jitter=False
        )
        
        # Test delay calculation
        assert strategy.get_delay(0) == 1.0
        assert strategy.get_delay(1) == 2.0
        assert strategy.get_delay(2) == 4.0
    
    @pytest.mark.asyncio
    async def test_retry_on_failure(self):
        """Test retry on failure"""
        strategy = RetryStrategy(max_retries=2, base_delay=0.01)
        
        call_count = 0
        
        async def failing_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("Test error")
            return "success"
        
        result = await strategy.execute_with_retry(
            failing_func,
            retry_on=(ValueError,)
        )
        
        assert result == "success"
        assert call_count == 3
    
    @pytest.mark.asyncio
    async def test_retry_exhaustion(self):
        """Test retry exhaustion"""
        strategy = RetryStrategy(max_retries=2, base_delay=0.01)
        
        async def always_failing():
            raise ValueError("Always fails")
        
        with pytest.raises(ValueError, match="Always fails"):
            await strategy.execute_with_retry(
                always_failing,
                retry_on=(ValueError,)
            )


class TestErrorContext:
    """Test error context tracking"""
    
    def test_context_creation(self):
        """Test creating error context from frame"""
        def test_function():
            local_var = "test_value"
            context = ErrorContext.from_current_frame(depth=1)
            return context
        
        context = test_function()
        assert context.function_name == "test_function"
        assert "local_var" in context.variables
        assert context.line_number is not None
    
    def test_context_in_exception(self):
        """Test error context in exception"""
        exc = BaseApplicationException(
            message="Test error",
            error_code=ErrorCode.UNKNOWN_ERROR
        )
        
        assert exc.context is not None
        assert exc.context.function_name is not None


class TestExceptionClasses:
    """Test custom exception classes"""
    
    def test_base_exception(self):
        """Test base application exception"""
        exc = BaseApplicationException(
            message="Test error",
            error_code=ErrorCode.UNKNOWN_ERROR,
            status_code=500,
            details={"key": "value"},
            suggestions=["Try again later"]
        )
        
        assert exc.message == "Test error"
        assert exc.error_code == ErrorCode.UNKNOWN_ERROR
        assert exc.status_code == 500
        assert exc.error_id is not None
        assert exc.suggestions == ["Try again later"]
        
        # Test serialization
        data = exc.to_dict()
        assert data["error"]["code"] == "ERR_1000"
        assert data["error"]["message"] == "Test error"
        assert data["error"]["suggestions"] == ["Try again later"]
    
    def test_validation_exception(self):
        """Test validation exception with suggestions"""
        exc = ValidationException(
            message="Validation failed",
            field="email",
            value="invalid",
            validation_errors=[
                {"field": "email", "type": "missing", "message": "Required"}
            ]
        )
        
        assert exc.status_code == 400
        assert exc.details["field"] == "email"
        assert len(exc.suggestions) > 0
        assert "email" in exc.suggestions[0]
    
    def test_external_service_exception(self):
        """Test external service exception"""
        exc = ExternalServiceException(
            service_name="TestAPI",
            message="Connection failed",
            status_code=503,
            retry_after=30,
            request_id="req-123"
        )
        
        assert exc.status_code == 502
        assert exc.details["service"] == "TestAPI"
        assert exc.details["retryable"] is True
        assert exc.details["retry_after"] == 30
        assert len(exc.suggestions) > 0
    
    def test_exception_chaining(self):
        """Test exception chaining"""
        original = ValueError("Original error")
        exc = BaseApplicationException(
            message="Wrapped error",
            cause=original
        )
        
        assert exc.__cause__ == original
        assert exc.traceback is not None
    
    def test_exception_methods(self):
        """Test exception utility methods"""
        exc = BaseApplicationException(message="Test")
        
        # Test adding details
        exc.add_detail("key", "value")
        assert exc.details["key"] == "value"
        
        # Test adding suggestions
        exc.add_suggestion("Try this")
        assert "Try this" in exc.suggestions
        
        # Test adding context
        exc.with_context(user_id="123", action="test")
        assert exc.context.variables["user_id"] == "123"


class TestErrorRecovery:
    """Test error recovery mechanisms"""
    
    @pytest.mark.asyncio
    async def test_recovery_registration(self):
        """Test registering recovery strategies"""
        recovery = ErrorRecoveryManager()
        
        def recovery_func(error, request):
            return {"recovered": True}
        
        recovery.register_recovery(ValueError, recovery_func)
        
        # Test recovery
        request = Mock(spec=Request)
        result = await recovery.attempt_recovery(ValueError("test"), request)
        
        assert result == {"recovered": True}
    
    @pytest.mark.asyncio
    async def test_fallback_responses(self):
        """Test fallback response mechanism"""
        recovery = ErrorRecoveryManager()
        
        recovery.register_fallback("/api/test", {"fallback": "data"})
        
        request = Mock(spec=Request)
        request.url.path = "/api/test"
        
        result = await recovery.attempt_recovery(Exception("test"), request)
        assert result == {"fallback": "data"}
    
    @pytest.mark.asyncio
    async def test_recovery_failure(self):
        """Test recovery failure handling"""
        recovery = ErrorRecoveryManager()
        
        def failing_recovery(error, request):
            raise Exception("Recovery failed")
        
        recovery.register_recovery(ValueError, failing_recovery)
        
        request = Mock(spec=Request)
        result = await recovery.attempt_recovery(ValueError("test"), request)
        
        assert result is None


class TestErrorHandlers:
    """Test error handler functions"""
    
    @pytest.mark.asyncio
    async def test_application_exception_handler(self):
        """Test application exception handler"""
        request = Mock(spec=Request)
        request.url.path = "/api/test"
        
        exc = BaseApplicationException(
            message="Test error",
            error_code=ErrorCode.UNKNOWN_ERROR,
            status_code=500
        )
        
        with patch('src.error_handlers.get_settings') as mock_settings:
            mock_settings.return_value.app_debug = False
            mock_settings.return_value.is_production.return_value = True
            
            response = await application_exception_handler(request, exc)
            
            assert response.status_code == 500
            content = json.loads(response.body)
            assert content["error"]["code"] == "ERR_1000"
            assert content["error"]["message"] == "Test error"
    
    @pytest.mark.asyncio
    async def test_validation_exception_handler(self):
        """Test validation exception handler"""
        from fastapi.exceptions import RequestValidationError
        from pydantic import ValidationError
        
        request = Mock(spec=Request)
        request.url.path = "/api/test"
        request.method = "POST"
        
        # Create validation error
        errors = [
            {
                "loc": ("body", "email"),
                "msg": "field required",
                "type": "value_error.missing"
            }
        ]
        
        exc = RequestValidationError(errors)
        response = await validation_exception_handler(request, exc)
        
        assert response.status_code == 422
        content = json.loads(response.body)
        assert content["error"]["code"] == "ERR_1001"
        assert "validation_errors" in content["error"]["details"]


class TestErrorHandlingDecorators:
    """Test error handling decorators"""
    
    @pytest.mark.asyncio
    async def test_with_error_handling_decorator(self):
        """Test error handling decorator"""
        
        @with_error_handling(
            operation="test_operation",
            error_code=ErrorCode.UNKNOWN_ERROR
        )
        async def test_func():
            raise ValueError("Test error")
        
        with pytest.raises(BaseApplicationException) as exc_info:
            await test_func()
        
        assert exc_info.value.error_code == ErrorCode.UNKNOWN_ERROR
        assert "test_operation" in exc_info.value.message
    
    @pytest.mark.asyncio
    async def test_with_retry_decorator(self):
        """Test error handling with retry"""
        call_count = 0
        
        @with_error_handling(
            operation="test_operation",
            retry_strategy=RetryStrategy(max_retries=2, base_delay=0.01)
        )
        async def test_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("Test error")
            return "success"
        
        result = await test_func()
        assert result == "success"
        assert call_count == 3


class TestErrorStatistics:
    """Test error statistics endpoint"""
    
    @pytest.mark.asyncio
    async def test_get_error_statistics(self):
        """Test getting error statistics"""
        from src.error_handlers import error_metrics
        
        # Record some errors
        error_metrics.record_error(
            error_code="ERR_1000",
            endpoint="/api/test",
            error_id="test-123",
            details={}
        )
        
        stats = await get_error_statistics()
        
        assert "error_rate" in stats
        assert "top_errors" in stats
        assert "recent_errors" in stats
        assert "circuit_breakers" in stats
        assert "timestamp" in stats


class TestUtilityFunctions:
    """Test utility functions"""
    
    def test_create_error_from_exception(self):
        """Test creating application exception from standard exception"""
        
        # Test FileNotFoundError
        exc = create_error_from_exception(FileNotFoundError("test.txt"))
        assert isinstance(exc, BaseApplicationException)
        assert exc.error_code == ErrorCode.RESOURCE_NOT_FOUND
        
        # Test PermissionError
        exc = create_error_from_exception(PermissionError("Access denied"))
        assert isinstance(exc, AuthorizationException)
        
        # Test generic exception
        exc = create_error_from_exception(Exception("Generic error"))
        assert exc.error_code == ErrorCode.UNKNOWN_ERROR
    
    def test_aggregate_exceptions(self):
        """Test aggregating multiple exceptions"""
        exc1 = ValidationException("Error 1")
        exc2 = DatabaseException("Error 2")
        exc3 = ValueError("Error 3")
        
        aggregated = aggregate_exceptions(
            [exc1, exc2, exc3],
            message="Multiple failures"
        )
        
        assert aggregated.message == "Multiple failures"
        assert len(aggregated.details["errors"]) == 3
        assert aggregated.details["errors"][0]["error_code"] == "ERR_1001"


class TestIntegration:
    """Integration tests for error handling"""
    
    def test_fastapi_integration(self):
        """Test FastAPI integration"""
        app = FastAPI()
        register_error_handlers(app)
        
        @app.get("/test-error")
        async def test_endpoint():
            raise BaseApplicationException(
                message="Test error",
                error_code=ErrorCode.UNKNOWN_ERROR
            )
        
        @app.get("/test-validation")
        async def test_validation():
            raise ValidationException(
                message="Validation failed",
                field="test_field"
            )
        
        client = TestClient(app)
        
        # Test application exception
        response = client.get("/test-error")
        assert response.status_code == 500
        data = response.json()
        assert data["error"]["code"] == "ERR_1000"
        
        # Test validation exception
        response = client.get("/test-validation")
        assert response.status_code == 400
        data = response.json()
        assert data["error"]["code"] == "ERR_1001"
    
    @pytest.mark.asyncio
    async def test_middleware_integration(self):
        """Test middleware integration"""
        app = FastAPI()
        middleware = ErrorHandlingMiddleware(app)
        
        request = Mock(spec=Request)
        request.url.path = "/api/test"
        request.method = "GET"
        request.client = Mock(host="127.0.0.1")
        request.headers = {"user-agent": "test"}
        request.state = Mock()
        
        async def call_next(req):
            response = Mock()
            response.status_code = 200
            response.headers = {}
            return response
        
        response = await middleware(request, call_next)
        
        assert "X-Request-Id" in response.headers
        assert "X-Process-Time" in response.headers
        assert "X-Error-Rate" in response.headers


if __name__ == "__main__":
    pytest.main([__file__, "-v"])