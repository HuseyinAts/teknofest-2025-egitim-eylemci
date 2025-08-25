"""
Production-Ready Exception Hierarchy
TEKNOFEST 2025 - Eğitim Teknolojileri

Comprehensive exception system with:
- Detailed error categorization
- Rich context preservation
- Automatic serialization support
- Internationalization support
- Error chaining and correlation
"""

from typing import Optional, Dict, Any, List, Union
from enum import Enum
import traceback
import uuid
import sys
import inspect
from datetime import datetime
from dataclasses import dataclass, field


class ErrorCode(str, Enum):
    """Standardized error codes for the application"""
    
    # General errors (1000-1999)
    UNKNOWN_ERROR = "ERR_1000"
    VALIDATION_ERROR = "ERR_1001"
    SERIALIZATION_ERROR = "ERR_1002"
    CONFIGURATION_ERROR = "ERR_1003"
    
    # Authentication & Authorization (2000-2999)
    AUTHENTICATION_REQUIRED = "ERR_2000"
    AUTHENTICATION_FAILED = "ERR_2001"
    AUTHORIZATION_FAILED = "ERR_2002"
    TOKEN_EXPIRED = "ERR_2003"
    TOKEN_INVALID = "ERR_2004"
    
    # Resource errors (3000-3999)
    RESOURCE_NOT_FOUND = "ERR_3000"
    RESOURCE_ALREADY_EXISTS = "ERR_3001"
    RESOURCE_LOCKED = "ERR_3002"
    RESOURCE_DELETED = "ERR_3003"
    
    # Business logic errors (4000-4999)
    BUSINESS_RULE_VIOLATION = "ERR_4000"
    INVALID_STATE_TRANSITION = "ERR_4001"
    QUOTA_EXCEEDED = "ERR_4002"
    RATE_LIMIT_EXCEEDED = "ERR_4003"
    
    # External service errors (5000-5999)
    EXTERNAL_SERVICE_ERROR = "ERR_5000"
    DATABASE_ERROR = "ERR_5001"
    CACHE_ERROR = "ERR_5002"
    MESSAGE_QUEUE_ERROR = "ERR_5003"
    API_CALL_FAILED = "ERR_5004"
    
    # AI/ML specific errors (6000-6999)
    MODEL_NOT_FOUND = "ERR_6000"
    MODEL_LOADING_FAILED = "ERR_6001"
    MODEL_INFERENCE_FAILED = "ERR_6002"
    MODEL_VALIDATION_FAILED = "ERR_6003"
    INSUFFICIENT_GPU_MEMORY = "ERR_6004"
    MODEL_TIMEOUT = "ERR_6005"
    MODEL_VERSION_MISMATCH = "ERR_6006"
    
    # Network and connectivity errors (8000-8999)
    NETWORK_ERROR = "ERR_8000"
    CONNECTION_TIMEOUT = "ERR_8001"
    DNS_RESOLUTION_FAILED = "ERR_8002"
    SSL_VERIFICATION_FAILED = "ERR_8003"
    
    # File and IO errors (9000-9999)
    FILE_NOT_FOUND = "ERR_9000"
    FILE_ACCESS_DENIED = "ERR_9001"
    DISK_SPACE_INSUFFICIENT = "ERR_9002"
    IO_ERROR = "ERR_9003"
    
    # Education domain errors (7000-7999)
    INVALID_CURRICULUM = "ERR_7000"
    INVALID_LEARNING_PATH = "ERR_7001"
    ASSESSMENT_FAILED = "ERR_7002"
    STUDENT_NOT_FOUND = "ERR_7003"
    COURSE_NOT_AVAILABLE = "ERR_7004"


@dataclass
class ErrorContext:
    """Context information for error tracking"""
    file_name: Optional[str] = None
    function_name: Optional[str] = None
    line_number: Optional[int] = None
    variables: Dict[str, Any] = field(default_factory=dict)
    stack_trace: Optional[List[str]] = None
    
    @classmethod
    def from_current_frame(cls, depth: int = 2) -> 'ErrorContext':
        """Create error context from current execution frame"""
        frame = inspect.currentframe()
        for _ in range(depth):
            if frame:
                frame = frame.f_back
        
        if frame:
            return cls(
                file_name=frame.f_code.co_filename,
                function_name=frame.f_code.co_name,
                line_number=frame.f_lineno,
                variables={k: str(v)[:100] for k, v in frame.f_locals.items() 
                          if not k.startswith('_')},
                stack_trace=traceback.format_stack(frame)
            )
        return cls()


class BaseApplicationException(Exception):
    """Enhanced base exception class for all application exceptions"""
    
    def __init__(
        self,
        message: str,
        error_code: ErrorCode = ErrorCode.UNKNOWN_ERROR,
        status_code: int = 500,
        details: Optional[Dict[str, Any]] = None,
        internal_message: Optional[str] = None,
        cause: Optional[Exception] = None,
        user_message: Optional[str] = None,
        suggestions: Optional[List[str]] = None,
        context: Optional[ErrorContext] = None
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.status_code = status_code
        self.details = details or {}
        self.internal_message = internal_message
        self.cause = cause
        self.user_message = user_message or message
        self.suggestions = suggestions or []
        self.error_id = str(uuid.uuid4())
        self.timestamp = datetime.utcnow().isoformat()
        self.traceback = traceback.format_exc() if cause else None
        self.context = context or ErrorContext.from_current_frame()
        
        # Chain exceptions
        if cause:
            self.__cause__ = cause
    
    def to_dict(self, include_internal: bool = False, locale: str = "en") -> Dict[str, Any]:
        """Convert exception to dictionary for API responses"""
        result = {
            "error": {
                "code": self.error_code.value,
                "message": self._get_localized_message(locale),
                "error_id": self.error_id,
                "timestamp": self.timestamp,
                "details": self.details
            }
        }
        
        # Add user-friendly information
        if self.user_message and self.user_message != self.message:
            result["error"]["user_message"] = self.user_message
        
        if self.suggestions:
            result["error"]["suggestions"] = self.suggestions
        
        # Add internal debugging information
        if include_internal:
            debug_info = {}
            
            if self.internal_message:
                debug_info["internal_message"] = self.internal_message
            
            if self.traceback:
                debug_info["traceback"] = self.traceback.split('\n')
            
            if self.context:
                debug_info["context"] = {
                    "file": self.context.file_name,
                    "function": self.context.function_name,
                    "line": self.context.line_number,
                    "variables": self.context.variables
                }
            
            if self.cause:
                debug_info["cause"] = {
                    "type": type(self.cause).__name__,
                    "message": str(self.cause)
                }
            
            result["error"]["debug"] = debug_info
        
        return result
    
    def _get_localized_message(self, locale: str) -> str:
        """Get localized error message"""
        # For now, return the default message
        # In production, integrate with i18n system
        return self.user_message if locale != "en" else self.message
    
    def __str__(self):
        base = f"[{self.error_code}] {self.message} (ID: {self.error_id})"
        if self.context and self.context.file_name:
            base += f" at {self.context.file_name}:{self.context.line_number}"
        return base
    
    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"error_code={self.error_code}, "
            f"message='{self.message}', "
            f"error_id='{self.error_id}')"
        )
    
    def add_detail(self, key: str, value: Any) -> 'BaseApplicationException':
        """Add additional detail to the exception"""
        self.details[key] = value
        return self
    
    def add_suggestion(self, suggestion: str) -> 'BaseApplicationException':
        """Add a suggestion for resolving the error"""
        self.suggestions.append(suggestion)
        return self
    
    def with_context(self, **kwargs) -> 'BaseApplicationException':
        """Add context variables to the exception"""
        if self.context:
            self.context.variables.update(kwargs)
        return self


# ==========================================
# Business Logic Exceptions
# ==========================================

class ValidationException(BaseApplicationException):
    """Enhanced validation exception with detailed field information"""
    
    def __init__(
        self,
        message: str,
        field: Optional[str] = None,
        value: Any = None,
        validation_errors: Optional[List[Dict[str, Any]]] = None,
        constraints: Optional[Dict[str, Any]] = None
    ):
        details = {}
        suggestions = []
        
        if field:
            details["field"] = field
            details["value"] = str(value)[:100] if value else None
            
        if validation_errors:
            details["validation_errors"] = validation_errors
            # Generate suggestions from validation errors
            for error in validation_errors:
                if error.get("type") == "missing":
                    suggestions.append(f"Provide a value for required field '{error.get('field')}'")
                elif error.get("type") == "type_error":
                    suggestions.append(f"Ensure '{error.get('field')}' is of the correct type")
        
        if constraints:
            details["constraints"] = constraints
            
        super().__init__(
            message=message,
            error_code=ErrorCode.VALIDATION_ERROR,
            status_code=400,
            details=details,
            suggestions=suggestions,
            user_message="The provided data is invalid. Please check and try again."
        )


class ResourceNotFoundException(BaseApplicationException):
    """Raised when a requested resource is not found"""
    
    def __init__(
        self,
        resource_type: str,
        resource_id: Optional[str] = None,
        message: Optional[str] = None
    ):
        msg = message or f"{resource_type} not found"
        details = {"resource_type": resource_type}
        if resource_id:
            details["resource_id"] = resource_id
            msg = f"{resource_type} with ID '{resource_id}' not found"
        
        super().__init__(
            message=msg,
            error_code=ErrorCode.RESOURCE_NOT_FOUND,
            status_code=404,
            details=details
        )


class DuplicateResourceException(BaseApplicationException):
    """Raised when attempting to create a duplicate resource"""
    
    def __init__(
        self,
        resource_type: str,
        identifier: str,
        message: Optional[str] = None
    ):
        msg = message or f"{resource_type} with identifier '{identifier}' already exists"
        
        super().__init__(
            message=msg,
            error_code=ErrorCode.RESOURCE_ALREADY_EXISTS,
            status_code=409,
            details={"resource_type": resource_type, "identifier": identifier}
        )


class BusinessRuleViolationException(BaseApplicationException):
    """Raised when a business rule is violated"""
    
    def __init__(
        self,
        rule: str,
        message: str,
        context: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            message=message,
            error_code=ErrorCode.BUSINESS_RULE_VIOLATION,
            status_code=422,
            details={"rule": rule, "context": context or {}}
        )


# ==========================================
# Authentication & Authorization Exceptions
# ==========================================

class AuthenticationException(BaseApplicationException):
    """Base class for authentication exceptions"""
    
    def __init__(
        self,
        message: str = "Authentication required",
        error_code: ErrorCode = ErrorCode.AUTHENTICATION_REQUIRED,
        **kwargs
    ):
        super().__init__(
            message=message,
            error_code=error_code,
            status_code=401,
            **kwargs
        )


class AuthorizationException(BaseApplicationException):
    """Raised when user lacks necessary permissions"""
    
    def __init__(
        self,
        message: str = "Insufficient permissions",
        required_permission: Optional[str] = None,
        user_permissions: Optional[List[str]] = None
    ):
        details = {}
        if required_permission:
            details["required_permission"] = required_permission
        if user_permissions:
            details["user_permissions"] = user_permissions
        
        super().__init__(
            message=message,
            error_code=ErrorCode.AUTHORIZATION_FAILED,
            status_code=403,
            details=details
        )


class TokenException(AuthenticationException):
    """Raised for token-related issues"""
    
    def __init__(
        self,
        message: str,
        token_type: str = "access",
        expired: bool = False
    ):
        error_code = ErrorCode.TOKEN_EXPIRED if expired else ErrorCode.TOKEN_INVALID
        super().__init__(
            message=message,
            error_code=error_code,
            details={"token_type": token_type, "expired": expired}
        )


# ==========================================
# External Service Exceptions
# ==========================================

class ExternalServiceException(BaseApplicationException):
    """Enhanced external service exception with retry information"""
    
    def __init__(
        self,
        service_name: str,
        message: str,
        status_code: Optional[int] = None,
        response_body: Optional[str] = None,
        cause: Optional[Exception] = None,
        retry_after: Optional[int] = None,
        request_id: Optional[str] = None
    ):
        details = {
            "service": service_name,
            "retryable": status_code in [502, 503, 504] if status_code else True
        }
        
        if status_code:
            details["external_status_code"] = status_code
        if response_body:
            details["response_body"] = response_body[:500]  # Limit response size
        if retry_after:
            details["retry_after"] = retry_after
        if request_id:
            details["request_id"] = request_id
        
        suggestions = [
            "The service may be temporarily unavailable. Please try again later.",
            f"If the problem persists, contact support with reference ID: {request_id or 'N/A'}"
        ]
        
        super().__init__(
            message=message,
            error_code=ErrorCode.EXTERNAL_SERVICE_ERROR,
            status_code=502,
            details=details,
            cause=cause,
            suggestions=suggestions,
            user_message=f"Unable to connect to {service_name}. Please try again."
        )


class DatabaseException(BaseApplicationException):
    """Raised for database-related errors"""
    
    def __init__(
        self,
        message: str,
        operation: Optional[str] = None,
        table: Optional[str] = None,
        cause: Optional[Exception] = None
    ):
        details = {}
        if operation:
            details["operation"] = operation
        if table:
            details["table"] = table
        
        super().__init__(
            message=message,
            error_code=ErrorCode.DATABASE_ERROR,
            status_code=503,
            details=details,
            cause=cause
        )


class RateLimitException(BaseApplicationException):
    """Raised when rate limit is exceeded"""
    
    def __init__(
        self,
        limit: int,
        window: int,
        retry_after: Optional[int] = None
    ):
        message = f"Rate limit exceeded: {limit} requests per {window} seconds"
        details = {
            "limit": limit,
            "window": window,
            "retry_after": retry_after
        }
        
        super().__init__(
            message=message,
            error_code=ErrorCode.RATE_LIMIT_EXCEEDED,
            status_code=429,
            details=details
        )


# ==========================================
# AI/ML Specific Exceptions
# ==========================================

class ModelException(BaseApplicationException):
    """Base class for AI/ML model exceptions"""
    
    def __init__(
        self,
        message: str,
        model_name: Optional[str] = None,
        error_code: ErrorCode = ErrorCode.MODEL_INFERENCE_FAILED,
        **kwargs
    ):
        details = kwargs.get("details", {})
        if model_name:
            details["model_name"] = model_name
        kwargs["details"] = details
        
        super().__init__(
            message=message,
            error_code=error_code,
            status_code=503,
            **kwargs
        )


class ModelNotFoundException(ModelException):
    """Raised when a model is not found"""
    
    def __init__(self, model_name: str):
        super().__init__(
            message=f"Model '{model_name}' not found",
            model_name=model_name,
            error_code=ErrorCode.MODEL_NOT_FOUND
        )


class ModelLoadingException(ModelException):
    """Raised when model loading fails"""
    
    def __init__(
        self,
        model_name: str,
        reason: str,
        cause: Optional[Exception] = None
    ):
        super().__init__(
            message=f"Failed to load model '{model_name}': {reason}",
            model_name=model_name,
            error_code=ErrorCode.MODEL_LOADING_FAILED,
            cause=cause
        )


class ModelInferenceException(ModelException):
    """Raised when model inference fails"""
    
    def __init__(
        self,
        model_name: str,
        input_data: Optional[Any] = None,
        cause: Optional[Exception] = None
    ):
        details = {"model_name": model_name}
        if input_data:
            details["input_shape"] = str(getattr(input_data, "shape", "unknown"))
        
        super().__init__(
            message=f"Model inference failed for '{model_name}'",
            model_name=model_name,
            error_code=ErrorCode.MODEL_INFERENCE_FAILED,
            details=details,
            cause=cause
        )


# ==========================================
# Education Domain Exceptions
# ==========================================

class EducationException(BaseApplicationException):
    """Base class for education domain exceptions"""
    
    def __init__(
        self,
        message: str,
        error_code: ErrorCode,
        grade: Optional[str] = None,
        subject: Optional[str] = None,
        **kwargs
    ):
        details = kwargs.get("details", {})
        if grade:
            details["grade"] = grade
        if subject:
            details["subject"] = subject
        kwargs["details"] = details
        
        super().__init__(
            message=message,
            error_code=error_code,
            status_code=400,
            **kwargs
        )


class InvalidCurriculumException(EducationException):
    """Raised when curriculum data is invalid"""
    
    def __init__(
        self,
        message: str,
        grade: Optional[str] = None,
        subject: Optional[str] = None
    ):
        super().__init__(
            message=message,
            error_code=ErrorCode.INVALID_CURRICULUM,
            grade=grade,
            subject=subject
        )


class StudentNotFoundException(EducationException):
    """Raised when a student is not found"""
    
    def __init__(self, student_id: str):
        super().__init__(
            message=f"Student with ID '{student_id}' not found",
            error_code=ErrorCode.STUDENT_NOT_FOUND,
            details={"student_id": student_id}
        )


class AssessmentException(EducationException):
    """Raised when assessment processing fails"""
    
    def __init__(
        self,
        message: str,
        assessment_type: Optional[str] = None,
        reason: Optional[str] = None
    ):
        details = {}
        if assessment_type:
            details["assessment_type"] = assessment_type
        if reason:
            details["reason"] = reason
        
        super().__init__(
            message=message,
            error_code=ErrorCode.ASSESSMENT_FAILED,
            details=details
        )


# ==========================================
# Utility Functions
# ==========================================

def create_error_from_exception(
    exc: Exception,
    error_code: ErrorCode = ErrorCode.UNKNOWN_ERROR,
    status_code: int = 500
) -> BaseApplicationException:
    """Create an application exception from a standard exception"""
    
    # Map common exceptions to specific error types
    if isinstance(exc, FileNotFoundError):
        return ResourceNotFoundException(
            resource_type="File",
            message=str(exc)
        )
    elif isinstance(exc, PermissionError):
        return AuthorizationException(
            message=f"Permission denied: {str(exc)}"
        )
    elif isinstance(exc, ConnectionError):
        return ExternalServiceException(
            service_name="Unknown",
            message=str(exc)
        )
    elif isinstance(exc, TimeoutError):
        return ExternalServiceException(
            service_name="Unknown",
            message="Operation timed out",
            cause=exc
        )
    elif isinstance(exc, ValueError):
        return ValidationException(
            message=str(exc)
        )
    else:
        return BaseApplicationException(
            message=str(exc),
            error_code=error_code,
            status_code=status_code,
            cause=exc
        )


def aggregate_exceptions(
    exceptions: List[Exception],
    message: str = "Multiple errors occurred"
) -> BaseApplicationException:
    """Aggregate multiple exceptions into a single exception"""
    
    error_details = []
    for i, exc in enumerate(exceptions):
        if isinstance(exc, BaseApplicationException):
            error_details.append({
                "index": i,
                "error_code": exc.error_code.value,
                "message": exc.message,
                "details": exc.details
            })
        else:
            error_details.append({
                "index": i,
                "type": type(exc).__name__,
                "message": str(exc)
            })
    
    return BaseApplicationException(
        message=message,
        error_code=ErrorCode.UNKNOWN_ERROR,
        status_code=500,
        details={"errors": error_details}
    )