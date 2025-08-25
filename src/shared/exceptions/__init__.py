"""
Custom Exception Classes for Clean Error Handling
TEKNOFEST 2025 - Exception Hierarchy
"""

from typing import Optional, Dict, Any


class ApplicationError(Exception):
    """Base application exception with structured error handling"""
    
    def __init__(
        self,
        message: str,
        code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        status_code: int = 500
    ):
        self.message = message
        self.code = code or self.__class__.__name__
        self.details = details or {}
        self.status_code = status_code
        super().__init__(self.message)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for API responses"""
        return {
            "error": {
                "code": self.code,
                "message": self.message,
                "details": self.details,
                "type": self.__class__.__name__
            }
        }


# Domain Layer Exceptions
class DomainError(ApplicationError):
    """Base exception for domain layer errors"""
    def __init__(self, message: str, code: Optional[str] = None, details: Optional[Dict] = None):
        super().__init__(message, code, details, status_code=422)


class StudentNotFoundException(DomainError):
    """Raised when a student is not found"""
    def __init__(self, student_id: str):
        super().__init__(
            message=f"Student with ID '{student_id}' not found",
            code="STUDENT_NOT_FOUND",
            details={"student_id": student_id}
        )


class CurriculumNotFoundException(DomainError):
    """Raised when curriculum is not found for a grade"""
    def __init__(self, grade: str):
        super().__init__(
            message=f"Curriculum not found for grade {grade}",
            code="CURRICULUM_NOT_FOUND",
            details={"grade": grade}
        )


class InvalidLearningPeriodError(DomainError):
    """Raised when learning period is invalid"""
    def __init__(self, weeks: int, max_weeks: int):
        super().__init__(
            message=f"Learning period {weeks} weeks exceeds maximum of {max_weeks} weeks",
            code="INVALID_LEARNING_PERIOD",
            details={"requested_weeks": weeks, "max_weeks": max_weeks}
        )


class InsufficientDataError(DomainError):
    """Raised when there's not enough data for analysis"""
    def __init__(self, required: int, provided: int):
        super().__init__(
            message=f"Insufficient data: {provided} provided, {required} required",
            code="INSUFFICIENT_DATA",
            details={"required": required, "provided": provided}
        )


# Validation Layer Exceptions
class ValidationError(ApplicationError):
    """Base exception for validation errors"""
    def __init__(self, message: str, field: Optional[str] = None, details: Optional[Dict] = None):
        super().__init__(
            message=message,
            code="VALIDATION_ERROR",
            details={"field": field, **(details or {})},
            status_code=400
        )


class InvalidInputError(ValidationError):
    """Raised when input validation fails"""
    def __init__(self, field: str, value: Any, constraint: str):
        super().__init__(
            message=f"Invalid value for field '{field}': {constraint}",
            field=field,
            details={"value": str(value), "constraint": constraint}
        )


class MissingRequiredFieldError(ValidationError):
    """Raised when a required field is missing"""
    def __init__(self, field: str):
        super().__init__(
            message=f"Required field '{field}' is missing",
            field=field
        )


# Repository Layer Exceptions
class RepositoryError(ApplicationError):
    """Base exception for repository layer errors"""
    def __init__(self, message: str, details: Optional[Dict] = None):
        super().__init__(
            message=message,
            code="REPOSITORY_ERROR",
            details=details,
            status_code=500
        )


class EntityNotFoundError(RepositoryError):
    """Raised when an entity is not found in repository"""
    def __init__(self, entity_type: str, entity_id: str):
        super().__init__(
            message=f"{entity_type} with ID '{entity_id}' not found",
            details={"entity_type": entity_type, "entity_id": entity_id}
        )


class DuplicateEntityError(RepositoryError):
    """Raised when trying to create a duplicate entity"""
    def __init__(self, entity_type: str, field: str, value: str):
        super().__init__(
            message=f"{entity_type} with {field} '{value}' already exists",
            details={"entity_type": entity_type, "field": field, "value": value}
        )


# Service Layer Exceptions
class ServiceError(ApplicationError):
    """Base exception for service layer errors"""
    def __init__(self, message: str, service: str, details: Optional[Dict] = None):
        super().__init__(
            message=message,
            code="SERVICE_ERROR",
            details={"service": service, **(details or {})},
            status_code=500
        )


class ExternalServiceError(ServiceError):
    """Raised when an external service fails"""
    def __init__(self, service: str, reason: str):
        super().__init__(
            message=f"External service '{service}' failed: {reason}",
            service=service,
            details={"reason": reason}
        )


# Authentication Exceptions
class AuthenticationError(ApplicationError):
    """Base exception for authentication errors"""
    def __init__(self, message: str, details: Optional[Dict] = None):
        super().__init__(
            message=message,
            code="AUTHENTICATION_ERROR",
            details=details,
            status_code=401
        )


class InvalidCredentialsError(AuthenticationError):
    """Raised when credentials are invalid"""
    def __init__(self):
        super().__init__(message="Invalid username or password")


class TokenExpiredError(AuthenticationError):
    """Raised when JWT token has expired"""
    def __init__(self):
        super().__init__(message="Authentication token has expired")


class InsufficientPermissionsError(ApplicationError):
    """Raised when user lacks required permissions"""
    def __init__(self, required_role: str):
        super().__init__(
            message=f"Insufficient permissions. Required role: {required_role}",
            code="INSUFFICIENT_PERMISSIONS",
            details={"required_role": required_role},
            status_code=403
        )


# Configuration Exceptions
class ConfigurationError(ApplicationError):
    """Raised when configuration is invalid or missing"""
    def __init__(self, config_key: str, reason: str):
        super().__init__(
            message=f"Configuration error for '{config_key}': {reason}",
            code="CONFIGURATION_ERROR",
            details={"config_key": config_key, "reason": reason},
            status_code=500
        )
