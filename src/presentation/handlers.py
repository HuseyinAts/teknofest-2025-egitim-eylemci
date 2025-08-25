"""
Error Handlers - Presentation Layer
TEKNOFEST 2025 - Centralized Error Handling
"""

import logging
from fastapi import Request, status
from fastapi.responses import JSONResponse

from src.shared.exceptions import (
    ValidationError,
    DomainError,
    AuthenticationError,
    EntityNotFoundError,
    RepositoryError,
    ServiceError,
    ApplicationError
)

logger = logging.getLogger(__name__)


async def validation_error_handler(request: Request, exc: ValidationError) -> JSONResponse:
    """Handle validation errors"""
    logger.warning(f"Validation error: {exc.message}")
    
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content=exc.to_dict()
    )


async def domain_error_handler(request: Request, exc: DomainError) -> JSONResponse:
    """Handle domain logic errors"""
    logger.warning(f"Domain error: {exc.message}")
    
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=exc.to_dict()
    )


async def authentication_error_handler(request: Request, exc: AuthenticationError) -> JSONResponse:
    """Handle authentication errors"""
    logger.warning(f"Authentication error: {exc.message}")
    
    return JSONResponse(
        status_code=status.HTTP_401_UNAUTHORIZED,
        content=exc.to_dict(),
        headers={"WWW-Authenticate": "Bearer"}
    )


async def not_found_error_handler(request: Request, exc: EntityNotFoundError) -> JSONResponse:
    """Handle entity not found errors"""
    logger.info(f"Entity not found: {exc.message}")
    
    return JSONResponse(
        status_code=status.HTTP_404_NOT_FOUND,
        content=exc.to_dict()
    )


async def repository_error_handler(request: Request, exc: RepositoryError) -> JSONResponse:
    """Handle repository/database errors"""
    logger.error(f"Repository error: {exc.message}", exc_info=True)
    
    # Don't expose internal database errors in production
    if request.app.state.settings.is_production():
        content = {
            "error": {
                "code": "DATABASE_ERROR",
                "message": "A database error occurred"
            }
        }
    else:
        content = exc.to_dict()
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=content
    )


async def service_error_handler(request: Request, exc: ServiceError) -> JSONResponse:
    """Handle service layer errors"""
    logger.error(f"Service error: {exc.message}")
    
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content=exc.to_dict()
    )


async def general_error_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle all other exceptions"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    
    # Generate request ID for tracking
    request_id = getattr(request.state, "request_id", "unknown")
    
    # Don't expose internal errors in production
    if hasattr(request.app.state, "settings") and request.app.state.settings.is_production():
        content = {
            "error": {
                "code": "INTERNAL_ERROR",
                "message": "An unexpected error occurred",
                "request_id": request_id
            }
        }
    else:
        content = {
            "error": {
                "code": "INTERNAL_ERROR",
                "message": str(exc),
                "type": type(exc).__name__,
                "request_id": request_id
            }
        }
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=content
    )


__all__ = [
    "validation_error_handler",
    "domain_error_handler",
    "authentication_error_handler",
    "not_found_error_handler",
    "repository_error_handler",
    "service_error_handler",
    "general_error_handler"
]
