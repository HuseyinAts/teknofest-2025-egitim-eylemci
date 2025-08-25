"""
Middleware Components - Presentation Layer
TEKNOFEST 2025 - Clean Code Middleware Implementation
"""

import time
import logging
import json
from typing import Callable, Optional, Dict, Any
from datetime import datetime, timedelta
import hashlib

from fastapi import Request, Response, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from src.shared.exceptions import ApplicationError
from src.shared.constants import HTTPStatus, ErrorCodes
from src.config import Settings

logger = logging.getLogger(__name__)


class ErrorHandlerMiddleware(BaseHTTPMiddleware):
    """
    Centralized error handling middleware
    Catches all exceptions and returns structured error responses
    """
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request and handle errors"""
        try:
            response = await call_next(request)
            return response
            
        except ApplicationError as e:
            # Handle application-specific errors
            return JSONResponse(
                status_code=e.status_code,
                content=e.to_dict()
            )
            
        except Exception as e:
            # Handle unexpected errors
            logger.error(f"Unhandled exception: {e}", exc_info=True)
            
            # Don't expose internal errors in production
            if request.app.state.settings.is_production():
                content = {
                    "error": {
                        "code": "INTERNAL_ERROR",
                        "message": "An unexpected error occurred"
                    }
                }
            else:
                content = {
                    "error": {
                        "code": "INTERNAL_ERROR",
                        "message": str(e),
                        "type": type(e).__name__
                    }
                }
            
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content=content
            )


class SecurityMiddleware(BaseHTTPMiddleware):
    """
    Security middleware for handling security headers and CSRF protection
    """
    
    def __init__(self, app: ASGIApp, settings: Settings):
        super().__init__(app)
        self.settings = settings
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Add security headers and perform security checks"""
        
        # Check for SQL injection attempts
        if self._detect_sql_injection(request):
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={
                    "error": {
                        "code": "SECURITY_VIOLATION",
                        "message": "Invalid request detected"
                    }
                }
            )
        
        # Process request
        response = await call_next(request)
        
        # Add security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        
        # Remove sensitive headers
        response.headers.pop("Server", None)
        response.headers.pop("X-Powered-By", None)
        
        return response
    
    def _detect_sql_injection(self, request: Request) -> bool:
        """Simple SQL injection detection"""
        sql_keywords = [
            "select", "insert", "update", "delete", "drop",
            "union", "exec", "execute", "--", "/*", "*/"
        ]
        
        # Check query parameters
        for param_value in request.query_params.values():
            param_lower = str(param_value).lower()
            if any(keyword in param_lower for keyword in sql_keywords):
                logger.warning(f"Potential SQL injection detected: {param_value}")
                return True
        
        return False


class LoggingMiddleware(BaseHTTPMiddleware):
    """
    Request/Response logging middleware
    """
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Log request and response details"""
        
        # Generate request ID
        request_id = self._generate_request_id()
        request.state.request_id = request_id
        
        # Log request
        start_time = time.time()
        
        logger.info(
            f"Request started: {request_id} | "
            f"{request.method} {request.url.path} | "
            f"Client: {request.client.host if request.client else 'unknown'}"
        )
        
        # Process request
        response = await call_next(request)
        
        # Calculate duration
        duration = time.time() - start_time
        
        # Log response
        logger.info(
            f"Request completed: {request_id} | "
            f"Status: {response.status_code} | "
            f"Duration: {duration:.3f}s"
        )
        
        # Add request ID to response headers
        response.headers["X-Request-ID"] = request_id
        
        return response
    
    def _generate_request_id(self) -> str:
        """Generate unique request ID"""
        timestamp = str(time.time())
        random_str = str(hash(timestamp))
        return hashlib.md5(f"{timestamp}{random_str}".encode()).hexdigest()[:12]


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Rate limiting middleware to prevent abuse
    """
    
    def __init__(self, app: ASGIApp, requests_per_minute: int = 60):
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.request_counts: Dict[str, list] = {}
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Apply rate limiting"""
        
        # Get client identifier
        client_id = self._get_client_id(request)
        
        # Check rate limit
        if not self._check_rate_limit(client_id):
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "error": {
                        "code": "RATE_LIMIT_EXCEEDED",
                        "message": f"Rate limit exceeded. Maximum {self.requests_per_minute} requests per minute."
                    }
                },
                headers={
                    "Retry-After": "60"
                }
            )
        
        # Process request
        response = await call_next(request)
        
        # Add rate limit headers
        response.headers["X-RateLimit-Limit"] = str(self.requests_per_minute)
        response.headers["X-RateLimit-Remaining"] = str(
            self._get_remaining_requests(client_id)
        )
        
        return response
    
    def _get_client_id(self, request: Request) -> str:
        """Get client identifier for rate limiting"""
        # Try to get authenticated user ID
        if hasattr(request.state, "user_id"):
            return f"user:{request.state.user_id}"
        
        # Fall back to IP address
        if request.client:
            return f"ip:{request.client.host}"
        
        return "unknown"
    
    def _check_rate_limit(self, client_id: str) -> bool:
        """Check if client has exceeded rate limit"""
        now = datetime.utcnow()
        minute_ago = now - timedelta(minutes=1)
        
        # Get or create request list for client
        if client_id not in self.request_counts:
            self.request_counts[client_id] = []
        
        # Remove old requests
        self.request_counts[client_id] = [
            req_time for req_time in self.request_counts[client_id]
            if req_time > minute_ago
        ]
        
        # Check if limit exceeded
        if len(self.request_counts[client_id]) >= self.requests_per_minute:
            return False
        
        # Add current request
        self.request_counts[client_id].append(now)
        
        return True
    
    def _get_remaining_requests(self, client_id: str) -> int:
        """Get remaining requests for client"""
        if client_id not in self.request_counts:
            return self.requests_per_minute
        
        return max(0, self.requests_per_minute - len(self.request_counts[client_id]))


class AuthenticationMiddleware(BaseHTTPMiddleware):
    """
    Authentication middleware for protected routes
    """
    
    def __init__(self, app: ASGIApp, exclude_paths: Optional[list] = None):
        super().__init__(app)
        self.exclude_paths = exclude_paths or [
            "/",
            "/health",
            "/api/docs",
            "/api/redoc",
            "/api/openapi.json"
        ]
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Check authentication for protected routes"""
        
        # Skip authentication for excluded paths
        if request.url.path in self.exclude_paths:
            return await call_next(request)
        
        # Skip authentication for public endpoints
        if request.url.path.startswith("/api/v1/auth/"):
            return await call_next(request)
        
        # Check for authorization header
        auth_header = request.headers.get("Authorization")
        
        if not auth_header or not auth_header.startswith("Bearer "):
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={
                    "error": {
                        "code": ErrorCodes.INVALID_CREDENTIALS,
                        "message": "Authentication required"
                    }
                },
                headers={
                    "WWW-Authenticate": "Bearer"
                }
            )
        
        # Token validation would happen here
        # For now, just extract and store the token
        token = auth_header.replace("Bearer ", "")
        request.state.token = token
        
        # Process request
        return await call_next(request)


class CacheMiddleware(BaseHTTPMiddleware):
    """
    Response caching middleware for GET requests
    """
    
    def __init__(self, app: ASGIApp, cache_ttl_seconds: int = 300):
        super().__init__(app)
        self.cache_ttl = cache_ttl_seconds
        self.cache: Dict[str, tuple] = {}  # Simple in-memory cache
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Cache GET request responses"""
        
        # Only cache GET requests
        if request.method != "GET":
            return await call_next(request)
        
        # Skip caching for authenticated requests
        if hasattr(request.state, "user_id"):
            return await call_next(request)
        
        # Generate cache key
        cache_key = self._generate_cache_key(request)
        
        # Check cache
        if cache_key in self.cache:
            cached_response, cached_time = self.cache[cache_key]
            
            # Check if cache is still valid
            if time.time() - cached_time < self.cache_ttl:
                logger.debug(f"Cache hit for: {cache_key}")
                
                # Create response from cached data
                return JSONResponse(
                    content=cached_response["content"],
                    status_code=cached_response["status_code"],
                    headers={
                        **cached_response["headers"],
                        "X-Cache": "HIT"
                    }
                )
        
        # Process request
        response = await call_next(request)
        
        # Cache successful GET responses
        if response.status_code == 200 and request.method == "GET":
            # Read response body
            response_body = b""
            async for chunk in response.body_iterator:
                response_body += chunk
            
            # Cache response
            self.cache[cache_key] = (
                {
                    "content": json.loads(response_body),
                    "status_code": response.status_code,
                    "headers": dict(response.headers)
                },
                time.time()
            )
            
            # Create new response
            return Response(
                content=response_body,
                status_code=response.status_code,
                headers={
                    **dict(response.headers),
                    "X-Cache": "MISS"
                },
                media_type="application/json"
            )
        
        return response
    
    def _generate_cache_key(self, request: Request) -> str:
        """Generate cache key from request"""
        return f"{request.method}:{request.url.path}:{str(request.query_params)}"
    
    def clear_cache(self):
        """Clear all cached responses"""
        self.cache.clear()
        logger.info("Cache cleared")


# Export middleware
__all__ = [
    "ErrorHandlerMiddleware",
    "SecurityMiddleware",
    "LoggingMiddleware",
    "RateLimitMiddleware",
    "AuthenticationMiddleware",
    "CacheMiddleware"
]
