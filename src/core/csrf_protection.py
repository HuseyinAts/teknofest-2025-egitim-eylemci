"""
CSRF Protection Middleware
TEKNOFEST 2025 - Production Ready CSRF Protection
"""

import secrets
import hashlib
import hmac
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
import logging

from fastapi import Request, Response, HTTPException, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

logger = logging.getLogger(__name__)


class CSRFProtectionMiddleware(BaseHTTPMiddleware):
    """
    CSRF Protection using Double Submit Cookie pattern with HMAC signing
    """
    
    def __init__(
        self,
        app: ASGIApp,
        secret_key: str,
        cookie_name: str = "csrf_token",
        header_name: str = "X-CSRF-Token",
        cookie_secure: bool = True,
        cookie_httponly: bool = True,
        cookie_samesite: str = "strict",
        cookie_max_age: int = 3600,
        excluded_paths: Optional[list] = None,
        safe_methods: list = None,
    ):
        super().__init__(app)
        self.secret_key = secret_key.encode()
        self.cookie_name = cookie_name
        self.header_name = header_name
        self.cookie_secure = cookie_secure
        self.cookie_httponly = cookie_httponly
        self.cookie_samesite = cookie_samesite
        self.cookie_max_age = cookie_max_age
        self.excluded_paths = excluded_paths or [
            "/health", "/metrics", "/docs", "/redoc", "/openapi.json"
        ]
        self.safe_methods = safe_methods or ["GET", "HEAD", "OPTIONS"]
        
    def generate_csrf_token(self) -> str:
        """Generate a new CSRF token"""
        # Generate random token
        token = secrets.token_urlsafe(32)
        # Create timestamp
        timestamp = str(int(datetime.utcnow().timestamp()))
        # Create signature
        message = f"{token}:{timestamp}".encode()
        signature = hmac.new(self.secret_key, message, hashlib.sha256).hexdigest()
        # Return combined token
        return f"{token}:{timestamp}:{signature}"
    
    def verify_csrf_token(self, token: str) -> bool:
        """Verify CSRF token with HMAC signature"""
        try:
            parts = token.split(":")
            if len(parts) != 3:
                return False
            
            token_value, timestamp, signature = parts
            
            # Check timestamp (token expiry)
            token_time = int(timestamp)
            current_time = int(datetime.utcnow().timestamp())
            if current_time - token_time > self.cookie_max_age:
                logger.warning("CSRF token expired")
                return False
            
            # Verify signature
            message = f"{token_value}:{timestamp}".encode()
            expected_signature = hmac.new(self.secret_key, message, hashlib.sha256).hexdigest()
            
            # Constant-time comparison
            return hmac.compare_digest(signature, expected_signature)
            
        except (ValueError, AttributeError) as e:
            logger.error(f"CSRF token verification error: {e}")
            return False
    
    def is_excluded_path(self, path: str) -> bool:
        """Check if path is excluded from CSRF protection"""
        for excluded in self.excluded_paths:
            if path.startswith(excluded):
                return True
        return False
    
    async def dispatch(self, request: Request, call_next):
        # Skip CSRF check for safe methods
        if request.method in self.safe_methods:
            return await call_next(request)
        
        # Skip for excluded paths
        if self.is_excluded_path(request.url.path):
            return await call_next(request)
        
        # Skip for API endpoints that use different auth
        if request.url.path.startswith("/api/") and "Bearer" in request.headers.get("Authorization", ""):
            return await call_next(request)
        
        # Get CSRF token from cookie
        cookie_token = request.cookies.get(self.cookie_name)
        
        # Get CSRF token from header or form
        header_token = request.headers.get(self.header_name)
        
        # If no cookie token, generate new one
        if not cookie_token:
            response = await call_next(request)
            new_token = self.generate_csrf_token()
            response.set_cookie(
                key=self.cookie_name,
                value=new_token,
                secure=self.cookie_secure,
                httponly=self.cookie_httponly,
                samesite=self.cookie_samesite,
                max_age=self.cookie_max_age,
            )
            # Also return token in response header for SPA
            response.headers["X-CSRF-Token"] = new_token
            return response
        
        # Verify tokens match and are valid
        if not header_token:
            logger.warning(f"Missing CSRF token in header for {request.url.path}")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="CSRF token missing"
            )
        
        if not self.verify_csrf_token(cookie_token):
            logger.warning(f"Invalid CSRF cookie token for {request.url.path}")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Invalid CSRF token"
            )
        
        if cookie_token != header_token:
            logger.warning(f"CSRF token mismatch for {request.url.path}")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="CSRF token mismatch"
            )
        
        # Process request
        response = await call_next(request)
        
        # Rotate token periodically (every request for high security)
        # Uncomment for token rotation:
        # new_token = self.generate_csrf_token()
        # response.set_cookie(...)
        
        return response


class CSRFTokenGenerator:
    """Utility class for CSRF token operations"""
    
    def __init__(self, secret_key: str):
        self.secret_key = secret_key.encode()
    
    def generate_token(self, session_id: str) -> str:
        """Generate a CSRF token bound to session"""
        timestamp = str(int(datetime.utcnow().timestamp()))
        message = f"{session_id}:{timestamp}".encode()
        signature = hmac.new(self.secret_key, message, hashlib.sha256).hexdigest()
        return f"{session_id}:{timestamp}:{signature}"
    
    def verify_token(self, token: str, session_id: str, max_age: int = 3600) -> bool:
        """Verify CSRF token bound to session"""
        try:
            parts = token.split(":")
            if len(parts) != 3:
                return False
            
            token_session, timestamp, signature = parts
            
            # Check session match
            if token_session != session_id:
                return False
            
            # Check timestamp
            token_time = int(timestamp)
            current_time = int(datetime.utcnow().timestamp())
            if current_time - token_time > max_age:
                return False
            
            # Verify signature
            message = f"{session_id}:{timestamp}".encode()
            expected_signature = hmac.new(self.secret_key, message, hashlib.sha256).hexdigest()
            
            return hmac.compare_digest(signature, expected_signature)
            
        except (ValueError, AttributeError):
            return False


def setup_csrf_protection(app: ASGIApp, settings: Any) -> None:
    """Setup CSRF protection middleware"""
    
    # Only enable in production or when explicitly enabled
    if not settings.is_production() and not getattr(settings, 'csrf_enabled', False):
        logger.info("CSRF protection disabled in development")
        return
    
    csrf_middleware = CSRFProtectionMiddleware(
        app,
        secret_key=settings.secret_key.get_secret_value(),
        cookie_secure=settings.is_production(),
        cookie_httponly=True,
        cookie_samesite="strict" if settings.is_production() else "lax",
        excluded_paths=[
            "/health",
            "/metrics", 
            "/docs",
            "/redoc",
            "/openapi.json",
            "/api/auth/login",  # Login endpoint generates CSRF token
            "/api/auth/register",  # Registration endpoint
            "/api/auth/csrf",  # CSRF token endpoint
        ]
    )
    
    app.add_middleware(CSRFProtectionMiddleware, **{
        "secret_key": settings.secret_key.get_secret_value(),
        "cookie_secure": settings.is_production(),
        "cookie_httponly": True,
        "cookie_samesite": "strict" if settings.is_production() else "lax",
        "excluded_paths": csrf_middleware.excluded_paths,
    })
    
    logger.info("CSRF protection middleware configured")