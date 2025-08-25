"""
Security Middleware and Utilities
TEKNOFEST 2025 - Production Security Layer
"""

import time
import hashlib
import hmac
import re
from typing import Dict, Optional, List, Any
from datetime import datetime, timedelta
from functools import wraps
import logging

from fastapi import Request, HTTPException, status
from fastapi.responses import Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp
import bcrypt
from sqlalchemy import text
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


class SecurityHeaders:
    """Security headers for HTTP responses"""
    
    HEADERS = {
        "X-Content-Type-Options": "nosniff",
        "X-Frame-Options": "DENY",
        "X-XSS-Protection": "1; mode=block",
        "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
        "Content-Security-Policy": "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'",
        "Referrer-Policy": "strict-origin-when-cross-origin",
        "Permissions-Policy": "geolocation=(), microphone=(), camera=()"
    }
    
    @classmethod
    def apply(cls, response: Response) -> Response:
        """Apply security headers to response"""
        for header, value in cls.HEADERS.items():
            response.headers[header] = value
        return response


class RateLimiter:
    """Token bucket rate limiter"""
    
    def __init__(self, max_requests: int = 100, window_seconds: int = 3600):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests: Dict[str, List[float]] = {}
        self.blocked_ips: Dict[str, float] = {}
        
    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP from request"""
        # Check for proxy headers
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()
        
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip
            
        return request.client.host if request.client else "unknown"
    
    def _clean_old_requests(self, ip: str, current_time: float):
        """Remove requests outside the time window"""
        if ip in self.requests:
            cutoff = current_time - self.window_seconds
            self.requests[ip] = [t for t in self.requests[ip] if t > cutoff]
    
    def is_blocked(self, ip: str) -> bool:
        """Check if IP is temporarily blocked"""
        if ip in self.blocked_ips:
            if time.time() < self.blocked_ips[ip]:
                return True
            else:
                del self.blocked_ips[ip]
        return False
    
    def check_rate_limit(self, request: Request) -> bool:
        """Check if request exceeds rate limit"""
        ip = self._get_client_ip(request)
        current_time = time.time()
        
        # Check if IP is blocked
        if self.is_blocked(ip):
            return False
        
        # Clean old requests
        self._clean_old_requests(ip, current_time)
        
        # Initialize if new IP
        if ip not in self.requests:
            self.requests[ip] = []
        
        # Check rate limit
        if len(self.requests[ip]) >= self.max_requests:
            # Block IP for 5 minutes
            self.blocked_ips[ip] = current_time + 300
            logger.warning(f"Rate limit exceeded for IP: {ip}")
            return False
        
        # Add current request
        self.requests[ip].append(current_time)
        return True
    
    def get_remaining_requests(self, request: Request) -> int:
        """Get remaining requests for client"""
        ip = self._get_client_ip(request)
        if ip in self.requests:
            return max(0, self.max_requests - len(self.requests[ip]))
        return self.max_requests


class SQLInjectionProtection:
    """SQL Injection prevention utilities"""
    
    # Dangerous SQL patterns
    DANGEROUS_PATTERNS = [
        r"(\b(union|select|insert|update|delete|drop|create|alter|exec|execute|script|javascript)\b)",
        r"(--|#|\/\*|\*\/)",  # SQL comments
        r"(\bor\b\s*\d+\s*=\s*\d+)",  # OR 1=1
        r"(\band\b\s*\d+\s*=\s*\d+)",  # AND 1=1
        r"(char\s*\(\s*\d+\s*\))",  # char() function
        r"(waitfor\s+delay)",  # Time-based attacks
        r"(benchmark\s*\()",  # MySQL benchmark
        r"(sleep\s*\()",  # Sleep function
    ]
    
    @classmethod
    def validate_input(cls, value: str) -> bool:
        """Validate input for SQL injection attempts"""
        if not value:
            return True
        
        value_lower = value.lower()
        
        for pattern in cls.DANGEROUS_PATTERNS:
            if re.search(pattern, value_lower, re.IGNORECASE):
                logger.warning(f"Potential SQL injection detected: {value[:100]}")
                return False
        
        return True
    
    @classmethod
    def sanitize_input(cls, value: str) -> str:
        """Sanitize input by escaping special characters"""
        if not value:
            return value
        
        # Escape special SQL characters
        value = value.replace("'", "''")
        value = value.replace('"', '""')
        value = value.replace("\\", "\\\\")
        value = value.replace("%", "\\%")
        value = value.replace("_", "\\_")
        
        return value
    
    @classmethod
    def create_parameterized_query(cls, query: str, params: Dict[str, Any]) -> tuple:
        """Create parameterized query to prevent SQL injection"""
        # Convert to SQLAlchemy text with bound parameters
        stmt = text(query)
        return stmt, params


class InputValidator:
    """Input validation utilities"""
    
    @staticmethod
    def validate_email(email: str) -> bool:
        """Validate email format"""
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return bool(re.match(pattern, email))
    
    @staticmethod
    def validate_username(username: str) -> bool:
        """Validate username format"""
        # 3-20 characters, alphanumeric and underscore only
        pattern = r'^[a-zA-Z0-9_]{3,20}$'
        return bool(re.match(pattern, username))
    
    @staticmethod
    def validate_password(password: str, min_length: int = 8, require_special: bool = True) -> tuple[bool, str]:
        """Validate password strength"""
        if len(password) < min_length:
            return False, f"Password must be at least {min_length} characters"
        
        if not re.search(r'[A-Z]', password):
            return False, "Password must contain at least one uppercase letter"
        
        if not re.search(r'[a-z]', password):
            return False, "Password must contain at least one lowercase letter"
        
        if not re.search(r'\d', password):
            return False, "Password must contain at least one number"
        
        if require_special and not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
            return False, "Password must contain at least one special character"
        
        return True, "Password is valid"
    
    @staticmethod
    def sanitize_html(html: str) -> str:
        """Remove potentially dangerous HTML tags"""
        # Remove script tags
        html = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL | re.IGNORECASE)
        # Remove iframe tags
        html = re.sub(r'<iframe[^>]*>.*?</iframe>', '', html, flags=re.DOTALL | re.IGNORECASE)
        # Remove object tags
        html = re.sub(r'<object[^>]*>.*?</object>', '', html, flags=re.DOTALL | re.IGNORECASE)
        # Remove embed tags
        html = re.sub(r'<embed[^>]*>', '', html, flags=re.IGNORECASE)
        # Remove on* event handlers
        html = re.sub(r'\s*on\w+\s*=\s*["\'][^"\']*["\']', '', html, flags=re.IGNORECASE)
        
        return html


class PasswordManager:
    """Secure password hashing and verification"""
    
    @staticmethod
    def hash_password(password: str, rounds: int = 12) -> str:
        """Hash password using bcrypt"""
        salt = bcrypt.gensalt(rounds=rounds)
        hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
        return hashed.decode('utf-8')
    
    @staticmethod
    def verify_password(password: str, hashed: str) -> bool:
        """Verify password against hash"""
        try:
            return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))
        except Exception as e:
            logger.error(f"Password verification error: {e}")
            return False
    
    @staticmethod
    def generate_secure_token(length: int = 32) -> str:
        """Generate cryptographically secure random token"""
        import secrets
        return secrets.token_urlsafe(length)


class SessionManager:
    """Secure session management"""
    
    def __init__(self, secret_key: str, lifetime_hours: int = 24):
        self.secret_key = secret_key
        self.lifetime_hours = lifetime_hours
        self.sessions: Dict[str, Dict] = {}
    
    def create_session(self, user_id: str, data: Dict = None) -> str:
        """Create new session"""
        session_id = PasswordManager.generate_secure_token()
        
        self.sessions[session_id] = {
            'user_id': user_id,
            'created_at': datetime.utcnow(),
            'expires_at': datetime.utcnow() + timedelta(hours=self.lifetime_hours),
            'data': data or {},
            'ip': None,
            'user_agent': None
        }
        
        return session_id
    
    def validate_session(self, session_id: str) -> Optional[Dict]:
        """Validate and return session data"""
        if session_id not in self.sessions:
            return None
        
        session = self.sessions[session_id]
        
        # Check expiration
        if datetime.utcnow() > session['expires_at']:
            del self.sessions[session_id]
            return None
        
        # Refresh expiration
        session['expires_at'] = datetime.utcnow() + timedelta(hours=self.lifetime_hours)
        
        return session
    
    def destroy_session(self, session_id: str):
        """Destroy session"""
        if session_id in self.sessions:
            del self.sessions[session_id]


class CSRFProtection:
    """CSRF token generation and validation"""
    
    def __init__(self, secret_key: str):
        self.secret_key = secret_key
    
    def generate_token(self, session_id: str) -> str:
        """Generate CSRF token for session"""
        message = f"{session_id}:{time.time()}"
        signature = hmac.new(
            self.secret_key.encode(),
            message.encode(),
            hashlib.sha256
        ).hexdigest()
        
        return f"{message}:{signature}"
    
    def validate_token(self, token: str, session_id: str, max_age: int = 3600) -> bool:
        """Validate CSRF token"""
        try:
            parts = token.split(':')
            if len(parts) != 3:
                return False
            
            stored_session, timestamp, signature = parts
            
            # Check session match
            if stored_session != session_id:
                return False
            
            # Check age
            if time.time() - float(timestamp) > max_age:
                return False
            
            # Verify signature
            message = f"{stored_session}:{timestamp}"
            expected_signature = hmac.new(
                self.secret_key.encode(),
                message.encode(),
                hashlib.sha256
            ).hexdigest()
            
            return hmac.compare_digest(signature, expected_signature)
            
        except Exception as e:
            logger.error(f"CSRF validation error: {e}")
            return False


class SecurityMiddleware(BaseHTTPMiddleware):
    """Main security middleware combining all protections"""
    
    def __init__(self, app: ASGIApp, settings):
        super().__init__(app)
        self.settings = settings
        self.rate_limiter = RateLimiter(
            max_requests=settings.rate_limit_requests,
            window_seconds=settings.rate_limit_period
        )
    
    async def dispatch(self, request: Request, call_next):
        # 1. Check rate limiting
        if self.settings.rate_limit_enabled:
            if not self.rate_limiter.check_rate_limit(request):
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail="Rate limit exceeded. Please try again later."
                )
        
        # 2. Validate input for SQL injection
        if request.method in ["POST", "PUT", "PATCH"]:
            try:
                body = await request.body()
                if body:
                    # Simple check on raw body
                    if not SQLInjectionProtection.validate_input(body.decode('utf-8', errors='ignore')):
                        raise HTTPException(
                            status_code=status.HTTP_400_BAD_REQUEST,
                            detail="Invalid input detected"
                        )
            except:
                pass  # Body already consumed or other error
        
        # 3. Process request
        response = await call_next(request)
        
        # 4. Add security headers
        response = SecurityHeaders.apply(response)
        
        # 5. Add rate limit headers
        if self.settings.rate_limit_enabled:
            remaining = self.rate_limiter.get_remaining_requests(request)
            response.headers["X-RateLimit-Limit"] = str(self.settings.rate_limit_requests)
            response.headers["X-RateLimit-Remaining"] = str(remaining)
            response.headers["X-RateLimit-Reset"] = str(int(time.time()) + self.settings.rate_limit_period)
        
        return response


def require_auth(func):
    """Decorator to require authentication"""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        # Implementation depends on your auth system
        # This is a placeholder
        request = kwargs.get('request')
        if not request:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication required"
            )
        
        # Check for auth header or session
        auth_header = request.headers.get('Authorization')
        if not auth_header:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Missing authentication"
            )
        
        return await func(*args, **kwargs)
    
    return wrapper


def validate_request_data(data_class):
    """Decorator to validate request data against a Pydantic model"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                # Validate data
                request = kwargs.get('request')
                if request and hasattr(request, 'json'):
                    data = await request.json()
                    validated_data = data_class(**data)
                    kwargs['validated_data'] = validated_data
            except Exception as e:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail=f"Invalid request data: {str(e)}"
                )
            
            return await func(*args, **kwargs)
        
        return wrapper
    return decorator
