"""
Production Security Headers Middleware
Implements comprehensive security headers following OWASP best practices
"""

from typing import Optional, List, Dict, Any
from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp
import hashlib
import secrets
import time
import logging

logger = logging.getLogger(__name__)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """
    Comprehensive security headers middleware implementing:
    - Content Security Policy (CSP)
    - Strict Transport Security (HSTS)
    - X-Frame-Options
    - X-Content-Type-Options
    - X-XSS-Protection
    - Referrer-Policy
    - Permissions-Policy
    - Cache-Control for sensitive data
    """
    
    def __init__(
        self,
        app: ASGIApp,
        enforce_https: bool = True,
        csp_enabled: bool = True,
        csp_report_only: bool = False,
        csp_report_uri: Optional[str] = None,
        hsts_max_age: int = 31536000,  # 1 year
        frame_options: str = "DENY",
        custom_headers: Optional[Dict[str, str]] = None
    ):
        super().__init__(app)
        self.enforce_https = enforce_https
        self.csp_enabled = csp_enabled
        self.csp_report_only = csp_report_only
        self.csp_report_uri = csp_report_uri
        self.hsts_max_age = hsts_max_age
        self.frame_options = frame_options
        self.custom_headers = custom_headers or {}
        
        # Generate nonce for inline scripts/styles
        self.nonce_generator = lambda: secrets.token_urlsafe(16)
    
    async def dispatch(self, request: Request, call_next):
        # Generate CSP nonce for this request
        csp_nonce = self.nonce_generator()
        request.state.csp_nonce = csp_nonce
        
        # Process request
        response = await call_next(request)
        
        # Add security headers
        self._add_security_headers(response, request, csp_nonce)
        
        return response
    
    def _add_security_headers(self, response: Response, request: Request, nonce: str):
        """Add all security headers to response"""
        
        # 1. Strict-Transport-Security (HSTS)
        if self.enforce_https:
            response.headers["Strict-Transport-Security"] = (
                f"max-age={self.hsts_max_age}; includeSubDomains; preload"
            )
        
        # 2. Content-Security-Policy (CSP)
        if self.csp_enabled:
            csp_header = self._build_csp_header(nonce)
            if self.csp_report_only:
                response.headers["Content-Security-Policy-Report-Only"] = csp_header
            else:
                response.headers["Content-Security-Policy"] = csp_header
        
        # 3. X-Frame-Options
        response.headers["X-Frame-Options"] = self.frame_options
        
        # 4. X-Content-Type-Options
        response.headers["X-Content-Type-Options"] = "nosniff"
        
        # 5. X-XSS-Protection (legacy but still useful)
        response.headers["X-XSS-Protection"] = "1; mode=block"
        
        # 6. Referrer-Policy
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        
        # 7. Permissions-Policy (formerly Feature-Policy)
        response.headers["Permissions-Policy"] = self._build_permissions_policy()
        
        # 8. Cache-Control for sensitive endpoints
        if self._is_sensitive_endpoint(request.url.path):
            response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, private"
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"
        
        # 9. Remove server header (if possible)
        response.headers.pop("Server", None)
        response.headers.pop("X-Powered-By", None)
        
        # 10. Add custom headers
        for key, value in self.custom_headers.items():
            response.headers[key] = value
    
    def _build_csp_header(self, nonce: str) -> str:
        """Build Content Security Policy header"""
        directives = [
            "default-src 'self'",
            f"script-src 'self' 'nonce-{nonce}' https://cdn.jsdelivr.net https://unpkg.com",
            f"style-src 'self' 'nonce-{nonce}' https://fonts.googleapis.com",
            "img-src 'self' data: https:",
            "font-src 'self' https://fonts.gstatic.com",
            "connect-src 'self' https://api.sentry.io wss://",
            "media-src 'self'",
            "object-src 'none'",
            "base-uri 'self'",
            "form-action 'self'",
            "frame-ancestors 'none'",
            "upgrade-insecure-requests",
        ]
        
        if self.csp_report_uri:
            directives.append(f"report-uri {self.csp_report_uri}")
        
        return "; ".join(directives)
    
    def _build_permissions_policy(self) -> str:
        """Build Permissions Policy header"""
        policies = [
            "accelerometer=()",
            "camera=()",
            "geolocation=()",
            "gyroscope=()",
            "magnetometer=()",
            "microphone=()",
            "payment=()",
            "usb=()",
            "interest-cohort=()",  # Opt out of FLoC
        ]
        return ", ".join(policies)
    
    def _is_sensitive_endpoint(self, path: str) -> bool:
        """Check if endpoint handles sensitive data"""
        sensitive_patterns = [
            "/api/auth",
            "/api/login",
            "/api/register",
            "/api/token",
            "/api/user",
            "/api/profile",
            "/api/admin",
            "/api/payment",
        ]
        return any(path.startswith(pattern) for pattern in sensitive_patterns)


class AdvancedSecurityMiddleware(BaseHTTPMiddleware):
    """
    Advanced security features including:
    - Request rate limiting
    - SQL injection detection
    - XSS prevention
    - Path traversal protection
    - Request size limits
    - Suspicious pattern detection
    """
    
    def __init__(
        self,
        app: ASGIApp,
        max_request_size: int = 10 * 1024 * 1024,  # 10MB
        enable_sql_injection_protection: bool = True,
        enable_xss_protection: bool = True,
        enable_path_traversal_protection: bool = True,
        blocked_user_agents: Optional[List[str]] = None,
        blocked_ips: Optional[List[str]] = None
    ):
        super().__init__(app)
        self.max_request_size = max_request_size
        self.enable_sql_injection_protection = enable_sql_injection_protection
        self.enable_xss_protection = enable_xss_protection
        self.enable_path_traversal_protection = enable_path_traversal_protection
        self.blocked_user_agents = blocked_user_agents or []
        self.blocked_ips = blocked_ips or []
        
        # SQL injection patterns
        self.sql_patterns = [
            r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|UNION|ALTER|CREATE)\b)",
            r"(--|\||;|\/\*|\*\/)",
            r"(xp_cmdshell|sp_executesql)",
            r"(EXEC\s*\(|EXECUTE\s*\()",
        ]
        
        # XSS patterns
        self.xss_patterns = [
            r"<script[^>]*>.*?</script>",
            r"javascript:",
            r"on\w+\s*=",
            r"<iframe[^>]*>",
            r"<object[^>]*>",
            r"<embed[^>]*>",
        ]
        
        # Path traversal patterns
        self.path_traversal_patterns = [
            r"\.\./",
            r"\.\.",
            r"%2e%2e",
            r"\.\.\\",
        ]
    
    async def dispatch(self, request: Request, call_next):
        # Check blocked IPs
        client_ip = request.client.host
        if client_ip in self.blocked_ips:
            return JSONResponse(
                status_code=403,
                content={"detail": "Access denied"}
            )
        
        # Check blocked user agents
        user_agent = request.headers.get("User-Agent", "")
        if any(blocked in user_agent for blocked in self.blocked_user_agents):
            return JSONResponse(
                status_code=403,
                content={"detail": "Access denied"}
            )
        
        # Check request size
        content_length = request.headers.get("Content-Length")
        if content_length and int(content_length) > self.max_request_size:
            return JSONResponse(
                status_code=413,
                content={"detail": "Request entity too large"}
            )
        
        # Check for malicious patterns in URL
        if self._check_malicious_patterns(str(request.url)):
            logger.warning(f"Malicious pattern detected in URL from {client_ip}")
            return JSONResponse(
                status_code=400,
                content={"detail": "Invalid request"}
            )
        
        # Process request
        response = await call_next(request)
        
        return response
    
    def _check_malicious_patterns(self, text: str) -> bool:
        """Check for malicious patterns in text"""
        import re
        
        text_lower = text.lower()
        
        # Check SQL injection patterns
        if self.enable_sql_injection_protection:
            for pattern in self.sql_patterns:
                if re.search(pattern, text_lower, re.IGNORECASE):
                    return True
        
        # Check XSS patterns
        if self.enable_xss_protection:
            for pattern in self.xss_patterns:
                if re.search(pattern, text_lower, re.IGNORECASE):
                    return True
        
        # Check path traversal patterns
        if self.enable_path_traversal_protection:
            for pattern in self.path_traversal_patterns:
                if re.search(pattern, text_lower, re.IGNORECASE):
                    return True
        
        return False


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Token bucket rate limiting middleware
    """
    
    def __init__(
        self,
        app: ASGIApp,
        requests_per_second: float = 10,
        burst_size: int = 20,
        key_func: Optional[callable] = None
    ):
        super().__init__(app)
        self.requests_per_second = requests_per_second
        self.burst_size = burst_size
        self.key_func = key_func or self._default_key_func
        self.buckets: Dict[str, Dict[str, Any]] = {}
    
    def _default_key_func(self, request: Request) -> str:
        """Default key function using client IP"""
        return request.client.host
    
    def _get_bucket(self, key: str) -> Dict[str, Any]:
        """Get or create token bucket for key"""
        now = time.time()
        
        if key not in self.buckets:
            self.buckets[key] = {
                "tokens": self.burst_size,
                "last_update": now
            }
        
        bucket = self.buckets[key]
        
        # Refill tokens
        time_passed = now - bucket["last_update"]
        tokens_to_add = time_passed * self.requests_per_second
        bucket["tokens"] = min(self.burst_size, bucket["tokens"] + tokens_to_add)
        bucket["last_update"] = now
        
        return bucket
    
    async def dispatch(self, request: Request, call_next):
        # Get rate limit key
        key = self.key_func(request)
        
        # Get token bucket
        bucket = self._get_bucket(key)
        
        # Check if request is allowed
        if bucket["tokens"] >= 1:
            bucket["tokens"] -= 1
            response = await call_next(request)
            
            # Add rate limit headers
            response.headers["X-RateLimit-Limit"] = str(self.burst_size)
            response.headers["X-RateLimit-Remaining"] = str(int(bucket["tokens"]))
            response.headers["X-RateLimit-Reset"] = str(int(time.time() + 60))
            
            return response
        else:
            # Rate limit exceeded
            retry_after = int((1 - bucket["tokens"]) / self.requests_per_second)
            return JSONResponse(
                status_code=429,
                content={"detail": "Rate limit exceeded"},
                headers={"Retry-After": str(retry_after)}
            )


def setup_security_middleware(app: ASGIApp, settings: Any) -> None:
    """Setup all security middleware for the application"""
    
    # Add rate limiting
    app.add_middleware(
        RateLimitMiddleware,
        requests_per_second=settings.rate_limit_requests / settings.rate_limit_period,
        burst_size=settings.rate_limit_requests
    )
    
    # Add advanced security
    app.add_middleware(
        AdvancedSecurityMiddleware,
        enable_sql_injection_protection=True,
        enable_xss_protection=True,
        enable_path_traversal_protection=True
    )
    
    # Add security headers
    app.add_middleware(
        SecurityHeadersMiddleware,
        enforce_https=settings.is_production(),
        csp_enabled=True,
        csp_report_only=not settings.is_production(),
        hsts_max_age=31536000,
        frame_options="DENY"
    )
    
    logger.info("Security middleware configured successfully")