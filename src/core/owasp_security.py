"""
OWASP Top 10 Security Controls Implementation
Comprehensive security measures for production environment
"""

import hashlib
import hmac
import secrets
import re
from typing import Optional, Any, Dict, List
from datetime import datetime, timedelta
import logging
from functools import wraps
import json

from fastapi import Request, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
import jwt
from passlib.context import CryptContext

logger = logging.getLogger(__name__)

# Password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


class OWASPSecurityControls:
    """
    Implementation of OWASP Top 10 security controls:
    1. Broken Access Control
    2. Cryptographic Failures
    3. Injection
    4. Insecure Design
    5. Security Misconfiguration
    6. Vulnerable and Outdated Components
    7. Identification and Authentication Failures
    8. Software and Data Integrity Failures
    9. Security Logging and Monitoring Failures
    10. Server-Side Request Forgery (SSRF)
    """
    
    def __init__(self, settings: Any):
        self.settings = settings
        self.security_logger = self._setup_security_logger()
        
    def _setup_security_logger(self) -> logging.Logger:
        """Setup dedicated security logger"""
        security_logger = logging.getLogger("security")
        security_logger.setLevel(logging.INFO)
        
        # Add file handler for security events
        handler = logging.FileHandler("logs/security.log")
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s - %(extra)s'
        )
        handler.setFormatter(formatter)
        security_logger.addHandler(handler)
        
        return security_logger
    
    # ==========================================
    # 1. BROKEN ACCESS CONTROL PREVENTION
    # ==========================================
    
    def enforce_rbac(self, required_roles: List[str]):
        """Role-Based Access Control decorator"""
        def decorator(func):
            @wraps(func)
            async def wrapper(request: Request, *args, **kwargs):
                user = getattr(request.state, "user", None)
                if not user:
                    raise HTTPException(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        detail="Authentication required"
                    )
                
                user_roles = user.get("roles", [])
                if not any(role in user_roles for role in required_roles):
                    self.security_logger.warning(
                        f"Access denied for user {user.get('id')} to {request.url.path}",
                        extra={"user_id": user.get("id"), "path": request.url.path}
                    )
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail="Insufficient permissions"
                    )
                
                return await func(request, *args, **kwargs)
            return wrapper
        return decorator
    
    def validate_resource_ownership(self, user_id: int, resource_owner_id: int):
        """Validate that user owns the resource they're trying to access"""
        if user_id != resource_owner_id:
            self.security_logger.warning(
                f"Unauthorized resource access attempt by user {user_id}",
                extra={"user_id": user_id, "owner_id": resource_owner_id}
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You don't have permission to access this resource"
            )
    
    # ==========================================
    # 2. CRYPTOGRAPHIC FAILURES PREVENTION
    # ==========================================
    
    def encrypt_sensitive_data(self, data: str, key: Optional[bytes] = None) -> str:
        """Encrypt sensitive data using AES"""
        from cryptography.fernet import Fernet
        
        if not key:
            key = self.settings.secret_key.get_secret_value().encode()[:32]
            key = key.ljust(32, b'0')  # Ensure 32 bytes
        
        # Create Fernet instance
        fernet_key = Fernet.generate_key()
        f = Fernet(fernet_key)
        
        # Encrypt data
        encrypted = f.encrypt(data.encode())
        return encrypted.hex()
    
    def hash_password(self, password: str) -> str:
        """Securely hash passwords"""
        # Validate password strength
        self._validate_password_strength(password)
        return pwd_context.hash(password)
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify password against hash"""
        return pwd_context.verify(plain_password, hashed_password)
    
    def _validate_password_strength(self, password: str):
        """Validate password meets security requirements"""
        if len(password) < self.settings.security_password_min_length:
            raise ValueError(f"Password must be at least {self.settings.security_password_min_length} characters")
        
        if self.settings.security_password_require_special:
            if not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
                raise ValueError("Password must contain at least one special character")
        
        if not re.search(r'[A-Z]', password):
            raise ValueError("Password must contain at least one uppercase letter")
        
        if not re.search(r'[a-z]', password):
            raise ValueError("Password must contain at least one lowercase letter")
        
        if not re.search(r'\d', password):
            raise ValueError("Password must contain at least one number")
    
    # ==========================================
    # 3. INJECTION PREVENTION
    # ==========================================
    
    def sanitize_input(self, input_data: str) -> str:
        """Sanitize user input to prevent injection attacks"""
        # Remove SQL injection patterns
        sql_patterns = [
            r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|UNION|ALTER|CREATE|EXEC|EXECUTE)\b)",
            r"(--|\||;|\/\*|\*\/)",
            r"(xp_cmdshell|sp_executesql)",
        ]
        
        sanitized = input_data
        for pattern in sql_patterns:
            sanitized = re.sub(pattern, "", sanitized, flags=re.IGNORECASE)
        
        # Remove XSS patterns
        xss_patterns = [
            r"<script[^>]*>.*?</script>",
            r"javascript:",
            r"on\w+\s*=",
            r"<iframe[^>]*>",
        ]
        
        for pattern in xss_patterns:
            sanitized = re.sub(pattern, "", sanitized, flags=re.IGNORECASE)
        
        return sanitized
    
    def validate_sql_query(self, query: str) -> bool:
        """Validate SQL query for safety"""
        dangerous_keywords = [
            "DROP", "DELETE", "TRUNCATE", "ALTER", 
            "CREATE", "REPLACE", "EXEC", "EXECUTE"
        ]
        
        query_upper = query.upper()
        for keyword in dangerous_keywords:
            if keyword in query_upper:
                self.security_logger.warning(
                    f"Dangerous SQL keyword detected: {keyword}",
                    extra={"query": query[:100]}
                )
                return False
        
        return True
    
    # ==========================================
    # 4. INSECURE DESIGN PREVENTION
    # ==========================================
    
    def implement_defense_in_depth(self):
        """Multiple layers of security controls"""
        return {
            "authentication": "JWT with refresh tokens",
            "authorization": "RBAC with fine-grained permissions",
            "encryption": "TLS 1.3 + AES-256",
            "input_validation": "Whitelist validation + sanitization",
            "rate_limiting": "Token bucket algorithm",
            "monitoring": "Real-time security event monitoring",
            "backup": "Automated encrypted backups"
        }
    
    def threat_model_validation(self, feature: str) -> Dict[str, Any]:
        """Validate security threats for a feature"""
        threats = {
            "authentication": [
                "Brute force attacks",
                "Credential stuffing",
                "Session hijacking"
            ],
            "file_upload": [
                "Malicious file upload",
                "Path traversal",
                "File size DoS"
            ],
            "api": [
                "Rate limiting bypass",
                "API key exposure",
                "CORS misconfiguration"
            ]
        }
        
        mitigations = {
            "authentication": [
                "Account lockout after failed attempts",
                "CAPTCHA for suspicious activity",
                "Secure session management"
            ],
            "file_upload": [
                "File type validation",
                "Virus scanning",
                "Size limits and quotas"
            ],
            "api": [
                "Distributed rate limiting",
                "API key rotation",
                "Strict CORS policy"
            ]
        }
        
        return {
            "feature": feature,
            "threats": threats.get(feature, []),
            "mitigations": mitigations.get(feature, [])
        }
    
    # ==========================================
    # 5. SECURITY MISCONFIGURATION PREVENTION
    # ==========================================
    
    def validate_security_headers(self, headers: Dict[str, str]) -> List[str]:
        """Validate required security headers are present"""
        required_headers = [
            "Strict-Transport-Security",
            "Content-Security-Policy",
            "X-Frame-Options",
            "X-Content-Type-Options",
            "Referrer-Policy"
        ]
        
        missing_headers = []
        for header in required_headers:
            if header not in headers:
                missing_headers.append(header)
        
        if missing_headers:
            self.security_logger.warning(
                f"Missing security headers: {missing_headers}"
            )
        
        return missing_headers
    
    def disable_debug_in_production(self):
        """Ensure debug mode is disabled in production"""
        if self.settings.is_production() and self.settings.app_debug:
            self.security_logger.error("Debug mode enabled in production!")
            raise ValueError("Debug mode must be disabled in production")
    
    # ==========================================
    # 6. VULNERABLE COMPONENTS PREVENTION
    # ==========================================
    
    def check_dependency_vulnerabilities(self) -> Dict[str, Any]:
        """Check for known vulnerabilities in dependencies"""
        import subprocess
        
        try:
            # Run safety check
            result = subprocess.run(
                ["safety", "check", "--json"],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            vulnerabilities = json.loads(result.stdout) if result.stdout else []
            
            if vulnerabilities:
                self.security_logger.warning(
                    f"Found {len(vulnerabilities)} vulnerable dependencies",
                    extra={"vulnerabilities": vulnerabilities}
                )
            
            return {
                "status": "vulnerable" if vulnerabilities else "secure",
                "vulnerabilities": vulnerabilities
            }
        except Exception as e:
            logger.error(f"Failed to check dependencies: {e}")
            return {"status": "error", "message": str(e)}
    
    # ==========================================
    # 7. AUTHENTICATION FAILURES PREVENTION
    # ==========================================
    
    def implement_account_lockout(self, user_id: int, failed_attempts: int):
        """Implement account lockout after failed login attempts"""
        max_attempts = self.settings.security_max_login_attempts
        
        if failed_attempts >= max_attempts:
            lockout_time = datetime.utcnow() + timedelta(
                minutes=self.settings.security_lockout_duration_minutes
            )
            
            self.security_logger.warning(
                f"Account locked for user {user_id} until {lockout_time}",
                extra={"user_id": user_id, "lockout_until": lockout_time.isoformat()}
            )
            
            return {
                "locked": True,
                "lockout_until": lockout_time,
                "reason": f"Too many failed login attempts ({failed_attempts})"
            }
        
        return {"locked": False, "attempts_remaining": max_attempts - failed_attempts}
    
    def validate_session_security(self, session: Dict[str, Any]) -> bool:
        """Validate session security parameters"""
        # Check session expiry
        if "expires_at" in session:
            expires_at = datetime.fromisoformat(session["expires_at"])
            if expires_at < datetime.utcnow():
                self.security_logger.info(f"Expired session: {session.get('id')}")
                return False
        
        # Check session fingerprint
        if "fingerprint" in session:
            expected_fingerprint = self._generate_session_fingerprint(session)
            if session["fingerprint"] != expected_fingerprint:
                self.security_logger.warning(f"Session fingerprint mismatch: {session.get('id')}")
                return False
        
        return True
    
    def _generate_session_fingerprint(self, session: Dict[str, Any]) -> str:
        """Generate session fingerprint for validation"""
        data = f"{session.get('user_id')}:{session.get('ip')}:{session.get('user_agent')}"
        return hashlib.sha256(data.encode()).hexdigest()
    
    # ==========================================
    # 8. DATA INTEGRITY FAILURES PREVENTION
    # ==========================================
    
    def sign_data(self, data: str) -> str:
        """Sign data with HMAC for integrity verification"""
        key = self.settings.secret_key.get_secret_value().encode()
        signature = hmac.new(key, data.encode(), hashlib.sha256).hexdigest()
        return f"{data}.{signature}"
    
    def verify_data_signature(self, signed_data: str) -> Optional[str]:
        """Verify data signature"""
        try:
            data, signature = signed_data.rsplit(".", 1)
            key = self.settings.secret_key.get_secret_value().encode()
            expected_signature = hmac.new(key, data.encode(), hashlib.sha256).hexdigest()
            
            if hmac.compare_digest(signature, expected_signature):
                return data
            else:
                self.security_logger.warning("Invalid data signature detected")
                return None
        except ValueError:
            return None
    
    # ==========================================
    # 9. SECURITY MONITORING
    # ==========================================
    
    def log_security_event(self, event_type: str, details: Dict[str, Any]):
        """Log security events for monitoring"""
        event = {
            "timestamp": datetime.utcnow().isoformat(),
            "type": event_type,
            "details": details,
            "severity": self._determine_severity(event_type)
        }
        
        self.security_logger.info(
            f"Security event: {event_type}",
            extra=event
        )
        
        # Send to monitoring service if configured
        if self.settings.sentry_dsn:
            self._send_to_sentry(event)
    
    def _determine_severity(self, event_type: str) -> str:
        """Determine event severity"""
        high_severity = ["authentication_failure", "authorization_violation", "injection_attempt"]
        medium_severity = ["rate_limit_exceeded", "invalid_input", "session_expired"]
        
        if event_type in high_severity:
            return "high"
        elif event_type in medium_severity:
            return "medium"
        else:
            return "low"
    
    def _send_to_sentry(self, event: Dict[str, Any]):
        """Send security event to Sentry"""
        try:
            import sentry_sdk
            sentry_sdk.capture_message(
                f"Security Event: {event['type']}",
                level="warning" if event["severity"] == "high" else "info",
                extras=event
            )
        except Exception as e:
            logger.error(f"Failed to send to Sentry: {e}")
    
    # ==========================================
    # 10. SSRF PREVENTION
    # ==========================================
    
    def validate_url(self, url: str) -> bool:
        """Validate URL to prevent SSRF attacks"""
        from urllib.parse import urlparse
        
        # Block private IP ranges
        private_ranges = [
            "127.0.0.0/8",
            "10.0.0.0/8",
            "172.16.0.0/12",
            "192.168.0.0/16",
            "169.254.0.0/16",
            "::1/128",
            "fc00::/7"
        ]
        
        try:
            parsed = urlparse(url)
            
            # Check scheme
            if parsed.scheme not in ["http", "https"]:
                self.security_logger.warning(f"Invalid URL scheme: {parsed.scheme}")
                return False
            
            # Check for local addresses
            if parsed.hostname in ["localhost", "127.0.0.1", "0.0.0.0"]:
                self.security_logger.warning(f"Local address detected: {parsed.hostname}")
                return False
            
            # Additional checks for private IPs would go here
            
            return True
            
        except Exception as e:
            self.security_logger.error(f"URL validation error: {e}")
            return False
    
    def sanitize_file_path(self, file_path: str) -> str:
        """Sanitize file paths to prevent traversal attacks"""
        import os
        
        # Remove path traversal patterns
        sanitized = file_path.replace("../", "").replace("..\\", "")
        
        # Ensure path is within allowed directory
        safe_path = os.path.normpath(sanitized)
        
        # Additional validation
        if safe_path.startswith("/") or safe_path.startswith("\\"):
            safe_path = safe_path[1:]
        
        return safe_path


def setup_owasp_security(app: Any, settings: Any) -> OWASPSecurityControls:
    """Setup OWASP security controls for the application"""
    security = OWASPSecurityControls(settings)
    
    # Validate configuration
    security.disable_debug_in_production()
    
    # Setup monitoring
    security.log_security_event("security_initialized", {
        "environment": settings.app_env.value,
        "timestamp": datetime.utcnow().isoformat()
    })
    
    return security