"""
Security Test Suite
TEKNOFEST 2025 - Comprehensive Security Testing
"""

import pytest
import time
from unittest.mock import Mock, patch
from datetime import datetime, timedelta

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.security import (
    SecurityHeaders,
    RateLimiter,
    SQLInjectionProtection,
    InputValidator,
    PasswordManager,
    SessionManager,
    CSRFProtection
)
from src.core.authentication import JWTAuthentication
from src.config import get_settings


class TestSecurityHeaders:
    """Test security headers"""
    
    def test_security_headers_applied(self):
        """Test that all security headers are applied"""
        mock_response = Mock()
        mock_response.headers = {}
        
        SecurityHeaders.apply(mock_response)
        
        assert "X-Content-Type-Options" in mock_response.headers
        assert mock_response.headers["X-Content-Type-Options"] == "nosniff"
        assert "X-Frame-Options" in mock_response.headers
        assert mock_response.headers["X-Frame-Options"] == "DENY"
        assert "Strict-Transport-Security" in mock_response.headers


class TestRateLimiter:
    """Test rate limiting functionality"""
    
    def test_rate_limit_allows_normal_traffic(self):
        """Test that normal traffic is allowed"""
        limiter = RateLimiter(max_requests=5, window_seconds=10)
        mock_request = Mock()
        mock_request.client = Mock(host="127.0.0.1")
        mock_request.headers = {}
        
        # Should allow 5 requests
        for _ in range(5):
            assert limiter.check_rate_limit(mock_request) is True
        
        # 6th request should be blocked
        assert limiter.check_rate_limit(mock_request) is False
    
    def test_rate_limit_blocks_excessive_requests(self):
        """Test that excessive requests are blocked"""
        limiter = RateLimiter(max_requests=3, window_seconds=10)
        mock_request = Mock()
        mock_request.client = Mock(host="192.168.1.1")
        mock_request.headers = {}
        
        # Exhaust limit
        for _ in range(3):
            limiter.check_rate_limit(mock_request)
        
        # Should be blocked
        assert limiter.check_rate_limit(mock_request) is False
        assert limiter.is_blocked("192.168.1.1") is True
    
    def test_rate_limit_different_ips(self):
        """Test that rate limits are per IP"""
        limiter = RateLimiter(max_requests=2, window_seconds=10)
        
        # First IP
        request1 = Mock()
        request1.client = Mock(host="192.168.1.1")
        request1.headers = {}
        
        # Second IP
        request2 = Mock()
        request2.client = Mock(host="192.168.1.2")
        request2.headers = {}
        
        # Both should be allowed their quota
        assert limiter.check_rate_limit(request1) is True
        assert limiter.check_rate_limit(request1) is True
        assert limiter.check_rate_limit(request1) is False  # Blocked
        
        assert limiter.check_rate_limit(request2) is True  # Different IP still allowed
        assert limiter.check_rate_limit(request2) is True


class TestSQLInjectionProtection:
    """Test SQL injection protection"""
    
    @pytest.mark.parametrize("dangerous_input", [
        "'; DROP TABLE users; --",
        "1 OR 1=1",
        "admin' --",
        "1; DELETE FROM users WHERE 1=1",
        "' UNION SELECT * FROM passwords --",
        "Robert'); DROP TABLE students;--",
        "' OR '1'='1",
        "admin'/*",
        "' or 1=1#",
        "' or 1=1--",
        "' or 1=1/*",
        "'; EXEC sp_MSForEachTable 'DROP TABLE ?'; --",
        "' WAITFOR DELAY '00:00:10' --",
        "'; BENCHMARK(1000000,MD5('test')); --",
    ])
    def test_sql_injection_detection(self, dangerous_input):
        """Test detection of SQL injection attempts"""
        assert SQLInjectionProtection.validate_input(dangerous_input) is False
    
    @pytest.mark.parametrize("safe_input", [
        "John Doe",
        "user@example.com",
        "123456",
        "This is a normal comment",
        "O'Brien",  # Legitimate apostrophe
        "Price is $100",
        "Math equation: 2+2=4",
    ])
    def test_safe_input_allowed(self, safe_input):
        """Test that legitimate input is allowed"""
        assert SQLInjectionProtection.validate_input(safe_input) is True
    
    def test_input_sanitization(self):
        """Test input sanitization"""
        dangerous = "'; DROP TABLE users; --"
        sanitized = SQLInjectionProtection.sanitize_input(dangerous)
        assert "'" not in sanitized or "''" in sanitized  # Quotes should be escaped
    
    def test_parameterized_query_creation(self):
        """Test parameterized query creation"""
        query = "SELECT * FROM users WHERE email = :email AND age > :age"
        params = {"email": "user@example.com", "age": 18}
        
        stmt, safe_params = SQLInjectionProtection.create_parameterized_query(query, params)
        assert stmt is not None
        assert safe_params == params


class TestInputValidator:
    """Test input validation"""
    
    @pytest.mark.parametrize("valid_email", [
        "user@example.com",
        "john.doe@company.co.uk",
        "test123@test.org",
        "admin+tag@domain.com",
    ])
    def test_valid_email(self, valid_email):
        """Test valid email formats"""
        assert InputValidator.validate_email(valid_email) is True
    
    @pytest.mark.parametrize("invalid_email", [
        "notanemail",
        "@example.com",
        "user@",
        "user @example.com",
        "user@.com",
        "",
    ])
    def test_invalid_email(self, invalid_email):
        """Test invalid email formats"""
        assert InputValidator.validate_email(invalid_email) is False
    
    @pytest.mark.parametrize("valid_username", [
        "john_doe",
        "user123",
        "admin",
        "test_user_99",
    ])
    def test_valid_username(self, valid_username):
        """Test valid username formats"""
        assert InputValidator.validate_username(valid_username) is True
    
    @pytest.mark.parametrize("invalid_username", [
        "ab",  # Too short
        "a" * 21,  # Too long
        "user@name",  # Invalid character
        "user name",  # Space
        "user-name",  # Hyphen
        "",
    ])
    def test_invalid_username(self, invalid_username):
        """Test invalid username formats"""
        assert InputValidator.validate_username(invalid_username) is False
    
    def test_password_validation_strength(self):
        """Test password strength validation"""
        # Too short
        is_valid, message = InputValidator.validate_password("Short1!")
        assert is_valid is False
        assert "at least 8 characters" in message
        
        # No uppercase
        is_valid, message = InputValidator.validate_password("password123!")
        assert is_valid is False
        assert "uppercase" in message
        
        # No lowercase
        is_valid, message = InputValidator.validate_password("PASSWORD123!")
        assert is_valid is False
        assert "lowercase" in message
        
        # No number
        is_valid, message = InputValidator.validate_password("Password!")
        assert is_valid is False
        assert "number" in message
        
        # No special character
        is_valid, message = InputValidator.validate_password("Password123")
        assert is_valid is False
        assert "special character" in message
        
        # Valid password
        is_valid, message = InputValidator.validate_password("ValidPass123!")
        assert is_valid is True
    
    def test_html_sanitization(self):
        """Test HTML sanitization"""
        dangerous_html = '''
        <div>
            <script>alert('XSS')</script>
            <iframe src="evil.com"></iframe>
            <p onclick="stealCookies()">Click me</p>
            <embed src="virus.exe">
        </div>
        '''
        
        sanitized = InputValidator.sanitize_html(dangerous_html)
        
        assert "<script" not in sanitized
        assert "<iframe" not in sanitized
        assert "onclick" not in sanitized
        assert "<embed" not in sanitized


class TestPasswordManager:
    """Test password hashing and verification"""
    
    def test_password_hashing(self):
        """Test password hashing"""
        password = "TestPassword123!"
        hashed = PasswordManager.hash_password(password)
        
        assert hashed != password  # Should be hashed
        assert len(hashed) > 20  # Should be long enough
        assert hashed.startswith("$2b$")  # bcrypt format
    
    def test_password_verification(self):
        """Test password verification"""
        password = "SecurePassword456!"
        hashed = PasswordManager.hash_password(password)
        
        # Correct password should verify
        assert PasswordManager.verify_password(password, hashed) is True
        
        # Wrong password should not verify
        assert PasswordManager.verify_password("WrongPassword", hashed) is False
    
    def test_different_hashes_for_same_password(self):
        """Test that same password produces different hashes"""
        password = "SamePassword789!"
        hash1 = PasswordManager.hash_password(password)
        hash2 = PasswordManager.hash_password(password)
        
        assert hash1 != hash2  # Different salts should produce different hashes
        assert PasswordManager.verify_password(password, hash1) is True
        assert PasswordManager.verify_password(password, hash2) is True
    
    def test_secure_token_generation(self):
        """Test secure token generation"""
        token1 = PasswordManager.generate_secure_token()
        token2 = PasswordManager.generate_secure_token()
        
        assert len(token1) > 20  # Should be long enough
        assert token1 != token2  # Should be unique


class TestSessionManager:
    """Test session management"""
    
    def test_session_creation(self):
        """Test session creation"""
        manager = SessionManager("secret_key", lifetime_hours=1)
        session_id = manager.create_session("user123", {"role": "admin"})
        
        assert session_id is not None
        assert len(session_id) > 20
        assert "user123" in manager.sessions[session_id]["user_id"]
    
    def test_session_validation(self):
        """Test session validation"""
        manager = SessionManager("secret_key", lifetime_hours=1)
        session_id = manager.create_session("user456")
        
        # Valid session should return data
        session = manager.validate_session(session_id)
        assert session is not None
        assert session["user_id"] == "user456"
        
        # Invalid session should return None
        assert manager.validate_session("invalid_session") is None
    
    def test_session_expiration(self):
        """Test session expiration"""
        manager = SessionManager("secret_key", lifetime_hours=1)
        session_id = manager.create_session("user789")
        
        # Manually expire the session
        manager.sessions[session_id]["expires_at"] = datetime.utcnow() - timedelta(hours=1)
        
        # Expired session should return None
        assert manager.validate_session(session_id) is None
        assert session_id not in manager.sessions  # Should be deleted
    
    def test_session_destruction(self):
        """Test session destruction"""
        manager = SessionManager("secret_key")
        session_id = manager.create_session("user999")
        
        assert session_id in manager.sessions
        manager.destroy_session(session_id)
        assert session_id not in manager.sessions


class TestCSRFProtection:
    """Test CSRF protection"""
    
    def test_csrf_token_generation(self):
        """Test CSRF token generation"""
        csrf = CSRFProtection("secret_key")
        token = csrf.generate_token("session123")
        
        assert token is not None
        assert ":" in token
        assert len(token) > 30
    
    def test_csrf_token_validation(self):
        """Test CSRF token validation"""
        csrf = CSRFProtection("secret_key")
        session_id = "session456"
        token = csrf.generate_token(session_id)
        
        # Valid token should pass
        assert csrf.validate_token(token, session_id) is True
        
        # Wrong session should fail
        assert csrf.validate_token(token, "wrong_session") is False
        
        # Modified token should fail
        assert csrf.validate_token(token + "x", session_id) is False
    
    def test_csrf_token_expiration(self):
        """Test CSRF token expiration"""
        csrf = CSRFProtection("secret_key")
        session_id = "session789"
        
        # Generate token with old timestamp
        old_time = time.time() - 7200  # 2 hours ago
        message = f"{session_id}:{old_time}"
        import hmac
        import hashlib
        signature = hmac.new(
            "secret_key".encode(),
            message.encode(),
            hashlib.sha256
        ).hexdigest()
        old_token = f"{message}:{signature}"
        
        # Expired token should fail
        assert csrf.validate_token(old_token, session_id, max_age=3600) is False


class TestJWTAuthentication:
    """Test JWT authentication"""
    
    @pytest.fixture
    def jwt_auth(self):
        """Create JWT auth instance"""
        settings = get_settings()
        return JWTAuthentication(settings)
    
    def test_access_token_creation(self, jwt_auth):
        """Test access token creation"""
        token = jwt_auth.create_access_token(
            user_id="user123",
            email="user@example.com",
            roles=["student"]
        )
        
        assert token is not None
        assert len(token) > 50
        assert "." in token  # JWT format
    
    def test_refresh_token_creation(self, jwt_auth):
        """Test refresh token creation"""
        token = jwt_auth.create_refresh_token("user456")
        
        assert token is not None
        assert len(token) > 50
    
    def test_token_verification(self, jwt_auth):
        """Test token verification"""
        # Create token
        token = jwt_auth.create_access_token(
            user_id="user789",
            email="test@example.com",
            roles=["admin"]
        )
        
        # Verify token
        token_data = jwt_auth.verify_token(token)
        
        assert token_data.user_id == "user789"
        assert token_data.email == "test@example.com"
        assert "admin" in token_data.roles
    
    def test_invalid_token_rejection(self, jwt_auth):
        """Test that invalid tokens are rejected"""
        with pytest.raises(Exception):
            jwt_auth.verify_token("invalid.token.here")
    
    def test_token_refresh(self, jwt_auth):
        """Test token refresh functionality"""
        # Create refresh token
        refresh_token = jwt_auth.create_refresh_token("user999")
        
        # Refresh to get new access token
        new_access_token = jwt_auth.refresh_access_token(refresh_token)
        
        assert new_access_token is not None
        assert len(new_access_token) > 50
    
    def test_login_attempt_tracking(self, jwt_auth):
        """Test login attempt tracking"""
        email = "test@example.com"
        
        # Should allow initially
        assert jwt_auth.check_login_attempts(email) is True
        
        # Record multiple failed attempts
        for _ in range(5):
            jwt_auth.record_login_attempt(email, success=False)
        
        # Should be blocked after max attempts
        assert jwt_auth.check_login_attempts(email) is False
        
        # Successful login should clear attempts
        jwt_auth.record_login_attempt(email, success=True)
        
        # Should be unblocked after lockout duration
        # (Would need to wait or mock time to test this properly)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
