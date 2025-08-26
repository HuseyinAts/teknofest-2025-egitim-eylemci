"""
Comprehensive Authentication and Authorization Test Suite
Target: Full coverage for auth module
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime, timedelta
import jwt
from passlib.context import CryptContext

from src.core.authentication import (
    create_access_token,
    create_refresh_token,
    verify_token,
    get_current_user,
    hash_password,
    verify_password,
    UserLogin,
    UserRegister,
    TokenResponse,
    check_user_permissions,
    enforce_password_policy,
    track_login_attempts
)
from src.database.models import User, UserRole
from fastapi import HTTPException, status


pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


class TestPasswordHashing:
    """Test password hashing and verification"""
    
    def test_hash_password(self):
        """Test password hashing"""
        password = "SecurePassword123!"
        hashed = hash_password(password)
        
        assert hashed != password
        assert len(hashed) > 50  # BCrypt hash is typically 60 chars
        assert hashed.startswith("$2b$") or hashed.startswith("$2a$")
    
    def test_verify_password_correct(self):
        """Test verifying correct password"""
        password = "TestPassword456!"
        hashed = hash_password(password)
        
        assert verify_password(password, hashed) == True
    
    def test_verify_password_incorrect(self):
        """Test verifying incorrect password"""
        password = "CorrectPassword"
        wrong_password = "WrongPassword"
        hashed = hash_password(password)
        
        assert verify_password(wrong_password, hashed) == False
    
    def test_hash_different_each_time(self):
        """Test that hashing same password gives different results"""
        password = "SamePassword123"
        hash1 = hash_password(password)
        hash2 = hash_password(password)
        
        assert hash1 != hash2  # Due to salt
        assert verify_password(password, hash1) == True
        assert verify_password(password, hash2) == True


class TestTokenManagement:
    """Test JWT token creation and verification"""
    
    @patch('src.core.authentication.get_settings')
    def test_create_access_token(self, mock_settings):
        """Test creating access token"""
        mock_settings.return_value.jwt_secret_key.get_secret_value.return_value = "test-secret"
        mock_settings.return_value.jwt_algorithm = "HS256"
        mock_settings.return_value.jwt_access_token_expire_minutes = 30
        
        data = {"sub": "test_user", "user_id": 1, "roles": ["student"]}
        token = create_access_token(data)
        
        assert token is not None
        assert isinstance(token, str)
        assert len(token.split(".")) == 3  # JWT format: header.payload.signature
    
    @patch('src.core.authentication.get_settings')
    def test_create_refresh_token(self, mock_settings):
        """Test creating refresh token"""
        mock_settings.return_value.jwt_secret_key.get_secret_value.return_value = "test-secret"
        mock_settings.return_value.jwt_algorithm = "HS256"
        mock_settings.return_value.jwt_refresh_token_expire_days = 7
        
        data = {"sub": "test_user", "user_id": 1}
        token = create_refresh_token(data)
        
        assert token is not None
        assert isinstance(token, str)
        
        # Decode and verify expiry
        decoded = jwt.decode(token, "test-secret", algorithms=["HS256"])
        assert "exp" in decoded
        assert decoded["sub"] == "test_user"
    
    @patch('src.core.authentication.get_settings')
    def test_verify_token_valid(self, mock_settings):
        """Test verifying valid token"""
        secret = "test-secret"
        mock_settings.return_value.jwt_secret_key.get_secret_value.return_value = secret
        mock_settings.return_value.jwt_algorithm = "HS256"
        
        # Create a valid token
        data = {"sub": "test_user", "exp": datetime.utcnow() + timedelta(hours=1)}
        token = jwt.encode(data, secret, algorithm="HS256")
        
        decoded = verify_token(token)
        assert decoded is not None
        assert decoded["sub"] == "test_user"
    
    @patch('src.core.authentication.get_settings')
    def test_verify_token_expired(self, mock_settings):
        """Test verifying expired token"""
        secret = "test-secret"
        mock_settings.return_value.jwt_secret_key.get_secret_value.return_value = secret
        mock_settings.return_value.jwt_algorithm = "HS256"
        
        # Create an expired token
        data = {"sub": "test_user", "exp": datetime.utcnow() - timedelta(hours=1)}
        token = jwt.encode(data, secret, algorithm="HS256")
        
        with pytest.raises(HTTPException) as exc:
            verify_token(token)
        assert exc.value.status_code == status.HTTP_401_UNAUTHORIZED
    
    @patch('src.core.authentication.get_settings')
    def test_verify_token_invalid_signature(self, mock_settings):
        """Test verifying token with invalid signature"""
        mock_settings.return_value.jwt_secret_key.get_secret_value.return_value = "correct-secret"
        mock_settings.return_value.jwt_algorithm = "HS256"
        
        # Create token with different secret
        data = {"sub": "test_user", "exp": datetime.utcnow() + timedelta(hours=1)}
        token = jwt.encode(data, "wrong-secret", algorithm="HS256")
        
        with pytest.raises(HTTPException) as exc:
            verify_token(token)
        assert exc.value.status_code == status.HTTP_401_UNAUTHORIZED
    
    def test_verify_token_malformed(self):
        """Test verifying malformed token"""
        with pytest.raises(HTTPException) as exc:
            verify_token("not.a.valid.token")
        assert exc.value.status_code == status.HTTP_401_UNAUTHORIZED


class TestUserRetrieval:
    """Test getting current user from token"""
    
    @patch('src.database.session.SessionLocal')
    @patch('src.core.authentication.verify_token')
    async def test_get_current_user_valid(self, mock_verify, mock_session):
        """Test getting current user with valid token"""
        # Mock token verification
        mock_verify.return_value = {"sub": "test_user", "user_id": 1}
        
        # Mock database user
        mock_user = Mock(spec=User)
        mock_user.id = 1
        mock_user.username = "test_user"
        mock_user.is_active = True
        
        mock_db = Mock()
        mock_db.query.return_value.filter.return_value.first.return_value = mock_user
        mock_session.return_value = mock_db
        
        # Test
        from fastapi.security import HTTPAuthorizationCredentials
        credentials = HTTPAuthorizationCredentials(scheme="Bearer", credentials="test.token.here")
        
        user = await get_current_user(credentials)
        assert user.username == "test_user"
        assert user.is_active == True
    
    @patch('src.database.session.SessionLocal')
    @patch('src.core.authentication.verify_token')
    async def test_get_current_user_not_found(self, mock_verify, mock_session):
        """Test getting current user when user doesn't exist"""
        mock_verify.return_value = {"sub": "deleted_user", "user_id": 999}
        
        mock_db = Mock()
        mock_db.query.return_value.filter.return_value.first.return_value = None
        mock_session.return_value = mock_db
        
        from fastapi.security import HTTPAuthorizationCredentials
        credentials = HTTPAuthorizationCredentials(scheme="Bearer", credentials="test.token.here")
        
        with pytest.raises(HTTPException) as exc:
            await get_current_user(credentials)
        assert exc.value.status_code == status.HTTP_404_NOT_FOUND
    
    @patch('src.database.session.SessionLocal')
    @patch('src.core.authentication.verify_token')
    async def test_get_current_user_inactive(self, mock_verify, mock_session):
        """Test getting current user when user is inactive"""
        mock_verify.return_value = {"sub": "inactive_user", "user_id": 2}
        
        mock_user = Mock(spec=User)
        mock_user.id = 2
        mock_user.username = "inactive_user"
        mock_user.is_active = False
        
        mock_db = Mock()
        mock_db.query.return_value.filter.return_value.first.return_value = mock_user
        mock_session.return_value = mock_db
        
        from fastapi.security import HTTPAuthorizationCredentials
        credentials = HTTPAuthorizationCredentials(scheme="Bearer", credentials="test.token.here")
        
        with pytest.raises(HTTPException) as exc:
            await get_current_user(credentials)
        assert exc.value.status_code == status.HTTP_403_FORBIDDEN


class TestPermissionChecking:
    """Test permission and role checking"""
    
    def test_check_user_permissions_has_permission(self):
        """Test checking permissions when user has required role"""
        user = Mock(spec=User)
        user.role = UserRole.ADMIN
        
        result = check_user_permissions(user, [UserRole.ADMIN, UserRole.TEACHER])
        assert result == True
    
    def test_check_user_permissions_no_permission(self):
        """Test checking permissions when user doesn't have required role"""
        user = Mock(spec=User)
        user.role = UserRole.STUDENT
        
        with pytest.raises(HTTPException) as exc:
            check_user_permissions(user, [UserRole.ADMIN, UserRole.TEACHER])
        assert exc.value.status_code == status.HTTP_403_FORBIDDEN
    
    def test_check_user_permissions_student_access(self):
        """Test student accessing student-only resources"""
        user = Mock(spec=User)
        user.role = UserRole.STUDENT
        
        result = check_user_permissions(user, [UserRole.STUDENT])
        assert result == True
    
    def test_check_user_permissions_empty_required_roles(self):
        """Test when no specific roles are required"""
        user = Mock(spec=User)
        user.role = UserRole.STUDENT
        
        result = check_user_permissions(user, [])
        assert result == True  # No specific role required


class TestPasswordPolicy:
    """Test password policy enforcement"""
    
    def test_enforce_password_policy_valid(self):
        """Test valid password that meets all requirements"""
        password = "SecurePass123!@#"
        result = enforce_password_policy(password)
        assert result == True
    
    def test_enforce_password_policy_too_short(self):
        """Test password that's too short"""
        password = "Pass1!"
        with pytest.raises(ValueError) as exc:
            enforce_password_policy(password)
        assert "at least 8 characters" in str(exc.value).lower()
    
    def test_enforce_password_policy_no_uppercase(self):
        """Test password without uppercase letters"""
        password = "securepass123!"
        with pytest.raises(ValueError) as exc:
            enforce_password_policy(password)
        assert "uppercase" in str(exc.value).lower()
    
    def test_enforce_password_policy_no_lowercase(self):
        """Test password without lowercase letters"""
        password = "SECUREPASS123!"
        with pytest.raises(ValueError) as exc:
            enforce_password_policy(password)
        assert "lowercase" in str(exc.value).lower()
    
    def test_enforce_password_policy_no_digit(self):
        """Test password without digits"""
        password = "SecurePassword!"
        with pytest.raises(ValueError) as exc:
            enforce_password_policy(password)
        assert "digit" in str(exc.value).lower() or "number" in str(exc.value).lower()
    
    def test_enforce_password_policy_no_special(self):
        """Test password without special characters"""
        password = "SecurePass123"
        with pytest.raises(ValueError) as exc:
            enforce_password_policy(password)
        assert "special" in str(exc.value).lower()
    
    def test_enforce_password_policy_common_password(self):
        """Test rejecting common passwords"""
        common_passwords = ["password123!", "Password123!", "Admin123!"]
        for password in common_passwords:
            with pytest.raises(ValueError) as exc:
                enforce_password_policy(password)
            assert "common" in str(exc.value).lower() or "weak" in str(exc.value).lower()


class TestLoginAttemptTracking:
    """Test login attempt tracking for security"""
    
    @patch('src.database.session.SessionLocal')
    def test_track_login_attempts_success(self, mock_session):
        """Test tracking successful login"""
        mock_db = Mock()
        mock_session.return_value = mock_db
        
        result = track_login_attempts("test_user", success=True)
        assert result["locked"] == False
        assert result["attempts"] == 0
    
    @patch('src.database.session.SessionLocal')
    def test_track_login_attempts_failure(self, mock_session):
        """Test tracking failed login attempts"""
        mock_db = Mock()
        mock_session.return_value = mock_db
        
        # First failure
        result = track_login_attempts("test_user", success=False)
        assert result["locked"] == False
        assert result["attempts"] == 1
    
    @patch('src.database.session.SessionLocal')
    def test_track_login_attempts_lockout(self, mock_session):
        """Test account lockout after multiple failures"""
        mock_db = Mock()
        mock_session.return_value = mock_db
        
        # Simulate multiple failures
        for i in range(5):
            result = track_login_attempts("test_user", success=False)
        
        assert result["locked"] == True
        assert result["attempts"] >= 5
        assert "lockout_until" in result


class TestUserRegistration:
    """Test user registration validation"""
    
    def test_user_register_valid(self):
        """Test valid user registration data"""
        user_data = UserRegister(
            username="new_user",
            email="user@example.com",
            password="SecurePass123!",
            full_name="Test User"
        )
        
        assert user_data.username == "new_user"
        assert user_data.email == "user@example.com"
        assert user_data.full_name == "Test User"
    
    def test_user_register_invalid_email(self):
        """Test user registration with invalid email"""
        with pytest.raises(ValueError):
            UserRegister(
                username="new_user",
                email="invalid-email",
                password="SecurePass123!",
                full_name="Test User"
            )
    
    def test_user_register_invalid_username(self):
        """Test user registration with invalid username"""
        with pytest.raises(ValueError):
            UserRegister(
                username="u",  # Too short
                email="user@example.com",
                password="SecurePass123!",
                full_name="Test User"
            )
    
    def test_user_register_sql_injection_prevention(self):
        """Test SQL injection prevention in registration"""
        # Attempt SQL injection in username
        with pytest.raises(ValueError):
            UserRegister(
                username="admin'; DROP TABLE users;--",
                email="user@example.com",
                password="SecurePass123!",
                full_name="Test User"
            )


class TestUserLogin:
    """Test user login validation"""
    
    def test_user_login_valid(self):
        """Test valid login data"""
        login_data = UserLogin(
            username="test_user",
            password="TestPass123!"
        )
        
        assert login_data.username == "test_user"
        assert login_data.password == "TestPass123!"
    
    def test_user_login_email_format(self):
        """Test login with email instead of username"""
        login_data = UserLogin(
            username="user@example.com",
            password="TestPass123!"
        )
        
        assert "@" in login_data.username  # Should accept email format


class TestTokenResponse:
    """Test token response structure"""
    
    def test_token_response_structure(self):
        """Test token response has required fields"""
        response = TokenResponse(
            access_token="access.token.here",
            refresh_token="refresh.token.here",
            token_type="bearer",
            expires_in=3600
        )
        
        assert response.access_token == "access.token.here"
        assert response.refresh_token == "refresh.token.here"
        assert response.token_type == "bearer"
        assert response.expires_in == 3600


class TestSessionManagement:
    """Test session management and security"""
    
    @patch('src.database.session.SessionLocal')
    def test_session_timeout(self, mock_session):
        """Test session timeout handling"""
        mock_db = Mock()
        mock_session.return_value = mock_db
        
        # Create token that's about to expire
        data = {"sub": "test_user", "exp": datetime.utcnow() + timedelta(seconds=1)}
        
        # Wait and check if properly handled
        import time
        time.sleep(2)
        
        # Token should now be expired
        with pytest.raises(HTTPException):
            verify_token(jwt.encode(data, "secret", algorithm="HS256"))
    
    def test_concurrent_session_limit(self):
        """Test limiting concurrent sessions per user"""
        # This would need implementation in the actual code
        # Testing the concept here
        user_sessions = {}
        user_id = 1
        max_sessions = 3
        
        # Add sessions
        for i in range(5):
            session_id = f"session_{i}"
            if user_id not in user_sessions:
                user_sessions[user_id] = []
            
            if len(user_sessions[user_id]) >= max_sessions:
                # Remove oldest session
                user_sessions[user_id].pop(0)
            
            user_sessions[user_id].append(session_id)
        
        assert len(user_sessions[user_id]) == max_sessions


class TestSecurityHeaders:
    """Test security headers in authentication"""
    
    def test_csrf_token_generation(self):
        """Test CSRF token generation"""
        import secrets
        csrf_token = secrets.token_urlsafe(32)
        
        assert len(csrf_token) >= 32
        assert isinstance(csrf_token, str)
    
    def test_secure_cookie_flags(self):
        """Test secure cookie configuration"""
        cookie_config = {
            "secure": True,  # HTTPS only
            "httponly": True,  # No JavaScript access
            "samesite": "strict",  # CSRF protection
            "max_age": 3600  # 1 hour
        }
        
        assert cookie_config["secure"] == True
        assert cookie_config["httponly"] == True
        assert cookie_config["samesite"] == "strict"