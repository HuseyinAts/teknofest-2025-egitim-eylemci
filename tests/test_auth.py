"""
Authentication Tests
TEKNOFEST 2025 - Eğitim Teknolojileri
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock
from jose import jwt, JWTError
from fastapi import HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.auth import (
    verify_password,
    get_password_hash,
    create_access_token,
    get_current_user,
    get_current_active_user,
    require_role,
    authenticate_user,
    create_user,
    TokenData,
    LoginRequest,
    LoginResponse,
    RegisterRequest,
    UserResponse
)
from src.database.models import User, UserRole
from src.config import get_settings


class TestPasswordHashing:
    """Test password hashing functions"""
    
    def test_get_password_hash(self):
        """Test password hashing"""
        password = "test_password_123"
        hashed = get_password_hash(password)
        
        assert hashed != password
        assert len(hashed) > 50
        assert hashed.startswith("$2b$")
    
    def test_verify_password_correct(self):
        """Test password verification with correct password"""
        password = "test_password_123"
        hashed = get_password_hash(password)
        
        assert verify_password(password, hashed) is True
    
    def test_verify_password_incorrect(self):
        """Test password verification with incorrect password"""
        password = "test_password_123"
        wrong_password = "wrong_password"
        hashed = get_password_hash(password)
        
        assert verify_password(wrong_password, hashed) is False
    
    def test_password_hash_uniqueness(self):
        """Test that same password produces different hashes"""
        password = "test_password_123"
        hash1 = get_password_hash(password)
        hash2 = get_password_hash(password)
        
        assert hash1 != hash2
        assert verify_password(password, hash1) is True
        assert verify_password(password, hash2) is True


class TestTokenCreation:
    """Test JWT token creation"""
    
    def test_create_access_token_default_expiry(self):
        """Test token creation with default expiry"""
        data = {"sub": "user123", "username": "testuser", "role": "student"}
        token = create_access_token(data)
        
        assert token is not None
        assert isinstance(token, str)
        assert len(token) > 100
    
    def test_create_access_token_custom_expiry(self):
        """Test token creation with custom expiry"""
        data = {"sub": "user123", "username": "testuser", "role": "teacher"}
        expires_delta = timedelta(hours=1)
        token = create_access_token(data, expires_delta)
        
        settings = get_settings()
        decoded = jwt.decode(
            token,
            settings.secret_key.get_secret_value(),
            algorithms=["HS256"]
        )
        
        assert decoded["sub"] == "user123"
        assert decoded["username"] == "testuser"
        assert decoded["role"] == "teacher"
        assert "exp" in decoded
    
    def test_token_expiry_time(self):
        """Test that token has correct expiry time"""
        data = {"sub": "user123"}
        expires_delta = timedelta(minutes=15)
        token = create_access_token(data, expires_delta)
        
        settings = get_settings()
        decoded = jwt.decode(
            token,
            settings.secret_key.get_secret_value(),
            algorithms=["HS256"]
        )
        
        exp_time = datetime.fromtimestamp(decoded["exp"])
        now = datetime.utcnow()
        time_diff = exp_time - now
        
        assert 14 <= time_diff.total_seconds() / 60 <= 16


class TestTokenData:
    """Test TokenData class"""
    
    def test_token_data_initialization(self):
        """Test TokenData initialization"""
        token_data = TokenData(
            user_id="123",
            username="testuser",
            role="student"
        )
        
        assert token_data.user_id == "123"
        assert token_data.username == "testuser"
        assert token_data.role == "student"


@pytest.mark.asyncio
class TestGetCurrentUser:
    """Test get_current_user function"""
    
    async def test_get_current_user_valid_token(self):
        """Test getting current user with valid token"""
        # Mock data
        user_id = "user123"
        username = "testuser"
        role = "student"
        
        # Create valid token
        token_data = {
            "sub": user_id,
            "username": username,
            "role": role
        }
        token = create_access_token(token_data)
        
        # Mock credentials
        mock_credentials = Mock()
        mock_credentials.credentials = token
        
        # Mock user
        mock_user = Mock(spec=User)
        mock_user.id = user_id
        mock_user.username = username
        mock_user.role = UserRole.STUDENT
        mock_user.is_active = True
        
        # Mock database
        mock_db = AsyncMock(spec=AsyncSession)
        mock_result = AsyncMock()
        mock_result.scalar_one_or_none.return_value = mock_user
        mock_db.execute.return_value = mock_result
        
        # Test
        user = await get_current_user(mock_credentials, mock_db)
        
        assert user.id == user_id
        assert user.username == username
        assert user.is_active is True
    
    async def test_get_current_user_invalid_token(self):
        """Test getting current user with invalid token"""
        # Mock invalid credentials
        mock_credentials = Mock()
        mock_credentials.credentials = "invalid_token"
        
        # Mock database
        mock_db = AsyncMock(spec=AsyncSession)
        
        # Test
        with pytest.raises(HTTPException) as exc_info:
            await get_current_user(mock_credentials, mock_db)
        
        assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED
        assert "Could not validate credentials" in str(exc_info.value.detail)
    
    async def test_get_current_user_expired_token(self):
        """Test getting current user with expired token"""
        # Create expired token
        token_data = {
            "sub": "user123",
            "username": "testuser"
        }
        expired_token = create_access_token(
            token_data,
            expires_delta=timedelta(seconds=-10)
        )
        
        # Mock credentials
        mock_credentials = Mock()
        mock_credentials.credentials = expired_token
        
        # Mock database
        mock_db = AsyncMock(spec=AsyncSession)
        
        # Test
        with pytest.raises(HTTPException) as exc_info:
            await get_current_user(mock_credentials, mock_db)
        
        assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED
    
    async def test_get_current_user_no_user_in_db(self):
        """Test getting current user when user doesn't exist in database"""
        # Create valid token
        token_data = {
            "sub": "nonexistent_user",
            "username": "testuser",
            "role": "student"
        }
        token = create_access_token(token_data)
        
        # Mock credentials
        mock_credentials = Mock()
        mock_credentials.credentials = token
        
        # Mock database - user not found
        mock_db = AsyncMock(spec=AsyncSession)
        mock_result = AsyncMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_db.execute.return_value = mock_result
        
        # Test
        with pytest.raises(HTTPException) as exc_info:
            await get_current_user(mock_credentials, mock_db)
        
        assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED
    
    async def test_get_current_user_inactive(self):
        """Test getting current user when user is inactive"""
        # Create valid token
        user_id = "user123"
        token_data = {
            "sub": user_id,
            "username": "testuser",
            "role": "student"
        }
        token = create_access_token(token_data)
        
        # Mock credentials
        mock_credentials = Mock()
        mock_credentials.credentials = token
        
        # Mock inactive user
        mock_user = Mock(spec=User)
        mock_user.id = user_id
        mock_user.is_active = False
        
        # Mock database
        mock_db = AsyncMock(spec=AsyncSession)
        mock_result = AsyncMock()
        mock_result.scalar_one_or_none.return_value = mock_user
        mock_db.execute.return_value = mock_result
        
        # Test
        with pytest.raises(HTTPException) as exc_info:
            await get_current_user(mock_credentials, mock_db)
        
        assert exc_info.value.status_code == status.HTTP_403_FORBIDDEN
        assert "Inactive user" in str(exc_info.value.detail)


@pytest.mark.asyncio
class TestGetCurrentActiveUser:
    """Test get_current_active_user function"""
    
    async def test_get_current_active_user_success(self):
        """Test getting current active user"""
        # Mock active user
        mock_user = Mock(spec=User)
        mock_user.is_active = True
        mock_user.username = "testuser"
        
        # Test
        user = await get_current_active_user(mock_user)
        
        assert user == mock_user
        assert user.is_active is True
    
    async def test_get_current_active_user_inactive(self):
        """Test getting current active user when user is inactive"""
        # Mock inactive user
        mock_user = Mock(spec=User)
        mock_user.is_active = False
        
        # Test
        with pytest.raises(HTTPException) as exc_info:
            await get_current_active_user(mock_user)
        
        assert exc_info.value.status_code == status.HTTP_403_FORBIDDEN
        assert "Inactive user" in str(exc_info.value.detail)


@pytest.mark.asyncio
class TestRequireRole:
    """Test require_role function"""
    
    async def test_require_role_authorized(self):
        """Test role requirement with authorized user"""
        # Mock user with admin role
        mock_user = Mock(spec=User)
        mock_user.role = UserRole.ADMIN
        mock_user.is_active = True
        
        # Create role checker
        role_checker = require_role([UserRole.ADMIN, UserRole.TEACHER])
        
        # Test
        with patch('src.api.auth.get_current_active_user', return_value=mock_user):
            user = await role_checker(mock_user)
        
        assert user == mock_user
    
    async def test_require_role_unauthorized(self):
        """Test role requirement with unauthorized user"""
        # Mock user with student role
        mock_user = Mock(spec=User)
        mock_user.role = UserRole.STUDENT
        mock_user.is_active = True
        
        # Create role checker for admin only
        role_checker = require_role([UserRole.ADMIN])
        
        # Test
        with pytest.raises(HTTPException) as exc_info:
            await role_checker(mock_user)
        
        assert exc_info.value.status_code == status.HTTP_403_FORBIDDEN
        assert "Insufficient permissions" in str(exc_info.value.detail)
    
    async def test_require_role_multiple_roles(self):
        """Test role requirement with multiple allowed roles"""
        # Mock teacher user
        mock_user = Mock(spec=User)
        mock_user.role = UserRole.TEACHER
        mock_user.is_active = True
        
        # Create role checker for admin and teacher
        role_checker = require_role([UserRole.ADMIN, UserRole.TEACHER])
        
        # Test
        with patch('src.api.auth.get_current_active_user', return_value=mock_user):
            user = await role_checker(mock_user)
        
        assert user == mock_user


@pytest.mark.asyncio
class TestAuthenticateUser:
    """Test authenticate_user function"""
    
    async def test_authenticate_user_success_with_username(self):
        """Test successful authentication with username"""
        # Mock user
        mock_user = Mock(spec=User)
        mock_user.username = "testuser"
        mock_user.hashed_password = get_password_hash("password123")
        mock_user.failed_login_attempts = 5
        
        # Mock database
        mock_db = AsyncMock(spec=AsyncSession)
        mock_result = AsyncMock()
        mock_result.scalar_one_or_none.return_value = mock_user
        mock_db.execute.return_value = mock_result
        mock_db.commit = AsyncMock()
        
        # Test
        user = await authenticate_user(mock_db, "testuser", "password123")
        
        assert user == mock_user
        assert user.failed_login_attempts == 0
        assert user.last_login_at is not None
        mock_db.commit.assert_called_once()
    
    async def test_authenticate_user_success_with_email(self):
        """Test successful authentication with email"""
        # Mock user
        mock_user = Mock(spec=User)
        mock_user.email = "test@example.com"
        mock_user.hashed_password = get_password_hash("password123")
        
        # Mock database
        mock_db = AsyncMock(spec=AsyncSession)
        mock_result = AsyncMock()
        mock_result.scalar_one_or_none.return_value = mock_user
        mock_db.execute.return_value = mock_result
        mock_db.commit = AsyncMock()
        
        # Test
        user = await authenticate_user(mock_db, "test@example.com", "password123")
        
        assert user == mock_user
        mock_db.commit.assert_called_once()
    
    async def test_authenticate_user_wrong_password(self):
        """Test authentication with wrong password"""
        # Mock user
        mock_user = Mock(spec=User)
        mock_user.hashed_password = get_password_hash("correct_password")
        
        # Mock database
        mock_db = AsyncMock(spec=AsyncSession)
        mock_result = AsyncMock()
        mock_result.scalar_one_or_none.return_value = mock_user
        mock_db.execute.return_value = mock_result
        
        # Test
        user = await authenticate_user(mock_db, "testuser", "wrong_password")
        
        assert user is None
    
    async def test_authenticate_user_not_found(self):
        """Test authentication when user doesn't exist"""
        # Mock database - user not found
        mock_db = AsyncMock(spec=AsyncSession)
        mock_result = AsyncMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_db.execute.return_value = mock_result
        
        # Test
        user = await authenticate_user(mock_db, "nonexistent", "password123")
        
        assert user is None


@pytest.mark.asyncio
class TestCreateUser:
    """Test create_user function"""
    
    async def test_create_user_success(self):
        """Test successful user creation"""
        # Mock database - no existing user
        mock_db = AsyncMock(spec=AsyncSession)
        mock_result = AsyncMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_db.execute.return_value = mock_result
        mock_db.add = Mock()
        mock_db.commit = AsyncMock()
        mock_db.refresh = AsyncMock()
        
        # Test
        user = await create_user(
            mock_db,
            username="newuser",
            email="new@example.com",
            password="password123",
            full_name="New User",
            role=UserRole.STUDENT
        )
        
        assert user.username == "newuser"
        assert user.email == "new@example.com"
        assert user.full_name == "New User"
        assert user.role == UserRole.STUDENT
        assert user.is_active is True
        assert user.is_verified is False
        assert user.points == 0
        assert user.level == 1
        mock_db.add.assert_called_once()
        mock_db.commit.assert_called_once()
    
    async def test_create_user_duplicate_username(self):
        """Test user creation with duplicate username"""
        # Mock existing user
        mock_existing_user = Mock(spec=User)
        mock_existing_user.username = "existinguser"
        
        # Mock database - user exists
        mock_db = AsyncMock(spec=AsyncSession)
        mock_result = AsyncMock()
        mock_result.scalar_one_or_none.return_value = mock_existing_user
        mock_db.execute.return_value = mock_result
        
        # Test
        with pytest.raises(HTTPException) as exc_info:
            await create_user(
                mock_db,
                username="existinguser",
                email="new@example.com",
                password="password123",
                full_name="New User"
            )
        
        assert exc_info.value.status_code == status.HTTP_400_BAD_REQUEST
        assert "already registered" in str(exc_info.value.detail)
    
    async def test_create_user_duplicate_email(self):
        """Test user creation with duplicate email"""
        # Mock existing user
        mock_existing_user = Mock(spec=User)
        mock_existing_user.email = "existing@example.com"
        
        # Mock database - user exists
        mock_db = AsyncMock(spec=AsyncSession)
        mock_result = AsyncMock()
        mock_result.scalar_one_or_none.return_value = mock_existing_user
        mock_db.execute.return_value = mock_result
        
        # Test
        with pytest.raises(HTTPException) as exc_info:
            await create_user(
                mock_db,
                username="newuser",
                email="existing@example.com",
                password="password123",
                full_name="New User"
            )
        
        assert exc_info.value.status_code == status.HTTP_400_BAD_REQUEST
        assert "already registered" in str(exc_info.value.detail)
    
    async def test_create_user_with_teacher_role(self):
        """Test creating user with teacher role"""
        # Mock database - no existing user
        mock_db = AsyncMock(spec=AsyncSession)
        mock_result = AsyncMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_db.execute.return_value = mock_result
        mock_db.add = Mock()
        mock_db.commit = AsyncMock()
        mock_db.refresh = AsyncMock()
        
        # Test
        user = await create_user(
            mock_db,
            username="teacher",
            email="teacher@example.com",
            password="password123",
            full_name="Teacher User",
            role=UserRole.TEACHER
        )
        
        assert user.role == UserRole.TEACHER


class TestPydanticModels:
    """Test Pydantic models"""
    
    def test_login_request_valid(self):
        """Test valid login request"""
        request = LoginRequest(
            username="testuser",
            password="password123"
        )
        
        assert request.username == "testuser"
        assert request.password == "password123"
    
    def test_login_request_short_password(self):
        """Test login request with short password"""
        with pytest.raises(Exception):
            LoginRequest(
                username="testuser",
                password="123"  # Too short
            )
    
    def test_login_response(self):
        """Test login response model"""
        response = LoginResponse(
            access_token="token123",
            user_id="user123",
            username="testuser",
            role="student"
        )
        
        assert response.access_token == "token123"
        assert response.token_type == "bearer"
        assert response.user_id == "user123"
        assert response.username == "testuser"
        assert response.role == "student"
    
    def test_register_request_valid(self):
        """Test valid registration request"""
        request = RegisterRequest(
            username="newuser",
            email="new@example.com",
            password="password123",
            full_name="New User"
        )
        
        assert request.username == "newuser"
        assert request.email == "new@example.com"
        assert request.password == "password123"
        assert request.full_name == "New User"
    
    def test_register_request_invalid_email(self):
        """Test registration request with invalid email"""
        with pytest.raises(Exception):
            RegisterRequest(
                username="newuser",
                email="invalid-email",  # Invalid email format
                password="password123",
                full_name="New User"
            )
    
    def test_user_response(self):
        """Test user response model"""
        response = UserResponse(
            id="user123",
            username="testuser",
            email="test@example.com",
            full_name="Test User",
            role="student",
            is_active=True,
            is_verified=False,
            points=100,
            level=2,
            streak_days=5,
            created_at=datetime.utcnow()
        )
        
        assert response.id == "user123"
        assert response.username == "testuser"
        assert response.email == "test@example.com"
        assert response.points == 100
        assert response.level == 2