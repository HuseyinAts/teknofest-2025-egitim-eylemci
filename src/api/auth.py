"""
Authentication and Authorization Module
TEKNOFEST 2025 - Eğitim Teknolojileri

Production-ready authentication with JWT tokens.
"""

from typing import Optional, List
from datetime import datetime, timedelta
import logging

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from src.database.session import get_db
from src.database.models import User, UserRole
from src.config import get_settings

logger = logging.getLogger(__name__)

# Security settings
settings = get_settings()
SECRET_KEY = settings.secret_key or "your-secret-key-here"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Security scheme
security = HTTPBearer()


class TokenData:
    """Token data model"""
    def __init__(self, user_id: str, username: str, role: str):
        self.user_id = user_id
        self.username = username
        self.role = role


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify password against hash"""
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """Hash password"""
    return pwd_context.hash(password)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Create JWT access token"""
    to_encode = data.copy()
    
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: AsyncSession = Depends(get_db)
) -> User:
    """
    Get current authenticated user from JWT token.
    
    Args:
        credentials: Bearer token credentials
        db: Database session
    
    Returns:
        Current user object
    
    Raises:
        HTTPException: If authentication fails
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        # Decode JWT token
        payload = jwt.decode(
            credentials.credentials,
            SECRET_KEY,
            algorithms=[ALGORITHM]
        )
        user_id: str = payload.get("sub")
        username: str = payload.get("username")
        
        if user_id is None:
            raise credentials_exception
        
        token_data = TokenData(
            user_id=user_id,
            username=username,
            role=payload.get("role", "student")
        )
        
    except JWTError:
        raise credentials_exception
    
    # Get user from database
    result = await db.execute(
        select(User).where(User.id == token_data.user_id)
    )
    user = result.scalar_one_or_none()
    
    if user is None:
        raise credentials_exception
    
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Inactive user"
        )
    
    return user


async def get_current_active_user(
    current_user: User = Depends(get_current_user)
) -> User:
    """
    Get current active user.
    
    Args:
        current_user: Current user from token
    
    Returns:
        Active user object
    
    Raises:
        HTTPException: If user is inactive
    """
    if not current_user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Inactive user"
        )
    return current_user


def require_role(roles: List[UserRole]):
    """
    Dependency to require specific user roles.
    
    Args:
        roles: List of allowed roles
    
    Returns:
        Dependency function
    
    Example:
        @app.get("/admin")
        async def admin_only(
            user: User = Depends(require_role([UserRole.ADMIN]))
        ):
            return {"message": "Admin access granted"}
    """
    async def role_checker(
        current_user: User = Depends(get_current_active_user)
    ) -> User:
        if current_user.role not in roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Insufficient permissions. Required role: {roles}"
            )
        return current_user
    
    return role_checker


async def authenticate_user(
    db: AsyncSession,
    username: str,
    password: str
) -> Optional[User]:
    """
    Authenticate user with username and password.
    
    Args:
        db: Database session
        username: Username or email
        password: Plain text password
    
    Returns:
        User object if authentication successful, None otherwise
    """
    # Try to find user by username or email
    result = await db.execute(
        select(User).where(
            (User.username == username) | (User.email == username)
        )
    )
    user = result.scalar_one_or_none()
    
    if not user:
        return None
    
    if not verify_password(password, user.hashed_password):
        return None
    
    # Update last login
    user.last_login_at = datetime.utcnow()
    user.failed_login_attempts = 0
    await db.commit()
    
    return user


async def create_user(
    db: AsyncSession,
    username: str,
    email: str,
    password: str,
    full_name: str,
    role: UserRole = UserRole.STUDENT
) -> User:
    """
    Create a new user.
    
    Args:
        db: Database session
        username: Username
        email: Email address
        password: Plain text password
        full_name: Full name
        role: User role
    
    Returns:
        Created user object
    
    Raises:
        HTTPException: If user already exists
    """
    # Check if user exists
    result = await db.execute(
        select(User).where(
            (User.username == username) | (User.email == email)
        )
    )
    
    if result.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username or email already registered"
        )
    
    # Create new user
    hashed_password = get_password_hash(password)
    
    user = User(
        username=username,
        email=email,
        hashed_password=hashed_password,
        full_name=full_name,
        role=role,
        is_active=True,
        is_verified=False,
        points=0,
        level=1,
        streak_days=0
    )
    
    db.add(user)
    await db.commit()
    await db.refresh(user)
    
    return user


# Login endpoint models
from pydantic import BaseModel, EmailStr, Field


class LoginRequest(BaseModel):
    """Login request model"""
    username: str = Field(..., description="Username or email")
    password: str = Field(..., min_length=6)


class LoginResponse(BaseModel):
    """Login response model"""
    access_token: str
    token_type: str = "bearer"
    user_id: str
    username: str
    role: str


class RegisterRequest(BaseModel):
    """Registration request model"""
    username: str = Field(..., min_length=3, max_length=50)
    email: EmailStr
    password: str = Field(..., min_length=8)
    full_name: str = Field(..., min_length=2, max_length=100)


class UserResponse(BaseModel):
    """User response model"""
    id: str
    username: str
    email: str
    full_name: str
    role: str
    is_active: bool
    is_verified: bool
    points: int
    level: int
    streak_days: int
    created_at: datetime