"""
JWT Authentication System
TEKNOFEST 2025 - Secure Authentication Layer
"""

from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import logging

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel, EmailStr, Field, validator
from sqlalchemy.orm import Session

from src.config import get_settings
from src.database.session import get_db
from src.core.security import PasswordManager, InputValidator

logger = logging.getLogger(__name__)

# Security instances
security = HTTPBearer()
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


class TokenData(BaseModel):
    """Token data model"""
    user_id: str
    email: Optional[str] = None
    roles: List[str] = Field(default_factory=list)
    exp: Optional[datetime] = None


class TokenResponse(BaseModel):
    """Token response model"""
    access_token: str
    refresh_token: Optional[str] = None
    token_type: str = "bearer"
    expires_in: int


class UserLogin(BaseModel):
    """User login request model"""
    email: EmailStr
    password: str
    
    @validator('password')
    def validate_password_format(cls, v):
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters")
        return v


class UserRegister(BaseModel):
    """User registration model"""
    email: EmailStr
    password: str
    confirm_password: str
    username: str
    full_name: Optional[str] = None
    
    @validator('username')
    def validate_username(cls, v):
        if not InputValidator.validate_username(v):
            raise ValueError("Username must be 3-20 characters, alphanumeric and underscore only")
        return v
    
    @validator('password')
    def validate_password_strength(cls, v):
        is_valid, message = InputValidator.validate_password(v)
        if not is_valid:
            raise ValueError(message)
        return v
    
    @validator('confirm_password')
    def passwords_match(cls, v, values):
        if 'password' in values and v != values['password']:
            raise ValueError("Passwords do not match")
        return v


class JWTAuthentication:
    """JWT Authentication handler"""
    
    def __init__(self, settings = None):
        self.settings = settings or get_settings()
        self.secret_key = self.settings.jwt_secret_key.get_secret_value()
        self.algorithm = self.settings.jwt_algorithm
        self.access_token_expire = self.settings.jwt_access_token_expire_minutes
        self.refresh_token_expire = self.settings.jwt_refresh_token_expire_days
        
        # Login attempt tracking
        self.login_attempts: Dict[str, List[datetime]] = {}
        self.blocked_users: Dict[str, datetime] = {}
    
    def create_access_token(
        self,
        user_id: str,
        email: str = None,
        roles: List[str] = None,
        additional_claims: Dict[str, Any] = None
    ) -> str:
        """Create JWT access token"""
        
        expires_delta = timedelta(minutes=self.access_token_expire)
        expire = datetime.utcnow() + expires_delta
        
        claims = {
            "sub": user_id,
            "email": email,
            "roles": roles or [],
            "exp": expire,
            "iat": datetime.utcnow(),
            "type": "access"
        }
        
        if additional_claims:
            claims.update(additional_claims)
        
        token = jwt.encode(claims, self.secret_key, algorithm=self.algorithm)
        
        logger.info(f"Access token created for user: {user_id}")
        return token
    
    def create_refresh_token(self, user_id: str) -> str:
        """Create JWT refresh token"""
        
        expires_delta = timedelta(days=self.refresh_token_expire)
        expire = datetime.utcnow() + expires_delta
        
        claims = {
            "sub": user_id,
            "exp": expire,
            "iat": datetime.utcnow(),
            "type": "refresh"
        }
        
        token = jwt.encode(claims, self.secret_key, algorithm=self.algorithm)
        
        logger.info(f"Refresh token created for user: {user_id}")
        return token
    
    def verify_token(self, token: str, token_type: str = "access") -> TokenData:
        """Verify and decode JWT token"""
        
        try:
            payload = jwt.decode(
                token,
                self.secret_key,
                algorithms=[self.algorithm]
            )
            
            # Verify token type
            if payload.get("type") != token_type:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid token type"
                )
            
            # Extract token data
            token_data = TokenData(
                user_id=payload.get("sub"),
                email=payload.get("email"),
                roles=payload.get("roles", []),
                exp=datetime.fromtimestamp(payload.get("exp"))
            )
            
            return token_data
            
        except JWTError as e:
            logger.error(f"Token verification failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired token"
            )
    
    def refresh_access_token(self, refresh_token: str) -> str:
        """Refresh access token using refresh token"""
        
        token_data = self.verify_token(refresh_token, token_type="refresh")
        
        # Create new access token
        new_access_token = self.create_access_token(
            user_id=token_data.user_id,
            email=token_data.email,
            roles=token_data.roles
        )
        
        return new_access_token
    
    def check_login_attempts(self, email: str) -> bool:
        """Check if user has exceeded login attempts"""
        
        # Check if user is blocked
        if email in self.blocked_users:
            if datetime.utcnow() < self.blocked_users[email]:
                return False
            else:
                # Unblock user
                del self.blocked_users[email]
                if email in self.login_attempts:
                    del self.login_attempts[email]
        
        # Clean old attempts (older than 1 hour)
        if email in self.login_attempts:
            cutoff = datetime.utcnow() - timedelta(hours=1)
            self.login_attempts[email] = [
                attempt for attempt in self.login_attempts[email]
                if attempt > cutoff
            ]
        
        return True
    
    def record_login_attempt(self, email: str, success: bool = False):
        """Record login attempt"""
        
        if success:
            # Clear attempts on successful login
            if email in self.login_attempts:
                del self.login_attempts[email]
            return
        
        # Record failed attempt
        if email not in self.login_attempts:
            self.login_attempts[email] = []
        
        self.login_attempts[email].append(datetime.utcnow())
        
        # Check if should block
        if len(self.login_attempts[email]) >= self.settings.security_max_login_attempts:
            # Block user for specified duration
            self.blocked_users[email] = datetime.utcnow() + timedelta(
                minutes=self.settings.security_lockout_duration_minutes
            )
            logger.warning(f"User {email} blocked due to too many login attempts")
    
    def hash_password(self, password: str) -> str:
        """Hash password using bcrypt"""
        return PasswordManager.hash_password(
            password,
            rounds=self.settings.security_bcrypt_rounds
        )
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify password against hash"""
        return PasswordManager.verify_password(plain_password, hashed_password)


# Dependency injection functions
jwt_auth = JWTAuthentication()


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db)
) -> TokenData:
    """Get current authenticated user from token"""
    
    token = credentials.credentials
    token_data = jwt_auth.verify_token(token)
    
    # Optional: Verify user still exists in database
    # user = db.query(User).filter(User.id == token_data.user_id).first()
    # if not user:
    #     raise HTTPException(
    #         status_code=status.HTTP_401_UNAUTHORIZED,
    #         detail="User not found"
    #     )
    
    return token_data


async def require_roles(required_roles: List[str]):
    """Require specific roles for access"""
    
    async def role_checker(
        current_user: TokenData = Depends(get_current_user)
    ) -> TokenData:
        
        if not any(role in current_user.roles for role in required_roles):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions"
            )
        
        return current_user
    
    return role_checker


async def get_optional_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> Optional[TokenData]:
    """Get current user if authenticated, None otherwise"""
    
    if not credentials:
        return None
    
    try:
        token = credentials.credentials
        token_data = jwt_auth.verify_token(token)
        return token_data
    except:
        return None


# Role-based access control decorators
def require_admin(current_user: TokenData = Depends(require_roles(["admin"]))):
    """Require admin role"""
    return current_user


def require_teacher(current_user: TokenData = Depends(require_roles(["teacher", "admin"]))):
    """Require teacher or admin role"""
    return current_user


def require_student(current_user: TokenData = Depends(require_roles(["student", "teacher", "admin"]))):
    """Require student, teacher or admin role"""
    return current_user
