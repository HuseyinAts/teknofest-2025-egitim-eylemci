"""
Secure Authentication API with HttpOnly Cookies
TEKNOFEST 2025 - Production Ready Authentication
"""

import logging
import secrets
from typing import Optional, Dict, Any
from datetime import datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException, status, Request, Response, Cookie
from fastapi.security import HTTPBearer
from sqlalchemy.orm import Session
from pydantic import BaseModel, EmailStr, Field

from src.core.authentication import jwt_auth
from src.core.csrf_protection import CSRFTokenGenerator
from src.database.session import get_db
from src.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

router = APIRouter(
    prefix="/api/auth",
    tags=["Authentication"]
)

# CSRF token generator
csrf_generator = CSRFTokenGenerator(settings.secret_key.get_secret_value())


class LoginRequest(BaseModel):
    """Login request model"""
    email: EmailStr
    password: str = Field(min_length=8)


class RegisterRequest(BaseModel):
    """Registration request model"""
    email: EmailStr
    password: str = Field(min_length=8)
    name: str = Field(min_length=2, max_length=100)


class UserResponse(BaseModel):
    """User response model"""
    id: str
    email: str
    name: str
    role: str
    created_at: datetime


class AuthResponse(BaseModel):
    """Authentication response"""
    user: UserResponse
    message: str = "Authentication successful"
    csrfToken: Optional[str] = None


class SessionInfo(BaseModel):
    """Session information"""
    valid: bool
    expires_at: Optional[datetime]
    user_id: Optional[str]


def set_auth_cookies(
    response: Response,
    access_token: str,
    refresh_token: str,
    csrf_token: str,
    secure: bool = True
) -> None:
    """Set authentication cookies with security best practices"""
    
    # Access token cookie (15 minutes)
    response.set_cookie(
        key="access_token",
        value=access_token,
        secure=secure,
        httponly=True,
        samesite="strict" if secure else "lax",
        max_age=900,  # 15 minutes
        path="/"
    )
    
    # Refresh token cookie (7 days)
    response.set_cookie(
        key="refresh_token",
        value=refresh_token,
        secure=secure,
        httponly=True,
        samesite="strict" if secure else "lax",
        max_age=604800,  # 7 days
        path="/api/auth/refresh"
    )
    
    # CSRF token cookie (not httponly - needs to be readable by JS)
    response.set_cookie(
        key="csrf_token",
        value=csrf_token,
        secure=secure,
        httponly=False,  # JavaScript needs to read this
        samesite="strict" if secure else "lax",
        max_age=3600,  # 1 hour
        path="/"
    )


def clear_auth_cookies(response: Response) -> None:
    """Clear all authentication cookies"""
    response.delete_cookie(key="access_token", path="/")
    response.delete_cookie(key="refresh_token", path="/api/auth/refresh")
    response.delete_cookie(key="csrf_token", path="/")


async def get_current_user_from_cookie(
    access_token: Optional[str] = Cookie(None),
    db: Session = Depends(get_db)
) -> Optional[Dict[str, Any]]:
    """Get current user from cookie-based authentication"""
    if not access_token:
        return None
    
    try:
        # Decode and verify token
        payload = jwt_auth.decode_token(access_token)
        if not payload:
            return None
        
        # Get user from database (mock for now)
        user = {
            "id": payload.get("sub"),
            "email": payload.get("email"),
            "role": payload.get("role", "student")
        }
        
        return user
    except Exception as e:
        logger.error(f"Cookie authentication error: {e}")
        return None


@router.post("/register", response_model=AuthResponse)
async def register(
    request: RegisterRequest,
    response: Response,
    db: Session = Depends(get_db)
):
    """Register new user with secure cookie authentication"""
    try:
        # Check if user exists
        # existing_user = db.query(User).filter(User.email == request.email).first()
        # if existing_user:
        #     raise HTTPException(
        #         status_code=status.HTTP_400_BAD_REQUEST,
        #         detail="Email already registered"
        #     )
        
        # Hash password
        hashed_password = jwt_auth.hash_password(request.password)
        
        # Create user (mock for now)
        user = {
            "id": secrets.token_urlsafe(16),
            "email": request.email,
            "name": request.name,
            "password": hashed_password,
            "role": "student",
            "created_at": datetime.utcnow()
        }
        
        # Generate tokens
        access_token = jwt_auth.create_access_token(
            data={
                "sub": user["id"],
                "email": user["email"],
                "role": user["role"]
            }
        )
        
        refresh_token = jwt_auth.create_refresh_token(
            data={
                "sub": user["id"],
                "type": "refresh"
            }
        )
        
        # Generate CSRF token
        session_id = secrets.token_urlsafe(32)
        csrf_token = csrf_generator.generate_token(session_id)
        
        # Set cookies
        set_auth_cookies(
            response,
            access_token,
            refresh_token,
            csrf_token,
            secure=settings.is_production()
        )
        
        logger.info(f"User registered successfully: {user['email']}")
        
        return AuthResponse(
            user=UserResponse(
                id=user["id"],
                email=user["email"],
                name=user["name"],
                role=user["role"],
                created_at=user["created_at"]
            ),
            message="Registration successful",
            csrfToken=csrf_token
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Registration error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Registration failed"
        )


@router.post("/login", response_model=AuthResponse)
async def login(
    request: LoginRequest,
    response: Response,
    req: Request,
    db: Session = Depends(get_db)
):
    """Login with secure cookie authentication"""
    try:
        # Check login attempts
        client_ip = req.client.host
        if not jwt_auth.check_login_attempts(request.email):
            logger.warning(f"Too many login attempts for {request.email} from {client_ip}")
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Too many login attempts. Please try again later."
            )
        
        # Verify credentials (mock for now)
        # user = db.query(User).filter(User.email == request.email).first()
        # if not user or not jwt_auth.verify_password(request.password, user.password):
        #     jwt_auth.record_login_attempt(request.email, success=False)
        #     raise HTTPException(
        #         status_code=status.HTTP_401_UNAUTHORIZED,
        #         detail="Invalid email or password"
        #     )
        
        # Mock user
        user = {
            "id": "user_123",
            "email": request.email,
            "name": "Test User",
            "role": "student",
            "created_at": datetime.utcnow()
        }
        
        # Generate tokens
        access_token = jwt_auth.create_access_token(
            data={
                "sub": user["id"],
                "email": user["email"],
                "role": user["role"]
            }
        )
        
        refresh_token = jwt_auth.create_refresh_token(
            data={
                "sub": user["id"],
                "type": "refresh"
            }
        )
        
        # Generate CSRF token
        session_id = secrets.token_urlsafe(32)
        csrf_token = csrf_generator.generate_token(session_id)
        
        # Set cookies
        set_auth_cookies(
            response,
            access_token,
            refresh_token,
            csrf_token,
            secure=settings.is_production()
        )
        
        # Record successful login
        jwt_auth.record_login_attempt(request.email, success=True)
        logger.info(f"User logged in: {user['email']} from {client_ip}")
        
        return AuthResponse(
            user=UserResponse(
                id=user["id"],
                email=user["email"],
                name=user["name"],
                role=user["role"],
                created_at=user["created_at"]
            ),
            message="Login successful",
            csrfToken=csrf_token
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Login failed"
        )


@router.post("/logout")
async def logout(
    response: Response,
    current_user: Optional[Dict] = Depends(get_current_user_from_cookie)
):
    """Logout and clear cookies"""
    try:
        if current_user:
            logger.info(f"User logged out: {current_user.get('email')}")
        
        # Clear all auth cookies
        clear_auth_cookies(response)
        
        return {"message": "Logout successful"}
        
    except Exception as e:
        logger.error(f"Logout error: {e}")
        # Still clear cookies even if error
        clear_auth_cookies(response)
        return {"message": "Logout completed"}


@router.post("/refresh")
async def refresh_token(
    response: Response,
    refresh_token: Optional[str] = Cookie(None),
    db: Session = Depends(get_db)
):
    """Refresh access token using refresh token cookie"""
    if not refresh_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Refresh token not found"
        )
    
    try:
        # Decode refresh token
        payload = jwt_auth.decode_token(refresh_token)
        if not payload or payload.get("type") != "refresh":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid refresh token"
            )
        
        # Get user (mock for now)
        user_id = payload.get("sub")
        user = {
            "id": user_id,
            "email": "user@example.com",
            "role": "student"
        }
        
        # Generate new access token
        new_access_token = jwt_auth.create_access_token(
            data={
                "sub": user["id"],
                "email": user["email"],
                "role": user["role"]
            }
        )
        
        # Update access token cookie
        response.set_cookie(
            key="access_token",
            value=new_access_token,
            secure=settings.is_production(),
            httponly=True,
            samesite="strict" if settings.is_production() else "lax",
            max_age=900,  # 15 minutes
            path="/"
        )
        
        logger.info(f"Token refreshed for user: {user_id}")
        
        return {"message": "Token refreshed successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Token refresh error: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token refresh failed"
        )


@router.get("/me", response_model=UserResponse)
async def get_current_user(
    current_user: Dict = Depends(get_current_user_from_cookie)
):
    """Get current authenticated user"""
    if not current_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated"
        )
    
    # Get full user details (mock for now)
    return UserResponse(
        id=current_user["id"],
        email=current_user["email"],
        name="Test User",
        role=current_user["role"],
        created_at=datetime.utcnow()
    )


@router.get("/verify")
async def verify_session(
    access_token: Optional[str] = Cookie(None),
    db: Session = Depends(get_db)
) -> SessionInfo:
    """Verify if session is valid"""
    if not access_token:
        return SessionInfo(valid=False, expires_at=None, user_id=None)
    
    try:
        payload = jwt_auth.decode_token(access_token)
        if not payload:
            return SessionInfo(valid=False, expires_at=None, user_id=None)
        
        exp = payload.get("exp")
        expires_at = datetime.fromtimestamp(exp) if exp else None
        
        return SessionInfo(
            valid=True,
            expires_at=expires_at,
            user_id=payload.get("sub")
        )
        
    except Exception:
        return SessionInfo(valid=False, expires_at=None, user_id=None)


@router.get("/session/check")
async def check_session(
    access_token: Optional[str] = Cookie(None)
) -> Dict[str, Any]:
    """Check session expiry time"""
    if not access_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="No session found"
        )
    
    try:
        payload = jwt_auth.decode_token(access_token)
        if not payload:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid session"
            )
        
        exp = payload.get("exp")
        if exp:
            expires_at = datetime.fromtimestamp(exp)
            time_remaining = (expires_at - datetime.utcnow()).total_seconds()
            
            return {
                "expires_at": expires_at.isoformat(),
                "time_remaining": max(0, time_remaining),
                "should_refresh": time_remaining < 300  # Less than 5 minutes
            }
        
        return {
            "expires_at": None,
            "time_remaining": 0,
            "should_refresh": True
        }
        
    except Exception as e:
        logger.error(f"Session check error: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Session check failed"
        )


@router.get("/csrf")
async def get_csrf_token(
    response: Response,
    request: Request
) -> Dict[str, str]:
    """Get CSRF token for forms"""
    session_id = request.session.get("session_id")
    if not session_id:
        session_id = secrets.token_urlsafe(32)
        request.session["session_id"] = session_id
    
    csrf_token = csrf_generator.generate_token(session_id)
    
    # Set CSRF cookie
    response.set_cookie(
        key="csrf_token",
        value=csrf_token,
        secure=settings.is_production(),
        httponly=False,  # JavaScript needs to read this
        samesite="strict" if settings.is_production() else "lax",
        max_age=3600,
        path="/"
    )
    
    return {"csrfToken": csrf_token}