"""
Production Rate Limiting Configuration
TEKNOFEST 2025 - Optimized rate limits for production
"""

from typing import Dict, Optional
from src.config import get_settings
from src.core.rate_limiter import RateLimiter
import redis
import logging

logger = logging.getLogger(__name__)
settings = get_settings()


# Production rate limit configurations per endpoint type
RATE_LIMITS = {
    # Authentication endpoints - strict limits
    "auth": {
        "login": "5/minute",  # 5 login attempts per minute
        "register": "3/minute",  # 3 registration attempts per minute
        "password_reset": "3/hour",  # 3 password reset requests per hour
        "refresh_token": "10/minute",  # 10 token refresh per minute
    },
    
    # AI/Model endpoints - resource intensive
    "ai": {
        "generate": "10/minute",  # 10 AI generation requests per minute
        "analyze": "20/minute",  # 20 analysis requests per minute
        "quiz": "30/minute",  # 30 quiz generation per minute
        "learning_path": "20/minute",  # 20 learning path requests per minute
    },
    
    # Read endpoints - more lenient
    "read": {
        "get_user": "100/minute",  # 100 user profile reads per minute
        "get_content": "200/minute",  # 200 content reads per minute
        "search": "50/minute",  # 50 search requests per minute
        "leaderboard": "60/minute",  # 60 leaderboard views per minute
    },
    
    # Write endpoints - moderate limits
    "write": {
        "update_profile": "20/minute",  # 20 profile updates per minute
        "submit_answer": "60/minute",  # 60 answer submissions per minute
        "create_content": "10/minute",  # 10 content creations per minute
        "upload_file": "5/minute",  # 5 file uploads per minute
    },
    
    # Admin endpoints - special handling
    "admin": {
        "default": "100/minute",  # Admin users get higher limits
        "bulk_operation": "5/minute",  # Bulk operations limited
    }
}

# Global rate limits (per IP)
GLOBAL_RATE_LIMITS = {
    "production": "1000/hour",  # 1000 requests per hour per IP
    "staging": "2000/hour",  # More lenient for staging
    "development": "10000/hour",  # Very lenient for development
}


def get_redis_client() -> Optional[redis.Redis]:
    """Get Redis client for distributed rate limiting"""
    if not settings.redis_url:
        logger.warning("Redis URL not configured, using in-memory rate limiting")
        return None
    
    try:
        client = redis.from_url(
            settings.redis_url,
            password=settings.redis_password.get_secret_value() if settings.redis_password else None,
            decode_responses=True,
            socket_connect_timeout=5,
            socket_keepalive=True,
            socket_keepalive_options={
                1: 1,  # TCP_KEEPIDLE
                2: 3,  # TCP_KEEPINTVL  
                3: 5,  # TCP_KEEPCNT
            }
        )
        # Test connection
        client.ping()
        logger.info("Redis connection established for rate limiting")
        return client
    except Exception as e:
        logger.error(f"Failed to connect to Redis: {e}")
        return None


def create_rate_limiter() -> RateLimiter:
    """Create production-ready rate limiter"""
    redis_client = get_redis_client() if settings.rate_limit_enabled else None
    
    # Determine default limits based on environment
    if settings.is_production():
        default_limit = 100
        default_window = 60
    elif settings.app_env == "staging":
        default_limit = 200
        default_window = 60
    else:
        default_limit = 1000
        default_window = 60
    
    rate_limiter = RateLimiter(
        redis_client=redis_client,
        algorithm="sliding_window",  # Best for production
        default_limit=default_limit,
        default_window=default_window,
        enable_stats=True
    )
    
    logger.info(f"Rate limiter initialized: enabled={settings.rate_limit_enabled}, "
                f"limits={default_limit}/{default_window}s, "
                f"redis={'connected' if redis_client else 'not available'}")
    
    return rate_limiter


def get_endpoint_rate_limit(endpoint_type: str, operation: str) -> str:
    """
    Get rate limit for specific endpoint.
    
    Args:
        endpoint_type: Type of endpoint (auth, ai, read, write, admin)
        operation: Specific operation within the endpoint type
    
    Returns:
        Rate limit string (e.g., "10/minute")
    """
    endpoint_limits = RATE_LIMITS.get(endpoint_type, {})
    return endpoint_limits.get(operation, f"{settings.rate_limit_requests}/{settings.rate_limit_period}s")


def apply_production_limits(app):
    """
    Apply production rate limits to FastAPI app.
    
    This should be called during app initialization in production.
    """
    if not settings.rate_limit_enabled:
        logger.warning("Rate limiting is disabled. Enable it for production!")
        return
    
    rate_limiter = create_rate_limiter()
    
    # Store rate limiter in app state for access in routes
    app.state.rate_limiter = rate_limiter
    
    # Log rate limit configuration
    logger.info("Production rate limits applied:")
    for endpoint_type, limits in RATE_LIMITS.items():
        for operation, limit in limits.items():
            logger.info(f"  {endpoint_type}.{operation}: {limit}")
    
    return rate_limiter


# Example usage in routes:
"""
from src.core.production_rate_limits import get_endpoint_rate_limit

@router.post("/login")
@app.state.rate_limiter.limit(get_endpoint_rate_limit("auth", "login"))
async def login(request: Request, credentials: LoginCredentials):
    # Login logic here
    pass
"""


# Monitoring function
def get_rate_limit_stats() -> Dict:
    """Get current rate limit statistics for monitoring"""
    if hasattr(app.state, 'rate_limiter'):
        limiter = app.state.rate_limiter
        return {
            "enabled": settings.rate_limit_enabled,
            "algorithm": limiter.algorithm,
            "stats": limiter.stats,
            "redis_connected": limiter.redis is not None
        }
    return {
        "enabled": False,
        "message": "Rate limiter not initialized"
    }