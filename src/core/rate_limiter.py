"""
Rate Limiter for API Endpoints
TEKNOFEST 2025 - Eğitim Teknolojileri

Production-ready rate limiting with Redis backend and in-memory fallback.
"""

import time
import asyncio
from typing import Optional, Dict, Tuple, Any
from datetime import datetime, timedelta
from functools import wraps
import logging
import hashlib

from fastapi import HTTPException, Request, status
from redis import Redis
from redis.asyncio import Redis as AsyncRedis

logger = logging.getLogger(__name__)


class RateLimitExceeded(HTTPException):
    """Rate limit exceeded exception"""
    def __init__(self, retry_after: int = 60):
        super().__init__(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded",
            headers={"Retry-After": str(retry_after)}
        )


class RateLimiter:
    """
    Production-ready rate limiter with multiple algorithms.
    
    Supports:
    - Token bucket algorithm
    - Sliding window algorithm
    - Fixed window algorithm
    - Per-user and per-IP limiting
    - Redis backend with in-memory fallback
    """
    
    def __init__(
        self,
        redis_client: Optional[Redis] = None,
        algorithm: str = "sliding_window",
        default_limit: int = 100,
        default_window: int = 60,
        enable_stats: bool = True
    ):
        """
        Initialize rate limiter.
        
        Args:
            redis_client: Redis client for distributed rate limiting
            algorithm: Algorithm to use (sliding_window, token_bucket, fixed_window)
            default_limit: Default request limit
            default_window: Default window in seconds
            enable_stats: Enable statistics tracking
        """
        self.redis = redis_client
        self.algorithm = algorithm
        self.default_limit = default_limit
        self.default_window = default_window
        self.enable_stats = enable_stats
        
        # In-memory fallback storage
        self.memory_storage: Dict[str, Any] = {}
        
        # Statistics
        self.stats = {
            "allowed": 0,
            "denied": 0,
            "total": 0
        }
    
    def limit(
        self,
        rate: str,
        key_func: Optional[callable] = None,
        error_message: Optional[str] = None
    ):
        """
        Decorator for rate limiting endpoints.
        
        Args:
            rate: Rate limit string (e.g., "10/minute", "100/hour")
            key_func: Function to generate rate limit key
            error_message: Custom error message
        
        Example:
            @rate_limiter.limit("10/minute")
            async def my_endpoint(request: Request):
                return {"status": "ok"}
        """
        # Parse rate string
        limit, period = self._parse_rate(rate)
        
        def decorator(func):
            @wraps(func)
            async def wrapper(request: Request = None, *args, **kwargs):
                # Generate rate limit key
                if key_func:
                    key = key_func(request, *args, **kwargs)
                else:
                    key = self._default_key_func(request)
                
                # Check rate limit
                allowed, retry_after = await self.check_rate_limit(
                    key, limit, period
                )
                
                if not allowed:
                    if self.enable_stats:
                        self.stats["denied"] += 1
                    raise RateLimitExceeded(retry_after)
                
                if self.enable_stats:
                    self.stats["allowed"] += 1
                
                # Execute function
                return await func(request, *args, **kwargs)
            
            return wrapper
        return decorator
    
    async def check_rate_limit(
        self,
        key: str,
        limit: int,
        window: int
    ) -> Tuple[bool, int]:
        """
        Check if request is within rate limit.
        
        Args:
            key: Rate limit key
            limit: Request limit
            window: Time window in seconds
        
        Returns:
            Tuple of (allowed, retry_after_seconds)
        """
        if self.enable_stats:
            self.stats["total"] += 1
        
        if self.algorithm == "sliding_window":
            return await self._sliding_window_check(key, limit, window)
        elif self.algorithm == "token_bucket":
            return await self._token_bucket_check(key, limit, window)
        elif self.algorithm == "fixed_window":
            return await self._fixed_window_check(key, limit, window)
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")
    
    async def _sliding_window_check(
        self,
        key: str,
        limit: int,
        window: int
    ) -> Tuple[bool, int]:
        """
        Sliding window rate limiting algorithm.
        
        More accurate than fixed window, prevents burst at window boundaries.
        """
        now = time.time()
        window_start = now - window
        
        if self.redis:
            try:
                return await self._redis_sliding_window(key, limit, window, now, window_start)
            except Exception as e:
                logger.warning(f"Redis error, falling back to memory: {e}")
        
        # In-memory fallback
        return self._memory_sliding_window(key, limit, window, now, window_start)
    
    async def _redis_sliding_window(
        self,
        key: str,
        limit: int,
        window: int,
        now: float,
        window_start: float
    ) -> Tuple[bool, int]:
        """Redis-based sliding window implementation"""
        pipe = self.redis.pipeline()
        
        # Remove old entries
        pipe.zremrangebyscore(key, 0, window_start)
        
        # Count current window requests
        pipe.zcard(key)
        
        # Add current request
        pipe.zadd(key, {str(now): now})
        
        # Set expiry
        pipe.expire(key, window + 1)
        
        # Execute pipeline
        if isinstance(self.redis, AsyncRedis):
            results = await pipe.execute()
        else:
            results = await asyncio.get_event_loop().run_in_executor(
                None, pipe.execute
            )
        
        request_count = results[1]
        
        if request_count < limit:
            return True, 0
        else:
            # Calculate retry after
            if isinstance(self.redis, AsyncRedis):
                oldest = await self.redis.zrange(key, 0, 0, withscores=True)
            else:
                oldest = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: self.redis.zrange(key, 0, 0, withscores=True)
                )
            
            if oldest:
                oldest_time = oldest[0][1]
                retry_after = int(window - (now - oldest_time) + 1)
                return False, retry_after
            
            return False, window
    
    def _memory_sliding_window(
        self,
        key: str,
        limit: int,
        window: int,
        now: float,
        window_start: float
    ) -> Tuple[bool, int]:
        """In-memory sliding window implementation"""
        if key not in self.memory_storage:
            self.memory_storage[key] = []
        
        # Remove old entries
        self.memory_storage[key] = [
            t for t in self.memory_storage[key]
            if t > window_start
        ]
        
        # Check limit
        if len(self.memory_storage[key]) < limit:
            self.memory_storage[key].append(now)
            return True, 0
        else:
            # Calculate retry after
            oldest = min(self.memory_storage[key])
            retry_after = int(window - (now - oldest) + 1)
            return False, retry_after
    
    async def _token_bucket_check(
        self,
        key: str,
        limit: int,
        window: int
    ) -> Tuple[bool, int]:
        """
        Token bucket rate limiting algorithm.
        
        Allows bursts up to the limit, refills gradually.
        """
        now = time.time()
        refill_rate = limit / window  # Tokens per second
        
        if self.redis:
            try:
                return await self._redis_token_bucket(key, limit, refill_rate, now)
            except Exception as e:
                logger.warning(f"Redis error, falling back to memory: {e}")
        
        # In-memory fallback
        return self._memory_token_bucket(key, limit, refill_rate, now)
    
    async def _redis_token_bucket(
        self,
        key: str,
        limit: int,
        refill_rate: float,
        now: float
    ) -> Tuple[bool, int]:
        """Redis-based token bucket implementation"""
        bucket_key = f"bucket:{key}"
        
        # Get current bucket state
        if isinstance(self.redis, AsyncRedis):
            data = await self.redis.get(bucket_key)
        else:
            data = await asyncio.get_event_loop().run_in_executor(
                None, self.redis.get, bucket_key
            )
        
        if data:
            import json
            bucket = json.loads(data)
            tokens = bucket["tokens"]
            last_refill = bucket["last_refill"]
        else:
            tokens = limit
            last_refill = now
        
        # Calculate new tokens
        time_passed = now - last_refill
        tokens = min(limit, tokens + time_passed * refill_rate)
        
        if tokens >= 1:
            # Consume token
            tokens -= 1
            bucket = {"tokens": tokens, "last_refill": now}
            
            if isinstance(self.redis, AsyncRedis):
                await self.redis.setex(
                    bucket_key,
                    int(limit / refill_rate) + 1,
                    json.dumps(bucket)
                )
            else:
                await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.redis.setex(
                        bucket_key,
                        int(limit / refill_rate) + 1,
                        json.dumps(bucket)
                    )
                )
            
            return True, 0
        else:
            # Calculate retry after
            retry_after = int((1 - tokens) / refill_rate) + 1
            return False, retry_after
    
    def _memory_token_bucket(
        self,
        key: str,
        limit: int,
        refill_rate: float,
        now: float
    ) -> Tuple[bool, int]:
        """In-memory token bucket implementation"""
        bucket_key = f"bucket:{key}"
        
        if bucket_key not in self.memory_storage:
            self.memory_storage[bucket_key] = {
                "tokens": limit,
                "last_refill": now
            }
        
        bucket = self.memory_storage[bucket_key]
        
        # Calculate new tokens
        time_passed = now - bucket["last_refill"]
        bucket["tokens"] = min(limit, bucket["tokens"] + time_passed * refill_rate)
        bucket["last_refill"] = now
        
        if bucket["tokens"] >= 1:
            bucket["tokens"] -= 1
            return True, 0
        else:
            retry_after = int((1 - bucket["tokens"]) / refill_rate) + 1
            return False, retry_after
    
    async def _fixed_window_check(
        self,
        key: str,
        limit: int,
        window: int
    ) -> Tuple[bool, int]:
        """
        Fixed window rate limiting algorithm.
        
        Simple but can allow bursts at window boundaries.
        """
        now = int(time.time())
        window_key = f"{key}:{now // window}"
        
        if self.redis:
            try:
                if isinstance(self.redis, AsyncRedis):
                    count = await self.redis.incr(window_key)
                    if count == 1:
                        await self.redis.expire(window_key, window)
                else:
                    count = await asyncio.get_event_loop().run_in_executor(
                        None, self.redis.incr, window_key
                    )
                    if count == 1:
                        await asyncio.get_event_loop().run_in_executor(
                            None, self.redis.expire, window_key, window
                        )
                
                if count <= limit:
                    return True, 0
                else:
                    retry_after = window - (now % window)
                    return False, retry_after
                    
            except Exception as e:
                logger.warning(f"Redis error, falling back to memory: {e}")
        
        # In-memory fallback
        if window_key not in self.memory_storage:
            self.memory_storage[window_key] = 0
        
        self.memory_storage[window_key] += 1
        
        if self.memory_storage[window_key] <= limit:
            return True, 0
        else:
            retry_after = window - (now % window)
            return False, retry_after
    
    def _parse_rate(self, rate: str) -> Tuple[int, int]:
        """
        Parse rate string to limit and window.
        
        Examples:
            "10/minute" -> (10, 60)
            "100/hour" -> (100, 3600)
            "1000/day" -> (1000, 86400)
        """
        parts = rate.split("/")
        if len(parts) != 2:
            raise ValueError(f"Invalid rate format: {rate}")
        
        limit = int(parts[0])
        
        period_map = {
            "second": 1,
            "minute": 60,
            "hour": 3600,
            "day": 86400
        }
        
        period = parts[1].lower()
        if period in period_map:
            window = period_map[period]
        else:
            # Try to parse as seconds
            window = int(period)
        
        return limit, window
    
    def _default_key_func(self, request: Request) -> str:
        """
        Default key generation function.
        
        Uses user ID if authenticated, otherwise IP address.
        """
        if request is None:
            return "anonymous"
        
        # Try to get user ID from request
        if hasattr(request, "user") and request.user:
            if hasattr(request.user, "id"):
                return f"user:{request.user.id}"
            elif hasattr(request.user, "username"):
                return f"user:{request.user.username}"
        
        # Fall back to IP address
        client_ip = request.client.host if request.client else "unknown"
        return f"ip:{client_ip}"
    
    async def reset(self, key: Optional[str] = None):
        """
        Reset rate limit for a key or all keys.
        
        Args:
            key: Specific key to reset, or None for all
        """
        if key:
            if self.redis:
                try:
                    if isinstance(self.redis, AsyncRedis):
                        await self.redis.delete(key)
                    else:
                        await asyncio.get_event_loop().run_in_executor(
                            None, self.redis.delete, key
                        )
                except Exception as e:
                    logger.error(f"Failed to reset Redis key: {e}")
            
            # Clear from memory
            keys_to_delete = [k for k in self.memory_storage if k.startswith(key)]
            for k in keys_to_delete:
                del self.memory_storage[k]
        else:
            # Reset all
            if self.redis:
                try:
                    if isinstance(self.redis, AsyncRedis):
                        await self.redis.flushdb()
                    else:
                        await asyncio.get_event_loop().run_in_executor(
                            None, self.redis.flushdb
                        )
                except Exception as e:
                    logger.error(f"Failed to flush Redis: {e}")
            
            self.memory_storage.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get rate limiter statistics"""
        if not self.enable_stats:
            return {}
        
        total = self.stats["total"]
        if total > 0:
            allow_rate = (self.stats["allowed"] / total) * 100
            deny_rate = (self.stats["denied"] / total) * 100
        else:
            allow_rate = deny_rate = 0
        
        return {
            **self.stats,
            "allow_rate": round(allow_rate, 2),
            "deny_rate": round(deny_rate, 2),
            "algorithm": self.algorithm,
            "redis_connected": self.redis is not None,
            "memory_keys": len(self.memory_storage)
        }
    
    def reset_stats(self):
        """Reset statistics"""
        self.stats = {
            "allowed": 0,
            "denied": 0,
            "total": 0
        }


# Global rate limiter instance
_rate_limiter = None


def get_rate_limiter(
    redis_client: Optional[Redis] = None,
    **kwargs
) -> RateLimiter:
    """
    Get or create global rate limiter instance.
    
    Args:
        redis_client: Redis client
        **kwargs: Additional arguments for RateLimiter
    
    Returns:
        RateLimiter instance
    """
    global _rate_limiter
    
    if _rate_limiter is None:
        _rate_limiter = RateLimiter(redis_client=redis_client, **kwargs)
    
    return _rate_limiter