"""
Cache Manager for Production
TEKNOFEST 2025 - Eğitim Teknolojileri

Provides unified caching interface with Redis backend and fallback to in-memory cache.
"""

import json
import hashlib
import pickle
from typing import Any, Optional, Union, Dict
from datetime import timedelta
import asyncio
from functools import wraps
import logging

from redis import Redis, ConnectionError as RedisConnectionError
from redis.asyncio import Redis as AsyncRedis
import aiocache
from aiocache import caches
from aiocache.serializers import JsonSerializer, PickleSerializer

logger = logging.getLogger(__name__)


class CacheManager:
    """
    Production-ready cache manager with Redis and in-memory fallback.
    
    Features:
    - Redis primary cache
    - In-memory fallback cache
    - Automatic serialization/deserialization
    - TTL support
    - Cache invalidation
    - Performance monitoring
    - Decorator support for caching
    """
    
    def __init__(
        self,
        redis_client: Optional[Union[Redis, AsyncRedis]] = None,
        redis_url: Optional[str] = None,
        default_ttl: int = 300,  # 5 minutes default
        max_memory_items: int = 1000,
        enable_stats: bool = True
    ):
        """
        Initialize cache manager.
        
        Args:
            redis_client: Existing Redis client
            redis_url: Redis connection URL
            default_ttl: Default TTL in seconds
            max_memory_items: Max items in memory cache
            enable_stats: Enable statistics tracking
        """
        self.redis = redis_client
        self.redis_url = redis_url
        self.default_ttl = default_ttl
        self.enable_stats = enable_stats
        
        # Initialize Redis if URL provided
        if not self.redis and redis_url:
            try:
                self.redis = Redis.from_url(redis_url, decode_responses=False)
                self.redis.ping()
                logger.info("Redis cache connected successfully")
            except Exception as e:
                logger.warning(f"Redis connection failed, using in-memory cache: {e}")
                self.redis = None
        
        # Setup in-memory cache as fallback
        self._setup_memory_cache(max_memory_items)
        
        # Statistics
        self.stats = {
            "hits": 0,
            "misses": 0,
            "sets": 0,
            "deletes": 0,
            "errors": 0
        }
    
    def _setup_memory_cache(self, max_items: int):
        """Setup in-memory cache"""
        caches.set_config({
            'default': {
                'cache': "aiocache.SimpleMemoryCache",
                'serializer': {
                    'class': "aiocache.serializers.JsonSerializer"
                },
                'ttl': self.default_ttl,
                'max_items': max_items
            },
            'pickle': {
                'cache': "aiocache.SimpleMemoryCache",
                'serializer': {
                    'class': "aiocache.serializers.PickleSerializer"
                },
                'ttl': self.default_ttl,
                'max_items': max_items
            }
        })
        self.memory_cache = caches.get('default')
        self.pickle_cache = caches.get('pickle')
    
    def _generate_key(self, namespace: str, key: str) -> str:
        """Generate cache key with namespace"""
        return f"{namespace}:{key}"
    
    def _serialize(self, value: Any) -> bytes:
        """Serialize value for storage"""
        try:
            return json.dumps(value).encode('utf-8')
        except (TypeError, ValueError):
            # Fallback to pickle for complex objects
            return pickle.dumps(value)
    
    def _deserialize(self, data: bytes) -> Any:
        """Deserialize value from storage"""
        if data is None:
            return None
        
        try:
            return json.loads(data.decode('utf-8'))
        except (json.JSONDecodeError, UnicodeDecodeError):
            # Try pickle if JSON fails
            try:
                return pickle.loads(data)
            except:
                return None
    
    async def get(
        self,
        key: str,
        namespace: str = "default",
        default: Any = None
    ) -> Any:
        """
        Get value from cache.
        
        Args:
            key: Cache key
            namespace: Cache namespace
            default: Default value if not found
        
        Returns:
            Cached value or default
        """
        full_key = self._generate_key(namespace, key)
        
        try:
            # Try Redis first
            if self.redis:
                try:
                    data = await self._redis_get(full_key)
                    if data is not None:
                        self._record_hit()
                        return self._deserialize(data)
                except RedisConnectionError:
                    logger.warning("Redis connection error, falling back to memory cache")
                    self.redis = None
            
            # Fallback to memory cache
            value = await self.memory_cache.get(full_key)
            if value is not None:
                self._record_hit()
                return value
            
            self._record_miss()
            return default
            
        except Exception as e:
            logger.error(f"Cache get error: {e}")
            self._record_error()
            return default
    
    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        namespace: str = "default"
    ) -> bool:
        """
        Set value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: TTL in seconds (None for default)
            namespace: Cache namespace
        
        Returns:
            Success status
        """
        full_key = self._generate_key(namespace, key)
        ttl = ttl or self.default_ttl
        
        try:
            # Try Redis first
            if self.redis:
                try:
                    serialized = self._serialize(value)
                    await self._redis_set(full_key, serialized, ttl)
                    self._record_set()
                    return True
                except RedisConnectionError:
                    logger.warning("Redis connection error, falling back to memory cache")
                    self.redis = None
            
            # Fallback to memory cache
            await self.memory_cache.set(full_key, value, ttl=ttl)
            self._record_set()
            return True
            
        except Exception as e:
            logger.error(f"Cache set error: {e}")
            self._record_error()
            return False
    
    async def delete(
        self,
        key: str,
        namespace: str = "default"
    ) -> bool:
        """
        Delete value from cache.
        
        Args:
            key: Cache key
            namespace: Cache namespace
        
        Returns:
            Success status
        """
        full_key = self._generate_key(namespace, key)
        
        try:
            # Try Redis first
            if self.redis:
                try:
                    await self._redis_delete(full_key)
                    self._record_delete()
                except RedisConnectionError:
                    logger.warning("Redis connection error")
            
            # Also delete from memory cache
            await self.memory_cache.delete(full_key)
            return True
            
        except Exception as e:
            logger.error(f"Cache delete error: {e}")
            self._record_error()
            return False
    
    async def clear(self, namespace: Optional[str] = None) -> bool:
        """
        Clear cache.
        
        Args:
            namespace: Clear specific namespace or all if None
        
        Returns:
            Success status
        """
        try:
            if namespace:
                pattern = f"{namespace}:*"
                # Clear Redis keys with pattern
                if self.redis:
                    try:
                        keys = await self._redis_keys(pattern)
                        if keys:
                            await self._redis_delete_many(keys)
                    except RedisConnectionError:
                        pass
                
                # Clear memory cache (limited capability)
                await self.memory_cache.clear()
            else:
                # Clear all
                if self.redis:
                    try:
                        await self._redis_flushdb()
                    except RedisConnectionError:
                        pass
                
                await self.memory_cache.clear()
            
            return True
            
        except Exception as e:
            logger.error(f"Cache clear error: {e}")
            return False
    
    async def exists(self, key: str, namespace: str = "default") -> bool:
        """Check if key exists in cache"""
        full_key = self._generate_key(namespace, key)
        
        try:
            if self.redis:
                try:
                    return await self._redis_exists(full_key)
                except RedisConnectionError:
                    pass
            
            return await self.memory_cache.exists(full_key)
            
        except Exception:
            return False
    
    async def get_many(
        self,
        keys: list,
        namespace: str = "default"
    ) -> Dict[str, Any]:
        """Get multiple values from cache"""
        result = {}
        
        for key in keys:
            value = await self.get(key, namespace)
            if value is not None:
                result[key] = value
        
        return result
    
    async def set_many(
        self,
        mapping: Dict[str, Any],
        ttl: Optional[int] = None,
        namespace: str = "default"
    ) -> bool:
        """Set multiple values in cache"""
        success = True
        
        for key, value in mapping.items():
            if not await self.set(key, value, ttl, namespace):
                success = False
        
        return success
    
    def cache(
        self,
        ttl: Optional[int] = None,
        namespace: str = "default",
        key_prefix: Optional[str] = None
    ):
        """
        Decorator for caching function results.
        
        Args:
            ttl: TTL in seconds
            namespace: Cache namespace
            key_prefix: Optional key prefix
        
        Example:
            @cache_manager.cache(ttl=300, namespace="api")
            async def get_user_data(user_id: str):
                return await fetch_from_database(user_id)
        """
        def decorator(func):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                # Generate cache key from function name and arguments
                key_parts = [key_prefix or func.__name__]
                
                # Add args to key
                for arg in args:
                    if hasattr(arg, 'id'):
                        key_parts.append(str(arg.id))
                    else:
                        key_parts.append(str(arg))
                
                # Add kwargs to key
                for k, v in sorted(kwargs.items()):
                    key_parts.append(f"{k}={v}")
                
                cache_key = ":".join(key_parts)
                
                # Try to get from cache
                cached = await self.get(cache_key, namespace)
                if cached is not None:
                    return cached
                
                # Execute function
                result = await func(*args, **kwargs)
                
                # Cache result
                await self.set(cache_key, result, ttl, namespace)
                
                return result
            
            return wrapper
        return decorator
    
    def invalidate(self, pattern: str, namespace: str = "default"):
        """
        Decorator to invalidate cache on function execution.
        
        Args:
            pattern: Cache key pattern to invalidate
            namespace: Cache namespace
        
        Example:
            @cache_manager.invalidate(pattern="user:*", namespace="api")
            async def update_user(user_id: str, data: dict):
                return await update_database(user_id, data)
        """
        def decorator(func):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                # Execute function
                result = await func(*args, **kwargs)
                
                # Invalidate cache
                if "*" in pattern:
                    # Pattern-based invalidation
                    await self.clear(namespace)
                else:
                    # Specific key invalidation
                    await self.delete(pattern, namespace)
                
                return result
            
            return wrapper
        return decorator
    
    # Redis-specific async methods
    async def _redis_get(self, key: str) -> Optional[bytes]:
        """Async Redis get"""
        if isinstance(self.redis, AsyncRedis):
            return await self.redis.get(key)
        else:
            return await asyncio.get_event_loop().run_in_executor(
                None, self.redis.get, key
            )
    
    async def _redis_set(self, key: str, value: bytes, ttl: int) -> bool:
        """Async Redis set with TTL"""
        if isinstance(self.redis, AsyncRedis):
            return await self.redis.setex(key, ttl, value)
        else:
            return await asyncio.get_event_loop().run_in_executor(
                None, self.redis.setex, key, ttl, value
            )
    
    async def _redis_delete(self, key: str) -> int:
        """Async Redis delete"""
        if isinstance(self.redis, AsyncRedis):
            return await self.redis.delete(key)
        else:
            return await asyncio.get_event_loop().run_in_executor(
                None, self.redis.delete, key
            )
    
    async def _redis_exists(self, key: str) -> bool:
        """Async Redis exists"""
        if isinstance(self.redis, AsyncRedis):
            return await self.redis.exists(key) > 0
        else:
            return await asyncio.get_event_loop().run_in_executor(
                None, lambda: self.redis.exists(key) > 0
            )
    
    async def _redis_keys(self, pattern: str) -> list:
        """Async Redis keys by pattern"""
        if isinstance(self.redis, AsyncRedis):
            return await self.redis.keys(pattern)
        else:
            return await asyncio.get_event_loop().run_in_executor(
                None, self.redis.keys, pattern
            )
    
    async def _redis_delete_many(self, keys: list) -> int:
        """Async Redis delete multiple keys"""
        if not keys:
            return 0
        
        if isinstance(self.redis, AsyncRedis):
            return await self.redis.delete(*keys)
        else:
            return await asyncio.get_event_loop().run_in_executor(
                None, lambda: self.redis.delete(*keys)
            )
    
    async def _redis_flushdb(self) -> bool:
        """Async Redis flush database"""
        if isinstance(self.redis, AsyncRedis):
            return await self.redis.flushdb()
        else:
            return await asyncio.get_event_loop().run_in_executor(
                None, self.redis.flushdb
            )
    
    # Statistics methods
    def _record_hit(self):
        """Record cache hit"""
        if self.enable_stats:
            self.stats["hits"] += 1
    
    def _record_miss(self):
        """Record cache miss"""
        if self.enable_stats:
            self.stats["misses"] += 1
    
    def _record_set(self):
        """Record cache set"""
        if self.enable_stats:
            self.stats["sets"] += 1
    
    def _record_delete(self):
        """Record cache delete"""
        if self.enable_stats:
            self.stats["deletes"] += 1
    
    def _record_error(self):
        """Record cache error"""
        if self.enable_stats:
            self.stats["errors"] += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        if not self.enable_stats:
            return {}
        
        total_requests = self.stats["hits"] + self.stats["misses"]
        hit_rate = (self.stats["hits"] / total_requests * 100) if total_requests > 0 else 0
        
        return {
            **self.stats,
            "total_requests": total_requests,
            "hit_rate": round(hit_rate, 2),
            "redis_connected": self.redis is not None
        }
    
    def reset_stats(self):
        """Reset statistics"""
        self.stats = {
            "hits": 0,
            "misses": 0,
            "sets": 0,
            "deletes": 0,
            "errors": 0
        }


# Global cache manager instance
_cache_manager = None


def get_cache_manager(
    redis_url: Optional[str] = None,
    **kwargs
) -> CacheManager:
    """
    Get or create global cache manager instance.
    
    Args:
        redis_url: Redis connection URL
        **kwargs: Additional arguments for CacheManager
    
    Returns:
        CacheManager instance
    """
    global _cache_manager
    
    if _cache_manager is None:
        _cache_manager = CacheManager(redis_url=redis_url, **kwargs)
    
    return _cache_manager