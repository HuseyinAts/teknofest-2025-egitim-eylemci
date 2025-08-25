"""
Advanced Redis Cache Layer with Performance Optimization
TEKNOFEST 2025 - High Performance Caching System
"""

import json
import pickle
import hashlib
import asyncio
from typing import Any, Optional, Union, Callable, Dict, List
from datetime import datetime, timedelta
from functools import wraps
import logging

import redis
from redis import asyncio as aioredis
from redis.exceptions import RedisError, ConnectionError
import msgpack

from src.config import get_settings

logger = logging.getLogger(__name__)


class CacheSerializer:
    """High-performance serialization for cache data"""
    
    @staticmethod
    def serialize(data: Any, format: str = 'msgpack') -> bytes:
        """Serialize data for cache storage"""
        try:
            if format == 'msgpack':
                return msgpack.packb(data, use_bin_type=True)
            elif format == 'json':
                return json.dumps(data).encode('utf-8')
            elif format == 'pickle':
                return pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
            else:
                raise ValueError(f"Unsupported format: {format}")
        except Exception as e:
            logger.error(f"Serialization error: {e}")
            raise
    
    @staticmethod
    def deserialize(data: bytes, format: str = 'msgpack') -> Any:
        """Deserialize data from cache"""
        try:
            if format == 'msgpack':
                return msgpack.unpackb(data, raw=False)
            elif format == 'json':
                return json.loads(data.decode('utf-8'))
            elif format == 'pickle':
                return pickle.loads(data)
            else:
                raise ValueError(f"Unsupported format: {format}")
        except Exception as e:
            logger.error(f"Deserialization error: {e}")
            raise


class CacheKeyBuilder:
    """Intelligent cache key generation"""
    
    @staticmethod
    def build(prefix: str, *args, **kwargs) -> str:
        """Build cache key from prefix and parameters"""
        key_parts = [prefix]
        
        # Add positional arguments
        for arg in args:
            if isinstance(arg, (str, int, float, bool)):
                key_parts.append(str(arg))
            else:
                # Hash complex objects
                key_parts.append(hashlib.md5(
                    str(arg).encode()
                ).hexdigest()[:8])
        
        # Add keyword arguments
        if kwargs:
            sorted_kwargs = sorted(kwargs.items())
            for k, v in sorted_kwargs:
                key_parts.append(f"{k}:{v}")
        
        return ":".join(key_parts)
    
    @staticmethod
    def build_pattern(prefix: str) -> str:
        """Build pattern for cache key matching"""
        return f"{prefix}:*"


class RedisCache:
    """High-performance Redis cache with advanced features"""
    
    def __init__(self, 
                 redis_url: str = None,
                 prefix: str = "teknofest",
                 ttl: int = 3600,
                 max_connections: int = 50,
                 serializer: str = 'msgpack'):
        
        settings = get_settings()
        self.redis_url = redis_url or settings.redis_url
        self.prefix = prefix
        self.default_ttl = ttl
        self.serializer = serializer
        
        # Connection pool for better performance
        self.pool = redis.ConnectionPool.from_url(
            self.redis_url,
            max_connections=max_connections,
            decode_responses=False,
            socket_keepalive=True,
            socket_keepalive_options={
                1: 1,  # TCP_KEEPIDLE
                2: 1,  # TCP_KEEPINTVL
                3: 5,  # TCP_KEEPCNT
            }
        )
        
        self.client = redis.Redis(connection_pool=self.pool)
        self.async_client = None
        
        # Performance metrics
        self.metrics = {
            'hits': 0,
            'misses': 0,
            'errors': 0,
            'total_time': 0
        }
    
    async def get_async_client(self):
        """Get or create async Redis client"""
        if not self.async_client:
            self.async_client = await aioredis.from_url(
                self.redis_url,
                max_connections=50,
                decode_responses=False
            )
        return self.async_client
    
    def _make_key(self, key: str) -> str:
        """Create namespaced cache key"""
        return f"{self.prefix}:{key}"
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        try:
            start = datetime.now()
            full_key = self._make_key(key)
            
            value = self.client.get(full_key)
            
            elapsed = (datetime.now() - start).total_seconds()
            self.metrics['total_time'] += elapsed
            
            if value is None:
                self.metrics['misses'] += 1
                return None
            
            self.metrics['hits'] += 1
            return CacheSerializer.deserialize(value, self.serializer)
            
        except (RedisError, ConnectionError) as e:
            logger.error(f"Cache get error for {key}: {e}")
            self.metrics['errors'] += 1
            return None
    
    def set(self, key: str, value: Any, ttl: int = None) -> bool:
        """Set value in cache with TTL"""
        try:
            full_key = self._make_key(key)
            ttl = ttl or self.default_ttl
            
            serialized = CacheSerializer.serialize(value, self.serializer)
            
            return self.client.setex(
                full_key,
                ttl,
                serialized
            )
            
        except (RedisError, ConnectionError) as e:
            logger.error(f"Cache set error for {key}: {e}")
            self.metrics['errors'] += 1
            return False
    
    def delete(self, key: str) -> bool:
        """Delete key from cache"""
        try:
            full_key = self._make_key(key)
            return bool(self.client.delete(full_key))
        except (RedisError, ConnectionError) as e:
            logger.error(f"Cache delete error for {key}: {e}")
            return False
    
    def delete_pattern(self, pattern: str) -> int:
        """Delete all keys matching pattern"""
        try:
            full_pattern = self._make_key(pattern)
            keys = self.client.keys(full_pattern)
            
            if keys:
                return self.client.delete(*keys)
            return 0
            
        except (RedisError, ConnectionError) as e:
            logger.error(f"Cache delete pattern error for {pattern}: {e}")
            return 0
    
    def get_many(self, keys: List[str]) -> Dict[str, Any]:
        """Get multiple values at once (mget)"""
        try:
            full_keys = [self._make_key(k) for k in keys]
            values = self.client.mget(full_keys)
            
            result = {}
            for key, value in zip(keys, values):
                if value is not None:
                    result[key] = CacheSerializer.deserialize(value, self.serializer)
                    self.metrics['hits'] += 1
                else:
                    self.metrics['misses'] += 1
            
            return result
            
        except (RedisError, ConnectionError) as e:
            logger.error(f"Cache mget error: {e}")
            self.metrics['errors'] += 1
            return {}
    
    def set_many(self, data: Dict[str, Any], ttl: int = None) -> bool:
        """Set multiple values at once (mset)"""
        try:
            ttl = ttl or self.default_ttl
            pipe = self.client.pipeline()
            
            for key, value in data.items():
                full_key = self._make_key(key)
                serialized = CacheSerializer.serialize(value, self.serializer)
                pipe.setex(full_key, ttl, serialized)
            
            pipe.execute()
            return True
            
        except (RedisError, ConnectionError) as e:
            logger.error(f"Cache mset error: {e}")
            self.metrics['errors'] += 1
            return False
    
    async def get_async(self, key: str) -> Optional[Any]:
        """Async get value from cache"""
        try:
            client = await self.get_async_client()
            full_key = self._make_key(key)
            
            value = await client.get(full_key)
            
            if value is None:
                self.metrics['misses'] += 1
                return None
            
            self.metrics['hits'] += 1
            return CacheSerializer.deserialize(value, self.serializer)
            
        except Exception as e:
            logger.error(f"Async cache get error for {key}: {e}")
            self.metrics['errors'] += 1
            return None
    
    async def set_async(self, key: str, value: Any, ttl: int = None) -> bool:
        """Async set value in cache"""
        try:
            client = await self.get_async_client()
            full_key = self._make_key(key)
            ttl = ttl or self.default_ttl
            
            serialized = CacheSerializer.serialize(value, self.serializer)
            
            await client.setex(full_key, ttl, serialized)
            return True
            
        except Exception as e:
            logger.error(f"Async cache set error for {key}: {e}")
            self.metrics['errors'] += 1
            return False
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get cache performance metrics"""
        total_requests = self.metrics['hits'] + self.metrics['misses']
        
        return {
            'hits': self.metrics['hits'],
            'misses': self.metrics['misses'],
            'errors': self.metrics['errors'],
            'hit_rate': (self.metrics['hits'] / total_requests * 100) if total_requests > 0 else 0,
            'avg_response_time': (self.metrics['total_time'] / total_requests) if total_requests > 0 else 0,
            'total_requests': total_requests
        }
    
    def flush(self) -> bool:
        """Flush all cache keys with prefix"""
        try:
            pattern = self._make_key("*")
            keys = self.client.keys(pattern)
            
            if keys:
                self.client.delete(*keys)
            
            return True
            
        except (RedisError, ConnectionError) as e:
            logger.error(f"Cache flush error: {e}")
            return False


class CacheDecorator:
    """Decorator for automatic caching"""
    
    def __init__(self, cache: RedisCache):
        self.cache = cache
    
    def cached(self, 
               prefix: str = None,
               ttl: int = None,
               key_builder: Callable = None):
        """Cache decorator for functions"""
        
        def decorator(func):
            cache_prefix = prefix or func.__name__
            
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                # Build cache key
                if key_builder:
                    cache_key = key_builder(*args, **kwargs)
                else:
                    cache_key = CacheKeyBuilder.build(cache_prefix, *args, **kwargs)
                
                # Try to get from cache
                cached_value = self.cache.get(cache_key)
                if cached_value is not None:
                    logger.debug(f"Cache hit for {cache_key}")
                    return cached_value
                
                # Execute function
                result = func(*args, **kwargs)
                
                # Store in cache
                self.cache.set(cache_key, result, ttl)
                logger.debug(f"Cached result for {cache_key}")
                
                return result
            
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                # Build cache key
                if key_builder:
                    cache_key = key_builder(*args, **kwargs)
                else:
                    cache_key = CacheKeyBuilder.build(cache_prefix, *args, **kwargs)
                
                # Try to get from cache
                cached_value = await self.cache.get_async(cache_key)
                if cached_value is not None:
                    logger.debug(f"Cache hit for {cache_key}")
                    return cached_value
                
                # Execute function
                result = await func(*args, **kwargs)
                
                # Store in cache
                await self.cache.set_async(cache_key, result, ttl)
                logger.debug(f"Cached result for {cache_key}")
                
                return result
            
            # Return appropriate wrapper
            if asyncio.iscoroutinefunction(func):
                return async_wrapper
            else:
                return sync_wrapper
        
        return decorator
    
    def invalidate(self, pattern: str):
        """Invalidate cache by pattern"""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                result = func(*args, **kwargs)
                
                # Invalidate cache
                self.cache.delete_pattern(pattern)
                logger.debug(f"Invalidated cache pattern: {pattern}")
                
                return result
            
            return wrapper
        return decorator


# Global cache instance
_cache_instance = None

def get_cache() -> RedisCache:
    """Get global cache instance"""
    global _cache_instance
    
    if _cache_instance is None:
        _cache_instance = RedisCache()
    
    return _cache_instance


# Cache decorator instance
cache_decorator = CacheDecorator(get_cache())
cached = cache_decorator.cached
invalidate_cache = cache_decorator.invalidate
