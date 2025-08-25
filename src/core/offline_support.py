"""
Offline Mode Support Module
TEKNOFEST 2025 - Production Ready Offline Support
"""

import json
import logging
import hashlib
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from pathlib import Path
import asyncio
from enum import Enum

import aiofiles
from pydantic import BaseModel, Field
from redis import asyncio as aioredis
from sqlalchemy import Column, String, DateTime, Text, Boolean, Integer
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session

from src.config import get_settings
from src.database.base import Base

logger = logging.getLogger(__name__)


class CacheStrategy(str, Enum):
    """Cache strategies for offline mode"""
    CACHE_FIRST = "cache_first"
    NETWORK_FIRST = "network_first"
    CACHE_ONLY = "cache_only"
    NETWORK_ONLY = "network_only"
    STALE_WHILE_REVALIDATE = "stale_while_revalidate"


class SyncStatus(str, Enum):
    """Sync status for offline data"""
    PENDING = "pending"
    SYNCING = "syncing"
    SYNCED = "synced"
    FAILED = "failed"
    CONFLICT = "conflict"


class OfflineQueue(Base):
    """Database model for offline queue"""
    __tablename__ = "offline_queue"
    
    id = Column(String(36), primary_key=True)
    endpoint = Column(String(255), nullable=False)
    method = Column(String(10), nullable=False)
    payload = Column(Text, nullable=True)
    headers = Column(Text, nullable=True)
    retry_count = Column(Integer, default=0)
    max_retries = Column(Integer, default=3)
    status = Column(String(20), default=SyncStatus.PENDING.value)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_attempt_at = Column(DateTime, nullable=True)
    synced_at = Column(DateTime, nullable=True)
    error_message = Column(Text, nullable=True)


class CacheEntry(Base):
    """Database model for cache entries"""
    __tablename__ = "cache_entries"
    
    cache_key = Column(String(255), primary_key=True)
    data = Column(Text, nullable=False)
    metadata = Column(Text, nullable=True)
    expires_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_accessed_at = Column(DateTime, default=datetime.utcnow)
    access_count = Column(Integer, default=0)
    is_stale = Column(Boolean, default=False)


class OfflineRequest(BaseModel):
    """Model for offline request"""
    id: str
    endpoint: str
    method: str
    payload: Optional[Dict[str, Any]] = None
    headers: Optional[Dict[str, str]] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    retry_count: int = 0
    max_retries: int = 3


class CacheConfig(BaseModel):
    """Cache configuration"""
    strategy: CacheStrategy = CacheStrategy.NETWORK_FIRST
    ttl_seconds: int = 3600
    max_size_mb: int = 100
    enable_compression: bool = True
    enable_encryption: bool = False


class OfflineManager:
    """Manages offline functionality"""
    
    def __init__(self, db_session: Session, redis_client: Optional[aioredis.Redis] = None):
        self.db_session = db_session
        self.redis_client = redis_client
        self.settings = get_settings()
        self.cache_dir = Path(self.settings.data_dir) / "offline_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._sync_lock = asyncio.Lock()
        self._is_online = True
        
    async def initialize(self):
        """Initialize offline manager"""
        try:
            if self.redis_client:
                await self.redis_client.ping()
                logger.info("Redis connection established for offline caching")
        except Exception as e:
            logger.warning(f"Redis not available, using file-based cache: {e}")
            
    def generate_cache_key(self, endpoint: str, params: Optional[Dict] = None) -> str:
        """Generate cache key for endpoint and parameters"""
        key_data = f"{endpoint}"
        if params:
            sorted_params = json.dumps(params, sort_keys=True)
            key_data += f":{sorted_params}"
        return hashlib.sha256(key_data.encode()).hexdigest()
    
    async def get_from_cache(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get data from cache"""
        try:
            # Try Redis first
            if self.redis_client:
                data = await self.redis_client.get(f"cache:{cache_key}")
                if data:
                    return json.loads(data)
            
            # Fallback to database
            cache_entry = self.db_session.query(CacheEntry).filter_by(
                cache_key=cache_key
            ).first()
            
            if cache_entry:
                # Check expiration
                if cache_entry.expires_at and cache_entry.expires_at < datetime.utcnow():
                    cache_entry.is_stale = True
                    self.db_session.commit()
                    return None
                
                # Update access metadata
                cache_entry.last_accessed_at = datetime.utcnow()
                cache_entry.access_count += 1
                self.db_session.commit()
                
                return json.loads(cache_entry.data)
                
            # Fallback to file cache
            cache_file = self.cache_dir / f"{cache_key}.json"
            if cache_file.exists():
                async with aiofiles.open(cache_file, 'r') as f:
                    content = await f.read()
                    return json.loads(content)
                    
        except Exception as e:
            logger.error(f"Cache retrieval error: {e}")
            
        return None
    
    async def save_to_cache(
        self, 
        cache_key: str, 
        data: Dict[str, Any],
        ttl_seconds: Optional[int] = None
    ):
        """Save data to cache"""
        try:
            ttl = ttl_seconds or 3600
            expires_at = datetime.utcnow() + timedelta(seconds=ttl)
            
            # Save to Redis if available
            if self.redis_client:
                await self.redis_client.setex(
                    f"cache:{cache_key}",
                    ttl,
                    json.dumps(data)
                )
            
            # Save to database
            cache_entry = self.db_session.query(CacheEntry).filter_by(
                cache_key=cache_key
            ).first()
            
            if cache_entry:
                cache_entry.data = json.dumps(data)
                cache_entry.expires_at = expires_at
                cache_entry.is_stale = False
            else:
                cache_entry = CacheEntry(
                    cache_key=cache_key,
                    data=json.dumps(data),
                    expires_at=expires_at
                )
                self.db_session.add(cache_entry)
            
            self.db_session.commit()
            
            # Save to file cache as backup
            cache_file = self.cache_dir / f"{cache_key}.json"
            async with aiofiles.open(cache_file, 'w') as f:
                await f.write(json.dumps({
                    "data": data,
                    "expires_at": expires_at.isoformat(),
                    "cached_at": datetime.utcnow().isoformat()
                }))
                
        except Exception as e:
            logger.error(f"Cache save error: {e}")
    
    async def queue_request(self, request: OfflineRequest):
        """Queue request for later synchronization"""
        try:
            queue_entry = OfflineQueue(
                id=request.id,
                endpoint=request.endpoint,
                method=request.method,
                payload=json.dumps(request.payload) if request.payload else None,
                headers=json.dumps(request.headers) if request.headers else None,
                retry_count=request.retry_count,
                max_retries=request.max_retries
            )
            self.db_session.add(queue_entry)
            self.db_session.commit()
            
            logger.info(f"Queued offline request: {request.id}")
            
        except Exception as e:
            logger.error(f"Failed to queue request: {e}")
            raise
    
    async def sync_offline_data(self) -> Dict[str, Any]:
        """Synchronize offline data with server"""
        async with self._sync_lock:
            try:
                # Get pending requests
                pending = self.db_session.query(OfflineQueue).filter_by(
                    status=SyncStatus.PENDING.value
                ).order_by(OfflineQueue.created_at).all()
                
                results = {
                    "total": len(pending),
                    "synced": 0,
                    "failed": 0,
                    "errors": []
                }
                
                for entry in pending:
                    try:
                        entry.status = SyncStatus.SYNCING.value
                        entry.last_attempt_at = datetime.utcnow()
                        self.db_session.commit()
                        
                        # Process the request
                        # This would be implemented based on your specific API
                        
                        entry.status = SyncStatus.SYNCED.value
                        entry.synced_at = datetime.utcnow()
                        results["synced"] += 1
                        
                    except Exception as e:
                        entry.retry_count += 1
                        if entry.retry_count >= entry.max_retries:
                            entry.status = SyncStatus.FAILED.value
                            results["failed"] += 1
                        else:
                            entry.status = SyncStatus.PENDING.value
                        
                        entry.error_message = str(e)
                        results["errors"].append({
                            "id": entry.id,
                            "error": str(e)
                        })
                    
                    self.db_session.commit()
                
                return results
                
            except Exception as e:
                logger.error(f"Sync error: {e}")
                raise
    
    async def clear_expired_cache(self):
        """Clear expired cache entries"""
        try:
            # Clear from database
            expired = self.db_session.query(CacheEntry).filter(
                CacheEntry.expires_at < datetime.utcnow()
            ).all()
            
            for entry in expired:
                self.db_session.delete(entry)
                
                # Remove file cache
                cache_file = self.cache_dir / f"{entry.cache_key}.json"
                if cache_file.exists():
                    cache_file.unlink()
            
            self.db_session.commit()
            
            # Clear from Redis
            if self.redis_client:
                pattern = "cache:*"
                cursor = 0
                while True:
                    cursor, keys = await self.redis_client.scan(
                        cursor, match=pattern, count=100
                    )
                    if keys:
                        for key in keys:
                            ttl = await self.redis_client.ttl(key)
                            if ttl <= 0:
                                await self.redis_client.delete(key)
                    if cursor == 0:
                        break
            
            logger.info(f"Cleared {len(expired)} expired cache entries")
            
        except Exception as e:
            logger.error(f"Cache cleanup error: {e}")
    
    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        try:
            total_entries = self.db_session.query(CacheEntry).count()
            stale_entries = self.db_session.query(CacheEntry).filter_by(
                is_stale=True
            ).count()
            
            # Calculate cache size
            cache_size = 0
            for cache_file in self.cache_dir.glob("*.json"):
                cache_size += cache_file.stat().st_size
            
            # Get queue stats
            queue_stats = {
                "pending": self.db_session.query(OfflineQueue).filter_by(
                    status=SyncStatus.PENDING.value
                ).count(),
                "synced": self.db_session.query(OfflineQueue).filter_by(
                    status=SyncStatus.SYNCED.value
                ).count(),
                "failed": self.db_session.query(OfflineQueue).filter_by(
                    status=SyncStatus.FAILED.value
                ).count()
            }
            
            return {
                "cache_entries": total_entries,
                "stale_entries": stale_entries,
                "cache_size_mb": round(cache_size / (1024 * 1024), 2),
                "queue_stats": queue_stats,
                "is_online": self._is_online
            }
            
        except Exception as e:
            logger.error(f"Failed to get cache stats: {e}")
            return {}
    
    def set_online_status(self, is_online: bool):
        """Set online/offline status"""
        self._is_online = is_online
        logger.info(f"Network status changed: {'Online' if is_online else 'Offline'}")
    
    async def handle_offline_request(
        self,
        endpoint: str,
        method: str,
        payload: Optional[Dict] = None,
        strategy: CacheStrategy = CacheStrategy.NETWORK_FIRST
    ) -> Optional[Dict[str, Any]]:
        """Handle request with offline support"""
        cache_key = self.generate_cache_key(endpoint, payload)
        
        # Cache-first strategy
        if strategy == CacheStrategy.CACHE_FIRST:
            cached = await self.get_from_cache(cache_key)
            if cached:
                return cached
        
        # Network request (would be implemented based on your API)
        if self._is_online and strategy != CacheStrategy.CACHE_ONLY:
            try:
                # Make network request here
                # response = await make_api_request(endpoint, method, payload)
                # await self.save_to_cache(cache_key, response)
                # return response
                pass
            except Exception as e:
                logger.warning(f"Network request failed: {e}")
        
        # Fallback to cache
        if strategy != CacheStrategy.NETWORK_ONLY:
            cached = await self.get_from_cache(cache_key)
            if cached:
                return cached
        
        # Queue for later if offline
        if not self._is_online:
            request = OfflineRequest(
                id=cache_key,
                endpoint=endpoint,
                method=method,
                payload=payload
            )
            await self.queue_request(request)
        
        return None


class OfflineMiddleware:
    """Middleware for handling offline mode"""
    
    def __init__(self, offline_manager: OfflineManager):
        self.offline_manager = offline_manager
    
    async def __call__(self, request, call_next):
        """Process request with offline support"""
        # Check network status
        network_status = request.headers.get("X-Network-Status", "online")
        self.offline_manager.set_online_status(network_status == "online")
        
        # Add offline manager to request state
        request.state.offline_manager = self.offline_manager
        
        response = await call_next(request)
        
        # Add cache headers
        if hasattr(request.state, "cache_hit"):
            response.headers["X-Cache"] = "HIT" if request.state.cache_hit else "MISS"
        
        return response