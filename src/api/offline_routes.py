"""
Offline Support API Routes
TEKNOFEST 2025 - Production Ready
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Header, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from src.database.session import get_db
from src.core.offline_support import (
    OfflineManager,
    CacheStrategy,
    OfflineRequest,
    SyncStatus
)
from src.config import get_settings

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/offline", tags=["offline"])


class SyncRequest(BaseModel):
    """Sync request model"""
    force: bool = False


class CacheRequest(BaseModel):
    """Cache management request"""
    action: str = Field(..., pattern="^(clear|stats|refresh)$")
    target: Optional[str] = None


class QueuedRequestModel(BaseModel):
    """Model for queued offline request"""
    endpoint: str
    method: str
    payload: Optional[Dict[str, Any]] = None
    headers: Optional[Dict[str, str]] = None


def get_offline_manager(db: Session = Depends(get_db)) -> OfflineManager:
    """Get offline manager instance"""
    manager = OfflineManager(db_session=db)
    return manager


@router.get("/status")
async def get_offline_status(
    manager: OfflineManager = Depends(get_offline_manager),
    x_network_status: Optional[str] = Header(None)
):
    """Get current offline status and statistics"""
    try:
        # Update network status if provided
        if x_network_status:
            manager.set_online_status(x_network_status == "online")
        
        stats = await manager.get_cache_stats()
        
        return {
            "success": True,
            "data": {
                "is_online": manager._is_online,
                "cache_stats": stats,
                "timestamp": datetime.utcnow().isoformat()
            }
        }
    except Exception as e:
        logger.error(f"Failed to get offline status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/sync")
async def sync_offline_data(
    request: SyncRequest,
    background_tasks: BackgroundTasks,
    manager: OfflineManager = Depends(get_offline_manager)
):
    """Manually trigger offline data synchronization"""
    try:
        if request.force or manager._is_online:
            # Run sync in background
            background_tasks.add_task(manager.sync_offline_data)
            
            return {
                "success": True,
                "message": "Synchronization started",
                "timestamp": datetime.utcnow().isoformat()
            }
        else:
            return {
                "success": False,
                "message": "Cannot sync while offline",
                "timestamp": datetime.utcnow().isoformat()
            }
    except Exception as e:
        logger.error(f"Sync initiation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sync/status")
async def get_sync_status(
    manager: OfflineManager = Depends(get_offline_manager),
    db: Session = Depends(get_db)
):
    """Get synchronization status and history"""
    try:
        from src.core.offline_support import OfflineQueue
        
        # Get recent sync history
        recent_syncs = db.query(OfflineQueue).order_by(
            OfflineQueue.last_attempt_at.desc()
        ).limit(10).all()
        
        sync_history = []
        for sync in recent_syncs:
            sync_history.append({
                "id": sync.id,
                "endpoint": sync.endpoint,
                "method": sync.method,
                "status": sync.status,
                "retry_count": sync.retry_count,
                "last_attempt": sync.last_attempt_at.isoformat() if sync.last_attempt_at else None,
                "synced_at": sync.synced_at.isoformat() if sync.synced_at else None,
                "error": sync.error_message
            })
        
        # Get queue statistics
        queue_stats = await manager.get_cache_stats()
        
        return {
            "success": True,
            "data": {
                "queue_stats": queue_stats.get("queue_stats", {}),
                "sync_history": sync_history,
                "is_syncing": False,  # This would be tracked in production
                "timestamp": datetime.utcnow().isoformat()
            }
        }
    except Exception as e:
        logger.error(f"Failed to get sync status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/queue")
async def queue_offline_request(
    request: QueuedRequestModel,
    manager: OfflineManager = Depends(get_offline_manager)
):
    """Manually queue a request for offline synchronization"""
    try:
        offline_request = OfflineRequest(
            id=f"{datetime.utcnow().timestamp()}",
            endpoint=request.endpoint,
            method=request.method,
            payload=request.payload,
            headers=request.headers
        )
        
        await manager.queue_request(offline_request)
        
        return {
            "success": True,
            "message": "Request queued successfully",
            "request_id": offline_request.id,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to queue request: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/queue")
async def get_queued_requests(
    status: Optional[str] = None,
    limit: int = 50,
    manager: OfflineManager = Depends(get_offline_manager),
    db: Session = Depends(get_db)
):
    """Get list of queued offline requests"""
    try:
        from src.core.offline_support import OfflineQueue
        
        query = db.query(OfflineQueue)
        
        if status:
            query = query.filter_by(status=status)
        
        requests = query.order_by(OfflineQueue.created_at.desc()).limit(limit).all()
        
        queued_requests = []
        for req in requests:
            queued_requests.append({
                "id": req.id,
                "endpoint": req.endpoint,
                "method": req.method,
                "status": req.status,
                "retry_count": req.retry_count,
                "created_at": req.created_at.isoformat(),
                "last_attempt": req.last_attempt_at.isoformat() if req.last_attempt_at else None
            })
        
        return {
            "success": True,
            "data": {
                "requests": queued_requests,
                "total": len(queued_requests),
                "timestamp": datetime.utcnow().isoformat()
            }
        }
    except Exception as e:
        logger.error(f"Failed to get queued requests: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/queue/{request_id}")
async def delete_queued_request(
    request_id: str,
    manager: OfflineManager = Depends(get_offline_manager),
    db: Session = Depends(get_db)
):
    """Delete a specific queued request"""
    try:
        from src.core.offline_support import OfflineQueue
        
        request = db.query(OfflineQueue).filter_by(id=request_id).first()
        
        if not request:
            raise HTTPException(status_code=404, detail="Request not found")
        
        db.delete(request)
        db.commit()
        
        return {
            "success": True,
            "message": "Request deleted successfully",
            "timestamp": datetime.utcnow().isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete queued request: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/cache")
async def manage_cache(
    request: CacheRequest,
    background_tasks: BackgroundTasks,
    manager: OfflineManager = Depends(get_offline_manager)
):
    """Manage offline cache"""
    try:
        if request.action == "clear":
            if request.target:
                # Clear specific cache entry
                # This would be implemented based on requirements
                pass
            else:
                # Clear all expired cache
                background_tasks.add_task(manager.clear_expired_cache)
            
            return {
                "success": True,
                "message": "Cache clear initiated",
                "timestamp": datetime.utcnow().isoformat()
            }
        
        elif request.action == "stats":
            stats = await manager.get_cache_stats()
            return {
                "success": True,
                "data": stats,
                "timestamp": datetime.utcnow().isoformat()
            }
        
        elif request.action == "refresh":
            # Refresh cache entries
            # This would be implemented based on requirements
            return {
                "success": True,
                "message": "Cache refresh initiated",
                "timestamp": datetime.utcnow().isoformat()
            }
        
    except Exception as e:
        logger.error(f"Cache management failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/cache/{cache_key}")
async def get_cache_entry(
    cache_key: str,
    manager: OfflineManager = Depends(get_offline_manager)
):
    """Get specific cache entry"""
    try:
        data = await manager.get_from_cache(cache_key)
        
        if data:
            return {
                "success": True,
                "data": data,
                "cache_hit": True,
                "timestamp": datetime.utcnow().isoformat()
            }
        else:
            return {
                "success": False,
                "message": "Cache entry not found",
                "cache_hit": False,
                "timestamp": datetime.utcnow().isoformat()
            }
    except Exception as e:
        logger.error(f"Failed to get cache entry: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/test-offline")
async def test_offline_mode(
    strategy: CacheStrategy = CacheStrategy.NETWORK_FIRST,
    manager: OfflineManager = Depends(get_offline_manager)
):
    """Test endpoint for offline functionality"""
    try:
        # Simulate handling an offline request
        test_endpoint = "/api/v1/test"
        test_data = {"test": True, "timestamp": datetime.utcnow().isoformat()}
        
        result = await manager.handle_offline_request(
            endpoint=test_endpoint,
            method="GET",
            payload=None,
            strategy=strategy
        )
        
        if result:
            return {
                "success": True,
                "data": result,
                "source": "cache" if not manager._is_online else "network",
                "timestamp": datetime.utcnow().isoformat()
            }
        else:
            # Save test data to cache
            cache_key = manager.generate_cache_key(test_endpoint)
            await manager.save_to_cache(cache_key, test_data, ttl_seconds=3600)
            
            return {
                "success": True,
                "data": test_data,
                "source": "generated",
                "timestamp": datetime.utcnow().isoformat()
            }
    except Exception as e:
        logger.error(f"Offline test failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))