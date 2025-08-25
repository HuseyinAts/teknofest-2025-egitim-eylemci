# Offline Mode Documentation
## TEKNOFEST 2025 - Production Ready Offline Support

## Overview

This document describes the comprehensive offline mode implementation for the TEKNOFEST 2025 Education Technology platform. The system provides full offline functionality with automatic synchronization, caching, and conflict resolution.

## Features

### 1. Service Worker Support
- **Progressive Web App (PWA)** capability
- **Background sync** for queued requests
- **Cache strategies** for different resource types
- **Automatic updates** with user notification

### 2. Multi-Layer Caching
- **Redis** for fast in-memory caching
- **PostgreSQL** for persistent cache storage
- **File system** backup for critical data
- **IndexedDB** for client-side storage

### 3. Offline Queue Management
- Automatic request queuing when offline
- Retry logic with exponential backoff
- Priority-based synchronization
- Conflict detection and resolution

### 4. Real-time Status Indicators
- Visual network status indicator
- Queue size badge
- Sync progress tracking
- Cache statistics display

## Architecture

```
┌─────────────────┐
│   Frontend      │
│  (React + PWA)  │
├─────────────────┤
│ Service Worker  │
├─────────────────┤
│  IndexedDB      │
└────────┬────────┘
         │
    ┌────▼────┐
    │  Nginx  │
    └────┬────┘
         │
┌────────▼────────┐
│   FastAPI       │
│  Application    │
├─────────────────┤
│ Offline Manager │
├─────────────────┤
│ Cache Layer     │
├────┬──────┬─────┤
│Redis│ DB  │Files│
└────┴──────┴─────┘
```

## Usage

### Frontend Implementation

#### 1. Basic Offline Hook
```typescript
import { useOffline } from '@/hooks/useOffline';

function MyComponent() {
  const { isOnline, queueSize, sync } = useOffline();
  
  return (
    <div>
      {!isOnline && <Alert>You are offline</Alert>}
      {queueSize > 0 && (
        <Button onClick={sync}>
          Sync {queueSize} pending requests
        </Button>
      )}
    </div>
  );
}
```

#### 2. Offline-Aware Fetch
```typescript
import { useOfflineFetch } from '@/hooks/useOffline';

function DataComponent() {
  const { fetch, loading, error } = useOfflineFetch();
  
  const loadData = async () => {
    const data = await fetch('/api/v1/data', {
      cacheStrategy: CacheStrategy.CACHE_FIRST,
      cacheTTL: 3600000 // 1 hour
    });
    // Use data...
  };
}
```

### Backend Implementation

#### 1. Cached Endpoint
```python
from src.core.offline_support import OfflineManager

@app.get("/api/v1/data")
async def get_data(
    manager: OfflineManager = Depends(get_offline_manager)
):
    cache_key = manager.generate_cache_key("/api/v1/data")
    
    # Try cache first
    cached = await manager.get_from_cache(cache_key)
    if cached:
        return cached
    
    # Generate fresh data
    data = generate_data()
    
    # Save to cache
    await manager.save_to_cache(cache_key, data, ttl_seconds=3600)
    
    return data
```

#### 2. Queue Offline Request
```python
@app.post("/api/v1/action")
async def perform_action(
    request: ActionRequest,
    manager: OfflineManager = Depends(get_offline_manager)
):
    if not manager._is_online:
        # Queue for later
        offline_req = OfflineRequest(
            id=str(uuid.uuid4()),
            endpoint="/api/v1/action",
            method="POST",
            payload=request.dict()
        )
        await manager.queue_request(offline_req)
        return {"queued": True}
    
    # Process normally
    return process_action(request)
```

## Cache Strategies

### 1. Cache First
- Check cache first
- Fallback to network if miss
- Best for: Static content, rarely changing data

### 2. Network First
- Try network first
- Fallback to cache if offline
- Best for: Dynamic content, user data

### 3. Stale While Revalidate
- Return cache immediately
- Update cache in background
- Best for: Frequently accessed, tolerance for stale data

### 4. Cache Only
- Only use cached data
- Best for: Offline-only features

### 5. Network Only
- Always use network
- Best for: Real-time data, sensitive operations

## Configuration

### Environment Variables
```env
# Offline Mode Settings
OFFLINE_MODE_ENABLED=true
CACHE_TTL=3600
SYNC_INTERVAL=300
MAX_OFFLINE_QUEUE_SIZE=1000
CACHE_SIZE_LIMIT_MB=500

# Redis Configuration
REDIS_URL=redis://localhost:6379/0
REDIS_MAX_CONNECTIONS=50

# Database
DATABASE_URL=postgresql://user:pass@localhost/db
```

### Docker Deployment
```bash
# Start with offline support
docker-compose -f docker-compose.offline.yml up -d

# Monitor offline sync
docker-compose -f docker-compose.offline.yml logs -f sync-worker

# Clear cache
docker-compose -f docker-compose.offline.yml exec api python -m src.cli clear-cache
```

## Monitoring

### Health Check Endpoints
- `/health` - Overall system health
- `/api/v1/offline/status` - Offline mode status
- `/api/v1/offline/sync/status` - Sync queue status
- `/api/v1/offline/cache` - Cache statistics

### Metrics
- Queue size
- Cache hit rate
- Sync success rate
- Average sync time
- Cache memory usage

## Testing

### Run Offline Tests
```bash
# Backend tests
pytest tests/test_offline_support.py -v

# Frontend tests
npm test -- --coverage src/services/offlineManager.test.ts

# E2E offline tests
npm run test:e2e:offline
```

### Manual Testing

1. **Test Offline Mode**
   - Disable network in DevTools
   - Verify UI shows offline indicator
   - Make changes and verify queuing

2. **Test Sync**
   - Re-enable network
   - Verify automatic sync
   - Check sync results

3. **Test Cache**
   - Load data online
   - Go offline
   - Verify cached data loads

## Troubleshooting

### Common Issues

#### 1. Service Worker Not Registering
```javascript
// Check browser console
navigator.serviceWorker.getRegistrations().then(console.log);

// Force update
navigator.serviceWorker.getRegistration().then(reg => reg.update());
```

#### 2. Cache Not Working
```bash
# Clear all caches
curl -X POST http://localhost:8000/api/v1/offline/cache \
  -H "Content-Type: application/json" \
  -d '{"action": "clear"}'
```

#### 3. Sync Failures
```bash
# Check sync status
curl http://localhost:8000/api/v1/offline/sync/status

# Force sync
curl -X POST http://localhost:8000/api/v1/offline/sync \
  -d '{"force": true}'
```

## Performance Optimization

### 1. Cache Size Management
- Automatic cleanup of expired entries
- LRU eviction policy
- Configurable size limits

### 2. Sync Optimization
- Batch synchronization
- Priority queues
- Compression for large payloads

### 3. Network Detection
- Multiple detection methods
- Debounced status updates
- Background connectivity checks

## Security Considerations

### 1. Data Encryption
- Encrypt sensitive cached data
- Use HTTPS for all communications
- Secure IndexedDB storage

### 2. Cache Invalidation
- Token-based cache keys
- User-specific cache isolation
- Automatic cleanup on logout

### 3. Conflict Resolution
- Version tracking
- Last-write-wins strategy
- User notification for conflicts

## Best Practices

### 1. Design for Offline First
- Assume network is unreliable
- Cache aggressively
- Queue all mutations

### 2. User Experience
- Clear offline indicators
- Progress feedback
- Error recovery options

### 3. Data Consistency
- Use transactions
- Implement idempotency
- Handle partial sync

## API Reference

### Offline Manager Methods

```python
# Generate cache key
cache_key = manager.generate_cache_key(endpoint, params)

# Cache operations
await manager.save_to_cache(key, data, ttl)
data = await manager.get_from_cache(key)
await manager.clear_expired_cache()

# Queue operations
await manager.queue_request(request)
results = await manager.sync_offline_data()

# Status
stats = await manager.get_cache_stats()
manager.set_online_status(is_online)
```

### React Hooks

```typescript
// Main offline hook
const {
  isOnline,
  queueSize,
  sync,
  syncing,
  cacheStats,
  clearCache
} = useOffline();

// Fetch hook
const { fetch, loading, error } = useOfflineFetch();

// Sync hook
const { sync, syncing, syncResults } = useOfflineSync();

// Cache hook
const { cacheStats, clearCache, clearing } = useOfflineCache();
```

## Deployment Checklist

- [ ] Service worker registered and active
- [ ] Redis configured with persistence
- [ ] Database migrations for offline tables
- [ ] Nginx cache directories created
- [ ] SSL certificates for HTTPS
- [ ] Monitoring alerts configured
- [ ] Backup strategy for offline data
- [ ] Documentation updated
- [ ] Team training completed

## Support

For issues or questions about offline mode:
1. Check the troubleshooting guide
2. Review logs: `docker-compose logs -f`
3. Contact the development team

---

Last updated: 2025
Version: 1.0.0