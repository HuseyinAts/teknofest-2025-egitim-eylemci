# 🚦 Rate Limiting Implementation Guide

## 📋 Table of Contents
1. [Overview](#overview)
2. [Features](#features)
3. [Quick Start](#quick-start)
4. [Configuration](#configuration)
5. [Strategies](#strategies)
6. [Usage Examples](#usage-examples)
7. [Testing](#testing)
8. [Monitoring](#monitoring)
9. [Troubleshooting](#troubleshooting)
10. [Best Practices](#best-practices)

---

## 🎯 Overview

Production-ready rate limiting system for TEKNOFEST 2025 API with multiple strategies, backends, and granular control.

### Key Features
- ✅ **Multiple Strategies**: Fixed Window, Sliding Window, Token Bucket
- ✅ **Multiple Backends**: Memory (dev), Redis (production)
- ✅ **Per-Endpoint Limits**: Different limits for different endpoints
- ✅ **User-Based Limits**: Track by IP or authenticated user
- ✅ **Standard Headers**: RFC 6585 compliant rate limit headers
- ✅ **Graceful Degradation**: Falls back to memory if Redis unavailable

---

## 🚀 Features

### 1. **Rate Limiting Strategies**

| Strategy | Description | Use Case |
|----------|-------------|----------|
| **Fixed Window** | Simple counter reset every period | Basic protection |
| **Sliding Window** | More accurate, no burst at window edge | Accurate limiting |
| **Token Bucket** | Allows controlled bursts | API with burst traffic |
| **Leaky Bucket** | Smooth rate enforcement | Consistent flow |

### 2. **Storage Backends**

| Backend | Description | Use Case |
|---------|-------------|----------|
| **Memory** | In-process storage | Development, single server |
| **Redis** | Distributed storage | Production, multi-server |

### 3. **Rate Limit Headers**

All responses include standard headers:
```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1629456789
Retry-After: 60
```

---

## ⚡ Quick Start

### 1. **Enable Rate Limiting**

In `.env`:
```env
# Enable rate limiting
RATE_LIMIT_ENABLED=true
RATE_LIMIT_REQUESTS=100
RATE_LIMIT_PERIOD=3600  # 1 hour in seconds

# Optional: Use Redis
REDIS_URL=redis://localhost:6379/0
REDIS_PASSWORD=your_redis_password
```

### 2. **Run API with Rate Limiting**

```bash
# Start the rate-limited API
python src/api_server_with_rate_limit.py
```

### 3. **Test Rate Limiting**

```bash
# Run test suite
python test_rate_limit.py
```

---

## ⚙️ Configuration

### Environment Variables

```env
# Basic Configuration
RATE_LIMIT_ENABLED=true          # Enable/disable rate limiting
RATE_LIMIT_REQUESTS=100          # Number of requests allowed
RATE_LIMIT_PERIOD=3600           # Time period in seconds

# Redis Configuration (Optional)
REDIS_URL=redis://localhost:6379/0
REDIS_PASSWORD=your_password
REDIS_MAX_CONNECTIONS=10

# Advanced Options
RATE_LIMIT_STRATEGY=sliding_window  # fixed_window, sliding_window, token_bucket
RATE_LIMIT_KEY_PREFIX=teknofest:    # Redis key prefix
```

### Programmatic Configuration

```python
from src.rate_limiter import RateLimiter, RateLimitStrategy, RateLimitBackend

# Create custom rate limiter
limiter = RateLimiter(
    requests=100,
    period=3600,
    strategy=RateLimitStrategy.SLIDING_WINDOW,
    backend=RateLimitBackend.REDIS,
    redis_url="redis://localhost:6379",
    redis_password="password",
    enabled=True
)
```

---

## 📊 Strategies

### Fixed Window
```python
# Simple counter that resets every period
@rate_limit(requests=100, period=3600, strategy=RateLimitStrategy.FIXED_WINDOW)
```
- ✅ Simple and fast
- ❌ Allows burst at window boundaries

### Sliding Window
```python
# More accurate, tracks exact request times
@rate_limit(requests=100, period=3600, strategy=RateLimitStrategy.SLIDING_WINDOW)
```
- ✅ No boundary burst issue
- ❌ More memory/storage intensive

### Token Bucket
```python
# Allows controlled bursts
@rate_limit(requests=100, period=3600, strategy=RateLimitStrategy.TOKEN_BUCKET)
```
- ✅ Handles burst traffic well
- ✅ Smooth rate limiting
- ❌ More complex to understand

---

## 💻 Usage Examples

### 1. **Global Rate Limiting**

Applied to all endpoints:

```python
from src.rate_limiter import RateLimiter, RateLimitMiddleware

# Create global limiter
global_limiter = RateLimiter(
    requests=100,
    period=3600,
    enabled=True
)

# Add middleware
app.add_middleware(
    RateLimitMiddleware,
    limiter=global_limiter,
    exclude_paths=["/health", "/docs"]
)
```

### 2. **Per-Endpoint Rate Limiting**

Different limits for different endpoints:

```python
from src.rate_limiter import rate_limit, RateLimitPresets

# Strict limit for auth
@app.post("/auth/login")
@RateLimitPresets.auth()  # 5 requests per 5 minutes
async def login(request: Request, data: LoginRequest):
    ...

# Normal limit for API
@app.post("/api/query")
@RateLimitPresets.normal()  # 100 requests per hour
async def query(request: Request, data: QueryRequest):
    ...

# Custom limit
@app.post("/api/heavy-operation")
@rate_limit(requests=10, period=60)  # 10 requests per minute
async def heavy_operation(request: Request):
    ...
```

### 3. **Rate Limit Presets**

Pre-configured limits for common use cases:

```python
from src.rate_limiter import RateLimitPresets

# Available presets:
RateLimitPresets.strict()   # 10 req/min - Sensitive endpoints
RateLimitPresets.auth()     # 5 req/5min - Authentication
RateLimitPresets.normal()   # 100 req/hour - Standard API
RateLimitPresets.relaxed()  # 1000 req/hour - Public endpoints
RateLimitPresets.api()      # 100 req/min - API endpoints
```

### 4. **Custom Key Function**

Rate limit by custom identifier:

```python
def get_api_key(request: Request) -> str:
    """Get API key from request"""
    return request.headers.get("X-API-Key", "anonymous")

@app.post("/api/endpoint")
@rate_limit(requests=1000, period=3600, key_func=get_api_key)
async def api_endpoint(request: Request):
    ...
```

### 5. **Handling Rate Limit Errors**

```python
from src.rate_limiter import RateLimitExceeded

@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(
        status_code=429,
        content={
            "error": "Too Many Requests",
            "message": f"Please retry after {exc.retry_after} seconds",
            "retry_after": exc.retry_after
        },
        headers=exc.headers
    )
```

---

## 🧪 Testing

### Run Test Suite

```bash
# Basic test
python test_rate_limit.py

# Test with specific endpoint
curl -X POST http://localhost:8003/api/question \
  -H "Content-Type: application/json" \
  -d '{"question": "test"}' \
  -w "\n%{http_code} - Remaining: %{header.x-ratelimit-remaining}\n"
```

### Load Testing

```bash
# Using Apache Bench
ab -n 1000 -c 10 http://localhost:8003/api/endpoint

# Using wrk
wrk -t4 -c100 -d30s http://localhost:8003/api/endpoint
```

### Check Rate Limit Status

```bash
# Get current rate limit status
curl http://localhost:8003/api/rate-limit-status
```

---

## 📈 Monitoring

### Metrics to Track

1. **Rate Limit Hits**
   - Total requests
   - Requests blocked (429 responses)
   - Requests by endpoint
   - Requests by user/IP

2. **Performance Metrics**
   - Rate limit check latency
   - Redis connection latency
   - Memory usage (for memory backend)

### Redis Monitoring

```bash
# Monitor Redis keys
redis-cli --scan --pattern "teknofest:rate_limit:*"

# Get key TTL
redis-cli TTL "teknofest:rate_limit:ip:192.168.1.1:/api/endpoint"

# Monitor Redis performance
redis-cli --stat
```

### Application Logs

```python
# Enable detailed logging
LOG_LEVEL=DEBUG

# View logs
tail -f logs/app.log | grep "rate_limit"
```

---

## 🔧 Troubleshooting

### Common Issues

#### 1. **Rate Limiting Not Working**

Check:
```bash
# Is it enabled?
echo $RATE_LIMIT_ENABLED

# Test endpoint
curl -I http://localhost:8003/api/rate-limit-status
```

Solution:
```env
RATE_LIMIT_ENABLED=true
```

#### 2. **Redis Connection Failed**

Check:
```bash
# Test Redis connection
redis-cli ping

# Check Redis URL
echo $REDIS_URL
```

Solution:
```env
REDIS_URL=redis://localhost:6379/0
REDIS_PASSWORD=your_password
```

#### 3. **Too Many 429 Errors**

Adjust limits:
```env
RATE_LIMIT_REQUESTS=1000  # Increase limit
RATE_LIMIT_PERIOD=60      # Decrease period
```

#### 4. **Memory Backend Growing**

The memory backend auto-cleans expired entries. For manual cleanup:
```python
# In your application
limiter.backend._cleanup_if_needed()
```

---

## 📚 Best Practices

### 1. **Choose Appropriate Limits**

| Endpoint Type | Recommended Limit | Period |
|--------------|-------------------|---------|
| Authentication | 5-10 | 5 minutes |
| Public API | 100-1000 | 1 hour |
| Private API | 1000-10000 | 1 hour |
| Search | 10-30 | 1 minute |
| Heavy Operations | 1-5 | 1 minute |

### 2. **Use Different Strategies**

- **Authentication**: Fixed window (simple, effective)
- **API Endpoints**: Sliding window (accurate)
- **File Uploads**: Token bucket (allows bursts)

### 3. **Implement Graceful Degradation**

```python
# Fallback to memory if Redis fails
backend = RateLimitBackend.REDIS if redis_available else RateLimitBackend.MEMORY
```

### 4. **Provide Clear Error Messages**

```json
{
  "error": "Rate limit exceeded",
  "message": "You have made 105 requests in the last hour. Maximum is 100.",
  "retry_after": 1800,
  "upgrade_url": "https://api.example.com/pricing"
}
```

### 5. **Consider User Experience**

- Show remaining requests in UI
- Implement client-side rate limiting
- Provide upgrade options for higher limits
- Cache responses to reduce requests

---

## 🔐 Security Considerations

1. **Prevent Bypass Attempts**
   - Rate limit by IP AND user ID
   - Check X-Forwarded-For header
   - Implement CAPTCHA for repeated violations

2. **DDoS Protection**
   - Use cloud-based DDoS protection
   - Implement progressive rate limiting
   - Block IPs with excessive violations

3. **API Key Rate Limiting**
   ```python
   # Different limits for different API key tiers
   if api_key.tier == "premium":
       limit = 10000
   else:
       limit = 100
   ```

---

## 📊 Performance Impact

| Backend | Latency | Memory Usage | Scalability |
|---------|---------|--------------|-------------|
| Memory | <1ms | O(n) clients | Single server |
| Redis | 1-5ms | O(1) | Multi-server |

### Optimization Tips

1. **Use Redis Pipeline** for batch operations
2. **Enable Redis Persistence** for rate limit data
3. **Use Connection Pooling** for Redis
4. **Implement Local Caching** for frequent checks

---

## 🚀 Production Checklist

Before deploying to production:

- [ ] Enable rate limiting (`RATE_LIMIT_ENABLED=true`)
- [ ] Configure Redis backend
- [ ] Set appropriate limits per endpoint
- [ ] Test with load testing tools
- [ ] Configure monitoring and alerts
- [ ] Document rate limits in API docs
- [ ] Implement client retry logic
- [ ] Set up Redis persistence
- [ ] Configure Redis memory limits
- [ ] Test failover scenarios
- [ ] Review security implications
- [ ] Plan for scaling

---

## 📝 Example Implementation

Complete example with all features:

```python
# .env
RATE_LIMIT_ENABLED=true
RATE_LIMIT_REQUESTS=100
RATE_LIMIT_PERIOD=3600
REDIS_URL=redis://localhost:6379/0
REDIS_PASSWORD=secure_password

# Run server
python src/api_server_with_rate_limit.py

# Test
python test_rate_limit.py
```

---

## 🆘 Support

- Check logs: `logs/app.log`
- Run tests: `python test_rate_limit.py`
- Check status: `GET /api/rate-limit-status`
- Redis monitor: `redis-cli monitor`

---

*Last Updated: August 2025*
*Version: 1.0.0*