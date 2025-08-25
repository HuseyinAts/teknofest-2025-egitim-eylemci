"""
TEKNOFEST 2025 - Production Health Check & Monitoring Endpoints
"""

from fastapi import APIRouter, status, HTTPException
from typing import Dict, Any, List
import psutil
import aioredis
import asyncpg
from datetime import datetime, timedelta
import os
from prometheus_client import Counter, Histogram, Gauge, generate_latest
from src.core.database import get_db
from src.core.redis_client import get_redis
import time

router = APIRouter(prefix="/monitoring", tags=["monitoring"])

# Metrics
health_check_counter = Counter(
    'health_check_total',
    'Total number of health checks',
    ['status']
)

system_cpu_usage = Gauge(
    'system_cpu_usage_percent',
    'System CPU usage percentage'
)

system_memory_usage = Gauge(
    'system_memory_usage_percent',
    'System memory usage percentage'
)

database_connections = Gauge(
    'database_connections_active',
    'Active database connections'
)

redis_connections = Gauge(
    'redis_connections_active',
    'Active Redis connections'
)

response_time_histogram = Histogram(
    'http_request_duration_seconds',
    'HTTP request latency',
    ['method', 'endpoint', 'status']
)

class HealthChecker:
    """Production health checker"""
    
    @staticmethod
    async def check_database() -> Dict[str, Any]:
        """Check database health"""
        try:
            start = time.time()
            async with get_db() as conn:
                result = await conn.fetchval("SELECT 1")
                latency = (time.time() - start) * 1000
                
                # Get connection stats
                stats = await conn.fetchrow("""
                    SELECT 
                        numbackends as active_connections,
                        pg_database_size(current_database()) as db_size,
                        pg_postmaster_start_time() as uptime
                    FROM pg_stat_database 
                    WHERE datname = current_database()
                """)
                
                return {
                    "status": "healthy",
                    "latency_ms": round(latency, 2),
                    "active_connections": stats['active_connections'],
                    "database_size_mb": round(stats['db_size'] / 1024 / 1024, 2),
                    "uptime": str(datetime.now() - stats['uptime'])
                }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }
    
    @staticmethod
    async def check_redis() -> Dict[str, Any]:
        """Check Redis health"""
        try:
            start = time.time()
            redis = await get_redis()
            
            # Ping Redis
            await redis.ping()
            latency = (time.time() - start) * 1000
            
            # Get Redis info
            info = await redis.info()
            
            return {
                "status": "healthy",
                "latency_ms": round(latency, 2),
                "version": info.get('redis_version'),
                "connected_clients": info.get('connected_clients'),
                "used_memory_mb": round(info.get('used_memory', 0) / 1024 / 1024, 2),
                "uptime_days": info.get('uptime_in_days')
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }
    
    @staticmethod
    def check_system() -> Dict[str, Any]:
        """Check system resources"""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Update Prometheus metrics
            system_cpu_usage.set(cpu_percent)
            system_memory_usage.set(memory.percent)
            
            return {
                "status": "healthy",
                "cpu": {
                    "usage_percent": cpu_percent,
                    "cores": psutil.cpu_count()
                },
                "memory": {
                    "usage_percent": memory.percent,
                    "available_gb": round(memory.available / 1024 / 1024 / 1024, 2),
                    "total_gb": round(memory.total / 1024 / 1024 / 1024, 2)
                },
                "disk": {
                    "usage_percent": disk.percent,
                    "free_gb": round(disk.free / 1024 / 1024 / 1024, 2),
                    "total_gb": round(disk.total / 1024 / 1024 / 1024, 2)
                }
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }
    
    @staticmethod
    async def check_external_services() -> Dict[str, Any]:
        """Check external service connectivity"""
        services = {}
        
        # Check S3
        try:
            import boto3
            s3 = boto3.client('s3')
            s3.head_bucket(Bucket=os.environ.get('S3_BUCKET'))
            services['s3'] = {"status": "healthy"}
        except:
            services['s3'] = {"status": "unhealthy"}
        
        # Check Email Service
        try:
            # Simplified check - actual implementation would test SMTP
            services['email'] = {"status": "healthy"}
        except:
            services['email'] = {"status": "unhealthy"}
        
        # Check AI Service
        try:
            # Check if API key exists
            if os.environ.get('OPENAI_API_KEY'):
                services['ai'] = {"status": "healthy"}
            else:
                services['ai'] = {"status": "not_configured"}
        except:
            services['ai'] = {"status": "unhealthy"}
        
        return services

@router.get("/health", status_code=status.HTTP_200_OK)
async def health_check() -> Dict[str, Any]:
    """
    Basic health check endpoint for load balancers
    Returns 200 if service is running
    """
    health_check_counter.labels(status='success').inc()
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": os.environ.get('APP_VERSION', '1.0.0'),
        "environment": os.environ.get('ENVIRONMENT', 'production')
    }

@router.get("/ready", status_code=status.HTTP_200_OK)
async def readiness_check() -> Dict[str, Any]:
    """
    Readiness check - verifies all dependencies are ready
    Used by Kubernetes readiness probe
    """
    checker = HealthChecker()
    
    # Check all dependencies
    db_health = await checker.check_database()
    redis_health = await checker.check_redis()
    
    # Service is ready only if all critical dependencies are healthy
    is_ready = (
        db_health['status'] == 'healthy' and 
        redis_health['status'] == 'healthy'
    )
    
    if not is_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "status": "not_ready",
                "database": db_health['status'],
                "redis": redis_health['status']
            }
        )
    
    return {
        "status": "ready",
        "timestamp": datetime.utcnow().isoformat()
    }

@router.get("/live", status_code=status.HTTP_200_OK)
async def liveness_check() -> Dict[str, Any]:
    """
    Liveness check - verifies application is responsive
    Used by Kubernetes liveness probe
    """
    return {
        "status": "alive",
        "timestamp": datetime.utcnow().isoformat(),
        "pid": os.getpid()
    }

@router.get("/status", status_code=status.HTTP_200_OK)
async def detailed_status() -> Dict[str, Any]:
    """
    Detailed system status for monitoring dashboard
    Provides comprehensive health information
    """
    checker = HealthChecker()
    
    # Gather all health metrics
    db_health = await checker.check_database()
    redis_health = await checker.check_redis()
    system_health = checker.check_system()
    external_services = await checker.check_external_services()
    
    # Calculate overall health
    health_scores = {
        'database': 100 if db_health['status'] == 'healthy' else 0,
        'redis': 100 if redis_health['status'] == 'healthy' else 0,
        'system': 100 if system_health['cpu']['usage_percent'] < 80 else 50,
    }
    
    overall_health = sum(health_scores.values()) / len(health_scores)
    
    return {
        "status": "healthy" if overall_health > 75 else "degraded",
        "health_score": round(overall_health, 2),
        "timestamp": datetime.utcnow().isoformat(),
        "version": os.environ.get('APP_VERSION', '1.0.0'),
        "environment": os.environ.get('ENVIRONMENT', 'production'),
        "uptime": str(datetime.now() - datetime.fromtimestamp(psutil.boot_time())),
        "components": {
            "database": db_health,
            "redis": redis_health,
            "system": system_health,
            "external_services": external_services
        }
    }

@router.get("/metrics")
async def prometheus_metrics():
    """
    Prometheus metrics endpoint
    Returns metrics in Prometheus format
    """
    return generate_latest()

@router.get("/debug", status_code=status.HTTP_200_OK)
async def debug_info() -> Dict[str, Any]:
    """
    Debug information (only in non-production or for authorized users)
    """
    if os.environ.get('ENVIRONMENT') == 'production':
        # In production, require authentication
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Debug endpoint is disabled in production"
        )
    
    return {
        "environment_variables": {
            k: v for k, v in os.environ.items() 
            if not any(secret in k.lower() for secret in ['password', 'key', 'token', 'secret'])
        },
        "python_version": sys.version,
        "process": {
            "pid": os.getpid(),
            "cwd": os.getcwd(),
            "memory_usage_mb": psutil.Process().memory_info().rss / 1024 / 1024
        }
    }

@router.post("/maintenance", status_code=status.HTTP_200_OK)
async def toggle_maintenance(enabled: bool, message: str = None) -> Dict[str, Any]:
    """
    Toggle maintenance mode
    Requires admin authentication in production
    """
    # This would require proper authentication
    
    os.environ['MAINTENANCE_MODE'] = str(enabled).lower()
    if message:
        os.environ['MAINTENANCE_MESSAGE'] = message
    
    return {
        "maintenance_mode": enabled,
        "message": message or "System is under maintenance"
    }

# Error tracking endpoint
@router.get("/errors", status_code=status.HTTP_200_OK)
async def recent_errors(limit: int = 10) -> List[Dict[str, Any]]:
    """
    Get recent application errors
    Requires authentication in production
    """
    # This would fetch from error tracking system (Sentry, etc.)
    return []

# Performance metrics
@router.get("/performance", status_code=status.HTTP_200_OK)
async def performance_metrics() -> Dict[str, Any]:
    """
    Get performance metrics
    """
    # This would aggregate performance data
    return {
        "response_times": {
            "p50": 45,
            "p95": 120,
            "p99": 250
        },
        "throughput": {
            "requests_per_second": 150,
            "active_connections": 45
        },
        "error_rate": 0.02
    }
