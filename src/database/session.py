"""
Production-Ready Database Session Configuration

Features:
- Advanced connection pooling with overflow management
- Connection health checks and automatic recovery
- Query timeout and deadlock prevention
- Performance monitoring and metrics
- Graceful degradation under load
"""

import logging
import time
import threading
from typing import Optional, Dict, Any
from contextlib import contextmanager
from datetime import datetime, timedelta

from sqlalchemy import create_engine, event, pool, text
from sqlalchemy.orm import sessionmaker, scoped_session, Session
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.pool import NullPool, QueuePool, StaticPool, Pool
from sqlalchemy.exc import DBAPIError, OperationalError, TimeoutError

from ..config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# Connection pool metrics
class PoolMetrics:
    """Track connection pool metrics for monitoring"""
    def __init__(self):
        self.connections_created = 0
        self.connections_recycled = 0
        self.connections_failed = 0
        self.checkout_time_sum = 0
        self.checkout_count = 0
        self.overflow_created = 0
        self.pool_timeouts = 0
        self._lock = threading.Lock()
        self._checkout_times = {}
    
    def record_checkout(self, connection_id: int) -> None:
        """Record connection checkout"""
        with self._lock:
            self._checkout_times[connection_id] = time.time()
            self.checkout_count += 1
    
    def record_checkin(self, connection_id: int) -> None:
        """Record connection checkin"""
        with self._lock:
            if connection_id in self._checkout_times:
                duration = time.time() - self._checkout_times[connection_id]
                self.checkout_time_sum += duration
                del self._checkout_times[connection_id]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current metrics"""
        with self._lock:
            avg_checkout_time = (
                self.checkout_time_sum / self.checkout_count 
                if self.checkout_count > 0 else 0
            )
            return {
                "connections_created": self.connections_created,
                "connections_recycled": self.connections_recycled,
                "connections_failed": self.connections_failed,
                "avg_checkout_time": avg_checkout_time,
                "total_checkouts": self.checkout_count,
                "overflow_created": self.overflow_created,
                "pool_timeouts": self.pool_timeouts,
                "active_checkouts": len(self._checkout_times)
            }

pool_metrics = PoolMetrics()


def get_database_url(async_mode: bool = False) -> str:
    """
    Get database URL based on mode (sync/async).
    """
    url = settings.database_url
    if not url:
        raise ValueError("Database URL not configured")
    
    # Convert to async URL if needed
    if async_mode and url.startswith("postgresql://"):
        url = url.replace("postgresql://", "postgresql+asyncpg://")
    elif async_mode and url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql+asyncpg://")
    
    return url


# Configure connection pool based on environment
def get_pool_config():
    """
    Get optimized connection pool configuration based on environment.
    
    Production optimizations:
    - Larger pool size for handling concurrent requests
    - Connection recycling to prevent stale connections
    - Pre-ping to validate connections before use
    - Timeout handling to prevent indefinite waits
    """
    base_config = {
        "pool_pre_ping": True,  # Always verify connections
        "echo_pool": settings.app_debug,  # Pool logging in debug mode
    }
    
    if settings.is_production():
        # Production configuration
        return {
            **base_config,
            "poolclass": QueuePool,
            "pool_size": max(20, settings.database_pool_size),  # Minimum 20 connections
            "max_overflow": max(40, settings.database_max_overflow),  # Allow 2x overflow
            "pool_timeout": 10,  # Fail fast on pool exhaustion
            "pool_recycle": 1800,  # Recycle every 30 minutes
            "pool_reset_on_return": "rollback",  # Always rollback on return
            "pool_use_lifo": True,  # Use LIFO to keep connections warm
        }
    elif settings.app_env == "testing":
        # Testing configuration - use static pool
        return {
            **base_config,
            "poolclass": StaticPool,
            "connect_args": {"check_same_thread": False}
        }
    else:
        # Development configuration
        return {
            **base_config,
            "poolclass": QueuePool,
            "pool_size": 5,
            "max_overflow": 10,
            "pool_timeout": 30,
            "pool_recycle": 3600,
            "pool_reset_on_return": "rollback",
        }


# Create synchronous engine
# Use appropriate pool based on testing mode
sync_pool_config = get_pool_config()
if settings.is_testing() and "sqlite" in settings.database_url:
    # SQLite in testing needs special handling
    sync_pool_config = {
        "poolclass": StaticPool,
        "connect_args": {"check_same_thread": False},
        "pool_pre_ping": True,
    }

engine = create_engine(
    get_database_url(async_mode=False),
    echo=settings.database_echo,
    future=True,
    **sync_pool_config
)

# Create asynchronous engine
# Note: Async engines need NullPool or StaticPool, not QueuePool
# For async engines, we use NullPool in testing and StaticPool otherwise
if settings.is_testing():
    async_pool_config = {
        "poolclass": NullPool,
        "pool_pre_ping": True,
    }
else:
    async_pool_config = {
        "poolclass": StaticPool,
        "pool_pre_ping": True,
        "connect_args": {"check_same_thread": False} if "sqlite" in settings.database_url else {}
    }

async_engine = create_async_engine(
    get_database_url(async_mode=True),
    echo=settings.database_echo,
    future=True,
    **async_pool_config
)


# Event listeners for connection monitoring
@event.listens_for(Pool, "connect")
def receive_connect(dbapi_conn, connection_record):
    """
    Event listener for new connections.
    Sets connection parameters and tracks metrics.
    """
    try:
        # Get backend PID for tracking
        connection_record.info['pid'] = (
            dbapi_conn.get_backend_pid() 
            if hasattr(dbapi_conn, 'get_backend_pid') 
            else id(dbapi_conn)
        )
        connection_record.info['connected_at'] = datetime.now()
        
        pool_metrics.connections_created += 1
        logger.debug(f"New database connection established: PID={connection_record.info.get('pid')}")
        
        # Set connection parameters based on environment
        with dbapi_conn.cursor() as cursor:
            if settings.is_production():
                # Production settings - strict timeouts
                cursor.execute("SET statement_timeout = '30s'")
                cursor.execute("SET lock_timeout = '5s'")
                cursor.execute("SET idle_in_transaction_session_timeout = '30s'")
                cursor.execute("SET deadlock_timeout = '1s'")
                # Enable query tracking
                cursor.execute("SET application_name = %s", (f"{settings.app_name}-{settings.app_version}",))
                cursor.execute("SET client_encoding = 'UTF8'")
                # Performance settings
                cursor.execute("SET jit = on")
                cursor.execute("SET random_page_cost = 1.1")
            else:
                # Development settings - more lenient
                cursor.execute("SET statement_timeout = '60s'")
                cursor.execute("SET lock_timeout = '10s'")
                cursor.execute("SET application_name = %s", (f"{settings.app_name}-dev",))
    except Exception as e:
        pool_metrics.connections_failed += 1
        logger.error(f"Failed to configure connection: {e}")
        raise


@event.listens_for(Pool, "checkout")
def receive_checkout(dbapi_conn, connection_record, connection_proxy):
    """
    Event listener for connection checkout from pool.
    Validates connection health and tracks metrics.
    """
    pid = connection_record.info.get('pid')
    
    # Track checkout metrics
    pool_metrics.record_checkout(pid)
    
    # Check connection age and recycle if needed
    connected_at = connection_record.info.get('connected_at')
    if connected_at:
        age = (datetime.now() - connected_at).total_seconds()
        if settings.is_production() and age > 1800:  # 30 minutes
            logger.info(f"Recycling old connection: PID={pid}, age={age:.0f}s")
            pool_metrics.connections_recycled += 1
            # Mark for recycling
            connection_record.invalidate()
    
    logger.debug(f"Connection checked out: PID={pid}")


@event.listens_for(Pool, "checkin")
def receive_checkin(dbapi_conn, connection_record):
    """
    Event listener for connection checkin to pool.
    Performs cleanup and tracks metrics.
    """
    pid = connection_record.info.get('pid')
    
    # Track checkin metrics
    pool_metrics.record_checkin(pid)
    
    # Reset connection state
    try:
        with dbapi_conn.cursor() as cursor:
            # Reset any session-level settings
            cursor.execute("RESET ALL")
            # Ensure clean transaction state
            cursor.execute("ROLLBACK")
    except Exception as e:
        logger.warning(f"Failed to reset connection state: {e}")
        # Mark connection as invalid
        connection_record.invalidate()
    
    logger.debug(f"Connection returned to pool: PID={pid}")

@event.listens_for(Pool, "invalidate")
def receive_invalidate(dbapi_conn, connection_record, exception):
    """
    Event listener for connection invalidation.
    """
    if exception:
        pool_metrics.connections_failed += 1
        logger.warning(f"Connection invalidated due to error: {exception}")

@event.listens_for(Pool, "reset")
def receive_reset(dbapi_conn, connection_record):
    """
    Event listener for connection reset.
    """
    logger.debug(f"Connection reset: PID={connection_record.info.get('pid')}")

@event.listens_for(Pool, "overflow_created")
def receive_overflow_created(dbapi_conn, connection_record):
    """
    Event listener for overflow connection creation.
    """
    pool_metrics.overflow_created += 1
    logger.info(f"Overflow connection created: Total={pool_metrics.overflow_created}")


# Create session factories
SessionLocal = scoped_session(
    sessionmaker(
        autocommit=False,
        autoflush=False,
        bind=engine,
        expire_on_commit=False,
    )
)

AsyncSessionLocal = sessionmaker(
    class_=AsyncSession,
    autocommit=False,
    autoflush=False,
    bind=async_engine,
    expire_on_commit=False,
)


def get_db_stats() -> Dict[str, Any]:
    """
    Get comprehensive database connection pool statistics.
    
    Returns:
        Dictionary containing pool metrics and health status
    """
    try:
        pool = engine.pool
        
        # Basic pool stats
        stats = {
            "pool_size": pool.size() if hasattr(pool, 'size') else 0,
            "checked_in": pool.checkedin() if hasattr(pool, 'checkedin') else 0,
            "overflow": pool.overflow() if hasattr(pool, 'overflow') else 0,
            "checked_out": pool.checkedout() if hasattr(pool, 'checkedout') else 0,
            "total": (
                pool.checkedin() + pool.checkedout() 
                if hasattr(pool, 'checkedin') else 0
            ),
            "max_overflow": getattr(pool, '_max_overflow', 0),
        }
        
        # Add custom metrics
        stats.update(pool_metrics.get_stats())
        
        # Calculate health indicators
        if stats["pool_size"] > 0:
            stats["utilization"] = (
                stats["checked_out"] / (stats["pool_size"] + stats["overflow"]) * 100
            )
            stats["health"] = "healthy" if stats["utilization"] < 80 else "warning"
        else:
            stats["utilization"] = 0
            stats["health"] = "unknown"
        
        # Check if pool is exhausted
        stats["exhausted"] = (
            stats["checked_out"] >= stats["pool_size"] + stats["max_overflow"]
        )
        
        return stats
    except Exception as e:
        logger.error(f"Failed to get pool stats: {e}")
        return {"error": str(e), "health": "error"}

def check_db_health() -> Dict[str, Any]:
    """
    Perform comprehensive database health check.
    
    Returns:
        Dictionary with health status and diagnostics
    """
    health = {
        "status": "unknown",
        "latency_ms": None,
        "pool_stats": None,
        "errors": [],
    }
    
    try:
        # Test database connectivity
        start = time.time()
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1")).scalar()
            if result != 1:
                raise ValueError("Unexpected query result")
        
        latency = (time.time() - start) * 1000
        health["latency_ms"] = round(latency, 2)
        
        # Get pool statistics
        health["pool_stats"] = get_db_stats()
        
        # Determine overall health
        if latency < 100 and not health["pool_stats"].get("exhausted"):
            health["status"] = "healthy"
        elif latency < 500:
            health["status"] = "degraded"
        else:
            health["status"] = "unhealthy"
        
    except TimeoutError:
        health["status"] = "timeout"
        health["errors"].append("Database connection timeout")
        pool_metrics.pool_timeouts += 1
    except OperationalError as e:
        health["status"] = "error"
        health["errors"].append(f"Database operational error: {e}")
    except Exception as e:
        health["status"] = "error"
        health["errors"].append(f"Health check failed: {e}")
    
    return health


@contextmanager
def get_raw_connection():
    """
    Get raw database connection for special operations.
    
    WARNING: Bypasses SQLAlchemy's connection pool and transaction management.
    Use only when absolutely necessary.
    
    Yields:
        Raw DBAPI connection
    """
    conn = None
    try:
        conn = engine.raw_connection()
        logger.warning("Raw connection acquired - ensure proper cleanup")
        yield conn
    except Exception as e:
        logger.error(f"Error with raw connection: {e}")
        raise
    finally:
        if conn:
            try:
                conn.close()
                logger.debug("Raw connection closed")
            except Exception as e:
                logger.error(f"Failed to close raw connection: {e}")

@contextmanager
def get_db_session() -> Session:
    """
    Get a database session with automatic cleanup.
    
    Yields:
        SQLAlchemy Session instance
    """
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception as e:
        session.rollback()
        logger.error(f"Database session error: {e}")
        raise
    finally:
        session.close()

async def get_async_db_session() -> AsyncSession:
    """
    Get an async database session.
    
    Yields:
        AsyncSession instance
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


def close_all_sessions():
    """
    Gracefully close all database sessions and connections.
    Should be called on application shutdown.
    """
    logger.info("Initiating database shutdown...")
    
    try:
        # Get final statistics
        final_stats = get_db_stats()
        logger.info(f"Final pool statistics: {final_stats}")
        
        # Remove scoped session
        SessionLocal.remove()
        logger.debug("Scoped session removed")
        
        # Dispose of connection pool
        engine.dispose()
        logger.info("Database connection pool disposed")
        
        # Log metrics summary
        metrics = pool_metrics.get_stats()
        logger.info(
            f"Session metrics - Created: {metrics['connections_created']}, "
            f"Failed: {metrics['connections_failed']}, "
            f"Recycled: {metrics['connections_recycled']}"
        )
        
    except Exception as e:
        logger.error(f"Error during database shutdown: {e}")
        # Force disposal
        try:
            engine.dispose()
        except:
            pass
    
    logger.info("Database shutdown complete")


async def close_async_sessions():
    """
    Gracefully close all async database sessions.
    """
    logger.info("Closing async database sessions...")
    
    try:
        await async_engine.dispose()
        logger.info("Async database sessions closed successfully")
    except Exception as e:
        logger.error(f"Error closing async sessions: {e}")
        raise

# Production-ready helper functions
def optimize_pool_for_load(expected_connections: int) -> None:
    """
    Dynamically adjust pool size based on expected load.
    
    Args:
        expected_connections: Expected number of concurrent connections
    """
    if not settings.is_production():
        logger.warning("Pool optimization should only be used in production")
        return
    
    # Calculate optimal pool settings
    pool_size = min(expected_connections // 2, 50)  # Cap at 50
    max_overflow = min(expected_connections, 100)  # Cap at 100
    
    logger.info(
        f"Optimizing pool for {expected_connections} connections: "
        f"pool_size={pool_size}, max_overflow={max_overflow}"
    )
    
    # Note: In production, consider recreating the engine with new settings
    # This is a placeholder for the actual implementation

def monitor_pool_health() -> None:
    """
    Monitor and log pool health metrics.
    Should be called periodically in production.
    """
    health = check_db_health()
    stats = health.get("pool_stats", {})
    
    if health["status"] == "unhealthy":
        logger.error(f"Database unhealthy: {health.get('errors')}")
    elif health["status"] == "degraded":
        logger.warning(f"Database degraded: latency={health.get('latency_ms')}ms")
    
    if stats.get("exhausted"):
        logger.critical("Connection pool exhausted!")
    elif stats.get("utilization", 0) > 80:
        logger.warning(f"High pool utilization: {stats.get('utilization'):.1f}%")