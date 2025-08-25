"""
Database connection management with security and pooling
Production-ready connection management system
"""

import os
import ssl
import logging
from typing import Optional, Dict, Any, List
from contextlib import contextmanager
from datetime import datetime, timedelta
import threading
import time
from urllib.parse import urlparse, parse_qs

from sqlalchemy import create_engine, event, pool, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import DBAPIError, OperationalError
from sqlalchemy.pool import QueuePool, NullPool, StaticPool
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_READ_COMMITTED

from ..config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class SecureConnectionManager:
    """
    Manages secure database connections with advanced pooling and monitoring.
    """
    
    def __init__(self):
        self.engines: Dict[str, Engine] = {}
        self.connection_stats: Dict[str, Dict[str, Any]] = {}
        self.lock = threading.Lock()
        self._monitoring_thread = None
        self._stop_monitoring = threading.Event()
        
        # SSL configuration
        self.ssl_config = self._get_ssl_config()
        
        # Connection pool configuration
        self.pool_config = self._get_pool_config()
    
    def _get_ssl_config(self) -> Dict[str, Any]:
        """Get SSL configuration for secure connections."""
        ssl_config = {}
        
        if settings.is_production():
            # Production SSL settings
            ssl_config = {
                'sslmode': 'require',  # or 'verify-full' for maximum security
                'sslcert': os.getenv('DATABASE_SSL_CERT', None),
                'sslkey': os.getenv('DATABASE_SSL_KEY', None),
                'sslrootcert': os.getenv('DATABASE_SSL_ROOT_CERT', None),
            }
            
            # Remove None values
            ssl_config = {k: v for k, v in ssl_config.items() if v is not None}
        
        return ssl_config
    
    def _get_pool_config(self) -> Dict[str, Any]:
        """Get connection pool configuration based on environment."""
        if settings.is_production():
            return {
                'poolclass': QueuePool,
                'pool_size': settings.database_pool_size or 20,
                'max_overflow': settings.database_max_overflow or 40,
                'pool_timeout': 30,
                'pool_recycle': 3600,  # Recycle connections after 1 hour
                'pool_pre_ping': True,  # Test connections before using
                'echo_pool': False,
            }
        elif settings.app_env.value == 'testing':
            return {
                'poolclass': StaticPool,
                'connect_args': {'check_same_thread': False}
            }
        else:
            # Development settings
            return {
                'poolclass': QueuePool,
                'pool_size': 5,
                'max_overflow': 10,
                'pool_timeout': 30,
                'pool_recycle': 3600,
                'pool_pre_ping': True,
                'echo_pool': True,  # Log pool checkouts/checkins in development
            }
    
    def get_engine(
        self, 
        database_url: Optional[str] = None,
        name: str = 'default',
        **kwargs
    ) -> Engine:
        """
        Get or create a database engine with secure connection.
        
        Args:
            database_url: Database URL (uses settings if not provided)
            name: Engine name for caching
            **kwargs: Additional engine arguments
        
        Returns:
            SQLAlchemy Engine instance
        """
        if name in self.engines:
            return self.engines[name]
        
        with self.lock:
            # Double-check after acquiring lock
            if name in self.engines:
                return self.engines[name]
            
            # Get database URL
            url = database_url or settings.database_url
            
            # Parse and secure the URL
            url = self._secure_database_url(url)
            
            # Create engine arguments
            engine_args = {
                **self.pool_config,
                **kwargs
            }
            
            # Add SSL configuration for PostgreSQL
            if 'postgresql' in url:
                connect_args = engine_args.get('connect_args', {})
                connect_args.update(self.ssl_config)
                engine_args['connect_args'] = connect_args
            
            # Add execution options
            engine_args['execution_options'] = {
                'isolation_level': 'READ_COMMITTED',
            }
            
            # Create engine
            engine = create_engine(url, **engine_args)
            
            # Set up event listeners
            self._setup_engine_events(engine, name)
            
            # Initialize connection stats
            self.connection_stats[name] = {
                'created_at': datetime.now(),
                'total_connections': 0,
                'active_connections': 0,
                'failed_connections': 0,
                'last_error': None
            }
            
            # Store engine
            self.engines[name] = engine
            
            logger.info(f"Created database engine: {name}")
            return engine
    
    def _secure_database_url(self, url: str) -> str:
        """
        Secure and validate database URL.
        
        Args:
            url: Original database URL
        
        Returns:
            Secured database URL
        """
        # Parse URL
        parsed = urlparse(url)
        
        # Validate scheme
        allowed_schemes = ['postgresql', 'postgresql+psycopg2', 'postgresql+asyncpg']
        if not any(parsed.scheme.startswith(s) for s in allowed_schemes):
            logger.warning(f"Unusual database scheme: {parsed.scheme}")
        
        # Add connection parameters for PostgreSQL
        if 'postgresql' in parsed.scheme:
            # Parse existing query parameters
            params = parse_qs(parsed.query)
            
            # Add security parameters
            if settings.is_production():
                params.setdefault('sslmode', ['require'])
                params.setdefault('connect_timeout', ['10'])
                params.setdefault('application_name', ['teknofest_app'])
            
            # Reconstruct query string
            query_parts = []
            for key, values in params.items():
                for value in values:
                    query_parts.append(f"{key}={value}")
            
            # Reconstruct URL with parameters
            if query_parts:
                url = url.split('?')[0] + '?' + '&'.join(query_parts)
        
        return url
    
    def _setup_engine_events(self, engine: Engine, name: str):
        """Set up event listeners for engine monitoring."""
        
        @event.listens_for(engine, "connect")
        def receive_connect(dbapi_conn, connection_record):
            """Handle new connection creation."""
            connection_record.info['connect_time'] = datetime.now()
            self.connection_stats[name]['total_connections'] += 1
            self.connection_stats[name]['active_connections'] += 1
            
            # Set connection parameters for PostgreSQL
            if hasattr(dbapi_conn, 'set_isolation_level'):
                dbapi_conn.set_isolation_level(ISOLATION_LEVEL_READ_COMMITTED)
            
            logger.debug(f"New connection created for engine: {name}")
        
        @event.listens_for(engine, "checkout")
        def receive_checkout(dbapi_conn, connection_record, connection_proxy):
            """Handle connection checkout from pool."""
            # Test connection if it's been idle for too long
            if 'connect_time' in connection_record.info:
                age = datetime.now() - connection_record.info['connect_time']
                if age > timedelta(hours=1):
                    # Connection is old, test it
                    try:
                        cursor = dbapi_conn.cursor()
                        cursor.execute("SELECT 1")
                        cursor.close()
                    except Exception:
                        # Connection is dead, invalidate it
                        connection_proxy.invalidate()
                        raise
        
        @event.listens_for(engine, "checkin")
        def receive_checkin(dbapi_conn, connection_record):
            """Handle connection checkin to pool."""
            pass
        
        @event.listens_for(engine, "close")
        def receive_close(dbapi_conn, connection_record):
            """Handle connection close."""
            if name in self.connection_stats:
                self.connection_stats[name]['active_connections'] -= 1
            logger.debug(f"Connection closed for engine: {name}")
    
    @contextmanager
    def get_connection(self, engine_name: str = 'default', **kwargs):
        """
        Get a database connection with automatic cleanup.
        
        Args:
            engine_name: Name of engine to use
            **kwargs: Additional connection arguments
        
        Yields:
            Database connection
        """
        engine = self.get_engine(name=engine_name)
        conn = None
        
        try:
            conn = engine.connect(**kwargs)
            yield conn
        except OperationalError as e:
            # Handle connection errors
            self.connection_stats[engine_name]['failed_connections'] += 1
            self.connection_stats[engine_name]['last_error'] = str(e)
            logger.error(f"Database connection error: {e}")
            raise
        finally:
            if conn:
                conn.close()
    
    def test_connection(self, engine_name: str = 'default') -> bool:
        """
        Test database connection.
        
        Args:
            engine_name: Name of engine to test
        
        Returns:
            True if connection is successful
        """
        try:
            with self.get_connection(engine_name) as conn:
                result = conn.execute(text("SELECT 1"))
                return result.scalar() == 1
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False
    
    def get_pool_status(self, engine_name: str = 'default') -> Dict[str, Any]:
        """
        Get connection pool status.
        
        Args:
            engine_name: Name of engine
        
        Returns:
            Pool status information
        """
        if engine_name not in self.engines:
            return {}
        
        engine = self.engines[engine_name]
        pool = engine.pool
        
        status = {
            'size': getattr(pool, 'size', 0),
            'checked_in': getattr(pool, 'checkedin', 0),
            'overflow': getattr(pool, 'overflow', 0),
            'total': getattr(pool, 'total', 0),
        }
        
        # Add connection stats
        if engine_name in self.connection_stats:
            status.update(self.connection_stats[engine_name])
        
        return status
    
    def close_all_connections(self):
        """Close all database connections and engines."""
        for name, engine in self.engines.items():
            try:
                engine.dispose()
                logger.info(f"Closed engine: {name}")
            except Exception as e:
                logger.error(f"Error closing engine {name}: {e}")
        
        self.engines.clear()
        self.connection_stats.clear()
    
    def start_monitoring(self, interval_seconds: int = 60):
        """
        Start connection pool monitoring.
        
        Args:
            interval_seconds: Monitoring interval
        """
        if self._monitoring_thread and self._monitoring_thread.is_alive():
            return
        
        self._stop_monitoring.clear()
        self._monitoring_thread = threading.Thread(
            target=self._monitor_connections,
            args=(interval_seconds,),
            daemon=True
        )
        self._monitoring_thread.start()
        logger.info("Started connection pool monitoring")
    
    def stop_monitoring(self):
        """Stop connection pool monitoring."""
        self._stop_monitoring.set()
        if self._monitoring_thread:
            self._monitoring_thread.join(timeout=5)
        logger.info("Stopped connection pool monitoring")
    
    def _monitor_connections(self, interval: int):
        """Monitor connection pools and log statistics."""
        while not self._stop_monitoring.is_set():
            try:
                for engine_name in list(self.engines.keys()):
                    status = self.get_pool_status(engine_name)
                    
                    # Log pool status
                    logger.info(
                        f"Pool [{engine_name}] - "
                        f"Size: {status.get('size', 0)}, "
                        f"Checked In: {status.get('checked_in', 0)}, "
                        f"Active: {status.get('active_connections', 0)}, "
                        f"Failed: {status.get('failed_connections', 0)}"
                    )
                    
                    # Check for pool exhaustion
                    if status.get('checked_in', 0) == 0 and status.get('size', 0) > 0:
                        logger.warning(f"Connection pool [{engine_name}] may be exhausted")
                    
                    # Check for high failure rate
                    total = status.get('total_connections', 1)
                    failed = status.get('failed_connections', 0)
                    if failed > 0 and (failed / total) > 0.1:
                        logger.warning(
                            f"High connection failure rate [{engine_name}]: "
                            f"{failed}/{total} ({failed/total*100:.1f}%)"
                        )
            
            except Exception as e:
                logger.error(f"Error in connection monitoring: {e}")
            
            # Wait for next interval
            self._stop_monitoring.wait(interval)


class ConnectionPoolManager:
    """
    Manages multiple connection pools for different workloads.
    """
    
    def __init__(self):
        self.pools: Dict[str, Engine] = {}
        self.connection_manager = SecureConnectionManager()
    
    def create_pool(
        self,
        name: str,
        pool_size: int = 10,
        max_overflow: int = 20,
        pool_type: str = 'default'
    ) -> Engine:
        """
        Create a named connection pool.
        
        Args:
            name: Pool name
            pool_size: Number of persistent connections
            max_overflow: Maximum overflow connections
            pool_type: Type of pool (default, read, write)
        
        Returns:
            Engine with configured pool
        """
        pool_config = {
            'pool_size': pool_size,
            'max_overflow': max_overflow,
        }
        
        # Configure based on pool type
        if pool_type == 'read':
            # Read-heavy workload
            pool_config.update({
                'pool_recycle': 7200,  # 2 hours
                'pool_timeout': 10,
            })
        elif pool_type == 'write':
            # Write-heavy workload
            pool_config.update({
                'pool_recycle': 1800,  # 30 minutes
                'pool_timeout': 30,
                'pool_pre_ping': True,
            })
        
        engine = self.connection_manager.get_engine(
            name=name,
            **pool_config
        )
        
        self.pools[name] = engine
        return engine
    
    def get_pool(self, name: str) -> Optional[Engine]:
        """Get a named connection pool."""
        return self.pools.get(name)
    
    def distribute_load(self, operation_type: str = 'read') -> Engine:
        """
        Distribute database load across pools.
        
        Args:
            operation_type: Type of operation (read/write)
        
        Returns:
            Appropriate engine for the operation
        """
        if operation_type == 'read' and 'read_pool' in self.pools:
            return self.pools['read_pool']
        elif operation_type == 'write' and 'write_pool' in self.pools:
            return self.pools['write_pool']
        
        # Default pool
        return self.connection_manager.get_engine()


# Global instances
_connection_manager = None
_pool_manager = None


def get_connection_manager() -> SecureConnectionManager:
    """Get global connection manager instance."""
    global _connection_manager
    if _connection_manager is None:
        _connection_manager = SecureConnectionManager()
        if settings.is_production():
            _connection_manager.start_monitoring()
    return _connection_manager


def get_pool_manager() -> ConnectionPoolManager:
    """Get global pool manager instance."""
    global _pool_manager
    if _pool_manager is None:
        _pool_manager = ConnectionPoolManager()
    return _pool_manager