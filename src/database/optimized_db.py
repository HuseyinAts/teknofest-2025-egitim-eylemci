"""
Optimized Database Layer with Connection Pooling and Query Optimization
TEKNOFEST 2025 - High Performance Database Operations
"""

import logging
import time
from typing import Any, Dict, List, Optional, Type, TypeVar, Union, Tuple
from contextlib import contextmanager
from datetime import datetime
import asyncio

from sqlalchemy import create_engine, event, pool, text, select, and_, or_
from sqlalchemy.orm import (
    Session, sessionmaker, scoped_session,
    Query, joinedload, selectinload, subqueryload,
    contains_eager, lazyload, noload
)
from sqlalchemy.ext.asyncio import (
    AsyncSession, create_async_engine, async_sessionmaker
)
from sqlalchemy.pool import QueuePool, NullPool, StaticPool
from sqlalchemy.exc import SQLAlchemyError, OperationalError, IntegrityError
from sqlalchemy.engine import Engine

from src.config import get_settings
from src.core.cache import get_cache, cached

logger = logging.getLogger(__name__)

T = TypeVar('T')


class DatabaseConfig:
    """Database configuration with optimized settings"""
    
    def __init__(self):
        self.settings = get_settings()
        
        # Connection pool settings
        self.pool_size = self.settings.database_pool_size
        self.max_overflow = self.settings.database_max_overflow
        self.pool_timeout = self.settings.database_pool_timeout
        self.pool_recycle = self.settings.database_pool_recycle
        self.pool_pre_ping = self.settings.database_pool_pre_ping
        
        # Performance settings
        self.echo = self.settings.database_echo
        self.echo_pool = self.settings.database_echo_pool
        self.statement_timeout = self.settings.database_statement_timeout
        self.lock_timeout = self.settings.database_lock_timeout
        
    def get_engine_kwargs(self) -> Dict[str, Any]:
        """Get optimized engine configuration"""
        return {
            'pool_size': self.pool_size,
            'max_overflow': self.max_overflow,
            'pool_timeout': self.pool_timeout,
            'pool_recycle': self.pool_recycle,
            'pool_pre_ping': self.pool_pre_ping,
            'echo': self.echo,
            'echo_pool': self.echo_pool,
            'pool_class': QueuePool,  # Most efficient for production
            'connect_args': {
                'connect_timeout': 10,
                'command_timeout': self.statement_timeout / 1000,
                'options': f'-c statement_timeout={self.statement_timeout} -c lock_timeout={self.lock_timeout}'
            }
        }


class PerformanceMonitor:
    """Monitor database query performance"""
    
    def __init__(self):
        self.query_times = []
        self.slow_queries = []
        self.query_count = 0
        self.slow_query_threshold = 0.1  # 100ms
    
    def record_query(self, query: str, duration: float):
        """Record query execution time"""
        self.query_count += 1
        self.query_times.append(duration)
        
        if duration > self.slow_query_threshold:
            self.slow_queries.append({
                'query': query[:200],  # First 200 chars
                'duration': duration,
                'timestamp': datetime.utcnow()
            })
            logger.warning(f"Slow query detected ({duration:.3f}s): {query[:100]}...")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get performance statistics"""
        if not self.query_times:
            return {'query_count': 0}
        
        return {
            'query_count': self.query_count,
            'avg_time': sum(self.query_times) / len(self.query_times),
            'min_time': min(self.query_times),
            'max_time': max(self.query_times),
            'slow_queries': len(self.slow_queries),
            'total_time': sum(self.query_times)
        }


class OptimizedSession:
    """Optimized database session with performance features"""
    
    def __init__(self, database_url: str = None):
        self.config = DatabaseConfig()
        self.database_url = database_url or self.config.settings.database_url
        self.monitor = PerformanceMonitor()
        self.cache = get_cache()
        
        # Create engine with optimized settings
        self.engine = self._create_engine()
        
        # Session factory with optimizations
        self.SessionLocal = sessionmaker(
            bind=self.engine,
            autocommit=False,
            autoflush=False,  # Control flushing manually for performance
            expire_on_commit=False,  # Don't expire objects after commit
            class_=Session
        )
        
        # Scoped session for thread safety
        self.session_factory = scoped_session(self.SessionLocal)
        
        # Setup event listeners
        self._setup_event_listeners()
    
    def _create_engine(self) -> Engine:
        """Create optimized database engine"""
        engine_kwargs = self.config.get_engine_kwargs()
        
        # SQLite optimizations
        if 'sqlite' in self.database_url:
            engine_kwargs['connect_args'] = {
                'check_same_thread': False,
                'timeout': 15
            }
            engine_kwargs['pool_class'] = StaticPool
            
            engine = create_engine(self.database_url, **engine_kwargs)
            
            # SQLite performance pragmas
            @event.listens_for(engine, "connect")
            def set_sqlite_pragma(dbapi_conn, connection_record):
                cursor = dbapi_conn.cursor()
                cursor.execute("PRAGMA journal_mode=WAL")  # Write-Ahead Logging
                cursor.execute("PRAGMA synchronous=NORMAL")  # Faster writes
                cursor.execute("PRAGMA cache_size=10000")  # Larger cache
                cursor.execute("PRAGMA temp_store=MEMORY")  # Use memory for temp tables
                cursor.execute("PRAGMA mmap_size=30000000000")  # Memory-mapped I/O
                cursor.close()
        else:
            # PostgreSQL/MySQL optimizations
            engine = create_engine(self.database_url, **engine_kwargs)
        
        return engine
    
    def _setup_event_listeners(self):
        """Setup performance monitoring event listeners"""
        
        @event.listens_for(self.engine, "before_execute")
        def before_execute(conn, clauseelement, multiparams, params, execution_options):
            conn.info['query_start_time'] = time.time()
        
        @event.listens_for(self.engine, "after_execute")
        def after_execute(conn, clauseelement, multiparams, params, execution_options, result):
            duration = time.time() - conn.info.get('query_start_time', time.time())
            self.monitor.record_query(str(clauseelement), duration)
    
    @contextmanager
    def get_session(self) -> Session:
        """Get optimized database session"""
        session = self.session_factory()
        
        try:
            # Enable query batching
            session.bulk_insert_mappings = True
            session.bulk_save_objects = True
            
            yield session
            session.commit()
        except SQLAlchemyError as e:
            session.rollback()
            logger.error(f"Database error: {e}")
            raise
        finally:
            session.close()
    
    def bulk_insert(self, model: Type[T], records: List[Dict[str, Any]], batch_size: int = 1000) -> int:
        """Optimized bulk insert"""
        inserted = 0
        
        with self.get_session() as session:
            for i in range(0, len(records), batch_size):
                batch = records[i:i + batch_size]
                
                # Use bulk_insert_mappings for best performance
                session.bulk_insert_mappings(model, batch)
                session.flush()  # Flush in batches
                
                inserted += len(batch)
                logger.info(f"Inserted batch {i//batch_size + 1}: {len(batch)} records")
        
        return inserted
    
    def bulk_update(self, model: Type[T], records: List[Dict[str, Any]], batch_size: int = 500) -> int:
        """Optimized bulk update"""
        updated = 0
        
        with self.get_session() as session:
            for i in range(0, len(records), batch_size):
                batch = records[i:i + batch_size]
                
                # Use bulk_update_mappings for best performance
                session.bulk_update_mappings(model, batch)
                session.flush()
                
                updated += len(batch)
        
        return updated
    
    def execute_raw(self, query: str, params: Dict[str, Any] = None) -> Any:
        """Execute raw SQL with performance monitoring"""
        with self.get_session() as session:
            start = time.time()
            
            result = session.execute(text(query), params or {})
            
            duration = time.time() - start
            self.monitor.record_query(query, duration)
            
            return result.fetchall()
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get database performance statistics"""
        pool_status = {
            'size': self.engine.pool.size(),
            'checked_in': self.engine.pool.checkedin(),
            'checked_out': self.engine.pool.checkedout(),
            'overflow': self.engine.pool.overflow(),
            'total': self.engine.pool.total()
        }
        
        return {
            'pool_status': pool_status,
            'query_stats': self.monitor.get_statistics(),
            'cache_stats': self.cache.get_metrics()
        }


class QueryOptimizer:
    """Query optimization strategies to prevent N+1 and improve performance"""
    
    @staticmethod
    def optimize_eager_loading(query: Query, *relationships) -> Query:
        """Add eager loading to prevent N+1 queries"""
        for relationship in relationships:
            if '.' in relationship:
                # Nested relationship
                parts = relationship.split('.')
                query = query.options(
                    selectinload(parts[0]).selectinload(parts[1])
                )
            else:
                # Direct relationship
                query = query.options(selectinload(relationship))
        
        return query
    
    @staticmethod
    def optimize_join_loading(query: Query, *relationships) -> Query:
        """Use join loading for one-to-one relationships"""
        for relationship in relationships:
            query = query.options(joinedload(relationship))
        
        return query
    
    @staticmethod
    def optimize_subquery_loading(query: Query, *relationships) -> Query:
        """Use subquery loading for collections"""
        for relationship in relationships:
            query = query.options(subqueryload(relationship))
        
        return query
    
    @staticmethod
    def paginate(query: Query, page: int = 1, per_page: int = 20) -> Tuple[List, int]:
        """Optimized pagination"""
        # Get total count efficiently
        total = query.count()
        
        # Get paginated results
        results = query.limit(per_page).offset((page - 1) * per_page).all()
        
        return results, total
    
    @staticmethod
    @cached(prefix="query_result", ttl=300)
    def cached_query(session: Session, model: Type[T], filters: Dict[str, Any]) -> List[T]:
        """Cached query execution"""
        query = session.query(model)
        
        for key, value in filters.items():
            query = query.filter(getattr(model, key) == value)
        
        return query.all()


class AsyncDatabaseSession:
    """Async database session for high-performance async operations"""
    
    def __init__(self, database_url: str = None):
        self.config = DatabaseConfig()
        settings = get_settings()
        
        # Convert to async URL
        if database_url:
            self.database_url = database_url
        elif 'sqlite' in settings.database_url:
            self.database_url = settings.database_url.replace('sqlite:///', 'sqlite+aiosqlite:///')
        else:
            self.database_url = settings.database_url.replace('postgresql://', 'postgresql+asyncpg://')
        
        # Create async engine
        self.engine = create_async_engine(
            self.database_url,
            pool_size=self.config.pool_size,
            max_overflow=self.config.max_overflow,
            pool_timeout=self.config.pool_timeout,
            pool_recycle=self.config.pool_recycle,
            pool_pre_ping=self.config.pool_pre_ping,
            echo=self.config.echo
        )
        
        # Async session factory
        self.async_session = async_sessionmaker(
            self.engine,
            class_=AsyncSession,
            expire_on_commit=False
        )
    
    async def execute(self, query: str, params: Dict[str, Any] = None) -> Any:
        """Execute async query"""
        async with self.async_session() as session:
            result = await session.execute(text(query), params or {})
            return result.fetchall()
    
    async def bulk_insert_async(self, model: Type[T], records: List[Dict[str, Any]]) -> int:
        """Async bulk insert"""
        async with self.async_session() as session:
            # Convert to model instances
            instances = [model(**record) for record in records]
            
            session.add_all(instances)
            await session.commit()
            
            return len(instances)
    
    async def get_or_create(self, model: Type[T], defaults: Dict[str, Any], **kwargs) -> Tuple[T, bool]:
        """Get or create pattern with async"""
        async with self.async_session() as session:
            # Try to get existing
            query = select(model).filter_by(**kwargs)
            result = await session.execute(query)
            instance = result.scalar_one_or_none()
            
            if instance:
                return instance, False
            
            # Create new
            instance = model(**kwargs, **defaults)
            session.add(instance)
            await session.commit()
            
            return instance, True


# Global instances
_db_session = None
_async_session = None

def get_db_session() -> OptimizedSession:
    """Get optimized database session"""
    global _db_session
    
    if _db_session is None:
        _db_session = OptimizedSession()
    
    return _db_session

def get_async_db_session() -> AsyncDatabaseSession:
    """Get async database session"""
    global _async_session
    
    if _async_session is None:
        _async_session = AsyncDatabaseSession()
    
    return _async_session
