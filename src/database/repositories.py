"""
Optimized Repository Pattern with Caching and Query Optimization
TEKNOFEST 2025 - High Performance Data Access Layer
"""

import logging
from typing import Any, Dict, List, Optional, Type, TypeVar, Generic, Tuple
from datetime import datetime
from abc import ABC, abstractmethod

from sqlalchemy import select, update, delete, func, and_, or_, desc, asc
from sqlalchemy.orm import Session, Query, selectinload, joinedload
from sqlalchemy.exc import SQLAlchemyError

from src.database.optimized_db import (
    get_db_session, QueryOptimizer, get_async_db_session
)
from src.core.cache import get_cache, cached, CacheKeyBuilder

logger = logging.getLogger(__name__)

T = TypeVar('T')


class BaseRepository(Generic[T], ABC):
    """Base repository with caching and optimization"""
    
    def __init__(self, model: Type[T], session: Session = None):
        self.model = model
        self.model_name = model.__name__.lower()
        self.db = session or get_db_session()
        self.cache = get_cache()
        self.optimizer = QueryOptimizer()
        
        # Cache configuration
        self.cache_ttl = 300  # 5 minutes default
        self.cache_prefix = f"repo:{self.model_name}"
    
    def _get_cache_key(self, method: str, *args, **kwargs) -> str:
        """Generate cache key for repository methods"""
        return CacheKeyBuilder.build(
            f"{self.cache_prefix}:{method}",
            *args,
            **kwargs
        )
    
    def _invalidate_cache(self):
        """Invalidate all cache for this repository"""
        pattern = f"{self.cache_prefix}:*"
        deleted = self.cache.delete_pattern(pattern)
        logger.debug(f"Invalidated {deleted} cache keys for {self.model_name}")
    
    @cached(prefix="repo:find_by_id", ttl=300)
    def find_by_id(self, id: Any, load_relationships: List[str] = None) -> Optional[T]:
        """Find entity by ID with optional eager loading"""
        with self.db.get_session() as session:
            query = session.query(self.model).filter(self.model.id == id)
            
            # Apply eager loading if specified
            if load_relationships:
                query = self.optimizer.optimize_eager_loading(query, *load_relationships)
            
            return query.first()
    
    def find_all(self,
                 filters: Dict[str, Any] = None,
                 order_by: str = None,
                 load_relationships: List[str] = None,
                 page: int = 1,
                 per_page: int = 20) -> Tuple[List[T], int]:
        """Find all with pagination and eager loading"""
        
        # Check cache first
        cache_key = self._get_cache_key(
            "find_all",
            filters=filters,
            order_by=order_by,
            page=page,
            per_page=per_page
        )
        
        cached_result = self.cache.get(cache_key)
        if cached_result:
            return cached_result
        
        with self.db.get_session() as session:
            query = session.query(self.model)
            
            # Apply filters
            if filters:
                for key, value in filters.items():
                    if hasattr(self.model, key):
                        if isinstance(value, list):
                            query = query.filter(getattr(self.model, key).in_(value))
                        elif value is None:
                            query = query.filter(getattr(self.model, key).is_(None))
                        else:
                            query = query.filter(getattr(self.model, key) == value)
            
            # Apply eager loading
            if load_relationships:
                query = self.optimizer.optimize_eager_loading(query, *load_relationships)
            
            # Apply ordering
            if order_by:
                if order_by.startswith('-'):
                    query = query.order_by(desc(getattr(self.model, order_by[1:])))
                else:
                    query = query.order_by(asc(getattr(self.model, order_by)))
            
            # Paginate
            results, total = self.optimizer.paginate(query, page, per_page)
            
            # Cache results
            self.cache.set(cache_key, (results, total), self.cache_ttl)
            
            return results, total
    
    def create(self, data: Dict[str, Any]) -> T:
        """Create new entity and invalidate cache"""
        with self.db.get_session() as session:
            instance = self.model(**data)
            session.add(instance)
            session.flush()  # Get ID without committing
            
            # Invalidate cache after create
            self._invalidate_cache()
            
            return instance
    
    def bulk_create(self, records: List[Dict[str, Any]], batch_size: int = 1000) -> int:
        """Bulk create with optimized performance"""
        count = self.db.bulk_insert(self.model, records, batch_size)
        
        # Invalidate cache after bulk create
        self._invalidate_cache()
        
        return count
    
    def update(self, id: Any, data: Dict[str, Any]) -> Optional[T]:
        """Update entity and invalidate cache"""
        with self.db.get_session() as session:
            instance = session.query(self.model).filter(self.model.id == id).first()
            
            if not instance:
                return None
            
            for key, value in data.items():
                if hasattr(instance, key):
                    setattr(instance, key, value)
            
            session.flush()
            
            # Invalidate cache after update
            self._invalidate_cache()
            
            return instance
    
    def bulk_update(self, updates: List[Dict[str, Any]], batch_size: int = 500) -> int:
        """Bulk update with optimized performance"""
        count = self.db.bulk_update(self.model, updates, batch_size)
        
        # Invalidate cache after bulk update
        self._invalidate_cache()
        
        return count
    
    def delete(self, id: Any, soft_delete: bool = True) -> bool:
        """Delete entity and invalidate cache"""
        with self.db.get_session() as session:
            instance = session.query(self.model).filter(self.model.id == id).first()
            
            if not instance:
                return False
            
            if soft_delete and hasattr(instance, 'deleted_at'):
                instance.deleted_at = datetime.utcnow()
            else:
                session.delete(instance)
            
            # Invalidate cache after delete
            self._invalidate_cache()
            
            return True
    
    def count(self, filters: Dict[str, Any] = None) -> int:
        """Count entities with optional filters"""
        # Check cache
        cache_key = self._get_cache_key("count", filters=filters)
        cached_count = self.cache.get(cache_key)
        
        if cached_count is not None:
            return cached_count
        
        with self.db.get_session() as session:
            query = session.query(func.count(self.model.id))
            
            if filters:
                for key, value in filters.items():
                    if hasattr(self.model, key):
                        query = query.filter(getattr(self.model, key) == value)
            
            count = query.scalar()
            
            # Cache count
            self.cache.set(cache_key, count, self.cache_ttl)
            
            return count
    
    def exists(self, filters: Dict[str, Any]) -> bool:
        """Check if entity exists"""
        return self.count(filters) > 0
    
    def find_by_attributes(self, **kwargs) -> List[T]:
        """Find by multiple attributes with caching"""
        cache_key = self._get_cache_key("find_by_attributes", **kwargs)
        cached_result = self.cache.get(cache_key)
        
        if cached_result:
            return cached_result
        
        with self.db.get_session() as session:
            query = session.query(self.model)
            
            for key, value in kwargs.items():
                if hasattr(self.model, key):
                    query = query.filter(getattr(self.model, key) == value)
            
            results = query.all()
            
            # Cache results
            self.cache.set(cache_key, results, self.cache_ttl)
            
            return results
    
    def search(self, 
               search_term: str,
               search_fields: List[str],
               filters: Dict[str, Any] = None,
               page: int = 1,
               per_page: int = 20) -> Tuple[List[T], int]:
        """Full-text search with pagination"""
        
        with self.db.get_session() as session:
            query = session.query(self.model)
            
            # Apply search conditions
            if search_term and search_fields:
                search_conditions = []
                for field in search_fields:
                    if hasattr(self.model, field):
                        search_conditions.append(
                            getattr(self.model, field).ilike(f"%{search_term}%")
                        )
                
                if search_conditions:
                    query = query.filter(or_(*search_conditions))
            
            # Apply additional filters
            if filters:
                for key, value in filters.items():
                    if hasattr(self.model, key):
                        query = query.filter(getattr(self.model, key) == value)
            
            # Paginate
            return self.optimizer.paginate(query, page, per_page)
    
    def aggregate(self, 
                  aggregations: Dict[str, str],
                  filters: Dict[str, Any] = None,
                  group_by: List[str] = None) -> List[Dict]:
        """Perform aggregations with optional grouping"""
        
        with self.db.get_session() as session:
            # Build aggregation query
            agg_fields = []
            
            for alias, expr in aggregations.items():
                if expr.startswith('count'):
                    agg_fields.append(func.count(self.model.id).label(alias))
                elif expr.startswith('sum'):
                    field = expr.split(':')[1]
                    agg_fields.append(func.sum(getattr(self.model, field)).label(alias))
                elif expr.startswith('avg'):
                    field = expr.split(':')[1]
                    agg_fields.append(func.avg(getattr(self.model, field)).label(alias))
                elif expr.startswith('max'):
                    field = expr.split(':')[1]
                    agg_fields.append(func.max(getattr(self.model, field)).label(alias))
                elif expr.startswith('min'):
                    field = expr.split(':')[1]
                    agg_fields.append(func.min(getattr(self.model, field)).label(alias))
            
            # Add group by fields
            if group_by:
                for field in group_by:
                    if hasattr(self.model, field):
                        agg_fields.append(getattr(self.model, field))
            
            query = session.query(*agg_fields)
            
            # Apply filters
            if filters:
                for key, value in filters.items():
                    if hasattr(self.model, key):
                        query = query.filter(getattr(self.model, key) == value)
            
            # Apply group by
            if group_by:
                for field in group_by:
                    if hasattr(self.model, field):
                        query = query.group_by(getattr(self.model, field))
            
            # Execute and format results
            results = query.all()
            
            return [dict(row._mapping) for row in results]
    
    async def find_by_id_async(self, id: Any) -> Optional[T]:
        """Async find by ID"""
        async_db = get_async_db_session()
        
        async with async_db.async_session() as session:
            query = select(self.model).filter(self.model.id == id)
            result = await session.execute(query)
            return result.scalar_one_or_none()
    
    async def create_async(self, data: Dict[str, Any]) -> T:
        """Async create"""
        async_db = get_async_db_session()
        
        async with async_db.async_session() as session:
            instance = self.model(**data)
            session.add(instance)
            await session.commit()
            
            # Invalidate cache
            self._invalidate_cache()
            
            return instance


class StudentRepository(BaseRepository):
    """Optimized Student repository"""
    
    def __init__(self):
        # Assuming Student model exists
        # super().__init__(Student)
        pass
    
    def find_active_students(self, grade: int = None) -> List:
        """Find active students with optimized query"""
        filters = {'is_active': True}
        if grade:
            filters['grade'] = grade
        
        # Load relationships to prevent N+1
        return self.find_all(
            filters=filters,
            load_relationships=['profile', 'enrollments.course']
        )[0]
    
    def get_student_with_progress(self, student_id: int):
        """Get student with all progress data - optimized"""
        # Use single query with eager loading
        with self.db.get_session() as session:
            query = session.query(self.model)\
                .filter(self.model.id == student_id)\
                .options(
                    selectinload('profile'),
                    selectinload('learning_paths').selectinload('milestones'),
                    selectinload('quiz_attempts').selectinload('answers'),
                    selectinload('achievements')
                )
            
            return query.first()


class CourseRepository(BaseRepository):
    """Optimized Course repository"""
    
    def get_popular_courses(self, limit: int = 10) -> List:
        """Get popular courses with caching"""
        cache_key = f"popular_courses:{limit}"
        cached = self.cache.get(cache_key)
        
        if cached:
            return cached
        
        with self.db.get_session() as session:
            # Optimized query with aggregation
            query = session.query(
                self.model,
                func.count(Enrollment.id).label('enrollment_count')
            ).join(Enrollment)\
             .group_by(self.model.id)\
             .order_by(desc('enrollment_count'))\
             .limit(limit)
            
            results = query.all()
            
            # Cache for 1 hour
            self.cache.set(cache_key, results, ttl=3600)
            
            return results
