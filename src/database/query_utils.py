"""
Query optimization utilities and helpers
"""

import logging
from typing import List, Optional, Any, Dict, Type, Union
from datetime import datetime, timedelta

from sqlalchemy import and_, or_, not_, func, text, desc, asc
from sqlalchemy.orm import Query, Session, load_only, defer, undefer, selectinload, joinedload, subqueryload
from sqlalchemy.sql import ClauseElement
from sqlalchemy.ext.hybrid import hybrid_property

from .base import Base

logger = logging.getLogger(__name__)


class QueryOptimizer:
    """Query optimization utilities."""
    
    @staticmethod
    def optimize_for_read(query: Query, *eager_load_attrs) -> Query:
        """
        Optimize query for read operations.
        
        Args:
            query: Base query to optimize
            eager_load_attrs: Attributes to eager load
        
        Returns:
            Optimized query
        """
        # Add eager loading for specified relationships
        for attr in eager_load_attrs:
            query = query.options(selectinload(attr))
        
        return query
    
    @staticmethod
    def optimize_for_count(query: Query) -> int:
        """
        Optimize query for counting records.
        
        Args:
            query: Query to count
        
        Returns:
            Count of records
        """
        # Use func.count for better performance
        return query.with_entities(func.count()).scalar()
    
    @staticmethod
    def optimize_for_existence(query: Query) -> bool:
        """
        Optimize query for checking existence.
        
        Args:
            query: Query to check
        
        Returns:
            True if records exist
        """
        return query.limit(1).count() > 0
    
    @staticmethod
    def paginate(
        query: Query,
        page: int = 1,
        per_page: int = 20,
        max_per_page: int = 100
    ) -> Dict[str, Any]:
        """
        Paginate query results.
        
        Args:
            query: Query to paginate
            page: Page number (1-indexed)
            per_page: Items per page
            max_per_page: Maximum items per page
        
        Returns:
            Dictionary with pagination data
        """
        # Ensure per_page doesn't exceed maximum
        per_page = min(per_page, max_per_page)
        
        # Calculate offset
        offset = (page - 1) * per_page
        
        # Get total count
        total = query.count()
        
        # Get items for current page
        items = query.offset(offset).limit(per_page).all()
        
        # Calculate pagination metadata
        total_pages = (total + per_page - 1) // per_page
        has_prev = page > 1
        has_next = page < total_pages
        
        return {
            'items': items,
            'page': page,
            'per_page': per_page,
            'total': total,
            'total_pages': total_pages,
            'has_prev': has_prev,
            'has_next': has_next,
            'prev_page': page - 1 if has_prev else None,
            'next_page': page + 1 if has_next else None
        }
    
    @staticmethod
    def batch_query(
        query: Query,
        batch_size: int = 1000
    ):
        """
        Execute query in batches for large datasets.
        
        Args:
            query: Query to execute
            batch_size: Size of each batch
        
        Yields:
            Batches of results
        """
        offset = 0
        while True:
            batch = query.offset(offset).limit(batch_size).all()
            if not batch:
                break
            yield batch
            offset += batch_size
    
    @staticmethod
    def load_only_fields(
        query: Query,
        model: Type[Base],
        *fields: str
    ) -> Query:
        """
        Load only specific fields from model.
        
        Args:
            query: Base query
            model: Model class
            fields: Field names to load
        
        Returns:
            Query with limited fields
        """
        load_fields = []
        for field in fields:
            if hasattr(model, field):
                load_fields.append(getattr(model, field))
        
        if load_fields:
            return query.options(load_only(*load_fields))
        return query
    
    @staticmethod
    def defer_fields(
        query: Query,
        model: Type[Base],
        *fields: str
    ) -> Query:
        """
        Defer loading of specific fields.
        
        Args:
            query: Base query
            model: Model class
            fields: Field names to defer
        
        Returns:
            Query with deferred fields
        """
        for field in fields:
            if hasattr(model, field):
                query = query.options(defer(getattr(model, field)))
        
        return query


class FilterBuilder:
    """Build complex filters for queries."""
    
    def __init__(self):
        self.filters = []
    
    def add(self, condition: ClauseElement) -> 'FilterBuilder':
        """Add a filter condition."""
        self.filters.append(condition)
        return self
    
    def add_if(
        self,
        condition: ClauseElement,
        apply: bool
    ) -> 'FilterBuilder':
        """Add filter condition if apply is True."""
        if apply:
            self.filters.append(condition)
        return self
    
    def add_range(
        self,
        field,
        min_value: Optional[Any] = None,
        max_value: Optional[Any] = None
    ) -> 'FilterBuilder':
        """Add range filter."""
        if min_value is not None:
            self.filters.append(field >= min_value)
        if max_value is not None:
            self.filters.append(field <= max_value)
        return self
    
    def add_in(
        self,
        field,
        values: List[Any]
    ) -> 'FilterBuilder':
        """Add IN filter."""
        if values:
            self.filters.append(field.in_(values))
        return self
    
    def add_like(
        self,
        field,
        pattern: str,
        case_sensitive: bool = False
    ) -> 'FilterBuilder':
        """Add LIKE filter."""
        if pattern:
            if case_sensitive:
                self.filters.append(field.like(f"%{pattern}%"))
            else:
                self.filters.append(field.ilike(f"%{pattern}%"))
        return self
    
    def add_date_range(
        self,
        field,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> 'FilterBuilder':
        """Add date range filter."""
        if start_date:
            self.filters.append(field >= start_date)
        if end_date:
            self.filters.append(field <= end_date)
        return self
    
    def build_and(self) -> ClauseElement:
        """Build AND condition from filters."""
        if not self.filters:
            return text('1=1')  # Always true
        return and_(*self.filters)
    
    def build_or(self) -> ClauseElement:
        """Build OR condition from filters."""
        if not self.filters:
            return text('1=0')  # Always false
        return or_(*self.filters)
    
    def apply(self, query: Query) -> Query:
        """Apply filters to query."""
        if self.filters:
            return query.filter(self.build_and())
        return query


class SortBuilder:
    """Build sorting for queries."""
    
    def __init__(self):
        self.sort_fields = []
    
    def add(
        self,
        field,
        descending: bool = False
    ) -> 'SortBuilder':
        """Add sort field."""
        if descending:
            self.sort_fields.append(desc(field))
        else:
            self.sort_fields.append(asc(field))
        return self
    
    def add_dynamic(
        self,
        model: Type[Base],
        field_name: str,
        descending: bool = False
    ) -> 'SortBuilder':
        """Add sort field dynamically by name."""
        if hasattr(model, field_name):
            field = getattr(model, field_name)
            self.add(field, descending)
        return self
    
    def apply(self, query: Query) -> Query:
        """Apply sorting to query."""
        if self.sort_fields:
            return query.order_by(*self.sort_fields)
        return query


class QueryCache:
    """Simple in-memory query result cache."""
    
    def __init__(self, ttl_seconds: int = 300):
        self._cache: Dict[str, tuple] = {}
        self.ttl_seconds = ttl_seconds
    
    def _is_expired(self, timestamp: datetime) -> bool:
        """Check if cache entry is expired."""
        return datetime.utcnow() > timestamp + timedelta(seconds=self.ttl_seconds)
    
    def get(self, key: str) -> Optional[Any]:
        """Get cached result."""
        if key in self._cache:
            result, timestamp = self._cache[key]
            if not self._is_expired(timestamp):
                logger.debug(f"Cache hit for key: {key}")
                return result
            else:
                # Remove expired entry
                del self._cache[key]
                logger.debug(f"Cache expired for key: {key}")
        return None
    
    def set(self, key: str, value: Any) -> None:
        """Set cached result."""
        self._cache[key] = (value, datetime.utcnow())
        logger.debug(f"Cache set for key: {key}")
    
    def clear(self) -> None:
        """Clear all cache entries."""
        self._cache.clear()
        logger.debug("Cache cleared")
    
    def cleanup(self) -> None:
        """Remove expired entries."""
        expired_keys = [
            key for key, (_, timestamp) in self._cache.items()
            if self._is_expired(timestamp)
        ]
        for key in expired_keys:
            del self._cache[key]
        if expired_keys:
            logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")


class BulkOperations:
    """Utilities for bulk database operations."""
    
    @staticmethod
    def bulk_insert_mappings(
        session: Session,
        model: Type[Base],
        mappings: List[Dict[str, Any]],
        chunk_size: int = 1000
    ) -> int:
        """
        Bulk insert using mappings.
        
        Args:
            session: Database session
            model: Model class
            mappings: List of dictionaries to insert
            chunk_size: Size of chunks for insertion
        
        Returns:
            Number of records inserted
        """
        total_inserted = 0
        
        for i in range(0, len(mappings), chunk_size):
            chunk = mappings[i:i + chunk_size]
            try:
                session.bulk_insert_mappings(model, chunk)
                session.flush()
                total_inserted += len(chunk)
                logger.debug(f"Inserted chunk of {len(chunk)} records")
            except Exception as e:
                logger.error(f"Error inserting chunk: {e}")
                raise
        
        return total_inserted
    
    @staticmethod
    def bulk_update_mappings(
        session: Session,
        model: Type[Base],
        mappings: List[Dict[str, Any]],
        chunk_size: int = 1000
    ) -> int:
        """
        Bulk update using mappings.
        
        Args:
            session: Database session
            model: Model class
            mappings: List of dictionaries to update (must include id)
            chunk_size: Size of chunks for update
        
        Returns:
            Number of records updated
        """
        total_updated = 0
        
        for i in range(0, len(mappings), chunk_size):
            chunk = mappings[i:i + chunk_size]
            try:
                session.bulk_update_mappings(model, chunk)
                session.flush()
                total_updated += len(chunk)
                logger.debug(f"Updated chunk of {len(chunk)} records")
            except Exception as e:
                logger.error(f"Error updating chunk: {e}")
                raise
        
        return total_updated
    
    @staticmethod
    def upsert(
        session: Session,
        model: Type[Base],
        defaults: Dict[str, Any],
        **kwargs
    ) -> tuple:
        """
        Update or insert a record.
        
        Args:
            session: Database session
            model: Model class
            defaults: Default values for insert/update
            kwargs: Fields to match on
        
        Returns:
            Tuple of (instance, created)
        """
        instance = session.query(model).filter_by(**kwargs).first()
        
        if instance:
            # Update existing
            for key, value in defaults.items():
                setattr(instance, key, value)
            created = False
        else:
            # Create new
            params = {**kwargs, **defaults}
            instance = model(**params)
            session.add(instance)
            created = True
        
        return instance, created


class QueryProfiler:
    """Profile query performance."""
    
    def __init__(self):
        self.queries = []
    
    def log_query(
        self,
        query: str,
        duration: float,
        params: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log a query execution."""
        self.queries.append({
            'query': query,
            'duration': duration,
            'params': params,
            'timestamp': datetime.utcnow()
        })
    
    def get_slow_queries(self, threshold_ms: float = 100) -> List[Dict[str, Any]]:
        """Get queries slower than threshold."""
        return [
            q for q in self.queries
            if q['duration'] > threshold_ms
        ]
    
    def get_summary(self) -> Dict[str, Any]:
        """Get query profiling summary."""
        if not self.queries:
            return {
                'total_queries': 0,
                'total_time': 0,
                'avg_time': 0,
                'slowest_query': None
            }
        
        total_time = sum(q['duration'] for q in self.queries)
        avg_time = total_time / len(self.queries)
        slowest = max(self.queries, key=lambda q: q['duration'])
        
        return {
            'total_queries': len(self.queries),
            'total_time': total_time,
            'avg_time': avg_time,
            'slowest_query': slowest
        }
    
    def clear(self) -> None:
        """Clear profiling data."""
        self.queries.clear()