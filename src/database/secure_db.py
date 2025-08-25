"""
Secure Database Utilities
TEKNOFEST 2025 - SQL Injection Protected Database Layer
"""

import logging
from typing import Any, Dict, List, Optional, Type, TypeVar, Union
from datetime import datetime

from sqlalchemy import text, and_, or_, select, update, delete
from sqlalchemy.orm import Session, Query
from sqlalchemy.exc import SQLAlchemyError, IntegrityError
from sqlalchemy.sql import Select
from pydantic import BaseModel

from src.core.security import SQLInjectionProtection

logger = logging.getLogger(__name__)

T = TypeVar('T')


class SecureQueryBuilder:
    """Build secure database queries with SQL injection protection"""
    
    def __init__(self, session: Session):
        self.session = session
        self.query_log = []
    
    def _validate_column_name(self, column_name: str) -> bool:
        """Validate column name to prevent SQL injection"""
        # Only allow alphanumeric, underscore and dot (for table.column)
        import re
        pattern = r'^[a-zA-Z0-9_]+(\.[a-zA-Z0-9_]+)?$'
        return bool(re.match(pattern, column_name))
    
    def _validate_table_name(self, table_name: str) -> bool:
        """Validate table name to prevent SQL injection"""
        import re
        pattern = r'^[a-zA-Z0-9_]+$'
        return bool(re.match(pattern, table_name))
    
    def safe_select(
        self,
        model: Type[T],
        filters: Dict[str, Any] = None,
        order_by: str = None,
        limit: int = None,
        offset: int = None
    ) -> List[T]:
        """Safely select records with parameterized queries"""
        
        try:
            query = self.session.query(model)
            
            # Apply filters
            if filters:
                for key, value in filters.items():
                    if not self._validate_column_name(key):
                        raise ValueError(f"Invalid column name: {key}")
                    
                    # Validate value for SQL injection
                    if isinstance(value, str):
                        if not SQLInjectionProtection.validate_input(value):
                            raise ValueError(f"Invalid input detected in filter value")
                    
                    # Use parameterized query
                    query = query.filter(getattr(model, key) == value)
            
            # Apply ordering
            if order_by:
                if not self._validate_column_name(order_by.lstrip('-')):
                    raise ValueError(f"Invalid order by column: {order_by}")
                
                if order_by.startswith('-'):
                    query = query.order_by(getattr(model, order_by[1:]).desc())
                else:
                    query = query.order_by(getattr(model, order_by))
            
            # Apply pagination
            if limit:
                query = query.limit(min(limit, 1000))  # Max limit to prevent abuse
            
            if offset:
                query = query.offset(offset)
            
            # Log query for audit
            self.query_log.append({
                'type': 'select',
                'model': model.__name__,
                'filters': filters,
                'timestamp': datetime.utcnow()
            })
            
            return query.all()
            
        except SQLAlchemyError as e:
            logger.error(f"Database query error: {e}")
            raise
    
    def safe_insert(self, model: Type[T], data: Dict[str, Any]) -> T:
        """Safely insert record with validation"""
        
        try:
            # Validate all string values for SQL injection
            for key, value in data.items():
                if isinstance(value, str):
                    if not SQLInjectionProtection.validate_input(value):
                        raise ValueError(f"Invalid input detected in {key}")
            
            # Create instance
            instance = model(**data)
            
            # Add to session
            self.session.add(instance)
            self.session.commit()
            
            # Log operation
            self.query_log.append({
                'type': 'insert',
                'model': model.__name__,
                'timestamp': datetime.utcnow()
            })
            
            return instance
            
        except IntegrityError as e:
            self.session.rollback()
            logger.error(f"Integrity error during insert: {e}")
            raise
        except SQLAlchemyError as e:
            self.session.rollback()
            logger.error(f"Database insert error: {e}")
            raise
    
    def safe_update(
        self,
        model: Type[T],
        filters: Dict[str, Any],
        updates: Dict[str, Any]
    ) -> int:
        """Safely update records with validation"""
        
        try:
            # Validate filter columns
            for key in filters.keys():
                if not self._validate_column_name(key):
                    raise ValueError(f"Invalid column name in filter: {key}")
            
            # Validate update columns
            for key in updates.keys():
                if not self._validate_column_name(key):
                    raise ValueError(f"Invalid column name in update: {key}")
            
            # Validate all string values
            for value in list(filters.values()) + list(updates.values()):
                if isinstance(value, str):
                    if not SQLInjectionProtection.validate_input(value):
                        raise ValueError("Invalid input detected")
            
            # Build query
            query = self.session.query(model)
            
            # Apply filters
            for key, value in filters.items():
                query = query.filter(getattr(model, key) == value)
            
            # Execute update
            count = query.update(updates)
            self.session.commit()
            
            # Log operation
            self.query_log.append({
                'type': 'update',
                'model': model.__name__,
                'filters': filters,
                'updates': updates,
                'affected_rows': count,
                'timestamp': datetime.utcnow()
            })
            
            return count
            
        except SQLAlchemyError as e:
            self.session.rollback()
            logger.error(f"Database update error: {e}")
            raise
    
    def safe_delete(
        self,
        model: Type[T],
        filters: Dict[str, Any],
        soft_delete: bool = True
    ) -> int:
        """Safely delete records with validation"""
        
        try:
            # Validate filter columns
            for key in filters.keys():
                if not self._validate_column_name(key):
                    raise ValueError(f"Invalid column name: {key}")
            
            # Validate filter values
            for value in filters.values():
                if isinstance(value, str):
                    if not SQLInjectionProtection.validate_input(value):
                        raise ValueError("Invalid input detected")
            
            # Build query
            query = self.session.query(model)
            
            # Apply filters
            for key, value in filters.items():
                query = query.filter(getattr(model, key) == value)
            
            if soft_delete and hasattr(model, 'deleted_at'):
                # Soft delete - just mark as deleted
                count = query.update({'deleted_at': datetime.utcnow()})
            else:
                # Hard delete
                count = query.delete()
            
            self.session.commit()
            
            # Log operation
            self.query_log.append({
                'type': 'delete',
                'model': model.__name__,
                'filters': filters,
                'soft_delete': soft_delete,
                'affected_rows': count,
                'timestamp': datetime.utcnow()
            })
            
            return count
            
        except SQLAlchemyError as e:
            self.session.rollback()
            logger.error(f"Database delete error: {e}")
            raise
    
    def safe_raw_query(
        self,
        query: str,
        params: Dict[str, Any] = None,
        fetch_all: bool = True
    ) -> Union[List, Any]:
        """Execute raw SQL query with parameterization"""
        
        try:
            # Validate query doesn't contain obvious injections
            if not SQLInjectionProtection.validate_input(query):
                raise ValueError("Potentially dangerous SQL query detected")
            
            # Validate parameters
            if params:
                for value in params.values():
                    if isinstance(value, str):
                        if not SQLInjectionProtection.validate_input(value):
                            raise ValueError("Invalid parameter value detected")
            
            # Create parameterized query
            stmt = text(query)
            
            # Execute query
            result = self.session.execute(stmt, params or {})
            
            # Log operation
            self.query_log.append({
                'type': 'raw_query',
                'query': query[:100],  # Log first 100 chars
                'has_params': bool(params),
                'timestamp': datetime.utcnow()
            })
            
            if fetch_all:
                return result.fetchall()
            else:
                return result.fetchone()
            
        except SQLAlchemyError as e:
            logger.error(f"Raw query error: {e}")
            raise
    
    def safe_bulk_insert(
        self,
        model: Type[T],
        records: List[Dict[str, Any]],
        batch_size: int = 100
    ) -> int:
        """Safely bulk insert records with validation"""
        
        try:
            inserted = 0
            
            # Process in batches
            for i in range(0, len(records), batch_size):
                batch = records[i:i + batch_size]
                
                # Validate each record
                for record in batch:
                    for key, value in record.items():
                        if isinstance(value, str):
                            if not SQLInjectionProtection.validate_input(value):
                                raise ValueError(f"Invalid input in bulk insert: {key}")
                
                # Create instances
                instances = [model(**record) for record in batch]
                
                # Bulk insert
                self.session.bulk_save_objects(instances)
                self.session.commit()
                
                inserted += len(instances)
            
            # Log operation
            self.query_log.append({
                'type': 'bulk_insert',
                'model': model.__name__,
                'records_count': inserted,
                'timestamp': datetime.utcnow()
            })
            
            return inserted
            
        except SQLAlchemyError as e:
            self.session.rollback()
            logger.error(f"Bulk insert error: {e}")
            raise
    
    def get_query_log(self) -> List[Dict]:
        """Get query execution log for audit"""
        return self.query_log


class SecureRepository:
    """Base repository with security features"""
    
    def __init__(self, session: Session, model: Type[T]):
        self.session = session
        self.model = model
        self.query_builder = SecureQueryBuilder(session)
    
    def find_by_id(self, id: Any) -> Optional[T]:
        """Find record by ID"""
        return self.query_builder.safe_select(
            self.model,
            filters={'id': id},
            limit=1
        )[0] if id else None
    
    def find_all(
        self,
        filters: Dict[str, Any] = None,
        order_by: str = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[T]:
        """Find all records with filters"""
        return self.query_builder.safe_select(
            self.model,
            filters=filters,
            order_by=order_by,
            limit=limit,
            offset=offset
        )
    
    def create(self, data: Dict[str, Any]) -> T:
        """Create new record"""
        return self.query_builder.safe_insert(self.model, data)
    
    def update(self, id: Any, data: Dict[str, Any]) -> bool:
        """Update record by ID"""
        count = self.query_builder.safe_update(
            self.model,
            filters={'id': id},
            updates=data
        )
        return count > 0
    
    def delete(self, id: Any, soft: bool = True) -> bool:
        """Delete record by ID"""
        count = self.query_builder.safe_delete(
            self.model,
            filters={'id': id},
            soft_delete=soft
        )
        return count > 0
    
    def bulk_create(self, records: List[Dict[str, Any]]) -> int:
        """Bulk create records"""
        return self.query_builder.safe_bulk_insert(
            self.model,
            records
        )
    
    def count(self, filters: Dict[str, Any] = None) -> int:
        """Count records with filters"""
        query = self.session.query(self.model)
        
        if filters:
            for key, value in filters.items():
                query = query.filter(getattr(self.model, key) == value)
        
        return query.count()
    
    def exists(self, filters: Dict[str, Any]) -> bool:
        """Check if record exists"""
        return self.count(filters) > 0


# Audit logging decorator
def audit_database_operation(operation_type: str):
    """Decorator to audit database operations"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = datetime.utcnow()
            
            try:
                result = func(*args, **kwargs)
                
                # Log successful operation
                logger.info(f"Database operation {operation_type} completed in {(datetime.utcnow() - start_time).total_seconds():.2f}s")
                
                return result
                
            except Exception as e:
                # Log failed operation
                logger.error(f"Database operation {operation_type} failed: {e}")
                raise
        
        return wrapper
    return decorator
