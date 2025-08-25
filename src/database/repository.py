"""
Generic repository pattern for database operations
"""

import logging
from typing import TypeVar, Generic, Type, Optional, List, Dict, Any, Union
from uuid import UUID
from datetime import datetime

from sqlalchemy import select, update, delete, func, and_, or_
from sqlalchemy.orm import Session, selectinload, joinedload, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.exc import IntegrityError, SQLAlchemyError

from .base import Base

logger = logging.getLogger(__name__)

ModelType = TypeVar("ModelType", bound=Base)


class BaseRepository(Generic[ModelType]):
    """
    Base repository providing CRUD operations for all models.
    """
    
    def __init__(self, model: Type[ModelType], session: Session):
        self.model = model
        self.session = session
    
    def get(self, id: Union[UUID, int]) -> Optional[ModelType]:
        """Get a single record by ID."""
        try:
            return self.session.query(self.model).filter(self.model.id == id).first()
        except SQLAlchemyError as e:
            logger.error(f"Error getting {self.model.__name__} with id {id}: {e}")
            raise
    
    def get_by(self, **kwargs) -> Optional[ModelType]:
        """Get a single record by field values."""
        try:
            return self.session.query(self.model).filter_by(**kwargs).first()
        except SQLAlchemyError as e:
            logger.error(f"Error getting {self.model.__name__} by {kwargs}: {e}")
            raise
    
    def get_all(
        self,
        skip: int = 0,
        limit: int = 100,
        order_by: Optional[str] = None,
        **filters
    ) -> List[ModelType]:
        """Get all records with optional pagination and filtering."""
        try:
            query = self.session.query(self.model)
            
            # Apply filters
            if filters:
                query = query.filter_by(**filters)
            
            # Apply ordering
            if order_by:
                if order_by.startswith('-'):
                    query = query.order_by(getattr(self.model, order_by[1:]).desc())
                else:
                    query = query.order_by(getattr(self.model, order_by))
            
            # Apply pagination
            query = query.offset(skip).limit(limit)
            
            return query.all()
        except SQLAlchemyError as e:
            logger.error(f"Error getting all {self.model.__name__}: {e}")
            raise
    
    def count(self, **filters) -> int:
        """Count records with optional filtering."""
        try:
            query = self.session.query(func.count(self.model.id))
            if filters:
                query = query.filter_by(**filters)
            return query.scalar()
        except SQLAlchemyError as e:
            logger.error(f"Error counting {self.model.__name__}: {e}")
            raise
    
    def create(self, **data) -> ModelType:
        """Create a new record."""
        try:
            instance = self.model(**data)
            self.session.add(instance)
            self.session.commit()
            self.session.refresh(instance)
            return instance
        except IntegrityError as e:
            self.session.rollback()
            logger.error(f"Integrity error creating {self.model.__name__}: {e}")
            raise
        except SQLAlchemyError as e:
            self.session.rollback()
            logger.error(f"Error creating {self.model.__name__}: {e}")
            raise
    
    def update(self, id: Union[UUID, int], **data) -> Optional[ModelType]:
        """Update a record by ID."""
        try:
            instance = self.get(id)
            if not instance:
                return None
            
            for key, value in data.items():
                if hasattr(instance, key):
                    setattr(instance, key, value)
            
            self.session.commit()
            self.session.refresh(instance)
            return instance
        except IntegrityError as e:
            self.session.rollback()
            logger.error(f"Integrity error updating {self.model.__name__}: {e}")
            raise
        except SQLAlchemyError as e:
            self.session.rollback()
            logger.error(f"Error updating {self.model.__name__}: {e}")
            raise
    
    def delete(self, id: Union[UUID, int]) -> bool:
        """Delete a record by ID."""
        try:
            instance = self.get(id)
            if not instance:
                return False
            
            self.session.delete(instance)
            self.session.commit()
            return True
        except SQLAlchemyError as e:
            self.session.rollback()
            logger.error(f"Error deleting {self.model.__name__}: {e}")
            raise
    
    def bulk_create(self, data: List[Dict[str, Any]]) -> List[ModelType]:
        """Create multiple records at once."""
        try:
            instances = [self.model(**item) for item in data]
            self.session.bulk_save_objects(instances, return_defaults=True)
            self.session.commit()
            return instances
        except IntegrityError as e:
            self.session.rollback()
            logger.error(f"Integrity error bulk creating {self.model.__name__}: {e}")
            raise
        except SQLAlchemyError as e:
            self.session.rollback()
            logger.error(f"Error bulk creating {self.model.__name__}: {e}")
            raise
    
    def bulk_update(self, updates: List[Dict[str, Any]]) -> int:
        """Update multiple records at once."""
        try:
            self.session.bulk_update_mappings(self.model, updates)
            self.session.commit()
            return len(updates)
        except SQLAlchemyError as e:
            self.session.rollback()
            logger.error(f"Error bulk updating {self.model.__name__}: {e}")
            raise
    
    def exists(self, **filters) -> bool:
        """Check if a record exists."""
        try:
            return self.session.query(
                self.session.query(self.model).filter_by(**filters).exists()
            ).scalar()
        except SQLAlchemyError as e:
            logger.error(f"Error checking existence of {self.model.__name__}: {e}")
            raise
    
    def search(
        self,
        search_fields: List[str],
        search_term: str,
        skip: int = 0,
        limit: int = 100
    ) -> List[ModelType]:
        """Search records by multiple fields."""
        try:
            conditions = []
            for field in search_fields:
                if hasattr(self.model, field):
                    conditions.append(
                        getattr(self.model, field).ilike(f"%{search_term}%")
                    )
            
            if not conditions:
                return []
            
            return self.session.query(self.model)\
                .filter(or_(*conditions))\
                .offset(skip)\
                .limit(limit)\
                .all()
        except SQLAlchemyError as e:
            logger.error(f"Error searching {self.model.__name__}: {e}")
            raise


class AsyncBaseRepository(Generic[ModelType]):
    """
    Async version of base repository.
    """
    
    def __init__(self, model: Type[ModelType], session: AsyncSession):
        self.model = model
        self.session = session
    
    async def get(self, id: Union[UUID, int]) -> Optional[ModelType]:
        """Get a single record by ID."""
        try:
            result = await self.session.execute(
                select(self.model).where(self.model.id == id)
            )
            return result.scalar_one_or_none()
        except SQLAlchemyError as e:
            logger.error(f"Error getting {self.model.__name__} with id {id}: {e}")
            raise
    
    async def get_by(self, **kwargs) -> Optional[ModelType]:
        """Get a single record by field values."""
        try:
            result = await self.session.execute(
                select(self.model).filter_by(**kwargs)
            )
            return result.scalar_one_or_none()
        except SQLAlchemyError as e:
            logger.error(f"Error getting {self.model.__name__} by {kwargs}: {e}")
            raise
    
    async def get_all(
        self,
        skip: int = 0,
        limit: int = 100,
        order_by: Optional[str] = None,
        **filters
    ) -> List[ModelType]:
        """Get all records with optional pagination and filtering."""
        try:
            query = select(self.model)
            
            # Apply filters
            if filters:
                query = query.filter_by(**filters)
            
            # Apply ordering
            if order_by:
                if order_by.startswith('-'):
                    query = query.order_by(getattr(self.model, order_by[1:]).desc())
                else:
                    query = query.order_by(getattr(self.model, order_by))
            
            # Apply pagination
            query = query.offset(skip).limit(limit)
            
            result = await self.session.execute(query)
            return result.scalars().all()
        except SQLAlchemyError as e:
            logger.error(f"Error getting all {self.model.__name__}: {e}")
            raise
    
    async def count(self, **filters) -> int:
        """Count records with optional filtering."""
        try:
            query = select(func.count(self.model.id))
            if filters:
                query = query.filter_by(**filters)
            result = await self.session.execute(query)
            return result.scalar()
        except SQLAlchemyError as e:
            logger.error(f"Error counting {self.model.__name__}: {e}")
            raise
    
    async def create(self, **data) -> ModelType:
        """Create a new record."""
        try:
            instance = self.model(**data)
            self.session.add(instance)
            await self.session.commit()
            await self.session.refresh(instance)
            return instance
        except IntegrityError as e:
            await self.session.rollback()
            logger.error(f"Integrity error creating {self.model.__name__}: {e}")
            raise
        except SQLAlchemyError as e:
            await self.session.rollback()
            logger.error(f"Error creating {self.model.__name__}: {e}")
            raise
    
    async def update(self, id: Union[UUID, int], **data) -> Optional[ModelType]:
        """Update a record by ID."""
        try:
            instance = await self.get(id)
            if not instance:
                return None
            
            for key, value in data.items():
                if hasattr(instance, key):
                    setattr(instance, key, value)
            
            await self.session.commit()
            await self.session.refresh(instance)
            return instance
        except IntegrityError as e:
            await self.session.rollback()
            logger.error(f"Integrity error updating {self.model.__name__}: {e}")
            raise
        except SQLAlchemyError as e:
            await self.session.rollback()
            logger.error(f"Error updating {self.model.__name__}: {e}")
            raise
    
    async def delete(self, id: Union[UUID, int]) -> bool:
        """Delete a record by ID."""
        try:
            instance = await self.get(id)
            if not instance:
                return False
            
            await self.session.delete(instance)
            await self.session.commit()
            return True
        except SQLAlchemyError as e:
            await self.session.rollback()
            logger.error(f"Error deleting {self.model.__name__}: {e}")
            raise
    
    async def exists(self, **filters) -> bool:
        """Check if a record exists."""
        try:
            query = select(self.model).filter_by(**filters).exists()
            result = await self.session.execute(select(query))
            return result.scalar()
        except SQLAlchemyError as e:
            logger.error(f"Error checking existence of {self.model.__name__}: {e}")
            raise
    
    async def search(
        self,
        search_fields: List[str],
        search_term: str,
        skip: int = 0,
        limit: int = 100
    ) -> List[ModelType]:
        """Search records by multiple fields."""
        try:
            conditions = []
            for field in search_fields:
                if hasattr(self.model, field):
                    conditions.append(
                        getattr(self.model, field).ilike(f"%{search_term}%")
                    )
            
            if not conditions:
                return []
            
            query = select(self.model)\
                .where(or_(*conditions))\
                .offset(skip)\
                .limit(limit)
            
            result = await self.session.execute(query)
            return result.scalars().all()
        except SQLAlchemyError as e:
            logger.error(f"Error searching {self.model.__name__}: {e}")
            raise


# Specific repositories for each model
from .models import User, LearningPath, Module, Assessment, StudySession, Progress, Achievement, Notification, AuditLog


class UserRepository(BaseRepository[User]):
    """Repository for User model with specific methods."""
    
    def __init__(self, session: Session):
        super().__init__(User, session)
    
    def get_by_email(self, email: str) -> Optional[User]:
        """Get user by email."""
        return self.get_by(email=email)
    
    def get_by_username(self, username: str) -> Optional[User]:
        """Get user by username."""
        return self.get_by(username=username)
    
    def get_active_users(self, skip: int = 0, limit: int = 100) -> List[User]:
        """Get all active users."""
        return self.get_all(skip=skip, limit=limit, is_active=True)
    
    def update_last_login(self, user_id: UUID, ip_address: str) -> Optional[User]:
        """Update user's last login information."""
        return self.update(
            user_id,
            last_login_at=datetime.utcnow(),
            last_login_ip=ip_address
        )


class LearningPathRepository(BaseRepository[LearningPath]):
    """Repository for LearningPath model."""
    
    def __init__(self, session: Session):
        super().__init__(LearningPath, session)
    
    def get_published(self, skip: int = 0, limit: int = 100) -> List[LearningPath]:
        """Get all published learning paths."""
        return self.get_all(skip=skip, limit=limit, is_published=True)
    
    def get_by_slug(self, slug: str) -> Optional[LearningPath]:
        """Get learning path by slug."""
        return self.get_by(slug=slug)
    
    def get_by_difficulty(
        self,
        difficulty: str,
        skip: int = 0,
        limit: int = 100
    ) -> List[LearningPath]:
        """Get learning paths by difficulty level."""
        return self.get_all(skip=skip, limit=limit, difficulty=difficulty)


class AssessmentRepository(BaseRepository[Assessment]):
    """Repository for Assessment model."""
    
    def __init__(self, session: Session):
        super().__init__(Assessment, session)
    
    def get_user_assessments(
        self,
        user_id: UUID,
        skip: int = 0,
        limit: int = 100
    ) -> List[Assessment]:
        """Get all assessments for a user."""
        return self.get_all(skip=skip, limit=limit, user_id=user_id, order_by='-completed_at')
    
    def get_passed_assessments(
        self,
        user_id: UUID,
        skip: int = 0,
        limit: int = 100
    ) -> List[Assessment]:
        """Get all passed assessments for a user."""
        return self.get_all(skip=skip, limit=limit, user_id=user_id, passed=True)


class ProgressRepository(BaseRepository[Progress]):
    """Repository for Progress model."""
    
    def __init__(self, session: Session):
        super().__init__(Progress, session)
    
    def get_user_progress(
        self,
        user_id: UUID,
        module_id: UUID
    ) -> Optional[Progress]:
        """Get progress for a specific user and module."""
        return self.get_by(user_id=user_id, module_id=module_id)
    
    def get_completed_modules(
        self,
        user_id: UUID,
        skip: int = 0,
        limit: int = 100
    ) -> List[Progress]:
        """Get all completed modules for a user."""
        return self.get_all(
            skip=skip,
            limit=limit,
            user_id=user_id,
            status='completed'
        )