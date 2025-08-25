"""
Database mixins for common functionality
"""

from datetime import datetime
from typing import Optional, Dict, Any
import json

from sqlalchemy import Column, DateTime, Boolean, String, Text, Integer
from sqlalchemy.sql import func
from sqlalchemy.ext.declarative import declared_attr
from sqlalchemy.orm import Session
from sqlalchemy.ext.hybrid import hybrid_property


class TimestampMixin:
    """Mixin for adding created_at and updated_at timestamps."""
    
    @declared_attr
    def created_at(cls):
        return Column(
            DateTime(timezone=True),
            server_default=func.now(),
            nullable=False,
            index=True
        )
    
    @declared_attr
    def updated_at(cls):
        return Column(
            DateTime(timezone=True),
            server_default=func.now(),
            onupdate=func.now(),
            nullable=False,
            index=True
        )


class SoftDeleteMixin:
    """Mixin for soft delete functionality."""
    
    @declared_attr
    def deleted_at(cls):
        return Column(DateTime(timezone=True), nullable=True, index=True)
    
    @declared_attr
    def deleted_by(cls):
        return Column(String(255), nullable=True)
    
    @declared_attr
    def deletion_reason(cls):
        return Column(Text, nullable=True)
    
    @hybrid_property
    def is_deleted(self) -> bool:
        """Check if record is soft deleted."""
        return self.deleted_at is not None
    
    def soft_delete(
        self,
        deleted_by: Optional[str] = None,
        reason: Optional[str] = None
    ) -> None:
        """Soft delete the record."""
        self.deleted_at = datetime.utcnow()
        self.deleted_by = deleted_by
        self.deletion_reason = reason
    
    def restore(self) -> None:
        """Restore a soft deleted record."""
        self.deleted_at = None
        self.deleted_by = None
        self.deletion_reason = None
    
    @classmethod
    def get_active_query(cls, session: Session):
        """Get query for active (non-deleted) records."""
        return session.query(cls).filter(cls.deleted_at.is_(None))
    
    @classmethod
    def get_deleted_query(cls, session: Session):
        """Get query for deleted records."""
        return session.query(cls).filter(cls.deleted_at.isnot(None))


class VersioningMixin:
    """Mixin for record versioning."""
    
    @declared_attr
    def version(cls):
        return Column(
            Integer,
            nullable=False,
            default=1,
            server_default='1'
        )
    
    @declared_attr
    def version_history(cls):
        return Column(Text, nullable=True)
    
    def increment_version(self) -> None:
        """Increment the version number."""
        self.version = (self.version or 0) + 1
    
    def add_version_history(self, changes: Dict[str, Any]) -> None:
        """Add changes to version history."""
        history = json.loads(self.version_history or '[]')
        history.append({
            'version': self.version,
            'timestamp': datetime.utcnow().isoformat(),
            'changes': changes
        })
        self.version_history = json.dumps(history)


class SlugMixin:
    """Mixin for URL-friendly slugs."""
    
    @declared_attr
    def slug(cls):
        return Column(
            String(255),
            unique=True,
            nullable=False,
            index=True
        )
    
    @staticmethod
    def generate_slug(text: str) -> str:
        """Generate a URL-friendly slug from text."""
        import re
        # Remove special characters and convert to lowercase
        slug = re.sub(r'[^\w\s-]', '', text.lower())
        # Replace spaces with hyphens
        slug = re.sub(r'[-\s]+', '-', slug)
        # Remove leading/trailing hyphens
        return slug.strip('-')
    
    def set_slug_from(self, field_name: str) -> None:
        """Set slug from another field."""
        if hasattr(self, field_name):
            value = getattr(self, field_name)
            if value:
                self.slug = self.generate_slug(value)


class AuditMixin:
    """Mixin for audit fields."""
    
    @declared_attr
    def created_by(cls):
        return Column(String(255), nullable=True)
    
    @declared_attr
    def updated_by(cls):
        return Column(String(255), nullable=True)
    
    @declared_attr
    def created_ip(cls):
        return Column(String(45), nullable=True)
    
    @declared_attr
    def updated_ip(cls):
        return Column(String(45), nullable=True)
    
    def set_audit_fields(
        self,
        user: Optional[str] = None,
        ip_address: Optional[str] = None,
        is_update: bool = False
    ) -> None:
        """Set audit fields."""
        if is_update:
            self.updated_by = user
            self.updated_ip = ip_address
        else:
            self.created_by = user
            self.created_ip = ip_address
            self.updated_by = user
            self.updated_ip = ip_address


class TagsMixin:
    """Mixin for tagging functionality."""
    
    @declared_attr
    def tags(cls):
        return Column(Text, nullable=True)
    
    def get_tags(self) -> list:
        """Get tags as a list."""
        if not self.tags:
            return []
        return json.loads(self.tags)
    
    def set_tags(self, tags: list) -> None:
        """Set tags from a list."""
        self.tags = json.dumps(tags) if tags else None
    
    def add_tag(self, tag: str) -> None:
        """Add a single tag."""
        tags = self.get_tags()
        if tag not in tags:
            tags.append(tag)
            self.set_tags(tags)
    
    def remove_tag(self, tag: str) -> None:
        """Remove a single tag."""
        tags = self.get_tags()
        if tag in tags:
            tags.remove(tag)
            self.set_tags(tags)
    
    def has_tag(self, tag: str) -> bool:
        """Check if has a specific tag."""
        return tag in self.get_tags()


class MetadataMixin:
    """Mixin for storing arbitrary metadata."""
    
    @declared_attr
    def metadata(cls):
        return Column(Text, nullable=True)
    
    def get_metadata(self) -> dict:
        """Get metadata as a dictionary."""
        if not self.metadata:
            return {}
        return json.loads(self.metadata)
    
    def set_metadata(self, metadata: dict) -> None:
        """Set metadata from a dictionary."""
        self.metadata = json.dumps(metadata) if metadata else None
    
    def update_metadata(self, **kwargs) -> None:
        """Update metadata with new values."""
        metadata = self.get_metadata()
        metadata.update(kwargs)
        self.set_metadata(metadata)
    
    def get_metadata_value(self, key: str, default: Any = None) -> Any:
        """Get a specific metadata value."""
        metadata = self.get_metadata()
        return metadata.get(key, default)
    
    def set_metadata_value(self, key: str, value: Any) -> None:
        """Set a specific metadata value."""
        metadata = self.get_metadata()
        metadata[key] = value
        self.set_metadata(metadata)


class StatusMixin:
    """Mixin for status tracking."""
    
    STATUSES = {
        'draft': 'Draft',
        'pending': 'Pending',
        'approved': 'Approved',
        'rejected': 'Rejected',
        'published': 'Published',
        'archived': 'Archived'
    }
    
    @declared_attr
    def status(cls):
        return Column(
            String(20),
            nullable=False,
            default='draft',
            index=True
        )
    
    @declared_attr
    def status_changed_at(cls):
        return Column(DateTime(timezone=True), nullable=True)
    
    @declared_attr
    def status_changed_by(cls):
        return Column(String(255), nullable=True)
    
    @declared_attr
    def status_reason(cls):
        return Column(Text, nullable=True)
    
    def change_status(
        self,
        new_status: str,
        changed_by: Optional[str] = None,
        reason: Optional[str] = None
    ) -> None:
        """Change the status with tracking."""
        if new_status not in self.STATUSES:
            raise ValueError(f"Invalid status: {new_status}")
        
        self.status = new_status
        self.status_changed_at = datetime.utcnow()
        self.status_changed_by = changed_by
        self.status_reason = reason
    
    @hybrid_property
    def is_published(self) -> bool:
        """Check if status is published."""
        return self.status == 'published'
    
    @hybrid_property
    def is_draft(self) -> bool:
        """Check if status is draft."""
        return self.status == 'draft'
    
    @hybrid_property
    def is_archived(self) -> bool:
        """Check if status is archived."""
        return self.status == 'archived'


class SearchableMixin:
    """Mixin for full-text search support."""
    
    @declared_attr
    def search_vector(cls):
        return Column(Text, nullable=True)
    
    def update_search_vector(self, *fields) -> None:
        """Update search vector from specified fields."""
        values = []
        for field in fields:
            if hasattr(self, field):
                value = getattr(self, field)
                if value:
                    values.append(str(value))
        
        self.search_vector = ' '.join(values).lower() if values else None
    
    @classmethod
    def search(cls, session: Session, query: str, limit: int = 10):
        """Search records using full-text search."""
        search_term = f"%{query.lower()}%"
        return session.query(cls)\
            .filter(cls.search_vector.ilike(search_term))\
            .limit(limit)\
            .all()


class CacheMixin:
    """Mixin for cache management."""
    
    @declared_attr
    def cache_key(cls):
        return Column(String(255), nullable=True, index=True)
    
    @declared_attr
    def cache_expires_at(cls):
        return Column(DateTime(timezone=True), nullable=True)
    
    def generate_cache_key(self) -> str:
        """Generate a cache key for this record."""
        import hashlib
        key_parts = [
            self.__class__.__name__,
            str(getattr(self, 'id', ''))
        ]
        key = ':'.join(key_parts)
        return hashlib.md5(key.encode()).hexdigest()
    
    def set_cache_expiry(self, seconds: int) -> None:
        """Set cache expiry time."""
        from datetime import timedelta
        self.cache_expires_at = datetime.utcnow() + timedelta(seconds=seconds)
    
    @hybrid_property
    def is_cache_valid(self) -> bool:
        """Check if cache is still valid."""
        if not self.cache_expires_at:
            return True
        return datetime.utcnow() < self.cache_expires_at
    
    def invalidate_cache(self) -> None:
        """Invalidate the cache."""
        self.cache_expires_at = datetime.utcnow()