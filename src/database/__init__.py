"""
Database module for TEKNOFEST 2025 Education Platform
Production-ready database configuration with migrations, connection pooling, and monitoring
"""

# Core components
from .base import (
    Base,
    get_db,
    get_async_db,
    get_db_context,
    get_async_db_context,
    init_db,
    init_async_db
)

# Session management
from .session import (
    SessionLocal,
    AsyncSessionLocal,
    engine,
    async_engine,
    get_database_url,
    get_db_stats,
    close_all_sessions,
    close_async_sessions
)

# Models
from .models import (
    User,
    UserRole,
    LearningPath,
    Module,
    StudySession,
    Assessment,
    Progress,
    Achievement,
    Notification,
    AuditLog,
    DifficultyLevel,
    ContentType
)

# Repository pattern
from .repository import (
    BaseRepository,
    AsyncBaseRepository,
    UserRepository,
    LearningPathRepository,
    AssessmentRepository,
    ProgressRepository
)

# Decorators and utilities
from .decorators import (
    transactional,
    async_transactional,
    with_retries,
    async_with_retries,
    atomic_transaction,
    handle_integrity_error,
    bulk_operation
)

# Mixins
from .mixins import (
    TimestampMixin,
    SoftDeleteMixin,
    VersioningMixin,
    SlugMixin,
    AuditMixin,
    TagsMixin,
    MetadataMixin,
    StatusMixin,
    SearchableMixin,
    CacheMixin
)

# Query utilities
from .query_utils import (
    QueryOptimizer,
    FilterBuilder,
    SortBuilder,
    QueryCache,
    BulkOperations,
    QueryProfiler
)

# Backup and maintenance
from .backup import (
    DatabaseBackup,
    DatabaseMaintenance
)

# Migrations and health
from .migrations import run_migrations, rollback_migration
from .health import check_database_health

__all__ = [
    # Core
    'Base',
    'get_db',
    'get_async_db',
    'get_db_context',
    'get_async_db_context',
    'init_db',
    'init_async_db',
    
    # Session
    'SessionLocal',
    'AsyncSessionLocal',
    'engine',
    'async_engine',
    'get_database_url',
    'get_db_stats',
    'close_all_sessions',
    'close_async_sessions',
    
    # Models
    'User',
    'UserRole',
    'LearningPath',
    'Module',
    'StudySession',
    'Assessment',
    'Progress',
    'Achievement',
    'Notification',
    'AuditLog',
    'DifficultyLevel',
    'ContentType',
    
    # Repository
    'BaseRepository',
    'AsyncBaseRepository',
    'UserRepository',
    'LearningPathRepository',
    'AssessmentRepository',
    'ProgressRepository',
    
    # Decorators
    'transactional',
    'async_transactional',
    'with_retries',
    'async_with_retries',
    'atomic_transaction',
    'handle_integrity_error',
    'bulk_operation',
    
    # Mixins
    'TimestampMixin',
    'SoftDeleteMixin',
    'VersioningMixin',
    'SlugMixin',
    'AuditMixin',
    'TagsMixin',
    'MetadataMixin',
    'StatusMixin',
    'SearchableMixin',
    'CacheMixin',
    
    # Query utilities
    'QueryOptimizer',
    'FilterBuilder',
    'SortBuilder',
    'QueryCache',
    'BulkOperations',
    'QueryProfiler',
    
    # Backup and maintenance
    'DatabaseBackup',
    'DatabaseMaintenance',
    
    # Migrations and health
    'run_migrations',
    'rollback_migration',
    'check_database_health',
]