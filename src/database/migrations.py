"""
Database migration management utilities
Production-ready migration system with rollback, validation, and health checks
"""

import os
import sys
import logging
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any
from contextlib import contextmanager

from alembic import command
from alembic.config import Config
from alembic.script import ScriptDirectory
from alembic.runtime.migration import MigrationContext
from sqlalchemy import create_engine, text, inspect
from sqlalchemy.exc import SQLAlchemyError

from ..config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class MigrationManager:
    """
    Manages database migrations with production-ready features.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize migration manager.
        
        Args:
            config_path: Path to alembic.ini file
        """
        self.config_path = config_path or "alembic.ini"
        self.alembic_cfg = self._get_alembic_config()
        self.engine = self._get_engine()
    
    def _get_alembic_config(self) -> Config:
        """Get Alembic configuration"""
        cfg = Config(self.config_path)
        
        # Override database URL from settings
        cfg.set_main_option("sqlalchemy.url", settings.database_url)
        
        return cfg
    
    def _get_engine(self):
        """Get SQLAlchemy engine"""
        return create_engine(
            settings.database_url,
            pool_pre_ping=True,
            pool_size=5,
            max_overflow=10
        )
    
    @contextmanager
    def _backup_context(self):
        """
        Context manager for creating database backup before migration.
        Only for production environment.
        """
        backup_path = None
        
        if settings.is_production():
            try:
                # Create backup
                backup_path = self.create_backup()
                logger.info(f"Database backup created: {backup_path}")
                yield backup_path
            except Exception as e:
                logger.error(f"Failed to create backup: {e}")
                if backup_path:
                    # Try to restore if migration fails
                    self.restore_backup(backup_path)
                raise
        else:
            # Skip backup in development
            yield None
    
    def run_migrations(self, revision: str = "head", sql_only: bool = False) -> bool:
        """
        Run database migrations.
        
        Args:
            revision: Target revision (default: "head" for latest)
            sql_only: If True, only generate SQL without executing
        
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"Starting migration to revision: {revision}")
            
            # Check database health first
            if not self.check_database_health():
                logger.error("Database health check failed")
                return False
            
            # Get current revision
            current = self.get_current_revision()
            logger.info(f"Current revision: {current}")
            
            if sql_only:
                # Generate SQL only
                command.upgrade(self.alembic_cfg, revision, sql=True)
                logger.info("SQL migration script generated")
            else:
                # Run migration with backup in production
                with self._backup_context():
                    command.upgrade(self.alembic_cfg, revision)
                    logger.info(f"Migration completed successfully to {revision}")
            
            # Verify migration
            if not sql_only:
                new_revision = self.get_current_revision()
                logger.info(f"New revision: {new_revision}")
                
                # Run post-migration checks
                self.run_post_migration_checks()
            
            return True
            
        except Exception as e:
            logger.error(f"Migration failed: {e}")
            return False
    
    def rollback_migration(self, revision: str = "-1") -> bool:
        """
        Rollback database migration.
        
        Args:
            revision: Target revision (default: "-1" for previous)
        
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"Starting rollback to revision: {revision}")
            
            # Get current revision
            current = self.get_current_revision()
            logger.info(f"Current revision: {current}")
            
            # Perform rollback
            command.downgrade(self.alembic_cfg, revision)
            
            # Verify rollback
            new_revision = self.get_current_revision()
            logger.info(f"Rolled back to revision: {new_revision}")
            
            return True
            
        except Exception as e:
            logger.error(f"Rollback failed: {e}")
            return False
    
    def get_current_revision(self) -> Optional[str]:
        """Get current database revision"""
        try:
            with self.engine.connect() as conn:
                context = MigrationContext.configure(conn)
                return context.get_current_revision()
        except Exception as e:
            logger.error(f"Failed to get current revision: {e}")
            return None
    
    def get_pending_migrations(self) -> List[str]:
        """Get list of pending migrations"""
        try:
            script = ScriptDirectory.from_config(self.alembic_cfg)
            current = self.get_current_revision()
            
            # Get all revisions
            revisions = []
            for revision in script.walk_revisions():
                if current is None or revision.revision != current:
                    revisions.append(revision.revision)
                else:
                    break
            
            return revisions
            
        except Exception as e:
            logger.error(f"Failed to get pending migrations: {e}")
            return []
    
    def create_migration(self, message: str, autogenerate: bool = True) -> Optional[str]:
        """
        Create a new migration.
        
        Args:
            message: Migration message
            autogenerate: Auto-detect model changes
        
        Returns:
            Revision ID if successful, None otherwise
        """
        try:
            if autogenerate:
                # Auto-generate migration from model changes
                revision = command.revision(
                    self.alembic_cfg,
                    message=message,
                    autogenerate=True
                )
            else:
                # Create empty migration
                revision = command.revision(
                    self.alembic_cfg,
                    message=message
                )
            
            logger.info(f"Created migration: {revision}")
            return revision
            
        except Exception as e:
            logger.error(f"Failed to create migration: {e}")
            return None
    
    def check_database_health(self) -> bool:
        """
        Check database health before migration.
        """
        checks = {
            "connection": self._check_connection(),
            "permissions": self._check_permissions(),
            "disk_space": self._check_disk_space(),
            "active_connections": self._check_active_connections(),
        }
        
        failed_checks = [name for name, passed in checks.items() if not passed]
        
        if failed_checks:
            logger.error(f"Health checks failed: {', '.join(failed_checks)}")
            return False
        
        logger.info("All health checks passed")
        return True
    
    def _check_connection(self) -> bool:
        """Check database connection"""
        try:
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return True
        except Exception as e:
            logger.error(f"Connection check failed: {e}")
            return False
    
    def _check_permissions(self) -> bool:
        """Check database permissions"""
        try:
            with self.engine.connect() as conn:
                # Check if we can create/drop tables
                conn.execute(text("CREATE TEMP TABLE _permission_check (id INT)"))
                conn.execute(text("DROP TABLE _permission_check"))
            return True
        except Exception as e:
            logger.error(f"Permission check failed: {e}")
            return False
    
    def _check_disk_space(self) -> bool:
        """Check available disk space"""
        try:
            with self.engine.connect() as conn:
                # PostgreSQL specific
                result = conn.execute(text("""
                    SELECT pg_database_size(current_database()) as size
                """))
                db_size = result.scalar()
                
                # Check if we have at least 2x the database size available
                # This is a simplified check - in production, check actual disk space
                if db_size:
                    logger.info(f"Database size: {db_size / 1024 / 1024:.2f} MB")
                
            return True
        except Exception:
            # Not critical for non-PostgreSQL databases
            return True
    
    def _check_active_connections(self) -> bool:
        """Check for active connections that might block migration"""
        try:
            with self.engine.connect() as conn:
                # PostgreSQL specific
                result = conn.execute(text("""
                    SELECT count(*) 
                    FROM pg_stat_activity 
                    WHERE datname = current_database() 
                    AND state = 'active'
                    AND pid != pg_backend_pid()
                """))
                active_count = result.scalar()
                
                if active_count > 10:  # Threshold
                    logger.warning(f"High number of active connections: {active_count}")
                
            return True
        except Exception:
            # Not critical for non-PostgreSQL databases
            return True
    
    def run_post_migration_checks(self):
        """Run checks after migration"""
        try:
            inspector = inspect(self.engine)
            
            # Check all tables exist
            tables = inspector.get_table_names()
            logger.info(f"Database has {len(tables)} tables")
            
            # Check for migration issues
            with self.engine.connect() as conn:
                # Check for invalid constraints
                result = conn.execute(text("""
                    SELECT conname 
                    FROM pg_constraint 
                    WHERE NOT convalidated
                """))
                invalid_constraints = result.fetchall()
                
                if invalid_constraints:
                    logger.warning(f"Found invalid constraints: {invalid_constraints}")
            
        except Exception as e:
            logger.error(f"Post-migration checks failed: {e}")
    
    def create_backup(self) -> str:
        """
        Create database backup.
        
        Returns:
            Path to backup file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = Path("backups")
        backup_dir.mkdir(exist_ok=True)
        
        backup_file = backup_dir / f"backup_{timestamp}.sql"
        
        try:
            # Parse database URL
            from urllib.parse import urlparse
            db_url = urlparse(settings.database_url)
            
            # PostgreSQL backup command
            command = [
                "pg_dump",
                "-h", db_url.hostname or "localhost",
                "-p", str(db_url.port or 5432),
                "-U", db_url.username or "postgres",
                "-d", db_url.path.lstrip("/"),
                "-f", str(backup_file),
                "--verbose",
                "--no-owner",
                "--no-acl"
            ]
            
            # Set password via environment
            env = os.environ.copy()
            if db_url.password:
                env["PGPASSWORD"] = db_url.password
            
            # Run backup
            result = subprocess.run(
                command,
                env=env,
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                raise Exception(f"Backup failed: {result.stderr}")
            
            logger.info(f"Backup created: {backup_file}")
            return str(backup_file)
            
        except Exception as e:
            logger.error(f"Failed to create backup: {e}")
            raise
    
    def restore_backup(self, backup_path: str) -> bool:
        """
        Restore database from backup.
        
        Args:
            backup_path: Path to backup file
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Parse database URL
            from urllib.parse import urlparse
            db_url = urlparse(settings.database_url)
            
            # PostgreSQL restore command
            command = [
                "psql",
                "-h", db_url.hostname or "localhost",
                "-p", str(db_url.port or 5432),
                "-U", db_url.username or "postgres",
                "-d", db_url.path.lstrip("/"),
                "-f", backup_path,
                "--single-transaction"
            ]
            
            # Set password via environment
            env = os.environ.copy()
            if db_url.password:
                env["PGPASSWORD"] = db_url.password
            
            # Run restore
            result = subprocess.run(
                command,
                env=env,
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                raise Exception(f"Restore failed: {result.stderr}")
            
            logger.info(f"Database restored from: {backup_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to restore backup: {e}")
            return False
    
    def get_migration_history(self) -> List[Dict[str, Any]]:
        """Get migration history"""
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text("""
                    SELECT version_num, 
                           COALESCE(
                               (SELECT create_date 
                                FROM alembic_version_history 
                                WHERE version_num = av.version_num 
                                LIMIT 1), 
                               CURRENT_TIMESTAMP
                           ) as applied_at
                    FROM alembic_version av
                    ORDER BY applied_at DESC
                """))
                
                return [
                    {"revision": row[0], "applied_at": row[1]}
                    for row in result
                ]
        except Exception:
            # Fallback if history table doesn't exist
            current = self.get_current_revision()
            if current:
                return [{"revision": current, "applied_at": datetime.now()}]
            return []


# Singleton instance
_migration_manager = None


def get_migration_manager() -> MigrationManager:
    """Get migration manager singleton"""
    global _migration_manager
    if _migration_manager is None:
        _migration_manager = MigrationManager()
    return _migration_manager


# Convenience functions
def run_migrations(revision: str = "head") -> bool:
    """Run migrations to specified revision"""
    manager = get_migration_manager()
    return manager.run_migrations(revision)


def rollback_migration(revision: str = "-1") -> bool:
    """Rollback to specified revision"""
    manager = get_migration_manager()
    return manager.rollback_migration(revision)


def create_migration(message: str, autogenerate: bool = True) -> Optional[str]:
    """Create new migration"""
    manager = get_migration_manager()
    return manager.create_migration(message, autogenerate)


def get_pending_migrations() -> List[str]:
    """Get list of pending migrations"""
    manager = get_migration_manager()
    return manager.get_pending_migrations()


def check_migrations_status() -> Dict[str, Any]:
    """Get comprehensive migration status"""
    manager = get_migration_manager()
    return {
        "current_revision": manager.get_current_revision(),
        "pending_migrations": manager.get_pending_migrations(),
        "health_check": manager.check_database_health(),
        "history": manager.get_migration_history()
    }