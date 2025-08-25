"""
Migration versioning and rollback system
Production-ready version control for database migrations
"""

import os
import json
import hashlib
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum

from sqlalchemy import create_engine, text, MetaData, Table, Column, String, DateTime, Integer, Boolean, Text
from sqlalchemy.exc import SQLAlchemyError
from alembic.script import ScriptDirectory
from alembic.config import Config
from alembic import command

logger = logging.getLogger(__name__)


class MigrationStatus(Enum):
    """Migration status enumeration."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


@dataclass
class MigrationVersion:
    """Represents a migration version."""
    revision: str
    description: str
    branch: Optional[str] = None
    dependencies: Optional[List[str]] = None
    checksum: Optional[str] = None
    created_at: Optional[datetime] = None
    applied_at: Optional[datetime] = None
    rolled_back_at: Optional[datetime] = None
    status: MigrationStatus = MigrationStatus.PENDING
    execution_time_ms: Optional[int] = None
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data['status'] = self.status.value
        if self.created_at:
            data['created_at'] = self.created_at.isoformat()
        if self.applied_at:
            data['applied_at'] = self.applied_at.isoformat()
        if self.rolled_back_at:
            data['rolled_back_at'] = self.rolled_back_at.isoformat()
        return data


class MigrationVersionControl:
    """
    Manages migration versioning, tracking, and rollback capabilities.
    """
    
    def __init__(self, database_url: str, alembic_config_path: str = "alembic.ini"):
        """
        Initialize version control system.
        
        Args:
            database_url: Database connection URL
            alembic_config_path: Path to Alembic configuration
        """
        self.database_url = database_url
        self.engine = create_engine(database_url)
        self.alembic_config = Config(alembic_config_path)
        self.script_dir = ScriptDirectory.from_config(self.alembic_config)
        
        # Initialize version tracking table
        self._init_version_table()
        
        # Migration history
        self.migration_history: List[MigrationVersion] = []
        self._load_history()
    
    def _init_version_table(self):
        """Initialize custom version tracking table."""
        metadata = MetaData()
        
        self.version_table = Table(
            'migration_version_control',
            metadata,
            Column('revision', String(32), primary_key=True),
            Column('description', Text),
            Column('branch', String(255)),
            Column('dependencies', Text),  # JSON array
            Column('checksum', String(64)),
            Column('created_at', DateTime),
            Column('applied_at', DateTime),
            Column('rolled_back_at', DateTime),
            Column('status', String(20)),
            Column('execution_time_ms', Integer),
            Column('error_message', Text),
            Column('rollback_revision', String(32)),
            Column('metadata_json', Text),  # Additional metadata
        )
        
        # Create table if it doesn't exist
        metadata.create_all(self.engine)
    
    def _load_history(self):
        """Load migration history from database."""
        try:
            with self.engine.connect() as conn:
                result = conn.execute(
                    text("SELECT * FROM migration_version_control ORDER BY applied_at DESC")
                )
                
                self.migration_history = []
                for row in result:
                    version = MigrationVersion(
                        revision=row.revision,
                        description=row.description or "",
                        branch=row.branch,
                        dependencies=json.loads(row.dependencies) if row.dependencies else None,
                        checksum=row.checksum,
                        created_at=row.created_at,
                        applied_at=row.applied_at,
                        rolled_back_at=row.rolled_back_at,
                        status=MigrationStatus(row.status) if row.status else MigrationStatus.PENDING,
                        execution_time_ms=row.execution_time_ms,
                        error_message=row.error_message
                    )
                    self.migration_history.append(version)
        
        except Exception as e:
            logger.error(f"Failed to load migration history: {e}")
    
    def calculate_checksum(self, revision: str) -> str:
        """
        Calculate checksum for a migration file.
        
        Args:
            revision: Migration revision
        
        Returns:
            SHA256 checksum of migration file
        """
        try:
            script = self.script_dir.get_revision(revision)
            if script and script.path:
                with open(script.path, 'rb') as f:
                    return hashlib.sha256(f.read()).hexdigest()
        except Exception as e:
            logger.error(f"Failed to calculate checksum for {revision}: {e}")
        
        return ""
    
    def verify_checksum(self, revision: str) -> bool:
        """
        Verify migration file hasn't been modified.
        
        Args:
            revision: Migration revision
        
        Returns:
            True if checksum matches or not found
        """
        stored_version = self.get_version(revision)
        if not stored_version or not stored_version.checksum:
            return True
        
        current_checksum = self.calculate_checksum(revision)
        if current_checksum != stored_version.checksum:
            logger.warning(f"Checksum mismatch for revision {revision}")
            return False
        
        return True
    
    def record_migration_start(self, revision: str, description: str = "") -> MigrationVersion:
        """
        Record the start of a migration.
        
        Args:
            revision: Migration revision
            description: Migration description
        
        Returns:
            MigrationVersion object
        """
        # Get migration script info
        script = self.script_dir.get_revision(revision)
        
        version = MigrationVersion(
            revision=revision,
            description=description or (script.doc if script else ""),
            branch=script.branch_labels[0] if script and script.branch_labels else None,
            dependencies=list(script.dependencies) if script and script.dependencies else None,
            checksum=self.calculate_checksum(revision),
            created_at=datetime.now(),
            status=MigrationStatus.IN_PROGRESS
        )
        
        # Store in database
        try:
            with self.engine.begin() as conn:
                conn.execute(
                    text("""
                        INSERT INTO migration_version_control 
                        (revision, description, branch, dependencies, checksum, 
                         created_at, status, metadata_json)
                        VALUES (:revision, :description, :branch, :dependencies, 
                                :checksum, :created_at, :status, :metadata)
                        ON CONFLICT (revision) 
                        DO UPDATE SET 
                            status = :status,
                            created_at = :created_at
                    """),
                    {
                        'revision': version.revision,
                        'description': version.description,
                        'branch': version.branch,
                        'dependencies': json.dumps(version.dependencies),
                        'checksum': version.checksum,
                        'created_at': version.created_at,
                        'status': version.status.value,
                        'metadata': json.dumps({'start_time': datetime.now().isoformat()})
                    }
                )
        
        except Exception as e:
            logger.error(f"Failed to record migration start: {e}")
        
        return version
    
    def record_migration_complete(
        self, 
        revision: str, 
        execution_time_ms: int,
        success: bool = True,
        error_message: str = None
    ):
        """
        Record migration completion.
        
        Args:
            revision: Migration revision
            execution_time_ms: Execution time in milliseconds
            success: Whether migration was successful
            error_message: Error message if failed
        """
        status = MigrationStatus.COMPLETED if success else MigrationStatus.FAILED
        
        try:
            with self.engine.begin() as conn:
                conn.execute(
                    text("""
                        UPDATE migration_version_control
                        SET status = :status,
                            applied_at = :applied_at,
                            execution_time_ms = :execution_time,
                            error_message = :error_message
                        WHERE revision = :revision
                    """),
                    {
                        'revision': revision,
                        'status': status.value,
                        'applied_at': datetime.now() if success else None,
                        'execution_time': execution_time_ms,
                        'error_message': error_message
                    }
                )
        
        except Exception as e:
            logger.error(f"Failed to record migration completion: {e}")
    
    def create_savepoint(self, name: str) -> bool:
        """
        Create a savepoint before migration.
        
        Args:
            name: Savepoint name
        
        Returns:
            True if successful
        """
        try:
            with self.engine.begin() as conn:
                conn.execute(text(f"SAVEPOINT {name}"))
            logger.info(f"Created savepoint: {name}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to create savepoint: {e}")
            return False
    
    def rollback_to_savepoint(self, name: str) -> bool:
        """
        Rollback to a savepoint.
        
        Args:
            name: Savepoint name
        
        Returns:
            True if successful
        """
        try:
            with self.engine.begin() as conn:
                conn.execute(text(f"ROLLBACK TO SAVEPOINT {name}"))
            logger.info(f"Rolled back to savepoint: {name}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to rollback to savepoint: {e}")
            return False
    
    def get_rollback_path(self, from_revision: str, to_revision: str) -> List[str]:
        """
        Get the path of migrations to rollback.
        
        Args:
            from_revision: Current revision
            to_revision: Target revision
        
        Returns:
            List of revisions to rollback in order
        """
        rollback_path = []
        
        try:
            # Get all revisions between current and target
            for script in self.script_dir.walk_revisions(from_revision, to_revision):
                rollback_path.append(script.revision)
        
        except Exception as e:
            logger.error(f"Failed to get rollback path: {e}")
        
        return rollback_path
    
    def validate_rollback(self, revision: str) -> Tuple[bool, str]:
        """
        Validate if rollback is safe.
        
        Args:
            revision: Target revision for rollback
        
        Returns:
            Tuple of (is_safe, message)
        """
        # Check if revision exists
        version = self.get_version(revision)
        if not version:
            return False, f"Revision {revision} not found in history"
        
        # Check if revision was successfully applied
        if version.status != MigrationStatus.COMPLETED:
            return False, f"Revision {revision} was not successfully applied"
        
        # Check for dependent migrations
        dependent_migrations = self._get_dependent_migrations(revision)
        if dependent_migrations:
            return False, f"Cannot rollback: {len(dependent_migrations)} migrations depend on this"
        
        # Check if downgrade method exists
        script = self.script_dir.get_revision(revision)
        if script and script.path:
            with open(script.path, 'r') as f:
                content = f.read()
                if 'def downgrade()' not in content:
                    return False, "Migration lacks downgrade method"
                
                # Check if downgrade is not empty
                if 'def downgrade():\n    pass' in content:
                    return False, "Downgrade method is empty"
        
        return True, "Rollback validation passed"
    
    def _get_dependent_migrations(self, revision: str) -> List[str]:
        """Get migrations that depend on given revision."""
        dependent = []
        
        for version in self.migration_history:
            if version.dependencies and revision in version.dependencies:
                if version.status == MigrationStatus.COMPLETED:
                    dependent.append(version.revision)
        
        return dependent
    
    def perform_rollback(self, revision: str, dry_run: bool = False) -> bool:
        """
        Perform migration rollback with safety checks.
        
        Args:
            revision: Target revision
            dry_run: If True, only simulate rollback
        
        Returns:
            True if successful
        """
        # Validate rollback
        is_safe, message = self.validate_rollback(revision)
        if not is_safe:
            logger.error(f"Rollback validation failed: {message}")
            return False
        
        if dry_run:
            logger.info(f"Dry run: Would rollback to {revision}")
            return True
        
        try:
            # Create savepoint
            savepoint_name = f"rollback_{revision}_{int(datetime.now().timestamp())}"
            self.create_savepoint(savepoint_name)
            
            # Perform rollback
            start_time = datetime.now()
            command.downgrade(self.alembic_config, revision)
            execution_time = int((datetime.now() - start_time).total_seconds() * 1000)
            
            # Record rollback
            with self.engine.begin() as conn:
                conn.execute(
                    text("""
                        UPDATE migration_version_control
                        SET status = :status,
                            rolled_back_at = :rolled_back_at
                        WHERE revision = :revision
                    """),
                    {
                        'revision': revision,
                        'status': MigrationStatus.ROLLED_BACK.value,
                        'rolled_back_at': datetime.now()
                    }
                )
            
            logger.info(f"Successfully rolled back to {revision} in {execution_time}ms")
            return True
        
        except Exception as e:
            logger.error(f"Rollback failed: {e}")
            # Try to rollback to savepoint
            self.rollback_to_savepoint(savepoint_name)
            return False
    
    def get_version(self, revision: str) -> Optional[MigrationVersion]:
        """Get version information for a revision."""
        for version in self.migration_history:
            if version.revision == revision:
                return version
        return None
    
    def get_migration_timeline(self) -> List[Dict[str, Any]]:
        """Get migration timeline for visualization."""
        timeline = []
        
        for version in sorted(self.migration_history, key=lambda v: v.created_at or datetime.min):
            timeline.append({
                'revision': version.revision,
                'description': version.description,
                'status': version.status.value,
                'created_at': version.created_at.isoformat() if version.created_at else None,
                'applied_at': version.applied_at.isoformat() if version.applied_at else None,
                'rolled_back_at': version.rolled_back_at.isoformat() if version.rolled_back_at else None,
                'execution_time_ms': version.execution_time_ms,
                'error_message': version.error_message
            })
        
        return timeline
    
    def export_version_history(self, output_path: str):
        """Export version history to JSON file."""
        history = {
            'exported_at': datetime.now().isoformat(),
            'database_url': self.database_url.split('@')[1] if '@' in self.database_url else 'unknown',
            'migrations': [v.to_dict() for v in self.migration_history]
        }
        
        output_file = Path(output_path)
        output_file.write_text(json.dumps(history, indent=2))
        logger.info(f"Exported version history to {output_path}")
    
    def cleanup_old_savepoints(self, older_than_days: int = 7):
        """Clean up old savepoints."""
        # This would be implemented based on your savepoint storage strategy
        pass