#!/usr/bin/env python
"""
Production migration script with safety checks and rollback capabilities
"""

import os
import sys
import logging
import argparse
import time
from datetime import datetime
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import get_settings
from src.database.migrations import MigrationManager
from src.database.migration_validator import MigrationValidator, MigrationMonitor
from src.database.migration_versioning import MigrationVersionControl
from src.database.connection_manager import get_connection_manager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/migration_production.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Get settings
settings = get_settings()


class ProductionMigrationRunner:
    """
    Runs migrations in production with comprehensive safety checks.
    """
    
    def __init__(self):
        self.migration_manager = MigrationManager()
        self.validator = MigrationValidator()
        self.version_control = MigrationVersionControl(settings.database_url)
        self.monitor = MigrationMonitor()
        self.connection_manager = get_connection_manager()
    
    def pre_migration_checks(self) -> bool:
        """Run pre-migration safety checks."""
        logger.info("Running pre-migration checks...")
        
        checks = {
            "Environment": self._check_environment(),
            "Database Connection": self._check_database_connection(),
            "Backup": self._check_backup_available(),
            "Disk Space": self._check_disk_space(),
            "Active Connections": self._check_active_connections(),
            "Pending Migrations": self._check_pending_migrations(),
        }
        
        # Print check results
        print("\n" + "="*60)
        print("PRE-MIGRATION CHECKS")
        print("="*60)
        
        all_passed = True
        for check_name, result in checks.items():
            status = "✓" if result['passed'] else "✗"
            color = "\033[92m" if result['passed'] else "\033[91m"
            reset = "\033[0m"
            print(f"{color}{status}{reset} {check_name}: {result['message']}")
            
            if not result['passed'] and result.get('critical', True):
                all_passed = False
        
        print("="*60 + "\n")
        
        return all_passed
    
    def _check_environment(self) -> dict:
        """Check if running in production environment."""
        is_production = settings.is_production()
        return {
            'passed': True,
            'message': f"Environment: {settings.app_env.value}",
            'critical': False
        }
    
    def _check_database_connection(self) -> dict:
        """Check database connection."""
        try:
            if self.connection_manager.test_connection():
                return {
                    'passed': True,
                    'message': "Database connection successful"
                }
            else:
                return {
                    'passed': False,
                    'message': "Database connection failed",
                    'critical': True
                }
        except Exception as e:
            return {
                'passed': False,
                'message': f"Connection error: {str(e)}",
                'critical': True
            }
    
    def _check_backup_available(self) -> dict:
        """Check if recent backup exists."""
        backup_dir = Path("backups")
        if not backup_dir.exists():
            return {
                'passed': False,
                'message': "No backup directory found",
                'critical': True
            }
        
        # Check for recent backup (within 24 hours)
        recent_backup = None
        for backup_file in backup_dir.glob("*.sql"):
            if (datetime.now() - datetime.fromtimestamp(backup_file.stat().st_mtime)).days < 1:
                recent_backup = backup_file
                break
        
        if recent_backup:
            return {
                'passed': True,
                'message': f"Recent backup found: {recent_backup.name}"
            }
        else:
            return {
                'passed': False,
                'message': "No recent backup (< 24 hours old) found",
                'critical': False  # Warning, not critical
            }
    
    def _check_disk_space(self) -> dict:
        """Check available disk space."""
        import shutil
        
        try:
            stats = shutil.disk_usage("/")
            free_gb = stats.free / (1024**3)
            
            if free_gb > 10:  # Require at least 10GB free
                return {
                    'passed': True,
                    'message': f"Disk space available: {free_gb:.1f} GB"
                }
            else:
                return {
                    'passed': False,
                    'message': f"Low disk space: {free_gb:.1f} GB",
                    'critical': True
                }
        except Exception as e:
            return {
                'passed': False,
                'message': f"Could not check disk space: {str(e)}",
                'critical': False
            }
    
    def _check_active_connections(self) -> dict:
        """Check number of active database connections."""
        pool_status = self.connection_manager.get_pool_status()
        active = pool_status.get('active_connections', 0)
        
        if active < 50:  # Threshold
            return {
                'passed': True,
                'message': f"Active connections: {active}"
            }
        else:
            return {
                'passed': False,
                'message': f"High number of active connections: {active}",
                'critical': False
            }
    
    def _check_pending_migrations(self) -> dict:
        """Check pending migrations."""
        pending = self.migration_manager.get_pending_migrations()
        
        if pending:
            return {
                'passed': True,
                'message': f"Found {len(pending)} pending migration(s)",
                'critical': False
            }
        else:
            return {
                'passed': True,
                'message': "No pending migrations",
                'critical': False
            }
    
    def validate_migrations(self, revisions: list) -> bool:
        """Validate all pending migrations."""
        logger.info("Validating migrations...")
        
        all_valid = True
        for revision in revisions:
            is_valid, results = self.validator.validate_migration(revision)
            
            if not is_valid:
                logger.error(f"Validation failed for {revision}")
                print(self.validator.generate_validation_report())
                all_valid = False
        
        return all_valid
    
    def create_backup(self) -> str:
        """Create database backup before migration."""
        logger.info("Creating database backup...")
        
        try:
            backup_path = self.migration_manager.create_backup()
            logger.info(f"Backup created: {backup_path}")
            return backup_path
        except Exception as e:
            logger.error(f"Backup creation failed: {e}")
            raise
    
    def run_migration(self, revision: str = "head", dry_run: bool = False) -> bool:
        """
        Run migration with monitoring and error handling.
        
        Args:
            revision: Target revision
            dry_run: If True, only simulate migration
        
        Returns:
            True if successful
        """
        # Start monitoring
        self.monitor.start_migration(revision)
        
        # Record migration start
        version = self.version_control.record_migration_start(revision)
        
        success = False
        error_message = None
        start_time = time.time()
        
        try:
            if dry_run:
                logger.info(f"DRY RUN: Would migrate to {revision}")
                # Generate SQL for review
                self.migration_manager.run_migrations(revision, sql_only=True)
                success = True
            else:
                # Run actual migration
                logger.info(f"Running migration to {revision}...")
                success = self.migration_manager.run_migrations(revision)
        
        except Exception as e:
            error_message = str(e)
            logger.error(f"Migration failed: {e}")
        
        finally:
            # Calculate execution time
            execution_time_ms = int((time.time() - start_time) * 1000)
            
            # Record completion
            self.version_control.record_migration_complete(
                revision, 
                execution_time_ms,
                success,
                error_message
            )
            
            # End monitoring
            self.monitor.end_migration(success, error_message)
        
        return success
    
    def rollback_migration(self, revision: str, dry_run: bool = False) -> bool:
        """
        Rollback migration with safety checks.
        
        Args:
            revision: Target revision for rollback
            dry_run: If True, only simulate rollback
        
        Returns:
            True if successful
        """
        logger.info(f"Initiating rollback to {revision}...")
        
        # Validate rollback
        is_safe, message = self.version_control.validate_rollback(revision)
        if not is_safe:
            logger.error(f"Rollback validation failed: {message}")
            return False
        
        # Perform rollback
        return self.version_control.perform_rollback(revision, dry_run)
    
    def run_production_migration(
        self,
        target_revision: str = "head",
        skip_validation: bool = False,
        skip_backup: bool = False,
        dry_run: bool = False,
        force: bool = False
    ) -> bool:
        """
        Run complete production migration workflow.
        
        Args:
            target_revision: Target revision
            skip_validation: Skip migration validation
            skip_backup: Skip backup creation
            dry_run: Simulate migration only
            force: Force migration even if checks fail
        
        Returns:
            True if successful
        """
        print("\n" + "="*60)
        print("PRODUCTION MIGRATION RUNNER")
        print("="*60)
        print(f"Target Revision: {target_revision}")
        print(f"Dry Run: {dry_run}")
        print(f"Started: {datetime.now().isoformat()}")
        print("="*60 + "\n")
        
        # Pre-migration checks
        if not force and not self.pre_migration_checks():
            logger.error("Pre-migration checks failed")
            return False
        
        # Get pending migrations
        pending = self.migration_manager.get_pending_migrations()
        if not pending:
            logger.info("No pending migrations")
            return True
        
        logger.info(f"Found {len(pending)} pending migration(s)")
        
        # Validate migrations
        if not skip_validation:
            if not self.validate_migrations(pending[:1]):  # Validate next migration
                if not force:
                    logger.error("Migration validation failed")
                    return False
                logger.warning("Validation failed but continuing due to --force flag")
        
        # Create backup
        backup_path = None
        if not skip_backup and not dry_run:
            try:
                backup_path = self.create_backup()
            except Exception as e:
                if not force:
                    logger.error(f"Backup creation failed: {e}")
                    return False
                logger.warning("Backup failed but continuing due to --force flag")
        
        # Run migration
        success = self.run_migration(target_revision, dry_run)
        
        if success:
            print("\n" + "="*60)
            print("✓ MIGRATION COMPLETED SUCCESSFULLY")
            print("="*60)
            
            # Show migration stats
            stats = self.monitor.get_migration_stats()
            if stats:
                print(f"Total Migrations Run: {stats.get('total_migrations', 0)}")
                print(f"Average Duration: {stats.get('average_duration_seconds', 0):.2f}s")
            
            if backup_path:
                print(f"Backup Location: {backup_path}")
            
            print(f"Completed: {datetime.now().isoformat()}")
            print("="*60 + "\n")
        else:
            print("\n" + "="*60)
            print("✗ MIGRATION FAILED")
            print("="*60)
            
            if backup_path:
                print(f"Backup available at: {backup_path}")
                print("Run with --rollback to restore from backup")
            
            print("="*60 + "\n")
        
        return success


def main():
    """Main entry point for production migration script."""
    parser = argparse.ArgumentParser(description='Production Database Migration Runner')
    
    parser.add_argument(
        'action',
        choices=['migrate', 'rollback', 'status', 'validate'],
        help='Action to perform'
    )
    
    parser.add_argument(
        '--revision',
        default='head',
        help='Target revision (default: head)'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Simulate migration without executing'
    )
    
    parser.add_argument(
        '--skip-validation',
        action='store_true',
        help='Skip migration validation'
    )
    
    parser.add_argument(
        '--skip-backup',
        action='store_true',
        help='Skip backup creation (not recommended)'
    )
    
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force migration even if checks fail (dangerous!)'
    )
    
    args = parser.parse_args()
    
    # Safety check for production
    if settings.is_production() and not args.dry_run:
        response = input("WARNING: This will modify the PRODUCTION database. Continue? (yes/no): ")
        if response.lower() != 'yes':
            print("Migration cancelled")
            sys.exit(0)
    
    # Create runner
    runner = ProductionMigrationRunner()
    
    # Execute action
    if args.action == 'migrate':
        success = runner.run_production_migration(
            target_revision=args.revision,
            skip_validation=args.skip_validation,
            skip_backup=args.skip_backup,
            dry_run=args.dry_run,
            force=args.force
        )
        sys.exit(0 if success else 1)
    
    elif args.action == 'rollback':
        success = runner.rollback_migration(args.revision, args.dry_run)
        sys.exit(0 if success else 1)
    
    elif args.action == 'status':
        # Show migration status
        from src.database.migrations import check_migrations_status
        status = check_migrations_status()
        
        print("\n" + "="*60)
        print("MIGRATION STATUS")
        print("="*60)
        print(f"Current Revision: {status.get('current_revision', 'None')}")
        print(f"Pending Migrations: {len(status.get('pending_migrations', []))}")
        print(f"Database Health: {'✓' if status.get('health_check') else '✗'}")
        print("="*60 + "\n")
    
    elif args.action == 'validate':
        # Validate specific revision or all pending
        if args.revision != 'head':
            is_valid, results = runner.validator.validate_migration(args.revision)
            print(runner.validator.generate_validation_report())
        else:
            pending = runner.migration_manager.get_pending_migrations()
            all_valid = runner.validate_migrations(pending)
            sys.exit(0 if all_valid else 1)


if __name__ == '__main__':
    main()