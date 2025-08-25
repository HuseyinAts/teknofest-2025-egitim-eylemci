"""
Test suite for database migration system
Comprehensive tests for production-ready migration functionality
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import json

from src.database.migrations import MigrationManager
from src.database.migration_validator import (
    MigrationValidator, 
    MigrationScheduler, 
    MigrationMonitor
)
from src.database.migration_versioning import (
    MigrationVersionControl,
    MigrationVersion,
    MigrationStatus
)
from src.database.connection_manager import SecureConnectionManager


class TestMigrationManager:
    """Test migration manager functionality."""
    
    @pytest.fixture
    def migration_manager(self, test_db_url):
        """Create migration manager for testing."""
        return MigrationManager()
    
    def test_get_current_revision(self, migration_manager):
        """Test getting current database revision."""
        revision = migration_manager.get_current_revision()
        assert revision is None or isinstance(revision, str)
    
    def test_get_pending_migrations(self, migration_manager):
        """Test getting list of pending migrations."""
        pending = migration_manager.get_pending_migrations()
        assert isinstance(pending, list)
    
    def test_check_database_health(self, migration_manager):
        """Test database health check."""
        health = migration_manager.check_database_health()
        assert isinstance(health, bool)
    
    @patch('src.database.migrations.subprocess.run')
    def test_create_backup(self, mock_run, migration_manager):
        """Test database backup creation."""
        mock_run.return_value = MagicMock(returncode=0, stderr="")
        
        backup_path = migration_manager.create_backup()
        assert backup_path is not None
        assert "backup" in backup_path
    
    def test_run_migrations_dry_run(self, migration_manager):
        """Test migration dry run (SQL generation only)."""
        success = migration_manager.run_migrations(revision="head", sql_only=True)
        assert isinstance(success, bool)
    
    def test_migration_with_invalid_revision(self, migration_manager):
        """Test migration with invalid revision."""
        success = migration_manager.run_migrations(revision="invalid_revision")
        assert success is False
    
    @patch('src.database.migrations.MigrationManager._backup_context')
    def test_migration_rollback_on_failure(self, mock_backup, migration_manager):
        """Test that migration rolls back on failure."""
        mock_backup.side_effect = Exception("Migration failed")
        
        success = migration_manager.run_migrations()
        assert success is False


class TestMigrationValidator:
    """Test migration validation functionality."""
    
    @pytest.fixture
    def validator(self):
        """Create migration validator."""
        return MigrationValidator()
    
    @pytest.fixture
    def dangerous_migration_content(self):
        """Create migration content with dangerous patterns."""
        return """
        def upgrade():
            op.execute("DROP DATABASE production")
            op.execute("DELETE FROM users")
            op.execute("UPDATE settings SET value = 'test'")
        
        def downgrade():
            pass
        """
    
    @pytest.fixture
    def safe_migration_content(self):
        """Create safe migration content."""
        return """
        def upgrade():
            op.add_column('users', sa.Column('created_at', sa.DateTime()))
            op.create_index('idx_users_email', 'users', ['email'])
        
        def downgrade():
            op.drop_index('idx_users_email')
            op.drop_column('users', 'created_at')
        """
    
    def test_check_dangerous_patterns(self, validator, dangerous_migration_content):
        """Test detection of dangerous SQL patterns."""
        results = validator._check_dangerous_patterns(dangerous_migration_content)
        
        assert len(results) > 0
        assert any(r['type'] == 'error' for r in results)
        assert any('DROP DATABASE' in r['message'] for r in results)
    
    def test_check_performance_patterns(self, validator):
        """Test detection of performance-impacting patterns."""
        content = """
        def upgrade():
            op.create_index('idx_large_table', 'large_table', ['column'])
            op.alter_table('users', 
                op.add_column('new_col', sa.String(), nullable=False))
        """
        
        results = validator._check_performance_patterns(content)
        assert len(results) > 0
        assert any('CONCURRENTLY' in r['message'] for r in results)
    
    def test_check_reversibility(self, validator, safe_migration_content):
        """Test checking migration reversibility."""
        mock_script = Mock()
        mock_script.dependencies = []
        mock_script.branch_labels = []
        
        results = validator._check_reversibility(safe_migration_content, mock_script)
        
        # Should pass for safe migration with proper downgrade
        assert not any(r['type'] == 'error' for r in results)
    
    def test_check_reversibility_missing_downgrade(self, validator):
        """Test detection of missing downgrade method."""
        content = """
        def upgrade():
            op.add_column('users', sa.Column('test', sa.String()))
        """
        
        mock_script = Mock()
        results = validator._check_reversibility(content, mock_script)
        
        assert any(r['type'] == 'error' for r in results)
        assert any('downgrade' in r['message'].lower() for r in results)
    
    def test_validation_report_generation(self, validator):
        """Test generation of validation report."""
        validator.validation_results = [
            {'type': 'error', 'check': 'test', 'message': 'Test error'},
            {'type': 'warning', 'check': 'test', 'message': 'Test warning'},
            {'type': 'info', 'check': 'test', 'message': 'Test info'}
        ]
        
        report = validator.generate_validation_report()
        
        assert 'MIGRATION VALIDATION REPORT' in report
        assert 'ERRORS:' in report
        assert 'WARNINGS:' in report
        assert 'Test error' in report


class TestMigrationVersionControl:
    """Test migration version control functionality."""
    
    @pytest.fixture
    def version_control(self, test_db_url):
        """Create version control instance."""
        return MigrationVersionControl(test_db_url)
    
    def test_calculate_checksum(self, version_control, tmp_path):
        """Test migration file checksum calculation."""
        # Create a temporary migration file
        migration_file = tmp_path / "test_migration.py"
        migration_file.write_text("test content")
        
        with patch.object(version_control.script_dir, 'get_revision') as mock_get:
            mock_script = Mock()
            mock_script.path = str(migration_file)
            mock_get.return_value = mock_script
            
            checksum = version_control.calculate_checksum("test_revision")
            assert len(checksum) == 64  # SHA256 hash length
    
    def test_record_migration_start(self, version_control):
        """Test recording migration start."""
        version = version_control.record_migration_start(
            "test_revision",
            "Test migration"
        )
        
        assert version.revision == "test_revision"
        assert version.status == MigrationStatus.IN_PROGRESS
        assert version.created_at is not None
    
    def test_validate_rollback_success(self, version_control):
        """Test successful rollback validation."""
        # Create a completed migration version
        version = MigrationVersion(
            revision="test_revision",
            description="Test",
            status=MigrationStatus.COMPLETED,
            applied_at=datetime.now()
        )
        version_control.migration_history = [version]
        
        # Mock the script with proper downgrade
        with patch.object(version_control.script_dir, 'get_revision') as mock_get:
            mock_script = Mock()
            mock_script.path = "test.py"
            mock_get.return_value = mock_script
            
            with patch('builtins.open', create=True) as mock_open:
                mock_open.return_value.__enter__.return_value.read.return_value = """
                def upgrade():
                    pass
                
                def downgrade():
                    op.drop_column('test', 'column')
                """
                
                is_safe, message = version_control.validate_rollback("test_revision")
                assert is_safe is True
    
    def test_get_rollback_path(self, version_control):
        """Test getting rollback path between revisions."""
        with patch.object(version_control.script_dir, 'walk_revisions') as mock_walk:
            mock_revisions = [
                Mock(revision="rev3"),
                Mock(revision="rev2"),
                Mock(revision="rev1")
            ]
            mock_walk.return_value = mock_revisions
            
            path = version_control.get_rollback_path("rev3", "rev1")
            assert path == ["rev3", "rev2", "rev1"]
    
    def test_migration_timeline(self, version_control):
        """Test getting migration timeline."""
        version_control.migration_history = [
            MigrationVersion(
                revision="rev1",
                description="First migration",
                created_at=datetime.now() - timedelta(days=2),
                applied_at=datetime.now() - timedelta(days=2),
                status=MigrationStatus.COMPLETED
            ),
            MigrationVersion(
                revision="rev2",
                description="Second migration",
                created_at=datetime.now() - timedelta(days=1),
                applied_at=datetime.now() - timedelta(days=1),
                status=MigrationStatus.COMPLETED
            )
        ]
        
        timeline = version_control.get_migration_timeline()
        
        assert len(timeline) == 2
        assert timeline[0]['revision'] == "rev1"
        assert timeline[1]['revision'] == "rev2"


class TestMigrationScheduler:
    """Test migration scheduling functionality."""
    
    @pytest.fixture
    def scheduler(self):
        """Create migration scheduler."""
        return MigrationScheduler()
    
    def test_add_maintenance_window(self, scheduler):
        """Test adding maintenance window."""
        start_time = datetime.now() + timedelta(hours=1)
        scheduler.add_maintenance_window(start_time, 60, "Test window")
        
        assert len(scheduler.maintenance_windows) == 1
        assert scheduler.maintenance_windows[0]['duration_minutes'] == 60
    
    def test_schedule_migration(self, scheduler):
        """Test scheduling a migration."""
        # Add a future maintenance window
        start_time = datetime.now() + timedelta(hours=1)
        scheduler.add_maintenance_window(start_time, 60)
        
        # Schedule a migration
        scheduled_time = scheduler.schedule_migration(
            "test_revision",
            priority=1,
            estimated_duration_minutes=30
        )
        
        assert scheduled_time == start_time
        assert len(scheduler.migration_queue) == 1
    
    def test_schedule_migration_no_window(self, scheduler):
        """Test scheduling when no suitable window exists."""
        scheduled_time = scheduler.schedule_migration("test_revision")
        assert scheduled_time is None
    
    def test_can_execute_now(self, scheduler):
        """Test checking if migration can execute now."""
        # Add current maintenance window
        scheduler.add_maintenance_window(
            datetime.now() - timedelta(minutes=10),
            30
        )
        
        can_execute = scheduler.can_execute_now("test_revision")
        assert can_execute is True
    
    def test_get_schedule(self, scheduler):
        """Test getting migration schedule."""
        # Add multiple migrations with different priorities
        scheduler.migration_queue = [
            {
                'revision': 'rev1',
                'priority': 5,
                'scheduled_time': datetime.now() + timedelta(hours=2)
            },
            {
                'revision': 'rev2',
                'priority': 1,
                'scheduled_time': datetime.now() + timedelta(hours=1)
            }
        ]
        
        schedule = scheduler.get_schedule()
        
        # Should be sorted by scheduled time then priority
        assert schedule[0]['revision'] == 'rev2'


class TestMigrationMonitor:
    """Test migration monitoring functionality."""
    
    @pytest.fixture
    def monitor(self, tmp_path):
        """Create migration monitor with temporary log file."""
        log_file = tmp_path / "migration_monitor.json"
        return MigrationMonitor(str(log_file))
    
    def test_start_migration(self, monitor):
        """Test starting migration monitoring."""
        monitor.start_migration("test_revision")
        
        assert monitor.current_migration is not None
        assert monitor.current_migration['revision'] == "test_revision"
    
    def test_record_metric(self, monitor):
        """Test recording migration metrics."""
        monitor.start_migration("test_revision")
        monitor.record_metric("rows_affected", 1000)
        monitor.record_metric("tables_modified", ["users", "settings"])
        
        assert monitor.current_migration['metrics']['rows_affected'] == 1000
        assert len(monitor.current_migration['metrics']['tables_modified']) == 2
    
    def test_end_migration_success(self, monitor):
        """Test ending successful migration."""
        monitor.start_migration("test_revision")
        monitor.record_metric("test_metric", "value")
        monitor.end_migration(success=True)
        
        assert monitor.current_migration is None
        assert len(monitor.metrics) == 1
        assert monitor.metrics[0]['success'] is True
    
    def test_end_migration_failure(self, monitor):
        """Test ending failed migration."""
        monitor.start_migration("test_revision")
        monitor.end_migration(success=False, error="Test error")
        
        assert monitor.metrics[0]['success'] is False
        assert monitor.metrics[0]['error'] == "Test error"
    
    def test_get_migration_stats(self, monitor):
        """Test getting migration statistics."""
        # Add some migration history
        monitor.metrics = [
            {'success': True, 'duration_seconds': 10},
            {'success': True, 'duration_seconds': 20},
            {'success': False, 'duration_seconds': 5}
        ]
        
        # Save to file
        monitor.log_file.write_text(json.dumps(monitor.metrics))
        
        stats = monitor.get_migration_stats()
        
        assert stats['total_migrations'] == 3
        assert stats['successful'] == 2
        assert stats['failed'] == 1
        assert stats['average_duration_seconds'] == 15  # (10+20)/2


class TestSecureConnectionManager:
    """Test secure connection management."""
    
    @pytest.fixture
    def connection_manager(self):
        """Create connection manager."""
        return SecureConnectionManager()
    
    def test_get_ssl_config_production(self, connection_manager):
        """Test SSL configuration for production."""
        with patch('src.database.connection_manager.settings.is_production', return_value=True):
            ssl_config = connection_manager._get_ssl_config()
            assert 'sslmode' in ssl_config
            assert ssl_config['sslmode'] == 'require'
    
    def test_get_pool_config_production(self, connection_manager):
        """Test pool configuration for production."""
        with patch('src.database.connection_manager.settings.is_production', return_value=True):
            pool_config = connection_manager._get_pool_config()
            
            assert pool_config['pool_size'] >= 20
            assert pool_config['pool_pre_ping'] is True
            assert pool_config['pool_recycle'] == 3600
    
    def test_secure_database_url(self, connection_manager):
        """Test securing database URL."""
        url = "postgresql://user:pass@localhost/db"
        
        with patch('src.database.connection_manager.settings.is_production', return_value=True):
            secured_url = connection_manager._secure_database_url(url)
            
            assert "sslmode" in secured_url
            assert "connect_timeout" in secured_url
    
    def test_get_pool_status(self, connection_manager):
        """Test getting connection pool status."""
        # Mock engine and pool
        mock_engine = Mock()
        mock_pool = Mock()
        mock_pool.size = 10
        mock_pool.checkedin = 8
        mock_pool.overflow = 2
        mock_engine.pool = mock_pool
        
        connection_manager.engines['test'] = mock_engine
        connection_manager.connection_stats['test'] = {
            'active_connections': 2,
            'failed_connections': 0
        }
        
        status = connection_manager.get_pool_status('test')
        
        assert status['size'] == 10
        assert status['checked_in'] == 8
        assert status['active_connections'] == 2


class TestIntegrationMigration:
    """Integration tests for migration system."""
    
    @pytest.mark.integration
    def test_full_migration_workflow(self, test_db_url):
        """Test complete migration workflow."""
        manager = MigrationManager()
        validator = MigrationValidator()
        version_control = MigrationVersionControl(test_db_url)
        
        # Get current revision
        current = manager.get_current_revision()
        
        # Check for pending migrations
        pending = manager.get_pending_migrations()
        
        if pending:
            # Validate first pending migration
            revision = pending[0]
            is_valid, results = validator.validate_migration(revision)
            
            if is_valid:
                # Record migration start
                version = version_control.record_migration_start(revision)
                
                # Run migration (dry run)
                success = manager.run_migrations(revision, sql_only=True)
                
                # Record completion
                version_control.record_migration_complete(
                    revision,
                    1000,
                    success
                )
                
                assert success is True