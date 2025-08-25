"""
Migration validation and safety checks
Production-ready validation system for database migrations
"""

import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime, timedelta
import json

from sqlalchemy import create_engine, text, inspect
from alembic.script import ScriptDirectory
from alembic.config import Config

logger = logging.getLogger(__name__)


class MigrationValidator:
    """
    Validates migrations for safety and correctness before execution.
    """
    
    # Dangerous SQL patterns to check
    DANGEROUS_PATTERNS = [
        (r'\bDROP\s+DATABASE\b', 'DROP DATABASE detected'),
        (r'\bDROP\s+SCHEMA\b', 'DROP SCHEMA detected'),
        (r'\bTRUNCATE\b.*\bCASCADE\b', 'TRUNCATE CASCADE detected'),
        (r'\bDELETE\s+FROM\b(?!.*\bWHERE\b)', 'DELETE without WHERE clause'),
        (r'\bUPDATE\b(?!.*\bWHERE\b)', 'UPDATE without WHERE clause'),
    ]
    
    # Performance impact patterns
    PERFORMANCE_PATTERNS = [
        (r'\bALTER\s+TABLE\b.*\bADD\s+COLUMN\b.*\bNOT\s+NULL\b(?!.*\bDEFAULT\b)', 
         'Adding NOT NULL column without DEFAULT may lock table'),
        (r'\bCREATE\s+INDEX\b(?!.*\bCONCURRENTLY\b)', 
         'CREATE INDEX without CONCURRENTLY may lock table'),
        (r'\bREINDEX\b', 'REINDEX can cause significant locks'),
        (r'\bALTER\s+TABLE\b.*\bRENAME\b', 'Table rename can break applications'),
    ]
    
    def __init__(self, config_path: str = "alembic.ini"):
        """Initialize validator with Alembic configuration."""
        self.config = Config(config_path)
        self.script_dir = ScriptDirectory.from_config(self.config)
        self.validation_results = []
    
    def validate_migration(
        self, 
        revision: str,
        check_dangerous: bool = True,
        check_performance: bool = True,
        check_reversibility: bool = True,
        dry_run: bool = True
    ) -> Tuple[bool, List[Dict[str, Any]]]:
        """
        Validate a specific migration revision.
        
        Args:
            revision: Migration revision to validate
            check_dangerous: Check for dangerous operations
            check_performance: Check for performance impacts
            check_reversibility: Check if migration is reversible
            dry_run: If True, only simulate validation
        
        Returns:
            Tuple of (is_valid, validation_results)
        """
        results = []
        is_valid = True
        
        try:
            # Get migration script
            script = self.script_dir.get_revision(revision)
            if not script:
                results.append({
                    'type': 'error',
                    'check': 'revision',
                    'message': f'Revision {revision} not found'
                })
                return False, results
            
            # Read migration file
            migration_file = Path(script.path)
            if not migration_file.exists():
                results.append({
                    'type': 'error',
                    'check': 'file',
                    'message': f'Migration file not found: {migration_file}'
                })
                return False, results
            
            content = migration_file.read_text()
            
            # Check for dangerous patterns
            if check_dangerous:
                dangerous_results = self._check_dangerous_patterns(content)
                results.extend(dangerous_results)
                if any(r['type'] == 'error' for r in dangerous_results):
                    is_valid = False
            
            # Check for performance impacts
            if check_performance:
                performance_results = self._check_performance_patterns(content)
                results.extend(performance_results)
            
            # Check reversibility
            if check_reversibility:
                reversibility_results = self._check_reversibility(content, script)
                results.extend(reversibility_results)
                if any(r['type'] == 'error' for r in reversibility_results):
                    is_valid = False
            
            # Check migration dependencies
            dependency_results = self._check_dependencies(script)
            results.extend(dependency_results)
            
            # Check for large data modifications
            data_results = self._check_data_modifications(content)
            results.extend(data_results)
            
            # Add summary
            results.append({
                'type': 'info',
                'check': 'summary',
                'message': f'Validation {"passed" if is_valid else "failed"} for revision {revision}'
            })
            
        except Exception as e:
            results.append({
                'type': 'error',
                'check': 'validation',
                'message': f'Validation error: {str(e)}'
            })
            is_valid = False
        
        self.validation_results = results
        return is_valid, results
    
    def _check_dangerous_patterns(self, content: str) -> List[Dict[str, Any]]:
        """Check for dangerous SQL patterns in migration."""
        results = []
        
        for pattern, message in self.DANGEROUS_PATTERNS:
            if re.search(pattern, content, re.IGNORECASE | re.MULTILINE):
                results.append({
                    'type': 'error',
                    'check': 'dangerous_pattern',
                    'message': f'DANGER: {message}',
                    'pattern': pattern
                })
        
        return results
    
    def _check_performance_patterns(self, content: str) -> List[Dict[str, Any]]:
        """Check for potential performance impacts."""
        results = []
        
        for pattern, message in self.PERFORMANCE_PATTERNS:
            if re.search(pattern, content, re.IGNORECASE | re.MULTILINE):
                results.append({
                    'type': 'warning',
                    'check': 'performance',
                    'message': f'PERFORMANCE: {message}',
                    'pattern': pattern
                })
        
        return results
    
    def _check_reversibility(self, content: str, script) -> List[Dict[str, Any]]:
        """Check if migration has proper downgrade method."""
        results = []
        
        # Check for downgrade function
        if 'def downgrade()' not in content:
            results.append({
                'type': 'error',
                'check': 'reversibility',
                'message': 'Migration lacks downgrade() function'
            })
        else:
            # Check if downgrade is not empty
            downgrade_match = re.search(
                r'def downgrade\(\)[^:]*:\s*(?:pass|$)', 
                content, 
                re.MULTILINE
            )
            if downgrade_match:
                results.append({
                    'type': 'warning',
                    'check': 'reversibility',
                    'message': 'Downgrade function appears to be empty'
                })
        
        # Check for data loss in downgrade
        if re.search(r'def downgrade.*?DROP\s+COLUMN', content, re.IGNORECASE | re.DOTALL):
            results.append({
                'type': 'warning',
                'check': 'reversibility',
                'message': 'Downgrade drops columns - potential data loss'
            })
        
        return results
    
    def _check_dependencies(self, script) -> List[Dict[str, Any]]:
        """Check migration dependencies and ordering."""
        results = []
        
        # Check if dependencies are properly set
        if script.dependencies:
            results.append({
                'type': 'info',
                'check': 'dependencies',
                'message': f'Migration depends on: {script.dependencies}'
            })
        
        # Check branch points
        if script.branch_labels:
            results.append({
                'type': 'info',
                'check': 'branch',
                'message': f'Branch labels: {script.branch_labels}'
            })
        
        return results
    
    def _check_data_modifications(self, content: str) -> List[Dict[str, Any]]:
        """Check for large data modifications that might timeout."""
        results = []
        
        # Check for bulk updates
        bulk_patterns = [
            (r'UPDATE.*FROM.*JOIN', 'Bulk UPDATE with JOIN detected'),
            (r'INSERT.*SELECT', 'Bulk INSERT detected'),
            (r'DELETE.*JOIN', 'Bulk DELETE with JOIN detected'),
        ]
        
        for pattern, message in bulk_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                results.append({
                    'type': 'warning',
                    'check': 'data_modification',
                    'message': f'{message} - consider batching for large tables'
                })
        
        return results
    
    def validate_all_pending(self) -> Tuple[bool, Dict[str, List[Dict[str, Any]]]]:
        """Validate all pending migrations."""
        all_results = {}
        all_valid = True
        
        try:
            # Get pending migrations
            # This would need to be connected to the migration manager
            pending_revisions = []  # Would get from migration manager
            
            for revision in pending_revisions:
                is_valid, results = self.validate_migration(revision)
                all_results[revision] = results
                if not is_valid:
                    all_valid = False
        
        except Exception as e:
            logger.error(f"Failed to validate pending migrations: {e}")
            all_valid = False
        
        return all_valid, all_results
    
    def generate_validation_report(self) -> str:
        """Generate a human-readable validation report."""
        if not self.validation_results:
            return "No validation results available"
        
        report = []
        report.append("=" * 60)
        report.append("MIGRATION VALIDATION REPORT")
        report.append("=" * 60)
        report.append(f"Generated: {datetime.now().isoformat()}")
        report.append("")
        
        # Group by type
        errors = [r for r in self.validation_results if r['type'] == 'error']
        warnings = [r for r in self.validation_results if r['type'] == 'warning']
        info = [r for r in self.validation_results if r['type'] == 'info']
        
        if errors:
            report.append("ERRORS:")
            report.append("-" * 40)
            for error in errors:
                report.append(f"  ✗ [{error['check']}] {error['message']}")
            report.append("")
        
        if warnings:
            report.append("WARNINGS:")
            report.append("-" * 40)
            for warning in warnings:
                report.append(f"  ⚠ [{warning['check']}] {warning['message']}")
            report.append("")
        
        if info:
            report.append("INFORMATION:")
            report.append("-" * 40)
            for item in info:
                report.append(f"  ℹ [{item['check']}] {item['message']}")
            report.append("")
        
        # Summary
        report.append("SUMMARY:")
        report.append("-" * 40)
        report.append(f"  Errors: {len(errors)}")
        report.append(f"  Warnings: {len(warnings)}")
        report.append(f"  Info: {len(info)}")
        
        validation_status = "PASSED" if not errors else "FAILED"
        report.append(f"  Status: {validation_status}")
        
        report.append("=" * 60)
        
        return "\n".join(report)


class MigrationScheduler:
    """
    Schedules and manages migration execution windows.
    """
    
    def __init__(self):
        self.maintenance_windows = []
        self.migration_queue = []
    
    def add_maintenance_window(
        self,
        start_time: datetime,
        duration_minutes: int,
        description: str = ""
    ):
        """Add a maintenance window for migrations."""
        self.maintenance_windows.append({
            'start_time': start_time,
            'end_time': start_time + timedelta(minutes=duration_minutes),
            'duration_minutes': duration_minutes,
            'description': description
        })
    
    def schedule_migration(
        self,
        revision: str,
        priority: int = 5,
        estimated_duration_minutes: int = 5,
        requires_downtime: bool = False
    ) -> Optional[datetime]:
        """
        Schedule a migration for execution.
        
        Args:
            revision: Migration revision
            priority: Priority (1-10, 1 is highest)
            estimated_duration_minutes: Estimated execution time
            requires_downtime: Whether migration requires downtime
        
        Returns:
            Scheduled execution time or None if cannot schedule
        """
        migration = {
            'revision': revision,
            'priority': priority,
            'estimated_duration': estimated_duration_minutes,
            'requires_downtime': requires_downtime,
            'scheduled_time': None
        }
        
        # Find suitable maintenance window
        for window in sorted(self.maintenance_windows, key=lambda w: w['start_time']):
            window_duration = (window['end_time'] - window['start_time']).total_seconds() / 60
            
            if window_duration >= estimated_duration_minutes:
                # Check if window is still in future
                if window['start_time'] > datetime.now():
                    migration['scheduled_time'] = window['start_time']
                    self.migration_queue.append(migration)
                    return window['start_time']
        
        return None
    
    def get_schedule(self) -> List[Dict[str, Any]]:
        """Get current migration schedule."""
        return sorted(
            self.migration_queue, 
            key=lambda m: (m['scheduled_time'] or datetime.max, m['priority'])
        )
    
    def can_execute_now(self, revision: str) -> bool:
        """Check if migration can be executed now."""
        # Check if in maintenance window
        now = datetime.now()
        for window in self.maintenance_windows:
            if window['start_time'] <= now <= window['end_time']:
                return True
        return False


class MigrationMonitor:
    """
    Monitors migration execution and performance.
    """
    
    def __init__(self, log_file: str = "logs/migration_monitor.json"):
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        self.current_migration = None
        self.metrics = []
    
    def start_migration(self, revision: str):
        """Start monitoring a migration."""
        self.current_migration = {
            'revision': revision,
            'start_time': datetime.now().isoformat(),
            'metrics': {}
        }
    
    def record_metric(self, name: str, value: Any):
        """Record a metric for current migration."""
        if self.current_migration:
            self.current_migration['metrics'][name] = value
    
    def end_migration(self, success: bool = True, error: str = None):
        """End monitoring current migration."""
        if self.current_migration:
            self.current_migration['end_time'] = datetime.now().isoformat()
            self.current_migration['success'] = success
            if error:
                self.current_migration['error'] = error
            
            # Calculate duration
            start = datetime.fromisoformat(self.current_migration['start_time'])
            end = datetime.fromisoformat(self.current_migration['end_time'])
            self.current_migration['duration_seconds'] = (end - start).total_seconds()
            
            # Save to log
            self._save_migration_log(self.current_migration)
            self.metrics.append(self.current_migration)
            self.current_migration = None
    
    def _save_migration_log(self, migration_data: Dict[str, Any]):
        """Save migration log to file."""
        logs = []
        if self.log_file.exists():
            try:
                logs = json.loads(self.log_file.read_text())
            except:
                pass
        
        logs.append(migration_data)
        
        # Keep only last 100 entries
        if len(logs) > 100:
            logs = logs[-100:]
        
        self.log_file.write_text(json.dumps(logs, indent=2))
    
    def get_migration_stats(self) -> Dict[str, Any]:
        """Get migration execution statistics."""
        if not self.log_file.exists():
            return {}
        
        try:
            logs = json.loads(self.log_file.read_text())
            
            successful = [l for l in logs if l.get('success')]
            failed = [l for l in logs if not l.get('success')]
            
            avg_duration = 0
            if successful:
                avg_duration = sum(l.get('duration_seconds', 0) for l in successful) / len(successful)
            
            return {
                'total_migrations': len(logs),
                'successful': len(successful),
                'failed': len(failed),
                'average_duration_seconds': avg_duration,
                'last_migration': logs[-1] if logs else None
            }
        
        except Exception as e:
            logger.error(f"Failed to get migration stats: {e}")
            return {}