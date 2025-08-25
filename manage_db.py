#!/usr/bin/env python
"""
Database Management CLI for TEKNOFEST 2025 Education Platform
Production-ready database management commands
"""

import sys
import click
import logging
from pathlib import Path
from datetime import datetime
from tabulate import tabulate

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.config import get_settings
from src.database import init_db
from src.database.migrations import (
    get_migration_manager,
    check_migrations_status
)
from src.database.seeds import DatabaseSeeder
from src.database.health import check_database_health

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Get settings
settings = get_settings()


@click.group()
@click.option('--env', default=None, help='Environment (development/staging/production)')
@click.pass_context
def cli(ctx, env):
    """Database management commands for TEKNOFEST 2025"""
    ctx.ensure_object(dict)
    ctx.obj['env'] = env or settings.app_env.value
    
    # Safety check for production
    if ctx.obj['env'] == 'production':
        click.echo(click.style("⚠️  WARNING: Running in PRODUCTION mode!", fg='red', bold=True))


@cli.group()
def migrate():
    """Migration management commands"""
    pass


@migrate.command('init')
def migrate_init():
    """Initialize migration system"""
    try:
        click.echo("Initializing migration system...")
        
        # Create migrations directory if it doesn't exist
        migrations_dir = Path("migrations")
        migrations_dir.mkdir(exist_ok=True)
        
        versions_dir = migrations_dir / "versions"
        versions_dir.mkdir(exist_ok=True)
        
        click.echo(click.style("✓ Migration system initialized", fg='green'))
        
    except Exception as e:
        click.echo(click.style(f"✗ Initialization failed: {e}", fg='red'))
        sys.exit(1)


@migrate.command('create')
@click.argument('message')
@click.option('--autogenerate/--no-autogenerate', default=True, help='Auto-detect model changes')
def migrate_create(message, autogenerate):
    """Create a new migration"""
    try:
        click.echo(f"Creating migration: {message}")
        
        manager = get_migration_manager()
        revision = manager.create_migration(message, autogenerate=autogenerate)
        
        if revision:
            click.echo(click.style(f"✓ Created migration: {revision}", fg='green'))
        else:
            click.echo(click.style("✗ Failed to create migration", fg='red'))
            sys.exit(1)
            
    except Exception as e:
        click.echo(click.style(f"✗ Migration creation failed: {e}", fg='red'))
        sys.exit(1)


@migrate.command('up')
@click.option('--revision', default='head', help='Target revision (default: head)')
@click.option('--sql', is_flag=True, help='Generate SQL only without executing')
@click.pass_context
def migrate_up(ctx, revision, sql):
    """Run migrations"""
    env = ctx.obj['env']
    
    # Confirmation for production
    if env == 'production' and not sql:
        if not click.confirm("Are you sure you want to run migrations in PRODUCTION?"):
            click.echo("Migration cancelled")
            return
    
    try:
        click.echo(f"Running migrations to {revision}...")
        
        manager = get_migration_manager()
        
        # Check pending migrations
        pending = manager.get_pending_migrations()
        if pending:
            click.echo(f"Found {len(pending)} pending migration(s)")
            for rev in pending[:5]:  # Show first 5
                click.echo(f"  - {rev}")
            if len(pending) > 5:
                click.echo(f"  ... and {len(pending) - 5} more")
        
        # Run migrations
        success = manager.run_migrations(revision=revision, sql_only=sql)
        
        if success:
            if sql:
                click.echo(click.style("✓ SQL script generated", fg='green'))
            else:
                click.echo(click.style("✓ Migrations completed successfully", fg='green'))
        else:
            click.echo(click.style("✗ Migration failed", fg='red'))
            sys.exit(1)
            
    except Exception as e:
        click.echo(click.style(f"✗ Migration failed: {e}", fg='red'))
        sys.exit(1)


@migrate.command('down')
@click.option('--revision', default='-1', help='Target revision (default: -1 for previous)')
@click.pass_context
def migrate_down(ctx, revision):
    """Rollback migrations"""
    env = ctx.obj['env']
    
    # Confirmation for production
    if env == 'production':
        if not click.confirm(click.style(
            "⚠️  Are you ABSOLUTELY sure you want to rollback in PRODUCTION?", 
            fg='red', bold=True
        )):
            click.echo("Rollback cancelled")
            return
    
    try:
        click.echo(f"Rolling back to {revision}...")
        
        manager = get_migration_manager()
        success = manager.rollback_migration(revision=revision)
        
        if success:
            click.echo(click.style("✓ Rollback completed successfully", fg='green'))
        else:
            click.echo(click.style("✗ Rollback failed", fg='red'))
            sys.exit(1)
            
    except Exception as e:
        click.echo(click.style(f"✗ Rollback failed: {e}", fg='red'))
        sys.exit(1)


@migrate.command('status')
def migrate_status():
    """Show migration status"""
    try:
        status = check_migrations_status()
        
        click.echo("\n" + "="*50)
        click.echo("MIGRATION STATUS")
        click.echo("="*50)
        
        # Current revision
        current = status.get('current_revision', 'None')
        click.echo(f"\nCurrent revision: {click.style(current, fg='cyan', bold=True)}")
        
        # Pending migrations
        pending = status.get('pending_migrations', [])
        if pending:
            click.echo(f"\nPending migrations: {click.style(str(len(pending)), fg='yellow', bold=True)}")
            for rev in pending[:10]:
                click.echo(f"  • {rev}")
            if len(pending) > 10:
                click.echo(f"  ... and {len(pending) - 10} more")
        else:
            click.echo(click.style("\n✓ No pending migrations", fg='green'))
        
        # Health check
        health = status.get('health_check', False)
        health_status = "✓ Healthy" if health else "✗ Unhealthy"
        health_color = 'green' if health else 'red'
        click.echo(f"\nDatabase health: {click.style(health_status, fg=health_color, bold=True)}")
        
        # History
        history = status.get('history', [])
        if history:
            click.echo(f"\nRecent migrations:")
            for item in history[:5]:
                applied_at = item.get('applied_at', 'Unknown')
                if isinstance(applied_at, datetime):
                    applied_at = applied_at.strftime('%Y-%m-%d %H:%M:%S')
                click.echo(f"  • {item['revision']} - {applied_at}")
        
        click.echo("\n" + "="*50)
        
    except Exception as e:
        click.echo(click.style(f"✗ Failed to get status: {e}", fg='red'))
        sys.exit(1)


@cli.group()
def seed():
    """Seed data management commands"""
    pass


@seed.command('run')
@click.option('--clear', is_flag=True, help='Clear existing data first')
@click.pass_context
def seed_run(ctx, clear):
    """Seed database with sample data"""
    env = ctx.obj['env']
    
    # Safety check for production
    if env == 'production':
        if not click.confirm("Are you sure you want to seed the PRODUCTION database?"):
            click.echo("Seeding cancelled")
            return
    
    try:
        seeder = DatabaseSeeder()
        
        if clear:
            if env == 'production':
                click.echo(click.style("Cannot clear data in production!", fg='red'))
                sys.exit(1)
            
            click.echo("Clearing existing data...")
            seeder.clear_all_data()
            click.echo(click.style("✓ Data cleared", fg='green'))
        
        click.echo(f"Seeding database for {env} environment...")
        seeder.seed_all(environment=env)
        
        # Show statistics
        stats = seeder.get_seed_stats()
        click.echo(click.style("\n✓ Database seeded successfully", fg='green'))
        click.echo("\nSeeded data statistics:")
        for key, value in stats.items():
            click.echo(f"  • {key}: {value}")
        
    except Exception as e:
        click.echo(click.style(f"✗ Seeding failed: {e}", fg='red'))
        sys.exit(1)


@seed.command('clear')
@click.pass_context
def seed_clear(ctx):
    """Clear all data from database"""
    env = ctx.obj['env']
    
    if env == 'production':
        click.echo(click.style("Cannot clear data in production!", fg='red'))
        sys.exit(1)
    
    if not click.confirm(click.style(
        f"⚠️  This will DELETE ALL DATA from the {env} database. Are you sure?", 
        fg='red', bold=True
    )):
        click.echo("Operation cancelled")
        return
    
    try:
        seeder = DatabaseSeeder()
        seeder.clear_all_data()
        click.echo(click.style("✓ All data cleared", fg='green'))
        
    except Exception as e:
        click.echo(click.style(f"✗ Clear failed: {e}", fg='red'))
        sys.exit(1)


@seed.command('stats')
def seed_stats():
    """Show seed data statistics"""
    try:
        seeder = DatabaseSeeder()
        stats = seeder.get_seed_stats()
        
        click.echo("\n" + "="*50)
        click.echo("SEED DATA STATISTICS")
        click.echo("="*50 + "\n")
        
        # Format as table
        table_data = [[k.replace('_', ' ').title(), v] for k, v in stats.items()]
        click.echo(tabulate(table_data, headers=['Entity', 'Count'], tablefmt='grid'))
        
        click.echo("\n" + "="*50)
        
    except Exception as e:
        click.echo(click.style(f"✗ Failed to get statistics: {e}", fg='red'))
        sys.exit(1)


@cli.command('health')
@click.option('--detailed', is_flag=True, help='Show detailed health check')
def health(detailed):
    """Check database health"""
    try:
        click.echo("Running database health check...")
        
        results = check_database_health()
        
        # Overall status
        status = results['status']
        status_color = {
            'healthy': 'green',
            'degraded': 'yellow',
            'unhealthy': 'red'
        }.get(status, 'white')
        
        click.echo("\n" + "="*60)
        click.echo("DATABASE HEALTH CHECK")
        click.echo("="*60)
        
        click.echo(f"\nOverall Status: {click.style(status.upper(), fg=status_color, bold=True)}")
        click.echo(f"Timestamp: {results['timestamp']}")
        click.echo(f"Execution Time: {results['execution_time_ms']}ms")
        
        # Summary
        if results.get('errors'):
            click.echo(f"\n{click.style('Errors:', fg='red', bold=True)}")
            for error in results['errors']:
                click.echo(f"  ✗ {error}")
        
        if results.get('warnings'):
            click.echo(f"\n{click.style('Warnings:', fg='yellow', bold=True)}")
            for warning in results['warnings']:
                click.echo(f"  ⚠ {warning}")
        
        # Detailed checks
        if detailed:
            click.echo(f"\n{click.style('Detailed Checks:', bold=True)}")
            
            for check_name, check_result in results['checks'].items():
                status_icon = {
                    'healthy': '✓',
                    'warning': '⚠',
                    'unhealthy': '✗',
                    'error': '✗',
                    'info': 'ℹ'
                }.get(check_result.get('status', 'unknown'), '?')
                
                status_color = {
                    'healthy': 'green',
                    'warning': 'yellow',
                    'unhealthy': 'red',
                    'error': 'red',
                    'info': 'cyan'
                }.get(check_result.get('status', 'unknown'), 'white')
                
                click.echo(f"\n  {status_icon} {check_name}: ", nl=False)
                click.echo(click.style(
                    check_result.get('status', 'unknown'), 
                    fg=status_color
                ))
                
                if 'message' in check_result:
                    click.echo(f"    Message: {check_result['message']}")
                
                if 'metrics' in check_result:
                    click.echo("    Metrics:")
                    for key, value in check_result['metrics'].items():
                        click.echo(f"      • {key}: {value}")
        
        click.echo("\n" + "="*60)
        
        # Exit with error if unhealthy
        if status == 'unhealthy':
            sys.exit(1)
        
    except Exception as e:
        click.echo(click.style(f"✗ Health check failed: {e}", fg='red'))
        sys.exit(1)


@cli.command('init')
@click.pass_context
def init(ctx):
    """Initialize database (create tables)"""
    env = ctx.obj['env']
    
    if env == 'production':
        click.echo(click.style(
            "⚠️  Use migrations for production! Run 'migrate up' instead.", 
            fg='red'
        ))
        sys.exit(1)
    
    try:
        click.echo("Initializing database...")
        init_db()
        click.echo(click.style("✓ Database initialized", fg='green'))
        
    except Exception as e:
        click.echo(click.style(f"✗ Initialization failed: {e}", fg='red'))
        sys.exit(1)


@cli.command('backup')
@click.option('--output', help='Output file path')
@click.pass_context
def backup(ctx, output):
    """Create database backup"""
    try:
        from src.database.migrations import get_migration_manager
        
        manager = get_migration_manager()
        
        click.echo("Creating database backup...")
        backup_path = manager.create_backup()
        
        if output:
            # Move to specified location
            import shutil
            shutil.move(backup_path, output)
            backup_path = output
        
        click.echo(click.style(f"✓ Backup created: {backup_path}", fg='green'))
        
    except Exception as e:
        click.echo(click.style(f"✗ Backup failed: {e}", fg='red'))
        sys.exit(1)


@cli.command('restore')
@click.argument('backup_file')
@click.pass_context
def restore(ctx, backup_file):
    """Restore database from backup"""
    env = ctx.obj['env']
    
    if env == 'production':
        if not click.confirm(click.style(
            "⚠️  Are you ABSOLUTELY sure you want to restore the PRODUCTION database?", 
            fg='red', bold=True
        )):
            click.echo("Restore cancelled")
            return
    
    try:
        from src.database.migrations import get_migration_manager
        
        manager = get_migration_manager()
        
        click.echo(f"Restoring database from {backup_file}...")
        success = manager.restore_backup(backup_file)
        
        if success:
            click.echo(click.style("✓ Database restored successfully", fg='green'))
        else:
            click.echo(click.style("✗ Restore failed", fg='red'))
            sys.exit(1)
        
    except Exception as e:
        click.echo(click.style(f"✗ Restore failed: {e}", fg='red'))
        sys.exit(1)


if __name__ == '__main__':
    cli(obj={})