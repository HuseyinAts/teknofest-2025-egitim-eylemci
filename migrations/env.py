"""
Alembic environment configuration
Production-ready migration system with support for online and offline migrations
"""

import os
import sys
from logging.config import fileConfig
from pathlib import Path

from sqlalchemy import engine_from_config
from sqlalchemy import pool
from alembic import context

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import your models and settings
from src.database.base import Base
from src.database.models import *  # Import all models to register them
from src.config import get_settings

# this is the Alembic Config object
config = context.config

# Get settings
settings = get_settings()

# Interpret the config file for Python logging
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Add your model's MetaData object here for 'autogenerate' support
target_metadata = Base.metadata

# Override database URL from environment if available
def get_database_url():
    """Get database URL from settings or environment"""
    # Use environment variable if set (for production)
    url = os.getenv('DATABASE_URL')
    if url:
        # Handle Heroku-style postgres:// URLs
        if url.startswith("postgres://"):
            url = url.replace("postgres://", "postgresql://", 1)
        return url
    
    # Otherwise use settings
    return settings.database_url


def include_object(object, name, type_, reflected, compare_to):
    """
    Filter objects for migrations.
    Exclude certain tables or schemas if needed.
    """
    # Exclude specific schemas
    if type_ == "schema":
        return name not in ["information_schema", "pg_catalog"]
    
    # Exclude specific tables
    if type_ == "table":
        # Add any tables you want to exclude from migrations
        excluded_tables = []
        return name not in excluded_tables
    
    return True


def run_migrations_offline() -> None:
    """
    Run migrations in 'offline' mode.
    
    This configures the context with just a URL
    and not an Engine, though an Engine is acceptable
    here as well. By skipping the Engine creation
    we don't even need a DBAPI to be available.
    
    Calls to context.execute() here emit the given string to the
    script output.
    """
    url = get_database_url()
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        include_object=include_object,
        compare_type=True,
        compare_server_default=True,
        # Include these for better migration generation
        include_schemas=True,
        render_as_batch=False,  # Set to True for SQLite
        version_table='alembic_version',
        version_table_schema=None,
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """
    Run migrations in 'online' mode.
    
    In this scenario we need to create an Engine
    and associate a connection with the context.
    """
    # Get database URL
    database_url = get_database_url()
    
    # Handle async URLs (convert back to sync for migrations)
    if "asyncpg" in database_url:
        database_url = database_url.replace("postgresql+asyncpg://", "postgresql://")
    
    # Update config with current database URL
    configuration = config.get_section(config.config_ini_section)
    configuration['sqlalchemy.url'] = database_url
    
    # Configure connection pool for production
    poolclass = pool.NullPool
    if settings.is_production():
        poolclass = pool.QueuePool
    
    connectable = engine_from_config(
        configuration,
        prefix="sqlalchemy.",
        poolclass=poolclass,
        pool_size=settings.database_pool_size,
        max_overflow=settings.database_max_overflow,
        pool_timeout=30,
        pool_recycle=3600,
        pool_pre_ping=True,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            include_object=include_object,
            compare_type=True,
            compare_server_default=True,
            include_schemas=True,
            render_as_batch=False,
            version_table='alembic_version',
            version_table_schema=None,
            # Transaction per migration
            transaction_per_migration=True,
            # Configure for proper constraint naming
            render_item=render_item,
        )

        with context.begin_transaction():
            # Set lock timeout for migrations
            if settings.is_production():
                connection.execute("SET lock_timeout = '10s'")
                connection.execute("SET statement_timeout = '30s'")
            
            context.run_migrations()


def render_item(type_, obj, autogen_context):
    """
    Custom rendering for certain database objects.
    Helps with PostgreSQL-specific features.
    """
    # Handle PostgreSQL-specific index methods
    if type_ == "index":
        if hasattr(obj, 'kwargs') and 'postgresql_using' in obj.kwargs:
            # Handle special index types like GIN, GIST, etc.
            return None
    
    # Default rendering
    return False


# Determine migration mode and run
if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()