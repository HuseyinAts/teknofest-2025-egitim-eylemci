# Database Migration System - Production Ready Guide

## 📋 Overview

This document describes the production-ready database migration system for the TEKNOFEST 2025 Education Platform. The system uses Alembic for migrations, PostgreSQL as the primary database, and includes comprehensive health checks, backup systems, and seed data management.

## 🏗️ Architecture

### Components

1. **Alembic** - Database migration tool
2. **SQLAlchemy** - ORM and database abstraction
3. **PostgreSQL** - Primary database (with asyncpg for async operations)
4. **Health Checker** - Comprehensive database health monitoring
5. **Seed System** - Environment-specific data seeding
6. **Backup System** - Automated backup and restore functionality

### Directory Structure

```
teknofest-2025-egitim-eylemci/
├── alembic.ini                 # Alembic configuration
├── migrations/                  # Migration files
│   ├── env.py                  # Alembic environment config
│   ├── script.py.mako          # Migration template
│   └── versions/               # Migration versions
├── src/
│   └── database/
│       ├── __init__.py         # Database module exports
│       ├── base.py             # Base configuration
│       ├── session.py          # Session management
│       ├── models.py           # Database models
│       ├── migrations.py       # Migration utilities
│       ├── seeds.py            # Seed data system
│       └── health.py           # Health checks
└── manage_db.py                # CLI management tool
```

## 🚀 Quick Start

### 1. Initial Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Initialize migration system
python manage_db.py migrate init

# Create initial migration
python manage_db.py migrate create "Initial schema" --autogenerate

# Run migrations
python manage_db.py migrate up
```

### 2. Seed Development Data

```bash
# Seed development environment
python manage_db.py seed run

# View seed statistics
python manage_db.py seed stats
```

### 3. Check Database Health

```bash
# Quick health check
python manage_db.py health

# Detailed health check
python manage_db.py health --detailed
```

## 📝 Database Models

### Core Models

- **User** - User accounts with roles (Student, Teacher, Admin, Parent)
- **LearningPath** - Educational content paths
- **Module** - Individual learning modules
- **StudySession** - User study tracking
- **Assessment** - Quizzes and evaluations
- **Achievement** - Gamification badges
- **Progress** - User progress tracking
- **Notification** - User notifications
- **AuditLog** - System audit trail

### Model Features

- UUID primary keys for all tables
- Timestamp mixins (created_at, updated_at)
- Proper indexes for performance
- Check constraints for data integrity
- JSONB fields for flexible data storage
- Enum types for predefined values

## 🔄 Migration Commands

### Creating Migrations

```bash
# Auto-generate migration from model changes
python manage_db.py migrate create "Add user preferences" --autogenerate

# Create empty migration
python manage_db.py migrate create "Custom migration" --no-autogenerate
```

### Running Migrations

```bash
# Migrate to latest version
python manage_db.py migrate up

# Migrate to specific revision
python manage_db.py migrate up --revision abc123

# Generate SQL without executing (for review)
python manage_db.py migrate up --sql
```

### Rolling Back

```bash
# Rollback one migration
python manage_db.py migrate down

# Rollback to specific revision
python manage_db.py migrate down --revision abc123
```

### Status and History

```bash
# Check migration status
python manage_db.py migrate status

# View pending migrations
python manage_db.py migrate status
```

## 🌱 Seed Data Management

### Environment-Specific Seeding

The system provides different seed data for each environment:

#### Development
- 100 test users
- 50 learning paths
- 30 achievements
- Sample study sessions
- Test notifications

```bash
python manage_db.py seed run --env development
```

#### Staging
- 50 test users
- 20 learning paths
- 20 achievements
- Limited test data

```bash
python manage_db.py seed run --env staging
```

#### Production
- Admin user only
- Essential achievements
- 5 sample learning paths

```bash
python manage_db.py seed run --env production
```

### Clearing Data

```bash
# Clear all data (development only)
python manage_db.py seed clear

# Seed with fresh data
python manage_db.py seed run --clear
```

## 🏥 Health Monitoring

### Health Checks

The system performs comprehensive health checks:

1. **Connectivity** - Basic database connection
2. **Performance** - Query response times
3. **Replication** - Replication status and lag
4. **Disk Usage** - Database and table sizes
5. **Connection Pool** - Pool statistics
6. **Long Queries** - Detect slow queries
7. **Table Integrity** - Check constraints and keys
8. **Indexes** - Unused and invalid indexes
9. **Locks** - Blocking locks detection
10. **Cache Hit Ratio** - Buffer cache efficiency

### Running Health Checks

```bash
# Basic health check
python manage_db.py health

# Detailed health check
python manage_db.py health --detailed
```

### Programmatic Health Checks

```python
from src.database.health import check_database_health

# Run complete health check
results = check_database_health()

# Check if healthy
if results['status'] == 'healthy':
    print("Database is healthy")
```

## 💾 Backup and Restore

### Creating Backups

```bash
# Create backup (auto-named with timestamp)
python manage_db.py backup

# Create backup with custom path
python manage_db.py backup --output /path/to/backup.sql
```

### Restoring from Backup

```bash
# Restore from backup
python manage_db.py restore /path/to/backup.sql
```

### Automated Backups

Production migrations automatically create backups before running:

```python
# In production, migrations create backups
manager = MigrationManager()
manager.run_migrations()  # Auto-backup if production
```

## 🔒 Production Best Practices

### 1. Pre-Migration Checklist

- [ ] Review migration SQL with `--sql` flag
- [ ] Test migration in staging environment
- [ ] Create manual backup
- [ ] Check database health
- [ ] Schedule during maintenance window
- [ ] Notify team members

### 2. Migration Process

```bash
# 1. Check current status
python manage_db.py migrate status

# 2. Review SQL
python manage_db.py migrate up --sql > migration.sql

# 3. Create backup
python manage_db.py backup

# 4. Run health check
python manage_db.py health

# 5. Execute migration
python manage_db.py migrate up

# 6. Verify
python manage_db.py health --detailed
```

### 3. Rollback Process

```bash
# 1. Identify issue
python manage_db.py migrate status

# 2. Rollback
python manage_db.py migrate down

# 3. Or restore from backup
python manage_db.py restore backup_20240101_120000.sql
```

### 4. Connection Pool Settings

Production configuration in `src/config.py`:

```python
database_pool_size: int = 5
database_max_overflow: int = 10
database_echo: bool = False
```

### 5. Security Considerations

- Never commit `.env` files with credentials
- Use environment variables for sensitive data
- Rotate database passwords regularly
- Enable SSL for database connections
- Implement row-level security where needed

## 🔧 Configuration

### Environment Variables

```bash
# .env file
DATABASE_URL=postgresql://user:pass@localhost:5432/dbname
DATABASE_POOL_SIZE=5
DATABASE_MAX_OVERFLOW=10
DATABASE_ECHO=false

# For async operations
DATABASE_ASYNC_URL=postgresql+asyncpg://user:pass@localhost:5432/dbname
```

### Alembic Configuration

Edit `alembic.ini`:

```ini
[alembic]
script_location = migrations
file_template = %%(year)d%%(month).2d%%(day).2d_%%(hour).2d%%(minute).2d-%%(rev)s_%%(slug)s
```

## 🐛 Troubleshooting

### Common Issues

#### 1. Migration Conflicts

```bash
# Check current revision
python manage_db.py migrate status

# Force revision
alembic stamp head
```

#### 2. Connection Pool Exhaustion

```python
# Check pool stats
from src.database.session import get_db_stats
stats = get_db_stats()
print(stats)
```

#### 3. Lock Issues

```bash
# Check for locks
python manage_db.py health --detailed

# Kill blocking queries (PostgreSQL)
SELECT pg_terminate_backend(pid) 
FROM pg_stat_activity 
WHERE state = 'idle in transaction';
```

#### 4. Failed Migration

```bash
# Rollback
python manage_db.py migrate down

# Or restore backup
python manage_db.py restore last_backup.sql
```

## 📊 Monitoring

### Key Metrics to Monitor

1. **Connection pool utilization**
2. **Query response times**
3. **Cache hit ratio (target: >90%)**
4. **Replication lag (if applicable)**
5. **Long-running queries**
6. **Database size growth**
7. **Lock wait times**

### Integration with Monitoring Tools

```python
# Export metrics for Prometheus/Grafana
from src.database.health import get_health_summary

metrics = get_health_summary()
# Export to monitoring system
```

## 🔍 Database Queries

### Useful Administrative Queries

```sql
-- Active connections
SELECT * FROM pg_stat_activity;

-- Database size
SELECT pg_size_pretty(pg_database_size('teknofest_dev'));

-- Table sizes
SELECT 
    schemaname,
    tablename,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) AS size
FROM pg_tables 
WHERE schemaname NOT IN ('pg_catalog', 'information_schema')
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;

-- Cache hit ratio
SELECT 
    sum(heap_blks_hit) / (sum(heap_blks_hit) + sum(heap_blks_read)) as cache_hit_ratio
FROM pg_statio_user_tables;

-- Unused indexes
SELECT 
    schemaname,
    tablename,
    indexname,
    idx_scan
FROM pg_stat_user_indexes
WHERE idx_scan = 0
ORDER BY pg_relation_size(indexrelid) DESC;
```

## 📚 Additional Resources

- [Alembic Documentation](https://alembic.sqlalchemy.org/)
- [SQLAlchemy Documentation](https://www.sqlalchemy.org/)
- [PostgreSQL Documentation](https://www.postgresql.org/docs/)
- [Database Design Best Practices](https://www.postgresql.org/docs/current/ddl.html)

## 🤝 Contributing

When contributing database changes:

1. Always create migrations for schema changes
2. Test migrations in development first
3. Include rollback migrations
4. Update seed data if needed
5. Document significant changes
6. Add appropriate indexes
7. Consider performance implications

## 📄 License

This migration system is part of the TEKNOFEST 2025 Education Platform.