# Database Seeding Guide

## Overview

The TEKNOFEST 2025 Education Platform includes a comprehensive database seeding system that provides production-ready seed data for different environments.

## Features

- **Environment-specific seeding**: Different data sets for development, staging, and production
- **Comprehensive data models**: All database models are seeded with realistic data
- **Turkish content**: Learning paths and content in Turkish language
- **Security**: Proper password hashing using bcrypt
- **Error handling**: Robust error handling and rollback mechanisms

## Quick Start

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Initialize Database

```bash
# Initialize database tables
make db-init

# Or directly
python scripts/seed_database.py --init-db
```

### Seed Database

```bash
# For development
make db-seed-dev

# For staging
make db-seed-staging

# For production (use with caution!)
make db-seed-prod
```

## Environments

### Development Environment

Creates extensive test data for development:
- Admin, teacher, and 100 student users
- 30 achievements
- 50 published + 10 draft learning paths
- Full user progress data
- Study sessions and notifications

```bash
python scripts/seed_database.py --env development
```

### Staging Environment

Creates moderate test data for staging:
- Admin and 50 test users
- 20 achievements
- 20 learning paths with modules
- User enrollments and progress
- Study sessions

```bash
python scripts/seed_database.py --env staging
```

### Production Environment

Creates minimal essential data for production:
- System users (admin, support, demo)
- 30 core achievements
- 15 published learning paths with full content
- Demo progress data

```bash
python scripts/seed_database.py --env production
```

## Default Credentials

### Production Users

| Role | Email | Password |
|------|-------|----------|
| Admin | admin@teknofest.com | Admin123! |
| Support | support@teknofest.com | Support123! |
| Demo Teacher | demo.teacher@teknofest.com | DemoTeacher123! |
| Demo Student | demo.student@teknofest.com | DemoStudent123! |

⚠️ **IMPORTANT**: Change these passwords immediately after initial setup!

### Development/Staging Users

| Role | Email | Password |
|------|-------|----------|
| Admin | admin@teknofest.com | Admin123! |
| Teacher | teacher@teknofest.com | Teacher123! |
| Students | student{0-99}@test.com | Test123! |

## Learning Paths

The seeder creates comprehensive learning paths in various categories:

### Programming & Software
- Python Programlama Temelleri
- İleri Python Programlama
- Web Geliştirme Temelleri
- React ile Modern Web Uygulamaları

### Data Science & AI
- Veri Bilimi ile Tanışma
- Makine Öğrenmesi 101
- Derin Öğrenme ve Yapay Sinir Ağları
- Yapay Zeka için Matematik

### Robotics & IoT
- Arduino ile Robotik Başlangıç
- IoT ve Akıllı Sistemler

### Other Topics
- Siber Güvenlik Temelleri
- Unity ile Oyun Geliştirme
- Bilimsel Hesaplama ve Simülasyon

## Commands

### View Statistics

```bash
# Show current database statistics
make db-stats

# Or directly
python scripts/seed_database.py --stats
```

### Reset Database

```bash
# Clear and reseed database (development only)
make db-reset

# Or step by step
make db-clear
make db-init
make db-seed-dev
```

### Clear Database

```bash
# Clear all data (development only)
make db-clear

# Or directly
python scripts/seed_database.py --clear
```

## Module Content

Each learning path includes multiple modules with:
- Various content types (video, text, quiz, exercise, project)
- Estimated completion times
- Difficulty levels
- Resources (PDFs, videos)
- Mandatory and optional modules

## Achievements

30 different achievements including:
- İlk Adım (First Step)
- Hızlı Öğrenci (Fast Learner)
- Süreklilik (Consistency)
- Uzman (Expert)
- Mükemmeliyetçi (Perfectionist)
- And many more...

## Safety Features

1. **Environment checks**: Production environment requires confirmation
2. **Rollback support**: Automatic rollback on errors
3. **Idempotent operations**: Safe to run multiple times
4. **Existing data checks**: Prevents duplicate entries

## Troubleshooting

### Database Connection Error

Ensure your database is running and configured:
```bash
# Check .env file
DATABASE_URL=postgresql://user:password@localhost/teknofest_db
```

### Permission Denied

Ensure you have proper database permissions:
```sql
GRANT ALL PRIVILEGES ON DATABASE teknofest_db TO your_user;
```

### Module Import Error

Install all dependencies:
```bash
pip install -r requirements.txt
```

## Advanced Usage

### Custom Seeding

```python
from src.database.seeds import DatabaseSeeder

seeder = DatabaseSeeder()
seeder.seed_all("development")

# Get statistics
stats = seeder.get_seed_stats()
print(stats)
```

### Selective Seeding

```python
from src.database.seeds import DatabaseSeeder
from src.database.base import get_db_context

seeder = DatabaseSeeder()

with get_db_context() as db:
    # Seed only specific data
    seeder._create_admin_user(db)
    seeder._create_achievements(db, count=10)
    seeder._create_learning_paths(db, count=5)
    db.commit()
```

## Best Practices

1. **Always backup** production database before seeding
2. **Test in staging** before applying to production
3. **Change default passwords** immediately after setup
4. **Use migrations** for schema changes, not init_db
5. **Monitor logs** during seeding for any issues

## Support

For issues or questions, please check:
- Project documentation
- GitHub Issues
- Contact the development team