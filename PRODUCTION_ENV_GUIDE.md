# 🔐 Production Environment Configuration Guide

## 📋 Table of Contents
1. [Overview](#overview)
2. [Security First](#security-first)
3. [Environment Setup](#environment-setup)
4. [Configuration Management](#configuration-management)
5. [Deployment](#deployment)
6. [Monitoring](#monitoring)
7. [Troubleshooting](#troubleshooting)

---

## 🎯 Overview

This guide provides comprehensive instructions for setting up production-ready environment configuration for the TEKNOFEST 2025 Educational AI platform.

### What's New
- ✅ Centralized configuration management
- ✅ Environment-specific settings
- ✅ Security validation
- ✅ Docker support
- ✅ Production hardening

---

## 🔒 Security First

### Critical Security Steps

#### 1. **Generate Secure Keys**
```python
# Run this to generate secure keys
python test_config.py

# Or manually:
import secrets
print(f"SECRET_KEY={secrets.token_hex(32)}")
print(f"JWT_SECRET_KEY={secrets.token_hex(32)}")
```

#### 2. **Update .env File**
```bash
# Copy template
cp .env.example .env

# Edit with secure values
nano .env
```

#### 3. **Protect Sensitive Files**
```bash
# Set proper permissions
chmod 600 .env
chmod 600 .env.production

# Verify .gitignore
cat .gitignore | grep env
```

#### 4. **Remove Hardcoded Secrets**
- ❌ NEVER commit real tokens to Git
- ❌ NEVER hardcode secrets in source code
- ✅ ALWAYS use environment variables
- ✅ ALWAYS validate configuration on startup

---

## 🚀 Environment Setup

### Development Environment

```bash
# 1. Create .env file
cp .env.example .env

# 2. Set development values
APP_ENV=development
APP_DEBUG=true
CORS_ORIGINS=http://localhost:3000,http://localhost:3001
RATE_LIMIT_ENABLED=false

# 3. Test configuration
python test_config.py
```

### Staging Environment

```bash
# Use staging-specific values
APP_ENV=staging
APP_DEBUG=false
CORS_ORIGINS=https://staging.yourdomain.com
RATE_LIMIT_ENABLED=true
RATE_LIMIT_REQUESTS=1000
```

### Production Environment

```bash
# Use production values
APP_ENV=production
APP_DEBUG=false
CORS_ORIGINS=https://yourdomain.com,https://www.yourdomain.com
RATE_LIMIT_ENABLED=true
RATE_LIMIT_REQUESTS=100

# Security headers
SECURITY_HEADERS_ENABLED=true
CONTENT_SECURITY_POLICY="default-src 'self';"

# Monitoring
SENTRY_DSN=your_sentry_dsn_here
METRICS_ENABLED=true
```

---

## ⚙️ Configuration Management

### Using the Config Module

```python
from src.config import settings, validate_environment

# Access settings
print(settings.app_env)
print(settings.api_port)
print(settings.is_production())

# Validate configuration
validate_environment()  # Raises exception if invalid

# Get sanitized URLs
print(settings.get_database_url(hide_password=True))
print(settings.get_redis_url(hide_password=True))
```

### Configuration Validation

The system automatically validates:
- ✅ Secret key strength (min 32 chars)
- ✅ Different secrets for different purposes
- ✅ No default passwords in production
- ✅ CORS origins (no wildcards in production)
- ✅ Rate limiting enabled in production
- ✅ Debug mode disabled in production

### Feature Flags

Control features via environment:
```bash
FEATURE_REGISTRATION_ENABLED=true
FEATURE_AI_CHAT=true
FEATURE_ANALYTICS=true
FEATURE_MAINTENANCE_MODE=false
```

---

## 🐳 Docker Deployment

### Development
```bash
# Use development compose
docker-compose up -d
```

### Production
```bash
# Use production compose with env file
docker-compose -f docker-compose.prod.yml --env-file .env.production up -d

# With monitoring stack
docker-compose -f docker-compose.prod.yml --profile monitoring up -d

# With backup service
docker-compose -f docker-compose.prod.yml --profile backup up -d
```

### Environment Variables in Docker

```yaml
# docker-compose.prod.yml
services:
  backend:
    env_file:
      - .env.production
    environment:
      - APP_ENV=${APP_ENV:-production}
      - DATABASE_URL=postgresql://${DB_USER}:${DB_PASSWORD}@postgres:5432/${DB_NAME}
```

---

## 📊 Monitoring

### Health Checks

```bash
# API health
curl http://localhost:8003/health

# With authentication
curl -H "Authorization: Bearer $TOKEN" http://localhost:8003/health
```

### Metrics (Prometheus)

```bash
# Enable metrics
METRICS_ENABLED=true
METRICS_PORT=9090

# Access metrics
curl http://localhost:9090/metrics
```

### Error Tracking (Sentry)

```bash
# Configure Sentry
SENTRY_DSN=https://your_key@sentry.io/project_id
SENTRY_ENVIRONMENT=production
SENTRY_TRACES_SAMPLE_RATE=0.1
```

### Logging

```bash
# Configure logging
LOG_LEVEL=INFO
LOG_FORMAT=json
LOG_FILE=/var/log/teknofest/app.log
LOG_MAX_SIZE=100  # MB
LOG_BACKUP_COUNT=10
```

---

## 🔧 Troubleshooting

### Common Issues

#### 1. **Configuration Not Loading**
```bash
# Check .env file exists
ls -la .env

# Validate syntax
python -c "from dotenv import load_dotenv; load_dotenv(); print('OK')"

# Test configuration
python test_config.py
```

#### 2. **Secret Key Errors**
```python
# Generate new keys
import secrets
print(secrets.token_hex(32))

# Update .env
SECRET_KEY=new_key_here
JWT_SECRET_KEY=different_key_here
```

#### 3. **Database Connection Issues**
```bash
# Test connection
python -c "
from src.config import settings
print(settings.database_url)
"

# Check PostgreSQL
docker-compose ps postgres
docker-compose logs postgres
```

#### 4. **CORS Errors**
```bash
# Check origins
echo $CORS_ORIGINS

# For development
CORS_ORIGINS=http://localhost:3000,http://localhost:3001

# For production
CORS_ORIGINS=https://yourdomain.com
```

### Validation Script

```bash
# Run comprehensive validation
python test_config.py

# Expected output:
# ✅ Configuration is production-ready!
```

---

## 📝 Environment Variables Reference

### Required Variables
| Variable | Description | Example |
|----------|-------------|---------|
| `SECRET_KEY` | Application secret (min 32 chars) | Generated |
| `JWT_SECRET_KEY` | JWT signing key (min 32 chars) | Generated |
| `DATABASE_URL` | PostgreSQL connection | postgresql://user:pass@host/db |
| `APP_ENV` | Environment type | production |

### Optional Variables
| Variable | Description | Default |
|----------|-------------|---------|
| `REDIS_URL` | Redis cache URL | redis://localhost:6379/0 |
| `SENTRY_DSN` | Error tracking | None |
| `HUGGING_FACE_HUB_TOKEN` | AI model token | None |
| `RATE_LIMIT_ENABLED` | Enable rate limiting | false |

### Security Variables
| Variable | Description | Production Value |
|----------|-------------|------------------|
| `APP_DEBUG` | Debug mode | false |
| `CORS_ORIGINS` | Allowed origins | Your domain |
| `RATE_LIMIT_ENABLED` | Rate limiting | true |
| `SECURITY_HEADERS_ENABLED` | Security headers | true |

---

## 🚨 Security Checklist

Before deploying to production:

- [ ] Generated unique SECRET_KEY (min 32 chars)
- [ ] Generated unique JWT_SECRET_KEY (different from SECRET_KEY)
- [ ] Updated database credentials
- [ ] Configured CORS origins (no wildcards)
- [ ] Enabled rate limiting
- [ ] Disabled debug mode
- [ ] Configured monitoring (Sentry/Prometheus)
- [ ] Set up logging
- [ ] Tested with `python test_config.py`
- [ ] Verified .env not in Git
- [ ] Set proper file permissions (chmod 600)
- [ ] Configured HTTPS only
- [ ] Enabled security headers
- [ ] Set up backup strategy
- [ ] Documented emergency procedures

---

## 📚 Additional Resources

- [Pydantic Settings Documentation](https://docs.pydantic.dev/latest/usage/settings/)
- [Docker Compose Environment Variables](https://docs.docker.com/compose/environment-variables/)
- [OWASP Security Headers](https://owasp.org/www-project-secure-headers/)
- [12 Factor App Config](https://12factor.net/config)

---

## 🆘 Support

For issues or questions:
1. Check `test_config.py` output
2. Review `UPDATE_ENV.txt` for quick fixes
3. See `SISTEM_ANALIZ_RAPORU.md` for system overview
4. Create GitHub issue with configuration details (hide secrets!)

---

*Last Updated: August 2025*
*Version: 1.0.0*