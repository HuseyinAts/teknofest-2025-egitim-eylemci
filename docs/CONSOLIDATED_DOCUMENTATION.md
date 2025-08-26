# Teknofest 2025 Education Platform - Consolidated Documentation

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Architecture](#architecture)
4. [Development](#development)
5. [Deployment](#deployment)
6. [Features](#features)
7. [API Documentation](#api-documentation)
8. [Configuration](#configuration)
9. [Testing](#testing)
10. [Monitoring](#monitoring)
11. [Troubleshooting](#troubleshooting)

---

## Overview

The Teknofest 2025 Education Platform is an AI-powered adaptive learning system designed specifically for Turkish K-12 students. It features personalized learning paths, intelligent assessment, and comprehensive gamification.

### Key Features
- 🤖 AI-powered personalized learning with Turkish NLP
- 📚 Adaptive curriculum based on MEB standards
- 🎮 Gamification and achievement system
- 📊 IRT-based assessment engine
- 🌐 Offline mode support
- 📈 Real-time progress tracking
- 🔒 Enterprise-grade security

### Technology Stack
- **Backend**: FastAPI, Python 3.11+
- **Frontend**: Next.js 14, React 18
- **Database**: PostgreSQL 15
- **Cache**: Redis 7
- **AI/ML**: PyTorch, Transformers, Turkish NLP models
- **Monitoring**: Sentry, OpenTelemetry, Prometheus
- **Container**: Docker, Kubernetes-ready

---

## Quick Start

### Prerequisites
- Python 3.11+
- Node.js 18+
- Docker & Docker Compose
- PostgreSQL 15
- Redis 7

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/HuseyinAts/teknofest-2025-egitim-eylemci.git
cd teknofest-2025-egitim-eylemci
```

2. **Set up environment**
```bash
# Copy environment template
cp configs/production.env .env

# Generate secure keys
python generate_secure_keys.py
```

3. **Install dependencies**
```bash
# Backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Frontend
cd frontend
npm install
cd ..
```

4. **Initialize database**
```bash
# Run migrations
alembic upgrade head

# Seed initial data
python scripts/seed_database.py
```

5. **Start services**
```bash
# Using Docker Compose
docker-compose up

# Or manually
# Terminal 1: Backend
uvicorn src.app:app --reload

# Terminal 2: Frontend
cd frontend && npm run dev
```

6. **Access the application**
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API Docs: http://localhost:8000/docs

---

## Architecture

### System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         Frontend (Next.js)                   │
│  ┌─────────┐  ┌──────────┐  ┌─────────┐  ┌──────────┐    │
│  │  Pages  │  │Components│  │  Store  │  │ Services │    │
│  └────┬────┘  └─────┬────┘  └────┬────┘  └─────┬────┘    │
└───────┼─────────────┼────────────┼─────────────┼──────────┘
        │             │            │             │
        └─────────────┴────────────┴─────────────┘
                           │
                    ┌──────▼──────┐
                    │   Nginx     │
                    │ Load Balancer│
                    └──────┬──────┘
                           │
        ┌──────────────────┼──────────────────┐
        │                  │                  │
┌───────▼─────┐    ┌───────▼─────┐    ┌──────▼──────┐
│   FastAPI   │    │   FastAPI   │    │   FastAPI   │
│  Worker 1   │    │  Worker 2   │    │  Worker N   │
└──────┬──────┘    └──────┬──────┘    └──────┬──────┘
       │                  │                   │
       └──────────────────┼───────────────────┘
                          │
         ┌────────────────┼────────────────┐
         │                │                │
    ┌────▼────┐    ┌──────▼──────┐  ┌─────▼─────┐
    │PostgreSQL│    │    Redis    │  │  AI/ML    │
    │ Primary  │    │   Cache     │  │  Models   │
    └─────────┘    └─────────────┘  └───────────┘
```

### Directory Structure

```
teknofest-2025-egitim-eylemci/
├── src/                    # Backend source code
│   ├── api/               # API endpoints
│   ├── core/              # Core functionality
│   ├── database/          # Database models and operations
│   ├── agents/            # AI agents
│   ├── nlp/              # Turkish NLP modules
│   └── utils/            # Utilities
├── frontend/              # Next.js frontend
│   ├── src/
│   │   ├── app/          # App router pages
│   │   ├── components/   # React components
│   │   ├── store/        # Redux store
│   │   └── lib/          # Libraries
├── tests/                 # Test suites
│   ├── unit/
│   ├── integration/
│   └── e2e/
├── migrations/            # Database migrations
├── configs/              # Configuration files
├── scripts/              # Utility scripts
├── monitoring/           # Monitoring configs
└── docs/                # Documentation
```

---

## Development

### Development Setup

1. **Install development dependencies**
```bash
pip install -r requirements-test.txt
npm install --save-dev
```

2. **Set up pre-commit hooks**
```bash
pre-commit install
```

3. **Run in development mode**
```bash
# Backend with hot reload
uvicorn src.app:app --reload --host 0.0.0.0 --port 8000

# Frontend with hot reload
cd frontend && npm run dev
```

### Code Style

- **Python**: Black, isort, flake8
- **JavaScript/TypeScript**: ESLint, Prettier
- **Commit messages**: Conventional Commits

### Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test types
pytest tests/unit/
pytest tests/integration/
pytest tests/e2e/

# Frontend tests
cd frontend && npm test
```

---

## Deployment

### Production Deployment

1. **Environment Setup**
```bash
# Create production environment file
cp configs/production.env .env.production

# Set production variables
export ENVIRONMENT=production
export DATABASE_URL=postgresql://user:pass@host/db
export REDIS_URL=redis://host:6379
export SECRET_KEY=$(python -c 'import secrets; print(secrets.token_hex(32))')
```

2. **Docker Deployment**
```bash
# Build production image
docker build -f Dockerfile.production -t teknofest-api:latest .

# Run with Docker Compose
docker-compose -f docker-compose.production.yml up -d
```

3. **Kubernetes Deployment**
```bash
# Apply configurations
kubectl apply -f k8s/production-deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/ingress.yaml
```

### Multi-Worker Deployment

```bash
# Start with multiple workers
docker-compose -f docker-compose.production-multiworker.yml up -d

# Scale workers
docker-compose scale api=4
```

### Database Migrations

```bash
# Create new migration
alembic revision --autogenerate -m "Description"

# Apply migrations
alembic upgrade head

# Rollback
alembic downgrade -1
```

---

## Features

### 1. Turkish NLP Integration

- **Morphological Analysis**: Zemberek integration
- **Tokenization**: Custom Turkish tokenizer
- **Named Entity Recognition**: Turkish NER models
- **Sentiment Analysis**: Turkish sentiment models

### 2. Adaptive Learning System

- **IRT Engine**: Item Response Theory implementation
- **Personalized Paths**: AI-generated learning paths
- **Difficulty Adjustment**: Real-time difficulty adaptation
- **Progress Tracking**: Comprehensive analytics

### 3. Gamification

- **Points System**: XP and level progression
- **Achievements**: Unlockable badges and rewards
- **Leaderboards**: School and national rankings
- **Streaks**: Daily learning streaks

### 4. Offline Mode

- **Service Worker**: PWA with offline support
- **Local Storage**: IndexedDB for offline data
- **Sync Queue**: Automatic sync when online
- **Cached Resources**: Offline-first approach

### 5. Security Features

- **JWT Authentication**: Secure token-based auth
- **Rate Limiting**: API rate limiting
- **Input Validation**: Comprehensive validation
- **SQL Injection Protection**: Parameterized queries
- **XSS Protection**: Content Security Policy

---

## API Documentation

### Authentication

```http
POST /api/v1/auth/register
Content-Type: application/json

{
  "username": "student123",
  "email": "student@example.com",
  "password": "SecurePass123!",
  "grade_level": 10
}
```

```http
POST /api/v1/auth/login
Content-Type: application/json

{
  "email": "student@example.com",
  "password": "SecurePass123!"
}
```

### Learning Paths

```http
GET /api/v1/learning-paths
Authorization: Bearer <token>

POST /api/v1/learning-paths/generate
Authorization: Bearer <token>
Content-Type: application/json

{
  "subject": "mathematics",
  "grade_level": 10,
  "learning_style": "visual"
}
```

### Quiz System

```http
GET /api/v1/quiz/categories

GET /api/v1/quiz/{quiz_id}
Authorization: Bearer <token>

POST /api/v1/quiz/{quiz_id}/submit
Authorization: Bearer <token>
Content-Type: application/json

{
  "answers": [
    {"question_id": 1, "answer": "A"},
    {"question_id": 2, "answer": "B"}
  ]
}
```

### AI Study Buddy

```http
POST /api/v1/study-buddy/chat
Authorization: Bearer <token>
Content-Type: application/json

{
  "message": "Üçgenin iç açıları toplamı nedir?",
  "context": {
    "subject": "geometry",
    "grade": 9
  }
}
```

---

## Configuration

### Environment Variables

```env
# Application
ENVIRONMENT=production
SECRET_KEY=your-secret-key
APP_VERSION=1.0.0

# Database
DATABASE_URL=postgresql://user:password@localhost/teknofest
DATABASE_POOL_SIZE=20
DATABASE_MAX_OVERFLOW=40

# Redis
REDIS_URL=redis://localhost:6379/0
REDIS_POOL_SIZE=10

# AI/ML
MODEL_PATH=/models
TURKISH_NLP_MODEL=dbmdz/bert-base-turkish-cased
INFERENCE_BATCH_SIZE=32

# Monitoring
SENTRY_DSN=https://your-sentry-dsn
OTEL_ENDPOINT=localhost:4317

# Security
JWT_SECRET_KEY=your-jwt-secret
JWT_ALGORITHM=HS256
JWT_EXPIRATION_HOURS=24

# Rate Limiting
RATE_LIMIT_REQUESTS=100
RATE_LIMIT_PERIOD=60
```

### Configuration Files

- `configs/production.yaml` - Production settings
- `config/sentry.yaml` - Sentry configuration
- `nginx/nginx.prod.conf` - Nginx configuration
- `monitoring/prometheus.yml` - Prometheus config
- `k8s/production-deployment.yaml` - Kubernetes config

---

## Testing

### Test Structure

```
tests/
├── unit/              # Unit tests
│   ├── test_services.py
│   ├── test_api_endpoints.py
│   └── test_models.py
├── integration/       # Integration tests
│   ├── test_database.py
│   ├── test_redis.py
│   └── test_external_apis.py
├── e2e/              # End-to-end tests
│   ├── test_user_journey.py
│   ├── test_full_system_integration.py
│   └── test_performance_e2e.py
└── load/             # Load tests
    └── locustfile.py
```

### Running Tests

```bash
# Unit tests
pytest tests/unit/ -v

# Integration tests
pytest tests/integration/ -v

# E2E tests
pytest tests/e2e/ -v

# Load tests
locust -f tests/load/locustfile.py --host=http://localhost:8000

# Coverage report
pytest --cov=src --cov-report=html
open htmlcov/index.html
```

### E2E Test Suite

```bash
# Run full E2E suite
./tests/e2e/run_e2e_suite.sh full

# Run specific test types
./tests/e2e/run_e2e_suite.sh performance
./tests/e2e/run_e2e_suite.sh integration
./tests/e2e/run_e2e_suite.sh user-journey
```

---

## Monitoring

### Sentry Integration

```python
# Automatic error tracking
import sentry_sdk

sentry_sdk.init(
    dsn="your-sentry-dsn",
    environment="production",
    traces_sample_rate=0.1
)
```

### Prometheus Metrics

- Request count and latency
- Database query performance
- AI model inference time
- Cache hit rates
- Error rates

### Health Checks

```http
GET /health
GET /ready
GET /metrics
```

### Logging

```python
import logging

logger = logging.getLogger(__name__)
logger.info("Application started")
logger.error("Error occurred", exc_info=True)
```

---

## Troubleshooting

### Common Issues

#### Database Connection Issues
```bash
# Check PostgreSQL status
docker-compose ps postgres

# View logs
docker-compose logs postgres

# Test connection
psql $DATABASE_URL
```

#### Redis Connection Issues
```bash
# Check Redis status
docker-compose ps redis

# Test connection
redis-cli ping
```

#### Frontend Build Issues
```bash
# Clear cache
rm -rf frontend/.next
rm -rf frontend/node_modules

# Reinstall dependencies
cd frontend && npm install
```

#### AI Model Loading Issues
```bash
# Check model files
ls -la models/

# Download models
python scripts/download_models.py
```

### Performance Optimization

1. **Database Optimization**
   - Add indexes for frequently queried columns
   - Use connection pooling
   - Optimize queries with EXPLAIN ANALYZE

2. **Caching Strategy**
   - Cache frequently accessed data
   - Use Redis for session storage
   - Implement CDN for static assets

3. **AI Model Optimization**
   - Use quantization for smaller models
   - Implement batch processing
   - Cache inference results

### Support

- **Documentation**: `/docs`
- **API Documentation**: `/api/docs`
- **GitHub Issues**: [Report bugs](https://github.com/HuseyinAts/teknofest-2025-egitim-eylemci/issues)
- **Email**: support@teknofest-education.com

---

## License

This project is licensed under the MIT License. See [LICENSE](../LICENSE) file for details.

## Contributors

- Hüseyin Ateş - Lead Developer
- Teknofest 2025 Team

---

*Last updated: December 2024*