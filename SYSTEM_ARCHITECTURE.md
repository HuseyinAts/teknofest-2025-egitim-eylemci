# 🏗️ TEKNOFEST 2025 - Sistem Mimarisi ve Teknik Tasarım

---

## 1. 🎯 MİMARİ GENEL BAKIŞ

### Mimari Prensipler
- **Microservices Architecture**: Bağımsız ölçeklenebilir servisler
- **Event-Driven Design**: Asenkron iletişim
- **API-First Approach**: RESTful ve GraphQL API'ler
- **Cloud-Native**: Container-based deployment
- **Domain-Driven Design**: İş mantığı odaklı

### Teknoloji Kararları
```yaml
Programming Languages:
  Backend: Python 3.11+ (FastAPI, Django)
  Frontend: TypeScript 5.0+ (React, Next.js)
  ML/AI: Python (PyTorch, Transformers)
  Scripts: Bash, Python

Databases:
  Primary: PostgreSQL 15
  NoSQL: MongoDB 6.0
  Cache: Redis 7.0
  Search: Elasticsearch 8.0
  
Message Queue:
  Primary: RabbitMQ
  Alternative: Apache Kafka
  
Container:
  Runtime: Docker 24+
  Orchestration: Kubernetes 1.28+
```

---

## 2. 📊 VERİTABANI TASARIMI

### PostgreSQL Schema

```sql
-- Users and Authentication
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email VARCHAR(255) UNIQUE NOT NULL,
    username VARCHAR(100) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    role VARCHAR(50) NOT NULL CHECK (role IN ('student', 'teacher', 'parent', 'admin')),
    is_active BOOLEAN DEFAULT true,
    email_verified BOOLEAN DEFAULT false,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_login TIMESTAMP,
    metadata JSONB DEFAULT '{}'
);

-- Student Profiles
CREATE TABLE student_profiles (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    grade INTEGER NOT NULL CHECK (grade BETWEEN 9 AND 12),
    school_name VARCHAR(255),
    learning_style VARCHAR(50),
    current_level DECIMAL(3,2) DEFAULT 0.5,
    target_level DECIMAL(3,2) DEFAULT 1.0,
    preferences JSONB DEFAULT '{}',
    achievements JSONB DEFAULT '[]',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Learning Paths
CREATE TABLE learning_paths (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    student_id UUID REFERENCES student_profiles(id) ON DELETE CASCADE,
    subject VARCHAR(100) NOT NULL,
    topic VARCHAR(255) NOT NULL,
    total_weeks INTEGER NOT NULL,
    current_week INTEGER DEFAULT 1,
    progress DECIMAL(5,2) DEFAULT 0.0,
    status VARCHAR(50) DEFAULT 'active',
    zpd_levels JSONB NOT NULL,
    weekly_plan JSONB NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP
);

-- Quizzes
CREATE TABLE quizzes (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    title VARCHAR(255) NOT NULL,
    subject VARCHAR(100) NOT NULL,
    topic VARCHAR(255),
    grade_level INTEGER NOT NULL,
    difficulty DECIMAL(3,2) NOT NULL,
    questions JSONB NOT NULL,
    time_limit INTEGER, -- in minutes
    created_by UUID REFERENCES users(id),
    is_adaptive BOOLEAN DEFAULT false,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB DEFAULT '{}'
);

-- Quiz Attempts
CREATE TABLE quiz_attempts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    quiz_id UUID REFERENCES quizzes(id) ON DELETE CASCADE,
    student_id UUID REFERENCES student_profiles(id) ON DELETE CASCADE,
    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP,
    time_spent INTEGER, -- in seconds
    score DECIMAL(5,2),
    answers JSONB NOT NULL,
    irt_analysis JSONB,
    feedback JSONB,
    status VARCHAR(50) DEFAULT 'in_progress'
);

-- Questions Bank
CREATE TABLE questions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    subject VARCHAR(100) NOT NULL,
    topic VARCHAR(255),
    grade_level INTEGER NOT NULL,
    question_text TEXT NOT NULL,
    question_type VARCHAR(50) NOT NULL,
    options JSONB,
    correct_answer JSONB NOT NULL,
    explanation TEXT,
    difficulty DECIMAL(3,2) NOT NULL,
    irt_params JSONB,
    tags TEXT[],
    usage_count INTEGER DEFAULT 0,
    success_rate DECIMAL(5,2),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_by UUID REFERENCES users(id)
);

-- Study Sessions
CREATE TABLE study_sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    student_id UUID REFERENCES student_profiles(id) ON DELETE CASCADE,
    subject VARCHAR(100) NOT NULL,
    topic VARCHAR(255),
    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    ended_at TIMESTAMP,
    duration INTEGER, -- in minutes
    activities JSONB DEFAULT '[]',
    performance_metrics JSONB,
    notes TEXT
);

-- Performance Analytics
CREATE TABLE performance_analytics (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    student_id UUID REFERENCES student_profiles(id) ON DELETE CASCADE,
    period_type VARCHAR(50) NOT NULL, -- daily, weekly, monthly
    period_date DATE NOT NULL,
    subjects JSONB NOT NULL,
    overall_score DECIMAL(5,2),
    time_spent INTEGER, -- in minutes
    quizzes_completed INTEGER DEFAULT 0,
    topics_covered INTEGER DEFAULT 0,
    strengths JSONB DEFAULT '[]',
    weaknesses JSONB DEFAULT '[]',
    recommendations JSONB DEFAULT '[]',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(student_id, period_type, period_date)
);

-- Notifications
CREATE TABLE notifications (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    type VARCHAR(50) NOT NULL,
    title VARCHAR(255) NOT NULL,
    message TEXT NOT NULL,
    data JSONB,
    is_read BOOLEAN DEFAULT false,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    read_at TIMESTAMP
);

-- Gamification
CREATE TABLE gamification_profiles (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    student_id UUID REFERENCES student_profiles(id) ON DELETE CASCADE,
    total_points INTEGER DEFAULT 0,
    current_level INTEGER DEFAULT 1,
    experience_points INTEGER DEFAULT 0,
    badges JSONB DEFAULT '[]',
    achievements JSONB DEFAULT '[]',
    streaks JSONB DEFAULT '{}',
    rank_in_school INTEGER,
    rank_overall INTEGER,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Content Library
CREATE TABLE content_items (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    title VARCHAR(255) NOT NULL,
    type VARCHAR(50) NOT NULL, -- video, pdf, simulation, etc.
    subject VARCHAR(100) NOT NULL,
    topic VARCHAR(255),
    grade_level INTEGER NOT NULL,
    url TEXT,
    file_path TEXT,
    duration INTEGER, -- in minutes for videos
    difficulty DECIMAL(3,2),
    tags TEXT[],
    metadata JSONB DEFAULT '{}',
    view_count INTEGER DEFAULT 0,
    rating DECIMAL(3,2),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_by UUID REFERENCES users(id)
);

-- Create indexes for performance
CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_users_role ON users(role);
CREATE INDEX idx_student_profiles_user_id ON student_profiles(user_id);
CREATE INDEX idx_learning_paths_student_id ON learning_paths(student_id);
CREATE INDEX idx_quiz_attempts_student_id ON quiz_attempts(student_id);
CREATE INDEX idx_quiz_attempts_quiz_id ON quiz_attempts(quiz_id);
CREATE INDEX idx_questions_subject_topic ON questions(subject, topic);
CREATE INDEX idx_study_sessions_student_id ON study_sessions(student_id);
CREATE INDEX idx_performance_analytics_student_period ON performance_analytics(student_id, period_type, period_date);
CREATE INDEX idx_notifications_user_id_unread ON notifications(user_id, is_read);
CREATE INDEX idx_content_items_subject_topic ON content_items(subject, topic);
```

### MongoDB Collections

```javascript
// User Preferences Collection
{
  "_id": ObjectId(),
  "user_id": "uuid",
  "preferences": {
    "theme": "dark",
    "language": "tr",
    "notifications": {
      "email": true,
      "sms": false,
      "push": true
    },
    "study_times": ["20:00-22:00"],
    "subjects_order": ["matematik", "fizik", "kimya"]
  },
  "updated_at": ISODate()
}

// Chat History Collection
{
  "_id": ObjectId(),
  "session_id": "uuid",
  "student_id": "uuid",
  "messages": [
    {
      "role": "user",
      "content": "Newton yasaları nedir?",
      "timestamp": ISODate()
    },
    {
      "role": "assistant",
      "content": "Newton'un üç hareket yasası...",
      "timestamp": ISODate(),
      "metadata": {
        "model": "qwen-turkish",
        "confidence": 0.95
      }
    }
  ],
  "subject": "fizik",
  "created_at": ISODate()
}

// Learning Resources Collection
{
  "_id": ObjectId(),
  "resource_id": "uuid",
  "type": "video",
  "title": "Integral Hesaplama",
  "subject": "matematik",
  "topics": ["integral", "calculus"],
  "grade_levels": [11, 12],
  "content": {
    "url": "https://cdn.example.com/video.mp4",
    "duration": 1200,
    "transcript": "...",
    "subtitles": {...}
  },
  "metadata": {
    "author": "Dr. Ahmet Yılmaz",
    "source": "EBA",
    "license": "CC-BY-SA"
  },
  "engagement": {
    "views": 15234,
    "likes": 892,
    "rating": 4.7,
    "comments": []
  },
  "created_at": ISODate()
}
```

### Redis Cache Structure

```yaml
# User Sessions
session:{user_id}:
  - token
  - role
  - permissions
  - last_activity
  TTL: 24 hours

# Learning Path Cache
learning_path:{student_id}:{subject}:
  - current_week
  - progress
  - next_topics
  TTL: 1 hour

# Quiz Cache
quiz:{quiz_id}:
  - questions
  - options
  - time_limit
  TTL: 30 minutes

# Leaderboard
leaderboard:global:
  - sorted set of student_id:points
  TTL: 5 minutes

leaderboard:school:{school_id}:
  - sorted set of student_id:points
  TTL: 5 minutes

# Real-time Analytics
analytics:active_users:
  - set of user_ids
  TTL: 1 minute

analytics:quiz_attempts:{date}:
  - counter
  TTL: 1 day
```

---

## 3. 🔄 API TASARIMI

### RESTful API Endpoints

```yaml
# Authentication
POST   /api/v1/auth/register
POST   /api/v1/auth/login
POST   /api/v1/auth/refresh
POST   /api/v1/auth/logout
POST   /api/v1/auth/forgot-password
POST   /api/v1/auth/reset-password

# User Management
GET    /api/v1/users/profile
PUT    /api/v1/users/profile
DELETE /api/v1/users/account
GET    /api/v1/users/preferences
PUT    /api/v1/users/preferences

# Learning Paths
GET    /api/v1/learning-paths
POST   /api/v1/learning-paths
GET    /api/v1/learning-paths/{id}
PUT    /api/v1/learning-paths/{id}/progress
DELETE /api/v1/learning-paths/{id}

# Quizzes
GET    /api/v1/quizzes
POST   /api/v1/quizzes/generate
GET    /api/v1/quizzes/{id}
POST   /api/v1/quizzes/{id}/attempt
PUT    /api/v1/quizzes/{id}/attempt/{attempt_id}
GET    /api/v1/quizzes/{id}/results

# Questions & Answers
POST   /api/v1/questions/ask
GET    /api/v1/questions/history
POST   /api/v1/questions/feedback

# Analytics
GET    /api/v1/analytics/dashboard
GET    /api/v1/analytics/performance
GET    /api/v1/analytics/progress
GET    /api/v1/analytics/recommendations

# Content
GET    /api/v1/content
GET    /api/v1/content/{id}
POST   /api/v1/content/search
GET    /api/v1/content/recommended

# Gamification
GET    /api/v1/gamification/profile
GET    /api/v1/gamification/leaderboard
GET    /api/v1/gamification/achievements
POST   /api/v1/gamification/claim-reward
```

### GraphQL Schema

```graphql
type Query {
  # User queries
  me: User!
  user(id: ID!): User
  
  # Learning queries
  learningPath(id: ID!): LearningPath
  myLearningPaths: [LearningPath!]!
  
  # Quiz queries
  quiz(id: ID!): Quiz
  quizzes(filter: QuizFilter): [Quiz!]!
  
  # Analytics queries
  performanceAnalytics(period: AnalyticsPeriod!): Analytics!
  recommendations: [Recommendation!]!
  
  # Content queries
  content(id: ID!): Content
  searchContent(query: String!, filters: ContentFilter): [Content!]!
}

type Mutation {
  # Auth mutations
  register(input: RegisterInput!): AuthPayload!
  login(input: LoginInput!): AuthPayload!
  
  # Learning mutations
  createLearningPath(input: LearningPathInput!): LearningPath!
  updateProgress(id: ID!, progress: Float!): LearningPath!
  
  # Quiz mutations
  generateQuiz(input: QuizGenerationInput!): Quiz!
  submitQuizAttempt(input: QuizAttemptInput!): QuizResult!
  
  # Q&A mutations
  askQuestion(question: String!, context: String): Answer!
  provideFeedback(answerId: ID!, rating: Int!): Feedback!
}

type Subscription {
  # Real-time updates
  learningProgress(studentId: ID!): ProgressUpdate!
  quizUpdate(quizId: ID!): QuizUpdate!
  notification(userId: ID!): Notification!
}
```

---

## 4. 🔐 GÜVENLİK MİMARİSİ

### Authentication Flow

```mermaid
sequenceDiagram
    participant User
    participant Frontend
    participant API Gateway
    participant Auth Service
    participant Database
    
    User->>Frontend: Login credentials
    Frontend->>API Gateway: POST /auth/login
    API Gateway->>Auth Service: Validate credentials
    Auth Service->>Database: Check user
    Database-->>Auth Service: User data
    Auth Service->>Auth Service: Generate JWT
    Auth Service-->>API Gateway: JWT + Refresh token
    API Gateway-->>Frontend: Auth tokens
    Frontend->>Frontend: Store in secure storage
    Frontend-->>User: Login success
```

### Security Measures

```yaml
Authentication:
  - JWT with RS256 signing
  - Refresh token rotation
  - MFA support (TOTP)
  - Session management

Authorization:
  - RBAC (Role-Based Access Control)
  - Resource-level permissions
  - API key management
  - OAuth2 support

Data Protection:
  - Encryption at rest (AES-256)
  - Encryption in transit (TLS 1.3)
  - PII data masking
  - GDPR compliance

API Security:
  - Rate limiting
  - IP whitelisting
  - Request signing
  - CORS policy

Application Security:
  - Input validation
  - SQL injection prevention
  - XSS protection
  - CSRF tokens
```

---

## 5. 🚀 DEPLOYMENT ARCHİTECTURE

### Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: teknofest-backend
spec:
  replicas: 3
  selector:
    matchLabels:
      app: backend
  template:
    metadata:
      labels:
        app: backend
    spec:
      containers:
      - name: api
        image: teknofest/backend:latest
        ports:
        - containerPort: 8000
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: db-secret
              key: url
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "512Mi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: backend-service
spec:
  selector:
    app: backend
  ports:
    - protocol: TCP
      port: 80
      targetPort: 8000
  type: LoadBalancer
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: backend-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: teknofest-backend
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

### Docker Compose (Development)

```yaml
version: '3.8'

services:
  postgres:
    image: postgres:15-alpine
    environment:
      POSTGRES_DB: teknofest
      POSTGRES_USER: admin
      POSTGRES_PASSWORD: ${DB_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"

  redis:
    image: redis:7-alpine
    command: redis-server --appendonly yes
    volumes:
      - redis_data:/data
    ports:
      - "6379:6379"

  mongodb:
    image: mongo:6
    environment:
      MONGO_INITDB_ROOT_USERNAME: admin
      MONGO_INITDB_ROOT_PASSWORD: ${MONGO_PASSWORD}
    volumes:
      - mongo_data:/data/db
    ports:
      - "27017:27017"

  backend:
    build: ./backend
    depends_on:
      - postgres
      - redis
      - mongodb
    environment:
      DATABASE_URL: postgresql://admin:${DB_PASSWORD}@postgres:5432/teknofest
      REDIS_URL: redis://redis:6379
      MONGODB_URL: mongodb://admin:${MONGO_PASSWORD}@mongodb:27017
    volumes:
      - ./backend:/app
    ports:
      - "8000:8000"
    command: uvicorn main:app --reload --host 0.0.0.0

  frontend:
    build: ./frontend
    depends_on:
      - backend
    environment:
      NEXT_PUBLIC_API_URL: http://localhost:8000
    volumes:
      - ./frontend:/app
      - /app/node_modules
    ports:
      - "3000:3000"
    command: npm run dev

  mcp-server:
    build: ./mcp
    depends_on:
      - backend
    environment:
      API_URL: http://backend:8000
      HUGGING_FACE_TOKEN: ${HF_TOKEN}
    volumes:
      - ./mcp:/app
    ports:
      - "8080:8080"

volumes:
  postgres_data:
  redis_data:
  mongo_data:
```

---

## 6. 📊 MONITORING VE OBSERVABILITY

### Metrics Collection

```yaml
Prometheus Metrics:
  - API response times
  - Request rates
  - Error rates
  - Database query performance
  - Cache hit rates
  - Model inference times
  - Queue depths
  - Resource utilization

Custom Metrics:
  - Active users
  - Quiz completion rates
  - Learning path progress
  - Question answering accuracy
  - Content engagement
  - Student performance trends
```

### Logging Strategy

```yaml
Log Levels:
  ERROR: System errors, exceptions
  WARN: Performance issues, deprecations
  INFO: User actions, API calls
  DEBUG: Detailed execution flow

Log Structure:
  timestamp: ISO 8601
  level: ERROR|WARN|INFO|DEBUG
  service: service name
  trace_id: correlation ID
  user_id: if applicable
  message: log message
  metadata: additional context

Log Destinations:
  - Elasticsearch (long-term storage)
  - CloudWatch/Stackdriver (cloud)
  - Local files (development)
```

### Alerting Rules

```yaml
Critical Alerts:
  - API availability < 99%
  - Database connection failures
  - Memory usage > 90%
  - Error rate > 5%
  - Response time > 2s (p95)

Warning Alerts:
  - CPU usage > 80%
  - Disk usage > 80%
  - Queue depth > 1000
  - Cache miss rate > 30%
  - Failed login attempts > 10/min
```

---

## 7. 🔄 CI/CD PIPELINE

### GitHub Actions Workflow

```yaml
name: CI/CD Pipeline

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install -r requirements-test.txt
      
      - name: Run tests
        run: |
          pytest --cov=app --cov-report=xml
      
      - name: Upload coverage
        uses: codecov/codecov-action@v3

  build:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Build Docker image
        run: |
          docker build -t teknofest/backend:${{ github.sha }} .
      
      - name: Push to registry
        run: |
          echo ${{ secrets.DOCKER_PASSWORD }} | docker login -u ${{ secrets.DOCKER_USERNAME }} --password-stdin
          docker push teknofest/backend:${{ github.sha }}

  deploy:
    needs: build
    if: github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    steps:
      - name: Deploy to Kubernetes
        run: |
          kubectl set image deployment/backend backend=teknofest/backend:${{ github.sha }}
          kubectl rollout status deployment/backend
```

---

## 8. 🎯 PERFORMANS OPTİMİZASYONU

### Database Optimization

```sql
-- Materialized Views for Analytics
CREATE MATERIALIZED VIEW student_performance_summary AS
SELECT 
    sp.id as student_id,
    sp.grade,
    AVG(qa.score) as avg_quiz_score,
    COUNT(DISTINCT qa.quiz_id) as quizzes_taken,
    SUM(ss.duration) as total_study_time,
    COUNT(DISTINCT DATE(ss.started_at)) as study_days
FROM student_profiles sp
LEFT JOIN quiz_attempts qa ON sp.id = qa.student_id
LEFT JOIN study_sessions ss ON sp.id = ss.student_id
GROUP BY sp.id, sp.grade
WITH DATA;

-- Refresh every hour
CREATE INDEX idx_perf_summary_student ON student_performance_summary(student_id);
```

### Caching Strategy

```python
# Cache decorator
def cache_result(ttl=3600):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            cache_key = f"{func.__name__}:{str(args)}:{str(kwargs)}"
            
            # Try to get from cache
            cached = await redis.get(cache_key)
            if cached:
                return json.loads(cached)
            
            # Execute function
            result = await func(*args, **kwargs)
            
            # Store in cache
            await redis.setex(
                cache_key, 
                ttl, 
                json.dumps(result)
            )
            
            return result
        return wrapper
    return decorator

# Usage
@cache_result(ttl=1800)
async def get_learning_path(student_id: str):
    # Expensive database query
    return await db.fetch_learning_path(student_id)
```

---

**Son Güncelleme**: 2025-08-20
**Versiyon**: 1.0.0
**Mimari Durum**: Design Phase