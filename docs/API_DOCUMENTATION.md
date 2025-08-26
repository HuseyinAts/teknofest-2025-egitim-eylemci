# 📚 TEKNOFEST 2025 API Documentation

## 🌟 Overview

TEKNOFEST 2025 Educational Technology Platform provides a comprehensive RESTful API for managing educational content, student assessments, and AI-powered learning experiences.

**Base URL**: `https://api.teknofest2025.com`  
**API Version**: `v1`  
**Documentation**: `https://api.teknofest2025.com/docs` (Swagger UI)  
**ReDoc**: `https://api.teknofest2025.com/redoc`

## 🔐 Authentication

The API uses JWT (JSON Web Token) authentication. Include the token in the Authorization header:

```http
Authorization: Bearer <your-jwt-token>
```

### Get Access Token

```http
POST /api/auth/login
Content-Type: application/json

{
  "username": "student@example.com",
  "password": "SecurePassword123!"
}
```

**Response:**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIs...",
  "refresh_token": "eyJhbGciOiJIUzI1NiIs...",
  "token_type": "bearer",
  "expires_in": 1800
}
```

### Refresh Token

```http
POST /api/auth/refresh
Content-Type: application/json

{
  "refresh_token": "eyJhbGciOiJIUzI1NiIs..."
}
```

## 📋 Endpoints

### 🏫 User Management

#### Register User
```http
POST /api/auth/register
```

**Request Body:**
```json
{
  "username": "student123",
  "email": "student@example.com",
  "password": "SecurePassword123!",
  "full_name": "Ahmet Yılmaz",
  "grade": 10,
  "school": "Atatürk Lisesi"
}
```

**Response:** `201 Created`
```json
{
  "id": "uuid-here",
  "username": "student123",
  "email": "student@example.com",
  "created_at": "2025-01-23T10:00:00Z"
}
```

#### Get User Profile
```http
GET /api/users/profile
Authorization: Bearer <token>
```

**Response:**
```json
{
  "id": "uuid-here",
  "username": "student123",
  "email": "student@example.com",
  "full_name": "Ahmet Yılmaz",
  "grade": 10,
  "points": 1250,
  "level": 5,
  "achievements": [...]
}
```

#### Update Profile
```http
PATCH /api/users/profile
Authorization: Bearer <token>
```

**Request Body:**
```json
{
  "full_name": "Ahmet Yılmaz",
  "grade": 11,
  "preferences": {
    "language": "tr",
    "theme": "dark"
  }
}
```

### 📚 Learning Paths

#### Get Learning Paths
```http
GET /api/learning-paths?subject=matematik&grade=10
Authorization: Bearer <token>
```

**Response:**
```json
{
  "data": [
    {
      "id": "path-001",
      "title": "Matematik Temelleri",
      "subject": "Matematik",
      "grade": 10,
      "difficulty": "medium",
      "duration_hours": 20,
      "modules": [...],
      "progress": 45
    }
  ],
  "total": 15,
  "page": 1,
  "per_page": 10
}
```

#### Create Custom Learning Path
```http
POST /api/learning-paths
Authorization: Bearer <token>
```

**Request Body:**
```json
{
  "title": "Özel Matematik Programım",
  "subject": "Matematik",
  "topics": ["Türev", "İntegral", "Limit"],
  "target_exam": "YKS",
  "duration_weeks": 8
}
```

### 📝 Assessments & Quizzes

#### Start Quiz
```http
POST /api/quiz/start
Authorization: Bearer <token>
```

**Request Body:**
```json
{
  "subject": "Fizik",
  "topic": "Mekanik",
  "difficulty": "hard",
  "question_count": 20,
  "time_limit_minutes": 30
}
```

**Response:**
```json
{
  "quiz_id": "quiz-123",
  "questions": [
    {
      "id": "q1",
      "text": "Bir cisim 10 m/s hızla yukarı atılıyor...",
      "options": ["A) 5m", "B) 10m", "C) 15m", "D) 20m"],
      "image_url": null,
      "points": 5
    }
  ],
  "started_at": "2025-01-23T10:00:00Z",
  "expires_at": "2025-01-23T10:30:00Z"
}
```

#### Submit Quiz Answers
```http
POST /api/quiz/{quiz_id}/submit
Authorization: Bearer <token>
```

**Request Body:**
```json
{
  "answers": [
    {"question_id": "q1", "answer": "B"},
    {"question_id": "q2", "answer": "D"}
  ],
  "time_spent_seconds": 1250
}
```

**Response:**
```json
{
  "score": 85,
  "correct_answers": 17,
  "total_questions": 20,
  "points_earned": 425,
  "performance_analysis": {
    "strengths": ["Kinematik", "Dinamik"],
    "weaknesses": ["Enerji Korunumu"],
    "recommendations": [...]
  }
}
```

### 🤖 AI Assistant

#### Chat with AI
```http
POST /api/ai/chat
Authorization: Bearer <token>
```

**Request Body:**
```json
{
  "message": "Türev konusunu anlamadım, bana yardım eder misin?",
  "context": "matematik",
  "session_id": "session-123"
}
```

**Response:**
```json
{
  "response": "Tabii ki! Türev, bir fonksiyonun değişim hızını gösteren matematiksel bir kavramdır...",
  "suggestions": [
    "Türev kuralları nelerdir?",
    "Basit bir türev örneği göster"
  ],
  "resources": [...]
}
```

#### Generate Practice Questions
```http
POST /api/ai/generate-questions
Authorization: Bearer <token>
```

**Request Body:**
```json
{
  "subject": "Kimya",
  "topic": "Periyodik Tablo",
  "difficulty": "medium",
  "count": 5,
  "question_type": "multiple_choice"
}
```

### 📊 Progress & Analytics

#### Get Progress Report
```http
GET /api/progress/report
Authorization: Bearer <token>
```

**Response:**
```json
{
  "overall_progress": 67,
  "subjects": {
    "matematik": {
      "progress": 75,
      "completed_topics": 15,
      "total_topics": 20,
      "average_score": 82
    }
  },
  "weekly_activity": [...],
  "strengths": ["Problem Çözme", "Analitik Düşünme"],
  "improvement_areas": ["Zaman Yönetimi"]
}
```

#### Get Leaderboard
```http
GET /api/leaderboard?type=weekly&subject=all
Authorization: Bearer <token>
```

**Response:**
```json
{
  "user_rank": 42,
  "total_users": 1500,
  "top_users": [
    {
      "rank": 1,
      "username": "champion123",
      "points": 5420,
      "level": 15
    }
  ]
}
```

### 📚 Educational Resources

#### Search Resources
```http
GET /api/resources/search?q=integral&type=video&grade=11
Authorization: Bearer <token>
```

**Response:**
```json
{
  "results": [
    {
      "id": "res-001",
      "title": "İntegral Teknikleri",
      "type": "video",
      "duration_minutes": 15,
      "instructor": "Prof. Dr. Ali Veli",
      "rating": 4.8,
      "view_count": 1250
    }
  ],
  "total": 42,
  "facets": {
    "types": {"video": 20, "document": 15, "interactive": 7},
    "difficulties": {"easy": 10, "medium": 22, "hard": 10}
  }
}
```

## 🔄 Webhooks

### Quiz Completion Webhook
```http
POST https://your-server.com/webhook/quiz-complete
Content-Type: application/json
X-Webhook-Signature: sha256=...

{
  "event": "quiz.completed",
  "timestamp": "2025-01-23T10:30:00Z",
  "data": {
    "user_id": "user-123",
    "quiz_id": "quiz-456",
    "score": 85,
    "points_earned": 425
  }
}
```

## 🚨 Error Handling

The API uses standard HTTP status codes and returns errors in the following format:

```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid input data",
    "details": [
      {
        "field": "email",
        "message": "Invalid email format"
      }
    ],
    "request_id": "req-123456"
  }
}
```

### Common Error Codes

| Status Code | Error Code | Description |
|------------|------------|-------------|
| 400 | `BAD_REQUEST` | Invalid request format |
| 401 | `UNAUTHORIZED` | Missing or invalid authentication |
| 403 | `FORBIDDEN` | Insufficient permissions |
| 404 | `NOT_FOUND` | Resource not found |
| 409 | `CONFLICT` | Resource already exists |
| 422 | `VALIDATION_ERROR` | Input validation failed |
| 429 | `RATE_LIMITED` | Too many requests |
| 500 | `INTERNAL_ERROR` | Server error |
| 503 | `SERVICE_UNAVAILABLE` | Service temporarily unavailable |

## 🔒 Rate Limiting

API endpoints are rate-limited to ensure fair usage:

- **Authentication endpoints**: 5 requests per minute
- **AI endpoints**: 30 requests per minute
- **General endpoints**: 100 requests per minute

Rate limit information is included in response headers:

```http
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1706007600
```

## 📊 Pagination

List endpoints support pagination using the following parameters:

- `page`: Page number (default: 1)
- `per_page`: Items per page (default: 20, max: 100)
- `sort`: Sort field (e.g., `created_at`, `score`)
- `order`: Sort order (`asc` or `desc`)

Example:
```http
GET /api/resources?page=2&per_page=50&sort=rating&order=desc
```

## 🔄 Versioning

The API version is included in the URL path. When breaking changes are introduced, a new version will be released:

- Current: `/api/v1/...`
- Future: `/api/v2/...`

Deprecated versions will be supported for at least 6 months with advance notice.

## 🧪 Testing

### Test Environment
- Base URL: `https://test-api.teknofest2025.com`
- Use test credentials provided in your developer account

### Postman Collection
Download our Postman collection for easy testing:
[Download Collection](https://api.teknofest2025.com/docs/postman-collection.json)

### cURL Examples

**Login:**
```bash
curl -X POST https://api.teknofest2025.com/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username":"test","password":"test123"}'
```

**Get Profile:**
```bash
curl -X GET https://api.teknofest2025.com/api/users/profile \
  -H "Authorization: Bearer YOUR_TOKEN"
```

## 📝 SDKs

Official SDKs are available for:

- **Python**: `pip install teknofest-sdk`
- **JavaScript/TypeScript**: `npm install @teknofest/sdk`
- **Java**: Maven dependency available
- **Go**: `go get github.com/teknofest/sdk-go`

## 🆘 Support

- **Documentation**: https://docs.teknofest2025.com
- **Status Page**: https://status.teknofest2025.com
- **Support Email**: api-support@teknofest2025.com
- **Developer Discord**: https://discord.gg/teknofest

## 📄 Terms of Service

By using this API, you agree to our [Terms of Service](https://teknofest2025.com/terms) and [Privacy Policy](https://teknofest2025.com/privacy).

---
*Last Updated: January 23, 2025*  
*API Version: 1.0.0*