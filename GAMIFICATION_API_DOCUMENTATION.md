# Gamification API Documentation
TEKNOFEST 2025 - Eğitim Teknolojileri

## Overview

The gamification system provides a comprehensive set of features to enhance student engagement through points, achievements, leaderboards, and streaks. The system is production-ready with caching, rate limiting, and performance optimization.

## Features

- **Points System**: Award and track points for various activities
- **Achievements**: Unlock badges and rewards based on milestones
- **Leaderboards**: Global, weekly, and subject-based rankings
- **Streaks**: Track consecutive days of learning
- **Levels**: Progress through levels based on points
- **Challenges**: Time-limited special challenges
- **Real-time Updates**: WebSocket support for live notifications

## Authentication

All endpoints require JWT authentication unless specified otherwise. Include the bearer token in the Authorization header:

```
Authorization: Bearer <your-jwt-token>
```

## Base URL

```
https://api.teknofest2025.edu.tr/api/v1/gamification
```

## Endpoints

### 1. Points Management

#### Award Points
`POST /points/award`

Award points to a user for completing activities.

**Rate Limit**: 10 requests/minute

**Request Body**:
```json
{
  "user_id": "uuid",
  "amount": 50,
  "source": "lesson_complete",
  "description": "Completed Python basics lesson",
  "metadata": {
    "lesson_id": "python_101",
    "duration": 45
  }
}
```

**Valid Sources**:
- `lesson_complete`: Lesson completion
- `quiz_complete`: Quiz completion
- `perfect_score`: Perfect quiz score
- `daily_login`: Daily login bonus
- `streak_bonus`: Streak milestone bonus
- `achievement`: Achievement unlock
- `challenge`: Challenge completion
- `helper`: Helping other students

**Response**:
```json
{
  "success": true,
  "transaction": {
    "user_id": "uuid",
    "amount": 75,
    "source": "lesson_complete",
    "description": "Completed Python basics lesson",
    "timestamp": "2025-01-22T10:30:00Z"
  },
  "old_points": 150,
  "new_points": 225,
  "old_level": 2,
  "new_level": 3,
  "level_up": true,
  "achievements_unlocked": [
    {
      "id": "ten_lessons",
      "name": "Öğrenme Yolunda",
      "description": "10 ders tamamladın!",
      "points": 50,
      "rarity": "common"
    }
  ]
}
```

### 2. Streak Management

#### Update Streak
`POST /streak/update`

Update the current user's learning streak.

**Rate Limit**: 5 requests/minute

**Response**:
```json
{
  "streak_days": 7,
  "updated": true,
  "streak_bonus": 35,
  "achievements_unlocked": [
    {
      "id": "week_streak",
      "name": "Haftalık Devamlılık",
      "description": "7 gün üst üste çalıştın!",
      "points": 30
    }
  ]
}
```

### 3. Leaderboards

#### Get Leaderboard
`GET /leaderboard`

Retrieve the leaderboard with various filters.

**Rate Limit**: 30 requests/minute

**Query Parameters**:
- `period`: `all`, `daily`, `weekly`, `monthly` (default: `all`)
- `subject`: Subject filter (optional)
- `limit`: Number of entries (10-1000, default: 100)
- `offset`: Pagination offset (default: 0)

**Response**:
```json
[
  {
    "rank": 1,
    "user_id": "uuid",
    "username": "top_student",
    "avatar_url": "https://example.com/avatar.jpg",
    "points": 5420,
    "level": 15,
    "achievements_count": 23,
    "streak_days": 45,
    "change": 2
  }
]
```

#### Get User Rank
`GET /user/{user_id}/rank`

Get a specific user's rank in the leaderboard.

**Rate Limit**: 20 requests/minute

**Response**:
```json
{
  "user_id": "uuid",
  "rank": 42,
  "total_users": 1500,
  "percentile": 97.2,
  "points": 3200,
  "level": 12,
  "period": "all",
  "subject": null
}
```

### 4. Achievements

#### Get User Achievements
`GET /user/{user_id}/achievements`

Get a user's achievements (unlocked and locked).

**Rate Limit**: 20 requests/minute

**Query Parameters**:
- `include_locked`: Include locked achievements (default: true)

**Response**:
```json
[
  {
    "id": "first_lesson",
    "name": "İlk Adım",
    "description": "İlk dersini tamamladın!",
    "icon_url": "https://example.com/icons/first_lesson.png",
    "points": 10,
    "rarity": "common",
    "unlocked": true,
    "unlocked_at": "2025-01-15T14:30:00Z"
  },
  {
    "id": "hundred_lessons",
    "name": "Bilgi Avcısı",
    "description": "100 ders tamamladın!",
    "icon_url": null,
    "points": 200,
    "rarity": "rare",
    "unlocked": false,
    "progress": {
      "lessons_completed": {
        "current": 42,
        "target": 100,
        "percentage": 42.0
      }
    }
  }
]
```

#### Get Available Achievements
`GET /achievements/available`

Get list of all available achievements in the system.

**Rate Limit**: 10 requests/minute

**Query Parameters**:
- `achievement_type`: Filter by type (optional)
- `rarity`: `common`, `rare`, `epic`, `legendary` (optional)

**Response**:
```json
[
  {
    "id": "level_50",
    "name": "Efsanevi Öğrenci",
    "description": "Seviye 50'ye ulaştın!",
    "type": "milestone",
    "points": 500,
    "rarity": "legendary",
    "criteria": {
      "level": 50
    }
  }
]
```

#### Check Achievement Progress
`POST /achievements/{achievement_id}/check`

Check progress towards a specific achievement.

**Rate Limit**: 10 requests/minute

**Response**:
```json
{
  "achievement_id": "hundred_lessons",
  "name": "Bilgi Avcısı",
  "description": "100 ders tamamladın!",
  "progress": {
    "lessons_completed": {
      "current": 42,
      "target": 100,
      "percentage": 42.0
    }
  },
  "points": 200,
  "rarity": "rare"
}
```

### 5. Statistics

#### Get User Gamification Stats
`GET /user/{user_id}/stats`

Get comprehensive gamification statistics for a user.

**Rate Limit**: 30 requests/minute

**Response**:
```json
{
  "user_id": "uuid",
  "username": "student123",
  "points": 3450,
  "level": 12,
  "next_level_points": 3750,
  "progress_to_next_level": 60.0,
  "streak_days": 15,
  "total_study_time": 4320,
  "achievements_unlocked": 18,
  "total_achievements": 50,
  "rank": 42,
  "percentile": 97.2,
  "recent_transactions": [
    {
      "amount": 50,
      "source": "quiz_complete",
      "description": "Completed math quiz",
      "timestamp": "2025-01-22T10:00:00Z"
    }
  ]
}
```

#### Get Global Statistics (Admin Only)
`GET /stats/global`

Get global gamification statistics.

**Rate Limit**: 10 requests/minute

**Authorization**: Admin role required

**Response**:
```json
{
  "total_users": 1500,
  "average_points": 1245.5,
  "average_level": 6.8,
  "average_streak": 4.2,
  "highest_points": 12450,
  "highest_level": 35,
  "longest_streak": 180,
  "total_achievements_unlocked": 8542,
  "most_common_achievements": [
    {
      "name": "İlk Adım",
      "count": 1450
    }
  ],
  "service_performance": {
    "points_awarded": 45320,
    "achievements_unlocked": 234,
    "leaderboard_updates": 1234,
    "cache_hits": 8901
  }
}
```

### 6. Admin Operations

#### Recalculate Gamification (Admin Only)
`POST /admin/recalculate`

Recalculate gamification statistics to fix inconsistencies.

**Rate Limit**: 1 request/minute

**Authorization**: Admin role required

**Request Body** (optional):
```json
{
  "user_id": "uuid"  // Specific user or null for all users
}
```

**Response**:
```json
{
  "success": true,
  "users_processed": 1,
  "levels_updated": 1,
  "achievements_unlocked": 3
}
```

### 7. Health Check

#### Health Check
`GET /health`

Check gamification service health.

**Response**:
```json
{
  "status": "healthy",
  "service": "gamification",
  "timestamp": "2025-01-22T12:00:00Z"
}
```

## Error Responses

### Rate Limit Exceeded
```json
{
  "detail": "Rate limit exceeded",
  "status_code": 429,
  "headers": {
    "Retry-After": "60"
  }
}
```

### Authentication Failed
```json
{
  "detail": "Could not validate credentials",
  "status_code": 401
}
```

### Insufficient Permissions
```json
{
  "detail": "Insufficient permissions. Required role: ['admin']",
  "status_code": 403
}
```

### Resource Not Found
```json
{
  "detail": "User not found",
  "status_code": 404
}
```

### Validation Error
```json
{
  "detail": [
    {
      "loc": ["body", "amount"],
      "msg": "ensure this value is greater than 0",
      "type": "value_error.number.not_gt"
    }
  ],
  "status_code": 422
}
```

## Rate Limiting

The API uses a sliding window rate limiting algorithm with the following default limits:

| Endpoint Pattern | Limit | Window |
|-----------------|-------|---------|
| `/points/award` | 10 | 1 minute |
| `/streak/update` | 5 | 1 minute |
| `/leaderboard` | 30 | 1 minute |
| `/user/*/achievements` | 20 | 1 minute |
| `/achievements/available` | 10 | 1 minute |
| `/admin/*` | 1 | 1 minute |

Rate limits are applied per user for authenticated endpoints and per IP for public endpoints.

## Caching

The following data is cached for performance:

| Data | TTL | Cache Key Pattern |
|------|-----|-------------------|
| Leaderboard | 5 minutes | `leaderboard:{period}:{subject}:{limit}:{offset}` |
| User achievements | 10 minutes | `achievements:{user_id}` |
| User stats | 5 minutes | `stats:{user_id}` |
| Achievement definitions | 1 hour | `achievements:definitions` |

## WebSocket Events

For real-time updates, connect to the WebSocket endpoint:

```
wss://api.teknofest2025.edu.tr/ws/gamification
```

### Event Types

#### Points Awarded
```json
{
  "type": "points_awarded",
  "user_id": "uuid",
  "amount": 50,
  "new_total": 3500,
  "source": "quiz_complete"
}
```

#### Achievement Unlocked
```json
{
  "type": "achievement_unlocked",
  "user_id": "uuid",
  "achievement": {
    "id": "week_streak",
    "name": "Haftalık Devamlılık",
    "points": 30
  }
}
```

#### Level Up
```json
{
  "type": "level_up",
  "user_id": "uuid",
  "old_level": 5,
  "new_level": 6
}
```

#### Leaderboard Update
```json
{
  "type": "leaderboard_update",
  "updates": [
    {
      "user_id": "uuid",
      "old_rank": 10,
      "new_rank": 9
    }
  ]
}
```

## Best Practices

1. **Cache User Stats**: Cache frequently accessed user statistics client-side
2. **Batch Updates**: Use batch endpoints when updating multiple users
3. **Rate Limit Handling**: Implement exponential backoff when rate limited
4. **WebSocket Fallback**: Have polling fallback if WebSocket connection fails
5. **Progress Tracking**: Update achievement progress asynchronously
6. **Error Handling**: Implement proper error handling for all API calls

## Performance Considerations

- Leaderboard queries are optimized with materialized views
- Achievement checks are cached for 10 minutes
- Point transactions are processed asynchronously
- Redis is used for distributed caching and rate limiting
- Database queries use proper indexing for optimal performance

## Security

- All endpoints require authentication except health check
- Role-based access control for admin operations
- Rate limiting prevents abuse
- Input validation on all endpoints
- SQL injection protection through parameterized queries
- XSS protection through proper output encoding

## Migration Guide

For migrating from the old system:

1. Run the database migration: `alembic upgrade 006`
2. Recalculate all user levels: `POST /admin/recalculate`
3. Update client code to use new endpoints
4. Configure caching and rate limiting
5. Set up monitoring for the new metrics

## Support

For issues or questions, contact the development team or create an issue in the project repository.