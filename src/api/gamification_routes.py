"""
Gamification API Routes
TEKNOFEST 2025 - Eğitim Teknolojileri

Production-ready gamification endpoints with rate limiting, caching, and monitoring.
"""

from typing import Optional, List, Dict
from datetime import datetime, timedelta
import logging

from fastapi import APIRouter, Depends, HTTPException, Query, Path, Body, status
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel, Field, validator
from redis import Redis

from src.database.session import get_db
from src.core.gamification_service import (
    GamificationService, GamificationConfig,
    PointSource, AchievementType
)
from src.api.auth import get_current_user, require_role
from src.database.models import User, UserRole
from src.core.cache_manager import CacheManager
from src.monitoring import metrics_collector
from src.exceptions import ValidationError, NotFoundError, RateLimitError
from src.core.rate_limiter import RateLimiter

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/gamification", tags=["gamification"])

# Initialize services
cache_manager = CacheManager()
rate_limiter = RateLimiter()


# Request/Response Models

class AwardPointsRequest(BaseModel):
    """Request model for awarding points"""
    user_id: str
    amount: int = Field(gt=0, le=1000)
    source: str
    description: str = Field(min_length=1, max_length=255)
    metadata: Optional[Dict] = None
    
    @validator('source')
    def validate_source(cls, v):
        """Validate point source"""
        valid_sources = [s.value for s in PointSource]
        if v not in valid_sources:
            raise ValueError(f"Invalid source. Must be one of: {valid_sources}")
        return v


class LeaderboardRequest(BaseModel):
    """Request model for leaderboard query"""
    period: str = Field(default="all", regex="^(all|daily|weekly|monthly)$")
    subject: Optional[str] = None
    limit: int = Field(default=100, ge=10, le=1000)
    offset: int = Field(default=0, ge=0)


class AchievementProgressResponse(BaseModel):
    """Response model for achievement progress"""
    achievement_id: str
    name: str
    description: str
    unlocked: bool
    progress: Dict
    points: int
    rarity: str


class GamificationStatsResponse(BaseModel):
    """Response model for gamification statistics"""
    user_id: str
    username: str
    points: int
    level: int
    next_level_points: int
    progress_to_next_level: float
    streak_days: int
    total_study_time: int
    achievements_unlocked: int
    total_achievements: int
    rank: int
    percentile: float
    recent_transactions: List[Dict]


# Dependency injection

async def get_gamification_service(
    db: AsyncSession = Depends(get_db),
    redis: Optional[Redis] = None
) -> GamificationService:
    """Get gamification service instance"""
    return GamificationService(
        db_session=db,
        cache_manager=cache_manager,
        redis_client=redis
    )


# Endpoints

@router.post("/points/award", response_model=Dict)
@rate_limiter.limit("10/minute")
async def award_points(
    request: AwardPointsRequest,
    current_user: User = Depends(get_current_user),
    service: GamificationService = Depends(get_gamification_service)
):
    """
    Award points to a user.
    
    Requires authentication. Teachers and admins can award points to any user.
    Students can only trigger automatic point awards for themselves.
    """
    try:
        # Authorization check
        if current_user.role == UserRole.STUDENT:
            if str(current_user.id) != request.user_id:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Students can only trigger their own point awards"
                )
            # Limit sources for students
            allowed_sources = [
                PointSource.LESSON_COMPLETE.value,
                PointSource.QUIZ_COMPLETE.value,
                PointSource.DAILY_LOGIN.value
            ]
            if request.source not in allowed_sources:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Invalid source for student. Allowed: {allowed_sources}"
                )
        
        # Award points
        result = await service.award_points(
            user_id=request.user_id,
            amount=request.amount,
            source=PointSource(request.source),
            description=request.description,
            metadata=request.metadata
        )
        
        # Log activity
        logger.info(
            f"Points awarded: {request.amount} to user {request.user_id} "
            f"by {current_user.username} ({request.source})"
        )
        
        # Track metrics
        metrics_collector.increment(
            "api.gamification.points_awarded",
            request.amount,
            tags={"source": request.source}
        )
        
        return result
        
    except NotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except ValidationError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to award points: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to award points"
        )


@router.post("/streak/update", response_model=Dict)
@rate_limiter.limit("5/minute")
async def update_streak(
    current_user: User = Depends(get_current_user),
    service: GamificationService = Depends(get_gamification_service)
):
    """
    Update the current user's streak.
    
    This endpoint should be called when a user completes any learning activity.
    """
    try:
        result = await service.update_streak(str(current_user.id))
        
        # Track metrics
        metrics_collector.increment("api.gamification.streak_updated")
        
        return result
        
    except Exception as e:
        logger.error(f"Failed to update streak: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update streak"
        )


@router.get("/leaderboard", response_model=List[Dict])
@rate_limiter.limit("30/minute")
async def get_leaderboard(
    period: str = Query(default="all", regex="^(all|daily|weekly|monthly)$"),
    subject: Optional[str] = Query(default=None),
    limit: int = Query(default=100, ge=10, le=1000),
    offset: int = Query(default=0, ge=0),
    service: GamificationService = Depends(get_gamification_service)
):
    """
    Get the leaderboard.
    
    Public endpoint with caching for performance.
    """
    try:
        entries = await service.get_leaderboard(
            period=period,
            subject=subject,
            limit=limit,
            offset=offset
        )
        
        # Track metrics
        metrics_collector.increment(
            "api.gamification.leaderboard_viewed",
            tags={"period": period, "subject": subject or "all"}
        )
        
        return [
            {
                "rank": entry.rank,
                "user_id": entry.user_id,
                "username": entry.username,
                "avatar_url": entry.avatar_url,
                "points": entry.points,
                "level": entry.level,
                "achievements_count": entry.achievements_count,
                "streak_days": entry.streak_days,
                "change": entry.change
            }
            for entry in entries
        ]
        
    except Exception as e:
        logger.error(f"Failed to get leaderboard: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve leaderboard"
        )


@router.get("/user/{user_id}/rank", response_model=Dict)
@rate_limiter.limit("20/minute")
async def get_user_rank(
    user_id: str = Path(..., description="User ID"),
    period: str = Query(default="all", regex="^(all|daily|weekly|monthly)$"),
    subject: Optional[str] = Query(default=None),
    service: GamificationService = Depends(get_gamification_service)
):
    """
    Get a user's rank in the leaderboard.
    """
    try:
        rank_info = await service.get_user_rank(
            user_id=user_id,
            period=period,
            subject=subject
        )
        
        return rank_info
        
    except NotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to get user rank: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve user rank"
        )


@router.get("/user/{user_id}/achievements", response_model=List[Dict])
@rate_limiter.limit("20/minute")
async def get_user_achievements(
    user_id: str = Path(..., description="User ID"),
    include_locked: bool = Query(default=True),
    current_user: User = Depends(get_current_user),
    service: GamificationService = Depends(get_gamification_service)
):
    """
    Get a user's achievements.
    
    Users can view their own achievements. Others can only see unlocked achievements.
    """
    try:
        # Privacy check
        is_own_profile = str(current_user.id) == user_id
        if not is_own_profile and include_locked:
            include_locked = False  # Others can't see locked achievements
        
        achievements = await service.get_user_achievements(
            user_id=user_id,
            include_locked=include_locked
        )
        
        return achievements
        
    except NotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to get user achievements: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve achievements"
        )


@router.get("/user/{user_id}/stats", response_model=GamificationStatsResponse)
@rate_limiter.limit("30/minute")
async def get_gamification_stats(
    user_id: str = Path(..., description="User ID"),
    current_user: User = Depends(get_current_user),
    service: GamificationService = Depends(get_gamification_service)
):
    """
    Get comprehensive gamification statistics for a user.
    
    Users can view detailed stats for themselves, others see limited info.
    """
    try:
        stats = await service.get_gamification_stats(user_id)
        
        # Privacy filter for other users
        if str(current_user.id) != user_id:
            # Remove sensitive information
            stats.pop("recent_transactions", None)
        
        return GamificationStatsResponse(**stats)
        
    except NotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to get gamification stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve statistics"
        )


@router.get("/achievements/available", response_model=List[Dict])
@rate_limiter.limit("10/minute")
async def get_available_achievements(
    achievement_type: Optional[str] = Query(default=None),
    rarity: Optional[str] = Query(default=None, regex="^(common|rare|epic|legendary)$"),
    service: GamificationService = Depends(get_gamification_service)
):
    """
    Get list of all available achievements.
    
    Public endpoint showing achievement definitions.
    """
    try:
        achievements = []
        
        for key, definition in service.achievement_definitions.items():
            # Filter by type if specified
            if achievement_type and definition["type"].value != achievement_type:
                continue
            
            # Filter by rarity if specified
            if rarity and definition["rarity"] != rarity:
                continue
            
            achievements.append({
                "id": key,
                "name": definition["name"],
                "description": definition["description"],
                "type": definition["type"].value,
                "points": definition["points"],
                "rarity": definition["rarity"],
                "criteria": definition["criteria"]
            })
        
        # Sort by rarity and points
        rarity_order = {"common": 0, "rare": 1, "epic": 2, "legendary": 3}
        achievements.sort(
            key=lambda x: (rarity_order.get(x["rarity"], 0), -x["points"])
        )
        
        return achievements
        
    except Exception as e:
        logger.error(f"Failed to get available achievements: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve achievements"
        )


@router.post("/achievements/{achievement_id}/check", response_model=Dict)
@rate_limiter.limit("10/minute")
async def check_achievement_progress(
    achievement_id: str = Path(..., description="Achievement ID"),
    current_user: User = Depends(get_current_user),
    service: GamificationService = Depends(get_gamification_service)
):
    """
    Check progress towards a specific achievement.
    """
    try:
        # Get achievement definition
        if achievement_id not in service.achievement_definitions:
            raise NotFoundError(f"Achievement {achievement_id} not found")
        
        definition = service.achievement_definitions[achievement_id]
        
        # Get progress
        progress = await service._get_achievement_progress(
            str(current_user.id),
            definition["criteria"]
        )
        
        return {
            "achievement_id": achievement_id,
            "name": definition["name"],
            "description": definition["description"],
            "progress": progress,
            "points": definition["points"],
            "rarity": definition["rarity"]
        }
        
    except NotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to check achievement progress: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to check progress"
        )


@router.get("/stats/global", response_model=Dict)
@rate_limiter.limit("10/minute")
async def get_global_stats(
    service: GamificationService = Depends(get_gamification_service),
    _: User = Depends(require_role([UserRole.ADMIN]))
):
    """
    Get global gamification statistics.
    
    Admin only endpoint.
    """
    try:
        from sqlalchemy import select, func
        from src.database.models import User, Achievement, user_achievements
        
        db = service.db
        
        # Get total users
        total_users = await db.scalar(
            select(func.count()).select_from(User).where(User.is_active == True)
        )
        
        # Get average stats
        avg_stats = await db.execute(
            select(
                func.avg(User.points).label("avg_points"),
                func.avg(User.level).label("avg_level"),
                func.avg(User.streak_days).label("avg_streak"),
                func.max(User.points).label("max_points"),
                func.max(User.level).label("max_level"),
                func.max(User.streak_days).label("max_streak")
            ).select_from(User).where(User.is_active == True)
        )
        stats = avg_stats.first()
        
        # Get achievement statistics
        total_achievements_unlocked = await db.scalar(
            select(func.count()).select_from(user_achievements)
        )
        
        # Get most common achievements
        most_common = await db.execute(
            select(
                Achievement.name,
                func.count(user_achievements.c.user_id).label("unlock_count")
            )
            .select_from(Achievement)
            .join(user_achievements)
            .group_by(Achievement.name)
            .order_by(func.count(user_achievements.c.user_id).desc())
            .limit(5)
        )
        
        # Get service performance stats
        performance = service.get_performance_stats()
        
        return {
            "total_users": total_users,
            "average_points": float(stats.avg_points or 0),
            "average_level": float(stats.avg_level or 0),
            "average_streak": float(stats.avg_streak or 0),
            "highest_points": int(stats.max_points or 0),
            "highest_level": int(stats.max_level or 0),
            "longest_streak": int(stats.max_streak or 0),
            "total_achievements_unlocked": total_achievements_unlocked,
            "most_common_achievements": [
                {"name": row.name, "count": row.unlock_count}
                for row in most_common
            ],
            "service_performance": performance
        }
        
    except Exception as e:
        logger.error(f"Failed to get global stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve global statistics"
        )


@router.post("/admin/recalculate", response_model=Dict)
@rate_limiter.limit("1/minute")
async def recalculate_gamification(
    user_id: Optional[str] = Body(default=None),
    service: GamificationService = Depends(get_gamification_service),
    _: User = Depends(require_role([UserRole.ADMIN]))
):
    """
    Recalculate gamification stats for a user or all users.
    
    Admin only endpoint for fixing inconsistencies.
    """
    try:
        from sqlalchemy import select
        from src.database.models import User
        
        db = service.db
        
        if user_id:
            # Recalculate for specific user
            result = await db.execute(
                select(User).where(User.id == user_id)
            )
            user = result.scalar_one_or_none()
            
            if not user:
                raise NotFoundError(f"User {user_id} not found")
            
            # Recalculate level based on points
            new_level = service._calculate_level(user.points)
            user.level = new_level
            
            # Check all achievements
            unlocked = await service._check_achievements(user)
            
            await db.commit()
            
            return {
                "success": True,
                "user_id": user_id,
                "points": user.points,
                "level": new_level,
                "achievements_checked": len(unlocked)
            }
        else:
            # Batch recalculation for all users
            result = await db.execute(
                select(User).where(User.is_active == True)
            )
            users = result.scalars().all()
            
            updated_count = 0
            achievements_unlocked = 0
            
            for user in users:
                # Recalculate level
                new_level = service._calculate_level(user.points)
                if new_level != user.level:
                    user.level = new_level
                    updated_count += 1
                
                # Check achievements
                unlocked = await service._check_achievements(user)
                achievements_unlocked += len(unlocked)
            
            await db.commit()
            
            return {
                "success": True,
                "users_processed": len(users),
                "levels_updated": updated_count,
                "achievements_unlocked": achievements_unlocked
            }
            
    except NotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to recalculate gamification: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to recalculate statistics"
        )


# Health check endpoint
@router.get("/health", response_model=Dict)
async def health_check():
    """Health check endpoint for gamification service"""
    return {
        "status": "healthy",
        "service": "gamification",
        "timestamp": datetime.now().isoformat()
    }