"""
Gamification Service Layer for Production
TEKNOFEST 2025 - Eğitim Teknolojileri

Provides comprehensive gamification features including points, achievements, 
leaderboards, streaks, and rewards.
"""

import asyncio
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta, date
import json
import hashlib
from dataclasses import dataclass, asdict
import logging
from enum import Enum
import math

import numpy as np
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, and_, or_, func, desc
from sqlalchemy.orm import selectinload
from redis import Redis
from pydantic import BaseModel, Field, validator

from src.database.models import (
    User, Achievement, user_achievements,
    StudySession, Assessment, Progress,
    Notification
)
from src.core.cache_manager import CacheManager
from src.monitoring import metrics_collector
from src.exceptions import ValidationError, NotFoundError

logger = logging.getLogger(__name__)


class AchievementType(Enum):
    """Types of achievements"""
    MILESTONE = "milestone"  # Reaching specific goals
    STREAK = "streak"  # Consecutive days/activities
    MASTERY = "mastery"  # Subject/skill mastery
    CHALLENGE = "challenge"  # Special challenges
    SOCIAL = "social"  # Social interactions
    SPECIAL = "special"  # Special events


class PointSource(Enum):
    """Sources of points"""
    LESSON_COMPLETE = "lesson_complete"
    QUIZ_COMPLETE = "quiz_complete"
    PERFECT_SCORE = "perfect_score"
    DAILY_LOGIN = "daily_login"
    STREAK_BONUS = "streak_bonus"
    ACHIEVEMENT = "achievement"
    CHALLENGE = "challenge"
    HELPER = "helper"  # Helping other students


@dataclass
class PointTransaction:
    """Point transaction record"""
    user_id: str
    amount: int
    source: PointSource
    description: str
    metadata: Optional[Dict] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class LeaderboardEntry:
    """Leaderboard entry"""
    rank: int
    user_id: str
    username: str
    avatar_url: Optional[str]
    points: int
    level: int
    achievements_count: int
    streak_days: int
    change: int = 0  # Position change from previous period


class GamificationConfig(BaseModel):
    """Configuration for gamification system"""
    points_multiplier: float = Field(default=1.0, ge=0.5, le=5.0)
    streak_threshold: int = Field(default=3, ge=1, le=30)
    level_base_points: int = Field(default=100, ge=50, le=1000)
    level_multiplier: float = Field(default=1.5, ge=1.1, le=3.0)
    max_daily_points: Optional[int] = Field(default=1000, ge=100)
    achievement_unlock_notification: bool = True
    leaderboard_size: int = Field(default=100, ge=10, le=1000)
    

class GamificationService:
    """
    Production-ready gamification service with comprehensive features.
    
    Features:
    - Points and XP system
    - Achievement tracking and unlocking
    - Leaderboards (global, weekly, subject-based)
    - Streak tracking
    - Level progression
    - Rewards and badges
    - Performance optimization with caching
    - Event-driven achievement checking
    """
    
    def __init__(
        self,
        db_session: AsyncSession,
        cache_manager: Optional[CacheManager] = None,
        redis_client: Optional[Redis] = None,
        config: Optional[GamificationConfig] = None
    ):
        self.db = db_session
        self.cache = cache_manager or CacheManager()
        self.redis = redis_client
        self.config = config or GamificationConfig()
        
        # Achievement definitions (would normally be in database)
        self.achievement_definitions = self._load_achievement_definitions()
        
        # Performance tracking
        self.performance_stats = {
            "points_awarded": 0,
            "achievements_unlocked": 0,
            "leaderboard_updates": 0,
            "cache_hits": 0
        }
        
        logger.info("Gamification Service initialized")
    
    def _load_achievement_definitions(self) -> Dict:
        """Load achievement definitions"""
        return {
            # Milestone achievements
            "first_lesson": {
                "type": AchievementType.MILESTONE,
                "name": "İlk Adım",
                "description": "İlk dersini tamamladın!",
                "points": 10,
                "criteria": {"lessons_completed": 1},
                "rarity": "common"
            },
            "ten_lessons": {
                "type": AchievementType.MILESTONE,
                "name": "Öğrenme Yolunda",
                "description": "10 ders tamamladın!",
                "points": 50,
                "criteria": {"lessons_completed": 10},
                "rarity": "common"
            },
            "hundred_lessons": {
                "type": AchievementType.MILESTONE,
                "name": "Bilgi Avcısı",
                "description": "100 ders tamamladın!",
                "points": 200,
                "criteria": {"lessons_completed": 100},
                "rarity": "rare"
            },
            
            # Streak achievements
            "week_streak": {
                "type": AchievementType.STREAK,
                "name": "Haftalık Devamlılık",
                "description": "7 gün üst üste çalıştın!",
                "points": 30,
                "criteria": {"streak_days": 7},
                "rarity": "common"
            },
            "month_streak": {
                "type": AchievementType.STREAK,
                "name": "Aylık Devamlılık",
                "description": "30 gün üst üste çalıştın!",
                "points": 150,
                "criteria": {"streak_days": 30},
                "rarity": "rare"
            },
            "hundred_day_streak": {
                "type": AchievementType.STREAK,
                "name": "Devamlılık Ustası",
                "description": "100 gün üst üste çalıştın!",
                "points": 500,
                "criteria": {"streak_days": 100},
                "rarity": "epic"
            },
            
            # Mastery achievements
            "perfect_quiz": {
                "type": AchievementType.MASTERY,
                "name": "Mükemmeliyetçi",
                "description": "Bir sınavdan tam puan aldın!",
                "points": 25,
                "criteria": {"perfect_scores": 1},
                "rarity": "common"
            },
            "ten_perfect": {
                "type": AchievementType.MASTERY,
                "name": "Kusursuz Performans",
                "description": "10 sınavdan tam puan aldın!",
                "points": 100,
                "criteria": {"perfect_scores": 10},
                "rarity": "rare"
            },
            "subject_master": {
                "type": AchievementType.MASTERY,
                "name": "Konu Uzmanı",
                "description": "Bir konuda uzmanlaştın!",
                "points": 200,
                "criteria": {"subject_mastery": 90},
                "rarity": "epic"
            },
            
            # Level achievements
            "level_5": {
                "type": AchievementType.MILESTONE,
                "name": "Yükselen Yıldız",
                "description": "Seviye 5'e ulaştın!",
                "points": 50,
                "criteria": {"level": 5},
                "rarity": "common"
            },
            "level_10": {
                "type": AchievementType.MILESTONE,
                "name": "Deneyimli Öğrenci",
                "description": "Seviye 10'a ulaştın!",
                "points": 100,
                "criteria": {"level": 10},
                "rarity": "rare"
            },
            "level_25": {
                "type": AchievementType.MILESTONE,
                "name": "Uzman Öğrenci",
                "description": "Seviye 25'e ulaştın!",
                "points": 250,
                "criteria": {"level": 25},
                "rarity": "epic"
            },
            "level_50": {
                "type": AchievementType.MILESTONE,
                "name": "Efsanevi Öğrenci",
                "description": "Seviye 50'ye ulaştın!",
                "points": 500,
                "criteria": {"level": 50},
                "rarity": "legendary"
            }
        }
    
    async def award_points(
        self,
        user_id: str,
        amount: int,
        source: PointSource,
        description: str,
        metadata: Optional[Dict] = None
    ) -> Dict:
        """
        Award points to a user.
        
        Args:
            user_id: User ID
            amount: Points to award
            source: Source of points
            description: Description of the transaction
            metadata: Additional metadata
        
        Returns:
            Transaction details with new totals
        """
        try:
            # Apply multipliers
            final_amount = int(amount * self.config.points_multiplier)
            
            # Check daily limit
            if self.config.max_daily_points:
                daily_points = await self._get_daily_points(user_id)
                if daily_points + final_amount > self.config.max_daily_points:
                    final_amount = max(0, self.config.max_daily_points - daily_points)
                    if final_amount == 0:
                        logger.info(f"User {user_id} reached daily point limit")
                        return {
                            "success": False,
                            "message": "Daily point limit reached",
                            "daily_limit": self.config.max_daily_points,
                            "daily_earned": daily_points
                        }
            
            # Update user points
            result = await self.db.execute(
                select(User).where(User.id == user_id).with_for_update()
            )
            user = result.scalar_one_or_none()
            
            if not user:
                raise NotFoundError(f"User {user_id} not found")
            
            old_points = user.points
            old_level = user.level
            
            user.points += final_amount
            
            # Calculate new level
            new_level = self._calculate_level(user.points)
            level_up = new_level > old_level
            user.level = new_level
            
            await self.db.commit()
            
            # Record transaction
            transaction = PointTransaction(
                user_id=user_id,
                amount=final_amount,
                source=source,
                description=description,
                metadata=metadata
            )
            
            # Store transaction in cache/database
            if self.redis:
                transaction_key = f"points:transactions:{user_id}"
                self.redis.lpush(
                    transaction_key,
                    json.dumps(asdict(transaction), default=str)
                )
                self.redis.ltrim(transaction_key, 0, 99)  # Keep last 100 transactions
                self.redis.expire(transaction_key, timedelta(days=30))
                
                # Update daily points cache
                daily_key = f"points:daily:{user_id}:{date.today()}"
                self.redis.incrby(daily_key, final_amount)
                self.redis.expire(daily_key, timedelta(days=1))
            
            # Check for new achievements
            achievements_unlocked = await self._check_achievements(user)
            
            # Update leaderboard cache
            await self._update_leaderboard_cache(user_id, user.points)
            
            # Send notification if level up
            if level_up:
                await self._send_notification(
                    user_id,
                    "level_up",
                    f"Tebrikler! Seviye {new_level} oldun!",
                    f"{old_points} puandan {user.points} puana yükseldin."
                )
            
            # Update metrics
            self.performance_stats["points_awarded"] += final_amount
            metrics_collector.increment(
                "gamification.points_awarded",
                final_amount,
                tags={"source": source.value}
            )
            
            return {
                "success": True,
                "transaction": asdict(transaction),
                "old_points": old_points,
                "new_points": user.points,
                "old_level": old_level,
                "new_level": new_level,
                "level_up": level_up,
                "achievements_unlocked": achievements_unlocked
            }
            
        except Exception as e:
            logger.error(f"Failed to award points: {e}")
            await self.db.rollback()
            raise
    
    async def update_streak(self, user_id: str) -> Dict:
        """
        Update user's streak.
        
        Args:
            user_id: User ID
        
        Returns:
            Streak information
        """
        try:
            result = await self.db.execute(
                select(User).where(User.id == user_id).with_for_update()
            )
            user = result.scalar_one_or_none()
            
            if not user:
                raise NotFoundError(f"User {user_id} not found")
            
            # Check last activity
            last_activity_key = f"streak:last_activity:{user_id}"
            today = date.today()
            
            if self.redis:
                last_activity = self.redis.get(last_activity_key)
                if last_activity:
                    last_date = date.fromisoformat(last_activity.decode())
                    days_diff = (today - last_date).days
                    
                    if days_diff == 0:
                        # Already updated today
                        return {
                            "streak_days": user.streak_days,
                            "updated": False,
                            "message": "Streak already updated today"
                        }
                    elif days_diff == 1:
                        # Consecutive day
                        user.streak_days += 1
                    else:
                        # Streak broken
                        user.streak_days = 1
                else:
                    # First activity
                    user.streak_days = 1
                
                # Update last activity
                self.redis.set(last_activity_key, today.isoformat())
                self.redis.expire(last_activity_key, timedelta(days=7))
            else:
                # Fallback without Redis
                user.streak_days = max(1, user.streak_days)
            
            await self.db.commit()
            
            # Award streak bonus points
            streak_bonus = 0
            if user.streak_days >= self.config.streak_threshold:
                streak_bonus = min(user.streak_days * 5, 100)
                await self.award_points(
                    user_id,
                    streak_bonus,
                    PointSource.STREAK_BONUS,
                    f"{user.streak_days} günlük seri bonusu"
                )
            
            # Check streak achievements
            achievements_unlocked = await self._check_streak_achievements(user)
            
            return {
                "streak_days": user.streak_days,
                "updated": True,
                "streak_bonus": streak_bonus,
                "achievements_unlocked": achievements_unlocked
            }
            
        except Exception as e:
            logger.error(f"Failed to update streak: {e}")
            await self.db.rollback()
            raise
    
    async def get_leaderboard(
        self,
        period: str = "all",  # all, weekly, monthly, daily
        subject: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[LeaderboardEntry]:
        """
        Get leaderboard.
        
        Args:
            period: Time period for leaderboard
            subject: Optional subject filter
            limit: Number of entries
            offset: Offset for pagination
        
        Returns:
            List of leaderboard entries
        """
        try:
            # Check cache first
            cache_key = f"leaderboard:{period}:{subject or 'all'}:{limit}:{offset}"
            if self.redis:
                cached = self.redis.get(cache_key)
                if cached:
                    self.performance_stats["cache_hits"] += 1
                    return [LeaderboardEntry(**entry) for entry in json.loads(cached)]
            
            # Build query
            query = select(User).where(User.is_active == True)
            
            # Apply period filter
            if period == "daily":
                start_date = datetime.now() - timedelta(days=1)
            elif period == "weekly":
                start_date = datetime.now() - timedelta(weeks=1)
            elif period == "monthly":
                start_date = datetime.now() - timedelta(days=30)
            else:
                start_date = None
            
            if start_date and subject:
                # Filter by recent activity in subject
                from src.database.models import StudySession
                subquery = (
                    select(StudySession.user_id)
                    .where(
                        and_(
                            StudySession.created_at >= start_date,
                            StudySession.subject == subject
                        )
                    )
                    .distinct()
                )
                query = query.where(User.id.in_(subquery))
            
            # Order by points and get top users
            query = query.order_by(desc(User.points)).limit(limit).offset(offset)
            
            result = await self.db.execute(query)
            users = result.scalars().all()
            
            # Get previous rankings for change calculation
            previous_rankings = await self._get_previous_rankings(period, subject)
            
            # Build leaderboard entries
            entries = []
            for rank, user in enumerate(users, start=offset + 1):
                # Get achievement count
                achievement_count = await self.db.scalar(
                    select(func.count())
                    .select_from(user_achievements)
                    .where(user_achievements.c.user_id == user.id)
                )
                
                # Calculate rank change
                previous_rank = previous_rankings.get(str(user.id), rank)
                change = previous_rank - rank
                
                entry = LeaderboardEntry(
                    rank=rank,
                    user_id=str(user.id),
                    username=user.username,
                    avatar_url=user.avatar_url,
                    points=user.points,
                    level=user.level,
                    achievements_count=achievement_count,
                    streak_days=user.streak_days,
                    change=change
                )
                entries.append(entry)
            
            # Cache the result
            if self.redis:
                self.redis.setex(
                    cache_key,
                    timedelta(minutes=5),
                    json.dumps([asdict(e) for e in entries])
                )
            
            # Update metrics
            self.performance_stats["leaderboard_updates"] += 1
            metrics_collector.increment("gamification.leaderboard_views")
            
            return entries
            
        except Exception as e:
            logger.error(f"Failed to get leaderboard: {e}")
            raise
    
    async def get_user_rank(
        self,
        user_id: str,
        period: str = "all",
        subject: Optional[str] = None
    ) -> Dict:
        """
        Get user's rank in leaderboard.
        
        Args:
            user_id: User ID
            period: Time period
            subject: Optional subject filter
        
        Returns:
            User's rank information
        """
        try:
            # Get user
            result = await self.db.execute(
                select(User).where(User.id == user_id)
            )
            user = result.scalar_one_or_none()
            
            if not user:
                raise NotFoundError(f"User {user_id} not found")
            
            # Count users with more points
            query = select(func.count()).select_from(User).where(
                and_(
                    User.is_active == True,
                    User.points > user.points
                )
            )
            
            higher_count = await self.db.scalar(query)
            rank = higher_count + 1
            
            # Get total users
            total_query = select(func.count()).select_from(User).where(
                User.is_active == True
            )
            total_users = await self.db.scalar(total_query)
            
            # Calculate percentile
            percentile = ((total_users - rank + 1) / total_users) * 100 if total_users > 0 else 0
            
            return {
                "user_id": user_id,
                "rank": rank,
                "total_users": total_users,
                "percentile": round(percentile, 2),
                "points": user.points,
                "level": user.level,
                "period": period,
                "subject": subject
            }
            
        except Exception as e:
            logger.error(f"Failed to get user rank: {e}")
            raise
    
    async def get_user_achievements(
        self,
        user_id: str,
        include_locked: bool = True
    ) -> List[Dict]:
        """
        Get user's achievements.
        
        Args:
            user_id: User ID
            include_locked: Include locked achievements
        
        Returns:
            List of achievements
        """
        try:
            # Get user's unlocked achievements
            result = await self.db.execute(
                select(Achievement)
                .join(user_achievements)
                .where(user_achievements.c.user_id == user_id)
            )
            unlocked = result.scalars().all()
            
            unlocked_ids = {str(a.id) for a in unlocked}
            
            achievements = []
            
            # Add unlocked achievements
            for achievement in unlocked:
                achievements.append({
                    "id": str(achievement.id),
                    "name": achievement.name,
                    "description": achievement.description,
                    "icon_url": achievement.icon_url,
                    "points": achievement.points,
                    "rarity": achievement.rarity,
                    "unlocked": True,
                    "unlocked_at": achievement.created_at.isoformat()
                })
            
            # Add locked achievements if requested
            if include_locked:
                for key, definition in self.achievement_definitions.items():
                    if key not in unlocked_ids:
                        achievements.append({
                            "id": key,
                            "name": definition["name"],
                            "description": definition["description"],
                            "icon_url": None,
                            "points": definition["points"],
                            "rarity": definition["rarity"],
                            "unlocked": False,
                            "progress": await self._get_achievement_progress(
                                user_id, 
                                definition["criteria"]
                            )
                        })
            
            # Sort by rarity and unlocked status
            rarity_order = {"common": 0, "rare": 1, "epic": 2, "legendary": 3}
            achievements.sort(
                key=lambda x: (not x["unlocked"], rarity_order.get(x["rarity"], 0))
            )
            
            return achievements
            
        except Exception as e:
            logger.error(f"Failed to get user achievements: {e}")
            raise
    
    async def get_gamification_stats(self, user_id: str) -> Dict:
        """
        Get comprehensive gamification stats for a user.
        
        Args:
            user_id: User ID
        
        Returns:
            Gamification statistics
        """
        try:
            # Get user
            result = await self.db.execute(
                select(User).where(User.id == user_id)
            )
            user = result.scalar_one_or_none()
            
            if not user:
                raise NotFoundError(f"User {user_id} not found")
            
            # Get rank
            rank_info = await self.get_user_rank(user_id)
            
            # Get achievement count
            achievement_count = await self.db.scalar(
                select(func.count())
                .select_from(user_achievements)
                .where(user_achievements.c.user_id == user_id)
            )
            
            # Calculate next level requirements
            next_level_points = self._calculate_points_for_level(user.level + 1)
            current_level_points = self._calculate_points_for_level(user.level)
            progress_to_next = (
                (user.points - current_level_points) / 
                (next_level_points - current_level_points) * 100
            ) if next_level_points > current_level_points else 100
            
            # Get recent point transactions
            recent_transactions = []
            if self.redis:
                transaction_key = f"points:transactions:{user_id}"
                raw_transactions = self.redis.lrange(transaction_key, 0, 9)
                recent_transactions = [
                    json.loads(t) for t in raw_transactions
                ]
            
            return {
                "user_id": user_id,
                "username": user.username,
                "points": user.points,
                "level": user.level,
                "next_level_points": next_level_points,
                "progress_to_next_level": round(progress_to_next, 2),
                "streak_days": user.streak_days,
                "total_study_time": user.total_study_time,
                "achievements_unlocked": achievement_count,
                "total_achievements": len(self.achievement_definitions),
                "rank": rank_info["rank"],
                "percentile": rank_info["percentile"],
                "recent_transactions": recent_transactions
            }
            
        except Exception as e:
            logger.error(f"Failed to get gamification stats: {e}")
            raise
    
    # Helper methods
    
    def _calculate_level(self, points: int) -> int:
        """Calculate level from points"""
        if points <= 0:
            return 1
        
        # Level formula: points = base * (multiplier ^ (level - 1))
        # Solving for level: level = log(points/base) / log(multiplier) + 1
        level = math.floor(
            math.log(points / self.config.level_base_points) / 
            math.log(self.config.level_multiplier) + 1
        )
        return max(1, level)
    
    def _calculate_points_for_level(self, level: int) -> int:
        """Calculate points required for a specific level"""
        if level <= 1:
            return 0
        return int(
            self.config.level_base_points * 
            (self.config.level_multiplier ** (level - 1))
        )
    
    async def _get_daily_points(self, user_id: str) -> int:
        """Get points earned today"""
        if self.redis:
            daily_key = f"points:daily:{user_id}:{date.today()}"
            points = self.redis.get(daily_key)
            return int(points) if points else 0
        return 0
    
    async def _check_achievements(self, user: User) -> List[Dict]:
        """Check and unlock achievements for user"""
        unlocked = []
        
        # Get user's current achievements
        result = await self.db.execute(
            select(Achievement.name)
            .join(user_achievements)
            .where(user_achievements.c.user_id == user.id)
        )
        current_achievements = set(result.scalars().all())
        
        # Check each achievement definition
        for key, definition in self.achievement_definitions.items():
            if definition["name"] in current_achievements:
                continue
            
            # Check criteria
            if await self._meets_criteria(user, definition["criteria"]):
                # Unlock achievement
                achievement = await self._unlock_achievement(
                    user, 
                    key, 
                    definition
                )
                if achievement:
                    unlocked.append({
                        "id": key,
                        "name": definition["name"],
                        "description": definition["description"],
                        "points": definition["points"],
                        "rarity": definition["rarity"]
                    })
        
        return unlocked
    
    async def _check_streak_achievements(self, user: User) -> List[Dict]:
        """Check streak-specific achievements"""
        unlocked = []
        
        for key, definition in self.achievement_definitions.items():
            if definition["type"] != AchievementType.STREAK:
                continue
            
            if "streak_days" in definition["criteria"]:
                if user.streak_days >= definition["criteria"]["streak_days"]:
                    achievement = await self._unlock_achievement(
                        user,
                        key,
                        definition
                    )
                    if achievement:
                        unlocked.append({
                            "id": key,
                            "name": definition["name"],
                            "description": definition["description"],
                            "points": definition["points"]
                        })
        
        return unlocked
    
    async def _meets_criteria(self, user: User, criteria: Dict) -> bool:
        """Check if user meets achievement criteria"""
        for key, value in criteria.items():
            if key == "level" and user.level < value:
                return False
            elif key == "streak_days" and user.streak_days < value:
                return False
            elif key == "lessons_completed":
                # Check lesson completion count
                count = await self.db.scalar(
                    select(func.count())
                    .select_from(Progress)
                    .where(
                        and_(
                            Progress.user_id == user.id,
                            Progress.status == "completed"
                        )
                    )
                )
                if count < value:
                    return False
            elif key == "perfect_scores":
                # Check perfect score count
                count = await self.db.scalar(
                    select(func.count())
                    .select_from(Assessment)
                    .where(
                        and_(
                            Assessment.user_id == user.id,
                            Assessment.score >= 100
                        )
                    )
                )
                if count < value:
                    return False
        
        return True
    
    async def _unlock_achievement(
        self,
        user: User,
        achievement_key: str,
        definition: Dict
    ) -> Optional[Achievement]:
        """Unlock an achievement for user"""
        try:
            # Check if achievement exists in database
            result = await self.db.execute(
                select(Achievement).where(Achievement.name == definition["name"])
            )
            achievement = result.scalar_one_or_none()
            
            if not achievement:
                # Create achievement
                achievement = Achievement(
                    name=definition["name"],
                    description=definition["description"],
                    points=definition["points"],
                    rarity=definition["rarity"],
                    criteria=definition["criteria"]
                )
                self.db.add(achievement)
                await self.db.flush()
            
            # Add to user's achievements
            await self.db.execute(
                user_achievements.insert().values(
                    user_id=user.id,
                    achievement_id=achievement.id
                )
            )
            
            # Award points
            await self.award_points(
                str(user.id),
                definition["points"],
                PointSource.ACHIEVEMENT,
                f"'{definition['name']}' başarımı kazanıldı"
            )
            
            # Send notification
            if self.config.achievement_unlock_notification:
                await self._send_notification(
                    str(user.id),
                    "achievement_unlocked",
                    f"🏆 Yeni Başarım: {definition['name']}",
                    definition["description"]
                )
            
            # Update metrics
            self.performance_stats["achievements_unlocked"] += 1
            metrics_collector.increment(
                "gamification.achievements_unlocked",
                tags={"achievement": achievement_key, "rarity": definition["rarity"]}
            )
            
            await self.db.commit()
            return achievement
            
        except Exception as e:
            logger.error(f"Failed to unlock achievement: {e}")
            await self.db.rollback()
            return None
    
    async def _get_achievement_progress(
        self,
        user_id: str,
        criteria: Dict
    ) -> Dict:
        """Get progress towards achievement criteria"""
        progress = {}
        
        for key, target in criteria.items():
            current = 0
            
            if key == "lessons_completed":
                current = await self.db.scalar(
                    select(func.count())
                    .select_from(Progress)
                    .where(
                        and_(
                            Progress.user_id == user_id,
                            Progress.status == "completed"
                        )
                    )
                )
            elif key == "streak_days":
                result = await self.db.execute(
                    select(User.streak_days).where(User.id == user_id)
                )
                current = result.scalar() or 0
            elif key == "level":
                result = await self.db.execute(
                    select(User.level).where(User.id == user_id)
                )
                current = result.scalar() or 1
            
            progress[key] = {
                "current": current,
                "target": target,
                "percentage": min(100, (current / target * 100) if target > 0 else 0)
            }
        
        return progress
    
    async def _update_leaderboard_cache(self, user_id: str, points: int) -> None:
        """Update leaderboard cache with new points"""
        if self.redis:
            # Update sorted set for fast ranking
            self.redis.zadd("leaderboard:all", {user_id: points})
            
            # Expire old entries
            self.redis.zremrangebyrank("leaderboard:all", 0, -1001)
    
    async def _get_previous_rankings(
        self,
        period: str,
        subject: Optional[str]
    ) -> Dict[str, int]:
        """Get previous period rankings for change calculation"""
        if not self.redis:
            return {}
        
        cache_key = f"leaderboard:previous:{period}:{subject or 'all'}"
        cached = self.redis.get(cache_key)
        
        if cached:
            return json.loads(cached)
        
        return {}
    
    async def _send_notification(
        self,
        user_id: str,
        notification_type: str,
        title: str,
        message: str
    ) -> None:
        """Send notification to user"""
        try:
            notification = Notification(
                user_id=user_id,
                type=notification_type,
                title=title,
                message=message
            )
            self.db.add(notification)
            await self.db.commit()
            
            # Also send real-time notification if available
            if self.redis:
                self.redis.publish(
                    f"notifications:{user_id}",
                    json.dumps({
                        "type": notification_type,
                        "title": title,
                        "message": message,
                        "timestamp": datetime.now().isoformat()
                    })
                )
            
        except Exception as e:
            logger.error(f"Failed to send notification: {e}")
    
    def get_performance_stats(self) -> Dict:
        """Get service performance statistics"""
        return self.performance_stats.copy()