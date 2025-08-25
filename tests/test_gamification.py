"""
Comprehensive tests for Gamification System
TEKNOFEST 2025 - Eğitim Teknolojileri
"""

import pytest
import asyncio
from datetime import datetime, timedelta, date
from unittest.mock import Mock, patch, AsyncMock
import json
import uuid

from sqlalchemy.ext.asyncio import AsyncSession
from redis import Redis

from src.core.gamification_service import (
    GamificationService, GamificationConfig,
    PointSource, AchievementType, PointTransaction,
    LeaderboardEntry
)
from src.database.models import User, Achievement, UserRole
from src.exceptions import ValidationError, NotFoundError


@pytest.fixture
async def mock_db_session():
    """Create a mock database session"""
    session = AsyncMock(spec=AsyncSession)
    return session


@pytest.fixture
def mock_redis():
    """Create a mock Redis client"""
    redis = Mock(spec=Redis)
    redis.get = Mock(return_value=None)
    redis.set = Mock(return_value=True)
    redis.setex = Mock(return_value=True)
    redis.delete = Mock(return_value=1)
    redis.lpush = Mock(return_value=1)
    redis.lrange = Mock(return_value=[])
    redis.zadd = Mock(return_value=1)
    redis.zremrangebyrank = Mock(return_value=0)
    redis.incrby = Mock(return_value=1)
    redis.expire = Mock(return_value=True)
    redis.ltrim = Mock(return_value=True)
    redis.publish = Mock(return_value=1)
    return redis


@pytest.fixture
def mock_cache_manager():
    """Create a mock cache manager"""
    cache = Mock()
    cache.get = AsyncMock(return_value=None)
    cache.set = AsyncMock(return_value=True)
    cache.delete = AsyncMock(return_value=True)
    return cache


@pytest.fixture
def gamification_config():
    """Create test configuration"""
    return GamificationConfig(
        points_multiplier=1.5,
        streak_threshold=3,
        level_base_points=100,
        level_multiplier=1.5,
        max_daily_points=500,
        achievement_unlock_notification=True,
        leaderboard_size=100
    )


@pytest.fixture
async def gamification_service(mock_db_session, mock_redis, mock_cache_manager, gamification_config):
    """Create gamification service instance"""
    service = GamificationService(
        db_session=mock_db_session,
        cache_manager=mock_cache_manager,
        redis_client=mock_redis,
        config=gamification_config
    )
    return service


@pytest.fixture
def sample_user():
    """Create a sample user"""
    user = Mock(spec=User)
    user.id = uuid.uuid4()
    user.username = "test_user"
    user.email = "test@example.com"
    user.points = 150
    user.level = 2
    user.streak_days = 5
    user.total_study_time = 120
    user.role = UserRole.STUDENT
    user.is_active = True
    user.avatar_url = "https://example.com/avatar.jpg"
    return user


class TestPointSystem:
    """Test point awarding and management"""
    
    @pytest.mark.asyncio
    async def test_award_points_success(self, gamification_service, mock_db_session, sample_user):
        """Test successful point awarding"""
        # Setup
        mock_db_session.execute.return_value.scalar_one_or_none.return_value = sample_user
        mock_db_session.commit = AsyncMock()
        
        # Execute
        result = await gamification_service.award_points(
            user_id=str(sample_user.id),
            amount=50,
            source=PointSource.LESSON_COMPLETE,
            description="Completed Python basics lesson",
            metadata={"lesson_id": "python_101"}
        )
        
        # Assert
        assert result["success"] is True
        assert result["old_points"] == 150
        assert result["new_points"] == 225  # 150 + (50 * 1.5 multiplier)
        assert result["transaction"]["amount"] == 75  # 50 * 1.5
        assert mock_db_session.commit.called
    
    @pytest.mark.asyncio
    async def test_award_points_daily_limit(self, gamification_service, mock_db_session, mock_redis, sample_user):
        """Test daily point limit enforcement"""
        # Setup
        mock_db_session.execute.return_value.scalar_one_or_none.return_value = sample_user
        mock_redis.get.return_value = b"450"  # Already earned 450 points today
        
        # Execute
        result = await gamification_service.award_points(
            user_id=str(sample_user.id),
            amount=100,
            source=PointSource.QUIZ_COMPLETE,
            description="Quiz completed"
        )
        
        # Assert
        assert result["new_points"] == 200  # 150 + 50 (limited to daily max)
        assert result["transaction"]["amount"] == 50  # Limited amount
    
    @pytest.mark.asyncio
    async def test_award_points_level_up(self, gamification_service, mock_db_session, sample_user):
        """Test level up when awarding points"""
        # Setup
        sample_user.points = 200
        sample_user.level = 2
        mock_db_session.execute.return_value.scalar_one_or_none.return_value = sample_user
        mock_db_session.commit = AsyncMock()
        
        # Execute
        result = await gamification_service.award_points(
            user_id=str(sample_user.id),
            amount=100,  # Will result in 150 points with multiplier
            source=PointSource.PERFECT_SCORE,
            description="Perfect score on advanced quiz"
        )
        
        # Assert
        assert result["level_up"] is True
        assert result["new_level"] == 3
        assert result["old_level"] == 2
    
    @pytest.mark.asyncio
    async def test_award_points_user_not_found(self, gamification_service, mock_db_session):
        """Test error when user not found"""
        # Setup
        mock_db_session.execute.return_value.scalar_one_or_none.return_value = None
        
        # Execute & Assert
        with pytest.raises(NotFoundError):
            await gamification_service.award_points(
                user_id="nonexistent",
                amount=50,
                source=PointSource.DAILY_LOGIN,
                description="Daily login"
            )


class TestStreakSystem:
    """Test streak tracking and management"""
    
    @pytest.mark.asyncio
    async def test_update_streak_first_day(self, gamification_service, mock_db_session, mock_redis, sample_user):
        """Test streak update for first activity"""
        # Setup
        sample_user.streak_days = 0
        mock_db_session.execute.return_value.scalar_one_or_none.return_value = sample_user
        mock_db_session.commit = AsyncMock()
        mock_redis.get.return_value = None  # No previous activity
        
        # Execute
        result = await gamification_service.update_streak(str(sample_user.id))
        
        # Assert
        assert result["streak_days"] == 1
        assert result["updated"] is True
        assert sample_user.streak_days == 1
    
    @pytest.mark.asyncio
    async def test_update_streak_consecutive_day(self, gamification_service, mock_db_session, mock_redis, sample_user):
        """Test streak increment for consecutive day"""
        # Setup
        sample_user.streak_days = 5
        mock_db_session.execute.return_value.scalar_one_or_none.return_value = sample_user
        mock_db_session.commit = AsyncMock()
        yesterday = (date.today() - timedelta(days=1)).isoformat()
        mock_redis.get.return_value = yesterday.encode()
        
        # Execute
        result = await gamification_service.update_streak(str(sample_user.id))
        
        # Assert
        assert result["streak_days"] == 6
        assert result["updated"] is True
        assert sample_user.streak_days == 6
    
    @pytest.mark.asyncio
    async def test_update_streak_broken(self, gamification_service, mock_db_session, mock_redis, sample_user):
        """Test streak reset when broken"""
        # Setup
        sample_user.streak_days = 10
        mock_db_session.execute.return_value.scalar_one_or_none.return_value = sample_user
        mock_db_session.commit = AsyncMock()
        three_days_ago = (date.today() - timedelta(days=3)).isoformat()
        mock_redis.get.return_value = three_days_ago.encode()
        
        # Execute
        result = await gamification_service.update_streak(str(sample_user.id))
        
        # Assert
        assert result["streak_days"] == 1  # Reset to 1
        assert result["updated"] is True
        assert sample_user.streak_days == 1
    
    @pytest.mark.asyncio
    async def test_update_streak_same_day(self, gamification_service, mock_db_session, mock_redis, sample_user):
        """Test streak update on same day (should not update)"""
        # Setup
        mock_db_session.execute.return_value.scalar_one_or_none.return_value = sample_user
        today = date.today().isoformat()
        mock_redis.get.return_value = today.encode()
        
        # Execute
        result = await gamification_service.update_streak(str(sample_user.id))
        
        # Assert
        assert result["updated"] is False
        assert result["message"] == "Streak already updated today"


class TestLeaderboard:
    """Test leaderboard functionality"""
    
    @pytest.mark.asyncio
    async def test_get_leaderboard_all_time(self, gamification_service, mock_db_session):
        """Test getting all-time leaderboard"""
        # Setup
        users = [
            Mock(id=uuid.uuid4(), username=f"user{i}", points=1000-i*100, 
                 level=10-i, streak_days=30-i*5, avatar_url=None)
            for i in range(5)
        ]
        mock_db_session.execute.return_value.scalars.return_value.all.return_value = users
        mock_db_session.scalar = AsyncMock(return_value=2)  # Achievement count
        
        # Execute
        leaderboard = await gamification_service.get_leaderboard(
            period="all",
            limit=10,
            offset=0
        )
        
        # Assert
        assert len(leaderboard) == 5
        assert leaderboard[0].rank == 1
        assert leaderboard[0].points == 1000
        assert leaderboard[-1].rank == 5
        assert leaderboard[-1].points == 600
    
    @pytest.mark.asyncio
    async def test_get_leaderboard_with_cache(self, gamification_service, mock_redis):
        """Test leaderboard retrieval from cache"""
        # Setup
        cached_data = [
            {
                "rank": 1,
                "user_id": str(uuid.uuid4()),
                "username": "cached_user",
                "avatar_url": None,
                "points": 5000,
                "level": 25,
                "achievements_count": 15,
                "streak_days": 50,
                "change": 2
            }
        ]
        mock_redis.get.return_value = json.dumps(cached_data).encode()
        
        # Execute
        leaderboard = await gamification_service.get_leaderboard()
        
        # Assert
        assert len(leaderboard) == 1
        assert leaderboard[0].username == "cached_user"
        assert leaderboard[0].points == 5000
        assert gamification_service.performance_stats["cache_hits"] == 1
    
    @pytest.mark.asyncio
    async def test_get_user_rank(self, gamification_service, mock_db_session, sample_user):
        """Test getting user's rank"""
        # Setup
        mock_db_session.execute.return_value.scalar_one_or_none.return_value = sample_user
        mock_db_session.scalar = AsyncMock(side_effect=[10, 100])  # 10 users ahead, 100 total
        
        # Execute
        rank_info = await gamification_service.get_user_rank(str(sample_user.id))
        
        # Assert
        assert rank_info["rank"] == 11
        assert rank_info["total_users"] == 100
        assert rank_info["percentile"] == 90.0


class TestAchievements:
    """Test achievement system"""
    
    @pytest.mark.asyncio
    async def test_get_user_achievements(self, gamification_service, mock_db_session):
        """Test getting user achievements"""
        # Setup
        unlocked_achievement = Mock(
            id=uuid.uuid4(),
            name="First Steps",
            description="Complete your first lesson",
            icon_url="https://example.com/icon.png",
            points=10,
            rarity="common",
            created_at=datetime.now()
        )
        mock_db_session.execute.return_value.scalars.return_value.all.return_value = [unlocked_achievement]
        
        # Execute
        achievements = await gamification_service.get_user_achievements(
            user_id=str(uuid.uuid4()),
            include_locked=True
        )
        
        # Assert
        assert len(achievements) > 0
        unlocked = [a for a in achievements if a["unlocked"]]
        locked = [a for a in achievements if not a["unlocked"]]
        assert len(unlocked) == 1
        assert len(locked) > 0
    
    @pytest.mark.asyncio
    async def test_check_achievements_level_based(self, gamification_service, mock_db_session, sample_user):
        """Test checking level-based achievements"""
        # Setup
        sample_user.level = 5
        mock_db_session.execute.return_value.scalars.return_value.all.return_value = []  # No current achievements
        mock_db_session.commit = AsyncMock()
        mock_db_session.flush = AsyncMock()
        
        # Mock achievement creation
        new_achievement = Mock(id=uuid.uuid4())
        mock_db_session.add = Mock()
        mock_db_session.execute = AsyncMock()
        
        # Execute
        with patch.object(gamification_service, '_unlock_achievement', return_value=new_achievement):
            unlocked = await gamification_service._check_achievements(sample_user)
        
        # Assert
        assert any(a["name"] == "Yükselen Yıldız" for a in unlocked)
    
    @pytest.mark.asyncio
    async def test_achievement_progress_tracking(self, gamification_service, mock_db_session):
        """Test tracking progress towards achievements"""
        # Setup
        user_id = str(uuid.uuid4())
        mock_db_session.scalar = AsyncMock(return_value=5)  # 5 lessons completed
        mock_db_session.execute.return_value.scalar.return_value = 3  # Level 3
        
        # Execute
        progress = await gamification_service._get_achievement_progress(
            user_id,
            {"lessons_completed": 10, "level": 5}
        )
        
        # Assert
        assert progress["lessons_completed"]["current"] == 5
        assert progress["lessons_completed"]["target"] == 10
        assert progress["lessons_completed"]["percentage"] == 50.0


class TestGamificationStats:
    """Test comprehensive gamification statistics"""
    
    @pytest.mark.asyncio
    async def test_get_gamification_stats(self, gamification_service, mock_db_session, mock_redis, sample_user):
        """Test getting comprehensive user stats"""
        # Setup
        mock_db_session.execute.return_value.scalar_one_or_none.return_value = sample_user
        mock_db_session.scalar = AsyncMock(side_effect=[10, 100, 5])  # Rank, total users, achievements
        
        transactions = [
            {"amount": 50, "source": "lesson_complete", "description": "Lesson 1", "timestamp": "2025-01-22T10:00:00"}
        ]
        mock_redis.lrange.return_value = [json.dumps(t).encode() for t in transactions]
        
        # Execute
        stats = await gamification_service.get_gamification_stats(str(sample_user.id))
        
        # Assert
        assert stats["user_id"] == str(sample_user.id)
        assert stats["username"] == sample_user.username
        assert stats["points"] == sample_user.points
        assert stats["level"] == sample_user.level
        assert stats["streak_days"] == sample_user.streak_days
        assert stats["rank"] == 11
        assert stats["percentile"] == 90.0
        assert stats["achievements_unlocked"] == 5
        assert len(stats["recent_transactions"]) == 1


class TestLevelCalculation:
    """Test level calculation logic"""
    
    def test_calculate_level_from_points(self, gamification_service):
        """Test level calculation from points"""
        # Test various point values
        assert gamification_service._calculate_level(0) == 1
        assert gamification_service._calculate_level(50) == 1
        assert gamification_service._calculate_level(100) == 1
        assert gamification_service._calculate_level(150) == 2
        assert gamification_service._calculate_level(225) == 3
        assert gamification_service._calculate_level(337) == 3
        assert gamification_service._calculate_level(338) == 4
    
    def test_calculate_points_for_level(self, gamification_service):
        """Test points required for specific levels"""
        assert gamification_service._calculate_points_for_level(1) == 0
        assert gamification_service._calculate_points_for_level(2) == 100
        assert gamification_service._calculate_points_for_level(3) == 150
        assert gamification_service._calculate_points_for_level(4) == 225
        assert gamification_service._calculate_points_for_level(5) == 337


class TestPerformanceAndCaching:
    """Test performance optimizations and caching"""
    
    @pytest.mark.asyncio
    async def test_cache_utilization(self, gamification_service, mock_redis):
        """Test that cache is properly utilized"""
        # Setup
        mock_redis.get.return_value = None
        mock_redis.setex = Mock()
        
        # Execute
        await gamification_service.get_leaderboard()
        
        # Assert
        assert mock_redis.get.called
        assert mock_redis.setex.called
    
    @pytest.mark.asyncio
    async def test_leaderboard_cache_update(self, gamification_service, mock_redis):
        """Test leaderboard cache updates on point changes"""
        # Setup
        user_id = str(uuid.uuid4())
        points = 500
        
        # Execute
        await gamification_service._update_leaderboard_cache(user_id, points)
        
        # Assert
        mock_redis.zadd.assert_called_with("leaderboard:all", {user_id: points})
        mock_redis.zremrangebyrank.assert_called_with("leaderboard:all", 0, -1001)
    
    def test_performance_stats_tracking(self, gamification_service):
        """Test performance statistics tracking"""
        # Get initial stats
        stats = gamification_service.get_performance_stats()
        
        # Assert
        assert "points_awarded" in stats
        assert "achievements_unlocked" in stats
        assert "leaderboard_updates" in stats
        assert "cache_hits" in stats


class TestErrorHandling:
    """Test error handling and edge cases"""
    
    @pytest.mark.asyncio
    async def test_database_error_handling(self, gamification_service, mock_db_session):
        """Test handling of database errors"""
        # Setup
        mock_db_session.execute.side_effect = Exception("Database connection error")
        
        # Execute & Assert
        with pytest.raises(Exception):
            await gamification_service.award_points(
                user_id="test",
                amount=50,
                source=PointSource.DAILY_LOGIN,
                description="Test"
            )
        
        # Verify rollback was called
        mock_db_session.rollback.assert_called()
    
    @pytest.mark.asyncio
    async def test_invalid_point_source(self, gamification_service):
        """Test handling of invalid point source"""
        # This should be caught at the API level, but test service handling
        with pytest.raises(AttributeError):
            await gamification_service.award_points(
                user_id="test",
                amount=50,
                source="invalid_source",  # Invalid source
                description="Test"
            )
    
    @pytest.mark.asyncio
    async def test_concurrent_updates(self, gamification_service, mock_db_session, sample_user):
        """Test handling of concurrent updates"""
        # Setup
        mock_db_session.execute.return_value.scalar_one_or_none.return_value = sample_user
        mock_db_session.commit = AsyncMock()
        
        # Execute multiple concurrent updates
        tasks = [
            gamification_service.award_points(
                user_id=str(sample_user.id),
                amount=10,
                source=PointSource.DAILY_LOGIN,
                description=f"Test {i}"
            )
            for i in range(5)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Assert no exceptions
        assert all(not isinstance(r, Exception) for r in results)