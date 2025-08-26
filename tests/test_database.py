"""
Database CRUD Tests
TEKNOFEST 2025 - Eğitim Teknolojileri
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy import select, delete

from src.database.models import (
    Base, User, UserRole, Quiz, Question, Answer,
    StudentProgress, Achievement, LearningPath,
    StudySession, Notification
)
from src.database.session import get_db, init_db


@pytest.fixture
async def test_db():
    """Create test database session"""
    # Create in-memory SQLite database for testing
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        echo=False
    )
    
    # Create tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    # Create session factory
    async_session = sessionmaker(
        engine, class_=AsyncSession, expire_on_commit=False
    )
    
    async with async_session() as session:
        yield session
        await session.rollback()
    
    await engine.dispose()


@pytest.mark.asyncio
class TestUserCRUD:
    """Test User model CRUD operations"""
    
    async def test_create_user(self, test_db):
        """Test creating a new user"""
        user = User(
            username="testuser",
            email="test@example.com",
            hashed_password="hashed_password_123",
            full_name="Test User",
            role=UserRole.STUDENT,
            is_active=True,
            points=0,
            level=1
        )
        
        test_db.add(user)
        await test_db.commit()
        await test_db.refresh(user)
        
        assert user.id is not None
        assert user.username == "testuser"
        assert user.email == "test@example.com"
        assert user.role == UserRole.STUDENT
        assert user.points == 0
        assert user.level == 1
    
    async def test_read_user(self, test_db):
        """Test reading user from database"""
        # Create user
        user = User(
            username="readtest",
            email="read@example.com",
            hashed_password="hashed",
            full_name="Read Test",
            role=UserRole.TEACHER
        )
        test_db.add(user)
        await test_db.commit()
        
        # Read user
        result = await test_db.execute(
            select(User).where(User.username == "readtest")
        )
        fetched_user = result.scalar_one()
        
        assert fetched_user.username == "readtest"
        assert fetched_user.email == "read@example.com"
        assert fetched_user.role == UserRole.TEACHER
    
    async def test_update_user(self, test_db):
        """Test updating user information"""
        # Create user
        user = User(
            username="updatetest",
            email="update@example.com",
            hashed_password="hashed",
            full_name="Update Test",
            points=100
        )
        test_db.add(user)
        await test_db.commit()
        
        # Update user
        user.points = 200
        user.level = 3
        user.streak_days = 7
        await test_db.commit()
        
        # Verify update
        result = await test_db.execute(
            select(User).where(User.username == "updatetest")
        )
        updated_user = result.scalar_one()
        
        assert updated_user.points == 200
        assert updated_user.level == 3
        assert updated_user.streak_days == 7
    
    async def test_delete_user(self, test_db):
        """Test deleting user"""
        # Create user
        user = User(
            username="deletetest",
            email="delete@example.com",
            hashed_password="hashed",
            full_name="Delete Test"
        )
        test_db.add(user)
        await test_db.commit()
        user_id = user.id
        
        # Delete user
        await test_db.delete(user)
        await test_db.commit()
        
        # Verify deletion
        result = await test_db.execute(
            select(User).where(User.id == user_id)
        )
        deleted_user = result.scalar_one_or_none()
        
        assert deleted_user is None
    
    async def test_user_unique_constraints(self, test_db):
        """Test unique constraints on username and email"""
        # Create first user
        user1 = User(
            username="unique",
            email="unique@example.com",
            hashed_password="hashed",
            full_name="First User"
        )
        test_db.add(user1)
        await test_db.commit()
        
        # Try to create user with same username
        user2 = User(
            username="unique",  # Same username
            email="different@example.com",
            hashed_password="hashed",
            full_name="Second User"
        )
        test_db.add(user2)
        
        with pytest.raises(Exception):  # Should raise IntegrityError
            await test_db.commit()
        
        await test_db.rollback()
        
        # Try to create user with same email
        user3 = User(
            username="different",
            email="unique@example.com",  # Same email
            hashed_password="hashed",
            full_name="Third User"
        )
        test_db.add(user3)
        
        with pytest.raises(Exception):  # Should raise IntegrityError
            await test_db.commit()


@pytest.mark.asyncio
class TestQuizCRUD:
    """Test Quiz and Question models CRUD operations"""
    
    async def test_create_quiz_with_questions(self, test_db):
        """Test creating quiz with questions"""
        # Create user first
        user = User(
            username="quizuser",
            email="quiz@example.com",
            hashed_password="hashed",
            full_name="Quiz User"
        )
        test_db.add(user)
        await test_db.commit()
        
        # Create quiz
        quiz = Quiz(
            title="Math Quiz",
            description="Basic mathematics quiz",
            subject="Mathematics",
            grade_level=10,
            created_by_id=user.id,
            difficulty_level=0.5,
            estimated_time=30,
            max_attempts=3,
            passing_score=70,
            is_active=True
        )
        test_db.add(quiz)
        await test_db.commit()
        
        # Create questions
        question1 = Question(
            quiz_id=quiz.id,
            question_text="What is 2+2?",
            question_type="multiple_choice",
            difficulty=0.3,
            points=10,
            order=1
        )
        
        question2 = Question(
            quiz_id=quiz.id,
            question_text="What is 5*5?",
            question_type="multiple_choice",
            difficulty=0.5,
            points=15,
            order=2
        )
        
        test_db.add_all([question1, question2])
        await test_db.commit()
        
        # Verify quiz and questions
        result = await test_db.execute(
            select(Quiz).where(Quiz.title == "Math Quiz")
        )
        fetched_quiz = result.scalar_one()
        
        assert fetched_quiz.title == "Math Quiz"
        assert fetched_quiz.difficulty_level == 0.5
        
        # Get questions
        result = await test_db.execute(
            select(Question).where(Question.quiz_id == fetched_quiz.id)
        )
        questions = result.scalars().all()
        
        assert len(questions) == 2
        assert questions[0].question_text == "What is 2+2?"
        assert questions[1].question_text == "What is 5*5?"
    
    async def test_create_answers(self, test_db):
        """Test creating answers for questions"""
        # Create user and quiz
        user = User(
            username="answeruser",
            email="answer@example.com",
            hashed_password="hashed",
            full_name="Answer User"
        )
        test_db.add(user)
        await test_db.commit()
        
        quiz = Quiz(
            title="Answer Quiz",
            subject="Math",
            created_by_id=user.id
        )
        test_db.add(quiz)
        await test_db.commit()
        
        question = Question(
            quiz_id=quiz.id,
            question_text="What is 2+2?",
            question_type="multiple_choice"
        )
        test_db.add(question)
        await test_db.commit()
        
        # Create answers
        answers = [
            Answer(question_id=question.id, answer_text="3", is_correct=False, order=1),
            Answer(question_id=question.id, answer_text="4", is_correct=True, order=2),
            Answer(question_id=question.id, answer_text="5", is_correct=False, order=3),
            Answer(question_id=question.id, answer_text="6", is_correct=False, order=4)
        ]
        
        test_db.add_all(answers)
        await test_db.commit()
        
        # Verify answers
        result = await test_db.execute(
            select(Answer).where(Answer.question_id == question.id)
        )
        fetched_answers = result.scalars().all()
        
        assert len(fetched_answers) == 4
        correct_answers = [a for a in fetched_answers if a.is_correct]
        assert len(correct_answers) == 1
        assert correct_answers[0].answer_text == "4"


@pytest.mark.asyncio
class TestStudentProgress:
    """Test StudentProgress model operations"""
    
    async def test_create_progress_record(self, test_db):
        """Test creating student progress record"""
        # Create user and quiz
        user = User(
            username="progressuser",
            email="progress@example.com",
            hashed_password="hashed",
            full_name="Progress User"
        )
        test_db.add(user)
        
        quiz = Quiz(
            title="Progress Quiz",
            subject="Science",
            created_by_id=user.id
        )
        test_db.add(quiz)
        await test_db.commit()
        
        # Create progress record
        progress = StudentProgress(
            student_id=user.id,
            quiz_id=quiz.id,
            score=85.5,
            completed_questions=8,
            total_questions=10,
            time_spent=1200,  # 20 minutes in seconds
            attempt_number=1,
            is_completed=True
        )
        test_db.add(progress)
        await test_db.commit()
        
        # Verify progress
        result = await test_db.execute(
            select(StudentProgress).where(
                StudentProgress.student_id == user.id
            )
        )
        fetched_progress = result.scalar_one()
        
        assert fetched_progress.score == 85.5
        assert fetched_progress.completed_questions == 8
        assert fetched_progress.is_completed is True
    
    async def test_update_progress(self, test_db):
        """Test updating student progress"""
        # Create initial progress
        user = User(
            username="updateprogress",
            email="updateprog@example.com",
            hashed_password="hashed",
            full_name="Update Progress"
        )
        test_db.add(user)
        
        quiz = Quiz(
            title="Update Quiz",
            subject="History",
            created_by_id=user.id
        )
        test_db.add(quiz)
        await test_db.commit()
        
        progress = StudentProgress(
            student_id=user.id,
            quiz_id=quiz.id,
            score=50,
            completed_questions=5,
            total_questions=10,
            is_completed=False
        )
        test_db.add(progress)
        await test_db.commit()
        
        # Update progress
        progress.score = 80
        progress.completed_questions = 10
        progress.is_completed = True
        progress.completed_at = datetime.utcnow()
        await test_db.commit()
        
        # Verify update
        result = await test_db.execute(
            select(StudentProgress).where(
                StudentProgress.student_id == user.id
            )
        )
        updated_progress = result.scalar_one()
        
        assert updated_progress.score == 80
        assert updated_progress.completed_questions == 10
        assert updated_progress.is_completed is True
        assert updated_progress.completed_at is not None


@pytest.mark.asyncio
class TestAchievements:
    """Test Achievement model operations"""
    
    async def test_create_achievement(self, test_db):
        """Test creating achievement"""
        achievement = Achievement(
            name="First Quiz",
            description="Complete your first quiz",
            icon="trophy",
            points=50,
            category="milestone",
            criteria={"quizzes_completed": 1}
        )
        test_db.add(achievement)
        await test_db.commit()
        
        assert achievement.id is not None
        assert achievement.name == "First Quiz"
        assert achievement.points == 50
    
    async def test_user_achievements(self, test_db):
        """Test linking achievements to users"""
        # Create user
        user = User(
            username="achiever",
            email="achiever@example.com",
            hashed_password="hashed",
            full_name="Achiever"
        )
        test_db.add(user)
        
        # Create achievements
        achievement1 = Achievement(
            name="Quick Learner",
            description="Complete 5 quizzes in a day",
            points=100
        )
        achievement2 = Achievement(
            name="Perfect Score",
            description="Get 100% on a quiz",
            points=75
        )
        test_db.add_all([achievement1, achievement2])
        await test_db.commit()
        
        # Link achievements to user (would typically be done through a junction table)
        # For this test, we'll just verify they exist
        result = await test_db.execute(select(Achievement))
        achievements = result.scalars().all()
        
        assert len(achievements) == 2
        assert sum(a.points for a in achievements) == 175


@pytest.mark.asyncio
class TestLearningPath:
    """Test LearningPath model operations"""
    
    async def test_create_learning_path(self, test_db):
        """Test creating learning path"""
        user = User(
            username="learner",
            email="learner@example.com",
            hashed_password="hashed",
            full_name="Learner"
        )
        test_db.add(user)
        await test_db.commit()
        
        path = LearningPath(
            student_id=user.id,
            title="Mathematics Fundamentals",
            description="Learn basic math concepts",
            subject="Mathematics",
            difficulty_level=0.3,
            estimated_hours=20,
            modules=[
                {"name": "Addition", "hours": 5},
                {"name": "Subtraction", "hours": 5},
                {"name": "Multiplication", "hours": 5},
                {"name": "Division", "hours": 5}
            ],
            is_active=True
        )
        test_db.add(path)
        await test_db.commit()
        
        assert path.id is not None
        assert path.title == "Mathematics Fundamentals"
        assert len(path.modules) == 4
        assert path.estimated_hours == 20
    
    async def test_update_learning_path_progress(self, test_db):
        """Test updating learning path progress"""
        user = User(
            username="pathuser",
            email="path@example.com",
            hashed_password="hashed",
            full_name="Path User"
        )
        test_db.add(user)
        
        path = LearningPath(
            student_id=user.id,
            title="Science Journey",
            subject="Science",
            progress_percentage=0,
            completed_modules=0,
            total_modules=5
        )
        test_db.add(path)
        await test_db.commit()
        
        # Update progress
        path.completed_modules = 2
        path.progress_percentage = 40
        path.last_accessed_at = datetime.utcnow()
        await test_db.commit()
        
        # Verify update
        result = await test_db.execute(
            select(LearningPath).where(LearningPath.student_id == user.id)
        )
        updated_path = result.scalar_one()
        
        assert updated_path.progress_percentage == 40
        assert updated_path.completed_modules == 2


@pytest.mark.asyncio
class TestStudySession:
    """Test StudySession model operations"""
    
    async def test_create_study_session(self, test_db):
        """Test creating study session"""
        user = User(
            username="studier",
            email="studier@example.com",
            hashed_password="hashed",
            full_name="Studier"
        )
        test_db.add(user)
        await test_db.commit()
        
        session = StudySession(
            student_id=user.id,
            subject="Physics",
            topic="Mechanics",
            duration_minutes=45,
            notes="Studied Newton's laws",
            productivity_score=0.8,
            started_at=datetime.utcnow(),
            ended_at=datetime.utcnow() + timedelta(minutes=45)
        )
        test_db.add(session)
        await test_db.commit()
        
        assert session.id is not None
        assert session.duration_minutes == 45
        assert session.productivity_score == 0.8
    
    async def test_query_study_sessions(self, test_db):
        """Test querying study sessions by date range"""
        user = User(
            username="regular",
            email="regular@example.com",
            hashed_password="hashed",
            full_name="Regular Student"
        )
        test_db.add(user)
        await test_db.commit()
        
        # Create multiple sessions
        base_time = datetime.utcnow()
        sessions = [
            StudySession(
                student_id=user.id,
                subject=f"Subject{i}",
                duration_minutes=30 + i*10,
                started_at=base_time - timedelta(days=i)
            )
            for i in range(5)
        ]
        test_db.add_all(sessions)
        await test_db.commit()
        
        # Query sessions from last 3 days
        three_days_ago = base_time - timedelta(days=3)
        result = await test_db.execute(
            select(StudySession).where(
                (StudySession.student_id == user.id) &
                (StudySession.started_at >= three_days_ago)
            )
        )
        recent_sessions = result.scalars().all()
        
        assert len(recent_sessions) == 3


@pytest.mark.asyncio
class TestNotifications:
    """Test Notification model operations"""
    
    async def test_create_notification(self, test_db):
        """Test creating notification"""
        user = User(
            username="notifuser",
            email="notif@example.com",
            hashed_password="hashed",
            full_name="Notification User"
        )
        test_db.add(user)
        await test_db.commit()
        
        notification = Notification(
            user_id=user.id,
            title="New Achievement!",
            message="You earned the 'Quick Learner' achievement",
            type="achievement",
            priority="medium",
            is_read=False
        )
        test_db.add(notification)
        await test_db.commit()
        
        assert notification.id is not None
        assert notification.is_read is False
        assert notification.priority == "medium"
    
    async def test_mark_notification_read(self, test_db):
        """Test marking notification as read"""
        user = User(
            username="reader",
            email="reader@example.com",
            hashed_password="hashed",
            full_name="Reader"
        )
        test_db.add(user)
        
        notification = Notification(
            user_id=user.id,
            title="Quiz Available",
            message="New quiz ready",
            is_read=False
        )
        test_db.add(notification)
        await test_db.commit()
        
        # Mark as read
        notification.is_read = True
        notification.read_at = datetime.utcnow()
        await test_db.commit()
        
        # Verify
        result = await test_db.execute(
            select(Notification).where(Notification.user_id == user.id)
        )
        updated_notif = result.scalar_one()
        
        assert updated_notif.is_read is True
        assert updated_notif.read_at is not None
    
    async def test_query_unread_notifications(self, test_db):
        """Test querying unread notifications"""
        user = User(
            username="busyuser",
            email="busy@example.com",
            hashed_password="hashed",
            full_name="Busy User"
        )
        test_db.add(user)
        await test_db.commit()
        
        # Create mixed notifications
        notifications = [
            Notification(user_id=user.id, title=f"Notif {i}", message="Test", is_read=i % 2 == 0)
            for i in range(6)
        ]
        test_db.add_all(notifications)
        await test_db.commit()
        
        # Query unread only
        result = await test_db.execute(
            select(Notification).where(
                (Notification.user_id == user.id) &
                (Notification.is_read == False)
            )
        )
        unread = result.scalars().all()
        
        assert len(unread) == 3  # Odd numbered indices are unread


@pytest.mark.asyncio
class TestDatabaseTransactions:
    """Test database transactions and rollback"""
    
    async def test_transaction_rollback(self, test_db):
        """Test transaction rollback on error"""
        user = User(
            username="transuser",
            email="trans@example.com",
            hashed_password="hashed",
            full_name="Transaction User"
        )
        test_db.add(user)
        await test_db.commit()
        
        initial_points = user.points
        
        try:
            # Start transaction
            user.points += 100
            
            # This should cause an error (invalid quiz)
            invalid_quiz = Quiz(
                title=None,  # Required field
                subject="Math",
                created_by_id="invalid_id"  # Invalid foreign key
            )
            test_db.add(invalid_quiz)
            await test_db.commit()
            
        except Exception:
            await test_db.rollback()
        
        # Verify rollback
        await test_db.refresh(user)
        assert user.points == initial_points
    
    async def test_bulk_operations(self, test_db):
        """Test bulk insert and delete operations"""
        # Bulk insert users
        users = [
            User(
                username=f"bulk{i}",
                email=f"bulk{i}@example.com",
                hashed_password="hashed",
                full_name=f"Bulk User {i}"
            )
            for i in range(10)
        ]
        test_db.add_all(users)
        await test_db.commit()
        
        # Verify bulk insert
        result = await test_db.execute(
            select(User).where(User.username.like("bulk%"))
        )
        bulk_users = result.scalars().all()
        assert len(bulk_users) == 10
        
        # Bulk delete
        await test_db.execute(
            delete(User).where(User.username.like("bulk%"))
        )
        await test_db.commit()
        
        # Verify bulk delete
        result = await test_db.execute(
            select(User).where(User.username.like("bulk%"))
        )
        remaining = result.scalars().all()
        assert len(remaining) == 0