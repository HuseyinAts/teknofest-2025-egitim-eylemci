"""
Integration Tests
TEKNOFEST 2025 - Eğitim Teknolojileri
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

from src.database.models import Base, User, UserRole, Quiz, Question, StudentProgress
from src.api.auth import authenticate_user, create_user, create_access_token, get_current_user
from src.agents.learning_path_agent_v2 import LearningPathAgent
from src.agents.study_buddy_agent_clean import StudyBuddyAgent
from src.core.irt_service import IRTService
from src.core.gamification_service import GamificationService
from src.data_processor import DataProcessor


@pytest.fixture
async def integration_db():
    """Create integration test database"""
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        echo=False
    )
    
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    async_session = sessionmaker(
        engine, class_=AsyncSession, expire_on_commit=False
    )
    
    async with async_session() as session:
        yield session
    
    await engine.dispose()


@pytest.mark.asyncio
@pytest.mark.integration
class TestAuthenticationFlow:
    """Test complete authentication flow"""
    
    async def test_user_registration_and_login(self, integration_db):
        """Test user registration and login flow"""
        # Register new user
        new_user = await create_user(
            db=integration_db,
            username="testuser",
            email="test@example.com",
            password="securepassword123",
            full_name="Test User",
            role=UserRole.STUDENT
        )
        
        assert new_user.id is not None
        assert new_user.username == "testuser"
        
        # Authenticate user
        authenticated_user = await authenticate_user(
            db=integration_db,
            username="testuser",
            password="securepassword123"
        )
        
        assert authenticated_user is not None
        assert authenticated_user.id == new_user.id
        
        # Create access token
        token = create_access_token({
            "sub": str(authenticated_user.id),
            "username": authenticated_user.username,
            "role": authenticated_user.role.value
        })
        
        assert token is not None
        assert len(token) > 100
        
        # Verify token and get current user
        mock_credentials = Mock()
        mock_credentials.credentials = token
        
        current_user = await get_current_user(mock_credentials, integration_db)
        assert current_user.id == authenticated_user.id
    
    async def test_failed_authentication_flow(self, integration_db):
        """Test failed authentication attempts"""
        # Create user
        user = await create_user(
            db=integration_db,
            username="testuser2",
            email="test2@example.com",
            password="correctpassword",
            full_name="Test User 2"
        )
        
        # Try wrong password
        failed_auth = await authenticate_user(
            db=integration_db,
            username="testuser2",
            password="wrongpassword"
        )
        
        assert failed_auth is None
        
        # Try non-existent user
        non_existent = await authenticate_user(
            db=integration_db,
            username="nonexistent",
            password="anypassword"
        )
        
        assert non_existent is None


@pytest.mark.asyncio
@pytest.mark.integration
class TestQuizWorkflow:
    """Test complete quiz workflow"""
    
    async def test_quiz_creation_and_completion(self, integration_db):
        """Test creating quiz and recording completion"""
        # Create teacher and student
        teacher = await create_user(
            db=integration_db,
            username="teacher",
            email="teacher@example.com",
            password="teacherpass",
            full_name="Teacher User",
            role=UserRole.TEACHER
        )
        
        student = await create_user(
            db=integration_db,
            username="student",
            email="student@example.com",
            password="studentpass",
            full_name="Student User",
            role=UserRole.STUDENT
        )
        
        # Create quiz
        quiz = Quiz(
            title="Math Quiz",
            description="Basic mathematics",
            subject="Mathematics",
            grade_level=10,
            created_by_id=teacher.id,
            difficulty_level=0.5,
            estimated_time=30,
            max_attempts=3,
            passing_score=70
        )
        integration_db.add(quiz)
        await integration_db.commit()
        
        # Add questions
        questions = []
        for i in range(5):
            q = Question(
                quiz_id=quiz.id,
                question_text=f"Question {i+1}",
                question_type="multiple_choice",
                difficulty=0.3 + i*0.1,
                points=10,
                order=i+1
            )
            questions.append(q)
        
        integration_db.add_all(questions)
        await integration_db.commit()
        
        # Student takes quiz
        progress = StudentProgress(
            student_id=student.id,
            quiz_id=quiz.id,
            score=80,
            completed_questions=5,
            total_questions=5,
            time_spent=1500,
            attempt_number=1,
            is_completed=True,
            completed_at=datetime.utcnow()
        )
        integration_db.add(progress)
        await integration_db.commit()
        
        # Update student points
        student.points += 80
        if student.points >= 100:
            student.level = 2
        await integration_db.commit()
        
        # Verify results
        await integration_db.refresh(student)
        assert student.points == 80
        assert progress.score == 80
        assert progress.is_completed is True


@pytest.mark.asyncio
@pytest.mark.integration
class TestLearningPathGeneration:
    """Test learning path generation workflow"""
    
    async def test_personalized_learning_path(self, integration_db):
        """Test generating personalized learning path"""
        # Create student
        student = await create_user(
            db=integration_db,
            username="learner",
            email="learner@example.com",
            password="password",
            full_name="Learner",
            role=UserRole.STUDENT
        )
        
        # Mock learning path agent
        with patch('src.agents.learning_path_agent_v2.LearningPathAgent') as MockAgent:
            mock_agent = Mock()
            mock_agent.create_personalized_path.return_value = {
                "student_id": student.id,
                "path": [
                    {
                        "week": 1,
                        "topics": ["Basic Math"],
                        "difficulty": 0.3,
                        "estimated_hours": 5
                    },
                    {
                        "week": 2,
                        "topics": ["Advanced Math"],
                        "difficulty": 0.5,
                        "estimated_hours": 6
                    }
                ],
                "total_hours": 11,
                "difficulty_progression": [0.3, 0.5]
            }
            MockAgent.return_value = mock_agent
            
            # Generate path
            agent = MockAgent()
            path = agent.create_personalized_path(
                student_profile={
                    "id": student.id,
                    "current_level": 0.3,
                    "target_level": 0.7,
                    "learning_style": "visual"
                },
                subject="Mathematics",
                duration_weeks=2
            )
            
            assert path is not None
            assert len(path["path"]) == 2
            assert path["total_hours"] == 11


@pytest.mark.asyncio
@pytest.mark.integration
class TestIRTIntegration:
    """Test IRT (Item Response Theory) integration"""
    
    async def test_adaptive_quiz_generation(self):
        """Test adaptive quiz generation based on student ability"""
        irt_service = IRTService()
        
        # Initial ability estimate
        student_ability = 0.5
        
        # Generate first question
        question_difficulty = irt_service.select_next_item(
            ability=student_ability,
            answered_items=[]
        )
        
        assert 0.3 <= question_difficulty <= 0.7
        
        # Student answers correctly
        new_ability = irt_service.update_ability(
            current_ability=student_ability,
            item_difficulty=question_difficulty,
            response_correct=True
        )
        
        assert new_ability > student_ability
        
        # Generate harder question
        next_difficulty = irt_service.select_next_item(
            ability=new_ability,
            answered_items=[question_difficulty]
        )
        
        assert next_difficulty > question_difficulty
    
    async def test_ability_convergence(self):
        """Test that ability estimate converges with more responses"""
        irt_service = IRTService()
        
        true_ability = 0.7
        estimated_ability = 0.5
        responses = []
        
        for i in range(20):
            # Generate question at estimated ability level
            item_difficulty = irt_service.select_next_item(
                ability=estimated_ability,
                answered_items=responses
            )
            
            # Simulate response based on true ability
            # Probability of correct response using IRT model
            prob_correct = 1 / (1 + np.exp(-(true_ability - item_difficulty)))
            response_correct = np.random.random() < prob_correct
            
            # Update ability estimate
            estimated_ability = irt_service.update_ability(
                current_ability=estimated_ability,
                item_difficulty=item_difficulty,
                response_correct=response_correct
            )
            
            responses.append(item_difficulty)
        
        # Check convergence
        assert abs(estimated_ability - true_ability) < 0.2


@pytest.mark.asyncio
@pytest.mark.integration
class TestGamificationFlow:
    """Test gamification system integration"""
    
    async def test_points_and_achievements(self, integration_db):
        """Test points accumulation and achievement unlocking"""
        # Create student
        student = await create_user(
            db=integration_db,
            username="gamer",
            email="gamer@example.com",
            password="password",
            full_name="Gamer",
            role=UserRole.STUDENT
        )
        
        gamification = GamificationService()
        
        # Complete activities and earn points
        activities = [
            ("quiz_completed", 50),
            ("perfect_score", 100),
            ("daily_streak", 20),
            ("quiz_completed", 50),
            ("module_completed", 75)
        ]
        
        total_points = 0
        for activity, points in activities:
            total_points += points
            student.points = total_points
            
            # Check for level up
            new_level = gamification.calculate_level(total_points)
            if new_level > student.level:
                student.level = new_level
        
        await integration_db.commit()
        await integration_db.refresh(student)
        
        assert student.points == 295
        assert student.level >= 2
        
        # Check achievements
        achievements = gamification.check_achievements(
            user_stats={
                "points": student.points,
                "level": student.level,
                "quizzes_completed": 2,
                "perfect_scores": 1,
                "streak_days": 1
            }
        )
        
        assert len(achievements) > 0


@pytest.mark.asyncio
@pytest.mark.integration
class TestStudyBuddyIntegration:
    """Test study buddy AI integration"""
    
    async def test_study_recommendations(self, integration_db):
        """Test generating study recommendations"""
        # Create student with history
        student = await create_user(
            db=integration_db,
            username="studious",
            email="studious@example.com",
            password="password",
            full_name="Studious Student",
            role=UserRole.STUDENT
        )
        
        # Add quiz history
        quiz = Quiz(
            title="Physics Quiz",
            subject="Physics",
            created_by_id=student.id
        )
        integration_db.add(quiz)
        await integration_db.commit()
        
        progress = StudentProgress(
            student_id=student.id,
            quiz_id=quiz.id,
            score=65,  # Below passing
            completed_questions=8,
            total_questions=10,
            is_completed=True
        )
        integration_db.add(progress)
        await integration_db.commit()
        
        # Mock study buddy
        with patch('src.agents.study_buddy_agent_clean.StudyBuddyAgent') as MockBuddy:
            mock_buddy = Mock()
            mock_buddy.generate_recommendations.return_value = {
                "recommendations": [
                    {
                        "subject": "Physics",
                        "topics": ["Mechanics", "Forces"],
                        "reason": "Low quiz score",
                        "priority": "high",
                        "estimated_hours": 3
                    }
                ],
                "total_study_time": 3,
                "focus_areas": ["Problem solving", "Conceptual understanding"]
            }
            MockBuddy.return_value = mock_buddy
            
            buddy = MockBuddy()
            recommendations = buddy.generate_recommendations(
                student_id=student.id,
                recent_performance=[{"quiz": "Physics", "score": 65}]
            )
            
            assert len(recommendations["recommendations"]) == 1
            assert recommendations["recommendations"][0]["priority"] == "high"


@pytest.mark.asyncio
@pytest.mark.integration
class TestDataProcessingPipeline:
    """Test data processing pipeline integration"""
    
    async def test_student_data_aggregation(self, integration_db):
        """Test aggregating student data for analysis"""
        # Create student with comprehensive data
        student = await create_user(
            db=integration_db,
            username="datastu",
            email="data@example.com",
            password="password",
            full_name="Data Student",
            role=UserRole.STUDENT
        )
        
        # Add multiple quizzes and progress
        subjects = ["Math", "Physics", "Chemistry"]
        for subject in subjects:
            quiz = Quiz(
                title=f"{subject} Quiz",
                subject=subject,
                created_by_id=student.id
            )
            integration_db.add(quiz)
            await integration_db.commit()
            
            progress = StudentProgress(
                student_id=student.id,
                quiz_id=quiz.id,
                score=70 + subjects.index(subject) * 10,
                completed_questions=10,
                total_questions=10,
                is_completed=True
            )
            integration_db.add(progress)
        
        await integration_db.commit()
        
        # Process student data
        processor = DataProcessor()
        
        with patch.object(processor, 'aggregate_student_data') as mock_aggregate:
            mock_aggregate.return_value = {
                "student_id": student.id,
                "average_score": 80,
                "subjects_studied": 3,
                "total_quizzes": 3,
                "strengths": ["Chemistry"],
                "weaknesses": ["Math"],
                "recommended_focus": "Mathematics fundamentals"
            }
            
            analysis = processor.aggregate_student_data(student.id)
            
            assert analysis["average_score"] == 80
            assert analysis["subjects_studied"] == 3
            assert "Math" in analysis["weaknesses"]


@pytest.mark.asyncio
@pytest.mark.integration
class TestEndToEndScenarios:
    """Test complete end-to-end scenarios"""
    
    async def test_new_student_onboarding(self, integration_db):
        """Test complete new student onboarding flow"""
        # 1. Register student
        student = await create_user(
            db=integration_db,
            username="newstudent",
            email="new@example.com",
            password="password123",
            full_name="New Student",
            role=UserRole.STUDENT
        )
        
        # 2. Detect learning style
        learning_style = "visual"  # Would be detected via questionnaire
        student.metadata = {"learning_style": learning_style}
        
        # 3. Take placement quiz
        placement_quiz = Quiz(
            title="Placement Test",
            subject="General",
            created_by_id=student.id,
            difficulty_level=0.5
        )
        integration_db.add(placement_quiz)
        await integration_db.commit()
        
        placement_progress = StudentProgress(
            student_id=student.id,
            quiz_id=placement_quiz.id,
            score=75,
            completed_questions=20,
            total_questions=20,
            is_completed=True
        )
        integration_db.add(placement_progress)
        
        # 4. Generate learning path based on placement
        initial_ability = 0.75  # Based on placement score
        
        # 5. Award onboarding achievement
        student.points = 50  # Welcome bonus
        
        await integration_db.commit()
        await integration_db.refresh(student)
        
        assert student.id is not None
        assert student.points == 50
        assert placement_progress.score == 75
    
    async def test_daily_study_routine(self, integration_db):
        """Test a typical daily study routine"""
        # Get existing student
        student = await create_user(
            db=integration_db,
            username="daily",
            email="daily@example.com",
            password="password",
            full_name="Daily Student",
            role=UserRole.STUDENT
        )
        
        daily_activities = []
        
        # Morning: Review yesterday's material
        review_quiz = Quiz(
            title="Daily Review",
            subject="Mixed",
            created_by_id=student.id
        )
        integration_db.add(review_quiz)
        await integration_db.commit()
        
        review_progress = StudentProgress(
            student_id=student.id,
            quiz_id=review_quiz.id,
            score=85,
            completed_questions=5,
            total_questions=5,
            time_spent=300,
            is_completed=True
        )
        integration_db.add(review_progress)
        daily_activities.append(("review", 85))
        
        # Afternoon: New lesson
        lesson_quiz = Quiz(
            title="New Material",
            subject="Physics",
            created_by_id=student.id
        )
        integration_db.add(lesson_quiz)
        await integration_db.commit()
        
        lesson_progress = StudentProgress(
            student_id=student.id,
            quiz_id=lesson_quiz.id,
            score=70,
            completed_questions=10,
            total_questions=10,
            time_spent=900,
            is_completed=True
        )
        integration_db.add(lesson_progress)
        daily_activities.append(("lesson", 70))
        
        # Evening: Practice problems
        practice_quiz = Quiz(
            title="Practice Problems",
            subject="Physics",
            created_by_id=student.id
        )
        integration_db.add(practice_quiz)
        await integration_db.commit()
        
        practice_progress = StudentProgress(
            student_id=student.id,
            quiz_id=practice_quiz.id,
            score=90,
            completed_questions=15,
            total_questions=15,
            time_spent=1200,
            is_completed=True
        )
        integration_db.add(practice_progress)
        daily_activities.append(("practice", 90))
        
        # Update daily stats
        total_points = sum(score for _, score in daily_activities)
        student.points += total_points
        student.streak_days += 1
        
        await integration_db.commit()
        await integration_db.refresh(student)
        
        assert student.points == 245
        assert student.streak_days == 1
        assert len(daily_activities) == 3
        
        # Calculate daily average
        daily_average = sum(score for _, score in daily_activities) / len(daily_activities)
        assert daily_average > 80  # Good performance


# Import numpy for IRT calculations
import numpy as np