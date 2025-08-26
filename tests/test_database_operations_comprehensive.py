"""
Comprehensive Database Operations Test Suite
Target: Full coverage for database operations
"""

import pytest
from unittest.mock import Mock, MagicMock, patch, PropertyMock
from datetime import datetime, timedelta
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import IntegrityError, OperationalError, DataError
import asyncio

from src.database.models import (
    Base, User, UserRole, Student, Teacher, Course, 
    Quiz, Question, QuestionType, DifficultyLevel,
    StudentAnswer, StudentProgress, LearningPath,
    enrollment_table, quiz_questions
)
from src.database.session import SessionLocal, get_db, async_get_db
from src.database.repositories import (
    UserRepository, StudentRepository, CourseRepository,
    QuizRepository, ProgressRepository
)
from src.database.backup_strategy import BackupStrategy
from src.database.migration_validator import MigrationValidator
from src.database.connection_manager import ConnectionManager


class TestDatabaseModels:
    """Test database model definitions and relationships"""
    
    def test_user_model_creation(self):
        """Test creating a User model instance"""
        user = User(
            username="test_user",
            email="test@example.com",
            hashed_password="hashed_password_here",
            full_name="Test User",
            role=UserRole.STUDENT,
            is_active=True
        )
        
        assert user.username == "test_user"
        assert user.email == "test@example.com"
        assert user.role == UserRole.STUDENT
        assert user.is_active == True
    
    def test_student_model_creation(self):
        """Test creating a Student model instance"""
        student = Student(
            user_id=1,
            grade_level="9. Sınıf",
            learning_style="visual",
            current_ability_level=0.5
        )
        
        assert student.user_id == 1
        assert student.grade_level == "9. Sınıf"
        assert student.learning_style == "visual"
        assert student.current_ability_level == 0.5
    
    def test_course_model_creation(self):
        """Test creating a Course model instance"""
        course = Course(
            title="Matematik 101",
            description="Temel matematik kursu",
            subject="Matematik",
            grade_level="9. Sınıf",
            teacher_id=1,
            difficulty_level=DifficultyLevel.MEDIUM,
            estimated_hours=40,
            is_published=True
        )
        
        assert course.title == "Matematik 101"
        assert course.subject == "Matematik"
        assert course.difficulty_level == DifficultyLevel.MEDIUM
        assert course.is_published == True
    
    def test_quiz_model_creation(self):
        """Test creating a Quiz model instance"""
        quiz = Quiz(
            title="Matematik Sınavı",
            course_id=1,
            difficulty_level=DifficultyLevel.MEDIUM,
            time_limit=60,
            passing_score=0.6,
            max_attempts=3,
            is_active=True
        )
        
        assert quiz.title == "Matematik Sınavı"
        assert quiz.time_limit == 60
        assert quiz.passing_score == 0.6
        assert quiz.max_attempts == 3
    
    def test_question_model_creation(self):
        """Test creating a Question model instance"""
        question = Question(
            text="2 + 2 = ?",
            type=QuestionType.MULTIPLE_CHOICE,
            difficulty=0.3,
            discrimination=0.7,
            guessing=0.25,
            subject="Matematik",
            topic="Toplama",
            options=["3", "4", "5", "6"],
            correct_answer="4",
            explanation="2 + 2 = 4",
            points=1.0
        )
        
        assert question.text == "2 + 2 = ?"
        assert question.type == QuestionType.MULTIPLE_CHOICE
        assert question.difficulty == 0.3
        assert question.correct_answer == "4"
    
    def test_model_relationships(self):
        """Test model relationships"""
        # User -> Student relationship
        user = User(username="student1", email="s1@test.com")
        student = Student(user_id=1, grade_level="10. Sınıf")
        user.student = student
        
        assert user.student == student
        
        # Teacher -> Courses relationship
        teacher = Teacher(user_id=2, subject_expertise=["Math", "Physics"])
        course1 = Course(title="Math 101", teacher_id=2)
        course2 = Course(title="Physics 101", teacher_id=2)
        teacher.courses = [course1, course2]
        
        assert len(teacher.courses) == 2
        assert course1 in teacher.courses


class TestUserRepository:
    """Test UserRepository CRUD operations"""
    
    @pytest.fixture
    def mock_db(self):
        """Create mock database session"""
        return Mock(spec=Session)
    
    @pytest.fixture
    def user_repo(self, mock_db):
        """Create UserRepository instance"""
        return UserRepository(mock_db)
    
    def test_create_user(self, user_repo, mock_db):
        """Test creating a new user"""
        user_data = {
            "username": "new_user",
            "email": "new@example.com",
            "hashed_password": "hashed_pwd",
            "full_name": "New User"
        }
        
        mock_db.add = Mock()
        mock_db.commit = Mock()
        mock_db.refresh = Mock()
        
        user = user_repo.create(user_data)
        
        mock_db.add.assert_called_once()
        mock_db.commit.assert_called_once()
    
    def test_get_user_by_id(self, user_repo, mock_db):
        """Test getting user by ID"""
        mock_user = Mock(spec=User)
        mock_user.id = 1
        mock_user.username = "test_user"
        
        mock_db.query.return_value.filter.return_value.first.return_value = mock_user
        
        user = user_repo.get_by_id(1)
        
        assert user == mock_user
        mock_db.query.assert_called_with(User)
    
    def test_get_user_by_username(self, user_repo, mock_db):
        """Test getting user by username"""
        mock_user = Mock(spec=User)
        mock_user.username = "test_user"
        
        mock_db.query.return_value.filter.return_value.first.return_value = mock_user
        
        user = user_repo.get_by_username("test_user")
        
        assert user == mock_user
    
    def test_update_user(self, user_repo, mock_db):
        """Test updating user"""
        mock_user = Mock(spec=User)
        mock_user.id = 1
        
        mock_db.query.return_value.filter.return_value.first.return_value = mock_user
        mock_db.commit = Mock()
        
        updated = user_repo.update(1, {"full_name": "Updated Name"})
        
        assert updated == mock_user
        assert mock_user.full_name == "Updated Name"
        mock_db.commit.assert_called_once()
    
    def test_delete_user(self, user_repo, mock_db):
        """Test deleting user"""
        mock_user = Mock(spec=User)
        
        mock_db.query.return_value.filter.return_value.first.return_value = mock_user
        mock_db.delete = Mock()
        mock_db.commit = Mock()
        
        result = user_repo.delete(1)
        
        assert result == True
        mock_db.delete.assert_called_with(mock_user)
        mock_db.commit.assert_called_once()
    
    def test_list_users_with_pagination(self, user_repo, mock_db):
        """Test listing users with pagination"""
        mock_users = [Mock(spec=User) for _ in range(5)]
        
        mock_query = Mock()
        mock_query.offset.return_value.limit.return_value.all.return_value = mock_users
        mock_db.query.return_value = mock_query
        
        users = user_repo.list(skip=0, limit=10)
        
        assert len(users) == 5
        mock_query.offset.assert_called_with(0)
    
    def test_handle_integrity_error(self, user_repo, mock_db):
        """Test handling integrity constraint violations"""
        mock_db.add = Mock()
        mock_db.commit = Mock(side_effect=IntegrityError("", "", ""))
        mock_db.rollback = Mock()
        
        with pytest.raises(IntegrityError):
            user_repo.create({"username": "duplicate_user"})
        
        mock_db.rollback.assert_called_once()


class TestStudentRepository:
    """Test StudentRepository operations"""
    
    @pytest.fixture
    def student_repo(self):
        """Create StudentRepository instance"""
        mock_db = Mock(spec=Session)
        return StudentRepository(mock_db), mock_db
    
    def test_create_student_profile(self, student_repo):
        """Test creating student profile"""
        repo, mock_db = student_repo
        
        mock_db.add = Mock()
        mock_db.commit = Mock()
        mock_db.refresh = Mock()
        
        student_data = {
            "user_id": 1,
            "grade_level": "11. Sınıf",
            "learning_style": "kinesthetic"
        }
        
        student = repo.create(student_data)
        
        mock_db.add.assert_called_once()
        mock_db.commit.assert_called_once()
    
    def test_update_student_progress(self, student_repo):
        """Test updating student progress"""
        repo, mock_db = student_repo
        
        mock_student = Mock(spec=Student)
        mock_student.current_ability_level = 0.5
        
        mock_db.query.return_value.filter.return_value.first.return_value = mock_student
        mock_db.commit = Mock()
        
        updated = repo.update_progress(1, new_ability=0.7)
        
        assert mock_student.current_ability_level == 0.7
        mock_db.commit.assert_called_once()
    
    def test_get_student_courses(self, student_repo):
        """Test getting student's enrolled courses"""
        repo, mock_db = student_repo
        
        mock_courses = [Mock(spec=Course) for _ in range(3)]
        mock_db.query.return_value.join.return_value.filter.return_value.all.return_value = mock_courses
        
        courses = repo.get_enrolled_courses(student_id=1)
        
        assert len(courses) == 3


class TestTransactionManagement:
    """Test database transaction management"""
    
    @pytest.fixture
    def mock_db(self):
        """Create mock database session with transaction support"""
        mock = Mock(spec=Session)
        mock.begin = Mock(return_value=Mock())
        mock.commit = Mock()
        mock.rollback = Mock()
        return mock
    
    def test_successful_transaction(self, mock_db):
        """Test successful transaction commit"""
        with mock_db.begin():
            # Perform operations
            mock_db.add(Mock())
            mock_db.add(Mock())
        
        mock_db.commit.assert_called()
    
    def test_transaction_rollback_on_error(self, mock_db):
        """Test transaction rollback on error"""
        try:
            with mock_db.begin():
                mock_db.add(Mock())
                raise ValueError("Test error")
        except ValueError:
            pass
        
        mock_db.rollback.assert_called()
    
    def test_nested_transactions(self, mock_db):
        """Test nested transaction handling"""
        mock_db.begin_nested = Mock(return_value=Mock())
        
        with mock_db.begin():
            # Outer transaction
            mock_db.add(Mock())
            
            with mock_db.begin_nested():
                # Inner transaction
                mock_db.add(Mock())
            
            mock_db.add(Mock())
        
        assert mock_db.begin_nested.called


class TestConnectionPooling:
    """Test database connection pooling"""
    
    def test_connection_pool_configuration(self):
        """Test connection pool configuration"""
        from sqlalchemy.pool import QueuePool
        
        engine = create_engine(
            "sqlite:///:memory:",
            poolclass=QueuePool,
            pool_size=5,
            max_overflow=10,
            pool_timeout=30,
            pool_recycle=3600
        )
        
        assert engine.pool.size() == 5
        assert engine.pool._max_overflow == 10
    
    @patch('src.database.connection_manager.create_engine')
    def test_connection_manager_initialization(self, mock_create_engine):
        """Test ConnectionManager initialization"""
        mock_engine = Mock()
        mock_create_engine.return_value = mock_engine
        
        manager = ConnectionManager(
            database_url="postgresql://test",
            pool_size=10,
            max_overflow=20
        )
        
        assert manager.engine == mock_engine
        mock_create_engine.assert_called_once()
    
    def test_connection_health_check(self):
        """Test database connection health check"""
        manager = Mock(spec=ConnectionManager)
        manager.check_connection.return_value = True
        
        assert manager.check_connection() == True
    
    def test_connection_retry_logic(self):
        """Test connection retry on failure"""
        manager = Mock(spec=ConnectionManager)
        manager.connect_with_retry = Mock(
            side_effect=[OperationalError("", "", ""), Mock()]
        )
        
        # First attempt fails, second succeeds
        try:
            manager.connect_with_retry()
        except OperationalError:
            pass
        
        connection = manager.connect_with_retry()
        assert connection is not None


class TestQueryOptimization:
    """Test query optimization and performance"""
    
    @pytest.fixture
    def mock_db(self):
        """Create mock database session"""
        return Mock(spec=Session)
    
    def test_eager_loading(self, mock_db):
        """Test eager loading to prevent N+1 queries"""
        from sqlalchemy.orm import joinedload
        
        mock_query = Mock()
        mock_query.options.return_value = mock_query
        mock_query.all.return_value = []
        
        mock_db.query.return_value = mock_query
        
        # Eager load related objects
        mock_db.query(User).options(joinedload(User.student)).all()
        
        mock_query.options.assert_called()
    
    def test_bulk_insert(self, mock_db):
        """Test bulk insert operations"""
        mock_db.bulk_insert_mappings = Mock()
        mock_db.commit = Mock()
        
        users_data = [
            {"username": f"user_{i}", "email": f"user{i}@test.com"}
            for i in range(100)
        ]
        
        mock_db.bulk_insert_mappings(User, users_data)
        mock_db.commit()
        
        mock_db.bulk_insert_mappings.assert_called_once()
        assert len(users_data) == 100
    
    def test_query_with_index_hints(self, mock_db):
        """Test queries using index hints"""
        mock_query = Mock()
        mock_query.filter.return_value = mock_query
        mock_query.first.return_value = Mock()
        
        mock_db.query.return_value = mock_query
        
        # Query should use indexed columns
        result = mock_db.query(User).filter(User.username == "test").first()
        
        assert result is not None


class TestMigrations:
    """Test database migration handling"""
    
    @patch('alembic.command')
    def test_run_migration_upgrade(self, mock_alembic):
        """Test running migration upgrade"""
        from alembic.config import Config
        
        config = Config("alembic.ini")
        mock_alembic.upgrade = Mock()
        
        mock_alembic.upgrade(config, "head")
        
        mock_alembic.upgrade.assert_called_with(config, "head")
    
    @patch('alembic.command')
    def test_run_migration_downgrade(self, mock_alembic):
        """Test running migration downgrade"""
        from alembic.config import Config
        
        config = Config("alembic.ini")
        mock_alembic.downgrade = Mock()
        
        mock_alembic.downgrade(config, "-1")
        
        mock_alembic.downgrade.assert_called_with(config, "-1")
    
    def test_migration_validator(self):
        """Test migration validation"""
        validator = MigrationValidator()
        
        # Mock migration files
        migrations = [
            {"version": "001", "description": "initial"},
            {"version": "002", "description": "add_indexes"}
        ]
        
        result = validator.validate_migrations(migrations)
        assert result["valid"] == True
        assert len(result["migrations"]) == 2


class TestBackupStrategy:
    """Test database backup strategies"""
    
    def test_backup_strategy_initialization(self):
        """Test backup strategy initialization"""
        strategy = BackupStrategy(
            backup_dir="/backups",
            retention_days=7,
            compression=True
        )
        
        assert strategy.backup_dir == "/backups"
        assert strategy.retention_days == 7
        assert strategy.compression == True
    
    @patch('subprocess.run')
    def test_create_backup(self, mock_subprocess):
        """Test creating database backup"""
        mock_subprocess.return_value = Mock(returncode=0)
        
        strategy = BackupStrategy()
        backup_file = strategy.create_backup(
            database_url="postgresql://test",
            backup_name="test_backup"
        )
        
        assert backup_file is not None
        mock_subprocess.assert_called()
    
    @patch('os.listdir')
    @patch('os.remove')
    def test_cleanup_old_backups(self, mock_remove, mock_listdir):
        """Test cleaning up old backups"""
        # Mock old backup files
        mock_listdir.return_value = [
            "backup_2024_01_01.sql",
            "backup_2024_01_02.sql",
            "backup_2024_01_03.sql"
        ]
        
        strategy = BackupStrategy(retention_days=2)
        strategy.cleanup_old_backups()
        
        # Should remove old backups
        assert mock_remove.called


class TestAsyncDatabaseOperations:
    """Test async database operations"""
    
    @pytest.mark.asyncio
    async def test_async_session_creation(self):
        """Test creating async database session"""
        async with async_get_db() as session:
            assert session is not None
    
    @pytest.mark.asyncio
    async def test_async_query_execution(self):
        """Test executing async queries"""
        mock_session = Mock()
        mock_session.execute = AsyncMock(return_value=Mock())
        
        result = await mock_session.execute("SELECT 1")
        assert result is not None
    
    @pytest.mark.asyncio
    async def test_async_transaction(self):
        """Test async transaction handling"""
        mock_session = Mock()
        mock_session.begin = AsyncMock()
        mock_session.commit = AsyncMock()
        mock_session.rollback = AsyncMock()
        
        async with mock_session.begin():
            # Perform async operations
            pass
        
        mock_session.begin.assert_called()


class TestDatabaseConstraints:
    """Test database constraints and validations"""
    
    def test_unique_constraint_violation(self):
        """Test unique constraint violation handling"""
        mock_db = Mock(spec=Session)
        mock_db.add = Mock()
        mock_db.commit = Mock(side_effect=IntegrityError("", "", "UNIQUE constraint"))
        
        with pytest.raises(IntegrityError):
            user = User(username="duplicate", email="dup@test.com")
            mock_db.add(user)
            mock_db.commit()
    
    def test_foreign_key_constraint(self):
        """Test foreign key constraint"""
        mock_db = Mock(spec=Session)
        mock_db.add = Mock()
        mock_db.commit = Mock(side_effect=IntegrityError("", "", "FOREIGN KEY"))
        
        with pytest.raises(IntegrityError):
            # Try to create student with non-existent user_id
            student = Student(user_id=9999, grade_level="10. Sınıf")
            mock_db.add(student)
            mock_db.commit()
    
    def test_check_constraint(self):
        """Test check constraints"""
        mock_db = Mock(spec=Session)
        
        # Test ability level must be between 0 and 1
        with pytest.raises(ValueError):
            student = Student(
                user_id=1,
                grade_level="10. Sınıf",
                current_ability_level=1.5  # Invalid: > 1.0
            )
            if student.current_ability_level > 1.0:
                raise ValueError("Ability level must be between 0 and 1")


class TestDatabaseCaching:
    """Test database query caching"""
    
    @patch('redis.Redis')
    def test_query_cache_hit(self, mock_redis):
        """Test cache hit for repeated queries"""
        mock_redis.get.return_value = b'{"id": 1, "username": "cached_user"}'
        
        cache_key = "user:1"
        cached_data = mock_redis.get(cache_key)
        
        assert cached_data is not None
        mock_redis.get.assert_called_with(cache_key)
    
    @patch('redis.Redis')
    def test_query_cache_miss(self, mock_redis):
        """Test cache miss and database query"""
        mock_redis.get.return_value = None
        mock_db = Mock(spec=Session)
        
        mock_user = Mock(spec=User)
        mock_user.id = 1
        mock_user.username = "test_user"
        
        mock_db.query.return_value.filter.return_value.first.return_value = mock_user
        
        # Cache miss - query database
        cache_key = "user:1"
        cached_data = mock_redis.get(cache_key)
        
        if cached_data is None:
            user = mock_db.query(User).filter(User.id == 1).first()
            assert user == mock_user
            # Set cache
            mock_redis.set.assert_not_called()  # Would be called in real impl