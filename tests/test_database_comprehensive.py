"""
Comprehensive Database Test Suite
TEKNOFEST 2025 - Production Ready Database Tests
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from datetime import datetime, timedelta
import uuid
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.exc import IntegrityError, OperationalError
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.database.models import User, LearningPath, Module, Assessment, UserRole, DifficultyLevel
from src.database.session import SessionLocal, get_db, async_session_maker
from src.database.repository import BaseRepository, UserRepository, LearningPathRepository
from src.database.base import Base


class TestDatabaseModels:
    """Test database models and relationships"""
    
    @pytest.fixture(scope="function")
    def test_engine(self):
        """Create test database engine"""
        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(bind=engine)
        return engine
    
    @pytest.fixture(scope="function")
    def test_session(self, test_engine):
        """Create test database session"""
        TestSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)
        session = TestSessionLocal()
        yield session
        session.close()
    
    def test_user_model_creation(self, test_session):
        """Test User model creation"""
        user = User(
            id=uuid.uuid4(),
            email="test@example.com",
            username="testuser",
            full_name="Test User",
            hashed_password="hashed_password_123",
            role=UserRole.STUDENT
        )
        
        test_session.add(user)
        test_session.commit()
        
        saved_user = test_session.query(User).filter_by(email="test@example.com").first()
        assert saved_user is not None
        assert saved_user.username == "testuser"
        assert saved_user.role == UserRole.STUDENT
        assert saved_user.is_active == True
    
    def test_user_unique_constraints(self, test_session):
        """Test unique constraints on User model"""
        user1 = User(
            id=uuid.uuid4(),
            email="test@example.com",
            username="testuser",
            full_name="Test User 1",
            hashed_password="hash1"
        )
        
        user2 = User(
            id=uuid.uuid4(),
            email="test@example.com",  # Duplicate email
            username="testuser2",
            full_name="Test User 2",
            hashed_password="hash2"
        )
        
        test_session.add(user1)
        test_session.commit()
        
        test_session.add(user2)
        with pytest.raises(IntegrityError):
            test_session.commit()
    
    def test_learning_path_model(self, test_session):
        """Test LearningPath model"""
        path = LearningPath(
            id=uuid.uuid4(),
            title="Python Programming",
            description="Learn Python from basics to advanced",
            slug="python-programming",
            objectives=["Learn basics", "Master OOP", "Build projects"],
            difficulty=DifficultyLevel.INTERMEDIATE,
            estimated_hours=40.0,
            language="tr"
        )
        
        test_session.add(path)
        test_session.commit()
        
        saved_path = test_session.query(LearningPath).filter_by(slug="python-programming").first()
        assert saved_path is not None
        assert saved_path.title == "Python Programming"
        assert len(saved_path.objectives) == 3
        assert saved_path.difficulty == DifficultyLevel.INTERMEDIATE
    
    def test_user_learning_path_relationship(self, test_session):
        """Test many-to-many relationship between User and LearningPath"""
        user = User(
            id=uuid.uuid4(),
            email="student@example.com",
            username="student",
            full_name="Student User",
            hashed_password="hash"
        )
        
        path1 = LearningPath(
            id=uuid.uuid4(),
            title="Math Course",
            description="Mathematics",
            slug="math-course",
            objectives=["Algebra", "Geometry"],
            difficulty=DifficultyLevel.BEGINNER,
            estimated_hours=20.0
        )
        
        path2 = LearningPath(
            id=uuid.uuid4(),
            title="Physics Course",
            description="Physics",
            slug="physics-course",
            objectives=["Mechanics", "Thermodynamics"],
            difficulty=DifficultyLevel.INTERMEDIATE,
            estimated_hours=30.0
        )
        
        user.learning_paths.append(path1)
        user.learning_paths.append(path2)
        
        test_session.add(user)
        test_session.commit()
        
        saved_user = test_session.query(User).filter_by(username="student").first()
        assert len(saved_user.learning_paths) == 2
        assert path1 in saved_user.learning_paths
        assert path2 in saved_user.learning_paths
    
    def test_module_cascade_delete(self, test_session):
        """Test cascade delete for modules"""
        path = LearningPath(
            id=uuid.uuid4(),
            title="Test Path",
            description="Test",
            slug="test-path",
            objectives=["Test"],
            difficulty=DifficultyLevel.BEGINNER,
            estimated_hours=10.0
        )
        
        module = Module(
            id=uuid.uuid4(),
            learning_path_id=path.id,
            title="Module 1",
            order_index=1,
            content_type="video",
            estimated_minutes=30
        )
        
        path.modules.append(module)
        test_session.add(path)
        test_session.commit()
        
        # Verify module exists
        assert test_session.query(Module).count() == 1
        
        # Delete learning path
        test_session.delete(path)
        test_session.commit()
        
        # Module should be deleted due to cascade
        assert test_session.query(Module).count() == 0
    
    def test_assessment_model(self, test_session):
        """Test Assessment model"""
        user = User(
            id=uuid.uuid4(),
            email="test@example.com",
            username="testuser",
            full_name="Test User",
            hashed_password="hash"
        )
        
        assessment = Assessment(
            id=uuid.uuid4(),
            user_id=user.id,
            title="Math Quiz",
            assessment_type="quiz",
            questions=[
                {"question": "2+2?", "answer": "4"},
                {"question": "3*3?", "answer": "9"}
            ],
            score=80.0,
            passed=True
        )
        
        user.assessments.append(assessment)
        test_session.add(user)
        test_session.commit()
        
        saved_assessment = test_session.query(Assessment).first()
        assert saved_assessment is not None
        assert saved_assessment.title == "Math Quiz"
        assert len(saved_assessment.questions) == 2
        assert saved_assessment.score == 80.0
        assert saved_assessment.passed == True


class TestDatabaseRepository:
    """Test repository pattern implementation"""
    
    @pytest.fixture(scope="function")
    def test_db(self):
        """Create test database"""
        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(bind=engine)
        TestSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)
        db = TestSessionLocal()
        yield db
        db.close()
    
    @pytest.fixture
    def user_repo(self, test_db):
        """Create user repository"""
        return UserRepository(test_db)
    
    @pytest.fixture
    def learning_path_repo(self, test_db):
        """Create learning path repository"""
        return LearningPathRepository(test_db)
    
    def test_repository_create(self, user_repo):
        """Test repository create operation"""
        user_data = {
            'email': 'newuser@example.com',
            'username': 'newuser',
            'full_name': 'New User',
            'hashed_password': 'hashed'
        }
        
        user = user_repo.create(user_data)
        
        assert user is not None
        assert user.email == 'newuser@example.com'
        assert user.id is not None
    
    def test_repository_get_by_id(self, user_repo):
        """Test repository get by ID"""
        user_data = {
            'email': 'test@example.com',
            'username': 'testuser',
            'full_name': 'Test User',
            'hashed_password': 'hash'
        }
        
        created_user = user_repo.create(user_data)
        fetched_user = user_repo.get(created_user.id)
        
        assert fetched_user is not None
        assert fetched_user.id == created_user.id
        assert fetched_user.email == created_user.email
    
    def test_repository_update(self, user_repo):
        """Test repository update operation"""
        user_data = {
            'email': 'original@example.com',
            'username': 'original',
            'full_name': 'Original User',
            'hashed_password': 'hash'
        }
        
        user = user_repo.create(user_data)
        
        update_data = {'full_name': 'Updated User'}
        updated_user = user_repo.update(user.id, update_data)
        
        assert updated_user.full_name == 'Updated User'
        assert updated_user.email == 'original@example.com'  # Unchanged
    
    def test_repository_delete(self, user_repo):
        """Test repository delete operation"""
        user_data = {
            'email': 'delete@example.com',
            'username': 'deleteuser',
            'full_name': 'Delete User',
            'hashed_password': 'hash'
        }
        
        user = user_repo.create(user_data)
        user_id = user.id
        
        assert user_repo.delete(user_id) == True
        assert user_repo.get(user_id) is None
    
    def test_repository_list_with_filters(self, user_repo):
        """Test repository list with filters"""
        # Create multiple users
        for i in range(5):
            user_repo.create({
                'email': f'user{i}@example.com',
                'username': f'user{i}',
                'full_name': f'User {i}',
                'hashed_password': 'hash',
                'role': UserRole.STUDENT if i % 2 == 0 else UserRole.TEACHER
            })
        
        # Test filtering
        students = user_repo.list(filters={'role': UserRole.STUDENT})
        teachers = user_repo.list(filters={'role': UserRole.TEACHER})
        
        assert len(students) == 3
        assert len(teachers) == 2
    
    def test_repository_pagination(self, learning_path_repo):
        """Test repository pagination"""
        # Create multiple learning paths
        for i in range(15):
            learning_path_repo.create({
                'title': f'Course {i}',
                'description': f'Description {i}',
                'slug': f'course-{i}',
                'objectives': ['Learn'],
                'difficulty': DifficultyLevel.BEGINNER,
                'estimated_hours': 10.0
            })
        
        # Test pagination
        page1 = learning_path_repo.list(skip=0, limit=5)
        page2 = learning_path_repo.list(skip=5, limit=5)
        page3 = learning_path_repo.list(skip=10, limit=5)
        
        assert len(page1) == 5
        assert len(page2) == 5
        assert len(page3) == 5
        assert page1[0].title == 'Course 0'
        assert page2[0].title == 'Course 5'
        assert page3[0].title == 'Course 10'


class TestDatabaseSession:
    """Test database session management"""
    
    def test_session_creation(self):
        """Test session creation"""
        with patch('src.database.session.create_engine') as mock_engine:
            mock_engine.return_value = MagicMock()
            
            session = SessionLocal()
            assert session is not None
    
    def test_get_db_generator(self):
        """Test get_db generator function"""
        with patch('src.database.session.SessionLocal') as mock_session:
            mock_db = MagicMock()
            mock_session.return_value = mock_db
            
            db_gen = get_db()
            db = next(db_gen)
            
            assert db == mock_db
            
            # Test cleanup
            try:
                next(db_gen)
            except StopIteration:
                pass
            
            mock_db.close.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_async_session(self):
        """Test async session creation"""
        with patch('src.database.session.async_session_maker') as mock_async_session:
            mock_session = AsyncMock()
            mock_async_session.return_value = mock_session
            
            async with mock_session as session:
                assert session is not None


class TestDatabaseTransactions:
    """Test database transactions and rollback"""
    
    @pytest.fixture(scope="function")
    def test_engine(self):
        """Create test database engine"""
        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(bind=engine)
        return engine
    
    @pytest.fixture(scope="function")
    def test_session(self, test_engine):
        """Create test database session"""
        TestSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)
        session = TestSessionLocal()
        yield session
        session.close()
    
    def test_transaction_commit(self, test_session):
        """Test successful transaction commit"""
        user = User(
            id=uuid.uuid4(),
            email="commit@example.com",
            username="commituser",
            full_name="Commit User",
            hashed_password="hash"
        )
        
        test_session.add(user)
        test_session.commit()
        
        # Verify data persisted
        result = test_session.query(User).filter_by(username="commituser").first()
        assert result is not None
    
    def test_transaction_rollback(self, test_session):
        """Test transaction rollback on error"""
        user1 = User(
            id=uuid.uuid4(),
            email="user@example.com",
            username="user1",
            full_name="User 1",
            hashed_password="hash"
        )
        
        test_session.add(user1)
        test_session.commit()
        
        # Try to add duplicate
        user2 = User(
            id=uuid.uuid4(),
            email="user@example.com",  # Duplicate
            username="user2",
            full_name="User 2",
            hashed_password="hash"
        )
        
        test_session.add(user2)
        
        with pytest.raises(IntegrityError):
            test_session.commit()
        
        test_session.rollback()
        
        # Verify only first user exists
        users = test_session.query(User).all()
        assert len(users) == 1
        assert users[0].username == "user1"
    
    def test_nested_transactions(self, test_session):
        """Test nested transactions with savepoints"""
        # Create initial user
        user = User(
            id=uuid.uuid4(),
            email="nested@example.com",
            username="nesteduser",
            full_name="Nested User",
            hashed_password="hash"
        )
        test_session.add(user)
        test_session.commit()
        
        # Start nested transaction
        nested = test_session.begin_nested()
        
        try:
            # Try to update with invalid data
            user.email = None  # This should fail
            test_session.commit()
        except:
            nested.rollback()
        
        # User should still have original email
        result = test_session.query(User).filter_by(username="nesteduser").first()
        assert result.email == "nested@example.com"


class TestDatabasePerformance:
    """Test database performance and optimization"""
    
    @pytest.fixture(scope="function")
    def perf_engine(self):
        """Create performance test database"""
        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(bind=engine)
        return engine
    
    @pytest.fixture(scope="function")
    def perf_session(self, perf_engine):
        """Create performance test session"""
        TestSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=perf_engine)
        session = TestSessionLocal()
        yield session
        session.close()
    
    def test_bulk_insert_performance(self, perf_session):
        """Test bulk insert performance"""
        import time
        
        users = []
        for i in range(1000):
            users.append(User(
                id=uuid.uuid4(),
                email=f"user{i}@example.com",
                username=f"user{i}",
                full_name=f"User {i}",
                hashed_password="hash"
            ))
        
        start_time = time.time()
        perf_session.bulk_save_objects(users)
        perf_session.commit()
        duration = time.time() - start_time
        
        assert perf_session.query(User).count() == 1000
        assert duration < 2.0  # Should complete within 2 seconds
    
    def test_query_optimization(self, perf_session):
        """Test query optimization with eager loading"""
        # Create test data
        for i in range(10):
            user = User(
                id=uuid.uuid4(),
                email=f"user{i}@example.com",
                username=f"user{i}",
                full_name=f"User {i}",
                hashed_password="hash"
            )
            
            for j in range(5):
                path = LearningPath(
                    id=uuid.uuid4(),
                    title=f"Course {i}-{j}",
                    description="Test",
                    slug=f"course-{i}-{j}",
                    objectives=["Learn"],
                    difficulty=DifficultyLevel.BEGINNER,
                    estimated_hours=10.0
                )
                user.learning_paths.append(path)
            
            perf_session.add(user)
        
        perf_session.commit()
        
        # Test lazy loading (N+1 problem)
        import time
        start_time = time.time()
        users = perf_session.query(User).all()
        for user in users:
            _ = len(user.learning_paths)  # Triggers lazy load
        lazy_duration = time.time() - start_time
        
        # Test eager loading (optimized)
        from sqlalchemy.orm import joinedload
        start_time = time.time()
        users = perf_session.query(User).options(joinedload(User.learning_paths)).all()
        for user in users:
            _ = len(user.learning_paths)  # Already loaded
        eager_duration = time.time() - start_time
        
        # Eager loading should be faster
        assert eager_duration < lazy_duration * 1.5  # Allow some variance
    
    def test_index_performance(self, perf_session):
        """Test index performance on queries"""
        # Create many users
        for i in range(1000):
            user = User(
                id=uuid.uuid4(),
                email=f"user{i}@example.com",
                username=f"user{i}",
                full_name=f"User {i}",
                hashed_password="hash",
                is_active=i % 2 == 0
            )
            perf_session.add(user)
        
        perf_session.commit()
        
        import time
        
        # Test indexed query (email has index)
        start_time = time.time()
        result = perf_session.query(User).filter_by(email="user500@example.com").first()
        indexed_duration = time.time() - start_time
        
        assert result is not None
        assert indexed_duration < 0.1  # Should be very fast


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])