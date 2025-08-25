"""
TEKNOFEST 2025 - Production Ready Test Configuration & Fixtures
Complete test infrastructure for all test types
"""
import pytest
import asyncio
import os
import sys
import json
import tempfile
import shutil
from typing import Generator, AsyncGenerator, Dict, Any, Optional
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch
import logging

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

# FastAPI and async imports
from fastapi import FastAPI
from fastapi.testclient import TestClient
from httpx import AsyncClient
import aiofiles

# Database imports
from sqlalchemy import create_engine, event, text
from sqlalchemy.orm import sessionmaker, Session, scoped_session
from sqlalchemy.pool import StaticPool
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker

# Load test environment
from dotenv import load_dotenv
load_dotenv(".env.test")

# Configure logging for tests
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ==================== Session & Event Loop Fixtures ====================

@pytest.fixture(scope="session")
def event_loop():
    """Create an event loop for the test session"""
    policy = asyncio.get_event_loop_policy()
    loop = policy.new_event_loop()
    asyncio.set_event_loop(loop)
    yield loop
    loop.close()

@pytest.fixture(scope="session")
def anyio_backend():
    """Backend for anyio async tests"""
    return "asyncio"

# ==================== Database Fixtures ====================

@pytest.fixture(scope="session")
def database_url():
    """Get test database URL"""
    return os.getenv("TEST_DATABASE_URL", "sqlite:///:memory:")

@pytest.fixture(scope="session")
def sync_engine(database_url):
    """Create synchronous database engine"""
    if "sqlite" in database_url:
        engine = create_engine(
            database_url,
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
            echo=False
        )
    else:
        engine = create_engine(database_url, echo=False)
    
    # Import and create tables
    try:
        from src.database.models import Base
        Base.metadata.create_all(bind=engine)
    except ImportError:
        logger.warning("Could not import database models")
    
    yield engine
    engine.dispose()

@pytest.fixture(scope="session")
async def async_engine(database_url):
    """Create asynchronous database engine"""
    if "sqlite" in database_url:
        async_url = database_url.replace("sqlite://", "sqlite+aiosqlite://")
        engine = create_async_engine(
            async_url,
            connect_args={"check_same_thread": False},
            echo=False
        )
    else:
        async_url = database_url.replace("postgresql://", "postgresql+asyncpg://")
        engine = create_async_engine(async_url, echo=False)
    
    async with engine.begin() as conn:
        try:
            from src.database.models import Base
            await conn.run_sync(Base.metadata.create_all)
        except ImportError:
            logger.warning("Could not import database models")
    
    yield engine
    await engine.dispose()

@pytest.fixture
def db_session(sync_engine):
    """Create a test database session"""
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=sync_engine)
    session = SessionLocal()
    
    # Begin nested transaction
    connection = sync_engine.connect()
    transaction = connection.begin()
    session.bind = connection
    
    yield session
    
    # Rollback transaction
    session.close()
    transaction.rollback()
    connection.close()

@pytest.fixture
async def async_db_session(async_engine):
    """Create an async test database session"""
    AsyncSessionLocal = async_sessionmaker(
        async_engine,
        class_=AsyncSession,
        expire_on_commit=False
    )
    
    async with AsyncSessionLocal() as session:
        async with session.begin():
            yield session
            await session.rollback()

# ==================== Application Fixtures ====================

@pytest.fixture
def app():
    """Create FastAPI test application"""
    try:
        from src.app import app as main_app
        return main_app
    except ImportError:
        # Create minimal test app if main app doesn't exist
        test_app = FastAPI(title="Test App")
        
        @test_app.get("/health")
        def health_check():
            return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}
        
        @test_app.get("/api/v1/test")
        def test_endpoint():
            return {"message": "test"}
        
        return test_app

@pytest.fixture
def client(app, db_session):
    """Create test client with database override"""
    def override_get_db():
        yield db_session
    
    if hasattr(app, "dependency_overrides"):
        try:
            from src.database.session import get_db
            app.dependency_overrides[get_db] = override_get_db
        except ImportError:
            pass
    
    with TestClient(app) as test_client:
        yield test_client
    
    if hasattr(app, "dependency_overrides"):
        app.dependency_overrides.clear()

@pytest.fixture
async def async_client(app, async_db_session):
    """Create async test client"""
    async def override_get_db():
        yield async_db_session
    
    if hasattr(app, "dependency_overrides"):
        try:
            from src.database.session import get_async_db
            app.dependency_overrides[get_async_db] = override_get_db
        except ImportError:
            pass
    
    async with AsyncClient(app=app, base_url="http://test") as ac:
        yield ac

# ==================== Authentication Fixtures ====================

@pytest.fixture
def test_user():
    """Create test user data"""
    return {
        "id": 1,
        "username": "test_student",
        "email": "student@teknofest.edu.tr",
        "full_name": "Test Öğrenci",
        "grade": 10,
        "learning_style": "visual",
        "role": "student",
        "is_active": True,
        "created_at": datetime.utcnow().isoformat()
    }

@pytest.fixture
def test_teacher():
    """Create test teacher data"""
    return {
        "id": 2,
        "username": "test_teacher",
        "email": "teacher@teknofest.edu.tr",
        "full_name": "Test Öğretmen",
        "subject": "Matematik",
        "role": "teacher",
        "is_active": True
    }

@pytest.fixture
def auth_token(test_user):
    """Generate test authentication token"""
    try:
        from src.core.security import create_access_token
        return create_access_token(data={"sub": test_user["username"]})
    except ImportError:
        return "test-jwt-token-123456789"

@pytest.fixture
def auth_headers(auth_token):
    """Get authorization headers"""
    return {
        "Authorization": f"Bearer {auth_token}",
        "Content-Type": "application/json"
    }

# ==================== Agent Fixtures ====================

@pytest.fixture
def mock_learning_path_agent():
    """Mock Learning Path Agent"""
    agent = Mock()
    agent.create_path = AsyncMock(return_value={
        "id": "path_123",
        "student_id": 1,
        "topics": ["Matematik", "Fizik", "Kimya"],
        "duration": 30,
        "difficulty": 0.5,
        "created_at": datetime.utcnow().isoformat()
    })
    agent.update_progress = AsyncMock(return_value={"status": "updated"})
    agent.get_recommendations = AsyncMock(return_value=[
        {"topic": "Geometri", "reason": "Prerequisite for Trigonometry"},
        {"topic": "Cebir", "reason": "Foundation for advanced math"}
    ])
    return agent

@pytest.fixture
def mock_study_buddy_agent():
    """Mock Study Buddy Agent"""
    agent = Mock()
    agent.answer_question = AsyncMock(return_value={
        "answer": "Test cevabı",
        "confidence": 0.95,
        "sources": ["Textbook page 45", "Online resource"],
        "follow_up_questions": ["Bu konuyu daha detaylı öğrenmek ister misin?"]
    })
    agent.generate_quiz = AsyncMock(return_value={
        "questions": [
            {
                "id": 1,
                "question": "2 + 2 = ?",
                "options": ["3", "4", "5", "6"],
                "correct": 1,
                "explanation": "İki artı iki dört eder"
            }
        ]
    })
    return agent

# ==================== Turkish NLP Fixtures ====================

@pytest.fixture
def turkish_text_samples():
    """Sample Turkish texts for NLP testing"""
    return {
        "simple": "Merhaba dünya",
        "complex": "Türkçe'nin zengin morfolojik yapısı, dil işleme açısından zorluklar yaratır.",
        "with_numbers": "2024 yılında 15 öğrenci başarılı oldu.",
        "educational": "Matematik dersinde integral konusunu işledik.",
        "with_punctuation": "Nasılsın? İyi misin! Umarım her şey yolundadır..."
    }

@pytest.fixture
def mock_nlp_processor():
    """Mock NLP processor"""
    processor = Mock()
    processor.tokenize = Mock(return_value=["test", "token", "list"])
    processor.analyze_morphology = Mock(return_value={
        "root": "test",
        "suffixes": ["-ler", "-de"],
        "pos": "NOUN"
    })
    processor.extract_entities = Mock(return_value=[
        {"text": "Ankara", "type": "LOCATION"},
        {"text": "2024", "type": "DATE"}
    ])
    return processor

# ==================== Model & AI Fixtures ====================

@pytest.fixture
def mock_ai_model():
    """Mock AI model for testing"""
    model = Mock()
    model.generate = AsyncMock(return_value={
        "text": "Generated response",
        "tokens": 50,
        "latency": 0.5
    })
    model.embed = AsyncMock(return_value=[0.1] * 768)
    model.classify = AsyncMock(return_value={
        "label": "positive",
        "confidence": 0.89
    })
    return model

@pytest.fixture
def model_config():
    """Model configuration for testing"""
    return {
        "name": "test-model",
        "version": "1.0.0",
        "max_tokens": 512,
        "temperature": 0.7,
        "device": "cpu",
        "batch_size": 8
    }

# ==================== Redis & Cache Fixtures ====================

@pytest.fixture
async def mock_redis():
    """Mock Redis client"""
    redis_mock = AsyncMock()
    redis_mock.get = AsyncMock(return_value=None)
    redis_mock.set = AsyncMock(return_value=True)
    redis_mock.delete = AsyncMock(return_value=1)
    redis_mock.exists = AsyncMock(return_value=False)
    redis_mock.expire = AsyncMock(return_value=True)
    redis_mock.ttl = AsyncMock(return_value=300)
    return redis_mock

@pytest.fixture
def cache_client(mock_redis):
    """Mock cache client"""
    return mock_redis

# ==================== File & Storage Fixtures ====================

@pytest.fixture
def temp_dir():
    """Create temporary directory for tests"""
    temp_path = tempfile.mkdtemp(prefix="test_teknofest_")
    yield Path(temp_path)
    shutil.rmtree(temp_path, ignore_errors=True)

@pytest.fixture
def sample_file(temp_dir):
    """Create a sample file for testing"""
    file_path = temp_dir / "test_file.txt"
    file_path.write_text("Test content for TEKNOFEST 2025")
    return file_path

@pytest.fixture
async def async_sample_file(temp_dir):
    """Create async sample file"""
    file_path = temp_dir / "async_test_file.txt"
    async with aiofiles.open(file_path, 'w') as f:
        await f.write("Async test content")
    return file_path

# ==================== Request & Response Fixtures ====================

@pytest.fixture
def sample_quiz_request():
    """Sample quiz generation request"""
    return {
        "topic": "Matematik - İntegral",
        "grade": 11,
        "difficulty": 0.6,
        "question_count": 10,
        "question_types": ["multiple_choice", "true_false"],
        "language": "tr",
        "include_explanations": True
    }

@pytest.fixture
def sample_learning_path_request():
    """Sample learning path request"""
    return {
        "student_id": 1,
        "subjects": ["Matematik", "Fizik"],
        "target_exam": "YKS",
        "daily_study_hours": 4,
        "start_date": datetime.utcnow().isoformat(),
        "end_date": (datetime.utcnow() + timedelta(days=180)).isoformat(),
        "preferences": {
            "morning_study": True,
            "break_duration": 15,
            "revision_frequency": "weekly"
        }
    }

# ==================== External API Mocks ====================

@pytest.fixture
def mock_external_apis():
    """Mock all external API calls"""
    with patch('httpx.AsyncClient.get') as mock_get, \
         patch('httpx.AsyncClient.post') as mock_post:
        
        mock_get.return_value = AsyncMock(
            status_code=200,
            json=AsyncMock(return_value={"status": "success", "data": []})
        )
        
        mock_post.return_value = AsyncMock(
            status_code=201,
            json=AsyncMock(return_value={"id": "123", "status": "created"})
        )
        
        yield {"get": mock_get, "post": mock_post}

# ==================== Performance Testing Fixtures ====================

@pytest.fixture
def performance_timer():
    """Timer for performance tests"""
    import time
    
    class Timer:
        def __init__(self):
            self.start_time = None
            self.end_time = None
        
        def start(self):
            self.start_time = time.perf_counter()
        
        def stop(self):
            self.end_time = time.perf_counter()
            return self.end_time - self.start_time
        
        @property
        def elapsed(self):
            if self.start_time and self.end_time:
                return self.end_time - self.start_time
            return None
    
    return Timer()

# ==================== Test Data Factories ====================

@pytest.fixture
def user_factory():
    """Factory for creating test users"""
    def _create_user(**kwargs):
        defaults = {
            "id": 1,
            "username": "testuser",
            "email": "test@example.com",
            "grade": 10,
            "is_active": True
        }
        defaults.update(kwargs)
        return defaults
    return _create_user

@pytest.fixture
def quiz_factory():
    """Factory for creating test quizzes"""
    def _create_quiz(**kwargs):
        defaults = {
            "id": 1,
            "title": "Test Quiz",
            "topic": "Matematik",
            "questions": [],
            "duration": 30,
            "created_by": 1
        }
        defaults.update(kwargs)
        return defaults
    return _create_quiz

# ==================== Cleanup Fixtures ====================

@pytest.fixture(autouse=True)
def cleanup_test_files(temp_dir):
    """Automatically cleanup test files after each test"""
    yield
    # Cleanup happens in temp_dir fixture

@pytest.fixture(autouse=True)
def reset_environment():
    """Reset environment variables after each test"""
    original_env = os.environ.copy()
    yield
    os.environ.clear()
    os.environ.update(original_env)

# ==================== Pytest Hooks ====================

def pytest_configure(config):
    """Configure pytest"""
    # Add custom markers description
    config.addinivalue_line(
        "markers", "benchmark: mark test as benchmark test"
    )
    config.addinivalue_line(
        "markers", "requires_gpu: mark test as requiring GPU"
    )
    config.addinivalue_line(
        "markers", "requires_internet: mark test as requiring internet"
    )

def pytest_collection_modifyitems(config, items):
    """Modify test collection"""
    # Skip GPU tests if no GPU available
    skip_gpu = pytest.mark.skip(reason="GPU not available")
    for item in items:
        if "requires_gpu" in item.keywords:
            try:
                import torch
                if not torch.cuda.is_available():
                    item.add_marker(skip_gpu)
            except ImportError:
                item.add_marker(skip_gpu)

# ==================== Shared Test Utilities ====================

@pytest.fixture
def assert_response():
    """Helper for asserting API responses"""
    def _assert(response, status_code=200, has_data=True):
        assert response.status_code == status_code
        if has_data:
            data = response.json()
            assert data is not None
            return data
        return None
    return _assert

@pytest.fixture
def create_test_data():
    """Create comprehensive test data set"""
    return {
        "users": [user_factory() for _ in range(5)],
        "quizzes": [quiz_factory() for _ in range(3)],
        "learning_paths": [],
        "questions": [],
        "answers": []
    }

# Log test configuration
logger.info("Test configuration loaded successfully")
logger.info(f"Test database: {os.getenv('TEST_DATABASE_URL', 'sqlite:///:memory:')}")
logger.info(f"Test environment: {os.getenv('ENVIRONMENT', 'test')}")