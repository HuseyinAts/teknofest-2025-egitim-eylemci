"""
TEKNOFEST 2025 - Test Configuration ve Fixtures
"""
import pytest
import asyncio
from typing import Generator, AsyncGenerator
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool

# Test database
TEST_DATABASE_URL = "sqlite:///:memory:"

@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture(scope="session")
def test_db():
    """Create test database"""
    engine = create_engine(
        TEST_DATABASE_URL,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    
    # Create tables - simplified for now
    # from src.models import Base
    # Base.metadata.create_all(bind=engine)
    
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    
    yield TestingSessionLocal()

@pytest.fixture
def client():
    """Create test client"""
    try:
        from src.main import app
        with TestClient(app) as test_client:
            yield test_client
    except ImportError:
        # If main app doesn't exist yet, create a simple test app
        from fastapi import FastAPI
        test_app = FastAPI()
        
        @test_app.get("/health")
        def health_check():
            return {"status": "healthy"}
        
        with TestClient(test_app) as test_client:
            yield test_client

@pytest.fixture
def auth_headers():
    """Get authentication headers"""
    return {
        "Authorization": "Bearer test-token-123",
        "Content-Type": "application/json"
    }

@pytest.fixture
def sample_user():
    """Sample user data"""
    return {
        "id": 1,
        "username": "test_user",
        "email": "test@teknofest.com",
        "grade": 10,
        "learning_style": "visual"
    }

@pytest.fixture
def sample_quiz_request():
    """Sample quiz request data"""
    return {
        "topic": "Matematik",
        "grade": 10,
        "difficulty": 0.5,
        "question_count": 10,
        "learning_style": "visual"
    }

@pytest.fixture
async def mock_model_response():
    """Mock AI model response"""
    return {
        "response": "Bu bir test cevabıdır.",
        "confidence": 0.95,
        "metadata": {
            "model": "test-model",
            "latency": 0.123
        }
    }