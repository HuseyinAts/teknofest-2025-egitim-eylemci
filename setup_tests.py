"""
TEKNOFEST 2025 - Test Altyapısı Kurulum ve Örnek Testler
Bu script test altyapısını kurar ve örnek testler oluşturur
"""

import os
import sys
import subprocess
from pathlib import Path

def create_test_structure():
    """Test klasör yapısını oluştur"""
    
    # Test directories
    test_dirs = [
        "tests",
        "tests/unit",
        "tests/integration",
        "tests/e2e",
        "tests/fixtures",
        "tests/mocks"
    ]
    
    for dir_path in test_dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created: {dir_path}")
    
    # Create __init__.py files
    for dir_path in test_dirs:
        init_file = Path(dir_path) / "__init__.py"
        init_file.touch()
    
    print("\n📁 Test klasör yapısı oluşturuldu!")

def create_pytest_config():
    """pytest.ini yapılandırması"""
    
    pytest_ini = """[pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = 
    -v
    --strict-markers
    --tb=short
    --cov=src
    --cov-report=term-missing
    --cov-report=html
    --cov-report=xml
    --cov-fail-under=80
markers =
    unit: Unit tests
    integration: Integration tests
    e2e: End-to-end tests
    slow: Slow tests
    model: Model related tests
    api: API endpoint tests
asyncio_mode = auto
"""
    
    with open("pytest.ini", "w") as f:
        f.write(pytest_ini)
    
    print("✅ pytest.ini yapılandırması oluşturuldu!")

def create_conftest():
    """Test fixtures ve configuration"""
    
    conftest_content = '''"""
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
    
    # Create tables
    from src.models import Base
    Base.metadata.create_all(bind=engine)
    
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    
    yield TestingSessionLocal()

@pytest.fixture
def client():
    """Create test client"""
    from src.main import app
    
    with TestClient(app) as test_client:
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
'''
    
    with open("tests/conftest.py", "w") as f:
        f.write(conftest_content)
    
    print("✅ conftest.py oluşturuldu!")

def create_unit_tests():
    """Unit test örnekleri"""
    
    # Test API Endpoints
    test_api = '''"""
Unit Tests - API Endpoints
"""
import pytest
from fastapi import status

class TestHealthEndpoint:
    """Health endpoint testleri"""
    
    def test_health_check(self, client):
        """Test /health endpoint"""
        response = client.get("/health")
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data
        assert "timestamp" in data
    
    def test_readiness_check(self, client):
        """Test /ready endpoint"""
        response = client.get("/monitoring/ready")
        assert response.status_code in [status.HTTP_200_OK, status.HTTP_503_SERVICE_UNAVAILABLE]

class TestLearningPathAPI:
    """Learning Path API testleri"""
    
    @pytest.mark.asyncio
    async def test_create_learning_path(self, client, auth_headers, sample_user):
        """Test learning path creation"""
        request_data = {
            "student_id": sample_user["id"],
            "topic": "Matematik",
            "grade": sample_user["grade"],
            "target_date": "2025-03-01"
        }
        
        response = client.post(
            "/api/v1/learning-path",
            json=request_data,
            headers=auth_headers
        )
        
        assert response.status_code == status.HTTP_201_CREATED
        data = response.json()
        assert "path_id" in data
        assert data["topic"] == "Matematik"
        assert len(data["milestones"]) > 0
    
    def test_learning_path_invalid_grade(self, client, auth_headers):
        """Test invalid grade"""
        request_data = {
            "student_id": 1,
            "topic": "Matematik",
            "grade": 15,  # Invalid grade
            "target_date": "2025-03-01"
        }
        
        response = client.post(
            "/api/v1/learning-path",
            json=request_data,
            headers=auth_headers
        )
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

class TestQuizGeneration:
    """Quiz generation testleri"""
    
    @pytest.mark.asyncio
    async def test_generate_adaptive_quiz(self, client, auth_headers, sample_quiz_request):
        """Test adaptive quiz generation"""
        response = client.post(
            "/api/v1/adaptive-quiz",
            json=sample_quiz_request,
            headers=auth_headers
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "quiz_id" in data
        assert "questions" in data
        assert len(data["questions"]) == sample_quiz_request["question_count"]
        
        # Check question structure
        for question in data["questions"]:
            assert "id" in question
            assert "text" in question
            assert "options" in question
            assert "difficulty" in question
    
    @pytest.mark.parametrize("difficulty", [0.1, 0.5, 0.9])
    def test_quiz_difficulty_levels(self, client, auth_headers, difficulty):
        """Test different difficulty levels"""
        request_data = {
            "topic": "Fizik",
            "grade": 11,
            "difficulty": difficulty,
            "question_count": 5
        }
        
        response = client.post(
            "/api/v1/adaptive-quiz",
            json=request_data,
            headers=auth_headers
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        # Check average difficulty
        avg_difficulty = sum(q["difficulty"] for q in data["questions"]) / len(data["questions"])
        assert abs(avg_difficulty - difficulty) < 0.2  # Within tolerance
'''
    
    with open("tests/unit/test_api_endpoints.py", "w") as f:
        f.write(test_api)
    
    # Test Models
    test_models = '''"""
Unit Tests - Data Models
"""
import pytest
from datetime import datetime, timedelta

class TestUserModel:
    """User model testleri"""
    
    def test_user_creation(self, test_db):
        """Test user creation"""
        from src.models import User
        
        user = User(
            username="test_student",
            email="student@test.com",
            grade=10,
            learning_style="visual"
        )
        
        test_db.add(user)
        test_db.commit()
        
        assert user.id is not None
        assert user.username == "test_student"
        assert user.created_at is not None
    
    def test_user_validation(self):
        """Test user validation"""
        from src.models import User
        
        with pytest.raises(ValueError):
            User(
                username="",  # Empty username
                email="invalid-email",  # Invalid email
                grade=15  # Invalid grade
            )

class TestQuizModel:
    """Quiz model testleri"""
    
    def test_quiz_creation(self, test_db):
        """Test quiz creation"""
        from src.models import Quiz
        
        quiz = Quiz(
            title="Matematik Test 1",
            topic="Matematik",
            grade=10,
            difficulty=0.5,
            question_count=10
        )
        
        test_db.add(quiz)
        test_db.commit()
        
        assert quiz.id is not None
        assert quiz.title == "Matematik Test 1"
        assert quiz.is_active == True
    
    def test_quiz_irt_parameters(self, test_db):
        """Test IRT parameters"""
        from src.models import Question
        
        question = Question(
            quiz_id=1,
            text="Test sorusu",
            difficulty=0.5,
            discrimination=1.2,
            guessing=0.25
        )
        
        test_db.add(question)
        test_db.commit()
        
        assert question.difficulty >= 0 and question.difficulty <= 1
        assert question.discrimination > 0
        assert question.guessing >= 0 and question.guessing <= 1

class TestProgressModel:
    """Progress tracking model testleri"""
    
    def test_progress_calculation(self, test_db):
        """Test progress calculation"""
        from src.models import Progress
        
        progress = Progress(
            user_id=1,
            topic="Matematik",
            completed_lessons=5,
            total_lessons=10,
            quiz_scores=[85, 90, 78, 92]
        )
        
        test_db.add(progress)
        test_db.commit()
        
        assert progress.completion_percentage == 50.0
        assert progress.average_score == 86.25
        assert progress.is_completed == False
'''
    
    with open("tests/unit/test_models.py", "w") as f:
        f.write(test_models)
    
    # Test Services
    test_services = '''"""
Unit Tests - Business Logic Services
"""
import pytest
from unittest.mock import Mock, patch, AsyncMock

class TestLearningPathService:
    """Learning path service testleri"""
    
    @pytest.mark.asyncio
    async def test_zpd_calculation(self):
        """Test Zone of Proximal Development calculation"""
        from src.services.learning_path_service import calculate_zpd
        
        current_level = 0.5
        performance = 0.8
        
        zpd = calculate_zpd(current_level, performance)
        
        assert zpd > current_level  # Should increase for good performance
        assert zpd <= 1.0  # Should not exceed max
    
    @pytest.mark.asyncio
    async def test_adaptive_content_selection(self):
        """Test adaptive content selection"""
        from src.services.learning_path_service import select_next_content
        
        student_profile = {
            "current_level": 0.6,
            "learning_style": "visual",
            "completed_topics": ["Algebra", "Geometry"]
        }
        
        next_content = await select_next_content(student_profile)
        
        assert next_content is not None
        assert next_content["difficulty"] >= 0.5  # Within ZPD
        assert next_content["difficulty"] <= 0.7
        assert next_content["style"] == "visual"

class TestIRTService:
    """IRT (Item Response Theory) service testleri"""
    
    def test_probability_calculation(self):
        """Test IRT probability calculation"""
        from src.services.irt_service import calculate_probability
        
        ability = 0.5
        difficulty = 0.5
        discrimination = 1.0
        guessing = 0.25
        
        prob = calculate_probability(ability, difficulty, discrimination, guessing)
        
        assert prob >= guessing  # Should be at least guessing probability
        assert prob <= 1.0  # Should not exceed 1
        assert abs(prob - 0.625) < 0.01  # Expected value for equal ability/difficulty
    
    def test_ability_estimation(self):
        """Test ability estimation from responses"""
        from src.services.irt_service import estimate_ability
        
        responses = [
            {"correct": True, "difficulty": 0.3},
            {"correct": True, "difficulty": 0.5},
            {"correct": False, "difficulty": 0.8},
            {"correct": True, "difficulty": 0.6}
        ]
        
        ability = estimate_ability(responses)
        
        assert ability >= 0 and ability <= 1
        assert ability > 0.5  # Should be above average for 3/4 correct

class TestGamificationService:
    """Gamification service testleri"""
    
    def test_xp_calculation(self):
        """Test XP calculation"""
        from src.services.gamification_service import calculate_xp
        
        quiz_score = 85
        difficulty = 0.7
        time_bonus = True
        
        xp = calculate_xp(quiz_score, difficulty, time_bonus)
        
        assert xp > 0
        assert xp > calculate_xp(85, 0.3, False)  # Higher difficulty = more XP
    
    def test_achievement_unlock(self):
        """Test achievement unlocking"""
        from src.services.gamification_service import check_achievements
        
        user_stats = {
            "total_xp": 1000,
            "streak_days": 7,
            "quizzes_completed": 25,
            "perfect_scores": 5
        }
        
        new_achievements = check_achievements(user_stats)
        
        assert "Week Warrior" in new_achievements  # 7 day streak
        assert "Quiz Master" in new_achievements  # 25 quizzes
'''
    
    with open("tests/unit/test_services.py", "w") as f:
        f.write(test_services)
    
    print("✅ Unit testler oluşturuldu!")

def create_integration_tests():
    """Integration test örnekleri"""
    
    test_integration = '''"""
Integration Tests - Full Flow Testing
"""
import pytest
from fastapi import status
import asyncio

class TestFullUserFlow:
    """Complete user journey test"""
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_complete_learning_flow(self, client, auth_headers):
        """Test complete learning flow from registration to completion"""
        
        # 1. Register user
        user_data = {
            "username": "integration_test_user",
            "email": "integration@test.com",
            "password": "Test123!",
            "grade": 10
        }
        
        response = client.post("/api/v1/auth/register", json=user_data)
        assert response.status_code == status.HTTP_201_CREATED
        user_id = response.json()["user_id"]
        
        # 2. Take learning style assessment
        assessment_data = {
            "user_id": user_id,
            "responses": [
                {"question_id": 1, "answer": "visual"},
                {"question_id": 2, "answer": "reading"},
                {"question_id": 3, "answer": "visual"}
            ]
        }
        
        response = client.post(
            "/api/v1/learning-style/assess",
            json=assessment_data,
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_200_OK
        learning_style = response.json()["primary_style"]
        
        # 3. Generate learning path
        path_data = {
            "student_id": user_id,
            "topic": "Matematik",
            "grade": 10,
            "learning_style": learning_style
        }
        
        response = client.post(
            "/api/v1/learning-path",
            json=path_data,
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_201_CREATED
        path_id = response.json()["path_id"]
        
        # 4. Take adaptive quiz
        quiz_data = {
            "path_id": path_id,
            "topic": "Matematik",
            "adaptive": True
        }
        
        response = client.post(
            "/api/v1/adaptive-quiz",
            json=quiz_data,
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_200_OK
        quiz_id = response.json()["quiz_id"]
        
        # 5. Submit quiz answers
        answers_data = {
            "quiz_id": quiz_id,
            "answers": [
                {"question_id": 1, "answer": "A"},
                {"question_id": 2, "answer": "B"},
                {"question_id": 3, "answer": "C"}
            ]
        }
        
        response = client.post(
            "/api/v1/quiz/submit",
            json=answers_data,
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_200_OK
        score = response.json()["score"]
        
        # 6. Check progress
        response = client.get(
            f"/api/v1/progress/{user_id}",
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_200_OK
        progress = response.json()
        assert progress["total_xp"] > 0
        assert len(progress["achievements"]) >= 0

@pytest.mark.integration
class TestDatabaseTransactions:
    """Database transaction tests"""
    
    @pytest.mark.asyncio
    async def test_concurrent_updates(self, test_db):
        """Test concurrent database updates"""
        from src.models import User, Progress
        
        async def update_progress(user_id, xp):
            progress = test_db.query(Progress).filter_by(user_id=user_id).first()
            progress.total_xp += xp
            test_db.commit()
        
        # Create user and progress
        user = User(username="concurrent_test", email="concurrent@test.com")
        test_db.add(user)
        test_db.commit()
        
        progress = Progress(user_id=user.id, total_xp=0)
        test_db.add(progress)
        test_db.commit()
        
        # Simulate concurrent updates
        tasks = [update_progress(user.id, 10) for _ in range(10)]
        await asyncio.gather(*tasks)
        
        # Check final state
        final_progress = test_db.query(Progress).filter_by(user_id=user.id).first()
        assert final_progress.total_xp == 100  # Should handle all updates

@pytest.mark.integration
class TestExternalServices:
    """External service integration tests"""
    
    @pytest.mark.asyncio
    @patch("src.services.ai_service.call_model")
    async def test_ai_model_integration(self, mock_model):
        """Test AI model integration"""
        mock_model.return_value = {
            "response": "Mocked AI response",
            "confidence": 0.95
        }
        
        from src.services.ai_service import generate_answer
        
        result = await generate_answer("Test question")
        
        assert result is not None
        assert "response" in result
        mock_model.assert_called_once()
'''
    
    with open("tests/integration/test_integration.py", "w") as f:
        f.write(test_integration)
    
    print("✅ Integration testler oluşturuldu!")

def create_performance_tests():
    """Performance test örnekleri"""
    
    test_performance = '''"""
Performance Tests - Load and Stress Testing
"""
import pytest
import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
import statistics

class TestAPIPerformance:
    """API performance tests"""
    
    @pytest.mark.slow
    def test_endpoint_response_time(self, client):
        """Test endpoint response times"""
        endpoints = [
            "/health",
            "/api/v1/topics",
            "/api/v1/grades"
        ]
        
        for endpoint in endpoints:
            times = []
            
            for _ in range(10):
                start = time.time()
                response = client.get(endpoint)
                elapsed = time.time() - start
                times.append(elapsed)
                
                assert response.status_code == 200
            
            avg_time = statistics.mean(times)
            p95_time = statistics.quantiles(times, n=20)[18]  # 95th percentile
            
            assert avg_time < 0.2  # Average under 200ms
            assert p95_time < 0.5  # 95th percentile under 500ms
    
    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_concurrent_requests(self, client):
        """Test handling concurrent requests"""
        async def make_request():
            response = client.get("/health")
            return response.status_code
        
        # Create 100 concurrent requests
        tasks = [make_request() for _ in range(100)]
        start = time.time()
        results = await asyncio.gather(*tasks)
        elapsed = time.time() - start
        
        # All requests should succeed
        assert all(status == 200 for status in results)
        
        # Should handle 100 requests in under 5 seconds
        assert elapsed < 5.0
        
        # Calculate requests per second
        rps = len(results) / elapsed
        assert rps > 20  # At least 20 requests per second
    
    @pytest.mark.slow
    def test_database_query_performance(self, test_db):
        """Test database query performance"""
        from src.models import User
        
        # Insert test data
        users = [
            User(username=f"user_{i}", email=f"user_{i}@test.com")
            for i in range(1000)
        ]
        test_db.bulk_save_objects(users)
        test_db.commit()
        
        # Test query performance
        start = time.time()
        result = test_db.query(User).filter(
            User.username.like("user_%")
        ).limit(100).all()
        elapsed = time.time() - start
        
        assert len(result) == 100
        assert elapsed < 0.1  # Query should be under 100ms
    
    @pytest.mark.slow
    def test_memory_usage(self):
        """Test memory usage under load"""
        import psutil
        import gc
        
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Simulate heavy operation
        data = []
        for _ in range(1000):
            data.append([i for i in range(1000)])
        
        peak_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Clean up
        del data
        gc.collect()
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Memory should not leak
        assert final_memory - initial_memory < 50  # Less than 50MB increase
        
        # Peak memory should be reasonable
        assert peak_memory - initial_memory < 200  # Less than 200MB peak

class TestCachePerformance:
    """Cache performance tests"""
    
    @pytest.mark.slow
    def test_cache_hit_rate(self, client):
        """Test cache hit rate"""
        endpoint = "/api/v1/topics"
        
        # First request (cache miss)
        start1 = time.time()
        response1 = client.get(endpoint)
        time1 = time.time() - start1
        
        # Second request (cache hit)
        start2 = time.time()
        response2 = client.get(endpoint)
        time2 = time.time() - start2
        
        assert response1.status_code == 200
        assert response2.status_code == 200
        
        # Cached request should be much faster
        assert time2 < time1 * 0.5  # At least 50% faster
'''
    
    with open("tests/test_performance.py", "w") as f:
        f.write(test_performance)
    
    print("✅ Performance testler oluşturuldu!")

def create_test_runner():
    """Test runner script"""
    
    runner = '''#!/usr/bin/env python
"""
TEKNOFEST 2025 - Test Runner
Tüm testleri çalıştırır ve rapor üretir
"""
import subprocess
import sys
import os
from pathlib import Path

def run_tests():
    """Run all tests with coverage"""
    
    print("🧪 TEKNOFEST 2025 - Test Suite")
    print("=" * 50)
    
    # Test commands
    commands = [
        # Unit tests
        ["pytest", "tests/unit", "-v", "--tb=short", "-m", "not slow"],
        
        # Integration tests
        ["pytest", "tests/integration", "-v", "--tb=short", "-m", "integration"],
        
        # Performance tests (optional)
        # ["pytest", "tests", "-v", "-m", "slow", "--timeout=60"],
        
        # Coverage report
        ["pytest", "tests", "--cov=src", "--cov-report=term-missing", "--cov-report=html"],
        
        # Generate XML report for CI/CD
        ["pytest", "tests", "--junitxml=test-results.xml"],
    ]
    
    failed = False
    
    for cmd in commands:
        print(f"\\n📍 Running: {' '.join(cmd)}")
        print("-" * 40)
        
        result = subprocess.run(cmd, capture_output=False)
        
        if result.returncode != 0:
            failed = True
            print(f"❌ Command failed: {' '.join(cmd)}")
        else:
            print(f"✅ Command succeeded: {' '.join(cmd)}")
    
    # Summary
    print("\\n" + "=" * 50)
    if failed:
        print("❌ Some tests failed!")
        print("📊 Check htmlcov/index.html for coverage report")
        sys.exit(1)
    else:
        print("✅ All tests passed!")
        print("📊 Coverage report: htmlcov/index.html")
        
        # Open coverage report
        if sys.platform == "win32":
            os.startfile("htmlcov/index.html")
        elif sys.platform == "darwin":
            subprocess.run(["open", "htmlcov/index.html"])
        else:
            subprocess.run(["xdg-open", "htmlcov/index.html"])
        
        sys.exit(0)

if __name__ == "__main__":
    run_tests()
'''
    
    with open("run_tests.py", "w") as f:
        f.write(runner)
    
    # Make executable on Unix
    if sys.platform != "win32":
        os.chmod("run_tests.py", 0o755)
    
    print("✅ Test runner oluşturuldu!")

def install_test_dependencies():
    """Test dependencies kurulumu"""
    
    print("\n📦 Test dependencies kuruluyor...")
    
    dependencies = [
        "pytest>=7.4.0",
        "pytest-cov>=4.1.0",
        "pytest-asyncio>=0.21.0",
        "pytest-mock>=3.11.0",
        "pytest-timeout>=2.1.0",
        "pytest-xdist>=3.3.0",  # Parallel testing
        "httpx>=0.24.0",  # Async HTTP testing
        "faker>=19.0.0",  # Test data generation
        "factory-boy>=3.3.0",  # Test fixtures
        "freezegun>=1.2.0",  # Time mocking
    ]
    
    # Create requirements-test.txt
    with open("requirements-test.txt", "w") as f:
        f.write("\n".join(dependencies))
    
    print("✅ requirements-test.txt oluşturuldu!")
    
    # Install dependencies
    print("\n📥 Dependencies yükleniyor...")
    result = subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements-test.txt"])
    
    if result.returncode == 0:
        print("✅ Test dependencies yüklendi!")
    else:
        print("⚠️ Bazı dependencies yüklenemedi. Manual olarak yükleyin:")
        print("pip install -r requirements-test.txt")

def main():
    """Ana fonksiyon"""
    
    print("🚀 TEKNOFEST 2025 - Test Altyapısı Kurulumu")
    print("=" * 50)
    
    # 1. Create test structure
    create_test_structure()
    
    # 2. Create pytest config
    create_pytest_config()
    
    # 3. Create conftest
    create_conftest()
    
    # 4. Create unit tests
    create_unit_tests()
    
    # 5. Create integration tests
    create_integration_tests()
    
    # 6. Create performance tests
    create_performance_tests()
    
    # 7. Create test runner
    create_test_runner()
    
    # 8. Install dependencies
    install_test_dependencies()
    
    print("\n" + "=" * 50)
    print("✅ TEST ALTYAPISI HAZIR!")
    print("\n📝 Kullanım:")
    print("  - Tüm testleri çalıştır: python run_tests.py")
    print("  - Sadece unit testler: pytest tests/unit -v")
    print("  - Coverage raporu: pytest --cov=src --cov-report=html")
    print("  - Parallel testing: pytest -n auto")
    print("\n📊 Hedef: %80+ test coverage")
    print("🎯 Şu an: 15+ test hazır!")

if __name__ == "__main__":
    main()
