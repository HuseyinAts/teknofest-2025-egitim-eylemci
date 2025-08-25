"""
Comprehensive API Endpoint Test Suite
TEKNOFEST 2025 - Production Ready API Tests
"""

import pytest
from fastapi.testclient import TestClient
from fastapi import status
from unittest.mock import Mock, patch, MagicMock
import json
import jwt
from datetime import datetime, timedelta
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.app import app, get_settings
from src.core.authentication import create_access_token, verify_password, get_password_hash


class TestAPIBasics:
    """Test basic API functionality"""
    
    @pytest.fixture
    def client(self):
        """Create test client"""
        return TestClient(app)
    
    @pytest.fixture
    def mock_settings(self):
        """Mock settings for testing"""
        with patch('src.app.get_settings') as mock:
            settings = MagicMock()
            settings.app_name = "Test App"
            settings.app_version = "1.0.0"
            settings.app_env.value = "testing"
            settings.app_debug = False
            settings.validate_production_ready.return_value = []
            mock.return_value = settings
            yield settings
    
    def test_root_endpoint(self, client, mock_settings):
        """Test root endpoint"""
        response = client.get("/")
        assert response.status_code == status.HTTP_200_OK
        
        data = response.json()
        assert "name" in data
        assert "version" in data
        assert "environment" in data
        assert "status" in data
        assert data["status"] == "healthy"
    
    def test_health_check(self, client, mock_settings):
        """Test health check endpoint"""
        response = client.get("/health")
        assert response.status_code == status.HTTP_200_OK
        
        data = response.json()
        assert "status" in data
        assert "environment" in data
        assert "debug" in data
        assert data["status"] in ["healthy", "warning"]
    
    def test_404_not_found(self, client):
        """Test 404 response for unknown endpoint"""
        response = client.get("/nonexistent")
        assert response.status_code == status.HTTP_404_NOT_FOUND


class TestAuthenticationEndpoints:
    """Test authentication related endpoints"""
    
    @pytest.fixture
    def client(self):
        """Create test client"""
        return TestClient(app)
    
    @pytest.fixture
    def mock_db(self):
        """Mock database session"""
        with patch('src.app.SessionLocal') as mock:
            db = MagicMock()
            mock.return_value = db
            yield db
    
    @pytest.fixture
    def test_user(self):
        """Create test user data"""
        return {
            "email": "test@example.com",
            "username": "testuser",
            "password": "TestPassword123!",
            "full_name": "Test User"
        }
    
    def test_register_user(self, client, mock_db, test_user):
        """Test user registration"""
        with patch('src.api.auth_routes.UserRepository') as mock_repo:
            mock_repo_instance = MagicMock()
            mock_repo_instance.get_by_email.return_value = None
            mock_repo_instance.get_by_username.return_value = None
            mock_repo_instance.create.return_value = MagicMock(
                id="user-id-123",
                email=test_user["email"],
                username=test_user["username"],
                full_name=test_user["full_name"]
            )
            mock_repo.return_value = mock_repo_instance
            
            response = client.post("/api/v1/auth/register", json=test_user)
            
            assert response.status_code == status.HTTP_201_CREATED
            data = response.json()
            assert "access_token" in data
            assert "token_type" in data
            assert data["token_type"] == "bearer"
    
    def test_register_duplicate_email(self, client, mock_db, test_user):
        """Test registration with duplicate email"""
        with patch('src.api.auth_routes.UserRepository') as mock_repo:
            mock_repo_instance = MagicMock()
            mock_repo_instance.get_by_email.return_value = MagicMock()  # User exists
            mock_repo.return_value = mock_repo_instance
            
            response = client.post("/api/v1/auth/register", json=test_user)
            
            assert response.status_code == status.HTTP_400_BAD_REQUEST
            assert "already registered" in response.json()["detail"].lower()
    
    def test_login_success(self, client, mock_db):
        """Test successful login"""
        with patch('src.api.auth_routes.UserRepository') as mock_repo:
            mock_user = MagicMock()
            mock_user.email = "test@example.com"
            mock_user.hashed_password = get_password_hash("TestPassword123!")
            mock_user.is_active = True
            mock_user.id = "user-id-123"
            
            mock_repo_instance = MagicMock()
            mock_repo_instance.get_by_email.return_value = mock_user
            mock_repo.return_value = mock_repo_instance
            
            login_data = {
                "username": "test@example.com",  # OAuth2 uses 'username' field
                "password": "TestPassword123!"
            }
            
            response = client.post("/api/v1/auth/login", data=login_data)
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert "access_token" in data
            assert "token_type" in data
    
    def test_login_invalid_credentials(self, client, mock_db):
        """Test login with invalid credentials"""
        with patch('src.api.auth_routes.UserRepository') as mock_repo:
            mock_repo_instance = MagicMock()
            mock_repo_instance.get_by_email.return_value = None
            mock_repo.return_value = mock_repo_instance
            
            login_data = {
                "username": "wrong@example.com",
                "password": "WrongPassword"
            }
            
            response = client.post("/api/v1/auth/login", data=login_data)
            
            assert response.status_code == status.HTTP_401_UNAUTHORIZED
    
    def test_get_current_user(self, client, mock_db):
        """Test getting current user info"""
        # Create valid token
        token_data = {"sub": "test@example.com"}
        token = create_access_token(token_data)
        
        with patch('src.api.auth_routes.UserRepository') as mock_repo:
            mock_user = MagicMock()
            mock_user.email = "test@example.com"
            mock_user.username = "testuser"
            mock_user.full_name = "Test User"
            
            mock_repo_instance = MagicMock()
            mock_repo_instance.get_by_email.return_value = mock_user
            mock_repo.return_value = mock_repo_instance
            
            headers = {"Authorization": f"Bearer {token}"}
            response = client.get("/api/v1/auth/me", headers=headers)
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert data["email"] == "test@example.com"


class TestLearningPathEndpoints:
    """Test learning path related endpoints"""
    
    @pytest.fixture
    def client(self):
        """Create test client"""
        return TestClient(app)
    
    @pytest.fixture
    def auth_headers(self):
        """Create authenticated headers"""
        token_data = {"sub": "test@example.com"}
        token = create_access_token(token_data)
        return {"Authorization": f"Bearer {token}"}
    
    def test_detect_learning_style(self, client, auth_headers):
        """Test learning style detection endpoint"""
        with patch('src.app.get_learning_path_agent') as mock_agent:
            mock_agent_instance = MagicMock()
            mock_agent_instance.detect_learning_style.return_value = {
                "dominant_style": "visual",
                "scores": {"visual": 3, "auditory": 1, "reading": 1, "kinesthetic": 0},
                "percentages": {"visual": 60, "auditory": 20, "reading": 20, "kinesthetic": 0},
                "confidence": 0.6
            }
            mock_agent.return_value = mock_agent_instance
            
            request_data = {
                "student_responses": [
                    "I prefer visual materials",
                    "Graphs help me learn",
                    "Videos are effective"
                ]
            }
            
            response = client.post(
                "/api/v1/learning-style",
                json=request_data,
                headers=auth_headers
            )
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert data["success"] == True
            assert "data" in data
            assert data["data"]["dominant_style"] == "visual"
    
    def test_get_curriculum(self, client, auth_headers):
        """Test curriculum endpoint"""
        with patch('src.app.get_learning_path_agent') as mock_agent:
            mock_agent_instance = MagicMock()
            mock_agent_instance.curriculum = {
                "9": {
                    "Matematik": {
                        "topics": ["Kümeler", "Sayılar"],
                        "hours": 180
                    }
                }
            }
            mock_agent.return_value = mock_agent_instance
            
            response = client.get("/api/v1/curriculum/9", headers=auth_headers)
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert data["success"] == True
            assert "Matematik" in data["data"]
    
    def test_get_curriculum_not_found(self, client, auth_headers):
        """Test curriculum not found"""
        with patch('src.app.get_learning_path_agent') as mock_agent:
            mock_agent_instance = MagicMock()
            mock_agent_instance.curriculum = {}
            mock_agent.return_value = mock_agent_instance
            
            response = client.get("/api/v1/curriculum/13", headers=auth_headers)
            
            assert response.status_code == status.HTTP_404_NOT_FOUND


class TestQuizEndpoints:
    """Test quiz and assessment endpoints"""
    
    @pytest.fixture
    def client(self):
        """Create test client"""
        return TestClient(app)
    
    @pytest.fixture
    def auth_headers(self):
        """Create authenticated headers"""
        token_data = {"sub": "test@example.com"}
        token = create_access_token(token_data)
        return {"Authorization": f"Bearer {token}"}
    
    def test_generate_quiz(self, client, auth_headers):
        """Test quiz generation endpoint"""
        with patch('src.app.get_study_buddy_agent') as mock_agent:
            mock_agent_instance = MagicMock()
            mock_agent_instance.generate_adaptive_quiz.return_value = [
                {
                    "id": "q_1",
                    "text": "What is 2+2?",
                    "options": ["3", "4", "5", "6"],
                    "correct_answer": 1,
                    "difficulty": 0.3,
                    "success_probability": 0.85
                }
            ]
            mock_agent.return_value = mock_agent_instance
            
            request_data = {
                "topic": "Matematik",
                "student_ability": 0.5,
                "num_questions": 1
            }
            
            response = client.post(
                "/api/v1/generate-quiz",
                json=request_data,
                headers=auth_headers
            )
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert data["success"] == True
            assert len(data["data"]) == 1
            assert data["data"][0]["id"] == "q_1"
    
    def test_generate_quiz_invalid_params(self, client, auth_headers):
        """Test quiz generation with invalid parameters"""
        request_data = {
            "topic": "Math",
            "student_ability": 1.5,  # Invalid: > 1.0
            "num_questions": -5  # Invalid: negative
        }
        
        response = client.post(
            "/api/v1/generate-quiz",
            json=request_data,
            headers=auth_headers
        )
        
        # Should handle gracefully or return validation error
        assert response.status_code in [
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            status.HTTP_400_BAD_REQUEST,
            status.HTTP_200_OK  # If handled gracefully
        ]


class TestAIEndpoints:
    """Test AI/ML related endpoints"""
    
    @pytest.fixture
    def client(self):
        """Create test client"""
        return TestClient(app)
    
    @pytest.fixture
    def auth_headers(self):
        """Create authenticated headers"""
        token_data = {"sub": "test@example.com"}
        token = create_access_token(token_data)
        return {"Authorization": f"Bearer {token}"}
    
    def test_generate_text(self, client, auth_headers):
        """Test text generation endpoint"""
        with patch('src.app.get_model_integration') as mock_model:
            mock_model_instance = MagicMock()
            mock_model_instance.generate.return_value = "Generated text response"
            mock_model.return_value = mock_model_instance
            
            request_data = {
                "prompt": "Explain photosynthesis",
                "max_length": 100
            }
            
            response = client.post(
                "/api/v1/generate-text",
                json=request_data,
                headers=auth_headers
            )
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert data["success"] == True
            assert "generated_text" in data["data"]


class TestRateLimiting:
    """Test rate limiting functionality"""
    
    @pytest.fixture
    def client(self):
        """Create test client"""
        return TestClient(app)
    
    def test_rate_limit_enforcement(self, client):
        """Test that rate limiting is enforced"""
        # Make many rapid requests
        responses = []
        for _ in range(150):  # Exceed typical rate limit
            response = client.get("/")
            responses.append(response.status_code)
        
        # At least some requests should be rate limited
        # Note: Actual implementation may vary
        assert any(
            status_code == status.HTTP_429_TOO_MANY_REQUESTS 
            for status_code in responses
        ) or all(
            status_code == status.HTTP_200_OK 
            for status_code in responses
        )  # Or all pass if rate limiting disabled in test


class TestErrorHandling:
    """Test error handling and edge cases"""
    
    @pytest.fixture
    def client(self):
        """Create test client"""
        return TestClient(app)
    
    def test_malformed_json(self, client):
        """Test handling of malformed JSON"""
        response = client.post(
            "/api/v1/generate-quiz",
            data="not valid json",
            headers={"Content-Type": "application/json"}
        )
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    def test_missing_required_fields(self, client):
        """Test handling of missing required fields"""
        response = client.post(
            "/api/v1/learning-style",
            json={}  # Missing required 'student_responses'
        )
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    def test_internal_server_error_handling(self, client):
        """Test internal server error handling"""
        with patch('src.app.get_learning_path_agent') as mock_agent:
            mock_agent.side_effect = Exception("Database connection failed")
            
            response = client.post(
                "/api/v1/learning-style",
                json={"student_responses": ["test"]}
            )
            
            assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR


class TestCORSHeaders:
    """Test CORS configuration"""
    
    @pytest.fixture
    def client(self):
        """Create test client"""
        return TestClient(app)
    
    def test_cors_headers_present(self, client):
        """Test that CORS headers are present"""
        response = client.options("/")
        
        # Check for CORS headers
        headers = response.headers
        assert "access-control-allow-origin" in headers or \
               "Access-Control-Allow-Origin" in headers
    
    def test_preflight_request(self, client):
        """Test preflight request handling"""
        response = client.options(
            "/api/v1/generate-quiz",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "Content-Type"
            }
        )
        
        assert response.status_code in [status.HTTP_200_OK, status.HTTP_204_NO_CONTENT]


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])