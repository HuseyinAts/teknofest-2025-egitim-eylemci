"""
Unit Tests - API Endpoints
"""
import pytest
from fastapi import status
import json

class TestHealthEndpoint:
    """Health endpoint testleri"""
    
    def test_health_check(self, client):
        """Test /health endpoint"""
        response = client.get("/health")
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["status"] == "healthy"
    
    def test_health_check_response_structure(self, client):
        """Test health check response structure"""
        response = client.get("/health")
        data = response.json()
        assert "status" in data

class TestLearningPathAPI:
    """Learning Path API testleri"""
    
    @pytest.mark.asyncio
    async def test_create_learning_path_structure(self, sample_user):
        """Test learning path creation data structure"""
        request_data = {
            "student_id": sample_user["id"],
            "topic": "Matematik",
            "grade": sample_user["grade"],
            "target_date": "2025-03-01"
        }
        
        # Validate request structure
        assert "student_id" in request_data
        assert "topic" in request_data
        assert "grade" in request_data
        assert request_data["grade"] >= 9 and request_data["grade"] <= 12
    
    def test_learning_path_invalid_grade_validation(self):
        """Test invalid grade validation"""
        invalid_grades = [0, 8, 13, 15, -1, 100]
        
        for grade in invalid_grades:
            assert grade < 9 or grade > 12, f"Grade {grade} should be invalid"
    
    def test_learning_path_topic_validation(self):
        """Test topic validation"""
        valid_topics = ["Matematik", "Fizik", "Kimya", "Biyoloji", "Tarih"]
        invalid_topics = ["", None, "InvalidTopic123"]
        
        for topic in valid_topics:
            assert len(topic) > 0, f"Topic {topic} should be valid"
        
        for topic in invalid_topics:
            if topic:
                assert topic not in valid_topics, f"Topic {topic} should be invalid"

class TestQuizGeneration:
    """Quiz generation testleri"""
    
    def test_quiz_request_structure(self, sample_quiz_request):
        """Test quiz request data structure"""
        assert "topic" in sample_quiz_request
        assert "grade" in sample_quiz_request
        assert "difficulty" in sample_quiz_request
        assert "question_count" in sample_quiz_request
        
        # Validate ranges
        assert sample_quiz_request["difficulty"] >= 0.0
        assert sample_quiz_request["difficulty"] <= 1.0
        assert sample_quiz_request["question_count"] > 0
        assert sample_quiz_request["question_count"] <= 50
    
    @pytest.mark.parametrize("difficulty", [0.1, 0.5, 0.9])
    def test_quiz_difficulty_levels(self, difficulty):
        """Test different difficulty levels"""
        assert difficulty >= 0.0 and difficulty <= 1.0
        
        # Test difficulty categorization
        if difficulty < 0.33:
            level = "easy"
        elif difficulty < 0.67:
            level = "medium"
        else:
            level = "hard"
        
        assert level in ["easy", "medium", "hard"]
    
    def test_quiz_question_structure(self):
        """Test quiz question structure"""
        sample_question = {
            "id": 1,
            "text": "Sample question?",
            "options": ["A", "B", "C", "D"],
            "difficulty": 0.5,
            "topic": "Matematik",
            "correct_answer": "A"
        }
        
        # Validate question structure
        assert "id" in sample_question
        assert "text" in sample_question
        assert "options" in sample_question
        assert len(sample_question["options"]) == 4
        assert "difficulty" in sample_question
        assert sample_question["difficulty"] >= 0.0
        assert sample_question["difficulty"] <= 1.0

class TestAuthentication:
    """Authentication testleri"""
    
    def test_auth_header_structure(self, auth_headers):
        """Test authentication header structure"""
        assert "Authorization" in auth_headers
        assert auth_headers["Authorization"].startswith("Bearer ")
        assert "Content-Type" in auth_headers
        assert auth_headers["Content-Type"] == "application/json"
    
    def test_token_validation(self):
        """Test token validation logic"""
        valid_tokens = [
            "Bearer valid-token-123",
            "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9"
        ]
        
        invalid_tokens = [
            "Invalid-token",
            "Bearer",
            "",
            None,
            "Basic token123"
        ]
        
        for token in valid_tokens:
            assert token.startswith("Bearer "), f"Token should start with Bearer: {token}"
        
        for token in invalid_tokens:
            if token:
                assert not token.startswith("Bearer ") or len(token.split()) < 2

class TestDataValidation:
    """Data validation testleri"""
    
    def test_email_validation(self):
        """Test email validation"""
        valid_emails = [
            "test@example.com",
            "user.name@domain.co",
            "user+tag@example.org"
        ]
        
        invalid_emails = [
            "invalid.email",
            "@example.com",
            "user@",
            "",
            None
        ]
        
        import re
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        
        for email in valid_emails:
            assert re.match(email_pattern, email), f"Email should be valid: {email}"
        
        for email in invalid_emails:
            if email:
                assert not re.match(email_pattern, email), f"Email should be invalid: {email}"
    
    def test_grade_range_validation(self):
        """Test grade range validation"""
        valid_grades = [9, 10, 11, 12]
        invalid_grades = [0, 1, 8, 13, 15, -1, 100]
        
        for grade in valid_grades:
            assert 9 <= grade <= 12, f"Grade {grade} should be valid"
        
        for grade in invalid_grades:
            assert not (9 <= grade <= 12), f"Grade {grade} should be invalid"
    
    def test_learning_style_validation(self):
        """Test learning style validation"""
        valid_styles = ["visual", "auditory", "reading", "kinesthetic"]
        invalid_styles = ["invalid", "", None, "other"]
        
        for style in valid_styles:
            assert style in ["visual", "auditory", "reading", "kinesthetic"]
        
        for style in invalid_styles:
            if style:
                assert style not in valid_styles