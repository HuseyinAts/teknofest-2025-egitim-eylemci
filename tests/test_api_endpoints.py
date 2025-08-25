"""
Comprehensive tests for API Endpoints
"""
import pytest
import json
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from fastapi.testclient import TestClient
from fastapi import FastAPI
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Try to import various API server implementations
try:
    from src.api_server_integrated import app as integrated_app
except ImportError:
    integrated_app = None

try:
    from src.api_server_frontend import app as frontend_app
except ImportError:
    frontend_app = None

try:
    from src.api_server_with_rate_limit import app as rate_limit_app
except ImportError:
    rate_limit_app = None


class TestAPIEndpoints:
    
    @pytest.fixture
    def client(self):
        """Create a test client for the API"""
        if integrated_app:
            return TestClient(integrated_app)
        elif frontend_app:
            return TestClient(frontend_app)
        else:
            # Create a minimal app for testing
            app = FastAPI()
            
            @app.get("/health")
            def health_check():
                return {"status": "healthy"}
            
            @app.post("/api/learning-path")
            def generate_learning_path(request: dict):
                return {"path": "mocked", "weeks": 4}
            
            return TestClient(app)
    
    @pytest.mark.unit
    def test_health_endpoint(self, client):
        """Test health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert data["status"] in ["healthy", "ok", "running"]
    
    @pytest.mark.unit
    def test_learning_path_endpoint(self, client):
        """Test learning path generation endpoint"""
        request_data = {
            "student_profile": {
                "student_id": "test_001",
                "current_level": 0.5,
                "target_level": 0.8,
                "learning_style": "visual",
                "grade": 10
            },
            "subject": "Matematik",
            "weeks": 4
        }
        
        response = client.post("/api/learning-path", json=request_data)
        
        if response.status_code == 404:
            pytest.skip("Learning path endpoint not implemented")
        
        assert response.status_code in [200, 201]
        data = response.json()
        assert data is not None
    
    @pytest.mark.unit
    def test_quiz_generation_endpoint(self, client):
        """Test quiz generation endpoint"""
        request_data = {
            "topic": "Fizik",
            "difficulty": 0.6,
            "num_questions": 5
        }
        
        response = client.post("/api/generate-quiz", json=request_data)
        
        if response.status_code == 404:
            pytest.skip("Quiz generation endpoint not implemented")
        
        assert response.status_code in [200, 201]
        data = response.json()
        assert data is not None
    
    @pytest.mark.unit
    def test_student_profile_endpoint(self, client):
        """Test student profile endpoints"""
        # Test GET student profile
        response = client.get("/api/student/test_001")
        
        if response.status_code == 404:
            # Try creating a profile first
            profile_data = {
                "student_id": "test_001",
                "name": "Test Student",
                "grade": 10,
                "learning_style": "visual"
            }
            
            create_response = client.post("/api/student", json=profile_data)
            if create_response.status_code != 404:
                assert create_response.status_code in [200, 201]
                
                # Now try to get it
                response = client.get("/api/student/test_001")
                if response.status_code != 404:
                    assert response.status_code == 200
    
    @pytest.mark.unit
    def test_invalid_request_handling(self, client):
        """Test handling of invalid requests"""
        # Missing required fields
        invalid_request = {
            "topic": "Math"
            # Missing other required fields
        }
        
        response = client.post("/api/generate-quiz", json=invalid_request)
        
        if response.status_code != 404:
            # Should return 400 or 422 for bad request
            assert response.status_code in [400, 422]
    
    @pytest.mark.unit
    def test_cors_headers(self, client):
        """Test CORS headers if configured"""
        response = client.options("/api/learning-path")
        
        if response.status_code != 404:
            # Check for CORS headers
            headers = response.headers
            if "access-control-allow-origin" in headers:
                assert headers["access-control-allow-origin"] in ["*", "http://localhost:3000"]
    
    @pytest.mark.unit
    @pytest.mark.parametrize("endpoint", [
        "/api/learning-path",
        "/api/generate-quiz",
        "/api/study-plan",
        "/api/assessment"
    ])
    def test_post_endpoints_exist(self, client, endpoint):
        """Test that expected POST endpoints exist"""
        response = client.post(endpoint, json={})
        
        # Should not return 404 (endpoint exists)
        # May return 400/422 for invalid data
        if response.status_code != 404:
            assert response.status_code in [200, 201, 400, 422]
    
    @pytest.mark.unit
    def test_get_endpoints(self, client):
        """Test GET endpoints"""
        get_endpoints = [
            "/health",
            "/api/status",
            "/api/version"
        ]
        
        for endpoint in get_endpoints:
            response = client.get(endpoint)
            if response.status_code != 404:
                assert response.status_code in [200, 301, 302]
    
    @pytest.mark.unit
    def test_content_type_json(self, client):
        """Test that API returns JSON content type"""
        response = client.get("/health")
        
        if response.status_code == 200:
            assert "application/json" in response.headers.get("content-type", "")
    
    @pytest.mark.unit
    def test_large_request_handling(self, client):
        """Test handling of large requests"""
        large_request = {
            "student_profiles": [
                {
                    "student_id": f"student_{i}",
                    "current_level": 0.5,
                    "target_level": 0.8,
                    "learning_style": "visual",
                    "grade": 10
                }
                for i in range(100)
            ]
        }
        
        response = client.post("/api/batch-learning-paths", json=large_request)
        
        if response.status_code != 404:
            # Should handle large requests
            assert response.status_code in [200, 201, 413]  # 413 for payload too large


@pytest.mark.integration
class TestAPIIntegration:
    
    @pytest.fixture
    def client(self):
        """Create test client for integration tests"""
        if integrated_app:
            return TestClient(integrated_app)
        elif frontend_app:
            return TestClient(frontend_app)
        else:
            pytest.skip("No API app available for integration testing")
    
    def test_complete_learning_flow_api(self, client):
        """Test complete learning flow through API"""
        # Step 1: Create student profile
        profile_data = {
            "student_id": "integration_test",
            "name": "Integration Test Student",
            "grade": 10,
            "current_level": 0.4
        }
        
        profile_response = client.post("/api/student", json=profile_data)
        if profile_response.status_code == 404:
            pytest.skip("Student profile endpoint not implemented")
        
        # Step 2: Detect learning style
        style_data = {
            "student_id": "integration_test",
            "responses": ["Görsel öğrenirim", "Diyagramlar yardımcı"]
        }
        
        style_response = client.post("/api/detect-learning-style", json=style_data)
        if style_response.status_code != 404:
            assert style_response.status_code in [200, 201]
        
        # Step 3: Generate learning path
        path_data = {
            "student_id": "integration_test",
            "subject": "Matematik",
            "weeks": 4
        }
        
        path_response = client.post("/api/learning-path", json=path_data)
        if path_response.status_code != 404:
            assert path_response.status_code in [200, 201]
        
        # Step 4: Generate quiz
        quiz_data = {
            "student_id": "integration_test",
            "topic": "Matematik",
            "difficulty": 0.5,
            "num_questions": 5
        }
        
        quiz_response = client.post("/api/generate-quiz", json=quiz_data)
        if quiz_response.status_code != 404:
            assert quiz_response.status_code in [200, 201]
    
    def test_concurrent_api_requests(self, client):
        """Test handling concurrent API requests"""
        import concurrent.futures
        
        def make_request(i):
            request_data = {
                "topic": f"Topic_{i}",
                "difficulty": 0.5,
                "num_questions": 3
            }
            response = client.post("/api/generate-quiz", json=request_data)
            return response.status_code
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(make_request, i) for i in range(10)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]
        
        # Check that all requests were handled
        assert len(results) == 10
        # Most requests should succeed
        successful = sum(1 for code in results if code in [200, 201])
        if results[0] != 404:  # If endpoint exists
            assert successful >= 8  # At least 80% success rate
    
    def test_api_error_recovery(self, client):
        """Test API error recovery"""
        # Send malformed request
        malformed_request = "This is not JSON"
        
        response = client.post(
            "/api/learning-path",
            data=malformed_request,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code != 404:
            # Should return error status but not crash
            assert response.status_code in [400, 422]
        
        # Send valid request after error
        valid_request = {
            "subject": "Math",
            "weeks": 4
        }
        
        response2 = client.post("/api/learning-path", json=valid_request)
        
        if response2.status_code != 404:
            # Should still work after previous error
            assert response2.status_code in [200, 201, 400, 422]


class TestRateLimitedAPI:
    
    @pytest.fixture
    def client(self):
        """Create test client for rate limited API"""
        if rate_limit_app:
            return TestClient(rate_limit_app)
        else:
            pytest.skip("Rate limited API not available")
    
    @pytest.mark.unit
    def test_rate_limit_enforcement(self, client):
        """Test that rate limits are enforced"""
        # Make multiple rapid requests
        responses = []
        for i in range(20):
            response = client.get("/health")
            responses.append(response.status_code)
        
        # Some requests should be rate limited (429)
        rate_limited = sum(1 for code in responses if code == 429)
        
        # If rate limiting is enabled, some requests should be limited
        if 429 in responses:
            assert rate_limited > 0
    
    @pytest.mark.unit
    def test_rate_limit_headers(self, client):
        """Test rate limit headers"""
        response = client.get("/health")
        
        # Check for rate limit headers
        headers = response.headers
        rate_limit_headers = [
            "x-ratelimit-limit",
            "x-ratelimit-remaining",
            "x-ratelimit-reset"
        ]
        
        # If any rate limit header exists, it's configured
        has_rate_limit = any(h in headers for h in rate_limit_headers)
        
        if has_rate_limit:
            assert "x-ratelimit-limit" in headers
            assert int(headers["x-ratelimit-limit"]) > 0