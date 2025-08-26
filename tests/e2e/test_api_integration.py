"""
End-to-end API integration tests.
"""

import pytest
import httpx
import asyncio
from typing import Dict, Any, List
import json
import time
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'src'))

from src.config import Config


class TestAPIIntegration:
    """Complete API integration testing suite."""
    
    BASE_URL = "http://localhost:8000/api/v1"
    
    @pytest.fixture(scope="class")
    def api_client(self):
        """Create API client for testing."""
        return httpx.AsyncClient(base_url=self.BASE_URL, timeout=30.0)
    
    @pytest.fixture(scope="function")
    def test_user_credentials(self):
        """Generate unique test user credentials."""
        unique_id = os.urandom(4).hex()
        return {
            'username': f'api_test_user_{unique_id}',
            'email': f'api_test_{unique_id}@example.com',
            'password': 'Test@Password123'
        }
    
    @pytest.fixture(scope="function")
    async def authenticated_client(self, api_client: httpx.AsyncClient, test_user_credentials: Dict[str, Any]):
        """Create authenticated API client."""
        # Register user
        register_response = await api_client.post(
            '/auth/register',
            json={
                **test_user_credentials,
                'grade_level': 9,
                'learning_style': 'visual'
            }
        )
        assert register_response.status_code == 201
        
        # Login
        login_response = await api_client.post(
            '/auth/login',
            json={
                'email': test_user_credentials['email'],
                'password': test_user_credentials['password']
            }
        )
        assert login_response.status_code == 200
        
        token = login_response.json()['access_token']
        api_client.headers['Authorization'] = f'Bearer {token}'
        
        yield api_client
        
        # Cleanup
        del api_client.headers['Authorization']
    
    @pytest.mark.asyncio
    async def test_health_check(self, api_client: httpx.AsyncClient):
        """Test API health check endpoint."""
        response = await api_client.get('/health')
        assert response.status_code == 200
        
        data = response.json()
        assert data['status'] == 'healthy'
        assert 'database' in data
        assert 'cache' in data
        assert 'timestamp' in data
    
    @pytest.mark.asyncio
    async def test_user_registration_flow(self, api_client: httpx.AsyncClient, test_user_credentials: Dict[str, Any]):
        """Test complete user registration flow."""
        # Register new user
        response = await api_client.post(
            '/auth/register',
            json={
                **test_user_credentials,
                'grade_level': 10,
                'learning_style': 'auditory'
            }
        )
        
        assert response.status_code == 201
        data = response.json()
        assert 'user_id' in data
        assert 'message' in data
        
        # Try to register same user again (should fail)
        duplicate_response = await api_client.post(
            '/auth/register',
            json={
                **test_user_credentials,
                'grade_level': 10,
                'learning_style': 'auditory'
            }
        )
        assert duplicate_response.status_code == 409
    
    @pytest.mark.asyncio
    async def test_authentication_flow(self, api_client: httpx.AsyncClient, test_user_credentials: Dict[str, Any]):
        """Test authentication flow with JWT tokens."""
        # Register user first
        await api_client.post(
            '/auth/register',
            json={
                **test_user_credentials,
                'grade_level': 9,
                'learning_style': 'kinesthetic'
            }
        )
        
        # Login with correct credentials
        login_response = await api_client.post(
            '/auth/login',
            json={
                'email': test_user_credentials['email'],
                'password': test_user_credentials['password']
            }
        )
        
        assert login_response.status_code == 200
        data = login_response.json()
        assert 'access_token' in data
        assert 'refresh_token' in data
        assert 'token_type' in data
        assert data['token_type'] == 'bearer'
        
        # Test token refresh
        refresh_response = await api_client.post(
            '/auth/refresh',
            json={'refresh_token': data['refresh_token']}
        )
        assert refresh_response.status_code == 200
        assert 'access_token' in refresh_response.json()
        
        # Test logout
        api_client.headers['Authorization'] = f'Bearer {data["access_token"]}'
        logout_response = await api_client.post('/auth/logout')
        assert logout_response.status_code == 200
    
    @pytest.mark.asyncio
    async def test_quiz_operations(self, authenticated_client: httpx.AsyncClient):
        """Test quiz CRUD operations."""
        # Create quiz
        quiz_data = {
            'title': 'Test Quiz',
            'subject': 'mathematics',
            'grade_level': 9,
            'questions': [
                {
                    'text': 'What is 2 + 2?',
                    'options': ['3', '4', '5', '6'],
                    'correct_answer': 1,
                    'difficulty': 1
                },
                {
                    'text': 'What is 10 - 5?',
                    'options': ['3', '4', '5', '6'],
                    'correct_answer': 2,
                    'difficulty': 1
                }
            ]
        }
        
        create_response = await authenticated_client.post('/quiz/create', json=quiz_data)
        assert create_response.status_code == 201
        quiz_id = create_response.json()['quiz_id']
        
        # Get quiz
        get_response = await authenticated_client.get(f'/quiz/{quiz_id}')
        assert get_response.status_code == 200
        quiz = get_response.json()
        assert quiz['title'] == quiz_data['title']
        assert len(quiz['questions']) == 2
        
        # Submit quiz answers
        submission = {
            'quiz_id': quiz_id,
            'answers': [1, 2],
            'time_taken': 120
        }
        
        submit_response = await authenticated_client.post('/quiz/submit', json=submission)
        assert submit_response.status_code == 200
        result = submit_response.json()
        assert 'score' in result
        assert 'correct_answers' in result
        assert result['score'] == 100  # Both answers correct
        
        # Get quiz history
        history_response = await authenticated_client.get('/quiz/history')
        assert history_response.status_code == 200
        history = history_response.json()
        assert len(history) > 0
    
    @pytest.mark.asyncio
    async def test_learning_path_operations(self, authenticated_client: httpx.AsyncClient):
        """Test learning path CRUD operations."""
        # Create learning path
        path_data = {
            'title': 'Advanced Mathematics',
            'description': 'Complete mathematics learning path',
            'subject': 'mathematics',
            'grade_level': 9,
            'difficulty': 'intermediate',
            'modules': [
                {
                    'title': 'Algebra Basics',
                    'order': 1,
                    'content': 'Introduction to algebra'
                },
                {
                    'title': 'Geometry Fundamentals',
                    'order': 2,
                    'content': 'Basic geometry concepts'
                }
            ]
        }
        
        create_response = await authenticated_client.post('/learning-paths/create', json=path_data)
        assert create_response.status_code == 201
        path_id = create_response.json()['path_id']
        
        # Get learning path
        get_response = await authenticated_client.get(f'/learning-paths/{path_id}')
        assert get_response.status_code == 200
        path = get_response.json()
        assert path['title'] == path_data['title']
        assert len(path['modules']) == 2
        
        # Update progress
        progress_update = {
            'path_id': path_id,
            'module_id': path['modules'][0]['id'],
            'completed': True,
            'score': 85
        }
        
        progress_response = await authenticated_client.post('/learning-paths/progress', json=progress_update)
        assert progress_response.status_code == 200
        
        # Get user progress
        user_progress_response = await authenticated_client.get('/learning-paths/my-progress')
        assert user_progress_response.status_code == 200
        progress = user_progress_response.json()
        assert len(progress) > 0
    
    @pytest.mark.asyncio
    async def test_gamification_features(self, authenticated_client: httpx.AsyncClient):
        """Test gamification features."""
        # Get user achievements
        achievements_response = await authenticated_client.get('/gamification/achievements')
        assert achievements_response.status_code == 200
        achievements = achievements_response.json()
        assert 'total_points' in achievements
        assert 'badges' in achievements
        assert 'level' in achievements
        
        # Get leaderboard
        leaderboard_response = await authenticated_client.get('/gamification/leaderboard')
        assert leaderboard_response.status_code == 200
        leaderboard = leaderboard_response.json()
        assert isinstance(leaderboard, list)
        
        # Claim daily reward
        daily_reward_response = await authenticated_client.post('/gamification/daily-reward')
        assert daily_reward_response.status_code in [200, 409]  # 409 if already claimed
    
    @pytest.mark.asyncio
    async def test_irt_adaptive_testing(self, authenticated_client: httpx.AsyncClient):
        """Test IRT adaptive testing functionality."""
        # Start adaptive test
        start_response = await authenticated_client.post(
            '/irt/start-test',
            json={
                'subject': 'mathematics',
                'grade_level': 9,
                'target_questions': 10
            }
        )
        assert start_response.status_code == 200
        test_id = start_response.json()['test_id']
        
        # Get next question multiple times
        for i in range(5):
            question_response = await authenticated_client.get(f'/irt/next-question/{test_id}')
            assert question_response.status_code == 200
            question = question_response.json()
            
            # Submit answer
            answer_response = await authenticated_client.post(
                f'/irt/submit-answer/{test_id}',
                json={
                    'question_id': question['id'],
                    'answer': 0,  # First option
                    'time_taken': 30
                }
            )
            assert answer_response.status_code == 200
            
            # Check if difficulty adjusted
            if i > 0:
                assert 'difficulty_adjustment' in answer_response.json()
        
        # Get ability estimate
        ability_response = await authenticated_client.get(f'/irt/ability-estimate/{test_id}')
        assert ability_response.status_code == 200
        ability = ability_response.json()
        assert 'theta' in ability
        assert 'confidence_interval' in ability
    
    @pytest.mark.asyncio
    async def test_rate_limiting(self, api_client: httpx.AsyncClient):
        """Test API rate limiting."""
        # Make multiple rapid requests
        responses = []
        for _ in range(15):  # Assuming rate limit is 10 per minute
            response = await api_client.get('/health')
            responses.append(response.status_code)
            await asyncio.sleep(0.1)
        
        # Check if rate limiting kicked in
        assert 429 in responses, "Rate limiting not working"
        
        # Wait for rate limit reset
        await asyncio.sleep(60)
        
        # Should work again
        response = await api_client.get('/health')
        assert response.status_code == 200
    
    @pytest.mark.asyncio
    async def test_concurrent_requests(self, authenticated_client: httpx.AsyncClient):
        """Test handling of concurrent requests."""
        # Create multiple concurrent requests
        tasks = []
        for i in range(10):
            task = authenticated_client.get('/quiz/list')
            tasks.append(task)
        
        # Execute concurrently
        responses = await asyncio.gather(*tasks)
        
        # All should succeed
        for response in responses:
            assert response.status_code == 200
    
    @pytest.mark.asyncio
    async def test_pagination(self, authenticated_client: httpx.AsyncClient):
        """Test pagination functionality."""
        # Get paginated quiz list
        page1_response = await authenticated_client.get('/quiz/list?page=1&limit=5')
        assert page1_response.status_code == 200
        page1_data = page1_response.json()
        
        assert 'items' in page1_data
        assert 'total' in page1_data
        assert 'page' in page1_data
        assert 'pages' in page1_data
        assert len(page1_data['items']) <= 5
        
        # Get next page if available
        if page1_data['pages'] > 1:
            page2_response = await authenticated_client.get('/quiz/list?page=2&limit=5')
            assert page2_response.status_code == 200
            page2_data = page2_response.json()
            
            # Ensure different items
            page1_ids = {item['id'] for item in page1_data['items']}
            page2_ids = {item['id'] for item in page2_data['items']}
            assert page1_ids.isdisjoint(page2_ids)
    
    @pytest.mark.asyncio
    async def test_search_functionality(self, authenticated_client: httpx.AsyncClient):
        """Test search functionality across different resources."""
        # Search quizzes
        quiz_search = await authenticated_client.get('/quiz/search?q=mathematics')
        assert quiz_search.status_code == 200
        
        # Search learning paths
        path_search = await authenticated_client.get('/learning-paths/search?q=algebra')
        assert path_search.status_code == 200
        
        # Search with filters
        filtered_search = await authenticated_client.get(
            '/quiz/search?q=test&subject=mathematics&grade_level=9'
        )
        assert filtered_search.status_code == 200
    
    @pytest.mark.asyncio
    async def test_data_export(self, authenticated_client: httpx.AsyncClient):
        """Test data export functionality."""
        # Request data export
        export_response = await authenticated_client.post(
            '/user/export-data',
            json={'format': 'json'}
        )
        assert export_response.status_code == 200
        export_data = export_response.json()
        
        assert 'user_info' in export_data
        assert 'quiz_history' in export_data
        assert 'learning_progress' in export_data
        assert 'achievements' in export_data