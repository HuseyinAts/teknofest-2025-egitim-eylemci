"""
Full system integration E2E tests for Teknofest 2025 Education Platform.
"""

import pytest
import asyncio
from playwright.async_api import async_playwright, Page, Browser
import aiohttp
import time
from typing import Dict, Any, List
import json
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'src'))

from src.database.session import DatabaseSession
from src.database.models import User, Quiz, Question, LearningPath, UserProgress


class TestFullSystemIntegration:
    """Full system integration tests covering all major workflows."""
    
    @pytest.fixture(scope="class")
    async def api_client(self):
        """Create API client for backend testing."""
        async with aiohttp.ClientSession() as session:
            yield session
    
    @pytest.fixture(scope="class")
    async def browser(self):
        """Initialize browser for E2E testing."""
        async with async_playwright() as p:
            browser = await p.chromium.launch(
                headless=True,
                args=['--no-sandbox', '--disable-setuid-sandbox']
            )
            yield browser
            await browser.close()
    
    @pytest.fixture(scope="function")
    async def page(self, browser: Browser):
        """Create a new page for each test."""
        context = await browser.new_context(
            viewport={'width': 1920, 'height': 1080},
            ignore_https_errors=True,
            locale='tr-TR'  # Turkish locale for testing
        )
        page = await context.new_page()
        yield page
        await context.close()
    
    @pytest.mark.asyncio
    async def test_complete_student_learning_journey(self, page: Page, api_client):
        """Test complete student learning journey from registration to certification."""
        # 1. Register new student
        registration_data = {
            'username': f'student_{int(time.time())}',
            'email': f'student_{int(time.time())}@test.com',
            'password': 'SecurePass123!',
            'grade_level': 10,
            'learning_style': 'visual',
            'subjects': ['matematik', 'fizik', 'kimya']
        }
        
        # Register via API
        async with api_client.post(
            'http://localhost:8000/api/v1/auth/register',
            json=registration_data
        ) as response:
            assert response.status == 201
            user_data = await response.json()
            access_token = user_data['access_token']
            user_id = user_data['user']['id']
        
        # 2. Complete initial assessment
        headers = {'Authorization': f'Bearer {access_token}'}
        
        # Get initial assessment
        async with api_client.get(
            'http://localhost:8000/api/v1/assessment/initial',
            headers=headers
        ) as response:
            assert response.status == 200
            assessment_data = await response.json()
            assessment_id = assessment_data['id']
        
        # Submit assessment answers
        assessment_answers = [
            {'question_id': q['id'], 'answer': q['options'][0]['id']}
            for q in assessment_data['questions']
        ]
        
        async with api_client.post(
            f'http://localhost:8000/api/v1/assessment/{assessment_id}/submit',
            json={'answers': assessment_answers},
            headers=headers
        ) as response:
            assert response.status == 200
            assessment_result = await response.json()
        
        # 3. Generate personalized learning path
        async with api_client.post(
            'http://localhost:8000/api/v1/learning-paths/generate',
            json={
                'user_id': user_id,
                'assessment_results': assessment_result,
                'preferences': registration_data
            },
            headers=headers
        ) as response:
            assert response.status == 201
            learning_path = await response.json()
            path_id = learning_path['id']
        
        # 4. Navigate through UI to verify learning path
        await page.goto('http://localhost:3000/login')
        await page.fill('input[name="email"]', registration_data['email'])
        await page.fill('input[name="password"]', registration_data['password'])
        await page.click('button[type="submit"]')
        await page.wait_for_url('**/dashboard')
        
        # Navigate to learning path
        await page.goto(f'http://localhost:3000/learning-paths/{path_id}')
        assert await page.is_visible('.learning-path-content')
        
        # 5. Complete learning modules
        modules = learning_path['modules']
        for module in modules[:3]:  # Complete first 3 modules
            # Start module via API
            async with api_client.post(
                f'http://localhost:8000/api/v1/modules/{module["id"]}/start',
                headers=headers
            ) as response:
                assert response.status == 200
            
            # Complete module quiz
            async with api_client.get(
                f'http://localhost:8000/api/v1/modules/{module["id"]}/quiz',
                headers=headers
            ) as response:
                quiz_data = await response.json()
            
            # Submit quiz answers
            quiz_answers = [
                {'question_id': q['id'], 'answer': q['options'][0]['id']}
                for q in quiz_data['questions']
            ]
            
            async with api_client.post(
                f'http://localhost:8000/api/v1/modules/{module["id"]}/quiz/submit',
                json={'answers': quiz_answers},
                headers=headers
            ) as response:
                assert response.status == 200
                quiz_result = await response.json()
                assert quiz_result['passed'] == True
        
        # 6. Check progress and achievements
        async with api_client.get(
            f'http://localhost:8000/api/v1/users/{user_id}/progress',
            headers=headers
        ) as response:
            assert response.status == 200
            progress = await response.json()
            assert progress['completed_modules'] >= 3
            assert progress['total_points'] > 0
        
        # 7. Verify gamification elements
        async with api_client.get(
            f'http://localhost:8000/api/v1/users/{user_id}/achievements',
            headers=headers
        ) as response:
            assert response.status == 200
            achievements = await response.json()
            assert len(achievements) > 0
        
        # 8. Test AI study buddy interaction
        async with api_client.post(
            'http://localhost:8000/api/v1/study-buddy/chat',
            json={
                'message': 'Matematik konusunda yardım eder misin?',
                'context': {'subject': 'matematik', 'grade': 10}
            },
            headers=headers
        ) as response:
            assert response.status == 200
            buddy_response = await response.json()
            assert 'response' in buddy_response
            assert len(buddy_response['response']) > 0
    
    @pytest.mark.asyncio
    async def test_turkish_nlp_processing(self, api_client):
        """Test Turkish NLP processing capabilities."""
        test_texts = [
            "Üçgenin iç açıları toplamı kaç derecedir?",
            "Kimyasal reaksiyonlar nasıl gerçekleşir?",
            "Türkiye'nin başkenti neresidir?",
            "Matematikteki türev ve integral kavramları nelerdir?"
        ]
        
        for text in test_texts:
            async with api_client.post(
                'http://localhost:8000/api/v1/nlp/analyze',
                json={'text': text, 'language': 'tr'}
            ) as response:
                assert response.status == 200
                analysis = await response.json()
                assert 'tokens' in analysis
                assert 'entities' in analysis
                assert 'subject' in analysis
                assert analysis['language'] == 'tr'
    
    @pytest.mark.asyncio
    async def test_concurrent_user_load(self, api_client):
        """Test system behavior under concurrent user load."""
        num_users = 50
        
        async def simulate_user(user_id: int):
            """Simulate a single user's actions."""
            # Register
            async with api_client.post(
                'http://localhost:8000/api/v1/auth/register',
                json={
                    'username': f'load_test_user_{user_id}',
                    'email': f'load_test_{user_id}@test.com',
                    'password': 'TestPass123!',
                    'grade_level': 9
                }
            ) as response:
                if response.status != 201:
                    return False
                data = await response.json()
                token = data['access_token']
            
            # Take a quiz
            headers = {'Authorization': f'Bearer {token}'}
            async with api_client.get(
                'http://localhost:8000/api/v1/quiz/random',
                headers=headers
            ) as response:
                if response.status != 200:
                    return False
            
            return True
        
        # Run concurrent users
        start_time = time.time()
        tasks = [simulate_user(i) for i in range(num_users)]
        results = await asyncio.gather(*tasks)
        end_time = time.time()
        
        # Verify results
        success_rate = sum(results) / len(results)
        assert success_rate > 0.95, f"Success rate {success_rate} is below threshold"
        assert (end_time - start_time) < 30, "Concurrent load test took too long"
    
    @pytest.mark.asyncio
    async def test_data_persistence_and_recovery(self, api_client, page: Page):
        """Test data persistence and recovery scenarios."""
        # Create test data
        test_user = {
            'username': 'persistence_test',
            'email': 'persistence@test.com',
            'password': 'PersistTest123!'
        }
        
        # Register and create data
        async with api_client.post(
            'http://localhost:8000/api/v1/auth/register',
            json=test_user
        ) as response:
            assert response.status == 201
            user_data = await response.json()
            token = user_data['access_token']
            user_id = user_data['user']['id']
        
        headers = {'Authorization': f'Bearer {token}'}
        
        # Create learning progress
        async with api_client.post(
            'http://localhost:8000/api/v1/progress/update',
            json={
                'module_id': 1,
                'completion_percentage': 75,
                'time_spent': 1800
            },
            headers=headers
        ) as response:
            assert response.status == 200
        
        # Simulate connection loss and recovery
        await page.goto('http://localhost:3000/login')
        await page.fill('input[name="email"]', test_user['email'])
        await page.fill('input[name="password"]', test_user['password'])
        await page.click('button[type="submit"]')
        await page.wait_for_url('**/dashboard')
        
        # Go offline
        await page.context.set_offline(True)
        
        # Try to save data offline
        await page.evaluate('''() => {
            localStorage.setItem('offline_progress', JSON.stringify({
                module_id: 2,
                completion_percentage: 50,
                timestamp: Date.now()
            }));
        }''')
        
        # Go back online
        await page.context.set_offline(False)
        await page.reload()
        
        # Verify data sync
        async with api_client.get(
            f'http://localhost:8000/api/v1/users/{user_id}/progress',
            headers=headers
        ) as response:
            assert response.status == 200
            progress = await response.json()
            assert progress is not None
    
    @pytest.mark.asyncio
    async def test_security_measures(self, api_client):
        """Test security measures and protection mechanisms."""
        # Test rate limiting
        attempts = []
        for i in range(100):
            async with api_client.post(
                'http://localhost:8000/api/v1/auth/login',
                json={'email': 'test@test.com', 'password': 'wrong'}
            ) as response:
                attempts.append(response.status)
        
        # Should have rate limiting kick in
        assert 429 in attempts, "Rate limiting not working"
        
        # Test SQL injection protection
        malicious_inputs = [
            "'; DROP TABLE users; --",
            "1' OR '1'='1",
            "<script>alert('XSS')</script>"
        ]
        
        for input_str in malicious_inputs:
            async with api_client.post(
                'http://localhost:8000/api/v1/auth/login',
                json={'email': input_str, 'password': 'test'}
            ) as response:
                assert response.status in [400, 401], "SQL injection protection failed"
        
        # Test JWT token validation
        fake_token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP0THsR8U"
        async with api_client.get(
            'http://localhost:8000/api/v1/users/profile',
            headers={'Authorization': f'Bearer {fake_token}'}
        ) as response:
            assert response.status == 401, "Invalid JWT accepted"
    
    @pytest.mark.asyncio
    async def test_monitoring_and_telemetry(self, api_client):
        """Test monitoring and telemetry endpoints."""
        # Health check
        async with api_client.get('http://localhost:8000/health') as response:
            assert response.status == 200
            health = await response.json()
            assert health['status'] == 'healthy'
            assert 'database' in health
            assert 'redis' in health
        
        # Metrics endpoint
        async with api_client.get('http://localhost:8000/metrics') as response:
            assert response.status == 200
            metrics = await response.text()
            assert 'http_requests_total' in metrics
            assert 'http_request_duration_seconds' in metrics
        
        # Readiness check
        async with api_client.get('http://localhost:8000/ready') as response:
            assert response.status == 200
            ready = await response.json()
            assert ready['ready'] == True
    
    @pytest.mark.asyncio
    async def test_api_versioning_compatibility(self, api_client):
        """Test API versioning and backward compatibility."""
        versions = ['v1', 'v2']
        
        for version in versions:
            # Test health endpoint for each version
            async with api_client.get(f'http://localhost:8000/api/{version}/health') as response:
                if version == 'v1':
                    assert response.status == 200
                else:
                    # v2 might not exist yet
                    assert response.status in [200, 404]
    
    @pytest.mark.asyncio
    async def test_cross_browser_compatibility(self):
        """Test compatibility across different browsers."""
        browsers = ['chromium', 'firefox', 'webkit']
        
        async with async_playwright() as p:
            for browser_name in browsers:
                browser = await getattr(p, browser_name).launch(headless=True)
                context = await browser.new_context()
                page = await context.new_page()
                
                try:
                    await page.goto('http://localhost:3000')
                    assert await page.is_visible('body')
                    
                    # Basic functionality test
                    await page.goto('http://localhost:3000/login')
                    assert await page.is_visible('input[name="email"]')
                    assert await page.is_visible('input[name="password"]')
                except Exception as e:
                    pytest.fail(f"Browser {browser_name} failed: {str(e)}")
                finally:
                    await browser.close()
    
    @pytest.mark.asyncio
    async def test_database_transactions_and_rollback(self):
        """Test database transaction handling and rollback scenarios."""
        async with DatabaseSession() as session:
            try:
                # Start transaction
                async with session.begin():
                    # Create test user
                    user = User(
                        username='transaction_test',
                        email='transaction@test.com',
                        password_hash='hashed_password',
                        grade_level=10
                    )
                    session.add(user)
                    await session.flush()
                    
                    # Create related data
                    learning_path = LearningPath(
                        user_id=user.id,
                        title='Test Path',
                        description='Test Description'
                    )
                    session.add(learning_path)
                    
                    # Simulate error
                    raise Exception("Simulated error")
            except Exception:
                # Transaction should be rolled back
                pass
            
            # Verify rollback
            result = await session.execute(
                "SELECT * FROM users WHERE username = 'transaction_test'"
            )
            assert result.first() is None, "Transaction not rolled back properly"