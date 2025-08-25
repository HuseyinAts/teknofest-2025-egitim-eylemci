# -*- coding: utf-8 -*-
"""
Production-Ready End-to-End Tests for TEKNOFEST 2025 Education System
Complete test coverage for all system components and workflows
"""

import pytest
import json
import os
import sys
import asyncio
import aiohttp
import time
import random
import hashlib
import jwt
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import tempfile
import shutil

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import all system components
from src.api_server_with_rate_limit import create_app
from src.mcp_server.production_server import ProductionMCPServer
from src.agent_coordinator import AgentCoordinator
from src.event_manager import EventManager
from src.rate_limiter import RateLimiter
from src.multi_region_config import MultiRegionConfig
from src.cache_manager import CacheManager
from src.database_router import DatabaseRouter
from src.failover_manager import FailoverManager
from src.resource_manager import ResourceManager
from src.error_handler import ErrorHandler
from src.model_integration_cloud import ModelIntegration
from src.data_processor import DataProcessor
from src.agents.learning_path_agent_v2 import LearningPathAgent
from src.agents.study_buddy_agent import StudyBuddyAgent

# ========================= TEST CONFIGURATION =========================

TEST_CONFIG = {
    'api_base_url': 'http://localhost:5000',
    'mcp_server_port': 3000,
    'database': {
        'host': 'localhost',
        'port': 5432,
        'name': 'teknofest_test',
        'user': 'test_user',
        'password': 'test_password'
    },
    'redis': {
        'host': 'localhost',
        'port': 6379,
        'db': 1
    },
    'rate_limits': {
        'per_minute': 60,
        'per_hour': 1000,
        'per_day': 10000
    },
    'timeouts': {
        'api': 30,
        'database': 10,
        'cache': 5,
        'model': 60
    }
}

# ========================= FIXTURES =========================

@pytest.fixture(scope='session')
async def test_environment():
    """Setup complete test environment"""
    env = {
        'temp_dir': tempfile.mkdtemp(prefix='teknofest_test_'),
        'start_time': datetime.utcnow(),
        'test_data_dir': Path(__file__).parent / 'test_data'
    }
    
    # Create test data directory
    os.makedirs(env['test_data_dir'], exist_ok=True)
    
    yield env
    
    # Cleanup
    shutil.rmtree(env['temp_dir'], ignore_errors=True)


@pytest.fixture
async def app_instance():
    """Create Flask application instance"""
    app = create_app()
    app.config.update({
        'TESTING': True,
        'DEBUG': False,
        'RATE_LIMIT_ENABLED': True,
        'DATABASE_URI': f"postgresql://{TEST_CONFIG['database']['user']}:"
                       f"{TEST_CONFIG['database']['password']}@"
                       f"{TEST_CONFIG['database']['host']}:"
                       f"{TEST_CONFIG['database']['port']}/"
                       f"{TEST_CONFIG['database']['name']}",
        'REDIS_URL': f"redis://{TEST_CONFIG['redis']['host']}:"
                    f"{TEST_CONFIG['redis']['port']}/{TEST_CONFIG['redis']['db']}"
    })
    return app


@pytest.fixture
async def test_client(app_instance):
    """Create test client"""
    return app_instance.test_client()


@pytest.fixture
async def authenticated_client(test_client):
    """Create authenticated test client"""
    # Generate test token
    token = jwt.encode(
        {
            'user_id': 'test_user_123',
            'role': 'admin',
            'exp': datetime.utcnow() + timedelta(hours=1)
        },
        'test_secret_key',
        algorithm='HS256'
    )
    
    class AuthenticatedClient:
        def __init__(self, client, token):
            self.client = client
            self.token = token
            self.headers = {'Authorization': f'Bearer {token}'}
        
        def get(self, *args, **kwargs):
            kwargs.setdefault('headers', {}).update(self.headers)
            return self.client.get(*args, **kwargs)
        
        def post(self, *args, **kwargs):
            kwargs.setdefault('headers', {}).update(self.headers)
            return self.client.post(*args, **kwargs)
        
        def put(self, *args, **kwargs):
            kwargs.setdefault('headers', {}).update(self.headers)
            return self.client.put(*args, **kwargs)
        
        def delete(self, *args, **kwargs):
            kwargs.setdefault('headers', {}).update(self.headers)
            return self.client.delete(*args, **kwargs)
    
    return AuthenticatedClient(test_client, token)


@pytest.fixture
async def production_mcp_server():
    """Create production MCP server instance"""
    server = ProductionMCPServer(
        name="teknofest-mcp-test",
        version="1.0.0"
    )
    await server.initialize()
    yield server
    await server.shutdown()


@pytest.fixture
async def system_components():
    """Initialize all system components"""
    components = {
        'agent_coordinator': AgentCoordinator(),
        'event_manager': EventManager(),
        'rate_limiter': RateLimiter(
            requests_per_minute=TEST_CONFIG['rate_limits']['per_minute'],
            requests_per_hour=TEST_CONFIG['rate_limits']['per_hour']
        ),
        'cache_manager': CacheManager(
            redis_host=TEST_CONFIG['redis']['host'],
            redis_port=TEST_CONFIG['redis']['port']
        ),
        'database_router': DatabaseRouter(master_config=TEST_CONFIG['database']),
        'failover_manager': FailoverManager(),
        'resource_manager': ResourceManager(),
        'error_handler': ErrorHandler(),
        'model_integration': ModelIntegration(),
        'data_processor': DataProcessor()
    }
    
    # Initialize components
    for component in components.values():
        if hasattr(component, 'initialize'):
            await component.initialize()
    
    yield components
    
    # Cleanup
    for component in components.values():
        if hasattr(component, 'shutdown'):
            await component.shutdown()


# ========================= END-TO-END WORKFLOW TESTS =========================

class TestE2EStudentJourney:
    """End-to-end tests for complete student journey"""
    
    @pytest.mark.asyncio
    @pytest.mark.e2e
    async def test_complete_student_onboarding_flow(
        self, authenticated_client, system_components, production_mcp_server
    ):
        """Test complete student onboarding from registration to first lesson"""
        
        # Step 1: Register new student
        student_data = {
            'name': 'E2E Test Student',
            'email': f'e2e_test_{int(time.time())}@test.com',
            'grade': 9,
            'school': 'Test High School',
            'subjects': ['Matematik', 'Fizik', 'Kimya'],
            'learning_goals': 'Üniversite sınavına hazırlık'
        }
        
        response = authenticated_client.post('/api/v1/students/register', json=student_data)
        assert response.status_code == 201
        registration = json.loads(response.data)
        assert 'student_id' in registration
        assert 'profile_url' in registration
        
        student_id = registration['student_id']
        
        # Step 2: Complete initial assessment
        assessment_data = {
            'student_id': student_id,
            'assessment_type': 'initial',
            'responses': {
                'learning_style': ['visual', 'interactive'],
                'study_hours': 3,
                'difficulty_preference': 'challenging',
                'device_type': 'laptop'
            }
        }
        
        response = authenticated_client.post('/api/v1/assessment/initial', json=assessment_data)
        assert response.status_code == 200
        assessment_result = json.loads(response.data)
        assert 'learning_profile' in assessment_result
        assert 'recommendations' in assessment_result
        
        # Step 3: Generate personalized learning path
        learning_path_request = {
            'student_id': student_id,
            'duration_weeks': 12,
            'focus_areas': ['Matematik', 'Fizik'],
            'target_exam': 'YKS',
            'current_level': assessment_result['learning_profile']['current_level']
        }
        
        response = authenticated_client.post('/api/v1/learning-path/generate', json=learning_path_request)
        assert response.status_code == 200
        learning_path = json.loads(response.data)
        assert 'path_id' in learning_path
        assert 'weekly_plan' in learning_path
        assert len(learning_path['weekly_plan']) == 12
        
        # Step 4: Access first week's content
        response = authenticated_client.get(f'/api/v1/learning-path/{student_id}/week/1')
        assert response.status_code == 200
        week_content = json.loads(response.data)
        assert 'topics' in week_content
        assert 'exercises' in week_content
        assert 'estimated_hours' in week_content
        
        # Step 5: Start first lesson
        lesson_request = {
            'student_id': student_id,
            'topic_id': week_content['topics'][0]['id'],
            'session_type': 'interactive'
        }
        
        response = authenticated_client.post('/api/v1/lessons/start', json=lesson_request)
        assert response.status_code == 200
        lesson = json.loads(response.data)
        assert 'session_id' in lesson
        assert 'content_url' in lesson
        assert 'interactive_elements' in lesson
        
        # Step 6: Submit lesson progress
        progress_data = {
            'session_id': lesson['session_id'],
            'progress_percentage': 50,
            'time_spent': 1800,  # 30 minutes
            'interactions': [
                {'type': 'video_watched', 'duration': 600},
                {'type': 'exercise_attempted', 'result': 'correct'}
            ]
        }
        
        response = authenticated_client.post('/api/v1/lessons/progress', json=progress_data)
        assert response.status_code == 200
        
        # Step 7: Generate practice quiz
        quiz_request = {
            'student_id': student_id,
            'topic_id': week_content['topics'][0]['id'],
            'difficulty': 'adaptive',
            'num_questions': 10
        }
        
        response = authenticated_client.post('/api/v1/quiz/generate', json=quiz_request)
        assert response.status_code == 200
        quiz = json.loads(response.data)
        assert 'quiz_id' in quiz
        assert 'questions' in quiz
        assert len(quiz['questions']) == 10
        
        # Step 8: Submit quiz answers
        answers = []
        for question in quiz['questions']:
            answers.append({
                'question_id': question['id'],
                'answer': random.choice(question['options']),
                'time_spent': random.randint(30, 180)
            })
        
        submission = {
            'quiz_id': quiz['quiz_id'],
            'student_id': student_id,
            'answers': answers,
            'total_time': sum(a['time_spent'] for a in answers)
        }
        
        response = authenticated_client.post('/api/v1/quiz/submit', json=submission)
        assert response.status_code == 200
        results = json.loads(response.data)
        assert 'score' in results
        assert 'feedback' in results
        assert 'recommendations' in results
        
        # Step 9: Get progress report
        response = authenticated_client.get(f'/api/v1/students/{student_id}/progress')
        assert response.status_code == 200
        progress = json.loads(response.data)
        assert 'overall_progress' in progress
        assert 'subjects' in progress
        assert 'achievements' in progress
        assert 'next_steps' in progress
        
        logger.info(f"✅ Complete student onboarding flow successful for {student_id}")
    
    @pytest.mark.asyncio
    @pytest.mark.e2e
    async def test_adaptive_learning_experience(
        self, authenticated_client, system_components, production_mcp_server
    ):
        """Test adaptive learning system adjusting to student performance"""
        
        # Create test student with known profile
        student_id = f'adaptive_test_{int(time.time())}'
        
        # Simulate multiple learning sessions with varying performance
        performance_history = []
        
        for session_num in range(5):
            # Generate adaptive content based on previous performance
            content_request = {
                'student_id': student_id,
                'session_number': session_num + 1,
                'previous_performance': performance_history[-1] if performance_history else None
            }
            
            response = authenticated_client.post('/api/v1/adaptive/content', json=content_request)
            assert response.status_code == 200
            content = json.loads(response.data)
            
            # Verify difficulty adjustment
            if session_num > 0:
                if performance_history[-1]['score'] > 80:
                    assert content['difficulty'] > performance_history[-1]['difficulty']
                elif performance_history[-1]['score'] < 50:
                    assert content['difficulty'] < performance_history[-1]['difficulty']
            
            # Simulate performance
            performance = {
                'session_id': content['session_id'],
                'score': random.randint(40, 95),
                'difficulty': content['difficulty'],
                'topics_mastered': random.sample(content['topics'], k=random.randint(1, len(content['topics'])))
            }
            
            response = authenticated_client.post('/api/v1/adaptive/performance', json=performance)
            assert response.status_code == 200
            
            performance_history.append(performance)
        
        # Verify learning path adaptation
        response = authenticated_client.get(f'/api/v1/adaptive/{student_id}/analysis')
        assert response.status_code == 200
        analysis = json.loads(response.data)
        
        assert 'learning_curve' in analysis
        assert 'difficulty_progression' in analysis
        assert 'mastery_levels' in analysis
        assert 'recommendations' in analysis
        
        logger.info(f"✅ Adaptive learning test completed with {len(performance_history)} sessions")
    
    @pytest.mark.asyncio
    @pytest.mark.e2e
    async def test_collaborative_learning_session(
        self, authenticated_client, system_components, production_mcp_server
    ):
        """Test collaborative learning features with multiple students"""
        
        # Create study group
        group_data = {
            'name': 'E2E Test Study Group',
            'subject': 'Matematik',
            'max_members': 5,
            'study_schedule': 'daily',
            'creator_id': 'student_001'
        }
        
        response = authenticated_client.post('/api/v1/groups/create', json=group_data)
        assert response.status_code == 201
        group = json.loads(response.data)
        group_id = group['group_id']
        
        # Add multiple students
        students = []
        for i in range(4):
            student_id = f'collab_student_{i}'
            join_request = {
                'group_id': group_id,
                'student_id': student_id,
                'role': 'member'
            }
            response = authenticated_client.post('/api/v1/groups/join', json=join_request)
            assert response.status_code == 200
            students.append(student_id)
        
        # Start collaborative session
        session_data = {
            'group_id': group_id,
            'topic': 'Diferansiyel Denklemler',
            'session_type': 'problem_solving',
            'duration_minutes': 60
        }
        
        response = authenticated_client.post('/api/v1/groups/session/start', json=session_data)
        assert response.status_code == 200
        session = json.loads(response.data)
        session_id = session['session_id']
        
        # Simulate student interactions
        interactions = []
        for student_id in students:
            interaction = {
                'session_id': session_id,
                'student_id': student_id,
                'action': random.choice(['question_asked', 'answer_provided', 'resource_shared']),
                'content': f'Interaction from {student_id}',
                'timestamp': datetime.utcnow().isoformat()
            }
            response = authenticated_client.post('/api/v1/groups/session/interact', json=interaction)
            assert response.status_code == 200
            interactions.append(interaction)
        
        # End session and get summary
        response = authenticated_client.post(f'/api/v1/groups/session/{session_id}/end')
        assert response.status_code == 200
        summary = json.loads(response.data)
        
        assert 'participation_stats' in summary
        assert 'learning_outcomes' in summary
        assert 'collaboration_score' in summary
        assert len(summary['participation_stats']) == len(students)
        
        logger.info(f"✅ Collaborative session completed with {len(students)} students")


class TestE2ESystemIntegration:
    """End-to-end tests for system integration"""
    
    @pytest.mark.asyncio
    @pytest.mark.e2e
    async def test_mcp_agent_orchestration(
        self, authenticated_client, production_mcp_server, system_components
    ):
        """Test MCP server orchestrating multiple agents"""
        
        # Register agents with MCP server
        agents = ['learning_path', 'study_buddy', 'assessment', 'content_generator']
        
        for agent_type in agents:
            await production_mcp_server.register_agent(agent_type, {
                'capabilities': [f'{agent_type}_task'],
                'max_concurrent': 5
            })
        
        # Create complex task requiring multiple agents
        complex_task = {
            'task_id': f'complex_{int(time.time())}',
            'type': 'comprehensive_learning_plan',
            'requirements': {
                'student_id': 'test_student_123',
                'subjects': ['Matematik', 'Fizik'],
                'duration': '3_months',
                'goal': 'exam_preparation'
            },
            'steps': [
                {'agent': 'assessment', 'action': 'evaluate_current_level'},
                {'agent': 'learning_path', 'action': 'generate_curriculum'},
                {'agent': 'content_generator', 'action': 'create_materials'},
                {'agent': 'study_buddy', 'action': 'setup_support'}
            ]
        }
        
        # Execute task through MCP
        response = await production_mcp_server.handle_request({
            'jsonrpc': '2.0',
            'method': 'execute_task',
            'params': complex_task,
            'id': 1
        })
        
        assert response['jsonrpc'] == '2.0'
        assert 'result' in response
        assert response['result']['status'] == 'completed'
        assert len(response['result']['step_results']) == 4
        
        # Verify each step completed
        for i, step_result in enumerate(response['result']['step_results']):
            assert step_result['agent'] == complex_task['steps'][i]['agent']
            assert step_result['status'] == 'success'
            assert 'output' in step_result
        
        logger.info("✅ MCP agent orchestration test completed successfully")
    
    @pytest.mark.asyncio
    @pytest.mark.e2e
    async def test_event_driven_workflow(
        self, authenticated_client, system_components
    ):
        """Test event-driven workflow across system components"""
        
        event_manager = system_components['event_manager']
        events_captured = []
        
        # Setup event listeners
        async def capture_event(event):
            events_captured.append(event)
            logger.info(f"Event captured: {event['type']}")
        
        event_types = [
            'student.registered',
            'assessment.completed',
            'learning_path.generated',
            'content.accessed',
            'quiz.submitted',
            'progress.updated'
        ]
        
        for event_type in event_types:
            event_manager.subscribe(event_type, capture_event)
        
        # Execute workflow that triggers events
        workflow_data = {
            'student_name': 'Event Test Student',
            'email': f'event_test_{int(time.time())}@test.com',
            'grade': 10
        }
        
        # Trigger registration (should emit student.registered)
        response = authenticated_client.post('/api/v1/students/register', json=workflow_data)
        assert response.status_code == 201
        student_data = json.loads(response.data)
        
        # Trigger assessment (should emit assessment.completed)
        response = authenticated_client.post('/api/v1/assessment/quick', json={
            'student_id': student_data['student_id'],
            'subjects': ['Matematik']
        })
        assert response.status_code == 200
        
        # Wait for async events
        await asyncio.sleep(0.5)
        
        # Verify events were captured
        assert len(events_captured) >= 2
        event_types_captured = [e['type'] for e in events_captured]
        assert 'student.registered' in event_types_captured
        assert 'assessment.completed' in event_types_captured
        
        logger.info(f"✅ Event-driven workflow test captured {len(events_captured)} events")
    
    @pytest.mark.asyncio
    @pytest.mark.e2e
    async def test_multi_region_failover(
        self, authenticated_client, system_components
    ):
        """Test multi-region setup with automatic failover"""
        
        failover_manager = system_components['failover_manager']
        
        # Configure multiple regions
        regions = [
            {'name': 'primary', 'endpoint': 'http://primary.test.com', 'priority': 1},
            {'name': 'secondary', 'endpoint': 'http://secondary.test.com', 'priority': 2},
            {'name': 'tertiary', 'endpoint': 'http://tertiary.test.com', 'priority': 3}
        ]
        
        for region in regions:
            failover_manager.add_region(region)
        
        # Verify primary is active
        active = failover_manager.get_active_region()
        assert active['name'] == 'primary'
        
        # Simulate primary failure
        failover_manager.report_failure('primary', 'Connection timeout')
        
        # Verify automatic failover to secondary
        active = failover_manager.get_active_region()
        assert active['name'] == 'secondary'
        
        # Simulate recovery of primary
        failover_manager.report_recovery('primary')
        
        # Verify failback to primary (if configured)
        if failover_manager.auto_failback_enabled:
            await asyncio.sleep(failover_manager.failback_delay)
            active = failover_manager.get_active_region()
            assert active['name'] == 'primary'
        
        logger.info("✅ Multi-region failover test completed successfully")


class TestE2EPerformanceAndScale:
    """End-to-end performance and scalability tests"""
    
    @pytest.mark.asyncio
    @pytest.mark.e2e
    @pytest.mark.performance
    async def test_concurrent_user_load(
        self, authenticated_client, system_components
    ):
        """Test system performance under concurrent user load"""
        
        num_concurrent_users = 50
        operations_per_user = 10
        
        async def simulate_user_session(user_id: int):
            """Simulate a complete user session"""
            results = {
                'user_id': user_id,
                'operations': [],
                'errors': []
            }
            
            try:
                # Register user
                response = authenticated_client.post('/api/v1/students/register', json={
                    'name': f'Load Test User {user_id}',
                    'email': f'load_test_{user_id}@test.com',
                    'grade': 9
                })
                results['operations'].append({
                    'type': 'registration',
                    'status': response.status_code,
                    'duration': response.elapsed_total_seconds() if hasattr(response, 'elapsed_total_seconds') else 0
                })
                
                if response.status_code != 201:
                    results['errors'].append(f"Registration failed: {response.status_code}")
                    return results
                
                student_data = json.loads(response.data)
                student_id = student_data['student_id']
                
                # Perform various operations
                for op in range(operations_per_user):
                    operation_type = random.choice([
                        'get_profile',
                        'update_progress',
                        'access_content',
                        'submit_quiz'
                    ])
                    
                    start_time = time.time()
                    
                    if operation_type == 'get_profile':
                        response = authenticated_client.get(f'/api/v1/students/{student_id}')
                    elif operation_type == 'update_progress':
                        response = authenticated_client.post('/api/v1/progress/update', json={
                            'student_id': student_id,
                            'topic': 'test_topic',
                            'progress': random.randint(0, 100)
                        })
                    elif operation_type == 'access_content':
                        response = authenticated_client.get(f'/api/v1/content/random')
                    else:  # submit_quiz
                        response = authenticated_client.post('/api/v1/quiz/quick-submit', json={
                            'student_id': student_id,
                            'score': random.randint(0, 100)
                        })
                    
                    duration = time.time() - start_time
                    
                    results['operations'].append({
                        'type': operation_type,
                        'status': response.status_code,
                        'duration': duration
                    })
                    
                    if response.status_code >= 400:
                        results['errors'].append(f"{operation_type} failed: {response.status_code}")
                    
                    # Small delay between operations
                    await asyncio.sleep(random.uniform(0.1, 0.5))
                
            except Exception as e:
                results['errors'].append(f"Exception: {str(e)}")
            
            return results
        
        # Run concurrent user sessions
        start_time = time.time()
        tasks = [simulate_user_session(i) for i in range(num_concurrent_users)]
        results = await asyncio.gather(*tasks)
        total_duration = time.time() - start_time
        
        # Analyze results
        total_operations = sum(len(r['operations']) for r in results)
        total_errors = sum(len(r['errors']) for r in results)
        successful_operations = sum(
            1 for r in results 
            for op in r['operations'] 
            if op['status'] < 400
        )
        
        avg_response_time = sum(
            op['duration'] for r in results 
            for op in r['operations']
        ) / total_operations if total_operations > 0 else 0
        
        # Performance assertions
        success_rate = successful_operations / total_operations if total_operations > 0 else 0
        assert success_rate > 0.95, f"Success rate {success_rate:.2%} below threshold"
        assert avg_response_time < 2.0, f"Average response time {avg_response_time:.2f}s too high"
        assert total_duration < 60, f"Total test duration {total_duration:.2f}s exceeded limit"
        
        logger.info(f"""
        ✅ Concurrent Load Test Results:
        - Total Users: {num_concurrent_users}
        - Total Operations: {total_operations}
        - Successful Operations: {successful_operations}
        - Success Rate: {success_rate:.2%}
        - Average Response Time: {avg_response_time:.3f}s
        - Total Duration: {total_duration:.2f}s
        - Errors: {total_errors}
        """)
    
    @pytest.mark.asyncio
    @pytest.mark.e2e
    @pytest.mark.performance
    async def test_database_transaction_performance(
        self, system_components
    ):
        """Test database transaction performance and integrity"""
        
        database_router = system_components['database_router']
        
        # Test concurrent transactions
        num_transactions = 100
        
        async def perform_transaction(tx_id: int):
            """Perform a database transaction"""
            try:
                async with database_router.transaction() as tx:
                    # Simulate complex transaction
                    await tx.execute(
                        "INSERT INTO test_transactions (id, data) VALUES ($1, $2)",
                        tx_id, f"transaction_data_{tx_id}"
                    )
                    
                    # Simulate processing
                    await asyncio.sleep(random.uniform(0.01, 0.05))
                    
                    # Update
                    await tx.execute(
                        "UPDATE test_transactions SET processed = true WHERE id = $1",
                        tx_id
                    )
                    
                    return {'success': True, 'tx_id': tx_id}
            except Exception as e:
                return {'success': False, 'tx_id': tx_id, 'error': str(e)}
        
        # Run concurrent transactions
        start_time = time.time()
        tasks = [perform_transaction(i) for i in range(num_transactions)]
        results = await asyncio.gather(*tasks)
        duration = time.time() - start_time
        
        # Verify results
        successful = [r for r in results if r['success']]
        failed = [r for r in results if not r['success']]
        
        success_rate = len(successful) / num_transactions
        tx_per_second = num_transactions / duration
        
        assert success_rate > 0.99, f"Transaction success rate {success_rate:.2%} too low"
        assert tx_per_second > 10, f"Transaction throughput {tx_per_second:.2f} tx/s too low"
        
        logger.info(f"""
        ✅ Database Transaction Test Results:
        - Total Transactions: {num_transactions}
        - Successful: {len(successful)}
        - Failed: {len(failed)}
        - Success Rate: {success_rate:.2%}
        - Throughput: {tx_per_second:.2f} tx/s
        - Duration: {duration:.2f}s
        """)
    
    @pytest.mark.asyncio
    @pytest.mark.e2e
    @pytest.mark.performance
    async def test_cache_efficiency(
        self, authenticated_client, system_components
    ):
        """Test cache efficiency and hit rates"""
        
        cache_manager = system_components['cache_manager']
        
        # Clear cache
        await cache_manager.clear()
        
        # Generate test data
        test_keys = [f"test_key_{i}" for i in range(100)]
        test_data = {key: f"value_{key}_{random.randint(1000, 9999)}" for key in test_keys}
        
        # Populate cache
        for key, value in test_data.items():
            await cache_manager.set(key, value, ttl=300)
        
        # Test cache hits
        hits = 0
        misses = 0
        
        for _ in range(1000):
            key = random.choice(test_keys + ['non_existent_key'])
            result = await cache_manager.get(key)
            
            if result is not None:
                hits += 1
                # Verify correct data
                if key in test_data:
                    assert result == test_data[key]
            else:
                misses += 1
        
        hit_rate = hits / (hits + misses)
        assert hit_rate > 0.85, f"Cache hit rate {hit_rate:.2%} below threshold"
        
        # Test cache eviction
        await cache_manager.set_max_size(50)
        await cache_manager.evict_lru()
        
        remaining_keys = await cache_manager.keys()
        assert len(remaining_keys) <= 50
        
        logger.info(f"""
        ✅ Cache Efficiency Test Results:
        - Total Requests: {hits + misses}
        - Cache Hits: {hits}
        - Cache Misses: {misses}
        - Hit Rate: {hit_rate:.2%}
        - Keys After Eviction: {len(remaining_keys)}
        """)


class TestE2ESecurity:
    """End-to-end security tests"""
    
    @pytest.mark.asyncio
    @pytest.mark.e2e
    @pytest.mark.security
    async def test_authentication_and_authorization(
        self, test_client
    ):
        """Test complete authentication and authorization flow"""
        
        # Test unauthorized access
        response = test_client.get('/api/v1/admin/dashboard')
        assert response.status_code == 401
        
        # Register new user
        registration = {
            'username': f'security_test_{int(time.time())}',
            'password': 'SecureP@ssw0rd123!',
            'email': f'security_{int(time.time())}@test.com',
            'role': 'student'
        }
        
        response = test_client.post('/api/v1/auth/register', json=registration)
        assert response.status_code == 201
        
        # Login with credentials
        response = test_client.post('/api/v1/auth/login', json={
            'username': registration['username'],
            'password': registration['password']
        })
        assert response.status_code == 200
        auth_data = json.loads(response.data)
        assert 'access_token' in auth_data
        assert 'refresh_token' in auth_data
        
        # Test with valid token
        headers = {'Authorization': f'Bearer {auth_data["access_token"]}'}
        response = test_client.get('/api/v1/profile', headers=headers)
        assert response.status_code == 200
        
        # Test role-based access (student shouldn't access admin)
        response = test_client.get('/api/v1/admin/dashboard', headers=headers)
        assert response.status_code == 403
        
        # Test token refresh
        response = test_client.post('/api/v1/auth/refresh', json={
            'refresh_token': auth_data['refresh_token']
        })
        assert response.status_code == 200
        new_auth = json.loads(response.data)
        assert 'access_token' in new_auth
        
        # Test logout
        response = test_client.post('/api/v1/auth/logout', headers=headers)
        assert response.status_code == 200
        
        # Verify token is invalidated
        response = test_client.get('/api/v1/profile', headers=headers)
        assert response.status_code == 401
        
        logger.info("✅ Authentication and authorization test completed")
    
    @pytest.mark.asyncio
    @pytest.mark.e2e
    @pytest.mark.security
    async def test_input_validation_and_sanitization(
        self, authenticated_client
    ):
        """Test input validation against various attack vectors"""
        
        attack_vectors = [
            # SQL Injection attempts
            {
                'name': "'; DROP TABLE students; --",
                'expected_status': [400, 422]
            },
            # XSS attempts
            {
                'name': '<script>alert("XSS")</script>',
                'expected_status': [400, 422]
            },
            # Command injection
            {
                'name': '$(rm -rf /)',
                'expected_status': [400, 422]
            },
            # Path traversal
            {
                'file_path': '../../../etc/passwd',
                'expected_status': [400, 403]
            },
            # JSON injection
            {
                'data': '{"extra": "field", "__proto__": {"isAdmin": true}}',
                'expected_status': [400, 422]
            }
        ]
        
        for vector in attack_vectors:
            response = authenticated_client.post('/api/v1/students/update', json=vector)
            assert response.status_code in vector['expected_status'], \
                f"Failed to block attack vector: {vector}"
        
        logger.info(f"✅ Input validation test passed for {len(attack_vectors)} attack vectors")
    
    @pytest.mark.asyncio
    @pytest.mark.e2e
    @pytest.mark.security
    async def test_rate_limiting_and_ddos_protection(
        self, test_client
    ):
        """Test rate limiting and DDoS protection mechanisms"""
        
        # Test rate limiting per IP
        responses = []
        for i in range(100):
            response = test_client.get('/api/v1/public/info', 
                                      headers={'X-Forwarded-For': '192.168.1.1'})
            responses.append(response.status_code)
        
        # Should have some rate limited responses
        rate_limited = [r for r in responses if r == 429]
        assert len(rate_limited) > 0, "Rate limiting not working"
        
        # Test different IPs have separate limits
        response = test_client.get('/api/v1/public/info',
                                  headers={'X-Forwarded-For': '192.168.1.2'})
        assert response.status_code != 429
        
        # Test burst protection
        burst_responses = []
        async def burst_request():
            return test_client.get('/api/v1/public/info')
        
        tasks = [burst_request() for _ in range(50)]
        burst_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Count successful vs rate limited
        successful = sum(1 for r in burst_results if not isinstance(r, Exception) and r.status_code == 200)
        limited = sum(1 for r in burst_results if not isinstance(r, Exception) and r.status_code == 429)
        
        assert limited > 0, "Burst protection not working"
        
        logger.info(f"""
        ✅ Rate Limiting Test Results:
        - Sequential Requests: {len(responses)}
        - Rate Limited: {len(rate_limited)}
        - Burst Requests: {len(burst_results)}
        - Burst Successful: {successful}
        - Burst Limited: {limited}
        """)


class TestE2EMonitoringAndObservability:
    """End-to-end monitoring and observability tests"""
    
    @pytest.mark.asyncio
    @pytest.mark.e2e
    async def test_health_check_system(
        self, authenticated_client, system_components
    ):
        """Test comprehensive health check system"""
        
        # Get overall health
        response = authenticated_client.get('/api/v1/health')
        assert response.status_code == 200
        health = json.loads(response.data)
        
        assert 'status' in health
        assert 'timestamp' in health
        assert 'version' in health
        
        # Get detailed health
        response = authenticated_client.get('/api/v1/health/detailed')
        assert response.status_code == 200
        detailed = json.loads(response.data)
        
        required_components = [
            'api', 'database', 'cache', 'mcp_server',
            'agents', 'model', 'storage'
        ]
        
        for component in required_components:
            assert component in detailed['components']
            component_health = detailed['components'][component]
            assert 'status' in component_health
            assert 'latency' in component_health
            assert component_health['status'] in ['healthy', 'degraded', 'unhealthy']
        
        logger.info(f"✅ Health check system verified for {len(required_components)} components")
    
    @pytest.mark.asyncio
    @pytest.mark.e2e
    async def test_metrics_and_telemetry(
        self, authenticated_client, system_components
    ):
        """Test metrics collection and telemetry"""
        
        resource_manager = system_components['resource_manager']
        
        # Generate some activity
        for _ in range(10):
            authenticated_client.get('/api/v1/students/test')
            authenticated_client.post('/api/v1/quiz/generate', json={'topic': 'test'})
        
        # Get metrics
        response = authenticated_client.get('/api/v1/metrics')
        assert response.status_code == 200
        metrics = json.loads(response.data)
        
        # Verify metric categories
        assert 'api' in metrics
        assert 'system' in metrics
        assert 'business' in metrics
        
        # Verify API metrics
        api_metrics = metrics['api']
        assert 'request_count' in api_metrics
        assert 'response_time' in api_metrics
        assert 'error_rate' in api_metrics
        
        # Verify system metrics
        system_metrics = metrics['system']
        assert 'cpu_usage' in system_metrics
        assert 'memory_usage' in system_metrics
        assert 'disk_usage' in system_metrics
        
        # Verify business metrics
        business_metrics = metrics['business']
        assert 'active_users' in business_metrics
        assert 'quizzes_generated' in business_metrics
        
        logger.info("✅ Metrics and telemetry system verified")
    
    @pytest.mark.asyncio
    @pytest.mark.e2e
    async def test_logging_and_audit_trail(
        self, authenticated_client, system_components
    ):
        """Test logging and audit trail functionality"""
        
        event_manager = system_components['event_manager']
        audit_events = []
        
        # Setup audit listener
        def audit_handler(event):
            if event.get('category') == 'audit':
                audit_events.append(event)
        
        event_manager.subscribe('audit.*', audit_handler)
        
        # Perform auditable actions
        test_user = f'audit_test_{int(time.time())}'
        
        # Create user (should be audited)
        response = authenticated_client.post('/api/v1/auth/register', json={
            'username': test_user,
            'password': 'Test123!',
            'email': f'{test_user}@test.com'
        })
        assert response.status_code == 201
        
        # Login (should be audited)
        response = authenticated_client.post('/api/v1/auth/login', json={
            'username': test_user,
            'password': 'Test123!'
        })
        assert response.status_code == 200
        
        # Access sensitive data (should be audited)
        token = json.loads(response.data)['access_token']
        headers = {'Authorization': f'Bearer {token}'}
        response = authenticated_client.get('/api/v1/admin/users', headers=headers)
        
        # Wait for async processing
        await asyncio.sleep(0.5)
        
        # Verify audit trail
        assert len(audit_events) >= 2
        
        for event in audit_events:
            assert 'timestamp' in event
            assert 'user' in event
            assert 'action' in event
            assert 'resource' in event
        
        logger.info(f"✅ Audit trail captured {len(audit_events)} events")


class TestE2EDisasterRecovery:
    """End-to-end disaster recovery tests"""
    
    @pytest.mark.asyncio
    @pytest.mark.e2e
    @pytest.mark.slow
    async def test_backup_and_restore(
        self, system_components, test_environment
    ):
        """Test backup and restore procedures"""
        
        database_router = system_components['database_router']
        
        # Create test data
        test_data = {
            'students': [f'student_{i}' for i in range(10)],
            'quizzes': [f'quiz_{i}' for i in range(20)],
            'progress': [f'progress_{i}' for i in range(30)]
        }
        
        # Insert test data
        for table, items in test_data.items():
            for item in items:
                await database_router.execute(
                    f"INSERT INTO {table} (id, data) VALUES ($1, $2)",
                    item, f"data_for_{item}"
                )
        
        # Perform backup
        backup_path = Path(test_environment['temp_dir']) / 'backup.sql'
        backup_result = await database_router.backup(backup_path)
        assert backup_result['success']
        assert backup_path.exists()
        
        # Simulate disaster - clear database
        for table in test_data.keys():
            await database_router.execute(f"TRUNCATE TABLE {table}")
        
        # Verify data is gone
        result = await database_router.fetch("SELECT COUNT(*) FROM students")
        assert result[0]['count'] == 0
        
        # Restore from backup
        restore_result = await database_router.restore(backup_path)
        assert restore_result['success']
        
        # Verify data is restored
        for table, items in test_data.items():
            result = await database_router.fetch(f"SELECT COUNT(*) FROM {table}")
            assert result[0]['count'] == len(items)
        
        logger.info("✅ Backup and restore test completed successfully")
    
    @pytest.mark.asyncio
    @pytest.mark.e2e
    async def test_graceful_degradation(
        self, authenticated_client, system_components
    ):
        """Test system graceful degradation under component failures"""
        
        # Normal operation
        response = authenticated_client.get('/api/v1/health')
        assert response.status_code == 200
        initial_health = json.loads(response.data)
        assert initial_health['status'] == 'healthy'
        
        # Simulate cache failure
        cache_manager = system_components['cache_manager']
        cache_manager.simulate_failure()
        
        # System should still work but degraded
        response = authenticated_client.get('/api/v1/health')
        assert response.status_code == 200
        degraded_health = json.loads(response.data)
        assert degraded_health['status'] in ['degraded', 'healthy']
        
        # Core functionality should still work
        response = authenticated_client.get('/api/v1/students/test')
        assert response.status_code in [200, 503]
        
        # Restore cache
        cache_manager.recover()
        
        # Verify recovery
        response = authenticated_client.get('/api/v1/health')
        assert response.status_code == 200
        recovered_health = json.loads(response.data)
        assert recovered_health['status'] == 'healthy'
        
        logger.info("✅ Graceful degradation test completed")


# ========================= TEST EXECUTION =========================

def run_e2e_tests():
    """Run all end-to-end tests"""
    pytest.main([
        __file__,
        '-v',
        '--tb=short',
        '--asyncio-mode=auto',
        '-m', 'e2e',
        '--junit-xml=test_results/e2e_results.xml',
        '--html=test_results/e2e_report.html',
        '--self-contained-html',
        '--cov=src',
        '--cov-report=html:test_results/coverage',
        '--cov-report=term-missing'
    ])


if __name__ == "__main__":
    run_e2e_tests()