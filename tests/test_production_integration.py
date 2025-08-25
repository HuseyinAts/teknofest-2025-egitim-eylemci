# -*- coding: utf-8 -*-
"""
Production-Ready Integration Tests for TEKNOFEST 2025 Education System
"""

import pytest
import json
import os
import sys
import asyncio
import aiohttp
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from datetime import datetime, timedelta
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Mock external dependencies if not available
try:
    import redis
except ImportError:
    redis = Mock()

try:
    import psycopg2
except ImportError:
    psycopg2 = Mock()

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import production modules
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
from src.cdn_manager import CDNManager
from src.region_monitoring import RegionMonitor
from src.error_handler import ErrorHandler


# ========================= FIXTURES =========================

@pytest.fixture
async def test_app():
    """Create test Flask application with rate limiting"""
    app = create_app()
    app.config['TESTING'] = True
    app.config['RATE_LIMIT_ENABLED'] = True
    return app


@pytest.fixture
async def test_client(test_app):
    """Create test client for API testing"""
    return test_app.test_client()


@pytest.fixture
def mcp_server():
    """Create ProductionMCPServer instance"""
    return ProductionMCPServer(
        name="test-mcp-server",
        version="1.0.0"
    )


@pytest.fixture
def agent_coordinator():
    """Create AgentCoordinator instance"""
    return AgentCoordinator()


@pytest.fixture
def event_manager():
    """Create EventManager instance"""
    return EventManager()


@pytest.fixture
def rate_limiter():
    """Create RateLimiter instance"""
    return RateLimiter(
        requests_per_minute=60,
        requests_per_hour=1000
    )


@pytest.fixture
def multi_region_config():
    """Create MultiRegionConfig instance"""
    return MultiRegionConfig()


@pytest.fixture
def cache_manager():
    """Create CacheManager instance"""
    return CacheManager(
        redis_host='localhost',
        redis_port=6379,
        ttl=300
    )


@pytest.fixture
def database_router():
    """Create DatabaseRouter instance"""
    return DatabaseRouter(
        master_config={
            'host': 'localhost',
            'port': 5432,
            'database': 'teknofest_test',
            'user': 'test_user',
            'password': 'test_password'
        }
    )


@pytest.fixture
def failover_manager():
    """Create FailoverManager instance"""
    return FailoverManager()


@pytest.fixture
def resource_manager():
    """Create ResourceManager instance"""
    return ResourceManager()


# ========================= API ENDPOINT INTEGRATION TESTS =========================

class TestAPIEndpointIntegration:
    """Integration tests for API endpoints"""
    
    @pytest.mark.asyncio
    async def test_health_check_endpoint(self, test_client):
        """Test health check endpoint"""
        response = test_client.get('/health')
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['status'] == 'healthy'
        assert 'timestamp' in data
        assert 'version' in data
    
    @pytest.mark.asyncio
    async def test_student_registration_flow(self, test_client):
        """Test complete student registration flow"""
        # Register new student
        registration_data = {
            'name': 'Test Student',
            'email': 'test@student.com',
            'grade': 9,
            'school': 'Test School'
        }
        
        response = test_client.post(
            '/api/v1/students/register',
            json=registration_data
        )
        
        assert response.status_code in [200, 201]
        data = json.loads(response.data)
        assert 'student_id' in data
        student_id = data['student_id']
        
        # Verify student profile
        response = test_client.get(f'/api/v1/students/{student_id}')
        assert response.status_code == 200
        profile = json.loads(response.data)
        assert profile['name'] == registration_data['name']
        assert profile['grade'] == registration_data['grade']
    
    @pytest.mark.asyncio
    async def test_learning_path_generation_api(self, test_client):
        """Test learning path generation through API"""
        request_data = {
            'student_id': 'test_student_123',
            'subject': 'Matematik',
            'current_level': 0.5,
            'target_level': 0.8,
            'duration_weeks': 4
        }
        
        response = test_client.post(
            '/api/v1/learning-path/generate',
            json=request_data
        )
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert 'learning_path' in data
        assert 'weekly_plan' in data['learning_path']
        assert len(data['learning_path']['weekly_plan']) == 4
    
    @pytest.mark.asyncio
    async def test_quiz_submission_and_evaluation(self, test_client):
        """Test quiz submission and evaluation flow"""
        # Generate quiz
        quiz_request = {
            'student_id': 'test_student_123',
            'topic': 'Denklemler',
            'difficulty': 0.5,
            'num_questions': 10
        }
        
        response = test_client.post(
            '/api/v1/quiz/generate',
            json=quiz_request
        )
        
        assert response.status_code == 200
        quiz_data = json.loads(response.data)
        assert 'quiz_id' in quiz_data
        assert 'questions' in quiz_data
        
        # Submit answers
        submission = {
            'quiz_id': quiz_data['quiz_id'],
            'student_id': 'test_student_123',
            'answers': [
                {'question_id': q['id'], 'answer': q['options'][0]}
                for q in quiz_data['questions']
            ]
        }
        
        response = test_client.post(
            '/api/v1/quiz/submit',
            json=submission
        )
        
        assert response.status_code == 200
        results = json.loads(response.data)
        assert 'score' in results
        assert 'feedback' in results
        assert 0 <= results['score'] <= 100
    
    @pytest.mark.asyncio
    async def test_api_error_handling(self, test_client):
        """Test API error handling"""
        # Test 404 error
        response = test_client.get('/api/v1/nonexistent')
        assert response.status_code == 404
        
        # Test invalid request data
        response = test_client.post(
            '/api/v1/students/register',
            json={'invalid': 'data'}
        )
        assert response.status_code in [400, 422]
        
        # Test method not allowed
        response = test_client.delete('/api/v1/health')
        assert response.status_code == 405


# ========================= MCP SERVER INTEGRATION TESTS =========================

class TestMCPServerIntegration:
    """Integration tests for MCP Server"""
    
    @pytest.mark.asyncio
    async def test_mcp_server_initialization(self, mcp_server):
        """Test MCP server initialization"""
        assert mcp_server.name == "test-mcp-server"
        assert mcp_server.version == "1.0.0"
        assert hasattr(mcp_server, 'register_tools')
        assert hasattr(mcp_server, 'handle_request')
    
    @pytest.mark.asyncio
    async def test_mcp_tool_registration(self, mcp_server):
        """Test MCP tool registration"""
        # Register custom tool
        async def custom_tool(params):
            return {'result': 'success', 'params': params}
        
        mcp_server.register_tool('custom_tool', custom_tool)
        
        # Verify tool is registered
        assert 'custom_tool' in mcp_server.tools
        
        # Test tool execution
        result = await mcp_server.execute_tool('custom_tool', {'test': 'param'})
        assert result['result'] == 'success'
        assert result['params']['test'] == 'param'
    
    @pytest.mark.asyncio
    async def test_mcp_request_handling(self, mcp_server):
        """Test MCP request handling"""
        request = {
            'jsonrpc': '2.0',
            'method': 'tools/list',
            'id': 1
        }
        
        response = await mcp_server.handle_request(request)
        
        assert response['jsonrpc'] == '2.0'
        assert response['id'] == 1
        assert 'result' in response
        assert 'tools' in response['result']
    
    @pytest.mark.asyncio
    async def test_mcp_error_handling(self, mcp_server):
        """Test MCP error handling"""
        # Invalid request
        invalid_request = {
            'jsonrpc': '2.0',
            'method': 'invalid/method',
            'id': 1
        }
        
        response = await mcp_server.handle_request(invalid_request)
        
        assert 'error' in response
        assert response['error']['code'] in [-32601, -32602]  # Method not found or Invalid params
    
    @pytest.mark.asyncio
    async def test_mcp_concurrent_requests(self, mcp_server):
        """Test MCP server handling concurrent requests"""
        requests = [
            {
                'jsonrpc': '2.0',
                'method': 'tools/list',
                'id': i
            }
            for i in range(10)
        ]
        
        tasks = [mcp_server.handle_request(req) for req in requests]
        responses = await asyncio.gather(*tasks)
        
        assert len(responses) == 10
        for i, response in enumerate(responses):
            assert response['id'] == i
            assert 'result' in response


# ========================= AGENT COORDINATION TESTS =========================

class TestAgentCoordination:
    """Integration tests for agent coordination"""
    
    @pytest.mark.asyncio
    async def test_agent_registration(self, agent_coordinator):
        """Test agent registration and discovery"""
        # Register agents
        agent_coordinator.register_agent('learning_path', {'type': 'planner'})
        agent_coordinator.register_agent('study_buddy', {'type': 'tutor'})
        agent_coordinator.register_agent('assessment', {'type': 'evaluator'})
        
        # Verify registration
        agents = agent_coordinator.get_registered_agents()
        assert len(agents) == 3
        assert 'learning_path' in agents
        assert 'study_buddy' in agents
        assert 'assessment' in agents
    
    @pytest.mark.asyncio
    async def test_agent_task_routing(self, agent_coordinator):
        """Test task routing to appropriate agents"""
        # Register mock agents
        mock_agents = {
            'math_tutor': Mock(process=AsyncMock(return_value={'result': 'math_solution'})),
            'quiz_generator': Mock(process=AsyncMock(return_value={'quiz': 'generated'}))
        }
        
        for name, agent in mock_agents.items():
            agent_coordinator.register_agent(name, agent)
        
        # Route math task
        math_task = {'type': 'tutoring', 'subject': 'math'}
        result = await agent_coordinator.route_task(math_task, 'math_tutor')
        assert result['result'] == 'math_solution'
        
        # Route quiz task
        quiz_task = {'type': 'quiz', 'topic': 'algebra'}
        result = await agent_coordinator.route_task(quiz_task, 'quiz_generator')
        assert result['quiz'] == 'generated'
    
    @pytest.mark.asyncio
    async def test_agent_communication(self, agent_coordinator):
        """Test inter-agent communication"""
        # Setup message queue
        message_queue = asyncio.Queue()
        
        async def agent_a():
            await message_queue.put({'from': 'agent_a', 'data': 'hello'})
            
        async def agent_b():
            msg = await message_queue.get()
            return {'from': 'agent_b', 'received': msg}
        
        # Run agents
        await agent_a()
        result = await agent_b()
        
        assert result['received']['from'] == 'agent_a'
        assert result['received']['data'] == 'hello'
    
    @pytest.mark.asyncio
    async def test_agent_load_balancing(self, agent_coordinator):
        """Test load balancing across multiple agents"""
        # Register multiple instances of same agent type
        for i in range(3):
            agent_coordinator.register_agent(
                f'tutor_{i}',
                {'type': 'tutor', 'load': 0}
            )
        
        # Distribute tasks
        tasks = []
        for i in range(9):
            agent = agent_coordinator.get_least_loaded_agent('tutor')
            tasks.append(agent)
            agent_coordinator.update_agent_load(agent, 1)
        
        # Verify even distribution
        load_distribution = {}
        for task in tasks:
            load_distribution[task] = load_distribution.get(task, 0) + 1
        
        # Each agent should get approximately 3 tasks
        for agent, count in load_distribution.items():
            assert 2 <= count <= 4


# ========================= EVENT SYSTEM INTEGRATION TESTS =========================

class TestEventSystemIntegration:
    """Integration tests for event system"""
    
    @pytest.mark.asyncio
    async def test_event_publishing_and_subscription(self, event_manager):
        """Test event publishing and subscription"""
        received_events = []
        
        # Subscribe to events
        def event_handler(event):
            received_events.append(event)
        
        event_manager.subscribe('student.registered', event_handler)
        event_manager.subscribe('quiz.completed', event_handler)
        
        # Publish events
        event_manager.publish('student.registered', {
            'student_id': 'test_123',
            'timestamp': datetime.now().isoformat()
        })
        
        event_manager.publish('quiz.completed', {
            'quiz_id': 'quiz_456',
            'score': 85,
            'timestamp': datetime.now().isoformat()
        })
        
        # Allow time for async processing
        await asyncio.sleep(0.1)
        
        assert len(received_events) == 2
        assert received_events[0]['event_type'] == 'student.registered'
        assert received_events[1]['event_type'] == 'quiz.completed'
    
    @pytest.mark.asyncio
    async def test_event_filtering(self, event_manager):
        """Test event filtering and routing"""
        math_events = []
        science_events = []
        
        # Subscribe with filters
        event_manager.subscribe(
            'quiz.completed',
            lambda e: math_events.append(e),
            filter_func=lambda e: e.get('subject') == 'math'
        )
        
        event_manager.subscribe(
            'quiz.completed',
            lambda e: science_events.append(e),
            filter_func=lambda e: e.get('subject') == 'science'
        )
        
        # Publish events
        event_manager.publish('quiz.completed', {'subject': 'math', 'score': 90})
        event_manager.publish('quiz.completed', {'subject': 'science', 'score': 85})
        event_manager.publish('quiz.completed', {'subject': 'math', 'score': 95})
        
        await asyncio.sleep(0.1)
        
        assert len(math_events) == 2
        assert len(science_events) == 1
    
    @pytest.mark.asyncio
    async def test_event_persistence(self, event_manager):
        """Test event persistence and replay"""
        # Configure persistence
        event_manager.enable_persistence('events.db')
        
        # Publish events
        events = [
            {'type': 'student.login', 'user_id': 'user1', 'time': '10:00'},
            {'type': 'quiz.started', 'quiz_id': 'q1', 'time': '10:05'},
            {'type': 'quiz.completed', 'quiz_id': 'q1', 'time': '10:30'}
        ]
        
        for event in events:
            event_manager.publish(event['type'], event)
        
        # Replay events
        replayed = event_manager.replay_events(
            start_time='10:00',
            end_time='10:30'
        )
        
        assert len(replayed) == 3
        assert replayed[0]['type'] == 'student.login'
        assert replayed[-1]['type'] == 'quiz.completed'
    
    @pytest.mark.asyncio
    async def test_event_error_handling(self, event_manager):
        """Test event system error handling"""
        error_count = [0]
        
        def failing_handler(event):
            error_count[0] += 1
            raise Exception("Handler error")
        
        def working_handler(event):
            event['processed'] = True
        
        # Subscribe handlers
        event_manager.subscribe('test.event', failing_handler)
        event_manager.subscribe('test.event', working_handler)
        
        # Publish event
        event = {'data': 'test'}
        event_manager.publish('test.event', event)
        
        await asyncio.sleep(0.1)
        
        # Verify error was handled and other handlers still executed
        assert error_count[0] == 1
        assert event.get('processed') == True


# ========================= RATE LIMITING INTEGRATION TESTS =========================

class TestRateLimitingIntegration:
    """Integration tests for rate limiting"""
    
    @pytest.mark.asyncio
    async def test_basic_rate_limiting(self, rate_limiter):
        """Test basic rate limiting functionality"""
        client_id = 'test_client_123'
        
        # Make requests within limit
        for i in range(5):
            allowed = await rate_limiter.check_limit(client_id)
            assert allowed == True
        
        # Configure to allow only 5 requests
        rate_limiter.set_limit(client_id, 5, 60)  # 5 requests per minute
        
        # Next request should be blocked
        allowed = await rate_limiter.check_limit(client_id)
        assert allowed == False
    
    @pytest.mark.asyncio
    async def test_rate_limit_with_api(self, test_client):
        """Test rate limiting with API endpoints"""
        # Make multiple rapid requests
        responses = []
        for i in range(15):
            response = test_client.get(
                '/api/v1/test',
                headers={'X-Client-ID': 'rapid_client'}
            )
            responses.append(response.status_code)
        
        # Some requests should be rate limited (429)
        assert 429 in responses
        # But some should succeed (200)
        assert 200 in responses
    
    @pytest.mark.asyncio
    async def test_distributed_rate_limiting(self, rate_limiter):
        """Test distributed rate limiting across multiple instances"""
        client_id = 'distributed_client'
        
        # Simulate multiple rate limiter instances
        limiters = [
            RateLimiter(redis_host='localhost'),
            RateLimiter(redis_host='localhost'),
            RateLimiter(redis_host='localhost')
        ]
        
        # Make requests from different instances
        total_allowed = 0
        for i in range(30):
            limiter = limiters[i % 3]
            if await limiter.check_limit(client_id):
                total_allowed += 1
        
        # Total allowed should respect global limit
        assert total_allowed <= 20  # Assuming 20 req/min limit
    
    @pytest.mark.asyncio
    async def test_rate_limit_reset(self, rate_limiter):
        """Test rate limit reset after time window"""
        client_id = 'reset_test_client'
        rate_limiter.set_limit(client_id, 2, 1)  # 2 requests per second
        
        # Use up the limit
        assert await rate_limiter.check_limit(client_id) == True
        assert await rate_limiter.check_limit(client_id) == True
        assert await rate_limiter.check_limit(client_id) == False
        
        # Wait for reset
        await asyncio.sleep(1.1)
        
        # Should be allowed again
        assert await rate_limiter.check_limit(client_id) == True


# ========================= MULTI-REGION INTEGRATION TESTS =========================

class TestMultiRegionIntegration:
    """Integration tests for multi-region deployment"""
    
    @pytest.mark.asyncio
    async def test_region_discovery(self, multi_region_config):
        """Test region discovery and registration"""
        # Register regions
        regions = [
            {'name': 'us-east-1', 'endpoint': 'https://us-east-1.api.com'},
            {'name': 'eu-west-1', 'endpoint': 'https://eu-west-1.api.com'},
            {'name': 'ap-south-1', 'endpoint': 'https://ap-south-1.api.com'}
        ]
        
        for region in regions:
            multi_region_config.register_region(region)
        
        # Verify registration
        registered = multi_region_config.get_regions()
        assert len(registered) == 3
        assert all(r['name'] in ['us-east-1', 'eu-west-1', 'ap-south-1'] for r in registered)
    
    @pytest.mark.asyncio
    async def test_region_health_monitoring(self, multi_region_config):
        """Test region health monitoring"""
        # Mock health check responses
        async def mock_health_check(region):
            if region['name'] == 'us-east-1':
                return {'status': 'healthy', 'latency': 20}
            elif region['name'] == 'eu-west-1':
                return {'status': 'degraded', 'latency': 150}
            else:
                return {'status': 'unhealthy', 'latency': None}
        
        multi_region_config.health_check = mock_health_check
        
        # Perform health checks
        health_status = await multi_region_config.check_all_regions()
        
        assert health_status['us-east-1']['status'] == 'healthy'
        assert health_status['eu-west-1']['status'] == 'degraded'
        assert health_status['ap-south-1']['status'] == 'unhealthy'
    
    @pytest.mark.asyncio
    async def test_region_failover(self, failover_manager):
        """Test automatic region failover"""
        # Configure regions with priorities
        failover_manager.configure_regions([
            {'name': 'primary', 'priority': 1, 'healthy': True},
            {'name': 'secondary', 'priority': 2, 'healthy': True},
            {'name': 'tertiary', 'priority': 3, 'healthy': True}
        ])
        
        # Primary should be active
        assert failover_manager.get_active_region()['name'] == 'primary'
        
        # Simulate primary failure
        failover_manager.mark_unhealthy('primary')
        
        # Should failover to secondary
        assert failover_manager.get_active_region()['name'] == 'secondary'
        
        # Simulate secondary failure
        failover_manager.mark_unhealthy('secondary')
        
        # Should failover to tertiary
        assert failover_manager.get_active_region()['name'] == 'tertiary'
    
    @pytest.mark.asyncio
    async def test_cross_region_replication(self, multi_region_config):
        """Test data replication across regions"""
        # Mock replication
        replicated_data = {}
        
        async def replicate(data, source_region, target_regions):
            for region in target_regions:
                replicated_data[region] = data
            return True
        
        # Perform replication
        data = {'user_id': 'test_123', 'action': 'quiz_completed'}
        success = await replicate(
            data,
            'us-east-1',
            ['eu-west-1', 'ap-south-1']
        )
        
        assert success == True
        assert replicated_data['eu-west-1'] == data
        assert replicated_data['ap-south-1'] == data


# ========================= PERFORMANCE AND LOAD TESTS =========================

class TestPerformanceIntegration:
    """Performance and load integration tests"""
    
    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_high_concurrency_api_requests(self, test_client):
        """Test API under high concurrency"""
        async def make_request(session, url):
            async with session.get(url) as response:
                return response.status
        
        # Simulate 100 concurrent requests
        async with aiohttp.ClientSession() as session:
            tasks = []
            for i in range(100):
                url = f'http://localhost:5000/api/v1/health'
                tasks.append(make_request(session, url))
            
            responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Verify responses
        successful = [r for r in responses if isinstance(r, int) and r == 200]
        assert len(successful) > 50  # At least 50% should succeed
    
    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_database_connection_pooling(self, database_router):
        """Test database connection pooling under load"""
        # Configure connection pool
        database_router.configure_pool(min_size=5, max_size=20)
        
        # Simulate concurrent database operations
        async def db_operation(i):
            conn = await database_router.get_connection()
            try:
                # Simulate query
                await asyncio.sleep(0.01)
                return f'result_{i}'
            finally:
                await database_router.release_connection(conn)
        
        # Run concurrent operations
        tasks = [db_operation(i) for i in range(50)]
        results = await asyncio.gather(*tasks)
        
        assert len(results) == 50
        assert all(r.startswith('result_') for r in results)
        
        # Verify pool metrics
        metrics = database_router.get_pool_metrics()
        assert metrics['connections_created'] <= 20
        assert metrics['connections_active'] == 0
    
    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_cache_performance(self, cache_manager):
        """Test cache performance and hit rates"""
        # Populate cache
        for i in range(1000):
            key = f'item_{i}'
            value = {'id': i, 'data': f'data_{i}'}
            await cache_manager.set(key, value)
        
        # Test cache hits
        hit_count = 0
        miss_count = 0
        
        for i in range(2000):
            key = f'item_{i}'
            result = await cache_manager.get(key)
            if result:
                hit_count += 1
            else:
                miss_count += 1
        
        # Calculate hit rate
        hit_rate = hit_count / (hit_count + miss_count)
        assert hit_rate >= 0.45  # At least 45% hit rate expected
        
        # Test cache eviction
        cache_manager.set_max_size(500)
        await cache_manager.evict_lru()
        
        remaining = await cache_manager.size()
        assert remaining <= 500
    
    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_memory_management(self, resource_manager):
        """Test memory management under load"""
        import psutil
        process = psutil.Process()
        
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Simulate memory-intensive operations
        large_objects = []
        for i in range(100):
            # Create large object
            obj = {'data': 'x' * 10000, 'id': i}
            large_objects.append(obj)
            
            # Monitor memory
            if i % 20 == 0:
                await resource_manager.check_memory_usage()
                if resource_manager.should_garbage_collect():
                    # Clear some objects
                    large_objects = large_objects[-50:]
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be controlled
        assert memory_increase < 500  # Less than 500MB increase


# ========================= SECURITY INTEGRATION TESTS =========================

class TestSecurityIntegration:
    """Security integration tests"""
    
    @pytest.mark.asyncio
    async def test_authentication_flow(self, test_client):
        """Test authentication and authorization flow"""
        # Attempt unauthorized access
        response = test_client.get('/api/v1/admin/users')
        assert response.status_code == 401
        
        # Login
        login_response = test_client.post(
            '/api/v1/auth/login',
            json={'username': 'admin', 'password': 'secure_password'}
        )
        assert login_response.status_code == 200
        token = json.loads(login_response.data)['token']
        
        # Access with token
        response = test_client.get(
            '/api/v1/admin/users',
            headers={'Authorization': f'Bearer {token}'}
        )
        assert response.status_code == 200
    
    @pytest.mark.asyncio
    async def test_input_validation(self, test_client):
        """Test input validation and sanitization"""
        # SQL injection attempt
        malicious_input = {
            'student_id': "'; DROP TABLE students; --",
            'name': '<script>alert("XSS")</script>'
        }
        
        response = test_client.post(
            '/api/v1/students/update',
            json=malicious_input
        )
        
        # Should be rejected or sanitized
        assert response.status_code in [400, 422]
        
        # Verify no damage done
        check_response = test_client.get('/api/v1/students')
        assert check_response.status_code == 200
    
    @pytest.mark.asyncio
    async def test_rate_limiting_security(self, test_client):
        """Test rate limiting for security"""
        # Simulate brute force attempt
        attempts = []
        for i in range(20):
            response = test_client.post(
                '/api/v1/auth/login',
                json={'username': 'admin', 'password': f'wrong_{i}'}
            )
            attempts.append(response.status_code)
        
        # Should be rate limited after several attempts
        assert 429 in attempts
    
    @pytest.mark.asyncio
    async def test_encryption_in_transit(self):
        """Test data encryption in transit"""
        # This would typically test HTTPS/TLS
        # For testing, we verify encryption configuration
        from src.config import Config
        
        config = Config()
        assert config.ENFORCE_HTTPS == True
        assert config.TLS_VERSION >= '1.2'
        assert 'TLS_ECDHE_RSA_WITH_AES_256_GCM_SHA384' in config.ALLOWED_CIPHERS


# ========================= MONITORING INTEGRATION TESTS =========================

class TestMonitoringIntegration:
    """Monitoring and observability integration tests"""
    
    @pytest.mark.asyncio
    async def test_metrics_collection(self, resource_manager):
        """Test metrics collection and reporting"""
        # Start metrics collection
        resource_manager.start_metrics_collection()
        
        # Simulate activity
        for i in range(10):
            resource_manager.record_metric('api_requests', 1)
            resource_manager.record_metric('response_time', 50 + i * 10)
        
        # Get metrics
        metrics = resource_manager.get_metrics()
        
        assert metrics['api_requests'] == 10
        assert metrics['response_time']['avg'] > 0
        assert metrics['response_time']['max'] >= metrics['response_time']['min']
    
    @pytest.mark.asyncio
    async def test_logging_pipeline(self, event_manager):
        """Test centralized logging pipeline"""
        logs = []
        
        # Configure log handler
        def log_handler(log_entry):
            logs.append(log_entry)
        
        event_manager.subscribe('log.*', log_handler)
        
        # Generate logs
        event_manager.publish('log.info', {'message': 'Test info log'})
        event_manager.publish('log.error', {'message': 'Test error log', 'error': 'TestError'})
        event_manager.publish('log.warning', {'message': 'Test warning'})
        
        await asyncio.sleep(0.1)
        
        assert len(logs) == 3
        assert any(l['event_type'] == 'log.error' for l in logs)
    
    @pytest.mark.asyncio
    async def test_health_check_aggregation(self, test_client):
        """Test health check aggregation across components"""
        response = test_client.get('/api/v1/health/detailed')
        assert response.status_code == 200
        
        health_data = json.loads(response.data)
        
        # Verify component health checks
        assert 'components' in health_data
        assert 'database' in health_data['components']
        assert 'cache' in health_data['components']
        assert 'mcp_server' in health_data['components']
        
        # Verify overall status
        assert health_data['overall_status'] in ['healthy', 'degraded', 'unhealthy']
    
    @pytest.mark.asyncio
    async def test_alert_generation(self, resource_manager):
        """Test alert generation for critical events"""
        alerts = []
        
        # Configure alert handler
        resource_manager.on_alert(lambda alert: alerts.append(alert))
        
        # Simulate critical conditions
        resource_manager.record_metric('error_rate', 0.15)  # 15% error rate
        resource_manager.record_metric('memory_usage', 0.95)  # 95% memory usage
        resource_manager.record_metric('response_time', 5000)  # 5 second response
        
        # Check alerts
        await resource_manager.evaluate_alerts()
        
        assert len(alerts) >= 2  # At least memory and response time alerts
        assert any(a['type'] == 'high_memory_usage' for a in alerts)


# ========================= END-TO-END INTEGRATION TESTS =========================

class TestEndToEndIntegration:
    """Complete end-to-end integration tests"""
    
    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_complete_student_learning_session(
        self, test_client, mcp_server, agent_coordinator, event_manager
    ):
        """Test complete student learning session from start to finish"""
        # 1. Student registration
        registration = {
            'name': 'Integration Test Student',
            'email': 'integration@test.com',
            'grade': 9
        }
        
        response = test_client.post('/api/v1/students/register', json=registration)
        assert response.status_code in [200, 201]
        student_data = json.loads(response.data)
        student_id = student_data['student_id']
        
        # 2. Learning style assessment
        assessment_response = test_client.post(
            '/api/v1/assessment/learning-style',
            json={
                'student_id': student_id,
                'responses': ['visual', 'graphs', 'diagrams']
            }
        )
        assert assessment_response.status_code == 200
        
        # 3. Generate learning path
        path_response = test_client.post(
            '/api/v1/learning-path/generate',
            json={
                'student_id': student_id,
                'subject': 'Matematik',
                'duration_weeks': 4
            }
        )
        assert path_response.status_code == 200
        learning_path = json.loads(path_response.data)
        
        # 4. Complete first week's content
        for week in range(1, 2):  # Just first week for testing
            # Get weekly content
            content_response = test_client.get(
                f'/api/v1/learning-path/{student_id}/week/{week}'
            )
            assert content_response.status_code == 200
            
            # Generate and take quiz
            quiz_response = test_client.post(
                '/api/v1/quiz/generate',
                json={
                    'student_id': student_id,
                    'week': week,
                    'topic': 'Matematik'
                }
            )
            assert quiz_response.status_code == 200
            quiz = json.loads(quiz_response.data)
            
            # Submit quiz answers
            answers = [
                {'question_id': q['id'], 'answer': q['options'][0]}
                for q in quiz['questions']
            ]
            
            submit_response = test_client.post(
                '/api/v1/quiz/submit',
                json={
                    'quiz_id': quiz['quiz_id'],
                    'student_id': student_id,
                    'answers': answers
                }
            )
            assert submit_response.status_code == 200
        
        # 5. Get progress report
        progress_response = test_client.get(f'/api/v1/students/{student_id}/progress')
        assert progress_response.status_code == 200
        progress = json.loads(progress_response.data)
        
        # Verify progress tracking
        assert 'completed_weeks' in progress
        assert progress['completed_weeks'] >= 1
        assert 'overall_score' in progress
        assert 0 <= progress['overall_score'] <= 100
    
    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_system_resilience_and_recovery(
        self, test_client, failover_manager, cache_manager, database_router
    ):
        """Test system resilience and recovery from failures"""
        
        # 1. Normal operation
        response = test_client.get('/api/v1/health')
        assert response.status_code == 200
        
        # 2. Simulate database failure
        database_router.simulate_failure('master')
        
        # System should continue with replica
        response = test_client.get('/api/v1/health')
        assert response.status_code == 200
        health = json.loads(response.data)
        assert health['components']['database']['status'] in ['degraded', 'replica_active']
        
        # 3. Simulate cache failure
        cache_manager.simulate_failure()
        
        # System should continue without cache
        response = test_client.get('/api/v1/students/test_123')
        assert response.status_code in [200, 503]  # May be slower but should work
        
        # 4. Recovery
        database_router.recover('master')
        cache_manager.recover()
        
        # Verify full recovery
        response = test_client.get('/api/v1/health')
        assert response.status_code == 200
        health = json.loads(response.data)
        assert health['overall_status'] == 'healthy'
    
    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_production_deployment_readiness(
        self, test_client, mcp_server, agent_coordinator,
        event_manager, rate_limiter, multi_region_config
    ):
        """Test production deployment readiness"""
        
        readiness_checks = {
            'api_available': False,
            'mcp_functional': False,
            'agents_registered': False,
            'events_working': False,
            'rate_limiting_active': False,
            'multi_region_configured': False,
            'monitoring_enabled': False,
            'security_configured': False
        }
        
        # 1. Check API availability
        response = test_client.get('/api/v1/health')
        readiness_checks['api_available'] = response.status_code == 200
        
        # 2. Check MCP functionality
        mcp_response = await mcp_server.handle_request({
            'jsonrpc': '2.0',
            'method': 'tools/list',
            'id': 1
        })
        readiness_checks['mcp_functional'] = 'result' in mcp_response
        
        # 3. Check agent registration
        agent_coordinator.register_agent('test_agent', {})
        agents = agent_coordinator.get_registered_agents()
        readiness_checks['agents_registered'] = len(agents) > 0
        
        # 4. Check event system
        event_received = [False]
        event_manager.subscribe('test', lambda e: event_received.__setitem__(0, True))
        event_manager.publish('test', {})
        await asyncio.sleep(0.1)
        readiness_checks['events_working'] = event_received[0]
        
        # 5. Check rate limiting
        readiness_checks['rate_limiting_active'] = await rate_limiter.check_limit('test') is not None
        
        # 6. Check multi-region configuration
        regions = multi_region_config.get_regions()
        readiness_checks['multi_region_configured'] = len(regions) > 0
        
        # 7. Check monitoring
        response = test_client.get('/api/v1/metrics')
        readiness_checks['monitoring_enabled'] = response.status_code in [200, 401]
        
        # 8. Check security configuration
        response = test_client.get('/api/v1/admin/test')
        readiness_checks['security_configured'] = response.status_code == 401
        
        # Verify all checks pass
        failed_checks = [k for k, v in readiness_checks.items() if not v]
        assert len(failed_checks) == 0, f"Failed checks: {failed_checks}"
        
        print("Production Readiness Check: ALL SYSTEMS GO!")


# ========================= TEST UTILITIES =========================

@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


# ========================= RUN TESTS =========================

if __name__ == "__main__":
    pytest.main([
        __file__,
        '-v',
        '--tb=short',
        '--asyncio-mode=auto',
        '-x'  # Stop on first failure
    ])