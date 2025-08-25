"""
External API Testing with Production-Ready Stubs
================================================
Comprehensive tests demonstrating API stub usage
"""

import pytest
import asyncio
import json
import time
from datetime import datetime
from unittest.mock import patch, AsyncMock
import uuid


class TestHuggingFaceAPI:
    """Test HuggingFace API integration"""
    
    @pytest.mark.asyncio
    async def test_text_generation(self, huggingface_stub):
        """Test text generation endpoint"""
        response = await huggingface_stub.request(
            'POST',
            'https://api-inference.huggingface.co/models/gpt2',
            headers={'Authorization': f'Bearer {huggingface_stub.config.api_key}'},
            json_data={
                'inputs': 'Once upon a time',
                'parameters': {
                    'max_new_tokens': 50,
                    'temperature': 0.8
                }
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert len(data) > 0
        assert 'generated_text' in data[0]
        assert data[0]['score'] > 0
    
    @pytest.mark.asyncio
    async def test_text_classification(self, huggingface_stub):
        """Test text classification endpoint"""
        response = await huggingface_stub.request(
            'POST',
            'https://api-inference.huggingface.co/models/distilbert-base-uncased-finetuned-sst-2-english',
            headers={'Authorization': f'Bearer {huggingface_stub.config.api_key}'},
            json_data={
                'inputs': 'I love this product, it works great!'
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert len(data) > 0
        assert len(data[0]) > 0  # Should have multiple labels
        assert 'label' in data[0][0]
        assert 'score' in data[0][0]
    
    @pytest.mark.asyncio
    async def test_embeddings(self, huggingface_stub):
        """Test sentence embeddings endpoint"""
        response = await huggingface_stub.request(
            'POST',
            'https://api-inference.huggingface.co/models/sentence-transformers/all-MiniLM-L6-v2',
            headers={'Authorization': f'Bearer {huggingface_stub.config.api_key}'},
            json_data={
                'inputs': 'This is a test sentence for embedding.'
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert len(data) > 0
        assert len(data[0]) == 384  # Expected embedding dimension
    
    @pytest.mark.asyncio
    async def test_model_info(self, huggingface_stub):
        """Test getting model information"""
        response = await huggingface_stub.request(
            'GET',
            'https://api-inference.huggingface.co/models/gpt2',
            headers={'Authorization': f'Bearer {huggingface_stub.config.api_key}'}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert 'model_id' in data
        assert 'pipeline_tag' in data
    
    @pytest.mark.asyncio
    async def test_qwen_model(self, huggingface_stub):
        """Test Qwen model for Turkish text generation"""
        response = await huggingface_stub.request(
            'POST',
            'https://api-inference.huggingface.co/models/Qwen/Qwen2.5-0.5B',
            headers={'Authorization': f'Bearer {huggingface_stub.config.api_key}'},
            json_data={
                'inputs': 'Türkiye\'nin başkenti',
                'parameters': {
                    'max_new_tokens': 30,
                    'temperature': 0.7
                }
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert 'generated_text' in data[0]


class TestOpenAIAPI:
    """Test OpenAI API integration"""
    
    @pytest.mark.asyncio
    async def test_chat_completion(self, openai_stub):
        """Test chat completion endpoint"""
        response = await openai_stub.request(
            'POST',
            'https://api.openai.com/v1/chat/completions',
            headers={'Authorization': f'Bearer {openai_stub.config.api_key}'},
            json_data={
                'model': 'gpt-3.5-turbo',
                'messages': [
                    {'role': 'system', 'content': 'You are a helpful assistant.'},
                    {'role': 'user', 'content': 'What is 2+2?'}
                ],
                'temperature': 0.7,
                'max_tokens': 100
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert 'choices' in data
        assert len(data['choices']) > 0
        assert 'message' in data['choices'][0]
        assert 'content' in data['choices'][0]['message']
        assert 'usage' in data
    
    @pytest.mark.asyncio
    async def test_streaming_response(self, openai_stub):
        """Test streaming chat completion"""
        response = await openai_stub.request(
            'POST',
            'https://api.openai.com/v1/chat/completions',
            headers={'Authorization': f'Bearer {openai_stub.config.api_key}'},
            json_data={
                'model': 'gpt-3.5-turbo',
                'messages': [{'role': 'user', 'content': 'Hello'}],
                'stream': True
            }
        )
        
        assert response.status_code == 200
        assert 'data:' in response.text
        assert '[DONE]' in response.text
    
    @pytest.mark.asyncio
    async def test_embeddings(self, openai_stub):
        """Test embeddings endpoint"""
        response = await openai_stub.request(
            'POST',
            'https://api.openai.com/v1/embeddings',
            headers={'Authorization': f'Bearer {openai_stub.config.api_key}'},
            json_data={
                'input': 'This is a test text for embedding',
                'model': 'text-embedding-ada-002'
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert 'data' in data
        assert len(data['data']) > 0
        assert 'embedding' in data['data'][0]
        assert len(data['data'][0]['embedding']) == 1536
    
    @pytest.mark.asyncio
    async def test_list_models(self, openai_stub):
        """Test listing available models"""
        response = await openai_stub.request(
            'GET',
            'https://api.openai.com/v1/models',
            headers={'Authorization': f'Bearer {openai_stub.config.api_key}'}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert 'data' in data
        assert len(data['data']) > 0
        assert any(m['id'] == 'gpt-3.5-turbo' for m in data['data'])


class TestRateLimiting:
    """Test rate limiting functionality"""
    
    @pytest.mark.asyncio
    async def test_rate_limit_enforcement(self, rate_limited_stub):
        """Test that rate limiting is enforced"""
        # Make requests up to the limit
        for i in range(5):
            response = await rate_limited_stub.request('GET', '/test')
            assert response.status_code == 200
        
        # Next request should be rate limited
        response = await rate_limited_stub.request('GET', '/test')
        assert response.status_code == 429
        assert 'retry_after' in response.json()
    
    @pytest.mark.asyncio
    async def test_token_bucket_refill(self, api_config):
        """Test token bucket rate limiting with refill"""
        api_config.rate_limit = 10
        api_config.rate_limit_strategy = RateLimitStrategy.TOKEN_BUCKET
        stub = BaseAPIStub(api_config)
        
        # Use tokens
        for i in range(5):
            allowed, _ = stub.rate_limiter.check_rate_limit()
            assert allowed
        
        # Wait for refill
        await asyncio.sleep(3)
        
        # Should have more tokens available
        allowed, _ = stub.rate_limiter.check_rate_limit()
        assert allowed
    
    @pytest.mark.asyncio
    async def test_sliding_window(self, api_config):
        """Test sliding window rate limiting"""
        from tests.external_api_stubs import RateLimitStrategy, BaseAPIStub
        
        api_config.rate_limit = 5
        api_config.rate_limit_strategy = RateLimitStrategy.SLIDING_WINDOW
        stub = BaseAPIStub(api_config)
        
        # Make 5 requests
        for i in range(5):
            allowed, _ = stub.rate_limiter.check_rate_limit()
            assert allowed
        
        # 6th request should be blocked
        allowed, retry_after = stub.rate_limiter.check_rate_limit()
        assert not allowed
        assert retry_after > 0
    
    @pytest.mark.asyncio
    async def test_per_client_rate_limiting(self, rate_limited_stub):
        """Test per-client rate limiting"""
        # Client 1 uses their quota
        for i in range(5):
            response = await rate_limited_stub.request(
                'GET', '/test',
                headers={'X-Client-Id': 'client1'}
            )
            assert response.status_code == 200
        
        # Client 1 is rate limited
        response = await rate_limited_stub.request(
            'GET', '/test',
            headers={'X-Client-Id': 'client1'}
        )
        assert response.status_code == 429
        
        # Client 2 can still make requests
        response = await rate_limited_stub.request(
            'GET', '/test',
            headers={'X-Client-Id': 'client2'}
        )
        assert response.status_code == 200


class TestAuthentication:
    """Test authentication mechanisms"""
    
    @pytest.mark.asyncio
    async def test_bearer_token_auth(self, api_config, auth_token):
        """Test Bearer token authentication"""
        stub = BaseAPIStub(api_config)
        
        # Request with valid token
        response = await stub.request(
            'GET', '/protected',
            headers={'Authorization': f'Bearer {auth_token}'}
        )
        assert response.status_code == 200
        
        # Request without token
        response = await stub.request('GET', '/protected')
        assert response.status_code == 401
    
    @pytest.mark.asyncio
    async def test_api_key_auth(self, api_config):
        """Test API key authentication"""
        stub = BaseAPIStub(api_config)
        
        # Request with correct API key
        response = await stub.request(
            'GET', '/protected',
            headers={'Authorization': 'ApiKey test_api_key'}
        )
        assert response.status_code == 200
        
        # Request with wrong API key
        response = await stub.request(
            'GET', '/protected',
            headers={'Authorization': 'ApiKey wrong_key'}
        )
        assert response.status_code == 401
    
    @pytest.mark.asyncio
    async def test_expired_token(self, api_config, expired_token):
        """Test expired token handling"""
        stub = BaseAPIStub(api_config)
        
        response = await stub.request(
            'GET', '/protected',
            headers={'Authorization': f'Bearer {expired_token}'}
        )
        assert response.status_code == 401
    
    @pytest.mark.asyncio
    async def test_no_auth_required(self, api_config_no_auth):
        """Test endpoints that don't require authentication"""
        stub = BaseAPIStub(api_config_no_auth)
        
        # Should work without any auth
        response = await stub.request('GET', '/public')
        assert response.status_code == 200


class TestWebhooks:
    """Test webhook functionality"""
    
    @pytest.mark.asyncio
    async def test_webhook_delivery(self, webhook_stub, webhook_payload):
        """Test webhook delivery"""
        response = await webhook_stub.send_webhook(
            'user.created',
            webhook_payload
        )
        
        assert response.status_code == 200
        assert len(webhook_stub.delivery_history) > 0
        
        # Check delivery record
        delivery = webhook_stub.delivery_history[-1]
        assert delivery['event_type'] == 'user.created'
        assert delivery['status'] == 'delivered'
    
    @pytest.mark.asyncio
    async def test_webhook_signature(self, webhook_stub):
        """Test webhook signature verification"""
        payload = {'test': 'data'}
        signature = webhook_stub._generate_signature(payload)
        
        # Verify correct signature
        assert webhook_stub.verify_signature(payload, signature)
        
        # Verify incorrect signature
        assert not webhook_stub.verify_signature(payload, 'wrong_signature')
    
    @pytest.mark.asyncio
    async def test_webhook_retry(self, webhook_stub):
        """Test webhook retry logic"""
        # Force failures to test retry
        with patch('random.random', return_value=0.05):  # Always fail
            response = await webhook_stub.send_webhook(
                'test.event',
                {'data': 'test'}
            )
        
        assert response.status_code == 503
        
        # Check retry attempts
        delivery = webhook_stub.delivery_history[-1]
        assert delivery['attempts'] == webhook_stub.retry_config['max_retries']
        assert delivery['status'] == 'failed'
    
    @pytest.mark.asyncio
    async def test_webhook_headers(self, webhook_stub):
        """Test webhook headers"""
        response = await webhook_stub.send_webhook(
            'custom.event',
            {'data': 'test'},
            headers={'X-Custom-Header': 'value'}
        )
        
        delivery = webhook_stub.delivery_history[-1]
        assert 'X-Webhook-Event' in delivery['headers']
        assert 'X-Webhook-Id' in delivery['headers']
        assert 'X-Webhook-Signature' in delivery['headers']
        assert delivery['headers']['X-Custom-Header'] == 'value'


class TestGraphQLAPI:
    """Test GraphQL API functionality"""
    
    @pytest.mark.asyncio
    async def test_graphql_query(self, graphql_stub, graphql_query):
        """Test GraphQL query execution"""
        result = await graphql_stub.execute(
            graphql_query,
            variables={'id': '123'}
        )
        
        assert 'data' in result
        assert 'user' in result['data']
        assert result['data']['user']['id'] == '123'
    
    @pytest.mark.asyncio
    async def test_graphql_mutation(self, graphql_stub, graphql_mutation):
        """Test GraphQL mutation execution"""
        result = await graphql_stub.execute(
            graphql_mutation,
            variables={
                'input': {
                    'name': 'New User',
                    'email': 'new@example.com'
                }
            }
        )
        
        assert 'data' in result
        assert 'createUser' in result['data']
        assert result['data']['createUser']['name'] == 'New User'
    
    @pytest.mark.asyncio
    async def test_graphql_list_query(self, graphql_stub):
        """Test GraphQL list query"""
        query = """
        query GetUsers($limit: Int) {
            users(limit: $limit) {
                id
                name
                email
            }
        }
        """
        
        result = await graphql_stub.execute(
            query,
            variables={'limit': 5}
        )
        
        assert 'data' in result
        assert 'users' in result['data']
        assert len(result['data']['users']) == 5


class TestErrorHandling:
    """Test error handling and recovery"""
    
    @pytest.mark.asyncio
    async def test_random_errors(self, api_config_high_error):
        """Test handling of random errors"""
        stub = BaseAPIStub(api_config_high_error)
        
        success_count = 0
        error_count = 0
        
        for i in range(20):
            response = await stub.request('GET', '/test')
            if response.status_code == 200:
                success_count += 1
            else:
                error_count += 1
        
        # With 50% error rate, should have some of each
        assert success_count > 0
        assert error_count > 0
        
        # Check metrics
        metrics = stub.get_metrics()
        assert metrics['error_rate'] > 0.3  # Should be around 0.5
    
    @pytest.mark.asyncio
    async def test_retry_on_error(self, retry_handler, api_config_high_error):
        """Test retry logic on errors"""
        stub = BaseAPIStub(api_config_high_error)
        
        # Use retry handler
        response = await retry_handler(
            stub.request,
            'GET', '/test'
        )
        
        # Should eventually succeed with retries
        assert response.status_code == 200
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_pattern(self, api_config):
        """Test circuit breaker pattern"""
        class CircuitBreaker:
            def __init__(self, threshold: int = 5):
                self.failure_count = 0
                self.threshold = threshold
                self.is_open = False
                self.last_failure_time = None
            
            async def call(self, func, *args, **kwargs):
                if self.is_open:
                    # Check if circuit should be reset
                    if self.last_failure_time and \
                       time.time() - self.last_failure_time > 30:
                        self.is_open = False
                        self.failure_count = 0
                    else:
                        raise Exception("Circuit breaker is open")
                
                try:
                    result = await func(*args, **kwargs)
                    self.failure_count = 0  # Reset on success
                    return result
                except Exception as e:
                    self.failure_count += 1
                    self.last_failure_time = time.time()
                    
                    if self.failure_count >= self.threshold:
                        self.is_open = True
                    
                    raise
        
        api_config.error_rate = 1.0  # Always fail
        stub = BaseAPIStub(api_config)
        breaker = CircuitBreaker(threshold=3)
        
        # Make requests until circuit opens
        for i in range(3):
            with pytest.raises(Exception):
                await breaker.call(stub.request, 'GET', '/test')
        
        # Circuit should be open now
        assert breaker.is_open
        
        # Next call should fail immediately
        with pytest.raises(Exception) as exc:
            await breaker.call(stub.request, 'GET', '/test')
        assert "Circuit breaker is open" in str(exc.value)


class TestCaching:
    """Test API response caching"""
    
    @pytest.mark.asyncio
    async def test_cache_hit(self, api_config):
        """Test cache hit for identical requests"""
        api_config.enable_caching = True
        api_config.cache_ttl = 60
        stub = BaseAPIStub(api_config)
        
        # First request - cache miss
        response1 = await stub.request('GET', '/data', params={'id': '123'})
        assert stub.metrics['cache_misses'] == 1
        
        # Second identical request - cache hit
        response2 = await stub.request('GET', '/data', params={'id': '123'})
        assert stub.metrics['cache_hits'] == 1
        
        # Responses should be identical
        assert response1.json() == response2.json()
    
    @pytest.mark.asyncio
    async def test_cache_expiry(self, api_config):
        """Test cache expiry"""
        api_config.enable_caching = True
        api_config.cache_ttl = 0.1  # 100ms TTL
        stub = BaseAPIStub(api_config)
        
        # First request
        await stub.request('GET', '/data')
        
        # Wait for cache to expire
        await asyncio.sleep(0.2)
        
        # Should be cache miss
        await stub.request('GET', '/data')
        assert stub.metrics['cache_misses'] == 2
    
    @pytest.mark.asyncio
    async def test_cache_key_generation(self, api_config):
        """Test cache key generation for different requests"""
        stub = BaseAPIStub(api_config)
        
        key1 = stub._get_cache_key('GET', '/api/v1', {'param': 'value'})
        key2 = stub._get_cache_key('GET', '/api/v1', {'param': 'value'})
        key3 = stub._get_cache_key('GET', '/api/v1', {'param': 'different'})
        key4 = stub._get_cache_key('POST', '/api/v1', {'param': 'value'})
        
        # Same requests should have same key
        assert key1 == key2
        
        # Different params should have different key
        assert key1 != key3
        
        # Different method should have different key
        assert key1 != key4


class TestMetricsAndMonitoring:
    """Test metrics and monitoring functionality"""
    
    @pytest.mark.asyncio
    async def test_metrics_collection(self, api_config):
        """Test that metrics are collected correctly"""
        stub = BaseAPIStub(api_config)
        
        # Make various requests
        for i in range(10):
            await stub.request('GET', f'/endpoint{i}')
        
        metrics = stub.get_metrics()
        
        assert metrics['total_requests'] == 10
        assert metrics['avg_latency'] > 0
        assert 'success_rate' in metrics
        assert 'cache_hit_rate' in metrics
    
    @pytest.mark.asyncio
    async def test_request_history(self, api_config):
        """Test request history tracking"""
        stub = BaseAPIStub(api_config)
        
        # Make requests with different parameters
        await stub.request('GET', '/users', params={'page': 1})
        await stub.request('POST', '/users', json_data={'name': 'Test'})
        await stub.request('DELETE', '/users/123')
        
        assert len(stub.request_history) == 3
        
        # Check history details
        assert stub.request_history[0]['method'] == 'GET'
        assert stub.request_history[1]['method'] == 'POST'
        assert stub.request_history[2]['method'] == 'DELETE'
    
    @pytest.mark.asyncio
    async def test_performance_monitoring(self, api_monitor, api_config):
        """Test performance monitoring"""
        stub = BaseAPIStub(api_config)
        
        # Monitor multiple requests
        for i in range(5):
            await api_monitor.monitor(
                stub.request,
                'GET', f'/endpoint{i}'
            )
        
        stats = api_monitor.get_stats()
        
        assert stats['total_calls'] == 5
        assert stats['successful_calls'] == 5
        assert stats['avg_latency'] > 0
        assert stats['max_latency'] >= stats['avg_latency']
        assert stats['min_latency'] <= stats['avg_latency']


class TestBatchProcessing:
    """Test batch API processing"""
    
    @pytest.mark.asyncio
    async def test_batch_requests(self, batch_processor, api_config):
        """Test batch request processing"""
        stub = BaseAPIStub(api_config)
        
        # Create items to process
        items = [{'id': i, 'data': f'item_{i}'} for i in range(25)]
        
        # Process function
        async def process_item(item):
            response = await stub.request(
                'POST', '/process',
                json_data=item
            )
            return response.status_code == 200
        
        # Process in batches
        results = await batch_processor.process_batch(items, process_item)
        
        assert len(results) == 25
        assert all(r for r in results)  # All should succeed
    
    @pytest.mark.asyncio
    async def test_batch_with_errors(self, batch_processor, api_config_high_error):
        """Test batch processing with errors"""
        stub = BaseAPIStub(api_config_high_error)
        
        items = list(range(10))
        
        async def process_item(item):
            try:
                response = await stub.request('GET', f'/item/{item}')
                return response.status_code == 200
            except:
                return False
        
        results = await batch_processor.process_batch(items, process_item)
        
        # Some should succeed, some should fail
        success_count = sum(1 for r in results if r)
        assert 0 < success_count < 10


class TestIntegrationScenarios:
    """Test realistic integration scenarios"""
    
    @pytest.mark.asyncio
    async def test_ml_pipeline_integration(self, huggingface_stub, openai_stub):
        """Test ML pipeline with multiple API calls"""
        # Step 1: Generate text with HuggingFace
        hf_response = await huggingface_stub.request(
            'POST',
            'https://api-inference.huggingface.co/models/gpt2',
            headers={'Authorization': 'Bearer hf_token'},
            json_data={
                'inputs': 'Artificial intelligence is',
                'parameters': {'max_new_tokens': 50}
            }
        )
        
        assert hf_response.status_code == 200
        generated_text = hf_response.json()[0]['generated_text']
        
        # Step 2: Enhance with OpenAI
        openai_response = await openai_stub.request(
            'POST',
            'https://api.openai.com/v1/chat/completions',
            headers={'Authorization': 'Bearer sk-test'},
            json_data={
                'model': 'gpt-3.5-turbo',
                'messages': [
                    {'role': 'user', 'content': f'Improve this text: {generated_text}'}
                ]
            }
        )
        
        assert openai_response.status_code == 200
        
        # Step 3: Generate embeddings
        embedding_response = await openai_stub.request(
            'POST',
            'https://api.openai.com/v1/embeddings',
            headers={'Authorization': 'Bearer sk-test'},
            json_data={
                'input': generated_text,
                'model': 'text-embedding-ada-002'
            }
        )
        
        assert embedding_response.status_code == 200
        assert len(embedding_response.json()['data'][0]['embedding']) == 1536
    
    @pytest.mark.asyncio
    async def test_event_driven_workflow(self, webhook_stub, graphql_stub):
        """Test event-driven workflow with webhooks and GraphQL"""
        # Step 1: Create user via GraphQL
        create_result = await graphql_stub.execute(
            """
            mutation CreateUser($input: UserInput!) {
                createUser(input: $input) {
                    id
                    name
                    email
                }
            }
            """,
            variables={
                'input': {
                    'name': 'Test User',
                    'email': 'test@example.com'
                }
            }
        )
        
        user_id = create_result['data']['createUser']['id']
        
        # Step 2: Send webhook notification
        webhook_response = await webhook_stub.send_webhook(
            'user.created',
            {
                'user_id': user_id,
                'timestamp': datetime.now().isoformat()
            }
        )
        
        assert webhook_response.status_code == 200
        
        # Step 3: Query user to verify
        query_result = await graphql_stub.execute(
            """
            query GetUser($id: ID!) {
                user(id: $id) {
                    id
                    name
                    email
                }
            }
            """,
            variables={'id': user_id}
        )
        
        assert query_result['data']['user']['id'] == user_id
    
    @pytest.mark.asyncio
    async def test_resilient_api_client(self, service_registry, retry_handler):
        """Test resilient API client with multiple services"""
        results = {}
        
        for service_name, service_info in service_registry.items():
            stub = service_info['stub']
            
            try:
                # Try to make authenticated request with retry
                response = await retry_handler(
                    stub.request,
                    'GET', f"{service_info['base_url']}/health",
                    headers={'Authorization': f'Bearer test_token'}
                )
                
                results[service_name] = {
                    'status': 'healthy',
                    'response_code': response.status_code
                }
            except Exception as e:
                results[service_name] = {
                    'status': 'unhealthy',
                    'error': str(e)
                }
        
        # At least some services should be healthy
        healthy_services = [s for s, r in results.items() if r['status'] == 'healthy']
        assert len(healthy_services) > 0