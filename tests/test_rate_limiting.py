"""
Comprehensive tests for Rate Limiting
"""
import pytest
import asyncio
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rate_limiter import RateLimiter


class TestRateLimiter:
    
    @pytest.fixture
    def rate_limiter(self):
        """Create a RateLimiter instance for testing"""
        return RateLimiter(max_requests=10, window_seconds=60)
    
    @pytest.fixture
    def strict_limiter(self):
        """Create a strict rate limiter for testing"""
        return RateLimiter(max_requests=3, window_seconds=1)
    
    @pytest.mark.unit
    def test_rate_limiter_initialization(self, rate_limiter):
        """Test rate limiter initialization"""
        assert rate_limiter is not None
        assert rate_limiter.max_requests == 10
        assert rate_limiter.window_seconds == 60
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_allow_request_under_limit(self, rate_limiter):
        """Test allowing requests under the limit"""
        client_id = "test_client_1"
        
        # Make requests under the limit
        for i in range(5):
            allowed = await rate_limiter.allow_request(client_id)
            assert allowed == True
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_block_request_over_limit(self, strict_limiter):
        """Test blocking requests over the limit"""
        client_id = "test_client_2"
        
        # Make requests up to the limit
        for i in range(3):
            allowed = await strict_limiter.allow_request(client_id)
            assert allowed == True
        
        # Next request should be blocked
        allowed = await strict_limiter.allow_request(client_id)
        assert allowed == False
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_window_reset(self, strict_limiter):
        """Test that rate limit window resets"""
        client_id = "test_client_3"
        
        # Exhaust the limit
        for i in range(3):
            await strict_limiter.allow_request(client_id)
        
        # Should be blocked now
        assert await strict_limiter.allow_request(client_id) == False
        
        # Wait for window to reset
        await asyncio.sleep(1.1)
        
        # Should be allowed again
        assert await strict_limiter.allow_request(client_id) == True
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_multiple_clients(self, rate_limiter):
        """Test rate limiting for multiple clients independently"""
        client1 = "client_1"
        client2 = "client_2"
        
        # Client 1 makes requests
        for i in range(5):
            assert await rate_limiter.allow_request(client1) == True
        
        # Client 2 should still be able to make requests
        for i in range(5):
            assert await rate_limiter.allow_request(client2) == True
    
    @pytest.mark.unit
    def test_get_remaining_requests(self, rate_limiter):
        """Test getting remaining requests for a client"""
        client_id = "test_client_4"
        
        if hasattr(rate_limiter, 'get_remaining'):
            initial = rate_limiter.get_remaining(client_id)
            assert initial == rate_limiter.max_requests
            
            # Make a request
            asyncio.run(rate_limiter.allow_request(client_id))
            
            remaining = rate_limiter.get_remaining(client_id)
            assert remaining == rate_limiter.max_requests - 1
    
    @pytest.mark.unit
    def test_get_reset_time(self, rate_limiter):
        """Test getting reset time for rate limit window"""
        client_id = "test_client_5"
        
        if hasattr(rate_limiter, 'get_reset_time'):
            # Make a request to start tracking
            asyncio.run(rate_limiter.allow_request(client_id))
            
            reset_time = rate_limiter.get_reset_time(client_id)
            assert reset_time is not None
            assert isinstance(reset_time, (int, float, datetime))
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_burst_requests(self, strict_limiter):
        """Test handling burst of requests"""
        client_id = "burst_client"
        results = []
        
        # Send burst of requests
        tasks = []
        for i in range(10):
            tasks.append(strict_limiter.allow_request(client_id))
        
        results = await asyncio.gather(*tasks)
        
        # First 3 should succeed, rest should fail
        allowed = sum(1 for r in results if r)
        blocked = sum(1 for r in results if not r)
        
        assert allowed == 3
        assert blocked == 7
    
    @pytest.mark.unit
    def test_clear_client_history(self, rate_limiter):
        """Test clearing client request history"""
        client_id = "test_client_6"
        
        # Make some requests
        for i in range(3):
            asyncio.run(rate_limiter.allow_request(client_id))
        
        if hasattr(rate_limiter, 'clear_client'):
            rate_limiter.clear_client(client_id)
            
            # Should be able to make max requests again
            for i in range(rate_limiter.max_requests):
                assert asyncio.run(rate_limiter.allow_request(client_id)) == True
    
    @pytest.mark.unit
    def test_clear_all_clients(self, rate_limiter):
        """Test clearing all client histories"""
        clients = ["client_a", "client_b", "client_c"]
        
        # Make requests for all clients
        for client in clients:
            for i in range(3):
                asyncio.run(rate_limiter.allow_request(client))
        
        if hasattr(rate_limiter, 'clear_all'):
            rate_limiter.clear_all()
            
            # All clients should have fresh limits
            for client in clients:
                assert asyncio.run(rate_limiter.allow_request(client)) == True


class TestAdvancedRateLimiting:
    
    @pytest.fixture
    def token_bucket_limiter(self):
        """Create a token bucket rate limiter if available"""
        try:
            from src.rate_limiter import TokenBucketRateLimiter
            return TokenBucketRateLimiter(capacity=10, refill_rate=2)
        except ImportError:
            return None
    
    @pytest.fixture
    def sliding_window_limiter(self):
        """Create a sliding window rate limiter if available"""
        try:
            from src.rate_limiter import SlidingWindowRateLimiter
            return SlidingWindowRateLimiter(max_requests=10, window_seconds=60)
        except ImportError:
            return None
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_token_bucket_refill(self, token_bucket_limiter):
        """Test token bucket refill mechanism"""
        if token_bucket_limiter is None:
            pytest.skip("TokenBucketRateLimiter not available")
        
        client_id = "token_client"
        
        # Consume all tokens
        for i in range(10):
            assert await token_bucket_limiter.allow_request(client_id) == True
        
        # Should be empty
        assert await token_bucket_limiter.allow_request(client_id) == False
        
        # Wait for refill (2 tokens per second)
        await asyncio.sleep(1)
        
        # Should have 2 tokens now
        assert await token_bucket_limiter.allow_request(client_id) == True
        assert await token_bucket_limiter.allow_request(client_id) == True
        assert await token_bucket_limiter.allow_request(client_id) == False
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_sliding_window_precision(self, sliding_window_limiter):
        """Test sliding window precision"""
        if sliding_window_limiter is None:
            pytest.skip("SlidingWindowRateLimiter not available")
        
        client_id = "sliding_client"
        
        # Make requests over time
        for i in range(5):
            assert await sliding_window_limiter.allow_request(client_id) == True
            await asyncio.sleep(0.2)
        
        # Should still have room for more
        for i in range(5):
            assert await sliding_window_limiter.allow_request(client_id) == True


@pytest.mark.integration
class TestRateLimitingIntegration:
    
    @pytest.fixture
    def rate_limiter(self):
        return RateLimiter(max_requests=20, window_seconds=10)
    
    @pytest.mark.asyncio
    async def test_concurrent_clients_isolation(self, rate_limiter):
        """Test that rate limits are isolated between clients"""
        async def client_requests(client_id, num_requests):
            results = []
            for i in range(num_requests):
                allowed = await rate_limiter.allow_request(client_id)
                results.append(allowed)
                await asyncio.sleep(0.01)
            return results
        
        # Run multiple clients concurrently
        tasks = [
            client_requests(f"client_{i}", 15)
            for i in range(3)
        ]
        
        all_results = await asyncio.gather(*tasks)
        
        # Each client should succeed with their requests (under limit)
        for results in all_results:
            assert all(results)  # All should be True
    
    @pytest.mark.asyncio
    async def test_rate_limit_with_retry(self, rate_limiter):
        """Test retry mechanism with rate limiting"""
        client_id = "retry_client"
        successful_requests = 0
        
        async def make_request_with_retry(max_retries=3):
            nonlocal successful_requests
            for attempt in range(max_retries):
                if await rate_limiter.allow_request(client_id):
                    successful_requests += 1
                    return True
                await asyncio.sleep(0.5)  # Wait before retry
            return False
        
        # Make many requests with retry
        tasks = [make_request_with_retry() for _ in range(30)]
        results = await asyncio.gather(*tasks)
        
        # Should have successful requests up to the limit
        assert successful_requests <= rate_limiter.max_requests
        assert successful_requests > 0
    
    @pytest.mark.asyncio
    async def test_rate_limit_metrics(self, rate_limiter):
        """Test rate limiting metrics collection"""
        metrics = {
            'total_requests': 0,
            'allowed_requests': 0,
            'blocked_requests': 0
        }
        
        client_id = "metrics_client"
        
        # Make requests and collect metrics
        for i in range(30):
            metrics['total_requests'] += 1
            if await rate_limiter.allow_request(client_id):
                metrics['allowed_requests'] += 1
            else:
                metrics['blocked_requests'] += 1
        
        # Verify metrics
        assert metrics['total_requests'] == 30
        assert metrics['allowed_requests'] == rate_limiter.max_requests
        assert metrics['blocked_requests'] == 30 - rate_limiter.max_requests
        assert metrics['allowed_requests'] + metrics['blocked_requests'] == metrics['total_requests']
    
    @pytest.mark.asyncio
    async def test_graceful_degradation(self, rate_limiter):
        """Test graceful degradation under load"""
        async def stressed_client(client_id):
            successes = 0
            for i in range(100):
                if await rate_limiter.allow_request(client_id):
                    successes += 1
                # No delay - maximum stress
            return successes
        
        # Run many stressed clients
        tasks = [stressed_client(f"stressed_{i}") for i in range(10)]
        results = await asyncio.gather(*tasks)
        
        # Each client should get fair share
        for result in results:
            assert result <= rate_limiter.max_requests
            assert result > 0  # Should get at least some requests through
    
    @pytest.mark.asyncio
    async def test_rate_limit_header_generation(self, rate_limiter):
        """Test generation of rate limit headers for HTTP responses"""
        client_id = "header_client"
        
        # Make some requests
        for i in range(5):
            await rate_limiter.allow_request(client_id)
        
        if hasattr(rate_limiter, 'get_headers'):
            headers = rate_limiter.get_headers(client_id)
            
            assert 'X-RateLimit-Limit' in headers
            assert 'X-RateLimit-Remaining' in headers
            assert 'X-RateLimit-Reset' in headers
            
            assert headers['X-RateLimit-Limit'] == str(rate_limiter.max_requests)
            assert int(headers['X-RateLimit-Remaining']) == rate_limiter.max_requests - 5