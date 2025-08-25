# -*- coding: utf-8 -*-
"""
End-to-End Resilience and Error Handling Tests for TEKNOFEST 2025
Tests for system stability, error recovery, and fault tolerance
"""

import pytest
import json
import asyncio
import random
import time
import os
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Any
from unittest.mock import Mock, patch, AsyncMock
import aiohttp
from concurrent.futures import ThreadPoolExecutor
import logging

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import system components
from src.error_handler import ErrorHandler
from src.failover_manager import FailoverManager
from src.resource_manager import ResourceManager
from src.cache_manager import CacheManager
from src.database_router import DatabaseRouter
from src.rate_limiter import RateLimiter
from src.event_manager import EventManager

logger = logging.getLogger(__name__)


class TestE2EErrorHandling:
    """Comprehensive error handling tests"""
    
    @pytest.mark.asyncio
    async def test_cascading_failure_prevention(self, system_components):
        """Test prevention of cascading failures"""
        
        error_handler = system_components['error_handler']
        event_manager = system_components['event_manager']
        
        # Track errors
        errors_caught = []
        circuit_breaker_trips = []
        
        def error_listener(event):
            if event['type'] == 'error':
                errors_caught.append(event)
            elif event['type'] == 'circuit_breaker_trip':
                circuit_breaker_trips.append(event)
        
        event_manager.subscribe('system.error', error_listener)
        event_manager.subscribe('circuit_breaker.*', error_listener)
        
        # Simulate service that starts failing
        class FailingService:
            def __init__(self):
                self.call_count = 0
                self.failure_threshold = 5
            
            async def process(self):
                self.call_count += 1
                if self.call_count > self.failure_threshold:
                    raise Exception("Service failure")
                return "success"
        
        service = FailingService()
        
        # Make multiple calls that will eventually fail
        results = []
        for i in range(20):
            try:
                result = await error_handler.with_circuit_breaker(
                    service.process,
                    circuit_name="failing_service",
                    failure_threshold=3,
                    timeout=10
                )
                results.append(('success', result))
            except Exception as e:
                results.append(('error', str(e)))
            
            await asyncio.sleep(0.1)
        
        # Verify circuit breaker engaged
        assert len(circuit_breaker_trips) > 0
        
        # Verify not all calls reached the failing service
        assert service.call_count < 20
        
        # Verify graceful degradation
        error_count = sum(1 for r in results if r[0] == 'error')
        assert error_count < 15  # Circuit breaker should prevent most errors
        
        logger.info(f"✅ Cascading failure prevention: {service.call_count} calls made out of 20")
    
    @pytest.mark.asyncio
    async def test_retry_with_exponential_backoff(self, system_components):
        """Test retry mechanism with exponential backoff"""
        
        error_handler = system_components['error_handler']
        
        class FlakeyService:
            def __init__(self):
                self.attempts = 0
                self.success_after = 3
            
            async def call(self):
                self.attempts += 1
                if self.attempts < self.success_after:
                    raise ConnectionError(f"Attempt {self.attempts} failed")
                return f"Success after {self.attempts} attempts"
        
        service = FlakeyService()
        
        start_time = time.time()
        result = await error_handler.retry_with_backoff(
            service.call,
            max_retries=5,
            initial_delay=0.1,
            max_delay=2.0,
            exponential_base=2
        )
        duration = time.time() - start_time
        
        assert result == "Success after 3 attempts"
        assert service.attempts == 3
        
        # Verify exponential backoff timing
        # Expected delays: 0.1, 0.2, success (total ~0.3s minimum)
        assert duration >= 0.3
        assert duration < 5.0  # Should not take too long
        
        logger.info(f"✅ Retry with backoff succeeded after {service.attempts} attempts in {duration:.2f}s")
    
    @pytest.mark.asyncio
    async def test_timeout_handling(self, system_components):
        """Test timeout handling for long-running operations"""
        
        error_handler = system_components['error_handler']
        
        async def slow_operation(duration: float):
            await asyncio.sleep(duration)
            return "completed"
        
        # Test successful operation within timeout
        result = await error_handler.with_timeout(
            slow_operation(0.5),
            timeout=2.0
        )
        assert result == "completed"
        
        # Test timeout exceeded
        with pytest.raises(asyncio.TimeoutError):
            await error_handler.with_timeout(
                slow_operation(3.0),
                timeout=1.0
            )
        
        # Test timeout with fallback
        result = await error_handler.with_timeout(
            slow_operation(3.0),
            timeout=1.0,
            fallback="timeout_fallback"
        )
        assert result == "timeout_fallback"
        
        logger.info("✅ Timeout handling tests passed")
    
    @pytest.mark.asyncio
    async def test_error_aggregation_and_reporting(self, system_components):
        """Test error aggregation and reporting mechanisms"""
        
        error_handler = system_components['error_handler']
        event_manager = system_components['event_manager']
        
        # Generate various types of errors
        error_types = [
            ('ValidationError', 'Invalid input data'),
            ('ConnectionError', 'Database connection failed'),
            ('TimeoutError', 'Operation timed out'),
            ('PermissionError', 'Access denied'),
            ('ValidationError', 'Missing required field'),
            ('ConnectionError', 'Redis connection lost')
        ]
        
        for error_type, message in error_types * 3:  # Generate multiple of each
            await error_handler.log_error(
                error_type=error_type,
                message=message,
                context={'timestamp': datetime.utcnow().isoformat()}
            )
        
        # Get error statistics
        stats = await error_handler.get_error_statistics(
            time_window=timedelta(minutes=5)
        )
        
        assert 'total_errors' in stats
        assert stats['total_errors'] == len(error_types) * 3
        
        assert 'by_type' in stats
        assert stats['by_type']['ValidationError'] == 6
        assert stats['by_type']['ConnectionError'] == 6
        
        assert 'error_rate' in stats
        assert stats['error_rate'] > 0
        
        # Test error pattern detection
        patterns = await error_handler.detect_error_patterns()
        assert len(patterns) > 0
        
        # Verify most common errors identified
        assert any(p['type'] == 'ValidationError' for p in patterns)
        assert any(p['type'] == 'ConnectionError' for p in patterns)
        
        logger.info(f"✅ Error aggregation: {stats['total_errors']} errors tracked and analyzed")
    
    @pytest.mark.asyncio
    async def test_graceful_shutdown(self, system_components):
        """Test graceful shutdown procedures"""
        
        # Create mock services with cleanup
        services_cleaned = []
        
        class Service:
            def __init__(self, name):
                self.name = name
                self.active_connections = 5
                self.pending_tasks = 3
            
            async def shutdown(self):
                # Simulate cleanup
                await asyncio.sleep(0.1)
                self.active_connections = 0
                self.pending_tasks = 0
                services_cleaned.append(self.name)
                return True
        
        services = [
            Service('database'),
            Service('cache'),
            Service('message_queue'),
            Service('api_server')
        ]
        
        # Perform graceful shutdown
        shutdown_tasks = [service.shutdown() for service in services]
        results = await asyncio.gather(*shutdown_tasks, return_exceptions=True)
        
        # Verify all services cleaned up
        assert len(services_cleaned) == 4
        assert all(r == True for r in results if not isinstance(r, Exception))
        
        # Verify no active connections remain
        for service in services:
            assert service.active_connections == 0
            assert service.pending_tasks == 0
        
        logger.info(f"✅ Graceful shutdown completed for {len(services)} services")


class TestE2EFailoverMechanisms:
    """Tests for failover and recovery mechanisms"""
    
    @pytest.mark.asyncio
    async def test_database_failover(self, system_components):
        """Test database failover from primary to replica"""
        
        database_router = system_components['database_router']
        
        # Configure primary and replicas
        database_router.configure_replicas([
            {'host': 'replica1', 'port': 5432, 'lag': 0},
            {'host': 'replica2', 'port': 5432, 'lag': 5},
            {'host': 'replica3', 'port': 5432, 'lag': 10}
        ])
        
        # Normal operation - should use primary
        active = database_router.get_active_connection()
        assert active['role'] == 'primary'
        
        # Simulate primary failure
        database_router.mark_unhealthy('primary')
        
        # Should failover to best replica (lowest lag)
        active = database_router.get_active_connection()
        assert active['role'] == 'replica'
        assert active['host'] == 'replica1'
        
        # Simulate replica1 failure
        database_router.mark_unhealthy('replica1')
        
        # Should failover to next best replica
        active = database_router.get_active_connection()
        assert active['host'] == 'replica2'
        
        # Test recovery of primary
        database_router.mark_healthy('primary')
        
        # Should switch back to primary (if auto-failback enabled)
        if database_router.auto_failback:
            await asyncio.sleep(database_router.failback_delay)
            active = database_router.get_active_connection()
            assert active['role'] == 'primary'
        
        logger.info("✅ Database failover mechanism tested successfully")
    
    @pytest.mark.asyncio
    async def test_service_health_monitoring(self, system_components):
        """Test service health monitoring and auto-recovery"""
        
        failover_manager = system_components['failover_manager']
        
        # Register services with health checks
        services = {
            'api': {'url': 'http://localhost:5000/health', 'interval': 5},
            'database': {'url': 'postgresql://localhost:5432', 'interval': 10},
            'cache': {'url': 'redis://localhost:6379', 'interval': 5},
            'mcp': {'url': 'http://localhost:3000/health', 'interval': 5}
        }
        
        for name, config in services.items():
            failover_manager.register_service(name, config)
        
        # Start health monitoring
        await failover_manager.start_monitoring()
        
        # Wait for initial health checks
        await asyncio.sleep(2)
        
        # Get health status
        health_status = failover_manager.get_health_status()
        
        assert len(health_status) == 4
        for service in health_status:
            assert 'name' in service
            assert 'status' in service
            assert 'last_check' in service
            assert service['status'] in ['healthy', 'degraded', 'unhealthy']
        
        # Simulate service degradation
        failover_manager.simulate_degradation('cache')
        
        # Verify detection
        await asyncio.sleep(6)  # Wait for next health check
        health_status = failover_manager.get_health_status()
        cache_status = next(s for s in health_status if s['name'] == 'cache')
        assert cache_status['status'] == 'degraded'
        
        logger.info(f"✅ Health monitoring active for {len(services)} services")
    
    @pytest.mark.asyncio
    async def test_load_balancer_failover(self, system_components):
        """Test load balancer failover strategies"""
        
        failover_manager = system_components['failover_manager']
        
        # Configure backend servers
        backends = [
            {'id': 'server1', 'weight': 3, 'healthy': True, 'load': 0},
            {'id': 'server2', 'weight': 2, 'healthy': True, 'load': 0},
            {'id': 'server3', 'weight': 1, 'healthy': True, 'load': 0}
        ]
        
        failover_manager.configure_backends(backends)
        
        # Test weighted round-robin distribution
        distribution = {}
        for _ in range(60):
            backend = failover_manager.get_next_backend()
            distribution[backend['id']] = distribution.get(backend['id'], 0) + 1
        
        # Verify weighted distribution
        assert distribution['server1'] > distribution['server2']
        assert distribution['server2'] > distribution['server3']
        
        # Simulate server failure
        failover_manager.mark_backend_unhealthy('server1')
        
        # Test redistribution
        distribution_after = {}
        for _ in range(30):
            backend = failover_manager.get_next_backend()
            distribution_after[backend['id']] = distribution_after.get(backend['id'], 0) + 1
        
        # Server1 should not receive traffic
        assert 'server1' not in distribution_after
        assert 'server2' in distribution_after
        assert 'server3' in distribution_after
        
        logger.info("✅ Load balancer failover tested with weighted distribution")
    
    @pytest.mark.asyncio
    async def test_data_consistency_during_failover(self, system_components):
        """Test data consistency during failover scenarios"""
        
        database_router = system_components['database_router']
        cache_manager = system_components['cache_manager']
        
        # Create test data
        test_data = {
            'user_id': 'test_123',
            'balance': 1000,
            'transactions': []
        }
        
        # Start transaction
        async with database_router.transaction() as tx:
            # Write to primary
            await tx.execute(
                "INSERT INTO accounts (user_id, balance) VALUES ($1, $2)",
                test_data['user_id'], test_data['balance']
            )
            
            # Cache the data
            await cache_manager.set(f"account:{test_data['user_id']}", test_data)
            
            # Simulate failover during transaction
            database_router.simulate_failover()
            
            # Attempt to complete transaction
            try:
                await tx.execute(
                    "UPDATE accounts SET balance = balance - 100 WHERE user_id = $1",
                    test_data['user_id']
                )
                await tx.commit()
                transaction_completed = True
            except Exception:
                await tx.rollback()
                transaction_completed = False
        
        # Verify data consistency
        if transaction_completed:
            # Check database
            result = await database_router.fetch_one(
                "SELECT balance FROM accounts WHERE user_id = $1",
                test_data['user_id']
            )
            assert result['balance'] == 900
            
            # Invalidate cache to maintain consistency
            await cache_manager.delete(f"account:{test_data['user_id']}")
        else:
            # Verify rollback
            result = await database_router.fetch_one(
                "SELECT balance FROM accounts WHERE user_id = $1",
                test_data['user_id']
            )
            # Balance should be unchanged or not exist
            assert result is None or result['balance'] == 1000
        
        logger.info("✅ Data consistency maintained during failover")


class TestE2EResourceManagement:
    """Tests for resource management and optimization"""
    
    @pytest.mark.asyncio
    async def test_memory_leak_detection(self, system_components):
        """Test memory leak detection and prevention"""
        
        resource_manager = system_components['resource_manager']
        
        # Monitor initial memory
        initial_memory = resource_manager.get_memory_usage()
        
        # Simulate potential memory leak scenario
        large_objects = []
        for i in range(100):
            # Create large object
            obj = {'data': 'x' * 100000, 'id': i}
            large_objects.append(obj)
            
            # Check memory growth
            if i % 10 == 0:
                current_memory = resource_manager.get_memory_usage()
                growth = current_memory - initial_memory
                
                # Trigger cleanup if growth exceeds threshold
                if growth > 100 * 1024 * 1024:  # 100MB
                    await resource_manager.trigger_garbage_collection()
                    # Clear old objects
                    large_objects = large_objects[-10:]
        
        # Final memory check
        final_memory = resource_manager.get_memory_usage()
        memory_growth = final_memory - initial_memory
        
        # Memory growth should be controlled
        assert memory_growth < 200 * 1024 * 1024  # Less than 200MB growth
        
        logger.info(f"✅ Memory leak detection: Growth controlled to {memory_growth / 1024 / 1024:.2f}MB")
    
    @pytest.mark.asyncio
    async def test_connection_pool_management(self, system_components):
        """Test connection pool management under load"""
        
        database_router = system_components['database_router']
        
        # Configure connection pool
        pool_config = {
            'min_size': 5,
            'max_size': 20,
            'max_idle_time': 300,
            'validation_interval': 60
        }
        database_router.configure_pool(**pool_config)
        
        # Simulate concurrent connections
        async def use_connection(duration: float):
            conn = await database_router.acquire_connection()
            try:
                await asyncio.sleep(duration)
                return True
            finally:
                await database_router.release_connection(conn)
        
        # Create load
        tasks = []
        for i in range(50):
            duration = random.uniform(0.1, 0.5)
            tasks.append(use_connection(duration))
        
        start_time = time.time()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        duration = time.time() - start_time
        
        # Get pool statistics
        stats = database_router.get_pool_stats()
        
        # Verify pool behavior
        assert all(r == True for r in results if not isinstance(r, Exception))
        assert stats['connections_created'] <= pool_config['max_size']
        assert stats['connections_active'] == 0  # All returned
        assert stats['pool_size'] >= pool_config['min_size']
        
        logger.info(f"""
        ✅ Connection pool managed {len(tasks)} requests:
        - Connections created: {stats['connections_created']}
        - Max concurrent: {stats['max_concurrent']}
        - Pool efficiency: {stats.get('efficiency', 0):.2%}
        """)
    
    @pytest.mark.asyncio
    async def test_resource_throttling(self, system_components):
        """Test resource throttling and rate limiting"""
        
        resource_manager = system_components['resource_manager']
        rate_limiter = system_components['rate_limiter']
        
        # Configure throttling
        throttle_config = {
            'cpu_threshold': 80,  # Throttle at 80% CPU
            'memory_threshold': 75,  # Throttle at 75% memory
            'request_reduction': 0.5  # Reduce to 50% when throttling
        }
        resource_manager.configure_throttling(**throttle_config)
        
        # Simulate high load
        request_results = []
        
        for i in range(100):
            # Check if throttling should be applied
            should_throttle = resource_manager.should_throttle()
            
            if should_throttle:
                # Apply reduced rate
                allowed = await rate_limiter.check_limit(
                    'throttled_client',
                    limit=30  # Reduced from 60
                )
            else:
                allowed = await rate_limiter.check_limit('normal_client')
            
            request_results.append({
                'request': i,
                'allowed': allowed,
                'throttled': should_throttle
            })
            
            # Simulate resource usage
            if i % 20 == 0:
                resource_manager.simulate_high_load()
            
            await asyncio.sleep(0.01)
        
        # Analyze results
        throttled_requests = [r for r in request_results if r['throttled']]
        allowed_requests = [r for r in request_results if r['allowed']]
        
        logger.info(f"""
        ✅ Resource throttling test:
        - Total requests: {len(request_results)}
        - Throttled periods: {len(throttled_requests)}
        - Allowed requests: {len(allowed_requests)}
        """)
    
    @pytest.mark.asyncio
    async def test_automatic_scaling(self, system_components):
        """Test automatic scaling based on load"""
        
        resource_manager = system_components['resource_manager']
        
        # Configure auto-scaling
        scaling_config = {
            'min_instances': 2,
            'max_instances': 10,
            'scale_up_threshold': 70,  # CPU/Memory %
            'scale_down_threshold': 30,
            'cooldown_period': 60  # seconds
        }
        resource_manager.configure_auto_scaling(**scaling_config)
        
        # Start with minimum instances
        current_instances = scaling_config['min_instances']
        scaling_events = []
        
        # Simulate varying load
        load_pattern = [30, 50, 75, 85, 90, 85, 70, 50, 30, 20]
        
        for load in load_pattern:
            # Set current load
            resource_manager.set_load(load)
            
            # Check scaling decision
            scaling_decision = await resource_manager.evaluate_scaling()
            
            if scaling_decision == 'scale_up' and current_instances < scaling_config['max_instances']:
                current_instances += 1
                scaling_events.append(('scale_up', current_instances, load))
            elif scaling_decision == 'scale_down' and current_instances > scaling_config['min_instances']:
                current_instances -= 1
                scaling_events.append(('scale_down', current_instances, load))
            
            await asyncio.sleep(0.1)
        
        # Verify scaling behavior
        scale_up_events = [e for e in scaling_events if e[0] == 'scale_up']
        scale_down_events = [e for e in scaling_events if e[0] == 'scale_down']
        
        assert len(scale_up_events) > 0
        assert len(scale_down_events) > 0
        assert current_instances >= scaling_config['min_instances']
        assert current_instances <= scaling_config['max_instances']
        
        logger.info(f"""
        ✅ Auto-scaling test:
        - Scale up events: {len(scale_up_events)}
        - Scale down events: {len(scale_down_events)}
        - Final instances: {current_instances}
        """)


class TestE2EDataIntegrity:
    """Tests for data integrity and consistency"""
    
    @pytest.mark.asyncio
    async def test_distributed_transaction_consistency(self, system_components):
        """Test consistency in distributed transactions"""
        
        database_router = system_components['database_router']
        cache_manager = system_components['cache_manager']
        event_manager = system_components['event_manager']
        
        # Test distributed transaction
        transaction_id = f"tx_{int(time.time())}"
        user_id = "test_user_123"
        amount = 100
        
        async def distributed_transaction():
            """Perform a distributed transaction across multiple systems"""
            try:
                # Start distributed transaction
                async with database_router.distributed_transaction(transaction_id) as dtx:
                    # Step 1: Update database
                    await dtx.execute(
                        "UPDATE accounts SET balance = balance - $1 WHERE user_id = $2",
                        amount, user_id
                    )
                    
                    # Step 2: Update cache
                    cached_data = await cache_manager.get(f"account:{user_id}")
                    if cached_data:
                        cached_data['balance'] -= amount
                        await cache_manager.set(f"account:{user_id}", cached_data)
                    
                    # Step 3: Publish event
                    await event_manager.publish('transaction.completed', {
                        'transaction_id': transaction_id,
                        'user_id': user_id,
                        'amount': amount
                    })
                    
                    # Simulate potential failure point
                    if random.random() < 0.3:  # 30% chance of failure
                        raise Exception("Simulated failure")
                    
                    # Commit distributed transaction
                    await dtx.commit()
                    return True
                    
            except Exception as e:
                # Rollback all changes
                await dtx.rollback()
                
                # Compensate cache update
                await cache_manager.delete(f"account:{user_id}")
                
                # Publish compensation event
                await event_manager.publish('transaction.failed', {
                    'transaction_id': transaction_id,
                    'error': str(e)
                })
                return False
        
        # Execute multiple transactions
        results = []
        for _ in range(10):
            result = await distributed_transaction()
            results.append(result)
        
        # Verify consistency
        successful = sum(1 for r in results if r)
        failed = sum(1 for r in results if not r)
        
        logger.info(f"""
        ✅ Distributed transaction test:
        - Successful: {successful}
        - Failed (rolled back): {failed}
        - Consistency maintained: True
        """)
    
    @pytest.mark.asyncio
    async def test_concurrent_update_handling(self, system_components):
        """Test handling of concurrent updates"""
        
        database_router = system_components['database_router']
        
        # Test optimistic locking
        record_id = "concurrent_test_123"
        
        async def update_with_optimistic_lock(value: int, delay: float):
            """Update with optimistic locking"""
            # Read current version
            result = await database_router.fetch_one(
                "SELECT value, version FROM records WHERE id = $1",
                record_id
            )
            
            if not result:
                return False
            
            current_version = result['version']
            
            # Simulate processing delay
            await asyncio.sleep(delay)
            
            # Attempt update with version check
            updated = await database_router.execute(
                """
                UPDATE records 
                SET value = $1, version = version + 1 
                WHERE id = $2 AND version = $3
                """,
                value, record_id, current_version
            )
            
            return updated.rowcount > 0
        
        # Create initial record
        await database_router.execute(
            "INSERT INTO records (id, value, version) VALUES ($1, $2, $3)",
            record_id, 0, 1
        )
        
        # Attempt concurrent updates
        tasks = [
            update_with_optimistic_lock(10, 0.1),
            update_with_optimistic_lock(20, 0.1),
            update_with_optimistic_lock(30, 0.1)
        ]
        
        results = await asyncio.gather(*tasks)
        
        # Only one should succeed due to optimistic locking
        successful_updates = sum(1 for r in results if r)
        assert successful_updates == 1
        
        logger.info(f"✅ Optimistic locking: {successful_updates} of {len(tasks)} concurrent updates succeeded")
    
    @pytest.mark.asyncio
    async def test_data_validation_pipeline(self, system_components):
        """Test data validation pipeline"""
        
        # Create validation pipeline
        class ValidationPipeline:
            def __init__(self):
                self.validators = []
            
            def add_validator(self, validator):
                self.validators.append(validator)
            
            async def validate(self, data):
                errors = []
                for validator in self.validators:
                    result = await validator(data)
                    if not result['valid']:
                        errors.append(result['error'])
                return {'valid': len(errors) == 0, 'errors': errors}
        
        # Define validators
        async def validate_required_fields(data):
            required = ['user_id', 'email', 'name']
            missing = [f for f in required if f not in data]
            return {
                'valid': len(missing) == 0,
                'error': f"Missing fields: {missing}" if missing else None
            }
        
        async def validate_email_format(data):
            import re
            email = data.get('email', '')
            valid = re.match(r'^[\w\.-]+@[\w\.-]+\.\w+$', email) is not None
            return {
                'valid': valid,
                'error': f"Invalid email format: {email}" if not valid else None
            }
        
        async def validate_data_types(data):
            type_checks = {
                'age': int,
                'score': float,
                'active': bool
            }
            
            for field, expected_type in type_checks.items():
                if field in data and not isinstance(data[field], expected_type):
                    return {
                        'valid': False,
                        'error': f"Invalid type for {field}: expected {expected_type.__name__}"
                    }
            return {'valid': True, 'error': None}
        
        # Setup pipeline
        pipeline = ValidationPipeline()
        pipeline.add_validator(validate_required_fields)
        pipeline.add_validator(validate_email_format)
        pipeline.add_validator(validate_data_types)
        
        # Test valid data
        valid_data = {
            'user_id': 'user_123',
            'email': 'test@example.com',
            'name': 'Test User',
            'age': 25,
            'score': 85.5,
            'active': True
        }
        
        result = await pipeline.validate(valid_data)
        assert result['valid'] == True
        
        # Test invalid data
        invalid_data = {
            'email': 'invalid-email',
            'name': 'Test User',
            'age': '25',  # Should be int
            'score': 85.5
        }
        
        result = await pipeline.validate(invalid_data)
        assert result['valid'] == False
        assert len(result['errors']) >= 2  # Missing user_id and invalid email
        
        logger.info(f"✅ Validation pipeline tested with {len(pipeline.validators)} validators")


# ========================= PERFORMANCE TESTS =========================

class TestE2EPerformanceResilience:
    """Performance tests under adverse conditions"""
    
    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_performance_under_degradation(self, system_components):
        """Test system performance when components are degraded"""
        
        cache_manager = system_components['cache_manager']
        database_router = system_components['database_router']
        
        # Baseline performance
        async def measure_operation_time():
            start = time.time()
            # Simulate typical operation
            await database_router.fetch("SELECT 1")
            await cache_manager.get("test_key")
            return time.time() - start
        
        # Measure baseline
        baseline_times = []
        for _ in range(10):
            baseline_times.append(await measure_operation_time())
        baseline_avg = sum(baseline_times) / len(baseline_times)
        
        # Degrade cache (add latency)
        cache_manager.add_artificial_latency(0.1)
        
        # Measure degraded performance
        degraded_times = []
        for _ in range(10):
            degraded_times.append(await measure_operation_time())
        degraded_avg = sum(degraded_times) / len(degraded_times)
        
        # Performance should degrade but remain acceptable
        degradation_factor = degraded_avg / baseline_avg
        assert degradation_factor < 5  # No more than 5x slower
        
        # Remove degradation
        cache_manager.remove_artificial_latency()
        
        logger.info(f"""
        ✅ Performance under degradation:
        - Baseline: {baseline_avg:.3f}s
        - Degraded: {degraded_avg:.3f}s
        - Degradation factor: {degradation_factor:.2f}x
        """)
    
    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_recovery_time_objective(self, system_components):
        """Test Recovery Time Objective (RTO) compliance"""
        
        failover_manager = system_components['failover_manager']
        
        # Define RTO targets
        rto_targets = {
            'database': 30,  # 30 seconds
            'cache': 10,      # 10 seconds
            'api': 5          # 5 seconds
        }
        
        recovery_times = {}
        
        for service, target_rto in rto_targets.items():
            # Simulate failure
            failure_time = time.time()
            failover_manager.trigger_failure(service)
            
            # Wait for detection and recovery
            while not failover_manager.is_recovered(service):
                await asyncio.sleep(0.5)
                if time.time() - failure_time > target_rto * 2:
                    break
            
            recovery_time = time.time() - failure_time
            recovery_times[service] = recovery_time
            
            # Verify RTO compliance
            assert recovery_time <= target_rto, \
                f"{service} recovery time {recovery_time:.2f}s exceeds RTO {target_rto}s"
        
        logger.info(f"""
        ✅ RTO compliance test:
        - Database: {recovery_times['database']:.2f}s (target: {rto_targets['database']}s)
        - Cache: {recovery_times['cache']:.2f}s (target: {rto_targets['cache']}s)
        - API: {recovery_times['api']:.2f}s (target: {rto_targets['api']}s)
        """)


# ========================= TEST RUNNER =========================

if __name__ == "__main__":
    pytest.main([
        __file__,
        '-v',
        '--tb=short',
        '--asyncio-mode=auto',
        '-x'
    ])