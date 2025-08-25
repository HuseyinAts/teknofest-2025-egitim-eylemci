"""
Test suite for production-ready database connection pooling
"""

import pytest
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from unittest.mock import Mock, patch

from src.database.session import (
    engine,
    SessionLocal,
    get_db_stats,
    check_db_health,
    pool_metrics,
    get_db_session,
    monitor_pool_health,
    PoolMetrics
)
from src.database.health import (
    DatabaseHealthChecker,
    check_database_health,
    get_health_summary
)


class TestConnectionPooling:
    """Test database connection pooling functionality"""
    
    def test_pool_configuration(self):
        """Test that pool is configured correctly"""
        pool = engine.pool
        assert pool is not None
        
        # Check pool configuration
        assert hasattr(pool, 'size')
        assert hasattr(pool, 'overflow')
        
        # Verify pre-ping is enabled
        assert engine.pool._pre_ping is True
    
    def test_get_db_stats(self):
        """Test database statistics retrieval"""
        stats = get_db_stats()
        
        assert isinstance(stats, dict)
        assert 'pool_size' in stats
        assert 'checked_in' in stats
        assert 'checked_out' in stats
        assert 'utilization' in stats
        assert 'health' in stats
    
    def test_check_db_health(self):
        """Test database health check"""
        health = check_db_health()
        
        assert isinstance(health, dict)
        assert 'status' in health
        assert 'latency_ms' in health
        assert 'pool_stats' in health
        assert health['status'] in ['healthy', 'degraded', 'unhealthy', 'error', 'timeout']
    
    def test_pool_metrics_tracking(self):
        """Test connection pool metrics tracking"""
        metrics = PoolMetrics()
        
        # Test checkout tracking
        metrics.record_checkout(1)
        assert metrics.checkout_count == 1
        
        # Test checkin tracking
        time.sleep(0.1)
        metrics.record_checkin(1)
        assert metrics.checkout_time_sum > 0
        
        # Test stats retrieval
        stats = metrics.get_stats()
        assert 'connections_created' in stats
        assert 'avg_checkout_time' in stats
        assert stats['avg_checkout_time'] > 0
    
    def test_connection_lifecycle(self):
        """Test connection checkout and checkin"""
        initial_stats = get_db_stats()
        
        # Get a connection
        with get_db_session() as session:
            # Connection should be checked out
            during_stats = get_db_stats()
            assert during_stats['checked_out'] >= initial_stats['checked_out']
        
        # Connection should be returned
        after_stats = get_db_stats()
        assert after_stats['checked_in'] >= initial_stats['checked_in']
    
    def test_concurrent_connections(self):
        """Test handling multiple concurrent connections"""
        def use_connection(thread_id):
            try:
                with get_db_session() as session:
                    # Simulate work
                    time.sleep(0.1)
                    return thread_id, True
            except Exception as e:
                return thread_id, False
        
        # Test with multiple threads
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(use_connection, i) for i in range(10)]
            results = [future.result() for future in as_completed(futures)]
        
        # All connections should succeed
        assert all(success for _, success in results)
        
        # Check final pool state
        final_stats = get_db_stats()
        assert final_stats['health'] in ['healthy', 'warning']
    
    def test_pool_exhaustion_handling(self):
        """Test behavior when pool is exhausted"""
        # This test would require mocking to simulate pool exhaustion
        # as we don't want to actually exhaust the pool in tests
        pass
    
    def test_connection_timeout(self):
        """Test connection timeout handling"""
        # Mock a slow database response
        with patch('src.database.session.engine.connect') as mock_connect:
            mock_connect.side_effect = TimeoutError("Connection timeout")
            
            health = check_db_health()
            assert health['status'] == 'timeout'
            assert pool_metrics.pool_timeouts > 0
    
    def test_monitor_pool_health(self):
        """Test pool health monitoring"""
        # This should not raise any exceptions
        monitor_pool_health()
        
        # Check that metrics were recorded
        stats = pool_metrics.get_stats()
        assert stats is not None


class TestDatabaseHealthChecker:
    """Test database health checking functionality"""
    
    @pytest.fixture
    def health_checker(self):
        """Create a health checker instance"""
        return DatabaseHealthChecker()
    
    def test_check_connectivity(self, health_checker):
        """Test basic connectivity check"""
        result = health_checker.check_connectivity()
        
        assert 'status' in result
        assert 'message' in result
        if result['status'] == 'healthy':
            assert 'response_time_ms' in result
    
    def test_check_performance(self, health_checker):
        """Test performance metrics check"""
        result = health_checker.check_performance()
        
        assert 'status' in result
        if result['status'] != 'error':
            assert 'metrics' in result
            metrics = result['metrics']
            assert 'simple_query_ms' in metrics
    
    def test_check_connection_pool(self, health_checker):
        """Test connection pool check"""
        result = health_checker.check_connection_pool()
        
        assert 'status' in result
        if result['status'] != 'error':
            assert 'pool_stats' in result
            assert 'database_stats' in result
    
    def test_check_all(self, health_checker):
        """Test comprehensive health check"""
        results = health_checker.check_all()
        
        assert 'timestamp' in results
        assert 'status' in results
        assert 'checks' in results
        assert 'execution_time_ms' in results
        
        # Verify individual checks were run
        checks = results['checks']
        expected_checks = [
            'connectivity',
            'performance',
            'connection_pool'
        ]
        
        for check in expected_checks:
            assert check in checks
    
    def test_health_summary(self, health_checker):
        """Test health summary retrieval"""
        # Run a check first
        health_checker.check_all()
        
        # Get summary
        summary = health_checker.get_health_summary()
        
        assert 'status' in summary
        assert 'timestamp' in summary
        assert 'execution_time_ms' in summary
    
    def test_health_history(self, health_checker):
        """Test health history tracking"""
        # Run multiple checks
        for _ in range(3):
            health_checker.check_all()
            time.sleep(0.1)
        
        # Get history
        history = health_checker.get_health_history(limit=2)
        
        assert len(history) <= 2
        for entry in history:
            assert 'timestamp' in entry
            assert 'status' in entry


class TestProductionReadiness:
    """Test production-ready features"""
    
    def test_connection_recycling(self):
        """Test that old connections are recycled"""
        # This would require time manipulation or mocking
        # to test effectively without waiting
        pass
    
    def test_connection_validation(self):
        """Test pre-ping connection validation"""
        # Get initial metrics
        initial_failed = pool_metrics.connections_failed
        
        # Force a connection check
        with get_db_session() as session:
            # Connection should be validated
            pass
        
        # Failed connections should not increase for valid connection
        assert pool_metrics.connections_failed == initial_failed
    
    def test_graceful_shutdown(self):
        """Test graceful shutdown of connections"""
        from src.database.session import close_all_sessions
        
        # This should not raise any exceptions
        close_all_sessions()
        
        # Engine should be disposed
        assert engine.pool.checkedout() == 0
    
    def test_high_load_simulation(self):
        """Simulate high load conditions"""
        def stress_test(duration=1):
            end_time = time.time() + duration
            success_count = 0
            error_count = 0
            
            while time.time() < end_time:
                try:
                    with get_db_session() as session:
                        # Simulate quick query
                        success_count += 1
                except Exception:
                    error_count += 1
            
            return success_count, error_count
        
        # Run stress test with multiple threads
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(stress_test, 0.5) for _ in range(5)]
            results = [future.result() for future in as_completed(futures)]
        
        total_success = sum(s for s, _ in results)
        total_errors = sum(e for _, e in results)
        
        # Most requests should succeed
        assert total_success > 0
        assert total_errors < total_success * 0.1  # Less than 10% error rate
    
    def test_metrics_accuracy(self):
        """Test that metrics are accurately tracked"""
        # Reset metrics
        test_metrics = PoolMetrics()
        
        # Perform operations
        for i in range(5):
            test_metrics.record_checkout(i)
            time.sleep(0.01)
            test_metrics.record_checkin(i)
        
        stats = test_metrics.get_stats()
        
        assert stats['total_checkouts'] == 5
        assert stats['avg_checkout_time'] > 0
        assert stats['active_checkouts'] == 0


@pytest.mark.integration
class TestIntegrationScenarios:
    """Integration tests for real-world scenarios"""
    
    def test_database_restart_recovery(self):
        """Test recovery from database restart"""
        # This would require database control in test environment
        pass
    
    def test_network_interruption_handling(self):
        """Test handling of network interruptions"""
        # This would require network simulation
        pass
    
    def test_long_running_transaction_cleanup(self):
        """Test cleanup of long-running transactions"""
        # Start a long transaction
        with get_db_session() as session:
            # Begin transaction
            session.begin()
            
            # Simulate long operation
            time.sleep(0.5)
            
            # Rollback should happen automatically on exit
        
        # Check that connection is properly returned
        stats = get_db_stats()
        assert stats['health'] in ['healthy', 'warning']