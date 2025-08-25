# -*- coding: utf-8 -*-
"""
Core Module Tests
TEKNOFEST 2025 - Eğitim Teknolojileri
"""

import unittest
import time
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Import core modules
from src.core.base_agent import BaseAgent
from src.core.config_manager import ConfigManager
from src.core.data_processor import DataProcessor
from src.core.error_handler import (
    ErrorHandler, TeknofestError, ValidationError,
    AuthenticationError, RateLimitError, with_error_handling
)
from src.core.logging_config import setup_logging, get_logger, JSONFormatter
from src.core.rate_limiter import RateLimiter, SlidingWindowRateLimiter
from src.core.cache_manager import CacheManager, cache_result


class TestAgent(BaseAgent):
    """Test implementation of BaseAgent"""
    
    def process_request(self, request):
        if self.validate_request(request):
            return {'result': 'success'}
        raise ValidationError("Invalid request")
    
    def validate_request(self, request):
        return 'action' in request


class TestBaseAgent(unittest.TestCase):
    """Test BaseAgent functionality"""
    
    def setUp(self):
        self.agent = TestAgent("test_agent", {"test": True})
    
    def test_agent_initialization(self):
        """Test agent initialization"""
        self.assertEqual(self.agent.agent_id, "test_agent")
        self.assertEqual(self.agent.config, {"test": True})
        self.assertIsNotNone(self.agent.created_at)
    
    def test_process_valid_request(self):
        """Test processing valid request"""
        request = {'action': 'test'}
        response = self.agent.process_request(request)
        self.assertEqual(response, {'result': 'success'})
    
    def test_process_invalid_request(self):
        """Test processing invalid request"""
        request = {'invalid': 'data'}
        with self.assertRaises(ValidationError):
            self.agent.process_request(request)
    
    def test_update_metrics(self):
        """Test metrics update"""
        self.agent.update_metrics(True, 0.5)
        metrics = self.agent.get_metrics()
        
        self.assertEqual(metrics['total_interactions'], 1)
        self.assertEqual(metrics['successful_interactions'], 1)
        self.assertEqual(metrics['average_response_time'], 0.5)
        self.assertEqual(metrics['success_rate'], 1.0)
    
    def test_reset_metrics(self):
        """Test metrics reset"""
        self.agent.update_metrics(True, 0.5)
        self.agent.reset_metrics()
        metrics = self.agent.get_metrics()
        
        self.assertEqual(metrics['total_interactions'], 0)
        self.assertEqual(metrics['successful_interactions'], 0)


class TestConfigManager(unittest.TestCase):
    """Test ConfigManager functionality"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.config_dir = Path(self.temp_dir) / "configs"
        self.config_dir.mkdir()
        
        # Create test config files
        yaml_config = """
database:
  host: localhost
  port: 5432
  name: test_db
"""
        yaml_file = self.config_dir / "test.yaml"
        yaml_file.write_text(yaml_config)
        
        json_config = {"api": {"key": "test_key", "timeout": 30}}
        json_file = self.config_dir / "api.json"
        json_file.write_text(json.dumps(json_config))
        
        self.config_manager = ConfigManager(str(self.config_dir))
    
    def test_load_configs(self):
        """Test loading configuration files"""
        self.assertIn('test', self.config_manager.configs)
        self.assertIn('api', self.config_manager.configs)
    
    def test_get_config_value(self):
        """Test getting configuration values"""
        self.assertEqual(self.config_manager.get('test.database.host'), 'localhost')
        self.assertEqual(self.config_manager.get('api.key'), 'test_key')
        self.assertEqual(self.config_manager.get('nonexistent', 'default'), 'default')
    
    def test_set_config_value(self):
        """Test setting configuration values"""
        self.config_manager.set('new.value', 'test')
        self.assertEqual(self.config_manager.get('new.value'), 'test')
    
    def test_get_database_url(self):
        """Test database URL generation"""
        url = self.config_manager.get_database_url()
        self.assertIn('postgresql://', url)
        self.assertIn('localhost:5432', url)
    
    def test_environment_check(self):
        """Test environment checking"""
        self.assertTrue(self.config_manager.is_development())
        self.assertFalse(self.config_manager.is_production())


class TestDataProcessor(unittest.TestCase):
    """Test DataProcessor functionality"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.processor = DataProcessor(self.temp_dir)
    
    def test_process_educational_content(self):
        """Test processing educational content"""
        content = {
            'subject': 'Matematik',
            'topic': 'Algebra',
            'grade_level': 9,
            'content_type': 'quiz',
            'questions': [
                {
                    'text': 'Test question',
                    'options': ['A', 'B', 'C', 'D'],
                    'correct_answer': 0
                }
            ]
        }
        
        processed = self.processor.process_educational_content(content)
        
        self.assertIn('id', processed)
        self.assertEqual(processed['subject'], 'Matematik')
        self.assertEqual(len(processed['questions']), 1)
    
    def test_analyze_student_performance(self):
        """Test student performance analysis"""
        performance_data = [
            {'topic': 'Algebra', 'score': 80, 'is_correct': True},
            {'topic': 'Algebra', 'score': 60, 'is_correct': False},
            {'topic': 'Geometry', 'score': 90, 'is_correct': True},
        ]
        
        analysis = self.processor.analyze_student_performance(performance_data)
        
        self.assertEqual(analysis['total_activities'], 3)
        self.assertIn('strengths', analysis)
        self.assertIn('weaknesses', analysis)
        self.assertIn('recommendations', analysis)
    
    def test_save_and_load_dataset(self):
        """Test saving and loading datasets"""
        data = {'test': 'data', 'value': 123}
        
        # Save as JSON
        path = self.processor.save_dataset(data, 'test.json', 'json')
        self.assertTrue(path.exists())
        
        # Load back
        loaded = self.processor.load_dataset('test.json', 'json')
        self.assertEqual(loaded, data)


class TestErrorHandler(unittest.TestCase):
    """Test ErrorHandler functionality"""
    
    def setUp(self):
        self.handler = ErrorHandler()
    
    def test_handle_teknofest_error(self):
        """Test handling TeknofestError"""
        error = ValidationError("Test error", "field1")
        response = self.handler.handle_error(error)
        
        self.assertFalse(response['success'])
        self.assertEqual(response['error']['code'], 'VALIDATION_ERROR')
        self.assertEqual(response['error']['message'], 'Test error')
    
    def test_handle_generic_error(self):
        """Test handling generic error"""
        error = ValueError("Generic error")
        response = self.handler.handle_error(error)
        
        self.assertFalse(response['success'])
        self.assertEqual(response['error']['code'], 'INTERNAL_ERROR')
        self.assertEqual(response['error']['type'], 'ValueError')
    
    def test_custom_error_handler(self):
        """Test custom error handler registration"""
        def custom_handler(error, context):
            return {'custom': True, 'message': str(error)}
        
        self.handler.register_handler(KeyError, custom_handler)
        
        error = KeyError("test_key")
        response = self.handler.handle_error(error)
        
        self.assertTrue(response['custom'])
        self.assertIn('test_key', response['message'])
    
    def test_error_statistics(self):
        """Test error statistics tracking"""
        self.handler.handle_error(ValidationError("Test"))
        self.handler.handle_error(AuthenticationError())
        self.handler.handle_error(ValidationError("Test2"))
        
        stats = self.handler.get_stats()
        
        self.assertEqual(stats['total_errors'], 3)
        self.assertEqual(stats['errors_by_type']['ValidationError'], 2)
        self.assertEqual(stats['errors_by_code']['VALIDATION_ERROR'], 2)
    
    def test_with_error_handling_decorator(self):
        """Test error handling decorator"""
        @with_error_handling()
        def test_function(x):
            if x < 0:
                raise ValueError("Negative value")
            return x * 2
        
        # Test successful call
        result = test_function(5)
        self.assertEqual(result, 10)
        
        # Test error call
        result = test_function(-1)
        self.assertFalse(result['success'])
        self.assertIn('error', result)


class TestRateLimiter(unittest.TestCase):
    """Test RateLimiter functionality"""
    
    def test_token_bucket_rate_limiter(self):
        """Test token bucket rate limiter"""
        limiter = RateLimiter(rate=5, per=1)  # 5 requests per second
        
        # Should allow first 5 requests
        for i in range(5):
            self.assertTrue(limiter.is_allowed("test_key"))
        
        # 6th request should be denied
        self.assertFalse(limiter.is_allowed("test_key"))
        
        # After waiting, should allow again
        time.sleep(0.3)
        self.assertTrue(limiter.is_allowed("test_key"))
    
    def test_sliding_window_rate_limiter(self):
        """Test sliding window rate limiter"""
        limiter = SlidingWindowRateLimiter(rate=3, window=1)
        
        # Should allow first 3 requests
        for i in range(3):
            self.assertTrue(limiter.is_allowed("test_key"))
        
        # 4th request should be denied
        self.assertFalse(limiter.is_allowed("test_key"))
        
        # Check remaining
        self.assertEqual(limiter.get_remaining("test_key"), 0)
    
    def test_rate_limiter_reset(self):
        """Test rate limiter reset"""
        limiter = RateLimiter(rate=5, per=60)
        
        # Use up some tokens
        for i in range(3):
            limiter.is_allowed("test_key")
        
        # Reset
        limiter.reset("test_key")
        
        # Should have full allowance again
        for i in range(5):
            self.assertTrue(limiter.is_allowed("test_key"))


class TestCacheManager(unittest.TestCase):
    """Test CacheManager functionality"""
    
    def setUp(self):
        self.cache = CacheManager(max_size=3, default_ttl=1)
    
    def test_cache_set_and_get(self):
        """Test cache set and get operations"""
        self.cache.set("key1", "value1")
        self.assertEqual(self.cache.get("key1"), "value1")
        self.assertIsNone(self.cache.get("nonexistent"))
    
    def test_cache_expiration(self):
        """Test cache TTL expiration"""
        self.cache.set("key1", "value1", ttl=0.1)
        self.assertEqual(self.cache.get("key1"), "value1")
        
        time.sleep(0.2)
        self.assertIsNone(self.cache.get("key1"))
    
    def test_lru_eviction(self):
        """Test LRU eviction"""
        # Fill cache to capacity
        self.cache.set("key1", "value1")
        self.cache.set("key2", "value2")
        self.cache.set("key3", "value3")
        
        # Access key1 to make it recently used
        self.cache.get("key1")
        
        # Add new item, should evict key2 (least recently used)
        self.cache.set("key4", "value4")
        
        self.assertIsNotNone(self.cache.get("key1"))
        self.assertIsNone(self.cache.get("key2"))  # Evicted
        self.assertIsNotNone(self.cache.get("key3"))
        self.assertIsNotNone(self.cache.get("key4"))
    
    def test_cache_statistics(self):
        """Test cache statistics"""
        self.cache.set("key1", "value1")
        self.cache.get("key1")  # Hit
        self.cache.get("key2")  # Miss
        
        stats = self.cache.get_stats()
        
        self.assertEqual(stats['hits'], 1)
        self.assertEqual(stats['misses'], 1)
        self.assertEqual(stats['size'], 1)
        self.assertAlmostEqual(stats['hit_rate'], 0.5)
    
    def test_cache_decorator(self):
        """Test cache result decorator"""
        call_count = 0
        
        @cache_result(self.cache, ttl=1)
        def expensive_function(x):
            nonlocal call_count
            call_count += 1
            return x * 2
        
        # First call should execute function
        result1 = expensive_function(5)
        self.assertEqual(result1, 10)
        self.assertEqual(call_count, 1)
        
        # Second call should use cache
        result2 = expensive_function(5)
        self.assertEqual(result2, 10)
        self.assertEqual(call_count, 1)  # Not incremented
        
        # Different argument should execute function
        result3 = expensive_function(3)
        self.assertEqual(result3, 6)
        self.assertEqual(call_count, 2)


class TestLoggingConfig(unittest.TestCase):
    """Test logging configuration"""
    
    def test_json_formatter(self):
        """Test JSON log formatter"""
        formatter = JSONFormatter()
        record = MagicMock()
        record.levelname = "INFO"
        record.name = "test_logger"
        record.getMessage.return_value = "Test message"
        record.module = "test_module"
        record.funcName = "test_func"
        record.lineno = 42
        record.exc_info = None
        record.__dict__ = {
            'name': 'test_logger',
            'levelname': 'INFO',
            'extra_field': 'extra_value'
        }
        
        formatted = formatter.format(record)
        log_obj = json.loads(formatted)
        
        self.assertEqual(log_obj['level'], 'INFO')
        self.assertEqual(log_obj['message'], 'Test message')
        self.assertEqual(log_obj['line'], 42)
    
    def test_setup_logging(self):
        """Test logging setup"""
        with tempfile.TemporaryDirectory() as temp_dir:
            setup_logging(
                level="DEBUG",
                log_dir=temp_dir,
                app_name="test_app",
                use_json=True,
                console=False,
                file=True
            )
            
            logger = get_logger("test_logger")
            logger.info("Test log message")
            
            # Check log file exists
            log_file = Path(temp_dir) / "test_app.log"
            self.assertTrue(log_file.exists())


if __name__ == '__main__':
    unittest.main()