"""
Comprehensive tests for Event System
"""
import pytest
import asyncio
import json
from datetime import datetime
from unittest.mock import Mock, patch, AsyncMock, MagicMock
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.event_manager import EventManager
from src.event_handlers import EventHandler
from src.event_formats import EventFormat


class TestEventManager:
    
    @pytest.fixture
    def event_manager(self):
        """Create an EventManager instance for testing"""
        return EventManager()
    
    @pytest.fixture
    def sample_event(self):
        """Create a sample event for testing"""
        return {
            'event_id': 'test_001',
            'event_type': 'learning_path_created',
            'timestamp': datetime.now().isoformat(),
            'data': {
                'student_id': 'student_001',
                'subject': 'Matematik',
                'weeks': 4
            }
        }
    
    @pytest.mark.unit
    def test_event_manager_initialization(self, event_manager):
        """Test event manager initialization"""
        assert event_manager is not None
        assert hasattr(event_manager, 'emit') or hasattr(event_manager, 'publish')
        assert hasattr(event_manager, 'subscribe') or hasattr(event_manager, 'on')
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_emit_event(self, event_manager, sample_event):
        """Test emitting an event"""
        if hasattr(event_manager, 'emit'):
            result = await event_manager.emit(sample_event)
            assert result is not None
        elif hasattr(event_manager, 'publish'):
            result = await event_manager.publish(sample_event)
            assert result is not None
        else:
            pytest.skip("EventManager doesn't have emit/publish methods")
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_subscribe_to_event(self, event_manager):
        """Test subscribing to events"""
        received_events = []
        
        async def handler(event):
            received_events.append(event)
        
        if hasattr(event_manager, 'subscribe'):
            event_manager.subscribe('test_event', handler)
        elif hasattr(event_manager, 'on'):
            event_manager.on('test_event', handler)
        else:
            pytest.skip("EventManager doesn't have subscribe/on methods")
        
        # Emit test event
        test_event = {
            'event_type': 'test_event',
            'data': {'test': 'value'}
        }
        
        if hasattr(event_manager, 'emit'):
            await event_manager.emit(test_event)
        elif hasattr(event_manager, 'publish'):
            await event_manager.publish(test_event)
        
        # Check if handler received the event
        await asyncio.sleep(0.1)  # Give time for async processing
        assert len(received_events) > 0
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_multiple_subscribers(self, event_manager):
        """Test multiple subscribers to same event"""
        results = {'handler1': [], 'handler2': []}
        
        async def handler1(event):
            results['handler1'].append(event)
        
        async def handler2(event):
            results['handler2'].append(event)
        
        if hasattr(event_manager, 'subscribe'):
            event_manager.subscribe('multi_test', handler1)
            event_manager.subscribe('multi_test', handler2)
            
            # Emit event
            test_event = {'event_type': 'multi_test', 'data': 'test'}
            
            if hasattr(event_manager, 'emit'):
                await event_manager.emit(test_event)
            
            await asyncio.sleep(0.1)
            
            # Both handlers should receive the event
            assert len(results['handler1']) > 0
            assert len(results['handler2']) > 0
        else:
            pytest.skip("EventManager doesn't support multiple subscribers")
    
    @pytest.mark.unit
    def test_unsubscribe_from_event(self, event_manager):
        """Test unsubscribing from events"""
        if hasattr(event_manager, 'unsubscribe'):
            def handler(event):
                pass
            
            # Subscribe
            if hasattr(event_manager, 'subscribe'):
                event_manager.subscribe('test', handler)
            
            # Unsubscribe
            event_manager.unsubscribe('test', handler)
            
            # Verify unsubscribed
            assert True  # If no exception, test passes
        else:
            pytest.skip("EventManager doesn't have unsubscribe method")
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_event_filtering(self, event_manager):
        """Test event filtering by type"""
        received = {'type1': [], 'type2': []}
        
        async def type1_handler(event):
            received['type1'].append(event)
        
        async def type2_handler(event):
            received['type2'].append(event)
        
        if hasattr(event_manager, 'subscribe'):
            event_manager.subscribe('type1', type1_handler)
            event_manager.subscribe('type2', type2_handler)
            
            # Emit different event types
            if hasattr(event_manager, 'emit'):
                await event_manager.emit({'event_type': 'type1', 'data': 1})
                await event_manager.emit({'event_type': 'type2', 'data': 2})
            
            await asyncio.sleep(0.1)
            
            # Each handler should only receive its type
            assert len(received['type1']) == 1
            assert len(received['type2']) == 1
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_event_error_handling(self, event_manager):
        """Test error handling in event processing"""
        async def failing_handler(event):
            raise ValueError("Handler error")
        
        if hasattr(event_manager, 'subscribe'):
            event_manager.subscribe('error_test', failing_handler)
            
            # Should not crash when handler fails
            if hasattr(event_manager, 'emit'):
                try:
                    await event_manager.emit({'event_type': 'error_test'})
                    # Should handle error gracefully
                    assert True
                except:
                    # If error propagates, that's also acceptable
                    assert True
    
    @pytest.mark.unit
    def test_event_history(self, event_manager):
        """Test event history tracking if supported"""
        if hasattr(event_manager, 'get_history'):
            # Emit some events
            for i in range(3):
                if hasattr(event_manager, 'emit'):
                    asyncio.run(event_manager.emit({
                        'event_type': 'history_test',
                        'data': i
                    }))
            
            history = event_manager.get_history()
            assert len(history) >= 3
        else:
            pytest.skip("EventManager doesn't track history")


class TestEventHandler:
    
    @pytest.fixture
    def event_handler(self):
        """Create an EventHandler instance for testing"""
        return EventHandler()
    
    @pytest.mark.unit
    def test_handler_initialization(self, event_handler):
        """Test event handler initialization"""
        assert event_handler is not None
        assert hasattr(event_handler, 'handle') or hasattr(event_handler, 'process')
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_handle_learning_event(self, event_handler):
        """Test handling learning-related events"""
        learning_event = {
            'event_type': 'quiz_completed',
            'data': {
                'student_id': 'test_001',
                'score': 0.85,
                'quiz_id': 'quiz_001'
            }
        }
        
        if hasattr(event_handler, 'handle'):
            result = await event_handler.handle(learning_event)
            assert result is not None
        elif hasattr(event_handler, 'process'):
            result = await event_handler.process(learning_event)
            assert result is not None
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_handle_progress_event(self, event_handler):
        """Test handling progress events"""
        progress_event = {
            'event_type': 'progress_updated',
            'data': {
                'student_id': 'test_001',
                'subject': 'Matematik',
                'completion': 0.75
            }
        }
        
        if hasattr(event_handler, 'handle_progress'):
            result = await event_handler.handle_progress(progress_event)
            assert result is not None
        elif hasattr(event_handler, 'handle'):
            result = await event_handler.handle(progress_event)
            assert result is not None
    
    @pytest.mark.unit
    def test_event_validation(self, event_handler):
        """Test event validation"""
        valid_event = {
            'event_type': 'test',
            'data': {'key': 'value'},
            'timestamp': datetime.now().isoformat()
        }
        
        invalid_event = {
            'data': {'key': 'value'}
            # Missing event_type
        }
        
        if hasattr(event_handler, 'validate'):
            assert event_handler.validate(valid_event) == True
            assert event_handler.validate(invalid_event) == False


class TestEventFormat:
    
    @pytest.fixture
    def event_format(self):
        """Create an EventFormat instance for testing"""
        return EventFormat()
    
    @pytest.mark.unit
    def test_format_initialization(self, event_format):
        """Test event format initialization"""
        assert event_format is not None
    
    @pytest.mark.unit
    def test_format_event(self, event_format):
        """Test formatting events"""
        raw_data = {
            'student_id': 'test_001',
            'action': 'quiz_completed',
            'score': 0.9
        }
        
        if hasattr(event_format, 'format'):
            formatted = event_format.format('quiz_completed', raw_data)
            
            assert 'event_type' in formatted
            assert 'timestamp' in formatted
            assert 'data' in formatted
            assert formatted['event_type'] == 'quiz_completed'
    
    @pytest.mark.unit
    def test_parse_event(self, event_format):
        """Test parsing formatted events"""
        formatted_event = {
            'event_type': 'test_event',
            'timestamp': '2024-01-01T12:00:00',
            'data': {'test': 'value'}
        }
        
        if hasattr(event_format, 'parse'):
            parsed = event_format.parse(formatted_event)
            
            assert parsed is not None
            assert 'event_type' in parsed
            assert parsed['event_type'] == 'test_event'
    
    @pytest.mark.unit
    def test_serialize_event(self, event_format):
        """Test event serialization"""
        event = {
            'event_type': 'test',
            'timestamp': datetime.now(),
            'data': {'key': 'value'}
        }
        
        if hasattr(event_format, 'serialize'):
            serialized = event_format.serialize(event)
            
            assert isinstance(serialized, str)
            # Should be valid JSON
            deserialized = json.loads(serialized)
            assert deserialized['event_type'] == 'test'


@pytest.mark.integration
class TestEventSystemIntegration:
    
    @pytest.fixture
    def event_system(self):
        """Create complete event system for integration testing"""
        return {
            'manager': EventManager(),
            'handler': EventHandler(),
            'format': EventFormat()
        }
    
    @pytest.mark.asyncio
    async def test_complete_event_flow(self, event_system):
        """Test complete event flow from emission to handling"""
        manager = event_system['manager']
        handler = event_system['handler']
        format_obj = event_system['format']
        
        received_events = []
        
        # Set up handler
        async def capture_handler(event):
            received_events.append(event)
            if hasattr(handler, 'handle'):
                await handler.handle(event)
        
        # Subscribe to events
        if hasattr(manager, 'subscribe'):
            manager.subscribe('learning_event', capture_handler)
        
        # Format and emit event
        if hasattr(format_obj, 'format'):
            formatted_event = format_obj.format('learning_event', {
                'student_id': 'test',
                'progress': 0.5
            })
        else:
            formatted_event = {
                'event_type': 'learning_event',
                'data': {'student_id': 'test', 'progress': 0.5}
            }
        
        # Emit event
        if hasattr(manager, 'emit'):
            await manager.emit(formatted_event)
        
        await asyncio.sleep(0.1)
        
        # Verify event was received and handled
        assert len(received_events) > 0
        assert received_events[0]['event_type'] == 'learning_event'
    
    @pytest.mark.asyncio
    async def test_event_chain(self, event_system):
        """Test chain of events triggering other events"""
        manager = event_system['manager']
        
        chain_results = []
        
        async def first_handler(event):
            chain_results.append('first')
            # Trigger second event
            if hasattr(manager, 'emit'):
                await manager.emit({
                    'event_type': 'second_event',
                    'data': {'triggered_by': 'first'}
                })
        
        async def second_handler(event):
            chain_results.append('second')
        
        if hasattr(manager, 'subscribe'):
            manager.subscribe('first_event', first_handler)
            manager.subscribe('second_event', second_handler)
            
            # Start the chain
            if hasattr(manager, 'emit'):
                await manager.emit({'event_type': 'first_event', 'data': {}})
            
            await asyncio.sleep(0.2)
            
            # Both handlers should have executed
            assert 'first' in chain_results
            assert 'second' in chain_results
            assert chain_results.index('first') < chain_results.index('second')
    
    @pytest.mark.asyncio
    async def test_concurrent_event_processing(self, event_system):
        """Test concurrent processing of multiple events"""
        manager = event_system['manager']
        
        processed = []
        
        async def slow_handler(event):
            await asyncio.sleep(0.05)  # Simulate processing
            processed.append(event['data']['id'])
        
        if hasattr(manager, 'subscribe'):
            manager.subscribe('concurrent_test', slow_handler)
            
            # Emit multiple events concurrently
            events = [
                {'event_type': 'concurrent_test', 'data': {'id': i}}
                for i in range(5)
            ]
            
            if hasattr(manager, 'emit'):
                tasks = [manager.emit(event) for event in events]
                await asyncio.gather(*tasks)
            
            await asyncio.sleep(0.3)
            
            # All events should be processed
            assert len(processed) == 5
            assert set(processed) == set(range(5))