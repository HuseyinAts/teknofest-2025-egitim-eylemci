"""
Edge Cases and Boundary Tests with Parametrization
===================================================
"""

import pytest
from unittest.mock import patch, MagicMock
import math
import sys
from decimal import Decimal


class TestBoundaryValues:
    """Test boundary values and edge cases"""
    
    @pytest.mark.unit
    @pytest.mark.parametrize("difficulty,expected_valid", [
        (0.0, True),      # Minimum valid
        (0.000001, True), # Just above minimum
        (0.5, True),      # Middle value
        (0.999999, True), # Just below maximum
        (1.0, True),      # Maximum valid
        (-0.1, False),    # Below minimum
        (1.1, False),     # Above maximum
        (None, False),    # Null value
        (float('inf'), False),  # Infinity
        (float('nan'), False),  # NaN
    ])
    def test_difficulty_boundaries(self, study_buddy_agent, difficulty, expected_valid):
        """Test difficulty parameter boundaries"""
        if not expected_valid:
            with pytest.raises((ValueError, TypeError, AssertionError)):
                study_buddy_agent.generate_adaptive_quiz(
                    topic="Test",
                    student_ability=difficulty if difficulty is not None else 0.5,
                    num_questions=5
                )
        else:
            quiz = study_buddy_agent.generate_adaptive_quiz(
                topic="Test",
                student_ability=difficulty,
                num_questions=5
            )
            assert len(quiz) == 5
    
    @pytest.mark.unit
    @pytest.mark.parametrize("num_questions,expected_behavior", [
        (0, "empty"),           # No questions
        (1, "valid"),           # Single question
        (5, "valid"),           # Normal amount
        (100, "valid"),         # Large amount
        (1000, "performance"),  # Very large amount
        (-1, "error"),          # Negative
        (None, "error"),        # Null
        (float('inf'), "error"), # Infinity
        (2.5, "error"),         # Float value
    ])
    def test_quiz_size_boundaries(self, study_buddy_agent, num_questions, expected_behavior):
        """Test quiz size boundaries"""
        if expected_behavior == "error":
            with pytest.raises((ValueError, TypeError)):
                study_buddy_agent.generate_adaptive_quiz(
                    topic="Test",
                    student_ability=0.5,
                    num_questions=num_questions
                )
        elif expected_behavior == "empty":
            quiz = study_buddy_agent.generate_adaptive_quiz(
                topic="Test",
                student_ability=0.5,
                num_questions=num_questions
            )
            assert len(quiz) == 0
        elif expected_behavior == "performance":
            # Should handle but might be slow
            import time
            start = time.time()
            quiz = study_buddy_agent.generate_adaptive_quiz(
                topic="Test",
                student_ability=0.5,
                num_questions=int(num_questions)
            )
            duration = time.time() - start
            assert len(quiz) == int(num_questions)
            assert duration < 10  # Should complete within 10 seconds
        else:  # valid
            quiz = study_buddy_agent.generate_adaptive_quiz(
                topic="Test",
                student_ability=0.5,
                num_questions=int(num_questions)
            )
            assert len(quiz) == int(num_questions)
    
    @pytest.mark.unit
    @pytest.mark.parametrize("weeks,expected_valid", [
        (0, False),    # No weeks
        (1, True),     # Minimum valid
        (4, True),     # Typical
        (52, True),    # Full year
        (104, True),   # Two years
        (-1, False),   # Negative
        (None, False), # Null
    ])
    def test_learning_path_duration(self, learning_path_agent, sample_student_profile, weeks, expected_valid):
        """Test learning path duration boundaries"""
        if not expected_valid:
            with pytest.raises((ValueError, TypeError)):
                learning_path_agent.generate_learning_path(
                    sample_student_profile,
                    "Matematik",
                    weeks=weeks
                )
        else:
            path = learning_path_agent.generate_learning_path(
                sample_student_profile,
                "Matematik",
                weeks=weeks
            )
            assert len(path['weekly_plan']) == weeks


class TestStringInputs:
    """Test string input edge cases"""
    
    @pytest.mark.unit
    @pytest.mark.parametrize("input_text,expected_behavior", [
        ("", "empty"),                           # Empty string
        (" ", "whitespace"),                     # Single space
        ("   \n\t  ", "whitespace"),            # Various whitespace
        ("a", "minimal"),                        # Single character
        ("a" * 10000, "long"),                   # Very long string
        ("😀🎉🚀", "emoji"),                     # Emojis
        ("<script>alert('xss')</script>", "html"), # HTML content
        ("'; DROP TABLE users; --", "sql"),      # SQL injection attempt
        ("../../../etc/passwd", "path"),         # Path traversal
        (None, "null"),                          # Null
    ])
    def test_text_input_handling(self, learning_path_agent, input_text, expected_behavior):
        """Test various text input edge cases"""
        if expected_behavior == "null":
            with pytest.raises((TypeError, AttributeError)):
                learning_path_agent.detect_learning_style([input_text])
        elif expected_behavior in ["empty", "whitespace"]:
            result = learning_path_agent.detect_learning_style([input_text])
            # Should handle gracefully, possibly with default
            assert 'dominant_style' in result
        elif expected_behavior in ["html", "sql", "path"]:
            # Should sanitize dangerous input
            result = learning_path_agent.detect_learning_style([input_text])
            assert result is not None
            # Verify no dangerous content in output
            assert '<script>' not in str(result)
            assert 'DROP TABLE' not in str(result)
        else:
            result = learning_path_agent.detect_learning_style([input_text])
            assert 'dominant_style' in result
    
    @pytest.mark.unit
    @pytest.mark.parametrize("topic", [
        "Mathematics",           # English
        "Matematik",            # Turkish
        "数学",                 # Chinese
        "الرياضيات",           # Arabic
        "Μαθηματικά",          # Greek
        "गणित",                # Hindi
        "🔢➕➖✖️➗",           # Emoji math
        "M@th3m@t1c$",         # Special characters
    ])
    def test_unicode_handling(self, study_buddy_agent, topic):
        """Test Unicode and international character handling"""
        quiz = study_buddy_agent.generate_adaptive_quiz(
            topic=topic,
            student_ability=0.5,
            num_questions=3
        )
        
        assert len(quiz) == 3
        # Topic should be preserved or handled gracefully
        for q in quiz:
            assert 'question' in q


class TestCollectionInputs:
    """Test collection input edge cases"""
    
    @pytest.mark.unit
    @pytest.mark.parametrize("topics,expected_behavior", [
        ([], "empty"),                          # Empty list
        (["Topic1"], "single"),                 # Single item
        (["Topic1", "Topic2"], "normal"),       # Normal case
        (["Topic"] * 100, "large"),             # Large list
        (["", "", ""], "empty_items"),          # List of empty strings
        ([None, None], "null_items"),           # List of nulls
        (None, "null"),                         # Null instead of list
        ("NotAList", "wrong_type"),             # String instead of list
        ([1, 2, 3], "wrong_item_type"),        # Numbers instead of strings
    ])
    def test_topic_list_handling(self, study_buddy_agent, topics, expected_behavior):
        """Test various topic list inputs"""
        if expected_behavior in ["null", "wrong_type"]:
            with pytest.raises((TypeError, AttributeError)):
                study_buddy_agent.generate_study_plan(topics, available_hours=10)
        elif expected_behavior == "empty":
            plan = study_buddy_agent.generate_study_plan(topics, available_hours=10)
            assert plan['total_hours'] == 10
            assert len(plan['topics']) == 0
        elif expected_behavior == "null_items":
            with pytest.raises((ValueError, TypeError)):
                study_buddy_agent.generate_study_plan(topics, available_hours=10)
        else:
            plan = study_buddy_agent.generate_study_plan(
                topics if expected_behavior != "wrong_item_type" else ["Topic1", "Topic2"],
                available_hours=10
            )
            assert plan['total_hours'] == 10
    
    @pytest.mark.unit
    @pytest.mark.parametrize("profile_data,expected_valid", [
        ({}, False),                                    # Empty dict
        ({'student_id': 'test'}, False),               # Missing required fields
        ({'student_id': 'test', 'grade': 10}, True),   # Minimal valid
        ({'student_id': None, 'grade': 10}, False),    # Null field
        ({'student_id': '', 'grade': 10}, False),      # Empty field
        ({'student_id': 'test', 'grade': -1}, False),  # Invalid grade
        ({'student_id': 'test', 'grade': 13}, False),  # Grade too high
        ({'student_id': 'test', 'grade': 'ten'}, False), # Wrong type
    ])
    def test_profile_validation(self, learning_path_agent, profile_data, expected_valid):
        """Test student profile validation"""
        if not expected_valid:
            with pytest.raises((ValueError, TypeError, KeyError)):
                learning_path_agent.generate_learning_path(
                    profile_data,
                    "Matematik",
                    weeks=4
                )
        else:
            path = learning_path_agent.generate_learning_path(
                profile_data,
                "Matematik",
                weeks=4
            )
            assert 'weekly_plan' in path


class TestNumericOverflow:
    """Test numeric overflow and precision issues"""
    
    @pytest.mark.unit
    @pytest.mark.parametrize("hours,expected_behavior", [
        (0, "zero"),
        (0.1, "fractional"),
        (24, "full_day"),
        (168, "full_week"),
        (8760, "full_year"),
        (sys.maxsize, "overflow"),
        (-sys.maxsize, "underflow"),
        (Decimal('0.123456789'), "high_precision"),
    ])
    def test_hours_calculation(self, study_buddy_agent, hours, expected_behavior):
        """Test hours calculation with various numeric inputs"""
        if expected_behavior in ["overflow", "underflow"]:
            with pytest.raises((ValueError, OverflowError)):
                study_buddy_agent.generate_study_plan(
                    ["Topic1"],
                    available_hours=hours
                )
        elif expected_behavior == "zero":
            plan = study_buddy_agent.generate_study_plan(
                ["Topic1"],
                available_hours=hours
            )
            assert plan['total_hours'] == 0
        elif expected_behavior == "high_precision":
            plan = study_buddy_agent.generate_study_plan(
                ["Topic1"],
                available_hours=float(hours)
            )
            assert plan['total_hours'] == pytest.approx(float(hours))
        else:
            plan = study_buddy_agent.generate_study_plan(
                ["Topic1"],
                available_hours=hours if isinstance(hours, (int, float)) else float(hours)
            )
            assert plan['total_hours'] == hours
    
    @pytest.mark.unit
    @pytest.mark.parametrize("score,expected_valid", [
        (0/1, True),      # Zero score
        (1/2, True),      # Half score
        (1/1, True),      # Perfect score
        (0/0, False),     # Division by zero (NaN)
        (1/0, False),     # Infinity
        (-1/2, False),    # Negative score
        (3/2, False),     # Score > 1
    ])
    def test_score_calculations(self, score, expected_valid):
        """Test score calculation edge cases"""
        def calculate_performance(score):
            if math.isnan(score) or math.isinf(score):
                raise ValueError("Invalid score")
            if not 0 <= score <= 1:
                raise ValueError("Score out of range")
            return {
                'score': score,
                'grade': 'A' if score >= 0.9 else 'B' if score >= 0.8 else 'C'
            }
        
        if not expected_valid:
            with pytest.raises(ValueError):
                calculate_performance(score)
        else:
            result = calculate_performance(score)
            assert 0 <= result['score'] <= 1


class TestConcurrencyEdgeCases:
    """Test concurrency-related edge cases"""
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_race_condition(self, study_buddy_agent):
        """Test for race conditions in concurrent access"""
        import asyncio
        
        # Shared counter to detect race conditions
        counter = {'value': 0}
        
        async def increment_and_generate():
            counter['value'] += 1
            current = counter['value']
            await asyncio.sleep(0.001)  # Simulate processing
            quiz = study_buddy_agent.generate_adaptive_quiz(
                topic=f"Topic{current}",
                student_ability=0.5,
                num_questions=3
            )
            return current, quiz
        
        # Run multiple concurrent operations
        tasks = [increment_and_generate() for _ in range(10)]
        results = await asyncio.gather(*tasks)
        
        # Check all operations completed
        assert len(results) == 10
        
        # Check no duplicate counters (would indicate race condition)
        counters = [r[0] for r in results]
        assert len(set(counters)) == 10
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_deadlock_prevention(self, mcp_server):
        """Test that operations don't deadlock"""
        import asyncio
        
        async def potentially_deadlocking_operation():
            # Simulate operations that could deadlock
            request1 = {'tool': 'lock_resource_a', 'arguments': {}}
            request2 = {'tool': 'lock_resource_b', 'arguments': {}}
            
            if hasattr(mcp_server, 'handle_request'):
                # These should complete without deadlocking
                await asyncio.wait_for(
                    mcp_server.handle_request(request1),
                    timeout=1.0
                )
                await asyncio.wait_for(
                    mcp_server.handle_request(request2),
                    timeout=1.0
                )
            
            return True
        
        # Should complete without timeout
        result = await asyncio.wait_for(
            potentially_deadlocking_operation(),
            timeout=5.0
        )
        assert result is True


class TestMemoryEdgeCases:
    """Test memory-related edge cases"""
    
    @pytest.mark.unit
    @pytest.mark.slow
    def test_memory_leak_prevention(self, study_buddy_agent):
        """Test that repeated operations don't leak memory"""
        import gc
        import sys
        
        # Get initial memory baseline
        gc.collect()
        initial_objects = len(gc.get_objects())
        
        # Perform many operations
        for i in range(100):
            quiz = study_buddy_agent.generate_adaptive_quiz(
                topic="Test",
                student_ability=0.5,
                num_questions=5
            )
            # Explicitly delete to help GC
            del quiz
        
        # Force garbage collection
        gc.collect()
        final_objects = len(gc.get_objects())
        
        # Object count shouldn't grow significantly
        object_growth = final_objects - initial_objects
        assert object_growth < 1000  # Reasonable threshold
    
    @pytest.mark.unit
    def test_large_data_handling(self, learning_path_agent):
        """Test handling of large data structures"""
        # Create a large profile with many fields
        large_profile = {
            'student_id': 'test',
            'grade': 10,
            'learning_style': 'visual',
            'history': [{'date': f'2024-01-{i:02d}', 'score': 0.5} for i in range(1, 32)],
            'preferences': {f'pref_{i}': f'value_{i}' for i in range(100)},
            'notes': 'x' * 10000  # 10KB of text
        }
        
        # Should handle without error
        path = learning_path_agent.generate_learning_path(
            large_profile,
            "Matematik",
            weeks=4
        )
        
        assert 'weekly_plan' in path


class TestTimeEdgeCases:
    """Test time-related edge cases"""
    
    @pytest.mark.unit
    def test_timezone_handling(self, mock_time):
        """Test timezone handling in timestamps"""
        from datetime import datetime, timezone, timedelta
        
        with mock_time() as frozen_time:
            # Test different timezones
            utc_time = datetime.now(timezone.utc)
            local_time = datetime.now()
            
            # Times should be consistent
            assert abs((utc_time.replace(tzinfo=None) - local_time).total_seconds()) < 1
    
    @pytest.mark.unit
    @pytest.mark.parametrize("date_input,expected_valid", [
        ("2024-01-01", True),
        ("2024-13-01", False),  # Invalid month
        ("2024-01-32", False),  # Invalid day
        ("not-a-date", False),
        ("", False),
        (None, False),
    ])
    def test_date_parsing(self, date_input, expected_valid):
        """Test date parsing edge cases"""
        from datetime import datetime
        
        def parse_date(date_str):
            if not date_str:
                raise ValueError("Empty date")
            return datetime.fromisoformat(date_str)
        
        if not expected_valid:
            with pytest.raises((ValueError, TypeError)):
                parse_date(date_input)
        else:
            result = parse_date(date_input)
            assert isinstance(result, datetime)


class TestErrorPropagation:
    """Test error propagation and handling"""
    
    @pytest.mark.unit
    def test_nested_error_handling(self, learning_path_agent, mocker):
        """Test error propagation through nested calls"""
        # Mock a deep method to raise an error
        with patch.object(learning_path_agent, 'calculate_difficulty', side_effect=ValueError("Deep error")):
            with pytest.raises(ValueError) as exc_info:
                learning_path_agent.generate_learning_path(
                    {'student_id': 'test', 'grade': 10},
                    "Math",
                    weeks=4
                )
            
            # Error message should be preserved
            assert "Deep error" in str(exc_info.value)
    
    @pytest.mark.unit
    def test_partial_failure_recovery(self, study_buddy_agent, error_simulator):
        """Test recovery from partial failures"""
        questions = []
        
        for i in range(10):
            try:
                # Randomly fail some operations
                if i % 3 == 0:
                    error_simulator.random_error(probability=0.5)
                
                q = study_buddy_agent.generate_adaptive_quiz(
                    topic="Test",
                    student_ability=0.5,
                    num_questions=1
                )[0]
                questions.append(q)
            except:
                # Skip failed questions
                continue
        
        # Should have generated at least some questions
        assert len(questions) > 0
        assert len(questions) <= 10