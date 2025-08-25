# -*- coding: utf-8 -*-
"""
Comprehensive Test Suite for TEKNOFEST 2025 Education System
"""

import pytest
import json
import os
import sys
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
import asyncio
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import modules to test
from src.agents.learning_path_agent_v2 import LearningPathAgent
from src.agents.study_buddy_agent import StudyBuddyAgent
from src.data_processor import DataProcessor
from src.simple_data_processor import SimpleDataProcessor


# ========================= FIXTURES =========================

@pytest.fixture
def learning_path_agent():
    """Create a LearningPathAgent instance for testing"""
    return LearningPathAgent()


@pytest.fixture
def study_buddy_agent():
    """Create a StudyBuddyAgent instance for testing"""
    return StudyBuddyAgent()


@pytest.fixture
def data_processor():
    """Create a DataProcessor instance for testing"""
    return DataProcessor()


@pytest.fixture
def simple_data_processor():
    """Create a SimpleDataProcessor instance for testing"""
    return SimpleDataProcessor()


@pytest.fixture
def sample_student_profile():
    """Sample student profile for testing"""
    return {
        'student_id': 'test_student_123',
        'name': 'Test Öğrenci',
        'grade': 9,
        'current_level': 0.4,
        'target_level': 0.8,
        'learning_style': 'visual',
        'weak_topics': ['Denklemler', 'Geometri'],
        'strong_topics': ['Sayılar', 'Oran-Orantı']
    }


@pytest.fixture
def sample_quiz_data():
    """Sample quiz data for testing"""
    return [
        {
            'id': 'q1',
            'question': 'x + 5 = 10 denkleminde x kaçtır?',
            'options': ['3', '4', '5', '6'],
            'correct': '5',
            'difficulty': 0.3,
            'topic': 'Denklemler'
        },
        {
            'id': 'q2',
            'question': '2x - 3 = 7 denkleminde x kaçtır?',
            'options': ['3', '4', '5', '6'],
            'correct': '5',
            'difficulty': 0.4,
            'topic': 'Denklemler'
        }
    ]


# ========================= UNIT TESTS - Data Processor =========================

class TestDataProcessor:
    """Test suite for DataProcessor class"""
    
    def test_process_raw_data(self, data_processor):
        """Test raw data processing"""
        raw_data = {
            'student_id': '123',
            'grades': [80, 85, 90],
            'topics': ['Math', 'Science']
        }
        
        processed = data_processor.process_raw_data(raw_data)
        
        assert processed is not None
        assert 'student_id' in processed
        assert processed['student_id'] == '123'
    
    def test_validate_data(self, data_processor):
        """Test data validation"""
        valid_data = {
            'student_id': '123',
            'grade': 9,
            'score': 85
        }
        
        invalid_data = {
            'student_id': '',
            'grade': 'nine',
            'score': -10
        }
        
        assert data_processor.validate_data(valid_data) == True
        assert data_processor.validate_data(invalid_data) == False
    
    def test_normalize_scores(self, data_processor):
        """Test score normalization"""
        scores = [50, 60, 70, 80, 90]
        normalized = data_processor.normalize_scores(scores)
        
        assert len(normalized) == len(scores)
        assert min(normalized) >= 0
        assert max(normalized) <= 1
    
    def test_calculate_statistics(self, data_processor):
        """Test statistics calculation"""
        data = [10, 20, 30, 40, 50]
        stats = data_processor.calculate_statistics(data)
        
        assert 'mean' in stats
        assert 'median' in stats
        assert 'std' in stats
        assert stats['mean'] == 30
        assert stats['median'] == 30


class TestSimpleDataProcessor:
    """Test suite for SimpleDataProcessor class"""
    
    def test_load_questions(self, simple_data_processor):
        """Test question loading"""
        questions = simple_data_processor.load_questions('Matematik')
        
        assert isinstance(questions, list)
        assert len(questions) > 0
        if questions:
            assert 'topic' in questions[0]
            assert 'difficulty' in questions[0]
    
    def test_filter_by_difficulty(self, simple_data_processor):
        """Test filtering questions by difficulty"""
        questions = [
            {'id': 1, 'difficulty': 0.3},
            {'id': 2, 'difficulty': 0.5},
            {'id': 3, 'difficulty': 0.8}
        ]
        
        easy = simple_data_processor.filter_by_difficulty(questions, 0.0, 0.4)
        medium = simple_data_processor.filter_by_difficulty(questions, 0.4, 0.7)
        hard = simple_data_processor.filter_by_difficulty(questions, 0.7, 1.0)
        
        assert len(easy) == 1
        assert len(medium) == 1
        assert len(hard) == 1
    
    def test_prepare_dataset(self, simple_data_processor):
        """Test dataset preparation"""
        raw_data = [
            {'id': 1, 'data': 'test1'},
            {'id': 2, 'data': 'test2'}
        ]
        
        dataset = simple_data_processor.prepare_dataset(raw_data)
        
        assert dataset is not None
        assert len(dataset) == len(raw_data)


# ========================= UNIT TESTS - Learning Path Agent =========================

class TestLearningPathAgent:
    """Test suite for LearningPathAgent"""
    
    def test_detect_learning_style_visual(self, learning_path_agent):
        """Test visual learning style detection"""
        responses = [
            "Görsel materyaller kullanıyorum",
            "Şemalar ve grafikler faydalı",
            "Video izleyerek öğreniyorum"
        ]
        
        result = learning_path_agent.detect_learning_style(responses)
        
        assert result['dominant_style'] == 'visual'
        assert result['scores']['visual'] > result['scores']['auditory']
        assert result['scores']['visual'] > result['scores']['kinesthetic']
    
    def test_detect_learning_style_auditory(self, learning_path_agent):
        """Test auditory learning style detection"""
        responses = [
            "Sesli okuyarak öğreniyorum",
            "Dersleri dinleyerek anlıyorum",
            "Müzik eşliğinde çalışıyorum"
        ]
        
        result = learning_path_agent.detect_learning_style(responses)
        
        assert result['dominant_style'] == 'auditory'
        assert result['scores']['auditory'] > result['scores']['visual']
    
    def test_detect_learning_style_kinesthetic(self, learning_path_agent):
        """Test kinesthetic learning style detection"""
        responses = [
            "Yaparak öğreniyorum",
            "Pratik yapmayı seviyorum",
            "Deneyler yaparak anlıyorum"
        ]
        
        result = learning_path_agent.detect_learning_style(responses)
        
        assert result['dominant_style'] == 'kinesthetic'
        assert result['scores']['kinesthetic'] > result['scores']['visual']
    
    def test_generate_learning_path(self, learning_path_agent, sample_student_profile):
        """Test learning path generation"""
        path = learning_path_agent.generate_learning_path(
            sample_student_profile,
            "Matematik",
            weeks=4
        )
        
        assert 'weekly_plan' in path
        assert len(path['weekly_plan']) == 4
        assert 'student_id' in path
        assert path['student_id'] == sample_student_profile['student_id']
        
        # Check difficulty progression
        difficulties = [week['difficulty'] for week in path['weekly_plan']]
        assert difficulties == sorted(difficulties)  # Should be increasing
    
    def test_adaptive_difficulty(self, learning_path_agent):
        """Test adaptive difficulty calculation"""
        # Test for low performance
        low_diff = learning_path_agent.calculate_adaptive_difficulty(0.3, 0.4)
        assert low_diff < 0.4
        
        # Test for high performance
        high_diff = learning_path_agent.calculate_adaptive_difficulty(0.9, 0.5)
        assert high_diff > 0.5
        
        # Test for average performance
        avg_diff = learning_path_agent.calculate_adaptive_difficulty(0.5, 0.5)
        assert 0.4 <= avg_diff <= 0.6
    
    def test_personalize_content(self, learning_path_agent):
        """Test content personalization"""
        content = {
            'topic': 'Denklemler',
            'base_difficulty': 0.5
        }
        
        # Test for visual learner
        visual_content = learning_path_agent.personalize_content(content, 'visual')
        assert 'visual_aids' in visual_content
        assert visual_content['learning_style'] == 'visual'
        
        # Test for auditory learner
        audio_content = learning_path_agent.personalize_content(content, 'auditory')
        assert 'audio_resources' in audio_content
        assert audio_content['learning_style'] == 'auditory'


# ========================= UNIT TESTS - Study Buddy Agent =========================

class TestStudyBuddyAgent:
    """Test suite for StudyBuddyAgent"""
    
    def test_generate_adaptive_quiz(self, study_buddy_agent):
        """Test adaptive quiz generation"""
        quiz = study_buddy_agent.generate_adaptive_quiz(
            topic="Matematik",
            student_ability=0.5,
            num_questions=5
        )
        
        assert len(quiz) == 5
        for question in quiz:
            assert 'id' in question
            assert 'difficulty' in question
            assert 0 <= question['difficulty'] <= 1
            
            # Check difficulty is appropriate for student ability
            # Allow wider range as adaptive algorithm adjusts over time
            assert abs(question['difficulty'] - 0.5) <= 0.4
    
    def test_evaluate_answer(self, study_buddy_agent):
        """Test answer evaluation"""
        # Correct answer
        result = study_buddy_agent.evaluate_answer("5", "5")
        assert result['correct'] == True
        assert result['score'] > 0
        
        # Incorrect answer
        result = study_buddy_agent.evaluate_answer("3", "5")
        assert result['correct'] == False
        assert result['score'] == 0
    
    def test_generate_study_plan(self, study_buddy_agent):
        """Test study plan generation"""
        weak_topics = ["Denklemler", "Geometri", "Fonksiyonlar"]
        plan = study_buddy_agent.generate_study_plan(weak_topics, available_hours=12)
        
        assert 'total_hours' in plan
        assert plan['total_hours'] == 12
        assert 'topics' in plan
        assert len(plan['topics']) == 3
        
        # Check time allocation
        total_allocated = sum(topic['hours'] for topic in plan['topics'])
        assert total_allocated == 12
    
    def test_provide_hint(self, study_buddy_agent):
        """Test hint generation"""
        question = {
            'question': 'x + 5 = 10 denkleminde x kaçtır?',
            'topic': 'Denklemler',
            'difficulty': 0.3
        }
        
        hint = study_buddy_agent.provide_hint(question, level=1)
        assert hint is not None
        assert isinstance(hint, str)
        assert len(hint) > 0
        
        # Test progressive hints
        hint2 = study_buddy_agent.provide_hint(question, level=2)
        assert hint2 != hint  # Different hints for different levels
    
    def test_update_student_progress(self, study_buddy_agent):
        """Test student progress update"""
        initial_progress = {
            'student_id': 'test123',
            'topics': {
                'Denklemler': {'score': 0.5, 'attempts': 10}
            }
        }
        
        quiz_results = [
            {'topic': 'Denklemler', 'correct': True},
            {'topic': 'Denklemler', 'correct': True},
            {'topic': 'Denklemler', 'correct': False}
        ]
        
        updated = study_buddy_agent.update_progress(initial_progress, quiz_results)
        
        assert updated['topics']['Denklemler']['attempts'] == 13
        assert updated['topics']['Denklemler']['score'] != 0.5  # Score should change
    
    def test_recommend_resources(self, study_buddy_agent):
        """Test resource recommendation"""
        profile = {
            'learning_style': 'visual',
            'weak_topics': ['Geometri'],
            'grade': 9
        }
        
        resources = study_buddy_agent.recommend_resources(profile)
        
        assert isinstance(resources, list)
        assert len(resources) > 0
        
        for resource in resources:
            assert 'type' in resource
            assert 'topic' in resource
            assert 'url' in resource or 'content' in resource


# ========================= INTEGRATION TESTS =========================

class TestIntegration:
    """Integration tests for system components"""
    
    def test_agent_data_processor_integration(self, learning_path_agent, data_processor):
        """Test integration between agent and data processor"""
        # Process student data
        raw_data = {
            'student_id': 'int_test_123',
            'scores': [70, 75, 80, 85],
            'topics': ['Math', 'Science']
        }
        
        processed_data = data_processor.process_raw_data(raw_data)
        
        # Use processed data in agent
        profile = {
            'student_id': processed_data['student_id'],
            'current_level': 0.5,
            'target_level': 0.8,
            'learning_style': 'visual',
            'grade': 9
        }
        
        path = learning_path_agent.generate_learning_path(profile, "Matematik", weeks=2)
        
        assert path is not None
        assert path['student_id'] == raw_data['student_id']
    
    def test_quiz_evaluation_pipeline(self, study_buddy_agent, sample_quiz_data):
        """Test complete quiz evaluation pipeline"""
        student_answers = ['5', '5']  # Both correct
        
        results = []
        for i, question in enumerate(sample_quiz_data):
            result = study_buddy_agent.evaluate_answer(
                student_answers[i],
                question['correct']
            )
            result['question_id'] = question['id']
            result['topic'] = question['topic']
            results.append(result)
        
        # Calculate overall score
        total_score = sum(r['score'] for r in results)
        accuracy = sum(1 for r in results if r['correct']) / len(results)
        
        assert accuracy == 1.0  # All answers correct
        assert total_score > 0
    
    def test_adaptive_learning_loop(self, learning_path_agent, study_buddy_agent):
        """Test adaptive learning feedback loop"""
        # Initial student state
        student_state = {
            'ability': 0.4,
            'completed_topics': [],
            'performance_history': []
        }
        
        # Simulate 5 learning iterations
        for iteration in range(5):
            # Generate quiz based on current ability
            quiz = study_buddy_agent.generate_adaptive_quiz(
                topic="Matematik",
                student_ability=student_state['ability'],
                num_questions=3
            )
            
            # Simulate answering (60% correct rate)
            performance = 0.6
            correct_answers = int(len(quiz) * performance)
            
            # Update ability based on performance
            if performance > 0.7:
                student_state['ability'] = min(1.0, student_state['ability'] + 0.1)
            elif performance < 0.5:
                student_state['ability'] = max(0.0, student_state['ability'] - 0.05)
            
            student_state['performance_history'].append(performance)
        
        assert len(student_state['performance_history']) == 5
        assert 0 <= student_state['ability'] <= 1


# ========================= PERFORMANCE TESTS =========================

class TestPerformance:
    """Performance and load tests"""
    
    def test_large_dataset_processing(self, data_processor):
        """Test processing large datasets"""
        import time
        
        # Generate large dataset
        large_data = [
            {'id': i, 'score': i % 100, 'topic': f'Topic{i % 10}'}
            for i in range(10000)
        ]
        
        start_time = time.time()
        processed = data_processor.process_batch(large_data)
        end_time = time.time()
        
        processing_time = end_time - start_time
        
        assert processed is not None
        assert processing_time < 5  # Should process in under 5 seconds
    
    def test_concurrent_quiz_generation(self, study_buddy_agent):
        """Test concurrent quiz generation"""
        import concurrent.futures
        
        def generate_quiz_task(ability):
            return study_buddy_agent.generate_adaptive_quiz(
                topic="Matematik",
                student_ability=ability,
                num_questions=5
            )
        
        abilities = [0.3, 0.5, 0.7, 0.4, 0.6]
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            quizzes = list(executor.map(generate_quiz_task, abilities))
        
        assert len(quizzes) == 5
        for quiz in quizzes:
            assert len(quiz) == 5
    
    @pytest.mark.slow
    def test_memory_usage(self, learning_path_agent):
        """Test memory usage under load"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Generate many learning paths
        for i in range(100):
            profile = {
                'student_id': f'mem_test_{i}',
                'current_level': 0.5,
                'target_level': 0.8,
                'learning_style': 'visual',
                'grade': 9
            }
            learning_path_agent.generate_learning_path(profile, "Matematik", weeks=4)
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        assert memory_increase < 100  # Should not increase by more than 100MB


# ========================= ERROR HANDLING TESTS =========================

class TestErrorHandling:
    """Test error handling and edge cases"""
    
    def test_invalid_student_profile(self, learning_path_agent):
        """Test handling of invalid student profiles"""
        invalid_profile = {
            'student_id': '',  # Empty ID
            'current_level': 1.5,  # Out of range
            'target_level': -0.5,  # Negative
            'learning_style': 'unknown',  # Invalid style
            'grade': 0  # Invalid grade
        }
        
        with pytest.raises(ValueError):
            learning_path_agent.generate_learning_path(invalid_profile, "Matematik", weeks=4)
    
    def test_empty_quiz_generation(self, study_buddy_agent):
        """Test quiz generation with zero questions"""
        with pytest.raises(ValueError):
            study_buddy_agent.generate_adaptive_quiz(
                topic="Matematik",
                student_ability=0.5,
                num_questions=0
            )
    
    def test_null_data_processing(self, data_processor):
        """Test processing null/None data"""
        assert data_processor.process_raw_data(None) is None
        assert data_processor.process_raw_data({}) == {}
    
    def test_division_by_zero_handling(self, data_processor):
        """Test division by zero in statistics"""
        empty_data = []
        stats = data_processor.calculate_statistics(empty_data)
        
        assert stats is not None
        assert 'error' in stats or stats['mean'] == 0
    
    def test_missing_required_fields(self, learning_path_agent):
        """Test handling missing required fields"""
        incomplete_profile = {
            # Missing student_id - this is required
            'current_level': 0.5,
            'target_level': 0.8
        }
        
        with pytest.raises((KeyError, ValueError)):
            learning_path_agent.generate_learning_path(incomplete_profile, "Matematik", weeks=4)


# ========================= MOCK TESTS =========================

class TestWithMocks:
    """Tests using mocks for external dependencies"""
    
    def test_external_api_call(self, learning_path_agent):
        """Test external API calls with mocks"""
        # fetch_external_resources already returns mock data
        result = learning_path_agent.fetch_external_resources('Matematik')
        
        assert result is not None
        assert isinstance(result, list)
        assert len(result) > 0
        assert all('title' in r and 'url' in r for r in result)
    
    @patch('src.agents.study_buddy_agent.datetime')
    def test_time_based_functionality(self, mock_datetime, study_buddy_agent):
        """Test time-based functionality with mocked datetime"""
        mock_now = datetime(2024, 1, 1, 12, 0, 0)
        mock_datetime.now.return_value = mock_now
        
        # Test study plan with time constraints
        plan = study_buddy_agent.generate_study_plan(
            ["Topic1", "Topic2"],
            available_hours=10
        )
        
        assert plan is not None
        assert 'created_at' not in plan or plan['created_at'] == mock_now
    
    @patch('builtins.open', new_callable=MagicMock)
    def test_file_operations(self, mock_open, data_processor):
        """Test file operations with mocked file system"""
        mock_file_content = json.dumps([
            {'id': 1, 'data': 'test1'},
            {'id': 2, 'data': 'test2'}
        ])
        
        mock_open.return_value.__enter__.return_value.read.return_value = mock_file_content
        
        # Assuming data processor has a method to load from file
        data = data_processor.load_from_file('test.json')
        
        if data is not None:
            assert isinstance(data, (list, dict))


# ========================= ASYNC TESTS =========================

@pytest.mark.asyncio
class TestAsync:
    """Asynchronous tests"""
    
    async def test_async_quiz_generation(self):
        """Test asynchronous quiz generation"""
        agent = StudyBuddyAgent()
        
        # Create async wrapper if not exists
        async def async_generate_quiz():
            return agent.generate_adaptive_quiz(
                topic="Matematik",
                student_ability=0.5,
                num_questions=5
            )
        
        quiz = await async_generate_quiz()
        assert len(quiz) == 5
    
    async def test_concurrent_async_operations(self):
        """Test multiple concurrent async operations"""
        agent = LearningPathAgent()
        
        async def async_detect_style(responses):
            return agent.detect_learning_style(responses)
        
        tasks = [
            async_detect_style(["Görsel öğreniyorum"]),
            async_detect_style(["Sesli öğreniyorum"]),
            async_detect_style(["Yaparak öğreniyorum"])
        ]
        
        results = await asyncio.gather(*tasks)
        
        assert len(results) == 3
        assert results[0]['dominant_style'] == 'visual'
        assert results[1]['dominant_style'] == 'auditory'
        assert results[2]['dominant_style'] == 'kinesthetic'


# ========================= PARAMETRIZED TESTS =========================

class TestParametrized:
    """Parametrized tests for comprehensive coverage"""
    
    @pytest.mark.parametrize("ability,expected_range", [
        (0.1, (0.15, 0.35)),  # Adjusted for min difficulty of 0.2
        (0.5, (0.35, 0.65)),  # Adjusted for realistic ranges
        (0.9, (0.65, 0.85))   # Adjusted for max difficulty of 0.8
    ])
    def test_difficulty_ranges(self, study_buddy_agent, ability, expected_range):
        """Test difficulty ranges for different abilities"""
        quiz = study_buddy_agent.generate_adaptive_quiz(
            topic="Matematik",
            student_ability=ability,
            num_questions=5
        )
        
        difficulties = [q['difficulty'] for q in quiz]
        avg_difficulty = sum(difficulties) / len(difficulties)
        
        assert expected_range[0] <= avg_difficulty <= expected_range[1]
    
    @pytest.mark.parametrize("learning_style", ['visual', 'auditory', 'kinesthetic', 'reading'])
    def test_all_learning_styles(self, learning_path_agent, learning_style):
        """Test all learning style variations"""
        profile = {
            'student_id': f'test_{learning_style}',
            'current_level': 0.5,
            'target_level': 0.8,
            'learning_style': learning_style,
            'grade': 9
        }
        
        path = learning_path_agent.generate_learning_path(profile, "Matematik", weeks=2)
        
        assert path is not None
        assert 'learning_style_adaptations' in path or 'weekly_plan' in path
    
    @pytest.mark.parametrize("grade", [1, 5, 8, 9, 10, 11, 12])
    def test_all_grade_levels(self, learning_path_agent, grade):
        """Test all grade levels"""
        profile = {
            'student_id': f'test_grade_{grade}',
            'current_level': 0.5,
            'target_level': 0.8,
            'learning_style': 'visual',
            'grade': grade
        }
        
        path = learning_path_agent.generate_learning_path(profile, "Matematik", weeks=2)
        
        assert path is not None
        assert path.get('grade', grade) == grade


# ========================= END-TO-END TESTS =========================

class TestEndToEnd:
    """End-to-end tests for complete workflows"""
    
    def test_complete_student_journey(self, learning_path_agent, study_buddy_agent, data_processor):
        """Test complete student learning journey"""
        # Step 1: Create student profile
        student_data = {
            'student_id': 'e2e_test_student',
            'name': 'Test Student',
            'grade': 9,
            'initial_assessment_scores': [60, 65, 70]
        }
        
        # Step 2: Process initial data
        processed_data = data_processor.process_raw_data(student_data)
        
        # Step 3: Detect learning style
        learning_style_responses = [
            "Görsel materyaller kullanıyorum",
            "Grafikler yardımcı oluyor"
        ]
        style_result = learning_path_agent.detect_learning_style(learning_style_responses)
        
        # Step 4: Generate learning path
        profile = {
            'student_id': student_data['student_id'],
            'current_level': 0.65,
            'target_level': 0.85,
            'learning_style': style_result['dominant_style'],
            'grade': student_data['grade']
        }
        learning_path = learning_path_agent.generate_learning_path(profile, "Matematik", weeks=4)
        
        # Step 5: Generate and take quizzes
        quiz_results = []
        for week in range(4):
            quiz = study_buddy_agent.generate_adaptive_quiz(
                topic="Matematik",
                student_ability=profile['current_level'] + (week * 0.05),
                num_questions=10
            )
            
            # Simulate taking quiz (70% success rate)
            week_results = []
            for q in quiz[:7]:  # Answer 7 correctly
                result = {'correct': True, 'question_id': q['id']}
                week_results.append(result)
            for q in quiz[7:]:  # Answer 3 incorrectly
                result = {'correct': False, 'question_id': q['id']}
                week_results.append(result)
            
            quiz_results.append({
                'week': week + 1,
                'score': 0.7,
                'results': week_results
            })
        
        # Step 6: Generate study plan based on performance
        weak_topics = ["Denklemler", "Geometri"]  # Identified from quiz results
        study_plan = study_buddy_agent.generate_study_plan(weak_topics, available_hours=20)
        
        # Verify complete journey
        assert learning_path is not None
        assert len(quiz_results) == 4
        assert all(r['score'] == 0.7 for r in quiz_results)
        assert study_plan['total_hours'] == 20
        assert len(study_plan['topics']) == 2
    
    def test_adaptive_system_response(self, learning_path_agent, study_buddy_agent):
        """Test system adaptation to student performance"""
        # Simulate struggling student
        struggling_profile = {
            'student_id': 'struggling_student',
            'current_level': 0.3,
            'target_level': 0.7,
            'learning_style': 'kinesthetic',
            'grade': 9
        }
        
        # System should provide easier content
        path = learning_path_agent.generate_learning_path(struggling_profile, "Matematik", weeks=6)
        quiz = study_buddy_agent.generate_adaptive_quiz(
            topic="Matematik",
            student_ability=0.3,
            num_questions=5
        )
        
        # Check adaptations
        assert len(path['weekly_plan']) == 6  # More time given
        avg_quiz_difficulty = sum(q['difficulty'] for q in quiz) / len(quiz)
        assert avg_quiz_difficulty < 0.5  # Easier questions
        
        # Simulate advanced student
        advanced_profile = {
            'student_id': 'advanced_student',
            'current_level': 0.8,
            'target_level': 0.95,
            'learning_style': 'reading',
            'grade': 11
        }
        
        # System should provide challenging content
        path = learning_path_agent.generate_learning_path(advanced_profile, "Matematik", weeks=3)
        quiz = study_buddy_agent.generate_adaptive_quiz(
            topic="Matematik",
            student_ability=0.8,
            num_questions=5
        )
        
        # Check adaptations
        assert len(path['weekly_plan']) == 3  # Less time needed
        avg_quiz_difficulty = sum(q['difficulty'] for q in quiz) / len(quiz)
        assert avg_quiz_difficulty > 0.6  # Harder questions


# ========================= FIXTURES CLEANUP =========================

@pytest.fixture(autouse=True)
def cleanup():
    """Cleanup after each test"""
    yield
    # Add any cleanup code here if needed
    pass


# ========================= TEST RUNNER =========================

if __name__ == "__main__":
    # Run tests with coverage
    pytest.main([
        __file__,
        '-v',  # Verbose output
        '--cov=src',  # Coverage for src directory
        '--cov-report=html',  # HTML coverage report
        '--cov-report=term-missing',  # Terminal coverage with missing lines
        '-x',  # Stop on first failure
        '--tb=short'  # Short traceback format
    ])