"""
TEKNOFEST 2025 - Production Ready Agent Tests
Comprehensive test suite for all agent modules with 100% coverage target
"""

import pytest
import asyncio
import json
import os
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from unittest.mock import Mock, AsyncMock, patch, MagicMock, call
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import agents with error handling
try:
    from src.agents.learning_path_agent_v2 import LearningPathAgent
    from src.agents.study_buddy_agent_clean import StudyBuddyAgent
    from src.agents.optimized_learning_path_agent import OptimizedLearningPathAgent
except ImportError as e:
    print(f"Warning: Could not import agents: {e}")
    # Create mock classes for testing infrastructure
    class LearningPathAgent:
        pass
    class StudyBuddyAgent:
        pass
    class OptimizedLearningPathAgent:
        pass


@pytest.mark.agent
class TestLearningPathAgent:
    """Comprehensive tests for Learning Path Agent - Production Ready"""
    
    @pytest.fixture
    def mock_settings(self):
        """Mock settings for agent"""
        settings = Mock()
        settings.model_name = "test-model"
        settings.max_tokens = 512
        settings.temperature = 0.7
        settings.database_url = "sqlite:///:memory:"
        return settings
    
    @pytest.fixture
    def agent(self, mock_settings):
        """Create agent instance with mocked dependencies"""
        with patch('src.agents.learning_path_agent_v2.Settings', return_value=mock_settings):
            with patch('src.agents.learning_path_agent_v2.load_model'):
                agent = LearningPathAgent()
                agent.model = Mock()
                agent.model.generate = AsyncMock(return_value={"text": "Generated content"})
                return agent
    
    @pytest.fixture
    def sample_student_profile(self):
        """Comprehensive student profile for testing"""
        return {
            'student_id': 'STU001',
            'name': 'Ahmet Yılmaz',
            'grade': 10,
            'age': 16,
            'current_level': 0.45,
            'target_level': 0.85,
            'learning_style': 'visual',
            'learning_pace': 'moderate',
            'weak_topics': ['Matematik-İntegral', 'Fizik-Optik'],
            'strong_topics': ['Türkçe-Gramer', 'Tarih-Osmanlı'],
            'study_hours_per_day': 4,
            'exam_target': 'YKS',
            'exam_date': (datetime.now() + timedelta(days=180)).isoformat(),
            'preferences': {
                'morning_study': True,
                'break_frequency': 'hourly',
                'group_study': False
            }
        }
    
    @pytest.fixture
    def curriculum_data(self):
        """Mock curriculum data"""
        return {
            '10': {
                'Matematik': {
                    'topics': ['Fonksiyonlar', 'Polinomlar', 'Trigonometri'],
                    'difficulty': 0.6,
                    'hours_required': 120
                },
                'Fizik': {
                    'topics': ['Optik', 'Dalgalar', 'Elektrik'],
                    'difficulty': 0.7,
                    'hours_required': 100
                }
            }
        }
    
    # ==================== Initialization Tests ====================
    
    def test_agent_initialization(self, agent):
        """Test proper agent initialization"""
        assert agent is not None
        assert hasattr(agent, 'vark_quiz')
        assert hasattr(agent, 'curriculum')
        assert hasattr(agent, 'model')
    
    def test_agent_configuration(self, agent, mock_settings):
        """Test agent configuration loading"""
        assert agent.model is not None
        # Add more configuration checks based on actual implementation
    
    # ==================== Learning Style Detection Tests ====================
    
    @pytest.mark.parametrize("responses,expected_style", [
        (["görsel", "grafik", "video"], "visual"),
        (["dinleme", "sesli", "müzik"], "auditory"),
        (["yaparak", "pratik", "hareket"], "kinesthetic"),
        (["okuma", "yazma", "not"], "reading")
    ])
    def test_detect_learning_style(self, agent, responses, expected_style):
        """Test learning style detection with various inputs"""
        result = agent.detect_learning_style(responses)
        
        assert result is not None
        assert 'dominant_style' in result
        assert 'scores' in result
        assert 'percentages' in result
        assert 'recommendations' in result
        assert result['dominant_style'] == expected_style
        assert result['scores'][expected_style] > 0
        assert sum(result['percentages'].values()) == pytest.approx(100, rel=1e-2)
    
    def test_detect_learning_style_mixed(self, agent):
        """Test mixed learning style detection"""
        responses = [
            "Hem görsel hem işitsel materyalleri severim",
            "Grafik ve sesli anlatım",
            "Video ve podcast"
        ]
        
        result = agent.detect_learning_style(responses)
        
        assert result['scores']['visual'] > 0
        assert result['scores']['auditory'] > 0
        assert len(result['recommendations']) > 0
    
    def test_detect_learning_style_invalid_input(self, agent):
        """Test learning style detection with invalid input"""
        with pytest.raises(ValueError):
            agent.detect_learning_style([])
        
        with pytest.raises(TypeError):
            agent.detect_learning_style(None)
    
    # ==================== ZPD Calculation Tests ====================
    
    def test_calculate_zpd_level_normal(self, agent):
        """Test normal ZPD level calculation"""
        current = 0.3
        target = 0.8
        weeks = 12
        
        levels = agent.calculate_zpd_level(current, target, weeks)
        
        assert len(levels) == weeks
        assert levels[0] >= current
        assert levels[-1] <= target
        assert all(levels[i] <= levels[i+1] for i in range(len(levels)-1))
    
    def test_calculate_zpd_level_edge_cases(self, agent):
        """Test ZPD calculation edge cases"""
        # Already at target
        levels = agent.calculate_zpd_level(0.8, 0.8, 10)
        assert all(level == 0.8 for level in levels)
        
        # Single week
        levels = agent.calculate_zpd_level(0.3, 0.5, 1)
        assert len(levels) == 1
        
        # Large gap
        levels = agent.calculate_zpd_level(0.1, 0.95, 20)
        assert levels[0] >= 0.1
        assert levels[-1] <= 0.95
    
    def test_calculate_zpd_level_invalid_input(self, agent):
        """Test ZPD calculation with invalid inputs"""
        with pytest.raises(ValueError):
            agent.calculate_zpd_level(-0.1, 0.8, 10)
        
        with pytest.raises(ValueError):
            agent.calculate_zpd_level(0.3, 1.5, 10)
        
        with pytest.raises(ValueError):
            agent.calculate_zpd_level(0.8, 0.3, 10)  # Target < current
        
        with pytest.raises(ValueError):
            agent.calculate_zpd_level(0.3, 0.8, 0)  # Zero weeks
    
    # ==================== Curriculum Integration Tests ====================
    
    def test_get_curriculum_topics(self, agent, curriculum_data):
        """Test curriculum topic retrieval"""
        agent.curriculum = curriculum_data
        
        topics = agent.get_curriculum_topics(10, 'Matematik')
        assert topics == ['Fonksiyonlar', 'Polinomlar', 'Trigonometri']
        
        topics = agent.get_curriculum_topics(10, 'Fizik')
        assert topics == ['Optik', 'Dalgalar', 'Elektrik']
    
    def test_get_curriculum_topics_missing(self, agent):
        """Test curriculum retrieval with missing data"""
        agent.curriculum = {}
        
        topics = agent.get_curriculum_topics(10, 'Matematik')
        assert topics == []
        
        topics = agent.get_curriculum_topics(15, 'Kimya')  # Invalid grade
        assert topics == []
    
    # ==================== Learning Path Creation Tests ====================
    
    @pytest.mark.asyncio
    async def test_create_learning_path_complete(self, agent, sample_student_profile):
        """Test complete learning path creation"""
        path = await agent.create_learning_path(sample_student_profile)
        
        assert path is not None
        assert 'path_id' in path
        assert 'student_id' in path
        assert 'total_weeks' in path
        assert 'weekly_plans' in path
        assert 'milestones' in path
        assert 'assessment_schedule' in path
        
        assert path['student_id'] == sample_student_profile['student_id']
        assert len(path['weekly_plans']) > 0
        assert all('topics' in week for week in path['weekly_plans'])
    
    @pytest.mark.asyncio
    async def test_create_learning_path_with_preferences(self, agent, sample_student_profile):
        """Test learning path creation with student preferences"""
        path = await agent.create_learning_path(sample_student_profile)
        
        # Check if preferences are respected
        for week in path['weekly_plans']:
            if 'schedule' in week:
                assert week['schedule']['morning_sessions'] == sample_student_profile['preferences']['morning_study']
    
    @pytest.mark.asyncio
    async def test_create_learning_path_adaptive(self, agent, sample_student_profile):
        """Test adaptive learning path features"""
        # Create initial path
        path1 = await agent.create_learning_path(sample_student_profile)
        
        # Update profile with progress
        sample_student_profile['current_level'] = 0.6
        sample_student_profile['completed_topics'] = ['Matematik-Fonksiyonlar']
        
        # Create updated path
        path2 = await agent.create_learning_path(sample_student_profile)
        
        assert path2['path_id'] != path1['path_id']
        assert path2['total_weeks'] <= path1['total_weeks']
    
    # ==================== Progress Tracking Tests ====================
    
    @pytest.mark.asyncio
    async def test_update_progress(self, agent):
        """Test progress update functionality"""
        progress_data = {
            'student_id': 'STU001',
            'completed_topics': ['Fonksiyonlar', 'Polinomlar'],
            'quiz_scores': [0.8, 0.75, 0.85],
            'study_time': 240,  # minutes
            'date': datetime.now().isoformat()
        }
        
        result = await agent.update_progress(progress_data)
        
        assert result['status'] == 'success'
        assert 'new_level' in result
        assert 'recommendations' in result
        assert result['new_level'] > 0
    
    @pytest.mark.asyncio
    async def test_get_progress_report(self, agent):
        """Test progress report generation"""
        student_id = 'STU001'
        
        report = await agent.get_progress_report(student_id)
        
        assert report is not None
        assert 'overall_progress' in report
        assert 'subject_progress' in report
        assert 'strengths' in report
        assert 'areas_for_improvement' in report
        assert 'next_steps' in report
    
    # ==================== Recommendation Tests ====================
    
    @pytest.mark.asyncio
    async def test_get_recommendations(self, agent, sample_student_profile):
        """Test recommendation generation"""
        recommendations = await agent.get_recommendations(sample_student_profile)
        
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        
        for rec in recommendations:
            assert 'type' in rec
            assert 'content' in rec
            assert 'priority' in rec
            assert 'reason' in rec
    
    @pytest.mark.asyncio
    async def test_get_personalized_content(self, agent, sample_student_profile):
        """Test personalized content generation"""
        topic = 'Matematik-İntegral'
        
        content = await agent.get_personalized_content(
            sample_student_profile,
            topic
        )
        
        assert content is not None
        assert 'topic' in content
        assert 'learning_style_adapted' in content
        assert 'materials' in content
        assert 'exercises' in content
        
        # Check if content matches learning style
        if sample_student_profile['learning_style'] == 'visual':
            assert any('video' in m.lower() or 'görsel' in m.lower() 
                      for m in content['materials'])
    
    # ==================== Performance Tests ====================
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_create_path_performance(self, agent, sample_student_profile, performance_timer):
        """Test learning path creation performance"""
        performance_timer.start()
        
        path = await agent.create_learning_path(sample_student_profile)
        
        elapsed = performance_timer.stop()
        
        assert path is not None
        assert elapsed < 5.0  # Should complete within 5 seconds
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_bulk_operations_performance(self, agent, performance_timer):
        """Test bulk operations performance"""
        students = [
            {'student_id': f'STU{i:03d}', 'grade': 10, 'current_level': 0.4}
            for i in range(10)
        ]
        
        performance_timer.start()
        
        tasks = [agent.create_learning_path(s) for s in students]
        paths = await asyncio.gather(*tasks)
        
        elapsed = performance_timer.stop()
        
        assert len(paths) == 10
        assert elapsed < 30.0  # Should handle 10 students in 30 seconds
    
    # ==================== Error Handling Tests ====================
    
    @pytest.mark.asyncio
    async def test_handle_model_failure(self, agent, sample_student_profile):
        """Test handling of model failures"""
        agent.model.generate = AsyncMock(side_effect=Exception("Model error"))
        
        with pytest.raises(Exception) as exc_info:
            await agent.create_learning_path(sample_student_profile)
        
        assert "Model error" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_handle_invalid_student_data(self, agent):
        """Test handling of invalid student data"""
        invalid_profile = {'student_id': None}
        
        with pytest.raises(ValueError):
            await agent.create_learning_path(invalid_profile)
    
    @pytest.mark.asyncio
    async def test_retry_logic(self, agent, sample_student_profile):
        """Test retry logic for transient failures"""
        call_count = 0
        
        async def mock_generate(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception("Temporary error")
            return {"text": "Success"}
        
        agent.model.generate = mock_generate
        
        # Should retry and eventually succeed
        path = await agent.create_learning_path(sample_student_profile)
        assert path is not None
        assert call_count == 3


@pytest.mark.agent
class TestStudyBuddyAgent:
    """Comprehensive tests for Study Buddy Agent - Production Ready"""
    
    @pytest.fixture
    def agent(self):
        """Create Study Buddy agent instance"""
        with patch('src.agents.study_buddy_agent_clean.Settings'):
            with patch('src.agents.study_buddy_agent_clean.load_model'):
                agent = StudyBuddyAgent()
                agent.model = Mock()
                agent.model.generate = AsyncMock(return_value={
                    "text": "Generated answer",
                    "confidence": 0.95
                })
                return agent
    
    @pytest.fixture
    def sample_question(self):
        """Sample question for testing"""
        return {
            'question': 'İntegral nedir ve nasıl hesaplanır?',
            'subject': 'Matematik',
            'grade': 11,
            'context': 'Calculus konusu içinde',
            'student_id': 'STU001',
            'learning_style': 'visual'
        }
    
    @pytest.fixture
    def quiz_request(self):
        """Sample quiz generation request"""
        return {
            'topic': 'Fizik - Optik',
            'grade': 10,
            'difficulty': 0.6,
            'question_count': 10,
            'question_types': ['multiple_choice', 'true_false', 'short_answer'],
            'include_explanations': True,
            'language': 'tr',
            'time_limit': 30
        }
    
    # ==================== Question Answering Tests ====================
    
    @pytest.mark.asyncio
    async def test_answer_question_basic(self, agent, sample_question):
        """Test basic question answering"""
        response = await agent.answer_question(sample_question)
        
        assert response is not None
        assert 'answer' in response
        assert 'confidence' in response
        assert 'sources' in response
        assert 'follow_up_questions' in response
        
        assert len(response['answer']) > 0
        assert 0 <= response['confidence'] <= 1
    
    @pytest.mark.asyncio
    async def test_answer_question_with_context(self, agent, sample_question):
        """Test question answering with context"""
        sample_question['previous_qa'] = [
            {'q': 'Türev nedir?', 'a': 'Türev, bir fonksiyonun değişim hızıdır.'}
        ]
        
        response = await agent.answer_question(sample_question)
        
        assert response is not None
        assert 'context_used' in response
        assert response['context_used'] == True
    
    @pytest.mark.asyncio
    async def test_answer_question_learning_style_adaptation(self, agent, sample_question):
        """Test answer adaptation to learning style"""
        # Visual learner
        sample_question['learning_style'] = 'visual'
        response_visual = await agent.answer_question(sample_question)
        
        # Auditory learner
        sample_question['learning_style'] = 'auditory'
        response_auditory = await agent.answer_question(sample_question)
        
        # Answers should be different based on learning style
        assert response_visual['answer'] != response_auditory['answer']
    
    @pytest.mark.asyncio
    async def test_answer_question_with_latex(self, agent):
        """Test mathematical formula handling"""
        math_question = {
            'question': 'x^2 + 2x + 1 = 0 denklemini çöz',
            'subject': 'Matematik',
            'requires_latex': True
        }
        
        response = await agent.answer_question(math_question)
        
        assert response is not None
        assert 'latex_formulas' in response
        assert len(response['latex_formulas']) > 0
    
    # ==================== Quiz Generation Tests ====================
    
    @pytest.mark.asyncio
    async def test_generate_quiz_complete(self, agent, quiz_request):
        """Test complete quiz generation"""
        quiz = await agent.generate_quiz(quiz_request)
        
        assert quiz is not None
        assert 'quiz_id' in quiz
        assert 'questions' in quiz
        assert 'metadata' in quiz
        
        assert len(quiz['questions']) == quiz_request['question_count']
        
        for question in quiz['questions']:
            assert 'id' in question
            assert 'question' in question
            assert 'type' in question
            assert 'difficulty' in question
            
            if question['type'] == 'multiple_choice':
                assert 'options' in question
                assert 'correct_answer' in question
                assert len(question['options']) >= 2
    
    @pytest.mark.asyncio
    async def test_generate_quiz_different_types(self, agent, quiz_request):
        """Test quiz generation with different question types"""
        quiz = await agent.generate_quiz(quiz_request)
        
        question_types = set(q['type'] for q in quiz['questions'])
        
        # Should have variety of question types
        assert len(question_types) >= 2
        assert 'multiple_choice' in question_types
    
    @pytest.mark.asyncio
    async def test_generate_quiz_with_explanations(self, agent, quiz_request):
        """Test quiz generation with explanations"""
        quiz = await agent.generate_quiz(quiz_request)
        
        for question in quiz['questions']:
            if quiz_request['include_explanations']:
                assert 'explanation' in question
                assert len(question['explanation']) > 0
    
    @pytest.mark.asyncio
    async def test_generate_adaptive_quiz(self, agent):
        """Test adaptive quiz generation based on performance"""
        initial_request = {
            'topic': 'Matematik',
            'grade': 10,
            'difficulty': 0.5,
            'question_count': 5,
            'adaptive': True
        }
        
        quiz = await agent.generate_adaptive_quiz(initial_request)
        
        assert quiz is not None
        assert 'questions' in quiz
        assert 'difficulty_progression' in quiz
        
        # Difficulty should adapt
        difficulties = [q['difficulty'] for q in quiz['questions']]
        assert min(difficulties) < max(difficulties)
    
    # ==================== Explanation Generation Tests ====================
    
    @pytest.mark.asyncio
    async def test_explain_concept(self, agent):
        """Test concept explanation generation"""
        concept = {
            'name': 'Fotosintez',
            'subject': 'Biyoloji',
            'grade': 9,
            'detail_level': 'intermediate'
        }
        
        explanation = await agent.explain_concept(concept)
        
        assert explanation is not None
        assert 'explanation' in explanation
        assert 'examples' in explanation
        assert 'key_points' in explanation
        assert 'related_concepts' in explanation
    
    @pytest.mark.asyncio
    async def test_generate_summary(self, agent):
        """Test content summarization"""
        content = {
            'text': 'Uzun bir metin...' * 100,
            'max_length': 200,
            'style': 'bullet_points'
        }
        
        summary = await agent.generate_summary(content)
        
        assert summary is not None
        assert 'summary' in summary
        assert len(summary['summary']) <= content['max_length'] * 1.1  # Allow 10% margin
    
    # ==================== Interactive Features Tests ====================
    
    @pytest.mark.asyncio
    async def test_chat_session(self, agent):
        """Test chat session management"""
        session_id = await agent.start_chat_session('STU001')
        
        assert session_id is not None
        
        # Send messages
        response1 = await agent.send_message(session_id, "Merhaba")
        assert response1 is not None
        
        response2 = await agent.send_message(session_id, "İntegral hakkında bilgi ver")
        assert response2 is not None
        
        # Get chat history
        history = await agent.get_chat_history(session_id)
        assert len(history) >= 2
        
        # End session
        result = await agent.end_chat_session(session_id)
        assert result['status'] == 'ended'
    
    @pytest.mark.asyncio
    async def test_provide_hint(self, agent):
        """Test hint generation for problems"""
        problem = {
            'question': '2x + 5 = 15',
            'student_attempt': '2x = 15',
            'hint_level': 1  # Progressive hints
        }
        
        hint = await agent.provide_hint(problem)
        
        assert hint is not None
        assert 'hint' in hint
        assert 'next_step' in hint
        assert len(hint['hint']) > 0
    
    # ==================== Performance Tests ====================
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_answer_response_time(self, agent, sample_question, performance_timer):
        """Test question answering response time"""
        performance_timer.start()
        
        response = await agent.answer_question(sample_question)
        
        elapsed = performance_timer.stop()
        
        assert response is not None
        assert elapsed < 2.0  # Should respond within 2 seconds
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_quiz_generation_performance(self, agent, quiz_request, performance_timer):
        """Test quiz generation performance"""
        performance_timer.start()
        
        quiz = await agent.generate_quiz(quiz_request)
        
        elapsed = performance_timer.stop()
        
        assert quiz is not None
        assert elapsed < 10.0  # Should generate quiz within 10 seconds
    
    # ==================== Error Handling Tests ====================
    
    @pytest.mark.asyncio
    async def test_handle_empty_question(self, agent):
        """Test handling of empty questions"""
        with pytest.raises(ValueError):
            await agent.answer_question({'question': ''})
    
    @pytest.mark.asyncio
    async def test_handle_invalid_quiz_request(self, agent):
        """Test handling of invalid quiz requests"""
        invalid_request = {
            'topic': 'Math',
            'question_count': -5  # Invalid count
        }
        
        with pytest.raises(ValueError):
            await agent.generate_quiz(invalid_request)
    
    @pytest.mark.asyncio
    async def test_handle_model_timeout(self, agent, sample_question):
        """Test handling of model timeouts"""
        async def slow_generate(*args, **kwargs):
            await asyncio.sleep(10)
            return {"text": "Late response"}
        
        agent.model.generate = slow_generate
        
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(
                agent.answer_question(sample_question),
                timeout=1.0
            )


@pytest.mark.agent
class TestOptimizedLearningPathAgent:
    """Tests for Optimized Learning Path Agent"""
    
    @pytest.fixture
    def agent(self):
        """Create optimized agent instance"""
        with patch('src.agents.optimized_learning_path_agent.Settings'):
            agent = OptimizedLearningPathAgent()
            agent.model = Mock()
            return agent
    
    @pytest.mark.asyncio
    async def test_optimization_algorithm(self, agent):
        """Test path optimization algorithm"""
        constraints = {
            'max_hours_per_day': 6,
            'subjects': ['Math', 'Physics', 'Chemistry'],
            'priority_weights': {'Math': 0.5, 'Physics': 0.3, 'Chemistry': 0.2},
            'deadline': (datetime.now() + timedelta(days=90)).isoformat()
        }
        
        optimized_path = await agent.optimize_path(constraints)
        
        assert optimized_path is not None
        assert 'schedule' in optimized_path
        assert 'efficiency_score' in optimized_path
        assert optimized_path['efficiency_score'] > 0.7
    
    @pytest.mark.asyncio
    async def test_multi_objective_optimization(self, agent):
        """Test multi-objective optimization"""
        objectives = {
            'maximize_coverage': 0.4,
            'minimize_time': 0.3,
            'balance_subjects': 0.3
        }
        
        result = await agent.optimize_multi_objective(objectives)
        
        assert result is not None
        assert 'pareto_optimal' in result
        assert result['pareto_optimal'] == True


@pytest.mark.integration
@pytest.mark.agent
class TestAgentIntegration:
    """Integration tests for agent interactions"""
    
    @pytest.fixture
    def learning_agent(self):
        """Create learning path agent"""
        with patch('src.agents.learning_path_agent_v2.Settings'):
            return LearningPathAgent()
    
    @pytest.fixture
    def study_agent(self):
        """Create study buddy agent"""
        with patch('src.agents.study_buddy_agent_clean.Settings'):
            return StudyBuddyAgent()
    
    @pytest.mark.asyncio
    async def test_agent_collaboration(self, learning_agent, study_agent):
        """Test collaboration between agents"""
        # Learning agent creates path
        student_profile = {
            'student_id': 'STU001',
            'grade': 10,
            'current_level': 0.4
        }
        
        path = await learning_agent.create_learning_path(student_profile)
        
        # Study buddy uses path to generate quiz
        quiz_request = {
            'topic': path['weekly_plans'][0]['topics'][0],
            'difficulty': path['weekly_plans'][0]['difficulty'],
            'question_count': 5
        }
        
        quiz = await study_agent.generate_quiz(quiz_request)
        
        assert quiz is not None
        assert len(quiz['questions']) == 5


# ==================== Test Utilities ====================

def create_mock_student_batch(count: int) -> List[Dict]:
    """Create batch of mock students for testing"""
    return [
        {
            'student_id': f'STU{i:04d}',
            'name': f'Student {i}',
            'grade': 9 + (i % 4),
            'current_level': 0.3 + (i % 5) * 0.1
        }
        for i in range(count)
    ]


def assert_valid_learning_path(path: Dict) -> None:
    """Assert that a learning path has all required fields"""
    required_fields = [
        'path_id', 'student_id', 'created_at',
        'total_weeks', 'weekly_plans', 'milestones'
    ]
    
    for field in required_fields:
        assert field in path, f"Missing required field: {field}"
    
    assert isinstance(path['weekly_plans'], list)
    assert len(path['weekly_plans']) > 0


def assert_valid_quiz(quiz: Dict) -> None:
    """Assert that a quiz has all required fields"""
    required_fields = ['quiz_id', 'questions', 'metadata']
    
    for field in required_fields:
        assert field in quiz, f"Missing required field: {field}"
    
    assert isinstance(quiz['questions'], list)
    assert len(quiz['questions']) > 0
    
    for question in quiz['questions']:
        assert 'id' in question
        assert 'question' in question
        assert 'type' in question


# ==================== Benchmark Tests ====================

@pytest.mark.benchmark
@pytest.mark.agent
class TestAgentBenchmarks:
    """Performance benchmarks for agents"""
    
    @pytest.mark.asyncio
    async def test_benchmark_path_creation(self, benchmark):
        """Benchmark learning path creation"""
        agent = LearningPathAgent()
        profile = {
            'student_id': 'BENCH001',
            'grade': 10,
            'current_level': 0.5
        }
        
        result = await benchmark(agent.create_learning_path, profile)
        assert result is not None
    
    @pytest.mark.asyncio
    async def test_benchmark_quiz_generation(self, benchmark):
        """Benchmark quiz generation"""
        agent = StudyBuddyAgent()
        request = {
            'topic': 'Math',
            'question_count': 20,
            'difficulty': 0.6
        }
        
        result = await benchmark(agent.generate_quiz, request)
        assert result is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--cov=src.agents", "--cov-report=term-missing"])