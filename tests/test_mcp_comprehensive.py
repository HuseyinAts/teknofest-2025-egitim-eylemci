# -*- coding: utf-8 -*-
"""
Comprehensive MCP Server Test Suite for TEKNOFEST 2025
"""

import pytest
import json
import asyncio
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from datetime import datetime
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import MCP server modules
from src.mcp_server.server import MCPServer
from src.mcp_server.tools.assessment import AssessmentTool
from src.mcp_server.tools.curriculum import CurriculumTool


# ========================= FIXTURES =========================

@pytest.fixture
def mcp_server():
    """Create an MCP Server instance for testing"""
    return MCPServer()


@pytest.fixture
def assessment_tool():
    """Create an AssessmentTool instance for testing"""
    return AssessmentTool()


@pytest.fixture
def curriculum_tool():
    """Create a CurriculumTool instance for testing"""
    return CurriculumTool()


@pytest.fixture
def sample_assessment_request():
    """Sample assessment request"""
    return {
        'student_id': 'test_student_123',
        'subject': 'Matematik',
        'grade': 9,
        'assessment_type': 'diagnostic',
        'num_questions': 10
    }


@pytest.fixture
def sample_curriculum_request():
    """Sample curriculum request"""
    return {
        'student_id': 'test_student_123',
        'subject': 'Matematik',
        'grade': 9,
        'learning_goals': ['Denklemler', 'Geometri'],
        'duration_weeks': 4
    }


# ========================= MCP SERVER CORE TESTS =========================

class TestMCPServerCore:
    """Test suite for MCP Server core functionality"""
    
    @pytest.mark.asyncio
    async def test_server_initialization(self, mcp_server):
        """Test MCP server initialization"""
        assert mcp_server is not None
        assert hasattr(mcp_server, 'tools')
        assert hasattr(mcp_server, 'handle_request')
    
    @pytest.mark.asyncio
    async def test_tool_registration(self, mcp_server):
        """Test tool registration in MCP server"""
        # Register a mock tool
        mock_tool = Mock()
        mock_tool.name = 'test_tool'
        mock_tool.execute = AsyncMock(return_value={'success': True})
        
        mcp_server.register_tool(mock_tool)
        
        assert 'test_tool' in mcp_server.tools
        assert mcp_server.tools['test_tool'] == mock_tool
    
    @pytest.mark.asyncio
    async def test_request_handling(self, mcp_server):
        """Test request handling"""
        request = {
            'tool': 'assessment',
            'params': {
                'student_id': 'test123',
                'subject': 'Matematik'
            }
        }
        
        # Mock the assessment tool
        mock_tool = AsyncMock(return_value={'success': True, 'data': []})
        mcp_server.tools['assessment'] = mock_tool
        
        response = await mcp_server.handle_request(request)
        
        assert response is not None
        assert 'success' in response or 'error' not in response
    
    @pytest.mark.asyncio
    async def test_error_handling(self, mcp_server):
        """Test error handling in request processing"""
        request = {
            'tool': 'non_existent_tool',
            'params': {}
        }
        
        response = await mcp_server.handle_request(request)
        
        assert 'error' in response
        assert response['error'] == 'Tool not found' or 'not found' in response['error'].lower()
    
    @pytest.mark.asyncio
    async def test_concurrent_requests(self, mcp_server):
        """Test handling multiple concurrent requests"""
        # Mock assessment tool
        mock_tool = AsyncMock(return_value={'success': True, 'data': []})
        mcp_server.tools['assessment'] = mock_tool
        
        requests = [
            {'tool': 'assessment', 'params': {'student_id': f'student_{i}'}}
            for i in range(5)
        ]
        
        # Process requests concurrently
        responses = await asyncio.gather(*[
            mcp_server.handle_request(req) for req in requests
        ])
        
        assert len(responses) == 5
        for response in responses:
            assert response is not None


# ========================= ASSESSMENT TOOL TESTS =========================

class TestAssessmentTool:
    """Test suite for Assessment Tool"""
    
    @pytest.mark.asyncio
    async def test_create_diagnostic_assessment(self, assessment_tool):
        """Test diagnostic assessment creation"""
        params = {
            'student_id': 'test123',
            'subject': 'Matematik',
            'grade': 9,
            'num_questions': 10
        }
        
        assessment = await assessment_tool.create_diagnostic_assessment(params)
        
        assert assessment is not None
        assert 'questions' in assessment
        assert len(assessment['questions']) == 10
        assert assessment['type'] == 'diagnostic'
    
    @pytest.mark.asyncio
    async def test_create_formative_assessment(self, assessment_tool):
        """Test formative assessment creation"""
        params = {
            'student_id': 'test123',
            'topic': 'Denklemler',
            'difficulty': 0.5,
            'num_questions': 5
        }
        
        assessment = await assessment_tool.create_formative_assessment(params)
        
        assert assessment is not None
        assert 'questions' in assessment
        assert len(assessment['questions']) == 5
        assert assessment['type'] == 'formative'
    
    @pytest.mark.asyncio
    async def test_create_summative_assessment(self, assessment_tool):
        """Test summative assessment creation"""
        params = {
            'student_id': 'test123',
            'topics': ['Denklemler', 'Geometri'],
            'duration_minutes': 60,
            'num_questions': 20
        }
        
        assessment = await assessment_tool.create_summative_assessment(params)
        
        assert assessment is not None
        assert 'questions' in assessment
        assert len(assessment['questions']) == 20
        assert assessment['type'] == 'summative'
        assert assessment['duration'] == 60
    
    @pytest.mark.asyncio
    async def test_evaluate_responses(self, assessment_tool):
        """Test response evaluation"""
        responses = [
            {'question_id': 'q1', 'answer': 'A', 'correct_answer': 'A'},
            {'question_id': 'q2', 'answer': 'B', 'correct_answer': 'C'},
            {'question_id': 'q3', 'answer': 'D', 'correct_answer': 'D'}
        ]
        
        evaluation = await assessment_tool.evaluate_responses(responses)
        
        assert evaluation is not None
        assert 'score' in evaluation
        assert 'correct_count' in evaluation
        assert evaluation['correct_count'] == 2
        assert evaluation['total_questions'] == 3
        assert evaluation['percentage'] == pytest.approx(66.67, rel=1)
    
    @pytest.mark.asyncio
    async def test_generate_feedback(self, assessment_tool):
        """Test feedback generation"""
        evaluation_result = {
            'score': 7,
            'total': 10,
            'percentage': 70,
            'weak_topics': ['Geometri'],
            'strong_topics': ['Cebir']
        }
        
        feedback = await assessment_tool.generate_feedback(evaluation_result)
        
        assert feedback is not None
        assert 'overall_feedback' in feedback
        assert 'recommendations' in feedback
        assert 'weak_topics' in feedback
        assert 'strong_topics' in feedback
    
    @pytest.mark.asyncio
    async def test_adaptive_question_selection(self, assessment_tool):
        """Test adaptive question selection based on performance"""
        student_performance = {
            'current_ability': 0.6,
            'recent_correct_rate': 0.8
        }
        
        next_questions = await assessment_tool.select_adaptive_questions(
            student_performance,
            num_questions=3
        )
        
        assert len(next_questions) == 3
        # Questions should be slightly harder due to high correct rate
        avg_difficulty = sum(q['difficulty'] for q in next_questions) / 3
        assert avg_difficulty > 0.6


# ========================= CURRICULUM TOOL TESTS =========================

class TestCurriculumTool:
    """Test suite for Curriculum Tool"""
    
    @pytest.mark.asyncio
    async def test_generate_curriculum(self, curriculum_tool):
        """Test curriculum generation"""
        params = {
            'grade': 9,
            'subject': 'Matematik',
            'duration_weeks': 8,
            'learning_goals': ['Denklemler', 'Fonksiyonlar', 'Geometri']
        }
        
        curriculum = await curriculum_tool.generate_curriculum(params)
        
        assert curriculum is not None
        assert 'weeks' in curriculum
        assert len(curriculum['weeks']) == 8
        assert 'learning_goals' in curriculum
        assert len(curriculum['learning_goals']) == 3
    
    @pytest.mark.asyncio
    async def test_personalize_curriculum(self, curriculum_tool):
        """Test curriculum personalization"""
        base_curriculum = {
            'weeks': [{'week': 1, 'topics': ['Denklemler']}],
            'learning_goals': ['Denklemler']
        }
        
        student_profile = {
            'learning_style': 'visual',
            'pace': 'slow',
            'interests': ['bilim', 'teknoloji']
        }
        
        personalized = await curriculum_tool.personalize_curriculum(
            base_curriculum,
            student_profile
        )
        
        assert personalized is not None
        assert 'adaptations' in personalized
        assert personalized['learning_style'] == 'visual'
        assert 'visual_resources' in personalized['adaptations']
    
    @pytest.mark.asyncio
    async def test_curriculum_pacing(self, curriculum_tool):
        """Test curriculum pacing adjustments"""
        params = {
            'original_weeks': 4,
            'student_pace': 'slow',
            'topics': ['Topic1', 'Topic2', 'Topic3', 'Topic4']
        }
        
        adjusted = await curriculum_tool.adjust_pacing(params)
        
        assert adjusted is not None
        assert adjusted['recommended_weeks'] > 4  # Slower pace needs more time
        assert 'weekly_distribution' in adjusted
    
    @pytest.mark.asyncio
    async def test_prerequisite_checking(self, curriculum_tool):
        """Test prerequisite checking for curriculum topics"""
        topics = ['İntegral', 'Türev', 'Limit']
        student_knowledge = ['Cebir', 'Fonksiyonlar']
        
        check_result = await curriculum_tool.check_prerequisites(
            topics,
            student_knowledge
        )
        
        assert check_result is not None
        assert 'missing_prerequisites' in check_result
        assert 'ready_topics' in check_result
        assert 'Limit' in check_result['missing_prerequisites']
    
    @pytest.mark.asyncio
    async def test_resource_recommendation(self, curriculum_tool):
        """Test educational resource recommendations"""
        topic = 'Geometri'
        learning_style = 'visual'
        grade = 9
        
        resources = await curriculum_tool.recommend_resources(
            topic,
            learning_style,
            grade
        )
        
        assert resources is not None
        assert isinstance(resources, list)
        assert len(resources) > 0
        
        for resource in resources:
            assert 'type' in resource
            assert 'title' in resource
            assert 'url' in resource or 'content' in resource


# ========================= INTEGRATION TESTS =========================

class TestMCPIntegration:
    """Integration tests for MCP components"""
    
    @pytest.mark.asyncio
    async def test_assessment_curriculum_integration(self, assessment_tool, curriculum_tool):
        """Test integration between assessment and curriculum tools"""
        # First, create an assessment
        assessment_params = {
            'student_id': 'int_test_123',
            'subject': 'Matematik',
            'grade': 9,
            'num_questions': 10
        }
        
        assessment = await assessment_tool.create_diagnostic_assessment(assessment_params)
        
        # Evaluate responses (simulate)
        responses = [
            {'question_id': q['id'], 'answer': 'A', 'correct_answer': 'A' if i < 6 else 'B'}
            for i, q in enumerate(assessment['questions'])
        ]
        
        evaluation = await assessment_tool.evaluate_responses(responses)
        
        # Use evaluation to generate curriculum
        curriculum_params = {
            'grade': 9,
            'subject': 'Matematik',
            'duration_weeks': 8,
            'learning_goals': evaluation.get('weak_topics', ['Genel Matematik']),
            'student_level': evaluation['percentage'] / 100
        }
        
        curriculum = await curriculum_tool.generate_curriculum(curriculum_params)
        
        assert curriculum is not None
        assert curriculum['focus_areas'] == evaluation.get('weak_topics', ['Genel Matematik'])
    
    @pytest.mark.asyncio
    async def test_full_mcp_workflow(self, mcp_server, assessment_tool, curriculum_tool):
        """Test complete MCP workflow"""
        # Register tools
        mcp_server.register_tool(assessment_tool)
        mcp_server.register_tool(curriculum_tool)
        
        # Step 1: Initial assessment
        assessment_request = {
            'tool': 'assessment',
            'action': 'create_diagnostic',
            'params': {
                'student_id': 'workflow_test',
                'subject': 'Matematik',
                'grade': 9
            }
        }
        
        assessment_response = await mcp_server.handle_request(assessment_request)
        assert 'error' not in assessment_response
        
        # Step 2: Generate curriculum based on assessment
        curriculum_request = {
            'tool': 'curriculum',
            'action': 'generate',
            'params': {
                'student_id': 'workflow_test',
                'assessment_results': assessment_response.get('data', {}),
                'duration_weeks': 12
            }
        }
        
        curriculum_response = await mcp_server.handle_request(curriculum_request)
        assert 'error' not in curriculum_response
        
        # Step 3: Create formative assessments for first topic
        formative_request = {
            'tool': 'assessment',
            'action': 'create_formative',
            'params': {
                'student_id': 'workflow_test',
                'topic': curriculum_response.get('data', {}).get('first_topic', 'Matematik'),
                'num_questions': 5
            }
        }
        
        formative_response = await mcp_server.handle_request(formative_request)
        assert 'error' not in formative_response


# ========================= PERFORMANCE TESTS =========================

class TestMCPPerformance:
    """Performance tests for MCP Server"""
    
    @pytest.mark.asyncio
    async def test_concurrent_tool_execution(self, mcp_server):
        """Test concurrent execution of multiple tools"""
        import time
        
        # Create mock tools with delays
        async def slow_tool_execute(params):
            await asyncio.sleep(0.1)  # Simulate processing
            return {'success': True, 'tool': params.get('tool_id')}
        
        # Register multiple mock tools
        for i in range(5):
            mock_tool = Mock()
            mock_tool.name = f'tool_{i}'
            mock_tool.execute = slow_tool_execute
            mcp_server.register_tool(mock_tool)
        
        # Create concurrent requests
        requests = [
            {'tool': f'tool_{i}', 'params': {'tool_id': i}}
            for i in range(5)
        ]
        
        start_time = time.time()
        responses = await asyncio.gather(*[
            mcp_server.handle_request(req) for req in requests
        ])
        end_time = time.time()
        
        # Should complete in ~0.1 seconds (concurrent) not 0.5 seconds (sequential)
        assert (end_time - start_time) < 0.3
        assert len(responses) == 5
    
    @pytest.mark.asyncio
    async def test_large_assessment_generation(self, assessment_tool):
        """Test generation of large assessments"""
        import time
        
        params = {
            'student_id': 'perf_test',
            'subject': 'Matematik',
            'grade': 9,
            'num_questions': 100  # Large number of questions
        }
        
        start_time = time.time()
        assessment = await assessment_tool.create_diagnostic_assessment(params)
        end_time = time.time()
        
        assert len(assessment['questions']) == 100
        assert (end_time - start_time) < 2  # Should complete within 2 seconds
    
    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_memory_efficiency(self, mcp_server):
        """Test memory efficiency under load"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create many requests
        for i in range(100):
            request = {
                'tool': 'assessment',
                'params': {
                    'student_id': f'mem_test_{i}',
                    'num_questions': 10
                }
            }
            await mcp_server.handle_request(request)
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        assert memory_increase < 50  # Should not increase by more than 50MB


# ========================= ERROR HANDLING TESTS =========================

class TestMCPErrorHandling:
    """Test error handling in MCP components"""
    
    @pytest.mark.asyncio
    async def test_invalid_tool_request(self, mcp_server):
        """Test handling of invalid tool requests"""
        request = {
            'tool': 'invalid_tool',
            'params': {}
        }
        
        response = await mcp_server.handle_request(request)
        
        assert 'error' in response
        assert response['success'] == False
    
    @pytest.mark.asyncio
    async def test_missing_required_params(self, assessment_tool):
        """Test handling of missing required parameters"""
        params = {
            # Missing 'student_id' and 'subject'
            'grade': 9
        }
        
        with pytest.raises((KeyError, ValueError)):
            await assessment_tool.create_diagnostic_assessment(params)
    
    @pytest.mark.asyncio
    async def test_invalid_param_values(self, assessment_tool):
        """Test handling of invalid parameter values"""
        params = {
            'student_id': '',  # Empty ID
            'subject': 'InvalidSubject',
            'grade': -1,  # Invalid grade
            'num_questions': 0  # Invalid question count
        }
        
        with pytest.raises(ValueError):
            await assessment_tool.create_diagnostic_assessment(params)
    
    @pytest.mark.asyncio
    async def test_timeout_handling(self, mcp_server):
        """Test timeout handling for long-running operations"""
        async def timeout_tool_execute(params):
            await asyncio.sleep(10)  # Simulate very long operation
            return {'success': True}
        
        mock_tool = Mock()
        mock_tool.name = 'timeout_tool'
        mock_tool.execute = timeout_tool_execute
        mcp_server.register_tool(mock_tool)
        
        request = {'tool': 'timeout_tool', 'params': {}}
        
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(
                mcp_server.handle_request(request),
                timeout=1
            )
    
    @pytest.mark.asyncio
    async def test_exception_propagation(self, mcp_server):
        """Test proper exception propagation"""
        async def failing_tool_execute(params):
            raise RuntimeError("Tool execution failed")
        
        mock_tool = Mock()
        mock_tool.name = 'failing_tool'
        mock_tool.execute = failing_tool_execute
        mcp_server.register_tool(mock_tool)
        
        request = {'tool': 'failing_tool', 'params': {}}
        
        response = await mcp_server.handle_request(request)
        
        assert 'error' in response
        assert 'Tool execution failed' in response['error']


# ========================= MOCK TESTS =========================

class TestMCPWithMocks:
    """Tests using mocks for external dependencies"""
    
    @pytest.mark.asyncio
    @patch('src.mcp_server.tools.assessment.external_api')
    async def test_external_api_mock(self, mock_api, assessment_tool):
        """Test with mocked external API"""
        mock_api.fetch_questions = AsyncMock(return_value=[
            {'id': 'q1', 'text': 'Question 1'},
            {'id': 'q2', 'text': 'Question 2'}
        ])
        
        params = {
            'student_id': 'test123',
            'use_external': True,
            'num_questions': 2
        }
        
        assessment = await assessment_tool.create_with_external_questions(params)
        
        if assessment is not None:
            assert len(assessment.get('questions', [])) == 2
            mock_api.fetch_questions.assert_called_once()
    
    @pytest.mark.asyncio
    @patch('src.mcp_server.tools.curriculum.database')
    async def test_database_mock(self, mock_db, curriculum_tool):
        """Test with mocked database"""
        mock_db.get_student_history = AsyncMock(return_value={
            'completed_topics': ['Cebir', 'Geometri'],
            'performance': 0.75
        })
        
        student_id = 'test123'
        history = await curriculum_tool.get_student_history(student_id)
        
        if history is not None:
            assert 'completed_topics' in history
            assert len(history['completed_topics']) == 2
            mock_db.get_student_history.assert_called_with(student_id)
    
    @pytest.mark.asyncio
    @patch('src.mcp_server.server.logger')
    async def test_logging(self, mock_logger, mcp_server):
        """Test logging functionality"""
        request = {'tool': 'test', 'params': {}}
        
        await mcp_server.handle_request(request)
        
        # Check if appropriate logging calls were made
        assert mock_logger.info.called or mock_logger.error.called


# ========================= PARAMETRIZED TESTS =========================

class TestMCPParametrized:
    """Parametrized tests for MCP components"""
    
    @pytest.mark.asyncio
    @pytest.mark.parametrize("grade,expected_topics", [
        (1, ['Sayılar', 'Toplama', 'Çıkarma']),
        (5, ['Kesirler', 'Ondalık Sayılar', 'Geometri']),
        (9, ['Denklemler', 'Fonksiyonlar', 'Geometri']),
        (12, ['İntegral', 'Türev', 'Limit'])
    ])
    async def test_grade_appropriate_content(self, curriculum_tool, grade, expected_topics):
        """Test grade-appropriate content generation"""
        params = {
            'grade': grade,
            'subject': 'Matematik',
            'duration_weeks': 4
        }
        
        curriculum = await curriculum_tool.generate_curriculum(params)
        
        # Check if appropriate topics are included
        curriculum_topics = []
        for week in curriculum.get('weeks', []):
            curriculum_topics.extend(week.get('topics', []))
        
        # At least one expected topic should be present
        assert any(topic in curriculum_topics for topic in expected_topics)
    
    @pytest.mark.asyncio
    @pytest.mark.parametrize("num_questions", [5, 10, 20, 50])
    async def test_variable_assessment_sizes(self, assessment_tool, num_questions):
        """Test assessments with different sizes"""
        params = {
            'student_id': 'test123',
            'subject': 'Matematik',
            'grade': 9,
            'num_questions': num_questions
        }
        
        assessment = await assessment_tool.create_diagnostic_assessment(params)
        
        assert len(assessment['questions']) == num_questions
    
    @pytest.mark.asyncio
    @pytest.mark.parametrize("learning_style,expected_feature", [
        ('visual', 'visual_aids'),
        ('auditory', 'audio_resources'),
        ('kinesthetic', 'hands_on_activities'),
        ('reading', 'text_resources')
    ])
    async def test_learning_style_adaptations(self, curriculum_tool, learning_style, expected_feature):
        """Test curriculum adaptations for different learning styles"""
        base_curriculum = {
            'weeks': [{'week': 1, 'topics': ['Test Topic']}]
        }
        
        student_profile = {
            'learning_style': learning_style
        }
        
        personalized = await curriculum_tool.personalize_curriculum(
            base_curriculum,
            student_profile
        )
        
        assert expected_feature in personalized.get('adaptations', {})


# ========================= CLEANUP =========================

@pytest.fixture(autouse=True)
def cleanup():
    """Cleanup after each test"""
    yield
    # Add cleanup code if needed
    pass


# ========================= TEST RUNNER =========================

if __name__ == "__main__":
    pytest.main([
        __file__,
        '-v',
        '--cov=src.mcp_server',
        '--cov-report=html',
        '--cov-report=term-missing',
        '-x',
        '--tb=short'
    ])