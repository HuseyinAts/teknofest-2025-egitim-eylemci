"""
Comprehensive tests for Study Buddy Agent
"""
import pytest
import json
from unittest.mock import Mock, patch, MagicMock
import sys
import os
from datetime import datetime, timedelta

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agents.study_buddy_agent import StudyBuddyAgent


class TestStudyBuddyAgent:
    
    @pytest.fixture
    def agent(self):
        """Create a StudyBuddyAgent instance for testing"""
        return StudyBuddyAgent()
    
    @pytest.fixture
    def sample_performance_data(self):
        """Sample performance data for testing"""
        return {
            'quiz_scores': [0.6, 0.65, 0.7, 0.72, 0.75],
            'topics_studied': ['Algebra', 'Geometry', 'Trigonometry'],
            'study_time_hours': 15,
            'last_study_date': datetime.now() - timedelta(days=2)
        }
    
    @pytest.mark.unit
    def test_agent_initialization(self, agent):
        """Test agent initialization"""
        assert agent is not None
        assert hasattr(agent, 'generate_adaptive_quiz')
        assert hasattr(agent, 'generate_study_plan')
        assert hasattr(agent, 'provide_feedback')
    
    @pytest.mark.unit
    def test_generate_adaptive_quiz_basic(self, agent):
        """Test basic adaptive quiz generation"""
        quiz = agent.generate_adaptive_quiz(
            topic="Matematik",
            student_ability=0.5,
            num_questions=5
        )
        
        assert len(quiz) == 5
        for question in quiz:
            assert 'id' in question
            assert 'difficulty' in question
            assert 'topic' in question
            assert 0 <= question['difficulty'] <= 1
    
    @pytest.mark.unit
    def test_adaptive_quiz_difficulty_matching(self, agent):
        """Test that quiz difficulty matches student ability"""
        low_ability = 0.3
        high_ability = 0.8
        
        low_quiz = agent.generate_adaptive_quiz("Math", low_ability, 3)
        high_quiz = agent.generate_adaptive_quiz("Math", high_ability, 3)
        
        low_avg_diff = sum(q['difficulty'] for q in low_quiz) / len(low_quiz)
        high_avg_diff = sum(q['difficulty'] for q in high_quiz) / len(high_quiz)
        
        # High ability student gets harder questions
        assert high_avg_diff > low_avg_diff
        
        # Questions should be near student ability level
        assert abs(low_avg_diff - low_ability) < 0.3
        assert abs(high_avg_diff - high_ability) < 0.3
    
    @pytest.mark.unit
    def test_generate_study_plan_basic(self, agent):
        """Test basic study plan generation"""
        weak_topics = ["Denklemler", "Fonksiyonlar", "Türev"]
        plan = agent.generate_study_plan(weak_topics, available_hours=12)
        
        assert 'total_hours' in plan
        assert plan['total_hours'] == 12
        assert 'topics' in plan
        assert len(plan['topics']) == len(weak_topics)
        
        # Check time allocation
        total_allocated = sum(topic.get('hours', 0) for topic in plan['topics'])
        assert total_allocated <= 12
    
    @pytest.mark.unit
    def test_study_plan_priority_allocation(self, agent):
        """Test that study plan prioritizes topics correctly"""
        weak_topics = ["Critical_Topic", "Important_Topic", "Basic_Topic"]
        plan = agent.generate_study_plan(weak_topics, available_hours=10)
        
        # First topics should get more time (assuming priority order)
        if len(plan['topics']) >= 2:
            assert plan['topics'][0].get('hours', 0) >= plan['topics'][-1].get('hours', 0)
    
    @pytest.mark.unit
    def test_provide_feedback_positive(self, agent):
        """Test positive feedback generation"""
        performance = {
            'score': 0.85,
            'improvement': 0.15,
            'streak': 5
        }
        
        feedback = agent.provide_feedback(performance)
        
        assert feedback is not None
        assert isinstance(feedback, dict) or isinstance(feedback, str)
        # Should contain positive indicators
        if isinstance(feedback, str):
            assert any(word in feedback.lower() for word in ['başarılı', 'tebrik', 'harika', 'mükemmel'])
    
    @pytest.mark.unit
    def test_provide_feedback_needs_improvement(self, agent):
        """Test feedback for poor performance"""
        performance = {
            'score': 0.35,
            'improvement': -0.1,
            'streak': 0
        }
        
        feedback = agent.provide_feedback(performance)
        
        assert feedback is not None
        # Should contain constructive feedback
        if isinstance(feedback, str):
            assert any(word in feedback.lower() for word in ['çalış', 'gelişim', 'odaklan', 'pratik'])
    
    @pytest.mark.unit
    @pytest.mark.parametrize("num_questions", [1, 5, 10, 20])
    def test_quiz_generation_various_sizes(self, agent, num_questions):
        """Test quiz generation with various sizes"""
        quiz = agent.generate_adaptive_quiz(
            topic="Test Topic",
            student_ability=0.5,
            num_questions=num_questions
        )
        
        assert len(quiz) == num_questions
    
    @pytest.mark.unit
    def test_study_plan_empty_topics(self, agent):
        """Test study plan with empty topics"""
        plan = agent.generate_study_plan([], available_hours=5)
        
        assert plan is not None
        assert 'total_hours' in plan
        assert plan['topics'] == [] or len(plan['topics']) == 0
    
    @pytest.mark.unit
    def test_study_plan_zero_hours(self, agent):
        """Test study plan with zero available hours"""
        weak_topics = ["Topic1", "Topic2"]
        plan = agent.generate_study_plan(weak_topics, available_hours=0)
        
        assert plan is not None
        assert plan['total_hours'] == 0
    
    @pytest.mark.unit
    def test_adaptive_quiz_edge_abilities(self, agent):
        """Test quiz generation with edge case abilities"""
        # Minimum ability
        min_quiz = agent.generate_adaptive_quiz("Math", 0.0, 3)
        # Maximum ability
        max_quiz = agent.generate_adaptive_quiz("Math", 1.0, 3)
        
        assert all(0 <= q['difficulty'] <= 1 for q in min_quiz)
        assert all(0 <= q['difficulty'] <= 1 for q in max_quiz)
    
    @pytest.mark.unit
    def test_generate_practice_problems(self, agent):
        """Test practice problem generation if method exists"""
        if hasattr(agent, 'generate_practice_problems'):
            problems = agent.generate_practice_problems(
                topic="Algebra",
                difficulty=0.5,
                count=5
            )
            
            assert len(problems) == 5
            for problem in problems:
                assert 'question' in problem or 'problem' in problem
                assert 'difficulty' in problem
    
    @pytest.mark.unit
    def test_analyze_weak_areas(self, agent):
        """Test weak area analysis if method exists"""
        if hasattr(agent, 'analyze_weak_areas'):
            quiz_results = [
                {'topic': 'Algebra', 'correct': False},
                {'topic': 'Algebra', 'correct': False},
                {'topic': 'Geometry', 'correct': True},
                {'topic': 'Geometry', 'correct': True},
                {'topic': 'Trigonometry', 'correct': False}
            ]
            
            weak_areas = agent.analyze_weak_areas(quiz_results)
            
            assert 'Algebra' in weak_areas
            assert 'Trigonometry' in weak_areas
            assert 'Geometry' not in weak_areas or weak_areas['Geometry'] < weak_areas['Algebra']
    
    @pytest.mark.unit
    def test_study_plan_with_deadlines(self, agent):
        """Test study plan with deadline consideration"""
        if hasattr(agent, 'generate_study_plan_with_deadline'):
            weak_topics = ["Topic1", "Topic2", "Topic3"]
            deadline = datetime.now() + timedelta(days=7)
            
            plan = agent.generate_study_plan_with_deadline(
                weak_topics,
                deadline=deadline,
                daily_hours=2
            )
            
            assert plan is not None
            assert 'schedule' in plan
            assert len(plan['schedule']) <= 7  # Should not exceed deadline


@pytest.mark.integration
class TestStudyBuddyAgentIntegration:
    
    @pytest.fixture
    def agent(self):
        return StudyBuddyAgent()
    
    def test_complete_study_session_flow(self, agent):
        """Test complete study session flow"""
        # Step 1: Generate initial quiz
        initial_quiz = agent.generate_adaptive_quiz("Matematik", 0.5, 5)
        
        # Step 2: Simulate quiz results
        quiz_results = [
            {'id': q['id'], 'correct': q['difficulty'] < 0.6}
            for q in initial_quiz
        ]
        
        # Step 3: Identify weak areas (if method exists)
        weak_topics = ["Algebra", "Geometry"]  # Simulated weak areas
        
        # Step 4: Generate study plan
        study_plan = agent.generate_study_plan(weak_topics, 10)
        
        # Step 5: Generate follow-up quiz
        followup_quiz = agent.generate_adaptive_quiz("Matematik", 0.55, 5)
        
        # Verify flow completeness
        assert len(initial_quiz) == 5
        assert len(study_plan['topics']) == len(weak_topics)
        assert len(followup_quiz) == 5
        
        # Follow-up quiz should be slightly harder (simulating improvement)
        initial_avg = sum(q['difficulty'] for q in initial_quiz) / len(initial_quiz)
        followup_avg = sum(q['difficulty'] for q in followup_quiz) / len(followup_quiz)
        assert abs(followup_avg - initial_avg) < 0.5  # Reasonable progression
    
    def test_adaptive_learning_progression(self, agent):
        """Test adaptive learning over multiple sessions"""
        ability_progression = [0.3, 0.4, 0.5, 0.6, 0.7]
        quizzes = []
        
        for ability in ability_progression:
            quiz = agent.generate_adaptive_quiz("Math", ability, 3)
            quizzes.append(quiz)
        
        # Verify progression in difficulty
        avg_difficulties = [
            sum(q['difficulty'] for q in quiz) / len(quiz)
            for quiz in quizzes
        ]
        
        # Difficulty should generally increase
        for i in range(1, len(avg_difficulties)):
            assert avg_difficulties[i] >= avg_difficulties[i-1] - 0.1  # Allow small variation
    
    def test_multi_topic_study_plan(self, agent):
        """Test study plan covering multiple subjects"""
        subjects = {
            'Matematik': ["Cebir", "Geometri"],
            'Fizik': ["Mekanik", "Elektrik"],
            'Kimya': ["Organik", "İnorganik"]
        }
        
        all_weak_topics = []
        for subject, topics in subjects.items():
            all_weak_topics.extend(topics)
        
        comprehensive_plan = agent.generate_study_plan(all_weak_topics, 30)
        
        assert len(comprehensive_plan['topics']) == len(all_weak_topics)
        assert comprehensive_plan['total_hours'] == 30
        
        # Verify time is distributed
        total_allocated = sum(
            topic.get('hours', 0) for topic in comprehensive_plan['topics']
        )
        assert total_allocated > 0
        assert total_allocated <= 30