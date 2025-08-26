"""
Comprehensive Test Suite for StudyBuddyAgent
Target: Complete coverage for AI study buddy functionality
"""

import pytest
from unittest.mock import Mock, MagicMock, patch, AsyncMock
from datetime import datetime, timedelta
import json
import numpy as np

from src.agents.study_buddy_agent_clean import (
    StudyBuddyAgent,
    QuizQuestion,
    StudySession,
    FeedbackType,
    StudyTip,
    PerformanceMetrics
)


class TestQuizGeneration:
    """Test quiz generation functionality"""
    
    @pytest.fixture
    def agent(self):
        """Create StudyBuddyAgent instance"""
        with patch('src.agents.study_buddy_agent_clean.DataProcessor'):
            with patch('src.agents.study_buddy_agent_clean.IRTService'):
                agent = StudyBuddyAgent()
                return agent
    
    def test_generate_adaptive_quiz_basic(self, agent):
        """Test generating basic adaptive quiz"""
        quiz = agent.generate_adaptive_quiz(
            topic="matematik",
            student_ability=0.5,
            num_questions=5
        )
        
        assert "questions" in quiz
        assert len(quiz["questions"]) == 5
        assert "quiz_id" in quiz
        assert all("difficulty" in q for q in quiz["questions"])
    
    def test_generate_adaptive_quiz_easy(self, agent):
        """Test generating easy quiz for low-ability student"""
        quiz = agent.generate_adaptive_quiz(
            topic="fen bilgisi",
            student_ability=0.2,
            num_questions=3
        )
        
        # Questions should be easier for low-ability student
        difficulties = [q["difficulty"] for q in quiz["questions"]]
        assert all(d <= 0.5 for d in difficulties)
    
    def test_generate_adaptive_quiz_hard(self, agent):
        """Test generating hard quiz for high-ability student"""
        quiz = agent.generate_adaptive_quiz(
            topic="fizik",
            student_ability=0.9,
            num_questions=3
        )
        
        # Questions should be harder for high-ability student
        difficulties = [q["difficulty"] for q in quiz["questions"]]
        assert any(d >= 0.7 for d in difficulties)
    
    def test_generate_quiz_with_time_limit(self, agent):
        """Test generating quiz with time limit"""
        quiz = agent.generate_adaptive_quiz(
            topic="kimya",
            student_ability=0.5,
            num_questions=10,
            time_limit=30
        )
        
        assert "time_limit" in quiz
        assert quiz["time_limit"] == 30
        assert "estimated_time" in quiz
    
    def test_generate_quiz_with_question_types(self, agent):
        """Test generating quiz with specific question types"""
        quiz = agent.generate_adaptive_quiz(
            topic="tarih",
            student_ability=0.6,
            num_questions=5,
            question_types=["multiple_choice", "true_false"]
        )
        
        question_types = [q["type"] for q in quiz["questions"]]
        assert all(qt in ["multiple_choice", "true_false"] for qt in question_types)
    
    def test_generate_quiz_empty_topic(self, agent):
        """Test quiz generation with empty topic"""
        with pytest.raises(ValueError):
            agent.generate_adaptive_quiz(
                topic="",
                student_ability=0.5,
                num_questions=5
            )
    
    def test_generate_quiz_invalid_ability(self, agent):
        """Test quiz generation with invalid ability level"""
        with pytest.raises(ValueError):
            agent.generate_adaptive_quiz(
                topic="matematik",
                student_ability=1.5,  # Invalid: > 1.0
                num_questions=5
            )
    
    def test_adaptive_difficulty_adjustment(self, agent):
        """Test adaptive difficulty adjustment during quiz"""
        # Simulate quiz progress
        responses = [
            {"correct": True, "time": 30},
            {"correct": True, "time": 25},
            {"correct": False, "time": 60}
        ]
        
        next_difficulty = agent.calculate_next_difficulty(
            current_ability=0.5,
            responses=responses
        )
        
        assert 0.0 <= next_difficulty <= 1.0
        # Should adjust based on performance
        assert next_difficulty != 0.5


class TestStudySession:
    """Test study session management"""
    
    @pytest.fixture
    def agent(self):
        """Create StudyBuddyAgent instance"""
        with patch('src.agents.study_buddy_agent_clean.DataProcessor'):
            with patch('src.agents.study_buddy_agent_clean.IRTService'):
                agent = StudyBuddyAgent()
                return agent
    
    def test_start_study_session(self, agent):
        """Test starting a new study session"""
        session = agent.start_study_session(
            student_id="student_123",
            subject="matematik",
            duration_minutes=45
        )
        
        assert session.student_id == "student_123"
        assert session.subject == "matematik"
        assert session.duration_minutes == 45
        assert session.is_active == True
        assert session.start_time is not None
    
    def test_end_study_session(self, agent):
        """Test ending a study session"""
        session = agent.start_study_session(
            student_id="student_123",
            subject="fizik",
            duration_minutes=30
        )
        
        # Simulate some activity
        session.questions_attempted = 10
        session.correct_answers = 7
        
        summary = agent.end_study_session(session.session_id)
        
        assert summary["completed"] == True
        assert summary["questions_attempted"] == 10
        assert summary["accuracy"] == 0.7
        assert "duration" in summary
    
    def test_pause_resume_session(self, agent):
        """Test pausing and resuming study session"""
        session = agent.start_study_session(
            student_id="student_123",
            subject="kimya",
            duration_minutes=60
        )
        
        # Pause session
        agent.pause_session(session.session_id)
        assert session.is_paused == True
        
        # Resume session
        agent.resume_session(session.session_id)
        assert session.is_paused == False
        assert session.is_active == True
    
    def test_track_session_progress(self, agent):
        """Test tracking progress during session"""
        session = agent.start_study_session(
            student_id="student_123",
            subject="biyoloji",
            duration_minutes=45
        )
        
        # Track question responses
        agent.track_response(session.session_id, correct=True, time_spent=30)
        agent.track_response(session.session_id, correct=False, time_spent=45)
        agent.track_response(session.session_id, correct=True, time_spent=20)
        
        progress = agent.get_session_progress(session.session_id)
        
        assert progress["questions_attempted"] == 3
        assert progress["correct_answers"] == 2
        assert progress["accuracy"] == 2/3
        assert progress["average_time"] == 31.67  # (30+45+20)/3
    
    def test_session_timeout(self, agent):
        """Test automatic session timeout"""
        session = agent.start_study_session(
            student_id="student_123",
            subject="coğrafya",
            duration_minutes=30
        )
        
        # Simulate timeout
        session.start_time = datetime.now() - timedelta(minutes=35)
        
        status = agent.check_session_status(session.session_id)
        assert status["timed_out"] == True
        assert status["is_active"] == False


class TestFeedbackGeneration:
    """Test feedback generation functionality"""
    
    @pytest.fixture
    def agent(self):
        """Create StudyBuddyAgent instance"""
        with patch('src.agents.study_buddy_agent_clean.DataProcessor'):
            with patch('src.agents.study_buddy_agent_clean.IRTService'):
                with patch('src.agents.study_buddy_agent_clean.ModelIntegration'):
                    agent = StudyBuddyAgent()
                    return agent
    
    def test_generate_instant_feedback(self, agent):
        """Test generating instant feedback for answers"""
        feedback = agent.generate_feedback(
            question="What is 2+2?",
            student_answer="4",
            correct_answer="4",
            feedback_type=FeedbackType.INSTANT
        )
        
        assert feedback["correct"] == True
        assert "message" in feedback
        assert "explanation" in feedback
        assert feedback["feedback_type"] == FeedbackType.INSTANT
    
    def test_generate_detailed_feedback(self, agent):
        """Test generating detailed feedback"""
        feedback = agent.generate_feedback(
            question="Explain photosynthesis",
            student_answer="Plants make food from sunlight",
            correct_answer="Photosynthesis is the process by which plants convert light energy into chemical energy",
            feedback_type=FeedbackType.DETAILED
        )
        
        assert "message" in feedback
        assert "explanation" in feedback
        assert "suggestions" in feedback
        assert len(feedback["suggestions"]) > 0
    
    def test_generate_encouraging_feedback(self, agent):
        """Test generating encouraging feedback for incorrect answers"""
        feedback = agent.generate_feedback(
            question="What is the capital of Turkey?",
            student_answer="Istanbul",
            correct_answer="Ankara",
            feedback_type=FeedbackType.ENCOURAGING
        )
        
        assert feedback["correct"] == False
        assert "message" in feedback
        # Should be encouraging despite being wrong
        assert any(word in feedback["message"].lower() 
                  for word in ["try", "close", "good effort", "almost"])
    
    def test_generate_adaptive_feedback(self, agent):
        """Test adaptive feedback based on performance"""
        # Student struggling
        feedback_struggling = agent.generate_adaptive_feedback(
            recent_scores=[0.2, 0.3, 0.25],
            current_topic="algebra"
        )
        
        assert "support" in feedback_struggling["strategy"]
        assert "simpler" in feedback_struggling["recommendation"].lower()
        
        # Student excelling
        feedback_excelling = agent.generate_adaptive_feedback(
            recent_scores=[0.9, 0.85, 0.95],
            current_topic="algebra"
        )
        
        assert "challenge" in feedback_excelling["strategy"]
        assert "advanced" in feedback_excelling["recommendation"].lower()


class TestStudyTips:
    """Test study tips and recommendations"""
    
    @pytest.fixture
    def agent(self):
        """Create StudyBuddyAgent instance"""
        with patch('src.agents.study_buddy_agent_clean.DataProcessor'):
            agent = StudyBuddyAgent()
            return agent
    
    def test_generate_study_tips_for_topic(self, agent):
        """Test generating study tips for specific topic"""
        tips = agent.generate_study_tips(
            topic="matematik",
            learning_style="visual",
            difficulty_level=0.6
        )
        
        assert len(tips) > 0
        assert all(isinstance(tip, dict) for tip in tips)
        assert all("tip" in t and "category" in t for t in tips)
    
    def test_generate_personalized_tips(self, agent):
        """Test generating personalized study tips"""
        tips = agent.generate_personalized_tips(
            student_id="student_123",
            recent_performance={"math": 0.7, "science": 0.5},
            weak_areas=["problem_solving", "time_management"]
        )
        
        assert len(tips) > 0
        # Should address weak areas
        assert any("problem" in tip["tip"].lower() or "time" in tip["tip"].lower() 
                  for tip in tips)
    
    def test_get_exam_preparation_tips(self, agent):
        """Test getting exam preparation tips"""
        tips = agent.get_exam_prep_tips(
            exam_type="YKS",
            days_until_exam=30,
            subjects=["matematik", "fizik", "kimya"]
        )
        
        assert "daily_plan" in tips
        assert "subject_allocation" in tips
        assert "practice_recommendations" in tips
        assert len(tips["subject_allocation"]) == 3
    
    def test_get_motivation_tips(self, agent):
        """Test getting motivation tips for struggling students"""
        tips = agent.get_motivation_tips(
            performance_trend="declining",
            study_streak_days=2
        )
        
        assert len(tips) > 0
        assert any("motivation" in t["category"].lower() for t in tips)


class TestPerformanceAnalytics:
    """Test performance analytics and tracking"""
    
    @pytest.fixture
    def agent(self):
        """Create StudyBuddyAgent instance"""
        with patch('src.agents.study_buddy_agent_clean.DataProcessor'):
            with patch('src.agents.study_buddy_agent_clean.IRTService'):
                agent = StudyBuddyAgent()
                return agent
    
    def test_calculate_performance_metrics(self, agent):
        """Test calculating performance metrics"""
        responses = [
            {"correct": True, "time": 30, "difficulty": 0.5},
            {"correct": False, "time": 45, "difficulty": 0.6},
            {"correct": True, "time": 25, "difficulty": 0.4},
            {"correct": True, "time": 35, "difficulty": 0.5},
            {"correct": False, "time": 60, "difficulty": 0.7}
        ]
        
        metrics = agent.calculate_performance_metrics(responses)
        
        assert metrics["accuracy"] == 0.6  # 3/5
        assert metrics["average_time"] == 39  # (30+45+25+35+60)/5
        assert metrics["average_difficulty"] == 0.54
        assert metrics["improvement_rate"] is not None
    
    def test_identify_knowledge_gaps(self, agent):
        """Test identifying knowledge gaps"""
        quiz_results = [
            {"topic": "algebra", "subtopic": "equations", "correct": False},
            {"topic": "algebra", "subtopic": "equations", "correct": False},
            {"topic": "algebra", "subtopic": "functions", "correct": True},
            {"topic": "geometry", "subtopic": "triangles", "correct": False},
            {"topic": "geometry", "subtopic": "circles", "correct": True}
        ]
        
        gaps = agent.identify_knowledge_gaps(quiz_results)
        
        assert "algebra-equations" in gaps
        assert "geometry-triangles" in gaps
        assert gaps["algebra-equations"]["weakness_score"] > 0.5
    
    def test_predict_success_probability(self, agent):
        """Test predicting success probability"""
        student_ability = 0.6
        question_difficulty = 0.5
        
        probability = agent.predict_success_probability(
            student_ability, 
            question_difficulty
        )
        
        assert 0.0 <= probability <= 1.0
        assert probability > 0.5  # Student ability > difficulty
    
    def test_calculate_learning_velocity(self, agent):
        """Test calculating learning velocity"""
        progress_history = [
            {"date": "2024-01-01", "ability": 0.4},
            {"date": "2024-01-08", "ability": 0.45},
            {"date": "2024-01-15", "ability": 0.52},
            {"date": "2024-01-22", "ability": 0.58}
        ]
        
        velocity = agent.calculate_learning_velocity(progress_history)
        
        assert velocity > 0  # Positive learning trend
        assert "rate" in velocity
        assert "trend" in velocity
        assert velocity["trend"] == "improving"


class TestAIInteractions:
    """Test AI-powered interactions"""
    
    @pytest.fixture
    def agent(self):
        """Create StudyBuddyAgent instance with mocked AI"""
        with patch('src.agents.study_buddy_agent_clean.ModelIntegration') as mock_model:
            mock_model.return_value.generate.return_value = "AI generated response"
            agent = StudyBuddyAgent()
            return agent
    
    def test_answer_student_question(self, agent):
        """Test answering student questions"""
        response = agent.answer_question(
            question="What is the Pythagorean theorem?",
            context="geometry",
            student_level=0.5
        )
        
        assert response is not None
        assert "answer" in response
        assert "confidence" in response
        assert len(response["answer"]) > 0
    
    def test_explain_concept(self, agent):
        """Test explaining concepts"""
        explanation = agent.explain_concept(
            concept="photosynthesis",
            student_age=14,
            learning_style="visual",
            detail_level="simple"
        )
        
        assert explanation is not None
        assert "explanation" in explanation
        assert "examples" in explanation
        assert "visual_aids" in explanation  # For visual learner
    
    def test_generate_practice_problems(self, agent):
        """Test generating practice problems"""
        problems = agent.generate_practice_problems(
            topic="quadratic equations",
            num_problems=5,
            difficulty_range=(0.4, 0.6)
        )
        
        assert len(problems) == 5
        assert all("problem" in p and "solution" in p for p in problems)
        assert all(0.4 <= p["difficulty"] <= 0.6 for p in problems)
    
    def test_provide_hint(self, agent):
        """Test providing hints for problems"""
        hint = agent.provide_hint(
            problem="Solve: x^2 + 5x + 6 = 0",
            hint_level=1,  # First hint
            previous_attempts=[]
        )
        
        assert hint is not None
        assert "hint" in hint
        assert "hint_level" in hint
        assert hint["hint_level"] == 1
        assert not hint["reveals_answer"]
    
    def test_progressive_hints(self, agent):
        """Test progressive hint system"""
        problem = "Find the derivative of f(x) = x^3 + 2x^2 - 5x + 3"
        
        hint1 = agent.provide_hint(problem, hint_level=1)
        hint2 = agent.provide_hint(problem, hint_level=2)
        hint3 = agent.provide_hint(problem, hint_level=3)
        
        # Hints should get progressively more helpful
        assert hint1["hint_level"] < hint2["hint_level"] < hint3["hint_level"]
        assert hint3["reveals_answer"] or hint3["very_detailed"]


class TestAdaptiveLearning:
    """Test adaptive learning features"""
    
    @pytest.fixture
    def agent(self):
        """Create StudyBuddyAgent instance"""
        with patch('src.agents.study_buddy_agent_clean.IRTService'):
            agent = StudyBuddyAgent()
            return agent
    
    def test_adapt_content_difficulty(self, agent):
        """Test adapting content difficulty based on performance"""
        # Student performing well
        adapted_content = agent.adapt_content(
            current_difficulty=0.5,
            recent_performance=[0.8, 0.9, 0.85],
            target_success_rate=0.7
        )
        
        assert adapted_content["new_difficulty"] > 0.5  # Should increase
        
        # Student struggling
        adapted_content = agent.adapt_content(
            current_difficulty=0.5,
            recent_performance=[0.3, 0.4, 0.35],
            target_success_rate=0.7
        )
        
        assert adapted_content["new_difficulty"] < 0.5  # Should decrease
    
    def test_recommend_next_topic(self, agent):
        """Test recommending next topic to study"""
        completed_topics = ["algebra_basics", "linear_equations"]
        performance = {"algebra_basics": 0.8, "linear_equations": 0.75}
        
        next_topic = agent.recommend_next_topic(
            completed_topics=completed_topics,
            performance=performance,
            curriculum=["algebra_basics", "linear_equations", "quadratic_equations", "functions"]
        )
        
        assert next_topic == "quadratic_equations"
        assert next_topic not in completed_topics
    
    def test_create_personalized_curriculum(self, agent):
        """Test creating personalized curriculum"""
        curriculum = agent.create_personalized_curriculum(
            student_id="student_123",
            grade_level="9",
            strengths=["algebra"],
            weaknesses=["geometry"],
            learning_pace="slow",
            available_hours_per_week=10
        )
        
        assert "topics" in curriculum
        assert "schedule" in curriculum
        assert "estimated_completion" in curriculum
        assert len(curriculum["topics"]) > 0
        # Should have more geometry practice due to weakness
        geometry_topics = [t for t in curriculum["topics"] if "geometry" in t.lower()]
        assert len(geometry_topics) > 0


class TestErrorHandling:
    """Test error handling in StudyBuddyAgent"""
    
    @pytest.fixture
    def agent(self):
        """Create StudyBuddyAgent instance"""
        return StudyBuddyAgent()
    
    def test_handle_invalid_student_id(self, agent):
        """Test handling invalid student ID"""
        with pytest.raises(ValueError):
            agent.start_study_session(
                student_id="",
                subject="math",
                duration_minutes=30
            )
    
    def test_handle_model_failure(self, agent):
        """Test handling AI model failure"""
        with patch.object(agent, 'model_integration') as mock_model:
            mock_model.generate.side_effect = Exception("Model error")
            
            # Should fallback gracefully
            response = agent.answer_question(
                question="Test question",
                context="math"
            )
            
            assert response["answer"] == "I'm having trouble generating a response right now."
            assert response["error"] == True
    
    def test_handle_database_error(self, agent):
        """Test handling database errors"""
        with patch.object(agent, 'save_session') as mock_save:
            mock_save.side_effect = Exception("Database error")
            
            # Should handle error gracefully
            session = agent.start_study_session(
                student_id="student_123",
                subject="math",
                duration_minutes=30
            )
            
            assert session is not None  # Session created despite save error