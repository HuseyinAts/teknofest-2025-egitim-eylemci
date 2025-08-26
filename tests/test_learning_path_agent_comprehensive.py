"""
Comprehensive Test Suite for LearningPathAgent
Target: Increase coverage to 80%+
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime, timedelta
import json

from src.agents.learning_path_agent_v2 import (
    LearningPathAgent,
    LearningStyle,
    DifficultyLevel,
    StudentProfile,
    LearningPath
)


class TestStudentProfile:
    """Test StudentProfile dataclass and methods"""
    
    def test_student_profile_creation(self):
        """Test creating a student profile"""
        profile = StudentProfile(
            student_id="test_123",
            learning_style=LearningStyle.VISUAL,
            current_level=0.5,
            strengths=["math", "science"],
            weaknesses=["reading"],
            preferences={"pace": "slow"}
        )
        
        assert profile.student_id == "test_123"
        assert profile.learning_style == LearningStyle.VISUAL
        assert profile.current_level == 0.5
        assert "math" in profile.strengths
        assert "reading" in profile.weaknesses
        assert profile.preferences["pace"] == "slow"
    
    def test_student_profile_to_dict(self):
        """Test converting student profile to dictionary"""
        profile = StudentProfile(
            student_id="test_123",
            learning_style=LearningStyle.VISUAL,
            current_level=0.5,
            strengths=["math"],
            weaknesses=["reading"],
            preferences={"pace": "slow"}
        )
        
        result = profile.to_dict()
        
        assert result["student_id"] == "test_123"
        assert result["learning_style"] == "visual"
        assert result["current_level"] == 0.5
        assert result["strengths"] == ["math"]
        assert result["weaknesses"] == ["reading"]
        assert result["preferences"] == {"pace": "slow"}


class TestLearningPath:
    """Test LearningPath dataclass and methods"""
    
    def test_learning_path_creation(self):
        """Test creating a learning path"""
        path = LearningPath(
            path_id="path_123",
            student_id="student_123",
            topics=["algebra", "geometry"],
            difficulty_progression=[0.3, 0.5, 0.7],
            estimated_duration=120,
            adaptive_checkpoints=[7, 14, 21],
            resources={"videos": ["url1", "url2"]},
            created_at=datetime.now()
        )
        
        assert path.path_id == "path_123"
        assert path.student_id == "student_123"
        assert len(path.topics) == 2
        assert len(path.difficulty_progression) == 3
        assert path.estimated_duration == 120
        assert len(path.adaptive_checkpoints) == 3
        assert "videos" in path.resources
    
    def test_learning_path_to_dict(self):
        """Test converting learning path to dictionary"""
        created_time = datetime.now()
        path = LearningPath(
            path_id="path_123",
            student_id="student_123",
            topics=["algebra"],
            difficulty_progression=[0.3],
            estimated_duration=60,
            adaptive_checkpoints=[7],
            resources={"videos": ["url1"]},
            created_at=created_time
        )
        
        result = path.to_dict()
        
        assert result["path_id"] == "path_123"
        assert result["student_id"] == "student_123"
        assert result["topics"] == ["algebra"]
        assert result["difficulty_progression"] == [0.3]
        assert result["estimated_duration"] == 60
        assert result["adaptive_checkpoints"] == [7]
        assert result["resources"] == {"videos": ["url1"]}
        assert result["created_at"] == created_time.isoformat()


class TestLearningPathAgent:
    """Comprehensive tests for LearningPathAgent"""
    
    @pytest.fixture
    def agent(self):
        """Create a LearningPathAgent instance for testing"""
        with patch('src.agents.learning_path_agent_v2.DataProcessor'):
            agent = LearningPathAgent()
            return agent
    
    def test_agent_initialization(self, agent):
        """Test agent initialization"""
        assert agent is not None
        assert hasattr(agent, 'vark_questions')
        assert hasattr(agent, 'curriculum')
        assert hasattr(agent, 'learning_strategies')
        assert hasattr(agent, 'student_profiles')
        assert hasattr(agent, 'learning_paths')
    
    def test_detect_learning_style_visual(self, agent):
        """Test detecting visual learning style"""
        responses = ["a", "a", "a", "a", "a"]  # All visual responses
        result = agent.detect_learning_style(responses)
        
        assert result["dominant_style"] == "visual"
        assert result["scores"]["visual"] > result["scores"]["auditory"]
        assert result["scores"]["visual"] > result["scores"]["reading"]
        assert result["scores"]["visual"] > result["scores"]["kinesthetic"]
    
    def test_detect_learning_style_auditory(self, agent):
        """Test detecting auditory learning style"""
        responses = ["b", "b", "b", "b", "b"]  # All auditory responses
        result = agent.detect_learning_style(responses)
        
        assert result["dominant_style"] == "auditory"
        assert result["scores"]["auditory"] > result["scores"]["visual"]
    
    def test_detect_learning_style_mixed(self, agent):
        """Test detecting mixed learning style"""
        responses = ["a", "b", "c", "d", "a"]  # Mixed responses
        result = agent.detect_learning_style(responses)
        
        assert "dominant_style" in result
        assert "scores" in result
        assert "confidence" in result
        assert len(result["scores"]) == 4
    
    def test_detect_learning_style_invalid_responses(self, agent):
        """Test detecting learning style with invalid responses"""
        responses = ["x", "y", "z"]  # Invalid responses
        result = agent.detect_learning_style(responses)
        
        assert result["confidence"] < 0.5  # Low confidence for invalid responses
    
    def test_calculate_zpd_level_beginner(self, agent):
        """Test calculating ZPD for beginner level"""
        result = agent.calculate_zpd_level(
            current_performance=0.2,
            task_difficulty=0.3,
            recent_progress=[0.1, 0.15, 0.2]
        )
        
        assert "recommended_difficulty" in result
        assert "zpd_range" in result
        assert result["zpd_range"]["min"] >= 0.0
        assert result["zpd_range"]["max"] <= 1.0
        assert result["recommended_difficulty"] > 0.2  # Should be slightly harder
    
    def test_calculate_zpd_level_advanced(self, agent):
        """Test calculating ZPD for advanced level"""
        result = agent.calculate_zpd_level(
            current_performance=0.8,
            task_difficulty=0.7,
            recent_progress=[0.7, 0.75, 0.8]
        )
        
        assert result["recommended_difficulty"] > 0.7
        assert result["recommended_difficulty"] <= 1.0
        assert result["confidence"] > 0.5
    
    def test_calculate_zpd_level_with_plateau(self, agent):
        """Test calculating ZPD when student has plateaued"""
        result = agent.calculate_zpd_level(
            current_performance=0.5,
            task_difficulty=0.5,
            recent_progress=[0.5, 0.5, 0.5]  # No progress
        )
        
        assert "adjustment_reason" in result
        assert "plateau" in result["adjustment_reason"].lower()
    
    def test_get_curriculum_topics_valid_grade(self, agent):
        """Test getting curriculum topics for valid grade"""
        topics = agent.get_curriculum_topics("9. Sınıf")
        
        assert isinstance(topics, list)
        assert len(topics) > 0
        if topics:  # If curriculum is loaded
            assert all(isinstance(topic, dict) for topic in topics)
    
    def test_get_curriculum_topics_invalid_grade(self, agent):
        """Test getting curriculum topics for invalid grade"""
        topics = agent.get_curriculum_topics("InvalidGrade")
        
        assert isinstance(topics, list)
        assert len(topics) == 0
    
    def test_create_learning_path_basic(self, agent):
        """Test creating a basic learning path"""
        profile = StudentProfile(
            student_id="test_student",
            learning_style=LearningStyle.VISUAL,
            current_level=0.5,
            strengths=["math"],
            weaknesses=["reading"],
            preferences={"pace": "normal"}
        )
        
        path = agent.create_learning_path(
            student_profile=profile,
            target_topics=["algebra", "geometry"],
            time_constraint=30
        )
        
        assert path.student_id == "test_student"
        assert len(path.topics) > 0
        assert len(path.difficulty_progression) > 0
        assert path.estimated_duration > 0
        assert path.path_id is not None
    
    def test_create_learning_path_with_zpd(self, agent):
        """Test creating learning path with ZPD optimization"""
        profile = StudentProfile(
            student_id="test_student",
            learning_style=LearningStyle.KINESTHETIC,
            current_level=0.3,
            strengths=[],
            weaknesses=["math"],
            preferences={"pace": "slow"}
        )
        
        path = agent.create_learning_path(
            student_profile=profile,
            target_topics=["basic_math"],
            time_constraint=60,
            optimize_for_zpd=True
        )
        
        assert path.difficulty_progression[0] <= 0.5  # Should start easy
        assert path.estimated_duration >= 60  # Should respect time constraint
    
    def test_update_progress_success(self, agent):
        """Test updating student progress after success"""
        # Create initial profile and path
        profile = StudentProfile(
            student_id="test_student",
            learning_style=LearningStyle.VISUAL,
            current_level=0.5,
            strengths=[],
            weaknesses=[],
            preferences={}
        )
        agent.student_profiles["test_student"] = profile
        
        path = LearningPath(
            path_id="path_123",
            student_id="test_student",
            topics=["math"],
            difficulty_progression=[0.5, 0.6, 0.7],
            estimated_duration=30,
            adaptive_checkpoints=[],
            resources={},
            created_at=datetime.now()
        )
        agent.learning_paths["path_123"] = path
        
        # Update progress
        updated_profile = agent.update_progress(
            student_id="test_student",
            assessment_results={"score": 0.8, "topics": ["math"]},
            completed_topics=["basic_algebra"]
        )
        
        assert updated_profile.current_level > 0.5  # Level should increase
        assert "basic_algebra" in updated_profile.progress.get("completed_topics", [])
    
    def test_update_progress_failure(self, agent):
        """Test updating student progress after failure"""
        profile = StudentProfile(
            student_id="test_student",
            learning_style=LearningStyle.VISUAL,
            current_level=0.5,
            strengths=[],
            weaknesses=[],
            preferences={}
        )
        agent.student_profiles["test_student"] = profile
        
        updated_profile = agent.update_progress(
            student_id="test_student",
            assessment_results={"score": 0.2, "topics": ["math"]},
            completed_topics=[]
        )
        
        assert updated_profile.current_level <= 0.5  # Level should not increase much
        assert "math" in updated_profile.weaknesses  # Should identify weakness
    
    def test_get_progress_report(self, agent):
        """Test generating progress report"""
        # Setup student profile
        profile = StudentProfile(
            student_id="test_student",
            learning_style=LearningStyle.VISUAL,
            current_level=0.6,
            strengths=["math"],
            weaknesses=["reading"],
            preferences={},
            progress={
                "completed_topics": ["algebra", "geometry"],
                "scores": [0.7, 0.8],
                "time_spent": 120
            }
        )
        agent.student_profiles["test_student"] = profile
        
        report = agent.get_progress_report("test_student")
        
        assert report["student_id"] == "test_student"
        assert report["current_level"] == 0.6
        assert report["total_topics_completed"] == 2
        assert report["average_score"] == 0.75
        assert report["strengths"] == ["math"]
        assert report["areas_for_improvement"] == ["reading"]
    
    def test_get_progress_report_nonexistent_student(self, agent):
        """Test generating report for non-existent student"""
        report = agent.get_progress_report("nonexistent_student")
        
        assert report == {}
    
    def test_get_recommendations_for_struggling_student(self, agent):
        """Test getting recommendations for struggling student"""
        profile = StudentProfile(
            student_id="test_student",
            learning_style=LearningStyle.VISUAL,
            current_level=0.2,
            strengths=[],
            weaknesses=["math", "science"],
            preferences={"pace": "slow"}
        )
        agent.student_profiles["test_student"] = profile
        
        recommendations = agent.get_recommendations(
            student_id="test_student",
            context="struggling"
        )
        
        assert "strategies" in recommendations
        assert "resources" in recommendations
        assert "next_topics" in recommendations
        assert any("visual" in s.lower() for s in recommendations["strategies"])
    
    def test_get_recommendations_for_advanced_student(self, agent):
        """Test getting recommendations for advanced student"""
        profile = StudentProfile(
            student_id="test_student",
            learning_style=LearningStyle.READING,
            current_level=0.9,
            strengths=["math", "science", "programming"],
            weaknesses=[],
            preferences={"pace": "fast"}
        )
        agent.student_profiles["test_student"] = profile
        
        recommendations = agent.get_recommendations(
            student_id="test_student",
            context="advanced"
        )
        
        assert "enrichment" in str(recommendations).lower() or "challenge" in str(recommendations).lower()
    
    def test_get_personalized_content_visual_learner(self, agent):
        """Test getting personalized content for visual learner"""
        profile = StudentProfile(
            student_id="test_student",
            learning_style=LearningStyle.VISUAL,
            current_level=0.5,
            strengths=[],
            weaknesses=[],
            preferences={}
        )
        agent.student_profiles["test_student"] = profile
        
        content = agent.get_personalized_content(
            student_id="test_student",
            topic="geometry",
            content_type="lesson"
        )
        
        assert "format" in content
        assert content["format"] == "visual"
        assert "adaptations" in content
        assert any("diagram" in a.lower() or "visual" in a.lower() 
                  for a in content["adaptations"])
    
    def test_get_personalized_content_kinesthetic_learner(self, agent):
        """Test getting personalized content for kinesthetic learner"""
        profile = StudentProfile(
            student_id="test_student",
            learning_style=LearningStyle.KINESTHETIC,
            current_level=0.5,
            strengths=[],
            weaknesses=[],
            preferences={}
        )
        agent.student_profiles["test_student"] = profile
        
        content = agent.get_personalized_content(
            student_id="test_student",
            topic="physics",
            content_type="exercise"
        )
        
        assert content["format"] == "kinesthetic"
        assert any("hands-on" in a.lower() or "interactive" in a.lower() 
                  for a in content["adaptations"])
    
    def test_optimize_path(self, agent):
        """Test optimizing learning path"""
        path = LearningPath(
            path_id="path_123",
            student_id="test_student",
            topics=["math", "science", "history"],
            difficulty_progression=[0.3, 0.5, 0.7],
            estimated_duration=90,
            adaptive_checkpoints=[],
            resources={},
            created_at=datetime.now()
        )
        
        performance_data = {
            "scores": [0.8, 0.6, 0.4],
            "time_spent": [20, 30, 40],
            "attempts": [1, 2, 3]
        }
        
        optimized_path = agent.optimize_path(path, performance_data)
        
        assert optimized_path is not None
        assert len(optimized_path.topics) > 0
        assert len(optimized_path.difficulty_progression) > 0
    
    @pytest.mark.asyncio
    async def test_optimize_path_async(self, agent):
        """Test async path optimization"""
        path = LearningPath(
            path_id="path_123",
            student_id="test_student",
            topics=["math"],
            difficulty_progression=[0.5],
            estimated_duration=30,
            adaptive_checkpoints=[],
            resources={},
            created_at=datetime.now()
        )
        
        performance_data = {"scores": [0.7]}
        
        optimized_path = await agent.optimize_path_async(path, performance_data)
        
        assert optimized_path is not None
    
    def test_optimize_multi_objective(self, agent):
        """Test multi-objective optimization"""
        paths = [
            LearningPath(
                path_id=f"path_{i}",
                student_id="test_student",
                topics=[f"topic_{i}"],
                difficulty_progression=[0.5],
                estimated_duration=30,
                adaptive_checkpoints=[],
                resources={},
                created_at=datetime.now()
            )
            for i in range(3)
        ]
        
        objectives = {
            "maximize_engagement": 0.8,
            "minimize_time": 0.5,
            "maximize_retention": 0.9
        }
        
        optimal_path = agent.optimize_multi_objective(paths, objectives)
        
        assert optimal_path is not None
        assert optimal_path.path_id in [p.path_id for p in paths]
    
    @pytest.mark.asyncio
    async def test_optimize_multi_objective_async(self, agent):
        """Test async multi-objective optimization"""
        paths = [
            LearningPath(
                path_id="path_1",
                student_id="test_student",
                topics=["math"],
                difficulty_progression=[0.5],
                estimated_duration=30,
                adaptive_checkpoints=[],
                resources={},
                created_at=datetime.now()
            )
        ]
        
        objectives = {"maximize_engagement": 1.0}
        
        optimal_path = await agent.optimize_multi_objective_async(paths, objectives)
        
        assert optimal_path is not None


class TestDifficultyLevel:
    """Test DifficultyLevel enum"""
    
    def test_difficulty_values(self):
        """Test difficulty level values"""
        assert DifficultyLevel.VERY_EASY.value == 0.2
        assert DifficultyLevel.EASY.value == 0.4
        assert DifficultyLevel.MEDIUM.value == 0.6
        assert DifficultyLevel.HARD.value == 0.8
        assert DifficultyLevel.VERY_HARD.value == 1.0
    
    def test_difficulty_ordering(self):
        """Test difficulty level ordering"""
        assert DifficultyLevel.VERY_EASY.value < DifficultyLevel.EASY.value
        assert DifficultyLevel.EASY.value < DifficultyLevel.MEDIUM.value
        assert DifficultyLevel.MEDIUM.value < DifficultyLevel.HARD.value
        assert DifficultyLevel.HARD.value < DifficultyLevel.VERY_HARD.value


class TestLearningStyle:
    """Test LearningStyle enum"""
    
    def test_learning_style_values(self):
        """Test learning style values"""
        assert LearningStyle.VISUAL.value == "visual"
        assert LearningStyle.AUDITORY.value == "auditory"
        assert LearningStyle.READING.value == "reading"
        assert LearningStyle.KINESTHETIC.value == "kinesthetic"
    
    def test_all_learning_styles_present(self):
        """Test all learning styles are defined"""
        styles = [style.value for style in LearningStyle]
        assert "visual" in styles
        assert "auditory" in styles
        assert "reading" in styles
        assert "kinesthetic" in styles
        assert len(styles) == 4