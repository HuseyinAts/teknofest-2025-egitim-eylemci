"""
Comprehensive tests for Learning Path Agent
"""
import pytest
import json
from unittest.mock import Mock, patch, MagicMock
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agents.learning_path_agent_v2 import LearningPathAgent


class TestLearningPathAgent:
    
    @pytest.fixture
    def agent(self):
        """Create a LearningPathAgent instance for testing"""
        return LearningPathAgent()
    
    @pytest.fixture
    def sample_profile(self):
        """Sample student profile for testing"""
        return {
            'student_id': 'test_student_001',
            'current_level': 0.4,
            'target_level': 0.9,
            'learning_style': 'visual',
            'grade': 10,
            'weak_topics': ['Algebra', 'Geometry'],
            'strong_topics': ['Trigonometry']
        }
    
    @pytest.mark.unit
    def test_agent_initialization(self, agent):
        """Test agent initialization"""
        assert agent is not None
        assert hasattr(agent, 'detect_learning_style')
        assert hasattr(agent, 'generate_learning_path')
    
    @pytest.mark.unit
    def test_detect_learning_style_visual(self, agent):
        """Test visual learning style detection"""
        responses = [
            "Görsel materyaller kullanmayı severim",
            "Şema ve diyagramlar bana yardımcı oluyor",
            "Renkli notlar alıyorum"
        ]
        
        result = agent.detect_learning_style(responses)
        
        assert result['dominant_style'] == 'visual'
        assert result['scores']['visual'] > result['scores']['auditory']
        assert result['scores']['visual'] > result['scores']['reading']
        assert result['scores']['visual'] > result['scores']['kinesthetic']
        assert 'confidence' in result
        assert 0 <= result['confidence'] <= 1
    
    @pytest.mark.unit
    def test_detect_learning_style_auditory(self, agent):
        """Test auditory learning style detection"""
        responses = [
            "Dinleyerek öğrenirim",
            "Sesli okuma yaparım",
            "Tartışmalar bana yardımcı olur"
        ]
        
        result = agent.detect_learning_style(responses)
        
        assert result['dominant_style'] == 'auditory'
        assert result['scores']['auditory'] > 0
    
    @pytest.mark.unit
    def test_detect_learning_style_kinesthetic(self, agent):
        """Test kinesthetic learning style detection"""
        responses = [
            "Yaparak öğrenirim",
            "Pratik uygulamalar faydalı",
            "Hareket ederek çalışırım"
        ]
        
        result = agent.detect_learning_style(responses)
        
        assert result['dominant_style'] == 'kinesthetic'
        assert result['scores']['kinesthetic'] > 0
    
    @pytest.mark.unit
    def test_generate_learning_path_basic(self, agent, sample_profile):
        """Test basic learning path generation"""
        path = agent.generate_learning_path(
            sample_profile,
            subject="Matematik",
            weeks=6
        )
        
        assert 'student_id' in path
        assert path['student_id'] == sample_profile['student_id']
        assert 'subject' in path
        assert path['subject'] == "Matematik"
        assert 'weekly_plan' in path
        assert len(path['weekly_plan']) == 6
        assert 'total_duration_weeks' in path
        assert path['total_duration_weeks'] == 6
    
    @pytest.mark.unit
    def test_learning_path_difficulty_progression(self, agent, sample_profile):
        """Test that difficulty increases progressively"""
        path = agent.generate_learning_path(
            sample_profile,
            subject="Fizik",
            weeks=4
        )
        
        difficulties = [week['difficulty'] for week in path['weekly_plan']]
        
        # Check that difficulty generally increases
        assert difficulties[0] <= difficulties[-1]
        # Check all difficulties are in valid range
        for diff in difficulties:
            assert 0 <= diff <= 1
    
    @pytest.mark.unit
    def test_learning_path_with_custom_parameters(self, agent):
        """Test learning path with custom parameters"""
        custom_profile = {
            'student_id': 'advanced_student',
            'current_level': 0.7,
            'target_level': 0.95,
            'learning_style': 'reading',
            'grade': 12
        }
        
        path = agent.generate_learning_path(
            custom_profile,
            subject="Kimya",
            weeks=8
        )
        
        assert len(path['weekly_plan']) == 8
        assert path['learning_style'] == 'reading'
        # Advanced student should start with higher difficulty
        assert path['weekly_plan'][0]['difficulty'] >= 0.6
    
    @pytest.mark.unit
    def test_empty_responses_handling(self, agent):
        """Test handling of empty responses"""
        result = agent.detect_learning_style([])
        
        assert result['dominant_style'] in ['visual', 'auditory', 'reading', 'kinesthetic']
        assert all(score >= 0 for score in result['scores'].values())
    
    @pytest.mark.unit
    def test_invalid_profile_handling(self, agent):
        """Test handling of invalid profile data"""
        invalid_profile = {
            'student_id': 'test',
            # Missing required fields
        }
        
        # Should handle gracefully without raising exception
        path = agent.generate_learning_path(
            invalid_profile,
            subject="Test Subject",
            weeks=2
        )
        
        assert path is not None
        assert 'weekly_plan' in path
    
    @pytest.mark.unit
    @pytest.mark.parametrize("weeks", [1, 4, 12, 52])
    def test_different_duration_plans(self, agent, sample_profile, weeks):
        """Test learning paths with different durations"""
        path = agent.generate_learning_path(
            sample_profile,
            subject="Test Subject",
            weeks=weeks
        )
        
        assert len(path['weekly_plan']) == weeks
        assert path['total_duration_weeks'] == weeks
    
    @pytest.mark.unit
    def test_learning_path_topics_coverage(self, agent, sample_profile):
        """Test that weak topics are prioritized in learning path"""
        path = agent.generate_learning_path(
            sample_profile,
            subject="Matematik",
            weeks=4
        )
        
        # Check that weak topics appear in early weeks
        early_weeks_topics = []
        for week in path['weekly_plan'][:2]:
            if 'topics' in week:
                early_weeks_topics.extend(week.get('topics', []))
        
        # At least one weak topic should be addressed early
        assert any(topic in str(early_weeks_topics).lower() 
                  for topic in ['algebra', 'geometri'])
    
    @pytest.mark.unit
    def test_learning_style_mixed_responses(self, agent):
        """Test with mixed learning style responses"""
        mixed_responses = [
            "Görsel öğrenirim",  # Visual
            "Dinleyerek anlarım",  # Auditory
            "Okuyarak çalışırım",  # Reading
            "Yaparak öğrenirim"  # Kinesthetic
        ]
        
        result = agent.detect_learning_style(mixed_responses)
        
        assert result['dominant_style'] is not None
        assert sum(result['scores'].values()) > 0
        assert result['confidence'] <= 1.0  # Lower confidence for mixed styles
    
    @pytest.mark.unit
    def test_generate_learning_path_adaptivity(self, agent):
        """Test adaptive features in learning path"""
        # Low performing student
        low_profile = {
            'student_id': 'low_performer',
            'current_level': 0.2,
            'target_level': 0.6,
            'learning_style': 'visual',
            'grade': 9
        }
        
        # High performing student
        high_profile = {
            'student_id': 'high_performer',
            'current_level': 0.8,
            'target_level': 0.95,
            'learning_style': 'reading',
            'grade': 11
        }
        
        low_path = agent.generate_learning_path(low_profile, "Math", 4)
        high_path = agent.generate_learning_path(high_profile, "Math", 4)
        
        # High performer should start with higher difficulty
        assert low_path['weekly_plan'][0]['difficulty'] < high_path['weekly_plan'][0]['difficulty']
        
        # Both should show progression
        assert low_path['weekly_plan'][-1]['difficulty'] > low_path['weekly_plan'][0]['difficulty']
        assert high_path['weekly_plan'][-1]['difficulty'] > high_path['weekly_plan'][0]['difficulty']


@pytest.mark.integration
class TestLearningPathAgentIntegration:
    
    @pytest.fixture
    def agent(self):
        return LearningPathAgent()
    
    def test_complete_learning_flow(self, agent):
        """Test complete learning flow from style detection to path generation"""
        # Step 1: Detect learning style
        responses = ["Görsel öğrenirim", "Diyagramlar faydalı"]
        style_result = agent.detect_learning_style(responses)
        
        # Step 2: Create profile with detected style
        profile = {
            'student_id': 'integration_test',
            'current_level': 0.5,
            'target_level': 0.8,
            'learning_style': style_result['dominant_style'],
            'grade': 10
        }
        
        # Step 3: Generate learning path
        path = agent.generate_learning_path(profile, "Matematik", 4)
        
        # Verify complete flow
        assert path['learning_style'] == style_result['dominant_style']
        assert len(path['weekly_plan']) == 4
        assert all('difficulty' in week for week in path['weekly_plan'])
    
    def test_multiple_subjects_handling(self, agent):
        """Test handling multiple subjects"""
        profile = {
            'student_id': 'multi_subject',
            'current_level': 0.5,
            'target_level': 0.8,
            'learning_style': 'visual',
            'grade': 10
        }
        
        subjects = ["Matematik", "Fizik", "Kimya", "Biyoloji"]
        paths = []
        
        for subject in subjects:
            path = agent.generate_learning_path(profile, subject, 3)
            paths.append(path)
        
        # Verify each subject has unique path
        assert len(paths) == len(subjects)
        assert all(path['subject'] in subjects for path in paths)
        assert len(set(path['subject'] for path in paths)) == len(subjects)