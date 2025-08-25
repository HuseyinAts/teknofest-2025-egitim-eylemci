"""
Unit Tests - Business Logic Services
"""
import pytest
from unittest.mock import Mock, patch, AsyncMock
import math

class TestLearningPathService:
    """Learning path service testleri"""
    
    def test_zpd_calculation(self):
        """Test Zone of Proximal Development calculation"""
        def calculate_zpd(current_level: float, performance: float) -> float:
            # Simple ZPD calculation
            if performance > 0.8:
                # Good performance, increase difficulty
                new_level = current_level + 0.1
            elif performance < 0.6:
                # Poor performance, decrease difficulty
                new_level = current_level - 0.1
            else:
                # Maintain current level
                new_level = current_level
            
            # Clamp between 0 and 1
            return max(0.0, min(1.0, new_level))
        
        # Test cases
        assert calculate_zpd(0.5, 0.9) == 0.6  # Good performance
        assert calculate_zpd(0.5, 0.5) == 0.4  # Poor performance
        assert calculate_zpd(0.5, 0.7) == 0.5  # Average performance
        assert calculate_zpd(0.95, 0.9) == 1.0  # Don't exceed max
        assert calculate_zpd(0.05, 0.4) == 0.0  # Don't go below min
    
    def test_adaptive_content_selection(self):
        """Test adaptive content selection"""
        def select_next_content(student_profile: dict) -> dict:
            current_level = student_profile["current_level"]
            learning_style = student_profile["learning_style"]
            
            # Simple content selection
            difficulty_range = 0.1
            min_difficulty = max(0.0, current_level - difficulty_range)
            max_difficulty = min(1.0, current_level + difficulty_range)
            
            return {
                "difficulty": (min_difficulty + max_difficulty) / 2,
                "style": learning_style,
                "topic": "Selected topic"
            }
        
        student_profile = {
            "current_level": 0.6,
            "learning_style": "visual",
            "completed_topics": ["Algebra", "Geometry"]
        }
        
        next_content = select_next_content(student_profile)
        
        assert next_content is not None
        assert next_content["difficulty"] >= 0.5
        assert next_content["difficulty"] <= 0.7
        assert next_content["style"] == "visual"

class TestIRTService:
    """IRT (Item Response Theory) service testleri"""
    
    def test_probability_calculation(self):
        """Test IRT probability calculation"""
        def calculate_probability(ability: float, difficulty: float, 
                                discrimination: float = 1.0, 
                                guessing: float = 0.25) -> float:
            # 3-parameter logistic model
            z = discrimination * (ability - difficulty)
            probability = guessing + (1 - guessing) / (1 + math.exp(-z))
            return probability
        
        # Test cases
        prob = calculate_probability(0.5, 0.5, 1.0, 0.25)
        assert prob >= 0.25  # At least guessing probability
        assert prob <= 1.0  # Not exceed 1
        assert abs(prob - 0.625) < 0.01  # Expected value
        
        # Test extreme cases
        high_ability = calculate_probability(1.0, 0.0, 1.0, 0.25)
        assert high_ability > 0.9  # High ability on easy question
        
        low_ability = calculate_probability(0.0, 1.0, 1.0, 0.25)
        assert low_ability < 0.35  # Low ability on hard question
    
    def test_ability_estimation(self):
        """Test ability estimation from responses"""
        def estimate_ability(responses: list) -> float:
            # Simple ability estimation
            correct_count = sum(1 for r in responses if r["correct"])
            total_count = len(responses)
            
            # Weight by difficulty
            weighted_score = 0
            for response in responses:
                if response["correct"]:
                    weighted_score += response["difficulty"]
            
            # Normalize
            if total_count > 0:
                ability = (correct_count / total_count) * 0.5 + \
                         (weighted_score / total_count) * 0.5
            else:
                ability = 0.5
            
            return min(1.0, max(0.0, ability))
        
        responses = [
            {"correct": True, "difficulty": 0.3},
            {"correct": True, "difficulty": 0.5},
            {"correct": False, "difficulty": 0.8},
            {"correct": True, "difficulty": 0.6}
        ]
        
        ability = estimate_ability(responses)
        
        assert 0 <= ability <= 1
        assert ability > 0.5  # Should be above average

class TestGamificationService:
    """Gamification service testleri"""
    
    def test_xp_calculation(self):
        """Test XP calculation"""
        def calculate_xp(quiz_score: float, difficulty: float, 
                        time_bonus: bool = False) -> int:
            # Base XP from score
            base_xp = int(quiz_score * 10)
            
            # Difficulty multiplier
            difficulty_multiplier = 1 + difficulty
            
            # Time bonus
            time_multiplier = 1.2 if time_bonus else 1.0
            
            total_xp = int(base_xp * difficulty_multiplier * time_multiplier)
            
            return max(0, total_xp)
        
        # Test cases
        xp_easy = calculate_xp(85, 0.3, False)
        xp_hard = calculate_xp(85, 0.7, False)
        xp_bonus = calculate_xp(85, 0.7, True)
        
        assert xp_easy > 0
        assert xp_hard > xp_easy  # Higher difficulty = more XP
        assert xp_bonus > xp_hard  # Time bonus increases XP
        
        # Test edge cases
        assert calculate_xp(0, 0.5, False) == 0
        assert calculate_xp(100, 1.0, True) > 2000
    
    def test_achievement_unlock(self):
        """Test achievement unlocking"""
        def check_achievements(user_stats: dict) -> list:
            achievements = []
            
            # Check various achievements
            if user_stats.get("streak_days", 0) >= 7:
                achievements.append("Week Warrior")
            
            if user_stats.get("quizzes_completed", 0) >= 25:
                achievements.append("Quiz Master")
            
            if user_stats.get("perfect_scores", 0) >= 5:
                achievements.append("Perfectionist")
            
            if user_stats.get("total_xp", 0) >= 1000:
                achievements.append("XP Hunter")
            
            return achievements
        
        user_stats = {
            "total_xp": 1000,
            "streak_days": 7,
            "quizzes_completed": 25,
            "perfect_scores": 5
        }
        
        new_achievements = check_achievements(user_stats)
        
        assert "Week Warrior" in new_achievements
        assert "Quiz Master" in new_achievements
        assert "Perfectionist" in new_achievements
        assert "XP Hunter" in new_achievements
    
    def test_level_calculation(self):
        """Test level calculation from XP"""
        def calculate_level(total_xp: int) -> dict:
            # XP required per level (exponential growth)
            xp_per_level = [0, 100, 250, 500, 1000, 2000, 4000, 8000, 16000, 32000]
            
            level = 0
            for i, required_xp in enumerate(xp_per_level):
                if total_xp >= required_xp:
                    level = i
                else:
                    break
            
            # Calculate progress to next level
            if level < len(xp_per_level) - 1:
                current_level_xp = xp_per_level[level]
                next_level_xp = xp_per_level[level + 1]
                progress = (total_xp - current_level_xp) / (next_level_xp - current_level_xp)
            else:
                progress = 1.0
            
            return {
                "level": level,
                "progress": min(1.0, max(0.0, progress)),
                "xp_to_next": xp_per_level[level + 1] - total_xp if level < len(xp_per_level) - 1 else 0
            }
        
        # Test cases
        level_data = calculate_level(150)
        assert level_data["level"] == 1
        assert 0 <= level_data["progress"] <= 1
        
        level_data = calculate_level(1500)
        assert level_data["level"] == 4
        
        level_data = calculate_level(50000)
        assert level_data["level"] == 9
        assert level_data["progress"] == 1.0