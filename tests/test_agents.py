# -*- coding: utf-8 -*-
"""
Test suite for TEKNOFEST Education Agents
"""

import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agents.learning_path_agent_v2 import LearningPathAgent
from src.agents.study_buddy_agent import StudyBuddyAgent

def test_learning_style_detection():
    """Test VARK learning style detection"""
    agent = LearningPathAgent()
    responses = ["görsel öğreniyorum", "şemalar faydalı"]
    
    result = agent.detect_learning_style(responses)
    
    assert result['dominant_style'] == 'visual'
    assert result['scores']['visual'] > 0
    print("[OK] Learning style detection test passed")

def test_learning_path_generation():
    """Test learning path generation"""
    agent = LearningPathAgent()
    profile = {
        'student_id': 'test123',
        'current_level': 0.3,
        'target_level': 0.8,
        'learning_style': 'visual',
        'grade': 9
    }
    
    path = agent.generate_learning_path(profile, "Matematik", weeks=4)
    
    assert 'weekly_plan' in path
    assert len(path['weekly_plan']) == 4  # 4 hafta
    assert path['weekly_plan'][0]['difficulty'] < path['weekly_plan'][-1]['difficulty']  # Artan zorluk
    print("[OK] Learning path generation test passed")

def test_adaptive_quiz():
    """Test adaptive quiz generation"""
    agent = StudyBuddyAgent()
    quiz = agent.generate_adaptive_quiz(
        topic="Matematik",
        student_ability=0.5,
        num_questions=5
    )
    
    assert len(quiz) == 5
    for q in quiz:
        assert 'id' in q
        assert 'difficulty' in q
        assert 0 <= q['difficulty'] <= 1
    print("[OK] Adaptive quiz generation test passed")

def test_study_plan():
    """Test study plan generation"""
    agent = StudyBuddyAgent()
    weak_topics = ["Denklemler", "Geometri"]
    plan = agent.generate_study_plan(weak_topics, available_hours=10)
    
    assert 'total_hours' in plan
    assert plan['total_hours'] == 10
    assert len(plan['topics']) == 2
    print("[OK] Study plan generation test passed")

if __name__ == "__main__":
    # Run all tests
    print("\n=== TEKNOFEST 2025 Agent Tests ===\n")
    
    test_functions = [
        test_learning_style_detection,
        test_learning_path_generation,
        test_adaptive_quiz,
        test_study_plan
    ]
    
    passed = 0
    failed = 0
    
    for test_func in test_functions:
        try:
            test_func()
            passed += 1
        except AssertionError as e:
            print(f"[FAIL] {test_func.__name__} failed: {e}")
            failed += 1
        except Exception as e:
            print(f"[ERROR] {test_func.__name__} error: {e}")
            failed += 1
    
    print(f"\n=== Results: {passed} passed, {failed} failed ===")
    
    if failed == 0:
        print("[SUCCESS] All tests passed!")
    else:
        print(f"[FAILED] {failed} tests failed")