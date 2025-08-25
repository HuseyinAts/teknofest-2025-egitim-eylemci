# -*- coding: utf-8 -*-
"""
Study Buddy Agent - AI Study Companion
TEKNOFEST 2025 - Education Technologies
"""

import random
from typing import List, Dict, Optional
import math
from datetime import datetime, timedelta
from src.container import scoped
from src.config import Settings

@scoped
class StudyBuddyAgent:
    """AI Study Buddy and Exam Master"""
    
    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings
        self.irt_params = self.load_irt_parameters()
        self.question_bank = self.load_question_bank()
        
    def load_irt_parameters(self) -> Dict:
        """Item Response Theory parameters"""
        return {
            'discrimination': 1.0,
            'guessing': 0.25,  # 4 choices
            'max_info': 2.5
        }
    
    def load_question_bank(self) -> Dict:
        """Question templates"""
        return {
            'Matematik': {
                'easy': ["Basic math question 1", "Basic math question 2"],
                'medium': ["Medium math question 1", "Medium math question 2"],
                'hard': ["Hard math question 1", "Hard math question 2"]
            },
            'Fizik': {
                'easy': ["Basic physics question 1", "Basic physics question 2"],
                'medium': ["Medium physics question 1", "Medium physics question 2"],
                'hard': ["Hard physics question 1", "Hard physics question 2"]
            }
        }
    
    def calculate_irt_probability(self, ability: float, difficulty: float) -> float:
        """Calculate success probability using IRT"""
        a = self.irt_params['discrimination']
        c = self.irt_params['guessing']
        
        exponent = a * (ability - difficulty)
        probability = c + (1 - c) / (1 + math.exp(-exponent))
        
        return probability
    
    def select_next_difficulty(self, current_ability: float, 
                             performance_history: List[bool] = None) -> float:
        """Adaptive difficulty selection"""
        
        if performance_history and len(performance_history) >= 3:
            recent_performance = sum(performance_history[-3:]) / 3
            
            if recent_performance > 0.8:
                return min(current_ability + 0.3, 1.0)
            elif recent_performance < 0.4:
                return max(current_ability - 0.2, 0.1)
        
        variation = random.uniform(-0.1, 0.1)
        optimal_difficulty = current_ability + variation
        
        return max(0.1, min(1.0, optimal_difficulty))
    
    def generate_adaptive_quiz(
        self, 
        topic: str, 
        student_ability: float = 0.5,
        num_questions: int = 10
    ) -> List[Dict]:
        """Generate adaptive quiz using IRT"""
        
        questions = []
        current_ability = student_ability
        performance_history = []
        
        topic_questions = self.question_bank.get(topic, self.question_bank['Matematik'])
        
        for i in range(num_questions):
            optimal_difficulty = self.select_next_difficulty(
                current_ability, 
                performance_history
            )
            
            if optimal_difficulty < 0.4:
                question_pool = topic_questions.get('easy', [])
                level = 'Easy'
            elif optimal_difficulty < 0.7:
                question_pool = topic_questions.get('medium', [])
                level = 'Medium'
            else:
                question_pool = topic_questions.get('hard', [])
                level = 'Hard'
            
            question_text = random.choice(question_pool) if question_pool else f"{topic} Question {i+1}"
            
            question = {
                'id': f'q_{i+1}',
                'number': i + 1,
                'text': question_text,
                'difficulty': round(optimal_difficulty, 2),
                'level': level,
                'options': ["Option A", "Option B", "Option C", "Option D"],
                'correct_answer': random.randint(0, 3),
                'success_probability': round(
                    self.calculate_irt_probability(current_ability, optimal_difficulty), 2
                )
            }
            
            questions.append(question)
            
            is_correct = random.random() < question['success_probability']
            performance_history.append(is_correct)
            
            if is_correct:
                current_ability = min(current_ability + 0.1, 1.0)
            else:
                current_ability = max(current_ability - 0.05, 0.0)
        
        return questions
    
    def generate_study_plan(self, weak_topics: List[str], 
                           available_hours: int) -> Dict:
        """Generate study plan for weak topics"""
        
        if not weak_topics:
            return {'message': 'Great! You are proficient in all topics!'}
        
        hours_per_topic = available_hours / len(weak_topics)
        study_plan = {
            'total_hours': available_hours,
            'start_date': datetime.now().strftime('%Y-%m-%d'),
            'end_date': (datetime.now() + timedelta(days=7)).strftime('%Y-%m-%d'),
            'topics': []
        }
        
        for i, topic in enumerate(weak_topics):
            topic_plan = {
                'topic': topic,
                'allocated_hours': round(hours_per_topic, 1),
                'priority': i + 1,
                'activities': [
                    f"{topic} fundamentals review",
                    f"{topic} practice problems",
                    f"{topic} mock tests"
                ],
                'resources': [
                    f"EBA {topic} videos",
                    f"MEB {topic} notes",
                    f"Khan Academy {topic} exercises"
                ]
            }
            study_plan['topics'].append(topic_plan)
        
        return study_plan
    
    def evaluate_quiz_performance(self, answers: List[Dict]) -> Dict:
        """Evaluate quiz performance"""
        
        correct_count = sum(1 for a in answers if a.get('is_correct', False))
        total_count = len(answers)
        score = (correct_count / total_count * 100) if total_count > 0 else 0
        
        if score >= 80:
            message = "Excellent performance!"
            recommendation = "You can move on to more challenging topics."
        elif score >= 60:
            message = "Good progress!"
            recommendation = "Keep practicing to achieve mastery."
        else:
            message = "Keep trying!"
            recommendation = "Review the fundamentals and practice more."
        
        return {
            'score': round(score, 1),
            'correct_answers': correct_count,
            'total_questions': total_count,
            'message': message,
            'recommendation': recommendation
        }


# Test code
if __name__ == "__main__":
    import json
    
    buddy = StudyBuddyAgent()
    
    # Generate adaptive quiz
    print("=== ADAPTIVE QUIZ ===\n")
    quiz = buddy.generate_adaptive_quiz(
        topic="Matematik",
        student_ability=0.5,
        num_questions=5
    )
    
    for q in quiz:
        print(f"Question {q['number']}:")
        print(f"  Text: {q['text']}")
        print(f"  Difficulty: {q['difficulty']} ({q['level']})")
        print(f"  Success Probability: {q['success_probability']*100:.0f}%")
        print()
    
    # Generate study plan
    print("\n=== STUDY PLAN ===\n")
    weak_topics = ["Equations", "Geometry", "Trigonometry"]
    study_plan = buddy.generate_study_plan(weak_topics, available_hours=12)
    print(json.dumps(study_plan, indent=2))
    
    # Evaluate performance
    print("\n=== PERFORMANCE EVALUATION ===\n")
    simulated_answers = [
        {'is_correct': True},
        {'is_correct': True},
        {'is_correct': False},
        {'is_correct': True},
        {'is_correct': False}
    ]
    
    evaluation = buddy.evaluate_quiz_performance(simulated_answers)
    print(json.dumps(evaluation, indent=2))