"""
Assessment generation tool for TEKNOFEST Education Assistant
"""

from typing import Dict, List, Optional
import random


class AssessmentTool:
    """Assessment generation and evaluation tool"""
    
    def __init__(self):
        self.assessments = {}
        self.results = {}
    
    def generate(self, topic: str, difficulty: str = 'medium', num_questions: int = 10) -> Dict:
        """Generate assessment for a specific topic"""
        return generate_assessment(topic, difficulty, num_questions)
    
    def evaluate(self, student_answers: List[Dict], assessment: Dict) -> Dict:
        """Evaluate student answers against assessment"""
        return evaluate_assessment(student_answers, assessment)
    
    def save_assessment(self, assessment_id: str, assessment: Dict):
        """Save assessment for later use"""
        self.assessments[assessment_id] = assessment
    
    def get_assessment(self, assessment_id: str) -> Optional[Dict]:
        """Retrieve saved assessment"""
        return self.assessments.get(assessment_id)
    
    def save_result(self, student_id: str, result: Dict):
        """Save evaluation result"""
        if student_id not in self.results:
            self.results[student_id] = []
        self.results[student_id].append(result)
    
    def get_student_results(self, student_id: str) -> List[Dict]:
        """Get all results for a student"""
        return self.results.get(student_id, [])


def generate_assessment(topic: str, difficulty: str = 'medium', num_questions: int = 10) -> Dict:
    """Generate assessment for a specific topic"""
    
    assessment = {
        'topic': topic,
        'difficulty': difficulty,
        'total_questions': num_questions,
        'total_points': num_questions * 10,
        'duration_minutes': num_questions * 3,
        'questions': [],
        'rubric': {},
        'passing_score': 60
    }
    
    # Define difficulty levels
    difficulty_levels = {
        'easy': {'min': 0.2, 'max': 0.4, 'points': 5},
        'medium': {'min': 0.4, 'max': 0.7, 'points': 10},
        'hard': {'min': 0.7, 'max': 0.9, 'points': 15}
    }
    
    level_config = difficulty_levels.get(difficulty, difficulty_levels['medium'])
    
    # Generate questions
    for i in range(num_questions):
        question_type = random.choice(['multiple_choice', 'true_false', 'short_answer'])
        
        question = {
            'id': f'Q{i+1}',
            'type': question_type,
            'text': f'{topic} Question {i+1}',
            'points': level_config['points'],
            'difficulty': random.uniform(level_config['min'], level_config['max']),
            'learning_objective': f'Assess {topic} understanding',
            'estimated_time': 3
        }
        
        if question_type == 'multiple_choice':
            question['options'] = ['Option A', 'Option B', 'Option C', 'Option D']
            question['correct_answer'] = random.randint(0, 3)
        elif question_type == 'true_false':
            question['correct_answer'] = random.choice([True, False])
        else:
            question['expected_keywords'] = [topic.lower(), 'answer', 'solution']
        
        assessment['questions'].append(question)
    
    # Generate rubric
    assessment['rubric'] = {
        'excellent': {'range': '90-100', 'description': 'Outstanding understanding'},
        'good': {'range': '75-89', 'description': 'Good grasp of concepts'},
        'satisfactory': {'range': '60-74', 'description': 'Adequate understanding'},
        'needs_improvement': {'range': '0-59', 'description': 'Requires additional study'}
    }
    
    return assessment


def evaluate_assessment(student_answers: List[Dict], assessment: Dict) -> Dict:
    """Evaluate student answers against assessment"""
    
    total_score = 0
    correct_count = 0
    results = []
    
    for i, answer in enumerate(student_answers):
        question = assessment['questions'][i] if i < len(assessment['questions']) else None
        
        if not question:
            continue
        
        is_correct = False
        if question['type'] == 'multiple_choice':
            is_correct = answer.get('answer') == question['correct_answer']
        elif question['type'] == 'true_false':
            is_correct = answer.get('answer') == question['correct_answer']
        else:
            # For short answer, check if keywords are present
            answer_text = str(answer.get('answer', '')).lower()
            keywords_found = sum(1 for kw in question.get('expected_keywords', []) 
                                if kw in answer_text)
            is_correct = keywords_found >= len(question.get('expected_keywords', [])) / 2
        
        score = question['points'] if is_correct else 0
        total_score += score
        if is_correct:
            correct_count += 1
        
        results.append({
            'question_id': question['id'],
            'correct': is_correct,
            'score': score,
            'max_score': question['points']
        })
    
    percentage = (total_score / assessment['total_points']) * 100 if assessment['total_points'] > 0 else 0
    
    return {
        'total_score': total_score,
        'max_score': assessment['total_points'],
        'percentage': round(percentage, 2),
        'correct_answers': correct_count,
        'total_questions': assessment['total_questions'],
        'passed': percentage >= assessment['passing_score'],
        'grade': _get_grade(percentage),
        'detailed_results': results
    }


def _get_grade(percentage: float) -> str:
    """Get letter grade from percentage"""
    if percentage >= 90:
        return 'A'
    elif percentage >= 80:
        return 'B'
    elif percentage >= 70:
        return 'C'
    elif percentage >= 60:
        return 'D'
    else:
        return 'F'