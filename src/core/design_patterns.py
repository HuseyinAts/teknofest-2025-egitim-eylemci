"""
Design Patterns Implementation
TEKNOFEST 2025 - Clean Architecture Patterns
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type, Callable
from enum import Enum
import logging
from datetime import datetime

from src.core.constants import (
    LearningStyle, AssessmentType, QuestionDifficulty,
    LEARNING_RESOURCES, ASSESSMENT_STRATEGIES
)

logger = logging.getLogger(__name__)


# ==========================================
# STRATEGY PATTERN
# ==========================================

class LearningStrategy(ABC):
    """Abstract base for learning strategies"""
    
    @abstractmethod
    def get_resources(self) -> List[str]:
        """Get learning resources for this strategy"""
        pass
    
    @abstractmethod
    def get_assessment_type(self) -> str:
        """Get assessment type for this strategy"""
        pass
    
    @abstractmethod
    def calculate_difficulty(self, student_level: float) -> float:
        """Calculate appropriate difficulty for student"""
        pass
    
    @abstractmethod
    def generate_content(self, topic: str) -> Dict[str, Any]:
        """Generate learning content"""
        pass


class VisualLearningStrategy(LearningStrategy):
    """Strategy for visual learners"""
    
    def get_resources(self) -> List[str]:
        return LEARNING_RESOURCES[LearningStyle.VISUAL]
    
    def get_assessment_type(self) -> str:
        return ASSESSMENT_STRATEGIES[LearningStyle.VISUAL]
    
    def calculate_difficulty(self, student_level: float) -> float:
        # Visual learners benefit from slightly easier start
        return min(student_level + 0.1, 1.0)
    
    def generate_content(self, topic: str) -> Dict[str, Any]:
        return {
            'type': 'visual',
            'topic': topic,
            'content': {
                'videos': [f'{topic} video tutorial'],
                'infographics': [f'{topic} infographic'],
                'diagrams': [f'{topic} flow chart']
            },
            'interactions': ['zoom', 'pan', 'annotate']
        }


class AuditoryLearningStrategy(LearningStrategy):
    """Strategy for auditory learners"""
    
    def get_resources(self) -> List[str]:
        return LEARNING_RESOURCES[LearningStyle.AUDITORY]
    
    def get_assessment_type(self) -> str:
        return ASSESSMENT_STRATEGIES[LearningStyle.AUDITORY]
    
    def calculate_difficulty(self, student_level: float) -> float:
        # Standard difficulty progression
        return student_level + 0.15
    
    def generate_content(self, topic: str) -> Dict[str, Any]:
        return {
            'type': 'auditory',
            'topic': topic,
            'content': {
                'podcasts': [f'{topic} discussion'],
                'audio_lessons': [f'{topic} lecture'],
                'discussions': ['group_chat', 'voice_notes']
            },
            'interactions': ['play', 'pause', 'speed_control']
        }


class ReadingLearningStrategy(LearningStrategy):
    """Strategy for reading/writing learners"""
    
    def get_resources(self) -> List[str]:
        return LEARNING_RESOURCES[LearningStyle.READING]
    
    def get_assessment_type(self) -> str:
        return ASSESSMENT_STRATEGIES[LearningStyle.READING]
    
    def calculate_difficulty(self, student_level: float) -> float:
        # Reading learners can handle slightly harder content
        return min(student_level + 0.2, 1.0)
    
    def generate_content(self, topic: str) -> Dict[str, Any]:
        return {
            'type': 'reading',
            'topic': topic,
            'content': {
                'articles': [f'{topic} detailed guide'],
                'ebooks': [f'{topic} textbook'],
                'notes': [f'{topic} summary']
            },
            'interactions': ['highlight', 'annotate', 'bookmark']
        }


class KinestheticLearningStrategy(LearningStrategy):
    """Strategy for kinesthetic learners"""
    
    def get_resources(self) -> List[str]:
        return LEARNING_RESOURCES[LearningStyle.KINESTHETIC]
    
    def get_assessment_type(self) -> str:
        return ASSESSMENT_STRATEGIES[LearningStyle.KINESTHETIC]
    
    def calculate_difficulty(self, student_level: float) -> float:
        # Gradual progression for hands-on learning
        return student_level + 0.12
    
    def generate_content(self, topic: str) -> Dict[str, Any]:
        return {
            'type': 'kinesthetic',
            'topic': topic,
            'content': {
                'simulations': [f'{topic} interactive sim'],
                'experiments': [f'{topic} lab activity'],
                'projects': [f'{topic} hands-on project']
            },
            'interactions': ['drag_drop', 'build', 'experiment']
        }


class LearningStrategyContext:
    """Context for learning strategy pattern"""
    
    def __init__(self):
        self._strategies = {
            LearningStyle.VISUAL: VisualLearningStrategy(),
            LearningStyle.AUDITORY: AuditoryLearningStrategy(),
            LearningStyle.READING: ReadingLearningStrategy(),
            LearningStyle.KINESTHETIC: KinestheticLearningStrategy()
        }
        self._current_strategy = None
    
    def set_strategy(self, learning_style: LearningStyle):
        """Set the current learning strategy"""
        self._current_strategy = self._strategies.get(
            learning_style, 
            VisualLearningStrategy()  # Default
        )
    
    def execute_strategy(self, topic: str, student_level: float) -> Dict[str, Any]:
        """Execute the current strategy"""
        if not self._current_strategy:
            raise ValueError("No strategy set")
        
        return {
            'resources': self._current_strategy.get_resources(),
            'assessment': self._current_strategy.get_assessment_type(),
            'difficulty': self._current_strategy.calculate_difficulty(student_level),
            'content': self._current_strategy.generate_content(topic)
        }


# ==========================================
# FACTORY PATTERN
# ==========================================

class QuizFactory:
    """Factory for creating different types of quizzes"""
    
    @staticmethod
    def create_quiz(
        quiz_type: AssessmentType,
        topic: str,
        difficulty: QuestionDifficulty,
        num_questions: int
    ) -> 'BaseQuiz':
        """Create appropriate quiz instance"""
        
        quiz_map = {
            AssessmentType.QUIZ: StandardQuiz,
            AssessmentType.EXAM: ExamQuiz,
            AssessmentType.HOMEWORK: HomeworkQuiz,
            AssessmentType.PROJECT: ProjectQuiz
        }
        
        quiz_class = quiz_map.get(quiz_type, StandardQuiz)
        return quiz_class(topic, difficulty, num_questions)


class BaseQuiz(ABC):
    """Abstract base for all quiz types"""
    
    def __init__(self, topic: str, difficulty: QuestionDifficulty, num_questions: int):
        self.topic = topic
        self.difficulty = difficulty
        self.num_questions = num_questions
        self.created_at = datetime.utcnow()
    
    @abstractmethod
    def generate_questions(self) -> List[Dict]:
        """Generate quiz questions"""
        pass
    
    @abstractmethod
    def calculate_score(self, answers: List) -> float:
        """Calculate quiz score"""
        pass
    
    @abstractmethod
    def get_time_limit(self) -> int:
        """Get time limit in seconds"""
        pass


class StandardQuiz(BaseQuiz):
    """Standard quiz implementation"""
    
    def generate_questions(self) -> List[Dict]:
        questions = []
        for i in range(self.num_questions):
            questions.append({
                'id': i + 1,
                'text': f'{self.topic} - Question {i + 1}',
                'type': 'multiple_choice',
                'difficulty': self.difficulty.value,
                'options': ['A', 'B', 'C', 'D'],
                'correct': 'A'
            })
        return questions
    
    def calculate_score(self, answers: List) -> float:
        correct = sum(1 for a in answers if a['is_correct'])
        return (correct / self.num_questions) * 100
    
    def get_time_limit(self) -> int:
        return self.num_questions * 60  # 1 minute per question


class ExamQuiz(BaseQuiz):
    """Exam quiz with stricter rules"""
    
    def generate_questions(self) -> List[Dict]:
        # More complex question generation for exams
        questions = []
        for i in range(self.num_questions):
            questions.append({
                'id': i + 1,
                'text': f'{self.topic} - Exam Question {i + 1}',
                'type': 'mixed',  # Multiple types
                'difficulty': min(self.difficulty.value + 1, 5),
                'weight': 2 if i < 5 else 1,  # First 5 questions worth more
                'options': ['A', 'B', 'C', 'D', 'E'],
                'correct': 'A'
            })
        return questions
    
    def calculate_score(self, answers: List) -> float:
        total_weight = sum(q.get('weight', 1) for q in answers)
        weighted_correct = sum(
            q.get('weight', 1) for q in answers 
            if q.get('is_correct')
        )
        return (weighted_correct / total_weight) * 100
    
    def get_time_limit(self) -> int:
        return self.num_questions * 90  # 1.5 minutes per question


class HomeworkQuiz(BaseQuiz):
    """Homework quiz with extended time"""
    
    def generate_questions(self) -> List[Dict]:
        questions = []
        for i in range(self.num_questions):
            questions.append({
                'id': i + 1,
                'text': f'{self.topic} - Homework {i + 1}',
                'type': 'open_ended',
                'difficulty': self.difficulty.value,
                'hints_available': True,
                'partial_credit': True
            })
        return questions
    
    def calculate_score(self, answers: List) -> float:
        # Support partial credit
        total_score = sum(a.get('score', 0) for a in answers)
        max_score = self.num_questions * 100
        return (total_score / max_score) * 100
    
    def get_time_limit(self) -> int:
        return 86400  # 24 hours


class ProjectQuiz(BaseQuiz):
    """Project-based assessment"""
    
    def generate_questions(self) -> List[Dict]:
        return [{
            'id': 1,
            'text': f'{self.topic} - Project Requirements',
            'type': 'project',
            'difficulty': self.difficulty.value,
            'deliverables': [
                'Research report',
                'Implementation',
                'Presentation'
            ],
            'rubric': self._generate_rubric()
        }]
    
    def _generate_rubric(self) -> Dict:
        return {
            'research': 30,
            'implementation': 40,
            'presentation': 20,
            'creativity': 10
        }
    
    def calculate_score(self, answers: List) -> float:
        # Rubric-based scoring
        rubric = self._generate_rubric()
        total = sum(
            answers[0].get(criterion, 0) * weight / 100
            for criterion, weight in rubric.items()
        )
        return total
    
    def get_time_limit(self) -> int:
        return 604800  # 1 week


# ==========================================
# OBSERVER PATTERN
# ==========================================

class Subject(ABC):
    """Subject in observer pattern"""
    
    def __init__(self):
        self._observers: List[Observer] = []
        self._state = None
    
    def attach(self, observer: 'Observer'):
        """Attach an observer"""
        if observer not in self._observers:
            self._observers.append(observer)
            logger.debug(f"Observer {observer.__class__.__name__} attached")
    
    def detach(self, observer: 'Observer'):
        """Detach an observer"""
        if observer in self._observers:
            self._observers.remove(observer)
            logger.debug(f"Observer {observer.__class__.__name__} detached")
    
    def notify(self):
        """Notify all observers"""
        for observer in self._observers:
            observer.update(self)


class Observer(ABC):
    """Observer in observer pattern"""
    
    @abstractmethod
    def update(self, subject: Subject):
        """Update method called by subject"""
        pass


class StudentProgress(Subject):
    """Student progress subject"""
    
    def __init__(self, student_id: str):
        super().__init__()
        self.student_id = student_id
        self.current_level = 0.5
        self.completed_topics = []
        self.achievements = []
    
    def update_progress(self, topic: str, score: float):
        """Update student progress"""
        self.completed_topics.append(topic)
        
        # Update level based on score
        if score >= 90:
            self.current_level = min(self.current_level + 0.1, 1.0)
        elif score >= 75:
            self.current_level = min(self.current_level + 0.05, 1.0)
        
        # Check for achievements
        if len(self.completed_topics) == 10:
            self.achievements.append("First 10 Topics!")
        
        # Notify observers
        self.notify()


class ProgressLogger(Observer):
    """Observer that logs progress"""
    
    def update(self, subject: StudentProgress):
        logger.info(
            f"Student {subject.student_id} progress updated: "
            f"Level={subject.current_level:.2f}, "
            f"Topics={len(subject.completed_topics)}"
        )


class AchievementNotifier(Observer):
    """Observer that sends achievement notifications"""
    
    def update(self, subject: StudentProgress):
        if subject.achievements:
            latest = subject.achievements[-1]
            logger.info(f"Achievement unlocked for {subject.student_id}: {latest}")
            # Send notification (email, push, etc.)
            self._send_notification(subject.student_id, latest)
    
    def _send_notification(self, student_id: str, achievement: str):
        # Notification logic here
        pass


class ParentNotifier(Observer):
    """Observer that notifies parents"""
    
    def update(self, subject: StudentProgress):
        if subject.current_level >= 0.8:
            logger.info(f"Notifying parent: {subject.student_id} excelling!")
            # Send parent notification
        elif subject.current_level < 0.3:
            logger.info(f"Notifying parent: {subject.student_id} needs help")
            # Send parent alert


# ==========================================
# SINGLETON PATTERN
# ==========================================

class ConfigurationManager:
    """Singleton configuration manager"""
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self.config = {}
            self.load_configuration()
            ConfigurationManager._initialized = True
    
    def load_configuration(self):
        """Load configuration"""
        self.config = {
            'app_name': 'TEKNOFEST 2025',
            'version': '3.0.0',
            'features': {},
            'limits': {}
        }
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value"""
        return self.config.get(key, default)
    
    def set(self, key: str, value: Any):
        """Set configuration value"""
        self.config[key] = value


# ==========================================
# BUILDER PATTERN
# ==========================================

class LearningPathBuilder:
    """Builder for complex learning path creation"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset builder to initial state"""
        self._learning_path = {
            'created_at': datetime.utcnow(),
            'milestones': [],
            'resources': [],
            'assessments': []
        }
        return self
    
    def set_student(self, student_id: str, grade: int):
        """Set student information"""
        self._learning_path['student_id'] = student_id
        self._learning_path['grade'] = grade
        return self
    
    def set_subject(self, subject: str):
        """Set subject"""
        self._learning_path['subject'] = subject
        return self
    
    def set_duration(self, weeks: int):
        """Set duration"""
        self._learning_path['duration_weeks'] = weeks
        return self
    
    def set_learning_style(self, style: LearningStyle):
        """Set learning style"""
        self._learning_path['learning_style'] = style.value
        return self
    
    def add_milestone(self, week: int, topic: str, goal: str):
        """Add a milestone"""
        self._learning_path['milestones'].append({
            'week': week,
            'topic': topic,
            'goal': goal,
            'completed': False
        })
        return self
    
    def add_resource(self, resource_type: str, url: str, title: str):
        """Add a learning resource"""
        self._learning_path['resources'].append({
            'type': resource_type,
            'url': url,
            'title': title
        })
        return self
    
    def add_assessment(self, assessment_type: AssessmentType, week: int):
        """Add an assessment"""
        self._learning_path['assessments'].append({
            'type': assessment_type.value,
            'week': week,
            'completed': False
        })
        return self
    
    def build(self) -> Dict[str, Any]:
        """Build the final learning path"""
        # Validate required fields
        required = ['student_id', 'subject', 'duration_weeks']
        for field in required:
            if field not in self._learning_path:
                raise ValueError(f"Missing required field: {field}")
        
        # Calculate total hours
        self._learning_path['total_hours'] = (
            self._learning_path['duration_weeks'] * 10
        )
        
        result = self._learning_path
        self.reset()  # Reset for next build
        return result


# ==========================================
# ADAPTER PATTERN
# ==========================================

class ExternalQuizSystem:
    """External quiz system with different interface"""
    
    def fetch_quiz_data(self, quiz_id: int) -> Dict:
        """Fetch quiz from external system"""
        return {
            'quiz_id': quiz_id,
            'title': 'External Quiz',
            'items': [
                {'q': 'Question 1', 'a': 'Answer 1'},
                {'q': 'Question 2', 'a': 'Answer 2'}
            ]
        }


class QuizAdapter:
    """Adapter for external quiz system"""
    
    def __init__(self, external_system: ExternalQuizSystem):
        self.external_system = external_system
    
    def get_quiz(self, quiz_id: int) -> BaseQuiz:
        """Adapt external quiz to our interface"""
        external_data = self.external_system.fetch_quiz_data(quiz_id)
        
        # Create our quiz format
        quiz = StandardQuiz(
            topic=external_data['title'],
            difficulty=QuestionDifficulty.MEDIUM,
            num_questions=len(external_data['items'])
        )
        
        # Adapt questions
        quiz.questions = [
            {
                'id': i + 1,
                'text': item['q'],
                'answer': item['a']
            }
            for i, item in enumerate(external_data['items'])
        ]
        
        return quiz
