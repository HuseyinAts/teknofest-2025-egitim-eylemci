"""
Domain Entities
TEKNOFEST 2025 - Core Business Entities
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict
from datetime import datetime
import uuid

from src.domain.value_objects import (
    LearningStyle,
    Grade,
    AbilityLevel,
    QuizScore,
    LearningProgress
)
from src.shared.constants import StudentStatus
from src.shared.exceptions import DomainError


@dataclass
class Student:
    """Student entity - Core domain model"""
    id: str
    email: str
    username: str
    full_name: str
    grade: Grade
    learning_style: Optional[LearningStyle] = None
    ability_level: AbilityLevel = field(default_factory=lambda: AbilityLevel(0.5))
    status: StudentStatus = StudentStatus.ACTIVE
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    
    def __post_init__(self):
        """Validate student data"""
        if not self.email or '@' not in self.email:
            raise DomainError("Invalid email address")
        
        if not self.username or len(self.username) < 3:
            raise DomainError("Username must be at least 3 characters")
    
    @classmethod
    def create_new(
        cls,
        email: str,
        username: str,
        full_name: str,
        grade: int
    ) -> 'Student':
        """Factory method to create a new student"""
        return cls(
            id=str(uuid.uuid4()),
            email=email.lower(),
            username=username.lower(),
            full_name=full_name,
            grade=Grade(grade),
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
    
    def update_learning_style(self, learning_style: LearningStyle) -> None:
        """Update student's learning style"""
        self.learning_style = learning_style
        self.updated_at = datetime.utcnow()
    
    def update_ability_level(self, new_level: AbilityLevel) -> None:
        """Update student's ability level"""
        self.ability_level = new_level
        self.updated_at = datetime.utcnow()
    
    def promote_grade(self) -> None:
        """Promote student to next grade"""
        if self.grade.level >= 12:
            raise DomainError("Student is already in the highest grade")
        
        self.grade = Grade(self.grade.level + 1)
        self.updated_at = datetime.utcnow()
    
    def deactivate(self) -> None:
        """Deactivate student account"""
        self.status = StudentStatus.INACTIVE
        self.updated_at = datetime.utcnow()
    
    def activate(self) -> None:
        """Activate student account"""
        self.status = StudentStatus.ACTIVE
        self.updated_at = datetime.utcnow()
    
    @property
    def is_active(self) -> bool:
        """Check if student is active"""
        return self.status == StudentStatus.ACTIVE
    
    @property
    def needs_learning_style_assessment(self) -> bool:
        """Check if student needs learning style assessment"""
        return self.learning_style is None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            "id": self.id,
            "email": self.email,
            "username": self.username,
            "full_name": self.full_name,
            "grade": self.grade.level,
            "learning_style": self.learning_style.to_dict() if self.learning_style else None,
            "ability_level": self.ability_level.value,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }


@dataclass
class LearningPath:
    """Learning path entity - Personalized learning journey"""
    id: str
    student_id: str
    grade: Grade
    modules: List['LearningModule']
    learning_style: LearningStyle
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    
    @classmethod
    def create_for_student(
        cls,
        student: Student,
        modules: List['LearningModule']
    ) -> 'LearningPath':
        """Create learning path for a student"""
        if not student.learning_style:
            raise DomainError("Student must have learning style assessed before creating path")
        
        return cls(
            id=str(uuid.uuid4()),
            student_id=student.id,
            grade=student.grade,
            modules=modules,
            learning_style=student.learning_style
        )
    
    def add_module(self, module: 'LearningModule') -> None:
        """Add a module to the learning path"""
        self.modules.append(module)
        self.updated_at = datetime.utcnow()
    
    def remove_module(self, module_id: str) -> None:
        """Remove a module from the learning path"""
        self.modules = [m for m in self.modules if m.id != module_id]
        self.updated_at = datetime.utcnow()
    
    def reorder_modules(self, new_order: List[str]) -> None:
        """Reorder modules based on new order of IDs"""
        module_map = {m.id: m for m in self.modules}
        self.modules = [module_map[mid] for mid in new_order if mid in module_map]
        self.updated_at = datetime.utcnow()
    
    @property
    def total_duration_hours(self) -> float:
        """Calculate total duration of all modules"""
        return sum(m.estimated_hours for m in self.modules)
    
    @property
    def completed_modules(self) -> List['LearningModule']:
        """Get list of completed modules"""
        return [m for m in self.modules if m.is_completed]
    
    @property
    def current_module(self) -> Optional['LearningModule']:
        """Get current module in progress"""
        for module in self.modules:
            if not module.is_completed:
                return module
        return None
    
    @property
    def progress(self) -> LearningProgress:
        """Get learning progress"""
        completed_count = len(self.completed_modules)
        total_count = len(self.modules)
        
        # Calculate current ability level based on completed modules
        if completed_count > 0:
            avg_score = sum(m.score for m in self.completed_modules if m.score) / completed_count
            current_level = AbilityLevel(avg_score / 100)
        else:
            current_level = AbilityLevel(0.5)
        
        # Calculate total time spent
        time_spent = sum(m.time_spent_minutes / 60 for m in self.modules if m.time_spent_minutes)
        
        # Get last activity
        activities = [m.completed_at for m in self.modules if m.completed_at]
        last_activity = max(activities) if activities else self.created_at
        
        return LearningProgress(
            completed_modules=completed_count,
            total_modules=total_count,
            current_level=current_level,
            time_spent_hours=time_spent,
            last_activity=last_activity
        )


@dataclass
class LearningModule:
    """Learning module entity - A unit of learning content"""
    id: str
    title: str
    subject: str
    topic: str
    content_type: str  # video, text, interactive, quiz
    estimated_hours: float
    difficulty_level: str
    prerequisites: List[str] = field(default_factory=list)
    is_completed: bool = False
    score: Optional[float] = None
    time_spent_minutes: int = 0
    completed_at: Optional[datetime] = None
    
    @classmethod
    def create_new(
        cls,
        title: str,
        subject: str,
        topic: str,
        content_type: str,
        estimated_hours: float,
        difficulty_level: str,
        prerequisites: Optional[List[str]] = None
    ) -> 'LearningModule':
        """Factory method to create new learning module"""
        return cls(
            id=str(uuid.uuid4()),
            title=title,
            subject=subject,
            topic=topic,
            content_type=content_type,
            estimated_hours=estimated_hours,
            difficulty_level=difficulty_level,
            prerequisites=prerequisites or []
        )
    
    def mark_completed(self, score: float, time_spent_minutes: int) -> None:
        """Mark module as completed"""
        if not 0 <= score <= 100:
            raise DomainError("Score must be between 0 and 100")
        
        self.is_completed = True
        self.score = score
        self.time_spent_minutes = time_spent_minutes
        self.completed_at = datetime.utcnow()
    
    def reset_progress(self) -> None:
        """Reset module progress"""
        self.is_completed = False
        self.score = None
        self.time_spent_minutes = 0
        self.completed_at = None
    
    @property
    def is_accessible(self, completed_module_ids: List[str]) -> bool:
        """Check if module is accessible based on prerequisites"""
        return all(prereq in completed_module_ids for prereq in self.prerequisites)
    
    @property
    def efficiency_ratio(self) -> Optional[float]:
        """Calculate learning efficiency (actual vs estimated time)"""
        if not self.is_completed or self.time_spent_minutes == 0:
            return None
        
        actual_hours = self.time_spent_minutes / 60
        return self.estimated_hours / actual_hours


@dataclass
class Curriculum:
    """Curriculum entity - Educational content structure"""
    id: str
    grade: Grade
    subjects: Dict[str, 'Subject']
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    
    @classmethod
    def create_for_grade(cls, grade: int, subjects: Dict[str, 'Subject']) -> 'Curriculum':
        """Create curriculum for a specific grade"""
        return cls(
            id=str(uuid.uuid4()),
            grade=Grade(grade),
            subjects=subjects
        )
    
    def get_subject(self, subject_name: str) -> Optional['Subject']:
        """Get a specific subject from curriculum"""
        return self.subjects.get(subject_name)
    
    def add_subject(self, subject: 'Subject') -> None:
        """Add a subject to curriculum"""
        self.subjects[subject.name] = subject
        self.updated_at = datetime.utcnow()
    
    @property
    def total_hours(self) -> int:
        """Calculate total hours for all subjects"""
        return sum(s.total_hours for s in self.subjects.values())
    
    @property
    def subject_list(self) -> List[str]:
        """Get list of subject names"""
        return list(self.subjects.keys())


@dataclass
class Subject:
    """Subject entity - A subject in the curriculum"""
    name: str
    topics: List[str]
    total_hours: int
    prerequisites: Dict[str, List[str]] = field(default_factory=dict)
    
    def has_topic(self, topic: str) -> bool:
        """Check if subject has a specific topic"""
        return topic in self.topics
    
    def get_prerequisites(self, topic: str) -> List[str]:
        """Get prerequisites for a topic"""
        return self.prerequisites.get(topic, [])
    
    @property
    def hours_per_topic(self) -> float:
        """Calculate average hours per topic"""
        if not self.topics:
            return 0
        return self.total_hours / len(self.topics)


@dataclass
class Quiz:
    """Quiz entity - Assessment tool"""
    id: str
    title: str
    topic: str
    questions: List['Question']
    difficulty: str
    time_limit_minutes: int
    passing_score: float = 60.0
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    @classmethod
    def create_adaptive(
        cls,
        topic: str,
        student_ability: AbilityLevel,
        num_questions: int = 10
    ) -> 'Quiz':
        """Create adaptive quiz based on student ability"""
        # Determine difficulty based on ability
        if student_ability.value < 0.3:
            difficulty = "easy"
        elif student_ability.value < 0.6:
            difficulty = "medium"
        elif student_ability.value < 0.8:
            difficulty = "hard"
        else:
            difficulty = "expert"
        
        return cls(
            id=str(uuid.uuid4()),
            title=f"{topic} - Adaptive Quiz",
            topic=topic,
            questions=[],  # Questions would be populated from question bank
            difficulty=difficulty,
            time_limit_minutes=num_questions * 2  # 2 minutes per question
        )
    
    def add_question(self, question: 'Question') -> None:
        """Add a question to the quiz"""
        self.questions.append(question)
    
    def evaluate_responses(self, responses: Dict[str, str]) -> QuizScore:
        """Evaluate quiz responses and return score"""
        correct_count = 0
        
        for question in self.questions:
            if responses.get(question.id) == question.correct_answer:
                correct_count += 1
        
        return QuizScore(
            correct_answers=correct_count,
            total_questions=len(self.questions),
            time_spent_seconds=0,  # Would be tracked separately
            difficulty=self.difficulty
        )
    
    @property
    def total_points(self) -> float:
        """Calculate total points for the quiz"""
        return sum(q.points for q in self.questions)


@dataclass
class Question:
    """Question entity - A single quiz question"""
    id: str
    text: str
    options: List[str]
    correct_answer: str
    points: float = 1.0
    difficulty: str = "medium"
    topic: str = ""
    explanation: Optional[str] = None
    
    def __post_init__(self):
        """Validate question data"""
        if self.correct_answer not in self.options:
            raise DomainError("Correct answer must be one of the options")
        
        if len(self.options) < 2:
            raise DomainError("Question must have at least 2 options")
        
        if self.points <= 0:
            raise DomainError("Question points must be positive")
    
    @classmethod
    def create_multiple_choice(
        cls,
        text: str,
        options: List[str],
        correct_answer: str,
        topic: str,
        difficulty: str = "medium",
        explanation: Optional[str] = None
    ) -> 'Question':
        """Create a multiple choice question"""
        return cls(
            id=str(uuid.uuid4()),
            text=text,
            options=options,
            correct_answer=correct_answer,
            topic=topic,
            difficulty=difficulty,
            explanation=explanation
        )
    
    def is_correct(self, answer: str) -> bool:
        """Check if an answer is correct"""
        return answer == self.correct_answer
