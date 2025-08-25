"""
Value Objects for Domain Layer
TEKNOFEST 2025 - Immutable Domain Values
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, List
from datetime import datetime
from enum import Enum

from src.shared.constants import LearningStyles, QuizDifficulty
from src.shared.exceptions import ValidationError


@dataclass(frozen=True)
class LearningStyle:
    """Immutable value object representing a learning style"""
    primary_style: LearningStyles
    secondary_style: Optional[LearningStyles]
    confidence: float
    scores: Dict[str, float] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate learning style data"""
        if not 0 <= self.confidence <= 1:
            raise ValidationError(
                "Confidence must be between 0 and 1",
                field="confidence",
                details={"value": self.confidence}
            )
        
        if self.primary_style == self.secondary_style:
            raise ValidationError(
                "Primary and secondary styles must be different",
                field="secondary_style"
            )
    
    @property
    def is_strong(self) -> bool:
        """Check if the learning style indication is strong"""
        return self.confidence >= 0.7
    
    @property
    def is_mixed(self) -> bool:
        """Check if multiple learning styles are equally strong"""
        return self.confidence < 0.5
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            "primary_style": self.primary_style.value,
            "secondary_style": self.secondary_style.value if self.secondary_style else None,
            "confidence": self.confidence,
            "scores": self.scores
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'LearningStyle':
        """Create from dictionary"""
        return cls(
            primary_style=LearningStyles(data["primary_style"]),
            secondary_style=LearningStyles(data["secondary_style"]) if data.get("secondary_style") else None,
            confidence=data["confidence"],
            scores=data.get("scores", {})
        )


@dataclass(frozen=True)
class Grade:
    """Value object representing a student grade level"""
    level: int
    
    def __post_init__(self):
        """Validate grade level"""
        if not 9 <= self.level <= 12:
            raise ValidationError(
                f"Grade level must be between 9 and 12, got {self.level}",
                field="level"
            )
    
    @property
    def display_name(self) -> str:
        """Get display name for the grade"""
        return f"{self.level}. Sınıf"
    
    @property
    def curriculum_key(self) -> str:
        """Get curriculum key for this grade"""
        return str(self.level)
    
    def __str__(self) -> str:
        return self.display_name


@dataclass(frozen=True)
class StudentResponse:
    """Value object for a student's response"""
    content: str
    timestamp: datetime
    question_id: Optional[str] = None
    
    def __post_init__(self):
        """Validate response content"""
        if not self.content or len(self.content.strip()) < 10:
            raise ValidationError(
                "Response content must be at least 10 characters",
                field="content"
            )
        
        if len(self.content) > 1000:
            raise ValidationError(
                "Response content must not exceed 1000 characters",
                field="content"
            )
    
    @property
    def word_count(self) -> int:
        """Get word count of the response"""
        return len(self.content.split())
    
    @property
    def is_recent(self) -> bool:
        """Check if response is recent (within last 24 hours)"""
        time_diff = datetime.utcnow() - self.timestamp
        return time_diff.total_seconds() < 86400  # 24 hours
    
    def extract_keywords(self) -> List[str]:
        """Extract keywords from response content"""
        # Simple keyword extraction (can be enhanced with NLP)
        words = self.content.lower().split()
        # Filter out common words
        stopwords = {'ve', 'ile', 'bir', 'bu', 'şu', 'o', 'ben', 'sen'}
        return [w for w in words if w not in stopwords and len(w) > 3]


@dataclass(frozen=True)
class AbilityLevel:
    """Value object representing student ability level"""
    value: float
    
    def __post_init__(self):
        """Validate ability level"""
        if not 0.0 <= self.value <= 1.0:
            raise ValidationError(
                f"Ability level must be between 0.0 and 1.0, got {self.value}",
                field="value"
            )
    
    @property
    def category(self) -> str:
        """Get ability category"""
        if self.value < 0.3:
            return "Beginner"
        elif self.value < 0.6:
            return "Intermediate"
        elif self.value < 0.8:
            return "Advanced"
        else:
            return "Expert"
    
    @property
    def percentage(self) -> float:
        """Get ability as percentage"""
        return self.value * 100
    
    def adjust(self, delta: float) -> 'AbilityLevel':
        """Create new ability level with adjustment"""
        new_value = max(0.0, min(1.0, self.value + delta))
        return AbilityLevel(new_value)


@dataclass(frozen=True)
class QuizScore:
    """Value object for quiz score"""
    correct_answers: int
    total_questions: int
    time_spent_seconds: int
    difficulty: QuizDifficulty
    
    def __post_init__(self):
        """Validate quiz score data"""
        if self.correct_answers > self.total_questions:
            raise ValidationError(
                "Correct answers cannot exceed total questions",
                field="correct_answers"
            )
        
        if self.total_questions <= 0:
            raise ValidationError(
                "Total questions must be positive",
                field="total_questions"
            )
        
        if self.time_spent_seconds < 0:
            raise ValidationError(
                "Time spent cannot be negative",
                field="time_spent_seconds"
            )
    
    @property
    def percentage(self) -> float:
        """Get score as percentage"""
        return (self.correct_answers / self.total_questions) * 100
    
    @property
    def is_passing(self) -> bool:
        """Check if score is passing (>= 60%)"""
        return self.percentage >= 60
    
    @property
    def average_time_per_question(self) -> float:
        """Get average time spent per question"""
        return self.time_spent_seconds / self.total_questions
    
    @property
    def performance_rating(self) -> str:
        """Get performance rating based on score and difficulty"""
        score_factor = self.percentage / 100
        difficulty_factor = {
            QuizDifficulty.EASY: 0.7,
            QuizDifficulty.MEDIUM: 1.0,
            QuizDifficulty.HARD: 1.3,
            QuizDifficulty.EXPERT: 1.5
        }[self.difficulty]
        
        adjusted_score = score_factor * difficulty_factor
        
        if adjusted_score >= 1.2:
            return "Excellent"
        elif adjusted_score >= 0.9:
            return "Good"
        elif adjusted_score >= 0.6:
            return "Satisfactory"
        else:
            return "Needs Improvement"


@dataclass(frozen=True)
class LearningProgress:
    """Value object for learning progress tracking"""
    completed_modules: int
    total_modules: int
    current_level: AbilityLevel
    time_spent_hours: float
    last_activity: datetime
    
    @property
    def completion_percentage(self) -> float:
        """Get completion percentage"""
        if self.total_modules == 0:
            return 0.0
        return (self.completed_modules / self.total_modules) * 100
    
    @property
    def is_active(self) -> bool:
        """Check if student is actively learning (active in last 7 days)"""
        days_inactive = (datetime.utcnow() - self.last_activity).days
        return days_inactive < 7
    
    @property
    def estimated_completion_days(self) -> Optional[int]:
        """Estimate days to completion based on current pace"""
        if self.completed_modules == 0 or self.time_spent_hours == 0:
            return None
        
        modules_per_hour = self.completed_modules / self.time_spent_hours
        remaining_modules = self.total_modules - self.completed_modules
        remaining_hours = remaining_modules / modules_per_hour
        
        # Assume 2 hours of study per day
        return int(remaining_hours / 2)
