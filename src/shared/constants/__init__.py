"""
Application Constants
TEKNOFEST 2025 - Clean Code Implementation
"""

from enum import Enum


class AppConstants:
    """Application-wide constants"""
    API_VERSION = "v1"
    API_PREFIX = "/api/v1"
    APP_NAME = "TEKNOFEST 2025 - Eğitim Teknolojileri"
    DEFAULT_PAGE_SIZE = 50
    MAX_PAGE_SIZE = 100
    MIN_PASSWORD_LENGTH = 8
    MAX_LOGIN_ATTEMPTS = 5
    SESSION_TIMEOUT_HOURS = 24
    JWT_EXPIRATION_MINUTES = 30


class EducationConstants:
    """Education domain constants"""
    MAX_LEARNING_WEEKS = 52
    MIN_CONFIDENCE_THRESHOLD = 0.7
    DEFAULT_QUIZ_QUESTIONS = 10
    MAX_QUIZ_QUESTIONS = 50
    MIN_QUIZ_QUESTIONS = 5
    
    # Grade levels
    MIN_GRADE = 9
    MAX_GRADE = 12
    
    # Learning metrics
    MIN_ABILITY_LEVEL = 0.0
    MAX_ABILITY_LEVEL = 1.0
    DEFAULT_ABILITY_LEVEL = 0.5
    
    # Response validation
    MIN_RESPONSE_LENGTH = 10
    MAX_RESPONSE_LENGTH = 1000
    MIN_RESPONSES_FOR_ANALYSIS = 1
    MAX_RESPONSES_FOR_ANALYSIS = 100


class LearningStyles(str, Enum):
    """Learning style types enumeration"""
    VISUAL = "visual"
    AUDITORY = "auditory"
    READING = "reading"
    KINESTHETIC = "kinesthetic"


class QuizDifficulty(str, Enum):
    """Quiz difficulty levels"""
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    EXPERT = "expert"


class StudentStatus(str, Enum):
    """Student account status"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    SUSPENDED = "suspended"
    GRADUATED = "graduated"
