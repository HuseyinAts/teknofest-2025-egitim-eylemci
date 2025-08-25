"""
Application Constants and Enums
TEKNOFEST 2025 - Clean Code Constants
"""

from enum import Enum, IntEnum
from typing import Final


# ==========================================
# NUMERIC CONSTANTS
# ==========================================

# Scoring and Thresholds
PASSING_SCORE_THRESHOLD: Final[int] = 75
EXCELLENT_SCORE_THRESHOLD: Final[int] = 90
MIN_PASSING_GRADE: Final[float] = 0.6
MAX_SCORE: Final[int] = 100

# Performance Limits
MAX_FUNCTION_LENGTH: Final[int] = 50
MAX_CLASS_LENGTH: Final[int] = 300
MAX_CYCLOMATIC_COMPLEXITY: Final[int] = 10
MAX_PARAMETERS: Final[int] = 5
MAX_NESTED_DEPTH: Final[int] = 3

# Cache Settings
DEFAULT_CACHE_TTL: Final[int] = 300  # 5 minutes
LONG_CACHE_TTL: Final[int] = 3600  # 1 hour
SHORT_CACHE_TTL: Final[int] = 60  # 1 minute
CACHE_KEY_MAX_LENGTH: Final[int] = 250

# Database Settings
DEFAULT_PAGE_SIZE: Final[int] = 20
MAX_PAGE_SIZE: Final[int] = 100
BATCH_SIZE: Final[int] = 1000
QUERY_TIMEOUT: Final[int] = 30  # seconds
CONNECTION_POOL_SIZE: Final[int] = 20
MAX_RETRY_ATTEMPTS: Final[int] = 3

# API Rate Limits
DEFAULT_RATE_LIMIT: Final[int] = 100
RATE_LIMIT_WINDOW: Final[int] = 3600  # 1 hour
MAX_LOGIN_ATTEMPTS: Final[int] = 5
LOCKOUT_DURATION: Final[int] = 900  # 15 minutes

# Learning Path Settings
DEFAULT_DURATION_WEEKS: Final[int] = 12
MIN_CONFIDENCE_SCORE: Final[float] = 0.7
ZPD_STEP_SIZE: Final[float] = 0.1
DEFAULT_STUDY_HOURS: Final[int] = 10

# Quiz Settings
MIN_QUESTIONS: Final[int] = 5
MAX_QUESTIONS: Final[int] = 50
DEFAULT_QUESTIONS: Final[int] = 10
QUIZ_TIME_LIMIT: Final[int] = 1800  # 30 minutes

# ==========================================
# STRING CONSTANTS
# ==========================================

# Error Messages
ERROR_INVALID_INPUT: Final[str] = "Geçersiz giriş değeri"
ERROR_NOT_FOUND: Final[str] = "Kayıt bulunamadı"
ERROR_UNAUTHORIZED: Final[str] = "Yetkisiz erişim"
ERROR_DATABASE: Final[str] = "Veritabanı hatası"
ERROR_CACHE: Final[str] = "Önbellek hatası"
ERROR_RATE_LIMIT: Final[str] = "İstek limiti aşıldı"

# Success Messages
SUCCESS_CREATED: Final[str] = "Başarıyla oluşturuldu"
SUCCESS_UPDATED: Final[str] = "Başarıyla güncellendi"
SUCCESS_DELETED: Final[str] = "Başarıyla silindi"
SUCCESS_COMPLETED: Final[str] = "İşlem tamamlandı"

# Date Formats
DATE_FORMAT: Final[str] = "%Y-%m-%d"
DATETIME_FORMAT: Final[str] = "%Y-%m-%d %H:%M:%S"
ISO_FORMAT: Final[str] = "%Y-%m-%dT%H:%M:%S.%fZ"

# ==========================================
# ENUMS
# ==========================================

class Environment(str, Enum):
    """Application environments"""
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"


class UserRole(str, Enum):
    """User roles for RBAC"""
    ADMIN = "admin"
    TEACHER = "teacher"
    STUDENT = "student"
    PARENT = "parent"
    GUEST = "guest"


class LearningStyle(str, Enum):
    """VARK learning styles"""
    VISUAL = "visual"
    AUDITORY = "auditory"
    READING = "reading"
    KINESTHETIC = "kinesthetic"
    MIXED = "mixed"


class QuestionDifficulty(IntEnum):
    """Question difficulty levels"""
    VERY_EASY = 1
    EASY = 2
    MEDIUM = 3
    HARD = 4
    VERY_HARD = 5


class TaskStatus(str, Enum):
    """Task/Assignment status"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class Grade(IntEnum):
    """Student grades"""
    GRADE_9 = 9
    GRADE_10 = 10
    GRADE_11 = 11
    GRADE_12 = 12


class Subject(str, Enum):
    """School subjects"""
    MATHEMATICS = "Matematik"
    PHYSICS = "Fizik"
    CHEMISTRY = "Kimya"
    BIOLOGY = "Biyoloji"
    TURKISH = "Türkçe"
    ENGLISH = "İngilizce"
    HISTORY = "Tarih"
    GEOGRAPHY = "Coğrafya"


class AssessmentType(str, Enum):
    """Assessment types"""
    QUIZ = "quiz"
    EXAM = "exam"
    HOMEWORK = "homework"
    PROJECT = "project"
    PRESENTATION = "presentation"


class CacheStrategy(str, Enum):
    """Cache strategies"""
    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    FIFO = "fifo"  # First In First Out
    TTL = "ttl"  # Time To Live


class HTTPMethod(str, Enum):
    """HTTP methods"""
    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    PATCH = "PATCH"
    DELETE = "DELETE"
    OPTIONS = "OPTIONS"
    HEAD = "HEAD"


class ResponseStatus(str, Enum):
    """API response status"""
    SUCCESS = "success"
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


class LogLevel(str, Enum):
    """Logging levels"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


# ==========================================
# VALIDATION CONSTANTS
# ==========================================

# Regex Patterns
EMAIL_REGEX: Final[str] = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
USERNAME_REGEX: Final[str] = r'^[a-zA-Z0-9_]{3,20}$'
PHONE_REGEX: Final[str] = r'^\+?[1-9]\d{1,14}$'
URL_REGEX: Final[str] = r'^https?://[^\s/$.?#].[^\s]*$'

# Length Constraints
MIN_PASSWORD_LENGTH: Final[int] = 8
MAX_PASSWORD_LENGTH: Final[int] = 128
MIN_USERNAME_LENGTH: Final[int] = 3
MAX_USERNAME_LENGTH: Final[int] = 20
MAX_EMAIL_LENGTH: Final[int] = 254
MAX_NAME_LENGTH: Final[int] = 100
MAX_DESCRIPTION_LENGTH: Final[int] = 1000

# ==========================================
# FEATURE FLAGS
# ==========================================

FEATURE_FLAGS = {
    'REGISTRATION_ENABLED': True,
    'AI_CHAT_ENABLED': True,
    'ANALYTICS_ENABLED': True,
    'MAINTENANCE_MODE': False,
    'BETA_FEATURES': False,
    'DEBUG_MODE': False,
}

# ==========================================
# LEARNING KEYWORDS
# ==========================================

LEARNING_STYLE_KEYWORDS = {
    LearningStyle.VISUAL: [
        'görsel', 'şema', 'grafik', 'resim', 
        'video', 'animasyon', 'renk', 'diyagram'
    ],
    LearningStyle.AUDITORY: [
        'dinle', 'anlat', 'konuş', 'ses', 
        'müzik', 'tartış', 'açıkla', 'podcast'
    ],
    LearningStyle.READING: [
        'oku', 'yaz', 'not', 'metin', 
        'kitap', 'makale', 'araştır', 'döküman'
    ],
    LearningStyle.KINESTHETIC: [
        'yap', 'uygula', 'deney', 'hareket', 
        'dokun', 'pratik', 'el', 'deneyim'
    ]
}

# ==========================================
# RESOURCE TYPES
# ==========================================

LEARNING_RESOURCES = {
    LearningStyle.VISUAL: [
        'Video dersleri',
        'İnfografikler',
        'Akış şemaları',
        '3D modeller',
        'Animasyonlar'
    ],
    LearningStyle.AUDITORY: [
        'Podcast\'ler',
        'Sesli kitaplar',
        'Grup tartışmaları',
        'Anlatımlı videolar',
        'Müzik eşliğinde öğrenme'
    ],
    LearningStyle.READING: [
        'E-kitaplar',
        'Makaleler',
        'Ders notları',
        'Araştırma raporları',
        'Blog yazıları'
    ],
    LearningStyle.KINESTHETIC: [
        'Simülasyonlar',
        'Laboratuvar deneyleri',
        'İnteraktif uygulamalar',
        'Projeler',
        'Oyunlaştırılmış içerik'
    ]
}

# ==========================================
# ASSESSMENT STRATEGIES
# ==========================================

ASSESSMENT_STRATEGIES = {
    LearningStyle.VISUAL: 'Görsel sunum ve diyagram oluşturma',
    LearningStyle.AUDITORY: 'Sözlü sunum ve tartışma',
    LearningStyle.READING: 'Yazılı rapor ve deneme',
    LearningStyle.KINESTHETIC: 'Proje ve uygulama'
}

# ==========================================
# ERROR CODES
# ==========================================

class ErrorCode(IntEnum):
    """Application error codes"""
    # General errors (1000-1999)
    UNKNOWN_ERROR = 1000
    VALIDATION_ERROR = 1001
    NOT_FOUND = 1002
    ALREADY_EXISTS = 1003
    
    # Authentication errors (2000-2999)
    INVALID_CREDENTIALS = 2000
    TOKEN_EXPIRED = 2001
    TOKEN_INVALID = 2002
    UNAUTHORIZED = 2003
    FORBIDDEN = 2004
    
    # Database errors (3000-3999)
    DATABASE_ERROR = 3000
    CONNECTION_ERROR = 3001
    QUERY_ERROR = 3002
    INTEGRITY_ERROR = 3003
    
    # API errors (4000-4999)
    RATE_LIMIT_EXCEEDED = 4000
    INVALID_REQUEST = 4001
    SERVICE_UNAVAILABLE = 4002
    TIMEOUT = 4003
    
    # Business logic errors (5000-5999)
    INSUFFICIENT_CREDITS = 5000
    QUIZ_ALREADY_TAKEN = 5001
    ENROLLMENT_CLOSED = 5002
    PREREQUISITE_NOT_MET = 5003
