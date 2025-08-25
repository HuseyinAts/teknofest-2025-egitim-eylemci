"""
Database Models - Production Ready Implementation
TEKNOFEST 2025 - Education Platform Database Schema
"""

import uuid
import enum
from datetime import datetime
from typing import Optional, List, Dict, Any
from sqlalchemy import (
    Column, String, Integer, Float, Boolean, DateTime, Text, JSON,
    ForeignKey, UniqueConstraint, Index, CheckConstraint, Table,
    Enum as SQLEnum
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship, validates, backref
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func
from sqlalchemy.ext.hybrid import hybrid_property

Base = declarative_base()


# ==================== Enums ====================

class UserRole(enum.Enum):
    """User role enumeration"""
    STUDENT = "student"
    TEACHER = "teacher"
    ADMIN = "admin"
    PARENT = "parent"
    GUEST = "guest"


class DifficultyLevel(enum.Enum):
    """Difficulty level enumeration"""
    VERY_EASY = 0.2
    EASY = 0.4
    MEDIUM = 0.6
    HARD = 0.8
    VERY_HARD = 1.0


class QuestionType(enum.Enum):
    """Question type enumeration"""
    MULTIPLE_CHOICE = "multiple_choice"
    TRUE_FALSE = "true_false"
    SHORT_ANSWER = "short_answer"
    ESSAY = "essay"
    MATCHING = "matching"
    FILL_BLANK = "fill_blank"


class LearningStyle(enum.Enum):
    """Learning style enumeration"""
    VISUAL = "visual"
    AUDITORY = "auditory"
    READING = "reading"
    KINESTHETIC = "kinesthetic"
    MIXED = "mixed"


# ==================== Association Tables ====================

# Many-to-many relationship between students and courses
enrollment_table = Table(
    'enrollments',
    Base.metadata,
    Column('student_id', Integer, ForeignKey('students.id', ondelete='CASCADE')),
    Column('course_id', Integer, ForeignKey('courses.id', ondelete='CASCADE')),
    Column('enrolled_at', DateTime, default=func.now()),
    Column('completed', Boolean, default=False),
    UniqueConstraint('student_id', 'course_id', name='unique_enrollment')
)

# Many-to-many relationship between quizzes and questions
quiz_questions = Table(
    'quiz_questions',
    Base.metadata,
    Column('quiz_id', Integer, ForeignKey('quizzes.id', ondelete='CASCADE')),
    Column('question_id', Integer, ForeignKey('questions.id', ondelete='CASCADE')),
    Column('order', Integer, default=0),
    UniqueConstraint('quiz_id', 'question_id', name='unique_quiz_question')
)


# ==================== Base Mixins ====================

class TimestampMixin:
    """Mixin for timestamp fields"""
    created_at = Column(DateTime, default=func.now(), nullable=False)
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now(), nullable=False)


class SoftDeleteMixin:
    """Mixin for soft delete functionality"""
    is_deleted = Column(Boolean, default=False, nullable=False)
    deleted_at = Column(DateTime, nullable=True)


# ==================== Main Models ====================

class User(Base, TimestampMixin, SoftDeleteMixin):
    """User model - base for all user types"""
    __tablename__ = 'users'
    
    # Primary key
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Authentication fields
    username = Column(String(50), unique=True, nullable=False, index=True)
    email = Column(String(100), unique=True, nullable=False, index=True)
    hashed_password = Column(String(255), nullable=False)
    
    # Profile fields
    full_name = Column(String(100), nullable=True)
    phone = Column(String(20), nullable=True)
    avatar_url = Column(String(255), nullable=True)
    bio = Column(Text, nullable=True)
    
    # Role and permissions
    role = Column(SQLEnum(UserRole), default=UserRole.STUDENT, nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    is_verified = Column(Boolean, default=False, nullable=False)
    
    # Tracking fields
    last_login = Column(DateTime, nullable=True)
    login_count = Column(Integer, default=0, nullable=False)
    
    # JSON fields for flexible data
    preferences = Column(JSON, default={}, nullable=True)
    metadata = Column(JSON, default={}, nullable=True)
    
    # Relationships
    student_profile = relationship("Student", back_populates="user", uselist=False, cascade="all, delete-orphan")
    teacher_profile = relationship("Teacher", back_populates="user", uselist=False, cascade="all, delete-orphan")
    taught_courses = relationship("Course", back_populates="teacher", foreign_keys="Course.teacher_id")
    created_quizzes = relationship("Quiz", back_populates="creator", foreign_keys="Quiz.created_by_id")
    
    # Indexes
    __table_args__ = (
        Index('idx_user_email_active', 'email', 'is_active'),
        Index('idx_user_role', 'role'),
        CheckConstraint('length(username) >= 3', name='check_username_length'),
    )
    
    @validates('email')
    def validate_email(self, key, value):
        """Validate email format"""
        if '@' not in value:
            raise ValueError("Invalid email address")
        return value.lower()
    
    def __repr__(self):
        return f"<User(id={self.id}, username={self.username}, role={self.role})>"


class Student(Base, TimestampMixin):
    """Student profile model"""
    __tablename__ = 'students'
    
    # Primary key
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Foreign key to User
    user_id = Column(Integer, ForeignKey('users.id', ondelete='CASCADE'), unique=True, nullable=False)
    
    # Academic information
    grade = Column(Integer, nullable=False)
    school = Column(String(200), nullable=True)
    student_number = Column(String(50), nullable=True, unique=True)
    
    # Learning profile
    learning_style = Column(SQLEnum(LearningStyle), default=LearningStyle.MIXED, nullable=False)
    current_level = Column(Float, default=0.5, nullable=False)
    target_level = Column(Float, default=0.8, nullable=False)
    study_hours_per_day = Column(Float, default=2.0, nullable=False)
    
    # Performance metrics
    total_points = Column(Integer, default=0, nullable=False)
    quiz_count = Column(Integer, default=0, nullable=False)
    average_score = Column(Float, default=0.0, nullable=False)
    streak_days = Column(Integer, default=0, nullable=False)
    
    # JSON fields
    weak_topics = Column(JSON, default=[], nullable=True)
    strong_topics = Column(JSON, default=[], nullable=True)
    achievements = Column(JSON, default=[], nullable=True)
    
    # Relationships
    user = relationship("User", back_populates="student_profile")
    enrolled_courses = relationship("Course", secondary=enrollment_table, back_populates="enrolled_students")
    progress_records = relationship("Progress", back_populates="student", cascade="all, delete-orphan")
    answers = relationship("Answer", back_populates="student", cascade="all, delete-orphan")
    learning_paths = relationship("LearningPath", back_populates="student", cascade="all, delete-orphan")
    
    # Constraints
    __table_args__ = (
        CheckConstraint('grade >= 1 AND grade <= 12', name='check_grade_range'),
        CheckConstraint('current_level >= 0 AND current_level <= 1', name='check_current_level'),
        CheckConstraint('target_level >= 0 AND target_level <= 1', name='check_target_level'),
        Index('idx_student_grade', 'grade'),
    )
    
    def __repr__(self):
        return f"<Student(id={self.id}, user_id={self.user_id}, grade={self.grade})>"


class Teacher(Base, TimestampMixin):
    """Teacher profile model"""
    __tablename__ = 'teachers'
    
    # Primary key
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Foreign key to User
    user_id = Column(Integer, ForeignKey('users.id', ondelete='CASCADE'), unique=True, nullable=False)
    
    # Professional information
    subject = Column(String(100), nullable=False)
    qualification = Column(String(200), nullable=True)
    years_of_experience = Column(Integer, default=0, nullable=False)
    school = Column(String(200), nullable=True)
    
    # Performance metrics
    total_students = Column(Integer, default=0, nullable=False)
    courses_created = Column(Integer, default=0, nullable=False)
    average_rating = Column(Float, default=0.0, nullable=False)
    
    # JSON fields
    specializations = Column(JSON, default=[], nullable=True)
    certifications = Column(JSON, default=[], nullable=True)
    
    # Relationships
    user = relationship("User", back_populates="teacher_profile")
    
    # Constraints
    __table_args__ = (
        CheckConstraint('years_of_experience >= 0', name='check_experience'),
        CheckConstraint('average_rating >= 0 AND average_rating <= 5', name='check_rating'),
    )
    
    def __repr__(self):
        return f"<Teacher(id={self.id}, user_id={self.user_id}, subject={self.subject})>"


class Course(Base, TimestampMixin, SoftDeleteMixin):
    """Course model"""
    __tablename__ = 'courses'
    
    # Primary key
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Course information
    title = Column(String(200), nullable=False)
    description = Column(Text, nullable=True)
    subject = Column(String(100), nullable=False)
    grade_level = Column(Integer, nullable=False)
    
    # Teacher
    teacher_id = Column(Integer, ForeignKey('users.id', ondelete='SET NULL'), nullable=True)
    
    # Course details
    difficulty = Column(Float, default=0.5, nullable=False)
    estimated_hours = Column(Integer, default=30, nullable=False)
    max_students = Column(Integer, default=100, nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    is_published = Column(Boolean, default=False, nullable=False)
    
    # Curriculum alignment
    curriculum_alignment = Column(String(50), nullable=True)
    prerequisites = Column(JSON, default=[], nullable=True)
    learning_objectives = Column(JSON, default=[], nullable=True)
    
    # Statistics
    enrolled_count = Column(Integer, default=0, nullable=False)
    completion_rate = Column(Float, default=0.0, nullable=False)
    average_score = Column(Float, default=0.0, nullable=False)
    
    # Relationships
    teacher = relationship("User", back_populates="taught_courses", foreign_keys=[teacher_id])
    enrolled_students = relationship("Student", secondary=enrollment_table, back_populates="enrolled_courses")
    modules = relationship("Module", back_populates="course", cascade="all, delete-orphan")
    quizzes = relationship("Quiz", back_populates="course", cascade="all, delete-orphan")
    
    # Indexes
    __table_args__ = (
        Index('idx_course_subject_grade', 'subject', 'grade_level'),
        Index('idx_course_active_published', 'is_active', 'is_published'),
        CheckConstraint('difficulty >= 0 AND difficulty <= 1', name='check_course_difficulty'),
        CheckConstraint('grade_level >= 1 AND grade_level <= 12', name='check_course_grade'),
    )
    
    def __repr__(self):
        return f"<Course(id={self.id}, title={self.title}, subject={self.subject})>"


class Module(Base, TimestampMixin):
    """Course module model"""
    __tablename__ = 'modules'
    
    # Primary key
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Foreign key
    course_id = Column(Integer, ForeignKey('courses.id', ondelete='CASCADE'), nullable=False)
    
    # Module information
    title = Column(String(200), nullable=False)
    description = Column(Text, nullable=True)
    order = Column(Integer, default=0, nullable=False)
    
    # Content
    content = Column(Text, nullable=True)
    video_url = Column(String(255), nullable=True)
    resources = Column(JSON, default=[], nullable=True)
    
    # Requirements
    estimated_time = Column(Integer, default=60, nullable=False)  # in minutes
    is_required = Column(Boolean, default=True, nullable=False)
    
    # Relationships
    course = relationship("Course", back_populates="modules")
    
    # Indexes
    __table_args__ = (
        Index('idx_module_course_order', 'course_id', 'order'),
        UniqueConstraint('course_id', 'order', name='unique_module_order'),
    )
    
    def __repr__(self):
        return f"<Module(id={self.id}, title={self.title}, course_id={self.course_id})>"


class Quiz(Base, TimestampMixin, SoftDeleteMixin):
    """Quiz model"""
    __tablename__ = 'quizzes'
    
    # Primary key
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Quiz information
    title = Column(String(200), nullable=False)
    description = Column(Text, nullable=True)
    instructions = Column(Text, nullable=True)
    
    # Relations
    course_id = Column(Integer, ForeignKey('courses.id', ondelete='CASCADE'), nullable=True)
    created_by_id = Column(Integer, ForeignKey('users.id', ondelete='SET NULL'), nullable=True)
    
    # Quiz settings
    question_count = Column(Integer, default=10, nullable=False)
    time_limit = Column(Integer, default=30, nullable=False)  # in minutes
    max_attempts = Column(Integer, default=3, nullable=False)
    passing_score = Column(Float, default=0.6, nullable=False)
    difficulty = Column(Float, default=0.5, nullable=False)
    
    # Quiz type and status
    quiz_type = Column(String(50), default='practice', nullable=False)
    is_published = Column(Boolean, default=False, nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    randomize_questions = Column(Boolean, default=True, nullable=False)
    show_answers = Column(Boolean, default=True, nullable=False)
    
    # Statistics
    total_attempts = Column(Integer, default=0, nullable=False)
    average_score = Column(Float, default=0.0, nullable=False)
    completion_rate = Column(Float, default=0.0, nullable=False)
    
    # Dates
    available_from = Column(DateTime, nullable=True)
    available_until = Column(DateTime, nullable=True)
    
    # Relationships
    course = relationship("Course", back_populates="quizzes")
    creator = relationship("User", back_populates="created_quizzes", foreign_keys=[created_by_id])
    questions = relationship("Question", secondary=quiz_questions, back_populates="quizzes")
    attempts = relationship("QuizAttempt", back_populates="quiz", cascade="all, delete-orphan")
    
    # Computed properties
    @hybrid_property
    def total_points(self):
        """Calculate total points for the quiz"""
        return sum(q.points for q in self.questions)
    
    # Constraints
    __table_args__ = (
        CheckConstraint('passing_score >= 0 AND passing_score <= 1', name='check_passing_score'),
        CheckConstraint('difficulty >= 0 AND difficulty <= 1', name='check_quiz_difficulty'),
        Index('idx_quiz_course_active', 'course_id', 'is_active'),
    )
    
    def __repr__(self):
        return f"<Quiz(id={self.id}, title={self.title}, questions={self.question_count})>"


class Question(Base, TimestampMixin):
    """Question model"""
    __tablename__ = 'questions'
    
    # Primary key
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Question content
    question_text = Column(Text, nullable=False)
    question_type = Column(SQLEnum(QuestionType), nullable=False)
    
    # Answer options (for multiple choice)
    options = Column(JSON, default=[], nullable=True)
    correct_answer = Column(Text, nullable=True)
    
    # Metadata
    subject = Column(String(100), nullable=True)
    topic = Column(String(100), nullable=True)
    subtopic = Column(String(100), nullable=True)
    
    # Scoring and difficulty
    points = Column(Integer, default=10, nullable=False)
    difficulty = Column(Float, default=0.5, nullable=False)
    discrimination = Column(Float, default=0.3, nullable=True)  # IRT parameter
    
    # Additional content
    explanation = Column(Text, nullable=True)
    hint = Column(Text, nullable=True)
    resources = Column(JSON, default=[], nullable=True)
    
    # Media
    image_url = Column(String(255), nullable=True)
    audio_url = Column(String(255), nullable=True)
    
    # Usage statistics
    usage_count = Column(Integer, default=0, nullable=False)
    correct_count = Column(Integer, default=0, nullable=False)
    average_time = Column(Float, default=0.0, nullable=False)  # in seconds
    
    # Tags for categorization
    tags = Column(JSON, default=[], nullable=True)
    
    # Relationships
    quizzes = relationship("Quiz", secondary=quiz_questions, back_populates="questions")
    answers = relationship("Answer", back_populates="question", cascade="all, delete-orphan")
    
    # Computed property
    @hybrid_property
    def success_rate(self):
        """Calculate success rate"""
        if self.usage_count == 0:
            return 0.0
        return self.correct_count / self.usage_count
    
    # Constraints
    __table_args__ = (
        CheckConstraint('difficulty >= 0 AND difficulty <= 1', name='check_question_difficulty'),
        CheckConstraint('points > 0', name='check_positive_points'),
        Index('idx_question_subject_topic', 'subject', 'topic'),
        Index('idx_question_type', 'question_type'),
    )
    
    def __repr__(self):
        return f"<Question(id={self.id}, type={self.question_type}, points={self.points})>"


class Answer(Base, TimestampMixin):
    """Student answer model"""
    __tablename__ = 'answers'
    
    # Primary key
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Foreign keys
    student_id = Column(Integer, ForeignKey('students.id', ondelete='CASCADE'), nullable=False)
    question_id = Column(Integer, ForeignKey('questions.id', ondelete='CASCADE'), nullable=False)
    quiz_id = Column(Integer, ForeignKey('quizzes.id', ondelete='CASCADE'), nullable=True)
    attempt_id = Column(Integer, ForeignKey('quiz_attempts.id', ondelete='CASCADE'), nullable=True)
    
    # Answer content
    given_answer = Column(Text, nullable=True)
    is_correct = Column(Boolean, default=False, nullable=False)
    points_earned = Column(Float, default=0.0, nullable=False)
    
    # Timing
    time_taken = Column(Integer, default=0, nullable=False)  # in seconds
    submitted_at = Column(DateTime, default=func.now(), nullable=False)
    
    # Feedback
    feedback = Column(Text, nullable=True)
    confidence_level = Column(Float, nullable=True)  # 0-1 scale
    
    # Relationships
    student = relationship("Student", back_populates="answers")
    question = relationship("Question", back_populates="answers")
    attempt = relationship("QuizAttempt", back_populates="answers")
    
    # Constraints
    __table_args__ = (
        UniqueConstraint('student_id', 'question_id', 'attempt_id', name='unique_answer_per_attempt'),
        Index('idx_answer_student_quiz', 'student_id', 'quiz_id'),
    )
    
    def __repr__(self):
        return f"<Answer(id={self.id}, student_id={self.student_id}, correct={self.is_correct})>"


class QuizAttempt(Base, TimestampMixin):
    """Quiz attempt tracking"""
    __tablename__ = 'quiz_attempts'
    
    # Primary key
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Foreign keys
    student_id = Column(Integer, ForeignKey('students.id', ondelete='CASCADE'), nullable=False)
    quiz_id = Column(Integer, ForeignKey('quizzes.id', ondelete='CASCADE'), nullable=False)
    
    # Attempt information
    attempt_number = Column(Integer, default=1, nullable=False)
    started_at = Column(DateTime, default=func.now(), nullable=False)
    completed_at = Column(DateTime, nullable=True)
    
    # Scoring
    score = Column(Float, default=0.0, nullable=False)
    points_earned = Column(Float, default=0.0, nullable=False)
    points_possible = Column(Float, default=0.0, nullable=False)
    passed = Column(Boolean, default=False, nullable=False)
    
    # Status
    is_completed = Column(Boolean, default=False, nullable=False)
    time_spent = Column(Integer, default=0, nullable=False)  # in seconds
    
    # Relationships
    quiz = relationship("Quiz", back_populates="attempts")
    answers = relationship("Answer", back_populates="attempt", cascade="all, delete-orphan")
    
    # Constraints
    __table_args__ = (
        UniqueConstraint('student_id', 'quiz_id', 'attempt_number', name='unique_attempt'),
        Index('idx_attempt_student_completed', 'student_id', 'is_completed'),
    )
    
    def __repr__(self):
        return f"<QuizAttempt(id={self.id}, student_id={self.student_id}, score={self.score})>"


class Progress(Base, TimestampMixin):
    """Student progress tracking"""
    __tablename__ = 'progress'
    
    # Primary key
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Foreign keys
    student_id = Column(Integer, ForeignKey('students.id', ondelete='CASCADE'), nullable=False)
    course_id = Column(Integer, ForeignKey('courses.id', ondelete='CASCADE'), nullable=True)
    module_id = Column(Integer, ForeignKey('modules.id', ondelete='CASCADE'), nullable=True)
    
    # Progress data
    completed_topics = Column(JSON, default=[], nullable=True)
    current_topic = Column(String(200), nullable=True)
    
    # Performance metrics
    quiz_scores = Column(JSON, default=[], nullable=True)
    average_score = Column(Float, default=0.0, nullable=False)
    current_level = Column(Float, default=0.0, nullable=False)
    
    # Time tracking
    time_spent = Column(Integer, default=0, nullable=False)  # in minutes
    last_activity = Column(DateTime, default=func.now(), nullable=False)
    
    # Streaks and engagement
    streak_days = Column(Integer, default=0, nullable=False)
    total_sessions = Column(Integer, default=0, nullable=False)
    
    # Relationships
    student = relationship("Student", back_populates="progress_records")
    
    # Constraints
    __table_args__ = (
        UniqueConstraint('student_id', 'course_id', 'module_id', name='unique_progress'),
        Index('idx_progress_student_course', 'student_id', 'course_id'),
    )
    
    def __repr__(self):
        return f"<Progress(id={self.id}, student_id={self.student_id}, level={self.current_level})>"


class LearningPath(Base, TimestampMixin):
    """Personalized learning path"""
    __tablename__ = 'learning_paths'
    
    # Primary key
    id = Column(Integer, primary_key=True, autoincrement=True)
    path_id = Column(String(50), unique=True, nullable=False, default=lambda: str(uuid.uuid4()))
    
    # Foreign key
    student_id = Column(Integer, ForeignKey('students.id', ondelete='CASCADE'), nullable=False)
    
    # Path information
    title = Column(String(200), nullable=False)
    description = Column(Text, nullable=True)
    
    # Timeline
    start_date = Column(DateTime, default=func.now(), nullable=False)
    end_date = Column(DateTime, nullable=True)
    total_weeks = Column(Integer, default=12, nullable=False)
    current_week = Column(Integer, default=0, nullable=False)
    
    # Content
    weekly_plans = Column(JSON, default=[], nullable=False)
    milestones = Column(JSON, default=[], nullable=False)
    assessment_schedule = Column(JSON, default=[], nullable=False)
    
    # Progress
    progress_percentage = Column(Float, default=0.0, nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    is_completed = Column(Boolean, default=False, nullable=False)
    
    # Adaptation parameters
    difficulty_adjustment = Column(Float, default=0.0, nullable=False)
    pace_adjustment = Column(Float, default=1.0, nullable=False)
    
    # Relationships
    student = relationship("Student", back_populates="learning_paths")
    
    # Constraints
    __table_args__ = (
        CheckConstraint('progress_percentage >= 0 AND progress_percentage <= 100', name='check_progress_percentage'),
        CheckConstraint('current_week >= 0 AND current_week <= total_weeks', name='check_current_week'),
        Index('idx_learning_path_student_active', 'student_id', 'is_active'),
    )
    
    def __repr__(self):
        return f"<LearningPath(id={self.id}, student_id={self.student_id}, progress={self.progress_percentage}%)>"


# ==================== Additional Models ====================

class Notification(Base, TimestampMixin):
    """User notifications"""
    __tablename__ = 'notifications'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey('users.id', ondelete='CASCADE'), nullable=False)
    
    title = Column(String(200), nullable=False)
    message = Column(Text, nullable=False)
    type = Column(String(50), default='info', nullable=False)
    
    is_read = Column(Boolean, default=False, nullable=False)
    read_at = Column(DateTime, nullable=True)
    
    action_url = Column(String(255), nullable=True)
    metadata = Column(JSON, default={}, nullable=True)
    
    __table_args__ = (
        Index('idx_notification_user_unread', 'user_id', 'is_read'),
    )


class ActivityLog(Base, TimestampMixin):
    """User activity logging"""
    __tablename__ = 'activity_logs'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey('users.id', ondelete='CASCADE'), nullable=False)
    
    action = Column(String(100), nullable=False)
    entity_type = Column(String(50), nullable=True)
    entity_id = Column(Integer, nullable=True)
    
    ip_address = Column(String(45), nullable=True)
    user_agent = Column(String(255), nullable=True)
    
    metadata = Column(JSON, default={}, nullable=True)
    
    __table_args__ = (
        Index('idx_activity_user_action', 'user_id', 'action'),
        Index('idx_activity_entity', 'entity_type', 'entity_id'),
    )


# ==================== Create all tables ====================

def create_tables(engine):
    """Create all database tables"""
    Base.metadata.create_all(bind=engine)


def drop_tables(engine):
    """Drop all database tables"""
    Base.metadata.drop_all(bind=engine)


# Export all models
__all__ = [
    'Base',
    'User', 'Student', 'Teacher',
    'Course', 'Module',
    'Quiz', 'Question', 'Answer', 'QuizAttempt',
    'Progress', 'LearningPath',
    'Notification', 'ActivityLog',
    'UserRole', 'DifficultyLevel', 'QuestionType', 'LearningStyle',
    'create_tables', 'drop_tables'
]