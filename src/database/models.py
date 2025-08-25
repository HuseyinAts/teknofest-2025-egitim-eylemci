"""
Database models for TEKNOFEST 2025 Education Platform
"""

import uuid
from datetime import datetime
from typing import Optional, List

from sqlalchemy import (
    Column, String, Integer, Float, Boolean, DateTime, Text, JSON,
    ForeignKey, Table, Index, UniqueConstraint, CheckConstraint,
    Enum as SQLEnum
)
from sqlalchemy.orm import relationship, backref
from sqlalchemy.dialects.postgresql import UUID, JSONB, ARRAY
from sqlalchemy.sql import func
import enum

from .base import Base


class UserRole(enum.Enum):
    """User role enumeration"""
    STUDENT = "student"
    TEACHER = "teacher"
    ADMIN = "admin"
    PARENT = "parent"


class DifficultyLevel(enum.Enum):
    """Difficulty level enumeration"""
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"


class ContentType(enum.Enum):
    """Content type enumeration"""
    VIDEO = "video"
    TEXT = "text"
    QUIZ = "quiz"
    EXERCISE = "exercise"
    PROJECT = "project"


# Association tables for many-to-many relationships
user_learning_paths = Table(
    'user_learning_paths',
    Base.metadata,
    Column('user_id', UUID(as_uuid=True), ForeignKey('users.id', ondelete='CASCADE')),
    Column('learning_path_id', UUID(as_uuid=True), ForeignKey('learning_paths.id', ondelete='CASCADE')),
    Column('enrolled_at', DateTime(timezone=True), server_default=func.now()),
    Column('progress', Float, default=0.0),
    Column('completed_at', DateTime(timezone=True), nullable=True),
    UniqueConstraint('user_id', 'learning_path_id', name='uq_user_learning_path')
)

user_achievements = Table(
    'user_achievements',
    Base.metadata,
    Column('user_id', UUID(as_uuid=True), ForeignKey('users.id', ondelete='CASCADE')),
    Column('achievement_id', UUID(as_uuid=True), ForeignKey('achievements.id', ondelete='CASCADE')),
    Column('earned_at', DateTime(timezone=True), server_default=func.now()),
    UniqueConstraint('user_id', 'achievement_id', name='uq_user_achievement')
)


class TimestampMixin:
    """Mixin for adding timestamp fields"""
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)


class User(Base, TimestampMixin):
    """User model"""
    __tablename__ = 'users'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(String(255), unique=True, nullable=False, index=True)
    username = Column(String(100), unique=True, nullable=False, index=True)
    full_name = Column(String(255), nullable=False)
    hashed_password = Column(String(255), nullable=False)
    role = Column(SQLEnum(UserRole), default=UserRole.STUDENT, nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    is_verified = Column(Boolean, default=False, nullable=False)
    
    # Profile information
    avatar_url = Column(String(500), nullable=True)
    bio = Column(Text, nullable=True)
    date_of_birth = Column(DateTime, nullable=True)
    phone_number = Column(String(20), nullable=True)
    
    # Learning preferences
    preferred_language = Column(String(10), default='tr', nullable=False)
    learning_style = Column(JSONB, nullable=True)  # Visual, auditory, kinesthetic, etc.
    interests = Column(ARRAY(String), nullable=True)
    
    # Statistics
    total_study_time = Column(Integer, default=0)  # in minutes
    streak_days = Column(Integer, default=0)
    points = Column(Integer, default=0)
    level = Column(Integer, default=1)
    
    # OAuth fields
    oauth_provider = Column(String(50), nullable=True)
    oauth_id = Column(String(255), nullable=True)
    
    # Security
    last_login_at = Column(DateTime(timezone=True), nullable=True)
    last_login_ip = Column(String(45), nullable=True)
    failed_login_attempts = Column(Integer, default=0)
    locked_until = Column(DateTime(timezone=True), nullable=True)
    
    # Relationships
    learning_paths = relationship('LearningPath', secondary=user_learning_paths, back_populates='users')
    study_sessions = relationship('StudySession', back_populates='user', cascade='all, delete-orphan')
    assessments = relationship('Assessment', back_populates='user', cascade='all, delete-orphan')
    achievements = relationship('Achievement', secondary=user_achievements, back_populates='users')
    notifications = relationship('Notification', back_populates='user', cascade='all, delete-orphan')
    
    # Indexes
    __table_args__ = (
        Index('ix_users_email_active', 'email', 'is_active'),
        Index('ix_users_role_active', 'role', 'is_active'),
        CheckConstraint('streak_days >= 0', name='check_streak_positive'),
        CheckConstraint('points >= 0', name='check_points_positive'),
        CheckConstraint('level >= 1', name='check_level_positive'),
    )


class LearningPath(Base, TimestampMixin):
    """Learning path model"""
    __tablename__ = 'learning_paths'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    title = Column(String(255), nullable=False)
    description = Column(Text, nullable=False)
    slug = Column(String(255), unique=True, nullable=False, index=True)
    
    # Content
    objectives = Column(ARRAY(String), nullable=False)
    prerequisites = Column(ARRAY(String), nullable=True)
    tags = Column(ARRAY(String), nullable=True)
    
    # Metadata
    difficulty = Column(SQLEnum(DifficultyLevel), default=DifficultyLevel.BEGINNER, nullable=False)
    estimated_hours = Column(Float, nullable=False)
    language = Column(String(10), default='tr', nullable=False)
    
    # AI-generated content
    ai_generated = Column(Boolean, default=False)
    ai_model = Column(String(100), nullable=True)
    ai_parameters = Column(JSONB, nullable=True)
    
    # Statistics
    enrollment_count = Column(Integer, default=0)
    completion_count = Column(Integer, default=0)
    average_rating = Column(Float, nullable=True)
    
    # Publishing
    is_published = Column(Boolean, default=False)
    published_at = Column(DateTime(timezone=True), nullable=True)
    
    # Relationships
    users = relationship('User', secondary=user_learning_paths, back_populates='learning_paths')
    modules = relationship('Module', back_populates='learning_path', cascade='all, delete-orphan')
    
    __table_args__ = (
        Index('ix_learning_paths_difficulty', 'difficulty'),
        Index('ix_learning_paths_published', 'is_published'),
        CheckConstraint('estimated_hours > 0', name='check_estimated_hours_positive'),
    )


class Module(Base, TimestampMixin):
    """Module within a learning path"""
    __tablename__ = 'modules'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    learning_path_id = Column(UUID(as_uuid=True), ForeignKey('learning_paths.id', ondelete='CASCADE'), nullable=False)
    title = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    order_index = Column(Integer, nullable=False)
    
    # Content
    content_type = Column(SQLEnum(ContentType), nullable=False)
    content_url = Column(String(500), nullable=True)
    content_data = Column(JSONB, nullable=True)
    
    # Requirements
    estimated_minutes = Column(Integer, default=30)
    is_mandatory = Column(Boolean, default=True)
    
    # Relationships
    learning_path = relationship('LearningPath', back_populates='modules')
    progress_records = relationship('Progress', back_populates='module', cascade='all, delete-orphan')
    
    __table_args__ = (
        UniqueConstraint('learning_path_id', 'order_index', name='uq_module_order'),
        Index('ix_modules_learning_path', 'learning_path_id'),
    )


class StudySession(Base, TimestampMixin):
    """Study session tracking"""
    __tablename__ = 'study_sessions'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.id', ondelete='CASCADE'), nullable=False)
    module_id = Column(UUID(as_uuid=True), ForeignKey('modules.id', ondelete='SET NULL'), nullable=True)
    
    # Session data
    started_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    ended_at = Column(DateTime(timezone=True), nullable=True)
    duration_minutes = Column(Integer, nullable=True)
    
    # Activity tracking
    interactions = Column(JSONB, nullable=True)  # Clicks, scrolls, pauses, etc.
    notes = Column(Text, nullable=True)
    
    # AI assistance
    ai_interactions = Column(Integer, default=0)
    ai_messages = Column(JSONB, nullable=True)
    
    # Relationships
    user = relationship('User', back_populates='study_sessions')
    
    __table_args__ = (
        Index('ix_study_sessions_user_date', 'user_id', 'started_at'),
    )


class Assessment(Base, TimestampMixin):
    """Assessment/Quiz model"""
    __tablename__ = 'assessments'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.id', ondelete='CASCADE'), nullable=False)
    module_id = Column(UUID(as_uuid=True), ForeignKey('modules.id', ondelete='SET NULL'), nullable=True)
    
    # Assessment data
    type = Column(String(50), nullable=False)  # quiz, exam, project
    questions = Column(JSONB, nullable=False)
    answers = Column(JSONB, nullable=False)
    
    # Results
    score = Column(Float, nullable=False)
    max_score = Column(Float, nullable=False)
    percentage = Column(Float, nullable=False)
    passed = Column(Boolean, nullable=False)
    
    # Timing
    started_at = Column(DateTime(timezone=True), nullable=False)
    completed_at = Column(DateTime(timezone=True), nullable=False)
    time_spent_seconds = Column(Integer, nullable=False)
    
    # Feedback
    feedback = Column(JSONB, nullable=True)
    ai_evaluation = Column(JSONB, nullable=True)
    
    # Relationships
    user = relationship('User', back_populates='assessments')
    
    __table_args__ = (
        Index('ix_assessments_user_date', 'user_id', 'completed_at'),
        CheckConstraint('score >= 0 AND score <= max_score', name='check_score_valid'),
        CheckConstraint('percentage >= 0 AND percentage <= 100', name='check_percentage_valid'),
    )


class Progress(Base, TimestampMixin):
    """User progress tracking"""
    __tablename__ = 'progress'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.id', ondelete='CASCADE'), nullable=False)
    module_id = Column(UUID(as_uuid=True), ForeignKey('modules.id', ondelete='CASCADE'), nullable=False)
    
    # Progress data
    status = Column(String(20), default='not_started')  # not_started, in_progress, completed
    progress_percentage = Column(Float, default=0.0)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    
    # Tracking
    time_spent_minutes = Column(Integer, default=0)
    attempt_count = Column(Integer, default=0)
    
    # Relationships
    module = relationship('Module', back_populates='progress_records')
    
    __table_args__ = (
        UniqueConstraint('user_id', 'module_id', name='uq_user_module_progress'),
        Index('ix_progress_user_module', 'user_id', 'module_id'),
        CheckConstraint('progress_percentage >= 0 AND progress_percentage <= 100', name='check_progress_percentage'),
    )


class Achievement(Base, TimestampMixin):
    """Achievement/Badge model"""
    __tablename__ = 'achievements'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(100), unique=True, nullable=False)
    description = Column(Text, nullable=False)
    icon_url = Column(String(500), nullable=True)
    
    # Requirements
    criteria = Column(JSONB, nullable=False)
    points = Column(Integer, default=0)
    
    # Rarity
    rarity = Column(String(20), default='common')  # common, rare, epic, legendary
    
    # Relationships
    users = relationship('User', secondary=user_achievements, back_populates='achievements')


class Notification(Base, TimestampMixin):
    """User notification model"""
    __tablename__ = 'notifications'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.id', ondelete='CASCADE'), nullable=False)
    
    # Notification data
    type = Column(String(50), nullable=False)
    title = Column(String(255), nullable=False)
    message = Column(Text, nullable=False)
    data = Column(JSONB, nullable=True)
    
    # Status
    is_read = Column(Boolean, default=False)
    read_at = Column(DateTime(timezone=True), nullable=True)
    
    # Relationships
    user = relationship('User', back_populates='notifications')
    
    __table_args__ = (
        Index('ix_notifications_user_read', 'user_id', 'is_read'),
        Index('ix_notifications_created', 'created_at'),
    )


class AuditLog(Base):
    """Audit log for tracking important actions"""
    __tablename__ = 'audit_logs'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.id', ondelete='SET NULL'), nullable=True)
    
    # Action details
    action = Column(String(100), nullable=False)
    entity_type = Column(String(50), nullable=True)
    entity_id = Column(UUID(as_uuid=True), nullable=True)
    
    # Data
    old_data = Column(JSONB, nullable=True)
    new_data = Column(JSONB, nullable=True)
    
    # Request info
    ip_address = Column(String(45), nullable=True)
    user_agent = Column(String(500), nullable=True)
    
    # Timestamp
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    
    __table_args__ = (
        Index('ix_audit_logs_user', 'user_id'),
        Index('ix_audit_logs_action', 'action'),
        Index('ix_audit_logs_entity', 'entity_type', 'entity_id'),
        Index('ix_audit_logs_created', 'created_at'),
    )


class IRTItemBank(Base):
    """IRT Item Bank for adaptive testing"""
    __tablename__ = 'irt_item_bank'
    
    id = Column(UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid())
    item_id = Column(String(255), nullable=False, unique=True)
    question_id = Column(UUID(as_uuid=True), ForeignKey('questions.id', ondelete='SET NULL'), nullable=True)
    
    # IRT Parameters (3PL Model)
    difficulty = Column(Float, nullable=False)  # b parameter
    discrimination = Column(Float, nullable=False, server_default='1.0')  # a parameter
    guessing = Column(Float, nullable=False, server_default='0.2')  # c parameter
    upper_asymptote = Column(Float, nullable=False, server_default='1.0')  # d parameter
    
    # Content metadata
    subject = Column(String(100), nullable=False)
    topic = Column(String(255), nullable=True)
    grade_level = Column(Integer, nullable=True)
    
    # Usage statistics
    usage_count = Column(Integer, nullable=False, server_default='0')
    exposure_rate = Column(Float, nullable=False, server_default='0.0')
    
    # Calibration data
    standard_errors = Column(JSONB, nullable=True)
    fit_statistics = Column(JSONB, nullable=True)
    calibration_sample_size = Column(Integer, nullable=True)
    calibration_method = Column(String(50), nullable=True)
    
    # Status
    is_active = Column(Boolean, nullable=False, server_default='true')
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    
    # Relationships
    question = relationship("Question", back_populates="irt_parameters")
    
    __table_args__ = (
        CheckConstraint('difficulty >= -4 AND difficulty <= 4', name='check_difficulty_range'),
        CheckConstraint('discrimination >= 0.1 AND discrimination <= 3', name='check_discrimination_range'),
        CheckConstraint('guessing >= 0 AND guessing <= 0.5', name='check_guessing_range'),
        CheckConstraint('upper_asymptote >= 0.5 AND upper_asymptote <= 1', name='check_upper_asymptote_range'),
        Index('idx_irt_item_bank_subject_topic', 'subject', 'topic'),
        Index('idx_irt_item_bank_difficulty', 'difficulty'),
        Index('idx_irt_item_bank_usage', 'usage_count', 'exposure_rate'),
    )


class IRTStudentAbility(Base):
    """Student ability estimates from IRT"""
    __tablename__ = 'irt_student_abilities'
    
    id = Column(UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid())
    student_id = Column(UUID(as_uuid=True), ForeignKey('student_profiles.id', ondelete='CASCADE'), nullable=False)
    
    # Ability estimate
    theta = Column(Float, nullable=False)
    standard_error = Column(Float, nullable=False)
    confidence_lower = Column(Float, nullable=False)
    confidence_upper = Column(Float, nullable=False)
    estimation_method = Column(String(50), nullable=False)
    
    # Context
    subject = Column(String(100), nullable=True)
    topic = Column(String(255), nullable=True)
    test_id = Column(String(255), nullable=True)
    session_id = Column(String(255), nullable=True)
    
    # Test data
    items_count = Column(Integer, nullable=False)
    response_pattern = Column(ARRAY(Integer), nullable=True)
    test_information = Column(Float, nullable=True)
    reliability = Column(Float, nullable=True)
    convergence_iterations = Column(Integer, nullable=True)
    
    # Metadata
    timestamp = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    metadata = Column(JSONB, nullable=True)
    
    # Relationships
    student = relationship("StudentProfile", back_populates="irt_abilities")
    
    __table_args__ = (
        CheckConstraint('theta >= -4 AND theta <= 4', name='check_theta_range'),
        CheckConstraint('standard_error >= 0', name='check_se_positive'),
        CheckConstraint('reliability >= 0 AND reliability <= 1', name='check_reliability_range'),
        Index('idx_irt_abilities_student_subject', 'student_id', 'subject'),
        Index('idx_irt_abilities_timestamp', 'timestamp'),
    )


class IRTTestSession(Base):
    """Adaptive test sessions"""
    __tablename__ = 'irt_test_sessions'
    
    id = Column(UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid())
    session_id = Column(String(255), nullable=False, unique=True)
    student_id = Column(UUID(as_uuid=True), ForeignKey('student_profiles.id', ondelete='CASCADE'), nullable=False)
    
    # Test configuration
    subject = Column(String(100), nullable=False)
    topic = Column(String(255), nullable=True)
    test_type = Column(String(50), nullable=False, server_default='adaptive')
    max_items = Column(Integer, nullable=False, server_default='20')
    min_items = Column(Integer, nullable=False, server_default='5')
    target_se = Column(Float, nullable=False, server_default='0.3')
    
    # Current state
    current_theta = Column(Float, nullable=False, server_default='0.0')
    current_se = Column(Float, nullable=False, server_default='1.0')
    items_administered = Column(ARRAY(String), nullable=False, server_default='{}')
    responses = Column(ARRAY(Integer), nullable=False, server_default='{}')
    response_times = Column(ARRAY(Float), nullable=True)
    current_item = Column(String(255), nullable=True)
    
    # Final results
    final_theta = Column(Float, nullable=True)
    final_se = Column(Float, nullable=True)
    
    # Status
    status = Column(String(50), nullable=False, server_default='in_progress')
    start_time = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    end_time = Column(DateTime(timezone=True), nullable=True)
    time_limit = Column(Interval, nullable=True)
    
    # Metadata
    metadata = Column(JSONB, nullable=True)
    
    # Relationships
    student = relationship("StudentProfile", back_populates="irt_sessions")
    
    __table_args__ = (
        CheckConstraint('max_items >= min_items', name='check_items_range'),
        CheckConstraint('target_se > 0 AND target_se <= 1', name='check_target_se_range'),
        Index('idx_irt_sessions_student', 'student_id'),
        Index('idx_irt_sessions_status', 'status'),
    )


class IRTCalibrationHistory(Base):
    """History of item calibrations"""
    __tablename__ = 'irt_calibration_history'
    
    id = Column(UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid())
    calibration_id = Column(String(255), nullable=False)
    
    # Calibration details
    subject = Column(String(100), nullable=False)
    topic = Column(String(255), nullable=True)
    calibration_method = Column(String(50), nullable=False)
    sample_size = Column(Integer, nullable=False)
    items_calibrated = Column(Integer, nullable=False)
    
    # Results
    convergence_status = Column(String(50), nullable=False)
    fit_statistics = Column(JSONB, nullable=True)
    parameters_before = Column(JSONB, nullable=True)
    parameters_after = Column(JSONB, nullable=False)
    calibration_time = Column(Float, nullable=False)
    
    # Metadata
    timestamp = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    performed_by = Column(UUID(as_uuid=True), ForeignKey('users.id', ondelete='SET NULL'), nullable=True)
    
    # Relationships
    user = relationship("User")