"""
Database Models - Infrastructure Layer
TEKNOFEST 2025 - SQLAlchemy ORM Models
"""

from datetime import datetime
from typing import Optional, Dict, Any
import json

from sqlalchemy import (
    Column, String, Float, Integer, Boolean, DateTime, 
    JSON, ForeignKey, Text, UniqueConstraint, Index
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship, declarative_base
from sqlalchemy.sql import func
import uuid

from src.domain.entities import (
    Student as DomainStudent,
    LearningPath as DomainLearningPath,
    LearningModule as DomainLearningModule,
    Quiz as DomainQuiz,
    Question as DomainQuestion
)
from src.domain.value_objects import (
    Grade, LearningStyle, AbilityLevel, LearningStyles
)
from src.shared.constants import StudentStatus, QuizDifficulty

Base = declarative_base()


class StudentModel(Base):
    """SQLAlchemy model for Student entity"""
    __tablename__ = "students"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(String(255), unique=True, nullable=False, index=True)
    username = Column(String(100), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)
    full_name = Column(String(255), nullable=False)
    grade = Column(Integer, nullable=False)
    learning_style = Column(JSON, nullable=True)
    ability_level = Column(Float, default=0.5)
    status = Column(String(50), default="active")
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    learning_paths = relationship("LearningPathModel", back_populates="student", cascade="all, delete-orphan")
    quiz_results = relationship("QuizResultModel", back_populates="student", cascade="all, delete-orphan")
    
    # Indexes
    __table_args__ = (
        Index("idx_student_grade", "grade"),
        Index("idx_student_status", "status"),
        Index("idx_student_created", "created_at"),
    )
    
    def to_domain(self) -> DomainStudent:
        """Convert to domain entity"""
        learning_style_obj = None
        if self.learning_style:
            learning_style_obj = LearningStyle(
                primary_style=LearningStyles(self.learning_style["primary_style"]),
                secondary_style=LearningStyles(self.learning_style["secondary_style"]) if self.learning_style.get("secondary_style") else None,
                confidence=self.learning_style["confidence"],
                scores=self.learning_style.get("scores", {})
            )
        
        return DomainStudent(
            id=str(self.id),
            email=self.email,
            username=self.username,
            full_name=self.full_name,
            grade=Grade(self.grade),
            learning_style=learning_style_obj,
            ability_level=AbilityLevel(self.ability_level),
            status=StudentStatus(self.status),
            created_at=self.created_at,
            updated_at=self.updated_at or self.created_at
        )
    
    @classmethod
    def from_domain(cls, student: DomainStudent) -> 'StudentModel':
        """Create from domain entity"""
        return cls(
            id=uuid.UUID(student.id) if isinstance(student.id, str) else student.id,
            email=student.email,
            username=student.username,
            full_name=student.full_name,
            grade=student.grade.level,
            learning_style=student.learning_style.to_dict() if student.learning_style else None,
            ability_level=student.ability_level.value,
            status=student.status.value,
            created_at=student.created_at,
            updated_at=student.updated_at
        )
    
    def update_from_domain(self, student: DomainStudent):
        """Update model from domain entity"""
        self.email = student.email
        self.username = student.username
        self.full_name = student.full_name
        self.grade = student.grade.level
        self.learning_style = student.learning_style.to_dict() if student.learning_style else None
        self.ability_level = student.ability_level.value
        self.status = student.status.value
        self.updated_at = datetime.utcnow()


class LearningPathModel(Base):
    """SQLAlchemy model for LearningPath entity"""
    __tablename__ = "learning_paths"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    student_id = Column(UUID(as_uuid=True), ForeignKey("students.id"), nullable=False, unique=True)
    grade = Column(Integer, nullable=False)
    learning_style = Column(JSON, nullable=False)
    modules = Column(JSON, nullable=False)  # Stored as JSON for simplicity
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    student = relationship("StudentModel", back_populates="learning_paths")
    
    # Indexes
    __table_args__ = (
        Index("idx_learning_path_student", "student_id"),
        Index("idx_learning_path_grade", "grade"),
    )
    
    def to_domain(self) -> DomainLearningPath:
        """Convert to domain entity"""
        # Parse learning style
        ls_data = self.learning_style
        learning_style = LearningStyle(
            primary_style=LearningStyles(ls_data["primary_style"]),
            secondary_style=LearningStyles(ls_data["secondary_style"]) if ls_data.get("secondary_style") else None,
            confidence=ls_data["confidence"],
            scores=ls_data.get("scores", {})
        )
        
        # Parse modules
        modules = []
        for module_data in self.modules:
            module = DomainLearningModule(
                id=module_data["id"],
                title=module_data["title"],
                subject=module_data["subject"],
                topic=module_data["topic"],
                content_type=module_data["content_type"],
                estimated_hours=module_data["estimated_hours"],
                difficulty_level=module_data["difficulty_level"],
                prerequisites=module_data.get("prerequisites", []),
                is_completed=module_data.get("is_completed", False),
                score=module_data.get("score"),
                time_spent_minutes=module_data.get("time_spent_minutes", 0),
                completed_at=datetime.fromisoformat(module_data["completed_at"]) if module_data.get("completed_at") else None
            )
            modules.append(module)
        
        return DomainLearningPath(
            id=str(self.id),
            student_id=str(self.student_id),
            grade=Grade(self.grade),
            modules=modules,
            learning_style=learning_style,
            created_at=self.created_at,
            updated_at=self.updated_at or self.created_at
        )
    
    @classmethod
    def from_domain(cls, learning_path: DomainLearningPath) -> 'LearningPathModel':
        """Create from domain entity"""
        # Serialize modules
        modules_data = []
        for module in learning_path.modules:
            module_data = {
                "id": module.id,
                "title": module.title,
                "subject": module.subject,
                "topic": module.topic,
                "content_type": module.content_type,
                "estimated_hours": module.estimated_hours,
                "difficulty_level": module.difficulty_level,
                "prerequisites": module.prerequisites,
                "is_completed": module.is_completed,
                "score": module.score,
                "time_spent_minutes": module.time_spent_minutes,
                "completed_at": module.completed_at.isoformat() if module.completed_at else None
            }
            modules_data.append(module_data)
        
        return cls(
            id=uuid.UUID(learning_path.id) if isinstance(learning_path.id, str) else learning_path.id,
            student_id=uuid.UUID(learning_path.student_id) if isinstance(learning_path.student_id, str) else learning_path.student_id,
            grade=learning_path.grade.level,
            learning_style=learning_path.learning_style.to_dict(),
            modules=modules_data,
            created_at=learning_path.created_at,
            updated_at=learning_path.updated_at
        )
    
    def update_from_domain(self, learning_path: DomainLearningPath):
        """Update model from domain entity"""
        # Serialize modules
        modules_data = []
        for module in learning_path.modules:
            module_data = {
                "id": module.id,
                "title": module.title,
                "subject": module.subject,
                "topic": module.topic,
                "content_type": module.content_type,
                "estimated_hours": module.estimated_hours,
                "difficulty_level": module.difficulty_level,
                "prerequisites": module.prerequisites,
                "is_completed": module.is_completed,
                "score": module.score,
                "time_spent_minutes": module.time_spent_minutes,
                "completed_at": module.completed_at.isoformat() if module.completed_at else None
            }
            modules_data.append(module_data)
        
        self.modules = modules_data
        self.updated_at = datetime.utcnow()


class QuizModel(Base):
    """SQLAlchemy model for Quiz entity"""
    __tablename__ = "quizzes"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    title = Column(String(255), nullable=False)
    topic = Column(String(100), nullable=False, index=True)
    questions = Column(JSON, nullable=False)
    difficulty = Column(String(50), nullable=False)
    time_limit_minutes = Column(Integer, nullable=False)
    passing_score = Column(Float, default=60.0)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    results = relationship("QuizResultModel", back_populates="quiz", cascade="all, delete-orphan")
    
    # Indexes
    __table_args__ = (
        Index("idx_quiz_topic", "topic"),
        Index("idx_quiz_difficulty", "difficulty"),
    )
    
    def to_domain(self) -> DomainQuiz:
        """Convert to domain entity"""
        # Parse questions
        questions = []
        for q_data in self.questions:
            question = DomainQuestion(
                id=q_data["id"],
                text=q_data["text"],
                options=q_data["options"],
                correct_answer=q_data["correct_answer"],
                points=q_data.get("points", 1.0),
                difficulty=q_data.get("difficulty", "medium"),
                topic=q_data.get("topic", self.topic),
                explanation=q_data.get("explanation")
            )
            questions.append(question)
        
        return DomainQuiz(
            id=str(self.id),
            title=self.title,
            topic=self.topic,
            questions=questions,
            difficulty=self.difficulty,
            time_limit_minutes=self.time_limit_minutes,
            passing_score=self.passing_score,
            created_at=self.created_at
        )
    
    @classmethod
    def from_domain(cls, quiz: DomainQuiz) -> 'QuizModel':
        """Create from domain entity"""
        # Serialize questions
        questions_data = []
        for question in quiz.questions:
            q_data = {
                "id": question.id,
                "text": question.text,
                "options": question.options,
                "correct_answer": question.correct_answer,
                "points": question.points,
                "difficulty": question.difficulty,
                "topic": question.topic,
                "explanation": question.explanation
            }
            questions_data.append(q_data)
        
        return cls(
            id=uuid.UUID(quiz.id) if isinstance(quiz.id, str) else quiz.id,
            title=quiz.title,
            topic=quiz.topic,
            questions=questions_data,
            difficulty=quiz.difficulty,
            time_limit_minutes=quiz.time_limit_minutes,
            passing_score=quiz.passing_score,
            created_at=quiz.created_at
        )


class QuizResultModel(Base):
    """SQLAlchemy model for Quiz Results"""
    __tablename__ = "quiz_results"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    student_id = Column(UUID(as_uuid=True), ForeignKey("students.id"), nullable=False)
    quiz_id = Column(UUID(as_uuid=True), ForeignKey("quizzes.id"), nullable=False)
    score = Column(Float, nullable=False)
    correct_answers = Column(Integer, nullable=False)
    total_questions = Column(Integer, nullable=False)
    time_spent_seconds = Column(Integer, nullable=False)
    responses = Column(JSON, nullable=False)
    completed_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    student = relationship("StudentModel", back_populates="quiz_results")
    quiz = relationship("QuizModel", back_populates="results")
    
    # Indexes
    __table_args__ = (
        Index("idx_quiz_result_student", "student_id"),
        Index("idx_quiz_result_quiz", "quiz_id"),
        Index("idx_quiz_result_completed", "completed_at"),
        UniqueConstraint("student_id", "quiz_id", "completed_at", name="uq_student_quiz_attempt"),
    )


class CurriculumModel(Base):
    """SQLAlchemy model for Curriculum"""
    __tablename__ = "curricula"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    grade = Column(Integer, unique=True, nullable=False)
    subjects = Column(JSON, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Indexes
    __table_args__ = (
        Index("idx_curriculum_grade", "grade"),
    )


class LearningModuleModel(Base):
    """SQLAlchemy model for Learning Modules"""
    __tablename__ = "learning_modules"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    title = Column(String(255), nullable=False)
    subject = Column(String(100), nullable=False, index=True)
    topic = Column(String(100), nullable=False, index=True)
    content_type = Column(String(50), nullable=False)
    content_url = Column(Text, nullable=True)
    estimated_hours = Column(Float, nullable=False)
    difficulty_level = Column(String(50), nullable=False)
    prerequisites = Column(JSON, default=list)
    metadata = Column(JSON, default=dict)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Indexes
    __table_args__ = (
        Index("idx_module_subject", "subject"),
        Index("idx_module_topic", "topic"),
        Index("idx_module_difficulty", "difficulty_level"),
    )


# Export all models
__all__ = [
    "Base",
    "StudentModel",
    "LearningPathModel",
    "QuizModel",
    "QuizResultModel",
    "CurriculumModel",
    "LearningModuleModel"
]
