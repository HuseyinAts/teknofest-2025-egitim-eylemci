"""
API Request/Response Schemas - Presentation Layer
TEKNOFEST 2025 - Pydantic Models for API
"""

from typing import Optional, List, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field, validator, EmailStr
import uuid

from src.shared.constants import LearningStyles


# Base Response Models
class ErrorResponse(BaseModel):
    """Standard error response"""
    error: Dict[str, Any] = Field(..., description="Error details")
    
    class Config:
        schema_extra = {
            "example": {
                "error": {
                    "code": "VALIDATION_ERROR",
                    "message": "Invalid input data",
                    "details": {"field": "student_id", "issue": "Invalid UUID format"}
                }
            }
        }


class SuccessResponse(BaseModel):
    """Standard success response"""
    success: bool = True
    message: Optional[str] = None
    data: Optional[Dict[str, Any]] = None


# Learning Style Models
class LearningStyleDetectionRequest(BaseModel):
    """Request for learning style detection"""
    student_id: str = Field(
        ...,
        description="Student UUID",
        regex="^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
    )
    responses: List[str] = Field(
        ...,
        min_items=1,
        max_items=100,
        description="Student responses to analyze"
    )
    
    @validator('responses', each_item=True)
    def validate_response_length(cls, v):
        if len(v.strip()) < 10:
            raise ValueError("Response must be at least 10 characters")
        if len(v) > 1000:
            raise ValueError("Response must not exceed 1000 characters")
        return v.strip()
    
    class Config:
        schema_extra = {
            "example": {
                "student_id": "550e8400-e29b-41d4-a716-446655440000",
                "responses": [
                    "Görsel materyallerle öğrenmeyi tercih ederim",
                    "Grafikler ve şemalar konuyu anlamamı kolaylaştırır"
                ]
            }
        }


class LearningStyleDetectionResponse(BaseModel):
    """Response for learning style detection"""
    student_id: str
    primary_style: str
    secondary_style: Optional[str]
    confidence: float = Field(..., ge=0, le=1)
    scores: Dict[str, float]
    recommendation: str
    analyzed_at: datetime = Field(default_factory=datetime.utcnow)
    
    @classmethod
    def from_service_result(cls, result):
        """Create from service result"""
        return cls(
            student_id=result.student_id,
            primary_style=result.primary_style,
            secondary_style=result.secondary_style,
            confidence=result.confidence,
            scores=result.scores,
            recommendation=result.recommendation
        )


# Learning Path Models
class LearningPathRequest(BaseModel):
    """Request for creating learning path"""
    student_id: str = Field(..., description="Student UUID")
    grade: int = Field(..., ge=9, le=12, description="Grade level")
    
    class Config:
        schema_extra = {
            "example": {
                "student_id": "550e8400-e29b-41d4-a716-446655440000",
                "grade": 9
            }
        }


class LearningModuleResponse(BaseModel):
    """Response for a learning module"""
    id: str
    title: str
    subject: str
    topic: str
    content_type: str
    estimated_hours: float
    difficulty_level: str
    is_completed: bool
    score: Optional[float]
    time_spent_minutes: int


class LearningPathResponse(BaseModel):
    """Response for learning path"""
    id: str
    student_id: str
    grade: int
    modules: List[LearningModuleResponse]
    total_modules: int
    completed_modules: int
    progress_percentage: float
    estimated_total_hours: float
    created_at: datetime
    updated_at: datetime
    
    @classmethod
    def from_domain(cls, learning_path):
        """Create from domain entity"""
        progress = learning_path.progress
        
        return cls(
            id=learning_path.id,
            student_id=learning_path.student_id,
            grade=learning_path.grade.level,
            modules=[
                LearningModuleResponse(
                    id=module.id,
                    title=module.title,
                    subject=module.subject,
                    topic=module.topic,
                    content_type=module.content_type,
                    estimated_hours=module.estimated_hours,
                    difficulty_level=module.difficulty_level,
                    is_completed=module.is_completed,
                    score=module.score,
                    time_spent_minutes=module.time_spent_minutes
                )
                for module in learning_path.modules
            ],
            total_modules=progress.total_modules,
            completed_modules=progress.completed_modules,
            progress_percentage=progress.completion_percentage,
            estimated_total_hours=learning_path.total_duration_hours,
            created_at=learning_path.created_at,
            updated_at=learning_path.updated_at
        )


# Module Progress Models
class ModuleProgressRequest(BaseModel):
    """Request for updating module progress"""
    student_id: str = Field(..., description="Student UUID")
    module_id: str = Field(..., description="Module UUID")
    score: float = Field(..., ge=0, le=100, description="Module score (0-100)")
    time_spent_minutes: int = Field(..., ge=0, description="Time spent in minutes")
    
    class Config:
        schema_extra = {
            "example": {
                "student_id": "550e8400-e29b-41d4-a716-446655440000",
                "module_id": "660e8400-e29b-41d4-a716-446655440001",
                "score": 85.5,
                "time_spent_minutes": 45
            }
        }


class ModuleProgressResponse(BaseModel):
    """Response for module progress update"""
    student_id: str
    module_id: str
    is_completed: bool
    total_progress_percentage: float
    completed_modules: int
    total_modules: int
    current_ability_level: float
    estimated_completion_days: Optional[int]


# Quiz Models
class QuizGenerationRequest(BaseModel):
    """Request for quiz generation"""
    student_id: str = Field(..., description="Student UUID")
    topic: str = Field(..., min_length=1, max_length=100, description="Quiz topic")
    num_questions: int = Field(default=10, ge=5, le=50, description="Number of questions")
    difficulty: Optional[str] = Field(None, regex="^(easy|medium|hard|expert)$")
    
    class Config:
        schema_extra = {
            "example": {
                "student_id": "550e8400-e29b-41d4-a716-446655440000",
                "topic": "Matematik",
                "num_questions": 10,
                "difficulty": "medium"
            }
        }


class QuizQuestionResponse(BaseModel):
    """Response for a quiz question"""
    id: str
    text: str
    options: List[str]
    points: float


class QuizResponse(BaseModel):
    """Response for generated quiz"""
    id: str
    title: str
    topic: str
    questions: List[QuizQuestionResponse]
    difficulty: str
    time_limit_minutes: int
    passing_score: float


class QuizSubmissionRequest(BaseModel):
    """Request for quiz submission"""
    student_id: str = Field(..., description="Student UUID")
    quiz_id: str = Field(..., description="Quiz UUID")
    responses: Dict[str, str] = Field(..., description="Question ID to answer mapping")
    time_spent_seconds: int = Field(..., ge=0, description="Time spent in seconds")
    
    class Config:
        schema_extra = {
            "example": {
                "student_id": "550e8400-e29b-41d4-a716-446655440000",
                "quiz_id": "770e8400-e29b-41d4-a716-446655440002",
                "responses": {
                    "q1": "option_a",
                    "q2": "option_c",
                    "q3": "option_b"
                },
                "time_spent_seconds": 600
            }
        }


class QuizResultResponse(BaseModel):
    """Response for quiz result"""
    quiz_id: str
    student_id: str
    score: float
    correct_answers: int
    total_questions: int
    performance_rating: str
    feedback: Dict[str, str]
    new_ability_level: float
    passed: bool


# Student Models
class StudentCreateRequest(BaseModel):
    """Request for creating a student"""
    email: EmailStr
    username: str = Field(..., min_length=3, max_length=50, regex="^[a-zA-Z0-9_-]+$")
    full_name: str = Field(..., min_length=2, max_length=100)
    grade: int = Field(..., ge=9, le=12)
    password: str = Field(..., min_length=8, max_length=100)
    
    @validator('password')
    def validate_password(cls, v):
        if not any(char.isdigit() for char in v):
            raise ValueError('Password must contain at least one digit')
        if not any(char.isupper() for char in v):
            raise ValueError('Password must contain at least one uppercase letter')
        if not any(char.islower() for char in v):
            raise ValueError('Password must contain at least one lowercase letter')
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "email": "student@example.com",
                "username": "student123",
                "full_name": "Ahmet Yılmaz",
                "grade": 9,
                "password": "SecurePass123!"
            }
        }


class StudentResponse(BaseModel):
    """Response for student data"""
    id: str
    email: str
    username: str
    full_name: str
    grade: int
    learning_style: Optional[Dict[str, Any]]
    ability_level: float
    status: str
    created_at: datetime
    updated_at: datetime


class StudentUpdateRequest(BaseModel):
    """Request for updating student"""
    full_name: Optional[str] = Field(None, min_length=2, max_length=100)
    grade: Optional[int] = Field(None, ge=9, le=12)
    
    class Config:
        schema_extra = {
            "example": {
                "full_name": "Ahmet Yılmaz",
                "grade": 10
            }
        }
