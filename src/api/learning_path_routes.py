"""
Learning Path API Routes - Production Ready
TEKNOFEST 2025 - Personalized Education System
"""

import logging
from typing import List, Dict, Optional, Any
from datetime import datetime
from fastapi import APIRouter, HTTPException, Depends, status, Query
from pydantic import BaseModel, Field, validator
from sqlalchemy.ext.asyncio import AsyncSession

from src.database.session import get_async_session
from src.database.models import Student, LearningPath as LearningPathModel, Progress
from src.agents.learning_path_agent_v2 import learning_path_agent
from src.core.authentication import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/v1/learning-path",
    tags=["Learning Path"],
    responses={404: {"description": "Not found"}}
)


# ==================== Request/Response Models ====================

class StudentProfileRequest(BaseModel):
    """Student profile for learning path creation"""
    name: str = Field(..., min_length=1, max_length=100)
    grade: int = Field(..., ge=1, le=12)
    age: int = Field(..., ge=5, le=25)
    current_level: float = Field(0.5, ge=0, le=1)
    target_level: float = Field(0.8, ge=0, le=1)
    learning_style: str = Field("mixed", regex="^(visual|auditory|reading|kinesthetic|mixed)$")
    learning_pace: str = Field("moderate", regex="^(slow|moderate|fast)$")
    weak_topics: List[str] = Field(default_factory=list)
    strong_topics: List[str] = Field(default_factory=list)
    study_hours_per_day: float = Field(2.0, gt=0, le=12)
    exam_target: str = Field("YKS")
    exam_date: Optional[str] = None
    preferences: Dict[str, Any] = Field(default_factory=dict)

    @validator('target_level')
    def target_must_be_greater(cls, v, values):
        if 'current_level' in values and v <= values['current_level']:
            raise ValueError('Target level must be greater than current level')
        return v


class LearningStyleRequest(BaseModel):
    """Learning style detection request"""
    responses: List[str] = Field(..., min_items=1, max_items=20)


class ProgressUpdateRequest(BaseModel):
    """Progress update request"""
    completed_topics: List[str] = Field(default_factory=list)
    quiz_scores: List[float] = Field(default_factory=list)
    study_time_minutes: int = Field(0, ge=0)
    current_topic: Optional[str] = None


class RecommendationRequest(BaseModel):
    """Recommendation request parameters"""
    include_content: bool = True
    include_methods: bool = True
    include_schedule: bool = True
    max_recommendations: int = Field(5, ge=1, le=20)


class ContentRequest(BaseModel):
    """Personalized content request"""
    topic: str = Field(..., min_length=1)
    preferred_format: Optional[str] = None
    time_available: int = Field(60, ge=10, le=300)


class ZPDCalculationRequest(BaseModel):
    """Zone of Proximal Development calculation request"""
    current_level: float = Field(..., ge=0, le=1)
    target_level: float = Field(..., ge=0, le=1)
    weeks: int = Field(..., ge=1, le=52)


# ==================== VARK Learning Style Detection ====================

@router.get("/vark-questions")
async def get_vark_questions():
    """Get VARK learning style assessment questions"""
    try:
        questions = learning_path_agent.vark_quiz
        return {
            "success": True,
            "data": {
                "questions": questions,
                "total_questions": len(questions),
                "instructions": "Lütfen her soruyu dikkatlice okuyun ve size en uygun seçeneği işaretleyin"
            }
        }
    except Exception as e:
        logger.error(f"Error fetching VARK questions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/detect-learning-style")
async def detect_learning_style(
    request: LearningStyleRequest,
    current_user=Depends(get_current_user)
):
    """Detect student's learning style from questionnaire responses"""
    try:
        result = learning_path_agent.detect_learning_style(request.responses)
        
        # Save to database if user is authenticated
        if current_user:
            async with get_async_session() as session:
                student = await session.query(Student).filter_by(
                    user_id=current_user.id
                ).first()
                if student:
                    student.learning_style = result['dominant_style']
                    await session.commit()
        
        return {"success": True, "data": result}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Learning style detection error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Learning Path Creation ====================

@router.post("/create")
async def create_learning_path(
    profile: StudentProfileRequest,
    current_user=Depends(get_current_user)
):
    """Create personalized learning path for student"""
    try:
        # Add student_id from authenticated user
        profile_dict = profile.dict()
        if current_user:
            profile_dict['student_id'] = str(current_user.id)
        
        # Create learning path
        learning_path = await learning_path_agent.create_learning_path(profile_dict)
        
        # Save to database
        async with get_async_session() as session:
            path_model = LearningPathModel(
                student_id=current_user.id if current_user else None,
                title=f"Learning Path - {profile.exam_target}",
                weekly_plans=learning_path['weekly_plans'],
                milestones=learning_path['milestones'],
                assessment_schedule=learning_path['assessment_schedule'],
                total_weeks=learning_path['total_weeks']
            )
            session.add(path_model)
            await session.commit()
            await session.refresh(path_model)
            
            learning_path['id'] = path_model.id
        
        return {
            "success": True,
            "data": learning_path,
            "message": "Learning path created successfully"
        }
    except Exception as e:
        logger.error(f"Learning path creation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{path_id}")
async def get_learning_path(
    path_id: str,
    current_user=Depends(get_current_user)
):
    """Get learning path details"""
    try:
        async with get_async_session() as session:
            path = await session.query(LearningPathModel).filter_by(
                path_id=path_id
            ).first()
            
            if not path:
                raise HTTPException(status_code=404, detail="Learning path not found")
            
            # Check ownership
            if current_user and path.student_id != current_user.id:
                raise HTTPException(status_code=403, detail="Access denied")
            
            return {
                "success": True,
                "data": {
                    "path_id": path.path_id,
                    "student_id": path.student_id,
                    "weekly_plans": path.weekly_plans,
                    "milestones": path.milestones,
                    "progress": path.progress_percentage,
                    "current_week": path.current_week,
                    "total_weeks": path.total_weeks,
                    "is_active": path.is_active
                }
            }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching learning path: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Progress Management ====================

@router.post("/progress/update")
async def update_progress(
    request: ProgressUpdateRequest,
    current_user=Depends(get_current_user)
):
    """Update student progress"""
    try:
        progress_data = {
            'student_id': str(current_user.id) if current_user else 'anonymous',
            'completed_topics': request.completed_topics,
            'quiz_scores': request.quiz_scores
        }
        
        result = await learning_path_agent.update_progress(progress_data)
        
        # Save to database
        if current_user:
            async with get_async_session() as session:
                progress = Progress(
                    student_id=current_user.id,
                    completed_topics=request.completed_topics,
                    quiz_scores=request.quiz_scores,
                    average_score=result['average_score'],
                    current_level=result['new_level'],
                    time_spent=request.study_time_minutes
                )
                session.add(progress)
                await session.commit()
        
        return {"success": True, "data": result}
    except Exception as e:
        logger.error(f"Progress update error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/progress/report")
async def get_progress_report(
    current_user=Depends(get_current_user)
):
    """Get comprehensive progress report"""
    try:
        if not current_user:
            raise HTTPException(status_code=401, detail="Authentication required")
        
        report = await learning_path_agent.get_progress_report(str(current_user.id))
        
        return {"success": True, "data": report}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Progress report error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Recommendations ====================

@router.post("/recommendations")
async def get_recommendations(
    request: RecommendationRequest,
    current_user=Depends(get_current_user)
):
    """Get personalized study recommendations"""
    try:
        # Get student profile
        profile = {}
        if current_user:
            async with get_async_session() as session:
                student = await session.query(Student).filter_by(
                    user_id=current_user.id
                ).first()
                if student:
                    profile = {
                        'learning_style': student.learning_style,
                        'current_level': student.current_level,
                        'weak_topics': student.weak_topics
                    }
        
        recommendations = await learning_path_agent.get_recommendations(profile)
        
        # Filter based on request
        filtered = []
        for rec in recommendations:
            if rec['type'] == 'content' and not request.include_content:
                continue
            if rec['type'] == 'method' and not request.include_methods:
                continue
            if rec['type'] == 'schedule' and not request.include_schedule:
                continue
            filtered.append(rec)
        
        # Limit recommendations
        filtered = filtered[:request.max_recommendations]
        
        return {
            "success": True,
            "data": {
                "recommendations": filtered,
                "total": len(filtered),
                "generated_at": datetime.now().isoformat()
            }
        }
    except Exception as e:
        logger.error(f"Recommendations error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Curriculum and Content ====================

@router.get("/curriculum/{grade}/{subject}")
async def get_curriculum_topics(
    grade: int = Query(..., ge=1, le=12),
    subject: str = Query(..., regex="^(Matematik|Fizik|Kimya|Biyoloji|Tarih|Coğrafya)$")
):
    """Get curriculum topics for specific grade and subject"""
    try:
        topics = learning_path_agent.get_curriculum_topics(grade, subject)
        
        if not topics:
            raise HTTPException(
                status_code=404,
                detail=f"No curriculum found for grade {grade}, subject {subject}"
            )
        
        return {
            "success": True,
            "data": {
                "grade": grade,
                "subject": subject,
                "topics": topics,
                "total_topics": len(topics)
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Curriculum fetch error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/content/personalized")
async def get_personalized_content(
    request: ContentRequest,
    current_user=Depends(get_current_user)
):
    """Get personalized content for a specific topic"""
    try:
        # Get student profile
        profile = {'learning_style': 'mixed'}
        if current_user:
            async with get_async_session() as session:
                student = await session.query(Student).filter_by(
                    user_id=current_user.id
                ).first()
                if student:
                    profile['learning_style'] = student.learning_style
        
        content = await learning_path_agent.get_personalized_content(
            profile,
            request.topic
        )
        
        return {"success": True, "data": content}
    except Exception as e:
        logger.error(f"Personalized content error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Zone of Proximal Development ====================

@router.post("/zpd/calculate")
async def calculate_zpd_levels(request: ZPDCalculationRequest):
    """Calculate Zone of Proximal Development progression levels"""
    try:
        levels = learning_path_agent.calculate_zpd_level(
            request.current_level,
            request.target_level,
            request.weeks
        )
        
        return {
            "success": True,
            "data": {
                "current_level": request.current_level,
                "target_level": request.target_level,
                "weeks": request.weeks,
                "progression_levels": levels,
                "weekly_increase": [
                    levels[i] - levels[i-1] if i > 0 else levels[0] - request.current_level
                    for i in range(len(levels))
                ]
            }
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"ZPD calculation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Learning Strategies ====================

@router.get("/strategies/{learning_style}")
async def get_learning_strategies(
    learning_style: str = Query(..., regex="^(visual|auditory|reading|kinesthetic|mixed)$")
):
    """Get learning strategies for specific learning style"""
    try:
        strategies = learning_path_agent.learning_strategies.get(learning_style, {})
        
        if not strategies:
            raise HTTPException(
                status_code=404,
                detail=f"No strategies found for learning style: {learning_style}"
            )
        
        return {
            "success": True,
            "data": {
                "learning_style": learning_style,
                "strategies": strategies
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Learning strategies error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Batch Operations ====================

@router.post("/batch/create-paths")
async def create_multiple_paths(
    profiles: List[StudentProfileRequest],
    current_user=Depends(get_current_user)
):
    """Create learning paths for multiple students (admin only)"""
    try:
        # Check admin permission
        if not current_user or current_user.role != 'admin':
            raise HTTPException(status_code=403, detail="Admin access required")
        
        results = []
        errors = []
        
        for i, profile in enumerate(profiles):
            try:
                profile_dict = profile.dict()
                path = await learning_path_agent.create_learning_path(profile_dict)
                results.append({
                    "index": i,
                    "success": True,
                    "path_id": path['path_id']
                })
            except Exception as e:
                errors.append({
                    "index": i,
                    "error": str(e),
                    "profile": profile.dict()
                })
        
        return {
            "success": len(errors) == 0,
            "data": {
                "created": len(results),
                "failed": len(errors),
                "results": results,
                "errors": errors
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch creation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))