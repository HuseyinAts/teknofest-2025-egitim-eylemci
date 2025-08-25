"""
Learning API Endpoints - Clean Code Implementation
TEKNOFEST 2025 - Learning Management Routes
"""

import logging
from typing import List, Optional
from fastapi import APIRouter, Depends, status, Query, Path, Body
from fastapi.responses import JSONResponse

from src.infrastructure.config.container import get_learning_path_service
from src.application.services.learning_path_service import (
    LearningPathService,
    LearningStyleAnalysisRequest,
    LearningStyleAnalysisResult
)
from src.presentation.api.v1.schemas import (
    LearningStyleDetectionRequest,
    LearningStyleDetectionResponse,
    LearningPathRequest,
    LearningPathResponse,
    ModuleProgressRequest,
    ModuleProgressResponse,
    ErrorResponse
)
from src.shared.exceptions import ApplicationError

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/style/detect",
    response_model=LearningStyleDetectionResponse,
    status_code=status.HTTP_200_OK,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid request"},
        404: {"model": ErrorResponse, "description": "Student not found"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    },
    summary="Detect Learning Style",
    description="Analyze student responses to determine their dominant learning style (VARK model)"
)
async def detect_learning_style(
    request: LearningStyleDetectionRequest = Body(
        ...,
        example={
            "student_id": "550e8400-e29b-41d4-a716-446655440000",
            "responses": [
                "Görsel materyallerle öğrenmeyi tercih ederim",
                "Grafikler ve şemalar konuyu anlamamı kolaylaştırır",
                "Videolar ve animasyonlar ilgimi çeker"
            ]
        }
    ),
    service: LearningPathService = Depends(get_learning_path_service)
) -> LearningStyleDetectionResponse:
    """
    Detect student's learning style based on their responses.
    
    The analysis uses the VARK model to identify:
    - **Visual**: Learning through images, diagrams, and visual representations
    - **Auditory**: Learning through listening and verbal explanations
    - **Reading/Writing**: Learning through text and written materials
    - **Kinesthetic**: Learning through hands-on experience and practice
    
    Returns the primary and secondary learning styles with confidence scores.
    """
    try:
        # Create service request
        analysis_request = LearningStyleAnalysisRequest(
            student_id=request.student_id,
            responses=request.responses
        )
        
        # Call service
        result = await service.analyze_learning_style(analysis_request)
        
        # Convert to response
        return LearningStyleDetectionResponse.from_service_result(result)
        
    except ApplicationError as e:
        # Application errors are handled by middleware
        raise
    except Exception as e:
        logger.error(f"Unexpected error in detect_learning_style: {e}")
        raise


@router.post(
    "/path/create",
    response_model=LearningPathResponse,
    status_code=status.HTTP_201_CREATED,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid request"},
        404: {"model": ErrorResponse, "description": "Student or curriculum not found"},
        422: {"model": ErrorResponse, "description": "Student needs learning style assessment"}
    },
    summary="Create Personalized Learning Path",
    description="Create a personalized learning path for a student based on their learning style"
)
async def create_learning_path(
    request: LearningPathRequest = Body(...),
    service: LearningPathService = Depends(get_learning_path_service)
) -> LearningPathResponse:
    """
    Create a personalized learning path for a student.
    
    Requirements:
    - Student must exist and be active
    - Student must have completed learning style assessment
    - Grade must have available curriculum
    
    The path will be optimized based on:
    - Student's learning style
    - Current ability level
    - Grade curriculum requirements
    - Prerequisite dependencies
    """
    try:
        # Call service
        learning_path = await service.create_personalized_path(
            student_id=request.student_id,
            grade=request.grade
        )
        
        # Convert to response
        return LearningPathResponse.from_domain(learning_path)
        
    except ApplicationError as e:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in create_learning_path: {e}")
        raise


@router.put(
    "/progress/module",
    response_model=ModuleProgressResponse,
    status_code=status.HTTP_200_OK,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid request"},
        404: {"model": ErrorResponse, "description": "Student or module not found"}
    },
    summary="Update Module Progress",
    description="Update progress for a module in student's learning path"
)
async def update_module_progress(
    request: ModuleProgressRequest = Body(...),
    service: LearningPathService = Depends(get_learning_path_service)
) -> ModuleProgressResponse:
    """
    Update progress for a specific module in student's learning path.
    
    This endpoint:
    - Marks the module as completed
    - Records the score and time spent
    - Updates student's ability level based on performance
    - Returns updated learning path progress
    """
    try:
        # Call service
        updated_path = await service.update_module_progress(
            student_id=request.student_id,
            module_id=request.module_id,
            score=request.score,
            time_spent_minutes=request.time_spent_minutes
        )
        
        # Get progress
        progress = updated_path.progress
        
        # Create response
        return ModuleProgressResponse(
            student_id=request.student_id,
            module_id=request.module_id,
            is_completed=True,
            total_progress_percentage=progress.completion_percentage,
            completed_modules=progress.completed_modules,
            total_modules=progress.total_modules,
            current_ability_level=progress.current_level.value,
            estimated_completion_days=progress.estimated_completion_days
        )
        
    except ApplicationError as e:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in update_module_progress: {e}")
        raise


@router.get(
    "/path/{student_id}",
    response_model=LearningPathResponse,
    status_code=status.HTTP_200_OK,
    responses={
        404: {"model": ErrorResponse, "description": "Learning path not found"}
    },
    summary="Get Student Learning Path",
    description="Get the current learning path for a student"
)
async def get_learning_path(
    student_id: str = Path(..., description="Student UUID"),
    service: LearningPathService = Depends(get_learning_path_service)
) -> LearningPathResponse:
    """
    Retrieve the current learning path for a student.
    
    Returns:
    - Complete module list with progress
    - Current learning style configuration
    - Overall progress statistics
    """
    try:
        # Get learning path from repository
        path = await service._path_repo.get_by_student_id(student_id)
        
        if not path:
            from src.shared.exceptions import EntityNotFoundError
            raise EntityNotFoundError("LearningPath", student_id)
        
        # Convert to response
        return LearningPathResponse.from_domain(path)
        
    except ApplicationError as e:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in get_learning_path: {e}")
        raise


@router.get(
    "/curriculum/{grade}",
    status_code=status.HTTP_200_OK,
    summary="Get Curriculum",
    description="Get curriculum information for a specific grade"
)
async def get_curriculum(
    grade: int = Path(..., ge=9, le=12, description="Grade level (9-12)"),
    service: LearningPathService = Depends(get_learning_path_service)
):
    """
    Get curriculum information for a specific grade.
    
    Returns:
    - Available subjects
    - Topics for each subject
    - Total hours
    - Prerequisites
    """
    try:
        from src.domain.value_objects import Grade
        
        # Get curriculum
        curriculum = await service._curriculum_repo.get_by_grade(Grade(grade))
        
        if not curriculum:
            from src.shared.exceptions import CurriculumNotFoundException
            raise CurriculumNotFoundException(str(grade))
        
        # Convert to response
        return {
            "grade": grade,
            "subjects": [
                {
                    "name": subject.name,
                    "topics": subject.topics,
                    "total_hours": subject.total_hours,
                    "hours_per_topic": subject.hours_per_topic
                }
                for subject in curriculum.subjects.values()
            ],
            "total_hours": curriculum.total_hours
        }
        
    except ApplicationError as e:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in get_curriculum: {e}")
        raise
