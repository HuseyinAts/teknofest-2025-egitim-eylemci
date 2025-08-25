"""
Quiz API Endpoints - Clean Code Implementation
TEKNOFEST 2025 - Quiz Management Routes
"""

import logging
from typing import List
from fastapi import APIRouter, Depends, status, Body

from src.infrastructure.config.container import get_quiz_service
from src.application.services.quiz_service import (
    QuizService,
    QuizGenerationRequest as ServiceQuizRequest,
    QuizSubmissionRequest as ServiceSubmissionRequest
)
from src.presentation.api.v1.schemas import (
    QuizGenerationRequest,
    QuizResponse,
    QuizQuestionResponse,
    QuizSubmissionRequest,
    QuizResultResponse,
    ErrorResponse
)
from src.shared.exceptions import ApplicationError

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/generate",
    response_model=QuizResponse,
    status_code=status.HTTP_201_CREATED,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid request"},
        404: {"model": ErrorResponse, "description": "Student not found"}
    },
    summary="Generate Adaptive Quiz",
    description="Generate an adaptive quiz based on student's ability level"
)
async def generate_quiz(
    request: QuizGenerationRequest = Body(...),
    service: QuizService = Depends(get_quiz_service)
) -> QuizResponse:
    """
    Generate an adaptive quiz for a student.
    
    The quiz difficulty is automatically adjusted based on:
    - Student's current ability level
    - Previous quiz performance
    - Topic complexity
    
    If no difficulty is specified, the system will select the appropriate level.
    """
    try:
        # Create service request
        service_request = ServiceQuizRequest(
            student_id=request.student_id,
            topic=request.topic,
            num_questions=request.num_questions,
            difficulty=request.difficulty
        )
        
        # Generate quiz
        quiz = await service.generate_adaptive_quiz(service_request)
        
        # Convert to response
        return QuizResponse(
            id=quiz.id,
            title=quiz.title,
            topic=quiz.topic,
            questions=[
                QuizQuestionResponse(
                    id=q.id,
                    text=q.text,
                    options=q.options,
                    points=q.points
                )
                for q in quiz.questions
            ],
            difficulty=quiz.difficulty,
            time_limit_minutes=quiz.time_limit_minutes,
            passing_score=quiz.passing_score
        )
        
    except ApplicationError as e:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in generate_quiz: {e}")
        raise


@router.post(
    "/submit",
    response_model=QuizResultResponse,
    status_code=status.HTTP_200_OK,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid request"},
        404: {"model": ErrorResponse, "description": "Student or quiz not found"}
    },
    summary="Submit Quiz Answers",
    description="Submit quiz answers and get evaluation results"
)
async def submit_quiz(
    request: QuizSubmissionRequest = Body(...),
    service: QuizService = Depends(get_quiz_service)
) -> QuizResultResponse:
    """
    Submit quiz answers for evaluation.
    
    This endpoint:
    - Evaluates the submitted answers
    - Calculates the score
    - Updates student's ability level
    - Provides detailed feedback for each question
    - Records the result in the database
    """
    try:
        # Create service request
        service_request = ServiceSubmissionRequest(
            student_id=request.student_id,
            quiz_id=request.quiz_id,
            responses=request.responses,
            time_spent_seconds=request.time_spent_seconds
        )
        
        # Submit quiz
        result = await service.submit_quiz(service_request)
        
        # Convert to response
        return QuizResultResponse(
            quiz_id=result.quiz_id,
            student_id=result.student_id,
            score=result.score,
            correct_answers=result.correct_answers,
            total_questions=result.total_questions,
            performance_rating=result.performance_rating,
            feedback=result.feedback,
            new_ability_level=result.new_ability_level,
            passed=result.score >= 60  # Passing score is 60%
        )
        
    except ApplicationError as e:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in submit_quiz: {e}")
        raise


@router.get(
    "/history/{student_id}",
    response_model=List[dict],
    status_code=status.HTTP_200_OK,
    responses={
        404: {"model": ErrorResponse, "description": "Student not found"}
    },
    summary="Get Quiz History",
    description="Get quiz history for a student"
)
async def get_quiz_history(
    student_id: str,
    limit: int = 50,
    service: QuizService = Depends(get_quiz_service)
):
    """
    Get quiz history for a student.
    
    Returns:
    - List of completed quizzes
    - Scores and performance metrics
    - Timestamps and duration
    """
    try:
        history = await service.get_quiz_history(student_id, limit)
        return history
        
    except ApplicationError as e:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in get_quiz_history: {e}")
        raise


@router.get(
    "/statistics/{quiz_id}",
    response_model=dict,
    status_code=status.HTTP_200_OK,
    responses={
        404: {"model": ErrorResponse, "description": "Quiz not found"}
    },
    summary="Get Quiz Statistics",
    description="Get statistics for a specific quiz"
)
async def get_quiz_statistics(
    quiz_id: str,
    service: QuizService = Depends(get_quiz_service)
):
    """
    Get statistics for a specific quiz.
    
    Returns:
    - Average score
    - Completion rate
    - Question-level statistics
    - Difficulty analysis
    """
    try:
        stats = await service.get_quiz_statistics(quiz_id)
        return stats
        
    except ApplicationError as e:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in get_quiz_statistics: {e}")
        raise
