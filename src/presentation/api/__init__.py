"""
API Router Configuration - Presentation Layer
TEKNOFEST 2025 - Clean API Structure
"""

from fastapi import APIRouter

from src.presentation.api.v1 import (
    learning_router,
    quiz_router,
    student_router,
    health_router
)


def create_api_v1_router() -> APIRouter:
    """
    Create and configure API v1 router with all endpoints
    """
    api_router = APIRouter()
    
    # Include all routers
    api_router.include_router(
        health_router,
        prefix="/health",
        tags=["Health"]
    )
    
    api_router.include_router(
        learning_router,
        prefix="/learning",
        tags=["Learning Management"]
    )
    
    api_router.include_router(
        quiz_router,
        prefix="/quiz",
        tags=["Quiz Management"]
    )
    
    api_router.include_router(
        student_router,
        prefix="/students",
        tags=["Student Management"]
    )
    
    return api_router


__all__ = ["create_api_v1_router"]
