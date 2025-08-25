"""
API Module Initialization
TEKNOFEST 2025 - V1 API Routers
"""

from src.presentation.api.v1.learning import router as learning_router
from src.presentation.api.v1.quiz import router as quiz_router
from src.presentation.api.v1.student import router as student_router
from src.presentation.api.v1.health import router as health_router

__all__ = [
    "learning_router",
    "quiz_router", 
    "student_router",
    "health_router"
]
