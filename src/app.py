"""
Main Application Entry Point with Dependency Injection
TEKNOFEST 2025 - Eğitim Teknolojileri
"""

import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.config import Settings, get_settings, validate_environment
from src.factory import ServiceFactory, get_factory
from src.agents.learning_path_agent_v2 import LearningPathAgent
from src.agents.study_buddy_agent_clean import StudyBuddyAgent
from src.data_processor import DataProcessor
from src.model_integration_optimized import ModelIntegration
from src.api.irt_routes import router as irt_router
from src.api.gamification_routes import router as gamification_router
from src.api.offline_routes import router as offline_router
from src.api.auth_routes import router as auth_router
from src.core.offline_support import OfflineManager, OfflineMiddleware
from src.core.security import SecurityMiddleware, SQLInjectionProtection
from src.core.authentication import jwt_auth, get_current_user, UserLogin, UserRegister, TokenResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QuizRequest(BaseModel):
    """Quiz generation request model"""
    topic: str
    student_ability: float = 0.5
    num_questions: int = 10


class LearningStyleRequest(BaseModel):
    """Learning style detection request"""
    student_responses: list[str]


class TextGenerationRequest(BaseModel):
    """Text generation request"""
    prompt: str
    max_length: int = 200


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    logger.info("Starting application...")
    
    try:
        validate_environment()
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        if get_settings().is_production():
            raise
    
    factory = get_factory()
    factory.initialize()
    
    logger.info("Application started successfully")
    
    yield
    
    logger.info("Shutting down application...")


def create_app() -> FastAPI:
    """Create FastAPI application with dependency injection"""
    settings = get_settings()
    
    app = FastAPI(
        title="TEKNOFEST 2025 - Eğitim Teknolojileri API",
        version=settings.app_version,
        debug=settings.app_debug,
        lifespan=lifespan
    )
    
    # SECURITY: Add security middleware first
    app.add_middleware(SecurityMiddleware, settings=settings)
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=settings.cors_allow_credentials,
        allow_methods=settings.cors_allow_methods,
        allow_headers=settings.cors_allow_headers,
    )
    
    # Add offline support middleware
    from src.database.session import SessionLocal
    db_session = SessionLocal()
    offline_manager = OfflineManager(db_session)
    app.add_middleware(OfflineMiddleware, offline_manager=offline_manager)
    
    return app


app = create_app()

# Include routers
app.include_router(auth_router)  # Auth routes first for security
app.include_router(irt_router)
app.include_router(gamification_router)
app.include_router(offline_router)


def get_learning_path_agent() -> LearningPathAgent:
    """Dependency injection for LearningPathAgent"""
    factory = get_factory()
    with factory.create_scope() as scope:
        return scope.get_service(LearningPathAgent)


def get_study_buddy_agent() -> StudyBuddyAgent:
    """Dependency injection for StudyBuddyAgent"""
    factory = get_factory()
    with factory.create_scope() as scope:
        return scope.get_service(StudyBuddyAgent)


def get_model_integration() -> ModelIntegration:
    """Dependency injection for ModelIntegration"""
    return get_factory().create_service(ModelIntegration)


def get_data_processor() -> DataProcessor:
    """Dependency injection for DataProcessor"""
    return get_factory().create_service(DataProcessor)


@app.get("/")
async def root():
    """Root endpoint"""
    settings = get_settings()
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "environment": settings.app_env.value,
        "status": "healthy"
    }


@app.get("/health")
async def health_check(settings: Settings = Depends(get_settings)):
    """Health check endpoint"""
    issues = settings.validate_production_ready()
    
    return {
        "status": "healthy" if not issues else "warning",
        "environment": settings.app_env.value,
        "debug": settings.app_debug,
        "issues": issues
    }


@app.post("/api/v1/learning-style")
async def detect_learning_style(
    request: LearningStyleRequest,
    agent: LearningPathAgent = Depends(get_learning_path_agent)
):
    """Detect student learning style"""
    try:
        result = agent.detect_learning_style(request.student_responses)
        return {"success": True, "data": result}
    except Exception as e:
        logger.error(f"Learning style detection error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/generate-quiz")
async def generate_quiz(
    request: QuizRequest,
    agent: StudyBuddyAgent = Depends(get_study_buddy_agent)
):
    """Generate adaptive quiz"""
    try:
        quiz = agent.generate_adaptive_quiz(
            topic=request.topic,
            student_ability=request.student_ability,
            num_questions=request.num_questions
        )
        return {"success": True, "data": quiz}
    except Exception as e:
        logger.error(f"Quiz generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/generate-text")
async def generate_text(
    request: TextGenerationRequest,
    model: ModelIntegration = Depends(get_model_integration)
):
    """Generate text using AI model"""
    try:
        text = model.generate(
            prompt=request.prompt,
            max_length=request.max_length
        )
        return {"success": True, "data": {"generated_text": text}}
    except Exception as e:
        logger.error(f"Text generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/curriculum/{grade}")
async def get_curriculum(
    grade: str,
    agent: LearningPathAgent = Depends(get_learning_path_agent)
):
    """Get curriculum for a grade"""
    try:
        curriculum = agent.curriculum.get(grade)
        if not curriculum:
            raise HTTPException(status_code=404, detail=f"Curriculum not found for grade {grade}")
        return {"success": True, "data": curriculum}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Curriculum fetch error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/data/stats")
async def get_data_stats(
    processor: DataProcessor = Depends(get_data_processor)
):
    """Get data processing statistics"""
    try:
        stats = {
            "data_dir": str(processor.data_dir),
            "raw_dir": str(processor.raw_dir),
            "processed_dir": str(processor.processed_dir),
            "exists": processor.data_dir.exists()
        }
        return {"success": True, "data": stats}
    except Exception as e:
        logger.error(f"Data stats error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    
    settings = get_settings()
    
    uvicorn.run(
        "src.app:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.api_reload,
        workers=settings.api_workers
    )