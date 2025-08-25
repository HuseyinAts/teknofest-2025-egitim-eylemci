# -*- coding: utf-8 -*-
"""
FastAPI Server for TEKNOFEST Education Assistant
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict
import uvicorn
import sys
import os
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Initialize FastAPI app
app = FastAPI(
    title="TEKNOFEST Education Assistant API",
    description="Personalized Turkish Education Assistant API",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request Models
class StudentRequest(BaseModel):
    student_id: str = Field(..., description="Student ID")
    topic: str = Field(..., description="Topic")
    grade_level: int = Field(..., ge=1, le=12, description="Grade level")

class QuizRequest(BaseModel):
    topic: str
    student_ability: Optional[float] = 0.5
    num_questions: Optional[int] = 10

class StudyPlanRequest(BaseModel):
    student_id: str
    weak_topics: List[str]
    available_hours: int = Field(..., ge=1, le=100)

# API Endpoints
@app.get("/")
async def root():
    """API Home"""
    return {
        "message": "TEKNOFEST 2025 Education Assistant API",
        "version": "1.0.0",
        "endpoints": [
            "/docs - API Documentation",
            "/health - Health check",
            "/learning-path - Generate learning path",
            "/quiz - Generate adaptive quiz",
            "/study-plan - Generate study plan"
        ]
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    }

@app.post("/learning-path")
async def create_learning_path(request: StudentRequest):
    """Create personalized learning path"""
    try:
        # Import agent
        try:
            from src.agents.learning_path_agent_v2 import LearningPathAgent
        except ImportError:
            from src.agents.learning_path_agent import LearningPathAgent
        
        agent = LearningPathAgent()
        
        # Create student profile
        profile = {
            'student_id': request.student_id,
            'learning_style': 'visual',
            'current_level': 0.3,
            'target_level': 0.9,
            'grade': request.grade_level
        }
        
        # Generate learning path
        path = agent.generate_learning_path(
            student_profile=profile,
            topic=request.topic,
            weeks=4
        )
        
        return {
            "success": True,
            "message": "Learning path created successfully",
            "data": path
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/quiz")
async def generate_quiz(request: QuizRequest):
    """Generate adaptive quiz"""
    try:
        # Import agent
        try:
            from src.agents.study_buddy_agent_clean import StudyBuddyAgent
        except ImportError:
            try:
                from src.agents.study_buddy_agent import StudyBuddyAgent
            except:
                raise ImportError("Study buddy agent not found")
        
        agent = StudyBuddyAgent()
        quiz = agent.generate_adaptive_quiz(
            topic=request.topic,
            student_ability=request.student_ability,
            num_questions=request.num_questions
        )
        
        # Generate quiz ID
        quiz_id = f"quiz_{request.topic}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        return {
            "success": True,
            "quiz_id": quiz_id,
            "topic": request.topic,
            "questions": quiz,
            "total_questions": len(quiz)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/study-plan")
async def generate_study_plan(request: StudyPlanRequest):
    """Generate study plan"""
    try:
        # Import agent
        try:
            from src.agents.study_buddy_agent_clean import StudyBuddyAgent
        except ImportError:
            try:
                from src.agents.study_buddy_agent import StudyBuddyAgent
            except:
                raise ImportError("Study buddy agent not found")
        
        agent = StudyBuddyAgent()
        plan = agent.generate_study_plan(
            weak_topics=request.weak_topics,
            available_hours=request.available_hours
        )
        
        return {
            "success": True,
            "student_id": request.student_id,
            "study_plan": plan
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run server
if __name__ == "__main__":
    print("Starting TEKNOFEST Education API Server...")
    print("API Documentation: http://localhost:8000/docs")
    print("Health Check: http://localhost:8000/health")
    
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        reload=False,
        log_level="info"
    )