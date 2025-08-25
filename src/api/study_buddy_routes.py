"""
Study Buddy API Routes - Production Ready
TEKNOFEST 2025 - AI-Powered Educational Assistant
"""

import logging
from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta
from fastapi import APIRouter, HTTPException, Depends, status, Query, BackgroundTasks
from pydantic import BaseModel, Field, validator
from sqlalchemy.ext.asyncio import AsyncSession

from src.database.session import get_async_session
from src.database.models import Quiz as QuizModel, Question, Answer, QuizAttempt, Student
from src.agents.study_buddy_agent_clean import study_buddy_agent
from src.core.authentication import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/v1/study-buddy",
    tags=["Study Buddy"],
    responses={404: {"description": "Not found"}}
)


# ==================== Request/Response Models ====================

class QuizGenerationRequest(BaseModel):
    """Quiz generation request"""
    topic: str = Field(..., min_length=1, max_length=200)
    grade: int = Field(10, ge=1, le=12)
    difficulty: float = Field(0.5, ge=0, le=1)
    question_count: int = Field(10, ge=1, le=50)
    question_types: List[str] = Field(
        default=['multiple_choice'],
        min_items=1
    )
    time_limit: int = Field(30, ge=5, le=180)  # minutes
    include_explanations: bool = True
    adaptive: bool = False
    
    @validator('question_types')
    def validate_question_types(cls, v):
        valid_types = {'multiple_choice', 'true_false', 'short_answer', 'essay', 'matching', 'fill_blank'}
        for qt in v:
            if qt not in valid_types:
                raise ValueError(f"Invalid question type: {qt}")
        return v


class AdaptiveQuizRequest(BaseModel):
    """Adaptive quiz generation request"""
    topic: str = Field(..., min_length=1)
    grade: int = Field(10, ge=1, le=12)
    initial_difficulty: float = Field(0.5, ge=0, le=1)
    question_count: int = Field(5, ge=3, le=20)
    adaptation_rate: float = Field(0.1, ge=0.05, le=0.3)


class QuestionAnswerRequest(BaseModel):
    """Question answering request"""
    question: str = Field(..., min_length=1)
    subject: str = Field("Genel")
    context: Optional[str] = None
    learning_style: str = Field("mixed", regex="^(visual|auditory|reading|kinesthetic|mixed)$")
    previous_qa: List[Dict] = Field(default_factory=list, max_items=10)
    requires_latex: bool = False
    response_mode: str = Field("detailed", regex="^(detailed|summary|hint_only|step_by_step)$")


class ConceptExplanationRequest(BaseModel):
    """Concept explanation request"""
    name: str = Field(..., min_length=1)
    subject: str = Field("Genel")
    grade: int = Field(10, ge=1, le=12)
    detail_level: str = Field("intermediate", regex="^(basic|intermediate|advanced)$")
    include_examples: bool = True
    include_visuals: bool = False


class ContentSummaryRequest(BaseModel):
    """Content summarization request"""
    text: str = Field(..., min_length=10, max_length=10000)
    max_length: int = Field(200, ge=50, le=1000)
    style: str = Field("paragraph", regex="^(paragraph|bullet_points|outline)$")
    language: str = Field("tr", regex="^(tr|en)$")


class ChatMessageRequest(BaseModel):
    """Chat message request"""
    message: str = Field(..., min_length=1, max_length=1000)
    session_id: Optional[str] = None


class HintRequest(BaseModel):
    """Hint request for problem solving"""
    question: str = Field(..., min_length=1)
    student_attempt: Optional[str] = None
    hint_level: int = Field(1, ge=1, le=5)
    subject: str = Field("Matematik")


class QuizSubmissionRequest(BaseModel):
    """Quiz submission with answers"""
    quiz_id: str = Field(..., min_length=1)
    answers: List[Dict[str, Any]] = Field(..., min_items=1)
    time_taken: int = Field(..., ge=0)  # seconds


# ==================== Quiz Generation ====================

@router.post("/quiz/generate")
async def generate_quiz(
    request: QuizGenerationRequest,
    background_tasks: BackgroundTasks,
    current_user=Depends(get_current_user)
):
    """Generate personalized quiz"""
    try:
        quiz_data = {
            'topic': request.topic,
            'grade': request.grade,
            'difficulty': request.difficulty,
            'question_count': request.question_count,
            'question_types': request.question_types,
            'time_limit': request.time_limit,
            'include_explanations': request.include_explanations,
            'student_id': str(current_user.id) if current_user else 'anonymous'
        }
        
        if request.adaptive:
            quiz = await study_buddy_agent.generate_adaptive_quiz(quiz_data)
        else:
            quiz = await study_buddy_agent.generate_quiz(quiz_data)
        
        # Save quiz to database in background
        if current_user:
            background_tasks.add_task(save_quiz_to_db, quiz, current_user.id)
        
        return {
            "success": True,
            "data": quiz,
            "message": "Quiz generated successfully"
        }
    except Exception as e:
        logger.error(f"Quiz generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/quiz/adaptive")
async def generate_adaptive_quiz(
    request: AdaptiveQuizRequest,
    current_user=Depends(get_current_user)
):
    """Generate adaptive quiz that adjusts difficulty based on performance"""
    try:
        quiz_data = {
            'topic': request.topic,
            'grade': request.grade,
            'difficulty': request.initial_difficulty,
            'question_count': request.question_count,
            'student_id': str(current_user.id) if current_user else 'anonymous'
        }
        
        quiz = await study_buddy_agent.generate_adaptive_quiz(quiz_data)
        
        return {
            "success": True,
            "data": quiz,
            "message": "Adaptive quiz generated successfully"
        }
    except Exception as e:
        logger.error(f"Adaptive quiz generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/quiz/submit")
async def submit_quiz(
    request: QuizSubmissionRequest,
    current_user=Depends(get_current_user)
):
    """Submit quiz answers and get results"""
    try:
        if not current_user:
            raise HTTPException(status_code=401, detail="Authentication required")
        
        # Calculate score
        correct_count = 0
        total_points = 0
        earned_points = 0
        
        async with get_async_session() as session:
            # Get quiz
            quiz = await session.query(QuizModel).filter_by(
                id=request.quiz_id
            ).first()
            
            if not quiz:
                raise HTTPException(status_code=404, detail="Quiz not found")
            
            # Create quiz attempt
            attempt = QuizAttempt(
                student_id=current_user.id,
                quiz_id=quiz.id,
                started_at=datetime.now() - timedelta(seconds=request.time_taken),
                completed_at=datetime.now(),
                time_spent=request.time_taken
            )
            
            # Process answers
            for answer_data in request.answers:
                question_id = answer_data['question_id']
                given_answer = answer_data['answer']
                
                # Get question
                question = await session.query(Question).filter_by(
                    id=question_id
                ).first()
                
                if question:
                    is_correct = given_answer == question.correct_answer
                    if is_correct:
                        correct_count += 1
                        earned_points += question.points
                    total_points += question.points
                    
                    # Save answer
                    answer = Answer(
                        student_id=current_user.id,
                        question_id=question_id,
                        quiz_id=quiz.id,
                        attempt_id=attempt.id,
                        given_answer=given_answer,
                        is_correct=is_correct,
                        points_earned=question.points if is_correct else 0,
                        time_taken=request.time_taken // request.question_count
                    )
                    session.add(answer)
            
            # Update attempt with score
            attempt.score = (earned_points / total_points) * 100 if total_points > 0 else 0
            attempt.points_earned = earned_points
            attempt.points_possible = total_points
            attempt.passed = attempt.score >= quiz.passing_score * 100
            attempt.is_completed = True
            
            session.add(attempt)
            await session.commit()
        
        return {
            "success": True,
            "data": {
                "quiz_id": request.quiz_id,
                "score": attempt.score,
                "correct_answers": correct_count,
                "total_questions": len(request.answers),
                "points_earned": earned_points,
                "total_points": total_points,
                "passed": attempt.passed,
                "time_taken": request.time_taken
            },
            "message": "Quiz submitted successfully"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Quiz submission error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Question Answering ====================

@router.post("/answer")
async def answer_question(
    request: QuestionAnswerRequest,
    current_user=Depends(get_current_user)
):
    """Answer student question with personalized response"""
    try:
        answer_data = {
            'question': request.question,
            'subject': request.subject,
            'context': request.context,
            'learning_style': request.learning_style,
            'previous_qa': request.previous_qa,
            'requires_latex': request.requires_latex
        }
        
        # Get student's learning style if authenticated
        if current_user:
            async with get_async_session() as session:
                student = await session.query(Student).filter_by(
                    user_id=current_user.id
                ).first()
                if student and student.learning_style:
                    answer_data['learning_style'] = student.learning_style
        
        response = await study_buddy_agent.answer_question(answer_data)
        
        return {
            "success": True,
            "data": response,
            "message": "Question answered successfully"
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Question answering error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Concept Explanation ====================

@router.post("/explain")
async def explain_concept(
    request: ConceptExplanationRequest,
    current_user=Depends(get_current_user)
):
    """Explain a concept in detail"""
    try:
        concept_data = {
            'name': request.name,
            'subject': request.subject,
            'grade': request.grade,
            'detail_level': request.detail_level
        }
        
        explanation = await study_buddy_agent.explain_concept(concept_data)
        
        return {
            "success": True,
            "data": explanation,
            "message": "Concept explained successfully"
        }
    except Exception as e:
        logger.error(f"Concept explanation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Content Summarization ====================

@router.post("/summarize")
async def summarize_content(
    request: ContentSummaryRequest,
    current_user=Depends(get_current_user)
):
    """Generate content summary"""
    try:
        summary_data = {
            'text': request.text,
            'max_length': request.max_length,
            'style': request.style
        }
        
        summary = await study_buddy_agent.generate_summary(summary_data)
        
        return {
            "success": True,
            "data": summary,
            "message": "Content summarized successfully"
        }
    except Exception as e:
        logger.error(f"Summarization error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Chat Sessions ====================

@router.post("/chat/start")
async def start_chat_session(
    current_user=Depends(get_current_user)
):
    """Start a new chat session"""
    try:
        student_id = str(current_user.id) if current_user else 'anonymous'
        session_id = await study_buddy_agent.start_chat_session(student_id)
        
        return {
            "success": True,
            "data": {
                "session_id": session_id,
                "status": "active",
                "created_at": datetime.now().isoformat()
            },
            "message": "Chat session started"
        }
    except Exception as e:
        logger.error(f"Chat session start error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/chat/message")
async def send_chat_message(
    request: ChatMessageRequest,
    current_user=Depends(get_current_user)
):
    """Send message in chat session"""
    try:
        if not request.session_id:
            # Start new session if not provided
            student_id = str(current_user.id) if current_user else 'anonymous'
            request.session_id = await study_buddy_agent.start_chat_session(student_id)
        
        response = await study_buddy_agent.send_message(
            request.session_id,
            request.message
        )
        
        return {
            "success": True,
            "data": {
                "session_id": request.session_id,
                "response": response
            }
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Chat message error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/chat/history/{session_id}")
async def get_chat_history(
    session_id: str,
    current_user=Depends(get_current_user)
):
    """Get chat history for a session"""
    try:
        history = await study_buddy_agent.get_chat_history(session_id)
        
        return {
            "success": True,
            "data": {
                "session_id": session_id,
                "messages": history,
                "total_messages": len(history)
            }
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Chat history error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/chat/end/{session_id}")
async def end_chat_session(
    session_id: str,
    current_user=Depends(get_current_user)
):
    """End a chat session"""
    try:
        summary = await study_buddy_agent.end_chat_session(session_id)
        
        return {
            "success": True,
            "data": summary,
            "message": "Chat session ended"
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Chat session end error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Hints and Help ====================

@router.post("/hint")
async def provide_hint(
    request: HintRequest,
    current_user=Depends(get_current_user)
):
    """Provide progressive hints for problem solving"""
    try:
        hint_data = {
            'question': request.question,
            'student_attempt': request.student_attempt,
            'hint_level': request.hint_level
        }
        
        hint = await study_buddy_agent.provide_hint(hint_data)
        
        return {
            "success": True,
            "data": hint,
            "message": "Hint provided"
        }
    except Exception as e:
        logger.error(f"Hint generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Question Bank ====================

@router.get("/questions/bank")
async def get_question_bank(
    subject: Optional[str] = Query(None),
    topic: Optional[str] = Query(None),
    limit: int = Query(10, ge=1, le=100)
):
    """Get questions from question bank"""
    try:
        question_bank = study_buddy_agent.question_bank
        
        questions = []
        for subj, topics in question_bank.items():
            if subject and subj != subject:
                continue
            
            for top, qs in topics.items():
                if topic and top != topic:
                    continue
                
                for q in qs[:limit]:
                    questions.append({
                        'subject': subj,
                        'topic': top,
                        'question': q['question'],
                        'options': q.get('options', []),
                        'difficulty': 0.5  # Default difficulty
                    })
        
        return {
            "success": True,
            "data": {
                "questions": questions[:limit],
                "total": len(questions)
            }
        }
    except Exception as e:
        logger.error(f"Question bank error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Subject Knowledge ====================

@router.get("/subjects")
async def get_subject_knowledge():
    """Get available subjects and their metadata"""
    try:
        knowledge = study_buddy_agent.subject_knowledge
        
        subjects = []
        for subject, data in knowledge.items():
            subjects.append({
                'name': subject,
                'topics': data.get('topics', []),
                'difficulty_range': data.get('difficulty_range', (0.3, 0.8)),
                'key_concepts': data.get('key_concepts', {})
            })
        
        return {
            "success": True,
            "data": {
                "subjects": subjects,
                "total": len(subjects)
            }
        }
    except Exception as e:
        logger.error(f"Subject knowledge error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Helper Functions ====================

async def save_quiz_to_db(quiz: Dict, user_id: int):
    """Save generated quiz to database"""
    try:
        async with get_async_session() as session:
            quiz_model = QuizModel(
                title=quiz.get('title', 'Generated Quiz'),
                description=quiz.get('description'),
                created_by_id=user_id,
                question_count=len(quiz.get('questions', [])),
                time_limit=quiz.get('time_limit', 30),
                difficulty=quiz.get('difficulty', 0.5),
                quiz_type='practice',
                is_published=True,
                is_active=True
            )
            session.add(quiz_model)
            
            # Add questions
            for q_data in quiz.get('questions', []):
                question = Question(
                    question_text=q_data.get('question'),
                    question_type=q_data.get('type', 'multiple_choice'),
                    options=q_data.get('options', []),
                    correct_answer=q_data.get('correct_answer'),
                    points=q_data.get('points', 10),
                    difficulty=q_data.get('difficulty', 0.5),
                    subject=quiz.get('subject'),
                    topic=quiz.get('topic')
                )
                session.add(question)
                quiz_model.questions.append(question)
            
            await session.commit()
            logger.info(f"Quiz saved to database: {quiz_model.id}")
    except Exception as e:
        logger.error(f"Error saving quiz to database: {e}")