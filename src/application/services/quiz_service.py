"""
Quiz Service - Application Layer
TEKNOFEST 2025 - Refactored from StudyBuddyAgent
"""

import logging
import random
from typing import List, Dict, Optional
from dataclasses import dataclass
from datetime import datetime

from src.domain.entities import Student, Quiz, Question
from src.domain.value_objects import AbilityLevel, QuizScore
from src.domain.interfaces import (
    IStudentRepository,
    IQuizRepository,
    IUnitOfWork
)
from src.shared.constants import QuizDifficulty, EducationConstants
from src.shared.exceptions import (
    StudentNotFoundException,
    DomainError,
    ValidationError
)

logger = logging.getLogger(__name__)


@dataclass
class QuizGenerationRequest:
    """DTO for quiz generation request"""
    student_id: str
    topic: str
    num_questions: int = EducationConstants.DEFAULT_QUIZ_QUESTIONS
    difficulty: Optional[str] = None  # If None, will be adaptive


@dataclass
class QuizSubmissionRequest:
    """DTO for quiz submission"""
    student_id: str
    quiz_id: str
    responses: Dict[str, str]  # question_id -> answer
    time_spent_seconds: int


@dataclass
class QuizResult:
    """DTO for quiz result"""
    quiz_id: str
    student_id: str
    score: float
    correct_answers: int
    total_questions: int
    performance_rating: str
    feedback: Dict[str, str]  # question_id -> feedback
    new_ability_level: float


class QuizService:
    """
    Service for quiz generation and evaluation
    Refactored from StudyBuddyAgent with Clean Code principles
    """
    
    def __init__(
        self,
        student_repository: IStudentRepository,
        quiz_repository: IQuizRepository,
        unit_of_work: IUnitOfWork
    ):
        self._student_repo = student_repository
        self._quiz_repo = quiz_repository
        self._uow = unit_of_work
        
        # Initialize question bank (in production, this would come from database)
        self._question_bank = self._initialize_question_bank()
    
    async def generate_adaptive_quiz(
        self,
        request: QuizGenerationRequest
    ) -> Quiz:
        """
        Generate an adaptive quiz based on student ability
        Replaces the old generate_adaptive_quiz method
        """
        # Validate request
        self._validate_quiz_request(request)
        
        # Get student
        student = await self._get_student(request.student_id)
        
        # Determine difficulty if not specified
        if request.difficulty:
            difficulty = request.difficulty
        else:
            difficulty = self._determine_difficulty(student.ability_level)
        
        # Get or generate quiz
        quiz = await self._quiz_repo.get_adaptive_quiz(
            topic=request.topic,
            student_ability=student.ability_level.value,
            num_questions=request.num_questions
        )
        
        # If no existing quiz, create new one
        if not quiz:
            quiz = self._create_quiz(
                topic=request.topic,
                difficulty=difficulty,
                num_questions=request.num_questions,
                student_ability=student.ability_level
            )
            
            # Save quiz
            async with self._uow:
                quiz = await self._quiz_repo.save(quiz)
                await self._uow.commit()
        
        logger.info(
            f"Generated {difficulty} quiz for student {request.student_id} on topic {request.topic}"
        )
        
        return quiz
    
    async def submit_quiz(
        self,
        request: QuizSubmissionRequest
    ) -> QuizResult:
        """
        Submit and evaluate a quiz
        """
        # Get student and quiz
        student = await self._get_student(request.student_id)
        quiz = await self._get_quiz(request.quiz_id)
        
        # Evaluate responses
        quiz_score = self._evaluate_quiz(
            quiz,
            request.responses,
            request.time_spent_seconds
        )
        
        # Generate feedback
        feedback = self._generate_feedback(quiz, request.responses)
        
        # Update student ability
        new_ability = self._update_student_ability(
            student.ability_level,
            quiz_score
        )
        student.update_ability_level(new_ability)
        
        # Save results
        async with self._uow:
            # Save quiz result
            await self._quiz_repo.save_quiz_result(
                student_id=request.student_id,
                quiz_id=request.quiz_id,
                score=quiz_score.percentage,
                time_spent=request.time_spent_seconds,
                responses=request.responses
            )
            
            # Update student
            await self._student_repo.save(student)
            
            await self._uow.commit()
        
        # Create result
        return QuizResult(
            quiz_id=request.quiz_id,
            student_id=request.student_id,
            score=quiz_score.percentage,
            correct_answers=quiz_score.correct_answers,
            total_questions=quiz_score.total_questions,
            performance_rating=quiz_score.performance_rating,
            feedback=feedback,
            new_ability_level=new_ability.value
        )
    
    async def get_quiz_history(
        self,
        student_id: str,
        limit: int = 50
    ) -> List[Dict]:
        """
        Get quiz history for a student
        """
        # Validate student exists
        await self._get_student(student_id)
        
        # Get history
        history = await self._quiz_repo.get_student_quiz_history(
            student_id=student_id,
            limit=limit
        )
        
        return history
    
    async def get_quiz_statistics(self, quiz_id: str) -> Dict:
        """
        Get statistics for a quiz
        """
        stats = await self._quiz_repo.get_quiz_statistics(quiz_id)
        
        if not stats:
            raise DomainError(f"No statistics available for quiz {quiz_id}")
        
        return stats
    
    def _validate_quiz_request(self, request: QuizGenerationRequest) -> None:
        """Validate quiz generation request"""
        if not request.topic:
            raise ValidationError("Topic is required", field="topic")
        
        if request.num_questions < EducationConstants.MIN_QUIZ_QUESTIONS:
            raise ValidationError(
                f"Minimum {EducationConstants.MIN_QUIZ_QUESTIONS} questions required",
                field="num_questions"
            )
        
        if request.num_questions > EducationConstants.MAX_QUIZ_QUESTIONS:
            raise ValidationError(
                f"Maximum {EducationConstants.MAX_QUIZ_QUESTIONS} questions allowed",
                field="num_questions"
            )
        
        if request.difficulty and request.difficulty not in [d.value for d in QuizDifficulty]:
            raise ValidationError(
                f"Invalid difficulty: {request.difficulty}",
                field="difficulty"
            )
    
    async def _get_student(self, student_id: str) -> Student:
        """Get student with error handling"""
        student = await self._student_repo.get_by_id(student_id)
        
        if not student:
            raise StudentNotFoundException(student_id)
        
        if not student.is_active:
            raise DomainError(f"Student {student_id} is not active")
        
        return student
    
    async def _get_quiz(self, quiz_id: str) -> Quiz:
        """Get quiz with error handling"""
        quiz = await self._quiz_repo.get_by_id(quiz_id)
        
        if not quiz:
            raise DomainError(f"Quiz {quiz_id} not found")
        
        return quiz
    
    def _determine_difficulty(self, ability_level: AbilityLevel) -> str:
        """Determine quiz difficulty based on student ability"""
        if ability_level.value < 0.3:
            return QuizDifficulty.EASY.value
        elif ability_level.value < 0.6:
            return QuizDifficulty.MEDIUM.value
        elif ability_level.value < 0.8:
            return QuizDifficulty.HARD.value
        else:
            return QuizDifficulty.EXPERT.value
    
    def _create_quiz(
        self,
        topic: str,
        difficulty: str,
        num_questions: int,
        student_ability: AbilityLevel
    ) -> Quiz:
        """Create a new quiz"""
        # Get questions from question bank
        questions = self._select_questions(
            topic=topic,
            difficulty=difficulty,
            num_questions=num_questions
        )
        
        # Create quiz
        quiz = Quiz.create_adaptive(
            topic=topic,
            student_ability=student_ability,
            num_questions=num_questions
        )
        
        # Add questions
        for question in questions:
            quiz.add_question(question)
        
        return quiz
    
    def _select_questions(
        self,
        topic: str,
        difficulty: str,
        num_questions: int
    ) -> List[Question]:
        """Select questions from question bank"""
        # Filter questions by topic and difficulty
        available_questions = [
            q for q in self._question_bank
            if q.topic.lower() == topic.lower() and q.difficulty == difficulty
        ]
        
        # If not enough questions, include adjacent difficulties
        if len(available_questions) < num_questions:
            difficulties = [d.value for d in QuizDifficulty]
            current_index = difficulties.index(difficulty)
            
            # Add easier questions if available
            if current_index > 0:
                easier_difficulty = difficulties[current_index - 1]
                available_questions.extend([
                    q for q in self._question_bank
                    if q.topic.lower() == topic.lower() and q.difficulty == easier_difficulty
                ])
            
            # Add harder questions if still not enough
            if len(available_questions) < num_questions and current_index < len(difficulties) - 1:
                harder_difficulty = difficulties[current_index + 1]
                available_questions.extend([
                    q for q in self._question_bank
                    if q.topic.lower() == topic.lower() and q.difficulty == harder_difficulty
                ])
        
        # Randomly select questions
        selected = random.sample(
            available_questions,
            min(num_questions, len(available_questions))
        )
        
        return selected
    
    def _evaluate_quiz(
        self,
        quiz: Quiz,
        responses: Dict[str, str],
        time_spent_seconds: int
    ) -> QuizScore:
        """Evaluate quiz responses"""
        correct_count = 0
        
        for question in quiz.questions:
            if question.is_correct(responses.get(question.id, "")):
                correct_count += 1
        
        return QuizScore(
            correct_answers=correct_count,
            total_questions=len(quiz.questions),
            time_spent_seconds=time_spent_seconds,
            difficulty=QuizDifficulty(quiz.difficulty)
        )
    
    def _generate_feedback(
        self,
        quiz: Quiz,
        responses: Dict[str, str]
    ) -> Dict[str, str]:
        """Generate feedback for each question"""
        feedback = {}
        
        for question in quiz.questions:
            user_answer = responses.get(question.id, "")
            
            if question.is_correct(user_answer):
                feedback[question.id] = "✅ Doğru cevap!"
            else:
                feedback[question.id] = (
                    f"❌ Yanlış cevap. Doğru cevap: {question.correct_answer}. "
                    f"{question.explanation if question.explanation else ''}"
                )
        
        return feedback
    
    def _update_student_ability(
        self,
        current_ability: AbilityLevel,
        quiz_score: QuizScore
    ) -> AbilityLevel:
        """Update student ability based on quiz performance"""
        # Calculate adjustment based on performance
        performance_factor = quiz_score.percentage / 100
        
        # Consider difficulty
        difficulty_multiplier = {
            QuizDifficulty.EASY: 0.5,
            QuizDifficulty.MEDIUM: 1.0,
            QuizDifficulty.HARD: 1.5,
            QuizDifficulty.EXPERT: 2.0
        }[quiz_score.difficulty]
        
        # Calculate adjustment
        if performance_factor >= 0.8:
            adjustment = 0.05 * difficulty_multiplier
        elif performance_factor >= 0.6:
            adjustment = 0.02 * difficulty_multiplier
        elif performance_factor >= 0.4:
            adjustment = 0
        else:
            adjustment = -0.02 * difficulty_multiplier
        
        return current_ability.adjust(adjustment)
    
    def _initialize_question_bank(self) -> List[Question]:
        """
        Initialize question bank with sample questions
        In production, this would come from database
        """
        questions = []
        
        # Sample questions for different topics and difficulties
        sample_data = [
            # Mathematics questions
            {
                "topic": "Matematik",
                "difficulty": QuizDifficulty.EASY.value,
                "questions": [
                    {
                        "text": "2 + 2 kaçtır?",
                        "options": ["3", "4", "5", "6"],
                        "correct": "4",
                        "explanation": "İki sayının toplamı"
                    },
                    {
                        "text": "5 x 3 kaçtır?",
                        "options": ["12", "15", "18", "20"],
                        "correct": "15",
                        "explanation": "Çarpma işlemi: 5 x 3 = 15"
                    }
                ]
            },
            {
                "topic": "Matematik",
                "difficulty": QuizDifficulty.MEDIUM.value,
                "questions": [
                    {
                        "text": "x² - 4 = 0 denkleminin kökleri nelerdir?",
                        "options": ["-2 ve 2", "-4 ve 4", "0 ve 4", "-2 ve 0"],
                        "correct": "-2 ve 2",
                        "explanation": "x² = 4, x = ±2"
                    }
                ]
            },
            # Physics questions
            {
                "topic": "Fizik",
                "difficulty": QuizDifficulty.EASY.value,
                "questions": [
                    {
                        "text": "Hız formülü nedir?",
                        "options": ["v = x/t", "v = x*t", "v = t/x", "v = x+t"],
                        "correct": "v = x/t",
                        "explanation": "Hız = Yol / Zaman"
                    }
                ]
            }
        ]
        
        # Create Question objects
        for subject_data in sample_data:
            for q_data in subject_data["questions"]:
                question = Question.create_multiple_choice(
                    text=q_data["text"],
                    options=q_data["options"],
                    correct_answer=q_data["correct"],
                    topic=subject_data["topic"],
                    difficulty=subject_data["difficulty"],
                    explanation=q_data.get("explanation")
                )
                questions.append(question)
        
        return questions
