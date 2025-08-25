"""
Repository Interfaces - Domain Layer
TEKNOFEST 2025 - Repository Pattern Implementation
"""

from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any, Generic, TypeVar
from datetime import datetime

from src.domain.entities import (
    Student,
    LearningPath,
    Curriculum,
    Quiz,
    LearningModule
)
from src.domain.value_objects import Grade, LearningStyle

T = TypeVar('T')


class IRepository(ABC, Generic[T]):
    """Base repository interface for all entities"""
    
    @abstractmethod
    async def get_by_id(self, entity_id: str) -> Optional[T]:
        """Get entity by ID"""
        pass
    
    @abstractmethod
    async def get_all(self, limit: int = 100, offset: int = 0) -> List[T]:
        """Get all entities with pagination"""
        pass
    
    @abstractmethod
    async def save(self, entity: T) -> T:
        """Save entity (create or update)"""
        pass
    
    @abstractmethod
    async def delete(self, entity_id: str) -> bool:
        """Delete entity by ID"""
        pass
    
    @abstractmethod
    async def exists(self, entity_id: str) -> bool:
        """Check if entity exists"""
        pass


class IStudentRepository(IRepository[Student]):
    """Repository interface for Student entities"""
    
    @abstractmethod
    async def get_by_email(self, email: str) -> Optional[Student]:
        """Get student by email address"""
        pass
    
    @abstractmethod
    async def get_by_username(self, username: str) -> Optional[Student]:
        """Get student by username"""
        pass
    
    @abstractmethod
    async def get_by_grade(self, grade: Grade, limit: int = 100) -> List[Student]:
        """Get all students in a specific grade"""
        pass
    
    @abstractmethod
    async def get_active_students(self, limit: int = 100) -> List[Student]:
        """Get all active students"""
        pass
    
    @abstractmethod
    async def search(self, query: str, limit: int = 50) -> List[Student]:
        """Search students by name, email, or username"""
        pass
    
    @abstractmethod
    async def update_learning_style(self, student_id: str, learning_style: LearningStyle) -> Student:
        """Update student's learning style"""
        pass
    
    @abstractmethod
    async def get_students_needing_assessment(self, limit: int = 100) -> List[Student]:
        """Get students who need learning style assessment"""
        pass


class ILearningPathRepository(IRepository[LearningPath]):
    """Repository interface for LearningPath entities"""
    
    @abstractmethod
    async def get_by_student_id(self, student_id: str) -> Optional[LearningPath]:
        """Get learning path for a specific student"""
        pass
    
    @abstractmethod
    async def get_by_grade(self, grade: Grade) -> List[LearningPath]:
        """Get all learning paths for a specific grade"""
        pass
    
    @abstractmethod
    async def get_active_paths(self, limit: int = 100) -> List[LearningPath]:
        """Get active learning paths (with recent activity)"""
        pass
    
    @abstractmethod
    async def update_module_progress(
        self,
        path_id: str,
        module_id: str,
        is_completed: bool,
        score: Optional[float] = None
    ) -> LearningPath:
        """Update progress for a specific module in a path"""
        pass
    
    @abstractmethod
    async def get_paths_by_learning_style(
        self,
        learning_style: LearningStyle,
        limit: int = 100
    ) -> List[LearningPath]:
        """Get learning paths matching a learning style"""
        pass


class ICurriculumRepository(IRepository[Curriculum]):
    """Repository interface for Curriculum entities"""
    
    @abstractmethod
    async def get_by_grade(self, grade: Grade) -> Optional[Curriculum]:
        """Get curriculum for a specific grade"""
        pass
    
    @abstractmethod
    async def get_all_grades(self) -> List[Grade]:
        """Get list of all available grades"""
        pass
    
    @abstractmethod
    async def get_subject_topics(self, grade: Grade, subject: str) -> List[str]:
        """Get topics for a specific subject in a grade"""
        pass
    
    @abstractmethod
    async def search_topics(self, query: str) -> List[Dict[str, Any]]:
        """Search for topics across all curricula"""
        pass
    
    @abstractmethod
    async def get_prerequisites(self, grade: Grade, subject: str, topic: str) -> List[str]:
        """Get prerequisites for a specific topic"""
        pass


class IQuizRepository(IRepository[Quiz]):
    """Repository interface for Quiz entities"""
    
    @abstractmethod
    async def get_by_topic(self, topic: str, limit: int = 50) -> List[Quiz]:
        """Get quizzes for a specific topic"""
        pass
    
    @abstractmethod
    async def get_by_difficulty(self, difficulty: str, limit: int = 50) -> List[Quiz]:
        """Get quizzes by difficulty level"""
        pass
    
    @abstractmethod
    async def get_adaptive_quiz(
        self,
        topic: str,
        student_ability: float,
        num_questions: int = 10
    ) -> Quiz:
        """Get or generate an adaptive quiz based on student ability"""
        pass
    
    @abstractmethod
    async def save_quiz_result(
        self,
        student_id: str,
        quiz_id: str,
        score: float,
        time_spent: int,
        responses: Dict[str, str]
    ) -> None:
        """Save quiz result for a student"""
        pass
    
    @abstractmethod
    async def get_student_quiz_history(
        self,
        student_id: str,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Get quiz history for a student"""
        pass
    
    @abstractmethod
    async def get_quiz_statistics(self, quiz_id: str) -> Dict[str, Any]:
        """Get statistics for a specific quiz"""
        pass


class ILearningModuleRepository(IRepository[LearningModule]):
    """Repository interface for LearningModule entities"""
    
    @abstractmethod
    async def get_by_subject(self, subject: str) -> List[LearningModule]:
        """Get modules for a specific subject"""
        pass
    
    @abstractmethod
    async def get_by_topic(self, topic: str) -> List[LearningModule]:
        """Get modules for a specific topic"""
        pass
    
    @abstractmethod
    async def get_by_difficulty(self, difficulty: str) -> List[LearningModule]:
        """Get modules by difficulty level"""
        pass
    
    @abstractmethod
    async def get_prerequisites(self, module_id: str) -> List[LearningModule]:
        """Get prerequisite modules for a specific module"""
        pass
    
    @abstractmethod
    async def search(self, query: str, filters: Optional[Dict[str, Any]] = None) -> List[LearningModule]:
        """Search modules with optional filters"""
        pass
    
    @abstractmethod
    async def get_recommended(
        self,
        student: Student,
        limit: int = 10
    ) -> List[LearningModule]:
        """Get recommended modules for a student based on their profile"""
        pass


class IUnitOfWork(ABC):
    """Unit of Work pattern for transaction management"""
    
    students: IStudentRepository
    learning_paths: ILearningPathRepository
    curricula: ICurriculumRepository
    quizzes: IQuizRepository
    modules: ILearningModuleRepository
    
    @abstractmethod
    async def __aenter__(self):
        """Enter transaction context"""
        pass
    
    @abstractmethod
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit transaction context"""
        pass
    
    @abstractmethod
    async def commit(self):
        """Commit transaction"""
        pass
    
    @abstractmethod
    async def rollback(self):
        """Rollback transaction"""
        pass
