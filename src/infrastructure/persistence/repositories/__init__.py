"""
Repository Implementations - Infrastructure Layer
TEKNOFEST 2025 - SQLAlchemy Repository Pattern Implementation
"""

import logging
from typing import Optional, List, Dict, Any
from datetime import datetime

from sqlalchemy import select, update, delete, and_, or_, func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload, joinedload

from src.domain.entities import (
    Student,
    LearningPath,
    Curriculum,
    Quiz,
    LearningModule,
    Subject
)
from src.domain.value_objects import Grade, LearningStyle, AbilityLevel
from src.domain.interfaces import (
    IStudentRepository,
    ILearningPathRepository,
    ICurriculumRepository,
    IQuizRepository,
    ILearningModuleRepository
)
from src.shared.exceptions import (
    EntityNotFoundError,
    DuplicateEntityError,
    RepositoryError
)
from src.infrastructure.persistence.models import (
    StudentModel,
    LearningPathModel,
    CurriculumModel,
    QuizModel,
    LearningModuleModel
)

logger = logging.getLogger(__name__)


class BaseRepository:
    """Base repository with common operations"""
    
    def __init__(self, session: AsyncSession, model_class):
        self._session = session
        self._model_class = model_class
    
    async def _execute_query(self, query):
        """Execute a query with error handling"""
        try:
            result = await self._session.execute(query)
            return result
        except Exception as e:
            logger.error(f"Database query error: {e}")
            raise RepositoryError(f"Database operation failed: {str(e)}")
    
    async def _commit(self):
        """Commit transaction with error handling"""
        try:
            await self._session.commit()
        except Exception as e:
            await self._session.rollback()
            logger.error(f"Database commit error: {e}")
            raise RepositoryError(f"Failed to save data: {str(e)}")


class StudentRepository(BaseRepository, IStudentRepository):
    """SQLAlchemy implementation of Student Repository"""
    
    def __init__(self, session: AsyncSession):
        super().__init__(session, StudentModel)
    
    async def get_by_id(self, entity_id: str) -> Optional[Student]:
        """Get student by ID"""
        query = select(StudentModel).where(StudentModel.id == entity_id)
        result = await self._execute_query(query)
        model = result.scalar_one_or_none()
        
        return model.to_domain() if model else None
    
    async def get_by_email(self, email: str) -> Optional[Student]:
        """Get student by email"""
        query = select(StudentModel).where(
            func.lower(StudentModel.email) == email.lower()
        )
        result = await self._execute_query(query)
        model = result.scalar_one_or_none()
        
        return model.to_domain() if model else None
    
    async def get_by_username(self, username: str) -> Optional[Student]:
        """Get student by username"""
        query = select(StudentModel).where(
            func.lower(StudentModel.username) == username.lower()
        )
        result = await self._execute_query(query)
        model = result.scalar_one_or_none()
        
        return model.to_domain() if model else None
    
    async def get_by_grade(self, grade: Grade, limit: int = 100) -> List[Student]:
        """Get students by grade"""
        query = (
            select(StudentModel)
            .where(StudentModel.grade == grade.level)
            .limit(limit)
        )
        result = await self._execute_query(query)
        models = result.scalars().all()
        
        return [model.to_domain() for model in models]
    
    async def get_active_students(self, limit: int = 100) -> List[Student]:
        """Get active students"""
        query = (
            select(StudentModel)
            .where(StudentModel.status == 'active')
            .limit(limit)
        )
        result = await self._execute_query(query)
        models = result.scalars().all()
        
        return [model.to_domain() for model in models]
    
    async def get_all(self, limit: int = 100, offset: int = 0) -> List[Student]:
        """Get all students with pagination"""
        query = select(StudentModel).limit(limit).offset(offset)
        result = await self._execute_query(query)
        models = result.scalars().all()
        
        return [model.to_domain() for model in models]
    
    async def save(self, entity: Student) -> Student:
        """Save student (create or update)"""
        # Check if student exists
        existing = await self.get_by_id(entity.id)
        
        if existing:
            # Update existing
            query = (
                update(StudentModel)
                .where(StudentModel.id == entity.id)
                .values(
                    email=entity.email,
                    username=entity.username,
                    full_name=entity.full_name,
                    grade=entity.grade.level,
                    learning_style=entity.learning_style.to_dict() if entity.learning_style else None,
                    ability_level=entity.ability_level.value,
                    status=entity.status.value,
                    updated_at=datetime.utcnow()
                )
            )
            await self._execute_query(query)
        else:
            # Check for duplicates
            if await self.get_by_email(entity.email):
                raise DuplicateEntityError("Student", "email", entity.email)
            
            if await self.get_by_username(entity.username):
                raise DuplicateEntityError("Student", "username", entity.username)
            
            # Create new
            model = StudentModel.from_domain(entity)
            self._session.add(model)
        
        await self._commit()
        return entity
    
    async def delete(self, entity_id: str) -> bool:
        """Delete student (soft delete)"""
        query = (
            update(StudentModel)
            .where(StudentModel.id == entity_id)
            .values(
                status='inactive',
                updated_at=datetime.utcnow()
            )
        )
        result = await self._execute_query(query)
        await self._commit()
        
        return result.rowcount > 0
    
    async def exists(self, entity_id: str) -> bool:
        """Check if student exists"""
        query = select(func.count()).where(StudentModel.id == entity_id)
        result = await self._execute_query(query)
        count = result.scalar()
        
        return count > 0
    
    async def search(self, query_str: str, limit: int = 50) -> List[Student]:
        """Search students"""
        search_pattern = f"%{query_str}%"
        query = (
            select(StudentModel)
            .where(
                or_(
                    StudentModel.full_name.ilike(search_pattern),
                    StudentModel.email.ilike(search_pattern),
                    StudentModel.username.ilike(search_pattern)
                )
            )
            .limit(limit)
        )
        result = await self._execute_query(query)
        models = result.scalars().all()
        
        return [model.to_domain() for model in models]
    
    async def update_learning_style(
        self,
        student_id: str,
        learning_style: LearningStyle
    ) -> Student:
        """Update student's learning style"""
        student = await self.get_by_id(student_id)
        
        if not student:
            raise EntityNotFoundError("Student", student_id)
        
        student.update_learning_style(learning_style)
        return await self.save(student)
    
    async def get_students_needing_assessment(self, limit: int = 100) -> List[Student]:
        """Get students without learning style assessment"""
        query = (
            select(StudentModel)
            .where(
                and_(
                    StudentModel.learning_style.is_(None),
                    StudentModel.status == 'active'
                )
            )
            .limit(limit)
        )
        result = await self._execute_query(query)
        models = result.scalars().all()
        
        return [model.to_domain() for model in models]


class LearningPathRepository(BaseRepository, ILearningPathRepository):
    """SQLAlchemy implementation of LearningPath Repository"""
    
    def __init__(self, session: AsyncSession):
        super().__init__(session, LearningPathModel)
    
    async def get_by_id(self, entity_id: str) -> Optional[LearningPath]:
        """Get learning path by ID"""
        query = (
            select(LearningPathModel)
            .options(selectinload(LearningPathModel.modules))
            .where(LearningPathModel.id == entity_id)
        )
        result = await self._execute_query(query)
        model = result.scalar_one_or_none()
        
        return model.to_domain() if model else None
    
    async def get_by_student_id(self, student_id: str) -> Optional[LearningPath]:
        """Get learning path by student ID"""
        query = (
            select(LearningPathModel)
            .options(selectinload(LearningPathModel.modules))
            .where(LearningPathModel.student_id == student_id)
        )
        result = await self._execute_query(query)
        model = result.scalar_one_or_none()
        
        return model.to_domain() if model else None
    
    async def get_by_grade(self, grade: Grade) -> List[LearningPath]:
        """Get learning paths by grade"""
        query = (
            select(LearningPathModel)
            .where(LearningPathModel.grade == grade.level)
        )
        result = await self._execute_query(query)
        models = result.scalars().all()
        
        return [model.to_domain() for model in models]
    
    async def get_active_paths(self, limit: int = 100) -> List[LearningPath]:
        """Get active learning paths"""
        # Active = updated in last 7 days
        from datetime import timedelta
        cutoff_date = datetime.utcnow() - timedelta(days=7)
        
        query = (
            select(LearningPathModel)
            .where(LearningPathModel.updated_at >= cutoff_date)
            .limit(limit)
        )
        result = await self._execute_query(query)
        models = result.scalars().all()
        
        return [model.to_domain() for model in models]
    
    async def get_all(self, limit: int = 100, offset: int = 0) -> List[LearningPath]:
        """Get all learning paths"""
        query = select(LearningPathModel).limit(limit).offset(offset)
        result = await self._execute_query(query)
        models = result.scalars().all()
        
        return [model.to_domain() for model in models]
    
    async def save(self, entity: LearningPath) -> LearningPath:
        """Save learning path"""
        existing = await self.get_by_id(entity.id)
        
        if existing:
            # Update existing
            model = await self._session.get(LearningPathModel, entity.id)
            model.update_from_domain(entity)
        else:
            # Create new
            model = LearningPathModel.from_domain(entity)
            self._session.add(model)
        
        await self._commit()
        return entity
    
    async def delete(self, entity_id: str) -> bool:
        """Delete learning path"""
        query = delete(LearningPathModel).where(LearningPathModel.id == entity_id)
        result = await self._execute_query(query)
        await self._commit()
        
        return result.rowcount > 0
    
    async def exists(self, entity_id: str) -> bool:
        """Check if learning path exists"""
        query = select(func.count()).where(LearningPathModel.id == entity_id)
        result = await self._execute_query(query)
        count = result.scalar()
        
        return count > 0
    
    async def update_module_progress(
        self,
        path_id: str,
        module_id: str,
        is_completed: bool,
        score: Optional[float] = None
    ) -> LearningPath:
        """Update module progress"""
        path = await self.get_by_id(path_id)
        
        if not path:
            raise EntityNotFoundError("LearningPath", path_id)
        
        # Find and update module
        for module in path.modules:
            if module.id == module_id:
                if is_completed:
                    module.mark_completed(score or 0, 0)
                else:
                    module.reset_progress()
                break
        
        return await self.save(path)
    
    async def get_paths_by_learning_style(
        self,
        learning_style: LearningStyle,
        limit: int = 100
    ) -> List[LearningPath]:
        """Get paths by learning style"""
        # This would need a JSON query in PostgreSQL
        # For now, return all and filter in memory
        all_paths = await self.get_all(limit=limit)
        
        return [
            path for path in all_paths
            if path.learning_style.primary_style == learning_style.primary_style
        ]


class CurriculumRepository(BaseRepository, ICurriculumRepository):
    """SQLAlchemy implementation of Curriculum Repository"""
    
    def __init__(self, session: AsyncSession):
        super().__init__(session, CurriculumModel)
        # Initialize with MEB curriculum data
        self._curriculum_data = self._load_meb_curriculum()
    
    async def get_by_id(self, entity_id: str) -> Optional[Curriculum]:
        """Get curriculum by ID"""
        # For now, using in-memory data
        for grade_level, data in self._curriculum_data.items():
            if data.get('id') == entity_id:
                return self._create_curriculum_from_data(grade_level, data)
        return None
    
    async def get_by_grade(self, grade: Grade) -> Optional[Curriculum]:
        """Get curriculum by grade"""
        grade_str = str(grade.level)
        
        if grade_str in self._curriculum_data:
            return self._create_curriculum_from_data(
                grade.level,
                self._curriculum_data[grade_str]
            )
        
        return None
    
    async def get_all_grades(self) -> List[Grade]:
        """Get all available grades"""
        return [Grade(int(g)) for g in self._curriculum_data.keys()]
    
    async def get_subject_topics(self, grade: Grade, subject: str) -> List[str]:
        """Get topics for a subject"""
        curriculum = await self.get_by_grade(grade)
        
        if curriculum:
            subject_obj = curriculum.get_subject(subject)
            if subject_obj:
                return subject_obj.topics
        
        return []
    
    async def search_topics(self, query: str) -> List[Dict[str, Any]]:
        """Search topics across all curricula"""
        results = []
        query_lower = query.lower()
        
        for grade_str, grade_data in self._curriculum_data.items():
            for subject_name, subject_data in grade_data.items():
                if subject_name == 'id':
                    continue
                
                for topic in subject_data.get('topics', []):
                    if query_lower in topic.lower():
                        results.append({
                            'grade': int(grade_str),
                            'subject': subject_name,
                            'topic': topic
                        })
        
        return results
    
    async def get_prerequisites(
        self,
        grade: Grade,
        subject: str,
        topic: str
    ) -> List[str]:
        """Get prerequisites for a topic"""
        curriculum = await self.get_by_grade(grade)
        
        if curriculum:
            subject_obj = curriculum.get_subject(subject)
            if subject_obj:
                return subject_obj.get_prerequisites(topic)
        
        return []
    
    async def get_all(self, limit: int = 100, offset: int = 0) -> List[Curriculum]:
        """Get all curricula"""
        curricula = []
        
        for grade_level in list(self._curriculum_data.keys())[offset:offset+limit]:
            curriculum = await self.get_by_grade(Grade(int(grade_level)))
            if curriculum:
                curricula.append(curriculum)
        
        return curricula
    
    async def save(self, entity: Curriculum) -> Curriculum:
        """Save curriculum"""
        # For now, just return the entity as we're using in-memory data
        return entity
    
    async def delete(self, entity_id: str) -> bool:
        """Delete curriculum"""
        # Not implemented for in-memory data
        return False
    
    async def exists(self, entity_id: str) -> bool:
        """Check if curriculum exists"""
        curriculum = await self.get_by_id(entity_id)
        return curriculum is not None
    
    def _load_meb_curriculum(self) -> Dict:
        """Load MEB curriculum data"""
        return {
            '9': {
                'id': 'curr-9',
                'Matematik': {
                    'topics': ['Kümeler', 'Sayılar', 'Denklemler', 'Fonksiyonlar'],
                    'hours': 180,
                    'prerequisites': {
                        'Sayılar': ['Kümeler'],
                        'Denklemler': ['Sayılar'],
                        'Fonksiyonlar': ['Denklemler']
                    }
                },
                'Fizik': {
                    'topics': ['Hareket', 'Kuvvet', 'Enerji', 'Elektrik'],
                    'hours': 144,
                    'prerequisites': {
                        'Kuvvet': ['Hareket'],
                        'Enerji': ['Kuvvet'],
                        'Elektrik': ['Enerji']
                    }
                },
                'Türkçe': {
                    'topics': ['Dil Bilgisi', 'Anlatım', 'Edebiyat', 'Metin İnceleme'],
                    'hours': 144,
                    'prerequisites': {}
                }
            },
            '10': {
                'id': 'curr-10',
                'Matematik': {
                    'topics': ['Polinomlar', 'Trigonometri', 'Analitik Geometri', 'Olasılık'],
                    'hours': 180,
                    'prerequisites': {
                        'Trigonometri': ['Polinomlar'],
                        'Analitik Geometri': ['Trigonometri']
                    }
                },
                'Fizik': {
                    'topics': ['Dalgalar', 'Optik', 'Manyetizma', 'Modern Fizik'],
                    'hours': 144,
                    'prerequisites': {
                        'Optik': ['Dalgalar'],
                        'Modern Fizik': ['Manyetizma']
                    }
                }
            },
            '11': {
                'id': 'curr-11',
                'Matematik': {
                    'topics': ['Limit', 'Türev', 'İntegral', 'Diziler'],
                    'hours': 180,
                    'prerequisites': {
                        'Türev': ['Limit'],
                        'İntegral': ['Türev']
                    }
                }
            },
            '12': {
                'id': 'curr-12',
                'Matematik': {
                    'topics': ['Kompleks Sayılar', 'Logaritma', 'Diferansiyel Denklemler'],
                    'hours': 180,
                    'prerequisites': {}
                }
            }
        }
    
    def _create_curriculum_from_data(self, grade: int, data: Dict) -> Curriculum:
        """Create curriculum from data"""
        subjects = {}
        
        for subject_name, subject_data in data.items():
            if subject_name == 'id':
                continue
            
            subjects[subject_name] = Subject(
                name=subject_name,
                topics=subject_data.get('topics', []),
                total_hours=subject_data.get('hours', 0),
                prerequisites=subject_data.get('prerequisites', {})
            )
        
        return Curriculum.create_for_grade(grade, subjects)
