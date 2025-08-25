"""
Learning Path Service - Application Layer
TEKNOFEST 2025 - Refactored from LearningPathAgent
"""

import logging
from typing import List, Dict, Optional
from dataclasses import dataclass

from src.domain.entities import Student, LearningPath, LearningModule, Curriculum
from src.domain.value_objects import LearningStyle, StudentResponse, Grade, AbilityLevel
from src.domain.interfaces import (
    IStudentRepository,
    ICurriculumRepository,
    ILearningPathRepository,
    ILearningModuleRepository,
    IUnitOfWork
)
from src.shared.constants import LearningStyles, EducationConstants
from src.shared.exceptions import (
    StudentNotFoundException,
    CurriculumNotFoundException,
    InsufficientDataError,
    DomainError
)

logger = logging.getLogger(__name__)


@dataclass
class LearningStyleAnalysisRequest:
    """DTO for learning style analysis request"""
    student_id: str
    responses: List[str]


@dataclass
class LearningStyleAnalysisResult:
    """DTO for learning style analysis result"""
    student_id: str
    primary_style: str
    secondary_style: Optional[str]
    confidence: float
    scores: Dict[str, float]
    recommendation: str


class LearningPathService:
    """
    Service for managing learning paths and learning style analysis
    Refactored from LearningPathAgent with Clean Code principles
    """
    
    def __init__(
        self,
        student_repository: IStudentRepository,
        curriculum_repository: ICurriculumRepository,
        learning_path_repository: ILearningPathRepository,
        module_repository: ILearningModuleRepository,
        unit_of_work: IUnitOfWork
    ):
        self._student_repo = student_repository
        self._curriculum_repo = curriculum_repository
        self._path_repo = learning_path_repository
        self._module_repo = module_repository
        self._uow = unit_of_work
        
        # Initialize keyword mappings for learning style detection
        self._keyword_mappings = self._initialize_keyword_mappings()
    
    async def analyze_learning_style(
        self,
        request: LearningStyleAnalysisRequest
    ) -> LearningStyleAnalysisResult:
        """
        Analyze student responses to determine learning style
        Replaces the old detect_learning_style method
        """
        # Validate input
        self._validate_responses(request.responses)
        
        # Get student
        student = await self._get_student(request.student_id)
        
        # Convert responses to domain objects
        student_responses = self._create_student_responses(request.responses)
        
        # Analyze responses
        learning_style = self._analyze_responses(student_responses)
        
        # Update student's learning style
        async with self._uow:
            student.update_learning_style(learning_style)
            await self._student_repo.save(student)
            await self._uow.commit()
        
        # Create result
        return self._create_analysis_result(request.student_id, learning_style)
    
    async def create_personalized_path(
        self,
        student_id: str,
        grade: int
    ) -> LearningPath:
        """
        Create a personalized learning path for a student
        """
        # Get student
        student = await self._get_student(student_id)
        
        # Ensure student has learning style assessed
        if not student.learning_style:
            raise DomainError(
                "Student must complete learning style assessment before creating a path"
            )
        
        # Get curriculum
        curriculum = await self._get_curriculum(Grade(grade))
        
        # Get recommended modules based on learning style
        modules = await self._get_personalized_modules(
            student,
            curriculum
        )
        
        # Create learning path
        learning_path = LearningPath.create_for_student(student, modules)
        
        # Save learning path
        async with self._uow:
            saved_path = await self._path_repo.save(learning_path)
            await self._uow.commit()
        
        logger.info(f"Created personalized learning path for student {student_id}")
        
        return saved_path
    
    async def update_module_progress(
        self,
        student_id: str,
        module_id: str,
        score: float,
        time_spent_minutes: int
    ) -> LearningPath:
        """
        Update progress for a module in student's learning path
        """
        # Get learning path
        learning_path = await self._path_repo.get_by_student_id(student_id)
        
        if not learning_path:
            raise DomainError(f"No learning path found for student {student_id}")
        
        # Find and update module
        module = self._find_module_in_path(learning_path, module_id)
        module.mark_completed(score, time_spent_minutes)
        
        # Update student ability level based on performance
        student = await self._get_student(student_id)
        new_ability = self._calculate_new_ability(student.ability_level, score)
        student.update_ability_level(new_ability)
        
        # Save updates
        async with self._uow:
            await self._path_repo.save(learning_path)
            await self._student_repo.save(student)
            await self._uow.commit()
        
        return learning_path
    
    def _validate_responses(self, responses: List[str]) -> None:
        """Validate student responses"""
        if not responses:
            raise InsufficientDataError(
                required=EducationConstants.MIN_RESPONSES_FOR_ANALYSIS,
                provided=0
            )
        
        if len(responses) > EducationConstants.MAX_RESPONSES_FOR_ANALYSIS:
            raise DomainError(
                f"Too many responses: {len(responses)}. Maximum allowed is {EducationConstants.MAX_RESPONSES_FOR_ANALYSIS}"
            )
        
        for response in responses:
            if len(response.strip()) < EducationConstants.MIN_RESPONSE_LENGTH:
                raise DomainError(
                    f"Response too short. Minimum length is {EducationConstants.MIN_RESPONSE_LENGTH} characters"
            )
    
    async def _get_student(self, student_id: str) -> Student:
        """Get student by ID with error handling"""
        student = await self._student_repo.get_by_id(student_id)
        
        if not student:
            raise StudentNotFoundException(student_id)
        
        if not student.is_active:
            raise DomainError(f"Student {student_id} is not active")
        
        return student
    
    async def _get_curriculum(self, grade: Grade) -> Curriculum:
        """Get curriculum for grade with error handling"""
        curriculum = await self._curriculum_repo.get_by_grade(grade)
        
        if not curriculum:
            raise CurriculumNotFoundException(str(grade.level))
        
        return curriculum
    
    def _create_student_responses(self, responses: List[str]) -> List[StudentResponse]:
        """Convert string responses to StudentResponse objects"""
        from datetime import datetime
        
        return [
            StudentResponse(
                content=response,
                timestamp=datetime.utcnow()
            )
            for response in responses
        ]
    
    def _analyze_responses(self, responses: List[StudentResponse]) -> LearningStyle:
        """
        Analyze student responses to determine learning style
        This is the core logic refactored from the old agent
        """
        # Initialize scores
        scores = {
            LearningStyles.VISUAL.value: 0,
            LearningStyles.AUDITORY.value: 0,
            LearningStyles.READING.value: 0,
            LearningStyles.KINESTHETIC.value: 0
        }
        
        # Analyze each response
        for response in responses:
            keywords = response.extract_keywords()
            
            for keyword in keywords:
                keyword_lower = keyword.lower()
                
                # Check keyword mappings
                for style, style_keywords in self._keyword_mappings.items():
                    if any(sk in keyword_lower for sk in style_keywords):
                        scores[style] += 1
        
        # Normalize scores
        total_score = sum(scores.values())
        if total_score > 0:
            normalized_scores = {k: v / total_score for k, v in scores.items()}
        else:
            normalized_scores = scores
        
        # Determine primary and secondary styles
        sorted_styles = sorted(normalized_scores.items(), key=lambda x: x[1], reverse=True)
        
        primary_style = LearningStyles(sorted_styles[0][0])
        secondary_style = LearningStyles(sorted_styles[1][0]) if len(sorted_styles) > 1 and sorted_styles[1][1] > 0 else None
        
        # Calculate confidence
        confidence = sorted_styles[0][1] if sorted_styles else 0.0
        
        return LearningStyle(
            primary_style=primary_style,
            secondary_style=secondary_style,
            confidence=min(confidence, 1.0),
            scores=normalized_scores
        )
    
    def _initialize_keyword_mappings(self) -> Dict[str, List[str]]:
        """Initialize keyword mappings for learning style detection"""
        return {
            LearningStyles.VISUAL.value: [
                'görsel', 'şema', 'grafik', 'resim', 'video', 
                'animasyon', 'renk', 'diyagram', 'harita', 'çizim',
                'tablo', 'infografik', 'illüstrasyon'
            ],
            LearningStyles.AUDITORY.value: [
                'dinle', 'anlat', 'konuş', 'ses', 'müzik', 
                'tartış', 'açıkla', 'sözlü', 'podcast', 'anlatım',
                'konferans', 'sunum', 'diyalog'
            ],
            LearningStyles.READING.value: [
                'oku', 'yaz', 'not', 'metin', 'kitap', 
                'makale', 'araştır', 'döküman', 'rapor', 'özet',
                'liste', 'tanım', 'açıklama'
            ],
            LearningStyles.KINESTHETIC.value: [
                'yap', 'uygula', 'deney', 'hareket', 'dokun', 
                'pratik', 'el', 'deneyim', 'aktivite', 'oyun',
                'simülasyon', 'model', 'proje'
            ]
        }
    
    def _create_analysis_result(
        self,
        student_id: str,
        learning_style: LearningStyle
    ) -> LearningStyleAnalysisResult:
        """Create analysis result DTO"""
        return LearningStyleAnalysisResult(
            student_id=student_id,
            primary_style=learning_style.primary_style.value,
            secondary_style=learning_style.secondary_style.value if learning_style.secondary_style else None,
            confidence=learning_style.confidence,
            scores=learning_style.scores,
            recommendation=self._get_style_recommendation(learning_style.primary_style)
        )
    
    def _get_style_recommendation(self, style: LearningStyles) -> str:
        """Get learning recommendation for a style"""
        recommendations = {
            LearningStyles.VISUAL: "Görsel materyaller, infografikler, videolar ve diyagramlar tercih edilmeli. Mind mapping ve renk kodlama teknikleri kullanılmalı.",
            LearningStyles.AUDITORY: "Sesli anlatımlar, podcast'ler ve grup tartışmaları önerilir. Bilgileri sesli tekrar etme ve açıklama teknikleri kullanılmalı.",
            LearningStyles.READING: "Yazılı kaynaklar, e-kitaplar ve detaylı notlar kullanılmalı. Özet çıkarma ve not alma teknikleri geliştirilmeli.",
            LearningStyles.KINESTHETIC: "Uygulamalı aktiviteler, deneyler ve simülasyonlar tercih edilmeli. Hands-on projeler ve interaktif öğrenme araçları kullanılmalı."
        }
        return recommendations.get(style, "Karma öğrenme yöntemleri önerilir")
    
    async def _get_personalized_modules(
        self,
        student: Student,
        curriculum: Curriculum
    ) -> List[LearningModule]:
        """Get personalized modules based on student profile and curriculum"""
        modules = []
        
        # Get all available modules for the grade
        for subject_name, subject in curriculum.subjects.items():
            for topic in subject.topics:
                # Get modules for this topic
                topic_modules = await self._module_repo.get_by_topic(topic)
                
                # Filter and prioritize based on learning style
                personalized_modules = self._filter_modules_by_style(
                    topic_modules,
                    student.learning_style
                )
                
                modules.extend(personalized_modules)
        
        # Sort modules by prerequisites and difficulty
        sorted_modules = self._sort_modules_by_prerequisites(modules)
        
        return sorted_modules
    
    def _filter_modules_by_style(
        self,
        modules: List[LearningModule],
        learning_style: LearningStyle
    ) -> List[LearningModule]:
        """Filter and prioritize modules based on learning style"""
        if not learning_style:
            return modules
        
        # Map learning styles to content types
        style_content_map = {
            LearningStyles.VISUAL: ['video', 'infographic', 'diagram'],
            LearningStyles.AUDITORY: ['audio', 'podcast', 'discussion'],
            LearningStyles.READING: ['text', 'article', 'ebook'],
            LearningStyles.KINESTHETIC: ['interactive', 'simulation', 'lab']
        }
        
        preferred_types = style_content_map.get(learning_style.primary_style, [])
        
        # Sort modules with preferred content types first
        return sorted(
            modules,
            key=lambda m: 0 if m.content_type in preferred_types else 1
        )
    
    def _sort_modules_by_prerequisites(
        self,
        modules: List[LearningModule]
    ) -> List[LearningModule]:
        """Sort modules respecting prerequisites"""
        sorted_modules = []
        module_ids = set()
        
        # Simple topological sort
        while modules:
            for module in modules[:]:
                if all(prereq in module_ids for prereq in module.prerequisites):
                    sorted_modules.append(module)
                    module_ids.add(module.id)
                    modules.remove(module)
        
        return sorted_modules
    
    def _find_module_in_path(
        self,
        learning_path: LearningPath,
        module_id: str
    ) -> LearningModule:
        """Find a module in learning path"""
        for module in learning_path.modules:
            if module.id == module_id:
                return module
        
        raise DomainError(f"Module {module_id} not found in learning path")
    
    def _calculate_new_ability(
        self,
        current_ability: AbilityLevel,
        score: float
    ) -> AbilityLevel:
        """Calculate new ability level based on performance"""
        # Simple adjustment based on score
        if score >= 90:
            adjustment = 0.1
        elif score >= 70:
            adjustment = 0.05
        elif score >= 50:
            adjustment = 0
        else:
            adjustment = -0.05
        
        return current_ability.adjust(adjustment)
