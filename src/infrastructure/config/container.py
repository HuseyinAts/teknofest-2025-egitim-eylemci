"""
Dependency Injection Container - Infrastructure Layer
TEKNOFEST 2025 - Clean Architecture DI Container
"""

import logging
from typing import Optional
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from dependency_injector import containers, providers

from src.config import Settings, get_settings
from src.domain.interfaces import (
    IStudentRepository,
    ILearningPathRepository,
    ICurriculumRepository,
    IQuizRepository,
    ILearningModuleRepository,
    IUnitOfWork
)
from src.infrastructure.persistence.repositories import (
    StudentRepository,
    LearningPathRepository,
    CurriculumRepository
)
from src.application.services.learning_path_service import LearningPathService
from src.application.services.quiz_service import QuizService

logger = logging.getLogger(__name__)


class UnitOfWork(IUnitOfWork):
    """Unit of Work implementation for transaction management"""
    
    def __init__(self, session_factory):
        self._session_factory = session_factory
        self._session: Optional[AsyncSession] = None
    
    async def __aenter__(self):
        """Enter transaction context"""
        self._session = self._session_factory()
        
        # Initialize repositories
        self.students = StudentRepository(self._session)
        self.learning_paths = LearningPathRepository(self._session)
        self.curricula = CurriculumRepository(self._session)
        # Add other repositories as implemented
        
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit transaction context"""
        if exc_type:
            await self.rollback()
        else:
            await self.commit()
        
        await self._session.close()
    
    async def commit(self):
        """Commit transaction"""
        await self._session.commit()
    
    async def rollback(self):
        """Rollback transaction"""
        await self._session.rollback()


class DatabaseProvider:
    """Database connection provider"""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.engine = None
        self.session_factory = None
    
    async def initialize(self):
        """Initialize database connection"""
        # Create async engine
        self.engine = create_async_engine(
            self.settings.database_url,
            echo=self.settings.app_debug,
            pool_pre_ping=True,
            pool_size=self.settings.database_pool_size,
            max_overflow=self.settings.database_max_overflow
        )
        
        # Create session factory
        self.session_factory = async_sessionmaker(
            self.engine,
            class_=AsyncSession,
            expire_on_commit=False
        )
        
        logger.info("Database connection initialized")
    
    async def close(self):
        """Close database connection"""
        if self.engine:
            await self.engine.dispose()
            logger.info("Database connection closed")
    
    def get_session(self) -> AsyncSession:
        """Get database session"""
        if not self.session_factory:
            raise RuntimeError("Database not initialized")
        return self.session_factory()
    
    @asynccontextmanager
    async def session_scope(self):
        """Provide a transactional scope for database operations"""
        async with self.session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise
            finally:
                await session.close()


class Container(containers.DeclarativeContainer):
    """
    Main Dependency Injection Container
    Manages all application dependencies with Clean Architecture
    """
    
    # Configuration
    config = providers.Configuration()
    
    # Settings
    settings = providers.Singleton(
        get_settings
    )
    
    # Database
    database_provider = providers.Singleton(
        DatabaseProvider,
        settings=settings
    )
    
    # Database Session
    db_session = providers.Factory(
        lambda db_provider: db_provider.get_session(),
        db_provider=database_provider
    )
    
    # Unit of Work
    unit_of_work = providers.Factory(
        UnitOfWork,
        session_factory=db_session
    )
    
    # Repositories
    student_repository = providers.Factory(
        StudentRepository,
        session=db_session
    )
    
    learning_path_repository = providers.Factory(
        LearningPathRepository,
        session=db_session
    )
    
    curriculum_repository = providers.Factory(
        CurriculumRepository,
        session=db_session
    )
    
    # Application Services
    learning_path_service = providers.Factory(
        LearningPathService,
        student_repository=student_repository,
        curriculum_repository=curriculum_repository,
        learning_path_repository=learning_path_repository,
        module_repository=providers.Object(None),  # Placeholder
        unit_of_work=unit_of_work
    )
    
    quiz_service = providers.Factory(
        QuizService,
        student_repository=student_repository,
        quiz_repository=providers.Object(None),  # Placeholder
        unit_of_work=unit_of_work
    )


class ServiceLocator:
    """Service locator for easy access to services"""
    
    _container: Optional[Container] = None
    
    @classmethod
    def initialize(cls, settings: Optional[Settings] = None):
        """Initialize the service locator"""
        if cls._container is None:
            cls._container = Container()
            
            if settings:
                cls._container.config.from_dict(settings.dict())
            else:
                cls._container.config.from_dict(get_settings().dict())
            
            logger.info("Service locator initialized")
    
    @classmethod
    async def startup(cls):
        """Startup tasks"""
        if not cls._container:
            cls.initialize()
        
        # Initialize database
        db_provider = cls._container.database_provider()
        await db_provider.initialize()
        
        logger.info("Application startup complete")
    
    @classmethod
    async def shutdown(cls):
        """Shutdown tasks"""
        if cls._container:
            # Close database
            db_provider = cls._container.database_provider()
            await db_provider.close()
        
        logger.info("Application shutdown complete")
    
    @classmethod
    def get_container(cls) -> Container:
        """Get the DI container"""
        if not cls._container:
            cls.initialize()
        return cls._container
    
    @classmethod
    def get_learning_path_service(cls) -> LearningPathService:
        """Get learning path service"""
        return cls.get_container().learning_path_service()
    
    @classmethod
    def get_quiz_service(cls) -> QuizService:
        """Get quiz service"""
        return cls.get_container().quiz_service()
    
    @classmethod
    def get_unit_of_work(cls) -> IUnitOfWork:
        """Get unit of work"""
        return cls.get_container().unit_of_work()


# Convenience functions
def get_container() -> Container:
    """Get the DI container"""
    return ServiceLocator.get_container()


def get_learning_path_service() -> LearningPathService:
    """Get learning path service for dependency injection"""
    return ServiceLocator.get_learning_path_service()


def get_quiz_service() -> QuizService:
    """Get quiz service for dependency injection"""
    return ServiceLocator.get_quiz_service()


async def initialize_container(settings: Optional[Settings] = None):
    """Initialize the DI container"""
    ServiceLocator.initialize(settings)
    await ServiceLocator.startup()


async def cleanup_container():
    """Cleanup the DI container"""
    await ServiceLocator.shutdown()
