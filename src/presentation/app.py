"""
Application Factory - Clean Architecture Implementation
TEKNOFEST 2025 - Refactored from app.py
"""

import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.config import Settings, get_settings
from src.shared.constants import AppConstants
from src.infrastructure.config.container import (
    initialize_container,
    cleanup_container,
    get_container
)
from src.presentation.middleware import (
    ErrorHandlerMiddleware,
    SecurityMiddleware,
    LoggingMiddleware,
    RateLimitMiddleware
)
from src.presentation.api import create_api_v1_router

logger = logging.getLogger(__name__)


class ApplicationFactory:
    """
    Factory for creating FastAPI application
    Following Clean Architecture principles
    """
    
    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        self.app: Optional[FastAPI] = None
    
    def create_app(self) -> FastAPI:
        """Create and configure FastAPI application"""
        if self.app:
            return self.app
        
        # Create app with lifespan
        self.app = self._create_base_app()
        
        # Configure middleware
        self._configure_middleware()
        
        # Configure routes
        self._configure_routes()
        
        # Configure exception handlers
        self._configure_exception_handlers()
        
        logger.info(f"Application created: {self.settings.app_name}")
        
        return self.app
    
    def _create_base_app(self) -> FastAPI:
        """Create base FastAPI application"""
        
        @asynccontextmanager
        async def lifespan(app: FastAPI):
            """Application lifespan manager"""
            logger.info("Starting application...")
            
            try:
                # Initialize DI container
                await initialize_container(self.settings)
                
                # Run any startup tasks
                await self._on_startup()
                
                logger.info("Application started successfully")
                
                yield
                
            finally:
                # Run shutdown tasks
                await self._on_shutdown()
                
                # Cleanup DI container
                await cleanup_container()
                
                logger.info("Application shutdown complete")
        
        # Create FastAPI app
        app = FastAPI(
            title=self.settings.app_name,
            version=self.settings.app_version,
            debug=self.settings.app_debug,
            lifespan=lifespan,
            docs_url="/api/docs" if not self.settings.is_production() else None,
            redoc_url="/api/redoc" if not self.settings.is_production() else None,
            openapi_url="/api/openapi.json" if not self.settings.is_production() else None
        )
        
        return app
    
    def _configure_middleware(self):
        """Configure application middleware in correct order"""
        
        # CORS Middleware (should be first)
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=self.settings.cors_origins,
            allow_credentials=self.settings.cors_allow_credentials,
            allow_methods=self.settings.cors_allow_methods,
            allow_headers=self.settings.cors_allow_headers,
        )
        
        # Custom middleware (order matters!)
        # 1. Error Handler (catches all errors)
        self.app.add_middleware(ErrorHandlerMiddleware)
        
        # 2. Security Middleware
        self.app.add_middleware(
            SecurityMiddleware,
            settings=self.settings
        )
        
        # 3. Rate Limiting
        if self.settings.rate_limit_enabled:
            self.app.add_middleware(
                RateLimitMiddleware,
                requests_per_minute=self.settings.rate_limit_requests_per_minute
            )
        
        # 4. Logging Middleware
        if not self.settings.is_production():
            self.app.add_middleware(LoggingMiddleware)
    
    def _configure_routes(self):
        """Configure API routes"""
        
        # Health check endpoint
        @self.app.get("/health")
        async def health_check():
            """Health check endpoint"""
            return {
                "status": "healthy",
                "environment": self.settings.app_env.value,
                "version": self.settings.app_version
            }
        
        # API v1 routes
        api_v1_router = create_api_v1_router()
        self.app.include_router(
            api_v1_router,
            prefix=AppConstants.API_PREFIX
        )
        
        # Root endpoint
        @self.app.get("/")
        async def root():
            """Root endpoint"""
            return {
                "name": self.settings.app_name,
                "version": self.settings.app_version,
                "api_docs": "/api/docs" if not self.settings.is_production() else None
            }
    
    def _configure_exception_handlers(self):
        """Configure custom exception handlers"""
        from src.presentation.handlers import (
            validation_error_handler,
            domain_error_handler,
            authentication_error_handler,
            not_found_error_handler,
            general_error_handler
        )
        from src.shared.exceptions import (
            ValidationError,
            DomainError,
            AuthenticationError,
            EntityNotFoundError
        )
        
        # Register exception handlers
        self.app.add_exception_handler(ValidationError, validation_error_handler)
        self.app.add_exception_handler(DomainError, domain_error_handler)
        self.app.add_exception_handler(AuthenticationError, authentication_error_handler)
        self.app.add_exception_handler(EntityNotFoundError, not_found_error_handler)
        self.app.add_exception_handler(Exception, general_error_handler)
    
    async def _on_startup(self):
        """Run startup tasks"""
        logger.info("Running startup tasks...")
        
        # Add any additional startup tasks here
        # For example: cache warming, health checks, etc.
    
    async def _on_shutdown(self):
        """Run shutdown tasks"""
        logger.info("Running shutdown tasks...")
        
        # Add any cleanup tasks here
        # For example: closing connections, saving state, etc.


def create_application(settings: Optional[Settings] = None) -> FastAPI:
    """
    Create FastAPI application
    This is the main entry point for the application
    """
    factory = ApplicationFactory(settings)
    return factory.create_app()


# Create the application instance
app = create_application()


if __name__ == "__main__":
    import uvicorn
    
    settings = get_settings()
    
    # Run the application
    uvicorn.run(
        "src.presentation.app:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.api_reload and not settings.is_production(),
        workers=settings.api_workers if settings.is_production() else 1,
        log_level="debug" if settings.app_debug else "info"
    )
