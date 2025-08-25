"""
Service Factory Pattern for Dependency Injection
TEKNOFEST 2025 - Eğitim Teknolojileri
"""

from typing import TypeVar, Type, Optional
from src.container import DependencyInjectionContainer, ServiceCollection, configure_services

T = TypeVar('T')


class ServiceFactory:
    """Factory for creating service instances with dependency injection"""
    
    def __init__(self):
        self._container = DependencyInjectionContainer()
        self._initialized = False
    
    def initialize(self):
        """Initialize the service container"""
        if not self._initialized:
            self._container.register_services(configure_services)
            self._container.build()
            self._initialized = True
    
    def create_service(self, service_type: Type[T]) -> T:
        """Create a service instance with all dependencies"""
        if not self._initialized:
            self.initialize()
        return self._container.get_required_service(service_type)
    
    def get_service(self, service_type: Type[T]) -> Optional[T]:
        """Get a service instance if registered"""
        if not self._initialized:
            self.initialize()
        return self._container.get_service(service_type)
    
    def create_scope(self):
        """Create a new service scope for scoped services"""
        if not self._initialized:
            self.initialize()
        return self._container.build().create_scope()
    
    @classmethod
    def get_default(cls) -> 'ServiceFactory':
        """Get the default factory instance"""
        if not hasattr(cls, '_default_instance'):
            cls._default_instance = cls()
            cls._default_instance.initialize()
        return cls._default_instance
    
    def reset(self):
        """Reset the factory (useful for testing)"""
        self._container.reset()
        self._initialized = False


def get_factory() -> ServiceFactory:
    """Get the default service factory"""
    return ServiceFactory.get_default()


def create_service(service_type: Type[T]) -> T:
    """Convenience function to create a service"""
    return get_factory().create_service(service_type)


def get_service(service_type: Type[T]) -> Optional[T]:
    """Convenience function to get a service"""
    return get_factory().get_service(service_type)