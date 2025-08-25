"""
Production-Ready Dependency Injection Container
TEKNOFEST 2025 - Eğitim Teknolojileri
"""

import inspect
import threading
from abc import ABC, abstractmethod
from enum import Enum
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar, Union

T = TypeVar('T')


class Scope(Enum):
    """Service lifecycle scopes"""
    SINGLETON = "singleton"
    TRANSIENT = "transient"
    SCOPED = "scoped"


class ServiceDescriptor:
    """Describes a service registration"""
    
    def __init__(
        self,
        service_type: Type,
        implementation: Union[Type, Callable, object],
        scope: Scope = Scope.SINGLETON,
        factory: Optional[Callable] = None
    ):
        self.service_type = service_type
        self.implementation = implementation
        self.scope = scope
        self.factory = factory
        self.instance = None
        self.lock = threading.Lock()


class IServiceProvider(ABC):
    """Service provider interface"""
    
    @abstractmethod
    def get_service(self, service_type: Type[T]) -> Optional[T]:
        """Get a service instance"""
        pass
    
    @abstractmethod
    def get_required_service(self, service_type: Type[T]) -> T:
        """Get a required service instance"""
        pass
    
    @abstractmethod
    def create_scope(self) -> 'IServiceScope':
        """Create a new service scope"""
        pass


class IServiceScope(ABC):
    """Service scope interface"""
    
    @abstractmethod
    def get_service(self, service_type: Type[T]) -> Optional[T]:
        """Get a service from scope"""
        pass
    
    @abstractmethod
    def dispose(self):
        """Dispose the scope"""
        pass


class ServiceCollection:
    """Collection of service descriptors"""
    
    def __init__(self):
        self._descriptors: List[ServiceDescriptor] = []
        self._service_map: Dict[Type, List[ServiceDescriptor]] = {}
    
    def add_singleton(
        self,
        service_type: Type[T],
        implementation: Optional[Union[Type[T], Callable[..., T]]] = None,
        factory: Optional[Callable[[IServiceProvider], T]] = None,
        instance: Optional[T] = None
    ) -> 'ServiceCollection':
        """Register a singleton service"""
        if instance is not None:
            descriptor = ServiceDescriptor(service_type, instance, Scope.SINGLETON)
            descriptor.instance = instance
        elif factory is not None:
            descriptor = ServiceDescriptor(service_type, factory, Scope.SINGLETON, factory=factory)
        else:
            impl = implementation or service_type
            descriptor = ServiceDescriptor(service_type, impl, Scope.SINGLETON)
        
        self._add_descriptor(descriptor)
        return self
    
    def add_transient(
        self,
        service_type: Type[T],
        implementation: Optional[Union[Type[T], Callable[..., T]]] = None,
        factory: Optional[Callable[[IServiceProvider], T]] = None
    ) -> 'ServiceCollection':
        """Register a transient service"""
        if factory is not None:
            descriptor = ServiceDescriptor(service_type, factory, Scope.TRANSIENT, factory=factory)
        else:
            impl = implementation or service_type
            descriptor = ServiceDescriptor(service_type, impl, Scope.TRANSIENT)
        
        self._add_descriptor(descriptor)
        return self
    
    def add_scoped(
        self,
        service_type: Type[T],
        implementation: Optional[Union[Type[T], Callable[..., T]]] = None,
        factory: Optional[Callable[[IServiceProvider], T]] = None
    ) -> 'ServiceCollection':
        """Register a scoped service"""
        if factory is not None:
            descriptor = ServiceDescriptor(service_type, factory, Scope.SCOPED, factory=factory)
        else:
            impl = implementation or service_type
            descriptor = ServiceDescriptor(service_type, impl, Scope.SCOPED)
        
        self._add_descriptor(descriptor)
        return self
    
    def _add_descriptor(self, descriptor: ServiceDescriptor):
        """Add a service descriptor"""
        self._descriptors.append(descriptor)
        
        if descriptor.service_type not in self._service_map:
            self._service_map[descriptor.service_type] = []
        self._service_map[descriptor.service_type].append(descriptor)
    
    def build_service_provider(self) -> 'ServiceProvider':
        """Build the service provider"""
        return ServiceProvider(self._descriptors, self._service_map)


class ServiceScope(IServiceScope):
    """Service scope implementation"""
    
    def __init__(self, provider: 'ServiceProvider'):
        self._provider = provider
        self._scoped_instances: Dict[Type, Any] = {}
        self._lock = threading.Lock()
    
    def __enter__(self):
        """Enter context manager"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager"""
        self.dispose()
        return False
    
    def get_service(self, service_type: Type[T]) -> Optional[T]:
        """Get a service from scope"""
        descriptors = self._provider._service_map.get(service_type, [])
        if not descriptors:
            return None
        
        descriptor = descriptors[-1]
        
        if descriptor.scope == Scope.SCOPED:
            with self._lock:
                if service_type in self._scoped_instances:
                    return self._scoped_instances[service_type]
                
                instance = self._provider._create_instance(descriptor, self)
                self._scoped_instances[service_type] = instance
                return instance
        
        return self._provider._get_service_internal(descriptor, self)
    
    def dispose(self):
        """Dispose the scope"""
        for instance in self._scoped_instances.values():
            if hasattr(instance, 'dispose'):
                instance.dispose()
            elif hasattr(instance, 'close'):
                instance.close()
        self._scoped_instances.clear()


class ServiceProvider(IServiceProvider):
    """Service provider implementation"""
    
    def __init__(self, descriptors: List[ServiceDescriptor], service_map: Dict[Type, List[ServiceDescriptor]]):
        self._descriptors = descriptors
        self._service_map = service_map
        self._root_scope = ServiceScope(self)
    
    def get_service(self, service_type: Type[T]) -> Optional[T]:
        """Get a service instance"""
        return self._root_scope.get_service(service_type)
    
    def get_required_service(self, service_type: Type[T]) -> T:
        """Get a required service instance"""
        service = self.get_service(service_type)
        if service is None:
            raise ValueError(f"Service of type {service_type.__name__} is not registered")
        return service
    
    def create_scope(self) -> IServiceScope:
        """Create a new service scope"""
        return ServiceScope(self)
    
    def _get_service_internal(self, descriptor: ServiceDescriptor, scope: ServiceScope) -> Any:
        """Internal method to get service"""
        if descriptor.scope == Scope.SINGLETON:
            if descriptor.instance is not None:
                return descriptor.instance
            
            with descriptor.lock:
                if descriptor.instance is not None:
                    return descriptor.instance
                
                descriptor.instance = self._create_instance(descriptor, scope)
                return descriptor.instance
        
        elif descriptor.scope == Scope.TRANSIENT:
            return self._create_instance(descriptor, scope)
        
        elif descriptor.scope == Scope.SCOPED:
            return scope.get_service(descriptor.service_type)
        
        return None
    
    def _create_instance(self, descriptor: ServiceDescriptor, scope: ServiceScope) -> Any:
        """Create an instance of a service"""
        if descriptor.factory:
            return descriptor.factory(scope)
        
        implementation = descriptor.implementation
        
        if not inspect.isclass(implementation):
            return implementation
        
        sig = inspect.signature(implementation.__init__)
        params = {}
        
        for param_name, param in sig.parameters.items():
            if param_name == 'self':
                continue
            
            param_type = param.annotation
            if param_type == inspect.Parameter.empty:
                if param.default != inspect.Parameter.empty:
                    params[param_name] = param.default
                continue
            
            service = scope.get_service(param_type)
            if service is not None:
                params[param_name] = service
            elif param.default != inspect.Parameter.empty:
                params[param_name] = param.default
            else:
                raise ValueError(
                    f"Cannot resolve dependency {param_type.__name__} for {implementation.__name__}"
                )
        
        return implementation(**params)


def inject(scope: Scope = Scope.SINGLETON):
    """Decorator for dependency injection"""
    def decorator(cls):
        cls._injection_scope = scope
        return cls
    return decorator


def singleton(cls):
    """Decorator for singleton services"""
    return inject(Scope.SINGLETON)(cls)


def transient(cls):
    """Decorator for transient services"""
    return inject(Scope.TRANSIENT)(cls)


def scoped(cls):
    """Decorator for scoped services"""
    return inject(Scope.SCOPED)(cls)


class DependencyInjectionContainer:
    """Main DI container for the application"""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self._services = ServiceCollection()
            self._provider: Optional[ServiceProvider] = None
            self._initialized = True
    
    def register_services(self, configurator: Callable[[ServiceCollection], None]):
        """Register services using a configurator function"""
        configurator(self._services)
    
    def build(self) -> ServiceProvider:
        """Build and return the service provider"""
        if self._provider is None:
            self._provider = self._services.build_service_provider()
        return self._provider
    
    def get_service(self, service_type: Type[T]) -> Optional[T]:
        """Get a service from the container"""
        if self._provider is None:
            raise RuntimeError("Container not built. Call build() first.")
        return self._provider.get_service(service_type)
    
    def get_required_service(self, service_type: Type[T]) -> T:
        """Get a required service from the container"""
        if self._provider is None:
            raise RuntimeError("Container not built. Call build() first.")
        return self._provider.get_required_service(service_type)
    
    @classmethod
    def reset(cls):
        """Reset the container (useful for testing)"""
        with cls._lock:
            cls._instance = None


def configure_services(services: ServiceCollection):
    """Configure application services"""
    from src.config import Settings, get_settings
    from src.agents.learning_path_agent_v2 import LearningPathAgent
    from src.agents.study_buddy_agent_clean import StudyBuddyAgent
    from src.model_integration_optimized import ModelIntegration
    from src.data_processor import DataProcessor
    
    services.add_singleton(Settings, factory=lambda _: get_settings())
    
    services.add_singleton(DataProcessor)
    
    services.add_singleton(ModelIntegration)
    
    services.add_scoped(LearningPathAgent)
    services.add_scoped(StudyBuddyAgent)
    
    try:
        import redis
        services.add_singleton(
            redis.Redis,
            factory=lambda provider: redis.Redis.from_url(
                provider.get_required_service(Settings).get_redis_url(hide_password=False)
            )
        )
    except ImportError:
        pass
    
    try:
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker
        
        services.add_singleton(
            create_engine,
            factory=lambda provider: create_engine(
                provider.get_required_service(Settings).get_database_url(hide_password=False),
                pool_size=provider.get_required_service(Settings).database_pool_size,
                max_overflow=provider.get_required_service(Settings).database_max_overflow,
                echo=provider.get_required_service(Settings).database_echo
            )
        )
        
        services.add_singleton(
            sessionmaker,
            factory=lambda provider: sessionmaker(
                bind=provider.get_service(create_engine),
                autocommit=False,
                autoflush=False
            )
        )
    except ImportError:
        pass


container = DependencyInjectionContainer()