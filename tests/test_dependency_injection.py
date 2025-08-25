"""
Comprehensive Dependency Injection Tests
TEKNOFEST 2025 - Eğitim Teknolojileri
"""

import pytest
import threading
from typing import Optional
from unittest.mock import Mock, patch

from src.container import (
    ServiceCollection,
    ServiceProvider,
    Scope,
    DependencyInjectionContainer,
    singleton,
    transient,
    scoped,
    inject
)
from src.factory import ServiceFactory
from src.config import Settings


class TestService:
    """Test service class"""
    def __init__(self):
        self.id = id(self)
        self.disposed = False
    
    def dispose(self):
        self.disposed = True


class DependentService:
    """Service with dependencies"""
    def __init__(self, test_service: TestService):
        self.test_service = test_service
        self.id = id(self)


@singleton
class SingletonService:
    """Singleton service"""
    def __init__(self):
        self.id = id(self)


@transient
class TransientService:
    """Transient service"""
    def __init__(self):
        self.id = id(self)


@scoped
class ScopedService:
    """Scoped service"""
    def __init__(self):
        self.id = id(self)


class TestServiceCollection:
    """Test ServiceCollection"""
    
    def test_add_singleton(self):
        """Test singleton registration"""
        collection = ServiceCollection()
        collection.add_singleton(TestService)
        
        assert len(collection._descriptors) == 1
        assert collection._descriptors[0].scope == Scope.SINGLETON
    
    def test_add_singleton_with_instance(self):
        """Test singleton registration with instance"""
        collection = ServiceCollection()
        instance = TestService()
        collection.add_singleton(TestService, instance=instance)
        
        descriptor = collection._descriptors[0]
        assert descriptor.instance is instance
    
    def test_add_singleton_with_factory(self):
        """Test singleton registration with factory"""
        collection = ServiceCollection()
        factory = lambda provider: TestService()
        collection.add_singleton(TestService, factory=factory)
        
        descriptor = collection._descriptors[0]
        assert descriptor.factory is factory
    
    def test_add_transient(self):
        """Test transient registration"""
        collection = ServiceCollection()
        collection.add_transient(TestService)
        
        assert len(collection._descriptors) == 1
        assert collection._descriptors[0].scope == Scope.TRANSIENT
    
    def test_add_scoped(self):
        """Test scoped registration"""
        collection = ServiceCollection()
        collection.add_scoped(TestService)
        
        assert len(collection._descriptors) == 1
        assert collection._descriptors[0].scope == Scope.SCOPED
    
    def test_build_service_provider(self):
        """Test building service provider"""
        collection = ServiceCollection()
        collection.add_singleton(TestService)
        
        provider = collection.build_service_provider()
        assert isinstance(provider, ServiceProvider)


class TestServiceProvider:
    """Test ServiceProvider"""
    
    def test_get_singleton_service(self):
        """Test getting singleton service"""
        collection = ServiceCollection()
        collection.add_singleton(TestService)
        provider = collection.build_service_provider()
        
        service1 = provider.get_service(TestService)
        service2 = provider.get_service(TestService)
        
        assert service1 is not None
        assert service1 is service2
        assert service1.id == service2.id
    
    def test_get_transient_service(self):
        """Test getting transient service"""
        collection = ServiceCollection()
        collection.add_transient(TestService)
        provider = collection.build_service_provider()
        
        service1 = provider.get_service(TestService)
        service2 = provider.get_service(TestService)
        
        assert service1 is not None
        assert service2 is not None
        assert service1 is not service2
        assert service1.id != service2.id
    
    def test_get_scoped_service(self):
        """Test getting scoped service"""
        collection = ServiceCollection()
        collection.add_scoped(TestService)
        provider = collection.build_service_provider()
        
        with provider.create_scope() as scope1:
            service1 = scope1.get_service(TestService)
            service2 = scope1.get_service(TestService)
            assert service1 is service2
        
        with provider.create_scope() as scope2:
            service3 = scope2.get_service(TestService)
            assert service3 is not service1
    
    def test_get_required_service(self):
        """Test getting required service"""
        collection = ServiceCollection()
        collection.add_singleton(TestService)
        provider = collection.build_service_provider()
        
        service = provider.get_required_service(TestService)
        assert service is not None
    
    def test_get_required_service_not_registered(self):
        """Test getting required service that's not registered"""
        collection = ServiceCollection()
        provider = collection.build_service_provider()
        
        with pytest.raises(ValueError, match="is not registered"):
            provider.get_required_service(TestService)
    
    def test_dependency_injection(self):
        """Test automatic dependency injection"""
        collection = ServiceCollection()
        collection.add_singleton(TestService)
        collection.add_singleton(DependentService)
        provider = collection.build_service_provider()
        
        dependent = provider.get_service(DependentService)
        assert dependent is not None
        assert dependent.test_service is not None
        assert isinstance(dependent.test_service, TestService)
    
    def test_factory_function(self):
        """Test factory function"""
        collection = ServiceCollection()
        created_services = []
        
        def factory(provider):
            service = TestService()
            created_services.append(service)
            return service
        
        collection.add_singleton(TestService, factory=factory)
        provider = collection.build_service_provider()
        
        service = provider.get_service(TestService)
        assert service is not None
        assert len(created_services) == 1
        assert service is created_services[0]
    
    def test_scope_disposal(self):
        """Test scope disposal"""
        collection = ServiceCollection()
        collection.add_scoped(TestService)
        provider = collection.build_service_provider()
        
        scope = provider.create_scope()
        service = scope.get_service(TestService)
        assert service is not None
        assert not service.disposed
        
        scope.dispose()
        assert service.disposed


class TestDependencyInjectionContainer:
    """Test DependencyInjectionContainer"""
    
    def test_singleton_pattern(self):
        """Test container is singleton"""
        container1 = DependencyInjectionContainer()
        container2 = DependencyInjectionContainer()
        assert container1 is container2
    
    def test_register_services(self):
        """Test registering services"""
        container = DependencyInjectionContainer()
        container.reset()
        container = DependencyInjectionContainer()
        
        def configurator(services: ServiceCollection):
            services.add_singleton(TestService)
        
        container.register_services(configurator)
        provider = container.build()
        
        service = provider.get_service(TestService)
        assert service is not None
    
    def test_get_service(self):
        """Test getting service from container"""
        container = DependencyInjectionContainer()
        container.reset()
        container = DependencyInjectionContainer()
        
        def configurator(services: ServiceCollection):
            services.add_singleton(TestService)
        
        container.register_services(configurator)
        container.build()
        
        service = container.get_service(TestService)
        assert service is not None
    
    def test_get_service_before_build(self):
        """Test getting service before build raises error"""
        container = DependencyInjectionContainer()
        container.reset()
        container = DependencyInjectionContainer()
        
        with pytest.raises(RuntimeError, match="Container not built"):
            container.get_service(TestService)
    
    def test_reset(self):
        """Test container reset"""
        container1 = DependencyInjectionContainer()
        container1.reset()
        container2 = DependencyInjectionContainer()
        assert container1 is not container2


class TestServiceFactory:
    """Test ServiceFactory"""
    
    def test_initialize(self):
        """Test factory initialization"""
        factory = ServiceFactory()
        assert not factory._initialized
        
        factory.initialize()
        assert factory._initialized
    
    def test_create_service(self):
        """Test creating service"""
        factory = ServiceFactory()
        factory.reset()
        
        with patch('src.factory.configure_services') as mock_config:
            def config(services):
                services.add_singleton(TestService)
            mock_config.side_effect = config
            
            factory._container.register_services(config)
            service = factory.create_service(TestService)
            assert service is not None
    
    def test_get_service(self):
        """Test getting service"""
        factory = ServiceFactory()
        factory.reset()
        
        def config(services):
            services.add_singleton(TestService)
        
        factory._container.register_services(config)
        service = factory.get_service(TestService)
        assert service is not None
    
    def test_create_scope(self):
        """Test creating scope"""
        factory = ServiceFactory()
        factory.reset()
        
        def config(services):
            services.add_scoped(TestService)
        
        factory._container.register_services(config)
        factory.initialize()
        
        scope = factory.create_scope()
        assert scope is not None
    
    def test_get_default(self):
        """Test getting default factory"""
        factory1 = ServiceFactory.get_default()
        factory2 = ServiceFactory.get_default()
        assert factory1 is factory2


class TestDecorators:
    """Test service decorators"""
    
    def test_singleton_decorator(self):
        """Test singleton decorator"""
        assert hasattr(SingletonService, '_injection_scope')
        assert SingletonService._injection_scope == Scope.SINGLETON
    
    def test_transient_decorator(self):
        """Test transient decorator"""
        assert hasattr(TransientService, '_injection_scope')
        assert TransientService._injection_scope == Scope.TRANSIENT
    
    def test_scoped_decorator(self):
        """Test scoped decorator"""
        assert hasattr(ScopedService, '_injection_scope')
        assert ScopedService._injection_scope == Scope.SCOPED
    
    def test_inject_decorator(self):
        """Test inject decorator with custom scope"""
        @inject(Scope.TRANSIENT)
        class CustomService:
            pass
        
        assert hasattr(CustomService, '_injection_scope')
        assert CustomService._injection_scope == Scope.TRANSIENT


class TestThreadSafety:
    """Test thread safety of DI container"""
    
    def test_singleton_thread_safety(self):
        """Test singleton is thread-safe"""
        collection = ServiceCollection()
        collection.add_singleton(TestService)
        provider = collection.build_service_provider()
        
        services = []
        
        def get_service():
            service = provider.get_service(TestService)
            services.append(service)
        
        threads = [threading.Thread(target=get_service) for _ in range(10)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        
        assert len(services) == 10
        assert all(s is services[0] for s in services)
    
    def test_container_singleton_thread_safety(self):
        """Test container singleton is thread-safe"""
        containers = []
        
        def get_container():
            container = DependencyInjectionContainer()
            containers.append(container)
        
        threads = [threading.Thread(target=get_container) for _ in range(10)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        
        assert len(containers) == 10
        assert all(c is containers[0] for c in containers)


class TestIntegration:
    """Integration tests for DI system"""
    
    @patch('src.container.get_settings')
    def test_real_services_registration(self, mock_settings):
        """Test real services can be registered"""
        mock_settings.return_value = Mock(spec=Settings)
        
        from src.container import configure_services
        
        collection = ServiceCollection()
        configure_services(collection)
        provider = collection.build_service_provider()
        
        from src.agents.learning_path_agent_v2 import LearningPathAgent
        from src.agents.study_buddy_agent_clean import StudyBuddyAgent
        
        with provider.create_scope() as scope:
            learning_agent = scope.get_service(LearningPathAgent)
            study_agent = scope.get_service(StudyBuddyAgent)
            
            assert learning_agent is not None
            assert study_agent is not None
    
    def test_circular_dependency_detection(self):
        """Test circular dependency is detected"""
        class ServiceA:
            def __init__(self, service_b: 'ServiceB'):
                self.service_b = service_b
        
        class ServiceB:
            def __init__(self, service_a: ServiceA):
                self.service_a = service_a
        
        collection = ServiceCollection()
        collection.add_singleton(ServiceA)
        collection.add_singleton(ServiceB)
        provider = collection.build_service_provider()
        
        with pytest.raises(ValueError):
            provider.get_service(ServiceA)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])