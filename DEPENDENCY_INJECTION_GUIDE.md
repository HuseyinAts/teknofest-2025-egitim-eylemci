# Production-Ready Dependency Injection Guide

## Overview

This project implements a production-ready dependency injection (DI) system for the TEKNOFEST 2025 Eğitim Teknolojileri project. The DI system provides:

- **Service lifecycle management** (Singleton, Transient, Scoped)
- **Automatic dependency resolution**
- **Thread-safe service creation**
- **Factory pattern integration**
- **Scope management for request-based services**
- **Decorator-based service registration**

## Architecture

### Core Components

1. **ServiceCollection**: Registry for service descriptors
2. **ServiceProvider**: Resolves and creates service instances
3. **ServiceScope**: Manages scoped service lifetimes
4. **DependencyInjectionContainer**: Global container management
5. **ServiceFactory**: Factory pattern for service creation

### Service Lifetimes

- **Singleton**: Single instance throughout application lifetime
- **Transient**: New instance for every request
- **Scoped**: Single instance per scope/request

## Usage Examples

### 1. Basic Service Registration

```python
from src.container import ServiceCollection, singleton, transient, scoped

# Using decorators
@singleton
class DatabaseService:
    def __init__(self):
        self.connection = create_connection()

@transient
class EmailService:
    def __init__(self, database: DatabaseService):
        self.database = database

@scoped
class RequestHandler:
    def __init__(self, email: EmailService):
        self.email = email
```

### 2. Manual Registration

```python
from src.container import ServiceCollection

def configure_services(services: ServiceCollection):
    # Singleton with implementation
    services.add_singleton(IDatabase, PostgresDatabase)
    
    # Singleton with factory
    services.add_singleton(
        Redis,
        factory=lambda provider: Redis.from_url(
            provider.get_required_service(Settings).redis_url
        )
    )
    
    # Singleton with instance
    services.add_singleton(Settings, instance=get_settings())
    
    # Transient service
    services.add_transient(EmailSender)
    
    # Scoped service
    services.add_scoped(RequestContext)
```

### 3. Using the Container

```python
from src.container import DependencyInjectionContainer

# Initialize container
container = DependencyInjectionContainer()
container.register_services(configure_services)
provider = container.build()

# Get services
database = provider.get_required_service(IDatabase)
email = provider.get_service(EmailSender)

# Using scopes
with provider.create_scope() as scope:
    request_handler = scope.get_service(RequestHandler)
    # Scoped services are disposed when scope exits
```

### 4. Factory Pattern Integration

```python
from src.factory import ServiceFactory, get_factory

# Get default factory
factory = get_factory()

# Create service with dependencies
agent = factory.create_service(LearningPathAgent)

# Create scoped services
with factory.create_scope() as scope:
    handler = scope.get_service(RequestHandler)
```

### 5. FastAPI Integration

```python
from fastapi import Depends
from src.factory import get_factory

def get_learning_agent() -> LearningPathAgent:
    """Dependency injection for FastAPI"""
    factory = get_factory()
    with factory.create_scope() as scope:
        return scope.get_service(LearningPathAgent)

@app.post("/api/learning-path")
async def create_learning_path(
    agent: LearningPathAgent = Depends(get_learning_agent)
):
    return agent.create_path()
```

## Service Configuration

The main service configuration is in `src/container.py`:

```python
def configure_services(services: ServiceCollection):
    # Settings (Singleton)
    services.add_singleton(Settings, factory=lambda _: get_settings())
    
    # Data Processing (Singleton)
    services.add_singleton(DataProcessor)
    services.add_singleton(ModelIntegration)
    
    # Agents (Scoped - per request)
    services.add_scoped(LearningPathAgent)
    services.add_scoped(StudyBuddyAgent)
    
    # Database (Singleton)
    services.add_singleton(
        create_engine,
        factory=lambda provider: create_engine(
            provider.get_required_service(Settings).database_url
        )
    )
```

## Best Practices

### 1. Service Design

```python
# Good: Constructor injection
class MyService:
    def __init__(self, database: DatabaseService, cache: CacheService):
        self.database = database
        self.cache = cache

# Bad: Service locator pattern
class MyService:
    def __init__(self):
        self.database = container.get_service(DatabaseService)
```

### 2. Lifecycle Management

```python
# Singleton for stateless services
@singleton
class CalculatorService:
    def add(self, a, b):
        return a + b

# Scoped for request-specific data
@scoped
class UserContext:
    def __init__(self):
        self.user_id = None
        self.permissions = []

# Transient for disposable resources
@transient
class TempFileHandler:
    def __init__(self):
        self.temp_file = create_temp_file()
```

### 3. Testing

```python
import pytest
from src.container import ServiceCollection

@pytest.fixture
def test_provider():
    """Test provider with mocked services"""
    collection = ServiceCollection()
    collection.add_singleton(DatabaseService, instance=MockDatabase())
    collection.add_singleton(CacheService, instance=MockCache())
    return collection.build_service_provider()

def test_my_service(test_provider):
    service = test_provider.get_required_service(MyService)
    assert service.process() == expected_result
```

## Advanced Features

### 1. Factory Functions

```python
def create_redis_client(provider: ServiceProvider) -> Redis:
    settings = provider.get_required_service(Settings)
    return Redis(
        host=settings.redis_host,
        port=settings.redis_port,
        password=settings.redis_password
    )

services.add_singleton(Redis, factory=create_redis_client)
```

### 2. Conditional Registration

```python
def configure_services(services: ServiceCollection):
    settings = get_settings()
    
    if settings.use_redis:
        services.add_singleton(ICacheService, RedisCache)
    else:
        services.add_singleton(ICacheService, MemoryCache)
```

### 3. Multiple Implementations

```python
# Register multiple implementations
services.add_singleton(INotifier, EmailNotifier)
services.add_singleton(INotifier, SmsNotifier)

# Get all implementations
notifiers = provider.get_services(INotifier)
for notifier in notifiers:
    notifier.send(message)
```

## Thread Safety

The DI system is fully thread-safe:

- Singleton creation uses locks to prevent race conditions
- Scoped instances are thread-local
- Container initialization is thread-safe

```python
import threading

def worker():
    service = provider.get_required_service(MyService)
    service.process()

threads = [threading.Thread(target=worker) for _ in range(10)]
for t in threads:
    t.start()
for t in threads:
    t.join()
```

## Migration Guide

### From Direct Instantiation

Before:
```python
class MyController:
    def __init__(self):
        self.database = DatabaseService()
        self.cache = CacheService()
        self.email = EmailService(self.database)
```

After:
```python
class MyController:
    def __init__(self, database: DatabaseService, 
                 cache: CacheService, 
                 email: EmailService):
        self.database = database
        self.cache = cache
        self.email = email
```

### From Global Instances

Before:
```python
# globals.py
database = DatabaseService()
cache = CacheService()

# usage.py
from globals import database, cache
```

After:
```python
# services.py
@singleton
class DatabaseService:
    pass

@singleton
class CacheService:
    pass

# usage.py
def my_function(database: DatabaseService = Depends(get_database)):
    pass
```

## Performance Considerations

1. **Singleton services** are created once and cached
2. **Scoped services** are created per scope/request
3. **Transient services** have minimal overhead
4. **Dependency resolution** is optimized with caching

## Troubleshooting

### Common Issues

1. **Circular Dependencies**
   ```python
   # Error: Circular dependency detected
   class A:
       def __init__(self, b: B): pass
   
   class B:
       def __init__(self, a: A): pass
   
   # Solution: Use factory or redesign
   ```

2. **Missing Dependencies**
   ```python
   # Error: Cannot resolve dependency
   # Solution: Ensure all dependencies are registered
   services.add_singleton(RequiredService)
   ```

3. **Scope Lifetime**
   ```python
   # Wrong: Singleton depending on scoped
   @singleton
   class MyService:
       def __init__(self, scoped: ScopedService): pass
   
   # Correct: Scoped depending on singleton
   @scoped
   class MyService:
       def __init__(self, singleton: SingletonService): pass
   ```

## Testing

Run the comprehensive test suite:

```bash
# Run all DI tests
pytest tests/test_dependency_injection.py -v

# Run specific test class
pytest tests/test_dependency_injection.py::TestServiceProvider -v

# Run with coverage
pytest tests/test_dependency_injection.py --cov=src.container
```

## Summary

The dependency injection system provides:

✅ **Production-ready** architecture  
✅ **Thread-safe** operations  
✅ **Flexible** service lifetimes  
✅ **Automatic** dependency resolution  
✅ **Easy** testing and mocking  
✅ **Clean** separation of concerns  
✅ **Scalable** service management  

This system ensures maintainable, testable, and scalable code for the TEKNOFEST 2025 project.