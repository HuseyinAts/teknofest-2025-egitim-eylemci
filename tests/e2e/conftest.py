"""
E2E test configuration and fixtures.
"""

import pytest
import asyncio
import os
import sys
from typing import Generator, AsyncGenerator
import docker
import time
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'src'))

from src.config import Config
from src.database.session import DatabaseSession
from src.database.models import Base


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
def docker_client():
    """Create Docker client for container management."""
    return docker.from_env()


@pytest.fixture(scope="session")
def test_database(docker_client):
    """Create test database container."""
    # Start PostgreSQL container
    container = docker_client.containers.run(
        "postgres:15",
        environment={
            "POSTGRES_USER": "test_user",
            "POSTGRES_PASSWORD": "test_password",
            "POSTGRES_DB": "test_teknofest"
        },
        ports={'5432/tcp': 5433},
        detach=True,
        remove=True,
        name="test_postgres_e2e"
    )
    
    # Wait for database to be ready
    time.sleep(5)
    
    # Create connection
    max_retries = 10
    for i in range(max_retries):
        try:
            conn = psycopg2.connect(
                host="localhost",
                port=5433,
                user="test_user",
                password="test_password",
                database="test_teknofest"
            )
            conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
            conn.close()
            break
        except psycopg2.OperationalError:
            if i == max_retries - 1:
                raise
            time.sleep(2)
    
    yield {
        'host': 'localhost',
        'port': 5433,
        'user': 'test_user',
        'password': 'test_password',
        'database': 'test_teknofest'
    }
    
    # Cleanup
    container.stop()


@pytest.fixture(scope="session")
def test_redis(docker_client):
    """Create test Redis container."""
    container = docker_client.containers.run(
        "redis:7",
        ports={'6379/tcp': 6380},
        detach=True,
        remove=True,
        name="test_redis_e2e"
    )
    
    # Wait for Redis to be ready
    time.sleep(3)
    
    yield {
        'host': 'localhost',
        'port': 6380
    }
    
    # Cleanup
    container.stop()


@pytest.fixture(scope="session")
def test_config(test_database, test_redis):
    """Create test configuration."""
    os.environ['DATABASE_URL'] = (
        f"postgresql://{test_database['user']}:{test_database['password']}"
        f"@{test_database['host']}:{test_database['port']}/{test_database['database']}"
    )
    os.environ['REDIS_URL'] = f"redis://{test_redis['host']}:{test_redis['port']}/0"
    os.environ['SECRET_KEY'] = 'test_secret_key_for_e2e_testing'
    os.environ['ENVIRONMENT'] = 'test'
    
    return Config()


@pytest.fixture(scope="session")
async def test_app(test_config):
    """Create test FastAPI application."""
    from src.app import create_app
    
    app = create_app(test_config)
    
    # Initialize database
    from src.database.session import engine
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    yield app
    
    # Cleanup database
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


@pytest.fixture(scope="session")
def api_server(test_app):
    """Start API server for E2E tests."""
    import uvicorn
    import threading
    
    def run_server():
        uvicorn.run(test_app, host="0.0.0.0", port=8000)
    
    thread = threading.Thread(target=run_server, daemon=True)
    thread.start()
    
    # Wait for server to start
    time.sleep(5)
    
    yield
    
    # Server will stop when tests complete


@pytest.fixture(scope="session")
def frontend_server():
    """Start frontend server for E2E tests."""
    import subprocess
    import os
    
    # Change to frontend directory
    frontend_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        'frontend'
    )
    
    # Start Next.js server
    process = subprocess.Popen(
        ['npm', 'run', 'dev'],
        cwd=frontend_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    # Wait for server to start
    time.sleep(10)
    
    yield
    
    # Stop server
    process.terminate()
    process.wait()


@pytest.fixture(autouse=True)
async def reset_database(test_config):
    """Reset database before each test."""
    from src.database.session import DatabaseSession
    from src.database.models import Base
    
    async with DatabaseSession() as session:
        # Clear all tables
        for table in reversed(Base.metadata.sorted_tables):
            await session.execute(table.delete())
        await session.commit()
    
    yield
    
    # Cleanup after test
    async with DatabaseSession() as session:
        for table in reversed(Base.metadata.sorted_tables):
            await session.execute(table.delete())
        await session.commit()


@pytest.fixture
def seed_test_data():
    """Seed test data for E2E tests."""
    async def _seed():
        from src.database.seeds import seed_all
        await seed_all()
    
    return _seed


# Pytest configuration
def pytest_configure(config):
    """Configure pytest for E2E tests."""
    config.addinivalue_line(
        "markers", "e2e: mark test as end-to-end test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "requires_browser: mark test as requiring browser"
    )