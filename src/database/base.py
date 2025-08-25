"""
Database base configuration with SQLAlchemy declarative base
"""

import asyncio
from typing import AsyncGenerator, Generator
from contextlib import asynccontextmanager, contextmanager

from sqlalchemy import MetaData
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session
from sqlalchemy.ext.asyncio import AsyncSession

from .session import SessionLocal, AsyncSessionLocal, engine, async_engine


# Naming convention for constraints (helps with migrations)
naming_convention = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s"
}

metadata = MetaData(naming_convention=naming_convention)

# Create declarative base
Base = declarative_base(metadata=metadata)


def get_db() -> Generator[Session, None, None]:
    """
    Dependency for getting database session.
    Usage:
        @app.get("/items/")
        def read_items(db: Session = Depends(get_db)):
            return db.query(Item).all()
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


async def get_async_db() -> AsyncGenerator[AsyncSession, None]:
    """
    Async dependency for getting database session.
    Usage:
        @app.get("/items/")
        async def read_items(db: AsyncSession = Depends(get_async_db)):
            result = await db.execute(select(Item))
            return result.scalars().all()
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


@contextmanager
def get_db_context():
    """
    Context manager for database session.
    Usage:
        with get_db_context() as db:
            items = db.query(Item).all()
    """
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


@asynccontextmanager
async def get_async_db_context():
    """
    Async context manager for database session.
    Usage:
        async with get_async_db_context() as db:
            result = await db.execute(select(Item))
            items = result.scalars().all()
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


def init_db():
    """
    Initialize database by creating all tables.
    Should only be used for development/testing.
    Use migrations for production.
    """
    import logging
    # Import all models to register them with SQLAlchemy
    from . import models
    
    logger = logging.getLogger(__name__)
    
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created successfully")
    except Exception as e:
        logger.error(f"Failed to create database tables: {e}")
        raise


async def init_async_db():
    """
    Async version of database initialization.
    """
    import logging
    # Import all models to register them with SQLAlchemy
    from . import models
    
    logger = logging.getLogger(__name__)
    
    try:
        async with async_engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("Database tables created successfully (async)")
    except Exception as e:
        logger.error(f"Failed to create database tables: {e}")
        raise