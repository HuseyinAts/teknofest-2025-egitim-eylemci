"""
Database transaction decorators and utilities
"""

import logging
import functools
from typing import Callable, Any, Optional, TypeVar, Union
import asyncio
from contextlib import contextmanager

from sqlalchemy.orm import Session
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.exc import SQLAlchemyError, IntegrityError, OperationalError

from .session import SessionLocal, AsyncSessionLocal

logger = logging.getLogger(__name__)

F = TypeVar('F', bound=Callable[..., Any])


def transactional(
    auto_commit: bool = True,
    auto_rollback: bool = True,
    retry_on: Optional[tuple] = None,
    max_retries: int = 3
):
    """
    Decorator for automatic transaction management.
    
    Args:
        auto_commit: Automatically commit on success
        auto_rollback: Automatically rollback on error
        retry_on: Tuple of exceptions to retry on
        max_retries: Maximum number of retries
    
    Usage:
        @transactional()
        def create_user(db: Session, name: str):
            user = User(name=name)
            db.add(user)
            return user
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Check if session is passed as argument
            db = None
            for arg in args:
                if isinstance(arg, Session):
                    db = arg
                    break
            
            if 'db' in kwargs and isinstance(kwargs['db'], Session):
                db = kwargs['db']
            
            # If no session provided, create one
            if db is None:
                with SessionLocal() as db:
                    kwargs['db'] = db
                    return _execute_with_transaction(
                        func, args, kwargs, db,
                        auto_commit, auto_rollback,
                        retry_on, max_retries
                    )
            else:
                # Session provided, use it
                return _execute_with_transaction(
                    func, args, kwargs, db,
                    auto_commit, auto_rollback,
                    retry_on, max_retries
                )
        
        return wrapper
    return decorator


def async_transactional(
    auto_commit: bool = True,
    auto_rollback: bool = True,
    retry_on: Optional[tuple] = None,
    max_retries: int = 3
):
    """
    Async version of transactional decorator.
    
    Usage:
        @async_transactional()
        async def create_user(db: AsyncSession, name: str):
            user = User(name=name)
            db.add(user)
            return user
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # Check if session is passed as argument
            db = None
            for arg in args:
                if isinstance(arg, AsyncSession):
                    db = arg
                    break
            
            if 'db' in kwargs and isinstance(kwargs['db'], AsyncSession):
                db = kwargs['db']
            
            # If no session provided, create one
            if db is None:
                async with AsyncSessionLocal() as db:
                    kwargs['db'] = db
                    return await _async_execute_with_transaction(
                        func, args, kwargs, db,
                        auto_commit, auto_rollback,
                        retry_on, max_retries
                    )
            else:
                # Session provided, use it
                return await _async_execute_with_transaction(
                    func, args, kwargs, db,
                    auto_commit, auto_rollback,
                    retry_on, max_retries
                )
        
        return wrapper
    return decorator


def _execute_with_transaction(
    func: Callable,
    args: tuple,
    kwargs: dict,
    db: Session,
    auto_commit: bool,
    auto_rollback: bool,
    retry_on: Optional[tuple],
    max_retries: int
) -> Any:
    """Execute function with transaction management."""
    retries = 0
    last_error = None
    
    while retries <= max_retries:
        try:
            result = func(*args, **kwargs)
            
            if auto_commit:
                db.commit()
                logger.debug(f"Transaction committed for {func.__name__}")
            
            return result
            
        except Exception as e:
            if auto_rollback:
                db.rollback()
                logger.debug(f"Transaction rolled back for {func.__name__}")
            
            # Check if we should retry
            if retry_on and isinstance(e, retry_on) and retries < max_retries:
                retries += 1
                last_error = e
                logger.warning(
                    f"Retrying {func.__name__} (attempt {retries}/{max_retries}) "
                    f"after error: {e}"
                )
                # Exponential backoff
                import time
                time.sleep(2 ** retries)
                continue
            
            logger.error(f"Transaction failed for {func.__name__}: {e}")
            raise
    
    # Max retries reached
    logger.error(
        f"Max retries ({max_retries}) reached for {func.__name__}. "
        f"Last error: {last_error}"
    )
    raise last_error


async def _async_execute_with_transaction(
    func: Callable,
    args: tuple,
    kwargs: dict,
    db: AsyncSession,
    auto_commit: bool,
    auto_rollback: bool,
    retry_on: Optional[tuple],
    max_retries: int
) -> Any:
    """Async version of execute with transaction."""
    retries = 0
    last_error = None
    
    while retries <= max_retries:
        try:
            result = await func(*args, **kwargs)
            
            if auto_commit:
                await db.commit()
                logger.debug(f"Transaction committed for {func.__name__}")
            
            return result
            
        except Exception as e:
            if auto_rollback:
                await db.rollback()
                logger.debug(f"Transaction rolled back for {func.__name__}")
            
            # Check if we should retry
            if retry_on and isinstance(e, retry_on) and retries < max_retries:
                retries += 1
                last_error = e
                logger.warning(
                    f"Retrying {func.__name__} (attempt {retries}/{max_retries}) "
                    f"after error: {e}"
                )
                # Exponential backoff
                await asyncio.sleep(2 ** retries)
                continue
            
            logger.error(f"Transaction failed for {func.__name__}: {e}")
            raise
    
    # Max retries reached
    logger.error(
        f"Max retries ({max_retries}) reached for {func.__name__}. "
        f"Last error: {last_error}"
    )
    raise last_error


def with_retries(
    exceptions: tuple = (OperationalError,),
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0
):
    """
    Decorator for retrying database operations.
    
    Args:
        exceptions: Exceptions to catch and retry
        max_attempts: Maximum retry attempts
        delay: Initial delay between retries
        backoff: Backoff multiplier for delay
    
    Usage:
        @with_retries()
        def get_user(user_id: int):
            return db.query(User).get(user_id)
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            attempt = 1
            current_delay = delay
            
            while attempt <= max_attempts:
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    if attempt == max_attempts:
                        logger.error(
                            f"Max attempts ({max_attempts}) reached for {func.__name__}. "
                            f"Error: {e}"
                        )
                        raise
                    
                    logger.warning(
                        f"Attempt {attempt}/{max_attempts} failed for {func.__name__}. "
                        f"Retrying in {current_delay}s. Error: {e}"
                    )
                    
                    import time
                    time.sleep(current_delay)
                    current_delay *= backoff
                    attempt += 1
        
        return wrapper
    return decorator


def async_with_retries(
    exceptions: tuple = (OperationalError,),
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0
):
    """Async version of with_retries decorator."""
    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            attempt = 1
            current_delay = delay
            
            while attempt <= max_attempts:
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    if attempt == max_attempts:
                        logger.error(
                            f"Max attempts ({max_attempts}) reached for {func.__name__}. "
                            f"Error: {e}"
                        )
                        raise
                    
                    logger.warning(
                        f"Attempt {attempt}/{max_attempts} failed for {func.__name__}. "
                        f"Retrying in {current_delay}s. Error: {e}"
                    )
                    
                    await asyncio.sleep(current_delay)
                    current_delay *= backoff
                    attempt += 1
        
        return wrapper
    return decorator


@contextmanager
def atomic_transaction(db: Session):
    """
    Context manager for atomic transactions.
    
    Usage:
        with atomic_transaction(db) as session:
            user = User(name="John")
            session.add(user)
            # Automatically commits or rolls back
    """
    try:
        yield db
        db.commit()
        logger.debug("Atomic transaction committed")
    except Exception as e:
        db.rollback()
        logger.error(f"Atomic transaction rolled back: {e}")
        raise
    finally:
        db.close()


def handle_integrity_error(
    default_return: Any = None,
    message: Optional[str] = None,
    raise_on_duplicate: bool = False
):
    """
    Decorator to handle integrity errors gracefully.
    
    Args:
        default_return: Value to return on integrity error
        message: Custom error message
        raise_on_duplicate: Raise exception on duplicate key errors
    
    Usage:
        @handle_integrity_error(default_return=None, message="User already exists")
        def create_user(email: str):
            # Create user logic
            pass
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except IntegrityError as e:
                error_msg = message or f"Integrity error in {func.__name__}: {e}"
                
                # Check if it's a duplicate key error
                if 'duplicate key' in str(e).lower():
                    if raise_on_duplicate:
                        logger.error(f"Duplicate key error: {error_msg}")
                        raise
                    else:
                        logger.warning(f"Duplicate key ignored: {error_msg}")
                else:
                    logger.error(error_msg)
                
                return default_return
        
        return wrapper
    return decorator


def bulk_operation(chunk_size: int = 1000):
    """
    Decorator for handling bulk database operations.
    
    Args:
        chunk_size: Size of chunks for bulk operations
    
    Usage:
        @bulk_operation(chunk_size=500)
        def bulk_insert_users(db: Session, users: List[dict]):
            # Function receives chunked data automatically
            pass
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(db: Session, data: list, *args, **kwargs):
            results = []
            
            # Process data in chunks
            for i in range(0, len(data), chunk_size):
                chunk = data[i:i + chunk_size]
                
                try:
                    result = func(db, chunk, *args, **kwargs)
                    results.extend(result if isinstance(result, list) else [result])
                    
                    # Commit after each chunk
                    db.commit()
                    logger.debug(f"Processed chunk {i//chunk_size + 1} of {func.__name__}")
                    
                except Exception as e:
                    db.rollback()
                    logger.error(
                        f"Error processing chunk {i//chunk_size + 1} in {func.__name__}: {e}"
                    )
                    raise
            
            return results
        
        return wrapper
    return decorator