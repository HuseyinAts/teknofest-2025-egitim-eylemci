"""
Student API Endpoints - Clean Code Implementation
TEKNOFEST 2025 - Student Management Routes
"""

import logging
from typing import List, Optional
from fastapi import APIRouter, Depends, status, Path, Query, Body

from src.infrastructure.config.container import get_container
from src.domain.entities import Student
from src.presentation.api.v1.schemas import (
    StudentCreateRequest,
    StudentResponse,
    StudentUpdateRequest,
    ErrorResponse
)
from src.shared.exceptions import ApplicationError

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/",
    response_model=StudentResponse,
    status_code=status.HTTP_201_CREATED,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid request"},
        409: {"model": ErrorResponse, "description": "Student already exists"}
    },
    summary="Create Student",
    description="Create a new student account"
)
async def create_student(
    request: StudentCreateRequest = Body(...)
) -> StudentResponse:
    """
    Create a new student account.
    
    Requirements:
    - Unique email address
    - Unique username
    - Valid grade level (9-12)
    - Strong password
    """
    try:
        container = get_container()
        
        async with container.unit_of_work() as uow:
            # Check if student exists
            existing = await uow.students.get_by_email(request.email)
            if existing:
                from src.shared.exceptions import DuplicateEntityError
                raise DuplicateEntityError("Student", "email", request.email)
            
            existing = await uow.students.get_by_username(request.username)
            if existing:
                from src.shared.exceptions import DuplicateEntityError
                raise DuplicateEntityError("Student", "username", request.username)
            
            # Create student
            student = Student.create_new(
                email=request.email,
                username=request.username,
                full_name=request.full_name,
                grade=request.grade
            )
            
            # Save student
            saved_student = await uow.students.save(student)
            await uow.commit()
        
        # Convert to response
        return StudentResponse(
            id=saved_student.id,
            email=saved_student.email,
            username=saved_student.username,
            full_name=saved_student.full_name,
            grade=saved_student.grade.level,
            learning_style=saved_student.learning_style.to_dict() if saved_student.learning_style else None,
            ability_level=saved_student.ability_level.value,
            status=saved_student.status.value,
            created_at=saved_student.created_at,
            updated_at=saved_student.updated_at
        )
        
    except ApplicationError as e:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in create_student: {e}")
        raise


@router.get(
    "/{student_id}",
    response_model=StudentResponse,
    status_code=status.HTTP_200_OK,
    responses={
        404: {"model": ErrorResponse, "description": "Student not found"}
    },
    summary="Get Student",
    description="Get student details by ID"
)
async def get_student(
    student_id: str = Path(..., description="Student UUID")
) -> StudentResponse:
    """
    Get student details by ID.
    
    Returns complete student profile including:
    - Basic information
    - Learning style (if assessed)
    - Current ability level
    - Account status
    """
    try:
        container = get_container()
        student_repo = container.student_repository()
        
        student = await student_repo.get_by_id(student_id)
        
        if not student:
            from src.shared.exceptions import StudentNotFoundException
            raise StudentNotFoundException(student_id)
        
        return StudentResponse(
            id=student.id,
            email=student.email,
            username=student.username,
            full_name=student.full_name,
            grade=student.grade.level,
            learning_style=student.learning_style.to_dict() if student.learning_style else None,
            ability_level=student.ability_level.value,
            status=student.status.value,
            created_at=student.created_at,
            updated_at=student.updated_at
        )
        
    except ApplicationError as e:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in get_student: {e}")
        raise


@router.put(
    "/{student_id}",
    response_model=StudentResponse,
    status_code=status.HTTP_200_OK,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid request"},
        404: {"model": ErrorResponse, "description": "Student not found"}
    },
    summary="Update Student",
    description="Update student information"
)
async def update_student(
    student_id: str = Path(..., description="Student UUID"),
    request: StudentUpdateRequest = Body(...)
) -> StudentResponse:
    """
    Update student information.
    
    Updatable fields:
    - Full name
    - Grade level
    """
    try:
        container = get_container()
        
        async with container.unit_of_work() as uow:
            # Get student
            student = await uow.students.get_by_id(student_id)
            
            if not student:
                from src.shared.exceptions import StudentNotFoundException
                raise StudentNotFoundException(student_id)
            
            # Update fields
            if request.full_name:
                student.full_name = request.full_name
            
            if request.grade:
                from src.domain.value_objects import Grade
                student.grade = Grade(request.grade)
            
            # Save updates
            updated_student = await uow.students.save(student)
            await uow.commit()
        
        return StudentResponse(
            id=updated_student.id,
            email=updated_student.email,
            username=updated_student.username,
            full_name=updated_student.full_name,
            grade=updated_student.grade.level,
            learning_style=updated_student.learning_style.to_dict() if updated_student.learning_style else None,
            ability_level=updated_student.ability_level.value,
            status=updated_student.status.value,
            created_at=updated_student.created_at,
            updated_at=updated_student.updated_at
        )
        
    except ApplicationError as e:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in update_student: {e}")
        raise


@router.get(
    "/",
    response_model=List[StudentResponse],
    status_code=status.HTTP_200_OK,
    summary="List Students",
    description="Get list of students with optional filters"
)
async def list_students(
    grade: Optional[int] = Query(None, ge=9, le=12, description="Filter by grade"),
    active_only: bool = Query(True, description="Only show active students"),
    limit: int = Query(50, ge=1, le=100, description="Maximum number of results"),
    offset: int = Query(0, ge=0, description="Number of results to skip")
) -> List[StudentResponse]:
    """
    Get list of students with optional filters.
    
    Filters:
    - Grade level
    - Active status
    - Pagination (limit/offset)
    """
    try:
        container = get_container()
        student_repo = container.student_repository()
        
        if grade:
            from src.domain.value_objects import Grade
            students = await student_repo.get_by_grade(Grade(grade), limit=limit)
        elif active_only:
            students = await student_repo.get_active_students(limit=limit)
        else:
            students = await student_repo.get_all(limit=limit, offset=offset)
        
        return [
            StudentResponse(
                id=student.id,
                email=student.email,
                username=student.username,
                full_name=student.full_name,
                grade=student.grade.level,
                learning_style=student.learning_style.to_dict() if student.learning_style else None,
                ability_level=student.ability_level.value,
                status=student.status.value,
                created_at=student.created_at,
                updated_at=student.updated_at
            )
            for student in students
        ]
        
    except ApplicationError as e:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in list_students: {e}")
        raise


@router.delete(
    "/{student_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    responses={
        404: {"model": ErrorResponse, "description": "Student not found"}
    },
    summary="Delete Student",
    description="Soft delete a student account"
)
async def delete_student(
    student_id: str = Path(..., description="Student UUID")
):
    """
    Soft delete a student account.
    
    This will:
    - Mark the account as inactive
    - Preserve data for audit purposes
    - Prevent login
    """
    try:
        container = get_container()
        
        async with container.unit_of_work() as uow:
            success = await uow.students.delete(student_id)
            
            if not success:
                from src.shared.exceptions import StudentNotFoundException
                raise StudentNotFoundException(student_id)
            
            await uow.commit()
        
        return None
        
    except ApplicationError as e:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in delete_student: {e}")
        raise
