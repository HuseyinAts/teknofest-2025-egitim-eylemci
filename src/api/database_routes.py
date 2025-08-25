"""
Database CRUD API Routes - Production Ready
TEKNOFEST 2025 - Database Operations
"""

import logging
from typing import List, Optional, Any, Dict
from datetime import datetime, timedelta
from fastapi import APIRouter, HTTPException, Depends, status, Query, BackgroundTasks
from pydantic import BaseModel, Field, EmailStr, validator
from sqlalchemy import select, update, delete, and_, or_, func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from src.database.session import get_async_session
from src.database.models import (
    User, Student, Teacher, Course, Module, 
    Quiz, Question, Answer, QuizAttempt,
    Progress, LearningPath, Notification, ActivityLog,
    UserRole, DifficultyLevel, QuestionType, LearningStyle
)
from src.core.authentication import get_current_user, get_password_hash
from src.core.security import check_permission

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/v1/database",
    tags=["Database Operations"],
    responses={404: {"description": "Not found"}}
)


# ==================== User Models ====================

class UserCreate(BaseModel):
    """User creation model"""
    username: str = Field(..., min_length=3, max_length=50)
    email: EmailStr
    password: str = Field(..., min_length=8)
    full_name: Optional[str] = Field(None, max_length=100)
    phone: Optional[str] = Field(None, regex="^[0-9+()-]+$", max_length=20)
    role: str = Field("student", regex="^(student|teacher|admin|parent)$")


class UserUpdate(BaseModel):
    """User update model"""
    full_name: Optional[str] = Field(None, max_length=100)
    phone: Optional[str] = Field(None, regex="^[0-9+()-]+$", max_length=20)
    bio: Optional[str] = Field(None, max_length=500)
    avatar_url: Optional[str] = Field(None, max_length=255)
    preferences: Optional[Dict[str, Any]] = None


class StudentCreate(BaseModel):
    """Student profile creation"""
    grade: int = Field(..., ge=1, le=12)
    school: Optional[str] = Field(None, max_length=200)
    student_number: Optional[str] = Field(None, max_length=50)
    learning_style: str = Field("mixed", regex="^(visual|auditory|reading|kinesthetic|mixed)$")
    target_level: float = Field(0.8, ge=0, le=1)
    study_hours_per_day: float = Field(2.0, gt=0, le=12)


class TeacherCreate(BaseModel):
    """Teacher profile creation"""
    subject: str = Field(..., min_length=1, max_length=100)
    qualification: Optional[str] = Field(None, max_length=200)
    years_of_experience: int = Field(0, ge=0)
    school: Optional[str] = Field(None, max_length=200)
    specializations: List[str] = Field(default_factory=list)


class CourseCreate(BaseModel):
    """Course creation model"""
    title: str = Field(..., min_length=1, max_length=200)
    description: Optional[str] = None
    subject: str = Field(..., min_length=1, max_length=100)
    grade_level: int = Field(..., ge=1, le=12)
    difficulty: float = Field(0.5, ge=0, le=1)
    estimated_hours: int = Field(30, gt=0)
    max_students: int = Field(100, gt=0)
    curriculum_alignment: Optional[str] = None
    prerequisites: List[str] = Field(default_factory=list)
    learning_objectives: List[str] = Field(default_factory=list)


class ModuleCreate(BaseModel):
    """Module creation model"""
    course_id: int
    title: str = Field(..., min_length=1, max_length=200)
    description: Optional[str] = None
    order: int = Field(0, ge=0)
    content: Optional[str] = None
    video_url: Optional[str] = None
    estimated_time: int = Field(60, gt=0)
    is_required: bool = True
    resources: List[Dict] = Field(default_factory=list)


# ==================== User Operations ====================

@router.post("/users/create", status_code=status.HTTP_201_CREATED)
async def create_user(
    user_data: UserCreate,
    db: AsyncSession = Depends(get_async_session),
    current_user=Depends(get_current_user)
):
    """Create a new user (admin only)"""
    try:
        # Check admin permission
        if not current_user or current_user.role != UserRole.ADMIN:
            raise HTTPException(status_code=403, detail="Admin access required")
        
        # Check if username exists
        existing = await db.execute(
            select(User).where(User.username == user_data.username)
        )
        if existing.scalar_one_or_none():
            raise HTTPException(status_code=400, detail="Username already exists")
        
        # Check if email exists
        existing = await db.execute(
            select(User).where(User.email == user_data.email.lower())
        )
        if existing.scalar_one_or_none():
            raise HTTPException(status_code=400, detail="Email already exists")
        
        # Create user
        user = User(
            username=user_data.username,
            email=user_data.email.lower(),
            hashed_password=get_password_hash(user_data.password),
            full_name=user_data.full_name,
            phone=user_data.phone,
            role=UserRole[user_data.role.upper()]
        )
        
        db.add(user)
        await db.commit()
        await db.refresh(user)
        
        # Create profile based on role
        if user_data.role == "student":
            student = Student(user_id=user.id, grade=9)
            db.add(student)
        elif user_data.role == "teacher":
            teacher = Teacher(user_id=user.id, subject="General")
            db.add(teacher)
        
        await db.commit()
        
        return {
            "success": True,
            "data": {
                "id": user.id,
                "username": user.username,
                "email": user.email,
                "role": user.role.value,
                "created_at": user.created_at.isoformat()
            },
            "message": "User created successfully"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"User creation error: {e}")
        await db.rollback()
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/users/{user_id}")
async def get_user(
    user_id: int,
    db: AsyncSession = Depends(get_async_session),
    current_user=Depends(get_current_user)
):
    """Get user details"""
    try:
        # Check permission
        if not current_user or (current_user.id != user_id and current_user.role != UserRole.ADMIN):
            raise HTTPException(status_code=403, detail="Access denied")
        
        result = await db.execute(
            select(User).where(User.id == user_id)
            .options(selectinload(User.student_profile))
            .options(selectinload(User.teacher_profile))
        )
        user = result.scalar_one_or_none()
        
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        user_data = {
            "id": user.id,
            "username": user.username,
            "email": user.email,
            "full_name": user.full_name,
            "phone": user.phone,
            "role": user.role.value,
            "is_active": user.is_active,
            "is_verified": user.is_verified,
            "created_at": user.created_at.isoformat(),
            "last_login": user.last_login.isoformat() if user.last_login else None
        }
        
        # Add profile data
        if user.student_profile:
            user_data["student_profile"] = {
                "grade": user.student_profile.grade,
                "school": user.student_profile.school,
                "learning_style": user.student_profile.learning_style.value,
                "current_level": user.student_profile.current_level,
                "average_score": user.student_profile.average_score
            }
        elif user.teacher_profile:
            user_data["teacher_profile"] = {
                "subject": user.teacher_profile.subject,
                "qualification": user.teacher_profile.qualification,
                "years_of_experience": user.teacher_profile.years_of_experience,
                "average_rating": user.teacher_profile.average_rating
            }
        
        return {"success": True, "data": user_data}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"User fetch error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/users/{user_id}")
async def update_user(
    user_id: int,
    user_update: UserUpdate,
    db: AsyncSession = Depends(get_async_session),
    current_user=Depends(get_current_user)
):
    """Update user details"""
    try:
        # Check permission
        if not current_user or (current_user.id != user_id and current_user.role != UserRole.ADMIN):
            raise HTTPException(status_code=403, detail="Access denied")
        
        result = await db.execute(
            select(User).where(User.id == user_id)
        )
        user = result.scalar_one_or_none()
        
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        # Update fields
        update_data = user_update.dict(exclude_unset=True)
        for field, value in update_data.items():
            setattr(user, field, value)
        
        user.updated_at = datetime.now()
        
        await db.commit()
        await db.refresh(user)
        
        return {
            "success": True,
            "message": "User updated successfully"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"User update error: {e}")
        await db.rollback()
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/users/{user_id}")
async def delete_user(
    user_id: int,
    db: AsyncSession = Depends(get_async_session),
    current_user=Depends(get_current_user)
):
    """Soft delete user (admin only)"""
    try:
        # Check admin permission
        if not current_user or current_user.role != UserRole.ADMIN:
            raise HTTPException(status_code=403, detail="Admin access required")
        
        result = await db.execute(
            select(User).where(User.id == user_id)
        )
        user = result.scalar_one_or_none()
        
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        # Soft delete
        user.is_deleted = True
        user.deleted_at = datetime.now()
        user.is_active = False
        
        await db.commit()
        
        return {
            "success": True,
            "message": "User deleted successfully"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"User deletion error: {e}")
        await db.rollback()
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Student Operations ====================

@router.post("/students/create", status_code=status.HTTP_201_CREATED)
async def create_student_profile(
    student_data: StudentCreate,
    db: AsyncSession = Depends(get_async_session),
    current_user=Depends(get_current_user)
):
    """Create student profile for current user"""
    try:
        if not current_user:
            raise HTTPException(status_code=401, detail="Authentication required")
        
        # Check if profile exists
        existing = await db.execute(
            select(Student).where(Student.user_id == current_user.id)
        )
        if existing.scalar_one_or_none():
            raise HTTPException(status_code=400, detail="Student profile already exists")
        
        # Create student profile
        student = Student(
            user_id=current_user.id,
            grade=student_data.grade,
            school=student_data.school,
            student_number=student_data.student_number,
            learning_style=LearningStyle[student_data.learning_style.upper()],
            target_level=student_data.target_level,
            study_hours_per_day=student_data.study_hours_per_day
        )
        
        db.add(student)
        await db.commit()
        await db.refresh(student)
        
        return {
            "success": True,
            "data": {
                "id": student.id,
                "user_id": student.user_id,
                "grade": student.grade,
                "learning_style": student.learning_style.value,
                "created_at": student.created_at.isoformat()
            },
            "message": "Student profile created successfully"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Student profile creation error: {e}")
        await db.rollback()
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/students/me")
async def get_my_student_profile(
    db: AsyncSession = Depends(get_async_session),
    current_user=Depends(get_current_user)
):
    """Get current user's student profile"""
    try:
        if not current_user:
            raise HTTPException(status_code=401, detail="Authentication required")
        
        result = await db.execute(
            select(Student).where(Student.user_id == current_user.id)
            .options(selectinload(Student.enrolled_courses))
            .options(selectinload(Student.learning_paths))
        )
        student = result.scalar_one_or_none()
        
        if not student:
            raise HTTPException(status_code=404, detail="Student profile not found")
        
        return {
            "success": True,
            "data": {
                "id": student.id,
                "grade": student.grade,
                "school": student.school,
                "learning_style": student.learning_style.value,
                "current_level": student.current_level,
                "target_level": student.target_level,
                "average_score": student.average_score,
                "total_points": student.total_points,
                "quiz_count": student.quiz_count,
                "streak_days": student.streak_days,
                "enrolled_courses": len(student.enrolled_courses),
                "active_paths": len([p for p in student.learning_paths if p.is_active])
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Student profile fetch error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Course Operations ====================

@router.post("/courses/create", status_code=status.HTTP_201_CREATED)
async def create_course(
    course_data: CourseCreate,
    db: AsyncSession = Depends(get_async_session),
    current_user=Depends(get_current_user)
):
    """Create a new course (teacher only)"""
    try:
        # Check teacher permission
        if not current_user or current_user.role not in [UserRole.TEACHER, UserRole.ADMIN]:
            raise HTTPException(status_code=403, detail="Teacher access required")
        
        # Create course
        course = Course(
            title=course_data.title,
            description=course_data.description,
            subject=course_data.subject,
            grade_level=course_data.grade_level,
            teacher_id=current_user.id,
            difficulty=course_data.difficulty,
            estimated_hours=course_data.estimated_hours,
            max_students=course_data.max_students,
            curriculum_alignment=course_data.curriculum_alignment,
            prerequisites=course_data.prerequisites,
            learning_objectives=course_data.learning_objectives
        )
        
        db.add(course)
        await db.commit()
        await db.refresh(course)
        
        return {
            "success": True,
            "data": {
                "id": course.id,
                "title": course.title,
                "subject": course.subject,
                "grade_level": course.grade_level,
                "created_at": course.created_at.isoformat()
            },
            "message": "Course created successfully"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Course creation error: {e}")
        await db.rollback()
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/courses")
async def list_courses(
    subject: Optional[str] = Query(None),
    grade: Optional[int] = Query(None, ge=1, le=12),
    is_active: bool = Query(True),
    page: int = Query(1, ge=1),
    limit: int = Query(10, ge=1, le=100),
    db: AsyncSession = Depends(get_async_session)
):
    """List available courses with filters"""
    try:
        # Build query
        query = select(Course).where(
            Course.is_active == is_active,
            Course.is_deleted == False
        )
        
        if subject:
            query = query.where(Course.subject == subject)
        if grade:
            query = query.where(Course.grade_level == grade)
        
        # Add pagination
        offset = (page - 1) * limit
        query = query.offset(offset).limit(limit)
        
        result = await db.execute(query)
        courses = result.scalars().all()
        
        # Get total count
        count_query = select(func.count(Course.id)).where(
            Course.is_active == is_active,
            Course.is_deleted == False
        )
        if subject:
            count_query = count_query.where(Course.subject == subject)
        if grade:
            count_query = count_query.where(Course.grade_level == grade)
        
        total_result = await db.execute(count_query)
        total = total_result.scalar()
        
        return {
            "success": True,
            "data": {
                "courses": [
                    {
                        "id": c.id,
                        "title": c.title,
                        "subject": c.subject,
                        "grade_level": c.grade_level,
                        "difficulty": c.difficulty,
                        "enrolled_count": c.enrolled_count,
                        "average_score": c.average_score
                    }
                    for c in courses
                ],
                "pagination": {
                    "page": page,
                    "limit": limit,
                    "total": total,
                    "total_pages": (total + limit - 1) // limit
                }
            }
        }
    except Exception as e:
        logger.error(f"Course listing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/courses/{course_id}/enroll")
async def enroll_in_course(
    course_id: int,
    db: AsyncSession = Depends(get_async_session),
    current_user=Depends(get_current_user)
):
    """Enroll student in course"""
    try:
        if not current_user:
            raise HTTPException(status_code=401, detail="Authentication required")
        
        # Get student profile
        student_result = await db.execute(
            select(Student).where(Student.user_id == current_user.id)
        )
        student = student_result.scalar_one_or_none()
        
        if not student:
            raise HTTPException(status_code=400, detail="Student profile required")
        
        # Get course
        course_result = await db.execute(
            select(Course).where(Course.id == course_id)
        )
        course = course_result.scalar_one_or_none()
        
        if not course:
            raise HTTPException(status_code=404, detail="Course not found")
        
        # Check if already enrolled
        if student in course.enrolled_students:
            raise HTTPException(status_code=400, detail="Already enrolled")
        
        # Check max students
        if course.enrolled_count >= course.max_students:
            raise HTTPException(status_code=400, detail="Course is full")
        
        # Enroll student
        course.enrolled_students.append(student)
        course.enrolled_count += 1
        
        await db.commit()
        
        return {
            "success": True,
            "message": "Enrolled successfully"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Enrollment error: {e}")
        await db.rollback()
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Progress Tracking ====================

@router.get("/progress/summary")
async def get_progress_summary(
    db: AsyncSession = Depends(get_async_session),
    current_user=Depends(get_current_user)
):
    """Get user's progress summary"""
    try:
        if not current_user:
            raise HTTPException(status_code=401, detail="Authentication required")
        
        # Get student profile
        student_result = await db.execute(
            select(Student).where(Student.user_id == current_user.id)
        )
        student = student_result.scalar_one_or_none()
        
        if not student:
            raise HTTPException(status_code=404, detail="Student profile not found")
        
        # Get progress records
        progress_result = await db.execute(
            select(Progress).where(Progress.student_id == student.id)
            .order_by(Progress.last_activity.desc())
            .limit(10)
        )
        progress_records = progress_result.scalars().all()
        
        # Calculate summary
        total_time = sum(p.time_spent for p in progress_records)
        avg_score = sum(p.average_score for p in progress_records) / len(progress_records) if progress_records else 0
        
        return {
            "success": True,
            "data": {
                "student_id": student.id,
                "current_level": student.current_level,
                "target_level": student.target_level,
                "total_study_time": total_time,
                "average_score": avg_score,
                "streak_days": student.streak_days,
                "recent_progress": [
                    {
                        "id": p.id,
                        "course_id": p.course_id,
                        "current_topic": p.current_topic,
                        "average_score": p.average_score,
                        "time_spent": p.time_spent,
                        "last_activity": p.last_activity.isoformat()
                    }
                    for p in progress_records
                ]
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Progress summary error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Activity Logging ====================

@router.post("/activity/log")
async def log_activity(
    action: str,
    entity_type: Optional[str] = None,
    entity_id: Optional[int] = None,
    metadata: Optional[Dict] = None,
    db: AsyncSession = Depends(get_async_session),
    current_user=Depends(get_current_user)
):
    """Log user activity"""
    try:
        if not current_user:
            raise HTTPException(status_code=401, detail="Authentication required")
        
        activity = ActivityLog(
            user_id=current_user.id,
            action=action,
            entity_type=entity_type,
            entity_id=entity_id,
            metadata=metadata or {}
        )
        
        db.add(activity)
        await db.commit()
        
        return {
            "success": True,
            "message": "Activity logged"
        }
    except Exception as e:
        logger.error(f"Activity logging error: {e}")
        await db.rollback()
        raise HTTPException(status_code=500, detail=str(e))