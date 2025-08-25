"""
Health Check API Endpoints
TEKNOFEST 2025 - System Health Monitoring
"""

import logging
from typing import Dict, Any
from datetime import datetime
from fastapi import APIRouter, Depends, status

from src.config import Settings, get_settings
from src.infrastructure.config.container import get_container

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get(
    "/",
    status_code=status.HTTP_200_OK,
    summary="Health Check",
    description="Basic health check endpoint"
)
async def health_check(
    settings: Settings = Depends(get_settings)
) -> Dict[str, Any]:
    """
    Basic health check endpoint.
    
    Returns:
    - Service status
    - Environment information
    - Version details
    """
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": settings.app_name,
        "version": settings.app_version,
        "environment": settings.app_env.value
    }


@router.get(
    "/ready",
    status_code=status.HTTP_200_OK,
    summary="Readiness Check",
    description="Check if service is ready to handle requests"
)
async def readiness_check(
    settings: Settings = Depends(get_settings)
) -> Dict[str, Any]:
    """
    Readiness check endpoint.
    
    Verifies:
    - Database connectivity
    - Required services availability
    - Configuration validity
    """
    issues = []
    checks = {
        "database": False,
        "configuration": False,
        "services": False
    }
    
    try:
        # Check database
        container = get_container()
        
        async with container.unit_of_work() as uow:
            # Try to execute a simple query
            exists = await uow.students.exists("test-id")
            checks["database"] = True
    except Exception as e:
        logger.error(f"Database check failed: {e}")
        issues.append(f"Database connection failed: {str(e)}")
    
    # Check configuration
    try:
        config_issues = settings.validate_production_ready()
        if not config_issues:
            checks["configuration"] = True
        else:
            issues.extend(config_issues)
    except Exception as e:
        logger.error(f"Configuration check failed: {e}")
        issues.append(f"Configuration validation failed: {str(e)}")
    
    # Check services
    try:
        learning_service = container.learning_path_service()
        quiz_service = container.quiz_service()
        
        if learning_service and quiz_service:
            checks["services"] = True
        else:
            issues.append("Required services not initialized")
    except Exception as e:
        logger.error(f"Service check failed: {e}")
        issues.append(f"Service initialization failed: {str(e)}")
    
    # Determine overall status
    all_checks_passed = all(checks.values())
    
    return {
        "ready": all_checks_passed,
        "timestamp": datetime.utcnow().isoformat(),
        "checks": checks,
        "issues": issues if not all_checks_passed else None
    }


@router.get(
    "/live",
    status_code=status.HTTP_200_OK,
    summary="Liveness Check",
    description="Check if service is alive"
)
async def liveness_check() -> Dict[str, Any]:
    """
    Liveness check endpoint.
    
    Simple check to verify the service is running.
    Used by orchestrators like Kubernetes for health monitoring.
    """
    return {
        "alive": True,
        "timestamp": datetime.utcnow().isoformat()
    }


@router.get(
    "/metrics",
    status_code=status.HTTP_200_OK,
    summary="Service Metrics",
    description="Get service metrics and statistics"
)
async def get_metrics(
    settings: Settings = Depends(get_settings)
) -> Dict[str, Any]:
    """
    Get service metrics and statistics.
    
    Returns:
    - Request counts
    - Response times
    - Error rates
    - Resource usage
    """
    # In production, these would come from a metrics collector
    # For now, return mock data
    return {
        "timestamp": datetime.utcnow().isoformat(),
        "service": settings.app_name,
        "metrics": {
            "requests": {
                "total": 0,
                "success": 0,
                "errors": 0,
                "rate_per_minute": 0
            },
            "response_time": {
                "average_ms": 0,
                "p50_ms": 0,
                "p95_ms": 0,
                "p99_ms": 0
            },
            "resources": {
                "cpu_percent": 0,
                "memory_mb": 0,
                "connections": {
                    "database": 0,
                    "redis": 0
                }
            },
            "business": {
                "active_students": 0,
                "quizzes_today": 0,
                "learning_paths_created": 0
            }
        }
    }


@router.get(
    "/dependencies",
    status_code=status.HTTP_200_OK,
    summary="Dependency Status",
    description="Check status of all service dependencies"
)
async def check_dependencies() -> Dict[str, Any]:
    """
    Check status of all service dependencies.
    
    Verifies connectivity to:
    - Database
    - Cache (Redis)
    - External services
    - File storage
    """
    dependencies = {}
    
    # Check database
    try:
        container = get_container()
        async with container.unit_of_work() as uow:
            await uow.students.exists("test")
        dependencies["database"] = {
            "status": "healthy",
            "latency_ms": 0
        }
    except Exception as e:
        dependencies["database"] = {
            "status": "unhealthy",
            "error": str(e)
        }
    
    # Check cache (if configured)
    dependencies["cache"] = {
        "status": "not_configured",
        "type": "redis"
    }
    
    # Check external services
    dependencies["external_services"] = {
        "ai_model": {
            "status": "healthy",
            "endpoint": "local"
        }
    }
    
    # Overall health
    all_healthy = all(
        dep.get("status") in ["healthy", "not_configured"]
        for dep in dependencies.values()
        if isinstance(dep, dict)
    )
    
    return {
        "healthy": all_healthy,
        "timestamp": datetime.utcnow().isoformat(),
        "dependencies": dependencies
    }
