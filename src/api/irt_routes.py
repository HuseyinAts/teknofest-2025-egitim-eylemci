"""
IRT API Routes for Production
TEKNOFEST 2025 - Eğitim Teknolojileri

RESTful API endpoints for Item Response Theory operations.
"""

from typing import Dict, List, Optional
from datetime import datetime
import logging

from fastapi import APIRouter, Depends, HTTPException, Query, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.irt_service import (
    IRTService, IRTItemRequest, IRTEstimationRequest,
    AdaptiveTestRequest
)
from src.core.irt_engine import EstimationMethod, IRTModel
from src.database.session import get_db
from src.core.cache_manager import CacheManager
from src.monitoring import metrics_collector
from src.error_handlers import handle_api_errors
from src.resilience import circuit_breaker, retry_with_backoff

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/v1/irt", tags=["IRT"])

# Service instances (will be dependency injected)
_irt_service: Optional[IRTService] = None
_cache_manager: Optional[CacheManager] = None


def get_irt_service(db: AsyncSession = Depends(get_db)) -> IRTService:
    """Dependency injection for IRT service"""
    global _irt_service, _cache_manager
    
    if _irt_service is None:
        _cache_manager = CacheManager()
        _irt_service = IRTService(db, _cache_manager)
    
    return _irt_service


# Request/Response Models

class IRTHealthResponse(BaseModel):
    """Health check response"""
    status: str
    items_loaded: int
    active_sessions: int
    cache_status: str
    timestamp: str


class AbilityEstimateResponse(BaseModel):
    """Ability estimation response"""
    student_id: str
    theta: float = Field(description="Ability estimate")
    standard_error: float
    confidence_interval: List[float]
    reliability: float
    items_count: int
    estimation_method: str
    timestamp: str


class AdaptiveTestResponse(BaseModel):
    """Adaptive test session response"""
    session_id: str
    status: str
    current_item: Optional[Dict] = None
    progress: Optional[float] = None
    current_theta: Optional[float] = None
    current_se: Optional[float] = None
    final_results: Optional[Dict] = None


class CalibrationResponse(BaseModel):
    """Item calibration response"""
    subject: str
    items_calibrated: int
    calibration_method: str
    duration_seconds: float
    timestamp: str


class TestInformationResponse(BaseModel):
    """Test information curve response"""
    theta_values: List[float]
    information_values: List[float]
    standard_errors: List[float]
    max_information: float
    optimal_theta: float
    reliability: float


# API Endpoints

@router.get("/health", response_model=IRTHealthResponse)
@handle_api_errors
async def health_check(
    service: IRTService = Depends(get_irt_service)
) -> IRTHealthResponse:
    """
    Health check for IRT service.
    
    Returns service status and key metrics.
    """
    stats = service.get_performance_stats()
    
    return IRTHealthResponse(
        status="healthy",
        items_loaded=len(service.engine.item_bank),
        active_sessions=stats.get("active_sessions", 0),
        cache_status="connected" if service.cache else "disabled",
        timestamp=datetime.now().isoformat()
    )


@router.post("/items", response_model=Dict)
@handle_api_errors
@retry_with_backoff(max_retries=3)
async def add_or_update_item(
    item_request: IRTItemRequest,
    service: IRTService = Depends(get_irt_service)
) -> Dict:
    """
    Add or update IRT parameters for an item.
    
    Args:
        item_request: Item parameters including difficulty, discrimination, guessing
    
    Returns:
        Created/updated item information
    """
    try:
        item = await service.add_or_update_item(item_request)
        
        metrics_collector.increment("irt.items.updated")
        
        return {
            "item_id": item.item_id,
            "difficulty": item.difficulty,
            "discrimination": item.discrimination,
            "guessing": item.guessing,
            "subject": item.subject,
            "topic": item.topic,
            "message": "Item parameters updated successfully"
        }
    
    except Exception as e:
        logger.error(f"Failed to add/update item: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/estimate", response_model=AbilityEstimateResponse)
@handle_api_errors
@circuit_breaker(failure_threshold=5, recovery_timeout=30)
async def estimate_ability(
    request: IRTEstimationRequest,
    save_to_db: bool = Query(default=True),
    service: IRTService = Depends(get_irt_service)
) -> AbilityEstimateResponse:
    """
    Estimate student ability from item responses.
    
    Args:
        request: Student responses and item IDs
        save_to_db: Whether to persist the estimate
    
    Returns:
        Ability estimate with confidence interval
    """
    try:
        ability = await service.estimate_ability(request, save_to_db)
        
        metrics_collector.histogram(
            "irt.ability.estimate",
            ability.theta,
            tags={"method": request.estimation_method.value}
        )
        
        return AbilityEstimateResponse(
            student_id=ability.student_id,
            theta=round(ability.theta, 3),
            standard_error=round(ability.standard_error, 3),
            confidence_interval=[
                round(ability.confidence_interval[0], 3),
                round(ability.confidence_interval[1], 3)
            ],
            reliability=round(ability.reliability, 3),
            items_count=len(ability.items_administered),
            estimation_method=ability.estimation_method.value,
            timestamp=ability.timestamp.isoformat()
        )
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Ability estimation failed: {e}")
        raise HTTPException(status_code=500, detail="Estimation failed")


@router.post("/adaptive/start", response_model=AdaptiveTestResponse)
@handle_api_errors
async def start_adaptive_test(
    request: AdaptiveTestRequest,
    service: IRTService = Depends(get_irt_service)
) -> AdaptiveTestResponse:
    """
    Start a new adaptive test session.
    
    Args:
        request: Test configuration including subject, limits
    
    Returns:
        Session ID and first item
    """
    try:
        session_data = await service.start_adaptive_test(request)
        
        metrics_collector.increment(
            "irt.adaptive.sessions.started",
            tags={"subject": request.subject}
        )
        
        return AdaptiveTestResponse(
            session_id=session_data["session_id"],
            status="started",
            current_item=session_data["first_item"],
            progress=0.0
        )
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to start adaptive test: {e}")
        raise HTTPException(status_code=500, detail="Failed to start test")


@router.post("/adaptive/{session_id}/respond", response_model=AdaptiveTestResponse)
@handle_api_errors
async def submit_adaptive_response(
    session_id: str,
    response: int = Query(..., ge=0, le=1),
    service: IRTService = Depends(get_irt_service)
) -> AdaptiveTestResponse:
    """
    Submit response for current item in adaptive test.
    
    Args:
        session_id: Active test session ID
        response: Student response (0 or 1)
    
    Returns:
        Next item or final results
    """
    try:
        result = await service.submit_adaptive_response(session_id, response)
        
        if result["status"] == "completed":
            metrics_collector.increment(
                "irt.adaptive.sessions.completed"
            )
            
            return AdaptiveTestResponse(
                session_id=session_id,
                status="completed",
                final_results={
                    "theta": round(result["final_theta"], 3),
                    "standard_error": round(result["final_se"], 3),
                    "confidence_interval": [
                        round(result["confidence_interval"][0], 3),
                        round(result["confidence_interval"][1], 3)
                    ],
                    "reliability": round(result["reliability"], 3),
                    "items_administered": result["items_administered"],
                    "duration_minutes": round(result["duration_minutes"], 1)
                }
            )
        else:
            return AdaptiveTestResponse(
                session_id=session_id,
                status="in_progress",
                current_item=result.get("next_item"),
                progress=result.get("progress"),
                current_theta=round(result.get("current_theta", 0), 3),
                current_se=round(result.get("current_se", 1), 3)
            )
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to process response: {e}")
        raise HTTPException(status_code=500, detail="Failed to process response")


@router.get("/students/{student_id}/history", response_model=List[Dict])
@handle_api_errors
async def get_ability_history(
    student_id: str,
    subject: Optional[str] = Query(None),
    limit: int = Query(default=10, ge=1, le=100),
    service: IRTService = Depends(get_irt_service)
) -> List[Dict]:
    """
    Get historical ability estimates for a student.
    
    Args:
        student_id: Student identifier
        subject: Filter by subject (optional)
        limit: Maximum number of records
    
    Returns:
        List of historical ability estimates
    """
    try:
        history = await service.get_student_ability_history(
            student_id, subject, limit
        )
        
        return [
            {
                "timestamp": record["timestamp"],
                "theta": round(record["theta"], 3),
                "standard_error": round(record["standard_error"], 3),
                "subject": record["subject"],
                "reliability": round(record["reliability"], 3) if record["reliability"] else None,
                "items_count": record["items_count"]
            }
            for record in history
        ]
    
    except Exception as e:
        logger.error(f"Failed to get ability history: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve history")


@router.post("/calibrate", response_model=CalibrationResponse)
@handle_api_errors
async def calibrate_items(
    subject: str = Query(...),
    min_responses: int = Query(default=30, ge=10, le=1000),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    service: IRTService = Depends(get_irt_service)
) -> CalibrationResponse:
    """
    Calibrate item parameters from response data.
    
    Args:
        subject: Subject to calibrate
        min_responses: Minimum responses per item
    
    Returns:
        Calibration summary
    
    Note: This runs as a background task for large datasets.
    """
    try:
        start_time = datetime.now()
        
        # For large calibrations, run in background
        if min_responses > 100:
            background_tasks.add_task(
                service.calibrate_items_from_responses,
                subject,
                min_responses
            )
            
            return CalibrationResponse(
                subject=subject,
                items_calibrated=0,
                calibration_method="background_processing",
                duration_seconds=0,
                timestamp=start_time.isoformat()
            )
        else:
            # Run synchronously for small datasets
            items = await service.calibrate_items_from_responses(
                subject,
                min_responses
            )
            
            duration = (datetime.now() - start_time).total_seconds()
            
            metrics_collector.histogram(
                "irt.calibration.duration",
                duration,
                tags={"subject": subject}
            )
            
            return CalibrationResponse(
                subject=subject,
                items_calibrated=len(items),
                calibration_method="marginal_maximum_likelihood",
                duration_seconds=round(duration, 2),
                timestamp=start_time.isoformat()
            )
    
    except Exception as e:
        logger.error(f"Calibration failed: {e}")
        raise HTTPException(status_code=500, detail="Calibration failed")


@router.post("/test-information", response_model=TestInformationResponse)
@handle_api_errors
async def calculate_test_information(
    item_ids: List[str],
    theta_min: float = Query(default=-4, ge=-5, le=0),
    theta_max: float = Query(default=4, ge=0, le=5),
    points: int = Query(default=100, ge=10, le=500),
    service: IRTService = Depends(get_irt_service)
) -> TestInformationResponse:
    """
    Calculate test information curve for a set of items.
    
    Args:
        item_ids: List of item identifiers
        theta_min: Minimum ability level
        theta_max: Maximum ability level
        points: Number of points to calculate
    
    Returns:
        Test information and standard error curves
    """
    try:
        if not item_ids:
            raise ValueError("At least one item ID required")
        
        info_data = await service.get_test_information_curve(
            item_ids,
            (theta_min, theta_max),
            points
        )
        
        return TestInformationResponse(
            theta_values=info_data["theta"],
            information_values=info_data["information"],
            standard_errors=info_data["standard_error"],
            max_information=round(info_data["max_information"], 3),
            optimal_theta=round(info_data["max_info_theta"], 3),
            reliability=round(info_data["reliability"], 3)
        )
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to calculate test information: {e}")
        raise HTTPException(status_code=500, detail="Calculation failed")


@router.get("/items/{item_id}", response_model=Dict)
@handle_api_errors
async def get_item_parameters(
    item_id: str,
    service: IRTService = Depends(get_irt_service)
) -> Dict:
    """
    Get IRT parameters for a specific item.
    
    Args:
        item_id: Item identifier
    
    Returns:
        Item parameters and statistics
    """
    try:
        if item_id not in service.engine.item_bank:
            raise HTTPException(status_code=404, detail="Item not found")
        
        item = service.engine.item_bank[item_id]
        
        return {
            "item_id": item.item_id,
            "difficulty": round(item.difficulty, 3),
            "discrimination": round(item.discrimination, 3),
            "guessing": round(item.guessing, 3),
            "upper_asymptote": round(item.upper_asymptote, 3),
            "subject": item.subject,
            "topic": item.topic,
            "grade_level": item.grade_level,
            "usage_count": item.usage_count,
            "exposure_rate": round(item.exposure_rate, 3) if item.exposure_rate else 0
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get item parameters: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve item")


@router.get("/statistics", response_model=Dict)
@handle_api_errors
async def get_irt_statistics(
    service: IRTService = Depends(get_irt_service)
) -> Dict:
    """
    Get overall IRT system statistics.
    
    Returns:
        System performance metrics and statistics
    """
    try:
        stats = service.get_performance_stats()
        
        return {
            "total_items": len(service.engine.item_bank),
            "estimations_performed": stats.get("estimations", 0),
            "active_sessions": stats.get("active_sessions", 0),
            "cache_hits": stats.get("cache_hits", 0),
            "avg_estimation_time_seconds": round(stats.get("avg_estimation_time", 0), 3),
            "items_calibrated": stats.get("items_calibrated", 0),
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Failed to get statistics: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve statistics")


# Initialize service on module load
@router.on_event("startup")
async def startup_event():
    """Initialize IRT service on startup"""
    try:
        logger.info("Initializing IRT service...")
        # Service will be initialized on first request
        logger.info("IRT service ready")
    except Exception as e:
        logger.error(f"Failed to initialize IRT service: {e}")
        raise