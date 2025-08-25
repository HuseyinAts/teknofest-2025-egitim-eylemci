"""
Model Management API Endpoints
Production-ready API for ML model versioning, deployment, and monitoring
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Query, File, UploadFile, status
from fastapi.responses import JSONResponse, StreamingResponse
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone
from pydantic import BaseModel, Field, validator
import json
import io
import tempfile
from pathlib import Path

from src.ml.model_registry import (
    ModelRegistry, ModelMetadata, ModelStatus,
    ModelFramework, ModelType, get_model_registry
)
from src.ml.model_versioning_service import (
    ModelVersioningService, DeploymentConfig, DeploymentStrategy,
    get_versioning_service
)
from src.api.auth import get_current_user, require_admin
from src.database.models import User
from src.config import get_settings

settings = get_settings()
router = APIRouter(prefix="/api/v1/models", tags=["Model Management"])


# Request/Response Models
class ModelRegistrationRequest(BaseModel):
    """Request model for registering a new model"""
    name: str = Field(..., description="Model name")
    version: str = Field(..., description="Semantic version (e.g., 1.0.0)")
    framework: ModelFramework = Field(..., description="ML framework")
    model_type: ModelType = Field(..., description="Model type")
    description: str = Field("", description="Model description")
    training_dataset: Optional[str] = Field(None, description="Training dataset identifier")
    training_params: Dict[str, Any] = Field(default_factory=dict, description="Training parameters")
    training_metrics: Dict[str, float] = Field(default_factory=dict, description="Training metrics")
    validation_metrics: Dict[str, float] = Field(default_factory=dict, description="Validation metrics")
    test_metrics: Dict[str, float] = Field(default_factory=dict, description="Test metrics")
    tags: List[str] = Field(default_factory=list, description="Model tags")
    labels: Dict[str, str] = Field(default_factory=dict, description="Model labels")
    
    @validator('version')
    def validate_version(cls, v):
        """Validate semantic version format"""
        import semver
        try:
            semver.VersionInfo.parse(v)
        except ValueError:
            raise ValueError(f"Invalid version format: {v}. Use semantic versioning (e.g., 1.0.0)")
        return v


class DeploymentRequest(BaseModel):
    """Request model for deploying a model"""
    version: str = Field(..., description="Model version to deploy")
    strategy: DeploymentStrategy = Field(DeploymentStrategy.IMMEDIATE, description="Deployment strategy")
    canary_percentage: float = Field(10.0, ge=0.0, le=100.0, description="Canary deployment percentage")
    gradual_rollout_minutes: int = Field(60, ge=1, description="Gradual rollout duration in minutes")
    health_check_interval: int = Field(30, ge=10, description="Health check interval in seconds")
    auto_rollback: bool = Field(True, description="Enable automatic rollback on failures")
    rollback_threshold: float = Field(0.1, ge=0.0, le=1.0, description="Error rate threshold for rollback")
    min_requests_for_rollback: int = Field(100, ge=10, description="Minimum requests before rollback decision")
    a_b_test_duration_hours: int = Field(24, ge=1, description="A/B test duration in hours")
    traffic_split: Dict[str, float] = Field(default_factory=dict, description="Traffic split for A/B testing")


class ModelPromotionRequest(BaseModel):
    """Request model for promoting a model"""
    version: str = Field(..., description="Model version to promote")
    target_status: ModelStatus = Field(..., description="Target status")
    reason: str = Field(..., description="Reason for promotion")


class ModelRollbackRequest(BaseModel):
    """Request model for rolling back a model"""
    target_version: str = Field(..., description="Version to rollback to")
    reason: str = Field(..., description="Reason for rollback")


class PredictionRequest(BaseModel):
    """Request model for making predictions"""
    input_data: Any = Field(..., description="Input data for prediction")
    version: Optional[str] = Field(None, description="Specific model version (optional)")


class ModelComparisonRequest(BaseModel):
    """Request model for comparing model versions"""
    version1: str = Field(..., description="First version to compare")
    version2: str = Field(..., description="Second version to compare")
    test_data: Optional[Any] = Field(None, description="Test data for comparison")


class ModelExportRequest(BaseModel):
    """Request model for exporting a model"""
    version: str = Field(..., description="Model version to export")
    format: str = Field("onnx", description="Export format (onnx, tflite, coreml)")


# Initialize services
registry = get_model_registry()
versioning_service = get_versioning_service()


@router.post("/register", status_code=status.HTTP_201_CREATED)
async def register_model(
    request: ModelRegistrationRequest,
    model_file: UploadFile = File(...),
    artifacts: Optional[UploadFile] = File(None),
    current_user: User = Depends(require_admin)
) -> Dict[str, Any]:
    """
    Register a new model version
    
    - Requires admin privileges
    - Uploads model file and optional artifacts
    - Creates metadata and tracks in registry
    """
    try:
        # Read model file
        model_content = await model_file.read()
        
        # Create metadata
        metadata = ModelMetadata(
            name=request.name,
            version=request.version,
            framework=request.framework,
            model_type=request.model_type,
            description=request.description,
            training_dataset=request.training_dataset,
            training_params=request.training_params,
            training_metrics=request.training_metrics,
            validation_metrics=request.validation_metrics,
            test_metrics=request.test_metrics,
            tags=request.tags,
            labels=request.labels,
            deployed_by=current_user.username
        )
        
        # Process artifacts if provided
        artifacts_dict = None
        if artifacts:
            artifacts_content = await artifacts.read()
            with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as tmp:
                tmp.write(artifacts_content)
                tmp_path = tmp.name
            
            # Extract artifacts (implementation would handle zip extraction)
            artifacts_dict = {"artifacts_path": tmp_path}
        
        # Register model
        # Note: In production, you'd deserialize the model based on framework
        model_id = registry.register_model(
            model=model_content,  # This would be deserialized
            metadata=metadata,
            artifacts=artifacts_dict
        )
        
        return {
            "model_id": model_id,
            "version": request.version,
            "status": "registered",
            "message": f"Model {request.name} version {request.version} registered successfully"
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to register model: {str(e)}"
        )


@router.post("/{model_id}/deploy")
async def deploy_model(
    model_id: str,
    request: DeploymentRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(require_admin)
) -> Dict[str, Any]:
    """
    Deploy a model version
    
    - Requires admin privileges
    - Supports multiple deployment strategies
    - Starts health monitoring
    """
    try:
        # Create deployment config
        config = DeploymentConfig(
            strategy=request.strategy,
            canary_percentage=request.canary_percentage,
            gradual_rollout_minutes=request.gradual_rollout_minutes,
            health_check_interval=request.health_check_interval,
            auto_rollback=request.auto_rollback,
            rollback_threshold=request.rollback_threshold,
            min_requests_for_rollback=request.min_requests_for_rollback,
            a_b_test_duration_hours=request.a_b_test_duration_hours,
            traffic_split=request.traffic_split
        )
        
        # Deploy model
        result = await versioning_service.deploy_model(
            model_id=model_id,
            version=request.version,
            config=config
        )
        
        # Add background monitoring
        if request.auto_rollback:
            background_tasks.add_task(
                versioning_service.rollback_if_needed,
                model_id,
                request.version
            )
        
        return {
            **result,
            "deployed_by": current_user.username,
            "deployed_at": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to deploy model: {str(e)}"
        )


@router.post("/{model_id}/predict")
async def predict(
    model_id: str,
    request: PredictionRequest,
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Make prediction using deployed model
    
    - Automatically selects appropriate version
    - Tracks performance metrics
    """
    try:
        prediction, version_used = await versioning_service.predict(
            model_id=model_id,
            input_data=request.input_data,
            version=request.version
        )
        
        return {
            "model_id": model_id,
            "version_used": version_used,
            "prediction": prediction,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@router.post("/{model_id}/promote")
async def promote_model(
    model_id: str,
    request: ModelPromotionRequest,
    current_user: User = Depends(require_admin)
) -> Dict[str, Any]:
    """
    Promote model to different status
    
    - Requires admin privileges
    - Validates promotion path
    - Handles production deployment
    """
    try:
        success = registry.promote_model(
            model_id=model_id,
            version=request.version,
            target_status=request.target_status,
            promoted_by=current_user.username
        )
        
        if success:
            return {
                "model_id": model_id,
                "version": request.version,
                "new_status": request.target_status.value,
                "promoted_by": current_user.username,
                "promoted_at": datetime.now(timezone.utc).isoformat(),
                "reason": request.reason
            }
        else:
            raise ValueError("Promotion failed")
            
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to promote model: {str(e)}"
        )


@router.post("/{model_id}/rollback")
async def rollback_model(
    model_id: str,
    request: ModelRollbackRequest,
    current_user: User = Depends(require_admin)
) -> Dict[str, Any]:
    """
    Rollback to previous model version
    
    - Requires admin privileges
    - Archives current production version
    - Deploys target version
    """
    try:
        success = registry.rollback_model(
            model_id=model_id,
            target_version=request.target_version,
            reason=request.reason,
            rolled_back_by=current_user.username
        )
        
        if success:
            return {
                "model_id": model_id,
                "rolled_back_to": request.target_version,
                "rolled_back_by": current_user.username,
                "rolled_back_at": datetime.now(timezone.utc).isoformat(),
                "reason": request.reason
            }
        else:
            raise ValueError("Rollback failed")
            
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to rollback model: {str(e)}"
        )


@router.get("/{model_id}/versions")
async def list_versions(
    model_id: str,
    status: Optional[ModelStatus] = Query(None, description="Filter by status"),
    limit: int = Query(20, ge=1, le=100, description="Maximum number of versions"),
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    List all versions of a model
    
    - Returns complete lineage
    - Optional status filtering
    """
    try:
        lineage = registry.get_model_lineage(model_id)
        
        # Filter by status if provided
        if status:
            lineage = [v for v in lineage if v['status'] == status.value]
        
        # Apply limit
        lineage = lineage[:limit]
        
        return {
            "model_id": model_id,
            "versions": lineage,
            "total_count": len(lineage)
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model not found: {str(e)}"
        )


@router.get("/{model_id}/metrics")
async def get_metrics(
    model_id: str,
    version: Optional[str] = Query(None, description="Specific version"),
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Get performance metrics for model
    
    - Real-time performance data
    - Success rates and latencies
    """
    try:
        metrics = await versioning_service.get_model_metrics(model_id, version)
        
        return {
            "model_id": model_id,
            "version": version,
            "metrics": metrics,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Metrics not found: {str(e)}"
        )


@router.post("/{model_id}/compare")
async def compare_versions(
    model_id: str,
    request: ModelComparisonRequest,
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Compare two model versions
    
    - Performance comparison
    - Metrics comparison
    - Optional prediction comparison
    """
    try:
        if request.test_data:
            # Compare with test data
            comparison = await versioning_service.compare_versions(
                model_id=model_id,
                version1=request.version1,
                version2=request.version2,
                test_data=request.test_data
            )
        else:
            # Compare metadata only
            comparison = registry.compare_models(
                model_id=model_id,
                version1=request.version1,
                version2=request.version2
            )
        
        return {
            "model_id": model_id,
            "comparison": comparison,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Comparison failed: {str(e)}"
        )


@router.post("/{model_id}/export")
async def export_model(
    model_id: str,
    request: ModelExportRequest,
    current_user: User = Depends(get_current_user)
) -> StreamingResponse:
    """
    Export model in different formats
    
    - ONNX, TFLite, CoreML formats
    - Returns downloadable file
    """
    try:
        export_path = registry.export_model(
            model_id=model_id,
            version=request.version,
            export_format=request.format
        )
        
        # Read exported file
        with open(export_path, 'rb') as f:
            content = f.read()
        
        # Return as streaming response
        return StreamingResponse(
            io.BytesIO(content),
            media_type="application/octet-stream",
            headers={
                "Content-Disposition": f"attachment; filename={model_id}_{request.version}.{request.format}"
            }
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Export failed: {str(e)}"
        )


@router.delete("/{model_id}/cleanup")
async def cleanup_versions(
    model_id: str,
    keep_count: int = Query(5, ge=1, description="Number of recent versions to keep"),
    keep_days: int = Query(30, ge=1, description="Keep versions newer than this many days"),
    current_user: User = Depends(require_admin)
) -> Dict[str, Any]:
    """
    Clean up old model versions
    
    - Requires admin privileges
    - Keeps recent and production versions
    - Archives old versions
    """
    try:
        cleaned = await versioning_service.cleanup_old_versions(
            model_id=model_id,
            keep_count=keep_count,
            keep_days=keep_days
        )
        
        return {
            "model_id": model_id,
            "cleaned_versions": cleaned,
            "cleaned_count": len(cleaned),
            "cleaned_by": current_user.username,
            "cleaned_at": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cleanup failed: {str(e)}"
        )


@router.get("/health")
async def health_check() -> Dict[str, Any]:
    """
    Health check endpoint for model management service
    """
    try:
        # Check registry health
        registry_healthy = registry is not None
        
        # Check versioning service health
        service_healthy = versioning_service is not None
        
        # Check Redis connection
        redis_healthy = False
        if versioning_service.redis_client:
            try:
                versioning_service.redis_client.ping()
                redis_healthy = True
            except:
                pass
        
        healthy = registry_healthy and service_healthy
        
        return {
            "status": "healthy" if healthy else "degraded",
            "components": {
                "registry": "healthy" if registry_healthy else "unhealthy",
                "versioning_service": "healthy" if service_healthy else "unhealthy",
                "redis": "healthy" if redis_healthy else "unhealthy"
            },
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Health check failed: {str(e)}"
        )


@router.get("/dashboard")
async def model_dashboard(
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Get dashboard data for model management
    
    - Active models
    - Deployment status
    - Performance summary
    """
    try:
        # Get active models
        active_models = []
        for key, info in versioning_service.active_models.items():
            active_models.append({
                "key": key,
                "version": info.get('version'),
                "deployed_at": info.get('deployed_at').isoformat() if info.get('deployed_at') else None,
                "traffic_percentage": info.get('traffic_percentage', 100)
            })
        
        # Get metrics summary
        metrics_summary = {}
        for key, metrics in versioning_service.model_metrics.items():
            metrics_summary[key] = {
                "request_count": metrics.request_count,
                "success_rate": metrics.success_rate,
                "avg_latency": metrics.avg_latency
            }
        
        return {
            "active_models": active_models,
            "metrics_summary": metrics_summary,
            "total_active": len(active_models),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Dashboard data failed: {str(e)}"
        )