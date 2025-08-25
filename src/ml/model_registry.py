"""
Production-Ready ML Model Registry and Versioning System
Handles model storage, versioning, metadata tracking, and deployment
"""

import os
import json
import hashlib
import shutil
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Union
from enum import Enum
import pickle
import joblib
import uuid
from dataclasses import dataclass, asdict, field
import semver
import yaml
import numpy as np
from sqlalchemy import Column, String, Float, DateTime, Boolean, JSON, Integer, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Session
from sqlalchemy.sql import func
import boto3
from botocore.exceptions import ClientError
import mlflow
from mlflow.tracking import MlflowClient
import torch
import tensorflow as tf
from huggingface_hub import HfApi, ModelCard, ModelCardData, Repository
import warnings
from contextlib import contextmanager
import tempfile
import zipfile
import logging

from src.database.base import Base
from src.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class ModelStatus(str, Enum):
    """Model lifecycle status"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    ARCHIVED = "archived"
    DEPRECATED = "deprecated"
    FAILED = "failed"


class ModelFramework(str, Enum):
    """Supported ML frameworks"""
    PYTORCH = "pytorch"
    TENSORFLOW = "tensorflow"
    SKLEARN = "sklearn"
    XGBOOST = "xgboost"
    HUGGINGFACE = "huggingface"
    CUSTOM = "custom"


class ModelType(str, Enum):
    """Model types for education platform"""
    CONTENT_GENERATION = "content_generation"
    QUESTION_ANSWERING = "question_answering"
    ASSESSMENT = "assessment"
    RECOMMENDATION = "recommendation"
    NLP = "nlp"
    CLASSIFICATION = "classification"
    REGRESSION = "regression"


@dataclass
class ModelMetadata:
    """Comprehensive model metadata"""
    model_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    version: str = "0.1.0"
    framework: ModelFramework = ModelFramework.CUSTOM
    model_type: ModelType = ModelType.NLP
    status: ModelStatus = ModelStatus.DEVELOPMENT
    
    # Training metadata
    training_dataset: Optional[str] = None
    training_params: Dict[str, Any] = field(default_factory=dict)
    training_metrics: Dict[str, float] = field(default_factory=dict)
    validation_metrics: Dict[str, float] = field(default_factory=dict)
    test_metrics: Dict[str, float] = field(default_factory=dict)
    
    # Model information
    model_size_mb: float = 0.0
    parameters_count: int = 0
    input_shape: Optional[List[int]] = None
    output_shape: Optional[List[int]] = None
    
    # Deployment info
    deployed_at: Optional[datetime] = None
    deployed_by: Optional[str] = None
    deployment_endpoint: Optional[str] = None
    serving_container: Optional[str] = None
    
    # Performance tracking
    inference_time_ms: Optional[float] = None
    throughput_rps: Optional[float] = None
    memory_usage_mb: Optional[float] = None
    
    # Versioning
    parent_version: Optional[str] = None
    git_commit: Optional[str] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    # Tags and labels
    tags: List[str] = field(default_factory=list)
    labels: Dict[str, str] = field(default_factory=dict)
    description: str = ""
    
    # Data quality
    data_drift_score: Optional[float] = None
    model_drift_score: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = asdict(self)
        # Convert datetime objects to ISO format
        for key in ['created_at', 'updated_at', 'deployed_at']:
            if data.get(key) and isinstance(data[key], datetime):
                data[key] = data[key].isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelMetadata':
        """Create from dictionary"""
        # Convert ISO strings back to datetime
        for key in ['created_at', 'updated_at', 'deployed_at']:
            if data.get(key) and isinstance(data[key], str):
                data[key] = datetime.fromisoformat(data[key])
        
        # Convert string enums
        if 'framework' in data and isinstance(data['framework'], str):
            data['framework'] = ModelFramework(data['framework'])
        if 'model_type' in data and isinstance(data['model_type'], str):
            data['model_type'] = ModelType(data['model_type'])
        if 'status' in data and isinstance(data['status'], str):
            data['status'] = ModelStatus(data['status'])
            
        return cls(**data)


class ModelVersion(Base):
    """Database model for tracking model versions"""
    __tablename__ = 'model_versions'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    model_id = Column(String(255), nullable=False, index=True)
    name = Column(String(255), nullable=False)
    version = Column(String(50), nullable=False)
    framework = Column(String(50), nullable=False)
    model_type = Column(String(50), nullable=False)
    status = Column(String(50), nullable=False, default=ModelStatus.DEVELOPMENT.value)
    
    # Storage locations
    local_path = Column(String(500), nullable=True)
    s3_path = Column(String(500), nullable=True)
    mlflow_run_id = Column(String(255), nullable=True)
    huggingface_repo = Column(String(255), nullable=True)
    
    # Metadata
    metadata = Column(JSON, nullable=False)
    metrics = Column(JSON, nullable=True)
    parameters = Column(JSON, nullable=True)
    
    # Model characteristics
    model_size_mb = Column(Float, nullable=True)
    parameters_count = Column(Integer, nullable=True)
    
    # Performance metrics
    inference_time_ms = Column(Float, nullable=True)
    throughput_rps = Column(Float, nullable=True)
    memory_usage_mb = Column(Float, nullable=True)
    
    # Deployment
    is_deployed = Column(Boolean, default=False)
    deployed_at = Column(DateTime(timezone=True), nullable=True)
    deployment_endpoint = Column(String(500), nullable=True)
    
    # Tracking
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    created_by = Column(String(255), nullable=True)
    
    # Checksums for integrity
    model_hash = Column(String(64), nullable=True)
    
    class Meta:
        indexes = [
            ('model_id', 'version'),
            ('status',),
            ('created_at',),
        ]
        unique_together = [('model_id', 'version')]


class ModelRegistry:
    """Central model registry for managing ML models"""
    
    def __init__(self, 
                 storage_path: Path = Path("./models"),
                 s3_bucket: Optional[str] = None,
                 mlflow_tracking_uri: Optional[str] = None):
        """Initialize model registry"""
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Storage backends
        self.s3_bucket = s3_bucket
        self.s3_client = None
        if s3_bucket:
            self._init_s3()
        
        # MLflow integration
        self.mlflow_tracking_uri = mlflow_tracking_uri
        if mlflow_tracking_uri:
            mlflow.set_tracking_uri(mlflow_tracking_uri)
            self.mlflow_client = MlflowClient()
        else:
            self.mlflow_client = None
        
        # HuggingFace integration
        self.hf_api = HfApi()
        self.hf_token = settings.hugging_face_hub_token.get_secret_value() if settings.hugging_face_hub_token else None
        
        # Model cache
        self._model_cache = {}
        self._metadata_cache = {}
        
        logger.info(f"Model Registry initialized with storage at {self.storage_path}")
    
    def _init_s3(self):
        """Initialize S3 client"""
        try:
            self.s3_client = boto3.client('s3')
            # Test connection
            self.s3_client.head_bucket(Bucket=self.s3_bucket)
            logger.info(f"S3 bucket {self.s3_bucket} connected successfully")
        except ClientError as e:
            logger.warning(f"S3 initialization failed: {e}")
            self.s3_client = None
    
    def register_model(self,
                       model: Any,
                       metadata: ModelMetadata,
                       artifacts: Optional[Dict[str, Any]] = None) -> str:
        """Register a new model version"""
        try:
            # Validate version
            if not self._validate_version(metadata.version):
                raise ValueError(f"Invalid version format: {metadata.version}")
            
            # Check for existing version
            if self._version_exists(metadata.model_id, metadata.version):
                raise ValueError(f"Version {metadata.version} already exists for model {metadata.model_id}")
            
            # Calculate model size and hash
            model_bytes = self._serialize_model(model, metadata.framework)
            metadata.model_size_mb = len(model_bytes) / (1024 * 1024)
            model_hash = hashlib.sha256(model_bytes).hexdigest()
            
            # Count parameters if possible
            metadata.parameters_count = self._count_parameters(model, metadata.framework)
            
            # Create storage paths
            model_dir = self.storage_path / metadata.model_id / metadata.version
            model_dir.mkdir(parents=True, exist_ok=True)
            
            # Save model locally
            model_path = model_dir / "model.pkl"
            with open(model_path, 'wb') as f:
                f.write(model_bytes)
            
            # Save metadata
            metadata_path = model_dir / "metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata.to_dict(), f, indent=2)
            
            # Save artifacts
            if artifacts:
                artifacts_dir = model_dir / "artifacts"
                artifacts_dir.mkdir(exist_ok=True)
                for name, artifact in artifacts.items():
                    artifact_path = artifacts_dir / name
                    if isinstance(artifact, (dict, list)):
                        with open(f"{artifact_path}.json", 'w') as f:
                            json.dump(artifact, f, indent=2)
                    else:
                        joblib.dump(artifact, f"{artifact_path}.pkl")
            
            # Upload to S3 if configured
            s3_path = None
            if self.s3_client and self.s3_bucket:
                s3_path = self._upload_to_s3(model_dir, metadata.model_id, metadata.version)
            
            # Track with MLflow if configured
            mlflow_run_id = None
            if self.mlflow_client:
                mlflow_run_id = self._track_with_mlflow(model, metadata, artifacts)
            
            # Create database entry
            self._save_to_database(
                metadata=metadata,
                local_path=str(model_path),
                s3_path=s3_path,
                mlflow_run_id=mlflow_run_id,
                model_hash=model_hash
            )
            
            logger.info(f"Model {metadata.model_id} version {metadata.version} registered successfully")
            return metadata.model_id
            
        except Exception as e:
            logger.error(f"Failed to register model: {e}")
            raise
    
    def load_model(self,
                   model_id: str,
                   version: Optional[str] = None,
                   status: Optional[ModelStatus] = None) -> tuple[Any, ModelMetadata]:
        """Load a model and its metadata"""
        try:
            # Check cache first
            cache_key = f"{model_id}:{version or 'latest'}"
            if cache_key in self._model_cache:
                return self._model_cache[cache_key], self._metadata_cache[cache_key]
            
            # Get version info from database
            model_info = self._get_model_info(model_id, version, status)
            if not model_info:
                raise ValueError(f"Model {model_id} version {version} not found")
            
            # Load metadata
            metadata = ModelMetadata.from_dict(model_info['metadata'])
            
            # Try loading from different sources
            model = None
            
            # 1. Try local storage
            if model_info.get('local_path') and os.path.exists(model_info['local_path']):
                with open(model_info['local_path'], 'rb') as f:
                    model = self._deserialize_model(f.read(), metadata.framework)
            
            # 2. Try S3
            elif model_info.get('s3_path') and self.s3_client:
                model = self._load_from_s3(model_info['s3_path'], metadata.framework)
            
            # 3. Try MLflow
            elif model_info.get('mlflow_run_id') and self.mlflow_client:
                model = self._load_from_mlflow(model_info['mlflow_run_id'], metadata.framework)
            
            if model is None:
                raise ValueError(f"Could not load model {model_id} version {version}")
            
            # Cache the model
            self._model_cache[cache_key] = model
            self._metadata_cache[cache_key] = metadata
            
            return model, metadata
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def promote_model(self,
                     model_id: str,
                     version: str,
                     target_status: ModelStatus,
                     promoted_by: str) -> bool:
        """Promote a model to a different status"""
        try:
            # Validate promotion path
            current_info = self._get_model_info(model_id, version)
            if not current_info:
                raise ValueError(f"Model {model_id} version {version} not found")
            
            current_status = ModelStatus(current_info['status'])
            if not self._validate_promotion(current_status, target_status):
                raise ValueError(f"Invalid promotion from {current_status} to {target_status}")
            
            # If promoting to production, demote current production model
            if target_status == ModelStatus.PRODUCTION:
                self._demote_production_models(model_id)
            
            # Update model status
            self._update_model_status(model_id, version, target_status, promoted_by)
            
            # Deploy if moving to production
            if target_status == ModelStatus.PRODUCTION:
                self._deploy_model(model_id, version)
            
            logger.info(f"Model {model_id} version {version} promoted to {target_status}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to promote model: {e}")
            raise
    
    def rollback_model(self,
                      model_id: str,
                      target_version: str,
                      reason: str,
                      rolled_back_by: str) -> bool:
        """Rollback to a previous model version"""
        try:
            # Get current production version
            current_prod = self._get_production_model(model_id)
            if not current_prod:
                raise ValueError(f"No production model found for {model_id}")
            
            # Validate target version exists
            target_info = self._get_model_info(model_id, target_version)
            if not target_info:
                raise ValueError(f"Target version {target_version} not found")
            
            # Archive current production
            self._update_model_status(
                model_id,
                current_prod['version'],
                ModelStatus.ARCHIVED,
                rolled_back_by
            )
            
            # Promote target version to production
            self._update_model_status(
                model_id,
                target_version,
                ModelStatus.PRODUCTION,
                rolled_back_by
            )
            
            # Deploy target version
            self._deploy_model(model_id, target_version)
            
            # Log rollback
            self._log_rollback(model_id, current_prod['version'], target_version, reason, rolled_back_by)
            
            logger.info(f"Model {model_id} rolled back from {current_prod['version']} to {target_version}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to rollback model: {e}")
            raise
    
    def compare_models(self,
                      model_id: str,
                      version1: str,
                      version2: str) -> Dict[str, Any]:
        """Compare two model versions"""
        try:
            # Load both models' metadata
            _, metadata1 = self.load_model(model_id, version1)
            _, metadata2 = self.load_model(model_id, version2)
            
            comparison = {
                'version1': version1,
                'version2': version2,
                'metrics_comparison': self._compare_metrics(metadata1, metadata2),
                'performance_comparison': {
                    'inference_time_diff': (metadata2.inference_time_ms or 0) - (metadata1.inference_time_ms or 0),
                    'throughput_diff': (metadata2.throughput_rps or 0) - (metadata1.throughput_rps or 0),
                    'memory_diff': (metadata2.memory_usage_mb or 0) - (metadata1.memory_usage_mb or 0),
                },
                'size_comparison': {
                    'model_size_diff': metadata2.model_size_mb - metadata1.model_size_mb,
                    'parameters_diff': metadata2.parameters_count - metadata1.parameters_count,
                },
                'status': {
                    'version1_status': metadata1.status.value,
                    'version2_status': metadata2.status.value,
                }
            }
            
            return comparison
            
        except Exception as e:
            logger.error(f"Failed to compare models: {e}")
            raise
    
    def get_model_lineage(self, model_id: str) -> List[Dict[str, Any]]:
        """Get complete lineage of a model"""
        try:
            lineage = []
            versions = self._get_all_versions(model_id)
            
            for version_info in versions:
                metadata = ModelMetadata.from_dict(version_info['metadata'])
                lineage.append({
                    'version': version_info['version'],
                    'status': version_info['status'],
                    'created_at': version_info['created_at'],
                    'parent_version': metadata.parent_version,
                    'metrics': metadata.validation_metrics,
                    'deployed': version_info.get('is_deployed', False)
                })
            
            return sorted(lineage, key=lambda x: x['created_at'])
            
        except Exception as e:
            logger.error(f"Failed to get model lineage: {e}")
            raise
    
    def export_model(self,
                    model_id: str,
                    version: str,
                    export_format: str = 'onnx',
                    output_path: Optional[Path] = None) -> Path:
        """Export model to different formats"""
        try:
            model, metadata = self.load_model(model_id, version)
            
            if output_path is None:
                output_path = self.storage_path / "exports" / f"{model_id}_{version}.{export_format}"
            
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            if export_format == 'onnx':
                self._export_to_onnx(model, metadata, output_path)
            elif export_format == 'tflite':
                self._export_to_tflite(model, metadata, output_path)
            elif export_format == 'coreml':
                self._export_to_coreml(model, metadata, output_path)
            else:
                raise ValueError(f"Unsupported export format: {export_format}")
            
            logger.info(f"Model exported to {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Failed to export model: {e}")
            raise
    
    def _serialize_model(self, model: Any, framework: ModelFramework) -> bytes:
        """Serialize model based on framework"""
        if framework == ModelFramework.PYTORCH:
            buffer = io.BytesIO()
            torch.save(model.state_dict() if hasattr(model, 'state_dict') else model, buffer)
            return buffer.getvalue()
        elif framework == ModelFramework.TENSORFLOW:
            with tempfile.TemporaryDirectory() as tmpdir:
                model.save(tmpdir)
                return self._zip_directory(tmpdir)
        elif framework == ModelFramework.HUGGINGFACE:
            with tempfile.TemporaryDirectory() as tmpdir:
                model.save_pretrained(tmpdir)
                return self._zip_directory(tmpdir)
        else:
            return pickle.dumps(model)
    
    def _deserialize_model(self, data: bytes, framework: ModelFramework) -> Any:
        """Deserialize model based on framework"""
        if framework == ModelFramework.PYTORCH:
            buffer = io.BytesIO(data)
            return torch.load(buffer, map_location='cpu')
        elif framework == ModelFramework.TENSORFLOW:
            with tempfile.TemporaryDirectory() as tmpdir:
                self._unzip_data(data, tmpdir)
                return tf.keras.models.load_model(tmpdir)
        elif framework == ModelFramework.HUGGINGFACE:
            with tempfile.TemporaryDirectory() as tmpdir:
                self._unzip_data(data, tmpdir)
                from transformers import AutoModel
                return AutoModel.from_pretrained(tmpdir)
        else:
            return pickle.loads(data)
    
    def _count_parameters(self, model: Any, framework: ModelFramework) -> int:
        """Count model parameters"""
        try:
            if framework == ModelFramework.PYTORCH:
                return sum(p.numel() for p in model.parameters() if p.requires_grad)
            elif framework == ModelFramework.TENSORFLOW:
                return model.count_params()
            elif framework == ModelFramework.HUGGINGFACE:
                return sum(p.numel() for p in model.parameters())
            else:
                return 0
        except:
            return 0
    
    def _validate_version(self, version: str) -> bool:
        """Validate semantic version"""
        try:
            semver.VersionInfo.parse(version)
            return True
        except:
            return False
    
    def _validate_promotion(self, current: ModelStatus, target: ModelStatus) -> bool:
        """Validate promotion path"""
        valid_paths = {
            ModelStatus.DEVELOPMENT: [ModelStatus.STAGING, ModelStatus.ARCHIVED],
            ModelStatus.STAGING: [ModelStatus.PRODUCTION, ModelStatus.DEVELOPMENT, ModelStatus.ARCHIVED],
            ModelStatus.PRODUCTION: [ModelStatus.STAGING, ModelStatus.ARCHIVED, ModelStatus.DEPRECATED],
            ModelStatus.ARCHIVED: [ModelStatus.DEVELOPMENT],
            ModelStatus.DEPRECATED: [],
            ModelStatus.FAILED: [ModelStatus.DEVELOPMENT]
        }
        return target in valid_paths.get(current, [])
    
    def _compare_metrics(self, metadata1: ModelMetadata, metadata2: ModelMetadata) -> Dict[str, Any]:
        """Compare metrics between two models"""
        comparison = {}
        
        for metric_type in ['training_metrics', 'validation_metrics', 'test_metrics']:
            metrics1 = getattr(metadata1, metric_type)
            metrics2 = getattr(metadata2, metric_type)
            
            if metrics1 and metrics2:
                comparison[metric_type] = {}
                all_keys = set(metrics1.keys()) | set(metrics2.keys())
                
                for key in all_keys:
                    val1 = metrics1.get(key, 0)
                    val2 = metrics2.get(key, 0)
                    comparison[metric_type][key] = {
                        'version1': val1,
                        'version2': val2,
                        'diff': val2 - val1,
                        'improvement': ((val2 - val1) / val1 * 100) if val1 != 0 else 0
                    }
        
        return comparison
    
    # Placeholder methods for database operations
    def _version_exists(self, model_id: str, version: str) -> bool:
        """Check if version exists"""
        # Implementation would check database
        return False
    
    def _save_to_database(self, **kwargs):
        """Save model info to database"""
        # Implementation would save to database
        pass
    
    def _get_model_info(self, model_id: str, version: Optional[str] = None, status: Optional[ModelStatus] = None):
        """Get model info from database"""
        # Implementation would query database
        pass
    
    def _update_model_status(self, model_id: str, version: str, status: ModelStatus, updated_by: str):
        """Update model status in database"""
        # Implementation would update database
        pass
    
    def _get_production_model(self, model_id: str):
        """Get current production model"""
        # Implementation would query database
        pass
    
    def _demote_production_models(self, model_id: str):
        """Demote current production models"""
        # Implementation would update database
        pass
    
    def _deploy_model(self, model_id: str, version: str):
        """Deploy model to production"""
        # Implementation would handle deployment
        pass
    
    def _log_rollback(self, model_id: str, from_version: str, to_version: str, reason: str, by: str):
        """Log rollback event"""
        # Implementation would log to database
        pass
    
    def _get_all_versions(self, model_id: str) -> List[Dict[str, Any]]:
        """Get all versions of a model"""
        # Implementation would query database
        return []


# Singleton instance
_registry_instance = None

def get_model_registry() -> ModelRegistry:
    """Get or create model registry instance"""
    global _registry_instance
    if _registry_instance is None:
        _registry_instance = ModelRegistry()
    return _registry_instance


if __name__ == "__main__":
    # Example usage
    registry = get_model_registry()
    
    # Create sample metadata
    metadata = ModelMetadata(
        name="turkish-edu-model",
        version="1.0.0",
        framework=ModelFramework.HUGGINGFACE,
        model_type=ModelType.CONTENT_GENERATION,
        description="Turkish education content generation model",
        training_metrics={"loss": 0.23, "accuracy": 0.94},
        validation_metrics={"loss": 0.25, "accuracy": 0.92},
        tags=["turkish", "education", "nlp"]
    )
    
    print("Model Registry initialized successfully!")