"""
Comprehensive tests for ML Model Versioning System
Tests registry, versioning service, deployment strategies, and API endpoints
"""

import pytest
import asyncio
import json
import tempfile
import shutil
from pathlib import Path
from datetime import datetime, timedelta, timezone
from unittest.mock import Mock, patch, MagicMock, AsyncMock
import numpy as np
import pickle

from src.ml.model_registry import (
    ModelRegistry, ModelMetadata, ModelStatus,
    ModelFramework, ModelType
)
from src.ml.model_versioning_service import (
    ModelVersioningService, DeploymentConfig, DeploymentStrategy,
    ModelHealth, ModelPerformanceMetrics
)


class TestModelRegistry:
    """Test model registry functionality"""
    
    @pytest.fixture
    def registry(self, tmp_path):
        """Create test registry instance"""
        return ModelRegistry(storage_path=tmp_path / "models")
    
    @pytest.fixture
    def sample_model(self):
        """Create sample model"""
        # Simple sklearn model for testing
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression()
        model.fit([[0, 0], [1, 1]], [0, 1])
        return model
    
    @pytest.fixture
    def sample_metadata(self):
        """Create sample metadata"""
        return ModelMetadata(
            name="test-model",
            version="1.0.0",
            framework=ModelFramework.SKLEARN,
            model_type=ModelType.CLASSIFICATION,
            description="Test model for unit tests",
            training_metrics={"accuracy": 0.95, "loss": 0.05},
            validation_metrics={"accuracy": 0.93, "loss": 0.07},
            tags=["test", "classification"]
        )
    
    def test_register_model(self, registry, sample_model, sample_metadata):
        """Test model registration"""
        # Register model
        model_id = registry.register_model(
            model=sample_model,
            metadata=sample_metadata,
            artifacts={"scaler": "test_scaler"}
        )
        
        assert model_id is not None
        assert model_id == sample_metadata.model_id
        
        # Check files created
        model_dir = registry.storage_path / model_id / sample_metadata.version
        assert model_dir.exists()
        assert (model_dir / "model.pkl").exists()
        assert (model_dir / "metadata.json").exists()
        assert (model_dir / "artifacts").exists()
    
    def test_load_model(self, registry, sample_model, sample_metadata):
        """Test model loading"""
        # Register model first
        model_id = registry.register_model(sample_model, sample_metadata)
        
        # Mock database query
        with patch.object(registry, '_get_model_info') as mock_get:
            mock_get.return_value = {
                'metadata': sample_metadata.to_dict(),
                'local_path': str(registry.storage_path / model_id / sample_metadata.version / "model.pkl"),
                'version': sample_metadata.version,
                'status': ModelStatus.DEVELOPMENT.value
            }
            
            # Load model
            loaded_model, loaded_metadata = registry.load_model(model_id, sample_metadata.version)
            
            assert loaded_metadata.name == sample_metadata.name
            assert loaded_metadata.version == sample_metadata.version
    
    def test_version_validation(self, registry):
        """Test semantic version validation"""
        assert registry._validate_version("1.0.0") == True
        assert registry._validate_version("2.1.3-alpha") == True
        assert registry._validate_version("0.0.1") == True
        assert registry._validate_version("invalid") == False
        assert registry._validate_version("1.0") == False
    
    def test_promotion_validation(self, registry):
        """Test promotion path validation"""
        assert registry._validate_promotion(
            ModelStatus.DEVELOPMENT,
            ModelStatus.STAGING
        ) == True
        
        assert registry._validate_promotion(
            ModelStatus.STAGING,
            ModelStatus.PRODUCTION
        ) == True
        
        assert registry._validate_promotion(
            ModelStatus.PRODUCTION,
            ModelStatus.DEVELOPMENT
        ) == False
        
        assert registry._validate_promotion(
            ModelStatus.DEPRECATED,
            ModelStatus.PRODUCTION
        ) == False
    
    def test_model_comparison(self, registry, sample_metadata):
        """Test model comparison"""
        # Create two metadata versions
        metadata1 = sample_metadata
        metadata2 = ModelMetadata(
            name="test-model",
            version="1.1.0",
            framework=ModelFramework.SKLEARN,
            model_type=ModelType.CLASSIFICATION,
            training_metrics={"accuracy": 0.97, "loss": 0.03},
            validation_metrics={"accuracy": 0.95, "loss": 0.05}
        )
        
        # Compare metrics
        comparison = registry._compare_metrics(metadata1, metadata2)
        
        assert "training_metrics" in comparison
        assert "validation_metrics" in comparison
        
        # Check accuracy improvement
        acc_diff = comparison["training_metrics"]["accuracy"]
        assert acc_diff["version1"] == 0.95
        assert acc_diff["version2"] == 0.97
        assert acc_diff["diff"] == pytest.approx(0.02)
        assert acc_diff["improvement"] == pytest.approx(2.105, rel=1e-2)
    
    def test_model_export(self, registry, sample_model, sample_metadata, tmp_path):
        """Test model export functionality"""
        # Register model
        model_id = registry.register_model(sample_model, sample_metadata)
        
        # Mock load_model
        with patch.object(registry, 'load_model') as mock_load:
            mock_load.return_value = (sample_model, sample_metadata)
            
            # Mock export methods
            with patch.object(registry, '_export_to_onnx'):
                export_path = registry.export_model(
                    model_id,
                    sample_metadata.version,
                    export_format='onnx',
                    output_path=tmp_path / "export.onnx"
                )
                
                assert export_path == tmp_path / "export.onnx"


class TestModelVersioningService:
    """Test model versioning service"""
    
    @pytest.fixture
    def service(self):
        """Create test service instance"""
        service = ModelVersioningService()
        # Mock Redis client
        service.redis_client = MagicMock()
        return service
    
    @pytest.fixture
    def deployment_config(self):
        """Create test deployment config"""
        return DeploymentConfig(
            strategy=DeploymentStrategy.CANARY,
            canary_percentage=20.0,
            auto_rollback=True,
            rollback_threshold=0.1,
            min_requests_for_rollback=10
        )
    
    @pytest.mark.asyncio
    async def test_deploy_immediate(self, service):
        """Test immediate deployment"""
        model_id = "test-model"
        version = "1.0.0"
        model = Mock()
        metadata = Mock(spec=ModelMetadata)
        
        # Mock registry
        with patch.object(service.registry, 'load_model') as mock_load:
            mock_load.return_value = (model, metadata)
            
            with patch.object(service.registry, 'promote_model') as mock_promote:
                mock_promote.return_value = True
                
                with patch.object(service, '_check_model_health') as mock_health:
                    mock_health.return_value = ModelHealth.HEALTHY
                    
                    config = DeploymentConfig(strategy=DeploymentStrategy.IMMEDIATE)
                    result = await service.deploy_model(model_id, version, config)
                    
                    assert result["strategy"] == "immediate"
                    assert result["status"] == "deployed"
                    assert f"{model_id}:production" in service.active_models
    
    @pytest.mark.asyncio
    async def test_deploy_canary(self, service, deployment_config):
        """Test canary deployment"""
        model_id = "test-model"
        version = "1.0.0"
        model = Mock()
        metadata = Mock(spec=ModelMetadata)
        
        with patch.object(service.registry, 'load_model') as mock_load:
            mock_load.return_value = (model, metadata)
            
            with patch.object(service, '_check_model_health') as mock_health:
                mock_health.return_value = ModelHealth.HEALTHY
                
                result = await service.deploy_model(model_id, version, deployment_config)
                
                assert result["strategy"] == "canary"
                assert result["canary_percentage"] == 20.0
                assert f"{model_id}:canary" in service.active_models
                
                # Check Redis calls
                service.redis_client.hset.assert_called()
    
    @pytest.mark.asyncio
    async def test_prediction_routing(self, service):
        """Test prediction routing based on deployment"""
        model_id = "test-model"
        
        # Setup production model
        service.active_models[f"{model_id}:production"] = {
            'model': Mock(),
            'metadata': Mock(),
            'version': "1.0.0"
        }
        
        # Setup canary model
        service.active_models[f"{model_id}:canary"] = {
            'model': Mock(),
            'metadata': Mock(),
            'version': "1.1.0",
            'traffic_percentage': 30.0
        }
        
        # Test version selection
        with patch('random.random', return_value=0.2):  # 20% < 30%
            version = await service._select_version(model_id, None)
            assert version == "1.1.0"  # Canary version
        
        with patch('random.random', return_value=0.5):  # 50% > 30%
            version = await service._select_version(model_id, None)
            assert version == "1.0.0"  # Production version
    
    @pytest.mark.asyncio
    async def test_auto_rollback(self, service):
        """Test automatic rollback on failure"""
        model_id = "test-model"
        version = "1.1.0"
        
        # Setup metrics with poor performance
        metrics = ModelPerformanceMetrics(
            model_id=model_id,
            version=version,
            request_count=100,
            error_count=20,  # 20% error rate
            success_rate=0.8
        )
        service.model_metrics[f"{model_id}:{version}"] = metrics
        
        # Setup deployment config
        config = DeploymentConfig(
            auto_rollback=True,
            rollback_threshold=0.15,  # 15% error threshold
            min_requests_for_rollback=50
        )
        service.deployment_configs[f"{model_id}:{version}"] = config
        
        # Mock finding previous version
        with patch.object(service, '_find_previous_stable_version') as mock_find:
            mock_find.return_value = "1.0.0"
            
            with patch.object(service, '_perform_rollback') as mock_rollback:
                should_rollback = await service.rollback_if_needed(model_id, version)
                
                assert should_rollback == True
                mock_rollback.assert_called_once_with(model_id, version, "1.0.0")
    
    def test_metrics_update(self, service):
        """Test performance metrics update"""
        model_id = "test-model"
        version = "1.0.0"
        
        # Update metrics
        service._update_metrics(model_id, version, 100.0, True)
        service._update_metrics(model_id, version, 150.0, True)
        service._update_metrics(model_id, version, 200.0, False)
        
        metrics_key = f"{model_id}:{version}"
        metrics = service.model_metrics[metrics_key]
        
        assert metrics.request_count == 3
        assert metrics.error_count == 1
        assert metrics.success_rate == pytest.approx(0.667, rel=1e-2)
        assert metrics.avg_latency == pytest.approx(150.0)
        assert metrics.max_latency == 200.0
        assert metrics.min_latency == 100.0
    
    @pytest.mark.asyncio
    async def test_model_comparison(self, service):
        """Test model version comparison"""
        model_id = "test-model"
        
        # Mock models and predictions
        model1 = Mock()
        model2 = Mock()
        metadata1 = Mock(spec=ModelMetadata)
        metadata2 = Mock(spec=ModelMetadata)
        
        with patch.object(service.registry, 'load_model') as mock_load:
            mock_load.side_effect = [
                (model1, metadata1),
                (model2, metadata2)
            ]
            
            with patch.object(service, '_execute_prediction') as mock_predict:
                mock_predict.side_effect = [
                    {"result": "pred1"},
                    {"result": "pred2"}
                ]
                
                with patch.object(service.registry, 'compare_models') as mock_compare:
                    mock_compare.return_value = {"metrics": "comparison"}
                    
                    result = await service.compare_versions(
                        model_id, "1.0.0", "1.1.0", {"test": "data"}
                    )
                    
                    assert "comparison" in result
                    assert "latency" in result["comparison"]
    
    @pytest.mark.asyncio
    async def test_cleanup_old_versions(self, service):
        """Test cleanup of old model versions"""
        model_id = "test-model"
        
        # Mock lineage
        old_date = datetime.now(timezone.utc) - timedelta(days=60)
        recent_date = datetime.now(timezone.utc) - timedelta(days=5)
        
        lineage = [
            {"version": "1.0.0", "status": "archived", "created_at": old_date},
            {"version": "1.1.0", "status": "archived", "created_at": old_date},
            {"version": "1.2.0", "status": "staging", "created_at": recent_date},
            {"version": "1.3.0", "status": "production", "created_at": recent_date},
        ]
        
        with patch.object(service.registry, 'get_model_lineage') as mock_lineage:
            mock_lineage.return_value = lineage
            
            with patch.object(service.registry, 'promote_model') as mock_promote:
                cleaned = await service.cleanup_old_versions(
                    model_id,
                    keep_count=1,
                    keep_days=30
                )
                
                # Should clean only old archived versions
                assert len(cleaned) == 2
                assert "1.0.0" in cleaned
                assert "1.1.0" in cleaned
    
    @pytest.mark.asyncio
    async def test_gradual_rollout(self, service):
        """Test gradual rollout deployment"""
        model_id = "test-model"
        version = "1.0.0"
        model = Mock()
        metadata = Mock(spec=ModelMetadata)
        
        with patch.object(service.registry, 'load_model') as mock_load:
            mock_load.return_value = (model, metadata)
            
            with patch.object(service, '_check_model_health') as mock_health:
                mock_health.return_value = ModelHealth.HEALTHY
                
                config = DeploymentConfig(
                    strategy=DeploymentStrategy.GRADUAL,
                    gradual_rollout_minutes=1  # Quick for testing
                )
                
                result = await service.deploy_model(model_id, version, config)
                
                assert result["strategy"] == "gradual"
                assert result["initial_percentage"] == 5.0
                assert f"{model_id}:gradual" in service.active_models
                
                # Check initial percentage
                gradual_info = service.active_models[f"{model_id}:gradual"]
                assert gradual_info["current_percentage"] == 5.0
                assert gradual_info["target_percentage"] == 100.0


class TestModelPerformanceMetrics:
    """Test performance metrics tracking"""
    
    def test_metrics_initialization(self):
        """Test metrics initialization"""
        metrics = ModelPerformanceMetrics(
            model_id="test-model",
            version="1.0.0"
        )
        
        assert metrics.request_count == 0
        assert metrics.error_count == 0
        assert metrics.success_rate == 1.0
        assert metrics.avg_latency == 0.0
    
    def test_metrics_update(self):
        """Test metrics update"""
        metrics = ModelPerformanceMetrics(
            model_id="test-model",
            version="1.0.0"
        )
        
        # Add successful requests
        metrics.update(100.0, True)
        metrics.update(200.0, True)
        
        assert metrics.request_count == 2
        assert metrics.error_count == 0
        assert metrics.success_rate == 1.0
        assert metrics.avg_latency == 150.0
        assert metrics.max_latency == 200.0
        assert metrics.min_latency == 100.0
        
        # Add failed request
        metrics.update(50.0, False)
        
        assert metrics.request_count == 3
        assert metrics.error_count == 1
        assert metrics.success_rate == pytest.approx(0.667, rel=1e-2)
        assert metrics.min_latency == 50.0


class TestDeploymentStrategies:
    """Test different deployment strategies"""
    
    @pytest.mark.asyncio
    async def test_blue_green_deployment(self):
        """Test blue-green deployment strategy"""
        service = ModelVersioningService()
        model_id = "test-model"
        
        # Setup blue environment (current production)
        service.active_models[f"{model_id}:production"] = {
            'model': Mock(),
            'version': "1.0.0"
        }
        
        # Deploy green environment
        model = Mock()
        metadata = Mock()
        
        result = await service._deploy_blue_green(
            model_id, "1.1.0", model, metadata
        )
        
        assert result["strategy"] == "blue_green"
        
        # Check blue is preserved
        assert f"{model_id}:blue" in service.active_models
        assert service.active_models[f"{model_id}:blue"]["version"] == "1.0.0"
        
        # Check green is now production
        assert service.active_models[f"{model_id}:production"]["version"] == "1.1.0"
    
    @pytest.mark.asyncio
    async def test_ab_test_deployment(self):
        """Test A/B test deployment"""
        service = ModelVersioningService()
        service.redis_client = MagicMock()
        
        model_id = "test-model"
        version = "1.1.0"
        model = Mock()
        metadata = Mock()
        
        config = DeploymentConfig(
            strategy=DeploymentStrategy.A_B_TEST,
            traffic_split={"1.0.0": 60, "1.1.0": 40},
            a_b_test_duration_hours=24
        )
        
        result = await service._deploy_ab_test(
            model_id, version, model, metadata, config
        )
        
        assert result["strategy"] == "a_b_test"
        assert result["traffic_split"] == config.traffic_split
        assert result["duration_hours"] == 24
        
        # Check model is added to A/B pool
        assert f"{model_id}:ab_{version}" in service.active_models
        
        # Check Redis configuration
        service.redis_client.hset.assert_called()


class TestHealthMonitoring:
    """Test model health monitoring"""
    
    @pytest.mark.asyncio
    async def test_health_check(self):
        """Test model health check"""
        service = ModelVersioningService()
        
        model = Mock()
        metadata = Mock(spec=ModelMetadata)
        metadata.model_type = ModelType.NLP
        
        with patch.object(service, '_execute_prediction') as mock_predict:
            # Healthy model
            mock_predict.return_value = {"result": "success"}
            health = await service._check_model_health(model, metadata)
            assert health == ModelHealth.HEALTHY
            
            # Unhealthy model
            mock_predict.side_effect = Exception("Model error")
            health = await service._check_model_health(model, metadata)
            assert health == ModelHealth.UNHEALTHY
    
    @pytest.mark.asyncio
    async def test_health_monitoring_task(self):
        """Test health monitoring background task"""
        service = ModelVersioningService()
        service.redis_client = MagicMock()
        
        model_id = "test-model"
        version = "1.0.0"
        
        # Add model to active models
        service.active_models[f"{model_id}:{version}"] = {
            'model': Mock(),
            'metadata': Mock(spec=ModelMetadata)
        }
        
        config = DeploymentConfig(
            health_check_interval=1,  # Quick for testing
            auto_rollback=False
        )
        
        # Start monitoring
        await service._start_health_monitoring(model_id, version, config)
        
        # Check task was created
        task_key = f"{model_id}:{version}"
        assert task_key in service.health_check_tasks
        assert service.health_check_tasks[task_key] is not None
        
        # Cancel task for cleanup
        service.health_check_tasks[task_key].cancel()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])