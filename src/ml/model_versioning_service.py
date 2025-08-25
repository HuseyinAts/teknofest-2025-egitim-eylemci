"""
Model Versioning Service with Deployment and Monitoring
Handles model lifecycle, A/B testing, and automated rollback
"""

import asyncio
import json
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import numpy as np
from dataclasses import dataclass, field
import redis
from prometheus_client import Counter, Histogram, Gauge
import logging
from contextlib import asynccontextmanager
import aiohttp
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
import yaml

from src.config import get_settings
from src.ml.model_registry import (
    ModelRegistry, ModelMetadata, ModelStatus, 
    ModelFramework, ModelType, get_model_registry
)
from src.database.base import Base

logger = logging.getLogger(__name__)
settings = get_settings()


# Metrics
model_inference_counter = Counter(
    'model_inference_total',
    'Total number of model inferences',
    ['model_id', 'version', 'status']
)

model_inference_duration = Histogram(
    'model_inference_duration_seconds',
    'Model inference duration',
    ['model_id', 'version']
)

model_error_counter = Counter(
    'model_errors_total',
    'Total number of model errors',
    ['model_id', 'version', 'error_type']
)

active_model_gauge = Gauge(
    'active_models',
    'Number of active models',
    ['status']
)

model_performance_gauge = Gauge(
    'model_performance_score',
    'Model performance score',
    ['model_id', 'version', 'metric']
)


class DeploymentStrategy(str, Enum):
    """Model deployment strategies"""
    IMMEDIATE = "immediate"
    CANARY = "canary"
    BLUE_GREEN = "blue_green"
    A_B_TEST = "a_b_test"
    GRADUAL = "gradual"


class ModelHealth(str, Enum):
    """Model health status"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class DeploymentConfig:
    """Configuration for model deployment"""
    strategy: DeploymentStrategy = DeploymentStrategy.IMMEDIATE
    canary_percentage: float = 10.0
    gradual_rollout_minutes: int = 60
    health_check_interval: int = 30
    auto_rollback: bool = True
    rollback_threshold: float = 0.1
    min_requests_for_rollback: int = 100
    a_b_test_duration_hours: int = 24
    traffic_split: Dict[str, float] = field(default_factory=dict)


@dataclass
class ModelPerformanceMetrics:
    """Real-time model performance metrics"""
    model_id: str
    version: str
    request_count: int = 0
    error_count: int = 0
    total_latency: float = 0.0
    max_latency: float = 0.0
    min_latency: float = float('inf')
    success_rate: float = 1.0
    p50_latency: float = 0.0
    p95_latency: float = 0.0
    p99_latency: float = 0.0
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def update(self, latency: float, success: bool):
        """Update metrics with new request"""
        self.request_count += 1
        if not success:
            self.error_count += 1
        self.total_latency += latency
        self.max_latency = max(self.max_latency, latency)
        self.min_latency = min(self.min_latency, latency)
        self.success_rate = (self.request_count - self.error_count) / self.request_count
        self.last_updated = datetime.now(timezone.utc)
    
    @property
    def avg_latency(self) -> float:
        """Calculate average latency"""
        return self.total_latency / self.request_count if self.request_count > 0 else 0


class ModelVersioningService:
    """Service for managing model versions and deployments"""
    
    def __init__(self):
        """Initialize versioning service"""
        self.registry = get_model_registry()
        self.redis_client = None
        self._init_redis()
        
        # Active models cache
        self.active_models: Dict[str, Dict[str, Any]] = {}
        self.model_metrics: Dict[str, ModelPerformanceMetrics] = {}
        
        # Deployment configurations
        self.deployment_configs: Dict[str, DeploymentConfig] = {}
        
        # Health check tasks
        self.health_check_tasks: Dict[str, asyncio.Task] = {}
        
        logger.info("Model Versioning Service initialized")
    
    def _init_redis(self):
        """Initialize Redis connection"""
        try:
            self.redis_client = redis.from_url(
                settings.redis_url,
                decode_responses=True
            )
            self.redis_client.ping()
            logger.info("Redis connected for model versioning")
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}")
            self.redis_client = None
    
    async def deploy_model(self,
                          model_id: str,
                          version: str,
                          config: Optional[DeploymentConfig] = None) -> Dict[str, Any]:
        """Deploy a model version with specified strategy"""
        try:
            config = config or DeploymentConfig()
            
            # Load model and metadata
            model, metadata = self.registry.load_model(model_id, version)
            
            # Validate model health before deployment
            health_status = await self._check_model_health(model, metadata)
            if health_status != ModelHealth.HEALTHY:
                raise ValueError(f"Model health check failed: {health_status}")
            
            # Store deployment config
            self.deployment_configs[f"{model_id}:{version}"] = config
            
            # Deploy based on strategy
            if config.strategy == DeploymentStrategy.IMMEDIATE:
                result = await self._deploy_immediate(model_id, version, model, metadata)
            elif config.strategy == DeploymentStrategy.CANARY:
                result = await self._deploy_canary(model_id, version, model, metadata, config)
            elif config.strategy == DeploymentStrategy.BLUE_GREEN:
                result = await self._deploy_blue_green(model_id, version, model, metadata)
            elif config.strategy == DeploymentStrategy.A_B_TEST:
                result = await self._deploy_ab_test(model_id, version, model, metadata, config)
            elif config.strategy == DeploymentStrategy.GRADUAL:
                result = await self._deploy_gradual(model_id, version, model, metadata, config)
            else:
                raise ValueError(f"Unknown deployment strategy: {config.strategy}")
            
            # Start health monitoring
            await self._start_health_monitoring(model_id, version, config)
            
            # Update metrics
            active_model_gauge.labels(status='deployed').inc()
            
            logger.info(f"Model {model_id}:{version} deployed with strategy {config.strategy}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to deploy model: {e}")
            model_error_counter.labels(
                model_id=model_id,
                version=version,
                error_type='deployment'
            ).inc()
            raise
    
    async def predict(self,
                     model_id: str,
                     input_data: Any,
                     version: Optional[str] = None) -> Tuple[Any, str]:
        """Make prediction with appropriate model version"""
        start_time = time.time()
        selected_version = None
        
        try:
            # Select version based on deployment strategy
            selected_version = await self._select_version(model_id, version)
            
            # Get model from cache or load
            model_key = f"{model_id}:{selected_version}"
            if model_key not in self.active_models:
                model, metadata = self.registry.load_model(model_id, selected_version)
                self.active_models[model_key] = {
                    'model': model,
                    'metadata': metadata,
                    'loaded_at': datetime.now(timezone.utc)
                }
            
            model_info = self.active_models[model_key]
            model = model_info['model']
            
            # Make prediction
            prediction = await self._execute_prediction(model, input_data)
            
            # Update metrics
            latency = time.time() - start_time
            self._update_metrics(model_id, selected_version, latency, True)
            
            # Record inference
            model_inference_counter.labels(
                model_id=model_id,
                version=selected_version,
                status='success'
            ).inc()
            
            model_inference_duration.labels(
                model_id=model_id,
                version=selected_version
            ).observe(latency)
            
            return prediction, selected_version
            
        except Exception as e:
            latency = time.time() - start_time
            if selected_version:
                self._update_metrics(model_id, selected_version, latency, False)
                model_error_counter.labels(
                    model_id=model_id,
                    version=selected_version,
                    error_type='inference'
                ).inc()
            
            logger.error(f"Prediction failed for {model_id}: {e}")
            raise
    
    async def rollback_if_needed(self, model_id: str, version: str) -> bool:
        """Check metrics and rollback if necessary"""
        try:
            config_key = f"{model_id}:{version}"
            config = self.deployment_configs.get(config_key)
            
            if not config or not config.auto_rollback:
                return False
            
            # Get performance metrics
            metrics_key = f"{model_id}:{version}"
            metrics = self.model_metrics.get(metrics_key)
            
            if not metrics or metrics.request_count < config.min_requests_for_rollback:
                return False
            
            # Check if rollback is needed
            if metrics.success_rate < (1 - config.rollback_threshold):
                logger.warning(f"Model {model_id}:{version} performance degraded. Success rate: {metrics.success_rate}")
                
                # Find previous stable version
                previous_version = await self._find_previous_stable_version(model_id, version)
                if previous_version:
                    # Perform rollback
                    await self._perform_rollback(model_id, version, previous_version)
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to check rollback: {e}")
            return False
    
    async def get_model_metrics(self, model_id: str, version: Optional[str] = None) -> Dict[str, Any]:
        """Get performance metrics for a model"""
        try:
            if version:
                metrics_key = f"{model_id}:{version}"
                metrics = self.model_metrics.get(metrics_key)
                if metrics:
                    return {
                        'model_id': model_id,
                        'version': version,
                        'request_count': metrics.request_count,
                        'error_count': metrics.error_count,
                        'success_rate': metrics.success_rate,
                        'avg_latency': metrics.avg_latency,
                        'max_latency': metrics.max_latency,
                        'min_latency': metrics.min_latency if metrics.min_latency != float('inf') else 0,
                        'last_updated': metrics.last_updated.isoformat()
                    }
            else:
                # Get metrics for all versions
                all_metrics = {}
                for key, metrics in self.model_metrics.items():
                    if key.startswith(f"{model_id}:"):
                        version = key.split(':')[1]
                        all_metrics[version] = {
                            'request_count': metrics.request_count,
                            'success_rate': metrics.success_rate,
                            'avg_latency': metrics.avg_latency
                        }
                return all_metrics
            
            return {}
            
        except Exception as e:
            logger.error(f"Failed to get metrics: {e}")
            return {}
    
    async def compare_versions(self,
                              model_id: str,
                              version1: str,
                              version2: str,
                              test_data: Any) -> Dict[str, Any]:
        """Compare two model versions"""
        try:
            results = {
                'model_id': model_id,
                'version1': version1,
                'version2': version2,
                'comparison': {}
            }
            
            # Load both models
            model1, metadata1 = self.registry.load_model(model_id, version1)
            model2, metadata2 = self.registry.load_model(model_id, version2)
            
            # Run predictions and measure performance
            start1 = time.time()
            pred1 = await self._execute_prediction(model1, test_data)
            time1 = time.time() - start1
            
            start2 = time.time()
            pred2 = await self._execute_prediction(model2, test_data)
            time2 = time.time() - start2
            
            # Compare results
            results['comparison'] = {
                'latency': {
                    'version1': time1,
                    'version2': time2,
                    'improvement': ((time1 - time2) / time1 * 100) if time1 > 0 else 0
                },
                'metadata': self.registry.compare_models(model_id, version1, version2),
                'predictions_match': pred1 == pred2 if isinstance(pred1, (str, int, float)) else None
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to compare versions: {e}")
            raise
    
    async def cleanup_old_versions(self,
                                  model_id: str,
                                  keep_count: int = 5,
                                  keep_days: int = 30) -> List[str]:
        """Clean up old model versions"""
        try:
            cleaned_versions = []
            
            # Get all versions
            lineage = self.registry.get_model_lineage(model_id)
            
            # Sort by creation date
            lineage.sort(key=lambda x: x['created_at'], reverse=True)
            
            # Keep production and staging versions
            protected_statuses = [ModelStatus.PRODUCTION, ModelStatus.STAGING]
            protected_versions = [
                v['version'] for v in lineage 
                if v['status'] in [s.value for s in protected_statuses]
            ]
            
            # Determine versions to clean
            cutoff_date = datetime.now(timezone.utc) - timedelta(days=keep_days)
            
            for i, version_info in enumerate(lineage):
                version = version_info['version']
                
                # Skip protected versions
                if version in protected_versions:
                    continue
                
                # Skip recent versions
                if i < keep_count:
                    continue
                
                # Skip versions newer than cutoff
                if version_info['created_at'] > cutoff_date:
                    continue
                
                # Archive this version
                self.registry.promote_model(
                    model_id,
                    version,
                    ModelStatus.ARCHIVED,
                    'system'
                )
                cleaned_versions.append(version)
                
                # Remove from cache
                cache_key = f"{model_id}:{version}"
                if cache_key in self.active_models:
                    del self.active_models[cache_key]
            
            logger.info(f"Cleaned {len(cleaned_versions)} old versions of {model_id}")
            return cleaned_versions
            
        except Exception as e:
            logger.error(f"Failed to cleanup versions: {e}")
            return []
    
    async def _deploy_immediate(self,
                               model_id: str,
                               version: str,
                               model: Any,
                               metadata: ModelMetadata) -> Dict[str, Any]:
        """Deploy model immediately"""
        # Update active model
        self.active_models[f"{model_id}:production"] = {
            'model': model,
            'metadata': metadata,
            'version': version,
            'deployed_at': datetime.now(timezone.utc)
        }
        
        # Update registry
        self.registry.promote_model(model_id, version, ModelStatus.PRODUCTION, 'system')
        
        return {
            'strategy': 'immediate',
            'model_id': model_id,
            'version': version,
            'status': 'deployed'
        }
    
    async def _deploy_canary(self,
                            model_id: str,
                            version: str,
                            model: Any,
                            metadata: ModelMetadata,
                            config: DeploymentConfig) -> Dict[str, Any]:
        """Deploy model with canary strategy"""
        # Keep both versions active
        self.active_models[f"{model_id}:canary"] = {
            'model': model,
            'metadata': metadata,
            'version': version,
            'deployed_at': datetime.now(timezone.utc),
            'traffic_percentage': config.canary_percentage
        }
        
        # Set traffic split in Redis
        if self.redis_client:
            self.redis_client.hset(
                f"model_traffic:{model_id}",
                "canary_version", version,
                "canary_percentage", config.canary_percentage
            )
        
        return {
            'strategy': 'canary',
            'model_id': model_id,
            'version': version,
            'canary_percentage': config.canary_percentage,
            'status': 'deployed'
        }
    
    async def _deploy_blue_green(self,
                                model_id: str,
                                version: str,
                                model: Any,
                                metadata: ModelMetadata) -> Dict[str, Any]:
        """Deploy model with blue-green strategy"""
        # Prepare green environment
        self.active_models[f"{model_id}:green"] = {
            'model': model,
            'metadata': metadata,
            'version': version,
            'deployed_at': datetime.now(timezone.utc)
        }
        
        # Switch traffic
        if f"{model_id}:production" in self.active_models:
            self.active_models[f"{model_id}:blue"] = self.active_models[f"{model_id}:production"]
        
        self.active_models[f"{model_id}:production"] = self.active_models[f"{model_id}:green"]
        
        return {
            'strategy': 'blue_green',
            'model_id': model_id,
            'version': version,
            'status': 'deployed'
        }
    
    async def _deploy_ab_test(self,
                             model_id: str,
                             version: str,
                             model: Any,
                             metadata: ModelMetadata,
                             config: DeploymentConfig) -> Dict[str, Any]:
        """Deploy model for A/B testing"""
        # Add to A/B test pool
        self.active_models[f"{model_id}:ab_{version}"] = {
            'model': model,
            'metadata': metadata,
            'version': version,
            'deployed_at': datetime.now(timezone.utc),
            'traffic_split': config.traffic_split.get(version, 50.0)
        }
        
        # Set A/B test configuration
        if self.redis_client:
            self.redis_client.hset(
                f"model_ab_test:{model_id}",
                version, json.dumps({
                    'traffic_split': config.traffic_split.get(version, 50.0),
                    'start_time': datetime.now(timezone.utc).isoformat(),
                    'duration_hours': config.a_b_test_duration_hours
                })
            )
        
        return {
            'strategy': 'a_b_test',
            'model_id': model_id,
            'version': version,
            'traffic_split': config.traffic_split,
            'duration_hours': config.a_b_test_duration_hours,
            'status': 'deployed'
        }
    
    async def _deploy_gradual(self,
                             model_id: str,
                             version: str,
                             model: Any,
                             metadata: ModelMetadata,
                             config: DeploymentConfig) -> Dict[str, Any]:
        """Deploy model with gradual rollout"""
        # Start with minimal traffic
        initial_percentage = 5.0
        
        self.active_models[f"{model_id}:gradual"] = {
            'model': model,
            'metadata': metadata,
            'version': version,
            'deployed_at': datetime.now(timezone.utc),
            'current_percentage': initial_percentage,
            'target_percentage': 100.0,
            'rollout_end_time': datetime.now(timezone.utc) + timedelta(minutes=config.gradual_rollout_minutes)
        }
        
        # Schedule gradual increase
        asyncio.create_task(self._gradual_rollout_task(model_id, version, config))
        
        return {
            'strategy': 'gradual',
            'model_id': model_id,
            'version': version,
            'initial_percentage': initial_percentage,
            'rollout_minutes': config.gradual_rollout_minutes,
            'status': 'deploying'
        }
    
    async def _gradual_rollout_task(self,
                                   model_id: str,
                                   version: str,
                                   config: DeploymentConfig):
        """Task to gradually increase traffic"""
        try:
            steps = 10
            step_duration = config.gradual_rollout_minutes * 60 / steps
            
            for i in range(1, steps + 1):
                await asyncio.sleep(step_duration)
                
                # Check if rollback needed
                if await self.rollback_if_needed(model_id, version):
                    logger.info(f"Gradual rollout stopped due to rollback for {model_id}:{version}")
                    return
                
                # Increase traffic percentage
                percentage = min(100, i * 10)
                model_key = f"{model_id}:gradual"
                
                if model_key in self.active_models:
                    self.active_models[model_key]['current_percentage'] = percentage
                    
                    if self.redis_client:
                        self.redis_client.hset(
                            f"model_traffic:{model_id}",
                            "gradual_percentage", percentage
                        )
                    
                    logger.info(f"Gradual rollout: {model_id}:{version} at {percentage}%")
            
            # Complete rollout
            await self._deploy_immediate(model_id, version, 
                                        self.active_models[model_key]['model'],
                                        self.active_models[model_key]['metadata'])
            
        except Exception as e:
            logger.error(f"Gradual rollout failed: {e}")
    
    async def _select_version(self, model_id: str, requested_version: Optional[str]) -> str:
        """Select appropriate model version based on deployment strategy"""
        if requested_version:
            return requested_version
        
        # Check for A/B test
        if self.redis_client:
            ab_test = self.redis_client.hgetall(f"model_ab_test:{model_id}")
            if ab_test:
                # Select based on traffic split
                import random
                total_split = sum(json.loads(v)['traffic_split'] for v in ab_test.values())
                rand = random.uniform(0, total_split)
                cumulative = 0
                
                for version, config in ab_test.items():
                    config_data = json.loads(config)
                    cumulative += config_data['traffic_split']
                    if rand <= cumulative:
                        return version
        
        # Check for canary deployment
        if f"{model_id}:canary" in self.active_models:
            import random
            canary_info = self.active_models[f"{model_id}:canary"]
            if random.random() * 100 < canary_info['traffic_percentage']:
                return canary_info['version']
        
        # Check for gradual rollout
        if f"{model_id}:gradual" in self.active_models:
            import random
            gradual_info = self.active_models[f"{model_id}:gradual"]
            if random.random() * 100 < gradual_info['current_percentage']:
                return gradual_info['version']
        
        # Default to production version
        if f"{model_id}:production" in self.active_models:
            return self.active_models[f"{model_id}:production"]['version']
        
        # Fallback to latest version
        lineage = self.registry.get_model_lineage(model_id)
        if lineage:
            return lineage[-1]['version']
        
        raise ValueError(f"No version available for model {model_id}")
    
    async def _execute_prediction(self, model: Any, input_data: Any) -> Any:
        """Execute model prediction"""
        # This would be implemented based on model framework
        # For now, return a mock prediction
        return {"prediction": "mock_result", "confidence": 0.95}
    
    async def _check_model_health(self, model: Any, metadata: ModelMetadata) -> ModelHealth:
        """Check model health"""
        try:
            # Run basic health check
            test_input = self._get_test_input(metadata)
            result = await self._execute_prediction(model, test_input)
            
            if result:
                return ModelHealth.HEALTHY
            else:
                return ModelHealth.DEGRADED
                
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return ModelHealth.UNHEALTHY
    
    async def _start_health_monitoring(self,
                                      model_id: str,
                                      version: str,
                                      config: DeploymentConfig):
        """Start health monitoring for deployed model"""
        task_key = f"{model_id}:{version}"
        
        # Cancel existing task if any
        if task_key in self.health_check_tasks:
            self.health_check_tasks[task_key].cancel()
        
        # Create new monitoring task
        task = asyncio.create_task(
            self._health_monitor_task(model_id, version, config)
        )
        self.health_check_tasks[task_key] = task
    
    async def _health_monitor_task(self,
                                  model_id: str,
                                  version: str,
                                  config: DeploymentConfig):
        """Background task for health monitoring"""
        try:
            while True:
                await asyncio.sleep(config.health_check_interval)
                
                # Check model health
                model_key = f"{model_id}:{version}"
                if model_key in self.active_models:
                    model_info = self.active_models[model_key]
                    health = await self._check_model_health(
                        model_info['model'],
                        model_info['metadata']
                    )
                    
                    # Update health status
                    if self.redis_client:
                        self.redis_client.hset(
                            f"model_health:{model_id}",
                            version, health.value
                        )
                    
                    # Check if rollback needed
                    if health == ModelHealth.UNHEALTHY and config.auto_rollback:
                        await self.rollback_if_needed(model_id, version)
                        break
                        
        except asyncio.CancelledError:
            logger.info(f"Health monitoring stopped for {model_id}:{version}")
        except Exception as e:
            logger.error(f"Health monitoring error: {e}")
    
    def _update_metrics(self, model_id: str, version: str, latency: float, success: bool):
        """Update performance metrics"""
        metrics_key = f"{model_id}:{version}"
        
        if metrics_key not in self.model_metrics:
            self.model_metrics[metrics_key] = ModelPerformanceMetrics(
                model_id=model_id,
                version=version
            )
        
        self.model_metrics[metrics_key].update(latency, success)
        
        # Update Prometheus metrics
        model_performance_gauge.labels(
            model_id=model_id,
            version=version,
            metric='success_rate'
        ).set(self.model_metrics[metrics_key].success_rate)
        
        model_performance_gauge.labels(
            model_id=model_id,
            version=version,
            metric='avg_latency'
        ).set(self.model_metrics[metrics_key].avg_latency)
    
    async def _find_previous_stable_version(self, model_id: str, current_version: str) -> Optional[str]:
        """Find previous stable version for rollback"""
        lineage = self.registry.get_model_lineage(model_id)
        
        # Find current version index
        current_idx = None
        for i, v in enumerate(lineage):
            if v['version'] == current_version:
                current_idx = i
                break
        
        if current_idx is None or current_idx == 0:
            return None
        
        # Find previous production or staging version
        for i in range(current_idx - 1, -1, -1):
            if lineage[i]['status'] in [ModelStatus.PRODUCTION.value, ModelStatus.STAGING.value]:
                return lineage[i]['version']
        
        return None
    
    async def _perform_rollback(self, model_id: str, from_version: str, to_version: str):
        """Perform model rollback"""
        logger.warning(f"Rolling back {model_id} from {from_version} to {to_version}")
        
        # Load target version
        model, metadata = self.registry.load_model(model_id, to_version)
        
        # Deploy immediately
        await self._deploy_immediate(model_id, to_version, model, metadata)
        
        # Archive failed version
        self.registry.promote_model(model_id, from_version, ModelStatus.FAILED, 'system')
        
        # Clean up caches
        for key in list(self.active_models.keys()):
            if key.startswith(f"{model_id}:") and from_version in str(self.active_models[key].get('version', '')):
                del self.active_models[key]
    
    def _get_test_input(self, metadata: ModelMetadata) -> Any:
        """Get test input for health check"""
        # This would be customized based on model type
        if metadata.model_type == ModelType.NLP:
            return "Test input for health check"
        else:
            return np.zeros(metadata.input_shape) if metadata.input_shape else [0]


# Singleton instance
_service_instance = None

def get_versioning_service() -> ModelVersioningService:
    """Get or create versioning service instance"""
    global _service_instance
    if _service_instance is None:
        _service_instance = ModelVersioningService()
    return _service_instance


if __name__ == "__main__":
    import asyncio
    
    async def main():
        service = get_versioning_service()
        
        # Example deployment
        config = DeploymentConfig(
            strategy=DeploymentStrategy.CANARY,
            canary_percentage=20,
            auto_rollback=True,
            rollback_threshold=0.1
        )
        
        print("Model Versioning Service initialized successfully!")
    
    asyncio.run(main())