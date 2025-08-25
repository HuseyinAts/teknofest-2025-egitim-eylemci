"""
Production-Ready Orchestrator for Multi-Worker Deployment
TEKNOFEST 2025 - Advanced Container Orchestration
"""

import os
import sys
import time
import json
import asyncio
import docker
import psutil
import signal
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from collections import deque, defaultdict
from enum import Enum
import aioredis
import aiodocker
from prometheus_client import Counter, Gauge, Histogram, start_http_server
import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
)
logger = logging.getLogger(__name__)

# Prometheus metrics
worker_count_gauge = Gauge('orchestrator_worker_count', 'Number of workers', ['type', 'status'])
scaling_events_counter = Counter('orchestrator_scaling_events', 'Scaling events', ['type', 'direction'])
health_check_duration = Histogram('orchestrator_health_check_duration', 'Health check duration')
container_restarts = Counter('orchestrator_container_restarts', 'Container restart count', ['service'])
resource_usage = Gauge('orchestrator_resource_usage', 'Resource usage', ['resource', 'service'])

class WorkerType(Enum):
    API = "api"
    CELERY_DEFAULT = "celery_default"
    CELERY_AI = "celery_ai"
    CELERY_DATA = "celery_data"
    
class ServiceStatus(Enum):
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    DEGRADED = "degraded"
    STARTING = "starting"
    STOPPING = "stopping"
    UNKNOWN = "unknown"

@dataclass
class ServiceHealth:
    """Health status of a service"""
    service_name: str
    status: ServiceStatus
    last_check: datetime
    consecutive_failures: int = 0
    error_message: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    
@dataclass
class ScalingPolicy:
    """Scaling policy for a service"""
    min_replicas: int
    max_replicas: int
    target_cpu_utilization: float = 70.0
    target_memory_utilization: float = 80.0
    scale_up_threshold: float = 80.0
    scale_down_threshold: float = 30.0
    scale_up_increment: int = 1
    scale_down_increment: int = 1
    cooldown_seconds: int = 60
    
@dataclass
class ServiceConfig:
    """Configuration for a service"""
    name: str
    image: str
    worker_type: WorkerType
    scaling_policy: ScalingPolicy
    health_check_endpoint: Optional[str] = None
    health_check_interval: int = 30
    restart_policy: Dict[str, Any] = field(default_factory=lambda: {
        "max_attempts": 3,
        "delay": 5,
        "window": 120
    })
    environment: Dict[str, str] = field(default_factory=dict)
    labels: Dict[str, str] = field(default_factory=dict)
    resources: Dict[str, Any] = field(default_factory=lambda: {
        "limits": {"cpus": "2", "memory": "2G"},
        "reservations": {"cpus": "1", "memory": "1G"}
    })

class ContainerOrchestrator:
    """Advanced container orchestrator with auto-scaling and health management"""
    
    def __init__(self):
        self.docker_client = None
        self.async_docker = None
        self.redis_client = None
        self.services: Dict[str, ServiceConfig] = {}
        self.health_status: Dict[str, ServiceHealth] = {}
        self.scaling_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.last_scale_time: Dict[str, datetime] = {}
        self.running = False
        self.tasks: List[asyncio.Task] = []
        
    async def initialize(self):
        """Initialize orchestrator components"""
        try:
            # Initialize Docker clients
            self.docker_client = docker.from_env()
            self.async_docker = aiodocker.Docker()
            
            # Initialize Redis client
            self.redis_client = await aioredis.create_redis_pool(
                os.getenv('REDIS_URL', 'redis://localhost:6379'),
                minsize=5,
                maxsize=10
            )
            
            # Load service configurations
            await self.load_service_configs()
            
            # Start Prometheus metrics server
            start_http_server(9091)
            
            logger.info("Orchestrator initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize orchestrator: {e}")
            raise
            
    async def load_service_configs(self):
        """Load service configurations"""
        # API Workers
        self.services['api'] = ServiceConfig(
            name='teknofest-api',
            image='teknofest/api:latest',
            worker_type=WorkerType.API,
            scaling_policy=ScalingPolicy(
                min_replicas=int(os.getenv('MIN_API_WORKERS', 2)),
                max_replicas=int(os.getenv('MAX_API_WORKERS', 20)),
                target_cpu_utilization=70.0,
                scale_up_threshold=float(os.getenv('SCALE_UP_THRESHOLD', 80)),
                scale_down_threshold=float(os.getenv('SCALE_DOWN_THRESHOLD', 30))
            ),
            health_check_endpoint='/health/live',
            health_check_interval=int(os.getenv('HEALTH_CHECK_INTERVAL', 30))
        )
        
        # Celery Workers
        for worker_type in ['default', 'ai', 'data']:
            self.services[f'celery-{worker_type}'] = ServiceConfig(
                name=f'teknofest-celery-{worker_type}',
                image='teknofest/worker:latest',
                worker_type=getattr(WorkerType, f'CELERY_{worker_type.upper()}'),
                scaling_policy=ScalingPolicy(
                    min_replicas=int(os.getenv(f'MIN_CELERY_{worker_type.upper()}_WORKERS', 1)),
                    max_replicas=int(os.getenv(f'MAX_CELERY_{worker_type.upper()}_WORKERS', 10)),
                ),
                environment={'WORKER_TYPE': worker_type}
            )
            
    async def start(self):
        """Start the orchestrator"""
        self.running = True
        logger.info("Starting orchestrator...")
        
        # Start monitoring tasks
        self.tasks.append(asyncio.create_task(self.health_check_loop()))
        self.tasks.append(asyncio.create_task(self.scaling_loop()))
        self.tasks.append(asyncio.create_task(self.metrics_collection_loop()))
        self.tasks.append(asyncio.create_task(self.recovery_loop()))
        self.tasks.append(asyncio.create_task(self.optimization_loop()))
        
        # Ensure minimum replicas
        await self.ensure_minimum_replicas()
        
        logger.info("Orchestrator started successfully")
        
    async def stop(self):
        """Stop the orchestrator"""
        self.running = False
        logger.info("Stopping orchestrator...")
        
        # Cancel all tasks
        for task in self.tasks:
            task.cancel()
            
        # Wait for tasks to complete
        await asyncio.gather(*self.tasks, return_exceptions=True)
        
        # Cleanup resources
        if self.redis_client:
            self.redis_client.close()
            await self.redis_client.wait_closed()
            
        if self.async_docker:
            await self.async_docker.close()
            
        logger.info("Orchestrator stopped")
        
    async def health_check_loop(self):
        """Main health check loop"""
        while self.running:
            try:
                with health_check_duration.time():
                    await self.check_all_services_health()
                    
                await asyncio.sleep(int(os.getenv('HEALTH_CHECK_INTERVAL', 30)))
                
            except Exception as e:
                logger.error(f"Health check error: {e}")
                await asyncio.sleep(10)
                
    async def check_all_services_health(self):
        """Check health of all services"""
        containers = await self.async_docker.containers.list()
        
        for service_name, config in self.services.items():
            service_containers = [
                c for c in containers 
                if c._container.get('Labels', {}).get('com.docker.compose.service') == service_name
            ]
            
            if not service_containers:
                self.health_status[service_name] = ServiceHealth(
                    service_name=service_name,
                    status=ServiceStatus.UNHEALTHY,
                    last_check=datetime.now(),
                    error_message="No containers found"
                )
                continue
                
            # Check each container
            healthy_count = 0
            total_count = len(service_containers)
            
            for container in service_containers:
                is_healthy = await self.check_container_health(container, config)
                if is_healthy:
                    healthy_count += 1
                    
            # Determine service status
            if healthy_count == total_count:
                status = ServiceStatus.HEALTHY
            elif healthy_count > 0:
                status = ServiceStatus.DEGRADED
            else:
                status = ServiceStatus.UNHEALTHY
                
            # Update health status
            if service_name not in self.health_status:
                self.health_status[service_name] = ServiceHealth(
                    service_name=service_name,
                    status=status,
                    last_check=datetime.now()
                )
            else:
                self.health_status[service_name].status = status
                self.health_status[service_name].last_check = datetime.now()
                
                if status != ServiceStatus.HEALTHY:
                    self.health_status[service_name].consecutive_failures += 1
                else:
                    self.health_status[service_name].consecutive_failures = 0
                    
            # Update metrics
            worker_count_gauge.labels(
                type=config.worker_type.value,
                status=status.value
            ).set(healthy_count)
            
    async def check_container_health(self, container, config: ServiceConfig) -> bool:
        """Check health of a single container"""
        try:
            # Get container stats
            stats = await container.stats(stream=False)
            
            # Check if container is running
            info = await container.show()
            if info['State']['Status'] != 'running':
                return False
                
            # Check resource usage
            cpu_usage = self.calculate_cpu_usage(stats)
            memory_usage = self.calculate_memory_usage(stats)
            
            # Store metrics
            resource_usage.labels(resource='cpu', service=config.name).set(cpu_usage)
            resource_usage.labels(resource='memory', service=config.name).set(memory_usage)
            
            # Check health endpoint if configured
            if config.health_check_endpoint:
                # Get container IP
                networks = info['NetworkSettings']['Networks']
                if networks:
                    ip = list(networks.values())[0]['IPAddress']
                    health_url = f"http://{ip}:8000{config.health_check_endpoint}"
                    
                    # Make health check request
                    import aiohttp
                    async with aiohttp.ClientSession() as session:
                        async with session.get(health_url, timeout=5) as response:
                            return response.status == 200
                            
            return True
            
        except Exception as e:
            logger.error(f"Container health check failed: {e}")
            return False
            
    def calculate_cpu_usage(self, stats: Dict) -> float:
        """Calculate CPU usage percentage"""
        try:
            cpu_delta = stats['cpu_stats']['cpu_usage']['total_usage'] - \
                       stats['precpu_stats']['cpu_usage']['total_usage']
            system_delta = stats['cpu_stats']['system_cpu_usage'] - \
                          stats['precpu_stats']['system_cpu_usage']
            
            if system_delta > 0 and cpu_delta > 0:
                cpu_count = len(stats['cpu_stats']['cpu_usage'].get('percpu_usage', [1]))
                cpu_percent = (cpu_delta / system_delta) * cpu_count * 100.0
                return round(cpu_percent, 2)
                
        except (KeyError, ZeroDivisionError):
            pass
            
        return 0.0
        
    def calculate_memory_usage(self, stats: Dict) -> float:
        """Calculate memory usage in MB"""
        try:
            memory_bytes = stats['memory_stats']['usage']
            return round(memory_bytes / 1024 / 1024, 2)
        except KeyError:
            return 0.0
            
    async def scaling_loop(self):
        """Main scaling loop"""
        while self.running:
            try:
                await self.evaluate_scaling_decisions()
                await asyncio.sleep(int(os.getenv('SCALING_INTERVAL', 60)))
                
            except Exception as e:
                logger.error(f"Scaling loop error: {e}")
                await asyncio.sleep(30)
                
    async def evaluate_scaling_decisions(self):
        """Evaluate and execute scaling decisions"""
        for service_name, config in self.services.items():
            try:
                # Check cooldown
                if service_name in self.last_scale_time:
                    elapsed = (datetime.now() - self.last_scale_time[service_name]).total_seconds()
                    if elapsed < config.scaling_policy.cooldown_seconds:
                        continue
                        
                # Get current replica count
                current_replicas = await self.get_replica_count(service_name)
                
                # Get metrics
                metrics = await self.get_service_metrics(service_name)
                
                # Determine scaling action
                target_replicas = self.calculate_target_replicas(
                    current_replicas,
                    metrics,
                    config.scaling_policy
                )
                
                # Execute scaling if needed
                if target_replicas != current_replicas:
                    await self.scale_service(service_name, target_replicas)
                    
            except Exception as e:
                logger.error(f"Scaling evaluation error for {service_name}: {e}")
                
    async def get_replica_count(self, service_name: str) -> int:
        """Get current replica count for a service"""
        containers = await self.async_docker.containers.list(
            filters={'label': f'com.docker.compose.service={service_name}'}
        )
        return len([c for c in containers if c._container['State'] == 'running'])
        
    async def get_service_metrics(self, service_name: str) -> Dict[str, float]:
        """Get aggregated metrics for a service"""
        metrics = {
            'cpu_usage': 0.0,
            'memory_usage': 0.0,
            'request_rate': 0.0,
            'error_rate': 0.0,
            'response_time': 0.0
        }
        
        # Get container metrics
        containers = await self.async_docker.containers.list(
            filters={'label': f'com.docker.compose.service={service_name}'}
        )
        
        if not containers:
            return metrics
            
        total_cpu = 0.0
        total_memory = 0.0
        
        for container in containers:
            try:
                stats = await container.stats(stream=False)
                total_cpu += self.calculate_cpu_usage(stats)
                total_memory += self.calculate_memory_usage(stats)
            except:
                pass
                
        metrics['cpu_usage'] = total_cpu / len(containers) if containers else 0
        metrics['memory_usage'] = total_memory / len(containers) if containers else 0
        
        # Get application metrics from Redis
        try:
            app_metrics = await self.redis_client.hgetall(f'metrics:{service_name}')
            if app_metrics:
                metrics['request_rate'] = float(app_metrics.get(b'request_rate', 0))
                metrics['error_rate'] = float(app_metrics.get(b'error_rate', 0))
                metrics['response_time'] = float(app_metrics.get(b'response_time', 0))
        except:
            pass
            
        return metrics
        
    def calculate_target_replicas(
        self,
        current: int,
        metrics: Dict[str, float],
        policy: ScalingPolicy
    ) -> int:
        """Calculate target replica count based on metrics and policy"""
        # Check CPU threshold
        if metrics['cpu_usage'] > policy.scale_up_threshold:
            target = current + policy.scale_up_increment
            return min(target, policy.max_replicas)
            
        if metrics['cpu_usage'] < policy.scale_down_threshold:
            target = current - policy.scale_down_increment
            return max(target, policy.min_replicas)
            
        # Check memory threshold
        memory_percent = (metrics['memory_usage'] / 2048) * 100  # Assuming 2GB limit
        if memory_percent > policy.target_memory_utilization:
            target = current + policy.scale_up_increment
            return min(target, policy.max_replicas)
            
        # Check request rate (if applicable)
        if metrics['request_rate'] > 100:  # High request rate
            target = current + 1
            return min(target, policy.max_replicas)
            
        return current
        
    async def scale_service(self, service_name: str, target_replicas: int):
        """Scale a service to target replica count"""
        current_replicas = await self.get_replica_count(service_name)
        
        if target_replicas == current_replicas:
            return
            
        direction = "up" if target_replicas > current_replicas else "down"
        logger.info(f"Scaling {service_name} {direction}: {current_replicas} -> {target_replicas}")
        
        try:
            # Use docker-compose scale command
            import subprocess
            result = subprocess.run(
                ['docker-compose', '-f', 'docker-compose.production-multiworker.yml',
                 'up', '-d', '--scale', f'{service_name}={target_replicas}', '--no-recreate'],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                # Record scaling event
                self.last_scale_time[service_name] = datetime.now()
                self.scaling_history[service_name].append({
                    'timestamp': datetime.now().isoformat(),
                    'from': current_replicas,
                    'to': target_replicas,
                    'reason': 'auto-scaling'
                })
                
                # Update metrics
                scaling_events_counter.labels(
                    type=self.services[service_name].worker_type.value,
                    direction=direction
                ).inc()
                
                # Store in Redis
                await self.redis_client.hset(
                    f'scaling:{service_name}',
                    'last_scale',
                    datetime.now().isoformat()
                )
                
                logger.info(f"Successfully scaled {service_name} to {target_replicas} replicas")
            else:
                logger.error(f"Failed to scale {service_name}: {result.stderr}")
                
        except Exception as e:
            logger.error(f"Scaling error for {service_name}: {e}")
            
    async def recovery_loop(self):
        """Recovery loop for unhealthy services"""
        while self.running:
            try:
                await self.recover_unhealthy_services()
                await asyncio.sleep(60)
                
            except Exception as e:
                logger.error(f"Recovery loop error: {e}")
                await asyncio.sleep(30)
                
    async def recover_unhealthy_services(self):
        """Attempt to recover unhealthy services"""
        for service_name, health in self.health_status.items():
            if health.status == ServiceStatus.UNHEALTHY and \
               health.consecutive_failures >= 3:
                logger.warning(f"Attempting to recover {service_name}")
                
                try:
                    # Restart unhealthy containers
                    containers = await self.async_docker.containers.list(
                        filters={'label': f'com.docker.compose.service={service_name}'}
                    )
                    
                    for container in containers:
                        info = await container.show()
                        if info['State']['Status'] != 'running':
                            logger.info(f"Restarting container {info['Name']}")
                            await container.restart()
                            container_restarts.labels(service=service_name).inc()
                            
                    # If no containers, ensure minimum replicas
                    if not containers:
                        config = self.services.get(service_name)
                        if config:
                            await self.scale_service(
                                service_name,
                                config.scaling_policy.min_replicas
                            )
                            
                except Exception as e:
                    logger.error(f"Recovery failed for {service_name}: {e}")
                    
    async def optimization_loop(self):
        """Optimization loop for resource efficiency"""
        while self.running:
            try:
                await self.optimize_resource_allocation()
                await asyncio.sleep(300)  # Run every 5 minutes
                
            except Exception as e:
                logger.error(f"Optimization loop error: {e}")
                await asyncio.sleep(60)
                
    async def optimize_resource_allocation(self):
        """Optimize resource allocation across services"""
        # Get system resources
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        
        logger.info(f"System resources - CPU: {cpu_percent}%, Memory: {memory.percent}%")
        
        # Rebalance if system is under pressure
        if cpu_percent > 85 or memory.percent > 85:
            logger.warning("System under pressure, optimizing resource allocation")
            
            # Find services that can be scaled down
            for service_name, config in self.services.items():
                metrics = await self.get_service_metrics(service_name)
                current_replicas = await self.get_replica_count(service_name)
                
                # Scale down low-utilization services
                if metrics['cpu_usage'] < 20 and current_replicas > config.scaling_policy.min_replicas:
                    target = max(
                        current_replicas - 1,
                        config.scaling_policy.min_replicas
                    )
                    await self.scale_service(service_name, target)
                    
    async def metrics_collection_loop(self):
        """Collect and store metrics"""
        while self.running:
            try:
                await self.collect_metrics()
                await asyncio.sleep(int(os.getenv('METRICS_COLLECTION_INTERVAL', 10)))
                
            except Exception as e:
                logger.error(f"Metrics collection error: {e}")
                await asyncio.sleep(10)
                
    async def collect_metrics(self):
        """Collect metrics from all services"""
        timestamp = datetime.now().isoformat()
        
        for service_name in self.services:
            metrics = await self.get_service_metrics(service_name)
            
            # Store in Redis
            await self.redis_client.hset(
                f'metrics:{service_name}:latest',
                mapping={
                    'timestamp': timestamp,
                    'cpu_usage': metrics['cpu_usage'],
                    'memory_usage': metrics['memory_usage'],
                    'request_rate': metrics['request_rate'],
                    'error_rate': metrics['error_rate'],
                    'response_time': metrics['response_time']
                }
            )
            
            # Store time series data
            await self.redis_client.zadd(
                f'metrics:{service_name}:history',
                {json.dumps(metrics): time.time()}
            )
            
            # Trim old data (keep last 24 hours)
            await self.redis_client.zremrangebyscore(
                f'metrics:{service_name}:history',
                0,
                time.time() - 86400
            )
            
    async def ensure_minimum_replicas(self):
        """Ensure all services have minimum replicas running"""
        for service_name, config in self.services.items():
            current_replicas = await self.get_replica_count(service_name)
            
            if current_replicas < config.scaling_policy.min_replicas:
                logger.info(f"Ensuring minimum replicas for {service_name}")
                await self.scale_service(
                    service_name,
                    config.scaling_policy.min_replicas
                )
                
    async def get_status(self) -> Dict:
        """Get orchestrator status"""
        return {
            'status': 'running' if self.running else 'stopped',
            'services': {
                name: {
                    'replicas': await self.get_replica_count(name),
                    'health': self.health_status.get(name, ServiceHealth(
                        service_name=name,
                        status=ServiceStatus.UNKNOWN,
                        last_check=datetime.now()
                    )).status.value,
                    'scaling_policy': asdict(config.scaling_policy),
                    'last_scale': self.last_scale_time.get(name, '').isoformat() if name in self.last_scale_time else None
                }
                for name, config in self.services.items()
            },
            'system': {
                'cpu_percent': psutil.cpu_percent(),
                'memory_percent': psutil.virtual_memory().percent,
                'disk_percent': psutil.disk_usage('/').percent
            }
        }

async def main():
    """Main entry point"""
    orchestrator = ContainerOrchestrator()
    
    # Initialize
    await orchestrator.initialize()
    
    # Setup signal handlers
    def signal_handler(sig, frame):
        logger.info(f"Received signal {sig}")
        asyncio.create_task(orchestrator.stop())
        
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        # Start orchestrator
        await orchestrator.start()
        
        # Keep running
        while orchestrator.running:
            await asyncio.sleep(60)
            
            # Log status periodically
            status = await orchestrator.get_status()
            logger.info(f"Orchestrator status: {json.dumps(status, indent=2)}")
            
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        await orchestrator.stop()

if __name__ == '__main__':
    asyncio.run(main())