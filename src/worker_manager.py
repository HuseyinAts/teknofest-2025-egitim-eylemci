"""
Production-Ready Worker Manager
TEKNOFEST 2025 - Multi-Worker Deployment
Advanced process management with health checks, auto-recovery, and load balancing
"""

import os
import sys
import time
import signal
import psutil
import asyncio
import threading
import multiprocessing
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque
import json
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class WorkerMetrics:
    """Metrics for a single worker"""
    pid: int
    cpu_percent: float = 0.0
    memory_mb: float = 0.0
    request_count: int = 0
    error_count: int = 0
    response_time_ms: float = 0.0
    thread_count: int = 0
    file_descriptors: int = 0
    connections: int = 0
    status: str = "running"
    start_time: datetime = field(default_factory=datetime.now)
    last_health_check: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return {
            'pid': self.pid,
            'cpu_percent': round(self.cpu_percent, 2),
            'memory_mb': round(self.memory_mb, 2),
            'request_count': self.request_count,
            'error_count': self.error_count,
            'response_time_ms': round(self.response_time_ms, 2),
            'thread_count': self.thread_count,
            'file_descriptors': self.file_descriptors,
            'connections': self.connections,
            'status': self.status,
            'uptime_seconds': (datetime.now() - self.start_time).total_seconds(),
            'last_health_check': self.last_health_check.isoformat()
        }

@dataclass
class WorkerConfig:
    """Configuration for worker management"""
    min_workers: int = 2
    max_workers: int = 16
    memory_limit_mb: int = 2048
    cpu_threshold: float = 90.0
    memory_threshold: float = 85.0
    health_check_interval: int = 30
    unhealthy_restart_threshold: int = 3
    graceful_shutdown_timeout: int = 30
    scale_up_threshold: float = 80.0
    scale_down_threshold: float = 30.0
    scale_cooldown_seconds: int = 60
    request_timeout_seconds: int = 30
    max_requests_per_worker: int = 1000
    enable_auto_scale: bool = True
    enable_health_checks: bool = True
    enable_load_balancing: bool = True
    
class WorkerPool:
    """Manages a pool of worker processes"""
    
    def __init__(self, config: WorkerConfig):
        self.config = config
        self.workers: Dict[int, multiprocessing.Process] = {}
        self.worker_metrics: Dict[int, WorkerMetrics] = {}
        self.unhealthy_counts: Dict[int, int] = {}
        self.last_scale_time = datetime.now()
        self.request_queue = deque(maxlen=1000)
        self.shutdown_event = threading.Event()
        self.monitoring_thread: Optional[threading.Thread] = None
        self.load_balancer: Optional['LoadBalancer'] = None
        
    def start(self, initial_workers: Optional[int] = None):
        """Start the worker pool"""
        logger.info("Starting worker pool...")
        
        # Start initial workers
        num_workers = initial_workers or self.config.min_workers
        for _ in range(num_workers):
            self._spawn_worker()
            
        # Start monitoring
        if self.config.enable_health_checks:
            self._start_monitoring()
            
        # Initialize load balancer
        if self.config.enable_load_balancing:
            self.load_balancer = LoadBalancer(self)
            
        logger.info(f"Worker pool started with {num_workers} workers")
        
    def stop(self):
        """Stop all workers gracefully"""
        logger.info("Stopping worker pool...")
        
        # Signal shutdown
        self.shutdown_event.set()
        
        # Stop monitoring
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
            
        # Graceful shutdown of workers
        for pid in list(self.workers.keys()):
            self._stop_worker(pid, graceful=True)
            
        logger.info("Worker pool stopped")
        
    def _spawn_worker(self) -> Optional[int]:
        """Spawn a new worker process"""
        try:
            # Create worker process
            worker = multiprocessing.Process(target=self._worker_main)
            worker.start()
            
            # Track worker
            self.workers[worker.pid] = worker
            self.worker_metrics[worker.pid] = WorkerMetrics(pid=worker.pid)
            
            logger.info(f"Spawned worker {worker.pid}")
            return worker.pid
            
        except Exception as e:
            logger.error(f"Failed to spawn worker: {e}")
            return None
            
    def _stop_worker(self, pid: int, graceful: bool = True):
        """Stop a specific worker"""
        if pid not in self.workers:
            return
            
        worker = self.workers[pid]
        
        if graceful:
            # Try graceful shutdown
            try:
                os.kill(pid, signal.SIGTERM)
                worker.join(timeout=self.config.graceful_shutdown_timeout)
            except:
                pass
                
        # Force kill if still alive
        if worker.is_alive():
            try:
                os.kill(pid, signal.SIGKILL)
                worker.join(timeout=5)
            except:
                pass
                
        # Cleanup
        del self.workers[pid]
        del self.worker_metrics[pid]
        if pid in self.unhealthy_counts:
            del self.unhealthy_counts[pid]
            
        logger.info(f"Stopped worker {pid}")
        
    def _worker_main(self):
        """Main function for worker process"""
        # Set process title
        try:
            import setproctitle
            setproctitle.setproctitle(f"teknofest-worker-{os.getpid()}")
        except ImportError:
            pass
            
        # Setup signal handlers
        signal.signal(signal.SIGTERM, self._worker_signal_handler)
        signal.signal(signal.SIGINT, self._worker_signal_handler)
        
        # Worker main loop
        logger.info(f"Worker {os.getpid()} started")
        
        # Import and run the actual application
        try:
            from src.app import app
            import uvicorn
            
            uvicorn.run(
                app,
                host="127.0.0.1",
                port=8000 + os.getpid() % 100,  # Unique port per worker
                log_level="info",
                access_log=False  # Reduce logging overhead
            )
        except Exception as e:
            logger.error(f"Worker {os.getpid()} error: {e}")
            
    def _worker_signal_handler(self, signum, frame):
        """Handle signals in worker process"""
        logger.info(f"Worker {os.getpid()} received signal {signum}")
        sys.exit(0)
        
    def _start_monitoring(self):
        """Start the monitoring thread"""
        self.monitoring_thread = threading.Thread(target=self._monitor_loop)
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()
        logger.info("Health monitoring started")
        
    def _monitor_loop(self):
        """Main monitoring loop"""
        while not self.shutdown_event.is_set():
            try:
                # Check worker health
                self._check_worker_health()
                
                # Auto-scale if enabled
                if self.config.enable_auto_scale:
                    self._auto_scale()
                    
                # Sleep until next check
                self.shutdown_event.wait(self.config.health_check_interval)
                
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                
    def _check_worker_health(self):
        """Check health of all workers"""
        for pid in list(self.workers.keys()):
            try:
                # Get process info
                process = psutil.Process(pid)
                
                # Update metrics
                metrics = self.worker_metrics[pid]
                metrics.cpu_percent = process.cpu_percent(interval=0.1)
                metrics.memory_mb = process.memory_info().rss / 1024 / 1024
                metrics.thread_count = process.num_threads()
                metrics.file_descriptors = len(process.open_files())
                metrics.connections = len(process.connections())
                metrics.status = process.status()
                metrics.last_health_check = datetime.now()
                
                # Check health criteria
                is_unhealthy = False
                reasons = []
                
                if metrics.cpu_percent > self.config.cpu_threshold:
                    is_unhealthy = True
                    reasons.append(f"High CPU: {metrics.cpu_percent:.1f}%")
                    
                if metrics.memory_mb > self.config.memory_limit_mb:
                    is_unhealthy = True
                    reasons.append(f"High memory: {metrics.memory_mb:.0f}MB")
                    
                if metrics.status != psutil.STATUS_RUNNING:
                    is_unhealthy = True
                    reasons.append(f"Bad status: {metrics.status}")
                    
                if metrics.request_count > self.config.max_requests_per_worker:
                    is_unhealthy = True
                    reasons.append(f"Too many requests: {metrics.request_count}")
                    
                # Handle unhealthy worker
                if is_unhealthy:
                    self._handle_unhealthy_worker(pid, reasons)
                else:
                    # Reset unhealthy count
                    if pid in self.unhealthy_counts:
                        del self.unhealthy_counts[pid]
                        
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                logger.warning(f"Worker {pid} not accessible, removing")
                self._stop_worker(pid, graceful=False)
                
    def _handle_unhealthy_worker(self, pid: int, reasons: List[str]):
        """Handle an unhealthy worker"""
        # Increment unhealthy count
        self.unhealthy_counts[pid] = self.unhealthy_counts.get(pid, 0) + 1
        
        logger.warning(f"Worker {pid} unhealthy ({self.unhealthy_counts[pid]}): {', '.join(reasons)}")
        
        # Restart if threshold exceeded
        if self.unhealthy_counts[pid] >= self.config.unhealthy_restart_threshold:
            logger.error(f"Restarting unhealthy worker {pid}")
            self._stop_worker(pid, graceful=True)
            self._spawn_worker()
            
    def _auto_scale(self):
        """Auto-scale workers based on metrics"""
        # Check cooldown
        if (datetime.now() - self.last_scale_time).total_seconds() < self.config.scale_cooldown_seconds:
            return
            
        current_workers = len(self.workers)
        
        # Calculate average metrics
        if not self.worker_metrics:
            return
            
        avg_cpu = sum(m.cpu_percent for m in self.worker_metrics.values()) / len(self.worker_metrics)
        avg_memory = sum(m.memory_mb for m in self.worker_metrics.values()) / len(self.worker_metrics)
        
        # Get system metrics
        system_cpu = psutil.cpu_percent(interval=1)
        system_memory = psutil.virtual_memory().percent
        
        # Scale up
        if (system_cpu > self.config.scale_up_threshold or 
            system_memory > self.config.scale_up_threshold or
            avg_cpu > self.config.scale_up_threshold):
            
            if current_workers < self.config.max_workers:
                # Calculate scale factor
                scale_factor = 1
                if system_cpu > 90 or system_memory > 90:
                    scale_factor = 2
                    
                new_workers = min(current_workers + scale_factor, self.config.max_workers)
                
                logger.info(f"Scaling up from {current_workers} to {new_workers} workers")
                logger.info(f"Metrics: CPU={system_cpu:.1f}%, Memory={system_memory:.1f}%")
                
                for _ in range(new_workers - current_workers):
                    self._spawn_worker()
                    
                self.last_scale_time = datetime.now()
                
        # Scale down
        elif (system_cpu < self.config.scale_down_threshold and 
              system_memory < self.config.scale_down_threshold and
              avg_cpu < self.config.scale_down_threshold):
            
            if current_workers > self.config.min_workers:
                new_workers = max(current_workers - 1, self.config.min_workers)
                
                logger.info(f"Scaling down from {current_workers} to {new_workers} workers")
                logger.info(f"Metrics: CPU={system_cpu:.1f}%, Memory={system_memory:.1f}%")
                
                # Stop least busy workers
                workers_by_cpu = sorted(
                    self.worker_metrics.items(),
                    key=lambda x: x[1].cpu_percent
                )
                
                for pid, _ in workers_by_cpu[:current_workers - new_workers]:
                    self._stop_worker(pid, graceful=True)
                    
                self.last_scale_time = datetime.now()
                
    def get_metrics(self) -> Dict:
        """Get current pool metrics"""
        return {
            'workers': len(self.workers),
            'healthy_workers': len(self.workers) - len(self.unhealthy_counts),
            'unhealthy_workers': len(self.unhealthy_counts),
            'worker_metrics': {
                pid: metrics.to_dict() 
                for pid, metrics in self.worker_metrics.items()
            },
            'system_metrics': {
                'cpu_percent': psutil.cpu_percent(interval=0.1),
                'memory_percent': psutil.virtual_memory().percent,
                'load_average': os.getloadavg(),
                'disk_usage': psutil.disk_usage('/').percent
            }
        }

class LoadBalancer:
    """Load balancer for distributing requests across workers"""
    
    def __init__(self, worker_pool: WorkerPool):
        self.worker_pool = worker_pool
        self.current_index = 0
        self.algorithm = os.getenv('LOAD_BALANCE_ALGORITHM', 'round_robin')
        
    def get_next_worker(self) -> Optional[int]:
        """Get the next worker to handle a request"""
        if not self.worker_pool.workers:
            return None
            
        if self.algorithm == 'round_robin':
            return self._round_robin()
        elif self.algorithm == 'least_connections':
            return self._least_connections()
        elif self.algorithm == 'least_cpu':
            return self._least_cpu()
        else:
            return self._round_robin()
            
    def _round_robin(self) -> int:
        """Round-robin load balancing"""
        workers = list(self.worker_pool.workers.keys())
        if not workers:
            return None
            
        worker = workers[self.current_index % len(workers)]
        self.current_index += 1
        return worker
        
    def _least_connections(self) -> int:
        """Least connections load balancing"""
        min_connections = float('inf')
        selected_worker = None
        
        for pid, metrics in self.worker_pool.worker_metrics.items():
            if metrics.connections < min_connections:
                min_connections = metrics.connections
                selected_worker = pid
                
        return selected_worker
        
    def _least_cpu(self) -> int:
        """Least CPU usage load balancing"""
        min_cpu = float('inf')
        selected_worker = None
        
        for pid, metrics in self.worker_pool.worker_metrics.items():
            if metrics.cpu_percent < min_cpu:
                min_cpu = metrics.cpu_percent
                selected_worker = pid
                
        return selected_worker

class WorkerManager:
    """Main worker manager interface"""
    
    def __init__(self, config_path: Optional[str] = None):
        # Load configuration
        self.config = self._load_config(config_path)
        self.worker_pool = WorkerPool(self.config)
        
    def _load_config(self, config_path: Optional[str]) -> WorkerConfig:
        """Load configuration from file or environment"""
        config = WorkerConfig()
        
        # Load from file if provided
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                config_data = json.load(f)
                for key, value in config_data.items():
                    if hasattr(config, key):
                        setattr(config, key, value)
                        
        # Override with environment variables
        config.min_workers = int(os.getenv('MIN_WORKERS', config.min_workers))
        config.max_workers = int(os.getenv('MAX_WORKERS', config.max_workers))
        config.memory_limit_mb = int(os.getenv('WORKER_MEMORY_LIMIT', config.memory_limit_mb))
        config.enable_auto_scale = os.getenv('AUTO_SCALE_ENABLED', 'true').lower() == 'true'
        config.enable_health_checks = os.getenv('HEALTH_CHECKS_ENABLED', 'true').lower() == 'true'
        
        return config
        
    def start(self):
        """Start the worker manager"""
        logger.info("Starting Worker Manager...")
        
        # Calculate initial workers
        cpu_count = multiprocessing.cpu_count()
        env = os.getenv('APP_ENV', 'development')
        
        if env == 'production':
            initial_workers = min((cpu_count * 2) + 1, self.config.max_workers)
        elif env == 'staging':
            initial_workers = min(cpu_count + 1, 8)
        else:
            initial_workers = min(2, cpu_count)
            
        # Start worker pool
        self.worker_pool.start(initial_workers)
        
        logger.info(f"Worker Manager started in {env} mode with {initial_workers} workers")
        
    def stop(self):
        """Stop the worker manager"""
        logger.info("Stopping Worker Manager...")
        self.worker_pool.stop()
        logger.info("Worker Manager stopped")
        
    def get_status(self) -> Dict:
        """Get current status and metrics"""
        return {
            'status': 'running',
            'config': {
                'min_workers': self.config.min_workers,
                'max_workers': self.config.max_workers,
                'auto_scale': self.config.enable_auto_scale,
                'health_checks': self.config.enable_health_checks
            },
            'metrics': self.worker_pool.get_metrics()
        }
        
    def scale(self, num_workers: int):
        """Manually scale to a specific number of workers"""
        current = len(self.worker_pool.workers)
        
        if num_workers > current:
            # Scale up
            for _ in range(num_workers - current):
                self.worker_pool._spawn_worker()
        elif num_workers < current:
            # Scale down
            workers_to_stop = current - num_workers
            for pid in list(self.worker_pool.workers.keys())[:workers_to_stop]:
                self.worker_pool._stop_worker(pid, graceful=True)
                
        logger.info(f"Scaled from {current} to {num_workers} workers")

def main():
    """Main entry point for standalone execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description='TEKNOFEST Worker Manager')
    parser.add_argument('--config', help='Path to configuration file')
    parser.add_argument('--workers', type=int, help='Initial number of workers')
    parser.add_argument('--daemon', action='store_true', help='Run as daemon')
    
    args = parser.parse_args()
    
    # Create and start manager
    manager = WorkerManager(args.config)
    
    # Setup signal handlers
    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}")
        manager.stop()
        sys.exit(0)
        
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        # Start manager
        manager.start()
        
        # Keep running
        while True:
            time.sleep(60)
            
            # Print status periodically
            status = manager.get_status()
            logger.info(f"Status: {status['metrics']['workers']} workers, "
                       f"{status['metrics']['healthy_workers']} healthy")
                       
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        manager.stop()

if __name__ == '__main__':
    main()