# -*- coding: utf-8 -*-
"""
Production-Ready Gunicorn Configuration
TEKNOFEST 2025 - Eğitim Teknolojileri
Multi-Worker Deployment with Auto-Scaling Support
"""

import multiprocessing
import os
from pathlib import Path

# ==========================================
# CORE CONFIGURATION
# ==========================================

# Server socket binding
bind = f"{os.getenv('API_HOST', '0.0.0.0')}:{os.getenv('API_PORT', '8000')}"
backlog = int(os.getenv('GUNICORN_BACKLOG', 2048))

# ==========================================
# WORKER CONFIGURATION
# ==========================================

# Calculate optimal worker count
def get_worker_count():
    """Calculate optimal worker count based on CPU and memory"""
    cpu_count = multiprocessing.cpu_count()
    
    # Production formula: (2 x CPU cores) + 1
    # But cap at reasonable limits based on environment
    env = os.getenv('APP_ENV', 'development')
    
    if env == 'production':
        # Production: more workers, but cap at 16
        return min((cpu_count * 2) + 1, 16)
    elif env == 'staging':
        # Staging: moderate workers
        return min(cpu_count + 1, 8)
    else:
        # Development: minimal workers
        return min(2, cpu_count)

workers = int(os.getenv('API_WORKERS', get_worker_count()))

# Use uvicorn worker for async FastAPI support
worker_class = 'uvicorn.workers.UvicornWorker'

# Worker connections for async workers
worker_connections = int(os.getenv('WORKER_CONNECTIONS', 1000))

# Worker lifecycle management
max_requests = int(os.getenv('MAX_REQUESTS', 1000))
max_requests_jitter = int(os.getenv('MAX_REQUESTS_JITTER', 100))

# Timeouts
timeout = int(os.getenv('API_TIMEOUT', 30))
graceful_timeout = int(os.getenv('GRACEFUL_TIMEOUT', 30))
keepalive = int(os.getenv('KEEPALIVE', 5))

# Thread pool for sync operations
threads = int(os.getenv('THREADS', 4))

# ==========================================
# LOGGING CONFIGURATION
# ==========================================

# Dynamic log paths based on environment
log_dir = Path(os.getenv('LOG_DIR', '/var/log/teknofest'))
log_dir.mkdir(parents=True, exist_ok=True)

# Log files
accesslog = os.getenv('ACCESS_LOG', str(log_dir / 'access.log'))
errorlog = os.getenv('ERROR_LOG', str(log_dir / 'error.log'))

# Use '-' for stdout/stderr in containers
if os.getenv('LOG_TO_STDOUT', 'false').lower() == 'true':
    accesslog = '-'
    errorlog = '-'

# Log level
loglevel = os.getenv('LOG_LEVEL', 'info').lower()

# Detailed access log format with response time
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)s %(p)s'

# Capture stdout/stderr from workers
capture_output = os.getenv('CAPTURE_OUTPUT', 'true').lower() == 'true'
enable_stdio_inheritance = True

# ==========================================
# PROCESS MANAGEMENT
# ==========================================

# Process naming for monitoring
proc_name = 'teknofest_api'

# Daemonization (usually false in containers)
daemon = os.getenv('DAEMON', 'false').lower() == 'true'

# PID file for process management
pidfile = os.getenv('PID_FILE', '/var/run/teknofest.pid')

# User/Group (for security)
user = os.getenv('RUN_AS_USER', None)
group = os.getenv('RUN_AS_GROUP', None)

# Working directory
chdir = os.getenv('WORK_DIR', '/app')

# Temporary upload directory
tmp_upload_dir = os.getenv('UPLOAD_DIR', '/tmp/uploads')

# ==========================================
# PERFORMANCE TUNING
# ==========================================

# Preload application for better memory usage
preload_app = os.getenv('PRELOAD_APP', 'true').lower() == 'true'

# Disable request logging for health checks
def health_check_filter(record):
    """Filter out health check requests from logs"""
    return 'GET /health' not in record.getMessage()

# Request limits
limit_request_line = int(os.getenv('LIMIT_REQUEST_LINE', 4094))
limit_request_fields = int(os.getenv('LIMIT_REQUEST_FIELDS', 100))
limit_request_field_size = int(os.getenv('LIMIT_REQUEST_FIELD_SIZE', 8190))

# ==========================================
# SSL/TLS CONFIGURATION
# ==========================================

if os.getenv('SSL_ENABLED', 'false').lower() == 'true':
    keyfile = os.getenv('SSL_KEY', '/app/certs/server.key')
    certfile = os.getenv('SSL_CERT', '/app/certs/server.crt')
    ssl_version = 5  # TLS 1.2+
    cert_reqs = 0  # No client cert required
    ca_certs = os.getenv('SSL_CA_CERT', None)
    ciphers = 'TLSv1.2:!aNULL:!eNULL:!EXPORT:!DES:!MD5:!PSK:!RC4'
    do_handshake_on_connect = False

# ==========================================
# MONITORING & METRICS
# ==========================================

# StatsD configuration for metrics
if os.getenv('STATSD_ENABLED', 'false').lower() == 'true':
    statsd_host = os.getenv('STATSD_HOST', 'localhost:8125')
    statsd_prefix = os.getenv('STATSD_PREFIX', 'teknofest')

# Prometheus metrics endpoint (handled by app)
prometheus_multiproc_dir = os.getenv('PROMETHEUS_MULTIPROC_DIR', '/tmp/prometheus')

# ==========================================
# LIFECYCLE HOOKS
# ==========================================

def on_starting(server):
    """Called just before the master process is initialized"""
    server.log.info("=" * 60)
    server.log.info("TEKNOFEST 2025 API Server Starting")
    server.log.info("=" * 60)
    server.log.info(f"Master PID: {os.getpid()}")
    server.log.info(f"Workers: {workers}")
    server.log.info(f"Worker Class: {worker_class}")
    server.log.info(f"Bind: {bind}")
    server.log.info(f"Environment: {os.getenv('APP_ENV', 'development')}")
    
    # Create necessary directories
    os.makedirs('/tmp/uploads', exist_ok=True)
    os.makedirs('/var/log/teknofest', exist_ok=True)
    
    # Initialize Prometheus multiprocess mode
    if os.getenv('PROMETHEUS_ENABLED', 'false').lower() == 'true':
        from prometheus_client import multiprocess
        multiprocess.MultiProcessCollector(registry=None)
    
    # Initialize worker health monitor
    global health_monitor
    if os.getenv('WORKER_HEALTH_MONITORING', 'true').lower() == 'true':
        health_monitor = WorkerHealthMonitor(server)
        server.log.info("Worker health monitoring initialized")

def on_reload(server):
    """Called to recycle workers during a reload via SIGHUP"""
    server.log.info("Reloading TEKNOFEST API server...")
    server.log.info(f"Current workers: {workers}")

def when_ready(server):
    """Called just after the server is started"""
    server.log.info("Server is ready. Spawning workers...")
    
    # Log server capabilities
    import socket
    hostname = socket.gethostname()
    server.log.info(f"Hostname: {hostname}")
    server.log.info(f"Ready to accept connections on {bind}")
    
    # Start health monitoring
    global health_monitor
    if health_monitor:
        health_monitor.start()
        server.log.info("Worker health monitoring started")

def pre_fork(server, worker):
    """Called just before a worker is forked"""
    server.log.info(f"Forking worker with PID {worker.pid}")

def post_fork(server, worker):
    """Called just after a worker has been forked"""
    worker.log.info(f"Worker spawned (PID: {worker.pid})")
    
    # Set worker-specific configurations
    import resource
    
    # Set max memory usage per worker (2GB)
    max_memory = 2 * 1024 * 1024 * 1024  # 2GB in bytes
    resource.setrlimit(resource.RLIMIT_AS, (max_memory, max_memory))
    
    # Initialize worker-specific connections
    # This ensures each worker has its own database connections
    worker.log.info(f"Worker {worker.pid} initialized")

def worker_int(worker):
    """Called just after a worker exited on SIGINT or SIGQUIT"""
    worker.log.info(f"Worker {worker.pid} received INT/QUIT signal")
    
    # Cleanup worker resources
    import signal
    signal.alarm(30)  # Give 30 seconds for graceful shutdown

def worker_abort(worker):
    """Called when a worker receives SIGABRT signal"""
    worker.log.info(f"Worker {worker.pid} aborted (timeout)")
    
    # Force cleanup if needed
    import gc
    gc.collect()

def pre_exec(server):
    """Called just before a new master process is forked"""
    server.log.info("Forking new master process...")

def pre_request(worker, req):
    """Called just before a worker processes a request"""
    # Skip logging for health checks
    if req.path != '/health':
        worker.log.debug(f"{req.method} {req.path}")
    
    # Add request ID for tracing
    import uuid
    req.headers.append(('X-Request-ID', str(uuid.uuid4())))

def post_request(worker, req, environ, resp):
    """Called after a worker processes a request"""
    # Skip logging for health checks
    if req.path != '/health':
        # Log request with response time
        request_time = time.time() - worker.start_time if hasattr(worker, 'start_time') else 0
        worker.log.info(f"{req.method} {req.path} - {resp.status} ({request_time:.3f}s)")

def child_exit(server, worker):
    """Called after a worker has been exited"""
    server.log.info(f"Worker {worker.pid} exited")
    
    # Clean up Prometheus metrics for this worker
    if os.getenv('PROMETHEUS_ENABLED', 'false').lower() == 'true':
        try:
            from prometheus_client import multiprocess
            multiprocess.mark_process_dead(worker.pid)
        except:
            pass

def worker_exit(server, worker):
    """Called after a worker has been exited, in the worker process"""
    server.log.info(f"Worker {worker.pid} cleanup complete")

def nworkers_changed(server, new_value, old_value):
    """Called when number of workers has changed"""
    server.log.info(f"Number of workers changed from {old_value} to {new_value}")
    
    # Auto-scale notification
    if new_value > old_value:
        server.log.info(f"Scaling UP: Added {new_value - old_value} workers")
    else:
        server.log.info(f"Scaling DOWN: Removed {old_value - new_value} workers")

def on_exit(server):
    """Called just before exiting"""
    server.log.info("=" * 60)
    server.log.info("TEKNOFEST API Server shutting down")
    server.log.info("=" * 60)
    
    # Stop health monitoring
    global health_monitor
    if health_monitor:
        health_monitor.stop()
        server.log.info("Worker health monitoring stopped")
    
    # Final cleanup
    import shutil
    if os.path.exists('/tmp/prometheus'):
        shutil.rmtree('/tmp/prometheus', ignore_errors=True)

# ==========================================
# AUTO-SCALING CONFIGURATION
# ==========================================

import time
import psutil
import threading
import signal
import queue

# ==========================================
# WORKER HEALTH MONITORING
# ==========================================

class WorkerHealthMonitor:
    """Monitor worker health and performance"""
    
    def __init__(self, server):
        self.server = server
        self.unhealthy_workers = {}
        self.worker_metrics = {}
        self.monitoring_thread = None
        self.stop_monitoring = threading.Event()
        
    def start(self):
        """Start monitoring thread"""
        if not self.monitoring_thread:
            self.monitoring_thread = threading.Thread(target=self._monitor_loop)
            self.monitoring_thread.daemon = True
            self.monitoring_thread.start()
            
    def stop(self):
        """Stop monitoring thread"""
        self.stop_monitoring.set()
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
            
    def _monitor_loop(self):
        """Main monitoring loop"""
        while not self.stop_monitoring.is_set():
            try:
                self._check_worker_health()
                self._check_auto_scale()
                time.sleep(int(os.getenv('HEALTH_CHECK_INTERVAL', 30)))
            except Exception as e:
                self.server.log.error(f"Health monitoring error: {e}")
                
    def _check_worker_health(self):
        """Check health of all workers"""
        for worker_id, worker in self.server.WORKERS.items():
            try:
                process = psutil.Process(worker.pid)
                
                # Check CPU usage
                cpu_percent = process.cpu_percent(interval=0.1)
                
                # Check memory usage
                memory_info = process.memory_info()
                memory_mb = memory_info.rss / 1024 / 1024
                
                # Check thread count
                thread_count = process.num_threads()
                
                # Store metrics
                self.worker_metrics[worker_id] = {
                    'cpu_percent': cpu_percent,
                    'memory_mb': memory_mb,
                    'thread_count': thread_count,
                    'status': process.status(),
                    'create_time': process.create_time()
                }
                
                # Check for unhealthy conditions
                if cpu_percent > 90:
                    self._mark_unhealthy(worker_id, f"High CPU: {cpu_percent}%")
                elif memory_mb > 2048:  # 2GB limit
                    self._mark_unhealthy(worker_id, f"High memory: {memory_mb}MB")
                elif thread_count > 100:
                    self._mark_unhealthy(worker_id, f"Too many threads: {thread_count}")
                else:
                    self._mark_healthy(worker_id)
                    
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                self._mark_unhealthy(worker_id, "Process not accessible")
                
    def _mark_unhealthy(self, worker_id, reason):
        """Mark a worker as unhealthy"""
        if worker_id not in self.unhealthy_workers:
            self.unhealthy_workers[worker_id] = {
                'reason': reason,
                'count': 1,
                'first_seen': time.time()
            }
            self.server.log.warning(f"Worker {worker_id} unhealthy: {reason}")
        else:
            self.unhealthy_workers[worker_id]['count'] += 1
            
            # Restart worker if unhealthy for too long
            if self.unhealthy_workers[worker_id]['count'] > 3:
                self.server.log.error(f"Restarting unhealthy worker {worker_id}")
                self._restart_worker(worker_id)
                
    def _mark_healthy(self, worker_id):
        """Mark a worker as healthy"""
        if worker_id in self.unhealthy_workers:
            del self.unhealthy_workers[worker_id]
            self.server.log.info(f"Worker {worker_id} recovered")
            
    def _restart_worker(self, worker_id):
        """Gracefully restart a worker"""
        try:
            worker = self.server.WORKERS.get(worker_id)
            if worker:
                os.kill(worker.pid, signal.SIGTERM)
                del self.unhealthy_workers[worker_id]
        except Exception as e:
            self.server.log.error(f"Failed to restart worker {worker_id}: {e}")
            
    def _check_auto_scale(self):
        """Auto-scale workers based on system metrics"""
        if os.getenv('AUTO_SCALE_ENABLED', 'false').lower() != 'true':
            return
            
        # Calculate average metrics across all workers
        if not self.worker_metrics:
            return
            
        avg_cpu = sum(m['cpu_percent'] for m in self.worker_metrics.values()) / len(self.worker_metrics)
        avg_memory = sum(m['memory_mb'] for m in self.worker_metrics.values()) / len(self.worker_metrics)
        
        # Get system-wide metrics
        system_cpu = psutil.cpu_percent(interval=1)
        system_memory = psutil.virtual_memory().percent
        
        # Load average (1, 5, 15 minutes)
        load_avg = os.getloadavg()
        cpu_count = multiprocessing.cpu_count()
        normalized_load = load_avg[0] / cpu_count
        
        # Queue depth (if available)
        queue_depth = self._get_queue_depth()
        
        # Scaling decision logic
        current_workers = len(self.server.WORKERS)
        max_workers = int(os.getenv('MAX_WORKERS', 16))
        min_workers = int(os.getenv('MIN_WORKERS', 2))
        
        # Enhanced scaling conditions
        scale_up_conditions = [
            system_cpu > 80,
            system_memory > 85,
            normalized_load > 0.8,
            queue_depth > current_workers * 10 if queue_depth else False,
            avg_cpu > 75
        ]
        
        scale_down_conditions = [
            system_cpu < 30,
            system_memory < 40,
            normalized_load < 0.3,
            queue_depth < current_workers * 2 if queue_depth else True,
            avg_cpu < 20
        ]
        
        # Scale up
        if any(scale_up_conditions) and current_workers < max_workers:
            # Calculate how many workers to add
            scale_factor = 1
            if sum(scale_up_conditions) >= 3:
                scale_factor = 2
            if sum(scale_up_conditions) >= 4:
                scale_factor = 3
                
            new_workers = min(current_workers + scale_factor, max_workers)
            self.server.log.info(
                f"AUTO-SCALE UP: CPU={system_cpu:.1f}%, "
                f"Memory={system_memory:.1f}%, "
                f"Load={normalized_load:.2f}, "
                f"Queue={queue_depth or 'N/A'}"
            )
            self.server.log.info(f"Scaling from {current_workers} to {new_workers} workers")
            self.server.num_workers = new_workers
            
        # Scale down
        elif all(scale_down_conditions) and current_workers > min_workers:
            new_workers = max(current_workers - 1, min_workers)
            self.server.log.info(
                f"AUTO-SCALE DOWN: CPU={system_cpu:.1f}%, "
                f"Memory={system_memory:.1f}%, "
                f"Load={normalized_load:.2f}"
            )
            self.server.log.info(f"Scaling from {current_workers} to {new_workers} workers")
            self.server.num_workers = new_workers
            
    def _get_queue_depth(self):
        """Get request queue depth (implementation specific)"""
        try:
            # Check for backlog in socket
            import socket
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            # This is a placeholder - actual implementation would check real queue
            return None
        except:
            return None

# Global health monitor instance
health_monitor = None

def check_auto_scale(server):
    """Legacy auto-scale function for compatibility"""
    global health_monitor
    if health_monitor:
        health_monitor._check_auto_scale()