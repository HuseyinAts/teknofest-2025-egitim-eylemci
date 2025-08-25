"""
Celery Configuration for Distributed Task Queue
TEKNOFEST 2025 - Production-Ready Multi-Worker Deployment
"""

import os
from celery import Celery, Task
from celery.signals import worker_ready, worker_shutdown, task_prerun, task_postrun, task_failure
from kombu import Queue, Exchange
from datetime import timedelta
import logging
from typing import Any

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Redis configuration
REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
CELERY_BROKER_URL = os.getenv('CELERY_BROKER_URL', REDIS_URL)
CELERY_RESULT_BACKEND = os.getenv('CELERY_RESULT_BACKEND', REDIS_URL)

# Create Celery instance
app = Celery(
    'teknofest',
    broker=CELERY_BROKER_URL,
    backend=CELERY_RESULT_BACKEND,
    include=['src.tasks']  # Include task modules
)

# ==========================================
# CELERY CONFIGURATION
# ==========================================

app.conf.update(
    # Task execution settings
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    
    # Result backend settings
    result_expires=3600,  # Results expire after 1 hour
    result_backend_always_retry=True,
    result_backend_max_retries=10,
    
    # Worker settings
    worker_prefetch_multiplier=4,
    worker_max_tasks_per_child=1000,  # Restart worker after 1000 tasks
    worker_disable_rate_limits=False,
    worker_send_task_events=True,
    task_send_sent_event=True,
    
    # Performance settings
    task_compression='gzip',
    result_compression='gzip',
    
    # Task routing
    task_routes={
        'src.tasks.ai_tasks.*': {'queue': 'ai_queue'},
        'src.tasks.data_tasks.*': {'queue': 'data_queue'},
        'src.tasks.email_tasks.*': {'queue': 'email_queue'},
        'src.tasks.report_tasks.*': {'queue': 'report_queue'},
        'src.tasks.maintenance_tasks.*': {'queue': 'maintenance_queue'},
    },
    
    # Queue configuration
    task_queues=(
        Queue('default', Exchange('default'), routing_key='default'),
        Queue('ai_queue', Exchange('ai'), routing_key='ai.#', priority=10),
        Queue('data_queue', Exchange('data'), routing_key='data.#', priority=5),
        Queue('email_queue', Exchange('email'), routing_key='email.#', priority=3),
        Queue('report_queue', Exchange('report'), routing_key='report.#', priority=2),
        Queue('maintenance_queue', Exchange('maintenance'), routing_key='maintenance.#', priority=1),
    ),
    
    # Task time limits
    task_soft_time_limit=300,  # 5 minutes soft limit
    task_time_limit=600,  # 10 minutes hard limit
    
    # Retry settings
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    task_autoretry_for=(Exception,),
    task_max_retries=3,
    task_default_retry_delay=60,  # 1 minute
    
    # Rate limiting
    task_annotations={
        'src.tasks.email_tasks.send_email': {'rate_limit': '10/m'},
        'src.tasks.ai_tasks.process_model': {'rate_limit': '5/m'},
    },
    
    # Monitoring
    worker_hijacking_frequency=30,  # Check for hijacked workers every 30 seconds
    
    # Security
    worker_pool='prefork',  # Use prefork pool for better isolation
    
    # Beat schedule for periodic tasks
    beat_schedule={
        'cleanup-old-results': {
            'task': 'src.tasks.maintenance_tasks.cleanup_old_results',
            'schedule': timedelta(hours=1),
            'options': {'expires': 3600}
        },
        'health-check': {
            'task': 'src.tasks.maintenance_tasks.health_check',
            'schedule': timedelta(minutes=5),
            'options': {'expires': 300}
        },
        'collect-metrics': {
            'task': 'src.tasks.maintenance_tasks.collect_metrics',
            'schedule': timedelta(minutes=1),
            'options': {'expires': 60}
        },
        'backup-database': {
            'task': 'src.tasks.maintenance_tasks.backup_database',
            'schedule': timedelta(hours=24),
            'options': {'expires': 3600}
        },
        'generate-daily-report': {
            'task': 'src.tasks.report_tasks.generate_daily_report',
            'schedule': timedelta(hours=24),
            'options': {'expires': 3600}
        },
    },
)

# ==========================================
# CUSTOM TASK BASE CLASS
# ==========================================

class BaseTask(Task):
    """Base task with automatic retry and error handling"""
    
    autoretry_for = (Exception,)
    max_retries = 3
    default_retry_delay = 60
    
    def on_failure(self, exc, task_id, args, kwargs, einfo):
        """Called when task fails"""
        logger.error(f"Task {self.name}[{task_id}] failed: {exc}")
        # Send alert for critical tasks
        if hasattr(self, 'critical') and self.critical:
            self.send_failure_alert(task_id, exc)
    
    def on_retry(self, exc, task_id, args, kwargs, einfo):
        """Called when task is retried"""
        logger.warning(f"Task {self.name}[{task_id}] retry {self.request.retries}: {exc}")
    
    def on_success(self, retval, task_id, args, kwargs):
        """Called when task succeeds"""
        logger.info(f"Task {self.name}[{task_id}] completed successfully")
    
    def send_failure_alert(self, task_id, exc):
        """Send alert for task failure"""
        # Implement alert logic (email, webhook, etc.)
        pass

# Set default task base
app.Task = BaseTask

# ==========================================
# SIGNAL HANDLERS
# ==========================================

@worker_ready.connect
def on_worker_ready(sender=None, **kwargs):
    """Called when worker is ready"""
    logger.info(f"Worker {sender.hostname} is ready")
    # Initialize worker-specific resources
    initialize_worker_resources()

@worker_shutdown.connect
def on_worker_shutdown(sender=None, **kwargs):
    """Called when worker is shutting down"""
    logger.info(f"Worker {sender.hostname} is shutting down")
    # Cleanup worker resources
    cleanup_worker_resources()

@task_prerun.connect
def on_task_prerun(sender=None, task_id=None, task=None, args=None, kwargs=None, **kw):
    """Called before task execution"""
    logger.debug(f"Starting task {task.name}[{task_id}]")
    # Set up task context
    setup_task_context(task_id)

@task_postrun.connect
def on_task_postrun(sender=None, task_id=None, task=None, args=None, kwargs=None, retval=None, state=None, **kw):
    """Called after task execution"""
    logger.debug(f"Completed task {task.name}[{task_id}] with state {state}")
    # Clean up task context
    cleanup_task_context(task_id)

@task_failure.connect
def on_task_failure(sender=None, task_id=None, exception=None, args=None, kwargs=None, traceback=None, einfo=None, **kw):
    """Called on task failure"""
    logger.error(f"Task {sender.name}[{task_id}] failed with {exception}")
    # Record failure metrics
    record_task_failure(task_id, exception)

# ==========================================
# HELPER FUNCTIONS
# ==========================================

def initialize_worker_resources():
    """Initialize resources for worker"""
    # Initialize database connections
    # Initialize model cache
    # Set up monitoring
    pass

def cleanup_worker_resources():
    """Cleanup worker resources"""
    # Close database connections
    # Clear caches
    # Flush metrics
    pass

def setup_task_context(task_id):
    """Set up context for task execution"""
    # Set up request ID
    # Initialize tracing
    # Set up logging context
    pass

def cleanup_task_context(task_id):
    """Clean up task execution context"""
    # Clear request context
    # Flush logs
    pass

def record_task_failure(task_id, exception):
    """Record task failure metrics"""
    # Send to monitoring system
    # Update failure counters
    pass

# ==========================================
# TASK DISCOVERY
# ==========================================

# Auto-discover tasks from task modules
app.autodiscover_tasks(['src.tasks'])