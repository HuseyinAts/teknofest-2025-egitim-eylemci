"""
Celery Tasks Module
TEKNOFEST 2025 - Distributed Task Queue
"""

from .ai_tasks import *
from .data_tasks import *
from .email_tasks import *
from .report_tasks import *
from .maintenance_tasks import *

__all__ = [
    'ai_tasks',
    'data_tasks',
    'email_tasks',
    'report_tasks',
    'maintenance_tasks'
]