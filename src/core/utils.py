"""
Clean Code Utility Functions
TEKNOFEST 2025 - DRY Principle Implementation
"""

import re
import hashlib
import json
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from datetime import datetime, timedelta
from functools import wraps, lru_cache
import logging

from src.core.constants import (
    EMAIL_REGEX, USERNAME_REGEX, PHONE_REGEX, URL_REGEX,
    MIN_PASSWORD_LENGTH, MAX_PASSWORD_LENGTH,
    DATE_FORMAT, DATETIME_FORMAT, ISO_FORMAT,
    MAX_FUNCTION_LENGTH, MAX_CYCLOMATIC_COMPLEXITY
)

logger = logging.getLogger(__name__)


# ==========================================
# VALIDATION UTILITIES
# ==========================================

class Validator:
    """Centralized validation utilities"""
    
    @staticmethod
    def validate_email(email: str) -> Tuple[bool, Optional[str]]:
        """Validate email format"""
        if not email:
            return False, "Email is required"
        
        if not re.match(EMAIL_REGEX, email):
            return False, "Invalid email format"
        
        return True, None
    
    @staticmethod
    def validate_username(username: str) -> Tuple[bool, Optional[str]]:
        """Validate username format"""
        if not username:
            return False, "Username is required"
        
        if not re.match(USERNAME_REGEX, username):
            return False, "Username must be 3-20 characters, alphanumeric and underscore only"
        
        return True, None
    
    @staticmethod
    def validate_phone(phone: str) -> Tuple[bool, Optional[str]]:
        """Validate phone number"""
        if not phone:
            return False, "Phone number is required"
        
        if not re.match(PHONE_REGEX, phone):
            return False, "Invalid phone number format"
        
        return True, None
    
    @staticmethod
    def validate_url(url: str) -> Tuple[bool, Optional[str]]:
        """Validate URL format"""
        if not url:
            return False, "URL is required"
        
        if not re.match(URL_REGEX, url):
            return False, "Invalid URL format"
        
        return True, None
    
    @staticmethod
    def validate_password_strength(password: str) -> Tuple[bool, List[str]]:
        """Validate password strength and return issues"""
        issues = []
        
        if len(password) < MIN_PASSWORD_LENGTH:
            issues.append(f"At least {MIN_PASSWORD_LENGTH} characters required")
        
        if len(password) > MAX_PASSWORD_LENGTH:
            issues.append(f"Maximum {MAX_PASSWORD_LENGTH} characters allowed")
        
        if not re.search(r'[A-Z]', password):
            issues.append("At least one uppercase letter required")
        
        if not re.search(r'[a-z]', password):
            issues.append("At least one lowercase letter required")
        
        if not re.search(r'\d', password):
            issues.append("At least one number required")
        
        if not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
            issues.append("At least one special character required")
        
        return len(issues) == 0, issues
    
    @staticmethod
    def validate_date_range(
        start_date: datetime, 
        end_date: datetime
    ) -> Tuple[bool, Optional[str]]:
        """Validate date range"""
        if start_date >= end_date:
            return False, "Start date must be before end date"
        
        if end_date - start_date > timedelta(days=365):
            return False, "Date range cannot exceed one year"
        
        return True, None


# ==========================================
# STRING UTILITIES
# ==========================================

class StringUtils:
    """String manipulation utilities"""
    
    @staticmethod
    def sanitize(text: str, max_length: Optional[int] = None) -> str:
        """Sanitize string input"""
        if not text:
            return ""
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        # Remove multiple spaces
        text = re.sub(r'\s+', ' ', text)
        
        # Remove control characters
        text = re.sub(r'[\x00-\x1F\x7F]', '', text)
        
        # Limit length if specified
        if max_length and len(text) > max_length:
            text = text[:max_length]
        
        return text
    
    @staticmethod
    def slugify(text: str) -> str:
        """Convert text to URL-friendly slug"""
        # Convert to lowercase
        text = text.lower()
        
        # Replace spaces with hyphens
        text = re.sub(r'\s+', '-', text)
        
        # Remove non-alphanumeric characters
        text = re.sub(r'[^a-z0-9-]', '', text)
        
        # Remove multiple hyphens
        text = re.sub(r'-+', '-', text)
        
        # Remove leading/trailing hyphens
        text = text.strip('-')
        
        return text
    
    @staticmethod
    def truncate(text: str, length: int, suffix: str = "...") -> str:
        """Truncate text to specified length"""
        if len(text) <= length:
            return text
        
        return text[:length - len(suffix)] + suffix
    
    @staticmethod
    @lru_cache(maxsize=128)
    def generate_hash(text: str, algorithm: str = 'sha256') -> str:
        """Generate hash of text"""
        hash_obj = hashlib.new(algorithm)
        hash_obj.update(text.encode('utf-8'))
        return hash_obj.hexdigest()
    
    @staticmethod
    def mask_sensitive(text: str, visible_chars: int = 4) -> str:
        """Mask sensitive information"""
        if len(text) <= visible_chars * 2:
            return '*' * len(text)
        
        return text[:visible_chars] + '*' * (len(text) - visible_chars * 2) + text[-visible_chars:]


# ==========================================
# DATE/TIME UTILITIES
# ==========================================

class DateTimeUtils:
    """Date and time utilities"""
    
    @staticmethod
    def format_date(date: datetime, format: str = DATE_FORMAT) -> str:
        """Format date to string"""
        return date.strftime(format)
    
    @staticmethod
    def format_datetime(dt: datetime, format: str = DATETIME_FORMAT) -> str:
        """Format datetime to string"""
        return dt.strftime(format)
    
    @staticmethod
    def parse_date(date_str: str, format: str = DATE_FORMAT) -> datetime:
        """Parse date from string"""
        return datetime.strptime(date_str, format)
    
    @staticmethod
    def parse_datetime(dt_str: str, format: str = DATETIME_FORMAT) -> datetime:
        """Parse datetime from string"""
        return datetime.strptime(dt_str, format)
    
    @staticmethod
    def to_iso(dt: datetime) -> str:
        """Convert datetime to ISO format"""
        return dt.strftime(ISO_FORMAT)
    
    @staticmethod
    def from_iso(iso_str: str) -> datetime:
        """Parse datetime from ISO format"""
        return datetime.strptime(iso_str, ISO_FORMAT)
    
    @staticmethod
    def time_ago(dt: datetime) -> str:
        """Get human-readable time ago"""
        now = datetime.utcnow()
        diff = now - dt
        
        if diff.days > 365:
            return f"{diff.days // 365} yıl önce"
        elif diff.days > 30:
            return f"{diff.days // 30} ay önce"
        elif diff.days > 0:
            return f"{diff.days} gün önce"
        elif diff.seconds > 3600:
            return f"{diff.seconds // 3600} saat önce"
        elif diff.seconds > 60:
            return f"{diff.seconds // 60} dakika önce"
        else:
            return "Az önce"
    
    @staticmethod
    def add_business_days(start_date: datetime, days: int) -> datetime:
        """Add business days to date"""
        current = start_date
        remaining = days
        
        while remaining > 0:
            current += timedelta(days=1)
            if current.weekday() < 5:  # Monday-Friday
                remaining -= 1
        
        return current


# ==========================================
# DATA TRANSFORMATION UTILITIES
# ==========================================

class DataTransformer:
    """Data transformation utilities"""
    
    @staticmethod
    def flatten_dict(
        nested_dict: Dict, 
        parent_key: str = '', 
        separator: str = '.'
    ) -> Dict:
        """Flatten nested dictionary"""
        items = []
        
        for key, value in nested_dict.items():
            new_key = f"{parent_key}{separator}{key}" if parent_key else key
            
            if isinstance(value, dict):
                items.extend(
                    DataTransformer.flatten_dict(value, new_key, separator).items()
                )
            else:
                items.append((new_key, value))
        
        return dict(items)
    
    @staticmethod
    def unflatten_dict(
        flat_dict: Dict, 
        separator: str = '.'
    ) -> Dict:
        """Unflatten dictionary"""
        result = {}
        
        for key, value in flat_dict.items():
            parts = key.split(separator)
            current = result
            
            for part in parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]
            
            current[parts[-1]] = value
        
        return result
    
    @staticmethod
    def chunk_list(lst: List, chunk_size: int) -> List[List]:
        """Split list into chunks"""
        return [
            lst[i:i + chunk_size] 
            for i in range(0, len(lst), chunk_size)
        ]
    
    @staticmethod
    def merge_dicts(*dicts: Dict) -> Dict:
        """Merge multiple dictionaries"""
        result = {}
        
        for d in dicts:
            result.update(d)
        
        return result
    
    @staticmethod
    def remove_none_values(data: Dict) -> Dict:
        """Remove None values from dictionary"""
        return {
            k: v for k, v in data.items() 
            if v is not None
        }
    
    @staticmethod
    def safe_get(
        data: Dict, 
        path: str, 
        default: Any = None, 
        separator: str = '.'
    ) -> Any:
        """Safely get nested dictionary value"""
        keys = path.split(separator)
        current = data
        
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return default
        
        return current


# ==========================================
# FUNCTIONAL UTILITIES
# ==========================================

class FunctionalUtils:
    """Functional programming utilities"""
    
    @staticmethod
    def retry(
        max_attempts: int = 3,
        delay: float = 1.0,
        backoff: float = 2.0,
        exceptions: Tuple[Exception, ...] = (Exception,)
    ):
        """Retry decorator with exponential backoff"""
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                attempt = 1
                current_delay = delay
                
                while attempt <= max_attempts:
                    try:
                        return func(*args, **kwargs)
                    except exceptions as e:
                        if attempt == max_attempts:
                            logger.error(f"Max retry attempts reached for {func.__name__}")
                            raise
                        
                        logger.warning(
                            f"Attempt {attempt} failed for {func.__name__}: {e}. "
                            f"Retrying in {current_delay}s..."
                        )
                        
                        import time
                        time.sleep(current_delay)
                        current_delay *= backoff
                        attempt += 1
                
            return wrapper
        return decorator
    
    @staticmethod
    def memoize(maxsize: int = 128):
        """Memoization decorator"""
        def decorator(func: Callable) -> Callable:
            cache = {}
            
            @wraps(func)
            def wrapper(*args, **kwargs):
                # Create cache key
                key = str(args) + str(kwargs)
                
                if key in cache:
                    return cache[key]
                
                result = func(*args, **kwargs)
                
                # Limit cache size
                if len(cache) >= maxsize:
                    # Remove oldest entry (simple FIFO)
                    cache.pop(next(iter(cache)))
                
                cache[key] = result
                return result
            
            wrapper.cache_clear = lambda: cache.clear()
            return wrapper
        
        return decorator
    
    @staticmethod
    def compose(*functions: Callable) -> Callable:
        """Compose multiple functions"""
        def composed(x):
            for func in reversed(functions):
                x = func(x)
            return x
        return composed
    
    @staticmethod
    def pipe(*functions: Callable) -> Callable:
        """Pipe functions (opposite of compose)"""
        def piped(x):
            for func in functions:
                x = func(x)
            return x
        return piped


# ==========================================
# CALCULATION UTILITIES
# ==========================================

class CalculationUtils:
    """Mathematical calculation utilities"""
    
    @staticmethod
    def calculate_percentage(value: float, total: float) -> float:
        """Calculate percentage safely"""
        if total == 0:
            return 0.0
        return (value / total) * 100
    
    @staticmethod
    def calculate_average(values: List[float]) -> float:
        """Calculate average safely"""
        if not values:
            return 0.0
        return sum(values) / len(values)
    
    @staticmethod
    def calculate_median(values: List[float]) -> float:
        """Calculate median"""
        if not values:
            return 0.0
        
        sorted_values = sorted(values)
        n = len(sorted_values)
        
        if n % 2 == 0:
            return (sorted_values[n//2 - 1] + sorted_values[n//2]) / 2
        else:
            return sorted_values[n//2]
    
    @staticmethod
    def clamp(value: float, min_val: float, max_val: float) -> float:
        """Clamp value between min and max"""
        return max(min_val, min(value, max_val))
    
    @staticmethod
    def normalize(value: float, min_val: float, max_val: float) -> float:
        """Normalize value to 0-1 range"""
        if max_val == min_val:
            return 0.0
        return (value - min_val) / (max_val - min_val)
    
    @staticmethod
    def lerp(start: float, end: float, t: float) -> float:
        """Linear interpolation"""
        return start + (end - start) * t


# ==========================================
# CODE QUALITY UTILITIES
# ==========================================

class CodeQualityChecker:
    """Code quality checking utilities"""
    
    @staticmethod
    def check_function_length(func: Callable) -> Tuple[bool, int]:
        """Check if function length is within limits"""
        import inspect
        
        source = inspect.getsource(func)
        lines = source.split('\n')
        
        # Filter out empty lines and comments
        code_lines = [
            line for line in lines 
            if line.strip() and not line.strip().startswith('#')
        ]
        
        line_count = len(code_lines)
        is_valid = line_count <= MAX_FUNCTION_LENGTH
        
        return is_valid, line_count
    
    @staticmethod
    def check_cyclomatic_complexity(func: Callable) -> Tuple[bool, int]:
        """Check cyclomatic complexity"""
        import inspect
        import ast
        
        source = inspect.getsource(func)
        tree = ast.parse(source)
        
        complexity = 1  # Base complexity
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.While, ast.For)):
                complexity += 1
            elif isinstance(node, ast.BoolOp):
                complexity += len(node.values) - 1
        
        is_valid = complexity <= MAX_CYCLOMATIC_COMPLEXITY
        
        return is_valid, complexity
    
    @staticmethod
    def check_function_parameters(func: Callable) -> Tuple[bool, int]:
        """Check number of function parameters"""
        import inspect
        
        sig = inspect.signature(func)
        param_count = len(sig.parameters)
        
        is_valid = param_count <= 5  # Max 5 parameters
        
        return is_valid, param_count
