"""
Secure Configuration Management
TEKNOFEST 2025 - Production Ready

This module provides secure handling of sensitive configuration values.
"""

import os
import logging
from typing import Optional, Dict, Any
from pathlib import Path
import json
from cryptography.fernet import Fernet
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class SecureConfig:
    """
    Secure configuration manager that handles sensitive data
    with encryption and proper access controls.
    """
    
    def __init__(self):
        self._config_cache: Dict[str, Any] = {}
        self._encryption_key = self._get_or_create_encryption_key()
        self._fernet = Fernet(self._encryption_key) if self._encryption_key else None
        
    def _get_or_create_encryption_key(self) -> Optional[bytes]:
        """Get or create encryption key for sensitive data."""
        key_file = Path(".encryption_key")
        
        # In production, get from environment variable
        env_key = os.getenv("CONFIG_ENCRYPTION_KEY")
        if env_key:
            return env_key.encode()
            
        # For development, use local file (NEVER commit this file)
        if key_file.exists():
            with open(key_file, 'rb') as f:
                return f.read()
        else:
            # Generate new key for first run
            key = Fernet.generate_key()
            with open(key_file, 'wb') as f:
                f.write(key)
            logger.warning("Generated new encryption key. Add to .gitignore!")
            return key
    
    def get_secret(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """
        Get secret value from environment with fallback.
        
        Args:
            key: Environment variable name
            default: Default value if not found
            
        Returns:
            Secret value or default
        """
        # Check cache first
        if key in self._config_cache:
            return self._config_cache[key]
            
        # Get from environment
        value = os.getenv(key, default)
        
        # Validate critical secrets
        if key in self._get_required_secrets() and not value:
            raise ValueError(f"Required secret '{key}' is not configured")
            
        # Cache for performance
        self._config_cache[key] = value
        return value
    
    def _get_required_secrets(self) -> list:
        """List of required secrets that must be configured."""
        return [
            "SECRET_KEY",
            "JWT_SECRET_KEY",
            "DATABASE_URL" if os.getenv("APP_ENV") == "production" else None,
        ]
    
    def encrypt_value(self, value: str) -> str:
        """Encrypt sensitive value for storage."""
        if not self._fernet:
            raise RuntimeError("Encryption not available")
        return self._fernet.encrypt(value.encode()).decode()
    
    def decrypt_value(self, encrypted_value: str) -> str:
        """Decrypt sensitive value."""
        if not self._fernet:
            raise RuntimeError("Encryption not available")
        return self._fernet.decrypt(encrypted_value.encode()).decode()
    
    def validate_production_config(self) -> Dict[str, bool]:
        """
        Validate that all production requirements are met.
        
        Returns:
            Dictionary of validation results
        """
        results = {}
        
        # Check environment
        results["environment"] = os.getenv("APP_ENV") == "production"
        
        # Check debug is disabled
        results["debug_disabled"] = os.getenv("APP_DEBUG", "true").lower() == "false"
        
        # Check secure keys
        secret_key = self.get_secret("SECRET_KEY")
        results["secret_key_secure"] = (
            secret_key and 
            len(secret_key) >= 64 and 
            not secret_key.startswith("CHANGE_THIS")
        )
        
        jwt_key = self.get_secret("JWT_SECRET_KEY")
        results["jwt_key_secure"] = (
            jwt_key and 
            len(jwt_key) >= 64 and 
            jwt_key != secret_key and
            not jwt_key.startswith("CHANGE_THIS")
        )
        
        # Check database
        db_url = self.get_secret("DATABASE_URL")
        results["database_configured"] = (
            db_url and 
            "postgresql" in db_url and 
            "localhost" not in db_url if results["environment"] else True
        )
        
        # Check CORS
        cors_origins = os.getenv("CORS_ORIGINS", "")
        results["cors_secure"] = (
            "*" not in cors_origins if results["environment"] else True
        )
        
        # Check SSL
        results["ssl_enabled"] = (
            os.getenv("SSL_ENABLED", "false").lower() == "true" 
            if results["environment"] else True
        )
        
        # Check rate limiting
        results["rate_limiting"] = (
            os.getenv("RATE_LIMIT_ENABLED", "false").lower() == "true"
            if results["environment"] else True
        )
        
        # Check monitoring
        results["monitoring_configured"] = bool(os.getenv("SENTRY_DSN"))
        
        return results
    
    def get_database_config(self) -> Dict[str, Any]:
        """Get database configuration with proper defaults."""
        is_production = os.getenv("APP_ENV") == "production"
        
        return {
            "url": self.get_secret("DATABASE_URL"),
            "pool_size": int(os.getenv("DATABASE_POOL_SIZE", "20" if is_production else "5")),
            "max_overflow": int(os.getenv("DATABASE_MAX_OVERFLOW", "40" if is_production else "10")),
            "pool_timeout": int(os.getenv("DATABASE_POOL_TIMEOUT", "10")),
            "pool_recycle": int(os.getenv("DATABASE_POOL_RECYCLE", "1800")),
            "pool_pre_ping": os.getenv("DATABASE_POOL_PRE_PING", "true").lower() == "true",
            "echo": os.getenv("DATABASE_ECHO", "false").lower() == "true" and not is_production,
        }
    
    def get_redis_config(self) -> Dict[str, Any]:
        """Get Redis configuration."""
        return {
            "url": os.getenv("REDIS_URL", "redis://localhost:6379/0"),
            "password": self.get_secret("REDIS_PASSWORD"),
            "max_connections": int(os.getenv("REDIS_MAX_CONNECTIONS", "10")),
        }
    
    def get_jwt_config(self) -> Dict[str, Any]:
        """Get JWT configuration."""
        return {
            "secret_key": self.get_secret("JWT_SECRET_KEY"),
            "algorithm": os.getenv("JWT_ALGORITHM", "HS256"),
            "access_token_expire_minutes": int(
                os.getenv("JWT_ACCESS_TOKEN_EXPIRE_MINUTES", "30")
            ),
            "refresh_token_expire_days": int(
                os.getenv("JWT_REFRESH_TOKEN_EXPIRE_DAYS", "7")
            ),
        }
    
    def get_cors_config(self) -> Dict[str, Any]:
        """Get CORS configuration."""
        origins = os.getenv("CORS_ORIGINS", "http://localhost:3000")
        return {
            "origins": [o.strip() for o in origins.split(",")],
            "credentials": os.getenv("CORS_ALLOW_CREDENTIALS", "true").lower() == "true",
            "methods": os.getenv("CORS_ALLOW_METHODS", "*").split(","),
            "headers": os.getenv("CORS_ALLOW_HEADERS", "*").split(","),
        }
    
    def export_safe_config(self) -> Dict[str, Any]:
        """
        Export configuration without sensitive values.
        Used for logging and debugging.
        """
        config = {
            "app_env": os.getenv("APP_ENV"),
            "app_debug": os.getenv("APP_DEBUG"),
            "api_port": os.getenv("API_PORT"),
            "database_configured": bool(self.get_secret("DATABASE_URL")),
            "redis_configured": bool(os.getenv("REDIS_URL")),
            "cors_origins_count": len(self.get_cors_config()["origins"]),
            "rate_limiting_enabled": os.getenv("RATE_LIMIT_ENABLED"),
            "monitoring_enabled": bool(os.getenv("SENTRY_DSN")),
            "ssl_enabled": os.getenv("SSL_ENABLED"),
        }
        return config

# Global instance
secure_config = SecureConfig()

def validate_production_ready():
    """Validate production configuration on startup."""
    if os.getenv("APP_ENV") == "production":
        results = secure_config.validate_production_config()
        failed = [k for k, v in results.items() if not v]
        
        if failed:
            logger.error(f"Production validation failed: {failed}")
            raise RuntimeError(f"Production configuration incomplete: {', '.join(failed)}")
        
        logger.info("✅ Production configuration validated successfully")
        return True
    return True