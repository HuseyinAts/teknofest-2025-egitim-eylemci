"""
Production-Ready Configuration Management for TEKNOFEST 2025
Centralized configuration with validation, type checking, and security
"""

import os
import secrets
from pathlib import Path
from typing import List, Optional, Union
from enum import Enum
from functools import lru_cache
import warnings

try:
    # Pydantic v2
    from pydantic import Field, field_validator, SecretStr
    from pydantic_settings import BaseSettings
    from pydantic.networks import AnyHttpUrl, PostgresDsn, RedisDsn
    PYDANTIC_V2 = True
except ImportError:
    # Pydantic v1
    from pydantic import BaseSettings, Field, validator as field_validator, SecretStr
    from pydantic.networks import AnyHttpUrl, PostgresDsn, RedisDsn
    PYDANTIC_V2 = False


class Environment(str, Enum):
    """Application environment types"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"


class LogLevel(str, Enum):
    """Logging levels"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class Settings(BaseSettings):
    """
    Application settings with validation and type checking.
    Values are loaded from environment variables or .env file.
    """
    
    # ==========================================
    # APPLICATION SETTINGS
    # ==========================================
    app_name: str = Field(default="teknofest-2025-egitim-eylemci", env="APP_NAME")
    app_env: Environment = Field(default=Environment.DEVELOPMENT, env="APP_ENV")
    app_version: str = Field(default="3.0.0", env="APP_VERSION")
    app_debug: bool = Field(default=False, env="APP_DEBUG")
    
    @field_validator("app_debug", mode='before' if PYDANTIC_V2 else 'pre')
    def validate_debug(cls, v, info):
        """Ensure debug is disabled in production"""
        if PYDANTIC_V2:
            values = info.data if hasattr(info, 'data') else {}
        else:
            values = info
        if values.get("app_env") == Environment.PRODUCTION and v:
            warnings.warn("Debug mode should be disabled in production!")
            return False
        return v
    
    # ==========================================
    # SECURITY SETTINGS
    # ==========================================
    secret_key: SecretStr = Field(
        ...,  # SECURITY: No default value - must be provided
        env="SECRET_KEY",
        description="Required: Application secret key for session management"
    )
    jwt_secret_key: SecretStr = Field(
        ...,  # SECURITY: No default value - must be provided
        env="JWT_SECRET_KEY",
        description="Required: JWT signing key - must be different from secret_key"
    )
    jwt_algorithm: str = Field(default="HS256", env="JWT_ALGORITHM")
    
    # SECURITY: Additional security settings
    security_bcrypt_rounds: int = Field(default=12, env="SECURITY_BCRYPT_ROUNDS", ge=10, le=16)
    security_password_min_length: int = Field(default=8, env="SECURITY_PASSWORD_MIN_LENGTH", ge=8)
    security_password_require_special: bool = Field(default=True, env="SECURITY_PASSWORD_REQUIRE_SPECIAL")
    security_session_lifetime_hours: int = Field(default=24, env="SECURITY_SESSION_LIFETIME_HOURS")
    security_max_login_attempts: int = Field(default=5, env="SECURITY_MAX_LOGIN_ATTEMPTS")
    security_lockout_duration_minutes: int = Field(default=15, env="SECURITY_LOCKOUT_DURATION_MINUTES")
    jwt_access_token_expire_minutes: int = Field(default=30, env="JWT_ACCESS_TOKEN_EXPIRE_MINUTES")
    jwt_refresh_token_expire_days: int = Field(default=7, env="JWT_REFRESH_TOKEN_EXPIRE_DAYS")
    
    @field_validator("secret_key", "jwt_secret_key")
    def validate_secrets(cls, v, info):
        """Validate secret keys are strong enough"""
        if not v:
            return v
        secret_value = v.get_secret_value()
        if len(secret_value) < 32:
            field_name = info.field_name if PYDANTIC_V2 else info
            raise ValueError(f"{field_name} must be at least 32 characters long")
        if secret_value.startswith("CHANGE_THIS"):
            field_name = info.field_name if PYDANTIC_V2 else info
            raise ValueError(f"{field_name} must be changed from default value")
        return v
    
    # ==========================================
    # API SERVER SETTINGS
    # ==========================================
    api_host: str = Field(default="0.0.0.0", env="API_HOST")
    api_port: int = Field(default=8003, env="API_PORT", ge=1, le=65535)
    api_workers: int = Field(default=1, env="API_WORKERS", ge=1)
    api_reload: bool = Field(default=False, env="API_RELOAD")
    
    @field_validator("api_workers", mode='before')
    def validate_workers(cls, v, info):
        """Set appropriate worker count for environment"""
        if PYDANTIC_V2:
            values = info.data if hasattr(info, 'data') else {}
        else:
            values = info
        if values.get("app_env") == Environment.PRODUCTION:
            # In production, use CPU count
            return max(1, min(v or os.cpu_count() or 1, 16))
        return 1  # Single worker for development
    
    # ==========================================
    # CORS SETTINGS
    # ==========================================
    cors_origins: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:3001", "http://localhost:3002"],
        env="CORS_ORIGINS"
    )
    cors_allow_credentials: bool = Field(default=True, env="CORS_ALLOW_CREDENTIALS")
    cors_allow_methods: List[str] = Field(default=["*"], env="CORS_ALLOW_METHODS")
    cors_allow_headers: List[str] = Field(default=["*"], env="CORS_ALLOW_HEADERS")
    
    @field_validator("cors_origins", mode='before')
    def validate_cors_origins(cls, v, info):
        """Parse comma-separated CORS origins and validate for production"""
        if PYDANTIC_V2:
            values = info.data if hasattr(info, 'data') else {}
        else:
            values = info
        if isinstance(v, str):
            v = [origin.strip() for origin in v.split(",")]
        
        if values.get("app_env") == Environment.PRODUCTION:
            # Don't allow wildcard in production
            if "*" in v or any("*" in origin for origin in v):
                raise ValueError("Wildcard CORS origins not allowed in production")
            # Ensure HTTPS in production
            for origin in v:
                if not origin.startswith("https://") and not origin.startswith("http://localhost"):
                    warnings.warn(f"Non-HTTPS origin in production: {origin}")
        
        return v
    
    # ==========================================
    # DATABASE SETTINGS
    # ==========================================
    database_url: Optional[str] = Field(
        default="postgresql://postgres:password@localhost:5432/teknofest_dev",
        env="DATABASE_URL"
    )
    database_pool_size: int = Field(default=20, env="DATABASE_POOL_SIZE", ge=1, le=100)
    database_max_overflow: int = Field(default=40, env="DATABASE_MAX_OVERFLOW", ge=0, le=200)
    database_pool_timeout: int = Field(default=10, env="DATABASE_POOL_TIMEOUT", ge=1, le=60)
    database_pool_recycle: int = Field(default=1800, env="DATABASE_POOL_RECYCLE", ge=300, le=7200)
    database_pool_pre_ping: bool = Field(default=True, env="DATABASE_POOL_PRE_PING")
    database_pool_use_lifo: bool = Field(default=True, env="DATABASE_POOL_USE_LIFO")
    database_statement_timeout: int = Field(default=30000, env="DATABASE_STATEMENT_TIMEOUT", ge=1000)  # milliseconds
    database_lock_timeout: int = Field(default=5000, env="DATABASE_LOCK_TIMEOUT", ge=1000)  # milliseconds
    database_idle_in_transaction_timeout: int = Field(default=30000, env="DATABASE_IDLE_IN_TRANSACTION_TIMEOUT", ge=1000)  # milliseconds
    database_echo: bool = Field(default=False, env="DATABASE_ECHO")
    database_echo_pool: bool = Field(default=False, env="DATABASE_ECHO_POOL")
    
    @field_validator("database_pool_size", mode='before')
    def validate_pool_size(cls, v, info):
        """Validate pool size based on environment"""
        if PYDANTIC_V2:
            values = info.data if hasattr(info, 'data') else {}
        else:
            values = info
        if values.get("app_env") == Environment.PRODUCTION:
            # Production should have larger pool
            return max(20, v or 20)
        return v or 5
    
    @field_validator("database_pool_pre_ping")
    def validate_pool_pre_ping(cls, v, info):
        """Ensure pre-ping is enabled in production"""
        if PYDANTIC_V2:
            values = info.data if hasattr(info, 'data') else {}
        else:
            values = info
        if values.get("app_env") == Environment.PRODUCTION and not v:
            warnings.warn("Connection pre-ping should be enabled in production")
            return True
        return v
    
    # Redis Settings
    redis_url: Optional[str] = Field(
        default="redis://localhost:6379/0",
        env="REDIS_URL"
    )
    redis_password: Optional[SecretStr] = Field(default=None, env="REDIS_PASSWORD")
    redis_max_connections: int = Field(default=10, env="REDIS_MAX_CONNECTIONS", ge=1)
    
    # ==========================================
    # AI MODEL SETTINGS
    # ==========================================
    hugging_face_hub_token: Optional[SecretStr] = Field(default=None, env="HUGGING_FACE_HUB_TOKEN")
    hugging_face_endpoint_url: Optional[AnyHttpUrl] = Field(default=None, env="HUGGING_FACE_ENDPOINT_URL")
    
    model_name: str = Field(
        default="Huseyin/teknofest-2025-turkish-edu-v2",
        env="MODEL_NAME"
    )
    model_device: str = Field(default="cpu", env="MODEL_DEVICE")
    model_max_length: int = Field(default=2048, env="MODEL_MAX_LENGTH", ge=128, le=8192)
    model_temperature: float = Field(default=0.7, env="MODEL_TEMPERATURE", ge=0.0, le=2.0)
    model_top_p: float = Field(default=0.9, env="MODEL_TOP_P", ge=0.0, le=1.0)
    model_top_k: int = Field(default=50, env="MODEL_TOP_K", ge=1, le=100)
    model_cache_dir: Path = Field(default=Path("./model_cache"), env="MODEL_CACHE_DIR")
    model_load_in_8bit: bool = Field(default=False, env="MODEL_LOAD_IN_8BIT")
    model_use_cache: bool = Field(default=True, env="MODEL_USE_CACHE")
    
    @field_validator("model_device")
    def validate_model_device(cls, v):
        """Validate model device is available"""
        import torch
        if v == "cuda" and not torch.cuda.is_available():
            warnings.warn("CUDA requested but not available, falling back to CPU")
            return "cpu"
        elif v == "mps" and not torch.backends.mps.is_available():
            warnings.warn("MPS requested but not available, falling back to CPU")
            return "cpu"
        return v
    
    # ==========================================
    # LOGGING SETTINGS
    # ==========================================
    log_level: LogLevel = Field(default=LogLevel.INFO, env="LOG_LEVEL")
    log_format: str = Field(default="simple", env="LOG_FORMAT")
    log_file: Optional[Path] = Field(default=Path("logs/app.log"), env="LOG_FILE")
    log_max_size: int = Field(default=10, env="LOG_MAX_SIZE")  # MB
    log_backup_count: int = Field(default=5, env="LOG_BACKUP_COUNT")
    
    # ==========================================
    # RATE LIMITING
    # ==========================================
    rate_limit_enabled: bool = Field(default=False, env="RATE_LIMIT_ENABLED")
    rate_limit_requests: int = Field(default=100, env="RATE_LIMIT_REQUESTS", ge=1)
    rate_limit_period: int = Field(default=3600, env="RATE_LIMIT_PERIOD", ge=1)
    
    @field_validator("rate_limit_enabled", mode='before')
    def validate_rate_limit(cls, v, info):
        """Enable rate limiting in production"""
        if PYDANTIC_V2:
            values = info.data if hasattr(info, 'data') else {}
        else:
            values = info
        if values.get("app_env") == Environment.PRODUCTION and not v:
            warnings.warn("Rate limiting should be enabled in production!")
        return v
    
    # ==========================================
    # MONITORING
    # ==========================================
    sentry_dsn: Optional[str] = Field(default=None, env="SENTRY_DSN")
    sentry_environment: str = Field(default="development", env="SENTRY_ENVIRONMENT")
    sentry_traces_sample_rate: float = Field(default=0.1, env="SENTRY_TRACES_SAMPLE_RATE", ge=0.0, le=1.0)
    
    metrics_enabled: bool = Field(default=False, env="METRICS_ENABLED")
    metrics_port: int = Field(default=9090, env="METRICS_PORT", ge=1, le=65535)
    
    # ==========================================
    # EMAIL SETTINGS
    # ==========================================
    email_enabled: bool = Field(default=False, env="EMAIL_ENABLED")
    email_provider: str = Field(default="smtp", env="EMAIL_PROVIDER")
    email_from: str = Field(default="noreply@example.com", env="EMAIL_FROM")
    
    smtp_host: Optional[str] = Field(default=None, env="SMTP_HOST")
    smtp_port: int = Field(default=587, env="SMTP_PORT")
    smtp_username: Optional[str] = Field(default=None, env="SMTP_USERNAME")
    smtp_password: Optional[SecretStr] = Field(default=None, env="SMTP_PASSWORD")
    smtp_use_tls: bool = Field(default=True, env="SMTP_USE_TLS")
    
    # ==========================================
    # FEATURE FLAGS
    # ==========================================
    feature_registration_enabled: bool = Field(default=True, env="FEATURE_REGISTRATION_ENABLED")
    feature_ai_chat: bool = Field(default=True, env="FEATURE_AI_CHAT")
    feature_analytics: bool = Field(default=True, env="FEATURE_ANALYTICS")
    feature_maintenance_mode: bool = Field(default=False, env="FEATURE_MAINTENANCE_MODE")
    
    # ==========================================
    # MCP INTEGRATION
    # ==========================================
    mcp_enabled: bool = Field(default=False, env="MCP_ENABLED")
    mcp_server_url: Optional[AnyHttpUrl] = Field(default=None, env="MCP_SERVER_URL")
    mcp_api_key: Optional[SecretStr] = Field(default=None, env="MCP_API_KEY")
    
    class Config:
        """Pydantic configuration"""
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        
        # Hide secrets in string representation
        json_encoders = {
            SecretStr: lambda v: v.get_secret_value() if v else None,
        }
    
    def is_production(self) -> bool:
        """Check if running in production"""
        return self.app_env == Environment.PRODUCTION
    
    def is_development(self) -> bool:
        """Check if running in development"""
        return self.app_env == Environment.DEVELOPMENT
    
    def get_database_url(self, hide_password: bool = True) -> str:
        """Get database URL with optional password hiding"""
        if not self.database_url:
            return ""
        
        url = str(self.database_url)
        if hide_password and "@" in url:
            # Hide password in connection string
            parts = url.split("@")
            if ":" in parts[0]:
                user_pass = parts[0].split("//")[1]
                user = user_pass.split(":")[0]
                parts[0] = f"{parts[0].split('//')[0]}//{user}:***"
            url = "@".join(parts)
        
        return url
    
    def get_redis_url(self, hide_password: bool = True) -> str:
        """Get Redis URL with optional password hiding"""
        if not self.redis_url:
            return ""
        
        url = str(self.redis_url)
        if self.redis_password and hide_password:
            url = url.replace(self.redis_password.get_secret_value(), "***")
        
        return url
    
    def validate_production_ready(self) -> List[str]:
        """Validate configuration is production-ready"""
        issues = []
        
        if self.is_production():
            # Security checks
            if self.app_debug:
                issues.append("Debug mode is enabled in production")
            
            if len(self.secret_key.get_secret_value()) < 32:
                issues.append("Secret key is too short")
            
            if self.secret_key.get_secret_value() == self.jwt_secret_key.get_secret_value():
                issues.append("Secret key and JWT secret should be different")
            
            # CORS checks
            if "*" in self.cors_origins:
                issues.append("Wildcard CORS origins in production")
            
            # Database checks
            if "password" in str(self.database_url):
                issues.append("Default database password detected")
            
            # Rate limiting
            if not self.rate_limit_enabled:
                issues.append("Rate limiting is disabled")
            
            # Monitoring
            if not self.sentry_dsn:
                issues.append("Sentry monitoring is not configured")
            
            # Model
            if not self.hugging_face_hub_token:
                issues.append("Hugging Face token is not configured")
        
        return issues


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance.
    This ensures settings are loaded only once.
    """
    return Settings()


def validate_environment():
    """
    Validate environment configuration on startup.
    Raises exception if critical issues found.
    """
    settings = get_settings()
    issues = settings.validate_production_ready()
    
    if issues and settings.is_production():
        print("⚠️  Production configuration issues detected:")
        for issue in issues:
            print(f"   - {issue}")
        
        # Critical issues that should stop production startup
        critical_issues = [i for i in issues if "secret" in i.lower() or "password" in i.lower()]
        if critical_issues:
            raise ValueError(f"Critical configuration issues: {', '.join(critical_issues)}")
    
    return True


# Export commonly used settings
settings = get_settings()