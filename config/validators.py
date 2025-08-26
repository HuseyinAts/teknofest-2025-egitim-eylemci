"""
Startup validation script for required environment variables and configuration.
"""

import os
import sys
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class ConfigValidator:
    """Validates required configuration and environment variables."""
    
    CRITICAL_VARS = [
        "SECRET_KEY",
        "JWT_SECRET_KEY",
        "DATABASE_URL",
    ]
    
    RECOMMENDED_VARS = [
        "REDIS_URL",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "SENTRY_DSN",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
    ]
    
    PRODUCTION_REQUIRED = [
        "SSL_CERT_PATH",
        "SSL_KEY_PATH",
        "ALLOWED_HOSTS",
        "CORS_ORIGINS",
    ]
    
    @classmethod
    def validate_critical(cls) -> Tuple[bool, List[str]]:
        """Validate critical environment variables."""
        missing = []
        
        for var in cls.CRITICAL_VARS:
            if not os.getenv(var):
                missing.append(var)
        
        # Additional validations
        if os.getenv("SECRET_KEY") and len(os.getenv("SECRET_KEY", "")) < 32:
            missing.append("SECRET_KEY (must be at least 32 characters)")
        
        if os.getenv("SECRET_KEY") == os.getenv("JWT_SECRET_KEY"):
            missing.append("JWT_SECRET_KEY (must be different from SECRET_KEY)")
        
        return len(missing) == 0, missing
    
    @classmethod
    def validate_recommended(cls) -> Tuple[bool, List[str]]:
        """Validate recommended environment variables."""
        missing = []
        
        for var in cls.RECOMMENDED_VARS:
            if not os.getenv(var):
                missing.append(var)
        
        return len(missing) == 0, missing
    
    @classmethod
    def validate_production(cls) -> Tuple[bool, List[str]]:
        """Validate production-specific requirements."""
        if os.getenv("ENVIRONMENT", "development") != "production":
            return True, []
        
        missing = []
        
        for var in cls.PRODUCTION_REQUIRED:
            if not os.getenv(var):
                missing.append(var)
        
        # Check SSL certificates exist
        if os.getenv("SSL_CERT_PATH"):
            if not os.path.exists(os.getenv("SSL_CERT_PATH")):
                missing.append("SSL_CERT_PATH (file not found)")
        
        if os.getenv("SSL_KEY_PATH"):
            if not os.path.exists(os.getenv("SSL_KEY_PATH")):
                missing.append("SSL_KEY_PATH (file not found)")
        
        return len(missing) == 0, missing
    
    @classmethod
    def validate_database_connection(cls) -> Tuple[bool, Optional[str]]:
        """Test database connection."""
        database_url = os.getenv("DATABASE_URL")
        if not database_url:
            return False, "DATABASE_URL not set"
        
        try:
            import asyncpg
            import asyncio
            
            async def test_connection():
                try:
                    conn = await asyncpg.connect(database_url, timeout=5)
                    await conn.close()
                    return True, None
                except Exception as e:
                    return False, str(e)
            
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(test_connection())
            loop.close()
            return result
        except ImportError:
            logger.warning("asyncpg not installed, skipping database connection test")
            return True, None
        except Exception as e:
            return False, str(e)
    
    @classmethod
    def validate_redis_connection(cls) -> Tuple[bool, Optional[str]]:
        """Test Redis connection."""
        redis_url = os.getenv("REDIS_URL")
        if not redis_url:
            return True, None  # Redis is optional
        
        try:
            import redis
            r = redis.from_url(redis_url, socket_connect_timeout=5)
            r.ping()
            return True, None
        except ImportError:
            logger.warning("redis not installed, skipping Redis connection test")
            return True, None
        except Exception as e:
            return False, str(e)
    
    @classmethod
    def run_all_validations(cls) -> bool:
        """Run all validations and report results."""
        all_valid = True
        
        # Critical validations
        valid, missing = cls.validate_critical()
        if not valid:
            logger.error(f"CRITICAL: Missing required environment variables: {', '.join(missing)}")
            all_valid = False
        
        # Recommended validations
        valid, missing = cls.validate_recommended()
        if not valid:
            logger.warning(f"WARNING: Missing recommended environment variables: {', '.join(missing)}")
        
        # Production validations
        valid, missing = cls.validate_production()
        if not valid:
            logger.error(f"CRITICAL: Missing production requirements: {', '.join(missing)}")
            all_valid = False
        
        # Database connection
        valid, error = cls.validate_database_connection()
        if not valid:
            logger.error(f"CRITICAL: Database connection failed: {error}")
            all_valid = False
        
        # Redis connection
        valid, error = cls.validate_redis_connection()
        if not valid:
            logger.warning(f"WARNING: Redis connection failed: {error}")
        
        return all_valid


def main():
    """Main validation entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    validator = ConfigValidator()
    
    logger.info("Starting configuration validation...")
    
    if validator.run_all_validations():
        logger.info("All critical validations passed!")
        sys.exit(0)
    else:
        logger.error("Critical validation failures detected. Please fix before starting the application.")
        sys.exit(1)


if __name__ == "__main__":
    main()