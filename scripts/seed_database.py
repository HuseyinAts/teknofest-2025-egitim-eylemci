#!/usr/bin/env python
"""
Production-ready database seeding script for TEKNOFEST 2025 Education Platform

Usage:
    python scripts/seed_database.py --env production
    python scripts/seed_database.py --env staging
    python scripts/seed_database.py --env development
    python scripts/seed_database.py --clear  # Clear all data (dev only)
    python scripts/seed_database.py --stats  # Show seed statistics
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.database.seeds import DatabaseSeeder, seed_database, clear_database, get_seed_statistics
from src.config import get_settings
from src.database.base import init_db, get_db_context

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def validate_environment(env: str) -> bool:
    """Validate environment parameter"""
    valid_envs = ['development', 'staging', 'production']
    return env in valid_envs


def confirm_action(message: str) -> bool:
    """Ask for user confirmation"""
    response = input(f"\n{message} (yes/no): ").lower().strip()
    return response in ['yes', 'y']


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(
        description='Database seeding utility for TEKNOFEST 2025 Education Platform'
    )
    
    parser.add_argument(
        '--env',
        type=str,
        choices=['development', 'staging', 'production'],
        help='Target environment for seeding'
    )
    
    parser.add_argument(
        '--clear',
        action='store_true',
        help='Clear all data before seeding (development only)'
    )
    
    parser.add_argument(
        '--stats',
        action='store_true',
        help='Show current seed statistics'
    )
    
    parser.add_argument(
        '--force',
        action='store_true',
        help='Skip confirmation prompts'
    )
    
    parser.add_argument(
        '--init-db',
        action='store_true',
        help='Initialize database tables before seeding'
    )
    
    args = parser.parse_args()
    
    try:
        settings = get_settings()
        
        # Show statistics
        if args.stats:
            logger.info("Fetching seed statistics...")
            stats = get_seed_statistics()
            
            print("\n" + "="*50)
            print("DATABASE SEED STATISTICS")
            print("="*50)
            for entity, count in stats.items():
                print(f"{entity.capitalize()}: {count}")
            print("="*50 + "\n")
            return
        
        # Clear database
        if args.clear:
            if settings.ENVIRONMENT == "production":
                logger.error("Cannot clear data in production environment!")
                sys.exit(1)
            
            if not args.force:
                if not confirm_action("WARNING: This will DELETE ALL DATA. Are you sure?"):
                    logger.info("Operation cancelled")
                    return
            
            logger.info("Clearing all database data...")
            clear_database()
            logger.info("Database cleared successfully")
            return
        
        # Determine environment
        if args.env:
            environment = args.env
        else:
            environment = settings.ENVIRONMENT
            logger.info(f"Using environment from settings: {environment}")
        
        # Validate environment
        if not validate_environment(environment):
            logger.error(f"Invalid environment: {environment}")
            sys.exit(1)
        
        # Production safeguards
        if environment == "production":
            if not args.force:
                print("\n" + "!"*50)
                print("WARNING: You are about to seed the PRODUCTION database!")
                print("This should only be done during initial setup.")
                print("!"*50)
                
                if not confirm_action("Do you want to continue?"):
                    logger.info("Operation cancelled")
                    return
        
        # Initialize database if requested
        if args.init_db:
            logger.info("Initializing database tables...")
            init_db()
            logger.info("Database tables initialized")
        
        # Perform seeding
        logger.info(f"Starting database seeding for environment: {environment}")
        
        seeder = DatabaseSeeder()
        
        # Show what will be created
        if environment == "production":
            print("\nProduction seeding will create:")
            print("- Admin user (admin@teknofest.com)")
            print("- Support user (support@teknofest.com)")
            print("- Demo teacher (demo.teacher@teknofest.com)")
            print("- Demo student (demo.student@teknofest.com)")
            print("- 30 achievements")
            print("- 15 learning paths with modules")
            print("- Demo progress data\n")
        elif environment == "staging":
            print("\nStaging seeding will create:")
            print("- Admin and 50 test users")
            print("- 20 achievements")
            print("- 20 learning paths with modules")
            print("- User enrollments and progress")
            print("- Study sessions\n")
        else:  # development
            print("\nDevelopment seeding will create:")
            print("- Admin, teacher, and 100 test users")
            print("- 30 achievements")
            print("- 50 published + 10 draft learning paths")
            print("- Full user data (enrollments, progress, sessions)")
            print("- Notifications and audit logs\n")
        
        if not args.force and not confirm_action("Proceed with seeding?"):
            logger.info("Operation cancelled")
            return
        
        # Execute seeding
        seeder.seed_all(environment)
        
        # Show results
        stats = seeder.get_seed_stats()
        
        print("\n" + "="*50)
        print("SEEDING COMPLETED SUCCESSFULLY")
        print("="*50)
        print(f"Environment: {environment}")
        print("\nCreated entities:")
        for entity, count in stats.items():
            print(f"  {entity.capitalize()}: {count}")
        print("="*50 + "\n")
        
        # Show login credentials for production
        if environment == "production":
            print("\nDefault Login Credentials:")
            print("-" * 30)
            print("Admin:")
            print("  Email: admin@teknofest.com")
            print("  Password: Admin123!")
            print("\nSupport:")
            print("  Email: support@teknofest.com")
            print("  Password: Support123!")
            print("\nDemo Teacher:")
            print("  Email: demo.teacher@teknofest.com")
            print("  Password: DemoTeacher123!")
            print("\nDemo Student:")
            print("  Email: demo.student@teknofest.com")
            print("  Password: DemoStudent123!")
            print("-" * 30)
            print("\nIMPORTANT: Change these passwords immediately!")
        
    except KeyboardInterrupt:
        logger.info("\nOperation cancelled by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Seeding failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()