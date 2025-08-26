#!/usr/bin/env python3
"""
Database Migration Script: SQLite to PostgreSQL
TEKNOFEST 2025 - Production Migration

This script migrates data from SQLite to PostgreSQL for production deployment.
"""

import os
import sys
import asyncio
import logging
from pathlib import Path
from datetime import datetime
import json

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from sqlalchemy import create_engine, text, inspect
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
import asyncpg
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DatabaseMigrator:
    """Handle migration from SQLite to PostgreSQL."""
    
    def __init__(self):
        self.sqlite_url = "sqlite:///./teknofest.db"
        self.postgres_url = os.getenv(
            "DATABASE_URL",
            "postgresql://teknofest:password@localhost:5432/teknofest_db"
        )
        self.async_postgres_url = os.getenv(
            "ASYNC_DATABASE_URL",
            "postgresql+asyncpg://teknofest:password@localhost:5432/teknofest_db"
        )
        
    async def check_postgres_connection(self):
        """Verify PostgreSQL connection."""
        try:
            # Parse connection string
            parts = self.postgres_url.replace("postgresql://", "").split("@")
            user_pass = parts[0].split(":")
            host_db = parts[1].split("/")
            
            conn = await asyncpg.connect(
                user=user_pass[0],
                password=user_pass[1],
                host=host_db[0].split(":")[0],
                port=int(host_db[0].split(":")[1]) if ":" in host_db[0] else 5432,
                database=host_db[1]
            )
            
            version = await conn.fetchval("SELECT version();")
            logger.info(f"✅ Connected to PostgreSQL: {version}")
            await conn.close()
            return True
            
        except Exception as e:
            logger.error(f"❌ PostgreSQL connection failed: {e}")
            return False
    
    async def create_postgres_schema(self):
        """Create PostgreSQL schema and tables."""
        try:
            engine = create_async_engine(self.async_postgres_url, echo=False)
            
            async with engine.begin() as conn:
                # Import models
                from src.database.models import Base
                
                # Create all tables
                await conn.run_sync(Base.metadata.create_all)
                logger.info("✅ PostgreSQL schema created")
                
            await engine.dispose()
            return True
            
        except Exception as e:
            logger.error(f"❌ Schema creation failed: {e}")
            return False
    
    def export_sqlite_data(self):
        """Export data from SQLite to JSON."""
        try:
            # Check if SQLite database exists
            if not Path(self.sqlite_url.replace("sqlite:///", "")).exists():
                logger.warning("⚠️ SQLite database not found. Skipping data export.")
                return {}
                
            engine = create_engine(self.sqlite_url)
            inspector = inspect(engine)
            tables = inspector.get_table_names()
            
            data = {}
            
            with engine.connect() as conn:
                for table in tables:
                    if table == "alembic_version":
                        continue
                        
                    result = conn.execute(text(f"SELECT * FROM {table}"))
                    rows = [dict(row._mapping) for row in result]
                    
                    # Convert datetime objects to strings
                    for row in rows:
                        for key, value in row.items():
                            if isinstance(value, datetime):
                                row[key] = value.isoformat()
                    
                    data[table] = rows
                    logger.info(f"  Exported {len(rows)} rows from {table}")
            
            # Save backup
            backup_file = f"backup_sqlite_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(backup_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            
            logger.info(f"✅ Data exported to {backup_file}")
            return data
            
        except Exception as e:
            logger.error(f"❌ Data export failed: {e}")
            return {}
    
    async def import_to_postgres(self, data):
        """Import data to PostgreSQL."""
        if not data:
            logger.info("ℹ️ No data to import")
            return True
            
        try:
            engine = create_async_engine(self.async_postgres_url, echo=False)
            
            async with AsyncSession(engine) as session:
                for table_name, rows in data.items():
                    if not rows:
                        continue
                    
                    # Construct INSERT statement
                    for row in rows:
                        # Convert ISO datetime strings back to datetime
                        for key, value in row.items():
                            if isinstance(value, str) and 'T' in value:
                                try:
                                    row[key] = datetime.fromisoformat(value)
                                except:
                                    pass
                        
                        # Insert row
                        await session.execute(
                            text(f"""
                                INSERT INTO {table_name} ({', '.join(row.keys())})
                                VALUES ({', '.join([f':{k}' for k in row.keys()])})
                                ON CONFLICT DO NOTHING
                            """),
                            row
                        )
                    
                    logger.info(f"  Imported {len(rows)} rows to {table_name}")
                
                await session.commit()
            
            await engine.dispose()
            logger.info("✅ Data imported to PostgreSQL")
            return True
            
        except Exception as e:
            logger.error(f"❌ Data import failed: {e}")
            return False
    
    async def run_migration(self):
        """Run complete migration process."""
        logger.info("=" * 60)
        logger.info("Starting Database Migration: SQLite → PostgreSQL")
        logger.info("=" * 60)
        
        # Step 1: Check PostgreSQL connection
        logger.info("\n📌 Step 1: Checking PostgreSQL connection...")
        if not await self.check_postgres_connection():
            logger.error("Migration aborted: PostgreSQL not available")
            return False
        
        # Step 2: Create PostgreSQL schema
        logger.info("\n📌 Step 2: Creating PostgreSQL schema...")
        if not await self.create_postgres_schema():
            logger.error("Migration aborted: Schema creation failed")
            return False
        
        # Step 3: Export SQLite data
        logger.info("\n📌 Step 3: Exporting SQLite data...")
        data = self.export_sqlite_data()
        
        # Step 4: Import to PostgreSQL
        logger.info("\n📌 Step 4: Importing data to PostgreSQL...")
        if not await self.import_to_postgres(data):
            logger.error("Migration failed: Data import error")
            return False
        
        # Step 5: Update configuration
        logger.info("\n📌 Step 5: Updating configuration...")
        env_file = Path(".env")
        if env_file.exists():
            with open(env_file, 'r') as f:
                lines = f.readlines()
            
            updated_lines = []
            for line in lines:
                if line.startswith("DATABASE_URL=") and "sqlite" in line:
                    updated_lines.append(f"DATABASE_URL={self.postgres_url}\n")
                elif line.startswith("ASYNC_DATABASE_URL=") and "sqlite" in line:
                    updated_lines.append(f"ASYNC_DATABASE_URL={self.async_postgres_url}\n")
                else:
                    updated_lines.append(line)
            
            with open(env_file, 'w') as f:
                f.writelines(updated_lines)
            
            logger.info("✅ Configuration updated")
        
        logger.info("\n" + "=" * 60)
        logger.info("✅ Migration completed successfully!")
        logger.info("=" * 60)
        
        logger.info("\n📝 Next steps:")
        logger.info("1. Verify data integrity in PostgreSQL")
        logger.info("2. Update docker-compose.yml to use PostgreSQL")
        logger.info("3. Test application with new database")
        logger.info("4. Remove or archive SQLite database file")
        
        return True

async def main():
    """Main migration entry point."""
    migrator = DatabaseMigrator()
    success = await migrator.run_migration()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    asyncio.run(main())