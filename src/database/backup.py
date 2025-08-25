"""
Database backup and restore utilities
"""

import os
import logging
import json
import gzip
import shutil
from datetime import datetime
from typing import Optional, List, Dict, Any
from pathlib import Path
import subprocess

from sqlalchemy import MetaData, Table, select
from sqlalchemy.orm import Session
from sqlalchemy.ext.asyncio import AsyncSession

from .base import Base, engine, async_engine
from .session import get_database_url

logger = logging.getLogger(__name__)


class DatabaseBackup:
    """Database backup utilities."""
    
    def __init__(self, backup_dir: str = "backups"):
        self.backup_dir = Path(backup_dir)
        self.backup_dir.mkdir(exist_ok=True)
    
    def _get_backup_filename(self, prefix: str = "backup") -> str:
        """Generate backup filename with timestamp."""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        return f"{prefix}_{timestamp}.sql"
    
    def backup_schema(self, output_file: Optional[str] = None) -> str:
        """
        Backup database schema only.
        
        Args:
            output_file: Output file path
        
        Returns:
            Path to backup file
        """
        if not output_file:
            output_file = self.backup_dir / self._get_backup_filename("schema")
        else:
            output_file = Path(output_file)
        
        try:
            # Generate CREATE statements for all tables
            metadata = MetaData()
            metadata.reflect(bind=engine)
            
            with open(output_file, 'w') as f:
                # Write header
                f.write(f"-- Database Schema Backup\n")
                f.write(f"-- Generated: {datetime.utcnow().isoformat()}\n")
                f.write(f"-- Tables: {len(metadata.tables)}\n\n")
                
                # Write CREATE statements
                for table in metadata.sorted_tables:
                    f.write(f"-- Table: {table.name}\n")
                    f.write(f"{table.create(engine).compile(engine.dialect)}\n\n")
            
            logger.info(f"Schema backup created: {output_file}")
            return str(output_file)
            
        except Exception as e:
            logger.error(f"Failed to backup schema: {e}")
            raise
    
    def backup_data_json(
        self,
        tables: Optional[List[str]] = None,
        output_file: Optional[str] = None,
        compress: bool = True
    ) -> str:
        """
        Backup data to JSON format.
        
        Args:
            tables: List of table names to backup (None for all)
            output_file: Output file path
            compress: Whether to compress the output
        
        Returns:
            Path to backup file
        """
        if not output_file:
            ext = ".json.gz" if compress else ".json"
            output_file = self.backup_dir / (self._get_backup_filename("data").replace(".sql", ext))
        else:
            output_file = Path(output_file)
        
        try:
            metadata = MetaData()
            metadata.reflect(bind=engine)
            
            backup_data = {
                "metadata": {
                    "timestamp": datetime.utcnow().isoformat(),
                    "tables": []
                },
                "data": {}
            }
            
            with Session(engine) as session:
                # Get tables to backup
                tables_to_backup = tables or list(metadata.tables.keys())
                
                for table_name in tables_to_backup:
                    if table_name not in metadata.tables:
                        logger.warning(f"Table {table_name} not found, skipping")
                        continue
                    
                    table = metadata.tables[table_name]
                    
                    # Fetch all rows
                    result = session.execute(select(table))
                    rows = result.fetchall()
                    
                    # Convert rows to dictionaries
                    table_data = []
                    for row in rows:
                        row_dict = {}
                        for column in table.columns:
                            value = getattr(row, column.name)
                            # Handle special types
                            if isinstance(value, datetime):
                                value = value.isoformat()
                            elif hasattr(value, '__dict__'):
                                value = str(value)
                            row_dict[column.name] = value
                        table_data.append(row_dict)
                    
                    backup_data["data"][table_name] = table_data
                    backup_data["metadata"]["tables"].append({
                        "name": table_name,
                        "row_count": len(table_data)
                    })
                    
                    logger.info(f"Backed up {len(table_data)} rows from {table_name}")
            
            # Write to file
            if compress:
                with gzip.open(output_file, 'wt', encoding='utf-8') as f:
                    json.dump(backup_data, f, indent=2, default=str)
            else:
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(backup_data, f, indent=2, default=str)
            
            logger.info(f"Data backup created: {output_file}")
            return str(output_file)
            
        except Exception as e:
            logger.error(f"Failed to backup data: {e}")
            raise
    
    def backup_full_pg_dump(self, output_file: Optional[str] = None) -> str:
        """
        Full PostgreSQL backup using pg_dump.
        
        Args:
            output_file: Output file path
        
        Returns:
            Path to backup file
        """
        if not output_file:
            output_file = self.backup_dir / self._get_backup_filename("full")
        else:
            output_file = Path(output_file)
        
        try:
            # Parse database URL
            db_url = get_database_url()
            
            # Extract connection parameters
            from urllib.parse import urlparse
            parsed = urlparse(db_url)
            
            # Build pg_dump command
            cmd = [
                "pg_dump",
                "-h", parsed.hostname or "localhost",
                "-p", str(parsed.port or 5432),
                "-U", parsed.username or "postgres",
                "-d", parsed.path.lstrip("/"),
                "-f", str(output_file),
                "--verbose",
                "--clean",
                "--if-exists"
            ]
            
            # Set password via environment variable
            env = os.environ.copy()
            if parsed.password:
                env["PGPASSWORD"] = parsed.password
            
            # Execute pg_dump
            result = subprocess.run(
                cmd,
                env=env,
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                raise Exception(f"pg_dump failed: {result.stderr}")
            
            # Compress if large
            if output_file.stat().st_size > 10 * 1024 * 1024:  # 10MB
                compressed_file = Path(str(output_file) + ".gz")
                with open(output_file, 'rb') as f_in:
                    with gzip.open(compressed_file, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                output_file.unlink()
                output_file = compressed_file
            
            logger.info(f"Full backup created: {output_file}")
            return str(output_file)
            
        except Exception as e:
            logger.error(f"Failed to create full backup: {e}")
            raise
    
    def restore_from_json(
        self,
        backup_file: str,
        tables: Optional[List[str]] = None,
        truncate: bool = True
    ) -> Dict[str, int]:
        """
        Restore data from JSON backup.
        
        Args:
            backup_file: Path to backup file
            tables: List of tables to restore (None for all)
            truncate: Whether to truncate tables before restore
        
        Returns:
            Dictionary with restore statistics
        """
        backup_file = Path(backup_file)
        
        if not backup_file.exists():
            raise FileNotFoundError(f"Backup file not found: {backup_file}")
        
        try:
            # Read backup data
            if backup_file.suffix == ".gz":
                with gzip.open(backup_file, 'rt', encoding='utf-8') as f:
                    backup_data = json.load(f)
            else:
                with open(backup_file, 'r', encoding='utf-8') as f:
                    backup_data = json.load(f)
            
            metadata = MetaData()
            metadata.reflect(bind=engine)
            
            stats = {}
            
            with Session(engine) as session:
                # Get tables to restore
                tables_to_restore = tables or list(backup_data["data"].keys())
                
                for table_name in tables_to_restore:
                    if table_name not in backup_data["data"]:
                        logger.warning(f"Table {table_name} not in backup, skipping")
                        continue
                    
                    if table_name not in metadata.tables:
                        logger.warning(f"Table {table_name} not in database, skipping")
                        continue
                    
                    table = metadata.tables[table_name]
                    table_data = backup_data["data"][table_name]
                    
                    # Truncate table if requested
                    if truncate:
                        session.execute(table.delete())
                        session.commit()
                        logger.info(f"Truncated table {table_name}")
                    
                    # Insert data
                    if table_data:
                        # Convert datetime strings back to datetime objects
                        for row in table_data:
                            for column in table.columns:
                                if column.name in row and row[column.name]:
                                    # Handle datetime columns
                                    if 'datetime' in str(column.type).lower():
                                        try:
                                            row[column.name] = datetime.fromisoformat(row[column.name])
                                        except:
                                            pass
                        
                        session.bulk_insert_mappings(table.class_, table_data)
                        session.commit()
                    
                    stats[table_name] = len(table_data)
                    logger.info(f"Restored {len(table_data)} rows to {table_name}")
            
            logger.info(f"Restore completed: {stats}")
            return stats
            
        except Exception as e:
            logger.error(f"Failed to restore from backup: {e}")
            raise
    
    def restore_from_pg_dump(self, backup_file: str) -> bool:
        """
        Restore from PostgreSQL dump.
        
        Args:
            backup_file: Path to backup file
        
        Returns:
            True if successful
        """
        backup_file = Path(backup_file)
        
        if not backup_file.exists():
            raise FileNotFoundError(f"Backup file not found: {backup_file}")
        
        try:
            # Decompress if needed
            if backup_file.suffix == ".gz":
                decompressed_file = backup_file.with_suffix("")
                with gzip.open(backup_file, 'rb') as f_in:
                    with open(decompressed_file, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                backup_file = decompressed_file
            
            # Parse database URL
            db_url = get_database_url()
            from urllib.parse import urlparse
            parsed = urlparse(db_url)
            
            # Build psql command
            cmd = [
                "psql",
                "-h", parsed.hostname or "localhost",
                "-p", str(parsed.port or 5432),
                "-U", parsed.username or "postgres",
                "-d", parsed.path.lstrip("/"),
                "-f", str(backup_file),
                "--set", "ON_ERROR_STOP=on"
            ]
            
            # Set password via environment variable
            env = os.environ.copy()
            if parsed.password:
                env["PGPASSWORD"] = parsed.password
            
            # Execute psql
            result = subprocess.run(
                cmd,
                env=env,
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                raise Exception(f"psql failed: {result.stderr}")
            
            logger.info(f"Database restored from {backup_file}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to restore from dump: {e}")
            raise
    
    def list_backups(self) -> List[Dict[str, Any]]:
        """
        List available backups.
        
        Returns:
            List of backup information
        """
        backups = []
        
        for file_path in self.backup_dir.glob("*"):
            if file_path.is_file():
                stat = file_path.stat()
                backups.append({
                    "filename": file_path.name,
                    "path": str(file_path),
                    "size": stat.st_size,
                    "created": datetime.fromtimestamp(stat.st_ctime),
                    "type": self._detect_backup_type(file_path)
                })
        
        # Sort by creation time (newest first)
        backups.sort(key=lambda x: x["created"], reverse=True)
        
        return backups
    
    def _detect_backup_type(self, file_path: Path) -> str:
        """Detect backup type from filename."""
        name = file_path.name.lower()
        if "schema" in name:
            return "schema"
        elif "data" in name:
            return "data"
        elif "full" in name:
            return "full"
        elif name.endswith(".sql") or name.endswith(".sql.gz"):
            return "sql"
        elif name.endswith(".json") or name.endswith(".json.gz"):
            return "json"
        else:
            return "unknown"
    
    def cleanup_old_backups(self, keep_days: int = 7) -> int:
        """
        Clean up old backup files.
        
        Args:
            keep_days: Number of days to keep backups
        
        Returns:
            Number of files deleted
        """
        cutoff_date = datetime.utcnow().timestamp() - (keep_days * 86400)
        deleted_count = 0
        
        for file_path in self.backup_dir.glob("*"):
            if file_path.is_file():
                if file_path.stat().st_ctime < cutoff_date:
                    try:
                        file_path.unlink()
                        logger.info(f"Deleted old backup: {file_path.name}")
                        deleted_count += 1
                    except Exception as e:
                        logger.error(f"Failed to delete {file_path.name}: {e}")
        
        if deleted_count > 0:
            logger.info(f"Cleaned up {deleted_count} old backup files")
        
        return deleted_count


class DatabaseMaintenance:
    """Database maintenance utilities."""
    
    @staticmethod
    def vacuum(full: bool = False) -> bool:
        """
        Run VACUUM on PostgreSQL database.
        
        Args:
            full: Whether to run VACUUM FULL
        
        Returns:
            True if successful
        """
        try:
            with engine.connect() as conn:
                conn.execute("COMMIT")  # End any transaction
                if full:
                    conn.execute("VACUUM FULL")
                    logger.info("VACUUM FULL completed")
                else:
                    conn.execute("VACUUM")
                    logger.info("VACUUM completed")
            return True
        except Exception as e:
            logger.error(f"VACUUM failed: {e}")
            return False
    
    @staticmethod
    def analyze() -> bool:
        """
        Run ANALYZE on PostgreSQL database.
        
        Returns:
            True if successful
        """
        try:
            with engine.connect() as conn:
                conn.execute("ANALYZE")
                logger.info("ANALYZE completed")
            return True
        except Exception as e:
            logger.error(f"ANALYZE failed: {e}")
            return False
    
    @staticmethod
    def reindex(table_name: Optional[str] = None) -> bool:
        """
        Reindex database or specific table.
        
        Args:
            table_name: Table to reindex (None for all)
        
        Returns:
            True if successful
        """
        try:
            with engine.connect() as conn:
                if table_name:
                    conn.execute(f"REINDEX TABLE {table_name}")
                    logger.info(f"REINDEX completed for table {table_name}")
                else:
                    conn.execute("REINDEX DATABASE")
                    logger.info("REINDEX completed for database")
            return True
        except Exception as e:
            logger.error(f"REINDEX failed: {e}")
            return False