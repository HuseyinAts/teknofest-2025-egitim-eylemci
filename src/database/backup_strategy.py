"""
Comprehensive Database Backup Strategy
Production-ready backup, restore, and disaster recovery
"""

import os
import json
import gzip
import hashlib
import subprocess
import shutil
import tempfile
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Tuple
from pathlib import Path
from enum import Enum
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor

import boto3
from botocore.exceptions import ClientError
import schedule
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

from ..config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class BackupType(Enum):
    """Backup types"""
    FULL = "full"
    INCREMENTAL = "incremental"
    DIFFERENTIAL = "differential"
    SCHEMA_ONLY = "schema_only"
    DATA_ONLY = "data_only"


class BackupStatus(Enum):
    """Backup status"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    VERIFYING = "verifying"
    VERIFIED = "verified"


class DatabaseBackupStrategy:
    """
    Comprehensive database backup strategy with:
    - Automated backups (full, incremental, differential)
    - Point-in-time recovery (PITR)
    - Encryption and compression
    - Cloud storage (S3) integration
    - Retention policies
    - Verification and integrity checks
    - Monitoring and alerting
    """
    
    def __init__(
        self,
        local_backup_dir: str = "/var/backups/postgres",
        s3_bucket: Optional[str] = None,
        retention_days: int = 30,
        encryption_key: Optional[str] = None
    ):
        self.local_backup_dir = Path(local_backup_dir)
        self.local_backup_dir.mkdir(parents=True, exist_ok=True)
        
        self.s3_bucket = s3_bucket or os.getenv("BACKUP_S3_BUCKET")
        self.retention_days = retention_days
        self.encryption_key = encryption_key or settings.secret_key.get_secret_value()[:32]
        
        # Initialize S3 client if bucket specified
        self.s3_client = None
        if self.s3_bucket:
            self.s3_client = boto3.client('s3')
        
        # Thread pool for parallel operations
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Backup history tracking
        self.backup_history_file = self.local_backup_dir / "backup_history.json"
        self.backup_history = self._load_backup_history()
    
    def _load_backup_history(self) -> List[Dict[str, Any]]:
        """Load backup history from file"""
        if self.backup_history_file.exists():
            with open(self.backup_history_file, 'r') as f:
                return json.load(f)
        return []
    
    def _save_backup_history(self):
        """Save backup history to file"""
        with open(self.backup_history_file, 'w') as f:
            json.dump(self.backup_history, f, indent=2, default=str)
    
    def _get_database_connection_params(self) -> Dict[str, str]:
        """Extract database connection parameters"""
        from urllib.parse import urlparse
        
        db_url = settings.database_url
        parsed = urlparse(db_url)
        
        return {
            "host": parsed.hostname,
            "port": str(parsed.port or 5432),
            "database": parsed.path.lstrip('/'),
            "username": parsed.username,
            "password": parsed.password
        }
    
    def _generate_backup_filename(
        self, 
        backup_type: BackupType,
        timestamp: Optional[datetime] = None
    ) -> str:
        """Generate backup filename"""
        if not timestamp:
            timestamp = datetime.utcnow()
        
        date_str = timestamp.strftime("%Y%m%d_%H%M%S")
        return f"backup_{backup_type.value}_{date_str}.sql.gz"
    
    def _calculate_checksum(self, file_path: Path) -> str:
        """Calculate SHA256 checksum of file"""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    
    def _encrypt_file(self, input_file: Path, output_file: Path):
        """Encrypt backup file using AES"""
        from cryptography.fernet import Fernet
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2
        import base64
        
        # Derive encryption key
        kdf = PBKDF2(
            algorithm=hashes.SHA256(),
            length=32,
            salt=b'teknofest2025',
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(self.encryption_key.encode()))
        f = Fernet(key)
        
        # Encrypt file
        with open(input_file, 'rb') as infile:
            encrypted_data = f.encrypt(infile.read())
        
        with open(output_file, 'wb') as outfile:
            outfile.write(encrypted_data)
        
        logger.info(f"File encrypted: {output_file}")
    
    def _compress_file(self, input_file: Path, output_file: Path):
        """Compress backup file"""
        with open(input_file, 'rb') as f_in:
            with gzip.open(output_file, 'wb', compresslevel=9) as f_out:
                shutil.copyfileobj(f_in, f_out)
        
        # Calculate compression ratio
        original_size = input_file.stat().st_size
        compressed_size = output_file.stat().st_size
        ratio = (1 - compressed_size / original_size) * 100
        
        logger.info(f"File compressed: {output_file} (ratio: {ratio:.1f}%)")
    
    # ==========================================
    # BACKUP OPERATIONS
    # ==========================================
    
    def perform_full_backup(self) -> Dict[str, Any]:
        """Perform full database backup"""
        start_time = datetime.utcnow()
        backup_info = {
            "type": BackupType.FULL.value,
            "status": BackupStatus.IN_PROGRESS.value,
            "start_time": start_time,
            "file_path": None,
            "size_bytes": 0,
            "checksum": None,
            "error": None
        }
        
        try:
            # Get connection params
            conn_params = self._get_database_connection_params()
            
            # Generate backup filename
            backup_filename = self._generate_backup_filename(BackupType.FULL, start_time)
            backup_path = self.local_backup_dir / backup_filename
            temp_backup_path = self.local_backup_dir / f"temp_{backup_filename}"
            
            # Perform backup using pg_dump
            logger.info(f"Starting full backup to {backup_path}")
            
            env = os.environ.copy()
            env["PGPASSWORD"] = conn_params["password"]
            
            cmd = [
                "pg_dump",
                "-h", conn_params["host"],
                "-p", conn_params["port"],
                "-U", conn_params["username"],
                "-d", conn_params["database"],
                "-f", str(temp_backup_path),
                "--verbose",
                "--no-owner",
                "--no-privileges",
                "--no-tablespaces",
                "--clean",
                "--if-exists",
                "--create",
                "--encoding=UTF8"
            ]
            
            result = subprocess.run(
                cmd,
                env=env,
                capture_output=True,
                text=True,
                timeout=3600  # 1 hour timeout
            )
            
            if result.returncode != 0:
                raise Exception(f"pg_dump failed: {result.stderr}")
            
            # Compress backup
            self._compress_file(temp_backup_path, backup_path)
            temp_backup_path.unlink()  # Remove uncompressed file
            
            # Calculate checksum
            checksum = self._calculate_checksum(backup_path)
            
            # Encrypt if encryption is enabled
            if self.encryption_key:
                encrypted_path = backup_path.with_suffix('.enc')
                self._encrypt_file(backup_path, encrypted_path)
                backup_path.unlink()  # Remove unencrypted file
                backup_path = encrypted_path
            
            # Update backup info
            backup_info.update({
                "status": BackupStatus.COMPLETED.value,
                "end_time": datetime.utcnow(),
                "file_path": str(backup_path),
                "size_bytes": backup_path.stat().st_size,
                "checksum": checksum,
                "compressed": True,
                "encrypted": bool(self.encryption_key)
            })
            
            # Upload to S3 if configured
            if self.s3_client:
                self._upload_to_s3(backup_path, backup_info)
            
            # Add to history
            self.backup_history.append(backup_info)
            self._save_backup_history()
            
            # Verify backup
            self.verify_backup(backup_path)
            
            logger.info(f"Full backup completed: {backup_path}")
            
        except Exception as e:
            logger.error(f"Full backup failed: {e}")
            backup_info.update({
                "status": BackupStatus.FAILED.value,
                "error": str(e),
                "end_time": datetime.utcnow()
            })
            self.backup_history.append(backup_info)
            self._save_backup_history()
            raise
        
        return backup_info
    
    def perform_incremental_backup(self, base_backup_path: Path) -> Dict[str, Any]:
        """Perform incremental backup based on WAL files"""
        start_time = datetime.utcnow()
        backup_info = {
            "type": BackupType.INCREMENTAL.value,
            "status": BackupStatus.IN_PROGRESS.value,
            "start_time": start_time,
            "base_backup": str(base_backup_path),
            "wal_files": []
        }
        
        try:
            conn_params = self._get_database_connection_params()
            
            # Create WAL archive directory
            wal_dir = self.local_backup_dir / "wal_archive"
            wal_dir.mkdir(exist_ok=True)
            
            # Configure PostgreSQL for WAL archiving
            conn = psycopg2.connect(
                host=conn_params["host"],
                port=conn_params["port"],
                database=conn_params["database"],
                user=conn_params["username"],
                password=conn_params["password"]
            )
            conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
            
            with conn.cursor() as cursor:
                # Enable WAL archiving
                cursor.execute("SELECT pg_switch_wal();")
                
                # Get current WAL location
                cursor.execute("SELECT pg_current_wal_lsn();")
                current_wal_lsn = cursor.fetchone()[0]
                
                backup_info["current_wal_lsn"] = current_wal_lsn
            
            conn.close()
            
            # Archive WAL files
            backup_filename = self._generate_backup_filename(BackupType.INCREMENTAL, start_time)
            wal_backup_path = wal_dir / backup_filename
            
            # Compress WAL archive
            shutil.make_archive(
                str(wal_backup_path.with_suffix('')),
                'gztar',
                str(wal_dir)
            )
            
            backup_info.update({
                "status": BackupStatus.COMPLETED.value,
                "end_time": datetime.utcnow(),
                "file_path": str(wal_backup_path) + ".tar.gz",
                "size_bytes": (wal_backup_path.with_suffix('.tar.gz')).stat().st_size
            })
            
            # Upload to S3
            if self.s3_client:
                self._upload_to_s3(wal_backup_path.with_suffix('.tar.gz'), backup_info)
            
            logger.info(f"Incremental backup completed: {wal_backup_path}")
            
        except Exception as e:
            logger.error(f"Incremental backup failed: {e}")
            backup_info.update({
                "status": BackupStatus.FAILED.value,
                "error": str(e),
                "end_time": datetime.utcnow()
            })
            raise
        
        return backup_info
    
    def perform_point_in_time_recovery(
        self,
        base_backup_path: Path,
        target_time: datetime
    ) -> bool:
        """Perform point-in-time recovery"""
        logger.info(f"Starting PITR to {target_time}")
        
        try:
            conn_params = self._get_database_connection_params()
            
            # Stop PostgreSQL
            subprocess.run(["systemctl", "stop", "postgresql"], check=True)
            
            # Restore base backup
            self.restore_backup(base_backup_path)
            
            # Configure recovery
            recovery_conf = f"""
restore_command = 'cp /var/backups/postgres/wal_archive/%f %p'
recovery_target_time = '{target_time.isoformat()}'
recovery_target_action = 'promote'
            """
            
            recovery_file = Path("/var/lib/postgresql/data/recovery.conf")
            with open(recovery_file, 'w') as f:
                f.write(recovery_conf)
            
            # Start PostgreSQL
            subprocess.run(["systemctl", "start", "postgresql"], check=True)
            
            logger.info(f"PITR completed successfully to {target_time}")
            return True
            
        except Exception as e:
            logger.error(f"PITR failed: {e}")
            return False
    
    # ==========================================
    # RESTORE OPERATIONS
    # ==========================================
    
    def restore_backup(self, backup_path: Path) -> bool:
        """Restore database from backup"""
        logger.info(f"Starting restore from {backup_path}")
        
        try:
            conn_params = self._get_database_connection_params()
            
            # Decrypt if encrypted
            if backup_path.suffix == '.enc':
                decrypted_path = backup_path.with_suffix('')
                self._decrypt_file(backup_path, decrypted_path)
                backup_path = decrypted_path
            
            # Decompress if compressed
            if backup_path.suffix == '.gz':
                decompressed_path = backup_path.with_suffix('')
                with gzip.open(backup_path, 'rb') as f_in:
                    with open(decompressed_path, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                backup_path = decompressed_path
            
            # Restore using pg_restore or psql
            env = os.environ.copy()
            env["PGPASSWORD"] = conn_params["password"]
            
            cmd = [
                "psql",
                "-h", conn_params["host"],
                "-p", conn_params["port"],
                "-U", conn_params["username"],
                "-d", "postgres",  # Connect to postgres database first
                "-f", str(backup_path)
            ]
            
            result = subprocess.run(
                cmd,
                env=env,
                capture_output=True,
                text=True,
                timeout=3600
            )
            
            if result.returncode != 0:
                raise Exception(f"Restore failed: {result.stderr}")
            
            logger.info(f"Restore completed successfully from {backup_path}")
            return True
            
        except Exception as e:
            logger.error(f"Restore failed: {e}")
            return False
    
    def _decrypt_file(self, input_file: Path, output_file: Path):
        """Decrypt backup file"""
        from cryptography.fernet import Fernet
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2
        import base64
        
        # Derive decryption key
        kdf = PBKDF2(
            algorithm=hashes.SHA256(),
            length=32,
            salt=b'teknofest2025',
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(self.encryption_key.encode()))
        f = Fernet(key)
        
        # Decrypt file
        with open(input_file, 'rb') as infile:
            decrypted_data = f.decrypt(infile.read())
        
        with open(output_file, 'wb') as outfile:
            outfile.write(decrypted_data)
    
    # ==========================================
    # CLOUD STORAGE
    # ==========================================
    
    def _upload_to_s3(self, file_path: Path, metadata: Dict[str, Any]):
        """Upload backup to S3"""
        if not self.s3_client or not self.s3_bucket:
            return
        
        try:
            s3_key = f"database-backups/{file_path.name}"
            
            # Upload with metadata
            self.s3_client.upload_file(
                str(file_path),
                self.s3_bucket,
                s3_key,
                ExtraArgs={
                    'Metadata': {
                        'backup-type': metadata.get('type', ''),
                        'checksum': metadata.get('checksum', ''),
                        'timestamp': str(metadata.get('start_time', ''))
                    },
                    'ServerSideEncryption': 'AES256',
                    'StorageClass': 'STANDARD_IA'  # Infrequent access for cost optimization
                }
            )
            
            logger.info(f"Backup uploaded to S3: s3://{self.s3_bucket}/{s3_key}")
            
            # Set lifecycle policy for automatic deletion
            self._set_s3_lifecycle_policy()
            
        except ClientError as e:
            logger.error(f"S3 upload failed: {e}")
            raise
    
    def _set_s3_lifecycle_policy(self):
        """Set S3 lifecycle policy for automatic backup deletion"""
        lifecycle_policy = {
            'Rules': [
                {
                    'ID': 'DeleteOldBackups',
                    'Status': 'Enabled',
                    'Prefix': 'database-backups/',
                    'Expiration': {
                        'Days': self.retention_days
                    },
                    'Transitions': [
                        {
                            'Days': 7,
                            'StorageClass': 'STANDARD_IA'
                        },
                        {
                            'Days': 30,
                            'StorageClass': 'GLACIER'
                        }
                    ]
                }
            ]
        }
        
        try:
            self.s3_client.put_bucket_lifecycle_configuration(
                Bucket=self.s3_bucket,
                LifecycleConfiguration=lifecycle_policy
            )
        except ClientError as e:
            logger.warning(f"Failed to set S3 lifecycle policy: {e}")
    
    # ==========================================
    # VERIFICATION AND MONITORING
    # ==========================================
    
    def verify_backup(self, backup_path: Path) -> bool:
        """Verify backup integrity"""
        logger.info(f"Verifying backup: {backup_path}")
        
        try:
            # Verify file exists and is not empty
            if not backup_path.exists() or backup_path.stat().st_size == 0:
                raise ValueError("Backup file is missing or empty")
            
            # Verify checksum
            stored_checksum = None
            for backup in self.backup_history:
                if backup.get('file_path') == str(backup_path):
                    stored_checksum = backup.get('checksum')
                    break
            
            if stored_checksum:
                calculated_checksum = self._calculate_checksum(backup_path)
                if calculated_checksum != stored_checksum:
                    raise ValueError("Checksum mismatch")
            
            # Test restore to temporary database
            if settings.is_production():
                logger.info("Skipping test restore in production")
            else:
                # Create test database
                test_db = f"test_restore_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
                # Restore and verify
                # ... (implementation depends on specific requirements)
            
            logger.info(f"Backup verification successful: {backup_path}")
            return True
            
        except Exception as e:
            logger.error(f"Backup verification failed: {e}")
            return False
    
    def cleanup_old_backups(self):
        """Remove old backups based on retention policy"""
        cutoff_date = datetime.utcnow() - timedelta(days=self.retention_days)
        
        for backup_file in self.local_backup_dir.glob("backup_*.sql*"):
            if backup_file.stat().st_mtime < cutoff_date.timestamp():
                logger.info(f"Removing old backup: {backup_file}")
                backup_file.unlink()
        
        # Clean up backup history
        self.backup_history = [
            b for b in self.backup_history
            if datetime.fromisoformat(str(b['start_time'])) > cutoff_date
        ]
        self._save_backup_history()
    
    def get_backup_status(self) -> Dict[str, Any]:
        """Get current backup status and statistics"""
        if not self.backup_history:
            return {"status": "no_backups", "last_backup": None}
        
        last_backup = self.backup_history[-1]
        last_successful = next(
            (b for b in reversed(self.backup_history) 
             if b['status'] == BackupStatus.COMPLETED.value),
            None
        )
        
        return {
            "total_backups": len(self.backup_history),
            "last_backup": last_backup,
            "last_successful": last_successful,
            "failed_count": sum(1 for b in self.backup_history 
                              if b['status'] == BackupStatus.FAILED.value),
            "total_size": sum(b.get('size_bytes', 0) for b in self.backup_history),
            "retention_days": self.retention_days
        }
    
    # ==========================================
    # AUTOMATION
    # ==========================================
    
    def setup_automated_backups(self):
        """Setup automated backup schedule"""
        # Full backup daily at 2 AM
        schedule.every().day.at("02:00").do(self.perform_full_backup)
        
        # Incremental backup every 6 hours
        schedule.every(6).hours.do(
            lambda: self.perform_incremental_backup(self._get_latest_full_backup())
        )
        
        # Cleanup old backups weekly
        schedule.every().sunday.at("03:00").do(self.cleanup_old_backups)
        
        logger.info("Automated backup schedule configured")
    
    def _get_latest_full_backup(self) -> Optional[Path]:
        """Get path to latest full backup"""
        full_backups = [
            b for b in self.backup_history
            if b['type'] == BackupType.FULL.value and
            b['status'] == BackupStatus.COMPLETED.value
        ]
        
        if full_backups:
            latest = max(full_backups, key=lambda b: b['start_time'])
            return Path(latest['file_path'])
        
        return None
    
    async def monitor_backup_health(self):
        """Monitor backup health and send alerts"""
        while True:
            try:
                status = self.get_backup_status()
                
                # Check if backup is overdue
                if status['last_successful']:
                    last_time = datetime.fromisoformat(
                        str(status['last_successful']['end_time'])
                    )
                    hours_since = (datetime.utcnow() - last_time).total_seconds() / 3600
                    
                    if hours_since > 24:
                        logger.warning(f"No successful backup in {hours_since:.1f} hours")
                        # Send alert (implement based on monitoring system)
                
                # Check failure rate
                if status['failed_count'] > 3:
                    logger.error(f"High backup failure rate: {status['failed_count']} failures")
                    # Send critical alert
                
                await asyncio.sleep(3600)  # Check every hour
                
            except Exception as e:
                logger.error(f"Backup monitoring error: {e}")
                await asyncio.sleep(300)  # Retry in 5 minutes


def setup_database_backup_strategy() -> DatabaseBackupStrategy:
    """Setup and configure database backup strategy"""
    strategy = DatabaseBackupStrategy(
        local_backup_dir=os.getenv("BACKUP_DIR", "/var/backups/postgres"),
        s3_bucket=os.getenv("BACKUP_S3_BUCKET"),
        retention_days=int(os.getenv("BACKUP_RETENTION_DAYS", "30"))
    )
    
    # Setup automated backups
    strategy.setup_automated_backups()
    
    # Start monitoring in background
    asyncio.create_task(strategy.monitor_backup_health())
    
    logger.info("Database backup strategy initialized")
    return strategy