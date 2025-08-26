#!/bin/bash

# Database Backup Script
# TEKNOFEST 2025 - Production Database Backup
# Run this script via cron for automated backups

set -e

# Configuration
BACKUP_DIR="${BACKUP_DIR:-/var/backups/postgres}"
DB_HOST="${DB_HOST:-localhost}"
DB_PORT="${DB_PORT:-5432}"
DB_NAME="${DB_NAME:-teknofest_db}"
DB_USER="${DB_USER:-teknofest}"
DB_PASSWORD="${DB_PASSWORD}"
S3_BUCKET="${BACKUP_S3_BUCKET}"
RETENTION_DAYS="${BACKUP_RETENTION_DAYS:-30}"
BACKUP_TYPE="${1:-full}"  # full, incremental, or schema

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[$(date '+%Y-%m-%d %H:%M:%S')] ERROR:${NC} $1" >&2
}

warning() {
    echo -e "${YELLOW}[$(date '+%Y-%m-%d %H:%M:%S')] WARNING:${NC} $1"
}

# Create backup directory if it doesn't exist
mkdir -p "$BACKUP_DIR"
mkdir -p "$BACKUP_DIR/logs"

# Log file
LOG_FILE="$BACKUP_DIR/logs/backup_$(date +%Y%m%d).log"
exec 1> >(tee -a "$LOG_FILE")
exec 2>&1

log "=========================================="
log "Starting database backup: $BACKUP_TYPE"
log "=========================================="

# Check prerequisites
check_requirements() {
    log "Checking requirements..."
    
    # Check if pg_dump is available
    if ! command -v pg_dump &> /dev/null; then
        error "pg_dump not found. Please install PostgreSQL client tools."
        exit 1
    fi
    
    # Check if AWS CLI is available (if S3 backup is enabled)
    if [ -n "$S3_BUCKET" ]; then
        if ! command -v aws &> /dev/null; then
            error "AWS CLI not found. Please install AWS CLI for S3 backups."
            exit 1
        fi
    fi
    
    # Check database connectivity
    export PGPASSWORD="$DB_PASSWORD"
    if ! psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -c "SELECT 1;" &> /dev/null; then
        error "Cannot connect to database"
        exit 1
    fi
    
    log "All requirements met"
}

# Perform full backup
perform_full_backup() {
    local timestamp=$(date +%Y%m%d_%H%M%S)
    local backup_file="$BACKUP_DIR/backup_full_${timestamp}.sql"
    local compressed_file="${backup_file}.gz"
    
    log "Performing full backup to $compressed_file"
    
    # Export password for pg_dump
    export PGPASSWORD="$DB_PASSWORD"
    
    # Perform backup
    pg_dump \
        -h "$DB_HOST" \
        -p "$DB_PORT" \
        -U "$DB_USER" \
        -d "$DB_NAME" \
        --verbose \
        --no-owner \
        --no-privileges \
        --no-tablespaces \
        --clean \
        --if-exists \
        --create \
        --encoding=UTF8 \
        -f "$backup_file" 2>&1 | while read line; do
            log "pg_dump: $line"
        done
    
    if [ ${PIPESTATUS[0]} -ne 0 ]; then
        error "pg_dump failed"
        rm -f "$backup_file"
        exit 1
    fi
    
    # Compress backup
    log "Compressing backup..."
    gzip -9 "$backup_file"
    
    # Calculate checksum
    local checksum=$(sha256sum "$compressed_file" | cut -d' ' -f1)
    echo "$checksum" > "${compressed_file}.sha256"
    log "Checksum: $checksum"
    
    # Get file size
    local size=$(du -h "$compressed_file" | cut -f1)
    log "Backup size: $size"
    
    # Upload to S3 if configured
    if [ -n "$S3_BUCKET" ]; then
        upload_to_s3 "$compressed_file"
    fi
    
    # Verify backup
    verify_backup "$compressed_file"
    
    log "Full backup completed: $compressed_file"
    echo "$compressed_file"
}

# Perform incremental backup (WAL-based)
perform_incremental_backup() {
    local timestamp=$(date +%Y%m%d_%H%M%S)
    local wal_dir="$BACKUP_DIR/wal_archive"
    local backup_file="$BACKUP_DIR/backup_incremental_${timestamp}.tar.gz"
    
    log "Performing incremental backup (WAL archive)"
    
    # Create WAL archive directory
    mkdir -p "$wal_dir"
    
    # Switch WAL file
    export PGPASSWORD="$DB_PASSWORD"
    psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" \
        -c "SELECT pg_switch_wal();" &> /dev/null
    
    # Archive WAL files
    log "Archiving WAL files to $backup_file"
    tar czf "$backup_file" -C "$BACKUP_DIR" wal_archive/
    
    # Calculate checksum
    local checksum=$(sha256sum "$backup_file" | cut -d' ' -f1)
    echo "$checksum" > "${backup_file}.sha256"
    
    # Upload to S3
    if [ -n "$S3_BUCKET" ]; then
        upload_to_s3 "$backup_file"
    fi
    
    log "Incremental backup completed: $backup_file"
    echo "$backup_file"
}

# Perform schema-only backup
perform_schema_backup() {
    local timestamp=$(date +%Y%m%d_%H%M%S)
    local backup_file="$BACKUP_DIR/backup_schema_${timestamp}.sql"
    local compressed_file="${backup_file}.gz"
    
    log "Performing schema-only backup"
    
    export PGPASSWORD="$DB_PASSWORD"
    
    pg_dump \
        -h "$DB_HOST" \
        -p "$DB_PORT" \
        -U "$DB_USER" \
        -d "$DB_NAME" \
        --schema-only \
        --verbose \
        --no-owner \
        --no-privileges \
        -f "$backup_file"
    
    if [ $? -ne 0 ]; then
        error "Schema backup failed"
        rm -f "$backup_file"
        exit 1
    fi
    
    # Compress
    gzip -9 "$backup_file"
    
    # Upload to S3
    if [ -n "$S3_BUCKET" ]; then
        upload_to_s3 "$compressed_file"
    fi
    
    log "Schema backup completed: $compressed_file"
    echo "$compressed_file"
}

# Upload backup to S3
upload_to_s3() {
    local file_path="$1"
    local file_name=$(basename "$file_path")
    local s3_path="s3://${S3_BUCKET}/database-backups/${file_name}"
    
    log "Uploading to S3: $s3_path"
    
    aws s3 cp "$file_path" "$s3_path" \
        --storage-class STANDARD_IA \
        --server-side-encryption AES256 \
        --metadata "backup-type=${BACKUP_TYPE},timestamp=$(date -Iseconds)"
    
    if [ $? -eq 0 ]; then
        log "Upload successful: $s3_path"
        
        # Also upload checksum if it exists
        if [ -f "${file_path}.sha256" ]; then
            aws s3 cp "${file_path}.sha256" "${s3_path}.sha256"
        fi
    else
        error "S3 upload failed"
        return 1
    fi
}

# Verify backup integrity
verify_backup() {
    local backup_file="$1"
    
    log "Verifying backup: $backup_file"
    
    # Check file exists and is not empty
    if [ ! -f "$backup_file" ]; then
        error "Backup file not found: $backup_file"
        return 1
    fi
    
    if [ ! -s "$backup_file" ]; then
        error "Backup file is empty: $backup_file"
        return 1
    fi
    
    # Verify checksum if available
    if [ -f "${backup_file}.sha256" ]; then
        local stored_checksum=$(cat "${backup_file}.sha256")
        local calculated_checksum=$(sha256sum "$backup_file" | cut -d' ' -f1)
        
        if [ "$stored_checksum" != "$calculated_checksum" ]; then
            error "Checksum mismatch!"
            return 1
        fi
        log "Checksum verified"
    fi
    
    # Test decompression
    if [[ "$backup_file" == *.gz ]]; then
        if ! gzip -t "$backup_file" 2>/dev/null; then
            error "Backup file is corrupted"
            return 1
        fi
        log "Compression integrity verified"
    fi
    
    log "Backup verification successful"
    return 0
}

# Clean up old backups
cleanup_old_backups() {
    log "Cleaning up backups older than $RETENTION_DAYS days"
    
    # Find and remove old local backups
    find "$BACKUP_DIR" -name "backup_*.sql.gz" -type f -mtime +$RETENTION_DAYS -delete
    find "$BACKUP_DIR" -name "backup_*.tar.gz" -type f -mtime +$RETENTION_DAYS -delete
    find "$BACKUP_DIR" -name "*.sha256" -type f -mtime +$RETENTION_DAYS -delete
    
    # Clean old logs
    find "$BACKUP_DIR/logs" -name "*.log" -type f -mtime +30 -delete
    
    log "Cleanup completed"
}

# Send notification (implement based on your notification system)
send_notification() {
    local status="$1"
    local message="$2"
    
    # Example: Send to Slack
    if [ -n "$SLACK_WEBHOOK_URL" ]; then
        curl -X POST "$SLACK_WEBHOOK_URL" \
            -H 'Content-Type: application/json' \
            -d "{\"text\": \"Database Backup $status: $message\"}"
    fi
    
    # Example: Send email
    if [ -n "$ALERT_EMAIL" ]; then
        echo "$message" | mail -s "Database Backup $status" "$ALERT_EMAIL"
    fi
}

# Main execution
main() {
    # Check requirements
    check_requirements
    
    # Perform backup based on type
    case "$BACKUP_TYPE" in
        full)
            BACKUP_FILE=$(perform_full_backup)
            ;;
        incremental)
            BACKUP_FILE=$(perform_incremental_backup)
            ;;
        schema)
            BACKUP_FILE=$(perform_schema_backup)
            ;;
        *)
            error "Invalid backup type: $BACKUP_TYPE"
            echo "Usage: $0 [full|incremental|schema]"
            exit 1
            ;;
    esac
    
    # Clean up old backups
    cleanup_old_backups
    
    # Report success
    log "=========================================="
    log "Backup completed successfully!"
    log "Backup file: $BACKUP_FILE"
    log "=========================================="
    
    # Send success notification
    send_notification "SUCCESS" "Database backup completed: $BACKUP_FILE"
    
    exit 0
}

# Error handling
trap 'error "Backup failed!"; send_notification "FAILED" "Database backup failed. Check logs: $LOG_FILE"; exit 1' ERR

# Run main function
main "$@"