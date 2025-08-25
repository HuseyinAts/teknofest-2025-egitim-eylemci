"""Add partitioning and monitoring capabilities

Revision ID: 004
Revises: 003
Create Date: 2025-01-21 10:03:00.000000

"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = '004'
down_revision = '003'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Enable required extensions
    op.execute("CREATE EXTENSION IF NOT EXISTS pg_stat_statements;")
    op.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm;")  # For fuzzy text search
    op.execute("CREATE EXTENSION IF NOT EXISTS btree_gin;")  # For composite GIN indexes
    op.execute("CREATE EXTENSION IF NOT EXISTS uuid-ossp;")  # For UUID generation
    
    # Create partitioned table for audit logs (monthly partitions)
    op.execute("""
        -- Rename existing audit_logs table
        ALTER TABLE IF EXISTS audit_logs RENAME TO audit_logs_old;
        
        -- Create partitioned audit_logs table
        CREATE TABLE audit_logs (
            id UUID DEFAULT uuid_generate_v4(),
            user_id UUID,
            action VARCHAR(100) NOT NULL,
            entity_type VARCHAR(50),
            entity_id UUID,
            old_data JSONB,
            new_data JSONB,
            ip_address VARCHAR(45),
            user_agent VARCHAR(500),
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL,
            PRIMARY KEY (id, created_at)
        ) PARTITION BY RANGE (created_at);
        
        -- Create indexes on parent table
        CREATE INDEX ix_audit_logs_user ON audit_logs (user_id, created_at);
        CREATE INDEX ix_audit_logs_action ON audit_logs (action, created_at);
        CREATE INDEX ix_audit_logs_entity ON audit_logs (entity_type, entity_id, created_at);
        CREATE INDEX ix_audit_logs_created ON audit_logs (created_at);
        
        -- Create partitions for current and next 3 months
        CREATE TABLE audit_logs_2025_01 PARTITION OF audit_logs
            FOR VALUES FROM ('2025-01-01') TO ('2025-02-01');
        CREATE TABLE audit_logs_2025_02 PARTITION OF audit_logs
            FOR VALUES FROM ('2025-02-01') TO ('2025-03-01');
        CREATE TABLE audit_logs_2025_03 PARTITION OF audit_logs
            FOR VALUES FROM ('2025-03-01') TO ('2025-04-01');
        CREATE TABLE audit_logs_2025_04 PARTITION OF audit_logs
            FOR VALUES FROM ('2025-04-01') TO ('2025-05-01');
        
        -- Copy data from old table if exists
        INSERT INTO audit_logs SELECT * FROM audit_logs_old WHERE created_at >= '2025-01-01';
        
        -- Drop old table
        DROP TABLE IF EXISTS audit_logs_old;
    """)
    
    # Create function to automatically create monthly partitions
    op.execute("""
        CREATE OR REPLACE FUNCTION create_monthly_partition()
        RETURNS void AS $$
        DECLARE
            partition_name TEXT;
            start_date DATE;
            end_date DATE;
        BEGIN
            -- Calculate dates for next month
            start_date := DATE_TRUNC('month', CURRENT_DATE + INTERVAL '1 month');
            end_date := DATE_TRUNC('month', CURRENT_DATE + INTERVAL '2 months');
            partition_name := 'audit_logs_' || TO_CHAR(start_date, 'YYYY_MM');
            
            -- Check if partition already exists
            IF NOT EXISTS (
                SELECT 1 FROM pg_class
                WHERE relname = partition_name
            ) THEN
                -- Create new partition
                EXECUTE format(
                    'CREATE TABLE %I PARTITION OF audit_logs FOR VALUES FROM (%L) TO (%L)',
                    partition_name, start_date, end_date
                );
                RAISE NOTICE 'Created partition: %', partition_name;
            END IF;
        END;
        $$ LANGUAGE plpgsql;
    """)
    
    # Create partitioned table for study_sessions (weekly partitions for better performance)
    op.execute("""
        -- Rename existing study_sessions table
        ALTER TABLE IF EXISTS study_sessions RENAME TO study_sessions_old;
        
        -- Create partitioned study_sessions table
        CREATE TABLE study_sessions (
            id UUID DEFAULT uuid_generate_v4(),
            user_id UUID NOT NULL,
            module_id UUID,
            started_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL,
            ended_at TIMESTAMP WITH TIME ZONE,
            duration_minutes INTEGER,
            interactions JSONB,
            notes TEXT,
            ai_interactions INTEGER DEFAULT 0,
            ai_messages JSONB,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL,
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL,
            PRIMARY KEY (id, started_at),
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
            FOREIGN KEY (module_id) REFERENCES modules(id) ON DELETE SET NULL
        ) PARTITION BY RANGE (started_at);
        
        -- Create indexes
        CREATE INDEX ix_study_sessions_user_date ON study_sessions (user_id, started_at);
        CREATE INDEX ix_study_sessions_user_month ON study_sessions (user_id, date_trunc('month', started_at));
        CREATE INDEX ix_study_sessions_duration ON study_sessions (duration_minutes) 
            WHERE duration_minutes IS NOT NULL;
        
        -- Create initial partitions
        CREATE TABLE study_sessions_2025_w04 PARTITION OF study_sessions
            FOR VALUES FROM ('2025-01-20') TO ('2025-01-27');
        CREATE TABLE study_sessions_2025_w05 PARTITION OF study_sessions
            FOR VALUES FROM ('2025-01-27') TO ('2025-02-03');
        
        -- Copy data from old table if exists
        INSERT INTO study_sessions 
        SELECT * FROM study_sessions_old WHERE started_at >= '2025-01-20';
        
        -- Drop old table
        DROP TABLE IF EXISTS study_sessions_old CASCADE;
    """)
    
    # Create monitoring views
    op.execute("""
        CREATE OR REPLACE VIEW database_health AS
        SELECT
            current_database() as database_name,
            pg_database_size(current_database()) as database_size,
            pg_size_pretty(pg_database_size(current_database())) as database_size_pretty,
            (SELECT count(*) FROM pg_stat_activity) as active_connections,
            (SELECT count(*) FROM pg_stat_activity WHERE state = 'active') as active_queries,
            (SELECT count(*) FROM pg_stat_activity WHERE state = 'idle in transaction') as idle_in_transaction,
            (SELECT count(*) FROM pg_locks) as lock_count,
            (SELECT count(*) FROM pg_locks WHERE granted = false) as waiting_locks,
            NOW() as checked_at
        ;
    """)
    
    op.execute("""
        CREATE OR REPLACE VIEW table_statistics AS
        SELECT
            schemaname,
            tablename,
            pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as total_size,
            pg_size_pretty(pg_relation_size(schemaname||'.'||tablename)) as table_size,
            pg_size_pretty(pg_indexes_size(schemaname||'.'||tablename)) as indexes_size,
            n_live_tup as row_count,
            n_dead_tup as dead_rows,
            last_vacuum,
            last_autovacuum,
            last_analyze,
            last_autoanalyze
        FROM pg_stat_user_tables
        ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;
    """)
    
    op.execute("""
        CREATE OR REPLACE VIEW slow_queries AS
        SELECT
            query,
            calls,
            total_exec_time,
            mean_exec_time,
            stddev_exec_time,
            rows,
            100.0 * shared_blks_hit / NULLIF(shared_blks_hit + shared_blks_read, 0) AS hit_ratio
        FROM pg_stat_statements
        WHERE query NOT LIKE '%pg_stat_statements%'
        ORDER BY mean_exec_time DESC
        LIMIT 20;
    """)
    
    op.execute("""
        CREATE OR REPLACE VIEW index_usage AS
        SELECT
            schemaname,
            tablename,
            indexname,
            idx_scan as index_scans,
            idx_tup_read as tuples_read,
            idx_tup_fetch as tuples_fetched,
            pg_size_pretty(pg_relation_size(indexrelid)) as index_size,
            CASE WHEN idx_scan = 0 THEN 'UNUSED' ELSE 'USED' END as usage_status
        FROM pg_stat_user_indexes
        ORDER BY schemaname, tablename, indexname;
    """)
    
    # Create function for table maintenance
    op.execute("""
        CREATE OR REPLACE FUNCTION maintain_tables()
        RETURNS void AS $$
        DECLARE
            table_record RECORD;
        BEGIN
            -- Vacuum and analyze tables with high dead tuple ratio
            FOR table_record IN
                SELECT schemaname, tablename
                FROM pg_stat_user_tables
                WHERE n_dead_tup > 1000 
                    AND n_dead_tup > n_live_tup * 0.1
            LOOP
                EXECUTE format('VACUUM ANALYZE %I.%I', 
                    table_record.schemaname, table_record.tablename);
                RAISE NOTICE 'Vacuumed table: %.%', 
                    table_record.schemaname, table_record.tablename;
            END LOOP;
            
            -- Reindex tables with high bloat
            FOR table_record IN
                SELECT schemaname, tablename
                FROM pg_stat_user_tables
                WHERE pg_relation_size(schemaname||'.'||tablename) > 100000000 -- 100MB
            LOOP
                EXECUTE format('REINDEX TABLE %I.%I CONCURRENTLY', 
                    table_record.schemaname, table_record.tablename);
                RAISE NOTICE 'Reindexed table: %.%', 
                    table_record.schemaname, table_record.tablename;
            END LOOP;
        END;
        $$ LANGUAGE plpgsql;
    """)
    
    # Create function to drop old partitions
    op.execute("""
        CREATE OR REPLACE FUNCTION drop_old_partitions(
            parent_table TEXT,
            retention_days INTEGER DEFAULT 90
        )
        RETURNS void AS $$
        DECLARE
            partition_record RECORD;
            drop_date DATE;
        BEGIN
            drop_date := CURRENT_DATE - retention_days;
            
            FOR partition_record IN
                SELECT 
                    schemaname,
                    tablename
                FROM pg_tables
                WHERE tablename LIKE parent_table || '_%'
                    AND schemaname = 'public'
            LOOP
                -- Extract date from partition name and check if it's old
                IF partition_record.tablename ~ '[0-9]{4}_[0-9]{2}' THEN
                    -- Check if partition is older than retention period
                    -- This is simplified - implement proper date extraction based on naming
                    EXECUTE format('DROP TABLE IF EXISTS %I.%I', 
                        partition_record.schemaname, partition_record.tablename);
                    RAISE NOTICE 'Dropped old partition: %.%', 
                        partition_record.schemaname, partition_record.tablename;
                END IF;
            END LOOP;
        END;
        $$ LANGUAGE plpgsql;
    """)
    
    # Create connection pool monitoring
    op.execute("""
        CREATE OR REPLACE VIEW connection_pool_stats AS
        SELECT
            datname as database,
            usename as username,
            application_name,
            client_addr,
            state,
            COUNT(*) as connection_count,
            MAX(NOW() - state_change) as max_idle_time,
            AVG(NOW() - state_change) as avg_idle_time
        FROM pg_stat_activity
        WHERE datname = current_database()
        GROUP BY datname, usename, application_name, client_addr, state
        ORDER BY connection_count DESC;
    """)
    
    # Create deadlock monitoring
    op.execute("""
        CREATE OR REPLACE VIEW deadlock_info AS
        SELECT 
            blocked_locks.pid AS blocked_pid,
            blocked_activity.usename AS blocked_user,
            blocking_locks.pid AS blocking_pid,
            blocking_activity.usename AS blocking_user,
            blocked_activity.query AS blocked_statement,
            blocking_activity.query AS current_statement_in_blocking_process,
            blocked_activity.application_name AS blocked_application,
            blocking_activity.application_name AS blocking_application
        FROM pg_catalog.pg_locks blocked_locks
        JOIN pg_catalog.pg_stat_activity blocked_activity 
            ON blocked_activity.pid = blocked_locks.pid
        JOIN pg_catalog.pg_locks blocking_locks 
            ON blocking_locks.locktype = blocked_locks.locktype
            AND blocking_locks.database IS NOT DISTINCT FROM blocked_locks.database
            AND blocking_locks.relation IS NOT DISTINCT FROM blocked_locks.relation
            AND blocking_locks.page IS NOT DISTINCT FROM blocked_locks.page
            AND blocking_locks.tuple IS NOT DISTINCT FROM blocked_locks.tuple
            AND blocking_locks.virtualxid IS NOT DISTINCT FROM blocked_locks.virtualxid
            AND blocking_locks.transactionid IS NOT DISTINCT FROM blocked_locks.transactionid
            AND blocking_locks.classid IS NOT DISTINCT FROM blocked_locks.classid
            AND blocking_locks.objid IS NOT DISTINCT FROM blocked_locks.objid
            AND blocking_locks.objsubid IS NOT DISTINCT FROM blocked_locks.objsubid
            AND blocking_locks.pid != blocked_locks.pid
        JOIN pg_catalog.pg_stat_activity blocking_activity 
            ON blocking_activity.pid = blocking_locks.pid
        WHERE NOT blocked_locks.granted;
    """)


def downgrade() -> None:
    # Drop monitoring views
    op.execute("DROP VIEW IF EXISTS deadlock_info;")
    op.execute("DROP VIEW IF EXISTS connection_pool_stats;")
    op.execute("DROP VIEW IF EXISTS index_usage;")
    op.execute("DROP VIEW IF EXISTS slow_queries;")
    op.execute("DROP VIEW IF EXISTS table_statistics;")
    op.execute("DROP VIEW IF EXISTS database_health;")
    
    # Drop maintenance functions
    op.execute("DROP FUNCTION IF EXISTS drop_old_partitions(TEXT, INTEGER);")
    op.execute("DROP FUNCTION IF EXISTS maintain_tables();")
    op.execute("DROP FUNCTION IF EXISTS create_monthly_partition();")
    
    # Restore original tables from partitioned ones
    op.execute("""
        -- Restore study_sessions
        CREATE TABLE study_sessions_restored AS 
        SELECT * FROM study_sessions;
        DROP TABLE study_sessions CASCADE;
        ALTER TABLE study_sessions_restored RENAME TO study_sessions;
        
        -- Restore audit_logs
        CREATE TABLE audit_logs_restored AS 
        SELECT * FROM audit_logs;
        DROP TABLE audit_logs CASCADE;
        ALTER TABLE audit_logs_restored RENAME TO audit_logs;
    """)
    
    # Drop extensions (be careful with this in production)
    # op.execute("DROP EXTENSION IF EXISTS pg_stat_statements;")
    # op.execute("DROP EXTENSION IF EXISTS pg_trgm;")
    # op.execute("DROP EXTENSION IF EXISTS btree_gin;")
    # op.execute("DROP EXTENSION IF EXISTS uuid-ossp;")