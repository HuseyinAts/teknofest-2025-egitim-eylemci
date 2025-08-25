"""
Database health check and monitoring utilities
Production-ready health checks for database connectivity, performance, and integrity
"""

import time
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from contextlib import contextmanager

from sqlalchemy import text, create_engine
from sqlalchemy.exc import SQLAlchemyError, OperationalError, IntegrityError
from sqlalchemy.pool import NullPool

from ..config import get_settings
from .session import get_db_stats, engine

logger = logging.getLogger(__name__)
settings = get_settings()


class DatabaseHealthChecker:
    """
    Comprehensive database health checking system.
    """
    
    def __init__(self, connection_string: Optional[str] = None):
        """
        Initialize health checker.
        
        Args:
            connection_string: Database connection string (uses settings if not provided)
        """
        self.connection_string = connection_string or settings.database_url
        self._last_check_results = {}
        self._check_history = []
    
    def check_all(self) -> Dict[str, Any]:
        """
        Run all health checks and return comprehensive status.
        
        Returns:
            Dictionary with health check results
        """
        start_time = time.time()
        
        results = {
            "timestamp": datetime.utcnow().isoformat(),
            "status": "healthy",
            "checks": {},
            "metrics": {},
            "warnings": [],
            "errors": []
        }
        
        # Run individual checks
        checks = [
            ("connectivity", self.check_connectivity),
            ("performance", self.check_performance),
            ("replication", self.check_replication),
            ("disk_usage", self.check_disk_usage),
            ("connection_pool", self.check_connection_pool),
            ("long_running_queries", self.check_long_running_queries),
            ("table_integrity", self.check_table_integrity),
            ("indexes", self.check_indexes),
            ("locks", self.check_locks),
            ("cache_hit_ratio", self.check_cache_hit_ratio),
        ]
        
        for check_name, check_func in checks:
            try:
                check_result = check_func()
                results["checks"][check_name] = check_result
                
                # Update overall status
                if check_result["status"] == "unhealthy":
                    results["status"] = "unhealthy"
                    results["errors"].append(f"{check_name}: {check_result.get('message')}")
                elif check_result["status"] == "warning":
                    if results["status"] == "healthy":
                        results["status"] = "degraded"
                    results["warnings"].append(f"{check_name}: {check_result.get('message')}")
                    
            except Exception as e:
                logger.error(f"Health check '{check_name}' failed: {e}")
                results["checks"][check_name] = {
                    "status": "error",
                    "message": str(e)
                }
                results["status"] = "unhealthy"
                results["errors"].append(f"{check_name}: {str(e)}")
        
        # Calculate execution time
        results["execution_time_ms"] = round((time.time() - start_time) * 1000, 2)
        
        # Store results
        self._last_check_results = results
        self._check_history.append(results)
        
        # Keep only last 100 checks in history
        if len(self._check_history) > 100:
            self._check_history = self._check_history[-100:]
        
        return results
    
    def check_connectivity(self) -> Dict[str, Any]:
        """Check basic database connectivity"""
        try:
            start_time = time.time()
            
            with self._get_connection() as conn:
                result = conn.execute(text("SELECT 1"))
                result.scalar()
            
            response_time = (time.time() - start_time) * 1000
            
            return {
                "status": "healthy" if response_time < 100 else "warning",
                "response_time_ms": round(response_time, 2),
                "message": "Database is reachable"
            }
            
        except OperationalError as e:
            return {
                "status": "unhealthy",
                "message": f"Cannot connect to database: {str(e)}"
            }
    
    def check_performance(self) -> Dict[str, Any]:
        """Check database performance metrics"""
        try:
            metrics = {}
            
            with self._get_connection() as conn:
                # Check query performance
                start_time = time.time()
                conn.execute(text("SELECT COUNT(*) FROM pg_stat_activity"))
                query_time = (time.time() - start_time) * 1000
                metrics["simple_query_ms"] = round(query_time, 2)
                
                # Get database statistics
                result = conn.execute(text("""
                    SELECT 
                        numbackends as active_connections,
                        xact_commit as transactions_committed,
                        xact_rollback as transactions_rolled_back,
                        blks_read as blocks_read,
                        blks_hit as blocks_hit,
                        tup_returned as tuples_returned,
                        tup_fetched as tuples_fetched,
                        tup_inserted as tuples_inserted,
                        tup_updated as tuples_updated,
                        tup_deleted as tuples_deleted
                    FROM pg_stat_database 
                    WHERE datname = current_database()
                """))
                
                stats = dict(result.fetchone())
                metrics.update(stats)
                
                # Calculate cache hit ratio
                if stats['blocks_read'] > 0:
                    cache_hit_ratio = stats['blocks_hit'] / (stats['blocks_hit'] + stats['blocks_read'])
                    metrics['cache_hit_ratio'] = round(cache_hit_ratio * 100, 2)
            
            # Determine status based on metrics
            status = "healthy"
            if query_time > 100:
                status = "warning"
            if query_time > 500:
                status = "unhealthy"
            
            return {
                "status": status,
                "metrics": metrics,
                "message": f"Query time: {query_time:.2f}ms"
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Performance check failed: {str(e)}"
            }
    
    def check_replication(self) -> Dict[str, Any]:
        """Check database replication status (PostgreSQL specific)"""
        try:
            with self._get_connection() as conn:
                # Check if replication is configured
                result = conn.execute(text("""
                    SELECT COUNT(*) as replica_count
                    FROM pg_stat_replication
                """))
                replica_count = result.scalar()
                
                if replica_count == 0:
                    return {
                        "status": "info",
                        "message": "No replication configured",
                        "replica_count": 0
                    }
                
                # Get replication lag
                result = conn.execute(text("""
                    SELECT 
                        client_addr,
                        state,
                        sync_state,
                        pg_wal_lsn_diff(pg_current_wal_lsn(), replay_lsn) as lag_bytes
                    FROM pg_stat_replication
                """))
                
                replicas = []
                max_lag = 0
                
                for row in result:
                    lag_bytes = row['lag_bytes'] or 0
                    max_lag = max(max_lag, lag_bytes)
                    
                    replicas.append({
                        "client": str(row['client_addr']),
                        "state": row['state'],
                        "sync_state": row['sync_state'],
                        "lag_bytes": lag_bytes
                    })
                
                # Determine status based on lag
                status = "healthy"
                if max_lag > 10 * 1024 * 1024:  # 10MB lag
                    status = "warning"
                if max_lag > 100 * 1024 * 1024:  # 100MB lag
                    status = "unhealthy"
                
                return {
                    "status": status,
                    "replica_count": replica_count,
                    "replicas": replicas,
                    "max_lag_bytes": max_lag,
                    "message": f"{replica_count} replica(s) connected"
                }
                
        except Exception as e:
            # Not all databases support replication
            return {
                "status": "info",
                "message": "Replication check not applicable"
            }
    
    def check_disk_usage(self) -> Dict[str, Any]:
        """Check database disk usage"""
        try:
            with self._get_connection() as conn:
                # Get database size
                result = conn.execute(text("""
                    SELECT 
                        pg_database_size(current_database()) as db_size,
                        pg_size_pretty(pg_database_size(current_database())) as db_size_pretty
                """))
                row = result.fetchone()
                db_size = row['db_size']
                db_size_pretty = row['db_size_pretty']
                
                # Get table sizes
                result = conn.execute(text("""
                    SELECT 
                        schemaname,
                        tablename,
                        pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size,
                        pg_total_relation_size(schemaname||'.'||tablename) as size_bytes
                    FROM pg_tables 
                    WHERE schemaname NOT IN ('pg_catalog', 'information_schema')
                    ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC
                    LIMIT 10
                """))
                
                largest_tables = [
                    {
                        "schema": row['schemaname'],
                        "table": row['tablename'],
                        "size": row['size'],
                        "size_bytes": row['size_bytes']
                    }
                    for row in result
                ]
                
                # Determine status based on size
                status = "healthy"
                if db_size > 10 * 1024 * 1024 * 1024:  # 10GB
                    status = "warning"
                if db_size > 100 * 1024 * 1024 * 1024:  # 100GB
                    status = "unhealthy"
                
                return {
                    "status": status,
                    "database_size": db_size_pretty,
                    "database_size_bytes": db_size,
                    "largest_tables": largest_tables,
                    "message": f"Database size: {db_size_pretty}"
                }
                
        except Exception as e:
            return {
                "status": "error",
                "message": f"Disk usage check failed: {str(e)}"
            }
    
    def check_connection_pool(self) -> Dict[str, Any]:
        """Check connection pool status"""
        try:
            # Get pool stats from SQLAlchemy
            pool_stats = get_db_stats()
            
            # Get database connection stats
            with self._get_connection() as conn:
                result = conn.execute(text("""
                    SELECT 
                        COUNT(*) as total_connections,
                        COUNT(*) FILTER (WHERE state = 'active') as active,
                        COUNT(*) FILTER (WHERE state = 'idle') as idle,
                        COUNT(*) FILTER (WHERE state = 'idle in transaction') as idle_in_transaction,
                        MAX(EXTRACT(EPOCH FROM (now() - query_start))) as max_query_duration
                    FROM pg_stat_activity
                    WHERE datname = current_database()
                """))
                
                db_stats = dict(result.fetchone())
            
            # Combine stats
            stats = {
                **pool_stats,
                **db_stats
            }
            
            # Determine status
            status = "healthy"
            if stats.get('total_connections', 0) > 50:
                status = "warning"
            if stats.get('total_connections', 0) > 100:
                status = "unhealthy"
            
            if stats.get('idle_in_transaction', 0) > 5:
                status = "warning"
            
            return {
                "status": status,
                "pool_stats": pool_stats,
                "database_stats": db_stats,
                "message": f"Total connections: {stats.get('total_connections', 0)}"
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Connection pool check failed: {str(e)}"
            }
    
    def check_long_running_queries(self) -> Dict[str, Any]:
        """Check for long-running queries"""
        try:
            with self._get_connection() as conn:
                result = conn.execute(text("""
                    SELECT 
                        pid,
                        usename,
                        application_name,
                        state,
                        query,
                        EXTRACT(EPOCH FROM (now() - query_start)) as duration_seconds
                    FROM pg_stat_activity
                    WHERE state != 'idle'
                    AND query NOT LIKE '%pg_stat_activity%'
                    AND query_start < now() - interval '1 minute'
                    ORDER BY query_start
                    LIMIT 10
                """))
                
                long_queries = []
                for row in result:
                    long_queries.append({
                        "pid": row['pid'],
                        "user": row['usename'],
                        "application": row['application_name'],
                        "state": row['state'],
                        "duration_seconds": round(row['duration_seconds'], 2),
                        "query": row['query'][:100] + "..." if len(row['query']) > 100 else row['query']
                    })
                
                # Determine status
                status = "healthy"
                if long_queries:
                    status = "warning"
                if any(q['duration_seconds'] > 300 for q in long_queries):  # 5 minutes
                    status = "unhealthy"
                
                return {
                    "status": status,
                    "long_running_count": len(long_queries),
                    "queries": long_queries,
                    "message": f"Found {len(long_queries)} long-running queries"
                }
                
        except Exception as e:
            return {
                "status": "error",
                "message": f"Long query check failed: {str(e)}"
            }
    
    def check_table_integrity(self) -> Dict[str, Any]:
        """Check table integrity and constraints"""
        try:
            issues = []
            
            with self._get_connection() as conn:
                # Check for tables without primary keys
                result = conn.execute(text("""
                    SELECT 
                        schemaname,
                        tablename
                    FROM pg_tables t
                    WHERE schemaname NOT IN ('pg_catalog', 'information_schema')
                    AND NOT EXISTS (
                        SELECT 1 
                        FROM pg_constraint c
                        WHERE c.conrelid = (schemaname||'.'||tablename)::regclass
                        AND c.contype = 'p'
                    )
                """))
                
                tables_without_pk = [
                    f"{row['schemaname']}.{row['tablename']}"
                    for row in result
                ]
                
                if tables_without_pk:
                    issues.append({
                        "type": "missing_primary_key",
                        "tables": tables_without_pk
                    })
                
                # Check for invalid constraints
                result = conn.execute(text("""
                    SELECT 
                        conname,
                        conrelid::regclass as table_name
                    FROM pg_constraint
                    WHERE NOT convalidated
                """))
                
                invalid_constraints = [
                    {
                        "constraint": row['conname'],
                        "table": str(row['table_name'])
                    }
                    for row in result
                ]
                
                if invalid_constraints:
                    issues.append({
                        "type": "invalid_constraints",
                        "constraints": invalid_constraints
                    })
            
            # Determine status
            status = "healthy" if not issues else "warning"
            
            return {
                "status": status,
                "issues": issues,
                "message": f"Found {len(issues)} integrity issues" if issues else "No integrity issues found"
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Integrity check failed: {str(e)}"
            }
    
    def check_indexes(self) -> Dict[str, Any]:
        """Check index health and usage"""
        try:
            with self._get_connection() as conn:
                # Check for unused indexes
                result = conn.execute(text("""
                    SELECT 
                        schemaname,
                        tablename,
                        indexname,
                        idx_scan,
                        pg_size_pretty(pg_relation_size(indexrelid)) as index_size
                    FROM pg_stat_user_indexes
                    WHERE idx_scan = 0
                    AND indexrelid NOT IN (
                        SELECT conindid 
                        FROM pg_constraint 
                        WHERE conindid != 0
                    )
                    ORDER BY pg_relation_size(indexrelid) DESC
                    LIMIT 10
                """))
                
                unused_indexes = [
                    {
                        "schema": row['schemaname'],
                        "table": row['tablename'],
                        "index": row['indexname'],
                        "size": row['index_size']
                    }
                    for row in result
                ]
                
                # Check for invalid indexes
                result = conn.execute(text("""
                    SELECT 
                        schemaname,
                        tablename,
                        indexname
                    FROM pg_indexes i
                    WHERE NOT EXISTS (
                        SELECT 1
                        FROM pg_class c
                        WHERE c.relname = i.indexname
                        AND c.relkind = 'i'
                    )
                """))
                
                invalid_indexes = [
                    f"{row['schemaname']}.{row['tablename']}.{row['indexname']}"
                    for row in result
                ]
                
                # Determine status
                status = "healthy"
                if unused_indexes:
                    status = "info"
                if invalid_indexes:
                    status = "warning"
                
                return {
                    "status": status,
                    "unused_indexes": unused_indexes,
                    "invalid_indexes": invalid_indexes,
                    "message": f"Found {len(unused_indexes)} unused indexes"
                }
                
        except Exception as e:
            return {
                "status": "error",
                "message": f"Index check failed: {str(e)}"
            }
    
    def check_locks(self) -> Dict[str, Any]:
        """Check for database locks"""
        try:
            with self._get_connection() as conn:
                result = conn.execute(text("""
                    SELECT 
                        blocked_locks.pid AS blocked_pid,
                        blocked_activity.usename AS blocked_user,
                        blocking_locks.pid AS blocking_pid,
                        blocking_activity.usename AS blocking_user,
                        blocked_activity.query AS blocked_query,
                        blocking_activity.query AS blocking_query
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
                    WHERE NOT blocked_locks.granted
                """))
                
                locks = [
                    {
                        "blocked_pid": row['blocked_pid'],
                        "blocked_user": row['blocked_user'],
                        "blocking_pid": row['blocking_pid'],
                        "blocking_user": row['blocking_user'],
                        "blocked_query": row['blocked_query'][:100] if row['blocked_query'] else None,
                        "blocking_query": row['blocking_query'][:100] if row['blocking_query'] else None
                    }
                    for row in result
                ]
                
                # Determine status
                status = "healthy" if not locks else "warning"
                if len(locks) > 5:
                    status = "unhealthy"
                
                return {
                    "status": status,
                    "lock_count": len(locks),
                    "locks": locks,
                    "message": f"Found {len(locks)} blocking locks"
                }
                
        except Exception as e:
            return {
                "status": "error",
                "message": f"Lock check failed: {str(e)}"
            }
    
    def check_cache_hit_ratio(self) -> Dict[str, Any]:
        """Check database cache hit ratio"""
        try:
            with self._get_connection() as conn:
                result = conn.execute(text("""
                    SELECT 
                        sum(heap_blks_read) as heap_read,
                        sum(heap_blks_hit) as heap_hit,
                        sum(idx_blks_read) as idx_read,
                        sum(idx_blks_hit) as idx_hit
                    FROM pg_statio_user_tables
                """))
                
                row = result.fetchone()
                
                # Calculate ratios
                heap_ratio = 0
                if row['heap_read'] + row['heap_hit'] > 0:
                    heap_ratio = row['heap_hit'] / (row['heap_read'] + row['heap_hit'])
                
                idx_ratio = 0
                if row['idx_read'] + row['idx_hit'] > 0:
                    idx_ratio = row['idx_hit'] / (row['idx_read'] + row['idx_hit'])
                
                overall_ratio = 0
                total_read = row['heap_read'] + row['idx_read']
                total_hit = row['heap_hit'] + row['idx_hit']
                if total_read + total_hit > 0:
                    overall_ratio = total_hit / (total_read + total_hit)
                
                # Determine status
                status = "healthy"
                if overall_ratio < 0.90:
                    status = "warning"
                if overall_ratio < 0.75:
                    status = "unhealthy"
                
                return {
                    "status": status,
                    "overall_ratio": round(overall_ratio * 100, 2),
                    "heap_ratio": round(heap_ratio * 100, 2),
                    "index_ratio": round(idx_ratio * 100, 2),
                    "message": f"Cache hit ratio: {overall_ratio * 100:.2f}%"
                }
                
        except Exception as e:
            return {
                "status": "error",
                "message": f"Cache check failed: {str(e)}"
            }
    
    @contextmanager
    def _get_connection(self):
        """Get a database connection for health checks"""
        # Create a separate engine for health checks with minimal pooling
        health_engine = create_engine(
            self.connection_string,
            poolclass=NullPool,
            connect_args={"connect_timeout": 5}
        )
        
        conn = health_engine.connect()
        try:
            yield conn
        finally:
            conn.close()
            health_engine.dispose()
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get a summary of the last health check"""
        if not self._last_check_results:
            return {"status": "unknown", "message": "No health check performed yet"}
        
        return {
            "status": self._last_check_results["status"],
            "timestamp": self._last_check_results["timestamp"],
            "execution_time_ms": self._last_check_results.get("execution_time_ms"),
            "error_count": len(self._last_check_results.get("errors", [])),
            "warning_count": len(self._last_check_results.get("warnings", [])),
        }
    
    def get_health_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get health check history"""
        return self._check_history[-limit:]


# Singleton instance
_health_checker = None


def get_health_checker() -> DatabaseHealthChecker:
    """Get health checker singleton"""
    global _health_checker
    if _health_checker is None:
        _health_checker = DatabaseHealthChecker()
    return _health_checker


# Convenience functions
def check_database_health() -> Dict[str, Any]:
    """Run complete database health check"""
    checker = get_health_checker()
    return checker.check_all()


def get_health_summary() -> Dict[str, Any]:
    """Get summary of last health check"""
    checker = get_health_checker()
    return checker.get_health_summary()


def is_database_healthy() -> bool:
    """Quick check if database is healthy"""
    summary = get_health_summary()
    return summary["status"] in ["healthy", "degraded"]