"""Add indexes and performance optimizations

Revision ID: 002
Revises: 001
Create Date: 2025-01-21 10:01:00.000000

"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = '002'
down_revision = '001'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Add composite indexes for common queries
    
    # User search and filtering
    op.create_index('ix_users_full_text_search', 'users', [sa.text("to_tsvector('simple', full_name || ' ' || username || ' ' || COALESCE(bio, ''))")], postgresql_using='gin')
    op.create_index('ix_users_created_at_desc', 'users', [sa.text('created_at DESC')])
    op.create_index('ix_users_points_desc', 'users', [sa.text('points DESC')], postgresql_where=sa.text('is_active = true'))
    op.create_index('ix_users_level_points', 'users', ['level', 'points'], postgresql_where=sa.text('is_active = true'))
    
    # Learning path search and filtering
    op.create_index('ix_learning_paths_full_text', 'learning_paths', [sa.text("to_tsvector('simple', title || ' ' || description || ' ' || COALESCE(array_to_string(tags, ' '), ''))")], postgresql_using='gin')
    op.create_index('ix_learning_paths_tags', 'learning_paths', ['tags'], postgresql_using='gin')
    op.create_index('ix_learning_paths_enrollment_desc', 'learning_paths', [sa.text('enrollment_count DESC')], postgresql_where=sa.text('is_published = true'))
    op.create_index('ix_learning_paths_rating_desc', 'learning_paths', [sa.text('average_rating DESC NULLS LAST')], postgresql_where=sa.text('is_published = true'))
    
    # Module navigation
    op.create_index('ix_modules_path_order', 'modules', ['learning_path_id', 'order_index'])
    op.create_index('ix_modules_content_type', 'modules', ['content_type'])
    
    # Progress tracking
    op.create_index('ix_progress_user_status', 'progress', ['user_id', 'status'])
    op.create_index('ix_progress_completed', 'progress', ['user_id', 'completed_at'], postgresql_where=sa.text("status = 'completed'"))
    
    # Study session analytics
    op.create_index('ix_study_sessions_user_month', 'study_sessions', ['user_id', sa.text("date_trunc('month', started_at)")])
    op.create_index('ix_study_sessions_duration', 'study_sessions', ['duration_minutes'], postgresql_where=sa.text('duration_minutes IS NOT NULL'))
    
    # Assessment performance
    op.create_index('ix_assessments_user_type', 'assessments', ['user_id', 'type'])
    op.create_index('ix_assessments_module_scores', 'assessments', ['module_id', 'percentage'], postgresql_where=sa.text('module_id IS NOT NULL'))
    
    # Notification queries
    op.create_index('ix_notifications_user_unread', 'notifications', ['user_id', 'created_at'], postgresql_where=sa.text('is_read = false'))
    
    # Audit log queries
    op.create_index('ix_audit_logs_user_action_date', 'audit_logs', ['user_id', 'action', 'created_at'])
    
    # Association table indexes
    op.create_index('ix_user_learning_paths_enrolled', 'user_learning_paths', ['enrolled_at'])
    op.create_index('ix_user_learning_paths_progress', 'user_learning_paths', ['user_id', 'progress'])
    op.create_index('ix_user_achievements_earned', 'user_achievements', ['earned_at'])
    
    # Add partial indexes for common filters
    op.create_index('ix_users_active_verified', 'users', ['id'], postgresql_where=sa.text('is_active = true AND is_verified = true'))
    op.create_index('ix_learning_paths_published_beginner', 'learning_paths', ['id'], postgresql_where=sa.text("is_published = true AND difficulty = 'beginner'"))
    op.create_index('ix_modules_mandatory', 'modules', ['learning_path_id', 'order_index'], postgresql_where=sa.text('is_mandatory = true'))
    
    # Add covering indexes for common queries
    op.create_index('ix_users_login_covering', 'users', ['email', 'hashed_password', 'is_active', 'is_verified', 'id'])
    op.create_index('ix_progress_covering', 'progress', ['user_id', 'module_id', 'status', 'progress_percentage'])
    
    # Create materialized view for user statistics (optional, for heavy analytics)
    op.execute("""
        CREATE MATERIALIZED VIEW IF NOT EXISTS user_statistics AS
        SELECT 
            u.id as user_id,
            u.username,
            u.level,
            u.points,
            u.streak_days,
            COUNT(DISTINCT ulp.learning_path_id) as enrolled_paths,
            COUNT(DISTINCT p.module_id) FILTER (WHERE p.status = 'completed') as completed_modules,
            AVG(a.percentage) as avg_assessment_score,
            SUM(ss.duration_minutes) as total_study_minutes,
            COUNT(DISTINCT ua.achievement_id) as achievement_count,
            MAX(ss.started_at) as last_activity
        FROM users u
        LEFT JOIN user_learning_paths ulp ON u.id = ulp.user_id
        LEFT JOIN progress p ON u.id = p.user_id
        LEFT JOIN assessments a ON u.id = a.user_id
        LEFT JOIN study_sessions ss ON u.id = ss.user_id
        LEFT JOIN user_achievements ua ON u.id = ua.user_id
        WHERE u.is_active = true
        GROUP BY u.id, u.username, u.level, u.points, u.streak_days
    """)
    
    op.create_index('ix_user_statistics_user_id', 'user_statistics', ['user_id'], unique=True)
    op.create_index('ix_user_statistics_points', 'user_statistics', ['points'])
    op.create_index('ix_user_statistics_level', 'user_statistics', ['level'])


def downgrade() -> None:
    # Drop materialized view
    op.execute("DROP MATERIALIZED VIEW IF EXISTS user_statistics")
    
    # Drop covering indexes
    op.drop_index('ix_progress_covering', table_name='progress')
    op.drop_index('ix_users_login_covering', table_name='users')
    
    # Drop partial indexes
    op.drop_index('ix_modules_mandatory', table_name='modules')
    op.drop_index('ix_learning_paths_published_beginner', table_name='learning_paths')
    op.drop_index('ix_users_active_verified', table_name='users')
    
    # Drop association table indexes
    op.drop_index('ix_user_achievements_earned', table_name='user_achievements')
    op.drop_index('ix_user_learning_paths_progress', table_name='user_learning_paths')
    op.drop_index('ix_user_learning_paths_enrolled', table_name='user_learning_paths')
    
    # Drop audit log indexes
    op.drop_index('ix_audit_logs_user_action_date', table_name='audit_logs')
    
    # Drop notification indexes
    op.drop_index('ix_notifications_user_unread', table_name='notifications')
    
    # Drop assessment indexes
    op.drop_index('ix_assessments_module_scores', table_name='assessments')
    op.drop_index('ix_assessments_user_type', table_name='assessments')
    
    # Drop study session indexes
    op.drop_index('ix_study_sessions_duration', table_name='study_sessions')
    op.drop_index('ix_study_sessions_user_month', table_name='study_sessions')
    
    # Drop progress indexes
    op.drop_index('ix_progress_completed', table_name='progress')
    op.drop_index('ix_progress_user_status', table_name='progress')
    
    # Drop module indexes
    op.drop_index('ix_modules_content_type', table_name='modules')
    op.drop_index('ix_modules_path_order', table_name='modules')
    
    # Drop learning path indexes
    op.drop_index('ix_learning_paths_rating_desc', table_name='learning_paths')
    op.drop_index('ix_learning_paths_enrollment_desc', table_name='learning_paths')
    op.drop_index('ix_learning_paths_tags', table_name='learning_paths')
    op.drop_index('ix_learning_paths_full_text', table_name='learning_paths')
    
    # Drop user indexes
    op.drop_index('ix_users_level_points', table_name='users')
    op.drop_index('ix_users_points_desc', table_name='users')
    op.drop_index('ix_users_created_at_desc', table_name='users')
    op.drop_index('ix_users_full_text_search', table_name='users')