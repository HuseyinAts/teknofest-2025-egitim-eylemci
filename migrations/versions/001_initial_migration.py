"""Initial migration with all tables

Revision ID: 001
Revises: 
Create Date: 2025-01-21 10:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
import uuid

# revision identifiers, used by Alembic.
revision = '001'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Create enum types
    op.execute("CREATE TYPE userrole AS ENUM ('student', 'teacher', 'admin', 'parent')")
    op.execute("CREATE TYPE difficultylevel AS ENUM ('beginner', 'intermediate', 'advanced', 'expert')")
    op.execute("CREATE TYPE contenttype AS ENUM ('video', 'text', 'quiz', 'exercise', 'project')")
    
    # Create users table
    op.create_table('users',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False, default=uuid.uuid4),
        sa.Column('email', sa.String(length=255), nullable=False),
        sa.Column('username', sa.String(length=100), nullable=False),
        sa.Column('full_name', sa.String(length=255), nullable=False),
        sa.Column('hashed_password', sa.String(length=255), nullable=False),
        sa.Column('role', postgresql.ENUM('student', 'teacher', 'admin', 'parent', name='userrole'), nullable=False),
        sa.Column('is_active', sa.Boolean(), nullable=False, default=True),
        sa.Column('is_verified', sa.Boolean(), nullable=False, default=False),
        sa.Column('avatar_url', sa.String(length=500), nullable=True),
        sa.Column('bio', sa.Text(), nullable=True),
        sa.Column('date_of_birth', sa.DateTime(), nullable=True),
        sa.Column('phone_number', sa.String(length=20), nullable=True),
        sa.Column('preferred_language', sa.String(length=10), nullable=False, default='tr'),
        sa.Column('learning_style', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('interests', postgresql.ARRAY(sa.String()), nullable=True),
        sa.Column('total_study_time', sa.Integer(), nullable=True, default=0),
        sa.Column('streak_days', sa.Integer(), nullable=True, default=0),
        sa.Column('points', sa.Integer(), nullable=True, default=0),
        sa.Column('level', sa.Integer(), nullable=True, default=1),
        sa.Column('oauth_provider', sa.String(length=50), nullable=True),
        sa.Column('oauth_id', sa.String(length=255), nullable=True),
        sa.Column('last_login_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('last_login_ip', sa.String(length=45), nullable=True),
        sa.Column('failed_login_attempts', sa.Integer(), nullable=True, default=0),
        sa.Column('locked_until', sa.DateTime(timezone=True), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.PrimaryKeyConstraint('id', name=op.f('pk_users')),
        sa.UniqueConstraint('email', name=op.f('uq_users_email')),
        sa.UniqueConstraint('username', name=op.f('uq_users_username')),
        sa.CheckConstraint('streak_days >= 0', name='check_streak_positive'),
        sa.CheckConstraint('points >= 0', name='check_points_positive'),
        sa.CheckConstraint('level >= 1', name='check_level_positive')
    )
    op.create_index(op.f('ix_users_email'), 'users', ['email'])
    op.create_index(op.f('ix_users_username'), 'users', ['username'])
    op.create_index('ix_users_email_active', 'users', ['email', 'is_active'])
    op.create_index('ix_users_role_active', 'users', ['role', 'is_active'])
    
    # Create learning_paths table
    op.create_table('learning_paths',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False, default=uuid.uuid4),
        sa.Column('title', sa.String(length=255), nullable=False),
        sa.Column('description', sa.Text(), nullable=False),
        sa.Column('slug', sa.String(length=255), nullable=False),
        sa.Column('objectives', postgresql.ARRAY(sa.String()), nullable=False),
        sa.Column('prerequisites', postgresql.ARRAY(sa.String()), nullable=True),
        sa.Column('tags', postgresql.ARRAY(sa.String()), nullable=True),
        sa.Column('difficulty', postgresql.ENUM('beginner', 'intermediate', 'advanced', 'expert', name='difficultylevel'), nullable=False),
        sa.Column('estimated_hours', sa.Float(), nullable=False),
        sa.Column('language', sa.String(length=10), nullable=False, default='tr'),
        sa.Column('ai_generated', sa.Boolean(), nullable=True, default=False),
        sa.Column('ai_model', sa.String(length=100), nullable=True),
        sa.Column('ai_parameters', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('enrollment_count', sa.Integer(), nullable=True, default=0),
        sa.Column('completion_count', sa.Integer(), nullable=True, default=0),
        sa.Column('average_rating', sa.Float(), nullable=True),
        sa.Column('is_published', sa.Boolean(), nullable=True, default=False),
        sa.Column('published_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.PrimaryKeyConstraint('id', name=op.f('pk_learning_paths')),
        sa.UniqueConstraint('slug', name=op.f('uq_learning_paths_slug')),
        sa.CheckConstraint('estimated_hours > 0', name='check_estimated_hours_positive')
    )
    op.create_index(op.f('ix_learning_paths_slug'), 'learning_paths', ['slug'])
    op.create_index('ix_learning_paths_difficulty', 'learning_paths', ['difficulty'])
    op.create_index('ix_learning_paths_published', 'learning_paths', ['is_published'])
    
    # Create achievements table
    op.create_table('achievements',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False, default=uuid.uuid4),
        sa.Column('name', sa.String(length=100), nullable=False),
        sa.Column('description', sa.Text(), nullable=False),
        sa.Column('icon_url', sa.String(length=500), nullable=True),
        sa.Column('criteria', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('points', sa.Integer(), nullable=True, default=0),
        sa.Column('rarity', sa.String(length=20), nullable=True, default='common'),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.PrimaryKeyConstraint('id', name=op.f('pk_achievements')),
        sa.UniqueConstraint('name', name=op.f('uq_achievements_name'))
    )
    
    # Create modules table
    op.create_table('modules',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False, default=uuid.uuid4),
        sa.Column('learning_path_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('title', sa.String(length=255), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('order_index', sa.Integer(), nullable=False),
        sa.Column('content_type', postgresql.ENUM('video', 'text', 'quiz', 'exercise', 'project', name='contenttype'), nullable=False),
        sa.Column('content_url', sa.String(length=500), nullable=True),
        sa.Column('content_data', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('estimated_minutes', sa.Integer(), nullable=True, default=30),
        sa.Column('is_mandatory', sa.Boolean(), nullable=True, default=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['learning_path_id'], ['learning_paths.id'], name=op.f('fk_modules_learning_path_id_learning_paths'), ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id', name=op.f('pk_modules')),
        sa.UniqueConstraint('learning_path_id', 'order_index', name='uq_module_order')
    )
    op.create_index('ix_modules_learning_path', 'modules', ['learning_path_id'])
    
    # Create notifications table
    op.create_table('notifications',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False, default=uuid.uuid4),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('type', sa.String(length=50), nullable=False),
        sa.Column('title', sa.String(length=255), nullable=False),
        sa.Column('message', sa.Text(), nullable=False),
        sa.Column('data', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('is_read', sa.Boolean(), nullable=True, default=False),
        sa.Column('read_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], name=op.f('fk_notifications_user_id_users'), ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id', name=op.f('pk_notifications'))
    )
    op.create_index('ix_notifications_user_read', 'notifications', ['user_id', 'is_read'])
    op.create_index('ix_notifications_created', 'notifications', ['created_at'])
    
    # Create study_sessions table
    op.create_table('study_sessions',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False, default=uuid.uuid4),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('module_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('started_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('ended_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('duration_minutes', sa.Integer(), nullable=True),
        sa.Column('interactions', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('notes', sa.Text(), nullable=True),
        sa.Column('ai_interactions', sa.Integer(), nullable=True, default=0),
        sa.Column('ai_messages', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], name=op.f('fk_study_sessions_user_id_users'), ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['module_id'], ['modules.id'], name=op.f('fk_study_sessions_module_id_modules'), ondelete='SET NULL'),
        sa.PrimaryKeyConstraint('id', name=op.f('pk_study_sessions'))
    )
    op.create_index('ix_study_sessions_user_date', 'study_sessions', ['user_id', 'started_at'])
    
    # Create assessments table
    op.create_table('assessments',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False, default=uuid.uuid4),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('module_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('type', sa.String(length=50), nullable=False),
        sa.Column('questions', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('answers', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('score', sa.Float(), nullable=False),
        sa.Column('max_score', sa.Float(), nullable=False),
        sa.Column('percentage', sa.Float(), nullable=False),
        sa.Column('passed', sa.Boolean(), nullable=False),
        sa.Column('started_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('completed_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('time_spent_seconds', sa.Integer(), nullable=False),
        sa.Column('feedback', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('ai_evaluation', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], name=op.f('fk_assessments_user_id_users'), ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['module_id'], ['modules.id'], name=op.f('fk_assessments_module_id_modules'), ondelete='SET NULL'),
        sa.PrimaryKeyConstraint('id', name=op.f('pk_assessments')),
        sa.CheckConstraint('score >= 0 AND score <= max_score', name='check_score_valid'),
        sa.CheckConstraint('percentage >= 0 AND percentage <= 100', name='check_percentage_valid')
    )
    op.create_index('ix_assessments_user_date', 'assessments', ['user_id', 'completed_at'])
    
    # Create progress table
    op.create_table('progress',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False, default=uuid.uuid4),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('module_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('status', sa.String(length=20), nullable=True, default='not_started'),
        sa.Column('progress_percentage', sa.Float(), nullable=True, default=0.0),
        sa.Column('completed_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('time_spent_minutes', sa.Integer(), nullable=True, default=0),
        sa.Column('attempt_count', sa.Integer(), nullable=True, default=0),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], name=op.f('fk_progress_user_id_users'), ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['module_id'], ['modules.id'], name=op.f('fk_progress_module_id_modules'), ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id', name=op.f('pk_progress')),
        sa.UniqueConstraint('user_id', 'module_id', name='uq_user_module_progress'),
        sa.CheckConstraint('progress_percentage >= 0 AND progress_percentage <= 100', name='check_progress_percentage')
    )
    op.create_index('ix_progress_user_module', 'progress', ['user_id', 'module_id'])
    
    # Create audit_logs table
    op.create_table('audit_logs',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False, default=uuid.uuid4),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('action', sa.String(length=100), nullable=False),
        sa.Column('entity_type', sa.String(length=50), nullable=True),
        sa.Column('entity_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('old_data', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('new_data', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('ip_address', sa.String(length=45), nullable=True),
        sa.Column('user_agent', sa.String(length=500), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], name=op.f('fk_audit_logs_user_id_users'), ondelete='SET NULL'),
        sa.PrimaryKeyConstraint('id', name=op.f('pk_audit_logs'))
    )
    op.create_index('ix_audit_logs_user', 'audit_logs', ['user_id'])
    op.create_index('ix_audit_logs_action', 'audit_logs', ['action'])
    op.create_index('ix_audit_logs_entity', 'audit_logs', ['entity_type', 'entity_id'])
    op.create_index('ix_audit_logs_created', 'audit_logs', ['created_at'])
    
    # Create association tables
    op.create_table('user_learning_paths',
        sa.Column('user_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('learning_path_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('enrolled_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.Column('progress', sa.Float(), nullable=True, default=0.0),
        sa.Column('completed_at', sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], name=op.f('fk_user_learning_paths_user_id_users'), ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['learning_path_id'], ['learning_paths.id'], name=op.f('fk_user_learning_paths_learning_path_id_learning_paths'), ondelete='CASCADE'),
        sa.UniqueConstraint('user_id', 'learning_path_id', name='uq_user_learning_path')
    )
    
    op.create_table('user_achievements',
        sa.Column('user_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('achievement_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('earned_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], name=op.f('fk_user_achievements_user_id_users'), ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['achievement_id'], ['achievements.id'], name=op.f('fk_user_achievements_achievement_id_achievements'), ondelete='CASCADE'),
        sa.UniqueConstraint('user_id', 'achievement_id', name='uq_user_achievement')
    )


def downgrade() -> None:
    # Drop association tables
    op.drop_table('user_achievements')
    op.drop_table('user_learning_paths')
    
    # Drop main tables
    op.drop_table('audit_logs')
    op.drop_table('progress')
    op.drop_table('assessments')
    op.drop_table('study_sessions')
    op.drop_table('notifications')
    op.drop_table('modules')
    op.drop_table('achievements')
    op.drop_table('learning_paths')
    op.drop_table('users')
    
    # Drop enum types
    op.execute("DROP TYPE IF EXISTS contenttype")
    op.execute("DROP TYPE IF EXISTS difficultylevel")
    op.execute("DROP TYPE IF EXISTS userrole")