"""Add gamification tables

Revision ID: 006
Revises: 005
Create Date: 2025-01-22

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '006'
down_revision = '005'
branch_labels = None
depends_on = None


def upgrade():
    # Create point_transactions table for tracking all point awards
    op.create_table(
        'point_transactions',
        sa.Column('id', postgresql.UUID(as_uuid=True), server_default=sa.text('gen_random_uuid()'), nullable=False),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('amount', sa.Integer(), nullable=False),
        sa.Column('source', sa.String(50), nullable=False),
        sa.Column('description', sa.String(255), nullable=False),
        sa.Column('metadata', postgresql.JSONB(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE')
    )
    
    # Create indexes for point_transactions
    op.create_index('idx_point_transactions_user_id', 'point_transactions', ['user_id'])
    op.create_index('idx_point_transactions_created_at', 'point_transactions', ['created_at'])
    op.create_index('idx_point_transactions_source', 'point_transactions', ['source'])
    
    # Create leaderboard_snapshots table for historical leaderboard data
    op.create_table(
        'leaderboard_snapshots',
        sa.Column('id', postgresql.UUID(as_uuid=True), server_default=sa.text('gen_random_uuid()'), nullable=False),
        sa.Column('period', sa.String(20), nullable=False),  # daily, weekly, monthly
        sa.Column('subject', sa.String(100), nullable=True),
        sa.Column('snapshot_date', sa.Date(), nullable=False),
        sa.Column('rankings', postgresql.JSONB(), nullable=False),  # Array of user rankings
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('period', 'subject', 'snapshot_date', name='uq_leaderboard_snapshot')
    )
    
    # Create indexes for leaderboard_snapshots
    op.create_index('idx_leaderboard_snapshots_period_date', 'leaderboard_snapshots', ['period', 'snapshot_date'])
    op.create_index('idx_leaderboard_snapshots_subject', 'leaderboard_snapshots', ['subject'])
    
    # Create daily_activities table for streak tracking
    op.create_table(
        'daily_activities',
        sa.Column('id', postgresql.UUID(as_uuid=True), server_default=sa.text('gen_random_uuid()'), nullable=False),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('activity_date', sa.Date(), nullable=False),
        sa.Column('points_earned', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('lessons_completed', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('quizzes_completed', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('study_time_minutes', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('achievements_unlocked', postgresql.ARRAY(sa.String()), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
        sa.UniqueConstraint('user_id', 'activity_date', name='uq_user_daily_activity')
    )
    
    # Create indexes for daily_activities
    op.create_index('idx_daily_activities_user_date', 'daily_activities', ['user_id', 'activity_date'])
    op.create_index('idx_daily_activities_date', 'daily_activities', ['activity_date'])
    
    # Create achievement_progress table for tracking progress towards achievements
    op.create_table(
        'achievement_progress',
        sa.Column('id', postgresql.UUID(as_uuid=True), server_default=sa.text('gen_random_uuid()'), nullable=False),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('achievement_key', sa.String(100), nullable=False),
        sa.Column('progress_data', postgresql.JSONB(), nullable=False),
        sa.Column('current_value', sa.Float(), nullable=False, server_default='0'),
        sa.Column('target_value', sa.Float(), nullable=False),
        sa.Column('percentage', sa.Float(), nullable=False, server_default='0'),
        sa.Column('last_updated', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
        sa.UniqueConstraint('user_id', 'achievement_key', name='uq_user_achievement_progress')
    )
    
    # Create indexes for achievement_progress
    op.create_index('idx_achievement_progress_user', 'achievement_progress', ['user_id'])
    op.create_index('idx_achievement_progress_percentage', 'achievement_progress', ['percentage'])
    
    # Create challenges table for special time-limited challenges
    op.create_table(
        'challenges',
        sa.Column('id', postgresql.UUID(as_uuid=True), server_default=sa.text('gen_random_uuid()'), nullable=False),
        sa.Column('name', sa.String(100), nullable=False, unique=True),
        sa.Column('description', sa.Text(), nullable=False),
        sa.Column('challenge_type', sa.String(50), nullable=False),  # daily, weekly, special
        sa.Column('requirements', postgresql.JSONB(), nullable=False),
        sa.Column('rewards', postgresql.JSONB(), nullable=False),
        sa.Column('start_date', sa.DateTime(timezone=True), nullable=False),
        sa.Column('end_date', sa.DateTime(timezone=True), nullable=False),
        sa.Column('is_active', sa.Boolean(), nullable=False, server_default='true'),
        sa.Column('max_participants', sa.Integer(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint('id')
    )
    
    # Create indexes for challenges
    op.create_index('idx_challenges_active_dates', 'challenges', ['is_active', 'start_date', 'end_date'])
    op.create_index('idx_challenges_type', 'challenges', ['challenge_type'])
    
    # Create user_challenges table for tracking user participation in challenges
    op.create_table(
        'user_challenges',
        sa.Column('id', postgresql.UUID(as_uuid=True), server_default=sa.text('gen_random_uuid()'), nullable=False),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('challenge_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('joined_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('progress', postgresql.JSONB(), nullable=True),
        sa.Column('completed', sa.Boolean(), nullable=False, server_default='false'),
        sa.Column('completed_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('rewards_claimed', sa.Boolean(), nullable=False, server_default='false'),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['challenge_id'], ['challenges.id'], ondelete='CASCADE'),
        sa.UniqueConstraint('user_id', 'challenge_id', name='uq_user_challenge')
    )
    
    # Create indexes for user_challenges
    op.create_index('idx_user_challenges_user', 'user_challenges', ['user_id'])
    op.create_index('idx_user_challenges_challenge', 'user_challenges', ['challenge_id'])
    op.create_index('idx_user_challenges_completed', 'user_challenges', ['completed'])
    
    # Add additional columns to achievements table if they don't exist
    op.add_column('achievements', sa.Column('category', sa.String(50), nullable=True))
    op.add_column('achievements', sa.Column('tier', sa.Integer(), nullable=True))
    op.add_column('achievements', sa.Column('prerequisites', postgresql.ARRAY(sa.String()), nullable=True))
    op.add_column('achievements', sa.Column('unlock_count', sa.Integer(), nullable=False, server_default='0'))
    op.add_column('achievements', sa.Column('first_unlocked_by', postgresql.UUID(as_uuid=True), nullable=True))
    op.add_column('achievements', sa.Column('first_unlocked_at', sa.DateTime(timezone=True), nullable=True))
    
    # Create a materialized view for real-time leaderboard (optional, for performance)
    op.execute("""
        CREATE MATERIALIZED VIEW IF NOT EXISTS leaderboard_realtime AS
        SELECT 
            u.id as user_id,
            u.username,
            u.avatar_url,
            u.points,
            u.level,
            u.streak_days,
            COUNT(DISTINCT ua.achievement_id) as achievements_count,
            RANK() OVER (ORDER BY u.points DESC) as global_rank,
            PERCENT_RANK() OVER (ORDER BY u.points) * 100 as percentile
        FROM users u
        LEFT JOIN user_achievements ua ON u.id = ua.user_id
        WHERE u.is_active = true
        GROUP BY u.id, u.username, u.avatar_url, u.points, u.level, u.streak_days
        WITH DATA;
    """)
    
    # Create index on materialized view
    op.execute("CREATE INDEX idx_leaderboard_realtime_points ON leaderboard_realtime(points DESC);")
    op.execute("CREATE INDEX idx_leaderboard_realtime_user_id ON leaderboard_realtime(user_id);")
    
    # Create trigger for updating daily activities
    op.execute("""
        CREATE OR REPLACE FUNCTION update_daily_activity()
        RETURNS TRIGGER AS $$
        BEGIN
            INSERT INTO daily_activities (user_id, activity_date, points_earned)
            VALUES (NEW.user_id, CURRENT_DATE, 0)
            ON CONFLICT (user_id, activity_date) DO NOTHING;
            
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql;
    """)
    
    # Create trigger for auto-refreshing materialized view (scheduled job recommended instead)
    op.execute("""
        CREATE OR REPLACE FUNCTION refresh_leaderboard()
        RETURNS void AS $$
        BEGIN
            REFRESH MATERIALIZED VIEW CONCURRENTLY leaderboard_realtime;
        END;
        $$ LANGUAGE plpgsql;
    """)
    
    # Add check constraints
    op.create_check_constraint(
        'check_points_positive',
        'point_transactions',
        'amount > 0'
    )
    
    op.create_check_constraint(
        'check_challenge_dates',
        'challenges',
        'end_date > start_date'
    )
    
    op.create_check_constraint(
        'check_achievement_progress',
        'achievement_progress',
        'percentage >= 0 AND percentage <= 100'
    )


def downgrade():
    # Drop materialized view
    op.execute("DROP MATERIALIZED VIEW IF EXISTS leaderboard_realtime CASCADE;")
    
    # Drop functions
    op.execute("DROP FUNCTION IF EXISTS update_daily_activity() CASCADE;")
    op.execute("DROP FUNCTION IF EXISTS refresh_leaderboard() CASCADE;")
    
    # Drop tables
    op.drop_table('user_challenges')
    op.drop_table('challenges')
    op.drop_table('achievement_progress')
    op.drop_table('daily_activities')
    op.drop_table('leaderboard_snapshots')
    op.drop_table('point_transactions')
    
    # Drop columns from achievements table
    op.drop_column('achievements', 'category')
    op.drop_column('achievements', 'tier')
    op.drop_column('achievements', 'prerequisites')
    op.drop_column('achievements', 'unlock_count')
    op.drop_column('achievements', 'first_unlocked_by')
    op.drop_column('achievements', 'first_unlocked_at')