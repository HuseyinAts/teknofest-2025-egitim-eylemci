"""Add IRT tables for adaptive testing

Revision ID: 005
Revises: 004
Create Date: 2025-08-21

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers
revision = '005'
down_revision = '004'
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Add IRT related tables with optimized indexes"""
    
    # Create IRT Item Bank table
    op.create_table(
        'irt_item_bank',
        sa.Column('id', postgresql.UUID(as_uuid=True), server_default=sa.text('gen_random_uuid()'), nullable=False),
        sa.Column('item_id', sa.String(255), nullable=False, unique=True),
        sa.Column('question_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('difficulty', sa.Float(), nullable=False),
        sa.Column('discrimination', sa.Float(), nullable=False, server_default='1.0'),
        sa.Column('guessing', sa.Float(), nullable=False, server_default='0.2'),
        sa.Column('upper_asymptote', sa.Float(), nullable=False, server_default='1.0'),
        sa.Column('subject', sa.String(100), nullable=False),
        sa.Column('topic', sa.String(255), nullable=True),
        sa.Column('grade_level', sa.Integer(), nullable=True),
        sa.Column('usage_count', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('exposure_rate', sa.Float(), nullable=False, server_default='0.0'),
        sa.Column('standard_errors', postgresql.JSONB(), nullable=True),
        sa.Column('fit_statistics', postgresql.JSONB(), nullable=True),
        sa.Column('calibration_sample_size', sa.Integer(), nullable=True),
        sa.Column('calibration_method', sa.String(50), nullable=True),
        sa.Column('is_active', sa.Boolean(), nullable=False, server_default='true'),
        sa.Column('created_at', sa.TIMESTAMP(), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=False),
        sa.Column('updated_at', sa.TIMESTAMP(), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['question_id'], ['questions.id'], ondelete='SET NULL'),
        sa.CheckConstraint('difficulty >= -4 AND difficulty <= 4', name='check_difficulty_range'),
        sa.CheckConstraint('discrimination >= 0.1 AND discrimination <= 3', name='check_discrimination_range'),
        sa.CheckConstraint('guessing >= 0 AND guessing <= 0.5', name='check_guessing_range'),
        sa.CheckConstraint('upper_asymptote >= 0.5 AND upper_asymptote <= 1', name='check_upper_asymptote_range')
    )
    
    # Create indexes for IRT Item Bank
    op.create_index('idx_irt_item_bank_item_id', 'irt_item_bank', ['item_id'])
    op.create_index('idx_irt_item_bank_subject_topic', 'irt_item_bank', ['subject', 'topic'])
    op.create_index('idx_irt_item_bank_grade_level', 'irt_item_bank', ['grade_level'])
    op.create_index('idx_irt_item_bank_difficulty', 'irt_item_bank', ['difficulty'])
    op.create_index('idx_irt_item_bank_usage', 'irt_item_bank', ['usage_count', 'exposure_rate'])
    op.create_index('idx_irt_item_bank_active', 'irt_item_bank', ['is_active'], postgresql_where=sa.text('is_active = true'))
    
    # Create IRT Student Ability table
    op.create_table(
        'irt_student_abilities',
        sa.Column('id', postgresql.UUID(as_uuid=True), server_default=sa.text('gen_random_uuid()'), nullable=False),
        sa.Column('student_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('theta', sa.Float(), nullable=False),
        sa.Column('standard_error', sa.Float(), nullable=False),
        sa.Column('confidence_lower', sa.Float(), nullable=False),
        sa.Column('confidence_upper', sa.Float(), nullable=False),
        sa.Column('estimation_method', sa.String(50), nullable=False),
        sa.Column('subject', sa.String(100), nullable=True),
        sa.Column('topic', sa.String(255), nullable=True),
        sa.Column('test_id', sa.String(255), nullable=True),
        sa.Column('session_id', sa.String(255), nullable=True),
        sa.Column('items_count', sa.Integer(), nullable=False),
        sa.Column('response_pattern', postgresql.ARRAY(sa.Integer()), nullable=True),
        sa.Column('test_information', sa.Float(), nullable=True),
        sa.Column('reliability', sa.Float(), nullable=True),
        sa.Column('convergence_iterations', sa.Integer(), nullable=True),
        sa.Column('timestamp', sa.TIMESTAMP(), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=False),
        sa.Column('metadata', postgresql.JSONB(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['student_id'], ['student_profiles.id'], ondelete='CASCADE'),
        sa.CheckConstraint('theta >= -4 AND theta <= 4', name='check_theta_range'),
        sa.CheckConstraint('standard_error >= 0', name='check_se_positive'),
        sa.CheckConstraint('reliability >= 0 AND reliability <= 1', name='check_reliability_range')
    )
    
    # Create indexes for Student Abilities
    op.create_index('idx_irt_abilities_student', 'irt_student_abilities', ['student_id'])
    op.create_index('idx_irt_abilities_student_subject', 'irt_student_abilities', ['student_id', 'subject'])
    op.create_index('idx_irt_abilities_timestamp', 'irt_student_abilities', ['timestamp'])
    op.create_index('idx_irt_abilities_test_id', 'irt_student_abilities', ['test_id'])
    op.create_index('idx_irt_abilities_session_id', 'irt_student_abilities', ['session_id'])
    
    # Create IRT Test Sessions table
    op.create_table(
        'irt_test_sessions',
        sa.Column('id', postgresql.UUID(as_uuid=True), server_default=sa.text('gen_random_uuid()'), nullable=False),
        sa.Column('session_id', sa.String(255), nullable=False, unique=True),
        sa.Column('student_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('subject', sa.String(100), nullable=False),
        sa.Column('topic', sa.String(255), nullable=True),
        sa.Column('test_type', sa.String(50), nullable=False, server_default='adaptive'),
        sa.Column('max_items', sa.Integer(), nullable=False, server_default='20'),
        sa.Column('min_items', sa.Integer(), nullable=False, server_default='5'),
        sa.Column('target_se', sa.Float(), nullable=False, server_default='0.3'),
        sa.Column('current_theta', sa.Float(), nullable=False, server_default='0.0'),
        sa.Column('current_se', sa.Float(), nullable=False, server_default='1.0'),
        sa.Column('items_administered', postgresql.ARRAY(sa.String()), nullable=False, server_default='{}'),
        sa.Column('responses', postgresql.ARRAY(sa.Integer()), nullable=False, server_default='{}'),
        sa.Column('response_times', postgresql.ARRAY(sa.Float()), nullable=True),
        sa.Column('current_item', sa.String(255), nullable=True),
        sa.Column('final_theta', sa.Float(), nullable=True),
        sa.Column('final_se', sa.Float(), nullable=True),
        sa.Column('status', sa.String(50), nullable=False, server_default='in_progress'),
        sa.Column('start_time', sa.TIMESTAMP(), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=False),
        sa.Column('end_time', sa.TIMESTAMP(), nullable=True),
        sa.Column('time_limit', sa.Interval(), nullable=True),
        sa.Column('metadata', postgresql.JSONB(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['student_id'], ['student_profiles.id'], ondelete='CASCADE'),
        sa.CheckConstraint('max_items >= min_items', name='check_items_range'),
        sa.CheckConstraint('target_se > 0 AND target_se <= 1', name='check_target_se_range')
    )
    
    # Create indexes for Test Sessions
    op.create_index('idx_irt_sessions_session_id', 'irt_test_sessions', ['session_id'])
    op.create_index('idx_irt_sessions_student', 'irt_test_sessions', ['student_id'])
    op.create_index('idx_irt_sessions_status', 'irt_test_sessions', ['status'])
    op.create_index('idx_irt_sessions_subject', 'irt_test_sessions', ['subject'])
    op.create_index('idx_irt_sessions_active', 'irt_test_sessions', ['status', 'student_id'], 
                    postgresql_where=sa.text("status = 'in_progress'"))
    
    # Create IRT Calibration History table
    op.create_table(
        'irt_calibration_history',
        sa.Column('id', postgresql.UUID(as_uuid=True), server_default=sa.text('gen_random_uuid()'), nullable=False),
        sa.Column('calibration_id', sa.String(255), nullable=False),
        sa.Column('subject', sa.String(100), nullable=False),
        sa.Column('topic', sa.String(255), nullable=True),
        sa.Column('calibration_method', sa.String(50), nullable=False),
        sa.Column('sample_size', sa.Integer(), nullable=False),
        sa.Column('items_calibrated', sa.Integer(), nullable=False),
        sa.Column('convergence_status', sa.String(50), nullable=False),
        sa.Column('fit_statistics', postgresql.JSONB(), nullable=True),
        sa.Column('parameters_before', postgresql.JSONB(), nullable=True),
        sa.Column('parameters_after', postgresql.JSONB(), nullable=False),
        sa.Column('calibration_time', sa.Float(), nullable=False),
        sa.Column('timestamp', sa.TIMESTAMP(), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=False),
        sa.Column('performed_by', postgresql.UUID(as_uuid=True), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['performed_by'], ['users.id'], ondelete='SET NULL')
    )
    
    # Create indexes for Calibration History
    op.create_index('idx_irt_calibration_subject', 'irt_calibration_history', ['subject'])
    op.create_index('idx_irt_calibration_timestamp', 'irt_calibration_history', ['timestamp'])
    
    # Add IRT columns to existing questions table
    op.add_column('questions', sa.Column('irt_difficulty', sa.Float(), nullable=True))
    op.add_column('questions', sa.Column('irt_discrimination', sa.Float(), nullable=True))
    op.add_column('questions', sa.Column('irt_guessing', sa.Float(), nullable=True))
    op.add_column('questions', sa.Column('irt_calibrated', sa.Boolean(), nullable=False, server_default='false'))
    op.add_column('questions', sa.Column('irt_fit_statistics', postgresql.JSONB(), nullable=True))
    
    # Add IRT columns to quiz_attempts table
    op.add_column('quiz_attempts', sa.Column('adaptive_test', sa.Boolean(), nullable=False, server_default='false'))
    op.add_column('quiz_attempts', sa.Column('initial_theta', sa.Float(), nullable=True))
    op.add_column('quiz_attempts', sa.Column('final_theta', sa.Float(), nullable=True))
    op.add_column('quiz_attempts', sa.Column('theta_trajectory', postgresql.ARRAY(sa.Float()), nullable=True))
    op.add_column('quiz_attempts', sa.Column('information_trajectory', postgresql.ARRAY(sa.Float()), nullable=True))
    
    # Create trigger for updating updated_at timestamp
    op.execute("""
        CREATE OR REPLACE FUNCTION update_irt_updated_at()
        RETURNS TRIGGER AS $$
        BEGIN
            NEW.updated_at = CURRENT_TIMESTAMP;
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql;
    """)
    
    op.execute("""
        CREATE TRIGGER update_irt_item_bank_updated_at
        BEFORE UPDATE ON irt_item_bank
        FOR EACH ROW
        EXECUTE FUNCTION update_irt_updated_at();
    """)
    
    # Create function for calculating test information
    op.execute("""
        CREATE OR REPLACE FUNCTION calculate_test_information(
            p_theta FLOAT,
            p_items JSONB
        ) RETURNS FLOAT AS $$
        DECLARE
            v_information FLOAT := 0;
            v_item JSONB;
            v_p FLOAT;
            v_q FLOAT;
            v_a FLOAT;
            v_b FLOAT;
            v_c FLOAT;
        BEGIN
            FOR v_item IN SELECT * FROM jsonb_array_elements(p_items)
            LOOP
                v_a := (v_item->>'discrimination')::FLOAT;
                v_b := (v_item->>'difficulty')::FLOAT;
                v_c := (v_item->>'guessing')::FLOAT;
                
                -- Calculate probability using 3PL model
                v_p := v_c + (1 - v_c) / (1 + exp(-v_a * (p_theta - v_b)));
                v_q := 1 - v_p;
                
                -- Calculate item information
                IF v_p > 0 AND v_q > 0 AND v_p > v_c THEN
                    v_information := v_information + 
                        (v_a * v_a * power(v_p - v_c, 2)) / 
                        (power(1 - v_c, 2) * v_p * v_q);
                END IF;
            END LOOP;
            
            RETURN v_information;
        END;
        $$ LANGUAGE plpgsql IMMUTABLE;
    """)
    
    # Create materialized view for item statistics
    op.execute("""
        CREATE MATERIALIZED VIEW irt_item_statistics AS
        SELECT 
            iib.item_id,
            iib.subject,
            iib.topic,
            iib.difficulty,
            iib.discrimination,
            iib.guessing,
            iib.usage_count,
            iib.exposure_rate,
            COUNT(DISTINCT its.student_id) as unique_students,
            AVG(CASE WHEN its.responses[array_position(its.items_administered, iib.item_id)] = 1 
                THEN 1 ELSE 0 END) as success_rate,
            STDDEV(CASE WHEN its.responses[array_position(its.items_administered, iib.item_id)] = 1 
                THEN 1 ELSE 0 END) as response_variance
        FROM irt_item_bank iib
        LEFT JOIN irt_test_sessions its ON iib.item_id = ANY(its.items_administered)
        WHERE iib.is_active = true
        GROUP BY iib.item_id, iib.subject, iib.topic, iib.difficulty, 
                 iib.discrimination, iib.guessing, iib.usage_count, iib.exposure_rate;
    """)
    
    # Create index on materialized view
    op.execute("CREATE INDEX idx_irt_item_stats_subject ON irt_item_statistics(subject);")
    op.execute("CREATE INDEX idx_irt_item_stats_difficulty ON irt_item_statistics(difficulty);")


def downgrade() -> None:
    """Remove IRT tables and related objects"""
    
    # Drop materialized view
    op.execute("DROP MATERIALIZED VIEW IF EXISTS irt_item_statistics CASCADE;")
    
    # Drop functions
    op.execute("DROP FUNCTION IF EXISTS calculate_test_information CASCADE;")
    op.execute("DROP FUNCTION IF EXISTS update_irt_updated_at CASCADE;")
    
    # Remove columns from existing tables
    op.drop_column('quiz_attempts', 'information_trajectory')
    op.drop_column('quiz_attempts', 'theta_trajectory')
    op.drop_column('quiz_attempts', 'final_theta')
    op.drop_column('quiz_attempts', 'initial_theta')
    op.drop_column('quiz_attempts', 'adaptive_test')
    
    op.drop_column('questions', 'irt_fit_statistics')
    op.drop_column('questions', 'irt_calibrated')
    op.drop_column('questions', 'irt_guessing')
    op.drop_column('questions', 'irt_discrimination')
    op.drop_column('questions', 'irt_difficulty')
    
    # Drop tables
    op.drop_table('irt_calibration_history')
    op.drop_table('irt_test_sessions')
    op.drop_table('irt_student_abilities')
    op.drop_table('irt_item_bank')