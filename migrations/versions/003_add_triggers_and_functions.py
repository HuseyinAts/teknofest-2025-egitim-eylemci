"""Add database triggers and functions for automation

Revision ID: 003
Revises: 002
Create Date: 2025-01-21 10:02:00.000000

"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = '003'
down_revision = '002'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Create function to update updated_at timestamp
    op.execute("""
        CREATE OR REPLACE FUNCTION update_updated_at_column()
        RETURNS TRIGGER AS $$
        BEGIN
            NEW.updated_at = NOW();
            RETURN NEW;
        END;
        $$ language 'plpgsql';
    """)
    
    # Add update triggers to all tables with updated_at column
    tables_with_updated_at = [
        'users', 'learning_paths', 'modules', 'study_sessions',
        'assessments', 'progress', 'achievements', 'notifications'
    ]
    
    for table in tables_with_updated_at:
        op.execute(f"""
            CREATE TRIGGER update_{table}_updated_at 
            BEFORE UPDATE ON {table}
            FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
        """)
    
    # Function to update user statistics after study session
    op.execute("""
        CREATE OR REPLACE FUNCTION update_user_study_time()
        RETURNS TRIGGER AS $$
        BEGIN
            IF NEW.duration_minutes IS NOT NULL AND NEW.duration_minutes > 0 THEN
                UPDATE users 
                SET total_study_time = total_study_time + NEW.duration_minutes
                WHERE id = NEW.user_id;
            END IF;
            RETURN NEW;
        END;
        $$ language 'plpgsql';
    """)
    
    op.execute("""
        CREATE TRIGGER update_user_study_time_trigger
        AFTER INSERT OR UPDATE OF duration_minutes ON study_sessions
        FOR EACH ROW 
        WHEN (NEW.duration_minutes IS NOT NULL)
        EXECUTE FUNCTION update_user_study_time();
    """)
    
    # Function to update learning path enrollment count
    op.execute("""
        CREATE OR REPLACE FUNCTION update_enrollment_count()
        RETURNS TRIGGER AS $$
        BEGIN
            IF TG_OP = 'INSERT' THEN
                UPDATE learning_paths 
                SET enrollment_count = enrollment_count + 1
                WHERE id = NEW.learning_path_id;
            ELSIF TG_OP = 'DELETE' THEN
                UPDATE learning_paths 
                SET enrollment_count = enrollment_count - 1
                WHERE id = OLD.learning_path_id;
            END IF;
            RETURN NULL;
        END;
        $$ language 'plpgsql';
    """)
    
    op.execute("""
        CREATE TRIGGER update_enrollment_count_trigger
        AFTER INSERT OR DELETE ON user_learning_paths
        FOR EACH ROW EXECUTE FUNCTION update_enrollment_count();
    """)
    
    # Function to update user points when achievement is earned
    op.execute("""
        CREATE OR REPLACE FUNCTION award_achievement_points()
        RETURNS TRIGGER AS $$
        DECLARE
            achievement_points INTEGER;
        BEGIN
            SELECT points INTO achievement_points 
            FROM achievements 
            WHERE id = NEW.achievement_id;
            
            IF achievement_points > 0 THEN
                UPDATE users 
                SET points = points + achievement_points
                WHERE id = NEW.user_id;
            END IF;
            RETURN NEW;
        END;
        $$ language 'plpgsql';
    """)
    
    op.execute("""
        CREATE TRIGGER award_achievement_points_trigger
        AFTER INSERT ON user_achievements
        FOR EACH ROW EXECUTE FUNCTION award_achievement_points();
    """)
    
    # Function to calculate user level based on points
    op.execute("""
        CREATE OR REPLACE FUNCTION calculate_user_level()
        RETURNS TRIGGER AS $$
        DECLARE
            new_level INTEGER;
        BEGIN
            -- Calculate level based on points (100 points per level)
            new_level := GREATEST(1, FLOOR(NEW.points / 100) + 1);
            
            IF new_level != NEW.level THEN
                NEW.level := new_level;
                
                -- Create notification for level up
                IF new_level > OLD.level THEN
                    INSERT INTO notifications (user_id, type, title, message)
                    VALUES (NEW.id, 'level_up', 'Level Up!', 
                            'Congratulations! You reached level ' || new_level || '!');
                END IF;
            END IF;
            RETURN NEW;
        END;
        $$ language 'plpgsql';
    """)
    
    op.execute("""
        CREATE TRIGGER calculate_user_level_trigger
        BEFORE UPDATE OF points ON users
        FOR EACH ROW 
        WHEN (NEW.points != OLD.points)
        EXECUTE FUNCTION calculate_user_level();
    """)
    
    # Function to update progress when module is completed
    op.execute("""
        CREATE OR REPLACE FUNCTION update_learning_path_progress()
        RETURNS TRIGGER AS $$
        DECLARE
            total_modules INTEGER;
            completed_modules INTEGER;
            new_progress FLOAT;
        BEGIN
            IF NEW.status = 'completed' AND (OLD.status IS NULL OR OLD.status != 'completed') THEN
                -- Get total modules in learning path
                SELECT COUNT(*) INTO total_modules
                FROM modules m
                WHERE m.learning_path_id = (
                    SELECT learning_path_id FROM modules WHERE id = NEW.module_id
                );
                
                -- Get completed modules by user in this learning path
                SELECT COUNT(*) INTO completed_modules
                FROM progress p
                JOIN modules m ON p.module_id = m.id
                WHERE p.user_id = NEW.user_id 
                    AND p.status = 'completed'
                    AND m.learning_path_id = (
                        SELECT learning_path_id FROM modules WHERE id = NEW.module_id
                    );
                
                -- Calculate progress percentage
                IF total_modules > 0 THEN
                    new_progress := (completed_modules::FLOAT / total_modules::FLOAT) * 100;
                    
                    -- Update user_learning_paths progress
                    UPDATE user_learning_paths
                    SET progress = new_progress,
                        completed_at = CASE 
                            WHEN new_progress >= 100 THEN NOW() 
                            ELSE NULL 
                        END
                    WHERE user_id = NEW.user_id 
                        AND learning_path_id = (
                            SELECT learning_path_id FROM modules WHERE id = NEW.module_id
                        );
                    
                    -- Update learning path completion count if completed
                    IF new_progress >= 100 THEN
                        UPDATE learning_paths
                        SET completion_count = completion_count + 1
                        WHERE id = (SELECT learning_path_id FROM modules WHERE id = NEW.module_id);
                    END IF;
                END IF;
            END IF;
            RETURN NEW;
        END;
        $$ language 'plpgsql';
    """)
    
    op.execute("""
        CREATE TRIGGER update_learning_path_progress_trigger
        AFTER UPDATE OF status ON progress
        FOR EACH ROW 
        WHEN (NEW.status = 'completed')
        EXECUTE FUNCTION update_learning_path_progress();
    """)
    
    # Function to validate email format
    op.execute("""
        CREATE OR REPLACE FUNCTION validate_email()
        RETURNS TRIGGER AS $$
        BEGIN
            IF NEW.email !~ '^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$' THEN
                RAISE EXCEPTION 'Invalid email format: %', NEW.email;
            END IF;
            RETURN NEW;
        END;
        $$ language 'plpgsql';
    """)
    
    op.execute("""
        CREATE TRIGGER validate_email_trigger
        BEFORE INSERT OR UPDATE OF email ON users
        FOR EACH ROW EXECUTE FUNCTION validate_email();
    """)
    
    # Function to auto-generate slug for learning paths
    op.execute("""
        CREATE OR REPLACE FUNCTION generate_slug()
        RETURNS TRIGGER AS $$
        DECLARE
            base_slug TEXT;
            final_slug TEXT;
            counter INTEGER := 0;
        BEGIN
            IF NEW.slug IS NULL OR NEW.slug = '' THEN
                -- Generate base slug from title
                base_slug := regexp_replace(
                    lower(unaccent(NEW.title)),
                    '[^a-z0-9]+', '-', 'g'
                );
                base_slug := trim(both '-' from base_slug);
                
                final_slug := base_slug;
                
                -- Check for duplicates and add counter if needed
                WHILE EXISTS (SELECT 1 FROM learning_paths WHERE slug = final_slug AND id != NEW.id) LOOP
                    counter := counter + 1;
                    final_slug := base_slug || '-' || counter;
                END LOOP;
                
                NEW.slug := final_slug;
            END IF;
            RETURN NEW;
        END;
        $$ language 'plpgsql';
    """)
    
    # Note: unaccent extension needs to be enabled
    op.execute("CREATE EXTENSION IF NOT EXISTS unaccent;")
    
    op.execute("""
        CREATE TRIGGER generate_slug_trigger
        BEFORE INSERT OR UPDATE OF title ON learning_paths
        FOR EACH ROW 
        WHEN (NEW.slug IS NULL OR NEW.slug = '')
        EXECUTE FUNCTION generate_slug();
    """)
    
    # Function to clean up old notifications
    op.execute("""
        CREATE OR REPLACE FUNCTION cleanup_old_notifications()
        RETURNS void AS $$
        BEGIN
            -- Delete read notifications older than 30 days
            DELETE FROM notifications
            WHERE is_read = true 
                AND read_at < NOW() - INTERVAL '30 days';
            
            -- Delete unread notifications older than 90 days
            DELETE FROM notifications
            WHERE is_read = false 
                AND created_at < NOW() - INTERVAL '90 days';
        END;
        $$ language 'plpgsql';
    """)
    
    # Function to refresh materialized view
    op.execute("""
        CREATE OR REPLACE FUNCTION refresh_user_statistics()
        RETURNS void AS $$
        BEGIN
            REFRESH MATERIALIZED VIEW CONCURRENTLY user_statistics;
        END;
        $$ language 'plpgsql';
    """)
    
    # Create unique index for concurrent refresh
    op.execute("CREATE UNIQUE INDEX IF NOT EXISTS user_statistics_concurrent_idx ON user_statistics (user_id);")


def downgrade() -> None:
    # Drop triggers
    tables_with_triggers = [
        ('users', ['update_users_updated_at', 'calculate_user_level_trigger', 'validate_email_trigger']),
        ('learning_paths', ['update_learning_paths_updated_at', 'generate_slug_trigger']),
        ('modules', ['update_modules_updated_at']),
        ('study_sessions', ['update_study_sessions_updated_at', 'update_user_study_time_trigger']),
        ('assessments', ['update_assessments_updated_at']),
        ('progress', ['update_progress_updated_at', 'update_learning_path_progress_trigger']),
        ('achievements', ['update_achievements_updated_at']),
        ('notifications', ['update_notifications_updated_at']),
        ('user_learning_paths', ['update_enrollment_count_trigger']),
        ('user_achievements', ['award_achievement_points_trigger'])
    ]
    
    for table, triggers in tables_with_triggers:
        for trigger in triggers:
            op.execute(f"DROP TRIGGER IF EXISTS {trigger} ON {table};")
    
    # Drop functions
    functions = [
        'update_updated_at_column',
        'update_user_study_time',
        'update_enrollment_count',
        'award_achievement_points',
        'calculate_user_level',
        'update_learning_path_progress',
        'validate_email',
        'generate_slug',
        'cleanup_old_notifications',
        'refresh_user_statistics'
    ]
    
    for func in functions:
        op.execute(f"DROP FUNCTION IF EXISTS {func} CASCADE;")
    
    # Drop extensions
    op.execute("DROP EXTENSION IF EXISTS unaccent;")
    
    # Drop unique index for materialized view
    op.execute("DROP INDEX IF EXISTS user_statistics_concurrent_idx;")