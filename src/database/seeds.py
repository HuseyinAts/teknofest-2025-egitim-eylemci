"""
Database seeding system for TEKNOFEST 2025 Education Platform
Production-ready seed data management with environment-specific configurations
"""

import json
import logging
import random
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from pathlib import Path
import hashlib

from sqlalchemy.orm import Session
from sqlalchemy import text
from faker import Faker
from passlib.context import CryptContext

from .base import get_db_context
from .models import (
    User, UserRole, LearningPath, Module, Achievement,
    DifficultyLevel, ContentType
)
from ..config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()
fake = Faker(['tr_TR', 'en_US'])

# Password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


class DatabaseSeeder:
    """
    Manages database seeding with different strategies for different environments.
    """
    
    def __init__(self, session: Optional[Session] = None):
        """
        Initialize seeder.
        
        Args:
            session: Database session (creates new if not provided)
        """
        self.session = session
        self.created_entities = {
            'users': [],
            'learning_paths': [],
            'modules': [],
            'achievements': []
        }
    
    def seed_all(self, environment: str = "development"):
        """
        Seed all data based on environment.
        
        Args:
            environment: Target environment (development, staging, production)
        """
        logger.info(f"Starting database seeding for environment: {environment}")
        
        try:
            if environment == "production":
                self.seed_production()
            elif environment == "staging":
                self.seed_staging()
            else:
                self.seed_development()
            
            logger.info("Database seeding completed successfully")
            
        except Exception as e:
            logger.error(f"Database seeding failed: {e}")
            raise
    
    def seed_production(self):
        """Seed minimal production data"""
        with get_db_context() as db:
            try:
                # Create system users
                admin = self._create_admin_user(db)
                support = self._create_support_user(db)
                demo_teacher = self._create_demo_teacher(db)
                demo_student = self._create_demo_student(db)
                
                # Create essential achievements (all types)
                self._create_achievements(db, count=30)
                
                # Create core learning paths with full content
                paths = self._create_learning_paths(db, count=15, published=True)
                
                # Create sample assessments and progress for demo users
                if demo_student:
                    self._create_demo_progress(db, demo_student, paths[:3])
                
                db.commit()
                logger.info("Production data seeded successfully")
                
            except Exception as e:
                db.rollback()
                logger.error(f"Production seeding failed: {e}")
                raise
    
    def seed_staging(self):
        """Seed staging environment with test data"""
        with get_db_context() as db:
            # Create admin and test users
            admin = self._create_admin_user(db)
            users = self._create_test_users(db, count=50)
            
            # Create achievements
            achievements = self._create_achievements(db, count=20)
            
            # Create learning paths with modules
            paths = self._create_learning_paths(db, count=20, published=True)
            
            # Enroll users in paths
            self._enroll_users_in_paths(db, users, paths)
            
            # Create sample study sessions
            self._create_study_sessions(db, users)
            
            db.commit()
            logger.info("Staging data seeded")
    
    def seed_development(self):
        """Seed development environment with extensive test data"""
        with get_db_context() as db:
            # Create various users
            admin = self._create_admin_user(db)
            teacher = self._create_teacher_user(db)
            students = self._create_test_users(db, count=100)
            
            # Create achievements
            achievements = self._create_achievements(db, count=30)
            
            # Create learning paths
            paths = self._create_learning_paths(db, count=50, published=True)
            draft_paths = self._create_learning_paths(db, count=10, published=False)
            
            # Enroll users
            self._enroll_users_in_paths(db, students, paths)
            
            # Create study sessions
            self._create_study_sessions(db, students)
            
            # Award achievements
            self._award_achievements(db, students, achievements)
            
            # Create notifications
            self._create_notifications(db, students)
            
            db.commit()
            logger.info("Development data seeded")
    
    def _create_admin_user(self, db: Session) -> User:
        """Create admin user"""
        email = "admin@teknofest.com"
        
        # Check if already exists
        existing = db.query(User).filter_by(email=email).first()
        if existing:
            logger.info("Admin user already exists")
            return existing
        
        admin = User(
            email=email,
            username="admin",
            full_name="System Administrator",
            hashed_password=self._hash_password("Admin123!"),
            role=UserRole.ADMIN,
            is_active=True,
            is_verified=True,
            preferred_language="tr",
            created_at=datetime.utcnow()
        )
        
        db.add(admin)
        db.flush()
        
        self.created_entities['users'].append(admin)
        logger.info(f"Created admin user: {email}")
        
        return admin
    
    def _create_support_user(self, db: Session) -> User:
        """Create support user for production"""
        email = "support@teknofest.com"
        
        existing = db.query(User).filter_by(email=email).first()
        if existing:
            return existing
        
        support = User(
            email=email,
            username="support",
            full_name="Destek Ekibi",
            hashed_password=self._hash_password("Support123!"),
            role=UserRole.ADMIN,
            is_active=True,
            is_verified=True,
            preferred_language="tr",
            bio="TEKNOFEST Destek Ekibi",
            created_at=datetime.utcnow()
        )
        
        db.add(support)
        db.flush()
        
        self.created_entities['users'].append(support)
        logger.info(f"Created support user: {email}")
        
        return support
    
    def _create_demo_teacher(self, db: Session) -> User:
        """Create demo teacher for production"""
        email = "demo.teacher@teknofest.com"
        
        existing = db.query(User).filter_by(email=email).first()
        if existing:
            return existing
        
        teacher = User(
            email=email,
            username="demo_teacher",
            full_name="Demo Öğretmen",
            hashed_password=self._hash_password("DemoTeacher123!"),
            role=UserRole.TEACHER,
            is_active=True,
            is_verified=True,
            preferred_language="tr",
            bio="Demo hesap - Öğretmen özellikleri için",
            interests=["eğitim", "teknoloji", "yapay zeka"],
            created_at=datetime.utcnow()
        )
        
        db.add(teacher)
        db.flush()
        
        self.created_entities['users'].append(teacher)
        logger.info(f"Created demo teacher: {email}")
        
        return teacher
    
    def _create_demo_student(self, db: Session) -> User:
        """Create demo student for production"""
        email = "demo.student@teknofest.com"
        
        existing = db.query(User).filter_by(email=email).first()
        if existing:
            return existing
        
        student = User(
            email=email,
            username="demo_student",
            full_name="Demo Öğrenci",
            hashed_password=self._hash_password("DemoStudent123!"),
            role=UserRole.STUDENT,
            is_active=True,
            is_verified=True,
            preferred_language="tr",
            bio="Demo hesap - Öğrenci özellikleri için",
            interests=["matematik", "fizik", "programlama", "robotik"],
            total_study_time=240,
            streak_days=7,
            points=1500,
            level=3,
            created_at=datetime.utcnow()
        )
        
        db.add(student)
        db.flush()
        
        self.created_entities['users'].append(student)
        logger.info(f"Created demo student: {email}")
        
        return student
    
    def _create_teacher_user(self, db: Session) -> User:
        """Create teacher user"""
        email = "teacher@teknofest.com"
        
        # Check if already exists
        existing = db.query(User).filter_by(email=email).first()
        if existing:
            return existing
        
        teacher = User(
            email=email,
            username="teacher",
            full_name="Örnek Öğretmen",
            hashed_password=self._hash_password("Teacher123!"),
            role=UserRole.TEACHER,
            is_active=True,
            is_verified=True,
            preferred_language="tr",
            bio="Deneyimli eğitmen",
            created_at=datetime.utcnow()
        )
        
        db.add(teacher)
        db.flush()
        
        self.created_entities['users'].append(teacher)
        logger.info(f"Created teacher user: {email}")
        
        return teacher
    
    def _create_test_users(self, db: Session, count: int = 10) -> List[User]:
        """Create test student users"""
        users = []
        
        for i in range(count):
            # Generate unique data
            username = fake.user_name() + str(i)
            email = f"student{i}@test.com"
            
            # Check if already exists
            existing = db.query(User).filter_by(email=email).first()
            if existing:
                users.append(existing)
                continue
            
            user = User(
                email=email,
                username=username,
                full_name=fake.name(),
                hashed_password=self._hash_password("Test123!"),
                role=UserRole.STUDENT,
                is_active=True,
                is_verified=random.choice([True, False]),
                preferred_language=random.choice(['tr', 'en']),
                bio=fake.text(max_nb_chars=200) if random.random() > 0.5 else None,
                date_of_birth=fake.date_of_birth(minimum_age=10, maximum_age=25),
                interests=random.sample([
                    'matematik', 'fizik', 'kimya', 'biyoloji',
                    'programlama', 'robotik', 'yapay zeka', 'veri bilimi'
                ], k=random.randint(2, 4)),
                total_study_time=random.randint(0, 10000),
                streak_days=random.randint(0, 30),
                points=random.randint(0, 5000),
                level=random.randint(1, 10),
                created_at=fake.date_time_between(start_date='-1y', end_date='now')
            )
            
            db.add(user)
            users.append(user)
        
        db.flush()
        self.created_entities['users'].extend(users)
        logger.info(f"Created {len(users)} test users")
        
        return users
    
    def _create_achievements(self, db: Session, count: int = 10) -> List[Achievement]:
        """Create achievements"""
        achievements = []
        
        achievement_templates = [
            # Başlangıç Başarıları
            {
                "name": "İlk Adım",
                "description": "İlk dersini tamamla",
                "criteria": {"modules_completed": 1},
                "points": 10,
                "rarity": "common"
            },
            {
                "name": "Keşfetmeye Başla",
                "description": "3 farklı öğrenme yoluna kayıt ol",
                "criteria": {"paths_enrolled": 3},
                "points": 25,
                "rarity": "common"
            },
            {
                "name": "İlk Test",
                "description": "İlk değerlendirme sınavını tamamla",
                "criteria": {"assessments_completed": 1},
                "points": 15,
                "rarity": "common"
            },
            {
                "name": "Hızlı Öğrenci",
                "description": "Bir günde 5 modül tamamla",
                "criteria": {"daily_modules": 5},
                "points": 50,
                "rarity": "rare"
            },
            {
                "name": "Süreklilik",
                "description": "7 gün üst üste çalış",
                "criteria": {"streak_days": 7},
                "points": 100,
                "rarity": "rare"
            },
            {
                "name": "Uzman",
                "description": "Bir öğrenme yolunu tamamla",
                "criteria": {"paths_completed": 1},
                "points": 200,
                "rarity": "epic"
            },
            {
                "name": "Usta",
                "description": "10 öğrenme yolunu tamamla",
                "criteria": {"paths_completed": 10},
                "points": 1000,
                "rarity": "legendary"
            },
            {
                "name": "Mükemmeliyetçi",
                "description": "Bir sınavdan 100 puan al",
                "criteria": {"perfect_score": True},
                "points": 150,
                "rarity": "epic"
            },
            {
                "name": "Çalışkan",
                "description": "Toplam 100 saat çalış",
                "criteria": {"total_hours": 100},
                "points": 500,
                "rarity": "epic"
            },
            {
                "name": "Erken Kalkan",
                "description": "Sabah 6'da bir ders tamamla",
                "criteria": {"early_bird": True},
                "points": 30,
                "rarity": "common"
            },
            {
                "name": "Gece Kuşu",
                "description": "Gece yarısından sonra çalış",
                "criteria": {"night_owl": True},
                "points": 30,
                "rarity": "common"
            },
            {
                "name": "Sosyal Öğrenci",
                "description": "5 arkadaşını platforma davet et",
                "criteria": {"referrals": 5},
                "points": 100,
                "rarity": "rare"
            }
        ]
        
        for i, template in enumerate(achievement_templates[:count]):
            # Check if already exists
            existing = db.query(Achievement).filter_by(name=template["name"]).first()
            if existing:
                achievements.append(existing)
                continue
            
            achievement = Achievement(
                name=template["name"],
                description=template["description"],
                criteria=template["criteria"],
                points=template["points"],
                rarity=template["rarity"],
                icon_url=f"/icons/achievement_{i}.png"
            )
            
            db.add(achievement)
            achievements.append(achievement)
        
        db.flush()
        self.created_entities['achievements'] = achievements
        logger.info(f"Created {len(achievements)} achievements")
        
        return achievements
    
    def _create_learning_paths(
        self, 
        db: Session, 
        count: int = 10, 
        published: bool = True
    ) -> List[LearningPath]:
        """Create learning paths with modules"""
        paths = []
        
        topics = [
            # Programlama ve Yazılım
            {
                "title": "Python Programlama Temelleri",
                "description": "Python ile programlamaya başlangıç yapın. Değişkenler, döngüler ve fonksiyonlar gibi temel kavramları öğrenin.",
                "objectives": [
                    "Python kurulumu ve geliştirme ortamı",
                    "Değişkenler ve veri tipleri",
                    "Kontrol yapıları (if, for, while)",
                    "Fonksiyonlar ve modüller",
                    "Hata yönetimi"
                ],
                "prerequisites": [],
                "tags": ["python", "programlama", "başlangıç"],
                "difficulty": DifficultyLevel.BEGINNER,
                "hours": 25
            },
            {
                "title": "İleri Python Programlama",
                "description": "Python'da ileri seviye konular: OOP, dekoratörler, generators ve async programlama.",
                "objectives": [
                    "Nesne yönelimli programlama",
                    "Dekoratörler ve context managers",
                    "Generators ve iterators",
                    "Async/await programlama",
                    "Meta programlama"
                ],
                "prerequisites": ["Python Programlama Temelleri"],
                "tags": ["python", "ileri seviye", "OOP"],
                "difficulty": DifficultyLevel.ADVANCED,
                "hours": 40
            },
            # Veri Bilimi ve Yapay Zeka
            {
                "title": "Veri Bilimi ile Tanışma",
                "description": "Veri bilimi dünyasına giriş yapın. Veri analizi, görselleştirme ve istatistiksel yöntemleri öğrenin.",
                "objectives": [
                    "Veri bilimi nedir ve neden önemlidir",
                    "NumPy ve Pandas ile veri işleme",
                    "Matplotlib ve Seaborn ile görselleştirme",
                    "Temel istatistik ve olasılık",
                    "Veri temizleme ve ön işleme"
                ],
                "prerequisites": ["Python Programlama Temelleri"],
                "tags": ["veri bilimi", "analitik", "python", "pandas"],
                "difficulty": DifficultyLevel.INTERMEDIATE,
                "hours": 45
            },
            {
                "title": "Makine Öğrenmesi 101",
                "description": "Makine öğrenmesi dünyasına kapsamlı bir giriş. Temel algoritmalar ve uygulamaları öğrenin.",
                "objectives": [
                    "Makine öğrenmesi temelleri ve türleri",
                    "Supervised learning algoritmaları",
                    "Unsupervised learning teknikleri",
                    "Model değerlendirme metrikleri",
                    "Scikit-learn ile uygulama"
                ],
                "prerequisites": ["Python Programlama Temelleri", "Veri Bilimi ile Tanışma"],
                "tags": ["makine öğrenmesi", "AI", "scikit-learn"],
                "difficulty": DifficultyLevel.INTERMEDIATE,
                "hours": 65
            },
            {
                "title": "Derin Öğrenme ve Yapay Sinir Ağları",
                "description": "TensorFlow ve PyTorch ile derin öğrenme uygulamaları geliştirin.",
                "objectives": [
                    "Yapay sinir ağları temelleri",
                    "CNN ile görüntü işleme",
                    "RNN ve LSTM ile zaman serileri",
                    "Transfer learning",
                    "Model optimizasyonu"
                ],
                "prerequisites": ["Makine Öğrenmesi 101"],
                "tags": ["deep learning", "tensorflow", "pytorch", "neural networks"],
                "difficulty": DifficultyLevel.ADVANCED,
                "hours": 85
            },
            
            # Web ve Mobil Geliştirme
            {
                "title": "Web Geliştirme Temelleri",
                "description": "HTML, CSS ve JavaScript ile modern web uygulamaları geliştirin.",
                "objectives": [
                    "HTML5 ve semantik yapı",
                    "CSS3 ve responsive tasarım",
                    "JavaScript ES6+ özellikleri",
                    "DOM manipülasyonu",
                    "AJAX ve Fetch API"
                ],
                "prerequisites": [],
                "tags": ["web", "html", "css", "javascript", "frontend"],
                "difficulty": DifficultyLevel.BEGINNER,
                "hours": 35
            },
            {
                "title": "React ile Modern Web Uygulamaları",
                "description": "React framework'ü ile profesyonel web uygulamaları geliştirin.",
                "objectives": [
                    "React temelleri ve JSX",
                    "Component yaşam döngüsü",
                    "State ve props yönetimi",
                    "React Hooks",
                    "Redux ile state yönetimi"
                ],
                "prerequisites": ["Web Geliştirme Temelleri"],
                "tags": ["react", "javascript", "frontend", "spa"],
                "difficulty": DifficultyLevel.INTERMEDIATE,
                "hours": 50
            },
            
            # Robotik ve IoT
            {
                "title": "Arduino ile Robotik Başlangıç",
                "description": "Arduino kullanarak robotik projeler geliştirin.",
                "objectives": [
                    "Arduino kartı ve sensörler",
                    "Temel elektronik bilgisi",
                    "Motor kontrolü",
                    "Sensör okuma ve veri işleme",
                    "Basit robot projesi"
                ],
                "prerequisites": [],
                "tags": ["robotik", "arduino", "elektronik", "maker"],
                "difficulty": DifficultyLevel.BEGINNER,
                "hours": 30
            },
            {
                "title": "IoT ve Akıllı Sistemler",
                "description": "Nesnelerin interneti ile akıllı sistemler tasarlayın.",
                "objectives": [
                    "IoT temelleri ve protokoller",
                    "MQTT ve HTTP protokolleri",
                    "ESP32/ESP8266 programlama",
                    "Bulut platformları entegrasyonu",
                    "Akıllı ev projesi"
                ],
                "prerequisites": ["Arduino ile Robotik Başlangıç"],
                "tags": ["IoT", "esp32", "mqtt", "smart home"],
                "difficulty": DifficultyLevel.INTERMEDIATE,
                "hours": 45
            },
            
            # Matematik ve Bilim
            {
                "title": "Yapay Zeka için Matematik",
                "description": "Yapay zeka ve makine öğrenmesi için gerekli matematik temelleri.",
                "objectives": [
                    "Lineer cebir temelleri",
                    "Kalkülüs ve türevler",
                    "Olasılık ve istatistik",
                    "Optimizasyon yöntemleri",
                    "Matris işlemleri"
                ],
                "prerequisites": [],
                "tags": ["matematik", "AI", "lineer cebir", "istatistik"],
                "difficulty": DifficultyLevel.INTERMEDIATE,
                "hours": 40
            },
            {
                "title": "Bilimsel Hesaplama ve Simülasyon",
                "description": "Python ile bilimsel hesaplama ve simülasyon teknikleri.",
                "objectives": [
                    "NumPy ile sayısal hesaplama",
                    "SciPy ile bilimsel araçlar",
                    "Diferansiyel denklem çözümleri",
                    "Monte Carlo simülasyonları",
                    "3D görselleştirme"
                ],
                "prerequisites": ["Python Programlama Temelleri", "Yapay Zeka için Matematik"],
                "tags": ["bilimsel hesaplama", "simülasyon", "numpy", "scipy"],
                "difficulty": DifficultyLevel.ADVANCED,
                "hours": 55
            },
            
            # Siber Güvenlik
            {
                "title": "Siber Güvenlik Temelleri",
                "description": "Bilgi güvenliği ve etik hacking temel konuları.",
                "objectives": [
                    "Güvenlik temel kavramları",
                    "Ağ güvenliği temelleri",
                    "Şifreleme yöntemleri",
                    "Web uygulama güvenliği",
                    "Güvenlik araçları kullanımı"
                ],
                "prerequisites": [],
                "tags": ["güvenlik", "cybersecurity", "ethical hacking"],
                "difficulty": DifficultyLevel.INTERMEDIATE,
                "hours": 40
            },
            
            # Oyun Geliştirme
            {
                "title": "Unity ile Oyun Geliştirme",
                "description": "Unity game engine kullanarak 2D ve 3D oyunlar geliştirin.",
                "objectives": [
                    "Unity editörü kullanımı",
                    "C# ile oyun programlama",
                    "Fizik ve collision detection",
                    "Animasyon ve particle sistemleri",
                    "Mobil oyun optimizasyonu"
                ],
                "prerequisites": [],
                "tags": ["oyun", "unity", "gamedev", "C#"],
                "difficulty": DifficultyLevel.INTERMEDIATE,
                "hours": 60
            }
        ]
        
        # Use all topics if count is larger
        topics_to_use = topics * ((count // len(topics)) + 1) if count > len(topics) else topics[:count]
        
        for i in range(min(count, len(topics_to_use))):
            topic = topics_to_use[i]
            
            # Generate unique slug
            slug = f"{topic['title'].lower().replace(' ', '-').replace('ı', 'i').replace('ğ', 'g').replace('ü', 'u').replace('ş', 's').replace('ö', 'o').replace('ç', 'c')}-{i}"
            
            # Check if already exists
            existing = db.query(LearningPath).filter_by(slug=slug).first()
            if existing:
                paths.append(existing)
                continue
            
            path = LearningPath(
                title=topic["title"] if i < len(topics) else f"{topic['title']} - Bölüm {i // len(topics) + 1}",
                description=topic["description"],
                slug=slug,
                objectives=topic["objectives"],
                prerequisites=topic.get("prerequisites", []),
                tags=topic.get("tags", ["teknoloji", "eğitim"]),
                difficulty=topic["difficulty"],
                estimated_hours=topic["hours"],
                language="tr",
                ai_generated=random.choice([True, False]),
                ai_model="GPT-4" if random.random() > 0.5 else "Claude",
                is_published=published,
                published_at=datetime.utcnow() if published else None,
                enrollment_count=random.randint(100, 5000) if published else 0,
                completion_count=random.randint(10, 500) if published else 0,
                average_rating=round(random.uniform(4.0, 5.0), 1) if published else None
            )
            
            db.add(path)
            db.flush()
            
            # Create modules for this path
            self._create_modules_for_path(db, path)
            
            paths.append(path)
        
        self.created_entities['learning_paths'].extend(paths)
        logger.info(f"Created {len(paths)} learning paths")
        
        return paths
    
    def _create_modules_for_path(self, db: Session, path: LearningPath):
        """Create comprehensive modules for a learning path"""
        
        # Module templates based on path difficulty
        if path.difficulty == DifficultyLevel.BEGINNER:
            module_templates = [
                ("Giriş ve Genel Bakış", ContentType.VIDEO, 30),
                ("Temel Kavramlar", ContentType.TEXT, 45),
                ("İlk Uygulama", ContentType.EXERCISE, 60),
                ("Alıştırmalar", ContentType.QUIZ, 20),
                ("Mini Proje", ContentType.PROJECT, 90)
            ]
        elif path.difficulty == DifficultyLevel.INTERMEDIATE:
            module_templates = [
                ("Konsept Tanıtımı", ContentType.VIDEO, 40),
                ("Detaylı Teori", ContentType.TEXT, 60),
                ("Kod Örnekleri", ContentType.EXERCISE, 75),
                ("Problem Çözme", ContentType.EXERCISE, 90),
                ("Değerlendirme", ContentType.QUIZ, 30),
                ("Uygulama Projesi", ContentType.PROJECT, 120)
            ]
        else:  # ADVANCED or EXPERT
            module_templates = [
                ("İleri Seviye Konseptler", ContentType.VIDEO, 50),
                ("Derinlemesine Analiz", ContentType.TEXT, 75),
                ("Karmaşık Uygulamalar", ContentType.EXERCISE, 100),
                ("Optimizasyon Teknikleri", ContentType.EXERCISE, 90),
                ("Gerçek Dünya Problemleri", ContentType.PROJECT, 150),
                ("Kapsamlı Değerlendirme", ContentType.QUIZ, 45),
                ("Capstone Projesi", ContentType.PROJECT, 180)
            ]
        
        module_count = len(module_templates) + random.randint(2, 5)
        
        for i in range(min(module_count, len(module_templates) + 5)):
            if i < len(module_templates):
                title_base, content_type, duration = module_templates[i]
                title = f"Bölüm {i+1}: {title_base}"
            else:
                title = f"Bölüm {i+1}: Ek İçerik - {fake.catch_phrase()}"
                content_type = random.choice(list(ContentType))
                duration = random.randint(30, 90)
            
            module = Module(
                learning_path_id=path.id,
                title=title,
                description=f"{path.title} konusunda {title.lower()} içeriği. Bu modülde önemli kavramları ve uygulamaları öğreneceksiniz.",
                order_index=i,
                content_type=content_type,
                content_url=f"/api/content/{path.slug}/module-{i+1}",
                content_data={
                    "duration_minutes": duration,
                    "difficulty": path.difficulty.value,
                    "resources": [
                        {"type": "pdf", "url": f"/resources/{path.slug}/module-{i+1}.pdf"},
                        {"type": "video", "url": f"/videos/{path.slug}/module-{i+1}.mp4"}
                    ] if content_type != ContentType.QUIZ else [],
                    "quiz_questions": 10 if content_type == ContentType.QUIZ else 0,
                    "has_certificate": i == module_count - 1
                },
                estimated_minutes=duration,
                is_mandatory=i < 3  # First 3 modules are mandatory
            )
            
            db.add(module)
            self.created_entities['modules'].append(module)
    
    def _enroll_users_in_paths(
        self, 
        db: Session, 
        users: List[User], 
        paths: List[LearningPath]
    ):
        """Enroll users in learning paths"""
        for user in users:
            # Each user enrolls in 1-5 paths
            enrolled_paths = random.sample(paths, k=min(random.randint(1, 5), len(paths)))
            
            for path in enrolled_paths:
                # Add to many-to-many relationship
                if path not in user.learning_paths:
                    user.learning_paths.append(path)
                    
                    # Update enrollment count
                    path.enrollment_count += 1
        
        db.flush()
        logger.info(f"Enrolled {len(users)} users in learning paths")
    
    def _create_study_sessions(self, db: Session, users: List[User]):
        """Create study sessions for users"""
        from .models import StudySession
        
        for user in users[:20]:  # Create for first 20 users
            session_count = random.randint(5, 20)
            
            for _ in range(session_count):
                started = fake.date_time_between(start_date='-30d', end_date='now')
                duration = random.randint(10, 120)
                
                session = StudySession(
                    user_id=user.id,
                    started_at=started,
                    ended_at=started + timedelta(minutes=duration),
                    duration_minutes=duration,
                    notes=fake.text(max_nb_chars=200) if random.random() > 0.7 else None,
                    ai_interactions=random.randint(0, 10)
                )
                
                db.add(session)
        
        db.flush()
        logger.info("Created study sessions")
    
    def _create_demo_progress(self, db: Session, user: User, paths: List[LearningPath]):
        """Create demo progress for demo users"""
        from .models import Progress, StudySession, Assessment
        
        for path in paths:
            # Enroll user in path
            if path not in user.learning_paths:
                user.learning_paths.append(path)
            
            # Get modules for this path
            modules = db.query(Module).filter_by(learning_path_id=path.id).order_by(Module.order_index).all()
            
            for i, module in enumerate(modules[:5]):  # Progress on first 5 modules
                progress_pct = 100.0 if i < 3 else random.uniform(30, 90)
                
                progress = Progress(
                    user_id=user.id,
                    module_id=module.id,
                    status="completed" if progress_pct == 100 else "in_progress",
                    progress_percentage=progress_pct,
                    completed_at=datetime.utcnow() - timedelta(days=random.randint(1, 30)) if progress_pct == 100 else None,
                    time_spent_minutes=random.randint(30, 120),
                    attempt_count=random.randint(1, 3)
                )
                db.add(progress)
                
                # Create study session for this module
                session = StudySession(
                    user_id=user.id,
                    module_id=module.id,
                    started_at=datetime.utcnow() - timedelta(days=random.randint(1, 30)),
                    ended_at=datetime.utcnow() - timedelta(days=random.randint(0, 29)),
                    duration_minutes=random.randint(30, 90),
                    ai_interactions=random.randint(1, 10),
                    notes="Demo kullanıcı çalışma notları"
                )
                db.add(session)
                
                # Create assessment if it's a quiz module
                if module.content_type == ContentType.QUIZ and progress_pct == 100:
                    assessment = Assessment(
                        user_id=user.id,
                        module_id=module.id,
                        type="quiz",
                        questions=[
                            {"id": j, "question": f"Soru {j+1}", "type": "multiple_choice"}
                            for j in range(10)
                        ],
                        answers=[
                            {"question_id": j, "answer": random.choice(["A", "B", "C", "D"])}
                            for j in range(10)
                        ],
                        score=random.uniform(70, 100),
                        max_score=100,
                        percentage=random.uniform(70, 100),
                        passed=True,
                        started_at=datetime.utcnow() - timedelta(days=random.randint(1, 30)),
                        completed_at=datetime.utcnow() - timedelta(days=random.randint(0, 29)),
                        time_spent_seconds=random.randint(600, 1800),
                        feedback={"message": "Harika performans!"}
                    )
                    db.add(assessment)
        
        db.flush()
        logger.info(f"Created demo progress for user: {user.email}")
    
    def _award_achievements(
        self, 
        db: Session, 
        users: List[User], 
        achievements: List[Achievement]
    ):
        """Award achievements to users"""
        for user in users[:30]:  # Award to first 30 users
            # Each user gets 1-3 achievements
            user_achievements = random.sample(
                achievements, 
                k=min(random.randint(1, 3), len(achievements))
            )
            
            for achievement in user_achievements:
                if achievement not in user.achievements:
                    user.achievements.append(achievement)
                    user.points += achievement.points
        
        db.flush()
        logger.info("Awarded achievements to users")
    
    def _create_notifications(self, db: Session, users: List[User]):
        """Create sample notifications"""
        from .models import Notification
        
        notification_templates = [
            ("new_content", "Yeni İçerik", "Yeni bir ders eklendi: {}"),
            ("achievement", "Başarı Kazandın", "Tebrikler! {} başarısını kazandın"),
            ("reminder", "Hatırlatma", "Bugün çalışmayı unutma!"),
            ("progress", "İlerleme", "Harika gidiyorsun! %{} tamamladın")
        ]
        
        for user in users[:10]:  # Create for first 10 users
            for _ in range(random.randint(1, 5)):
                template = random.choice(notification_templates)
                
                notification = Notification(
                    user_id=user.id,
                    type=template[0],
                    title=template[1],
                    message=template[2].format(fake.word()),
                    is_read=random.choice([True, False]),
                    created_at=fake.date_time_between(start_date='-7d', end_date='now')
                )
                
                db.add(notification)
        
        db.flush()
        logger.info("Created notifications")
    
    def _hash_password(self, password: str) -> str:
        """Hash password using bcrypt"""
        return pwd_context.hash(password)
    
    def clear_all_data(self):
        """Clear all data from database (use with caution!)"""
        if settings.is_production():
            raise Exception("Cannot clear data in production!")
        
        with get_db_context() as db:
            # Disable foreign key checks
            db.execute(text("SET session_replication_role = 'replica';"))
            
            # Get all tables
            tables = [
                'notifications', 'audit_logs', 'progress', 'assessments',
                'study_sessions', 'user_achievements', 'achievements',
                'modules', 'user_learning_paths', 'learning_paths', 'users'
            ]
            
            # Truncate tables in order
            for table in tables:
                try:
                    db.execute(text(f"TRUNCATE TABLE {table} CASCADE;"))
                    logger.info(f"Cleared table: {table}")
                except Exception as e:
                    logger.warning(f"Could not clear table {table}: {e}")
            
            # Re-enable foreign key checks
            db.execute(text("SET session_replication_role = 'origin';"))
            
            db.commit()
            logger.info("All data cleared")
    
    def get_seed_stats(self) -> Dict[str, int]:
        """Get statistics about seeded data"""
        with get_db_context() as db:
            stats = {
                "users": db.query(User).count(),
                "learning_paths": db.query(LearningPath).count(),
                "modules": db.query(Module).count(),
                "achievements": db.query(Achievement).count(),
            }
            
            return stats


# Convenience functions
def seed_database(environment: str = None):
    """Seed database with appropriate data"""
    if environment is None:
        environment = "production" if settings.is_production() else "development"
    
    seeder = DatabaseSeeder()
    seeder.seed_all(environment)


def clear_database():
    """Clear all data from database (development only)"""
    seeder = DatabaseSeeder()
    seeder.clear_all_data()


def get_seed_statistics():
    """Get statistics about seeded data"""
    seeder = DatabaseSeeder()
    return seeder.get_seed_stats()