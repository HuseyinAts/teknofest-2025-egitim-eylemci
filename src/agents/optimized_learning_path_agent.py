"""
Optimized Learning Path Agent with Caching and Performance Improvements
TEKNOFEST 2025 - High Performance Learning Path Generation
"""

from typing import Dict, List, Optional, Any, Tuple
import json
from pathlib import Path
from datetime import datetime, timedelta
import asyncio
import logging

from src.container import scoped
from src.config import Settings
from src.core.cache import get_cache, cached, CacheKeyBuilder
from src.database.optimized_db import get_db_session, QueryOptimizer
from src.database.repositories import StudentRepository, CourseRepository

logger = logging.getLogger(__name__)


@scoped
class OptimizedLearningPathAgent:
    """Optimized Learning Path Agent with caching and performance improvements"""
    
    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings
        self.cache = get_cache()
        self.db = get_db_session()
        self.optimizer = QueryOptimizer()
        
        # Lazy load heavy data
        self._vark_quiz = None
        self._curriculum = None
        
        # Performance metrics
        self.metrics = {
            'cache_hits': 0,
            'cache_misses': 0,
            'generation_time': 0,
            'db_queries': 0
        }
    
    @property
    def vark_quiz(self):
        """Lazy load VARK questions with caching"""
        if self._vark_quiz is None:
            cache_key = "vark_quiz:questions"
            self._vark_quiz = self.cache.get(cache_key)
            
            if self._vark_quiz is None:
                self._vark_quiz = self._load_vark_questions()
                self.cache.set(cache_key, self._vark_quiz, ttl=86400)  # Cache for 24 hours
                self.metrics['cache_misses'] += 1
            else:
                self.metrics['cache_hits'] += 1
        
        return self._vark_quiz
    
    @property
    def curriculum(self):
        """Lazy load curriculum with caching"""
        if self._curriculum is None:
            cache_key = "meb:curriculum"
            self._curriculum = self.cache.get(cache_key)
            
            if self._curriculum is None:
                self._curriculum = self._load_meb_curriculum()
                self.cache.set(cache_key, self._curriculum, ttl=86400)  # Cache for 24 hours
                self.metrics['cache_misses'] += 1
            else:
                self.metrics['cache_hits'] += 1
        
        return self._curriculum
    
    def _load_vark_questions(self) -> List[Dict]:
        """Load VARK questions from file or database"""
        return [
            {
                'id': 1,
                'question': 'Yeni bir konuyu öğrenirken tercih ettiğiniz yöntem?',
                'options': {
                    'V': 'Görsel materyaller ve şemalar',
                    'A': 'Sesli anlatımlar ve tartışmalar',
                    'R': 'Kitap ve yazılı kaynaklar',
                    'K': 'Uygulamalı deney ve pratik'
                }
            },
            {
                'id': 2,
                'question': 'Bir problemi çözerken ilk yaklaşımınız?',
                'options': {
                    'V': 'Diyagram veya grafik çizerim',
                    'A': 'Başkalarıyla tartışırım',
                    'R': 'Notlar alır, araştırırım',
                    'K': 'Deneme yanılma yöntemi kullanırım'
                }
            }
        ]
    
    def _load_meb_curriculum(self) -> Dict:
        """Load MEB curriculum from cache or database"""
        return {
            '9': {
                'Matematik': {
                    'topics': ['Kümeler', 'Sayılar', 'Denklemler', 'Fonksiyonlar'],
                    'hours': 180,
                    'prerequisites': {
                        'Sayılar': ['Kümeler'],
                        'Denklemler': ['Sayılar'],
                        'Fonksiyonlar': ['Denklemler']
                    }
                },
                'Fizik': {
                    'topics': ['Hareket', 'Kuvvet', 'Enerji', 'Elektrik'],
                    'hours': 144,
                    'prerequisites': {
                        'Kuvvet': ['Hareket'],
                        'Enerji': ['Kuvvet'],
                        'Elektrik': ['Enerji']
                    }
                }
            }
        }
    
    @cached(prefix="learning_style", ttl=3600)
    def detect_learning_style(self, student_id: str, student_responses: List[str]) -> Dict:
        """Detect learning style with caching"""
        start_time = datetime.now()
        
        scores = {
            'visual': 0,
            'auditory': 0,
            'reading': 0,
            'kinesthetic': 0
        }
        
        # Optimized keyword matching using sets for O(1) lookup
        visual_keywords = set(['görsel', 'şema', 'grafik', 'resim', 'video', 'animasyon', 'renk'])
        auditory_keywords = set(['dinle', 'anlat', 'konuş', 'ses', 'müzik', 'tartış', 'açıkla'])
        reading_keywords = set(['oku', 'yaz', 'not', 'metin', 'kitap', 'makale', 'araştır'])
        kinesthetic_keywords = set(['yap', 'uygula', 'deney', 'hareket', 'dokun', 'pratik', 'el'])
        
        # Process responses in parallel
        for response in student_responses:
            words = set(response.lower().split())
            
            scores['visual'] += len(words & visual_keywords)
            scores['auditory'] += len(words & auditory_keywords)
            scores['reading'] += len(words & reading_keywords)
            scores['kinesthetic'] += len(words & kinesthetic_keywords)
        
        # Calculate dominant style
        dominant_style = max(scores, key=scores.get)
        total_score = sum(scores.values())
        
        percentages = {
            k: (v/total_score * 100 if total_score > 0 else 0)
            for k, v in scores.items()
        }
        
        result = {
            'student_id': student_id,
            'dominant_style': dominant_style,
            'scores': scores,
            'percentages': percentages,
            'confidence': percentages[dominant_style] / 100.0 if total_score > 0 else 0.0,
            'recommendation': self.get_style_recommendation(dominant_style),
            'processing_time': (datetime.now() - start_time).total_seconds()
        }
        
        # Update metrics
        self.metrics['generation_time'] += result['processing_time']
        
        return result
    
    def get_style_recommendation(self, style: str) -> str:
        """Get learning style recommendation from cache"""
        cache_key = f"recommendation:{style}"
        recommendation = self.cache.get(cache_key)
        
        if recommendation:
            return recommendation
        
        recommendations = {
            'visual': 'Görsel materyaller, infografikler ve videolar tercih edilmeli',
            'auditory': 'Sesli anlatımlar, podcast ve grup tartışmaları önerilir',
            'reading': 'Yazılı kaynaklar, e-kitaplar ve detaylı notlar kullanılmalı',
            'kinesthetic': 'Uygulamalı aktiviteler, deneyler ve simülasyonlar tercih edilmeli'
        }
        
        recommendation = recommendations.get(style, 'Karma öğrenme yöntemleri önerilir')
        self.cache.set(cache_key, recommendation, ttl=86400)
        
        return recommendation
    
    @cached(prefix="learning_path", ttl=1800)
    def generate_learning_path(self, student_profile: Dict) -> Dict:
        """Generate optimized learning path with caching"""
        start_time = datetime.now()
        
        # Extract profile data
        student_id = student_profile.get('student_id')
        grade = student_profile.get('grade', 9)
        subject = student_profile.get('subject', 'Matematik')
        learning_style = student_profile.get('learning_style', 'visual')
        current_level = student_profile.get('current_level', 0.5)
        target_level = student_profile.get('target_level', 0.9)
        duration_weeks = student_profile.get('duration_weeks', 12)
        
        # Get curriculum from cache
        subject_curriculum = self.curriculum.get(str(grade), {}).get(subject, {})
        
        # Calculate ZPD levels efficiently
        zpd_levels = self._calculate_zpd_levels_optimized(
            current_level, target_level, duration_weeks
        )
        
        # Generate milestones with batching
        milestones = self._generate_milestones_batch(
            subject_curriculum.get('topics', []),
            zpd_levels,
            learning_style
        )
        
        # Create learning path
        learning_path = {
            'student_id': student_id,
            'created_at': datetime.utcnow().isoformat(),
            'duration_weeks': duration_weeks,
            'subject': subject,
            'grade': grade,
            'learning_style': learning_style,
            'current_level': current_level,
            'target_level': target_level,
            'zpd_progression': zpd_levels,
            'milestones': milestones,
            'total_hours': subject_curriculum.get('hours', 0),
            'processing_time': (datetime.now() - start_time).total_seconds(),
            'cache_key': CacheKeyBuilder.build('learning_path', student_id, subject, grade)
        }
        
        # Update metrics
        self.metrics['generation_time'] += learning_path['processing_time']
        
        return learning_path
    
    def _calculate_zpd_levels_optimized(self, current: float, target: float, weeks: int) -> List[float]:
        """Optimized ZPD level calculation"""
        if current >= target:
            return [current] * weeks
        
        # Use numpy-like calculation for better performance
        step = (target - current) / weeks
        return [current + (step * week) for week in range(weeks + 1)]
    
    def _generate_milestones_batch(self, topics: List[str], zpd_levels: List[float], 
                                   learning_style: str) -> List[Dict]:
        """Generate milestones in batch for better performance"""
        milestones = []
        
        # Pre-calculate resources for each learning style
        style_resources = self._get_style_resources_cached(learning_style)
        
        for i, topic in enumerate(topics):
            week = (i * len(zpd_levels)) // len(topics)
            
            milestone = {
                'week': week + 1,
                'topic': topic,
                'difficulty_level': zpd_levels[min(week, len(zpd_levels) - 1)],
                'resources': style_resources,
                'assessment_type': self._get_assessment_type(learning_style),
                'estimated_hours': 10,
                'prerequisites': []  # Would be loaded from curriculum
            }
            
            milestones.append(milestone)
        
        return milestones
    
    @cached(prefix="style_resources", ttl=86400)
    def _get_style_resources_cached(self, learning_style: str) -> List[str]:
        """Get cached resources for learning style"""
        resources_map = {
            'visual': ['Video dersleri', 'İnfografikler', 'Akış şemaları', '3D modeller'],
            'auditory': ['Podcast\'ler', 'Sesli kitaplar', 'Grup tartışmaları', 'Anlatımlı videolar'],
            'reading': ['E-kitaplar', 'Makaleler', 'Ders notları', 'Araştırma raporları'],
            'kinesthetic': ['Simülasyonlar', 'Laboratuvar deneyleri', 'İnteraktif uygulamalar', 'Projeler']
        }
        
        return resources_map.get(learning_style, ['Karma kaynaklar'])
    
    def _get_assessment_type(self, learning_style: str) -> str:
        """Get assessment type for learning style"""
        assessment_map = {
            'visual': 'Görsel sunum ve diyagram oluşturma',
            'auditory': 'Sözlü sunum ve tartışma',
            'reading': 'Yazılı rapor ve deneme',
            'kinesthetic': 'Proje ve uygulama'
        }
        
        return assessment_map.get(learning_style, 'Karma değerlendirme')
    
    async def generate_learning_path_async(self, student_profile: Dict) -> Dict:
        """Async learning path generation for better performance"""
        # Run CPU-intensive operations in executor
        loop = asyncio.get_event_loop()
        
        # Parallel execution of independent operations
        tasks = [
            loop.run_in_executor(None, self.detect_learning_style, 
                                student_profile.get('student_id'), 
                                student_profile.get('responses', [])),
            loop.run_in_executor(None, self._load_student_history,
                                student_profile.get('student_id'))
        ]
        
        results = await asyncio.gather(*tasks)
        
        # Combine results
        student_profile['learning_style'] = results[0]['dominant_style']
        student_profile['history'] = results[1]
        
        # Generate path
        return self.generate_learning_path(student_profile)
    
    def _load_student_history(self, student_id: str) -> Dict:
        """Load student history with caching"""
        cache_key = f"student_history:{student_id}"
        history = self.cache.get(cache_key)
        
        if history:
            self.metrics['cache_hits'] += 1
            return history
        
        # Simulate database query (would be actual DB call)
        history = {
            'completed_topics': [],
            'quiz_scores': [],
            'learning_time': 0
        }
        
        self.cache.set(cache_key, history, ttl=3600)
        self.metrics['cache_misses'] += 1
        self.metrics['db_queries'] += 1
        
        return history
    
    def get_performance_metrics(self) -> Dict:
        """Get agent performance metrics"""
        cache_metrics = self.cache.get_metrics()
        
        return {
            'agent_metrics': self.metrics,
            'cache_metrics': cache_metrics,
            'db_metrics': self.db.get_performance_stats(),
            'optimization_suggestions': self._get_optimization_suggestions()
        }
    
    def _get_optimization_suggestions(self) -> List[str]:
        """Get performance optimization suggestions"""
        suggestions = []
        
        cache_hit_rate = self.metrics['cache_hits'] / max(
            self.metrics['cache_hits'] + self.metrics['cache_misses'], 1
        ) * 100
        
        if cache_hit_rate < 70:
            suggestions.append("Consider increasing cache TTL for better hit rate")
        
        if self.metrics['db_queries'] > 100:
            suggestions.append("High database query count - consider batch operations")
        
        if self.metrics['generation_time'] > 10:
            suggestions.append("Long generation time - consider async processing")
        
        return suggestions
