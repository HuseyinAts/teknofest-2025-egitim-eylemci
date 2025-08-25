"""
Kişiselleştirilmiş Öğrenme Yolu Agent'ı
TEKNOFEST 2025 - Eğitim Teknolojileri
"""

from typing import Dict, List, Optional
import json
from pathlib import Path
from datetime import datetime, timedelta
from src.container import scoped
from src.config import Settings

@scoped
class LearningPathAgent:
    """Kişiselleştirilmiş Öğrenme Yolu Agent'ı"""
    
    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings
        self.vark_quiz = self.load_vark_questions()
        self.curriculum = self.load_meb_curriculum()
        
    def load_vark_questions(self) -> List[Dict]:
        """VARK öğrenme stili test sorularını yükle"""
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
    
    def load_meb_curriculum(self) -> Dict:
        """MEB müfredatını yükle"""
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
                },
                'Türkçe': {
                    'topics': ['Dil Bilgisi', 'Anlatım', 'Edebiyat', 'Metin İnceleme'],
                    'hours': 144,
                    'prerequisites': {}
                }
            },
            '10': {
                'Matematik': {
                    'topics': ['Polinomlar', 'Trigonometri', 'Analitik Geometri'],
                    'hours': 180
                }
            }
        }
    
    def detect_learning_style(self, student_responses: List[str]) -> Dict:
        """VARK öğrenme stili tespiti"""
        scores = {
            'visual': 0,
            'auditory': 0, 
            'reading': 0,
            'kinesthetic': 0
        }
        
        # Türkçe anahtar kelime analizi
        visual_keywords = ['görsel', 'şema', 'grafik', 'resim', 'video', 'animasyon', 'renk']
        auditory_keywords = ['dinle', 'anlat', 'konuş', 'ses', 'müzik', 'tartış', 'açıkla']
        reading_keywords = ['oku', 'yaz', 'not', 'metin', 'kitap', 'makale', 'araştır']
        kinesthetic_keywords = ['yap', 'uygula', 'deney', 'hareket', 'dokun', 'pratik', 'el']
        
        for response in student_responses:
            response_lower = response.lower()
            
            if any(word in response_lower for word in visual_keywords):
                scores['visual'] += 1
            if any(word in response_lower for word in auditory_keywords):
                scores['auditory'] += 1
            if any(word in response_lower for word in reading_keywords):
                scores['reading'] += 1
            if any(word in response_lower for word in kinesthetic_keywords):
                scores['kinesthetic'] += 1
        
        # En yüksek skoru bul
        dominant_style = max(scores, key=scores.get)
        
        # Yüzdelik hesapla
        total_score = sum(scores.values())
        percentages = {k: (v/total_score * 100 if total_score > 0 else 0) 
                      for k, v in scores.items()}
        
        # Calculate confidence
        confidence = percentages[dominant_style] / 100.0 if total_score > 0 else 0.0
        
        return {
            'dominant_style': dominant_style,
            'primary_style': dominant_style,  # Alias for compatibility
            'scores': scores,
            'percentages': percentages,
            'confidence': confidence,
            'recommendation': self.get_style_recommendation(dominant_style)
        }
    
    def get_style_recommendation(self, style: str) -> str:
        """Öğrenme stiline göre öneri"""
        recommendations = {
            'visual': 'Görsel materyaller, infografikler ve videolar tercih edilmeli',
            'auditory': 'Sesli anlatımlar, podcast ve grup tartışmaları önerilir',
            'reading': 'Yazılı kaynaklar, e-kitaplar ve detaylı notlar kullanılmalı',
            'kinesthetic': 'Uygulamalı aktiviteler, deneyler ve simülasyonlar tercih edilmeli'
        }
        return recommendations.get(style, 'Karma öğrenme yöntemleri önerilir')
    
    def calculate_zpd_level(self, current_level: float, target_level: float, 
                           weeks: int) -> List[float]:
        """Zone of Proximal Development seviyeleri hesapla"""
        if current_level >= target_level:
            return [current_level] * weeks
        
        step = (target_level - current_level) / weeks
        levels = []
        
        for week in range(weeks):
            # Kademeli artış
            level = current_level + (step * (week + 1))
            # Fazla zorluk artışını engelle
            level = min(level, current_level + (0.3 * (week + 1)))
            levels.append(round(level, 2))
        
        return levels
    
    def get_adaptive_resources(self, topic: str, difficulty: float, 
                              learning_style: str) -> List[Dict]:
        """Adaptif kaynaklar öner"""
        resources = []
        
        # Öğrenme stiline göre kaynak türleri
        style_resources = {
            'visual': [
                {'type': 'video', 'source': 'EBA', 'name': f'{topic} Video Dersi'},
                {'type': 'infographic', 'source': 'Custom', 'name': f'{topic} İnfografik'},
                {'type': 'animation', 'source': 'Khan Academy TR', 'name': f'{topic} Animasyon'}
            ],
            'auditory': [
                {'type': 'podcast', 'source': 'EBA', 'name': f'{topic} Sesli Anlatım'},
                {'type': 'discussion', 'source': 'Forum', 'name': f'{topic} Tartışma Grubu'},
                {'type': 'audio_book', 'source': 'MEB', 'name': f'{topic} Sesli Kitap'}
            ],
            'reading': [
                {'type': 'ebook', 'source': 'MEB', 'name': f'{topic} E-Kitap'},
                {'type': 'article', 'source': 'EBA', 'name': f'{topic} Makale'},
                {'type': 'notes', 'source': 'Custom', 'name': f'{topic} Ders Notları'}
            ],
            'kinesthetic': [
                {'type': 'simulation', 'source': 'PhET', 'name': f'{topic} Simülasyon'},
                {'type': 'experiment', 'source': 'Lab', 'name': f'{topic} Deney'},
                {'type': 'practice', 'source': 'Custom', 'name': f'{topic} Uygulama'}
            ]
        }
        
        # Zorluk seviyesine göre kaynak seç
        base_resources = style_resources.get(learning_style, [])
        
        for resource in base_resources:
            resource['difficulty'] = difficulty
            resource['estimated_time'] = 30 if difficulty < 0.5 else 45
            resources.append(resource)
        
        return resources
    
    def generate_learning_path(self, student_profile: Dict, topic: str = None, 
                              weeks: int = 4, subject: str = None, **kwargs) -> Dict:
        """ZPD tabanlı öğrenme yolu oluştur"""
        
        # Validate student profile
        if not student_profile:
            raise ValueError("Student profile is required")
        
        if not student_profile.get('student_id'):
            raise ValueError("Student ID is required")
        
        # topic veya subject parametresini kullan
        if subject and not topic:
            topic = subject
        elif not topic and not subject:
            topic = "Matematik"  # Default topic
        
        # Öğrenci bilgilerini al ve validate et
        current_level = student_profile.get('current_level', 0.3)
        target_level = student_profile.get('target_level', 1.0)
        learning_style = student_profile.get('learning_style', 'visual')
        grade = student_profile.get('grade', 9)
        
        # Validate parameters
        if current_level < 0 or current_level > 1:
            raise ValueError("current_level must be between 0 and 1")
        if target_level < 0 or target_level > 1:
            raise ValueError("target_level must be between 0 and 1")
        if grade < 1 or grade > 12:
            raise ValueError("grade must be between 1 and 12")
        if learning_style not in ['visual', 'auditory', 'kinesthetic', 'reading']:
            raise ValueError("Invalid learning style")
        
        # ZPD seviyelerini hesapla
        zpd_levels = self.calculate_zpd_level(current_level, target_level, weeks)
        
        # Öğrenme yolu oluştur
        path = {
            'student_id': student_profile.get('student_id', 'unknown'),
            'topic': topic,
            'subject': topic,  # Compatibility
            'grade': grade,
            'total_weeks': weeks,
            'created_at': datetime.now().isoformat(),
            'learning_style': learning_style,
            'weekly_plan': []
        }
        
        # Haftalık plan oluştur
        start_date = datetime.now()
        
        for week in range(weeks):
            week_start = start_date + timedelta(weeks=week)
            week_end = week_start + timedelta(days=6)
            
            week_content = {
                'week': week + 1,
                'start_date': week_start.strftime('%Y-%m-%d'),
                'end_date': week_end.strftime('%Y-%m-%d'),
                'topic': f"{topic} - Bölüm {week + 1}",
                'difficulty': zpd_levels[week],
                'objectives': self.get_weekly_objectives(topic, week + 1),
                'resources': self.get_adaptive_resources(
                    topic,
                    zpd_levels[week],
                    learning_style
                ),
                'assessment': {
                    'quiz_count': 3 + week,  # Kademeli artış
                    'practice_count': 5 + (week * 2),
                    'project': week == weeks - 1  # Son hafta proje
                },
                'estimated_hours': 3 + (zpd_levels[week] * 2)
            }
            
            path['weekly_plan'].append(week_content)
        
        # Toplam istatistikler
        path['statistics'] = {
            'total_resources': sum(len(w['resources']) for w in path['weekly_plan']),
            'total_quizzes': sum(w['assessment']['quiz_count'] for w in path['weekly_plan']),
            'total_practices': sum(w['assessment']['practice_count'] for w in path['weekly_plan']),
            'total_hours': sum(w['estimated_hours'] for w in path['weekly_plan']),
            'difficulty_progression': zpd_levels
        }
        
        return path
    
    def get_weekly_objectives(self, topic: str, week: int) -> List[str]:
        """Haftalık öğrenme hedefleri"""
        objectives_template = {
            1: [
                f"{topic} temel kavramlarını tanımlayabilme",
                f"{topic} ile ilgili basit problemleri çözebilme"
            ],
            2: [
                f"{topic} kavramları arasındaki ilişkileri açıklayabilme",
                f"{topic} ile ilgili orta düzey problemleri çözebilme"
            ],
            3: [
                f"{topic} konusunu derinlemesine analiz edebilme",
                f"{topic} ile ilgili karmaşık problemleri çözebilme"
            ],
            4: [
                f"{topic} konusunu diğer konularla ilişkilendirebilme",
                f"{topic} ile ilgili proje geliştirebilme"
            ]
        }
        
        return objectives_template.get(week, [f"{topic} - Hafta {week} hedefleri"])
    
    def evaluate_progress(self, student_id: str, quiz_results: List[float]) -> Dict:
        """Öğrenci ilerlemesini değerlendir"""
        if not quiz_results:
            return {'status': 'no_data', 'message': 'Değerlendirme verisi yok'}
        
        avg_score = sum(quiz_results) / len(quiz_results)
        trend = 'improving' if len(quiz_results) > 1 and quiz_results[-1] > quiz_results[0] else 'stable'
        
        evaluation = {
            'student_id': student_id,
            'average_score': round(avg_score, 2),
            'last_score': quiz_results[-1],
            'trend': trend,
            'quiz_count': len(quiz_results),
            'recommendation': ''
        }
        
        # Öneri oluştur
        if avg_score < 0.5:
            evaluation['recommendation'] = 'Temel konuların tekrarı önerilir'
        elif avg_score < 0.7:
            evaluation['recommendation'] = 'Orta düzey pratik yapılmalı'
        else:
            evaluation['recommendation'] = 'İleri düzey konulara geçilebilir'
        
        return evaluation
    
    def calculate_adaptive_difficulty(self, student_performance: float, current_difficulty: float) -> float:
        """Calculate adaptive difficulty based on student performance"""
        # Low performance: decrease difficulty
        if student_performance < 0.4:
            new_difficulty = max(0.1, current_difficulty - 0.1)
        # High performance: increase difficulty
        elif student_performance > 0.8:
            new_difficulty = min(1.0, current_difficulty + 0.1)
        # Average performance: slight adjustment
        else:
            adjustment = (student_performance - 0.5) * 0.1
            new_difficulty = current_difficulty + adjustment
        
        # Keep within bounds
        new_difficulty = max(0.1, min(1.0, new_difficulty))
        
        return round(new_difficulty, 2)
    
    def personalize_content(self, content: Dict, learning_style: str) -> Dict:
        """Personalize content based on learning style"""
        personalized = content.copy()
        personalized['learning_style'] = learning_style
        
        # Add style-specific enhancements
        if learning_style == 'visual':
            personalized['visual_aids'] = [
                'Diagrams',
                'Infographics',
                'Mind maps',
                'Color-coded notes'
            ]
            personalized['recommended_tools'] = ['Canva', 'MindMeister', 'Lucidchart']
        
        elif learning_style == 'auditory':
            personalized['audio_resources'] = [
                'Podcasts',
                'Audio lectures',
                'Discussion groups',
                'Voice recordings'
            ]
            personalized['recommended_tools'] = ['Audacity', 'Discord', 'Zoom']
        
        elif learning_style == 'kinesthetic':
            personalized['hands_on_activities'] = [
                'Lab experiments',
                'Interactive simulations',
                'Physical models',
                'Role-playing'
            ]
            personalized['recommended_tools'] = ['PhET', 'Labster', 'Minecraft Education']
        
        elif learning_style == 'reading':
            personalized['text_resources'] = [
                'E-books',
                'Research papers',
                'Study guides',
                'Written summaries'
            ]
            personalized['recommended_tools'] = ['Kindle', 'Google Scholar', 'Notion']
        
        # Add adaptive features
        personalized['adaptive_features'] = {
            'pace': 'self-paced' if learning_style in ['reading', 'kinesthetic'] else 'structured',
            'feedback_type': 'visual' if learning_style == 'visual' else 'verbal',
            'assessment_style': 'practical' if learning_style == 'kinesthetic' else 'theoretical'
        }
        
        return personalized
    
    def fetch_external_resources(self, topic: str) -> Optional[List[Dict]]:
        """Fetch external educational resources (mock implementation)"""
        # This is a mock implementation
        # In production, this would call external APIs
        
        resources = [
            {
                'title': f'{topic} - Khan Academy',
                'url': f'https://khanacademy.org/topic/{topic.lower()}',
                'type': 'video',
                'language': 'tr',
                'difficulty': 0.5
            },
            {
                'title': f'{topic} - EBA Ders',
                'url': f'https://eba.gov.tr/{topic.lower()}',
                'type': 'interactive',
                'language': 'tr',
                'difficulty': 0.6
            },
            {
                'title': f'{topic} - Wikipedia',
                'url': f'https://tr.wikipedia.org/wiki/{topic}',
                'type': 'article',
                'language': 'tr',
                'difficulty': 0.4
            }
        ]
        
        return resources


# Test kodu
if __name__ == "__main__":
    agent = LearningPathAgent()
    
    # Örnek öğrenci profili
    student_profile = {
        'student_id': '12345',
        'learning_style': 'visual',
        'current_level': 0.3,
        'target_level': 0.9,
        'grade': 9
    }
    
    # Öğrenme stili testi
    test_responses = [
        'Görsel materyallerle daha iyi öğrenirim',
        'Grafik ve şemalar kullanmayı severim',
        'Video izleyerek konuları anlarım'
    ]
    
    style_result = agent.detect_learning_style(test_responses)
    print("Öğrenme Stili Analizi:")
    print(json.dumps(style_result, indent=2, ensure_ascii=False))
    
    # Öğrenme yolu oluştur
    learning_path = agent.generate_learning_path(student_profile, 'Matematik', weeks=4)
    print("\nÖğrenme Yolu:")
    print(json.dumps(learning_path, indent=2, ensure_ascii=False, default=str))