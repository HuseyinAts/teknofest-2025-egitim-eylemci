"""
Ki_iselle_tirilmi_ Örenme Yolu Olu_turan Agent
TEKNOFEST 2025 - Eitim Teknolojileri Eylemcisi
"""

import json
from typing import Dict, List
from dataclasses import dataclass

@dataclass
class StudentProfile:
    """Örenci profil veri yap1s1"""
    student_id: str
    learning_style: str  # VARK: visual, auditory, reading, kinesthetic
    grade_level: int
    current_knowledge: Dict[str, float]
    
class LearningPathAgent:
    """Ki_iselle_tirilmi_ örenme yolu olu_turan agent"""
    
    def __init__(self):
        self.curriculum_graph = self._build_curriculum_graph()
        self.learning_styles = ['visual', 'auditory', 'reading', 'kinesthetic']
        
    def _build_curriculum_graph(self) -> Dict:
        """MEB müfredat1na göre konu grafii"""
        return {
            'matematik_9': {
                'konular': ['Kümeler', 'Say1lar', 'Denklemler'],
                'prerequisites': {'Say1lar': ['Kümeler'], 'Denklemler': ['Say1lar']},
                'zorluk': {'Kümeler': 1, 'Say1lar': 2, 'Denklemler': 3}
            }
        }
    
    def analyze_learning_style(self, responses: List[str]) -> str:
        """VARK modeline göre örenme stili analizi"""
        scores = {style: 0 for style in self.learning_styles}
        
        for response in responses:
            if 'görsel' in response or '_ema' in response:
                scores['visual'] += 1
            elif 'dinle' in response or 'anlat' in response:
                scores['auditory'] += 1
            elif 'oku' in response or 'yaz' in response:
                scores['reading'] += 1
            elif 'uygula' in response or 'hareket' in response:
                scores['kinesthetic'] += 1
                
        return max(scores, key=scores.get)
    
    def create_personalized_path(
        self, 
        profile: StudentProfile,
        target_topic: str,
        duration_weeks: int = 4
    ) -> List[Dict]:
        """Zone of Proximal Development ile örenme yolu olu_tur"""
        
        path = []
        current_level = profile.current_knowledge.get(target_topic, 0)
        target_level = 1.0
        
        # ZPD hesaplama
        step_size = (target_level - current_level) / duration_weeks
        
        for week in range(1, duration_weeks + 1):
            difficulty = current_level + (step_size * week)
            
            week_plan = {
                'hafta': week,
                'konu': f"{target_topic} - Bölüm {week}",
                'zorluk': round(difficulty, 2),
                'örenme_stili': profile.learning_style,
                'kaynaklar': self._get_resources(
                    target_topic, 
                    profile.learning_style, 
                    difficulty
                ),
                'deerlendirme': {
                    'quiz_say1s1': 5,
                    'pratik_say1s1': 10,
                    'proje': week == duration_weeks
                }
            }
            path.append(week_plan)
            
        return path
    
    def _get_resources(
        self, 
        topic: str, 
        style: str, 
        difficulty: float
    ) -> List[str]:
        """Örenme stiline uygun kaynaklar"""
        resources = {
            'visual': ['Video ders', '0nfografik', 'Animasyon'],
            'auditory': ['Podcast', 'Sesli kitap', 'Tart1_ma forumu'],
            'reading': ['E-kitap', 'Makale', 'Ders notu'],
            'kinesthetic': ['Simülasyon', 'Deney', 'Uygulama']
        }
        return resources.get(style, [])

# Test kodu
if __name__ == "__main__":
    agent = LearningPathAgent()
    
    # Örnek örenci profili
    profile = StudentProfile(
        student_id="12345",
        learning_style="visual",
        grade_level=9,
        current_knowledge={'Matematik': 0.3}
    )
    
    # Örenme yolu olu_tur
    path = agent.create_personalized_path(profile, 'Matematik', 4)
    print(json.dumps(path, indent=2, ensure_ascii=False))