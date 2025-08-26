"""
🧠 GELİŞMİŞ CURRICULUM LEARNING SİSTEMİ
Türkçe dilinin karmaşıklık seviyesine göre kademeli öğrenme

ÖNERİ: Basit cümlelerden karmaşık metinlere doğru progressive training
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging
import re
from collections import Counter
from datasets import Dataset

logger = logging.getLogger(__name__)

@dataclass
class CurriculumStage:
    """Curriculum learning stage tanımı"""
    name: str
    description: str
    min_complexity: float
    max_complexity: float
    target_samples: int
    learning_rate_multiplier: float
    epochs: int

class TurkishComplexityAnalyzer:
    """Türkçe metin karmaşıklık analizi"""
    
    def __init__(self):
        # Morphological complexity indicators
        self.simple_suffixes = ['de', 'da', 'den', 'dan', 'ye', 'ya', 'ler', 'lar']
        self.complex_suffixes = ['leriyle', 'larıyla', 'lerinden', 'larından', 'lerine', 'larine']
        
        # Vocabulary complexity levels
        self.basic_words = set([
            'ben', 'sen', 'o', 'biz', 'siz', 'onlar', 'bu', 'şu', 'o',
            'var', 'yok', 'geldi', 'gitti', 'yaptı', 'dedi', 'güzel', 'iyi', 'kötü'
        ])
        
        self.advanced_words = set([
            'karmaşık', 'gelişmek', 'uygulama', 'araştırma', 'geliştirmek',
            'değerlendirme', 'analiz', 'sentez', 'yorumlama', 'değişim'
        ])
        
        # Sentence structure complexity
        self.complex_patterns = [
            r'.*ki\s+.*',  # Relative clauses with 'ki'
            r'.*\w+ken\s+.*',  # While doing something
            r'.*rken\s+.*',  # While doing (different form)
            r'.*dığı\s+.*',  # That which/what form
            r'.*eceği\s+.*',  # Future participle
        ]
    
    def calculate_morphological_complexity(self, text: str) -> float:
        """Morphological complexity skoru"""
        
        words = text.lower().split()
        if not words:
            return 0.0
        
        complexity_score = 0
        
        for word in words:
            # Simple suffix check
            simple_count = sum(1 for suffix in self.simple_suffixes if word.endswith(suffix))
            complex_count = sum(1 for suffix in self.complex_suffixes if word.endswith(suffix))
            
            # Word length as complexity indicator
            length_complexity = min(len(word) / 15, 1.0)  # Normalize to max 15 chars
            
            # Agglutination complexity (multiple suffixes)
            suffix_count = sum(1 for suffix in self.simple_suffixes + self.complex_suffixes 
                             if suffix in word)
            agglutination_complexity = min(suffix_count / 3, 1.0)
            
            word_complexity = (length_complexity * 0.4 + 
                             agglutination_complexity * 0.4 + 
                             complex_count * 0.2)
            
            complexity_score += word_complexity
        
        return complexity_score / len(words)
    
    def calculate_lexical_complexity(self, text: str) -> float:
        """Lexical complexity skoru"""
        
        words = set(text.lower().split())
        if not words:
            return 0.0
        
        basic_count = len(words & self.basic_words)
        advanced_count = len(words & self.advanced_words)
        total_words = len(words)
        
        # Higher score for more advanced words, lower for basic words
        basic_ratio = basic_count / total_words
        advanced_ratio = advanced_count / total_words
        
        return (1 - basic_ratio) * 0.6 + advanced_ratio * 0.4
    
    def calculate_syntactic_complexity(self, text: str) -> float:
        """Syntactic complexity skoru"""
        
        # Sentence count
        sentences = re.split(r'[.!?]+', text)
        valid_sentences = [s.strip() for s in sentences if len(s.strip()) > 5]
        
        if not valid_sentences:
            return 0.0
        
        complexity_score = 0
        
        for sentence in valid_sentences:
            sentence_complexity = 0
            
            # Complex pattern detection
            for pattern in self.complex_patterns:
                if re.match(pattern, sentence.lower()):
                    sentence_complexity += 0.3
            
            # Sentence length complexity
            words_in_sentence = len(sentence.split())
            length_complexity = min(words_in_sentence / 20, 1.0)  # Max 20 words
            
            # Clause complexity (comma count as proxy)
            comma_count = sentence.count(',')
            clause_complexity = min(comma_count / 3, 1.0)
            
            sentence_complexity += length_complexity * 0.4 + clause_complexity * 0.3
            complexity_score += sentence_complexity
        
        return complexity_score / len(valid_sentences)
    
    def calculate_overall_complexity(self, text: str) -> Dict[str, float]:
        """Genel complexity analizi"""
        
        morphological = self.calculate_morphological_complexity(text)
        lexical = self.calculate_lexical_complexity(text)
        syntactic = self.calculate_syntactic_complexity(text)
        
        # Weighted overall score
        overall = (morphological * 0.4 + lexical * 0.3 + syntactic * 0.3)
        
        return {
            'overall': overall,
            'morphological': morphological,
            'lexical': lexical,
            'syntactic': syntactic
        }


class TurkishCurriculumLearning:
    """Türkçe için curriculum learning orchestrator"""
    
    def __init__(self):
        self.complexity_analyzer = TurkishComplexityAnalyzer()
        
        # Curriculum stages definition
        self.stages = [
            CurriculumStage(
                name="stage1_basic",
                description="Temel Türkçe yapılar, basit cümleler",
                min_complexity=0.0,
                max_complexity=0.3,
                target_samples=15000,
                learning_rate_multiplier=1.2,  # Higher LR for basics
                epochs=3
            ),
            CurriculumStage(
                name="stage2_intermediate",
                description="Orta seviye morphology, compound sentences",
                min_complexity=0.3,
                max_complexity=0.6,
                target_samples=20000,
                learning_rate_multiplier=1.0,  # Standard LR
                epochs=4
            ),
            CurriculumStage(
                name="stage3_complex",
                description="Karmaşık agglutination, academic texts",
                min_complexity=0.6,
                max_complexity=0.8,
                target_samples=12000,
                learning_rate_multiplier=0.8,  # Lower LR for complex
                epochs=3
            ),
            CurriculumStage(
                name="stage4_advanced",
                description="En karmaşık yapılar, formal/literary texts",
                min_complexity=0.8,
                max_complexity=1.0,
                target_samples=8000,
                learning_rate_multiplier=0.6,  # Lowest LR for most complex
                epochs=2
            )
        ]
    
    def analyze_dataset_complexity(self, dataset: Dataset) -> Dict[str, any]:
        """Dataset'in complexity dağılımını analiz et"""
        
        logger.info("📊 Dataset complexity analysis başlıyor...")
        
        complexity_scores = []
        complexity_details = []
        
        for item in dataset:
            text = item.get('text', '')
            if len(text) > 30:  # Minimum length check
                complexity = self.complexity_analyzer.calculate_overall_complexity(text)
                complexity_scores.append(complexity['overall'])
                complexity_details.append({
                    'text': text[:100] + '...' if len(text) > 100 else text,
                    'complexity': complexity
                })
        
        # Statistics
        complexity_array = np.array(complexity_scores)
        
        stats = {
            'total_samples': len(complexity_scores),
            'mean_complexity': complexity_array.mean(),
            'std_complexity': complexity_array.std(),
            'complexity_distribution': {
                'basic (0.0-0.3)': np.sum((complexity_array >= 0.0) & (complexity_array < 0.3)),
                'intermediate (0.3-0.6)': np.sum((complexity_array >= 0.3) & (complexity_array < 0.6)),
                'complex (0.6-0.8)': np.sum((complexity_array >= 0.6) & (complexity_array < 0.8)),
                'advanced (0.8-1.0)': np.sum((complexity_array >= 0.8) & (complexity_array <= 1.0))
            },
            'complexity_details': complexity_details
        }
        
        logger.info(f"✅ Complexity analysis tamamlandı:")
        logger.info(f"  📊 Mean complexity: {stats['mean_complexity']:.3f}")
        logger.info(f"  📊 Distribution: {stats['complexity_distribution']}")
        
        return stats
    
    def create_curriculum_datasets(self, dataset: Dataset) -> Dict[str, Dataset]:
        """Curriculum stages için ayrı dataset'ler oluştur"""
        
        logger.info("📚 Curriculum datasets oluşturuluyor...")
        
        # Analyze all samples
        samples_with_complexity = []
        
        for item in dataset:
            text = item.get('text', '')
            if len(text) >= 30:  # Memory'den: minimum 30 karakter
                complexity = self.complexity_analyzer.calculate_overall_complexity(text)
                samples_with_complexity.append({
                    **item,
                    'complexity_score': complexity['overall'],
                    'complexity_details': complexity
                })
        
        # Sort by complexity
        samples_with_complexity.sort(key=lambda x: x['complexity_score'])
        
        # Create stage datasets
        stage_datasets = {}
        
        for stage in self.stages:
            stage_samples = []
            
            for sample in samples_with_complexity:
                complexity = sample['complexity_score']
                
                if stage.min_complexity <= complexity < stage.max_complexity:
                    stage_samples.append({
                        'text': sample['text'],
                        'source': sample.get('source', 'unknown'),
                        'complexity_score': complexity
                    })
                
                if len(stage_samples) >= stage.target_samples:
                    break
            
            if stage_samples:
                stage_datasets[stage.name] = Dataset.from_list(stage_samples)
                logger.info(f"✅ {stage.name}: {len(stage_samples)} samples (complexity: {stage.min_complexity:.1f}-{stage.max_complexity:.1f})")
            else:
                logger.warning(f"⚠️ {stage.name}: No samples found in complexity range")
        
        return stage_datasets
    
    def get_curriculum_training_schedule(self) -> List[Dict]:
        """Curriculum training schedule döndür"""
        
        schedule = []
        
        for stage in self.stages:
            schedule.append({
                'stage_name': stage.name,
                'description': stage.description,
                'complexity_range': f"{stage.min_complexity:.1f}-{stage.max_complexity:.1f}",
                'target_samples': stage.target_samples,
                'epochs': stage.epochs,
                'learning_rate_multiplier': stage.learning_rate_multiplier,
                'training_order': len(schedule) + 1
            })
        
        return schedule
    
    def adapt_training_args_for_stage(self, base_args, stage: CurriculumStage):
        """Stage'e göre training arguments'ları adapte et"""
        
        # Learning rate adjustment
        base_args.learning_rate = base_args.learning_rate * stage.learning_rate_multiplier
        base_args.num_train_epochs = stage.epochs
        
        # Stage-specific optimizations
        if stage.name == "stage1_basic":
            # Basic stage - more aggressive learning
            base_args.warmup_ratio = 0.2  # Longer warmup
            base_args.weight_decay = 0.005  # Less regularization
        
        elif stage.name == "stage4_advanced":
            # Advanced stage - more conservative
            base_args.warmup_ratio = 0.05  # Shorter warmup
            base_args.weight_decay = 0.02  # More regularization
        
        logger.info(f"🎯 {stage.name} adapted: LR={base_args.learning_rate:.2e}, Epochs={base_args.num_train_epochs}")
        
        return base_args


def integrate_curriculum_learning(original_dataset: Dataset) -> Tuple[Dict[str, Dataset], List[Dict]]:
    """Curriculum learning'i mevcut pipeline'a entegre et"""
    
    logger.info("🧠 Curriculum Learning entegrasyonu başlıyor...")
    
    # Create curriculum system
    curriculum = TurkishCurriculumLearning()
    
    # Analyze dataset complexity
    complexity_stats = curriculum.analyze_dataset_complexity(original_dataset)
    
    # Create stage datasets
    stage_datasets = curriculum.create_curriculum_datasets(original_dataset)
    
    # Get training schedule
    training_schedule = curriculum.get_curriculum_training_schedule()
    
    logger.info(f"✅ Curriculum Learning hazır:")
    logger.info(f"  📚 {len(stage_datasets)} stage dataset created")
    logger.info(f"  📋 {len(training_schedule)} training stages planned")
    
    return stage_datasets, training_schedule


# Test function
def test_curriculum_learning():
    """Curriculum learning test"""
    
    print("🧪 Curriculum Learning test ediliyor...")
    
    # Sample Turkish texts with different complexity levels
    sample_texts = [
        # Basic (0.0-0.3)
        {"text": "Bu güzel bir gün. Ben evdeyim. Sen neredesin?"},
        {"text": "Okula gittim. Arkadaşlarımla oynadım. Eve döndüm."},
        
        # Intermediate (0.3-0.6)
        {"text": "Türkiye'nin başkenti Ankara'dır. İstanbul en kalabalık şehridir."},
        {"text": "Eğitim sistemimizin geliştirilmesi için çalışmalar devam ediyor."},
        
        # Complex (0.6-0.8)
        {"text": "Teknolojik gelişmelerin eğitim alanındaki uygulamalarının değerlendirilmesi gerekmektedir."},
        {"text": "Araştırmacıların yaptığı çalışmalar, öğrencilerin başarısını artırdığını göstermektedir."},
        
        # Advanced (0.8-1.0)
        {"text": "Epistemolojik paradigmaların eğitimsel uygulamalardaki manifestasyonlarının analizi, pedagojik yaklaşımların teorik temellerinin sorgulanmasını gerektirmektedir."},
        {"text": "Multidisipliner araştırma metodolojilerinin entegrasyonu, bilimsel bilginin konstruktivistik yaklaşımlarla sentezlenmesinde kritik öneme sahiptir."}
    ]
    
    # Create dataset
    dataset = Dataset.from_list(sample_texts)
    
    # Test curriculum learning
    stage_datasets, schedule = integrate_curriculum_learning(dataset)
    
    print(f"✅ Stage datasets: {len(stage_datasets)}")
    print(f"✅ Training schedule: {len(schedule)} stages")
    
    # Show complexity analysis
    analyzer = TurkishComplexityAnalyzer()
    
    print("\n📊 Complexity Analysis Examples:")
    for i, text_item in enumerate(sample_texts[:4]):
        text = text_item['text']
        complexity = analyzer.calculate_overall_complexity(text)
        print(f"  {i+1}. Overall: {complexity['overall']:.3f} | Morph: {complexity['morphological']:.3f} | Lex: {complexity['lexical']:.3f}")
    
    print("🎉 Curriculum learning test completed!")


if __name__ == "__main__":
    test_curriculum_learning()