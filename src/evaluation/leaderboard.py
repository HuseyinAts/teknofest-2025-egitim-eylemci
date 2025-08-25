# -*- coding: utf-8 -*-
"""
TEKNOFEST 2025 Leaderboard Evaluator
Yarışma değerlendirme ve skor hesaplama sistemi
"""

import time
import numpy as np
from typing import Dict, List, Any, Optional
import json
from datetime import datetime
import statistics

class LeaderboardEvaluator:
    """TEKNOFEST yarışma değerlendirme sistemi"""
    
    def __init__(self):
        self.evaluation_history = []
        self.weights = {
            'accuracy': 0.4,
            'speed': 0.3,
            'reliability': 0.3
        }
        
    def evaluate(self, model, test_data):
        """Model değerlendirmesi"""
        metrics = {
            'accuracy': self.calculate_accuracy(model, test_data),
            'speed': self.measure_speed(model, test_data),
            'reliability': self.check_reliability(model, test_data)
        }
        
        # Ağırlıklı skor
        final_score = (
            metrics['accuracy'] * self.weights['accuracy'] +
            metrics['speed'] * self.weights['speed'] +
            metrics['reliability'] * self.weights['reliability']
        )
        
        # Değerlendirme kaydı
        evaluation = {
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics,
            'final_score': final_score,
            'model_info': self.get_model_info(model)
        }
        
        self.evaluation_history.append(evaluation)
        
        return {
            'final_score': round(final_score, 3),
            'metrics': metrics,
            'rank': self.calculate_rank(final_score)
        }
    
    def calculate_accuracy(self, model, test_data) -> float:
        """Doğruluk hesaplama"""
        if not test_data:
            return 0.0
            
        correct = 0
        total = len(test_data)
        
        for item in test_data:
            try:
                # Model tahminini al
                prediction = model.predict(item.get('input', ''))
                
                # Doğru cevapla karşılaştır
                if self.check_answer(prediction, item.get('expected', '')):
                    correct += 1
            except:
                # Hata durumunda yanlış say
                pass
        
        accuracy = (correct / total) if total > 0 else 0
        return round(accuracy, 3)
    
    def measure_speed(self, model, test_data) -> float:
        """Hız ölçümü (normalized 0-1)"""
        if not test_data:
            return 0.0
            
        response_times = []
        
        # Örnek veri üzerinde hız testi
        sample_size = min(10, len(test_data))
        
        for i in range(sample_size):
            item = test_data[i]
            
            start_time = time.time()
            try:
                _ = model.predict(item.get('input', ''))
                elapsed = time.time() - start_time
                response_times.append(elapsed)
            except:
                response_times.append(10.0)  # Timeout değeri
        
        if not response_times:
            return 0.0
        
        # Ortalama süre (ms)
        avg_time = statistics.mean(response_times) * 1000
        
        # Normalize (0-100ms arası ideal)
        if avg_time <= 100:
            speed_score = 1.0
        elif avg_time <= 500:
            speed_score = 1.0 - ((avg_time - 100) / 400) * 0.5
        elif avg_time <= 1000:
            speed_score = 0.5 - ((avg_time - 500) / 500) * 0.3
        else:
            speed_score = max(0.2 - ((avg_time - 1000) / 1000) * 0.2, 0)
        
        return round(speed_score, 3)
    
    def check_reliability(self, model, test_data) -> float:
        """Güvenilirlik kontrolü"""
        reliability_checks = {
            'consistency': self.check_consistency(model, test_data),
            'error_handling': self.check_error_handling(model),
            'edge_cases': self.check_edge_cases(model),
            'response_quality': self.check_response_quality(model, test_data)
        }
        
        # Ağırlıklı güvenilirlik skoru
        reliability = (
            reliability_checks['consistency'] * 0.3 +
            reliability_checks['error_handling'] * 0.2 +
            reliability_checks['edge_cases'] * 0.2 +
            reliability_checks['response_quality'] * 0.3
        )
        
        return round(reliability, 3)
    
    def check_consistency(self, model, test_data) -> float:
        """Tutarlılık kontrolü"""
        if not test_data:
            return 0.0
            
        # Aynı soruya verilen cevapların tutarlılığı
        consistency_scores = []
        
        sample_size = min(5, len(test_data))
        for i in range(sample_size):
            item = test_data[i]
            responses = []
            
            # Aynı soruyu 3 kez sor
            for _ in range(3):
                try:
                    response = model.predict(item.get('input', ''))
                    responses.append(response)
                except:
                    responses.append(None)
            
            # Tutarlılık kontrolü
            if all(r == responses[0] for r in responses if r is not None):
                consistency_scores.append(1.0)
            elif len(set(r for r in responses if r is not None)) <= 2:
                consistency_scores.append(0.5)
            else:
                consistency_scores.append(0.0)
        
        return statistics.mean(consistency_scores) if consistency_scores else 0.0
    
    def check_error_handling(self, model) -> float:
        """Hata yönetimi kontrolü"""
        error_cases = [
            None,  # Null input
            "",    # Empty string
            " " * 100,  # Only spaces
            "a" * 10000,  # Very long input
            {"invalid": "input"},  # Wrong type
        ]
        
        handled_correctly = 0
        
        for case in error_cases:
            try:
                response = model.predict(case)
                if response is not None:
                    handled_correctly += 1
            except:
                # Graceful error handling
                handled_correctly += 0.5
        
        return handled_correctly / len(error_cases)
    
    def check_edge_cases(self, model) -> float:
        """Uç durum kontrolü"""
        edge_cases = [
            {"input": "What is 0/0?", "type": "math_undefined"},
            {"input": "Solve x^2 = -1 in real numbers", "type": "no_solution"},
            {"input": "1+1" * 100, "type": "repetitive"},
            {"input": "😀🎉🔥", "type": "emoji_only"},
            {"input": "Türkçe karakter: ğüşıöç", "type": "turkish_chars"}
        ]
        
        handled = 0
        for case in edge_cases:
            try:
                response = model.predict(case['input'])
                if response and len(str(response)) > 0:
                    handled += 1
            except:
                pass
        
        return handled / len(edge_cases)
    
    def check_response_quality(self, model, test_data) -> float:
        """Cevap kalitesi kontrolü"""
        if not test_data:
            return 0.5
            
        quality_scores = []
        sample_size = min(10, len(test_data))
        
        for i in range(sample_size):
            item = test_data[i]
            try:
                response = model.predict(item.get('input', ''))
                
                # Kalite kriterleri
                if response:
                    score = 0
                    
                    # Cevap var mı?
                    if len(str(response)) > 0:
                        score += 0.25
                    
                    # Makul uzunlukta mı?
                    if 10 <= len(str(response)) <= 500:
                        score += 0.25
                    
                    # Yapılandırılmış mı?
                    if isinstance(response, (dict, list)) or '\n' in str(response):
                        score += 0.25
                    
                    # Beklenen formata uygun mu?
                    if item.get('format') and self.check_format(response, item['format']):
                        score += 0.25
                    
                    quality_scores.append(score)
                else:
                    quality_scores.append(0)
            except:
                quality_scores.append(0)
        
        return statistics.mean(quality_scores) if quality_scores else 0.0
    
    def check_answer(self, prediction, expected) -> bool:
        """Cevap doğruluğu kontrolü"""
        if prediction is None or expected is None:
            return False
            
        # String normalizasyonu
        pred_str = str(prediction).lower().strip()
        exp_str = str(expected).lower().strip()
        
        # Tam eşleşme
        if pred_str == exp_str:
            return True
        
        # Sayısal değerler için yaklaşık eşleşme
        try:
            pred_num = float(pred_str)
            exp_num = float(exp_str)
            return abs(pred_num - exp_num) < 0.01
        except:
            pass
        
        # Kısmi eşleşme (cevap içeriyor mu?)
        if exp_str in pred_str or pred_str in exp_str:
            return True
        
        return False
    
    def check_format(self, response, expected_format: str) -> bool:
        """Format kontrolü"""
        if expected_format == 'number':
            try:
                float(response)
                return True
            except:
                return False
        elif expected_format == 'list':
            return isinstance(response, list)
        elif expected_format == 'dict':
            return isinstance(response, dict)
        elif expected_format == 'string':
            return isinstance(response, str)
        
        return True
    
    def get_model_info(self, model) -> Dict:
        """Model bilgileri"""
        info = {
            'name': getattr(model, 'name', 'Unknown'),
            'version': getattr(model, 'version', '1.0'),
            'type': model.__class__.__name__
        }
        
        # Model parametreleri
        if hasattr(model, 'get_params'):
            info['params'] = model.get_params()
        
        return info
    
    def calculate_rank(self, score: float) -> Dict:
        """Sıralama hesaplama"""
        if score >= 0.9:
            return {'rank': 'A+', 'percentile': 95, 'medal': 'gold'}
        elif score >= 0.8:
            return {'rank': 'A', 'percentile': 85, 'medal': 'silver'}
        elif score >= 0.7:
            return {'rank': 'B+', 'percentile': 75, 'medal': 'bronze'}
        elif score >= 0.6:
            return {'rank': 'B', 'percentile': 60, 'medal': None}
        elif score >= 0.5:
            return {'rank': 'C', 'percentile': 40, 'medal': None}
        else:
            return {'rank': 'D', 'percentile': 20, 'medal': None}
    
    def generate_report(self) -> Dict:
        """Değerlendirme raporu"""
        if not self.evaluation_history:
            return {'message': 'No evaluations yet'}
        
        latest = self.evaluation_history[-1]
        
        report = {
            'timestamp': latest['timestamp'],
            'final_score': latest['final_score'],
            'metrics': latest['metrics'],
            'model_info': latest['model_info'],
            'rank': self.calculate_rank(latest['final_score']),
            'history_count': len(self.evaluation_history)
        }
        
        # İstatistikler
        if len(self.evaluation_history) > 1:
            scores = [e['final_score'] for e in self.evaluation_history]
            report['statistics'] = {
                'mean_score': round(statistics.mean(scores), 3),
                'max_score': round(max(scores), 3),
                'min_score': round(min(scores), 3),
                'std_dev': round(statistics.stdev(scores), 3) if len(scores) > 1 else 0
            }
        
        return report
    
    def compare_models(self, models: List, test_data) -> List[Dict]:
        """Model karşılaştırması"""
        results = []
        
        for model in models:
            evaluation = self.evaluate(model, test_data)
            results.append({
                'model': self.get_model_info(model),
                'score': evaluation['final_score'],
                'metrics': evaluation['metrics'],
                'rank': evaluation['rank']
            })
        
        # Skorlara göre sırala
        results.sort(key=lambda x: x['score'], reverse=True)
        
        # Sıralama ekle
        for i, result in enumerate(results):
            result['position'] = i + 1
        
        return results


# Mock model for testing
class MockModel:
    """Test için örnek model"""
    
    def __init__(self, name="TestModel", accuracy=0.8):
        self.name = name
        self.version = "1.0"
        self.accuracy = accuracy
        
    def predict(self, input_text):
        """Basit tahmin fonksiyonu"""
        if input_text is None or input_text == "":
            raise ValueError("Invalid input")
        
        # Simüle edilmiş tahmin
        import random
        if random.random() < self.accuracy:
            return "correct_answer"
        return "wrong_answer"
    
    def get_params(self):
        return {'accuracy': self.accuracy}


if __name__ == "__main__":
    # Test
    evaluator = LeaderboardEvaluator()
    
    # Test verisi
    test_data = [
        {'input': 'What is 2+2?', 'expected': '4'},
        {'input': 'Capital of Turkey?', 'expected': 'Ankara'},
        {'input': 'Solve x+5=10', 'expected': '5'},
        {'input': 'What is Python?', 'expected': 'programming language'},
        {'input': '10/2=?', 'expected': '5'}
    ]
    
    # Model oluştur ve değerlendir
    model = MockModel(name="TEKNOFEST_AI", accuracy=0.85)
    
    print("=== TEKNOFEST 2025 Leaderboard Evaluation ===\n")
    
    result = evaluator.evaluate(model, test_data)
    print(f"Final Score: {result['final_score']}")
    print(f"Metrics: {result['metrics']}")
    print(f"Rank: {result['rank']}")
    
    # Rapor
    print("\n=== Evaluation Report ===")
    report = evaluator.generate_report()
    print(json.dumps(report, indent=2))