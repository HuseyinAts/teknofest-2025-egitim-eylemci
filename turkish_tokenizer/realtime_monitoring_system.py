"""
⚡ REAL-TIME PERFORMANCE MONİTORİNG SİSTEMİ
Training sırasında anlık performans takibi ve otomatik optimizasyon

ÖNERİ: Training devam ederken Turkish performance metrics'leri izle
"""

import torch
import numpy as np
import json
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from collections import deque, defaultdict
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import psutil
import threading

logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    """Performance metrics tanımı"""
    timestamp: float
    step: int
    loss: float
    learning_rate: float
    
    # Turkish-specific metrics
    turkish_char_accuracy: Optional[float] = None
    vowel_harmony_score: Optional[float] = None
    morphology_preservation: Optional[float] = None
    
    # System metrics
    gpu_memory_used: Optional[float] = None
    gpu_utilization: Optional[float] = None
    cpu_percent: Optional[float] = None
    
    # Training efficiency
    tokens_per_second: Optional[float] = None
    samples_per_second: Optional[float] = None
    
    # Quality indicators
    gradient_norm: Optional[float] = None
    model_coherence: Optional[float] = None

class TurkishPerformanceEvaluator:
    """Türkçe-spesifik performance evaluation"""
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.turkish_chars = set('çğıöşüÇĞİÖŞÜ')
        self.vowel_groups = {
            'front': set('eiöüEİÖÜ'),
            'back': set('aıouAIOU')
        }
    
    def evaluate_turkish_generation(self, model, device: str = "cuda") -> Dict[str, float]:
        """Model'in Türkçe generation kalitesini değerlendir"""
        
        turkish_prompts = [
            "Türkiye'nin başkenti",
            "Eğitim sistemimizde",
            "Bu güzel bir",
            "Teknoloji alanında"
        ]
        
        model.eval()
        total_scores = defaultdict(list)
        
        with torch.no_grad():
            for prompt in turkish_prompts:
                try:
                    # Generate text
                    inputs = self.tokenizer(prompt, return_tensors="pt").to(device)
                    outputs = model.generate(
                        **inputs,
                        max_length=50,
                        num_return_sequences=1,
                        temperature=0.7,
                        do_sample=True,
                        pad_token_id=self.tokenizer.eos_token_id
                    )
                    
                    # Decode generated text
                    generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                    generated_part = generated[len(prompt):].strip()
                    
                    if len(generated_part) > 10:  # Valid generation
                        # Evaluate Turkish quality
                        scores = self._evaluate_text_quality(generated_part)
                        for key, value in scores.items():
                            total_scores[key].append(value)
                
                except Exception as e:
                    logger.warning(f"Generation evaluation error: {e}")
        
        # Average scores
        averaged_scores = {}
        for key, values in total_scores.items():
            averaged_scores[key] = np.mean(values) if values else 0.0
        
        return averaged_scores
    
    def _evaluate_text_quality(self, text: str) -> Dict[str, float]:
        """Text quality evaluation"""
        
        # Turkish character usage
        turkish_char_ratio = sum(1 for c in text if c in self.turkish_chars) / max(len(text), 1)
        
        # Vowel harmony check
        harmony_score = self._check_vowel_harmony(text)
        
        # Morphological plausibility
        morphology_score = self._check_morphology(text)
        
        # Coherence (simple heuristic)
        coherence_score = self._check_coherence(text)
        
        return {
            'turkish_char_accuracy': turkish_char_ratio,
            'vowel_harmony_score': harmony_score,
            'morphology_preservation': morphology_score,
            'model_coherence': coherence_score
        }
    
    def _check_vowel_harmony(self, text: str) -> float:
        """Vowel harmony compliance check"""
        
        words = text.split()
        harmony_violations = 0
        total_words = 0
        
        for word in words:
            if len(word) > 3:  # Check only meaningful words
                vowels = [c for c in word.lower() if c in 'aeiıoöuü']
                
                if len(vowels) > 1:
                    total_words += 1
                    
                    front_count = sum(1 for v in vowels if v in self.vowel_groups['front'])
                    back_count = sum(1 for v in vowels if v in self.vowel_groups['back'])
                    
                    # Violation if mixed front and back vowels
                    if front_count > 0 and back_count > 0:
                        harmony_violations += 1
        
        return 1.0 - (harmony_violations / max(total_words, 1))
    
    def _check_morphology(self, text: str) -> float:
        """Basic morphological plausibility check"""
        
        turkish_suffixes = ['ler', 'lar', 'de', 'da', 'den', 'dan', 'ye', 'ya', 'nin', 'nın']
        
        words = text.split()
        morphology_score = 0
        
        for word in words:
            if len(word) > 4:
                # Check for valid Turkish suffix patterns
                for suffix in turkish_suffixes:
                    if word.lower().endswith(suffix):
                        morphology_score += 1
                        break
        
        return morphology_score / max(len(words), 1)
    
    def _check_coherence(self, text: str) -> float:
        """Basic coherence check"""
        
        # Simple heuristics for coherence
        words = text.split()
        
        if len(words) < 3:
            return 0.5
        
        # Check for reasonable word length distribution
        avg_word_length = np.mean([len(w) for w in words])
        length_score = 1.0 if 3 <= avg_word_length <= 8 else 0.5
        
        # Check for reasonable punctuation
        punct_ratio = sum(1 for c in text if c in '.,!?;') / max(len(text), 1)
        punct_score = 1.0 if 0.01 <= punct_ratio <= 0.1 else 0.5
        
        return (length_score + punct_score) / 2


class RealTimeMonitor:
    """Real-time training monitor"""
    
    def __init__(self, tokenizer, save_interval: int = 100):
        self.tokenizer = tokenizer
        self.save_interval = save_interval
        
        # Performance tracking
        self.metrics_history = deque(maxlen=1000)  # Last 1000 steps
        self.performance_evaluator = TurkishPerformanceEvaluator(tokenizer)
        
        # System monitoring
        self.system_monitor = SystemMonitor()
        self.start_time = time.time()
        
        # Auto-optimization
        self.auto_optimizer = AutoTrainingOptimizer()
        
        # Threading for background monitoring
        self.monitoring_active = False
        self.monitor_thread = None
    
    def start_monitoring(self):
        """Background monitoring başlat"""
        
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._background_monitor, daemon=True)
        self.monitor_thread.start()
        
        logger.info("✅ Real-time monitoring started")
    
    def stop_monitoring(self):
        """Monitoring'i durdur"""
        
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        
        logger.info("🔴 Real-time monitoring stopped")
    
    def record_step_metrics(self, model, step: int, loss: float, 
                          learning_rate: float, **kwargs):
        """Her step'te metrics kaydet"""
        
        # System metrics
        system_info = self.system_monitor.get_current_metrics()
        
        # Turkish evaluation (periodic - expensive)
        turkish_metrics = {}
        if step % 50 == 0:  # Every 50 steps
            try:
                turkish_metrics = self.performance_evaluator.evaluate_turkish_generation(model)
            except Exception as e:
                logger.warning(f"Turkish evaluation failed: {e}")
        
        # Create metrics record
        metrics = PerformanceMetrics(
            timestamp=time.time(),
            step=step,
            loss=loss,
            learning_rate=learning_rate,
            
            # Turkish metrics (if available)
            turkish_char_accuracy=turkish_metrics.get('turkish_char_accuracy'),
            vowel_harmony_score=turkish_metrics.get('vowel_harmony_score'),
            morphology_preservation=turkish_metrics.get('morphology_preservation'),
            model_coherence=turkish_metrics.get('model_coherence'),
            
            # System metrics
            gpu_memory_used=system_info.get('gpu_memory_used'),
            gpu_utilization=system_info.get('gpu_utilization'),
            cpu_percent=system_info.get('cpu_percent'),
            
            # Additional metrics from kwargs
            **kwargs
        )
        
        self.metrics_history.append(metrics)
        
        # Auto-optimization check
        if step % 100 == 0:
            optimization_suggestions = self.auto_optimizer.analyze_performance(
                list(self.metrics_history)[-100:]  # Last 100 steps
            )
            
            if optimization_suggestions:
                logger.info(f"🔧 Auto-optimization suggestions: {optimization_suggestions}")
    
    def _background_monitor(self):
        """Background monitoring thread"""
        
        while self.monitoring_active:
            try:
                # Periodic system health check
                system_info = self.system_monitor.get_current_metrics()
                
                # Memory warning
                if system_info.get('gpu_memory_used', 0) > 0.9:
                    logger.warning("🚨 GPU memory usage >90%! Consider reducing batch size.")
                
                # Performance degradation warning
                if len(self.metrics_history) > 10:
                    recent_losses = [m.loss for m in list(self.metrics_history)[-10:]]
                    if len(set([f"{l:.3f}" for l in recent_losses])) == 1:  # No change
                        logger.warning("⚠️ Loss stagnation detected in last 10 steps")
                
                time.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Background monitoring error: {e}")
                time.sleep(60)  # Wait longer on error
    
    def generate_performance_report(self) -> Dict[str, Any]:
        """Comprehensive performance report"""
        
        if not self.metrics_history:
            return {'error': 'No metrics available'}
        
        metrics_list = list(self.metrics_history)
        
        # Loss analysis
        losses = [m.loss for m in metrics_list]
        loss_trend = np.polyfit(range(len(losses)), losses, 1)[0]  # Slope
        
        # Turkish metrics analysis
        turkish_scores = [m.turkish_char_accuracy for m in metrics_list 
                         if m.turkish_char_accuracy is not None]
        
        harmony_scores = [m.vowel_harmony_score for m in metrics_list 
                         if m.vowel_harmony_score is not None]
        
        # Training efficiency
        total_time = time.time() - self.start_time
        total_steps = len(metrics_list)
        steps_per_second = total_steps / total_time if total_time > 0 else 0
        
        # System performance
        gpu_memory_usage = [m.gpu_memory_used for m in metrics_list 
                           if m.gpu_memory_used is not None]
        
        report = {
            'training_summary': {
                'total_steps': total_steps,
                'training_time_hours': total_time / 3600,
                'steps_per_second': steps_per_second,
                'current_loss': losses[-1] if losses else None,
                'loss_trend': 'improving' if loss_trend < -0.001 else 'stable' if abs(loss_trend) < 0.001 else 'degrading'
            },
            
            'turkish_performance': {
                'avg_char_accuracy': np.mean(turkish_scores) if turkish_scores else None,
                'avg_vowel_harmony': np.mean(harmony_scores) if harmony_scores else None,
                'evaluation_count': len(turkish_scores)
            },
            
            'system_performance': {
                'avg_gpu_memory': np.mean(gpu_memory_usage) if gpu_memory_usage else None,
                'max_gpu_memory': np.max(gpu_memory_usage) if gpu_memory_usage else None,
                'memory_efficiency': 'good' if np.mean(gpu_memory_usage) < 0.8 else 'high' if gpu_memory_usage else 'unknown'
            },
            
            'recommendations': self.auto_optimizer.get_final_recommendations(metrics_list)
        }
        
        return report
    
    def save_metrics(self, filepath: str):
        """Metrics'leri kaydet"""
        
        metrics_data = []
        for m in self.metrics_history:
            metrics_data.append({
                'timestamp': m.timestamp,
                'step': m.step,
                'loss': m.loss,
                'learning_rate': m.learning_rate,
                'turkish_char_accuracy': m.turkish_char_accuracy,
                'vowel_harmony_score': m.vowel_harmony_score,
                'morphology_preservation': m.morphology_preservation,
                'gpu_memory_used': m.gpu_memory_used,
                'gpu_utilization': m.gpu_utilization,
                'cpu_percent': m.cpu_percent
            })
        
        with open(filepath, 'w') as f:
            json.dump(metrics_data, f, indent=2, default=str)
        
        logger.info(f"📊 Metrics saved to {filepath}")


class SystemMonitor:
    """System resource monitoring"""
    
    def get_current_metrics(self) -> Dict[str, float]:
        """Current system metrics"""
        
        metrics = {}
        
        # CPU usage
        try:
            metrics['cpu_percent'] = psutil.cpu_percent(interval=0.1)
        except:
            metrics['cpu_percent'] = None
        
        # GPU metrics (if available)
        if torch.cuda.is_available():
            try:
                gpu_memory = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()
                metrics['gpu_memory_used'] = gpu_memory
                
                # GPU utilization approximation
                metrics['gpu_utilization'] = min(gpu_memory * 1.2, 1.0)  # Rough estimate
            except:
                metrics['gpu_memory_used'] = None
                metrics['gpu_utilization'] = None
        
        return metrics


class AutoTrainingOptimizer:
    """Automatic training optimization suggestions"""
    
    def analyze_performance(self, recent_metrics: List[PerformanceMetrics]) -> List[str]:
        """Recent performance analysis ve öneriler"""
        
        suggestions = []
        
        if len(recent_metrics) < 10:
            return suggestions
        
        # Loss stagnation check
        recent_losses = [m.loss for m in recent_metrics[-10:]]
        loss_variance = np.var(recent_losses)
        
        if loss_variance < 0.001:  # Very low variance = stagnation
            suggestions.append("Loss stagnation detected - consider increasing learning rate")
        
        # Memory efficiency check
        memory_usages = [m.gpu_memory_used for m in recent_metrics 
                        if m.gpu_memory_used is not None]
        
        if memory_usages and np.mean(memory_usages) > 0.95:
            suggestions.append("High memory usage - consider reducing batch size")
        elif memory_usages and np.mean(memory_usages) < 0.6:
            suggestions.append("Low memory usage - consider increasing batch size")
        
        # Turkish performance check
        turkish_scores = [m.turkish_char_accuracy for m in recent_metrics 
                         if m.turkish_char_accuracy is not None]
        
        if turkish_scores and np.mean(turkish_scores) < 0.5:
            suggestions.append("Low Turkish quality - consider curriculum learning or data filtering")
        
        return suggestions
    
    def get_final_recommendations(self, all_metrics: List[PerformanceMetrics]) -> List[str]:
        """Final training recommendations"""
        
        recommendations = []
        
        if not all_metrics:
            return recommendations
        
        # Overall training assessment
        total_steps = len(all_metrics)
        final_loss = all_metrics[-1].loss
        
        if total_steps < 1000:
            recommendations.append("Consider longer training for better convergence")
        
        if final_loss > 2.0:
            recommendations.append("High final loss - consider adjusting hyperparameters")
        elif final_loss < 1.0:
            recommendations.append("Excellent convergence achieved!")
        
        # Turkish performance assessment
        turkish_evaluations = [m for m in all_metrics if m.turkish_char_accuracy is not None]
        
        if turkish_evaluations:
            avg_turkish_score = np.mean([m.turkish_char_accuracy for m in turkish_evaluations])
            
            if avg_turkish_score > 0.8:
                recommendations.append("Excellent Turkish language performance!")
            elif avg_turkish_score < 0.5:
                recommendations.append("Turkish performance needs improvement - consider more Turkish-specific data")
        
        return recommendations


def integrate_realtime_monitoring(trainer_class, tokenizer):
    """Trainer'a real-time monitoring entegre et"""
    
    class MonitoredTrainer(trainer_class):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.monitor = RealTimeMonitor(tokenizer)
            self.monitor.start_monitoring()
        
        def training_step(self, model, inputs):
            # Standard training step
            loss = super().training_step(model, inputs)
            
            # Record metrics
            if hasattr(self, 'state') and self.state.global_step % 10 == 0:
                try:
                    self.monitor.record_step_metrics(
                        model=model,
                        step=self.state.global_step,
                        loss=loss.item() if hasattr(loss, 'item') else float(loss),
                        learning_rate=self.optimizer.param_groups[0]['lr']
                    )
                except Exception as e:
                    logger.warning(f"Monitoring error: {e}")
            
            return loss
        
        def train(self, *args, **kwargs):
            result = super().train(*args, **kwargs)
            
            # Generate final report
            report = self.monitor.generate_performance_report()
            
            # Save metrics
            import os
            output_dir = getattr(self.args, 'output_dir', './training_output')
            os.makedirs(output_dir, exist_ok=True)
            
            self.monitor.save_metrics(f"{output_dir}/training_metrics.json")
            
            with open(f"{output_dir}/performance_report.json", 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            self.monitor.stop_monitoring()
            
            logger.info("📊 Real-time monitoring completed")
            logger.info(f"📈 Final report: {report['training_summary']}")
            
            return result
    
    return MonitoredTrainer


# Test function  
def test_realtime_monitoring():
    """Real-time monitoring test"""
    
    print("🧪 Real-time monitoring test ediliyor...")
    
    # Mock components
    class MockTokenizer:
        def __init__(self):
            self.eos_token_id = 0
        
        def __call__(self, text, return_tensors=None):
            return {'input_ids': torch.randint(0, 1000, (1, 10))}
        
        def decode(self, tokens, skip_special_tokens=False):
            return "Bu test cümlesidir güzel bir örnek."
    
    class MockModel:
        def eval(self): pass
        def generate(self, **kwargs):
            return torch.randint(0, 1000, (1, 20))
    
    tokenizer = MockTokenizer()
    model = MockModel()
    
    # Test monitoring
    monitor = RealTimeMonitor(tokenizer, save_interval=10)
    monitor.start_monitoring()
    
    # Simulate training steps
    for step in range(50):
        loss = 3.0 - (step * 0.05) + np.random.normal(0, 0.1)  # Decreasing loss with noise
        lr = 2e-4 * (0.99 ** (step // 10))  # Decaying learning rate
        
        monitor.record_step_metrics(
            model=model,
            step=step,
            loss=loss,
            learning_rate=lr
        )
        
        time.sleep(0.01)  # Simulate training time
    
    # Generate report
    report = monitor.generate_performance_report()
    
    print(f"✅ Training steps: {report['training_summary']['total_steps']}")
    print(f"✅ Final loss trend: {report['training_summary']['loss_trend']}")
    print(f"✅ Turkish evaluations: {report['turkish_performance']['evaluation_count']}")
    
    monitor.stop_monitoring()
    
    print("🎉 Real-time monitoring test completed!")


if __name__ == "__main__":
    test_realtime_monitoring()