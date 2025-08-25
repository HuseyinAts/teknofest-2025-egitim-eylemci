"""
Model Eğitim Optimizasyon Modülü
Checkpoint, logging, hata kurtarma ve doğrulama döngüleri
"""

import os
import torch
import json
import logging
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, asdict
from datetime import datetime
import traceback
from pathlib import Path
import numpy as np
from functools import wraps
import time
import psutil
import GPUtil

# Logging yapılandırması
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Eğitim konfigürasyonu"""
    model_name: str
    batch_size: int = 16
    learning_rate: float = 2e-5
    num_epochs: int = 3
    warmup_steps: int = 500
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    fp16: bool = True
    save_steps: int = 1000
    eval_steps: int = 500
    logging_steps: int = 100
    save_total_limit: int = 3
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    checkpoint_dir: str = "./checkpoints"
    output_dir: str = "./output"
    seed: int = 42
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, config_dict: Dict):
        return cls(**config_dict)
    
    def save(self, filepath: str):
        """Konfigürasyonu kaydet"""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
            
    @classmethod
    def load(cls, filepath: str):
        """Konfigürasyonu yükle"""
        with open(filepath, 'r') as f:
            return cls.from_dict(json.load(f))


@dataclass
class TrainingMetrics:
    """Eğitim metrikleri"""
    epoch: int
    step: int
    loss: float
    learning_rate: float
    grad_norm: Optional[float] = None
    eval_loss: Optional[float] = None
    eval_accuracy: Optional[float] = None
    eval_f1: Optional[float] = None
    perplexity: Optional[float] = None
    training_time: Optional[float] = None
    gpu_memory_mb: Optional[int] = None
    cpu_percent: Optional[float] = None
    

class RetryDecorator:
    """Yeniden deneme dekoratörü"""
    
    @staticmethod
    def retry(max_attempts: int = 3, delay: float = 1.0, backoff: float = 2.0):
        """
        Fonksiyon başarısız olursa yeniden dene
        
        Args:
            max_attempts: Maksimum deneme sayısı
            delay: İlk bekleme süresi (saniye)
            backoff: Her denemede bekleme süresini çarpan
        """
        def decorator(func: Callable):
            @wraps(func)
            def wrapper(*args, **kwargs):
                attempt = 1
                current_delay = delay
                
                while attempt <= max_attempts:
                    try:
                        return func(*args, **kwargs)
                    except Exception as e:
                        if attempt == max_attempts:
                            logger.error(f"Function {func.__name__} failed after {max_attempts} attempts")
                            raise
                        
                        logger.warning(f"Attempt {attempt} failed: {e}. Retrying in {current_delay} seconds...")
                        time.sleep(current_delay)
                        current_delay *= backoff
                        attempt += 1
                        
            return wrapper
        return decorator


class CheckpointManager:
    """Checkpoint yönetimi"""
    
    def __init__(self, checkpoint_dir: str, save_total_limit: int = 3):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.save_total_limit = save_total_limit
        self.checkpoints = []
        
    def save_checkpoint(self, 
                       model: torch.nn.Module,
                       optimizer: torch.optim.Optimizer,
                       epoch: int,
                       step: int,
                       metrics: TrainingMetrics,
                       config: TrainingConfig,
                       tokenizer=None) -> str:
        """Checkpoint kaydet"""
        
        checkpoint_name = f"checkpoint-epoch{epoch}-step{step}"
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        
        # Model state'ini kaydet
        model_path = checkpoint_path / "model.pt"
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch,
            'step': step,
            'metrics': asdict(metrics)
        }, model_path)
        
        # Config'i kaydet
        config_path = checkpoint_path / "config.json"
        config.save(str(config_path))
        
        # Tokenizer'ı kaydet (varsa)
        if tokenizer is not None:
            try:
                tokenizer.save_pretrained(str(checkpoint_path))
            except Exception as e:
                logger.warning(f"Could not save tokenizer: {e}")
                
        # Metrikleri kaydet
        metrics_path = checkpoint_path / "metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(asdict(metrics), f, indent=2)
            
        # Checkpoint listesini güncelle
        self.checkpoints.append({
            'path': str(checkpoint_path),
            'epoch': epoch,
            'step': step,
            'metrics': metrics
        })
        
        # Eski checkpoint'leri sil
        self._cleanup_old_checkpoints()
        
        logger.info(f"Checkpoint saved: {checkpoint_path}")
        return str(checkpoint_path)
        
    def load_checkpoint(self, 
                       checkpoint_path: str,
                       model: torch.nn.Module,
                       optimizer: Optional[torch.optim.Optimizer] = None) -> Dict:
        """Checkpoint yükle"""
        
        checkpoint_path = Path(checkpoint_path)
        model_path = checkpoint_path / "model.pt"
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
            
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # Model state'ini yükle
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Optimizer state'ini yükle (varsa)
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
        logger.info(f"Checkpoint loaded from: {checkpoint_path}")
        
        return {
            'epoch': checkpoint.get('epoch', 0),
            'step': checkpoint.get('step', 0),
            'metrics': checkpoint.get('metrics', {})
        }
        
    def get_best_checkpoint(self, metric: str = 'eval_loss', greater_is_better: bool = False) -> Optional[str]:
        """En iyi checkpoint'i getir"""
        
        if not self.checkpoints:
            return None
            
        # Metriğe göre sırala
        sorted_checkpoints = sorted(
            self.checkpoints,
            key=lambda x: getattr(x['metrics'], metric, float('inf')),
            reverse=greater_is_better
        )
        
        return sorted_checkpoints[0]['path'] if sorted_checkpoints else None
        
    def _cleanup_old_checkpoints(self):
        """Eski checkpoint'leri temizle"""
        
        if len(self.checkpoints) <= self.save_total_limit:
            return
            
        # En eski checkpoint'leri sil
        to_delete = self.checkpoints[:-self.save_total_limit]
        
        for checkpoint in to_delete:
            try:
                import shutil
                shutil.rmtree(checkpoint['path'])
                logger.info(f"Deleted old checkpoint: {checkpoint['path']}")
            except Exception as e:
                logger.warning(f"Could not delete checkpoint: {e}")
                
        # Listeyi güncelle
        self.checkpoints = self.checkpoints[-self.save_total_limit:]


class SystemMonitor:
    """Sistem kaynaklarını izle"""
    
    @staticmethod
    def get_system_stats() -> Dict[str, Any]:
        """Sistem istatistiklerini al"""
        stats = {}
        
        # CPU kullanımı
        stats['cpu_percent'] = psutil.cpu_percent(interval=0.1)
        stats['cpu_count'] = psutil.cpu_count()
        
        # Bellek kullanımı
        memory = psutil.virtual_memory()
        stats['memory_percent'] = memory.percent
        stats['memory_available_gb'] = memory.available / (1024**3)
        stats['memory_used_gb'] = memory.used / (1024**3)
        
        # GPU kullanımı (varsa)
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]  # İlk GPU
                stats['gpu_name'] = gpu.name
                stats['gpu_memory_used_mb'] = int(gpu.memoryUsed)
                stats['gpu_memory_total_mb'] = int(gpu.memoryTotal)
                stats['gpu_utilization'] = gpu.load * 100
                stats['gpu_temperature'] = gpu.temperature
        except Exception:
            stats['gpu_available'] = False
            
        return stats


class TrainingOptimizer:
    """Ana eğitim optimizasyon sınıfı"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.checkpoint_manager = CheckpointManager(config.checkpoint_dir, config.save_total_limit)
        self.system_monitor = SystemMonitor()
        self.training_history = []
        self.best_metric = float('inf') if not config.greater_is_better else float('-inf')
        self.best_checkpoint_path = None
        
    @RetryDecorator.retry(max_attempts=3, delay=2.0)
    def save_checkpoint_with_retry(self, *args, **kwargs):
        """Checkpoint kaydetmeyi yeniden deneme ile yap"""
        return self.checkpoint_manager.save_checkpoint(*args, **kwargs)
        
    def should_save_checkpoint(self, step: int) -> bool:
        """Checkpoint kaydedilmeli mi?"""
        return step % self.config.save_steps == 0
        
    def should_evaluate(self, step: int) -> bool:
        """Değerlendirme yapılmalı mı?"""
        return step % self.config.eval_steps == 0
        
    def should_log(self, step: int) -> bool:
        """Log yazılmalı mı?"""
        return step % self.config.logging_steps == 0
        
    def update_best_model(self, metrics: TrainingMetrics, checkpoint_path: str):
        """En iyi modeli güncelle"""
        
        metric_value = getattr(metrics, self.config.metric_for_best_model, None)
        if metric_value is None:
            return
            
        is_better = (
            (metric_value > self.best_metric and self.config.greater_is_better) or
            (metric_value < self.best_metric and not self.config.greater_is_better)
        )
        
        if is_better:
            self.best_metric = metric_value
            self.best_checkpoint_path = checkpoint_path
            logger.info(f"New best model! {self.config.metric_for_best_model}: {metric_value}")
            
    def log_metrics(self, metrics: TrainingMetrics):
        """Metrikleri logla"""
        
        # Sistem istatistiklerini ekle
        system_stats = self.system_monitor.get_system_stats()
        metrics.gpu_memory_mb = system_stats.get('gpu_memory_used_mb')
        metrics.cpu_percent = system_stats.get('cpu_percent')
        
        # History'e ekle
        self.training_history.append(asdict(metrics))
        
        # Log yaz
        log_str = (
            f"Epoch: {metrics.epoch} | "
            f"Step: {metrics.step} | "
            f"Loss: {metrics.loss:.4f} | "
            f"LR: {metrics.learning_rate:.2e}"
        )
        
        if metrics.eval_loss is not None:
            log_str += f" | Eval Loss: {metrics.eval_loss:.4f}"
            
        if metrics.eval_accuracy is not None:
            log_str += f" | Eval Acc: {metrics.eval_accuracy:.4f}"
            
        if metrics.gpu_memory_mb is not None:
            log_str += f" | GPU Mem: {metrics.gpu_memory_mb}MB"
            
        logger.info(log_str)
        
    def handle_training_error(self, error: Exception, step: int, epoch: int) -> bool:
        """
        Eğitim hatalarını yönet
        
        Returns:
            True: Eğitime devam et
            False: Eğitimi durdur
        """
        
        error_type = type(error).__name__
        error_msg = str(error)
        
        # CUDA OOM hatası
        if "CUDA out of memory" in error_msg:
            logger.error("CUDA OOM detected! Suggestions:")
            logger.error("1. Reduce batch_size")
            logger.error("2. Enable gradient_checkpointing")
            logger.error("3. Reduce sequence_length")
            logger.error("4. Use fp16/bf16 training")
            
            # Belleği temizle
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            return False
            
        # Gradient explosion
        elif "nan" in error_msg.lower() or "inf" in error_msg.lower():
            logger.error("Gradient explosion detected! Reducing learning rate...")
            # Learning rate'i azalt ve devam et
            return True
            
        # Diğer hatalar
        else:
            logger.error(f"Training error at epoch {epoch}, step {step}:")
            logger.error(f"Error type: {error_type}")
            logger.error(f"Error message: {error_msg}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            
            # Kritik hata değilse devam et
            if error_type in ['KeyError', 'IndexError', 'ValueError']:
                return True
            else:
                return False
                
    def save_training_history(self, filepath: str):
        """Eğitim geçmişini kaydet"""
        
        with open(filepath, 'w') as f:
            json.dump(self.training_history, f, indent=2)
            
        logger.info(f"Training history saved to: {filepath}")
        
    def load_training_history(self, filepath: str):
        """Eğitim geçmişini yükle"""
        
        with open(filepath, 'r') as f:
            self.training_history = json.load(f)
            
        logger.info(f"Training history loaded from: {filepath}")
        
    def generate_report(self) -> Dict[str, Any]:
        """Eğitim raporu oluştur"""
        
        if not self.training_history:
            return {"message": "No training history available"}
            
        history_df = pd.DataFrame(self.training_history)
        
        report = {
            'total_steps': len(self.training_history),
            'total_epochs': history_df['epoch'].max(),
            'best_metric': self.best_metric,
            'best_checkpoint': self.best_checkpoint_path,
            'final_loss': history_df['loss'].iloc[-1],
            'average_loss': history_df['loss'].mean(),
            'loss_std': history_df['loss'].std(),
        }
        
        if 'eval_loss' in history_df.columns:
            report['final_eval_loss'] = history_df['eval_loss'].dropna().iloc[-1] if not history_df['eval_loss'].dropna().empty else None
            report['best_eval_loss'] = history_df['eval_loss'].min()
            
        if 'eval_accuracy' in history_df.columns:
            report['final_eval_accuracy'] = history_df['eval_accuracy'].dropna().iloc[-1] if not history_df['eval_accuracy'].dropna().empty else None
            report['best_eval_accuracy'] = history_df['eval_accuracy'].max()
            
        # GPU kullanımı
        if 'gpu_memory_mb' in history_df.columns:
            report['avg_gpu_memory_mb'] = history_df['gpu_memory_mb'].mean()
            report['max_gpu_memory_mb'] = history_df['gpu_memory_mb'].max()
            
        return report


class EarlyStopping:
    """Early stopping mekanizması"""
    
    def __init__(self, patience: int = 5, min_delta: float = 0.001, greater_is_better: bool = False):
        self.patience = patience
        self.min_delta = min_delta
        self.greater_is_better = greater_is_better
        self.counter = 0
        self.best_score = None
        self.should_stop = False
        
    def __call__(self, score: float) -> bool:
        """
        Skorun iyileşip iyileşmediğini kontrol et
        
        Returns:
            True: Eğitim durdurulmalı
            False: Eğitim devam etmeli
        """
        
        if self.best_score is None:
            self.best_score = score
            return False
            
        if self.greater_is_better:
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta
            
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            logger.info(f"Early stopping triggered! No improvement for {self.patience} evaluations.")
            self.should_stop = True
            return True
            
        return False


class GradientAccumulator:
    """Gradient accumulation yönetimi"""
    
    def __init__(self, accumulation_steps: int = 1):
        self.accumulation_steps = accumulation_steps
        self.step = 0
        
    def should_step(self) -> bool:
        """Optimizer step atılmalı mı?"""
        self.step += 1
        return self.step % self.accumulation_steps == 0
        
    def get_scale_factor(self) -> float:
        """Loss ölçekleme faktörü"""
        return 1.0 / self.accumulation_steps


# Pandas import (rapor için)
try:
    import pandas as pd
except ImportError:
    logger.warning("pandas not installed. Some reporting features will be limited.")