#!/usr/bin/env python3
"""
🚀 ULTRA MEMORY MANAGER FOR A100 40GB
Comprehensive Memory Crisis Resolution System
TEKNOFEST 2025 - Turkish LLM Memory Optimization

CRITICAL FEATURES:
- Progressive dataset loading with memory limits
- Real-time A100 memory monitoring
- Automatic garbage collection
- Memory-efficient streaming
- Crisis prevention and recovery
"""

import gc
import os
import sys
import time
import json
import gzip
import logging
import threading
import psutil
from typing import Dict, List, Optional, Iterator, Any, Tuple, Union
from pathlib import Path
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import warnings
warnings.filterwarnings("ignore")

# Import torch with memory optimization
try:
    import torch
    TORCH_AVAILABLE = True
    
    # A100 optimizations
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
except ImportError:
    TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class MemoryConfig:
    """Memory management configuration for A100 optimization"""
    # A100 40GB specific limits
    gpu_memory_limit_gb: float = 38.0  # Keep 2GB buffer
    system_memory_limit_gb: float = 75.0  # Keep 5GB buffer  
    
    # Progressive loading settings
    max_batch_size_mb: int = 512  # 512MB chunks
    streaming_chunk_size: int = 10000  # 10K samples per chunk
    memory_check_frequency: int = 100  # Check every 100 samples
    
    # Crisis thresholds
    critical_memory_threshold: float = 0.9  # 90% usage triggers crisis mode
    warning_memory_threshold: float = 0.8  # 80% usage warning
    emergency_gc_threshold: float = 0.95  # 95% triggers emergency GC
    
    # Optimization flags
    enable_gradient_checkpointing: bool = True
    enable_mixed_precision: bool = True
    enable_cpu_offload: bool = True
    enable_progressive_gc: bool = True


class A100MemoryMonitor:
    """Real-time A100 memory monitoring system"""
    
    def __init__(self, config: MemoryConfig):
        self.config = config
        self.monitoring_active = False
        self.memory_history = []
        self.crisis_count = 0
        self.last_gc_time = time.time()
        
    def start_monitoring(self, update_interval: int = 5):
        """Start real-time memory monitoring"""
        self.monitoring_active = True
        
        def monitor_loop():
            while self.monitoring_active:
                try:
                    memory_stats = self.get_memory_stats()
                    self.memory_history.append(memory_stats)
                    
                    # Keep only last 100 measurements
                    if len(self.memory_history) > 100:
                        self.memory_history.pop(0)
                    
                    # Check for memory crisis
                    self._check_memory_crisis(memory_stats)
                    
                    time.sleep(update_interval)
                    
                except Exception as e:
                    logger.warning(f"Memory monitoring error: {e}")
                    break
        
        monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        monitor_thread.start()
        logger.info("✅ A100 memory monitoring started")
    
    def stop_monitoring(self):
        """Stop memory monitoring"""
        self.monitoring_active = False
        
    def get_memory_stats(self) -> Dict[str, float]:
        """Get comprehensive memory statistics"""
        stats = {
            'timestamp': time.time(),
            'system_memory_percent': 0.0,
            'system_memory_gb': 0.0,
            'gpu_memory_percent': 0.0,
            'gpu_memory_allocated_gb': 0.0,
            'gpu_memory_reserved_gb': 0.0,
            'gpu_memory_free_gb': 0.0,
            'crisis_level': 'normal'
        }
        
        try:
            # System memory
            system_memory = psutil.virtual_memory()
            stats['system_memory_percent'] = system_memory.percent
            stats['system_memory_gb'] = system_memory.used / (1024**3)
            
            # GPU memory (A100 specific)
            if TORCH_AVAILABLE and torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated(0) / (1024**3)
                reserved = torch.cuda.memory_reserved(0) / (1024**3)
                total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                free = total - reserved
                
                stats['gpu_memory_allocated_gb'] = allocated
                stats['gpu_memory_reserved_gb'] = reserved  
                stats['gpu_memory_free_gb'] = free
                stats['gpu_memory_percent'] = (reserved / total) * 100
                
                # Determine crisis level
                if stats['gpu_memory_percent'] > 95:
                    stats['crisis_level'] = 'emergency'
                elif stats['gpu_memory_percent'] > 90:
                    stats['crisis_level'] = 'critical' 
                elif stats['gpu_memory_percent'] > 80:
                    stats['crisis_level'] = 'warning'
                    
        except Exception as e:
            logger.warning(f"Memory stats collection error: {e}")
            
        return stats
    
    def _check_memory_crisis(self, stats: Dict[str, float]):
        """Check and respond to memory crisis"""
        crisis_level = stats['crisis_level']
        
        if crisis_level == 'emergency':
            self.crisis_count += 1
            logger.error(f"🚨 MEMORY EMERGENCY! GPU: {stats['gpu_memory_percent']:.1f}%")
            self.emergency_memory_cleanup()
            
        elif crisis_level == 'critical':
            logger.warning(f"⚠️ Memory Critical: GPU: {stats['gpu_memory_percent']:.1f}%")
            self.force_garbage_collection()
            
        elif crisis_level == 'warning':
            logger.info(f"💡 Memory Warning: GPU: {stats['gpu_memory_percent']:.1f}%")
            
    def emergency_memory_cleanup(self):
        """Emergency memory cleanup for A100"""
        try:
            logger.info("🆘 Starting emergency memory cleanup...")
            
            # Force garbage collection
            gc.collect()
            
            if TORCH_AVAILABLE and torch.cuda.is_available():
                # Clear CUDA cache
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                
                # Force memory defragmentation
                try:
                    torch.cuda.memory._dump_snapshot("emergency_cleanup.pickle")
                except:
                    pass
                
            # Clear Python caches
            sys.intern.__dict__.clear()
            
            # Wait and check result
            time.sleep(2)
            new_stats = self.get_memory_stats()
            logger.info(f"🔧 Emergency cleanup result: GPU {new_stats['gpu_memory_percent']:.1f}%")
            
        except Exception as e:
            logger.error(f"Emergency cleanup failed: {e}")
    
    def force_garbage_collection(self):
        """Force comprehensive garbage collection"""
        try:
            # Only run GC if enough time has passed
            current_time = time.time()
            if current_time - self.last_gc_time < 30:  # 30 second cooldown
                return
                
            logger.info("🧹 Running forced garbage collection...")
            
            # Multiple GC passes for thorough cleanup
            for _ in range(3):
                collected = gc.collect()
                if collected == 0:
                    break
                    
            if TORCH_AVAILABLE and torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            self.last_gc_time = current_time
            
        except Exception as e:
            logger.warning(f"Force GC failed: {e}")


class ProgressiveDatasetLoader:
    """Memory-efficient progressive dataset loader"""
    
    def __init__(self, config: MemoryConfig, memory_monitor: A100MemoryMonitor):
        self.config = config
        self.memory_monitor = memory_monitor
        self.loaded_chunks = {}
        self.chunk_cache = {}
        
    def load_streaming_dataset(self, 
                             dataset_paths: List[str],
                             max_samples: Optional[int] = None) -> Iterator[Dict[str, Any]]:
        """Load dataset with progressive streaming and memory management"""
        
        total_loaded = 0
        chunk_count = 0
        
        for dataset_path in dataset_paths:
            if not os.path.exists(dataset_path):
                logger.warning(f"Dataset not found: {dataset_path}")
                continue
                
            logger.info(f"📂 Loading dataset progressively: {dataset_path}")
            
            try:
                for chunk in self._stream_dataset_chunks(dataset_path):
                    # Memory check before processing chunk
                    memory_stats = self.memory_monitor.get_memory_stats()
                    
                    if memory_stats['crisis_level'] in ['critical', 'emergency']:
                        logger.warning("⚠️ Memory crisis detected, pausing dataset loading")
                        self.memory_monitor.force_garbage_collection()
                        time.sleep(5)  # Allow memory to stabilize
                        continue
                    
                    # Process chunk samples
                    for sample in chunk:
                        if max_samples and total_loaded >= max_samples:
                            return
                            
                        yield sample
                        total_loaded += 1
                        
                        # Periodic memory check
                        if total_loaded % self.config.memory_check_frequency == 0:
                            self._check_and_cleanup_memory()
                    
                    chunk_count += 1
                    logger.debug(f"Processed chunk {chunk_count}, total samples: {total_loaded}")
                    
            except Exception as e:
                logger.error(f"Error loading dataset {dataset_path}: {e}")
                continue
        
        logger.info(f"✅ Progressive loading complete: {total_loaded} samples")
    
    def _stream_dataset_chunks(self, dataset_path: str) -> Iterator[List[Dict[str, Any]]]:
        """Stream dataset in memory-efficient chunks"""
        
        file_path = Path(dataset_path)
        
        if file_path.suffix == '.json':
            yield from self._stream_json_chunks(file_path)
        elif file_path.suffix == '.gz':
            yield from self._stream_gz_chunks(file_path)
        elif file_path.suffix == '.jsonl':
            yield from self._stream_jsonl_chunks(file_path)
        else:
            logger.warning(f"Unsupported file format: {file_path.suffix}")
            return
    
    def _stream_json_chunks(self, file_path: Path) -> Iterator[List[Dict[str, Any]]]:
        """Stream JSON file in chunks"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            if isinstance(data, list):
                # Process in chunks
                chunk_size = self.config.streaming_chunk_size
                for i in range(0, len(data), chunk_size):
                    chunk = data[i:i + chunk_size]
                    yield chunk
            else:
                # Single object, wrap in list
                yield [data]
                
        except Exception as e:
            logger.error(f"JSON streaming error: {e}")
    
    def _stream_gz_chunks(self, file_path: Path) -> Iterator[List[Dict[str, Any]]]:
        """Stream compressed file in chunks"""
        try:
            chunk = []
            chunk_size = self.config.streaming_chunk_size
            
            with gzip.open(file_path, 'rt', encoding='utf-8') as f:
                for line in f:
                    try:
                        item = json.loads(line.strip())
                        chunk.append(item)
                        
                        if len(chunk) >= chunk_size:
                            yield chunk
                            chunk = []
                            
                    except json.JSONDecodeError:
                        continue
                        
                # Yield remaining items
                if chunk:
                    yield chunk
                    
        except Exception as e:
            logger.error(f"GZ streaming error: {e}")
    
    def _stream_jsonl_chunks(self, file_path: Path) -> Iterator[List[Dict[str, Any]]]:
        """Stream JSONL file in chunks"""
        try:
            chunk = []
            chunk_size = self.config.streaming_chunk_size
            
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        item = json.loads(line.strip())
                        chunk.append(item)
                        
                        if len(chunk) >= chunk_size:
                            yield chunk
                            chunk = []
                            
                    except json.JSONDecodeError:
                        continue
                        
                # Yield remaining items
                if chunk:
                    yield chunk
                    
        except Exception as e:
            logger.error(f"JSONL streaming error: {e}")
    
    def _check_and_cleanup_memory(self):
        """Check memory and cleanup if needed"""
        memory_stats = self.memory_monitor.get_memory_stats()
        
        if memory_stats['gpu_memory_percent'] > 85:
            logger.debug("🧹 Preventive memory cleanup triggered")
            gc.collect()
            
            if TORCH_AVAILABLE and torch.cuda.is_available():
                torch.cuda.empty_cache()


class UltraMemoryManager:
    """Ultra Memory Manager - Master class for A100 memory optimization"""
    
    def __init__(self, config: Optional[MemoryConfig] = None):
        self.config = config or MemoryConfig()
        self.memory_monitor = A100MemoryMonitor(self.config)
        self.dataset_loader = ProgressiveDatasetLoader(self.config, self.memory_monitor)
        self.optimization_active = False
        
        # Initialize A100 optimizations
        self._initialize_a100_optimizations()
        
    def _initialize_a100_optimizations(self):
        """Initialize A100-specific memory optimizations"""
        try:
            if TORCH_AVAILABLE and torch.cuda.is_available():
                # Memory pool optimization
                os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
                
                # A100 specific optimizations
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                torch.backends.cudnn.benchmark = True
                
                logger.info("✅ A100 memory optimizations initialized")
                
        except Exception as e:
            logger.warning(f"A100 optimization failed: {e}")
    
    def start_memory_optimization(self):
        """Start comprehensive memory optimization"""
        if self.optimization_active:
            logger.warning("Memory optimization already active")
            return
            
        logger.info("🚀 Starting Ultra Memory Manager for A100...")
        
        # Start memory monitoring
        self.memory_monitor.start_monitoring(update_interval=5)
        
        # Enable progressive GC if configured
        if self.config.enable_progressive_gc:
            self._start_progressive_gc()
            
        self.optimization_active = True
        logger.info("✅ Ultra Memory Manager active")
    
    def stop_memory_optimization(self):
        """Stop memory optimization"""
        self.optimization_active = False
        self.memory_monitor.stop_monitoring()
        logger.info("🛑 Ultra Memory Manager stopped")
    
    def _start_progressive_gc(self):
        """Start progressive garbage collection"""
        def progressive_gc_loop():
            while self.optimization_active:
                try:
                    time.sleep(60)  # Run every minute
                    
                    memory_stats = self.memory_monitor.get_memory_stats()
                    if memory_stats['gpu_memory_percent'] > 70:
                        logger.debug("🧹 Progressive GC triggered")
                        gc.collect()
                        
                        if TORCH_AVAILABLE and torch.cuda.is_available():
                            torch.cuda.empty_cache()
                            
                except Exception as e:
                    logger.warning(f"Progressive GC error: {e}")
                    break
        
        gc_thread = threading.Thread(target=progressive_gc_loop, daemon=True)
        gc_thread.start()
    
    def get_optimal_batch_size(self, model_size_gb: float = 8.0) -> int:
        """Calculate optimal batch size for A100 memory"""
        try:
            memory_stats = self.memory_monitor.get_memory_stats()
            available_gb = memory_stats['gpu_memory_free_gb']
            
            # Conservative estimation
            available_for_batch = available_gb - model_size_gb - 4.0  # 4GB buffer
            
            if available_for_batch <= 0:
                return 1  # Minimum batch size
                
            # Estimate batch size (rough approximation)
            samples_per_gb = 32  # Conservative estimate
            optimal_batch = int(available_for_batch * samples_per_gb)
            
            # Clamp to reasonable range
            return max(1, min(optimal_batch, 64))
            
        except Exception as e:
            logger.warning(f"Batch size calculation failed: {e}")
            return 8  # Safe default
    
    def create_memory_efficient_dataset(self, 
                                      dataset_paths: List[str],
                                      max_samples: Optional[int] = None) -> Iterator[Dict[str, Any]]:
        """Create memory-efficient dataset with progressive loading"""
        return self.dataset_loader.load_streaming_dataset(dataset_paths, max_samples)
    
    def get_memory_report(self) -> Dict[str, Any]:
        """Get comprehensive memory report"""
        stats = self.memory_monitor.get_memory_stats()
        
        report = {
            'current_stats': stats,
            'crisis_count': self.memory_monitor.crisis_count,
            'optimization_active': self.optimization_active,
            'config': {
                'gpu_limit_gb': self.config.gpu_memory_limit_gb,
                'system_limit_gb': self.config.system_memory_limit_gb,
                'streaming_chunk_size': self.config.streaming_chunk_size
            },
            'recommendations': self._generate_memory_recommendations(stats)
        }
        
        return report
    
    def _generate_memory_recommendations(self, stats: Dict[str, float]) -> List[str]:
        """Generate memory optimization recommendations"""
        recommendations = []
        
        if stats['gpu_memory_percent'] > 90:
            recommendations.append("🚨 Reduce batch size immediately")
            recommendations.append("🔧 Enable gradient checkpointing")
            recommendations.append("💾 Consider CPU offloading")
            
        elif stats['gpu_memory_percent'] > 80:
            recommendations.append("⚠️ Consider reducing sequence length")
            recommendations.append("🔄 Enable mixed precision training")
            
        elif stats['gpu_memory_percent'] < 50:
            recommendations.append("📈 Consider increasing batch size")
            recommendations.append("🚀 GPU memory underutilized")
            
        if stats['system_memory_percent'] > 85:
            recommendations.append("💻 System RAM high, reduce data loading workers")
            
        return recommendations


# Convenience functions for easy integration
def create_memory_manager(gpu_limit_gb: float = 38.0) -> UltraMemoryManager:
    """Create optimized memory manager for A100"""
    config = MemoryConfig(gpu_memory_limit_gb=gpu_limit_gb)
    return UltraMemoryManager(config)


def get_a100_memory_stats() -> Dict[str, float]:
    """Quick function to get A100 memory stats"""
    monitor = A100MemoryMonitor(MemoryConfig())
    return monitor.get_memory_stats()


# Example usage and testing
if __name__ == "__main__":
    # Initialize memory manager
    memory_manager = create_memory_manager()
    
    try:
        # Start optimization
        memory_manager.start_memory_optimization()
        
        # Test progressive dataset loading
        dataset_paths = [
            "/content/turkish_llm_10k_dataset.jsonl.gz",
            "/content/competition_dataset.json"
        ]
        
        print("🧪 Testing progressive dataset loading...")
        sample_count = 0
        
        for sample in memory_manager.create_memory_efficient_dataset(dataset_paths, max_samples=1000):
            sample_count += 1
            if sample_count % 100 == 0:
                print(f"Loaded {sample_count} samples")
                
        print(f"✅ Test complete: {sample_count} samples loaded")
        
        # Print memory report
        report = memory_manager.get_memory_report()
        print(f"\n📊 Memory Report:")
        print(f"GPU Usage: {report['current_stats']['gpu_memory_percent']:.1f}%")
        print(f"Crisis Count: {report['crisis_count']}")
        print(f"Recommendations: {len(report['recommendations'])}")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        
    finally:
        memory_manager.stop_memory_optimization()