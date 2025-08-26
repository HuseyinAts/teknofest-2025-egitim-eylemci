"""
🔄 HYBRID ENSEMBLE TRAINING APPROACH
Birden fazla model paralel train edip en iyisini seç

ÖNERİ: Risk mitigation için multiple approaches simultaneously
"""

import torch
import torch.nn as nn
import asyncio
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import threading

logger = logging.getLogger(__name__)

@dataclass
class EnsembleConfig:
    """Ensemble training configuration"""
    model_variants: List[Dict[str, Any]]
    parallel_execution: bool = True
    selection_criteria: str = "best_loss"  # "best_loss", "best_turkish", "best_overall"
    max_concurrent: int = 2  # Colab memory limit
    early_stopping_patience: int = 3

class ModelVariant:
    """Single model variant for ensemble"""
    
    def __init__(self, variant_id: str, config: Dict[str, Any]):
        self.variant_id = variant_id
        self.config = config
        self.model = None
        self.trainer = None
        self.results = {}
        self.status = "initialized"
        self.training_thread = None
    
    def setup_model(self):
        """Model setup"""
        logger.info(f"🔧 Setting up variant {self.variant_id}")
        
        # Model configuration based on variant type
        if self.config['type'] == 'conservative':
            # Conservative approach - original tokenizer
            from colab_pro_a100_optimized_trainer import ColabProA100Trainer, ColabProA100Config
            
            trainer_config = ColabProA100Config()
            trainer_config.learning_rate = 1e-4  # Lower LR
            trainer_config.num_epochs = 2  # Fewer epochs
            trainer_config.use_ewc = True
            trainer_config.use_self_synthesis = True
            
        elif self.config['type'] == 'aggressive':
            # Aggressive approach - higher LR, more optimization
            from colab_pro_a100_optimized_trainer import ColabProA100Trainer, ColabProA100Config
            
            trainer_config = ColabProA100Config()
            trainer_config.learning_rate = 3e-4  # Higher LR
            trainer_config.num_epochs = 4  # More epochs
            trainer_config.lora_r = 128  # Higher rank
            trainer_config.use_ewc = False  # Less conservative
            
        elif self.config['type'] == 'curriculum':
            # Curriculum learning approach
            from advanced_curriculum_learning import integrate_curriculum_learning
            from colab_pro_a100_optimized_trainer import ColabProA100Trainer, ColabProA100Config
            
            trainer_config = ColabProA100Config()
            trainer_config.learning_rate = 2e-4
            trainer_config.num_epochs = 3
            # Will use curriculum learning
            
        else:
            # Default balanced approach
            from colab_pro_a100_optimized_trainer import ColabProA100Trainer, ColabProA100Config
            trainer_config = ColabProA100Config()
        
        # Customize output directory
        trainer_config.output_dir = f"/content/drive/MyDrive/ensemble_{self.variant_id}"
        
        # Create trainer
        self.trainer = ColabProA100Trainer(trainer_config)
        self.status = "configured"
        
        logger.info(f"✅ Variant {self.variant_id} configured ({self.config['type']})")
    
    def train_async(self) -> Dict[str, Any]:
        """Asynchronous training execution"""
        
        logger.info(f"🚀 Starting training for variant {self.variant_id}")
        self.status = "training"
        
        try:
            start_time = time.time()
            
            # Execute training
            results = self.trainer.train()
            
            end_time = time.time()
            
            # Enhance results with variant info
            enhanced_results = {
                **results,
                'variant_id': self.variant_id,
                'variant_type': self.config['type'],
                'training_duration': end_time - start_time,
                'config_used': self.trainer.config.__dict__
            }
            
            self.results = enhanced_results
            self.status = "completed"
            
            logger.info(f"✅ Variant {self.variant_id} completed - Loss: {results.get('final_loss', 'N/A'):.4f}")
            
            return enhanced_results
            
        except Exception as e:
            logger.error(f"❌ Variant {self.variant_id} failed: {e}")
            self.status = "failed"
            self.results = {'error': str(e)}
            return self.results

class EnsembleTrainer:
    """Ensemble training orchestrator"""
    
    def __init__(self, config: EnsembleConfig):
        self.config = config
        self.variants = []
        self.results = []
        self.best_model = None
        self.ensemble_results = {}
        
        # Setup variants
        for i, variant_config in enumerate(config.model_variants):
            variant = ModelVariant(f"variant_{i}", variant_config)
            self.variants.append(variant)
    
    def train_ensemble_parallel(self) -> Dict[str, Any]:
        """Parallel ensemble training"""
        
        logger.info(f"🔄 Starting ensemble training with {len(self.variants)} variants")
        
        # Setup all models first
        for variant in self.variants:
            variant.setup_model()
        
        # Execute training in parallel (with concurrency limit)
        with ThreadPoolExecutor(max_workers=self.config.max_concurrent) as executor:
            
            # Submit all training jobs
            future_to_variant = {
                executor.submit(variant.train_async): variant 
                for variant in self.variants
            }
            
            # Collect results as they complete
            completed_results = []
            
            for future in future_to_variant:
                variant = future_to_variant[future]
                
                try:
                    result = future.result(timeout=7200)  # 2 hour timeout per variant
                    completed_results.append(result)
                    
                    logger.info(f"✅ {variant.variant_id} completed")
                    
                except Exception as e:
                    logger.error(f"❌ {variant.variant_id} failed: {e}")
                    completed_results.append({
                        'variant_id': variant.variant_id,
                        'error': str(e),
                        'status': 'failed'
                    })
        
        self.results = completed_results
        
        # Select best model
        self.best_model = self._select_best_model()
        
        # Compile ensemble results
        self.ensemble_results = self._compile_ensemble_results()
        
        logger.info(f"🏆 Ensemble training completed - Best: {self.best_model['variant_id']}")
        
        return self.ensemble_results
    
    def train_ensemble_sequential(self) -> Dict[str, Any]:
        """Sequential ensemble training (memory-safe)"""
        
        logger.info(f"🔄 Starting sequential ensemble training")
        
        completed_results = []
        
        for variant in self.variants:
            logger.info(f"🎯 Training {variant.variant_id}")
            
            # Setup and train one at a time
            variant.setup_model()
            result = variant.train_async()
            completed_results.append(result)
            
            # Cleanup to free memory
            if hasattr(variant.trainer, 'model'):
                del variant.trainer.model
            torch.cuda.empty_cache()
            
            logger.info(f"✅ {variant.variant_id} completed and cleaned up")
        
        self.results = completed_results
        self.best_model = self._select_best_model()
        self.ensemble_results = self._compile_ensemble_results()
        
        return self.ensemble_results
    
    def _select_best_model(self) -> Dict[str, Any]:
        """Best model selection"""
        
        valid_results = [r for r in self.results if 'error' not in r]
        
        if not valid_results:
            logger.error("❌ No successful training runs!")
            return {'error': 'All variants failed'}
        
        # Selection based on criteria
        if self.config.selection_criteria == "best_loss":
            best = min(valid_results, key=lambda x: x.get('final_loss', float('inf')))
        
        elif self.config.selection_criteria == "best_turkish":
            # Prioritize Turkish performance (if available)
            def turkish_score(result):
                turkish_metrics = result.get('turkish_performance', {})
                return turkish_metrics.get('overall_turkish_score', 0)
            
            best = max(valid_results, key=turkish_score)
        
        else:  # best_overall
            # Combined scoring
            def overall_score(result):
                loss_score = 1.0 / (1.0 + result.get('final_loss', 3.0))  # Lower loss = higher score
                turkish_score = result.get('turkish_performance', {}).get('overall_turkish_score', 0.5)
                time_score = 1.0 / (1.0 + result.get('training_duration', 3600) / 3600)  # Faster = higher score
                
                return loss_score * 0.5 + turkish_score * 0.3 + time_score * 0.2
            
            best = max(valid_results, key=overall_score)
        
        logger.info(f"🏆 Best model: {best['variant_id']} (Loss: {best.get('final_loss', 'N/A')})")
        
        return best
    
    def _compile_ensemble_results(self) -> Dict[str, Any]:
        """Ensemble results compilation"""
        
        successful_runs = [r for r in self.results if 'error' not in r]
        failed_runs = [r for r in self.results if 'error' in r]
        
        # Statistics
        if successful_runs:
            losses = [r['final_loss'] for r in successful_runs if 'final_loss' in r]
            durations = [r['training_duration'] for r in successful_runs if 'training_duration' in r]
            
            stats = {
                'successful_variants': len(successful_runs),
                'failed_variants': len(failed_runs),
                'average_loss': sum(losses) / len(losses) if losses else None,
                'best_loss': min(losses) if losses else None,
                'worst_loss': max(losses) if losses else None,
                'average_duration': sum(durations) / len(durations) if durations else None,
                'total_training_time': sum(durations) if durations else None
            }
        else:
            stats = {
                'successful_variants': 0,
                'failed_variants': len(failed_runs),
                'error': 'All variants failed'
            }
        
        return {
            'ensemble_summary': stats,
            'best_model': self.best_model,
            'all_results': self.results,
            'selection_criteria': self.config.selection_criteria,
            'ensemble_config': self.config.__dict__
        }
    
    def save_ensemble_results(self, output_path: str = "/content/drive/MyDrive/ensemble_results.json"):
        """Save ensemble results"""
        
        with open(output_path, 'w') as f:
            json.dump(self.ensemble_results, f, indent=2, default=str)
        
        logger.info(f"📊 Ensemble results saved to {output_path}")


def create_default_ensemble_config() -> EnsembleConfig:
    """Default ensemble configuration for Colab"""
    
    variants = [
        {
            'type': 'conservative',
            'description': 'Safe approach with lower LR and EWC',
            'expected_loss': '1.8-2.3',
            'risk': 'low'
        },
        {
            'type': 'aggressive', 
            'description': 'Higher LR and optimization for faster convergence',
            'expected_loss': '1.5-2.8',
            'risk': 'medium'
        },
        {
            'type': 'curriculum',
            'description': 'Curriculum learning for structured progression',
            'expected_loss': '1.6-2.2',
            'risk': 'low-medium'
        }
    ]
    
    return EnsembleConfig(
        model_variants=variants,
        parallel_execution=False,  # Sequential for Colab memory safety
        selection_criteria="best_overall",
        max_concurrent=1,
        early_stopping_patience=3
    )


def run_ensemble_training() -> Dict[str, Any]:
    """Colab'da ensemble training çalıştır"""
    
    print("🔄 HYBRID ENSEMBLE TRAINING BAŞLIYOR")
    print("=" * 60)
    print("🎯 Strategy: Multiple approaches simultaneously")
    print("✅ Risk Mitigation: En az bir variant başarılı olacak")
    print("🏆 Selection: En iyi performansı otomatik seç")
    print("=" * 60)
    
    # Create ensemble configuration
    config = create_default_ensemble_config()
    
    # Create ensemble trainer
    ensemble = EnsembleTrainer(config)
    
    try:
        # Run ensemble training
        if config.parallel_execution:
            results = ensemble.train_ensemble_parallel()
        else:
            results = ensemble.train_ensemble_sequential()
        
        # Save results
        ensemble.save_ensemble_results()
        
        # Print summary
        print("\n🎉 ENSEMBLE TRAINING COMPLETED")
        print("=" * 50)
        
        summary = results['ensemble_summary']
        print(f"✅ Successful variants: {summary['successful_variants']}")
        print(f"❌ Failed variants: {summary['failed_variants']}")
        
        if summary['successful_variants'] > 0:
            print(f"🏆 Best loss: {summary['best_loss']:.4f}")
            print(f"📊 Average loss: {summary['average_loss']:.4f}")
            print(f"⏱️ Total training time: {summary['total_training_time']/3600:.2f} hours")
            
            best_model = results['best_model']
            print(f"🥇 Winner: {best_model['variant_id']} ({best_model.get('variant_type', 'unknown')})")
        else:
            print("❌ All variants failed - check individual logs")
        
        return results
        
    except Exception as e:
        logger.error(f"Ensemble training failed: {e}")
        return {'error': str(e)}


# Test function
def test_ensemble_approach():
    """Ensemble approach test"""
    
    print("🧪 Ensemble approach test ediliyor...")
    
    # Mock variant configs
    test_config = EnsembleConfig(
        model_variants=[
            {'type': 'conservative', 'description': 'Safe approach'},
            {'type': 'aggressive', 'description': 'Fast approach'}
        ],
        parallel_execution=False,
        selection_criteria="best_loss",
        max_concurrent=1
    )
    
    # Mock results simulation
    mock_results = [
        {
            'variant_id': 'variant_0',
            'variant_type': 'conservative', 
            'final_loss': 2.1,
            'training_duration': 3600,
            'turkish_performance': {'overall_turkish_score': 0.8}
        },
        {
            'variant_id': 'variant_1',
            'variant_type': 'aggressive',
            'final_loss': 1.8,
            'training_duration': 2400,
            'turkish_performance': {'overall_turkish_score': 0.7}
        }
    ]
    
    # Test selection logic
    ensemble = EnsembleTrainer(test_config)
    ensemble.results = mock_results
    
    best_model = ensemble._select_best_model()
    ensemble_results = ensemble._compile_ensemble_results()
    
    print(f"✅ Best model selected: {best_model['variant_id']}")
    print(f"✅ Best loss: {best_model['final_loss']}")
    print(f"✅ Ensemble summary: {ensemble_results['ensemble_summary']}")
    
    print("🎉 Ensemble test completed!")


if __name__ == "__main__":
    test_ensemble_approach()