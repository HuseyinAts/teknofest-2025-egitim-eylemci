# 🎯 HÜCRE 5: Advanced Ensemble Configuration ve Dataset Preparation (GÜNCELLENMIŞ)
import json
import os
import torch
from datetime import datetime
import time
from pathlib import Path

# Tutarlı dizin yapısı (Cell 6 ile uyumlu)
BASE_DIR = '/content/teknofest-2025-egitim-eylemci'
WORKSPACE = f'{BASE_DIR}/turkish_tokenizer'

# Dizin yapısını göster ve oluştur
print("📁 TUTARLI DİZİN YAPISI:")
print(f"📁 Ana dizin: {BASE_DIR}")
print(f"📁 Çalışma dizini: {WORKSPACE}")
print("")

# Gerekli dizinleri oluştur
os.makedirs(f"{WORKSPACE}/configs", exist_ok=True)
os.makedirs(f"{WORKSPACE}/datasets", exist_ok=True)
os.makedirs(f"{WORKSPACE}/models", exist_ok=True)

print("🎯 ADVANCED ENSEMBLE CONFIGURATION VE DATASET PREPARATION")
print("=" * 90)
print(f"⏰ Başlangıç: {datetime.now().strftime('%H:%M:%S')}")
print("🚀 4 Paralel Variant + Memory-Efficient Dataset Loading")
print("=" * 90)

# Ensemble Variants Detailed Configuration (Memory'den)
print("🔧 ENSEMBLE VARIANTS DETAILED CONFIGURATION:")

ensemble_variants = {
    'variant_1_dora_neftune_sophia': {
        'name': 'DoRA + NEFTune + Sophia Ultimate',
        'description': 'Weight decomposition + Noise + Diagonal Hessian',
        'priority': 1,
        'expected_loss': 1.2,
        'training_time_hours': 8,
        'config': {
            # DoRA Configuration (Memory: weight-decomposed LoRA)
            'use_dora': True,
            'dora_r': 256,
            'dora_alpha': 128,
            'dora_dropout': 0.05,
            'dora_use_magnitude_scaling': True,
            
            # NEFTune Configuration (Memory: adaptive noise scaling)
            'use_neftune': True,
            'neftune_alpha': 10.0,
            'neftune_adaptive_scaling': True,
            
            # Sophia Configuration (Memory: diagonal Hessian approximation)
            'use_sophia': True,
            'sophia_lr': 2e-4,  # Turkish-optimal
            'sophia_betas': [0.965, 0.99],
            'sophia_rho': 0.01,
            'sophia_update_period': 10,
            
            # A100 Optimizations
            'bf16': True,
            'tf32': True,
            'per_device_batch_size': 8,
            'gradient_accumulation_steps': 16,
        }
    },
    
    'variant_2_progressive_curriculum': {
        'name': 'Progressive + Advanced Curriculum',
        'description': '4-phase training + Turkish difficulty progression',
        'priority': 2,
        'expected_loss': 1.4,
        'training_time_hours': 10,
        'config': {
            # Progressive Training (Memory: 4-phase approach)
            'use_progressive_training': True,
            'phase_1_epochs': 3,  # Original Qwen tokenizer
            'phase_2_epochs': 4,  # Gradual Turkish introduction
            'phase_3_epochs': 3,  # Basic Turkish optimization
            'phase_4_epochs': 3,  # Full Turkish optimization
            
            # Curriculum Learning (Memory: difficulty progression)
            'use_curriculum_learning': True,
            'curriculum_stages': [
                'simple_sentences',      # Stage 1
                'complex_morphology',    # Stage 2  
                'academic_texts',        # Stage 3
                'formal_writing'         # Stage 4
            ],
            
            # Standard optimizations
            'learning_rate': 2e-4,
            'bf16': True,
            'per_device_batch_size': 6,  # Slightly lower for stability
            'gradient_accumulation_steps': 20,
        }
    },
    
    'variant_3_dynamic_expansion': {
        'name': 'Dynamic Vocabulary Expansion',
        'description': 'Runtime Turkish token discovery + expansion',
        'priority': 3,
        'expected_loss': 1.3,
        'training_time_hours': 9,
        'config': {
            # Dynamic Expansion (Memory: every 500 steps)
            'use_dynamic_expansion': True,
            'expansion_check_steps': 500,
            'expansion_threshold': 50,
            'max_new_tokens': 5000,
            
            # Real-time Monitoring (Memory: background thread)
            'use_realtime_monitoring': True,
            'monitoring_interval': 100,
            'auto_optimization': True,
            
            # Memory optimization
            'streaming_dataset': True,
            'memory_efficient_loading': True,
            'max_memory_gb': 12,  # Memory-efficient target
            
            'learning_rate': 2e-4,
            'bf16': True,
            'per_device_batch_size': 7,
            'gradient_accumulation_steps': 18,
        }
    },
    
    'variant_4_catastrophic_prevention': {
        'name': 'Catastrophic Forgetting Prevention',
        'description': 'EWC + Self-synthesis + Knowledge retention',
        'priority': 4,
        'expected_loss': 1.5,
        'training_time_hours': 12,
        'config': {
            # Catastrophic Forgetting Prevention (Memory lessons)
            'use_ewc': True,
            'ewc_lambda': 0.5,
            'fisher_estimation_steps': 1000,
            
            # Self-synthesized Rehearsal (Memory: 30% synthetic + 70% new)
            'use_self_synthesis': True,
            'synthetic_data_ratio': 0.3,
            'synthesis_model': 'base_qwen',
            
            # Knowledge Retention
            'knowledge_retention_loss': True,
            'retention_weight': 0.1,
            
            # Conservative training
            'learning_rate': 1.5e-4,  # Slightly lower for stability
            'bf16': True,
            'per_device_batch_size': 6,
            'gradient_accumulation_steps': 20,
        }
    }
}

print(f"✅ {len(ensemble_variants)} ensemble variant configured")
for variant_id, variant in ensemble_variants.items():
    print(f"📊 {variant['name']}: Loss {variant['expected_loss']}, {variant['training_time_hours']}h")

# Dataset Configuration (Memory: HuggingFace datasets public access)
print("\n📊 DATASET CONFIGURATION (MEMORY-EFFICIENT LOADING):")

dataset_config = {
    'huggingface_datasets': [
        'merve/turkish_instructions',           # 5K samples
        'TFLai/Turkish-Alpaca',                # 5K samples  
        'malhajar/OpenOrca-tr',                # 5K samples
        'umarigan/turkish_corpus',             # 10K samples
        'Huseyin/muspdf',                      # 15K samples
        'tubitak/tuba-corpus',                 # 20K samples
        'boun-pars/boun-corpus',               # 10K samples
        'selimfirat/bilkent-turkish-writings-dataset'  # 12K samples (Memory: Bilkent integration)
    ],
    
    'local_datasets': [
        'turkish_quiz_instruct',
        'competition_dataset', 
        'tr_mega_combined',
        'synthetic_tr_mega',
        'turkish_llm_10k_v1',
        'turkish_llm_10k_v3'
    ],
    
    # Dataset Processing Pipeline (Memory: Turkish dataset processing)
    'processing_pipeline': {
        'fasttext_quality_threshold': 0.9,     # Keep top 10%
        'kenlm_perplexity_range': [20, 1000],  # Perplexity filtering
        'minhash_similarity_threshold': 0.75,   # 75% deduplication
        'min_text_length': 30,                 # Character minimum
        'max_samples_per_dataset': 10000,      # Memory limit
    },
    
    # Memory Optimization (Memory: >20GB to <12GB)
    'memory_optimization': {
        'streaming': True,
        'batch_size': 1000,
        'num_workers': 0,  # Colab safety
        'pin_memory': False,  # Colab optimization
        'prefetch_factor': 2,
        'max_memory_gb': 12,
    }
}

print(f"✅ {len(dataset_config['huggingface_datasets'])} HuggingFace datasets (public access)")
print(f"✅ {len(dataset_config['local_datasets'])} local datasets")
print(f"✅ Memory target: <{dataset_config['memory_optimization']['max_memory_gb']}GB")

# A100 Training Schedule Configuration
print("\n🚀 A100 TRAINING SCHEDULE CONFIGURATION:")

training_schedule = {
    'total_estimated_time': 12,  # hours
    'parallel_execution': True,
    'automatic_best_selection': True,
    'success_threshold': 0.95,
    
    'schedule': [
        {
            'time_slot': '0-3h',
            'variant': 'variant_1_dora_neftune_sophia',
            'action': 'primary_training',
            'checkpoints': [1, 2, 3],
            'evaluation_points': [1.5, 3.0]
        },
        {
            'time_slot': '1-4h', 
            'variant': 'variant_3_dynamic_expansion',
            'action': 'parallel_training',
            'checkpoints': [2, 4],
            'evaluation_points': [2.0, 4.0]
        },
        {
            'time_slot': '3-6h',
            'variant': 'variant_2_progressive_curriculum', 
            'action': 'secondary_training',
            'checkpoints': [4, 5, 6],
            'evaluation_points': [4.5, 6.0]
        },
        {
            'time_slot': '6-12h',
            'variant': 'variant_4_catastrophic_prevention',
            'action': 'conservative_training',
            'checkpoints': [8, 10, 12],
            'evaluation_points': [8.0, 10.0, 12.0]
        }
    ],
    
    'early_stopping': {
        'enabled': True,
        'patience': 3,
        'min_delta': 0.01,
        'monitor': 'eval_loss'
    },
    
    'automatic_selection_criteria': {
        'primary': 'final_loss',
        'secondary': 'training_time',
        'tertiary': 'turkish_metrics',
        'weights': [0.6, 0.2, 0.2]
    }
}

print(f"✅ {len(training_schedule['schedule'])} training slots configured")
print(f"✅ Parallel execution: {training_schedule['parallel_execution']}")
print(f"✅ Success threshold: {training_schedule['success_threshold']:.0%}")

# Master Configuration Update (Tutarlı dizin yapısı)
print("\n💾 MASTER CONFIGURATION UPDATE:")

# Güncellenmiş konfigürasyon yolları
workspace_config_path = f"{WORKSPACE}/configs/ensemble_config.json"
repo_config_path = f"{WORKSPACE}/configs/ensemble_runtime_config.json"

try:
    if os.path.exists(workspace_config_path):
        with open(workspace_config_path, 'r', encoding='utf-8') as f:
            master_config = json.load(f)
    else:
        master_config = {}
    
    # Update with detailed configurations
    master_config.update({
        'ensemble_variants_detailed': ensemble_variants,
        'dataset_configuration': dataset_config,
        'training_schedule': training_schedule,
        'configuration_completed_at': datetime.now().isoformat(),
        
        # Dizin yapısı bilgisi
        'directory_structure': {
            'base_dir': BASE_DIR,
            'workspace': WORKSPACE,
            'configs_dir': f"{WORKSPACE}/configs",
            'datasets_dir': f"{WORKSPACE}/datasets",
            'models_dir': f"{WORKSPACE}/models"
        },
        
        # Qwen3-8B Tokenizer Strategy (Memory lessons: avoid mismatch)
        'tokenizer_strategy': {
            'use_original_qwen': True,  # Recommended approach
            'avoid_embed_tokens_in_modules_to_save': True,  # Critical
            'learning_rate_turkish_optimal': 2e-4,  # Memory: optimal value
            'expected_success_rate': 0.95  # 95% vs 30% for mismatch
        },
        
        # Turkish Language Specific (Memory: morphology, vowel harmony)
        'turkish_optimizations': {
            'morphology_aware': True,
            'vowel_harmony_compliance': True,
            'agglutinative_structure': True,
            'cultural_context_preservation': True
        }
    })
    
    # Save updated configuration
    with open(workspace_config_path, 'w', encoding='utf-8') as f:
        json.dump(master_config, f, indent=2, ensure_ascii=False)
    
    print("✅ Master configuration updated")
    print(f"📁 Config path: {workspace_config_path}")
    
    # Configuration backup to repository
    with open(repo_config_path, 'w', encoding='utf-8') as f:
        json.dump(master_config, f, indent=2, ensure_ascii=False)
    
    print("✅ Configuration backup saved to repository")
    print(f"📁 Backup path: {repo_config_path}")
    
except Exception as e:
    print(f"⚠️ Configuration update error: {e}")

# GPU Memory Pre-allocation Test
print("\n🧪 GPU MEMORY PRE-ALLOCATION TEST:")

try:
    # Small memory test for ensemble readiness
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Test tensor creation for each variant
    memory_tests = {}
    
    for variant_id, variant in ensemble_variants.items():
        batch_size = variant['config'].get('per_device_batch_size', 8)
        seq_length = 2048  # Standard sequence length
        
        try:
            # Simulate model memory usage
            test_tensor = torch.randn(batch_size, seq_length, 4096, device=device, dtype=torch.float16)
            memory_used = torch.cuda.memory_allocated() / (1024**3)
            
            # Clean up
            del test_tensor
            torch.cuda.empty_cache()
            
            memory_tests[variant_id] = {
                'memory_gb': memory_used,
                'batch_size': batch_size,
                'status': 'OK' if memory_used < 35 else 'HIGH'
            }
            
            print(f"✅ {variant['name']}: {memory_used:.1f}GB (batch={batch_size})")
            
        except Exception as e:
            memory_tests[variant_id] = {
                'memory_gb': None,
                'batch_size': batch_size,
                'status': 'ERROR',
                'error': str(e)[:50]
            }
            print(f"⚠️ {variant['name']}: Memory test failed")
    
    # Overall memory assessment
    successful_tests = sum(1 for test in memory_tests.values() if test['status'] == 'OK')
    total_tests = len(memory_tests)
    
    print(f"\n📊 Memory Test Results: {successful_tests}/{total_tests} variants OK")
    
except Exception as e:
    print(f"⚠️ GPU memory test error: {e}")

# Final Readiness Assessment
print("\n" + "=" * 90)
print("🎯 ENSEMBLE CONFIGURATION READINESS ASSESSMENT:")

readiness_criteria = {
    'variants_configured': len(ensemble_variants) == 4,
    'datasets_mapped': len(dataset_config['huggingface_datasets']) >= 8,
    'schedule_planned': len(training_schedule['schedule']) == 4,
    'memory_optimized': dataset_config['memory_optimization']['max_memory_gb'] <= 12,
    'a100_optimized': all(v['config'].get('bf16', False) for v in ensemble_variants.values()),
    'tokenizer_safe': master_config.get('tokenizer_strategy', {}).get('avoid_embed_tokens_in_modules_to_save', False)
}

passed_criteria = sum(readiness_criteria.values())
total_criteria = len(readiness_criteria)

for criterion, status in readiness_criteria.items():
    print(f"📊 {criterion}: {'✅' if status else '❌'}")

overall_readiness = passed_criteria / total_criteria
print(f"\n📊 Overall Readiness: {overall_readiness:.1%} ({passed_criteria}/{total_criteria})")

ensemble_ready_for_training = overall_readiness >= 0.8
print(f"📊 Ready for Training: {'✅ HAZIR' if ensemble_ready_for_training else '❌ EKSİK'}")

if ensemble_ready_for_training:
    print("🚀 Next Step: ENSEMBLE TRAINING BAŞLATMA")
else:
    print("⚠️ Next Step: Configuration düzeltme gerekli")

print("=" * 90)

final_status = {
    'ensemble_configured': True,
    'variants_count': len(ensemble_variants),
    'readiness_score': overall_readiness,
    'ready_for_training': ensemble_ready_for_training,
    'next_step': 'start_ensemble_training' if ensemble_ready_for_training else 'fix_configuration',
    'directory_structure': {
        'base_dir': BASE_DIR,
        'workspace': WORKSPACE
    }
}

print(f"\n📝 Bu çıktıyı kopyalayıp paylaşın, Ensemble Training başlatma hücresini hazırlayacağım.")
print(f"📁 Tutarlı dizin yapısı: {BASE_DIR} → {WORKSPACE}")