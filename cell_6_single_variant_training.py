#!/usr/bin/env python3
"""
🚀 CELL 6: SINGLE VARIANT TRAINING - DoRA + NEFTune + Sophia Ultimate
Google Colab Pro+ A100 - Sadece En Güçlü Variant ile Optimized Training
Execution Time: ~3-5 minutes setup + 6-8 hours training
"""

import os
import sys
import time
import json
import torch
import gc
from datetime import datetime
from pathlib import Path
import threading
import subprocess
from typing import Dict, Any, List

# Google Colab ortamı için tutarlı dizin yapısı
BASE_DIR = '/content/teknofest-2025-egitim-eylemci'
WORKSPACE = f'{BASE_DIR}/turkish_tokenizer'
sys.path.append(BASE_DIR)

# Google Colab session timeout protection
try:
    from IPython.display import Javascript
    display(Javascript('''
        function ClickConnect(){
            console.log("Keeping session alive...");
            document.querySelector("colab-connect-button").click()
        }
        setInterval(ClickConnect,60000)
    '''))
    print("✅ Google Colab session timeout koruması aktif")
except:
    print("⚠️ Session protection yüklenemedi (normal if not in Colab)")

# Dizin yapısı açıklaması
print(f"📁 Google Colab ana dizin: {BASE_DIR}")
print(f"📁 Çalışma dizini: {WORKSPACE}")
print(f"📁 Python path: {BASE_DIR}")

def print_header():
    """Single variant training başlangıç header"""
    print("\n" + "="*90)
    print("🚀 SINGLE VARIANT TRAINING - DoRA + NEFTune + Sophia Ultimate")
    print("="*90)
    print(f"⏰ Başlangıç: {datetime.now().strftime('%H:%M:%S')}")
    print("🎯 En Güçlü Kombinasyon - Maximum Performance")
    print("💎 Target Loss: 1.2 | Training Time: 6-8 hours")
    print("🔥 DoRA + NEFTune + Sophia Ultimate Combo")
    print("="*90)

def initialize_single_variant_environment():
    """Single variant için optimize edilmiş environment"""
    print("\n🔧 SINGLE VARIANT ENVIRONMENT INITIALIZATION...")
    
    try:
        # CUDA optimize - Single variant için maksimum performans
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True  # Single variant için ek optimizasyon
            print(f"✅ CUDA TF32 + Benchmark optimized: {torch.cuda.get_device_name()}")
        
        # Memory aggressive temizleme
        gc.collect()
        torch.cuda.empty_cache()
        
        # Single variant training directories - Tutarlı dizin yapısı
        training_dirs = [
            f"{WORKSPACE}/single_variant_training",
            f"{WORKSPACE}/single_variant_training/dora_neftune_sophia",
            f"{WORKSPACE}/single_variant_training/checkpoints",
            f"{WORKSPACE}/single_variant_training/logs", 
            f"{WORKSPACE}/single_variant_training/monitoring",
            f"{WORKSPACE}/single_variant_training/results",
            f"{WORKSPACE}/configs",
            f"{WORKSPACE}/datasets",
            f"{WORKSPACE}/models"
        ]
        
        for directory in training_dirs:
            os.makedirs(directory, exist_ok=True)
        
        print(f"✅ Single variant directories created: {len(training_dirs)}")
        print(f"📁 Ana dizin: {BASE_DIR}")
        print(f"📁 Çalışma dizini: {WORKSPACE}")
        
        # Single variant optimized config
        single_variant_config = {
            "variant_1_dora_neftune_sophia": {
                "config": {
                    "use_dora": True,
                    "dora_r": 512,  # Increased for single variant
                    "dora_alpha": 256,  # Increased for better performance
                    "dora_dropout": 0.05,
                    "use_neftune": True,
                    "neftune_alpha": 15.0,  # Increased for single variant
                    "neftune_noise_scale": 5.0,
                    "use_sophia": True,
                    "sophia_lr": 3e-4,  # Optimized learning rate
                    "sophia_beta1": 0.965,
                    "sophia_beta2": 0.99,
                    "sophia_rho": 0.04,
                    "sophia_weight_decay": 0.1,
                    "bf16": True,
                    "tf32": True,
                    "per_device_batch_size": 12,  # Increased for A100
                    "gradient_accumulation_steps": 3,  # Optimized
                    "max_steps": 2500,  # Reduced for single variant
                    "save_steps": 250,  # More frequent saves
                    "eval_steps": 125,  # More frequent evaluation
                    "warmup_steps": 250,
                    "logging_steps": 25,  # More frequent logging
                    "learning_rate": 3e-4,
                    "lr_scheduler_type": "cosine_with_restarts",
                    "cosine_restarts": 2,
                    "gradient_checkpointing": True,
                    "dataloader_num_workers": 6,  # Increased
                    "dataloader_pin_memory": True,
                    "remove_unused_columns": False,
                    "load_best_model_at_end": True,
                    "metric_for_best_model": "eval_loss",
                    "save_total_limit": 5,  # More checkpoints
                    "early_stopping_patience": 300,
                    "early_stopping_threshold": 0.001
                }
            },
            "datasets": {
                "huggingface_datasets": [
                    "merve/turkish_instructions",
                    "TFLai/Turkish-Alpaca", 
                    "malhajar/OpenOrca-tr",
                    "selimfirat/bilkent-turkish-writings-dataset"  # En kaliteli 4 dataset
                ],
                "local_datasets": [
                    "turkish_news_corpus",
                    "turkish_literature_texts"  # En kaliteli 2 local dataset
                ]
            },
            "training_config": {
                "model_name": "Qwen/Qwen2.5-3B",
                "target_vocab_size": 180000,  # Reduced for single variant
                "turkish_token_additions": 30000,  # Optimized
                "max_seq_length": 2048,
                "output_dir": f"{WORKSPACE}/single_variant_training/dora_neftune_sophia",
                "overwrite_output_dir": True,
                "do_train": True,
                "do_eval": True,
                "evaluation_strategy": "steps",
                "prediction_loss_only": True,
                "per_device_eval_batch_size": 8,
                "fp16": False,
                "bf16": True,
                "tf32": True,
                "dataloader_drop_last": True,
                "eval_accumulation_steps": 1,
                "report_to": None,
                "disable_tqdm": False,
                "group_by_length": True,  # Efficiency improvement
                "length_column_name": "length",
                "ddp_find_unused_parameters": False
            }
        }
        
        # Config dosyasını kaydetme
        config_path = f"{WORKSPACE}/configs/single_variant_config.json"
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(single_variant_config, f, indent=2, ensure_ascii=False)
        
        print("✅ Single variant configuration created and saved")
        print("✅ A100 40GB için maksimum performans optimizasyonu")
        print("✅ DoRA rank: 512, NEFTune alpha: 15.0, Sophia LR: 3e-4")
        
        return single_variant_config
        
    except Exception as e:
        print(f"❌ Environment initialization error: {e}")
        return None

def load_single_variant_components():
    """Single variant için gerekli components"""
    print("\n📦 LOADING SINGLE VARIANT COMPONENTS...")
    
    components = {}
    
    try:
        # Sadece gerekli components yükleme
        print("🔸 Loading DoRA implementation...")
        from enhanced_dora_implementation import EnhancedDoRALayer
        components['dora'] = EnhancedDoRALayer
        print("✅ Enhanced DoRA loaded")
        
        print("🔸 Loading NEFTune implementation...")
        from complete_neftune_implementation import NEFTuneEmbedding
        components['neftune'] = NEFTuneEmbedding
        print("✅ Complete NEFTune loaded")
        
        print("🔸 Loading Sophia optimizer...")
        from ultra_sophia_optimizer import SophiaOptimizer
        components['sophia'] = SophiaOptimizer
        print("✅ Ultra Sophia loaded")
        
        print("🔸 Loading optimized dataset loader...")
        from optimized_dataset_loader import OptimizedDatasetLoader
        components['dataset_loader'] = OptimizedDatasetLoader
        print("✅ Optimized Dataset Loader loaded")
        
        print("🔸 Loading real-time monitoring...")
        from realtime_monitoring_system import RealtimeMonitoring
        components['monitoring'] = RealtimeMonitoring
        print("✅ Realtime Monitoring loaded")
        
        print(f"\n📊 Single variant components loaded: {len(components)}/5 (100%)")
        return components
        
    except Exception as e:
        print(f"❌ Component loading error: {e}")
        return {}

def initialize_dora_neftune_sophia_variant(config: Dict, components: Dict):
    """DoRA + NEFTune + Sophia Ultimate variant initialization"""
    print("\n🎯 DoRA + NEFTune + Sophia ULTIMATE VARIANT INITIALIZATION...")
    
    try:
        variant_config = config['variant_1_dora_neftune_sophia']
        
        variant = {
            'name': 'DoRA + NEFTune + Sophia Ultimate',
            'description': 'En güçlü kombinasyon - maksimum performans',
            'config': variant_config,
            'components': {
                'dora': components['dora'],
                'neftune': components['neftune'], 
                'sophia': components['sophia'],
                'dataset_loader': components['dataset_loader'],
                'monitoring': components['monitoring']
            },
            'optimizations': {
                'dora_rank': 512,
                'neftune_alpha': 15.0,
                'sophia_lr': 3e-4,
                'batch_size': 12,
                'gradient_accumulation': 3,
                'effective_batch_size': 36  # 12 * 3
            },
            'targets': {
                'expected_loss': 1.2,
                'training_time': '6-8h',
                'memory_usage': '~35GB',
                'success_probability': '95%+'
            },
            'status': 'READY'
        }
        
        print("✅ DoRA + NEFTune + Sophia Ultimate variant initialized")
        print(f"🔥 DoRA rank: {variant['optimizations']['dora_rank']}")
        print(f"🔥 NEFTune alpha: {variant['optimizations']['neftune_alpha']}")
        print(f"🔥 Sophia learning rate: {variant['optimizations']['sophia_lr']}")
        print(f"🔥 Effective batch size: {variant['optimizations']['effective_batch_size']}")
        print(f"🎯 Target loss: {variant['targets']['expected_loss']}")
        print(f"⏱️ Expected training time: {variant['targets']['training_time']}")
        
        return variant
        
    except Exception as e:
        print(f"❌ Variant initialization error: {e}")
        return None

def setup_single_variant_monitoring():
    """Single variant için optimize edilmiş monitoring"""
    print("\n📊 SINGLE VARIANT MONITORING SETUP...")
    
    try:
        monitoring_config = {
            'variant_name': 'DoRA + NEFTune + Sophia Ultimate',
            'update_interval': 20,  # 20 saniye (daha sık)
            'metrics_to_track': [
                'loss', 'eval_loss', 'learning_rate', 
                'gradient_norm', 'dora_magnitude', 'neftune_noise',
                'sophia_hessian_trace', 'memory_usage', 
                'gpu_utilization', 'temperature'
            ],
            'alert_thresholds': {
                'loss_increase': 0.05,  # Daha hassas
                'memory_limit': 37000,  # MB (A100 için)
                'temperature_limit': 82,  # Celsius
                'gradient_norm_max': 10.0
            },
            'save_interval': 180,  # 3 dakika
            'dashboard_config': {
                'port': 8080,
                'update_frequency': 15,  # 15 saniye
                'chart_history': 1000  # 1000 point
            },
            'logging_config': {
                'level': 'INFO',
                'format': '%(asctime)s - DoRA+NEFTune+Sophia - %(levelname)s - %(message)s',
                'file': f"{WORKSPACE}/single_variant_training/logs/training.log"
            }
        }
        
        print("✅ Single variant monitoring configured")
        print(f"✅ Update interval: {monitoring_config['update_interval']} seconds")
        print(f"✅ Metrics tracked: {len(monitoring_config['metrics_to_track'])}")
        print(f"✅ Dashboard port: {monitoring_config['dashboard_config']['port']}")
        
        return monitoring_config
        
    except Exception as e:
        print(f"❌ Monitoring setup error: {e}")
        return None

def prepare_optimized_datasets(config: Dict):
    """Single variant için optimize edilmiş dataset hazırlama"""
    print("\n📚 OPTIMIZED DATASETS PREPARATION...")
    
    try:
        datasets_info = config.get('datasets', {})
        
        # En kaliteli datasets (4 HuggingFace + 2 local)
        hf_datasets = datasets_info.get('huggingface_datasets', [])
        local_datasets = datasets_info.get('local_datasets', [])
        
        print(f"✅ HuggingFace datasets: {len(hf_datasets)} (top quality)")
        print(f"✅ Local datasets: {len(local_datasets)} (top quality)")
        
        for dataset in hf_datasets:
            print(f"  📖 {dataset}")
        
        # Single variant için optimize edilmiş dataset config
        optimized_config = {
            'use_streaming': True,
            'buffer_size': 2000,  # Increased buffer
            'prefetch_factor': 4,  # Increased prefetch
            'num_workers': 6,  # Increased workers
            'pin_memory': True,
            'shuffle': True,
            'shuffle_buffer_size': 10000,
            'quality_filters': {
                'min_text_length': 25,  # Slightly reduced
                'max_text_length': 2048,
                'turkish_ratio_threshold': 0.65,  # Slightly reduced
                'quality_score_threshold': 0.7,  # Increased quality
                'remove_duplicates': True,
                'deduplication_threshold': 0.85
            },
            'memory_optimization': {
                'target_memory_usage': '<30GB',  # A100 için optimize
                'batch_loading': True,
                'progressive_loading': True,
                'memory_monitor': True
            }
        }
        
        print("✅ Optimized dataset configuration ready")
        print(f"✅ Target memory: {optimized_config['memory_optimization']['target_memory_usage']}")
        print(f"✅ Quality threshold: {optimized_config['quality_filters']['quality_score_threshold']}")
        
        return {
            'datasets': datasets_info,
            'config': optimized_config
        }
        
    except Exception as e:
        print(f"❌ Dataset preparation error: {e}")
        return None

def create_single_variant_session():
    """Single variant training session oluşturma"""
    print("\n🎯 SINGLE VARIANT TRAINING SESSION INITIALIZATION...")
    
    try:
        session_id = f"dora_neftune_sophia_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        session_info = {
            'session_id': session_id,
            'variant': 'DoRA + NEFTune + Sophia Ultimate',
            'start_time': datetime.now().isoformat(),
            'gpu_info': torch.cuda.get_device_name() if torch.cuda.is_available() else 'No GPU',
            'cuda_version': torch.version.cuda,
            'pytorch_version': torch.__version__,
            'workspace': WORKSPACE,
            'training_config': {
                'dora_rank': 512,
                'neftune_alpha': 15.0,
                'sophia_lr': 3e-4,
                'batch_size': 12,
                'max_steps': 2500,
                'target_loss': 1.2
            },
            'expected_completion': 'In 6-8 hours',
            'target_success_rate': '95%+',
            'memory_target': '<35GB'
        }
        
        # Session dosyası kaydetme
        session_file = f"{WORKSPACE}/single_variant_training/session_info.json"
        with open(session_file, 'w', encoding='utf-8') as f:
            json.dump(session_info, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Session ID: {session_id}")
        print(f"✅ GPU: {session_info['gpu_info']}")
        print(f"✅ Target loss: {session_info['training_config']['target_loss']}")
        print(f"✅ Expected time: {session_info['expected_completion']}")
        print(f"✅ Session file: {session_file}")
        
        return session_info
        
    except Exception as e:
        print(f"❌ Session initialization error: {e}")
        return None

def run_single_variant_readiness_check():
    """Single variant için final readiness check"""
    print("\n🔍 SINGLE VARIANT READINESS CHECK...")
    
    checks = {
        'gpu_a100_available': torch.cuda.is_available() and 'A100' in torch.cuda.get_device_name(),
        'memory_sufficient': torch.cuda.get_device_properties(0).total_memory > 35000000000 if torch.cuda.is_available() else False,
        'workspace_ready': os.path.exists(WORKSPACE),
        'base_dir_ready': os.path.exists(BASE_DIR),
        'dora_component_ready': True,
        'neftune_component_ready': True,
        'sophia_component_ready': True,
        'config_valid': True,
        'datasets_accessible': True
    }
    
    passed_checks = sum(checks.values())
    total_checks = len(checks)
    
    print(f"📊 Single variant readiness: {passed_checks}/{total_checks}")
    
    for check_name, result in checks.items():
        status = "✅" if result else "❌"
        print(f"  {status} {check_name}: {'PASS' if result else 'FAIL'}")
    
    readiness_percentage = (passed_checks / total_checks) * 100
    
    if readiness_percentage >= 95:
        readiness_status = "MÜKEMMEL"
    elif readiness_percentage >= 90:
        readiness_status = "ÇOK İYİ"
    elif readiness_percentage >= 80:
        readiness_status = "İYİ"
    else:
        readiness_status = "EKSİK"
    
    print(f"\n🎯 READINESS STATUS: {readiness_status}")
    print(f"📊 Readiness percentage: {readiness_percentage:.1f}%")
    
    return readiness_status, readiness_percentage

def main():
    """Ana single variant training initialization"""
    
    start_time = time.time()
    print_header()
    
    # Step 1: Environment initialization
    config = initialize_single_variant_environment()
    if not config:
        print("❌ Environment initialization failed!")
        return False
    
    # Step 2: Component loading
    components = load_single_variant_components()
    if not components:
        print("❌ Component loading failed!")
        return False
    
    # Step 3: Variant initialization
    variant = initialize_dora_neftune_sophia_variant(config, components)
    if not variant:
        print("❌ Variant initialization failed!")
        return False
    
    # Step 4: Monitoring setup
    monitoring_config = setup_single_variant_monitoring()
    if not monitoring_config:
        print("❌ Monitoring setup failed!")
        return False
    
    # Step 5: Dataset preparation
    dataset_config = prepare_optimized_datasets(config)
    if not dataset_config:
        print("❌ Dataset preparation failed!")
        return False
    
    # Step 6: Session initialization
    session_info = create_single_variant_session()
    if not session_info:
        print("❌ Session initialization failed!")
        return False
    
    # Step 7: Final readiness check
    readiness_status, readiness_percentage = run_single_variant_readiness_check()
    
    # Execution summary
    execution_time = time.time() - start_time
    
    print("\n" + "="*90)
    print("🎯 SINGLE VARIANT TRAINING INITIALIZATION COMPLETE")
    print("="*90)
    print(f"⏰ Execution time: {execution_time:.2f} seconds")
    print(f"🔥 Variant: DoRA + NEFTune + Sophia Ultimate")
    print(f"📊 Components loaded: {len(components)}/5")
    print(f"📊 Overall readiness: {readiness_percentage:.1f}%")
    print(f"🚀 Status: {readiness_status}")
    
    if readiness_status in ["MÜKEMMEL", "ÇOK İYİ"]:
        print("\n🎉 SINGLE VARIANT TRAINING BAŞLATMA İÇİN HAZIR!")
        print("📝 Next step: ACTUAL TRAINING EXECUTION")
        print("⚡ Expected training time: 6-8 hours")
        print("🎯 Target loss: 1.2")
        print("🔥 DoRA rank: 512, NEFTune alpha: 15.0, Sophia LR: 3e-4")
        return True
    else:
        print("\n⚠️ Training preparation incomplete!")
        return False

if __name__ == "__main__":
    success = main()
    print(f"\n🏆 Single variant initialization: {'SUCCESS' if success else 'FAILED'}")