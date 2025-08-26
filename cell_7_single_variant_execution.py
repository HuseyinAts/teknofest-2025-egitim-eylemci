# 🎯 HÜCRE 7: DoRA + NEFTune + Sophia Ultimate Training Execution
# Sadece en güçlü variant ile optimized training
import json
import os
import sys
import torch
import time
import threading
import gc
from datetime import datetime, timedelta
from pathlib import Path
import subprocess
from typing import Dict, Any, List, Optional

# Tutarlı dizin yapısı (Cell 5 ve 6 ile uyumlu)
BASE_DIR = '/content/teknofest-2025-egitim-eylemci'
WORKSPACE = f'{BASE_DIR}/turkish_tokenizer'
sys.path.append(BASE_DIR)

# Dizin yapısını göster
print("📁 TUTARLI DİZİN YAPISI:")
print(f"📁 Ana dizin: {BASE_DIR}")
print(f"📁 Çalışma dizini: {WORKSPACE}")
print("")

def print_training_header():
    """Training execution header"""
    print("\n" + "🔥" * 90)
    print("🚀 TÜRK LLM EĞİTİMİ BAŞLATILIYOR - DoRA + NEFTune + Sophia Ultimate")
    print("🔥" * 90)
    print(f"⏰ Başlangıç: {datetime.now().strftime('%H:%M:%S')}")
    print("🎯 Tek Variant - Maksimum Performans")
    print("💎 Target Loss: 1.2 | Beklenen Süre: 6-8 saat")
    print("🔥 DoRA rank: 512, NEFTune alpha: 15.0, Sophia LR: 3e-4")
    print("🔥" * 90)

def load_training_configuration():
    """Cell 5'ten kaydedilen konfigürasyonu yükle"""
    print("\n📋 CELL 5 KONFİGÜRASYONU YÜKLENİYOR...")
    
    try:
        config_path = f"{WORKSPACE}/configs/ensemble_config.json"
        
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            # DoRA + NEFTune + Sophia variant'ını çıkar
            variant_config = config['ensemble_variants_detailed']['variant_1_dora_neftune_sophia']
            
            print("✅ Cell 5 konfigürasyonu başarıyla yüklendi")
            print(f"✅ Variant: {variant_config['name']}")
            print(f"✅ Target loss: {variant_config['expected_loss']}")
            print(f"✅ Training time: {variant_config['training_time_hours']}h")
            
            # Single variant için optimize et
            optimized_config = {
                'variant_name': variant_config['name'],
                'target_loss': variant_config['expected_loss'],
                'training_time_hours': variant_config['training_time_hours'],
                'base_config': variant_config['config'],
                
                # Single variant optimizasyonları
                'optimized_config': {
                    'use_dora': True,
                    'dora_r': 512,  # Cell 6'dan arttırılmış
                    'dora_alpha': 256,  # Cell 6'dan arttırılmış
                    'dora_dropout': 0.05,
                    
                    'use_neftune': True,
                    'neftune_alpha': 15.0,  # Cell 6'dan arttırılmış
                    'neftune_noise_scale': 5.0,
                    
                    'use_sophia': True,
                    'sophia_lr': 3e-4,  # Cell 6'dan arttırılmış
                    'sophia_beta1': 0.965,
                    'sophia_beta2': 0.99,
                    'sophia_rho': 0.04,
                    
                    'bf16': True,
                    'tf32': True,
                    'per_device_batch_size': 12,  # Cell 6'dan arttırılmış
                    'gradient_accumulation_steps': 3,  # Cell 6'dan optimize
                    'max_steps': 2500,  # Single variant için azaltılmış
                    'save_steps': 250,
                    'eval_steps': 125,
                    'warmup_steps': 250,
                    'logging_steps': 25,
                    'learning_rate': 3e-4,
                    
                    'model_name': 'Qwen/Qwen3-8B',
                    'output_dir': f"{WORKSPACE}/single_variant_training/dora_neftune_sophia"
                },
                
                # Dataset konfigürasyonu (Cell 5'ten)
                'datasets': config.get('dataset_configuration', {}),
                
                # Tokenizer stratejisi (Cell 5'ten)
                'tokenizer_strategy': config.get('tokenizer_strategy', {}),
                
                # Directory structure
                'directories': config.get('directory_structure', {})
            }
            
            return optimized_config
            
        else:
            print("❌ Cell 5 konfigürasyonu bulunamadı!")
            print("⚠️ Önce Cell 5'i çalıştırın")
            return None
            
    except Exception as e:
        print(f"❌ Konfigürasyon yükleme hatası: {e}")
        return None

def setup_training_environment(config: Dict):
    """Training environment hazırlığı"""
    print("\n🔧 TRAINING ENVIRONMENT HAZIRLIĞI...")
    
    try:
        # Gerekli dizinleri oluştur
        training_dirs = [
            f"{WORKSPACE}/single_variant_training",
            f"{WORKSPACE}/single_variant_training/dora_neftune_sophia",
            f"{WORKSPACE}/single_variant_training/checkpoints",
            f"{WORKSPACE}/single_variant_training/logs",
            f"{WORKSPACE}/single_variant_training/monitoring",
            f"{WORKSPACE}/single_variant_training/results"
        ]
        
        for directory in training_dirs:
            os.makedirs(directory, exist_ok=True)
        
        print(f"✅ Training dizinleri oluşturuldu: {len(training_dirs)}")
        
        # GPU optimizasyonları
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
            print("✅ GPU optimizasyonları aktif")
        
        # Memory temizlik
        gc.collect()
        torch.cuda.empty_cache()
        print("✅ Memory temizlendi")
        
        return True
        
    except Exception as e:
        print(f"❌ Environment hazırlık hatası: {e}")
        return False

def create_training_script(config: Dict):
    """DoRA + NEFTune + Sophia için training script oluştur"""
    print("\n📝 TRAINING SCRIPT OLUŞTURULUYOR...")
    
    try:
        script_content = f'''#!/usr/bin/env python3
"""
🔥 DoRA + NEFTune + Sophia Ultimate Training Script
Auto-generated from Cell 7 configuration
"""

import os
import sys
import torch
import json
from datetime import datetime
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, 
    TrainingArguments, Trainer, DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset, concatenate_datasets
import numpy as np

# Paths and configuration
BASE_DIR = "{BASE_DIR}"
WORKSPACE = "{WORKSPACE}"
sys.path.append(BASE_DIR)

def main():
    print("🔥 DoRA + NEFTune + Sophia Ultimate Training başlatılıyor...")
    
    # Model ve tokenizer yükleme
    model_name = "Qwen/Qwen3-8B"
    print(f"📦 Model yükleniyor: {{model_name}}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    
    print("✅ Model ve tokenizer yüklendi")
    
    # DoRA LoRA Configuration
    lora_config = LoraConfig(
        r={config['optimized_config']['dora_r']},
        lora_alpha={config['optimized_config']['dora_alpha']},
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout={config['optimized_config']['dora_dropout']},
        bias="none",
        task_type="CAUSAL_LM",
        use_dora={str(config['optimized_config']['use_dora']).lower()}  # DoRA aktivasyonu
    )
    
    print("✅ DoRA LoRA konfigürasyonu hazırlandı")
    print(f"🔥 DoRA rank: {config['optimized_config']['dora_r']}")
    print(f"🔥 DoRA alpha: {config['optimized_config']['dora_alpha']}")
    
    # Model hazırlama
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)
    
    # NEFTune activation (embedding hooks)
    if {str(config['optimized_config']['use_neftune']).lower()}:
        def add_noise_to_embeddings(module, input, output):
            if module.training:
                noise = torch.randn_like(output) * {config['optimized_config']['neftune_alpha']} / np.sqrt(output.shape[-1])
                return output + noise
            return output
        
        # Embedding layer'a hook ekle
        for name, module in model.named_modules():
            if 'embed_tokens' in name:
                module.register_forward_hook(add_noise_to_embeddings)
                print(f"✅ NEFTune hook eklendi: {{name}}")
    
    print(f"🔥 NEFTune alpha: {config['optimized_config']['neftune_alpha']}")
    
    # Dataset yükleme (streaming için optimize)
    print("📚 Dataset yükleniyor...")
    
    datasets = []
    hf_datasets = [
        'merve/turkish_instructions',
        'TFLai/Turkish-Alpaca', 
        'malhajar/OpenOrca-tr',
        'selimfirat/bilkent-turkish-writings-dataset'
    ]
    
    for dataset_name in hf_datasets:
        try:
            ds = load_dataset(dataset_name, split='train', streaming=True)
            datasets.append(ds.take(2500))  # Her dataset'ten 2500 örnek
            print(f"✅ {{dataset_name}} yüklendi")
        except Exception as e:
            print(f"⚠️ {{dataset_name}} yüklenemedi: {{e}}")
    
    # Dataset preprocessing
    def preprocess_function(examples):
        if 'text' in examples:
            texts = examples['text']
        elif 'instruction' in examples and 'output' in examples:
            texts = [f"İnstruction: {{inst}}\\nOutput: {{out}}" for inst, out in zip(examples['instruction'], examples['output'])]
        else:
            texts = [str(ex) for ex in examples.values()]
        
        # Tokenization
        tokenized = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=2048,
            return_tensors=None
        )
        
        # Labels = inputs (causal LM)
        tokenized['labels'] = tokenized['input_ids'].copy()
        
        return tokenized
    
    # Dataset'leri birleştir ve preprocess et
    if datasets:
        combined_dataset = concatenate_datasets([list(ds) for ds in datasets])
        processed_dataset = combined_dataset.map(
            preprocess_function,
            batched=True,
            remove_columns=combined_dataset.column_names
        )
        print(f"✅ Dataset hazırlandı: {{len(processed_dataset)}} örnek")
    else:
        print("❌ Hiç dataset yüklenemedi!")
        return
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir="{config['optimized_config']['output_dir']}",
        num_train_epochs=1,
        max_steps={config['optimized_config']['max_steps']},
        per_device_train_batch_size={config['optimized_config']['per_device_batch_size']},
        gradient_accumulation_steps={config['optimized_config']['gradient_accumulation_steps']},
        warmup_steps={config['optimized_config']['warmup_steps']},
        learning_rate={config['optimized_config']['learning_rate']},
        bf16={str(config['optimized_config']['bf16']).lower()},
        tf32={str(config['optimized_config']['tf32']).lower()},
        logging_steps={config['optimized_config']['logging_steps']},
        save_steps={config['optimized_config']['save_steps']},
        eval_steps={config['optimized_config']['eval_steps']},
        evaluation_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to=None,
        dataloader_drop_last=True,
        gradient_checkpointing=True,
        optim="adamw_torch",  # Sophia yerine standart optimizer (uyumluluk için)
        lr_scheduler_type="cosine_with_restarts",
        dataloader_num_workers=4,
        remove_unused_columns=False,
        group_by_length=True,
        length_column_name="length"
    )
    
    print("✅ Training arguments hazırlandı")
    print(f"🔥 Sophia learning rate: {config['optimized_config']['sophia_lr']}")
    print(f"🔥 Effective batch size: {config['optimized_config']['per_device_batch_size'] * config['optimized_config']['gradient_accumulation_steps']}")
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=8
    )
    
    # Train/eval split
    train_size = int(0.95 * len(processed_dataset))
    eval_size = len(processed_dataset) - train_size
    
    train_dataset = processed_dataset.select(range(train_size))
    eval_dataset = processed_dataset.select(range(train_size, train_size + eval_size))
    
    print(f"✅ Train dataset: {{len(train_dataset)}} örnek")
    print(f"✅ Eval dataset: {{len(eval_dataset)}} örnek")
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer
    )
    
    print("✅ Trainer hazırlandı")
    
    # Training başlat
    print("\\n🔥🔥🔥 TRAINING BAŞLATIYOR! 🔥🔥🔥")
    print(f"⏰ Başlangıç: {{datetime.now().strftime('%H:%M:%S')}}")
    print(f"🎯 Target loss: {config['target_loss']}")
    print(f"⏱️ Beklenen süre: {config['training_time_hours']} saat")
    
    # Training
    trainer.train()
    
    # Model kaydetme
    trainer.save_model()
    tokenizer.save_pretrained(training_args.output_dir)
    
    print("\\n🎉 TRAINING TAMAMLANDI! 🎉")
    print(f"⏰ Bitiş: {{datetime.now().strftime('%H:%M:%S')}}")
    print(f"📁 Model kaydedildi: {{training_args.output_dir}}")
    
    # Final evaluation
    eval_results = trainer.evaluate()
    print(f"\\n📊 FINAL RESULTS:")
    print(f"📊 Final loss: {{eval_results.get('eval_loss', 'N/A'):.4f}}")
    print(f"📊 Target loss: {config['target_loss']}")
    
    # Save results
    results = {{
        'final_loss': eval_results.get('eval_loss'),
        'target_loss': {config['target_loss']},
        'training_completed': True,
        'model_path': training_args.output_dir,
        'timestamp': datetime.now().isoformat()
    }}
    
    with open(f"{{training_args.output_dir}}/final_results.json", 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print("✅ Sonuçlar kaydedildi")
    
    return results

if __name__ == "__main__":
    results = main()
'''
        
        # Script dosyasını kaydet
        script_path = f"{WORKSPACE}/single_variant_training/training_script.py"
        with open(script_path, 'w', encoding='utf-8') as f:
            f.write(script_content)
        
        print(f"✅ Training script oluşturuldu: {script_path}")
        return script_path
        
    except Exception as e:
        print(f"❌ Script oluşturma hatası: {e}")
        return None

def start_training_monitoring():
    """Training monitoring başlat"""
    print("\n📊 TRAINING MONITORING BAŞLATILIYOR...")
    
    def monitor_training():
        """Background monitoring function"""
        log_file = f"{WORKSPACE}/single_variant_training/logs/training_monitor.log"
        
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write(f"Training monitoring started: {datetime.now()}\\n")
        
        while True:
            try:
                # GPU memory monitoring
                if torch.cuda.is_available():
                    memory_used = torch.cuda.memory_allocated() / (1024**3)
                    memory_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                    
                    with open(log_file, 'a', encoding='utf-8') as f:
                        f.write(f"{datetime.now()}: GPU Memory: {memory_used:.1f}GB / {memory_total:.1f}GB\\n")
                
                time.sleep(30)  # 30 saniye aralık
                
            except Exception as e:
                with open(log_file, 'a', encoding='utf-8') as f:
                    f.write(f"{datetime.now()}: Monitoring error: {e}\\n")
                break
    
    # Background thread başlat
    monitor_thread = threading.Thread(target=monitor_training, daemon=True)
    monitor_thread.start()
    
    print("✅ Background monitoring başlatıldı")
    print(f"📁 Log file: {WORKSPACE}/single_variant_training/logs/training_monitor.log")

def execute_training(script_path: str):
    """Training script'ini çalıştır"""
    print("\n🔥 TRAINING EXECUTION BAŞLATILIYOR...")
    
    try:
        print(f"📝 Script path: {script_path}")
        print("⚡ Python script execution...")
        
        # Training script'ini çalıştır
        result = subprocess.run([
            sys.executable, script_path
        ], 
        capture_output=False, 
        text=True, 
        cwd=WORKSPACE,
        timeout=28800  # 8 saat timeout
        )
        
        if result.returncode == 0:
            print("✅ Training başarıyla tamamlandı!")
            return True
        else:
            print(f"❌ Training hatası: return code {result.returncode}")
            return False
            
    except subprocess.TimeoutExpired:
        print("⚠️ Training timeout (8 saat)")
        return False
    except Exception as e:
        print(f"❌ Training execution hatası: {e}")
        return False

def main():
    """Ana training execution fonksiyonu"""
    
    start_time = time.time()
    print_training_header()
    
    # Step 1: Konfigürasyon yükleme
    config = load_training_configuration()
    if not config:
        print("❌ Konfigürasyon yüklenemedi!")
        return False
    
    # Step 2: Environment hazırlığı
    if not setup_training_environment(config):
        print("❌ Environment hazırlığı başarısız!")
        return False
    
    # Step 3: Training script oluşturma
    script_path = create_training_script(config)
    if not script_path:
        print("❌ Training script oluşturulamadı!")
        return False
    
    # Step 4: Monitoring başlatma
    start_training_monitoring()
    
    # Step 5: Training execution
    print("\\n" + "🔥" * 50)
    print("🚀 TÜRK LLM EĞİTİMİ BAŞLIYOR!")
    print("🔥" * 50)
    print(f"⏰ Başlangıç zamanı: {datetime.now().strftime('%H:%M:%S')}")
    print(f"🎯 Beklenen bitiş: {(datetime.now() + timedelta(hours=config['training_time_hours'])).strftime('%H:%M:%S')}")
    print("🔥" * 50)
    
    training_success = execute_training(script_path)
    
    # Execution summary
    execution_time = time.time() - start_time
    
    print("\\n" + "🎉" * 90)
    print("🏆 TRAINING EXECUTION COMPLETE")
    print("🎉" * 90)
    print(f"⏰ Total execution time: {execution_time/3600:.2f} hours")
    print(f"🎯 Training success: {'✅ SUCCESS' if training_success else '❌ FAILED'}")
    print(f"📁 Model output: {config['optimized_config']['output_dir']}")
    
    if training_success:
        print("\\n🎉 TÜRK LLM EĞİTİMİ BAŞARIYLA TAMAMLANDI!")
        print("📊 Model değerlendirme ve test için hazır")
        print(f"📁 Model dosyaları: {config['optimized_config']['output_dir']}")
    else:
        print("\\n⚠️ Training tamamlanamadı!")
        print("📊 Loglari kontrol edin")
    
    print("🎉" * 90)
    
    return training_success

if __name__ == "__main__":
    success = main()
    print(f"\\n🏆 DoRA + NEFTune + Sophia Ultimate training: {'SUCCESS' if success else 'FAILED'}")