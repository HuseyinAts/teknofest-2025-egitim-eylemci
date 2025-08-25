"""
QWEN3 Turkish Training - Google Colab Pro+ Optimized
Optimized for V100/A100 GPUs with 16-40GB VRAM
"""

import os
import sys
import json
import gc
import time
import warnings
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
import logging
import torch
import psutil
from datetime import datetime

warnings.filterwarnings('ignore')

# ============================================================================
# GOOGLE COLAB SETUP
# ============================================================================

IS_COLAB = 'google.colab' in sys.modules

if IS_COLAB:
    print("🔍 Google Colab environment detected!")
    from google.colab import drive, output
    
    # Mount Google Drive
    try:
        drive.mount('/content/drive', force_remount=True)
        DRIVE_PATH = Path('/content/drive/MyDrive/qwen_training')
        DRIVE_PATH.mkdir(parents=True, exist_ok=True)
        print(f"✅ Google Drive mounted at {DRIVE_PATH}")
    except Exception as e:
        print(f"⚠️ Google Drive mounting failed: {e}")
        DRIVE_PATH = Path('/content/qwen_training')
        DRIVE_PATH.mkdir(parents=True, exist_ok=True)
    
    # Check GPU
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"🖥️ GPU: {gpu_name} ({gpu_memory:.1f}GB)")
        
        # Detect Colab tier
        if 'A100' in gpu_name:
            COLAB_TIER = 'pro_plus_a100'
        elif 'V100' in gpu_name:
            COLAB_TIER = 'pro_plus_v100'
        elif 'P100' in gpu_name:
            COLAB_TIER = 'pro'
        elif 'T4' in gpu_name:
            COLAB_TIER = 'free'
        else:
            COLAB_TIER = 'unknown'
        
        print(f"📊 Detected Colab tier: {COLAB_TIER}")
else:
    DRIVE_PATH = Path('./qwen_training')
    COLAB_TIER = 'local'

# ============================================================================
# PACKAGE INSTALLATION
# ============================================================================

def install_packages():
    """Install required packages for Colab"""
    
    packages = [
        "transformers>=4.44.0",
        "datasets",
        "accelerate>=0.30.0",
        "peft>=0.11.0",
        "bitsandbytes>=0.43.0",
        "sentencepiece",
        "tiktoken",
        "trl>=0.8.0",
        "flash-attn",  # Pro+ can use Flash Attention
        "wandb",
        "safetensors",
        "einops",
    ]
    
    print("📦 Installing packages...")
    for package in packages:
        os.system(f"pip install -q {package}")
    
    # Install Flash Attention for Pro+
    if COLAB_TIER in ['pro_plus_a100', 'pro_plus_v100']:
        try:
            os.system("pip install -q ninja")
            os.system("pip install -q flash-attn --no-build-isolation")
            print("✅ Flash Attention installed")
        except:
            print("⚠️ Flash Attention installation failed")
    
    print("✅ Package installation complete!")

# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class TrainingConfig:
    """Training configuration optimized for Colab Pro+"""
    
    # Model settings - Pro+ can handle larger models
    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    
    # Training parameters
    num_epochs: int = 5
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    
    # Batch settings - optimized per GPU tier
    batch_size: int = 4  # Will be auto-adjusted
    gradient_accumulation_steps: int = 4
    max_length: int = 512
    
    # Optimization
    use_flash_attention: bool = True
    gradient_checkpointing: bool = True
    mixed_precision: str = "bf16"  # bf16 for A100/V100
    optim: str = "adamw_torch_fused"
    
    # LoRA settings
    use_lora: bool = True
    lora_rank: int = 64  # Higher rank for Pro+
    lora_alpha: int = 128
    lora_dropout: float = 0.1
    
    # Quantization
    use_4bit: bool = False  # Pro+ has enough memory
    use_8bit: bool = False
    
    # Dataset
    dataset_name: str = "teknofest/turkish-qa-dataset"  # Example dataset
    max_train_samples: Optional[int] = 10000  # More samples for Pro+
    
    # Checkpointing
    save_steps: int = 500
    eval_steps: int = 250
    save_total_limit: int = 3
    output_dir: str = str(DRIVE_PATH / "checkpoints")
    
    # Logging
    logging_steps: int = 10
    report_to: str = "wandb"  # Pro+ can use wandb
    
    def __post_init__(self):
        """Auto-adjust based on GPU tier"""
        if COLAB_TIER == 'pro_plus_a100':
            self.batch_size = 8
            self.gradient_accumulation_steps = 2
            self.max_length = 1024
            self.lora_rank = 128
            self.mixed_precision = "bf16"
        elif COLAB_TIER == 'pro_plus_v100':
            self.batch_size = 4
            self.gradient_accumulation_steps = 4
            self.max_length = 512
            self.lora_rank = 64
            self.mixed_precision = "fp16"
        elif COLAB_TIER == 'pro':
            self.batch_size = 2
            self.gradient_accumulation_steps = 8
            self.max_length = 384
            self.lora_rank = 32
            self.use_4bit = True
        elif COLAB_TIER == 'free':
            self.model_name = "microsoft/phi-2"  # Smaller model
            self.batch_size = 1
            self.gradient_accumulation_steps = 16
            self.max_length = 256
            self.lora_rank = 16
            self.use_4bit = True
            self.max_train_samples = 1000

# ============================================================================
# MEMORY OPTIMIZATION
# ============================================================================

def clear_memory():
    """Clear GPU and CPU memory"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def get_memory_stats():
    """Get current memory usage"""
    stats = {
        'cpu_percent': psutil.virtual_memory().percent,
        'cpu_gb': psutil.virtual_memory().used / 1e9
    }
    
    if torch.cuda.is_available():
        stats.update({
            'gpu_allocated_gb': torch.cuda.memory_allocated() / 1e9,
            'gpu_reserved_gb': torch.cuda.memory_reserved() / 1e9,
            'gpu_free_gb': (torch.cuda.get_device_properties(0).total_memory - 
                          torch.cuda.memory_reserved()) / 1e9
        })
    
    return stats

# ============================================================================
# MODEL LOADING
# ============================================================================

def load_model_and_tokenizer(config: TrainingConfig):
    """Load model and tokenizer with optimization"""
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
    
    print(f"📚 Loading model: {config.model_name}")
    
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        trust_remote_code=True,
        use_fast=True
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Model arguments
    model_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": torch.bfloat16 if config.mixed_precision == "bf16" else torch.float16,
        "low_cpu_mem_usage": True,
        "device_map": "auto",
    }
    
    # Flash Attention for Pro+
    if config.use_flash_attention and COLAB_TIER in ['pro_plus_a100', 'pro_plus_v100']:
        model_kwargs["attn_implementation"] = "flash_attention_2"
    
    # Quantization
    if config.use_4bit or config.use_8bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=config.use_4bit,
            load_in_8bit=config.use_8bit,
            bnb_4bit_compute_dtype=torch.bfloat16 if config.mixed_precision == "bf16" else torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(config.model_name, **model_kwargs)
    
    # LoRA setup
    if config.use_lora:
        if config.use_4bit or config.use_8bit:
            model = prepare_model_for_kbit_training(
                model,
                use_gradient_checkpointing=config.gradient_checkpointing
            )
        
        peft_config = LoraConfig(
            r=config.lora_rank,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )
        
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
    
    # Gradient checkpointing
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    
    print(f"✅ Model loaded successfully!")
    print(f"📊 Memory: {get_memory_stats()}")
    
    return model, tokenizer

# ============================================================================
# DATASET PREPARATION
# ============================================================================

def prepare_datasets(tokenizer, config: TrainingConfig):
    """Prepare training datasets"""
    from datasets import load_dataset
    
    print(f"📂 Loading dataset: {config.dataset_name}")
    
    try:
        # Load dataset
        dataset = load_dataset(config.dataset_name, split="train")
        
        # Limit samples if specified
        if config.max_train_samples:
            dataset = dataset.select(range(min(config.max_train_samples, len(dataset))))
        
        # Split into train/eval
        dataset_split = dataset.train_test_split(test_size=0.1, seed=42)
        train_dataset = dataset_split['train']
        eval_dataset = dataset_split['test']
        
    except Exception as e:
        print(f"⚠️ Dataset loading failed: {e}")
        print("Using fallback dataset...")
        
        # Fallback Turkish dataset
        samples = [
            {"text": "Yapay zeka, bilgisayarların insan benzeri düşünme ve öğrenme yeteneklerine sahip olmasını sağlayan teknolojilerin genel adıdır."},
            {"text": "Makine öğrenmesi, veri ve istatistiksel yöntemler kullanarak bilgisayarların belirli görevlerde performanslarını geliştirmelerini sağlar."},
            {"text": "Derin öğrenme, yapay sinir ağlarını kullanarak karmaşık problemleri çözen bir makine öğrenmesi tekniğidir."},
            {"text": "Türkiye'de teknoloji sektörü hızla büyümekte ve yeni girişimler ortaya çıkmaktadır."},
            {"text": "Python programlama dili, veri bilimi ve yapay zeka projelerinde en çok kullanılan dillerden biridir."},
        ] * 200
        
        from datasets import Dataset
        dataset = Dataset.from_list(samples)
        dataset_split = dataset.train_test_split(test_size=0.1, seed=42)
        train_dataset = dataset_split['train']
        eval_dataset = dataset_split['test']
    
    # Tokenization function
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=config.max_length
        )
    
    # Tokenize datasets
    train_dataset = train_dataset.map(
        tokenize_function,
        batched=True,
        num_proc=2,  # Colab has 2 CPUs
        remove_columns=train_dataset.column_names
    )
    
    eval_dataset = eval_dataset.map(
        tokenize_function,
        batched=True,
        num_proc=2,
        remove_columns=eval_dataset.column_names
    )
    
    print(f"✅ Dataset prepared: {len(train_dataset)} train, {len(eval_dataset)} eval")
    
    return train_dataset, eval_dataset

# ============================================================================
# TRAINING
# ============================================================================

def train_model(model, tokenizer, train_dataset, eval_dataset, config: TrainingConfig):
    """Train the model"""
    from transformers import (
        TrainingArguments,
        Trainer,
        DataCollatorForLanguageModeling,
        EarlyStoppingCallback,
        TrainerCallback
    )
    import wandb
    
    # Initialize wandb for Pro+
    if config.report_to == "wandb" and COLAB_TIER in ['pro_plus_a100', 'pro_plus_v100']:
        try:
            wandb.init(
                project="qwen-turkish-training",
                name=f"qwen-{COLAB_TIER}-{datetime.now().strftime('%Y%m%d_%H%M')}",
                config=config.__dict__
            )
        except:
            print("⚠️ Wandb initialization failed")
            config.report_to = "none"
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=8
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        eval_steps=config.eval_steps,
        evaluation_strategy="steps",
        save_strategy="steps",
        save_total_limit=config.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=config.mixed_precision == "bf16",
        fp16=config.mixed_precision == "fp16",
        gradient_checkpointing=config.gradient_checkpointing,
        optim=config.optim,
        report_to=config.report_to,
        push_to_hub=False,
        dataloader_num_workers=2,  # Colab has 2 CPUs
        dataloader_pin_memory=True,
        remove_unused_columns=False,
        label_names=["labels"],
    )
    
    # Custom callback for memory monitoring
    class MemoryCallback(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            if state.global_step % 100 == 0:
                stats = get_memory_stats()
                print(f"Step {state.global_step} - Memory: GPU {stats.get('gpu_allocated_gb', 0):.1f}GB, CPU {stats['cpu_gb']:.1f}GB")
                clear_memory()
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=3),
            MemoryCallback()
        ],
    )
    
    # Start training
    print("🚀 Starting training...")
    print(f"📊 Total training steps: {len(train_dataset) // (config.batch_size * config.gradient_accumulation_steps) * config.num_epochs}")
    
    try:
        trainer.train()
        
        # Save final model
        final_dir = Path(config.output_dir) / "final_model"
        model.save_pretrained(final_dir)
        tokenizer.save_pretrained(final_dir)
        print(f"✅ Model saved to {final_dir}")
        
        # Push to hub if Pro+
        if COLAB_TIER in ['pro_plus_a100', 'pro_plus_v100']:
            try:
                from huggingface_hub import login
                # You need to set your HF token
                # login(token="your_hf_token")
                # model.push_to_hub("your-username/qwen-turkish-finetuned")
                print("💡 To push to HuggingFace Hub, uncomment and add your token")
            except:
                pass
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        raise
    
    print("✅ Training complete!")
    return trainer

# ============================================================================
# INFERENCE TEST
# ============================================================================

def test_model(model, tokenizer, prompt="Türkiye'nin başkenti neresidir?"):
    """Test the trained model"""
    print(f"\n🧪 Testing model with prompt: {prompt}")
    
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.7,
            do_sample=True,
            top_p=0.95,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"📝 Response: {response}")
    return response

# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    """Main training pipeline for Google Colab Pro+"""
    
    print("=" * 60)
    print("🚀 QWEN TURKISH TRAINING - GOOGLE COLAB PRO+ OPTIMIZED")
    print("=" * 60)
    print(f"📍 Environment: {'Google Colab' if IS_COLAB else 'Local'}")
    print(f"🎯 Tier: {COLAB_TIER}")
    print(f"💾 Output: {DRIVE_PATH}")
    print("=" * 60)
    
    # Install packages
    install_packages()
    
    # Clear memory before starting
    clear_memory()
    
    # Initialize configuration
    config = TrainingConfig()
    print("\n⚙️ Configuration:")
    print(json.dumps(config.__dict__, indent=2))
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(config)
    
    # Prepare datasets
    train_dataset, eval_dataset = prepare_datasets(tokenizer, config)
    
    # Train model
    trainer = train_model(model, tokenizer, train_dataset, eval_dataset, config)
    
    # Test the model
    test_prompts = [
        "Yapay zeka nedir?",
        "Python programlama dilinin avantajları nelerdir?",
        "Türkiye'de teknoloji sektörü nasıl gelişiyor?",
    ]
    
    for prompt in test_prompts:
        test_model(model, tokenizer, prompt)
    
    # Print final stats
    print("\n" + "=" * 60)
    print("🎉 TRAINING COMPLETE!")
    print("=" * 60)
    print(f"📊 Final Memory: {get_memory_stats()}")
    
    if trainer.state.best_metric:
        print(f"📈 Best eval loss: {trainer.state.best_metric:.4f}")
    
    print(f"💾 Model saved to: {config.output_dir}/final_model")
    
    if IS_COLAB:
        print("\n💡 Tips for Colab Pro+:")
        print("1. Your model is saved to Google Drive for persistence")
        print("2. Use wandb.ai to monitor training remotely")
        print("3. Enable GPU runtime: Runtime > Change runtime type > GPU > A100/V100")
        print("4. For long training, use 'Prevent disconnection' browser extensions")
    
    print("=" * 60)

if __name__ == "__main__":
    main()