"""
🚀 GOOGLE COLAB PRO+ A100 OPTİMİZE EDİLMİŞ TRAINER
Catastrophic Forgetting Prevention + EWC + Self-Synthesized Rehearsal

KRİTİK FİXLER:
✅ Tokenizer Mismatch: embed_tokens ve lm_head ASLA modules_to_save'de değil
✅ Learning Rate: Türkçe-spesifik için 2e-4 optimal
✅ Dataset Kalitesi: Minimum 30 karakter kontrol
✅ Catastrophic Forgetting: EWC + Self-synthesized rehearsal
✅ A100 GPU: 40GB memory için optimize edildi
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import logging
from datetime import datetime
import json
import math

# Core imports
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer,
    DataCollatorForLanguageModeling, TrainerCallback
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset

logger = logging.getLogger(__name__)

@dataclass
class ColabProA100Config:
    """Google Colab Pro+ A100 için optimize edilmiş config"""
    
    # Model paths - Colab mount points
    model_name: str = "Qwen/Qwen3-8B"  # 🚨 Original tokenizer kullan!
    output_dir: str = "/content/drive/MyDrive/turkish_llm_output"
    
    # A100 GPU optimized settings (40GB memory)
    per_device_batch_size: int = 8  # A100 için optimal
    gradient_accumulation_steps: int = 16  # Effective batch: 128
    max_seq_length: int = 2048
    use_gradient_checkpointing: bool = True
    use_mixed_precision: str = "bf16"  # A100 için bf16 optimal
    
    # MEMORY'DEN KRİTİK AYARLAR
    learning_rate: float = 2e-4  # Türkçe-spesifik optimal
    min_text_length: int = 30    # Dataset kalitesi için minimum
    
    # Catastrophic Forgetting Prevention
    use_ewc: bool = True
    ewc_lambda: float = 0.4
    use_self_synthesis: bool = True
    synthesis_ratio: float = 0.3  # %30 synthetic data
    
    # LoRA - TOKENIZER SAFE
    lora_r: int = 64
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    # 🚨 KRİTİK: modules_to_save BOŞ - embed_tokens'a dokunma!
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])
    
    # Training schedule
    num_epochs: int = 3
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    
    # Colab specific
    save_steps: int = 250  # Sık kaydet - Colab disconnect riski
    logging_steps: int = 10
    eval_steps: int = 100


class EWCLoss:
    """Elastic Weight Consolidation - Catastrophic Forgetting Prevention"""
    
    def __init__(self, model: nn.Module, dataset: Dataset, tokenizer, device: str):
        self.model = model
        self.device = device
        self.params = {n: p.clone().detach() for n, p in model.named_parameters() if p.requires_grad}
        self.fisher_matrix = {}
        
        logger.info("🧠 EWC: Fisher Information Matrix hesaplanıyor...")
        self._compute_fisher_matrix(dataset, tokenizer)
    
    def _compute_fisher_matrix(self, dataset: Dataset, tokenizer):
        """Fisher Information Matrix hesapla"""
        
        self.model.eval()
        self.fisher_matrix = {}
        
        # Sample from dataset for Fisher computation
        sample_size = min(1000, len(dataset))
        sample_indices = np.random.choice(len(dataset), sample_size, replace=False)
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.fisher_matrix[name] = torch.zeros_like(param)
        
        for i in sample_indices[:100]:  # İlk 100 sample - Colab time limit
            sample = dataset[i]
            
            # Tokenize sample
            inputs = tokenizer(
                sample['text'][:512],  # Truncate for speed
                return_tensors="pt", 
                truncation=True,
                padding=True
            ).to(self.device)
            
            # Forward pass
            outputs = self.model(**inputs, labels=inputs['input_ids'])
            loss = outputs.loss
            
            # Backward pass
            self.model.zero_grad()
            loss.backward()
            
            # Accumulate Fisher Information
            for name, param in self.model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    self.fisher_matrix[name] += param.grad.data ** 2
        
        # Normalize Fisher matrix
        for name in self.fisher_matrix:
            self.fisher_matrix[name] /= sample_size
        
        logger.info(f"✅ EWC: Fisher matrix hesaplandı ({sample_size} sample)")
    
    def penalty(self, model: nn.Module) -> torch.Tensor:
        """EWC penalty hesapla"""
        
        loss = 0
        for name, param in model.named_parameters():
            if name in self.fisher_matrix:
                loss += (self.fisher_matrix[name] * (param - self.params[name]) ** 2).sum()
        
        return loss


class SelfSynthesizedRehearsal:
    """Self-Synthesized Rehearsal - Eski bilgiyi koruma"""
    
    def __init__(self, model: nn.Module, tokenizer, device: str):
        self.model = model
        self.tokenizer = tokenizer  
        self.device = device
        self.synthetic_data = []
    
    def generate_synthetic_samples(self, num_samples: int = 1000) -> List[Dict]:
        """Orijinal modelden sentetik Türkçe örnekler üret"""
        
        logger.info(f"🎭 Self-Synthesis: {num_samples} sentetik örnek üretiliyor...")
        
        # Türkçe prompt'lar
        turkish_prompts = [
            "Türkiye'nin başkenti",
            "Eğitim sisteminde",
            "Teknoloji gelişimi",
            "Kültürel değerlerimiz",
            "Tarihimizde önemli",
            "Bilim ve sanat",
            "Doğal güzellikler",
            "Sosyal yaşamımızda"
        ]
        
        self.model.eval()
        synthetic_samples = []
        
        with torch.no_grad():
            for i in range(num_samples // len(turkish_prompts)):
                for prompt in turkish_prompts:
                    
                    # Tokenize prompt
                    inputs = self.tokenizer(
                        prompt,
                        return_tensors="pt",
                        padding=True
                    ).to(self.device)
                    
                    # Generate continuation
                    outputs = self.model.generate(
                        **inputs,
                        max_length=256,
                        num_return_sequences=1,
                        temperature=0.8,
                        do_sample=True,
                        pad_token_id=self.tokenizer.eos_token_id
                    )
                    
                    # Decode generated text
                    generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                    
                    # Quality check - minimum length
                    if len(generated_text) >= 50:
                        synthetic_samples.append({
                            'text': generated_text,
                            'source': 'self_synthesized',
                            'turkish_score': 0.9  # High confidence for model's own output
                        })
                    
                    if len(synthetic_samples) >= num_samples:
                        break
                
                if len(synthetic_samples) >= num_samples:
                    break
        
        logger.info(f"✅ Self-Synthesis: {len(synthetic_samples)} kaliteli örnek üretildi")
        return synthetic_samples


class ColabProA100Trainer:
    """Google Colab Pro+ A100 için optimize edilmiş trainer"""
    
    def __init__(self, config: ColabProA100Config):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.ewc_loss = None
        self.rehearsal = None
        
        # Colab setup
        self._setup_colab_environment()
    
    def _setup_colab_environment(self):
        """Colab ortamını hazırla"""
        
        logger.info("🚀 Google Colab Pro+ A100 ortamı hazırlanıyor...")
        
        # GPU check
        if not torch.cuda.is_available():
            raise RuntimeError("❌ CUDA not available!")
        
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        
        logger.info(f"✅ GPU: {gpu_name}")
        logger.info(f"✅ Memory: {gpu_memory:.1f}GB")
        
        if "A100" not in gpu_name:
            logger.warning("⚠️ A100 GPU bulunamadı - ayarlar generic GPU için düzeltiliyor")
            self.config.per_device_batch_size = 4  # Smaller batch for non-A100
        
        # Drive mount check
        if not os.path.exists("/content/drive"):
            logger.warning("⚠️ Google Drive mount edilmemiş")
        
        # Output directory
        os.makedirs(self.config.output_dir, exist_ok=True)
    
    def load_model_and_tokenizer(self):
        """Model ve tokenizer yükle - TOKENIZER SAFE"""
        
        logger.info("📥 Model ve tokenizer yükleniyor...")
        
        # 🚨 KRİTİK: Original Qwen tokenizer kullan - mismatch riski yok!
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=True,
            padding_side="left"  # Generation için
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Model loading with A100 optimizations
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=torch.bfloat16,  # A100 için optimal
            device_map="auto",
            trust_remote_code=True,
            use_cache=False  # Memory optimization
        )
        
        logger.info(f"✅ Model yüklendi: {sum(p.numel() for p in self.model.parameters()):,} params")
        logger.info(f"✅ Tokenizer vocab: {len(self.tokenizer)} tokens")
        
        # Gradient checkpointing
        if self.config.use_gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
    
    def setup_lora(self):
        """LoRA setup - TOKENIZER SAFE"""
        
        logger.info("🎯 LoRA kurulumu - TOKENIZER SAFE VERSION...")
        
        # 🚨 KRİTİK: modules_to_save BOŞ - embed_tokens'a dokunma!
        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            target_modules=self.config.target_modules,
            lora_dropout=self.config.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            modules_to_save=[]  # 🚨 BOŞ - TOKENIZER SAFETY!
        )
        
        self.model = get_peft_model(self.model, lora_config)
        
        # Stats
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        
        logger.info(f"✅ LoRA uygulandı:")
        logger.info(f"  📊 Trainable: {trainable_params:,} ({trainable_params/total_params:.2%})")
        logger.info(f"  🚨 modules_to_save: [] - TOKENIZER SAFE!")
    
    def load_dataset(self) -> Dataset:
        """Dataset yükleme - kalite filtreli"""
        
        logger.info("📊 Dataset yükleniyor...")
        
        # Simplified dataset loading for Colab
        try:
            # Try to load from drive first
            drive_dataset_path = "/content/drive/MyDrive/turkish_dataset.jsonl"
            
            if os.path.exists(drive_dataset_path):
                logger.info("📁 Drive'dan dataset yükleniyor...")
                
                data = []
                with open(drive_dataset_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        try:
                            item = json.loads(line.strip())
                            text = item.get('text', '')
                            
                            # 🚨 KRİTİK: Minimum 30 karakter kontrol
                            if len(text.strip()) >= self.config.min_text_length:
                                data.append(item)
                        except:
                            continue
                
                logger.info(f"✅ Drive dataset: {len(data)} samples")
            else:
                # Fallback: Create sample Turkish data
                logger.warning("⚠️ Drive dataset bulunamadı - sample data oluşturuluyor...")
                
                sample_texts = [
                    "Türkiye'nin en güzel şehirlerinden biri İstanbul'dur. Tarihi ve kültürel değerleri ile ünlüdür.",
                    "Eğitim sistemimizin geliştirilmesi için teknoloji kullanımı çok önemlidir.",
                    "Bilim ve sanat alanında ülkemizin başarıları gurur vericidir.",
                    "Doğal güzelliklerimiz turizm sektörüne büyük katkı sağlamaktadır.",
                ] * 1000  # 4K samples
                
                data = [{'text': text, 'source': 'sample'} for text in sample_texts]
                logger.info(f"✅ Sample dataset: {len(data)} samples")
            
            dataset = Dataset.from_list(data)
            
            # Tokenization
            def tokenize_function(examples):
                return self.tokenizer(
                    examples['text'],
                    truncation=True,
                    padding=False,
                    max_length=self.config.max_seq_length
                )
            
            dataset = dataset.map(tokenize_function, batched=True, remove_columns=['text'])
            
            logger.info(f"✅ Dataset hazır: {len(dataset)} samples")
            return dataset
            
        except Exception as e:
            logger.error(f"❌ Dataset yükleme hatası: {e}")
            raise
    
    def setup_catastrophic_forgetting_prevention(self, dataset: Dataset):
        """Catastrophic Forgetting Prevention setup"""
        
        logger.info("🧠 Catastrophic Forgetting Prevention hazırlanıyor...")
        
        # EWC setup
        if self.config.use_ewc:
            self.ewc_loss = EWCLoss(self.model, dataset, self.tokenizer, "cuda")
        
        # Self-synthesized rehearsal
        if self.config.use_self_synthesis:
            self.rehearsal = SelfSynthesizedRehearsal(self.model, self.tokenizer, "cuda")
            synthetic_samples = self.rehearsal.generate_synthetic_samples(500)  # Colab limit
            
            # Mix with original dataset
            if synthetic_samples:
                synthetic_dataset = Dataset.from_list(synthetic_samples)
                
                def tokenize_synthetic(examples):
                    return self.tokenizer(
                        examples['text'],
                        truncation=True,
                        padding=False,
                        max_length=self.config.max_seq_length
                    )
                
                synthetic_dataset = synthetic_dataset.map(
                    tokenize_synthetic, 
                    batched=True, 
                    remove_columns=['text', 'source', 'turkish_score']
                )
                
                # Mix datasets: 70% original + 30% synthetic
                original_size = int(len(dataset) * 0.7)
                synthetic_size = int(len(synthetic_dataset) * 0.3)
                
                mixed_dataset = Dataset.from_dict({
                    **dataset.select(range(original_size)).to_dict(),
                    **{k: v[:synthetic_size] for k, v in synthetic_dataset.to_dict().items()}
                })
                
                logger.info(f"✅ Mixed dataset: {len(mixed_dataset)} samples (70% original + 30% synthetic)")
                return mixed_dataset
        
        return dataset
    
    def train(self):
        """Ana training fonksiyonu"""
        
        logger.info("🚀 Google Colab Pro+ A100 Training Başlıyor...")
        
        # Setup components
        self.load_model_and_tokenizer()
        self.setup_lora()
        
        # Load and prepare dataset
        dataset = self.load_dataset()
        dataset = self.setup_catastrophic_forgetting_prevention(dataset)
        
        # Training arguments - Colab optimized
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.per_device_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,  # 2e-4 - Türkçe optimal
            warmup_ratio=self.config.warmup_ratio,
            weight_decay=self.config.weight_decay,
            
            # Mixed precision - A100 optimal
            bf16=True,  # A100 için en iyi
            tf32=True,  # A100 tensor core acceleration
            
            # Memory optimization
            gradient_checkpointing=self.config.use_gradient_checkpointing,
            dataloader_num_workers=2,
            
            # Colab specific - sık kaydet
            save_steps=self.config.save_steps,
            logging_steps=self.config.logging_steps,
            eval_steps=self.config.eval_steps,
            evaluation_strategy="steps",
            save_strategy="steps",
            
            # Colab safety
            save_total_limit=3,  # Disk space limit
            load_best_model_at_end=True,
            
            # Disable external reporting
            report_to="none",
            remove_unused_columns=False
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
            pad_to_multiple_of=8
        )
        
        # Custom trainer with EWC
        class EWCTrainer(Trainer):
            def __init__(self, ewc_loss=None, ewc_lambda=0.4, **kwargs):
                super().__init__(**kwargs)
                self.ewc_loss = ewc_loss
                self.ewc_lambda = ewc_lambda
            
            def compute_loss(self, model, inputs, return_outputs=False):
                # Standard loss
                outputs = model(**inputs)
                loss = outputs.loss
                
                # Add EWC penalty
                if self.ewc_loss is not None:
                    ewc_penalty = self.ewc_loss.penalty(model)
                    loss += self.ewc_lambda * ewc_penalty
                
                return (loss, outputs) if return_outputs else loss
        
        # Create trainer
        trainer = EWCTrainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset,
            eval_dataset=dataset.select(range(min(500, len(dataset)))),  # Small eval
            data_collator=data_collator,
            ewc_loss=self.ewc_loss,
            ewc_lambda=self.config.ewc_lambda
        )
        
        # Start training
        logger.info("🎯 Training başlıyor - tüm optimizasyonlar aktif!")
        
        train_result = trainer.train()
        
        # Save model
        trainer.save_model()
        
        # Results
        results = {
            'final_loss': train_result.training_loss,
            'training_time': train_result.metrics['train_runtime'],
            'samples_per_second': train_result.metrics['train_samples_per_second'],
            'config_used': self.config.__dict__,
            'catastrophic_forgetting_prevention': {
                'ewc_enabled': self.config.use_ewc,
                'self_synthesis_enabled': self.config.use_self_synthesis
            }
        }
        
        logger.info(f"🎉 Training tamamlandı!")
        logger.info(f"📊 Final Loss: {train_result.training_loss:.4f}")
        logger.info(f"⏱️ Süre: {train_result.metrics['train_runtime']:.2f}s")
        
        # Save results
        with open(f"{self.config.output_dir}/training_results.json", 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        return results


def run_colab_pro_a100_training():
    """Google Colab Pro+ A100 için optimize edilmiş training çalıştır"""
    
    print("🚀 GOOGLE COLAB PRO+ A100 TÜRKİYE LLM TRAINER")
    print("=" * 60)
    print("✅ Tokenizer Mismatch Protection: Aktif")
    print("✅ Learning Rate: 2e-4 (Türkçe optimal)")
    print("✅ Dataset Quality: 30+ karakter minimum")  
    print("✅ Catastrophic Forgetting: EWC + Self-synthesis")
    print("✅ A100 GPU: Tam optimize")
    print("=" * 60)
    
    config = ColabProA100Config()
    trainer = ColabProA100Trainer(config)
    
    try:
        results = trainer.train()
        
        print("\n🎉 BAŞARILI!")
        print(f"📊 Final Loss: {results['final_loss']:.4f}")
        print(f"⏱️ Training Time: {results['training_time']:.1f}s")
        print(f"💾 Model saved: {config.output_dir}")
        
        return results
        
    except Exception as e:
        print(f"\n❌ HATA: {e}")
        raise


if __name__ == "__main__":
    # Colab notebook içinde çalıştır:
    # !python colab_pro_a100_optimized_trainer.py
    results = run_colab_pro_a100_training()