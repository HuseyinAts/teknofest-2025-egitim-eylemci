#!/usr/bin/env python3
"""
🔥 TEKNOFEST 2025 TURKISH LLM TRAINING PIPELINE - COMPREHENSIVE FIXES
Complete pipeline with all critical errors resolved for Google Colab Pro+ A100 40GB

CRITICAL FIXES APPLIED:
✅ Flash Attention 2 proper installation and fallback handling
✅ BitsAndBytes quantization with new API (BitsAndBytesConfig) 
✅ DoRA implementation with proper module naming (disabled problematic features)
✅ TrainerCallback import error resolution
✅ String formatting fixes for non-numeric values
✅ Memory management and error recovery
✅ Type safety improvements throughout
"""

import os
import sys
import json
import time
import subprocess
import logging
from datetime import datetime, timedelta
from pathlib import Path
import gc
from typing import Dict, List, Optional, Union, Any

# Setup enhanced logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

class TurkishLLMTrainingPipelineFixed:
    """Comprehensive fixed pipeline for Turkish LLM training"""
    
    def __init__(self):
        self.base_dir = Path('/content/qwen3_turkish_pipeline')
        self.base_dir.mkdir(exist_ok=True)
        os.chdir(self.base_dir)
        
        self.pipeline_stats = {
            'start_time': datetime.now(),
            'stage': 'initialization',
            'success': False,
            'error': None,
            'final_loss': None,
            'training_time_hours': None
        }
        
        self.dependencies_installed = {
            'flash_attn': False,
            'bitsandbytes': False,
            'transformers': False,
            'peft': False
        }
        
    def install_dependencies_with_fixes(self):
        """Install dependencies with comprehensive error handling"""
        logger.info("🔧 INSTALLING DEPENDENCIES WITH FIXES")
        logger.info("="*50)
        
        # Core dependencies with specific versions
        core_deps = [
            "torch>=2.0.0",
            "transformers>=4.36.0", 
            "datasets>=2.14.0",
            "accelerate>=0.24.0",
            "safetensors>=0.4.0",
            "sentencepiece>=0.1.99",
            "numpy>=1.24.0"
        ]
        
        for dep in core_deps:
            try:
                print(f"Installing {dep}...")
                subprocess.run([sys.executable, "-m", "pip", "install", dep], 
                             check=True, capture_output=True, text=True)
                print(f"✅ {dep} installed")
            except subprocess.CalledProcessError as e:
                logger.warning(f"⚠️ {dep} installation failed: {e}")
        
        # PEFT with DoRA support
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "peft>=0.7.0"], 
                         check=True, capture_output=True, text=True)
            self.dependencies_installed['peft'] = True
            print("✅ PEFT installed with DoRA support")
        except subprocess.CalledProcessError as e:
            logger.warning(f"⚠️ PEFT installation failed: {e}")
            
        # BitsAndBytes with new API
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", 
                          "bitsandbytes>=0.41.0", "--upgrade"], 
                         check=True, capture_output=True, text=True)
            self.dependencies_installed['bitsandbytes'] = True
            print("✅ BitsAndBytes installed with new API")
        except subprocess.CalledProcessError as e:
            logger.warning(f"⚠️ BitsAndBytes installation failed: {e}")
        
        # Flash Attention 2 with fallback
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", 
                          "flash-attn", "--no-build-isolation"], 
                         check=True, capture_output=True, text=True)
            self.dependencies_installed['flash_attn'] = True
            print("✅ Flash Attention 2 installed")
        except subprocess.CalledProcessError as e:
            logger.warning(f"⚠️ Flash Attention 2 installation failed: {e}")
            logger.info("Will use standard attention as fallback")
        
        self._check_versions()
        
    def _check_versions(self):
        """Check installed dependency versions"""
        try:
            import torch
            import transformers
            logger.info(f"✅ PyTorch: {torch.__version__}")
            logger.info(f"✅ Transformers: {transformers.__version__}")
            
            try:
                import peft
                logger.info(f"✅ PEFT: {peft.__version__}")
            except ImportError:
                logger.warning("⚠️ PEFT not available")
                
        except ImportError as e:
            logger.error(f"Critical dependency missing: {e}")
    
    def check_gpu_environment(self):
        """Validate GPU environment"""
        logger.info("🔍 CHECKING GPU ENVIRONMENT")
        
        try:
            import torch
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
                logger.info(f"✅ GPU: {gpu_name}")
                logger.info(f"✅ GPU Memory: {gpu_memory:.1f}GB")
                return True
            else:
                logger.error("❌ No GPU available")
                return False
        except Exception as e:
            logger.error(f"❌ GPU check failed: {e}")
            return False
    
    def load_model_with_fixes(self):
        """Load model with comprehensive error handling"""
        logger.info("🤖 LOADING MODEL WITH FIXES")
        
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM
            
            # Load tokenizer
            model_name = "Qwen/Qwen2.5-8B-Instruct"
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                
            # Model loading with comprehensive fallbacks
            model_kwargs = {
                "torch_dtype": torch.bfloat16,
                "device_map": "auto",
                "use_cache": False,
                "trust_remote_code": True,
                "low_cpu_mem_usage": True
            }
            
            model = None
            
            # Try Flash Attention 2 if available
            if self.dependencies_installed.get('flash_attn', False):
                try:
                    model_kwargs["attn_implementation"] = "flash_attention_2"
                    
                    # Try with quantization if available
                    if self.dependencies_installed.get('bitsandbytes', False):
                        try:
                            from transformers import BitsAndBytesConfig
                            bnb_config = BitsAndBytesConfig(
                                load_in_4bit=True,
                                bnb_4bit_use_double_quant=True,
                                bnb_4bit_quant_type="nf4",
                                bnb_4bit_compute_dtype=torch.bfloat16
                            )
                            model_kwargs["quantization_config"] = bnb_config
                            
                            model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
                            logger.info("✅ Model loaded with Flash Attention 2 + 4-bit quantization")
                            
                        except Exception as quant_error:
                            logger.warning(f"Quantization failed: {quant_error}")
                            model_kwargs.pop("quantization_config", None)
                            model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
                            logger.info("✅ Model loaded with Flash Attention 2 only")
                    else:
                        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
                        logger.info("✅ Model loaded with Flash Attention 2")
                        
                except Exception as flash_error:
                    logger.warning(f"Flash Attention 2 failed: {flash_error}")
                    model = None
            
            # Fallback to standard attention
            if model is None:
                model_kwargs.pop("attn_implementation", None)
                model_kwargs.pop("quantization_config", None)
                
                model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
                logger.info("✅ Model loaded with standard attention")
            
            logger.info(f"📊 Vocabulary size: {len(tokenizer.vocab) if hasattr(tokenizer, 'vocab') else 'Unknown'}")
            
            return model, tokenizer
            
        except Exception as e:
            logger.error(f"Model loading failed: {e}")
            raise
    
    def setup_fixed_lora(self, model):
        """Setup LoRA with DoRA fixes (disabled problematic features)"""
        logger.info("🔧 SETTING UP FIXED LORA")
        
        try:
            from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
            
            # Fixed LoRA config - DoRA disabled to avoid module naming conflicts
            lora_config = LoraConfig(
                r=128,  # Reasonable rank
                lora_alpha=64,  # Alpha value
                target_modules=[
                    "q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"
                ],
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
                use_dora=False,  # FIXED: Disabled to avoid module naming conflicts
                use_rslora=True  # Use rank-stabilized LoRA instead
            )
            
            model = prepare_model_for_kbit_training(model)
            model = get_peft_model(model, lora_config)
            
            # Calculate trainable parameters
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in model.parameters())
            
            logger.info("✅ LoRA Configuration Applied:")
            logger.info(f"   ├─ Rank: {lora_config.r}")
            logger.info(f"   ├─ Alpha: {lora_config.lora_alpha}")
            logger.info(f"   ├─ DoRA: Disabled (FIXED)")
            logger.info(f"   ├─ RSLoRA: Enabled")
            logger.info(f"   └─ Trainable: {trainable_params:,}/{total_params:,} ({trainable_params/total_params*100:.2f}%)")
            
            return model
            
        except Exception as e:
            logger.error(f"LoRA setup failed: {e}")
            raise
    
    def prepare_training_data(self) -> List[str]:
        """Prepare Turkish training data"""
        logger.info("📚 PREPARING TRAINING DATA")
        
        training_texts = [
            "Türkiye'de eğitim sistemi sürekli gelişmektedir.",
            "Yapay zeka teknolojileri eğitimde aktif olarak kullanılmaktadır.",
            "Öğrenciler için interaktif öğrenme materyalleri hazırlanmaktadır.",
            "Dil modelleri Türkçe metinleri başarılı bir şekilde anlayabilmektedir.",
            "Morfolojik analiz Türkçe doğal dil işlemede kritik öneme sahiptir.",
            "TEKNOFEST 2025 yarışmasına yoğun bir şekilde hazırlanıyoruz.",
            "Teknoloji festivalinde gençler projelerini sergiliyorlar.",
            "İnovasyon ve teknoloji alanında Türkiye önemli adımlar atıyor.",
            "Milli teknoloji hamlesinde eğitim çok önemli bir role sahip.",
            "Öğrenme sürecinde kişiselleştirme çok önemlidir.",
            "Adaptif öğrenme sistemleri her öğrencinin ihtiyacına göre şekillenir.",
            "Eğitimde yapay zeka destekli araçlar kullanım oranı artmaktadır.",
            "Uzaktan eğitim teknolojileri pandemi sonrası yaygınlaştı.",
            "Öğrencilerimizin başarılarından dolayı gururluyuz.",
            "Eğitim teknolojilerindeki gelişmeler hızla ilerlemektedir.",
            "Öğretmenlerimizin deneyimlerinden yararlanarak ileriyoruz.",
            "Müfredatımızı günün şartlarına göre güncellemeye devam ediyoruz.",
            "Algoritmaların eğitim süreçlerindeki etkisi araştırılmaktadır.",
            "Veri analitiği ile öğrenci performansı ölçümlenmektedir.",
            "Makine öğrenmesi yöntemleri eğitim alanında uygulanmaktadır."
        ]
        
        logger.info(f"✅ Prepared {len(training_texts)} training examples")
        return training_texts
    
    def create_fixed_trainer(self, model, tokenizer, train_dataset, eval_dataset):
        """Create trainer with comprehensive fixes"""
        logger.info("🏋️ CREATING FIXED TRAINER")
        
        try:
            from transformers import (
                TrainingArguments, Trainer, 
                DataCollatorForLanguageModeling,
                TrainerCallback  # FIXED: Proper import
            )
            
            # Fixed training arguments
            training_args = TrainingArguments(
                output_dir=str(self.base_dir / "qwen3_turkish_final"),
                num_train_epochs=2,  # Conservative for testing
                per_device_train_batch_size=4,  # Conservative batch size
                gradient_accumulation_steps=4,
                learning_rate=2e-4,
                warmup_ratio=0.1,
                logging_steps=10,
                save_steps=50,
                eval_steps=50,
                evaluation_strategy="steps",  # FIXED: Correct parameter name
                save_strategy="steps",
                bf16=True,
                tf32=True,
                gradient_checkpointing=True,
                remove_unused_columns=False,
                load_best_model_at_end=True,
                metric_for_best_model="eval_loss",
                greater_is_better=False,
                save_total_limit=2,
                max_steps=200,  # Limited for testing
                report_to=None,
                prediction_loss_only=True
            )
            
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=tokenizer,
                mlm=False
            )
            
            # FIXED: Callback with proper error handling
            class FixedCallback(TrainerCallback):
                def __init__(self, pipeline_stats):
                    self.pipeline_stats = pipeline_stats
                    self.start_time = datetime.now()
                
                def on_train_begin(self, args, state, control, **kwargs):
                    self.start_time = datetime.now()
                    logger.info("🚀 Training started")
                
                def on_log(self, args, state, control, logs=None, **kwargs):
                    if logs:
                        step = state.global_step
                        # FIXED: Safe string formatting
                        loss = logs.get('train_loss', 'N/A')
                        loss_str = f"{loss:.4f}" if isinstance(loss, (int, float)) else str(loss)
                        
                        elapsed = datetime.now() - self.start_time
                        time_str = f"{elapsed.total_seconds()/60:.1f}min"
                        
                        logger.info(f"📊 Step {step}: Loss={loss_str}, Time={time_str}")
                
                def on_train_end(self, args, state, control, **kwargs):
                    # FIXED: Safe final statistics
                    final_loss = None
                    if state.log_history:
                        for log_entry in reversed(state.log_history):
                            if 'train_loss' in log_entry:
                                final_loss = log_entry['train_loss']
                                break
                    
                    training_time = datetime.now() - self.start_time
                    training_hours = training_time.total_seconds() / 3600
                    
                    self.pipeline_stats.update({
                        'success': True,
                        'final_loss': final_loss,
                        'training_time_hours': training_hours,
                        'stage': 'completed'
                    })
                    
                    # FIXED: Safe formatting
                    loss_str = f"{final_loss:.4f}" if isinstance(final_loss, (int, float)) else "N/A"
                    time_str = f"{training_hours:.2f}h" if isinstance(training_hours, (int, float)) else "N/A"
                    
                    logger.info(f"🎉 Training completed!")
                    logger.info(f"   ├─ Final Loss: {loss_str}")
                    logger.info(f"   └─ Training Time: {time_str}")
            
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                data_collator=data_collator,
                callbacks=[FixedCallback(self.pipeline_stats)]
            )
            
            logger.info("✅ Fixed trainer created successfully")
            return trainer
            
        except Exception as e:
            logger.error(f"Trainer creation failed: {e}")
            raise
    
    def run_complete_pipeline(self) -> Dict[str, Any]:
        """Run the complete fixed pipeline"""
        logger.info("🔥" * 60)
        logger.info("🚀 FIXED TURKISH LLM TRAINING PIPELINE")
        logger.info("🔥" * 60)
        
        start_time = datetime.now()
        logger.info(f"⏰ Started: {start_time.strftime('%H:%M:%S')}")
        
        try:
            # Stage 1: Install dependencies
            self.install_dependencies_with_fixes()
            
            # Stage 2: Check environment
            if not self.check_gpu_environment():
                raise RuntimeError("GPU environment validation failed")
            
            # Stage 3: Load model
            model, tokenizer = self.load_model_with_fixes()
            
            # Stage 4: Setup LoRA
            model = self.setup_fixed_lora(model)
            
            # Stage 5: Prepare data
            training_texts = self.prepare_training_data()
            
            # Tokenize data
            def tokenize_function(examples):
                tokenized = tokenizer(
                    examples,
                    truncation=True,
                    padding=True,
                    max_length=512,
                    return_tensors="pt"
                )
                tokenized['labels'] = tokenized['input_ids'].clone()
                return {k: v.tolist() if hasattr(v, 'tolist') else v for k, v in tokenized.items()}
            
            from datasets import Dataset
            dataset_dict = tokenize_function(training_texts)
            dataset = Dataset.from_dict(dataset_dict)
            
            train_size = int(0.9 * len(dataset))
            train_dataset = dataset.select(range(train_size))
            eval_dataset = dataset.select(range(train_size, len(dataset)))
            
            # Stage 6: Create trainer
            trainer = self.create_fixed_trainer(model, tokenizer, train_dataset, eval_dataset)
            
            # Stage 7: Run training
            logger.info("🚀 STARTING TRAINING")
            trainer.train()
            
            # Save model
            final_model_dir = self.base_dir / "qwen3_turkish_final_model"
            trainer.save_model(str(final_model_dir))
            tokenizer.save_pretrained(str(final_model_dir))
            
            logger.info("✅ Training completed successfully!")
            logger.info(f"📁 Model saved to: {final_model_dir}")
            
            # Final statistics
            end_time = datetime.now()
            total_time = end_time - start_time
            
            self.pipeline_stats.update({
                'success': True,
                'stage': 'completed',
                'total_time_hours': total_time.total_seconds() / 3600
            })
            
            # FIXED: Safe final output
            final_loss = self.pipeline_stats.get('final_loss', 'N/A')
            training_time = self.pipeline_stats.get('training_time_hours', 'N/A')
            
            loss_str = f"{final_loss:.4f}" if isinstance(final_loss, (int, float)) else str(final_loss)
            time_str = f"{training_time:.2f}h" if isinstance(training_time, (int, float)) else str(training_time)
            
            logger.info("🎉" * 60)
            logger.info("🏆 PIPELINE COMPLETED SUCCESSFULLY")
            logger.info("🎉" * 60)
            logger.info(f"📊 Final Loss: {loss_str}")
            logger.info(f"⏱️ Training Time: {time_str}")
            logger.info("🎉" * 60)
            
            return self.pipeline_stats
            
        except Exception as e:
            # Comprehensive error handling
            end_time = datetime.now()
            total_time = end_time - start_time
            
            error_msg = str(e)
            logger.error(f"💥 PIPELINE FAILED: {error_msg}")
            
            self.pipeline_stats.update({
                'success': False,
                'error': error_msg,
                'stage': 'failed',
                'total_time_hours': total_time.total_seconds() / 3600
            })
            
            # Save error details
            error_file = self.base_dir / "pipeline_error.json"
            try:
                with open(error_file, 'w', encoding='utf-8') as f:
                    json.dump({
                        'error': error_msg,
                        'stage': self.pipeline_stats.get('stage', 'unknown'),
                        'timestamp': datetime.now().isoformat(),
                        'dependencies': self.dependencies_installed
                    }, f, indent=2, ensure_ascii=False)
                logger.error(f"📝 Error details saved to: {error_file}")
            except Exception:
                pass
            
            return self.pipeline_stats


# Simple interface function
def run_fixed_pipeline():
    """Run the fixed pipeline"""
    pipeline = TurkishLLMTrainingPipelineFixed()
    return pipeline.run_complete_pipeline()


if __name__ == "__main__":
    print("🔥 Starting TEKNOFEST 2025 Turkish LLM Training (FIXED VERSION)...")
    print("="*80)
    
    # Run the fixed pipeline
    results = run_fixed_pipeline()
    
    print("\n🎉 TRAINING COMPLETED!")
    print(f"📊 Success: {results.get('success', False)}")
    
    # FIXED: Safe output formatting
    final_loss = results.get('final_loss', 'N/A')
    loss_str = f"{final_loss:.4f}" if isinstance(final_loss, (int, float)) else str(final_loss)
    print(f"📊 Final Loss: {loss_str}")
    
    training_time = results.get('training_time_hours', 'N/A')
    time_str = f"{training_time:.2f}h" if isinstance(training_time, (int, float)) else str(training_time)
    print(f"⏱️ Training Time: {time_str}")