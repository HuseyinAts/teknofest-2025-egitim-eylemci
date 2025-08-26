# Parallel Hybrid Approach - Ultra Detailed Implementation
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
import json
import threading
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import time
from datetime import datetime

class ParallelHybridTrainer:
    """
    Paralel Hibrit Yaklaşım - Aynı anda iki model eğitimi
    
    Branch A: Orijinal Qwen tokenizer ile eğitim
    Branch B: Türkçe tokenizer ile eğitim  
    Final: En iyi performing model seçimi veya ensemble
    """
    
    def __init__(self, base_model="Qwen/Qwen3-8B"):
        self.base_model = base_model
        self.branch_a_results = {}
        self.branch_b_results = {}
        self.final_results = {}
        
    def setup_branch_a_original(self):
        """
        BRANCH A: Original Qwen Tokenizer ile eğitim setup
        Safe, reliable approach
        """
        
        print("🔧 BRANCH A SETUP: Original Qwen Tokenizer")
        print("-" * 50)
        
        branch_a_config = {
            "model_name": self.base_model,
            "tokenizer_type": "original",
            "training_config": {
                "num_epochs": 3,
                "batch_size": 8,
                "gradient_accumulation_steps": 2,
                "learning_rate": 2e-4,
                "warmup_ratio": 0.1,
                "max_length": 1024,
                "lora_r": 16,
                "lora_alpha": 32,
                "modules_to_save": [],  # Safe approach
                "expected_loss": "1.5-2.5",
                "expected_time": "8-12 hours",
                "risk_level": "LOW"
            }
        }
        
        return branch_a_config
    
    def setup_branch_b_turkish(self, turkish_tokenizer_path):
        """
        BRANCH B: Turkish Tokenizer ile eğitim setup
        High reward, high risk approach
        """
        
        print("🔧 BRANCH B SETUP: Turkish Custom Tokenizer")
        print("-" * 50)
        
        # Vocabulary analysis for risk assessment
        risk_assessment = self._assess_turkish_tokenizer_risk(turkish_tokenizer_path)
        
        branch_b_config = {
            "model_name": self.base_model,
            "tokenizer_type": "turkish_custom",
            "tokenizer_path": turkish_tokenizer_path,
            "training_config": {
                "num_epochs": 4,                    # More epochs needed
                "batch_size": 6,                    # Smaller batch for stability
                "gradient_accumulation_steps": 3,   # Maintain effective batch
                "learning_rate": 1e-4,              # Lower LR for stability
                "warmup_ratio": 0.2,                # Longer warmup
                "max_length": 1024,
                "lora_r": 32,                       # Higher rank
                "lora_alpha": 64,
                "modules_to_save": ["embed_tokens", "lm_head"],  # Required for tokenizer change
                "expected_loss": "2.0-4.0",
                "expected_time": "12-18 hours",
                "risk_level": risk_assessment["risk_level"],
                "vocabulary_coverage": risk_assessment["coverage"]
            }
        }
        
        return branch_b_config
    
    def _assess_turkish_tokenizer_risk(self, turkish_tokenizer_path):
        """Turkish tokenizer risk assessment"""
        
        try:
            # Load tokenizers for comparison
            original_tokenizer = AutoTokenizer.from_pretrained(self.base_model, trust_remote_code=True)
            
            import sentencepiece as spm
            sp_model = spm.SentencePieceProcessor()
            sp_model.load(turkish_tokenizer_path)
            
            turkish_tokenizer = LlamaTokenizer(
                vocab_file=turkish_tokenizer_path,
                legacy=False
            )
            
            # Quick vocabulary overlap check
            orig_vocab = set(original_tokenizer.get_vocab().keys())
            turk_vocab = set(turkish_tokenizer.get_vocab().keys())
            overlap = len(orig_vocab & turk_vocab)
            coverage = overlap / len(turk_vocab)
            
            # Risk assessment
            if coverage > 0.6:
                risk_level = "MEDIUM"
            elif coverage > 0.3:
                risk_level = "HIGH"
            else:
                risk_level = "VERY HIGH"
            
            print(f"📊 Turkish Tokenizer Risk Assessment:")
            print(f"  • Vocabulary overlap: {overlap:,} tokens")
            print(f"  • Coverage ratio: {coverage:.2%}")
            print(f"  • Risk level: {risk_level}")
            
            return {
                "coverage": coverage,
                "overlap": overlap,
                "risk_level": risk_level,
                "status": "success"
            }
            
        except Exception as e:
            print(f"❌ Risk assessment failed: {e}")
            return {
                "coverage": 0.0,
                "risk_level": "VERY HIGH",
                "status": "failed",
                "error": str(e)
            }
    
    def train_branch_a(self, turkish_dataset, config):
        """
        Branch A training - Original tokenizer
        Bu function ayrı process'te çalışacak
        """
        
        print("🚀 BRANCH A TRAINING BAŞLIYOR")
        print("=" * 60)
        
        try:
            # Model ve tokenizer setup
            tokenizer = AutoTokenizer.from_pretrained(self.base_model, trust_remote_code=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            model = AutoModelForCausalLM.from_pretrained(
                self.base_model,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
                use_cache=False
            )
            
            # LoRA setup
            lora_config = LoraConfig(
                r=config["training_config"]["lora_r"],
                lora_alpha=config["training_config"]["lora_alpha"],
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                lora_dropout=0.05,
                bias="none",
                task_type=TaskType.CAUSAL_LM,
                modules_to_save=config["training_config"]["modules_to_save"]
            )
            
            model = prepare_model_for_kbit_training(model)
            model = get_peft_model(model, lora_config)
            
            # Training setup
            from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
            
            training_args = TrainingArguments(
                output_dir="./branch_a_original",
                num_train_epochs=config["training_config"]["num_epochs"],
                per_device_train_batch_size=config["training_config"]["batch_size"],
                gradient_accumulation_steps=config["training_config"]["gradient_accumulation_steps"],
                learning_rate=config["training_config"]["learning_rate"],
                warmup_ratio=config["training_config"]["warmup_ratio"],
                logging_steps=25,
                save_steps=500,
                eval_steps=250,
                evaluation_strategy="steps",
                bf16=True,
                tf32=True,
                gradient_checkpointing=True,
                remove_unused_columns=False,
                load_best_model_at_end=True,
                metric_for_best_model="eval_loss",
                greater_is_better=False,
            )
            
            # Data preparation
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=tokenizer,
                mlm=False,
                pad_to_multiple_of=8
            )
            
            def tokenize_function(examples):
                return tokenizer(
                    examples["text"],
                    truncation=True,
                    padding="max_length",
                    max_length=config["training_config"]["max_length"],
                    return_tensors="pt"
                )
            
            # Dataset preparation
            train_dataset = turkish_dataset.map(
                tokenize_function,
                batched=True,
                remove_columns=["text"],
                desc="Branch A Tokenization"
            )
            
            # Trainer setup
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                tokenizer=tokenizer,
                data_collator=data_collator,
            )
            
            # Training execution
            start_time = time.time()
            print("🏃‍♂️ Branch A training başladı...")
            
            result = trainer.train()
            
            training_time = time.time() - start_time
            
            # Model kaydetme
            final_path = "./branch_a_complete"
            trainer.save_model(final_path)
            
            # Results storage
            branch_a_result = {
                "status": "SUCCESS",
                "training_loss": result.training_loss,
                "training_time": training_time,
                "epochs": config["training_config"]["num_epochs"],
                "final_step": result.global_step,
                "model_path": final_path,
                "tokenizer_type": "original_qwen",
                "config": config,
                "timestamp": datetime.now().isoformat()
            }
            
            print(f"✅ BRANCH A TAMAMLANDI!")
            print(f"  • Final Loss: {result.training_loss:.4f}")
            print(f"  • Training Time: {training_time/3600:.2f} hours")
            
            return branch_a_result
            
        except Exception as e:
            error_result = {
                "status": "FAILED",
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
                "config": config
            }
            print(f"❌ BRANCH A FAILED: {e}")
            return error_result
    
    def train_branch_b(self, turkish_dataset, config):
        """
        Branch B training - Turkish tokenizer
        Bu function ayrı process'te çalışacak
        """
        
        print("🚀 BRANCH B TRAINING BAŞLIYOR")
        print("=" * 60)
        
        try:
            # Turkish tokenizer setup
            import sentencepiece as spm
            sp_model = spm.SentencePieceProcessor()
            sp_model.load(config["tokenizer_path"])
            
            tokenizer = LlamaTokenizer(
                vocab_file=config["tokenizer_path"],
                legacy=False,
                add_bos_token=True,
                add_eos_token=True,
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            print(f"✅ Turkish tokenizer loaded: {len(tokenizer):,} tokens")
            
            # Model setup
            model = AutoModelForCausalLM.from_pretrained(
                self.base_model,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
                use_cache=False
            )
            
            # Resize embeddings for new vocabulary
            print("🔧 Resizing token embeddings...")
            model.resize_token_embeddings(len(tokenizer))
            
            # LoRA setup - more aggressive for Turkish adaptation
            lora_config = LoraConfig(
                r=config["training_config"]["lora_r"],
                lora_alpha=config["training_config"]["lora_alpha"],
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                lora_dropout=0.1,
                bias="none",
                task_type=TaskType.CAUSAL_LM,
                modules_to_save=config["training_config"]["modules_to_save"]
            )
            
            model = prepare_model_for_kbit_training(model)
            model = get_peft_model(model, lora_config)
            
            # Training setup
            from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
            
            training_args = TrainingArguments(
                output_dir="./branch_b_turkish",
                num_train_epochs=config["training_config"]["num_epochs"],
                per_device_train_batch_size=config["training_config"]["batch_size"],
                gradient_accumulation_steps=config["training_config"]["gradient_accumulation_steps"],
                learning_rate=config["training_config"]["learning_rate"],
                warmup_ratio=config["training_config"]["warmup_ratio"],
                logging_steps=25,
                save_steps=500,
                eval_steps=250,
                evaluation_strategy="steps",
                bf16=True,
                tf32=True,
                gradient_checkpointing=True,
                remove_unused_columns=False,
                load_best_model_at_end=True,
                metric_for_best_model="eval_loss",
                greater_is_better=False,
            )
            
            # Data preparation
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=tokenizer,
                mlm=False,
                pad_to_multiple_of=8
            )
            
            def tokenize_function_turkish(examples):
                return tokenizer(
                    examples["text"],
                    truncation=True,
                    padding="max_length",
                    max_length=config["training_config"]["max_length"],
                    return_tensors="pt"
                )
            
            # Dataset preparation
            train_dataset = turkish_dataset.map(
                tokenize_function_turkish,
                batched=True,
                remove_columns=["text"],
                desc="Branch B Turkish Tokenization"
            )
            
            # Trainer setup
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                tokenizer=tokenizer,
                data_collator=data_collator,
            )
            
            # Training execution
            start_time = time.time()
            print("🏃‍♂️ Branch B training başladı...")
            
            result = trainer.train()
            
            training_time = time.time() - start_time
            
            # Model kaydetme
            final_path = "./branch_b_complete"
            trainer.save_model(final_path)
            
            # Results storage
            branch_b_result = {
                "status": "SUCCESS",
                "training_loss": result.training_loss,
                "training_time": training_time,
                "epochs": config["training_config"]["num_epochs"],
                "final_step": result.global_step,
                "model_path": final_path,
                "tokenizer_type": "turkish_custom",
                "vocabulary_coverage": config["training_config"]["vocabulary_coverage"],
                "config": config,
                "timestamp": datetime.now().isoformat()
            }
            
            print(f"✅ BRANCH B TAMAMLANDI!")
            print(f"  • Final Loss: {result.training_loss:.4f}")
            print(f"  • Training Time: {training_time/3600:.2f} hours")
            
            return branch_b_result
            
        except Exception as e:
            error_result = {
                "status": "FAILED",
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
                "config": config
            }
            print(f"❌ BRANCH B FAILED: {e}")
            return error_result
    
    def run_parallel_training(self, turkish_dataset, turkish_tokenizer_path):
        """
        İki branch'i paralel olarak çalıştır
        """
        
        print("🚀 PARALLEL HYBRID TRAINING BAŞLIYOR")
        print("=" * 80)
        
        # Setup configurations
        config_a = self.setup_branch_a_original()
        config_b = self.setup_branch_b_turkish(turkish_tokenizer_path)
        
        print(f"\n📊 BRANCH COMPARISON:")
        print(f"Branch A - Risk: {config_a['training_config']['risk_level']}")
        print(f"Branch B - Risk: {config_b['training_config']['risk_level']}")
        print(f"Expected time: {max(12, 18)} hours (paralel)")
        
        # Paralel execution using ProcessPoolExecutor
        print("\n🔄 Starting parallel training...")
        
        start_time = time.time()
        
        with ProcessPoolExecutor(max_workers=2) as executor:
            # Submit both training jobs
            future_a = executor.submit(self.train_branch_a, turkish_dataset, config_a)
            future_b = executor.submit(self.train_branch_b, turkish_dataset, config_b)
            
            print("⏳ Her iki branch de çalışıyor...")
            print("📊 Progress tracking:")
            
            # Wait for completion
            results = {}
            
            # Branch A completion check
            try:
                results['branch_a'] = future_a.result(timeout=43200)  # 12 hours max
                print("✅ Branch A completed!")
            except Exception as e:
                results['branch_a'] = {"status": "FAILED", "error": str(e)}
                print(f"❌ Branch A failed: {e}")
            
            # Branch B completion check  
            try:
                results['branch_b'] = future_b.result(timeout=64800)  # 18 hours max
                print("✅ Branch B completed!")
            except Exception as e:
                results['branch_b'] = {"status": "FAILED", "error": str(e)}
                print(f"❌ Branch B failed: {e}")
        
        total_time = time.time() - start_time
        
        # Store results
        self.branch_a_results = results['branch_a']
        self.branch_b_results = results['branch_b']
        
        # Analysis and comparison
        final_analysis = self.analyze_parallel_results(total_time)
        
        return final_analysis
    
    def analyze_parallel_results(self, total_time):
        """
        İki branch'in sonuçlarını analiz et ve en iyisini seç
        """
        
        print("\n" + "=" * 80)
        print("📊 PARALLEL RESULTS ANALYSIS")
        print("=" * 80)
        
        print(f"\n⏱️ TOTAL PARALLEL TIME: {total_time/3600:.2f} hours")
        
        # Branch A analysis
        print(f"\n🔵 BRANCH A (Original Tokenizer):")
        if self.branch_a_results["status"] == "SUCCESS":
            print(f"  ✅ Status: SUCCESS")
            print(f"  📉 Loss: {self.branch_a_results['training_loss']:.4f}")
            print(f"  ⏱️ Time: {self.branch_a_results['training_time']/3600:.2f}h")
            print(f"  🎯 Tokenizer: Original Qwen")
        else:
            print(f"  ❌ Status: FAILED")
            print(f"  🚨 Error: {self.branch_a_results.get('error', 'Unknown')}")
        
        # Branch B analysis
        print(f"\n🔴 BRANCH B (Turkish Tokenizer):")
        if self.branch_b_results["status"] == "SUCCESS":
            print(f"  ✅ Status: SUCCESS")
            print(f"  📉 Loss: {self.branch_b_results['training_loss']:.4f}")
            print(f"  ⏱️ Time: {self.branch_b_results['training_time']/3600:.2f}h")
            print(f"  🎯 Tokenizer: Turkish Custom")
            print(f"  📊 Vocab Coverage: {self.branch_b_results.get('vocabulary_coverage', 0):.2%}")
        else:
            print(f"  ❌ Status: FAILED")
            print(f"  🚨 Error: {self.branch_b_results.get('error', 'Unknown')}")
        
        # Winner selection
        winner = self.select_winner()
        
        print(f"\n🏆 WINNER SELECTION:")
        print("-" * 50)
        print(f"Selected: {winner['branch'].upper()}")
        print(f"Reason: {winner['reason']}")
        print(f"Final Loss: {winner['loss']:.4f}")
        print(f"Model Path: {winner['model_path']}")
        
        # Final recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        print("-" * 50)
        if winner['branch'] == 'branch_a':
            print("✅ Original tokenizer approach proved more reliable")
            print("✅ Use this model for production")
            print("⚠️ Consider Turkish tokenizer optimization in future")
        else:
            print("🎉 Turkish tokenizer approach succeeded!")
            print("✅ Superior Turkish language performance expected")
            print("✅ Use this model for Turkish-focused applications")
        
        # Save analysis
        analysis_result = {
            "total_parallel_time": total_time,
            "branch_a_results": self.branch_a_results,
            "branch_b_results": self.branch_b_results,
            "winner": winner,
            "timestamp": datetime.now().isoformat(),
            "approach": "parallel_hybrid"
        }
        
        with open("./parallel_hybrid_analysis.json", 'w') as f:
            json.dump(analysis_result, f, indent=2, default=str)
        
        return analysis_result
    
    def select_winner(self):
        """Winner selection logic"""
        
        a_success = self.branch_a_results["status"] == "SUCCESS"
        b_success = self.branch_b_results["status"] == "SUCCESS"
        
        if a_success and b_success:
            # Her ikisi de başarılı - loss karşılaştır
            a_loss = self.branch_a_results["training_loss"]
            b_loss = self.branch_b_results["training_loss"]
            
            if b_loss < a_loss * 1.2:  # %20 tolerans ile Turkish tokenizer'ı tercih et
                return {
                    "branch": "branch_b",
                    "reason": "Turkish tokenizer achieved comparable/better loss",
                    "loss": b_loss,
                    "model_path": self.branch_b_results["model_path"]
                }
            else:
                return {
                    "branch": "branch_a", 
                    "reason": "Original tokenizer achieved significantly better loss",
                    "loss": a_loss,
                    "model_path": self.branch_a_results["model_path"]
                }
        
        elif a_success and not b_success:
            return {
                "branch": "branch_a",
                "reason": "Only original tokenizer succeeded",
                "loss": self.branch_a_results["training_loss"],
                "model_path": self.branch_a_results["model_path"]
            }
        
        elif not a_success and b_success:
            return {
                "branch": "branch_b",
                "reason": "Only Turkish tokenizer succeeded",
                "loss": self.branch_b_results["training_loss"], 
                "model_path": self.branch_b_results["model_path"]
            }
        
        else:
            return {
                "branch": "none",
                "reason": "Both branches failed",
                "loss": float('inf'),
                "model_path": None
            }

# Usage example
def main_parallel_hybrid():
    """Main function for parallel hybrid approach"""
    
    trainer = ParallelHybridTrainer()
    
    print("🎯 PARALLEL HYBRID APPROACH")
    print("Bu yaklaşım iki modeli aynı anda eğitir:")
    print("• Branch A: Original Qwen tokenizer (Safe)")
    print("• Branch B: Turkish custom tokenizer (High reward/risk)")
    print("• Final: En iyi performans gösteren model seçilir")
    
    # You would need to provide these:
    # turkish_dataset = load_turkish_dataset()
    # turkish_tokenizer_path = "/path/to/turkish_mixtral_v3_fixed.model"
    
    # Uncomment when ready:
    # analysis = trainer.run_parallel_training(turkish_dataset, turkish_tokenizer_path)
    
    return trainer

if __name__ == "__main__":
    main_parallel_hybrid()