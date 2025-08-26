# Sequential Hybrid Approach - Ultra Detailed Implementation
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
import json
import time
from datetime import datetime
import os

class SequentialHybridTrainer:
    """
    Sekansiyel Hibrit Yaklaşım - Aşama aşama tokenizer geçişi
    
    Aşama 1: Orijinal Qwen tokenizer ile Türkçe öğretme (Foundation)
    Aşama 2: Vocabulary mapping ile geçiş hazırlığı 
    Aşama 3: Türkçe tokenizer ile fine-tuning
    """
    
    def __init__(self, base_model="Qwen/Qwen3-8B"):
        self.base_model = base_model
        self.model = None
        self.original_tokenizer = None
        self.turkish_tokenizer = None
        self.phase_results = {}
        self.checkpoints = {}
        
    def phase1_foundation_training(self, turkish_dataset, epochs=3):
        """
        AŞAMA 1: FOUNDATION TRAINING
        Orijinal Qwen tokenizer ile güçlü Türkçe foundation oluşturma
        
        Hedefler:
        - Türkçe patterns öğretme
        - Pre-trained knowledge koruma
        - Stable baseline oluşturma
        """
        
        print("🎯 AŞAMA 1: FOUNDATION TRAINING BAŞLIYOR")
        print("=" * 60)
        
        # Orijinal tokenizer ve model yükleme
        self.original_tokenizer = AutoTokenizer.from_pretrained(
            self.base_model, trust_remote_code=True
        )
        if self.original_tokenizer.pad_token is None:
            self.original_tokenizer.pad_token = self.original_tokenizer.eos_token
            
        self.model = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            use_cache=False
        )
        
        # Conservative LoRA configuration
        lora_config_phase1 = LoraConfig(
            r=16,                           # Düşük rank - stability için
            lora_alpha=32,                  # 2*rank oranı
            target_modules=[                # Sadece attention layers
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            modules_to_save=[]              # 🚨 Embedding'lere dokunma!
        )
        
        # Model preparation
        self.model = prepare_model_for_kbit_training(self.model)
        self.model = get_peft_model(self.model, lora_config_phase1)
        
        print(f"📊 Phase 1 Model Stats:")
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"  • Trainable: {trainable_params/1e6:.1f}M ({100*trainable_params/total_params:.2f}%)")
        print(f"  • Vocabulary: {len(self.original_tokenizer):,} tokens")
        
        # Training configuration
        from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
        
        training_args_phase1 = TrainingArguments(
            output_dir="./phase1_foundation",
            num_train_epochs=epochs,
            per_device_train_batch_size=8,
            gradient_accumulation_steps=2,
            learning_rate=2e-4,             # Yüksek LR - vocab learning yok
            warmup_ratio=0.1,
            weight_decay=0.01,
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
            tokenizer=self.original_tokenizer,
            mlm=False,
            pad_to_multiple_of=8
        )
        
        def tokenize_function(examples):
            return self.original_tokenizer(
                examples["text"],
                truncation=True,
                padding="max_length",
                max_length=1024,
                return_tensors="pt"
            )
        
        # Dataset tokenization
        train_dataset = turkish_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=["text"],
            desc="Phase 1 Tokenization"
        )
        
        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args_phase1,
            train_dataset=train_dataset,
            tokenizer=self.original_tokenizer,
            data_collator=data_collator,
        )
        
        # Training execution
        print("🚀 Phase 1 Training başlıyor...")
        start_time = time.time()
        
        result = trainer.train()
        
        training_time = time.time() - start_time
        
        # Results storage
        self.phase_results["phase1"] = {
            "training_loss": result.training_loss,
            "training_time": training_time,
            "epochs": epochs,
            "final_step": result.global_step,
            "tokenizer": "original_qwen",
            "status": "completed"
        }
        
        # Save checkpoint
        checkpoint_path = "./phase1_foundation_complete"
        trainer.save_model(checkpoint_path)
        self.checkpoints["phase1"] = checkpoint_path
        
        print(f"✅ AŞAMA 1 TAMAMLANDI!")
        print(f"  • Final Loss: {result.training_loss:.4f}")
        print(f"  • Training Time: {training_time/3600:.2f} hours")
        print(f"  • Checkpoint: {checkpoint_path}")
        
        return result
    
    def phase2_vocabulary_analysis(self, turkish_tokenizer_path):
        """
        AŞAMA 2: VOCABULARY ANALYSIS & MAPPING
        İki tokenizer arasında intelligent mapping oluşturma
        """
        
        print("\n🔬 AŞAMA 2: VOCABULARY ANALYSIS BAŞLIYOR")
        print("=" * 60)
        
        # Load Turkish tokenizer
        try:
            import sentencepiece as spm
            sp_model = spm.SentencePieceProcessor()
            sp_model.load(turkish_tokenizer_path)
            
            self.turkish_tokenizer = LlamaTokenizer(
                vocab_file=turkish_tokenizer_path,
                legacy=False,
                add_bos_token=True,
                add_eos_token=True,
            )
            if self.turkish_tokenizer.pad_token is None:
                self.turkish_tokenizer.pad_token = self.turkish_tokenizer.eos_token
                
            print("✅ Turkish tokenizer loaded successfully")
            
        except Exception as e:
            print(f"❌ Turkish tokenizer loading failed: {e}")
            return None
        
        # Vocabulary analysis
        original_vocab = self.original_tokenizer.get_vocab()
        turkish_vocab = self.turkish_tokenizer.get_vocab()
        
        print(f"\n📊 Vocabulary Comparison:")
        print(f"  • Original Qwen: {len(original_vocab):,} tokens")
        print(f"  • Turkish: {len(turkish_vocab):,} tokens")
        
        # Find exact matches
        exact_matches = {}
        for token, turkish_id in turkish_vocab.items():
            if token in original_vocab:
                exact_matches[token] = {
                    'original_id': original_vocab[token],
                    'turkish_id': turkish_id,
                    'token': token
                }
        
        # Find partial matches (substrings, similar tokens)
        partial_matches = {}
        for turkish_token in turkish_vocab.keys():
            if turkish_token not in exact_matches:
                for original_token in original_vocab.keys():
                    if (len(turkish_token) > 3 and len(original_token) > 3 and
                        (turkish_token in original_token or original_token in turkish_token or
                         self._calculate_similarity(turkish_token, original_token) > 0.7)):
                        partial_matches[turkish_token] = {
                            'original_token': original_token,
                            'original_id': original_vocab[original_token],
                            'turkish_id': turkish_vocab[turkish_token],
                            'similarity': self._calculate_similarity(turkish_token, original_token)
                        }
                        break
        
        # Analysis results
        coverage_stats = {
            'exact_matches': len(exact_matches),
            'partial_matches': len(partial_matches),
            'total_turkish_vocab': len(turkish_vocab),
            'exact_coverage': len(exact_matches) / len(turkish_vocab),
            'total_coverage': (len(exact_matches) + len(partial_matches)) / len(turkish_vocab)
        }
        
        print(f"\n📈 Coverage Analysis:")
        print(f"  • Exact matches: {coverage_stats['exact_matches']:,} ({coverage_stats['exact_coverage']:.2%})")
        print(f"  • Partial matches: {coverage_stats['partial_matches']:,}")
        print(f"  • Total coverage: {coverage_stats['total_coverage']:.2%}")
        
        # Store analysis results
        self.phase_results["phase2"] = {
            "exact_matches": exact_matches,
            "partial_matches": partial_matches,
            "coverage_stats": coverage_stats,
            "status": "completed"
        }
        
        # Save analysis
        analysis_path = "./phase2_vocabulary_analysis.json"
        with open(analysis_path, 'w', encoding='utf-8') as f:
            json.dump({
                'coverage_stats': coverage_stats,
                'exact_match_count': len(exact_matches),
                'partial_match_count': len(partial_matches),
                'analysis_timestamp': datetime.now().isoformat()
            }, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Analysis saved: {analysis_path}")
        
        return coverage_stats['total_coverage']
    
    def _calculate_similarity(self, str1, str2):
        """Simple string similarity calculation"""
        from difflib import SequenceMatcher
        return SequenceMatcher(None, str1, str2).ratio()
    
    def phase3_gradual_adaptation(self, turkish_dataset, coverage_ratio):
        """
        AŞAMA 3: GRADUAL ADAPTATION
        Turkish tokenizer'a kademeli geçiş
        """
        
        print(f"\n🔄 AŞAMA 3: GRADUAL ADAPTATION BAŞLIYOR")
        print(f"Coverage Ratio: {coverage_ratio:.2%}")
        print("=" * 60)
        
        if coverage_ratio < 0.3:
            print("⚠️ WARNING: Low coverage ratio (<30%). High risk!")
            response = input("Continue anyway? (y/N): ")
            if response.lower() != 'y':
                return None
        
        # Load Phase 1 model
        print("📂 Loading Phase 1 model...")
        phase1_path = self.checkpoints["phase1"]
        
        # Create embedding mapping
        print("🧠 Creating smart embedding mapping...")
        new_embeddings = self._create_smart_embeddings()
        
        if new_embeddings is None:
            print("❌ Embedding mapping failed!")
            return None
        
        # Resize model vocabulary
        print("🔧 Resizing model vocabulary...")
        self.model.resize_token_embeddings(len(self.turkish_tokenizer))
        
        # Apply new embeddings
        with torch.no_grad():
            self.model.get_input_embeddings().weight.copy_(new_embeddings)
            
            # Also update output embeddings if exists
            if hasattr(self.model, 'lm_head'):
                new_output_embeddings = new_embeddings.clone()
                self.model.lm_head.weight.copy_(new_output_embeddings)
        
        print("✅ Embedding mapping applied")
        
        # Phase 3 LoRA configuration - more aggressive
        lora_config_phase3 = LoraConfig(
            r=32,                           # Higher rank for adaptation
            lora_alpha=64,                  # 2*rank
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ],
            lora_dropout=0.1,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            modules_to_save=["embed_tokens", "lm_head"]  # Now include embeddings
        )
        
        # Apply Phase 3 LoRA
        self.model = get_peft_model(self.model, lora_config_phase3)
        
        # Phase 3 training configuration
        from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
        
        # Very conservative training for adaptation
        training_args_phase3 = TrainingArguments(
            output_dir="./phase3_adaptation",
            num_train_epochs=2,              # Just 2 epochs for adaptation
            per_device_train_batch_size=4,   # Smaller batch
            gradient_accumulation_steps=4,   # Maintain effective batch size
            learning_rate=5e-5,              # Very low LR
            warmup_ratio=0.3,                # Long warmup
            weight_decay=0.01,
            logging_steps=10,
            save_steps=250,
            eval_steps=125,
            evaluation_strategy="steps",
            bf16=True,
            tf32=True,
            gradient_checkpointing=True,
            remove_unused_columns=False,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
        )
        
        # Data collator for Turkish tokenizer
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.turkish_tokenizer,
            mlm=False,
            pad_to_multiple_of=8
        )
        
        def tokenize_function_turkish(examples):
            return self.turkish_tokenizer(
                examples["text"],
                truncation=True,
                padding="max_length",
                max_length=1024,
                return_tensors="pt"
            )
        
        # Tokenize with Turkish tokenizer
        train_dataset_phase3 = turkish_dataset.map(
            tokenize_function_turkish,
            batched=True,
            remove_columns=["text"],
            desc="Phase 3 Turkish Tokenization"
        )
        
        # Create trainer
        trainer_phase3 = Trainer(
            model=self.model,
            args=training_args_phase3,
            train_dataset=train_dataset_phase3,
            tokenizer=self.turkish_tokenizer,
            data_collator=data_collator,
        )
        
        # Training execution
        print("🚀 Phase 3 Adaptation training başlıyor...")
        start_time = time.time()
        
        try:
            result = trainer_phase3.train()
            training_time = time.time() - start_time
            
            # Store results
            self.phase_results["phase3"] = {
                "training_loss": result.training_loss,
                "training_time": training_time,
                "epochs": 2,
                "final_step": result.global_step,
                "tokenizer": "turkish_custom",
                "coverage_ratio": coverage_ratio,
                "status": "completed"
            }
            
            # Save final model
            final_path = "./phase3_turkish_adapted"
            trainer_phase3.save_model(final_path)
            self.checkpoints["phase3"] = final_path
            
            print(f"✅ AŞAMA 3 TAMAMLANDI!")
            print(f"  • Final Loss: {result.training_loss:.4f}")
            print(f"  • Training Time: {training_time/3600:.2f} hours")
            print(f"  • Final Model: {final_path}")
            
            return result
            
        except Exception as e:
            print(f"❌ Phase 3 training failed: {e}")
            self.phase_results["phase3"] = {
                "status": "failed",
                "error": str(e),
                "coverage_ratio": coverage_ratio
            }
            return None
    
    def _create_smart_embeddings(self):
        """Create smart embedding initialization"""
        
        if "phase2" not in self.phase_results:
            print("❌ Phase 2 analysis not found!")
            return None
        
        exact_matches = self.phase_results["phase2"]["exact_matches"]
        partial_matches = self.phase_results["phase2"]["partial_matches"]
        
        # Get current embeddings
        current_embeddings = self.model.get_input_embeddings().weight.data
        embedding_dim = current_embeddings.size(1)
        new_vocab_size = len(self.turkish_tokenizer)
        
        # Initialize new embedding matrix
        device = current_embeddings.device
        dtype = current_embeddings.dtype
        new_embeddings = torch.randn(
            new_vocab_size, embedding_dim,
            device=device, dtype=dtype
        ) * 0.02  # Small variance
        
        # Copy exact matches
        exact_copied = 0
        for token_info in exact_matches.values():
            orig_id = token_info['original_id']
            turk_id = token_info['turkish_id']
            if orig_id < current_embeddings.size(0) and turk_id < new_vocab_size:
                new_embeddings[turk_id] = current_embeddings[orig_id].clone()
                exact_copied += 1
        
        # Handle partial matches
        partial_copied = 0
        for token, match_info in partial_matches.items():
            orig_id = match_info['original_id']
            turk_id = match_info['turkish_id']
            similarity = match_info['similarity']
            
            if orig_id < current_embeddings.size(0) and turk_id < new_vocab_size:
                # Weighted combination based on similarity
                base_embedding = current_embeddings[orig_id].clone()
                noise = torch.randn_like(base_embedding) * (1 - similarity) * 0.1
                new_embeddings[turk_id] = base_embedding + noise
                partial_copied += 1
        
        print(f"✅ Embedding mapping complete:")
        print(f"  • Exact mappings: {exact_copied}")
        print(f"  • Partial mappings: {partial_copied}")
        print(f"  • Total mapped: {exact_copied + partial_copied}/{new_vocab_size}")
        
        return new_embeddings
    
    def generate_final_report(self):
        """Generate comprehensive final report"""
        
        print("\n" + "=" * 80)
        print("📊 SEQUENTIAL HYBRID APPROACH - FINAL REPORT")
        print("=" * 80)
        
        total_time = 0
        for phase, results in self.phase_results.items():
            if "training_time" in results:
                total_time += results["training_time"]
        
        print(f"\n⏱️ TOPLAM SÜRE: {total_time/3600:.2f} hours")
        
        print(f"\n📈 PHASE RESULTS:")
        print("-" * 50)
        
        for phase, results in self.phase_results.items():
            print(f"\n{phase.upper()}:")
            if results["status"] == "completed":
                print(f"  ✅ Status: SUCCESS")
                if "training_loss" in results:
                    print(f"  📉 Loss: {results['training_loss']:.4f}")
                if "training_time" in results:
                    print(f"  ⏱️ Time: {results['training_time']/3600:.2f}h")
                if "tokenizer" in results:
                    print(f"  🔤 Tokenizer: {results['tokenizer']}")
            else:
                print(f"  ❌ Status: {results['status'].upper()}")
                if "error" in results:
                    print(f"  🚨 Error: {results['error']}")
        
        # Performance comparison
        if "phase1" in self.phase_results and "phase3" in self.phase_results:
            phase1_loss = self.phase_results["phase1"].get("training_loss", 0)
            phase3_loss = self.phase_results["phase3"].get("training_loss", 0)
            
            if phase1_loss > 0 and phase3_loss > 0:
                improvement = ((phase1_loss - phase3_loss) / phase1_loss) * 100
                print(f"\n📊 PERFORMANCE IMPROVEMENT:")
                print(f"  Phase 1 Loss: {phase1_loss:.4f}")
                print(f"  Phase 3 Loss: {phase3_loss:.4f}")
                print(f"  Improvement: {improvement:+.1f}%")
        
        # Success assessment
        success_phases = sum(1 for r in self.phase_results.values() if r["status"] == "completed")
        total_phases = len(self.phase_results)
        
        print(f"\n🎯 SUCCESS RATE: {success_phases}/{total_phases} phases completed")
        
        if success_phases == total_phases:
            print("🏆 SEQUENTIAL HYBRID APPROACH: FULL SUCCESS!")
        elif success_phases >= 2:
            print("✅ SEQUENTIAL HYBRID APPROACH: PARTIAL SUCCESS")
        else:
            print("❌ SEQUENTIAL HYBRID APPROACH: FAILED")
        
        return self.phase_results

# Usage example
def main_sequential_hybrid():
    """Main function for sequential hybrid approach"""
    
    trainer = SequentialHybridTrainer()
    
    # You would need to provide turkish_dataset and turkish_tokenizer_path
    # turkish_dataset = load_turkish_dataset()
    # turkish_tokenizer_path = "/path/to/turkish_mixtral_v3_fixed.model"
    
    print("🎯 SEQUENTIAL HYBRID APPROACH BAŞLIYOR")
    print("Bu yaklaşım 3 aşamada gerçekleştirilir:")
    print("1. Foundation training (Original tokenizer)")
    print("2. Vocabulary analysis & mapping")  
    print("3. Gradual adaptation (Turkish tokenizer)")
    
    # Uncomment when ready:
    # result1 = trainer.phase1_foundation_training(turkish_dataset)
    # coverage = trainer.phase2_vocabulary_analysis(turkish_tokenizer_path)
    # result3 = trainer.phase3_gradual_adaptation(turkish_dataset, coverage)
    # final_report = trainer.generate_final_report()
    
    return trainer

if __name__ == "__main__":
    main_sequential_hybrid()