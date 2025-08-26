# Advanced: Gradual Tokenizer Adaptation for Qwen3-8B
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
import sentencepiece as spm
import numpy as np
import json

print("=" * 60)
print("🔄 GRADUAL TOKENIZER ADAPTATION - ADVANCED SOLUTION")
print("=" * 60)

class GradualTokenizerAdapter:
    """Advanced class for gradual tokenizer adaptation"""
    
    def __init__(self, model_name="Qwen/Qwen3-8B"):
        self.model_name = model_name
        self.original_tokenizer = None
        self.custom_tokenizer = None
        self.model = None
        
    def load_original_setup(self):
        """Load model with original tokenizer first"""
        
        print("📝 Phase 1: Loading Original Qwen Setup...")
        
        # Load original tokenizer
        self.original_tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            use_fast=True
        )
        
        if self.original_tokenizer.pad_token is None:
            self.original_tokenizer.pad_token = self.original_tokenizer.eos_token
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            use_cache=False
        )
        
        print("✅ Original setup loaded successfully")
        print(f"  • Original vocab size: {len(self.original_tokenizer)}")
        print(f"  • Model vocab size: {self.model.config.vocab_size}")
        
        return self.model, self.original_tokenizer
    
    def setup_initial_lora(self):
        """Setup LoRA for initial training with original tokenizer"""
        
        print("\n🔧 Phase 1: Setting up LoRA for original tokenizer...")
        
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            modules_to_save=[]  # Don't touch embeddings yet
        )
        
        self.model = prepare_model_for_kbit_training(self.model)
        self.model = get_peft_model(self.model, lora_config)
        
        print("✅ Initial LoRA setup complete")
        return self.model
    
    def train_phase1_original_tokenizer(self, turkish_dataset):
        """Phase 1: Train with original tokenizer on Turkish data"""
        
        print("\n🎯 Phase 1 Training: Original Tokenizer on Turkish Data")
        print("-" * 50)
        
        # This phase teaches the model Turkish patterns
        # while keeping the original vocabulary
        
        from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
        
        # Training arguments for phase 1
        training_args = TrainingArguments(
            output_dir="./phase1_original_tokenizer",
            num_train_epochs=2,
            per_device_train_batch_size=8,
            gradient_accumulation_steps=2,
            learning_rate=1e-4,  # Conservative LR
            warmup_ratio=0.1,
            logging_steps=25,
            save_steps=500,
            eval_steps=250,
            bf16=True,
            remove_unused_columns=False,
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.original_tokenizer,
            mlm=False,
        )
        
        # Tokenize dataset with original tokenizer
        def tokenize_function(examples):
            return self.original_tokenizer(
                examples["text"],
                truncation=True,
                padding="max_length",
                max_length=1024,
            )
        
        tokenized_dataset = turkish_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=["text"],
        )
        
        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_dataset,
            tokenizer=self.original_tokenizer,
            data_collator=data_collator,
        )
        
        print("🚀 Starting Phase 1 training...")
        result = trainer.train()
        
        print(f"✅ Phase 1 complete - Loss: {result.training_loss:.4f}")
        
        # Save phase 1 model
        trainer.save_model("./phase1_complete")
        
        return result
    
    def load_custom_tokenizer(self, tokenizer_path):
        """Load custom Turkish tokenizer"""
        
        print("\n📝 Phase 2: Loading Custom Turkish Tokenizer...")
        
        try:
            # Load SentencePiece model
            sp_model = spm.SentencePieceProcessor()
            sp_model.load(tokenizer_path)
            
            # Create LlamaTokenizer wrapper
            self.custom_tokenizer = LlamaTokenizer(
                vocab_file=tokenizer_path,
                legacy=False,
                add_bos_token=True,
                add_eos_token=True,
            )
            
            if self.custom_tokenizer.pad_token is None:
                self.custom_tokenizer.pad_token = self.custom_tokenizer.eos_token
            
            print("✅ Custom tokenizer loaded")
            print(f"  • Custom vocab size: {len(self.custom_tokenizer)}")
            print(f"  • Original vocab size: {len(self.original_tokenizer)}")
            
            return self.custom_tokenizer
            
        except Exception as e:
            print(f"❌ Custom tokenizer loading failed: {e}")
            return None
    
    def create_vocabulary_mapping(self):
        """Create intelligent mapping between vocabularies"""
        
        print("\n🔗 Creating Vocabulary Mapping...")
        
        # Get vocabulary overlaps
        original_vocab = set(self.original_tokenizer.get_vocab().keys())
        custom_vocab = set(self.custom_tokenizer.get_vocab().keys())
        
        # Find common tokens
        common_tokens = original_vocab & custom_vocab
        overlap_ratio = len(common_tokens) / len(original_vocab)
        
        print(f"📊 Vocabulary Analysis:")
        print(f"  • Original vocab: {len(original_vocab)} tokens")
        print(f"  • Custom vocab: {len(custom_vocab)} tokens")
        print(f"  • Common tokens: {len(common_tokens)} tokens")
        print(f"  • Overlap ratio: {overlap_ratio:.2%}")
        
        # Create mapping strategy
        if overlap_ratio > 0.5:
            print("✅ Good overlap - can create meaningful mapping")
            return self._create_smart_mapping()
        else:
            print("⚠️ Low overlap - gradual adaptation will be challenging")
            return None
    
    def _create_smart_mapping(self):
        """Create smart embedding initialization"""
        
        print("🧠 Creating smart embedding mapping...")
        
        # Initialize new embedding layer
        old_embeddings = self.model.get_input_embeddings().weight.data
        new_vocab_size = len(self.custom_tokenizer)
        embedding_dim = old_embeddings.size(1)
        
        # Create new embedding matrix
        new_embeddings = torch.randn(new_vocab_size, embedding_dim, dtype=old_embeddings.dtype)
        
        # Map common tokens
        mapped_count = 0
        for token, new_id in self.custom_tokenizer.get_vocab().items():
            if token in self.original_tokenizer.get_vocab():
                old_id = self.original_tokenizer.get_vocab()[token]
                new_embeddings[new_id] = old_embeddings[old_id]
                mapped_count += 1
        
        print(f"✅ Mapped {mapped_count} tokens to preserve learned representations")
        
        return new_embeddings
    
    def gradual_adaptation_phase2(self, new_embeddings, turkish_dataset):
        """Phase 2: Gradual adaptation to new tokenizer"""
        
        print("\n🔄 Phase 2: Gradual Adaptation to Custom Tokenizer")
        print("-" * 50)
        
        # Resize model embeddings
        self.model.resize_token_embeddings(len(self.custom_tokenizer))
        
        # Initialize with mapped embeddings
        if new_embeddings is not None:
            with torch.no_grad():
                self.model.get_input_embeddings().weight.copy_(new_embeddings)
            print("✅ Embeddings initialized with vocabulary mapping")
        
        # Create new LoRA config that includes embeddings
        from peft import get_peft_model
        
        lora_config_phase2 = LoraConfig(
            r=32,  # Higher rank for vocabulary adaptation
            lora_alpha=64,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.1,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            modules_to_save=["embed_tokens", "lm_head"]  # Now include embeddings
        )
        
        # Apply new LoRA
        self.model = get_peft_model(self.model, lora_config_phase2)
        
        print("🔧 Phase 2 LoRA configuration applied")
        
        # Training with very conservative settings
        from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
        
        training_args = TrainingArguments(
            output_dir="./phase2_custom_tokenizer",
            num_train_epochs=1,  # Just 1 epoch for adaptation
            per_device_train_batch_size=4,  # Smaller batch
            gradient_accumulation_steps=4,
            learning_rate=5e-5,  # Very low LR
            warmup_ratio=0.2,    # Long warmup
            logging_steps=10,
            save_steps=250,
            bf16=True,
            remove_unused_columns=False,
        )
        
        # Data collator with custom tokenizer
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.custom_tokenizer,
            mlm=False,
        )
        
        # Tokenize with custom tokenizer
        def tokenize_function(examples):
            return self.custom_tokenizer(
                examples["text"],
                truncation=True,
                padding="max_length",
                max_length=1024,
            )
        
        tokenized_dataset = turkish_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=["text"],
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_dataset,
            tokenizer=self.custom_tokenizer,
            data_collator=data_collator,
        )
        
        print("🚀 Starting Phase 2 adaptation...")
        result = trainer.train()
        
        print(f"✅ Phase 2 complete - Loss: {result.training_loss:.4f}")
        
        # Save final model
        trainer.save_model("./phase2_adapted")
        
        return result

def main_gradual_adaptation():
    """Main function for gradual adaptation approach"""
    
    print("🎯 GRADUAL TOKENIZER ADAPTATION APPROACH")
    print("This method minimizes vocabulary learning disruption")
    print("=" * 60)
    
    # Initialize adapter
    adapter = GradualTokenizerAdapter()
    
    # Phase 1: Original tokenizer training
    model, original_tokenizer = adapter.load_original_setup()
    model = adapter.setup_initial_lora()
    
    # Load Turkish dataset (you need to implement this)
    # turkish_dataset = load_turkish_dataset()
    
    # Uncomment below lines when dataset is ready:
    # result1 = adapter.train_phase1_original_tokenizer(turkish_dataset)
    
    # Phase 2: Custom tokenizer adaptation
    # custom_tokenizer = adapter.load_custom_tokenizer("/path/to/turkish_mixtral_v3_fixed.model")
    # mapping = adapter.create_vocabulary_mapping()
    # new_embeddings = adapter._create_smart_mapping() if mapping else None
    # result2 = adapter.gradual_adaptation_phase2(new_embeddings, turkish_dataset)
    
    print("\n✅ Gradual adaptation complete!")
    print("📊 Expected improvement: Loss should be 2.0-3.5 vs 5+ with direct switch")
    
    return adapter

if __name__ == "__main__":
    main_gradual_adaptation()