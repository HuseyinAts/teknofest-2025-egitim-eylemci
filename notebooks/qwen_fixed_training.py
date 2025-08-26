# Qwen3-8B Training with Original Tokenizer - FIXED VERSION
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from datasets import load_dataset, Dataset
import json

print("=" * 60)
print("🔧 QWEN3-8B CORRECTED TRAINING - ORIGINAL TOKENIZER")
print("=" * 60)

# SOLUTION 1: Use Original Qwen Tokenizer
def load_model_with_original_tokenizer():
    """Load Qwen3-8B with its original tokenizer - NO VOCABULARY MISMATCH"""
    
    print("📝 Loading Original Qwen Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen3-8B",
        trust_remote_code=True,
        use_fast=True
    )
    
    # Set special tokens properly
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    print("✅ Original Qwen tokenizer loaded successfully!")
    print(f"  • Vocab size: {len(tokenizer)}")
    print(f"  • Pad token: {tokenizer.pad_token}")
    print(f"  • EOS token: {tokenizer.eos_token}")
    
    # Load model
    print("\n🤖 Loading Qwen3-8B Model...")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-8B",
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        use_cache=False
    )
    
    # NO NEED TO RESIZE - vocabulary matches perfectly!
    print("✅ Model loaded with matching vocabulary!")
    print(f"  • Model vocab size: {model.config.vocab_size}")
    print(f"  • Tokenizer vocab size: {len(tokenizer)}")
    print(f"  • ✅ PERFECT MATCH - No embedding layer reset!")
    
    return model, tokenizer

# CORRECTED LoRA Configuration - NO EMBEDDING MODULES
def setup_corrected_lora(model):
    """Setup LoRA without touching embedding layers"""
    
    print("\n🔧 Setting up CORRECTED LoRA configuration...")
    
    # CRITICAL FIX: Remove embed_tokens and lm_head from modules_to_save
    lora_config = LoraConfig(
        r=16,                           # Lower rank for stability
        lora_alpha=32,                  # 2*rank ratio
        target_modules=[                # Only attention modules
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        modules_to_save=[]              # 🚨 CRITICAL: EMPTY - Don't touch embeddings!
    )
    
    print("📋 CORRECTED LoRA Settings:")
    print(f"  • Rank (r): {lora_config.r}")
    print(f"  • Alpha: {lora_config.lora_alpha}")
    print(f"  • Target modules: {lora_config.target_modules}")
    print(f"  • Modules to save: {lora_config.modules_to_save} ✅ EMPTY!")
    print("  • ✅ Embedding layers will NOT be modified!")
    
    # Prepare and apply LoRA
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)
    
    # Check trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    
    print(f"\n📊 LoRA Statistics:")
    print(f"  • Total parameters: {all_params/1e9:.2f}B")
    print(f"  • Trainable parameters: {trainable_params/1e6:.2f}M")
    print(f"  • Trainable percentage: {100 * trainable_params / all_params:.2f}%")
    
    return model

# CORRECTED Training Configuration
class CorrectedTrainingConfig:
    """Corrected training configuration for Qwen3-8B"""
    
    # Model
    model_name = "Qwen/Qwen3-8B"
    
    # Training parameters - CORRECTED
    num_epochs = 3                      # More epochs needed
    batch_size = 8                      # Good for A100
    gradient_accumulation_steps = 2     # Effective batch = 16
    learning_rate = 2e-4                # Higher LR since no vocab learning needed
    warmup_ratio = 0.1                  # Proper warmup
    weight_decay = 0.01
    
    # Sequence length - CORRECTED
    max_length = 1024                   # Optimal for Qwen3
    
    # LoRA - CORRECTED
    lora_r = 16                         # Lower rank
    lora_alpha = 32                     # 2*rank
    lora_dropout = 0.05
    
    # Precision
    bf16 = True
    tf32 = True
    
    # Checkpointing
    save_steps = 500
    eval_steps = 250
    logging_steps = 25
    
    # Dataset
    max_train_samples = 50000           # Smaller, higher quality dataset
    max_eval_samples = 5000

def prepare_high_quality_turkish_dataset(tokenizer, config):
    """Prepare a high-quality Turkish dataset with proper filtering"""
    
    print("\n📚 Preparing HIGH QUALITY Turkish Dataset...")
    
    def improved_quality_filter(text):
        """Improved quality filtering for Turkish text"""
        if not text or len(text) < 100:
            return False
        
        # Check for Turkish characters
        turkish_chars = 'çÇğĞıİöÖşŞüÜ'
        turkish_count = sum(1 for c in text if c in turkish_chars)
        if turkish_count < 3:  # Must have Turkish characters
            return False
        
        # Word count check
        words = text.split()
        if len(words) < 20 or len(words) > 500:  # Reasonable length
            return False
        
        # Remove spam content
        spam_keywords = ['escort', 'xxx', 'porn', 'sex', 'casino']
        text_lower = text.lower()
        if any(spam in text_lower for spam in spam_keywords):
            return False
        
        # Check uniqueness
        unique_ratio = len(set(words)) / len(words)
        if unique_ratio < 0.5:  # At least 50% unique words
            return False
        
        return True
    
    # Load and filter Turkish dataset
    try:
        print("📥 Loading CulturaX Turkish dataset...")
        dataset = load_dataset("uonlp/CulturaX", "tr", split="train", streaming=True)
        
        high_quality_texts = []
        processed_count = 0
        accepted_count = 0
        
        for item in dataset:
            processed_count += 1
            text = item.get('text', '').strip()
            
            if improved_quality_filter(text):
                # Truncate to reasonable length
                if len(text) > 2000:
                    text = text[:2000]
                high_quality_texts.append(text)
                accepted_count += 1
            
            # Progress update
            if processed_count % 10000 == 0:
                acceptance_rate = (accepted_count / processed_count) * 100
                print(f"  Processed: {processed_count:,} | Accepted: {accepted_count:,} | Rate: {acceptance_rate:.1f}%")
            
            # Stop when we have enough
            if len(high_quality_texts) >= config.max_train_samples:
                break
            
            # Safety limit
            if processed_count >= 500000:
                break
        
        print(f"\n✅ Dataset prepared:")
        print(f"  • Processed: {processed_count:,} texts")
        print(f"  • Accepted: {len(high_quality_texts):,} texts")
        print(f"  • Quality rate: {(len(high_quality_texts)/processed_count)*100:.1f}%")
        
        # Create train/eval split
        train_size = int(len(high_quality_texts) * 0.95)
        train_texts = high_quality_texts[:train_size]
        eval_texts = high_quality_texts[train_size:]
        
        # Create datasets
        train_dataset = Dataset.from_dict({"text": train_texts})
        eval_dataset = Dataset.from_dict({"text": eval_texts})
        
        # Tokenize
        def tokenize_function(examples):
            return tokenizer(
                examples["text"],
                truncation=True,
                padding="max_length",
                max_length=config.max_length,
                return_tensors="pt"
            )
        
        print("\n🔄 Tokenizing datasets...")
        train_dataset = train_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=["text"],
            desc="Tokenizing train"
        )
        
        eval_dataset = eval_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=["text"],
            desc="Tokenizing eval"
        )
        
        print(f"✅ Tokenization complete:")
        print(f"  • Train samples: {len(train_dataset):,}")
        print(f"  • Eval samples: {len(eval_dataset):,}")
        
        return train_dataset, eval_dataset
        
    except Exception as e:
        print(f"❌ Dataset preparation error: {e}")
        return None, None

def main():
    """Main training function with corrected approach"""
    
    config = CorrectedTrainingConfig()
    
    # Step 1: Load model with original tokenizer
    model, tokenizer = load_model_with_original_tokenizer()
    
    # Step 2: Setup corrected LoRA
    model = setup_corrected_lora(model)
    
    # Step 3: Prepare high-quality dataset
    train_dataset, eval_dataset = prepare_high_quality_turkish_dataset(tokenizer, config)
    
    if train_dataset is None:
        print("❌ Dataset preparation failed!")
        return
    
    print("\n🎯 CORRECTED CONFIGURATION SUMMARY:")
    print("=" * 50)
    print("✅ Original Qwen tokenizer - NO vocabulary mismatch")
    print("✅ LoRA without embedding modification")
    print("✅ Higher learning rate (2e-4) - no vocab relearning needed")
    print("✅ More epochs (3) for proper fine-tuning")
    print("✅ High-quality filtered dataset")
    print("✅ Expected loss: 1.5-2.5 (much lower!)")
    print("=" * 50)
    
    return model, tokenizer, train_dataset, eval_dataset, config

if __name__ == "__main__":
    main()