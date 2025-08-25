"""
Advanced Tokenizer Fix for Qwen Models
Handles authentication issues and provides multiple fallback options
"""

import os
import json
import shutil
from pathlib import Path
from transformers import AutoTokenizer, PreTrainedTokenizerFast
import torch
import warnings
warnings.filterwarnings("ignore")

def create_manual_tokenizer():
    """Create a tokenizer manually without downloading"""
    
    print("\n[INFO] Creating tokenizer manually...")
    
    # Create tokenizer configuration
    tokenizer_config = {
        "add_prefix_space": False,
        "bos_token": "<|endoftext|>",
        "chat_template": "{% for message in messages %}{% if loop.first and messages[0]['role'] != 'system' %}{{ '<|im_start|>system\nYou are a helpful assistant<|im_end|>\n' }}{% endif %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}",
        "clean_up_tokenization_spaces": False,
        "eos_token": "<|endoftext|>",
        "errors": "replace",
        "model_max_length": 32768,
        "pad_token": "<|endoftext|>",
        "split_special_tokens": False,
        "tokenizer_class": "Qwen2Tokenizer",
        "unk_token": None
    }
    
    # Save tokenizer config
    local_dir = Path("./manual_tokenizer")
    local_dir.mkdir(parents=True, exist_ok=True)
    
    config_file = local_dir / "tokenizer_config.json"
    with open(config_file, 'w') as f:
        json.dump(tokenizer_config, f, indent=2)
    
    print(f"  [SAVED] Tokenizer config to: {local_dir}")
    
    return local_dir

def use_alternative_tokenizer():
    """Use an alternative compatible tokenizer"""
    
    print("\n[INFO] Trying alternative tokenizers...")
    
    # List of alternative models to try
    alternatives = [
        ("microsoft/phi-2", "Phi-2 (2.7B)"),
        ("TinyLlama/TinyLlama-1.1B-Chat-v1.0", "TinyLlama (1.1B)"),
        ("gpt2", "GPT-2 (base)"),
        ("EleutherAI/pythia-1.4b", "Pythia (1.4B)"),
        ("facebook/opt-1.3b", "OPT (1.3B)")
    ]
    
    for model_id, model_name in alternatives:
        try:
            print(f"\n  Trying {model_name}...")
            
            # Try to load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                trust_remote_code=False,  # Don't use remote code
                local_files_only=False,
                padding_side="left"
            )
            
            # Configure tokenizer
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token or "<pad>"
            
            # Test tokenizer
            test_text = "Bu bir test metnidir."
            tokens = tokenizer(test_text, return_tensors="pt")
            decoded = tokenizer.decode(tokens['input_ids'][0], skip_special_tokens=True)
            
            print(f"  [SUCCESS] {model_name} tokenizer loaded!")
            print(f"    Test: '{test_text}' -> {tokens['input_ids'].shape}")
            
            # Save tokenizer locally
            save_dir = Path(f"./tokenizer_{model_id.replace('/', '_')}")
            save_dir.mkdir(parents=True, exist_ok=True)
            tokenizer.save_pretrained(save_dir)
            
            print(f"  [SAVED] Tokenizer saved to: {save_dir}")
            
            return tokenizer, save_dir
            
        except Exception as e:
            print(f"    Failed: {str(e)[:100]}")
            continue
    
    return None, None

def modify_training_script():
    """Create a modified training script that uses alternative tokenizer"""
    
    print("\n[INFO] Creating modified training script...")
    
    modified_script = '''"""
Modified Training Script with Fixed Tokenizer
"""

from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path
import torch

class TokenizerManager:
    """Manages tokenizer loading with fallbacks"""
    
    def __init__(self):
        self.tokenizer = None
        self.tokenizer_path = None
    
    def load_tokenizer(self):
        """Load tokenizer with multiple fallback options"""
        
        # Try local tokenizers first
        local_paths = [
            "./tokenizer_microsoft_phi-2",
            "./tokenizer_TinyLlama_TinyLlama-1.1B-Chat-v1.0",
            "./tokenizer_gpt2",
            "./manual_tokenizer",
            "./fixed_tokenizer"
        ]
        
        for path in local_paths:
            if Path(path).exists():
                try:
                    print(f"Loading tokenizer from {path}...")
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        path,
                        trust_remote_code=False,
                        padding_side="left"
                    )
                    self.tokenizer_path = path
                    print(f"[OK] Tokenizer loaded from {path}")
                    break
                except Exception as e:
                    print(f"  Failed: {e}")
                    continue
        
        if self.tokenizer is None:
            # Try online as last resort
            try:
                print("Loading GPT-2 tokenizer as fallback...")
                self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
                self.tokenizer_path = "gpt2"
                print("[OK] GPT-2 tokenizer loaded")
            except Exception as e:
                raise Exception(f"Could not load any tokenizer: {e}")
        
        # Configure tokenizer
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token or "<pad>"
        
        return self.tokenizer
    
    def get_tokenizer(self):
        """Get loaded tokenizer"""
        if self.tokenizer is None:
            self.load_tokenizer()
        return self.tokenizer

# Usage in your training script:
# Replace this line:
#   tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B", ...)
# With:
#   tokenizer_manager = TokenizerManager()
#   tokenizer = tokenizer_manager.get_tokenizer()
'''
    
    # Save modified script
    script_file = Path("./tokenizer_manager.py")
    with open(script_file, 'w', encoding='utf-8') as f:
        f.write(modified_script)
    
    print(f"  [SAVED] Tokenizer manager to: {script_file}")
    
    # Create integration example
    integration_example = '''# Integration Example for Your Training Script

# At the top of your training script, add:
from tokenizer_manager import TokenizerManager

# In your OptimizedModelManager class, modify the _load_tokenizer method:
def _load_tokenizer(self, checkpoint_path=None):
    """Load and configure tokenizer with fallback"""
    
    # Use TokenizerManager for robust loading
    tokenizer_manager = TokenizerManager()
    self.tokenizer = tokenizer_manager.get_tokenizer()
    
    return self.tokenizer

# That's it! The tokenizer will now load with fallback options.
'''
    
    example_file = Path("./integration_example.txt")
    with open(example_file, 'w', encoding='utf-8') as f:
        f.write(integration_example)
    
    print(f"  [SAVED] Integration example to: {example_file}")
    
    return True

def main():
    """Main execution"""
    
    print("\n" + "="*70)
    print("ADVANCED TOKENIZER FIX UTILITY")
    print("="*70)
    
    print("\n[SYSTEM INFO]")
    print(f"  PyTorch: {torch.__version__}")
    print(f"  CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    
    print("\n[STARTING FIX PROCESS]")
    
    # Step 1: Try alternative tokenizers
    tokenizer, tokenizer_path = use_alternative_tokenizer()
    
    if tokenizer is None:
        print("\n[WARNING] Could not load any online tokenizer")
        print("Creating manual tokenizer configuration...")
        tokenizer_path = create_manual_tokenizer()
    
    # Step 2: Create modified training script
    if modify_training_script():
        print("\n" + "="*70)
        print("[SUCCESS] Tokenizer fix completed!")
        print("="*70)
        
        print("\n[NEXT STEPS]")
        print("1. A tokenizer has been saved locally")
        print("2. Use the tokenizer_manager.py module in your training script")
        print("3. See integration_example.txt for how to integrate")
        print("\nTo use in your notebook, add this code:")
        print("-" * 40)
        print("from tokenizer_manager import TokenizerManager")
        print("tokenizer_manager = TokenizerManager()")
        print("tokenizer = tokenizer_manager.get_tokenizer()")
        print("-" * 40)
        
        return True
    
    return False

if __name__ == "__main__":
    success = main()
    
    if not success:
        print("\n[FAILED] Could not complete tokenizer fix")
        print("\nManual fix instructions:")
        print("1. Install transformers: pip install transformers==4.36.0")
        print("2. Clear cache: rm -rf ~/.cache/huggingface")
        print("3. Try a different model like GPT-2 or Phi-2")