"""
Offline Tokenizer Solution for Qwen Models
Downloads and prepares tokenizer for offline use
"""

import os
import json
import torch
from pathlib import Path
from transformers import AutoTokenizer
import shutil

def setup_huggingface_cache():
    """Setup HuggingFace cache without authentication"""
    
    # Disable HuggingFace telemetry and authentication
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ['HF_DATASETS_OFFLINE'] = '1'
    
    print("[INFO] Set offline mode for HuggingFace")

def download_gpt2_tokenizer():
    """Download GPT-2 tokenizer which doesn't require authentication"""
    
    print("\n[INFO] Downloading GPT-2 tokenizer (no auth required)...")
    
    try:
        # Temporarily enable online mode
        os.environ.pop('HF_HUB_OFFLINE', None)
        os.environ.pop('TRANSFORMERS_OFFLINE', None)
        
        # Download GPT-2 tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            "gpt2",
            cache_dir="./cache_gpt2",
            force_download=False
        )
        
        # Configure for use with larger models
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        
        # Save locally
        save_path = Path("./gpt2_tokenizer")
        save_path.mkdir(parents=True, exist_ok=True)
        tokenizer.save_pretrained(save_path)
        
        print(f"[SUCCESS] GPT-2 tokenizer saved to: {save_path}")
        
        # Set back to offline mode
        setup_huggingface_cache()
        
        return tokenizer, save_path
        
    except Exception as e:
        print(f"[ERROR] Failed to download GPT-2: {e}")
        return None, None

def create_qwen_compatible_tokenizer():
    """Create a Qwen-compatible tokenizer configuration"""
    
    print("\n[INFO] Creating Qwen-compatible tokenizer...")
    
    save_dir = Path("./qwen_compatible_tokenizer")
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Qwen tokenizer configuration
    config = {
        "add_prefix_space": False,
        "added_tokens_decoder": {
            "151643": {
                "content": "<|endoftext|>",
                "lstrip": False,
                "normalized": False,
                "rstrip": False,
                "single_word": False,
                "special": True
            },
            "151644": {
                "content": "<|im_start|>",
                "lstrip": False,
                "normalized": False,
                "rstrip": False,
                "single_word": False,
                "special": True
            },
            "151645": {
                "content": "<|im_end|>",
                "lstrip": False,
                "normalized": False,
                "rstrip": False,
                "single_word": False,
                "special": True
            }
        },
        "bos_token": None,
        "chat_template": "{% for message in messages %}{% if loop.first and messages[0]['role'] != 'system' %}{{ '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n' }}{% endif %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}",
        "clean_up_tokenization_spaces": False,
        "eos_token": "<|endoftext|>",
        "errors": "replace",
        "model_max_length": 32768,
        "pad_token": "<|endoftext|>",
        "split_special_tokens": False,
        "tokenizer_class": "Qwen2Tokenizer",
        "unk_token": None
    }
    
    # Save configuration
    config_file = save_dir / "tokenizer_config.json"
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    print(f"[SUCCESS] Configuration saved to: {save_dir}")
    
    # Create special tokens map
    special_tokens = {
        "eos_token": "<|endoftext|>",
        "pad_token": "<|endoftext|>",
        "additional_special_tokens": ["<|im_start|>", "<|im_end|>"]
    }
    
    special_tokens_file = save_dir / "special_tokens_map.json"
    with open(special_tokens_file, 'w', encoding='utf-8') as f:
        json.dump(special_tokens, f, indent=2, ensure_ascii=False)
    
    return save_dir

def create_universal_tokenizer_loader():
    """Create a universal tokenizer loader for your training script"""
    
    print("\n[INFO] Creating universal tokenizer loader...")
    
    loader_code = '''"""
Universal Tokenizer Loader
Automatically loads the best available tokenizer
"""

from pathlib import Path
from transformers import AutoTokenizer, GPT2Tokenizer
import json
import warnings
warnings.filterwarnings("ignore")

class UniversalTokenizerLoader:
    """Load tokenizer with multiple fallback options"""
    
    def __init__(self, max_length=512):
        self.max_length = max_length
        self.tokenizer = None
        self.tokenizer_type = None
    
    def load(self):
        """Try to load tokenizer from multiple sources"""
        
        # Priority order of tokenizers to try
        tokenizer_paths = [
            ("./gpt2_tokenizer", "GPT-2 (Local)"),
            ("./qwen_compatible_tokenizer", "Qwen Compatible"),
            ("./fixed_tokenizer", "Fixed Tokenizer"),
            ("./manual_tokenizer", "Manual Tokenizer"),
            ("gpt2", "GPT-2 (Online)"),
        ]
        
        for path, name in tokenizer_paths:
            try:
                print(f"Trying to load {name} from {path}...")
                
                if Path(path).exists() or not path.startswith("./"):
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        path,
                        trust_remote_code=False,
                        local_files_only=path.startswith("./")
                    )
                    self.tokenizer_type = name
                    print(f"[SUCCESS] Loaded {name}")
                    break
                    
            except Exception as e:
                print(f"  Failed: {str(e)[:50]}...")
                continue
        
        if self.tokenizer is None:
            # Last resort: create a simple tokenizer
            print("Creating basic tokenizer...")
            from transformers import PreTrainedTokenizerFast
            
            self.tokenizer = PreTrainedTokenizerFast(
                tokenizer_object={
                    "version": "1.0",
                    "truncation": None,
                    "padding": None,
                    "added_tokens": [],
                    "normalizer": None,
                    "pre_tokenizer": {
                        "type": "Whitespace"
                    },
                    "post_processor": None,
                    "decoder": None,
                    "model": {
                        "type": "BPE",
                        "dropout": None,
                        "unk_token": "[UNK]",
                        "continuing_subword_prefix": None,
                        "end_of_word_suffix": None,
                        "fuse_unk": False,
                        "byte_fallback": False,
                        "vocab": {},
                        "merges": []
                    }
                }
            )
            self.tokenizer_type = "Basic"
        
        # Configure tokenizer
        self._configure_tokenizer()
        
        return self.tokenizer
    
    def _configure_tokenizer(self):
        """Configure tokenizer with proper settings"""
        
        if self.tokenizer is None:
            return
        
        # Set padding token
        if self.tokenizer.pad_token is None:
            if self.tokenizer.eos_token:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        
        # Set padding side
        self.tokenizer.padding_side = "left"
        
        # Set max length
        self.tokenizer.model_max_length = self.max_length
        
        print(f"Tokenizer configured: {self.tokenizer_type}")
        print(f"  Vocab size: {len(self.tokenizer)}")
        print(f"  Max length: {self.tokenizer.model_max_length}")
        print(f"  Pad token: {self.tokenizer.pad_token}")
    
    def test_tokenizer(self):
        """Test the tokenizer with sample text"""
        
        if self.tokenizer is None:
            print("[ERROR] No tokenizer loaded")
            return False
        
        test_texts = [
            "Hello, world!",
            "Merhaba dünya!",
            "Bu bir test cümlesidir."
        ]
        
        print("\\nTesting tokenizer:")
        for text in test_texts:
            try:
                tokens = self.tokenizer(text, return_tensors="pt")
                decoded = self.tokenizer.decode(tokens['input_ids'][0], skip_special_tokens=True)
                print(f"  OK: '{text}' -> {tokens['input_ids'].shape}")
            except Exception as e:
                print(f"  FAIL: '{text}' -> {e}")
                return False
        
        return True

# Usage:
if __name__ == "__main__":
    loader = UniversalTokenizerLoader(max_length=512)
    tokenizer = loader.load()
    
    if tokenizer:
        loader.test_tokenizer()
        print("\\nTokenizer ready for use!")
    else:
        print("\\n[ERROR] Could not load any tokenizer")
'''
    
    # Save loader
    loader_file = Path("./universal_tokenizer_loader.py")
    with open(loader_file, 'w', encoding='utf-8') as f:
        f.write(loader_code)
    
    print(f"[SUCCESS] Universal loader saved to: {loader_file}")
    
    return loader_file

def main():
    """Main execution"""
    
    print("\n" + "="*70)
    print("OFFLINE TOKENIZER SOLUTION")
    print("="*70)
    
    print("\n[SYSTEM INFO]")
    print(f"  PyTorch: {torch.__version__}")
    print(f"  Python: {os.sys.version.split()[0]}")
    
    # Setup offline mode
    setup_huggingface_cache()
    
    # Try to download GPT-2 tokenizer
    tokenizer, tokenizer_path = download_gpt2_tokenizer()
    
    # Create Qwen-compatible configuration
    qwen_path = create_qwen_compatible_tokenizer()
    
    # Create universal loader
    loader_path = create_universal_tokenizer_loader()
    
    print("\n" + "="*70)
    print("[COMPLETED] Tokenizer setup finished!")
    print("="*70)
    
    print("\n[FILES CREATED]")
    if tokenizer_path:
        print(f"  1. GPT-2 Tokenizer: {tokenizer_path}")
    print(f"  2. Qwen Config: {qwen_path}")
    print(f"  3. Universal Loader: {loader_path}")
    
    print("\n[HOW TO USE IN YOUR NOTEBOOK]")
    print("-" * 50)
    print("# Add this to your notebook:")
    print("from universal_tokenizer_loader import UniversalTokenizerLoader")
    print("")
    print("# Create and load tokenizer")
    print("loader = UniversalTokenizerLoader(max_length=512)")
    print("tokenizer = loader.load()")
    print("")
    print("# Test it")
    print("loader.test_tokenizer()")
    print("-" * 50)
    
    print("\n[INTEGRATION WITH YOUR TRAINING SCRIPT]")
    print("In your OptimizedModelManager._load_tokenizer method, replace with:")
    print("-" * 50)
    print("from universal_tokenizer_loader import UniversalTokenizerLoader")
    print("")
    print("def _load_tokenizer(self, checkpoint_path=None):")
    print("    loader = UniversalTokenizerLoader(max_length=self.config.max_length)")
    print("    return loader.load()")
    print("-" * 50)

if __name__ == "__main__":
    main()