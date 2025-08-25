"""
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
