"""
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
        
        print("\nTesting tokenizer:")
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
        print("\nTokenizer ready for use!")
    else:
        print("\n[ERROR] Could not load any tokenizer")
