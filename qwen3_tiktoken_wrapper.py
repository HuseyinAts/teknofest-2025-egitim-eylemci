"""
Qwen3 Tiktoken Tokenizer Wrapper
Transformers kütüphanesi ile uyumlu tiktoken wrapper
"""

import json
from pathlib import Path
from typing import List, Optional, Dict, Any, Union
import warnings
warnings.filterwarnings("ignore")

try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False
    print("[WARNING] Tiktoken yüklü değil. pip install tiktoken")

class Qwen3TiktokenTokenizer:
    """Qwen3 için tiktoken tabanlı tokenizer"""
    
    def __init__(self, tokenizer_dir: str = "./qwen3_tiktoken_tokenizer"):
        self.tokenizer_dir = Path(tokenizer_dir)
        self.encoding = None
        self.special_tokens = {}
        self.pad_token = "<|endoftext|>"
        self.eos_token = "<|endoftext|>"
        self.pad_token_id = 100257
        self.eos_token_id = 100257
        self.model_max_length = 32768
        self.padding_side = "left"
        
        if TIKTOKEN_AVAILABLE:
            self._load_tokenizer()
        else:
            print("[ERROR] Tiktoken kullanılamıyor")
    
    def _load_tokenizer(self):
        """Tiktoken encoding'i yükle"""
        try:
            # cl100k_base encoding kullan (Qwen3 default)
            self.encoding = tiktoken.get_encoding("cl100k_base")
            
            # Special tokens yükle
            if (self.tokenizer_dir / "special_tokens_map.json").exists():
                with open(self.tokenizer_dir / "special_tokens_map.json", 'r') as f:
                    special_tokens_map = json.load(f)
                    self.special_tokens = {
                        "<|endoftext|>": 100257,
                        "<|im_start|>": 100264,
                        "<|im_end|>": 100265,
                        "<|im_sep|>": 100266
                    }
            
            print(f"[OK] Tiktoken tokenizer yüklendi (vocab size: {self.encoding.n_vocab})")
            
        except Exception as e:
            print(f"[ERROR] Tokenizer yüklenemedi: {e}")
    
    def __call__(self, 
                 text: Union[str, List[str]], 
                 padding: bool = True,
                 truncation: bool = True,
                 max_length: Optional[int] = None,
                 return_tensors: Optional[str] = None,
                 **kwargs) -> Dict[str, Any]:
        """Tokenize text (transformers uyumlu)"""
        
        if not self.encoding:
            raise ValueError("Tokenizer yüklenmemiş")
        
        # Tek metin veya liste kontrolü
        if isinstance(text, str):
            texts = [text]
        else:
            texts = text
        
        max_len = max_length or self.model_max_length
        
        # Tokenize
        all_input_ids = []
        all_attention_masks = []
        
        for txt in texts:
            # Encode
            tokens = self.encoding.encode(txt)
            
            # Truncate
            if truncation and len(tokens) > max_len:
                tokens = tokens[:max_len]
            
            # Padding
            if padding:
                if self.padding_side == "left":
                    original_length = len(tokens)
                    pad_length = max_len - original_length
                    tokens = [self.pad_token_id] * pad_length + tokens
                    attention_mask = [0] * pad_length + [1] * original_length
                else:
                    original_length = len(tokens)
                    tokens = tokens + [self.pad_token_id] * (max_len - original_length)
                    attention_mask = [1] * original_length + [0] * (max_len - original_length)
            else:
                attention_mask = [1] * len(tokens)
            
            all_input_ids.append(tokens)
            all_attention_masks.append(attention_mask)
        
        # Return format
        result = {
            'input_ids': all_input_ids[0] if isinstance(text, str) else all_input_ids,
            'attention_mask': all_attention_masks[0] if isinstance(text, str) else all_attention_masks
        }
        
        # Convert to tensors if requested
        if return_tensors == "pt":
            import torch
            result['input_ids'] = torch.tensor(result['input_ids'])
            result['attention_mask'] = torch.tensor(result['attention_mask'])
        
        return result
    
    def encode(self, text: str, **kwargs) -> List[int]:
        """Encode text to token ids"""
        if not self.encoding:
            raise ValueError("Tokenizer yüklenmemiş")
        return self.encoding.encode(text)
    
    def decode(self, token_ids, skip_special_tokens: bool = True, **kwargs) -> str:
        """Decode token ids to text"""
        if not self.encoding:
            raise ValueError("Tokenizer yüklenmemiş")
        
        # Tensor ise listeye çevir
        if hasattr(token_ids, 'tolist'):
            token_ids = token_ids.tolist()
        
        # Eğer tek bir integer ise listeye çevir
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        
        # Nested list kontrolü
        if isinstance(token_ids, list) and len(token_ids) > 0 and isinstance(token_ids[0], list):
            token_ids = token_ids[0]
        
        # Special token'ları filtrele
        if skip_special_tokens and isinstance(token_ids, list):
            token_ids = [t for t in token_ids if t not in [self.pad_token_id, self.eos_token_id]]
        
        return self.encoding.decode(token_ids)
    
    def save_pretrained(self, save_directory: str):
        """Tokenizer'ı kaydet"""
        save_dir = Path(save_directory)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Config kaydet
        config = {
            "tokenizer_class": "Qwen3TiktokenTokenizer",
            "encoding_name": "cl100k_base",
            "pad_token": self.pad_token,
            "eos_token": self.eos_token,
            "model_max_length": self.model_max_length
        }
        
        with open(save_dir / "tokenizer_config.json", 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"[OK] Tokenizer kaydedildi: {save_dir}")
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs):
        """Tokenizer'ı yükle"""
        return cls(tokenizer_dir=pretrained_model_name_or_path)
    
    def __len__(self):
        """Vocab size"""
        return self.encoding.n_vocab if self.encoding else 0

# Test fonksiyonu
def test_tokenizer():
    """Tokenizer'ı test et"""
    print("\n[TEST] Tokenizer test ediliyor...")
    
    tokenizer = Qwen3TiktokenTokenizer()
    
    if not tokenizer.encoding:
        print("[ERROR] Tokenizer yüklenemedi")
        return False
    
    # Test metinleri
    test_texts = [
        "Merhaba dünya!",
        "Bu bir test cümlesidir.",
        "Qwen3 modeli için tokenizer testi."
    ]
    
    for text in test_texts:
        # Tokenize
        tokens = tokenizer(text, return_tensors="pt")
        print(f"\n  Text: {text}")
        print(f"  Tokens shape: {tokens['input_ids'].shape}")
        
        # Decode
        decoded = tokenizer.decode(tokens['input_ids'][0], skip_special_tokens=True)
        print(f"  Decoded: {decoded}")
    
    print("\n[OK] Tokenizer testi başarılı!")
    return True

if __name__ == "__main__":
    test_tokenizer()
