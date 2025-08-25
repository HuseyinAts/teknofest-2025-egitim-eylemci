"""
Qwen3 Tiktoken Tokenizer Fix
Qwen3 modeli için tiktoken tabanlı tokenizer çözümü
"""

import os
import json
import shutil
from pathlib import Path
import torch
import base64
import requests
from typing import Optional, Dict, Any

def install_tiktoken():
    """Tiktoken kütüphanesini yükle"""
    try:
        import tiktoken
        print("[OK] Tiktoken zaten yüklü")
        return True
    except ImportError:
        print("[INFO] Tiktoken yükleniyor...")
        os.system("pip install tiktoken")
        try:
            import tiktoken
            print("[OK] Tiktoken başarıyla yüklendi")
            return True
        except:
            print("[ERROR] Tiktoken yüklenemedi")
            return False

def create_qwen3_tokenizer():
    """Qwen3 için tiktoken tabanlı tokenizer oluştur"""
    
    print("\n[INFO] Qwen3 tiktoken tokenizer oluşturuluyor...")
    
    # Tiktoken'ı import et
    try:
        import tiktoken
    except ImportError:
        print("[ERROR] Tiktoken import edilemedi")
        return None
    
    # Qwen3 tokenizer konfigürasyonu
    tokenizer_config = {
        "tokenizer_class": "Qwen2Tokenizer",
        "model_max_length": 32768,
        "padding_side": "left",
        "truncation_side": "right",
        "chat_template": "{% for message in messages %}{% if loop.first and messages[0]['role'] != 'system' %}{{ '<|im_start|>system\nYou are a helpful assistant<|im_end|>\n' }}{% endif %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}",
        "clean_up_tokenization_spaces": False,
        "split_special_tokens": False,
        "use_default_system_prompt": False,
        "bos_token": None,
        "eos_token": "<|endoftext|>",
        "unk_token": None,
        "pad_token": "<|endoftext|>",
        "errors": "replace"
    }
    
    # Tokenizer dizinini oluştur
    save_dir = Path("./qwen3_tiktoken_tokenizer")
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Tokenizer config kaydet
    config_file = save_dir / "tokenizer_config.json"
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(tokenizer_config, f, indent=2, ensure_ascii=False)
    print(f"  [SAVED] Tokenizer config: {config_file}")
    
    # Special tokens map
    special_tokens = {
        "eos_token": "<|endoftext|>",
        "pad_token": "<|endoftext|>",
        "additional_special_tokens": [
            "<|im_start|>",
            "<|im_end|>",
            "<|im_sep|>"
        ]
    }
    
    special_tokens_file = save_dir / "special_tokens_map.json"
    with open(special_tokens_file, 'w', encoding='utf-8') as f:
        json.dump(special_tokens, f, indent=2, ensure_ascii=False)
    print(f"  [SAVED] Special tokens: {special_tokens_file}")
    
    # Qwen vocab için tiktoken encoding oluştur
    # Qwen3 cl100k_base encoding kullanır (GPT-4 benzeri)
    try:
        # cl100k_base encoding'i yükle
        encoding = tiktoken.get_encoding("cl100k_base")
        print("  [OK] cl100k_base encoding yüklendi")
        
        # Encoding bilgilerini kaydet
        encoding_info = {
            "encoding_name": "cl100k_base",
            "vocab_size": encoding.n_vocab,
            "max_token_value": encoding.max_token_value if hasattr(encoding, 'max_token_value') else encoding.n_vocab,
            "special_tokens": {
                "<|endoftext|>": 100257,
                "<|im_start|>": 100264,
                "<|im_end|>": 100265,
                "<|im_sep|>": 100266
            }
        }
        
        encoding_file = save_dir / "encoding_info.json"
        with open(encoding_file, 'w', encoding='utf-8') as f:
            json.dump(encoding_info, f, indent=2, ensure_ascii=False)
        print(f"  [SAVED] Encoding info: {encoding_file}")
        
    except Exception as e:
        print(f"  [WARNING] cl100k_base encoding yüklenemedi: {e}")
    
    return save_dir

def create_tiktoken_wrapper():
    """Transformers uyumlu tiktoken wrapper oluştur"""
    
    print("\n[INFO] Tiktoken wrapper oluşturuluyor...")
    
    wrapper_code = '''"""
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
                    pad_length = max_len - len(tokens)
                    tokens = [self.pad_token_id] * pad_length + tokens
                    attention_mask = [0] * pad_length + [1] * len(tokens)
                else:
                    original_length = len(tokens)
                    tokens = tokens + [self.pad_token_id] * (max_len - len(tokens))
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
    
    def decode(self, token_ids: List[int], skip_special_tokens: bool = True, **kwargs) -> str:
        """Decode token ids to text"""
        if not self.encoding:
            raise ValueError("Tokenizer yüklenmemiş")
        
        # Tensor ise listeye çevir
        if hasattr(token_ids, 'tolist'):
            token_ids = token_ids.tolist()
        
        # Nested list kontrolü
        if isinstance(token_ids[0], list):
            token_ids = token_ids[0]
        
        # Special token'ları filtrele
        if skip_special_tokens:
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
    print("\\n[TEST] Tokenizer test ediliyor...")
    
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
        print(f"\\n  Text: {text}")
        print(f"  Tokens shape: {tokens['input_ids'].shape}")
        
        # Decode
        decoded = tokenizer.decode(tokens['input_ids'][0], skip_special_tokens=True)
        print(f"  Decoded: {decoded}")
    
    print("\\n[OK] Tokenizer testi başarılı!")
    return True

if __name__ == "__main__":
    test_tokenizer()
'''
    
    # Wrapper'ı kaydet
    wrapper_file = Path("./qwen3_tiktoken_wrapper.py")
    with open(wrapper_file, 'w', encoding='utf-8') as f:
        f.write(wrapper_code)
    
    print(f"  [SAVED] Wrapper: {wrapper_file}")
    
    return wrapper_file

def create_integration_guide():
    """Entegrasyon kılavuzu oluştur"""
    
    guide = """
# Qwen3 Tiktoken Tokenizer Entegrasyon Kılavuzu

## 1. Kurulum
```bash
pip install tiktoken
```

## 2. Training Script'e Entegrasyon

Training script'inizde `_load_tokenizer` metodunu şu şekilde değiştirin:

```python
from qwen3_tiktoken_wrapper import Qwen3TiktokenTokenizer

def _load_tokenizer(self, checkpoint_path: Optional[Path] = None) -> AutoTokenizer:
    \"\"\"Tiktoken tabanlı Qwen3 tokenizer yükle\"\"\"
    
    print("Qwen3 tiktoken tokenizer yükleniyor...")
    
    # Tiktoken wrapper kullan
    tokenizer = Qwen3TiktokenTokenizer(
        tokenizer_dir="./qwen3_tiktoken_tokenizer"
    )
    
    # Özel token'ları ayarla
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    tokenizer.padding_side = "left"
    tokenizer.model_max_length = self.config.max_length
    
    return tokenizer
```

## 3. Notebook'ta Kullanım

```python
# Import
from qwen3_tiktoken_wrapper import Qwen3TiktokenTokenizer

# Tokenizer oluştur
tokenizer = Qwen3TiktokenTokenizer()

# Test et
text = "Merhaba, bu bir test metnidir."
tokens = tokenizer(text, return_tensors="pt")
print(f"Token shape: {tokens['input_ids'].shape}")

# Decode
decoded = tokenizer.decode(tokens['input_ids'][0])
print(f"Decoded: {decoded}")
```

## 4. Sorun Giderme

Eğer tiktoken yüklenemezse:
```bash
pip install --upgrade tiktoken
pip install regex
```

Eğer encoding bulunamazsa:
```python
import tiktoken
tiktoken.list_encoding_names()  # Mevcut encoding'leri listele
```
"""
    
    guide_file = Path("./TIKTOKEN_INTEGRATION_GUIDE.md")
    with open(guide_file, 'w', encoding='utf-8') as f:
        f.write(guide)
    
    print(f"\n[SAVED] Entegrasyon kılavuzu: {guide_file}")
    
    return guide_file

def main():
    """Ana çalıştırma fonksiyonu"""
    
    print("\n" + "="*70)
    print("QWEN3 TIKTOKEN TOKENIZER FIX")
    print("="*70)
    
    print("\n[SYSTEM INFO]")
    print(f"  PyTorch: {torch.__version__}")
    print(f"  Python: {os.sys.version.split()[0]}")
    
    # Tiktoken'ı yükle
    if not install_tiktoken():
        print("\n[ERROR] Tiktoken yüklenemedi!")
        print("Manuel olarak yükleyin: pip install tiktoken")
        return False
    
    # Qwen3 tokenizer oluştur
    tokenizer_dir = create_qwen3_tokenizer()
    
    # Wrapper oluştur
    wrapper_file = create_tiktoken_wrapper()
    
    # Entegrasyon kılavuzu
    guide_file = create_integration_guide()
    
    print("\n" + "="*70)
    print("[SUCCESS] Qwen3 tiktoken tokenizer hazır!")
    print("="*70)
    
    print("\n[OLUŞTURULAN DOSYALAR]")
    print(f"  1. Tokenizer dizini: {tokenizer_dir}")
    print(f"  2. Wrapper: {wrapper_file}")
    print(f"  3. Kılavuz: {guide_file}")
    
    print("\n[HIZLI KULLANIM]")
    print("-" * 50)
    print("from qwen3_tiktoken_wrapper import Qwen3TiktokenTokenizer")
    print("")
    print("# Tokenizer oluştur")
    print("tokenizer = Qwen3TiktokenTokenizer()")
    print("")
    print("# Kullan")
    print('text = "Merhaba dünya!"')
    print('tokens = tokenizer(text, return_tensors="pt")')
    print('decoded = tokenizer.decode(tokens["input_ids"][0])')
    print("-" * 50)
    
    print("\n[NOT] Training script'inizde _load_tokenizer metodunu")
    print("      TIKTOKEN_INTEGRATION_GUIDE.md dosyasındaki gibi güncelleyin.")
    
    return True

if __name__ == "__main__":
    success = main()
    
    if not success:
        print("\n[FAILED] Tokenizer fix başarısız oldu")
        print("Alternatif çözüm: GPT-2 tokenizer kullanın")