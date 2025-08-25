
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
    """Tiktoken tabanlı Qwen3 tokenizer yükle"""
    
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
