"""
Qwen3-8B Model ile Turkish Mixtral Tokenizer Entegrasyon Adapter'ı
Bu modül, Turkish Mixtral tokenizer'ı Qwen modeliyle kullanabilmek için gerekli adaptasyonu sağlar.
"""

import torch
import numpy as np
from typing import List, Dict, Optional, Union, Tuple
from pathlib import Path
import json
import logging
from transformers import PreTrainedTokenizer, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QwenTurkishTokenizerAdapter:
    """
    Turkish Mixtral tokenizer'ı Qwen modeli ile kullanmak için adapter sınıfı.
    
    Bu adapter:
    1. Turkish tokenizer'dan gelen token ID'lerini Qwen embedding space'ine map eder
    2. Vocabulary alignment yapar
    3. Special token'ları handle eder
    """
    
    def __init__(self, 
                 qwen_model_path: str = "Qwen/Qwen2.5-7B-Instruct",
                 turkish_tokenizer_path: str = "notebooks/turkish_mixtral_v3_fixed.model"):
        """
        Initialize adapter
        
        Args:
            qwen_model_path: Qwen model path or HuggingFace ID
            turkish_tokenizer_path: Turkish Mixtral tokenizer model path
        """
        self.qwen_model_path = qwen_model_path
        self.turkish_tokenizer_path = turkish_tokenizer_path
        
        # Tokenizer'ları yükle
        self._load_tokenizers()
        
        # Vocabulary mapping oluştur
        self.vocab_mapping = {}
        self.reverse_mapping = {}
        self._create_vocabulary_mapping()
        
    def _load_tokenizers(self):
        """Her iki tokenizer'ı da yükle"""
        try:
            # Qwen tokenizer
            from transformers import AutoTokenizer
            self.qwen_tokenizer = AutoTokenizer.from_pretrained(
                self.qwen_model_path,
                trust_remote_code=True
            )
            
            # Turkish Mixtral tokenizer
            from src.turkish_mixtral_tokenizer import TurkishMixtralTokenizer
            self.turkish_tokenizer = TurkishMixtralTokenizer(
                model_path=self.turkish_tokenizer_path
            )
            
            logger.info("✅ Her iki tokenizer başarıyla yüklendi")
            
        except Exception as e:
            logger.error(f"❌ Tokenizer yükleme hatası: {e}")
            raise
            
    def _create_vocabulary_mapping(self):
        """
        Turkish tokenizer vocabulary'sini Qwen vocabulary'sine map et.
        Yaklaşım: Subword similarity ve frequency-based mapping
        """
        logger.info("Vocabulary mapping oluşturuluyor...")
        
        # Öncelikli mapping stratejileri:
        # 1. Exact match (aynı token)
        # 2. Subword similarity 
        # 3. Character-level fallback
        
        # Special token mapping
        special_mappings = {
            '<s>': self.qwen_tokenizer.bos_token_id or 1,
            '</s>': self.qwen_tokenizer.eos_token_id or 2,
            '<unk>': self.qwen_tokenizer.unk_token_id or 0,
            '<PAD>': self.qwen_tokenizer.pad_token_id or 0,
        }
        
        # Turkish'e özel entity token'ları için Qwen'de yakın anlamlı token'lar bul
        entity_mappings = {
            '<AI>': self._find_qwen_token_for_text("AI"),
            '<ML>': self._find_qwen_token_for_text("ML"),
            '<NLP>': self._find_qwen_token_for_text("NLP"),
            '<BSc>': self._find_qwen_token_for_text("Bachelor"),
            '<MSc>': self._find_qwen_token_for_text("Master"),
            '<PhD>': self._find_qwen_token_for_text("PhD"),
            '<NUMBER>': self._find_qwen_token_for_text("123"),
            '<YEAR>': self._find_qwen_token_for_text("2024"),
            '<PERCENTAGE>': self._find_qwen_token_for_text("85%"),
        }
        
        self.vocab_mapping.update(special_mappings)
        self.vocab_mapping.update(entity_mappings)
        
        # Sık kullanılan Türkçe token'lar için mapping
        common_turkish_tokens = [
            "▁ve", "▁ile", "▁için", "▁bir", "▁bu", "▁olan",
            "lar", "ler", "de", "da", "den", "dan", "in", "ın",
            "mak", "mek", "yor", "miş", "mış", "ecek", "acak"
        ]
        
        for token in common_turkish_tokens:
            qwen_id = self._find_qwen_token_for_text(token.replace("▁", " ").strip())
            if qwen_id is not None:
                # Turkish tokenizer'da bu token'ın ID'sini bul
                try:
                    turkish_id = self.turkish_tokenizer.sp_model.piece_to_id(token)
                    self.vocab_mapping[turkish_id] = qwen_id
                except:
                    pass
                    
        logger.info(f"✅ {len(self.vocab_mapping)} token mapping oluşturuldu")
        
    def _find_qwen_token_for_text(self, text: str) -> Optional[int]:
        """Verilen text için Qwen tokenizer'da en uygun token ID'yi bul"""
        try:
            tokens = self.qwen_tokenizer.encode(text, add_special_tokens=False)
            if tokens:
                return tokens[0]  # İlk token'ı al
        except:
            pass
        return None
        
    def adapt_tokens(self, turkish_token_ids: List[int]) -> List[int]:
        """
        Turkish tokenizer ID'lerini Qwen token ID'lerine dönüştür
        
        Args:
            turkish_token_ids: Turkish tokenizer'dan gelen token ID'leri
            
        Returns:
            Qwen model için uygun token ID'leri
        """
        qwen_ids = []
        
        for tid in turkish_token_ids:
            if tid in self.vocab_mapping:
                # Direct mapping varsa kullan
                qwen_ids.append(self.vocab_mapping[tid])
            else:
                # Mapping yoksa, token'ı decode edip Qwen ile tekrar encode et
                try:
                    # Turkish token'ı text'e çevir
                    text = self.turkish_tokenizer.sp_model.id_to_piece(tid)
                    # Qwen ile tokenize et
                    qwen_tokens = self.qwen_tokenizer.encode(text, add_special_tokens=False)
                    qwen_ids.extend(qwen_tokens)
                except:
                    # Fallback: unknown token
                    qwen_ids.append(self.qwen_tokenizer.unk_token_id or 0)
                    
        return qwen_ids
        
    def prepare_input_for_model(self, text: str, max_length: int = 512) -> Dict[str, torch.Tensor]:
        """
        Text'i Turkish tokenizer ile tokenize edip Qwen model için hazırla
        
        Args:
            text: İşlenecek metin
            max_length: Maksimum sequence uzunluğu
            
        Returns:
            Model input dictionary (input_ids, attention_mask)
        """
        # Turkish tokenizer ile tokenize et
        turkish_ids = self.turkish_tokenizer.encode(text, add_special_tokens=True)
        
        # Qwen ID'lerine adapte et
        qwen_ids = self.adapt_tokens(turkish_ids)
        
        # Truncate if needed
        if len(qwen_ids) > max_length:
            qwen_ids = qwen_ids[:max_length]
            
        # Padding
        attention_mask = [1] * len(qwen_ids)
        
        if len(qwen_ids) < max_length:
            padding_length = max_length - len(qwen_ids)
            qwen_ids.extend([self.qwen_tokenizer.pad_token_id] * padding_length)
            attention_mask.extend([0] * padding_length)
            
        return {
            'input_ids': torch.tensor([qwen_ids]),
            'attention_mask': torch.tensor([attention_mask])
        }
        
    def decode_from_model(self, token_ids: Union[List[int], torch.Tensor]) -> str:
        """
        Model çıktısını decode et
        
        Args:
            token_ids: Model'den gelen token ID'leri
            
        Returns:
            Decoded text
        """
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
            
        # Qwen tokenizer ile decode et
        return self.qwen_tokenizer.decode(token_ids, skip_special_tokens=True)


class QwenWithTurkishTokenizer(PreTrainedModel):
    """
    Qwen modeli + Turkish tokenizer wrapper sınıfı
    Bu sınıf, training sırasında kullanılabilir
    """
    
    def __init__(self, qwen_model, adapter: QwenTurkishTokenizerAdapter):
        super().__init__(qwen_model.config)
        self.model = qwen_model
        self.adapter = adapter
        
    def forward(self, texts: List[str] = None, **kwargs) -> CausalLMOutputWithPast:
        """
        Forward pass with Turkish tokenizer
        
        Args:
            texts: Input texts (string list)
            **kwargs: Additional model arguments
        """
        if texts is not None:
            # Text'leri Turkish tokenizer ile işle
            batch_inputs = []
            batch_masks = []
            
            for text in texts:
                inputs = self.adapter.prepare_input_for_model(text)
                batch_inputs.append(inputs['input_ids'])
                batch_masks.append(inputs['attention_mask'])
                
            input_ids = torch.cat(batch_inputs, dim=0)
            attention_mask = torch.cat(batch_masks, dim=0)
            
            kwargs['input_ids'] = input_ids
            kwargs['attention_mask'] = attention_mask
            
        # Model forward pass
        return self.model(**kwargs)
        
    def generate_with_turkish(self, text: str, max_new_tokens: int = 100, **kwargs) -> str:
        """
        Turkish tokenizer kullanarak text generation
        
        Args:
            text: Input prompt
            max_new_tokens: Maximum tokens to generate
            **kwargs: Generation arguments
            
        Returns:
            Generated text
        """
        # Input'u hazırla
        inputs = self.adapter.prepare_input_for_model(text)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                inputs['input_ids'].to(self.model.device),
                attention_mask=inputs['attention_mask'].to(self.model.device),
                max_new_tokens=max_new_tokens,
                **kwargs
            )
            
        # Decode output
        generated_text = self.adapter.decode_from_model(outputs[0])
        
        return generated_text


def create_training_arguments_with_turkish_tokenizer():
    """
    Turkish tokenizer ile Qwen modelini eğitmek için TrainingArguments oluştur
    """
    from transformers import TrainingArguments
    
    training_args = TrainingArguments(
        output_dir="./qwen-turkish-finetuned",
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,
        warmup_steps=100,
        num_train_epochs=3,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=10,
        save_strategy="steps",
        save_steps=500,
        evaluation_strategy="steps",
        eval_steps=500,
        load_best_model_at_end=True,
        report_to="none",
        # Turkish tokenizer için özel ayarlar
        remove_unused_columns=False,  # Custom tokenizer kullanıyoruz
        label_names=["labels"],
    )
    
    return training_args


# Kullanım örneği
if __name__ == "__main__":
    print("="*60)
    print("QWEN + TURKISH TOKENIZER ENTEGRASYON TESTİ")
    print("="*60)
    
    # Adapter'ı oluştur
    adapter = QwenTurkishTokenizerAdapter()
    
    # Test metinleri
    test_texts = [
        "Merhaba, bugün yapay zeka hakkında konuşacağız.",
        "Öğrencilerin %85'i online eğitim platformlarını kullanıyor.",
        "PhD öğrencileri için hazırlanan AI ve ML dersleri başlıyor."
    ]
    
    print("\n📝 Tokenizasyon Testleri:")
    print("-"*40)
    
    for text in test_texts:
        print(f"\nText: {text}")
        
        # Turkish tokenizer ile tokenize et
        turkish_ids = adapter.turkish_tokenizer.encode(text)
        print(f"Turkish tokens ({len(turkish_ids)}): {turkish_ids[:10]}...")
        
        # Qwen'e adapte et
        qwen_ids = adapter.adapt_tokens(turkish_ids)
        print(f"Adapted Qwen tokens ({len(qwen_ids)}): {qwen_ids[:10]}...")
        
        # Model input hazırla
        model_input = adapter.prepare_input_for_model(text, max_length=128)
        print(f"Model input shape: {model_input['input_ids'].shape}")
        
    print("\n✅ Entegrasyon testi başarılı!")
    print("\n💡 Not: Bu adapter'ı kullanarak Qwen modelini Turkish tokenizer ile eğitebilirsiniz.")