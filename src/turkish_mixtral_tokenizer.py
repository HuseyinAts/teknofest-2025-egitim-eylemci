"""
Turkish Mixtral Tokenizer Integration
Türkçe için optimize edilmiş SentencePiece tokenizer
"""

import os
from pathlib import Path
from typing import List, Dict, Optional, Union
import sentencepiece as spm
from transformers import PreTrainedTokenizer
import logging

logger = logging.getLogger(__name__)


class TurkishMixtralTokenizer:
    """Turkish Mixtral SentencePiece Tokenizer Wrapper"""
    
    def __init__(self, model_path: str = None, vocab_path: str = None):
        """
        Initialize Turkish Mixtral tokenizer
        
        Args:
            model_path: Path to .model file
            vocab_path: Path to .vocab file (optional)
        """
        self.model_path = model_path or "notebooks/turkish_mixtral_v3_fixed.model"
        self.vocab_path = vocab_path or "notebooks/turkish_mixtral_v3_fixed.vocab"
        
        # SentencePiece processor
        self.sp_model = None
        self.vocab = {}
        self.special_tokens = {
            'pad_token': '<PAD>',
            'unk_token': '<unk>',
            'bos_token': '<s>',
            'eos_token': '</s>',
            'mask_token': '<MASK>'
        }
        
        # Entity tokens for education domain
        self.entity_tokens = [
            '<PhD>', '<MSc>', '<BSc>',  # Education levels
            '<AI>', '<ML>', '<NLP>',     # Tech terms
            '<NUMBER>', '<YEAR>', '<DATE>', '<TIME>',  # Temporal
            '<TECH_TERM>', '<BRAND_NAME>'  # General entities
        ]
        
        self._load_model()
        
    def _load_model(self):
        """Load SentencePiece model"""
        try:
            if not Path(self.model_path).exists():
                raise FileNotFoundError(f"Model file not found: {self.model_path}")
                
            self.sp_model = spm.SentencePieceProcessor()
            self.sp_model.load(self.model_path)
            
            # Load vocabulary if exists
            if Path(self.vocab_path).exists():
                self._load_vocab()
                
            logger.info(f"Loaded Turkish Mixtral tokenizer with {self.sp_model.get_piece_size()} tokens")
            
        except Exception as e:
            logger.error(f"Failed to load tokenizer: {e}")
            raise
            
    def _load_vocab(self):
        """Load vocabulary file"""
        with open(self.vocab_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    token, score = parts
                    self.vocab[token] = float(score)
                    
    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """
        Encode text to token IDs
        
        Args:
            text: Input text
            add_special_tokens: Whether to add <s> and </s>
            
        Returns:
            List of token IDs
        """
        if self.sp_model is None:
            raise RuntimeError("Tokenizer not loaded")
            
        # Preprocess for entity tokens
        text = self._preprocess_entities(text)
        
        # Encode with SentencePiece
        tokens = self.sp_model.encode(text, out_type=int)
        
        if add_special_tokens:
            bos_id = self.sp_model.piece_to_id('<s>')
            eos_id = self.sp_model.piece_to_id('</s>')
            tokens = [bos_id] + tokens + [eos_id]
            
        return tokens
        
    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        Decode token IDs to text
        
        Args:
            token_ids: List of token IDs
            skip_special_tokens: Whether to skip special tokens
            
        Returns:
            Decoded text
        """
        if self.sp_model is None:
            raise RuntimeError("Tokenizer not loaded")
            
        # Filter special tokens if needed
        if skip_special_tokens:
            special_ids = {
                self.sp_model.piece_to_id('<s>'),
                self.sp_model.piece_to_id('</s>'),
                self.sp_model.piece_to_id('<unk>'),
                self.sp_model.piece_to_id('<PAD>')
            }
            token_ids = [tid for tid in token_ids if tid not in special_ids]
            
        return self.sp_model.decode(token_ids)
        
    def _preprocess_entities(self, text: str) -> str:
        """Preprocess text for entity tokens"""
        # Example: Replace common patterns with entity tokens
        import re
        
        # Replace years
        text = re.sub(r'\b(19|20)\d{2}\b', '<YEAR>', text)
        
        # Replace numbers
        text = re.sub(r'\b\d+\b', '<NUMBER>', text)
        
        # Replace percentages
        text = re.sub(r'\b\d+%\b', '<PERCENTAGE>', text)
        
        return text
        
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text to subword tokens
        
        Args:
            text: Input text
            
        Returns:
            List of subword tokens
        """
        if self.sp_model is None:
            raise RuntimeError("Tokenizer not loaded")
            
        text = self._preprocess_entities(text)
        return self.sp_model.encode(text, out_type=str)
        
    def get_vocab_size(self) -> int:
        """Get vocabulary size"""
        return self.sp_model.get_piece_size() if self.sp_model else 0
        
    def batch_encode(self, texts: List[str], max_length: int = 512, 
                     padding: bool = True, truncation: bool = True) -> Dict[str, List[List[int]]]:
        """
        Batch encode multiple texts
        
        Args:
            texts: List of texts to encode
            max_length: Maximum sequence length
            padding: Whether to pad sequences
            truncation: Whether to truncate long sequences
            
        Returns:
            Dictionary with 'input_ids' and 'attention_mask'
        """
        encoded_texts = []
        attention_masks = []
        
        for text in texts:
            tokens = self.encode(text)
            
            # Truncate if needed
            if truncation and len(tokens) > max_length:
                tokens = tokens[:max_length]
                
            # Create attention mask (1 for real tokens, 0 for padding)
            attention_mask = [1] * len(tokens)
            
            # Pad if needed
            if padding and len(tokens) < max_length:
                pad_id = self.sp_model.piece_to_id('<PAD>')
                padding_length = max_length - len(tokens)
                tokens.extend([pad_id] * padding_length)
                attention_mask.extend([0] * padding_length)
                
            encoded_texts.append(tokens)
            attention_masks.append(attention_mask)
            
        return {
            'input_ids': encoded_texts,
            'attention_mask': attention_masks
        }
        
    def save_pretrained(self, save_directory: str):
        """Save tokenizer configuration"""
        os.makedirs(save_directory, exist_ok=True)
        
        # Copy model files
        import shutil
        shutil.copy2(self.model_path, os.path.join(save_directory, "spm.model"))
        if Path(self.vocab_path).exists():
            shutil.copy2(self.vocab_path, os.path.join(save_directory, "spm.vocab"))
            
        # Save config
        import json
        config = {
            "tokenizer_class": "TurkishMixtralTokenizer",
            "vocab_size": self.get_vocab_size(),
            "special_tokens": self.special_tokens,
            "entity_tokens": self.entity_tokens
        }
        
        with open(os.path.join(save_directory, "tokenizer_config.json"), 'w') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
            
    @classmethod
    def from_pretrained(cls, pretrained_path: str):
        """Load tokenizer from saved directory"""
        model_path = os.path.join(pretrained_path, "spm.model")
        vocab_path = os.path.join(pretrained_path, "spm.vocab")
        return cls(model_path=model_path, vocab_path=vocab_path)


class TurkishMixtralTokenizerForTransformers(PreTrainedTokenizer):
    """Transformers library compatible wrapper"""
    
    def __init__(self, model_path: str = None, **kwargs):
        self.tokenizer = TurkishMixtralTokenizer(model_path)
        
        super().__init__(
            bos_token='<s>',
            eos_token='</s>',
            unk_token='<unk>',
            pad_token='<PAD>',
            mask_token='<MASK>',
            **kwargs
        )
        
    def _tokenize(self, text: str) -> List[str]:
        return self.tokenizer.tokenize(text)
        
    def _convert_token_to_id(self, token: str) -> int:
        return self.tokenizer.sp_model.piece_to_id(token)
        
    def _convert_id_to_token(self, index: int) -> str:
        return self.tokenizer.sp_model.id_to_piece(index)
        
    def get_vocab(self) -> Dict[str, int]:
        vocab = {}
        for i in range(self.tokenizer.get_vocab_size()):
            vocab[self.tokenizer.sp_model.id_to_piece(i)] = i
        return vocab
        
    @property
    def vocab_size(self) -> int:
        return self.tokenizer.get_vocab_size()
        
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None):
        self.tokenizer.save_pretrained(save_directory)
        return (os.path.join(save_directory, "spm.model"),)


# Usage example
if __name__ == "__main__":
    # Initialize tokenizer
    tokenizer = TurkishMixtralTokenizer()
    
    # Test Turkish text
    test_texts = [
        "Merhaba, bugün yapay zeka ve makine öğrenmesi hakkında konuşacağız.",
        "2024 yılında Türkiye'de eğitim teknolojileri hızla gelişiyor.",
        "Öğrencilerin %85'i online eğitim platformlarını kullanıyor."
    ]
    
    for text in test_texts:
        tokens = tokenizer.tokenize(text)
        token_ids = tokenizer.encode(text)
        decoded = tokenizer.decode(token_ids)
        
        print(f"Original: {text}")
        print(f"Tokens ({len(tokens)}): {tokens[:10]}...")
        print(f"Decoded: {decoded}")
        print("-" * 50)
        
    # Batch encoding example
    batch_result = tokenizer.batch_encode(test_texts, max_length=128)
    print(f"\nBatch encoding shape: {len(batch_result['input_ids'])}x{len(batch_result['input_ids'][0])}")