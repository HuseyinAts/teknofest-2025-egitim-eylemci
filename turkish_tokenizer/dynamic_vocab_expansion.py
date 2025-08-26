"""
🎯 DİNAMİK VOCABULARY EXPANSION SİSTEMİ
Training sırasında yeni Türkçe token'ları otomatik keşfet ve ekle

ÖNERİ: Training devam ederken yeni Türkçe pattern'ları tespit et
"""

import torch
import torch.nn as nn
import numpy as np
from collections import defaultdict, Counter
from typing import Dict, List, Set, Tuple
import logging
import re

logger = logging.getLogger(__name__)

class DynamicTurkishVocabExpander:
    """Training sırasında dinamik vocabulary expansion"""
    
    def __init__(self, tokenizer, model, expansion_threshold: int = 50):
        self.tokenizer = tokenizer
        self.model = model
        self.expansion_threshold = expansion_threshold
        
        # Turkish pattern detection
        self.turkish_morphemes = Counter()
        self.new_turkish_patterns = set()
        self.expansion_candidates = defaultdict(int)
        
        # Vowel harmony patterns
        self.vowel_groups = {
            'front_unrounded': set('ei'),
            'front_rounded': set('öü'),
            'back_unrounded': set('aı'),
            'back_rounded': set('ou')
        }
    
    def analyze_training_batch(self, input_texts: List[str]):
        """Her batch'te yeni Türkçe pattern'ları analiz et"""
        
        for text in input_texts:
            # Inefficient tokenization patterns tespit et
            tokens = self.tokenizer.tokenize(text)
            
            # Çok fazla parçalanmış Türkçe kelimeler bul
            words = text.split()
            for word in words:
                if self._is_turkish_word(word):
                    word_tokens = self.tokenizer.tokenize(word)
                    
                    # Eğer kelime 4+ token'a bölünüyorsa expansion candidate
                    if len(word_tokens) >= 4:
                        self.expansion_candidates[word] += 1
                        
                        # Morphological analysis
                        morphemes = self._extract_morphemes(word)
                        for morpheme in morphemes:
                            self.turkish_morphemes[morpheme] += 1
    
    def _is_turkish_word(self, word: str) -> bool:
        """Türkçe kelime kontrolü"""
        turkish_chars = set('çğıöşüÇĞİÖŞÜ')
        return any(c in turkish_chars for c in word) or self._check_vowel_harmony(word)
    
    def _check_vowel_harmony(self, word: str) -> bool:
        """Ünlü uyumu kontrolü"""
        vowels = [c.lower() for c in word if c.lower() in 'aeiıoöuü']
        
        if len(vowels) < 2:
            return False
        
        # Front/back harmony check
        front_vowels = sum(1 for v in vowels if v in 'eiöü')
        back_vowels = sum(1 for v in vowels if v in 'aıou')
        
        # Strong harmony = one type dominant
        return max(front_vowels, back_vowels) / len(vowels) > 0.8
    
    def _extract_morphemes(self, word: str) -> List[str]:
        """Basit morpheme extraction"""
        
        # Turkish suffixes
        suffixes = ['ler', 'lar', 'de', 'da', 'den', 'dan', 'ye', 'ya', 
                   'nin', 'nın', 'nün', 'nun', 'mek', 'mak', 'yor', 'dı', 'di']
        
        morphemes = []
        remaining = word.lower()
        
        # Suffix detection
        for suffix in sorted(suffixes, key=len, reverse=True):
            if remaining.endswith(suffix) and len(remaining) > len(suffix) + 2:
                morphemes.append(suffix)
                remaining = remaining[:-len(suffix)]
                break
        
        if remaining:
            morphemes.insert(0, remaining)
        
        return morphemes if len(morphemes) > 1 else [word]
    
    def get_expansion_recommendations(self) -> Dict[str, int]:
        """Expansion önerileri al"""
        
        # Threshold'u geçen candidates
        candidates = {word: count for word, count in self.expansion_candidates.items() 
                     if count >= self.expansion_threshold}
        
        # Morpheme frequency analysis
        frequent_morphemes = {morpheme: count for morpheme, count in self.turkish_morphemes.items()
                            if count >= self.expansion_threshold // 2}
        
        logger.info(f"🎯 Expansion candidates: {len(candidates)} words, {len(frequent_morphemes)} morphemes")
        
        return {
            'inefficient_words': candidates,
            'frequent_morphemes': frequent_morphemes,
            'total_expansion_potential': len(candidates) + len(frequent_morphemes)
        }
    
    def expand_vocabulary_runtime(self, new_tokens: List[str]):
        """Runtime'da vocabulary expansion"""
        
        if not new_tokens:
            return
        
        logger.info(f"🔧 Runtime vocabulary expansion: {len(new_tokens)} new tokens")
        
        # Current vocabulary size
        current_size = len(self.tokenizer)
        
        # Add new tokens to tokenizer
        self.tokenizer.add_tokens(new_tokens)
        new_size = len(self.tokenizer)
        
        # Resize model embeddings
        self.model.resize_token_embeddings(new_size)
        
        # Initialize new embeddings smartly
        with torch.no_grad():
            embedding_layer = self.model.get_input_embeddings()
            
            # Use average of Turkish tokens for initialization
            turkish_token_ids = self._get_turkish_token_ids()
            if turkish_token_ids:
                avg_embedding = embedding_layer.weight[turkish_token_ids].mean(dim=0)
                
                # Initialize new tokens with average + noise
                for i in range(current_size, new_size):
                    noise = torch.randn_like(avg_embedding) * 0.02
                    embedding_layer.weight[i] = avg_embedding + noise
        
        logger.info(f"✅ Vocabulary expanded: {current_size} -> {new_size}")
    
    def _get_turkish_token_ids(self) -> List[int]:
        """Türkçe token ID'lerini bul"""
        
        turkish_chars = set('çğıöşüÇĞİÖŞÜ')
        turkish_ids = []
        
        for token_id, token in enumerate(self.tokenizer.get_vocab()):
            if any(c in turkish_chars for c in token):
                turkish_ids.append(token_id)
        
        return turkish_ids


class AdaptiveVocabularyCallback:
    """Training sırasında vocabulary expansion callback"""
    
    def __init__(self, expander: DynamicTurkishVocabExpander, 
                 expansion_frequency: int = 1000):
        self.expander = expander
        self.expansion_frequency = expansion_frequency
        self.step_count = 0
        self.expansions_done = 0
    
    def on_train_batch_end(self, batch_texts: List[str]):
        """Her batch sonunda analysis"""
        
        self.step_count += 1
        self.expander.analyze_training_batch(batch_texts)
        
        # Periodic expansion check
        if self.step_count % self.expansion_frequency == 0:
            recommendations = self.expander.get_expansion_recommendations()
            
            if recommendations['total_expansion_potential'] > 20:
                # Select top candidates for expansion
                new_tokens = []
                
                # Add most frequent inefficient words
                inefficient = recommendations['inefficient_words']
                top_words = sorted(inefficient.items(), key=lambda x: x[1], reverse=True)[:10]
                new_tokens.extend([word for word, _ in top_words])
                
                # Add frequent morphemes
                morphemes = recommendations['frequent_morphemes']
                top_morphemes = sorted(morphemes.items(), key=lambda x: x[1], reverse=True)[:5]
                new_tokens.extend([morpheme for morpheme, _ in top_morphemes])
                
                if new_tokens:
                    self.expander.expand_vocabulary_runtime(new_tokens)
                    self.expansions_done += 1
                    
                    logger.info(f"🎯 Dynamic expansion #{self.expansions_done}: {len(new_tokens)} tokens")


def integrate_dynamic_expansion(trainer, tokenizer, model):
    """Trainer'a dynamic expansion entegre et"""
    
    # Create expander
    expander = DynamicTurkishVocabExpander(tokenizer, model)
    callback = AdaptiveVocabularyCallback(expander, expansion_frequency=500)
    
    # Trainer'a callback ekle
    class DynamicExpansionTrainer(trainer.__class__):
        def __init__(self, *args, vocab_callback=None, **kwargs):
            super().__init__(*args, **kwargs)
            self.vocab_callback = vocab_callback
        
        def training_step(self, model, inputs):
            # Standard training step
            loss = super().training_step(model, inputs)
            
            # Extract texts for vocabulary analysis
            if self.vocab_callback and hasattr(inputs, 'input_ids'):
                try:
                    batch_texts = []
                    for input_ids in inputs['input_ids']:
                        text = self.tokenizer.decode(input_ids, skip_special_tokens=True)
                        batch_texts.append(text)
                    
                    self.vocab_callback.on_train_batch_end(batch_texts)
                except:
                    pass  # Silent fail for analysis
            
            return loss
    
    # Replace trainer class
    trainer.__class__ = DynamicExpansionTrainer
    trainer.vocab_callback = callback
    
    logger.info("✅ Dynamic vocabulary expansion integrated")
    
    return trainer, expander


# Test function
def test_dynamic_expansion():
    """Dynamic expansion test"""
    
    print("🧪 Dynamic Vocabulary Expansion testi...")
    
    # Mock tokenizer and model
    class MockTokenizer:
        def __init__(self):
            self.vocab = {f"token_{i}": i for i in range(1000)}
        
        def tokenize(self, text):
            return text.split()[:5]  # Simulate tokenization
        
        def __len__(self):
            return len(self.vocab)
        
        def add_tokens(self, tokens):
            start_id = len(self.vocab)
            for i, token in enumerate(tokens):
                self.vocab[token] = start_id + i
        
        def get_vocab(self):
            return self.vocab
    
    class MockModel:
        def __init__(self):
            self.embedding_size = 768
            self._embeddings = torch.randn(1000, self.embedding_size)
        
        def get_input_embeddings(self):
            return self
        
        @property
        def weight(self):
            return self._embeddings
        
        def resize_token_embeddings(self, new_size):
            if new_size > self._embeddings.size(0):
                additional = torch.randn(new_size - self._embeddings.size(0), self.embedding_size)
                self._embeddings = torch.cat([self._embeddings, additional], dim=0)
    
    tokenizer = MockTokenizer()
    model = MockModel()
    
    # Test expansion
    expander = DynamicTurkishVocabExpander(tokenizer, model, expansion_threshold=2)
    
    # Simulate training data
    turkish_texts = [
        "Bu çok güzel bir günlük aktivitesidir",
        "Türkiye'nin en büyük şehirlerinden biridir", 
        "Eğitim sistemimizin geliştirilmesi gerekiyor",
        "Bu çok güzel bir günlük aktivitesidir",  # Repeat for frequency
        "Türkiye'nin en büyük şehirlerinden biridir",
    ]
    
    # Analyze batches
    for i in range(3):
        expander.analyze_training_batch(turkish_texts)
    
    # Get recommendations
    recommendations = expander.get_expansion_recommendations()
    
    print(f"✅ Inefficient words: {len(recommendations['inefficient_words'])}")
    print(f"✅ Frequent morphemes: {len(recommendations['frequent_morphemes'])}")
    print(f"✅ Total expansion potential: {recommendations['total_expansion_potential']}")
    
    # Test expansion
    if recommendations['inefficient_words']:
        top_words = list(recommendations['inefficient_words'].keys())[:3]
        original_size = len(tokenizer)
        expander.expand_vocabulary_runtime(top_words)
        new_size = len(tokenizer)
        
        print(f"✅ Vocabulary expanded: {original_size} -> {new_size}")
    
    print("🎉 Dynamic expansion test completed!")


if __name__ == "__main__":
    test_dynamic_expansion()