"""
Turkish BPE Tokenizer Implementation
TEKNOFEST 2025 - Custom Turkish Tokenizer with ByteLevelBPETokenizer
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from collections import Counter, defaultdict
import regex as re
from dataclasses import dataclass, field
import pickle
import numpy as np
from tqdm import tqdm

# Import trie optimization
try:
    import sys
    sys.path.append(str(Path(__file__).parent.parent.parent / 'turkish_tokenizer'))
    from trie_suffix_optimization import TurkishSuffixTrie, OptimizedTurkishBPE
    TRIE_OPTIMIZATION_AVAILABLE = True
except ImportError:
    TRIE_OPTIMIZATION_AVAILABLE = False


@dataclass
class TokenizerConfig:
    """Configuration for Turkish BPE Tokenizer"""
    vocab_size: int = 32000
    min_frequency: int = 2
    special_tokens: List[str] = field(default_factory=lambda: [
        "<pad>", "<unk>", "<bos>", "<eos>", "<mask>",
        "<turk_num>", "<turk_date>", "<turk_currency>", "<turk_person>",
        "<turk_location>", "<turk_org>", "<turk_time>", "<turk_percent>"
    ])
    pre_tokenizer_regex: str = r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
    continuing_subword_prefix: str = "##"
    end_of_word_suffix: str = "</w>"
    lowercase: bool = False
    trim_offsets: bool = True
    turkish_specific: bool = True
    byte_fallback: bool = True


class TurkishBPETokenizer:
    """ByteLevel BPE Tokenizer optimized for Turkish language"""
    
    def __init__(self, config: Optional[TokenizerConfig] = None):
        self.config = config or TokenizerConfig()
        self.vocab = {}
        self.merges = []
        self.special_tokens_map = {}
        self.byte_encoder = self._bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        
        # Turkish-specific character handling
        self.turkish_chars = set("çğıöşüÇĞİÖŞÜ")
        self.turkish_char_map = {
            'ı': 'i', 'İ': 'I', 'ş': 's', 'Ş': 'S',
            'ğ': 'g', 'Ğ': 'G', 'ü': 'u', 'Ü': 'U',
            'ö': 'o', 'Ö': 'O', 'ç': 'c', 'Ç': 'C'
        }
        
        # Initialize special tokens
        self._init_special_tokens()
        
        # Pre-compiled regex patterns
        self.word_tokenizer = re.compile(self.config.pre_tokenizer_regex)
        self.turkish_patterns = self._compile_turkish_patterns()
        
        # Initialize trie optimization if available
        if TRIE_OPTIMIZATION_AVAILABLE:
            self.suffix_trie = TurkishSuffixTrie()
            self.trie_optimization = True
            print("✅ Trie-based suffix optimization enabled - O(log n) performance!")
        else:
            self.suffix_trie = None
            self.trie_optimization = False
            print("⚠️ Trie optimization not available - using O(n) linear search")
        
    def _bytes_to_unicode(self) -> Dict[int, str]:
        """Create byte to unicode mapping for byte-level BPE"""
        bs = list(range(ord("!"), ord("~")+1)) + \
             list(range(ord("¡"), ord("¬")+1)) + \
             list(range(ord("®"), ord("ÿ")+1))
        cs = bs[:]
        n = 0
        
        for b in range(2**8):
            if b not in bs:
                bs.append(b)
                cs.append(2**8+n)
                n += 1
                
        cs = [chr(n) for n in cs]
        return dict(zip(bs, cs))
    
    def _compile_turkish_patterns(self) -> Dict[str, re.Pattern]:
        """Compile Turkish-specific regex patterns"""
        patterns = {
            'number': re.compile(r'\d+(?:[.,]\d+)*'),
            'date': re.compile(r'\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}'),
            'time': re.compile(r'\d{1,2}:\d{2}(?::\d{2})?'),
            'currency': re.compile(r'[₺$€£¥]\s*\d+(?:[.,]\d+)*|\d+(?:[.,]\d+)*\s*(?:TL|USD|EUR|GBP)'),
            'percent': re.compile(r'%\s*\d+(?:[.,]\d+)*|\d+(?:[.,]\d+)*\s*%'),
            'email': re.compile(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'),
            'url': re.compile(r'https?://[^\s]+|www\.[^\s]+'),
            'mention': re.compile(r'@[a-zA-Z0-9_]+'),
            'hashtag': re.compile(r'#[a-zA-Z0-9_çğıöşüÇĞİÖŞÜ]+'),
        }
        return patterns
    
    def _init_special_tokens(self):
        """Initialize special tokens with IDs"""
        for i, token in enumerate(self.config.special_tokens):
            self.special_tokens_map[token] = i
            self.vocab[token] = i
            
    def pre_tokenize(self, text: str) -> List[str]:
        """Pre-tokenize text with Turkish-specific handling"""
        # Normalize Turkish characters if needed
        if self.config.turkish_specific:
            text = self._normalize_turkish(text)
            
        # Replace special patterns with tokens
        text = self._replace_special_patterns(text)
        
        # Apply word tokenization
        words = []
        for match in self.word_tokenizer.finditer(text):
            word = match.group(0)
            
            # Handle Turkish suffixes
            if self.config.turkish_specific:
                word = self._handle_turkish_suffixes(word)
                
            words.append(word)
            
        return words
    
    def _normalize_turkish(self, text: str) -> str:
        """Normalize Turkish-specific characters and patterns"""
        # Preserve original for now, just clean up
        text = text.replace("İ", "İ")  # Ensure proper Turkish capital I
        text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
        return text
    
    def _replace_special_patterns(self, text: str) -> str:
        """Replace special patterns with tokens"""
        for pattern_name, pattern in self.turkish_patterns.items():
            token_name = f"<turk_{pattern_name}>"
            if token_name in self.special_tokens_map:
                text = pattern.sub(token_name, text)
        return text
    
    def _handle_turkish_suffixes(self, word: str) -> str:
        """Handle Turkish agglutinative suffixes with trie optimization"""
        
        if self.trie_optimization and self.suffix_trie:
            # Use O(log n) trie-based suffix recognition
            suffix, suffix_type, pos = self.suffix_trie.find_longest_suffix(word)
            
            if suffix and pos > 0:  # Found a valid suffix
                # Mark suffix boundary for better subword learning
                return word[:pos] + self.config.continuing_subword_prefix + word[pos:]
        else:
            # Fallback to O(n) linear search
            suffixes = [
                "lar", "ler",  # Plural
                "dan", "den", "tan", "ten",  # Ablative
                "da", "de", "ta", "te",  # Locative
                "nın", "nin", "nun", "nün",  # Genitive
                "ı", "i", "u", "ü",  # Accusative
                "a", "e",  # Dative
                "la", "le",  # Instrumental
                "ca", "ce", "ça", "çe",  # Adverbial
                "lı", "li", "lu", "lü",  # With/having
                "sız", "siz", "suz", "süz",  # Without
                "lik", "lık", "luk", "lük",  # -ness
            ]
            
            # O(n) linear search through suffixes
            for suffix in suffixes:
                if word.endswith(suffix) and len(word) > len(suffix) + 2:
                    # Insert subword boundary marker
                    pos = len(word) - len(suffix)
                    return word[:pos] + self.config.continuing_subword_prefix + word[pos:]
                
        return word
    
    def _get_word_frequency(self, corpus: List[str]) -> Dict[str, int]:
        """Get word frequencies from corpus"""
        word_freq = Counter()
        
        for text in tqdm(corpus, desc="Counting words"):
            words = self.pre_tokenize(text)
            word_freq.update(words)
            
        return word_freq
    
    def _get_byte_pair_frequencies(self, word_freq: Dict[str, int]) -> Dict[Tuple[str, str], int]:
        """Get frequencies of adjacent byte pairs"""
        pair_freq = defaultdict(int)
        
        for word, freq in word_freq.items():
            # Convert word to bytes and then to unicode representation
            word_bytes = word.encode('utf-8')
            word_tokens = [self.byte_encoder[b] for b in word_bytes]
            
            # Add end of word token
            word_tokens.append(self.config.end_of_word_suffix)
            
            # Count pairs
            for i in range(len(word_tokens) - 1):
                pair = (word_tokens[i], word_tokens[i + 1])
                pair_freq[pair] += freq
                
        return pair_freq
    
    def _merge_pair(self, pair: Tuple[str, str], word_tokens: List[str]) -> List[str]:
        """Merge a specific pair in word tokens"""
        merged = []
        i = 0
        
        while i < len(word_tokens):
            if i < len(word_tokens) - 1 and \
               word_tokens[i] == pair[0] and word_tokens[i + 1] == pair[1]:
                merged.append(pair[0] + pair[1])
                i += 2
            else:
                merged.append(word_tokens[i])
                i += 1
                
        return merged
    
    def train(self, corpus: List[str], vocab_size: Optional[int] = None):
        """Train BPE tokenizer on Turkish corpus"""
        vocab_size = vocab_size or self.config.vocab_size
        
        print("Training Turkish BPE Tokenizer...")
        print(f"Target vocabulary size: {vocab_size}")
        
        # Get word frequencies
        word_freq = self._get_word_frequency(corpus)
        
        # Filter by minimum frequency
        word_freq = {w: f for w, f in word_freq.items() 
                    if f >= self.config.min_frequency}
        
        print(f"Unique words after filtering: {len(word_freq)}")
        
        # Initialize vocabulary with bytes
        for byte_val in self.byte_encoder.values():
            if byte_val not in self.vocab:
                self.vocab[byte_val] = len(self.vocab)
                
        # Add end of word suffix
        if self.config.end_of_word_suffix not in self.vocab:
            self.vocab[self.config.end_of_word_suffix] = len(self.vocab)
            
        # Learn merges
        word_splits = {}
        for word in word_freq:
            word_bytes = word.encode('utf-8')
            word_tokens = [self.byte_encoder[b] for b in word_bytes]
            word_tokens.append(self.config.end_of_word_suffix)
            word_splits[word] = word_tokens
            
        pbar = tqdm(total=vocab_size - len(self.vocab), desc="Learning merges")
        
        while len(self.vocab) < vocab_size:
            # Get pair frequencies
            pair_freq = defaultdict(int)
            for word, freq in word_freq.items():
                word_tokens = word_splits[word]
                for i in range(len(word_tokens) - 1):
                    pair = (word_tokens[i], word_tokens[i + 1])
                    pair_freq[pair] += freq
                    
            if not pair_freq:
                break
                
            # Find most frequent pair
            best_pair = max(pair_freq, key=pair_freq.get)
            
            # Create new token
            new_token = best_pair[0] + best_pair[1]
            
            # Add to vocabulary
            if new_token not in self.vocab:
                self.vocab[new_token] = len(self.vocab)
                self.merges.append(best_pair)
                
            # Update word splits
            for word in word_splits:
                word_splits[word] = self._merge_pair(best_pair, word_splits[word])
                
            pbar.update(1)
            
        pbar.close()
        print(f"Training complete. Vocabulary size: {len(self.vocab)}")
        print(f"Number of merges: {len(self.merges)}")
        
    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """Encode text to token IDs"""
        token_ids = []
        
        if add_special_tokens:
            token_ids.append(self.special_tokens_map["<bos>"])
            
        # Pre-tokenize
        words = self.pre_tokenize(text)
        
        for word in words:
            # Check if it's a special token
            if word in self.special_tokens_map:
                token_ids.append(self.special_tokens_map[word])
                continue
                
            # Convert to bytes and encode
            word_bytes = word.encode('utf-8')
            word_tokens = [self.byte_encoder[b] for b in word_bytes]
            
            # Apply merges
            for merge in self.merges:
                word_tokens = self._merge_pair(merge, word_tokens)
                
            # Convert to IDs
            for token in word_tokens:
                if token in self.vocab:
                    token_ids.append(self.vocab[token])
                else:
                    token_ids.append(self.special_tokens_map["<unk>"])
                    
        if add_special_tokens:
            token_ids.append(self.special_tokens_map["<eos>"])
            
        return token_ids
    
    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """Decode token IDs to text"""
        tokens = []
        
        # Reverse mappings
        id_to_token = {v: k for k, v in self.vocab.items()}
        id_to_special = {v: k for k, v in self.special_tokens_map.items()}
        
        for token_id in token_ids:
            if token_id in id_to_special:
                if not skip_special_tokens:
                    tokens.append(id_to_special[token_id])
            elif token_id in id_to_token:
                tokens.append(id_to_token[token_id])
                
        # Join tokens and convert back from byte encoding
        text = "".join(tokens)
        
        # Remove end of word markers
        text = text.replace(self.config.end_of_word_suffix, " ")
        
        # Convert from byte representation
        text_bytes = []
        for char in text:
            if char in self.byte_decoder:
                text_bytes.append(self.byte_decoder[char])
            else:
                text_bytes.append(ord(char))
                
        try:
            text = bytes(text_bytes).decode('utf-8', errors='ignore')
        except:
            text = text  # Fallback to original if decoding fails
            
        # Clean up
        text = text.replace(self.config.continuing_subword_prefix, "")
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into subword tokens"""
        token_ids = self.encode(text, add_special_tokens=False)
        
        id_to_token = {v: k for k, v in self.vocab.items()}
        id_to_special = {v: k for k, v in self.special_tokens_map.items()}
        
        tokens = []
        for token_id in token_ids:
            if token_id in id_to_special:
                tokens.append(id_to_special[token_id])
            elif token_id in id_to_token:
                token = id_to_token[token_id]
                # Clean token representation
                if token != self.config.end_of_word_suffix:
                    tokens.append(token)
                    
        return tokens
    
    def save(self, directory: str):
        """Save tokenizer to directory"""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        
        # Save vocabulary
        with open(directory / "vocab.json", "w", encoding="utf-8") as f:
            json.dump(self.vocab, f, ensure_ascii=False, indent=2)
            
        # Save merges
        with open(directory / "merges.txt", "w", encoding="utf-8") as f:
            for pair in self.merges:
                f.write(f"{pair[0]} {pair[1]}\n")
                
        # Save config
        config_dict = {
            "vocab_size": self.config.vocab_size,
            "min_frequency": self.config.min_frequency,
            "special_tokens": self.config.special_tokens,
            "continuing_subword_prefix": self.config.continuing_subword_prefix,
            "end_of_word_suffix": self.config.end_of_word_suffix,
            "lowercase": self.config.lowercase,
            "turkish_specific": self.config.turkish_specific,
        }
        
        with open(directory / "tokenizer_config.json", "w", encoding="utf-8") as f:
            json.dump(config_dict, f, ensure_ascii=False, indent=2)
            
        print(f"Tokenizer saved to {directory}")
        
    @classmethod
    def load(cls, directory: str) -> 'TurkishBPETokenizer':
        """Load tokenizer from directory"""
        directory = Path(directory)
        
        # Load config
        with open(directory / "tokenizer_config.json", "r", encoding="utf-8") as f:
            config_dict = json.load(f)
            
        config = TokenizerConfig(**config_dict)
        tokenizer = cls(config)
        
        # Load vocabulary
        with open(directory / "vocab.json", "r", encoding="utf-8") as f:
            tokenizer.vocab = json.load(f)
            
        # Load merges
        tokenizer.merges = []
        with open(directory / "merges.txt", "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    tokenizer.merges.append(tuple(parts))
                    
        print(f"Tokenizer loaded from {directory}")
        return tokenizer


def train_turkish_tokenizer(corpus_path: str, save_path: str, vocab_size: int = 32000):
    """Train a Turkish BPE tokenizer on a corpus"""
    print("Loading Turkish corpus...")
    
    # Load corpus (example implementation)
    corpus = []
    if os.path.exists(corpus_path):
        with open(corpus_path, "r", encoding="utf-8") as f:
            corpus = [line.strip() for line in f if line.strip()]
    else:
        # Use sample Turkish texts for demonstration
        corpus = [
            "Merhaba, bugün hava çok güzel.",
            "Türkiye'nin başkenti Ankara'dır.",
            "İstanbul Boğazı'nda vapurla gezinti yapmak çok keyifli.",
            "Öğrenciler sınavlara hazırlanıyor.",
            "Yapay zeka teknolojileri hızla gelişiyor.",
            "TEKNOFEST 2025 yarışmasına hazırlanıyoruz.",
            "Türkçe doğal dil işleme çalışmaları önem kazanıyor.",
            "Eğitim teknolojileri öğrenmeyi kolaylaştırıyor.",
        ]
        
    # Initialize tokenizer
    config = TokenizerConfig(
        vocab_size=vocab_size,
        turkish_specific=True,
        min_frequency=1  # Low for small corpus
    )
    
    tokenizer = TurkishBPETokenizer(config)
    
    # Train tokenizer
    tokenizer.train(corpus, vocab_size)
    
    # Save tokenizer
    tokenizer.save(save_path)
    
    # Test tokenizer
    test_sentences = [
        "Merhaba dünya!",
        "Türkiye'de eğitim sistemi gelişiyor.",
        "Öğrencilerin başarısı için çalışıyoruz.",
        "İstanbul'dan Ankara'ya gidiyorum.",
    ]
    
    print("\nTokenizer Test Results:")
    print("=" * 50)
    
    for sentence in test_sentences:
        tokens = tokenizer.tokenize(sentence)
        token_ids = tokenizer.encode(sentence)
        decoded = tokenizer.decode(token_ids)
        
        print(f"\nOriginal: {sentence}")
        print(f"Tokens: {tokens}")
        print(f"Token IDs: {token_ids}")
        print(f"Decoded: {decoded}")
        
    return tokenizer


if __name__ == "__main__":
    # Train tokenizer
    tokenizer = train_turkish_tokenizer(
        corpus_path="data/turkish_corpus.txt",
        save_path="models/turkish_bpe_tokenizer",
        vocab_size=32000
    )