#!/usr/bin/env python3
"""
Qwen3-8B Turkish Vocabulary Creator
Generates optimal Turkish vocabulary for tokenizer extension

HEDEF: 20,000 Turkish token addition to Qwen3's 151,936 vocabulary
FOKUS: Morphological boundaries, frequent suffixes, vowel harmony

Ultra Features:
- Morphological boundary detection (EN YÜKSEK ÖNCELİK)
- Turkish suffix system analysis
- Vowel harmony compliance
- Agglutinative structure optimization
- Smart token selection with frequency analysis
"""

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from typing import Dict, List, Tuple, Set, Optional
import logging
from tqdm import tqdm
from transformers import AutoTokenizer
import unicodedata

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Qwen3TurkishVocabCreator:
    """Creates optimal Turkish vocabulary for Qwen3-8B extension"""
    
    def __init__(self, qwen_model: str = "Qwen/Qwen3-8B"):
        self.qwen_model = qwen_model
        self.qwen_tokenizer = None
        self.target_extension_size = 20000
        
        # Turkish language features
        self.turkish_vowels = {
            'front_unrounded': ['e', 'i'],
            'front_rounded': ['ö', 'ü'], 
            'back_unrounded': ['a', 'ı'],
            'back_rounded': ['o', 'u']
        }
        
        self.all_vowels = set('aeiıoöuü')
        self.turkish_consonants = set('bcçdfgğhjklmnprsştuvyz')
        self.turkish_chars = self.all_vowels | self.turkish_consonants
        
        # Morphological patterns
        self.possessive_suffixes = [
            'im', 'in', 'i', 'imiz', 'iniz', 'leri', 'ları',
            'um', 'un', 'u', 'umuz', 'unuz', 'ları'
        ]
        
        self.case_suffixes = [
            'den', 'dan', 'e', 'a', 'de', 'da', 'i', 'ı', 'u', 'ü',
            'ten', 'tan', 'le', 'la', 'nin', 'nın', 'nun', 'nün'
        ]
        
        self.verb_suffixes = [
            'yor', 'iyor', 'uyor', 'üyor',  # Present continuous
            'di', 'dı', 'du', 'dü', 'ti', 'tı', 'tu', 'tü',  # Past
            'miş', 'mış', 'muş', 'müş',  # Evidential
            'ecek', 'acak', 'eceğ', 'acağ',  # Future
            'ir', 'ır', 'ur', 'ür', 'ar', 'er'  # Aorist
        ]
        
        self.derivational_suffixes = [
            'lik', 'lık', 'luk', 'lük', 'ci', 'cı', 'cu', 'cü',
            'sel', 'sal', 'li', 'lı', 'lu', 'lü', 'siz', 'sız', 'suz', 'süz'
        ]
        
        # High-frequency Turkish words
        self.high_freq_words = [
            'için', 'olan', 'gibi', 'çok', 'büyük', 'küçük', 'yeni', 'eski',
            'iyi', 'kötü', 'uzun', 'kısa', 'yüksek', 'alçak', 'hızlı', 'yavaş',
            'sıcak', 'soğuk', 'açık', 'kapalı', 'güzel', 'çirkin', 'temiz', 'kirli',
            'doğru', 'yanlış', 'kolay', 'zor', 'ucuz', 'pahalı', 'yakın', 'uzak'
        ]
    
    def load_qwen_tokenizer(self) -> bool:
        """Load Qwen3-8B tokenizer for overlap analysis"""
        try:
            self.qwen_tokenizer = AutoTokenizer.from_pretrained(
                self.qwen_model,
                trust_remote_code=True
            )
            logger.info(f"Loaded Qwen tokenizer: {len(self.qwen_tokenizer.vocab)} tokens")
            return True
        except Exception as e:
            logger.error(f"Failed to load Qwen tokenizer: {e}")
            return False
    
    def is_vowel_harmony_compliant(self, word: str) -> bool:
        """Check if word follows Turkish vowel harmony rules"""
        if len(word) < 2:
            return True
        
        vowels_in_word = [c for c in word.lower() if c in self.all_vowels]
        if len(vowels_in_word) < 2:
            return True
        
        # Basic vowel harmony check
        first_vowel = vowels_in_word[0]
        
        # Determine vowel group
        if first_vowel in ['a', 'ı', 'o', 'u']:  # Back vowels
            expected_groups = ['a', 'ı', 'o', 'u']
        else:  # Front vowels ['e', 'i', 'ö', 'ü']
            expected_groups = ['e', 'i', 'ö', 'ü']
        
        # Check if all vowels are in the same group (simplified check)
        harmony_score = sum(1 for v in vowels_in_word if v in expected_groups)
        return harmony_score / len(vowels_in_word) >= 0.7  # Allow some exceptions
    
    def extract_morphological_boundaries(self, text: str) -> List[str]:
        """Extract morphological boundaries from Turkish text"""
        boundaries = []
        
        # Split text into words
        words = re.findall(r'\b\w+\b', text.lower())
        
        for word in words:
            if len(word) < 3 or not any(c in self.turkish_chars for c in word):
                continue
            
            # Find suffix boundaries
            for suffix_list in [self.possessive_suffixes, self.case_suffixes, 
                              self.verb_suffixes, self.derivational_suffixes]:
                for suffix in suffix_list:
                    if word.endswith(suffix) and len(word) > len(suffix) + 2:
                        root = word[:-len(suffix)]
                        if self.is_vowel_harmony_compliant(word):
                            boundaries.append(root)
                            boundaries.append(suffix)
                            break
                else:
                    continue
                break
            else:
                # No suffix found, add whole word if it's Turkish
                if self.is_vowel_harmony_compliant(word):
                    boundaries.append(word)
        
        return boundaries
    
    def analyze_turkish_corpus(self, corpus_files: List[str]) -> Dict:
        """Analyze Turkish corpus for optimal vocabulary"""
        logger.info("Analyzing Turkish corpus for vocabulary extraction...")
        
        # Frequency counters
        word_freq = Counter()
        morpheme_freq = Counter()
        suffix_freq = Counter()
        root_freq = Counter()
        
        total_texts = 0
        
        for corpus_file in corpus_files:
            if not Path(corpus_file).exists():
                logger.warning(f"Corpus file not found: {corpus_file}")
                continue
            
            logger.info(f"Processing corpus: {corpus_file}")
            
            try:
                with open(corpus_file, 'r', encoding='utf-8') as f:
                    for line in tqdm(f, desc=f"Processing {corpus_file}"):
                        line = line.strip()
                        if not line:
                            continue
                        
                        # Extract morphological boundaries
                        boundaries = self.extract_morphological_boundaries(line)
                        
                        for boundary in boundaries:
                            if len(boundary) >= 2:
                                morpheme_freq[boundary] += 1
                                
                                # Classify as suffix or root
                                if boundary in (self.possessive_suffixes + self.case_suffixes + 
                                              self.verb_suffixes + self.derivational_suffixes):
                                    suffix_freq[boundary] += 1
                                else:
                                    root_freq[boundary] += 1
                        
                        # Also count whole words
                        words = re.findall(r'\b\w+\b', line.lower())
                        for word in words:
                            if (len(word) >= 2 and 
                                any(c in self.turkish_chars for c in word) and
                                self.is_vowel_harmony_compliant(word)):
                                word_freq[word] += 1
                        
                        total_texts += 1
                        
                        if total_texts % 10000 == 0:
                            logger.info(f"Processed {total_texts} texts...")
                            
            except Exception as e:
                logger.error(f"Error processing {corpus_file}: {e}")
        
        logger.info(f"Analysis complete: {total_texts} texts processed")
        logger.info(f"Unique words: {len(word_freq)}")
        logger.info(f"Unique morphemes: {len(morpheme_freq)}")
        logger.info(f"Unique suffixes: {len(suffix_freq)}")
        logger.info(f"Unique roots: {len(root_freq)}")
        
        return {
            'words': word_freq,
            'morphemes': morpheme_freq,
            'suffixes': suffix_freq,
            'roots': root_freq,
            'total_texts': total_texts
        }
    
    def filter_qwen_overlap(self, candidates: Dict[str, int]) -> Dict[str, int]:
        """Filter out tokens that already exist in Qwen vocabulary"""
        if not self.qwen_tokenizer:
            logger.warning("Qwen tokenizer not loaded, skipping overlap filtering")
            return candidates
        
        qwen_vocab = set(self.qwen_tokenizer.vocab.keys())
        filtered = {}
        overlap_count = 0
        
        for token, freq in candidates.items():
            # Check various forms that might exist in Qwen
            token_variants = [
                token, token.lower(), token.upper(), token.capitalize(),
                f"▁{token}", f" {token}", f"{token} "  # Common tokenizer prefixes
            ]
            
            if not any(variant in qwen_vocab for variant in token_variants):
                filtered[token] = freq
            else:
                overlap_count += 1
        
        logger.info(f"Filtered {overlap_count} overlapping tokens")
        logger.info(f"Remaining candidates: {len(filtered)}")
        
        return filtered
    
    def select_optimal_tokens(self, corpus_analysis: Dict) -> Dict[str, int]:
        """Select optimal 20K Turkish tokens for Qwen extension"""
        logger.info("Selecting optimal Turkish tokens for vocabulary extension...")
        
        all_candidates = {}
        
        # Priority 1: High-frequency Turkish words (5000 tokens)
        high_freq_words = dict(corpus_analysis['words'].most_common(10000))
        high_freq_filtered = self.filter_qwen_overlap(high_freq_words)
        
        # Add high-frequency words
        priority_words = dict(list(high_freq_filtered.items())[:5000])
        all_candidates.update(priority_words)
        logger.info(f"Added {len(priority_words)} high-frequency words")
        
        # Priority 2: Turkish suffixes (3000 tokens)
        suffix_candidates = dict(corpus_analysis['suffixes'].most_common(5000))
        suffix_filtered = self.filter_qwen_overlap(suffix_candidates)
        
        suffix_selection = {}
        for suffix, freq in suffix_filtered.items():
            if len(suffix_selection) >= 3000:
                break
            if len(suffix) >= 2 and suffix not in all_candidates:
                suffix_selection[suffix] = freq
        
        all_candidates.update(suffix_selection)
        logger.info(f"Added {len(suffix_selection)} Turkish suffixes")
        
        # Priority 3: Turkish roots (7000 tokens)
        root_candidates = dict(corpus_analysis['roots'].most_common(15000))
        root_filtered = self.filter_qwen_overlap(root_candidates)
        
        root_selection = {}
        for root, freq in root_filtered.items():
            if len(root_selection) >= 7000:
                break
            if len(root) >= 3 and root not in all_candidates:
                root_selection[root] = freq
        
        all_candidates.update(root_selection)
        logger.info(f"Added {len(root_selection)} Turkish roots")
        
        # Priority 4: Complex morphemes (3000 tokens)
        morpheme_candidates = dict(corpus_analysis['morphemes'].most_common(10000))
        morpheme_filtered = self.filter_qwen_overlap(morpheme_candidates)
        
        morpheme_selection = {}
        for morpheme, freq in morpheme_filtered.items():
            if len(morpheme_selection) >= 3000:
                break
            if (len(morpheme) >= 3 and 
                morpheme not in all_candidates and
                self.is_vowel_harmony_compliant(morpheme)):
                morpheme_selection[morpheme] = freq
        
        all_candidates.update(morpheme_selection)
        logger.info(f"Added {len(morpheme_selection)} complex morphemes")
        
        # Priority 5: Special Turkish patterns (2000 tokens)
        special_patterns = self._generate_special_patterns()
        special_filtered = self.filter_qwen_overlap(special_patterns)
        
        special_selection = {}
        for pattern, freq in special_filtered.items():
            if len(special_selection) >= 2000:
                break
            if pattern not in all_candidates:
                special_selection[pattern] = freq
        
        all_candidates.update(special_selection)
        logger.info(f"Added {len(special_selection)} special patterns")
        
        # Final selection
        final_tokens = dict(list(all_candidates.items())[:self.target_extension_size])
        
        logger.info(f"Final Turkish vocabulary: {len(final_tokens)} tokens")
        return final_tokens
    
    def _generate_special_patterns(self) -> Dict[str, int]:
        """Generate special Turkish patterns and compounds"""
        patterns = {}
        
        # Compound patterns
        compounds = [
            'okul', 'ev', 'araba', 'kitap', 'masa', 'sandalye', 'pencere', 'kapı',
            'bahçe', 'ağaç', 'çiçek', 'köpek', 'kedi', 'kuş', 'balık', 'tavuk'
        ]
        
        # Generate compound combinations
        for base in compounds:
            for suffix in ['li', 'siz', 'lik', 'ci']:
                if self.is_vowel_harmony_compliant(base + suffix):
                    patterns[base + suffix] = 100
        
        # Double consonant patterns
        double_patterns = ['ll', 'mm', 'nn', 'pp', 'ss', 'tt', 'kk', 'gg']
        for pattern in double_patterns:
            patterns[pattern] = 50
        
        # Turkish-specific letter combinations
        turkish_combinations = [
            'ğı', 'ğe', 'ği', 'ğa', 'çı', 'çe', 'çi', 'ça',
            'şı', 'şe', 'şi', 'şa', 'öy', 'üy', 'ıy', 'ey'
        ]
        
        for combo in turkish_combinations:
            patterns[combo] = 200
        
        return patterns
    
    def save_vocabulary(self, turkish_vocab: Dict[str, int], output_dir: str) -> bool:
        """Save Turkish vocabulary for tokenizer extension"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        try:
            # Assign token IDs starting from Qwen's max ID
            qwen_vocab_size = len(self.qwen_tokenizer.vocab) if self.qwen_tokenizer else 151936
            
            extension_vocab = {}
            for i, (token, freq) in enumerate(turkish_vocab.items()):
                extension_vocab[token] = qwen_vocab_size + i
            
            # Save main vocabulary file
            vocab_file = output_path / 'qwen3_turkish_extension_vocab.json'
            with open(vocab_file, 'w', encoding='utf-8') as f:
                json.dump(extension_vocab, f, ensure_ascii=False, indent=2)
            
            # Save metadata
            metadata = {
                'total_tokens': len(extension_vocab),
                'qwen_base_size': qwen_vocab_size,
                'new_vocab_size': qwen_vocab_size + len(extension_vocab),
                'extension_percentage': (len(extension_vocab) / qwen_vocab_size) * 100,
                'creation_timestamp': str(Path(__file__).stat().st_mtime)
            }
            
            metadata_file = output_path / 'extension_metadata.json'
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            
            # Save token list for training
            token_list_file = output_path / 'turkish_tokens_list.txt'
            with open(token_list_file, 'w', encoding='utf-8') as f:
                for token in extension_vocab.keys():
                    f.write(f"{token}\n")
            
            logger.info(f"Vocabulary saved to {output_path}")
            logger.info(f"Extension size: {len(extension_vocab)} tokens")
            logger.info(f"New total vocabulary: {metadata['new_vocab_size']}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to save vocabulary: {e}")
            return False


def create_qwen3_turkish_vocabulary(corpus_files: List[str], 
                                  output_dir: str = "qwen3_turkish_vocab") -> bool:
    """Main function to create Turkish vocabulary for Qwen3-8B extension"""
    
    logger.info("🚀 QWEN3-8B TURKISH VOCABULARY CREATION STARTED")
    logger.info("=" * 60)
    
    # Initialize creator
    creator = Qwen3TurkishVocabCreator()
    
    # Load Qwen tokenizer
    if not creator.load_qwen_tokenizer():
        logger.error("Failed to load Qwen tokenizer")
        return False
    
    # Analyze Turkish corpus
    corpus_analysis = creator.analyze_turkish_corpus(corpus_files)
    
    # Select optimal tokens
    optimal_tokens = creator.select_optimal_tokens(corpus_analysis)
    
    # Save vocabulary
    success = creator.save_vocabulary(optimal_tokens, output_dir)
    
    if success:
        logger.info("✅ QWEN3-8B TURKISH VOCABULARY CREATION COMPLETED")
        logger.info(f"📁 Output directory: {output_dir}")
        logger.info(f"🎯 Ready for tokenizer extension!")
    else:
        logger.error("❌ VOCABULARY CREATION FAILED")
    
    return success


if __name__ == "__main__":
    # Example usage
    corpus_files = [
        "turkish_corpus_1.txt",
        "turkish_corpus_2.txt", 
        "turkish_corpus_3.txt"
    ]
    
    success = create_qwen3_turkish_vocabulary(corpus_files)
    
    if success:
        print("\n🎉 Turkish vocabulary ready for Qwen3-8B extension!")
        print("Next: Run qwen_turkish_extender.py to extend the tokenizer")
    else:
        print("\n💥 Vocabulary creation failed!")