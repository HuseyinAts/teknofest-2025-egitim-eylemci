"""
Turkish Vocabulary Analyzer for Qwen3-8B Extension
Analyzes Turkish corpus to extract optimal vocabulary for tokenizer extension

Key Features:
- Morphological boundary detection (highest priority)
- Frequent suffix identification 
- Vowel harmony rule analysis
- Agglutinative structure optimization
- Target: 30K-50K Turkish tokens for aggressive optimization
"""

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from typing import Dict, List, Tuple, Set
import logging
from tqdm import tqdm
from transformers import AutoTokenizer
import unicodedata

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TurkishVocabularyAnalyzer:
    """Advanced vocabulary analyzer for Turkish tokenizer extension"""
    
    def __init__(self, qwen_model_name: str = "Qwen/Qwen3-8B"):
        self.qwen_tokenizer = None
        self.qwen_vocab_size = 151936  # Original Qwen3-8B vocab size
        
        # Turkish language features
        self.turkish_vowels = {
            'front_unrounded': ['e', 'i'],
            'front_rounded': ['ö', 'ü'], 
            'back_unrounded': ['a', 'ı'],
            'back_rounded': ['o', 'u']
        }
        
        self.all_vowels = set('aeiıoöuü')
        self.turkish_consonants = set('bcçdfgğhjklmnprsştuvyz')
        
        # Common Turkish morphemes and suffixes
        self.common_suffixes = [
            # Plural
            'ler', 'lar',
            # Cases  
            'de', 'da', 'den', 'dan', 'ye', 'ya', 'nin', 'nın', 'nün', 'nun',
            'in', 'ın', 'ün', 'un', 'e', 'a', 'i', 'ı', 'u', 'ü', 'o', 'ö',
            # Possessives
            'im', 'ım', 'üm', 'um', 'in', 'ın', 'ün', 'un', 'i', 'ı', 'ü', 'u',
            'miz', 'mız', 'müz', 'muz', 'niz', 'nız', 'nüz', 'nuz', 'leri', 'ları',
            # Verb forms
            'mek', 'mak', 'acak', 'ecek', 'dı', 'di', 'du', 'dü', 'tı', 'ti', 'tu', 'tü',
            'mış', 'miş', 'muş', 'müş', 'sa', 'se', 'yor', 'iyor', 'üyor', 'uyor',
            'bilir', 'bilmez', 'abilir', 'ebilir', 'amamak', 'ememek',
            # Adjective/Adverb forming
            'lı', 'li', 'lu', 'lü', 'sız', 'siz', 'suz', 'süz', 'ca', 'ce',
            # Question particles
            'mi', 'mı', 'mu', 'mü',
        ]
        
        # Morphological patterns
        self.morphological_patterns = [
            # Root + suffix patterns
            r'\w+(?:ler|lar)',  # Plurals
            r'\w+(?:de|da|den|dan)',  # Locatives
            r'\w+(?:ye|ya|e|a)',  # Datives
            r'\w+(?:nin|nın|nün|nun)',  # Genitives
            r'\w+(?:mek|mak)',  # Infinitives
            r'\w+(?:yor|iyor|üyor|uyor)',  # Present continuous
            r'\w+(?:dı|di|du|dü|tı|ti|tu|tü)',  # Past tense
            r'\w+(?:acak|ecek)',  # Future tense
            r'\w+(?:mış|miş|muş|müş)',  # Evidential
        ]
        
        self.vocabulary_stats = {
            'total_words': 0,
            'unique_words': 0,
            'turkish_words': 0,
            'morphologically_complex': 0,
            'new_tokens_identified': 0,
            'coverage_improvement': 0
        }
        
    def load_qwen_tokenizer(self):
        """Load Qwen3-8B tokenizer"""
        try:
            logger.info("Loading Qwen3-8B tokenizer...")
            self.qwen_tokenizer = AutoTokenizer.from_pretrained(
                "Qwen/Qwen3-8B",
                trust_remote_code=True
            )
            logger.info(f"Qwen tokenizer loaded. Vocab size: {len(self.qwen_tokenizer.vocab)}")
            return True
        except Exception as e:
            logger.error(f"Failed to load Qwen tokenizer: {e}")
            return False
    
    def analyze_vowel_harmony(self, word: str) -> Dict[str, any]:
        """Analyze vowel harmony in Turkish word"""
        vowels = [char for char in word.lower() if char in self.all_vowels]
        
        if len(vowels) < 2:
            return {'harmony': True, 'type': 'insufficient_data', 'vowels': vowels}
        
        # Check fronting harmony (e, i, ö, ü vs a, ı, o, u)
        front_vowels = set('eiöü')
        back_vowels = set('aıou')
        
        front_count = sum(1 for v in vowels if v in front_vowels)
        back_count = sum(1 for v in vowels if v in back_vowels)
        
        # Turkish vowel harmony rules
        harmony_score = 0
        if front_count > 0 and back_count == 0:
            harmony_score = 1  # All front
        elif back_count > 0 and front_count == 0:
            harmony_score = 1  # All back
        else:
            # Mixed - calculate harmony violations
            harmony_score = max(front_count, back_count) / len(vowels)
        
        return {
            'harmony': harmony_score > 0.8,
            'score': harmony_score,
            'vowels': vowels,
            'front_count': front_count,
            'back_count': back_count
        }
    
    def detect_morphological_boundaries(self, word: str) -> List[str]:
        """Detect morphological boundaries in Turkish words"""
        if len(word) < 4:
            return [word]
        
        boundaries = []
        word_lower = word.lower()
        
        # Try to identify root + suffix combinations
        for suffix in sorted(self.common_suffixes, key=len, reverse=True):
            if word_lower.endswith(suffix) and len(word_lower) > len(suffix) + 2:
                root = word_lower[:-len(suffix)]
                boundaries.extend([root, suffix])
                break
        
        if not boundaries:
            # Try pattern-based segmentation
            for pattern in self.morphological_patterns:
                matches = re.finditer(pattern, word_lower)
                for match in matches:
                    # Simple heuristic segmentation
                    matched_text = match.group()
                    if len(matched_text) == len(word_lower):
                        # Try to split at morpheme boundary
                        for suffix in self.common_suffixes:
                            if matched_text.endswith(suffix):
                                root = matched_text[:-len(suffix)]
                                if len(root) >= 3:
                                    boundaries.extend([root, suffix])
                                    break
        
        return boundaries if boundaries else [word]
    
    def calculate_turkish_specificity(self, word: str) -> float:
        """Calculate how Turkish-specific a word is"""
        score = 0.0
        word_lower = word.lower()
        
        # Turkish characters present
        turkish_chars = set('çğıöşüâîû')
        if any(char in turkish_chars for char in word_lower):
            score += 0.3
        
        # Vowel harmony
        harmony_info = self.analyze_vowel_harmony(word)
        if harmony_info['harmony']:
            score += 0.2
        
        # Morphological complexity (agglutination)
        if len(word) > 8:  # Longer words are more likely agglutinative
            score += 0.2
        
        # Turkish suffix patterns
        for suffix in self.common_suffixes:
            if word_lower.endswith(suffix):
                score += 0.2
                break
        
        # Consonant clusters typical in Turkish
        if re.search(r'[çğş]', word_lower):
            score += 0.1
        
        return min(score, 1.0)
    
    def analyze_tokenization_efficiency(self, text: str) -> Dict:
        """Analyze current Qwen tokenizer efficiency on Turkish text"""
        if not self.qwen_tokenizer:
            return {'error': 'Qwen tokenizer not loaded'}
        
        # Original tokenization
        tokens = self.qwen_tokenizer.tokenize(text)
        token_ids = self.qwen_tokenizer.encode(text)
        
        # Calculate efficiency metrics
        char_to_token_ratio = len(text) / len(tokens) if tokens else 0
        word_to_token_ratio = len(text.split()) / len(tokens) if tokens else 0
        
        # Identify undertokenized words (words split into many subwords)
        words = text.split()
        undertokenized_words = []
        
        for word in words:
            word_tokens = self.qwen_tokenizer.tokenize(word)
            if len(word_tokens) > 3:  # Word split into more than 3 tokens
                undertokenized_words.append({
                    'word': word,
                    'tokens': word_tokens,
                    'token_count': len(word_tokens)
                })
        
        return {
            'total_tokens': len(tokens),
            'total_chars': len(text),
            'total_words': len(words),
            'char_to_token_ratio': char_to_token_ratio,
            'word_to_token_ratio': word_to_token_ratio,
            'undertokenized_words': undertokenized_words,
            'efficiency_score': char_to_token_ratio / 4.0  # Normalized efficiency
        }
    
    def extract_vocabulary_candidates(self, corpus_data: List[Dict], 
                                    target_vocab_size: int = 40000) -> Dict:
        """Extract Turkish vocabulary candidates for tokenizer extension"""
        
        logger.info(f"Extracting vocabulary candidates for {target_vocab_size} new tokens...")
        
        # Collect all text
        all_texts = [item['text'] for item in corpus_data]
        combined_text = ' '.join(all_texts)
        
        # Word frequency analysis
        words = re.findall(r'\b\w+\b', combined_text.lower())
        word_freq = Counter(words)
        
        # Morpheme and subword analysis
        morphemes = Counter()
        roots = Counter()
        suffixes = Counter()
        
        # Analyze each word
        vocabulary_candidates = {}
        
        for word, freq in tqdm(word_freq.items(), desc="Analyzing words"):
            if len(word) < 3 or freq < 3:  # Filter very rare or short words
                continue
            
            # Calculate Turkish specificity
            turkish_score = self.calculate_turkish_specificity(word)
            
            if turkish_score < 0.3:  # Skip non-Turkish words
                continue
            
            # Morphological analysis
            boundaries = self.detect_morphological_boundaries(word)
            
            # Add to candidates
            vocabulary_candidates[word] = {
                'frequency': freq,
                'turkish_score': turkish_score,
                'length': len(word),
                'morphemes': boundaries,
                'vowel_harmony': self.analyze_vowel_harmony(word),
                'priority_score': freq * turkish_score * (len(word) / 10)
            }
            
            # Collect morphemes
            for morpheme in boundaries:
                morphemes[morpheme] += freq
                if len(morpheme) >= 3:
                    if morpheme in self.common_suffixes:
                        suffixes[morpheme] += freq
                    else:
                        roots[morpheme] += freq
        
        # Current tokenizer efficiency analysis
        logger.info("Analyzing current tokenization efficiency...")
        tokenization_analysis = {}
        
        sample_texts = all_texts[:100]  # Sample for efficiency analysis
        for i, text in enumerate(sample_texts):
            analysis = self.analyze_tokenization_efficiency(text)
            tokenization_analysis[i] = analysis
        
        # Calculate average efficiency
        avg_efficiency = np.mean([a.get('efficiency_score', 0) for a in tokenization_analysis.values()])
        
        # Priority-based candidate selection
        logger.info("Selecting top candidates...")
        
        # Sort candidates by priority score
        sorted_candidates = sorted(
            vocabulary_candidates.items(),
            key=lambda x: x[1]['priority_score'],
            reverse=True
        )
        
        # Select top candidates
        selected_words = []
        selected_morphemes = []
        selected_roots = []
        selected_suffixes = []
        
        # 1. High-frequency Turkish words (40% of target)
        word_quota = int(target_vocab_size * 0.4)
        for word, info in sorted_candidates[:word_quota]:
            selected_words.append((word, info))
        
        # 2. Common Turkish suffixes (30% of target) 
        suffix_quota = int(target_vocab_size * 0.3)
        common_turkish_suffixes = [
            (suffix, freq) for suffix, freq in suffixes.most_common(suffix_quota)
            if suffix in self.common_suffixes
        ]
        selected_suffixes.extend(common_turkish_suffixes)
        
        # 3. High-frequency roots (20% of target)
        root_quota = int(target_vocab_size * 0.2)
        common_roots = [(root, freq) for root, freq in roots.most_common(root_quota)]
        selected_roots.extend(common_roots)
        
        # 4. Morphologically complex words (10% of target)
        morpheme_quota = target_vocab_size - len(selected_words) - len(selected_suffixes) - len(selected_roots)
        
        complex_words = [
            (word, info) for word, info in sorted_candidates
            if len(info['morphemes']) > 1 and len(word) > 6
        ][:morpheme_quota]
        selected_morphemes.extend(complex_words)
        
        # Compile final vocabulary
        final_vocabulary = {
            'words': selected_words,
            'suffixes': selected_suffixes, 
            'roots': selected_roots,
            'complex_morphemes': selected_morphemes,
            'total_new_tokens': len(selected_words) + len(selected_suffixes) + len(selected_roots) + len(selected_morphemes)
        }
        
        # Update statistics
        self.vocabulary_stats.update({
            'total_words': len(words),
            'unique_words': len(word_freq),
            'turkish_words': len(vocabulary_candidates),
            'morphologically_complex': len(complex_words),
            'new_tokens_identified': final_vocabulary['total_new_tokens'],
            'current_efficiency': avg_efficiency
        })
        
        return {
            'vocabulary': final_vocabulary,
            'statistics': self.vocabulary_stats,
            'tokenization_analysis': tokenization_analysis,
            'recommendations': self.generate_recommendations(final_vocabulary, avg_efficiency)
        }
    
    def generate_recommendations(self, vocabulary: Dict, current_efficiency: float) -> Dict:
        """Generate recommendations for tokenizer extension"""
        
        total_tokens = vocabulary['total_new_tokens']
        expected_efficiency_gain = min(0.6, total_tokens / 50000)  # Diminishing returns
        
        return {
            'recommended_vocab_size': total_tokens,
            'expected_efficiency_gain': f"{expected_efficiency_gain:.1%}",
            'expected_token_reduction': f"{30 + expected_efficiency_gain * 50:.0f}%",
            'training_recommendations': {
                'learning_rate': '2e-4 (aggressive for Turkish-only)',
                'epochs': '8-12 epochs for vocabulary adaptation',
                'lora_rank': '256 (DoRA configuration)',
                'batch_size': '128 with gradient accumulation'
            },
            'quality_thresholds': {
                'min_frequency': 3,
                'min_turkish_score': 0.3,
                'prioritize_morphemes': True,
                'vowel_harmony_bonus': True
            }
        }
    
    def save_vocabulary_analysis(self, analysis_results: Dict, output_dir: str = "vocab_analysis"):
        """Save vocabulary analysis results"""
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Save complete analysis
        with open(output_path / 'turkish_vocabulary_analysis.json', 'w', encoding='utf-8') as f:
            json.dump(analysis_results, f, ensure_ascii=False, indent=2, default=str)
        
        # Save vocabulary for tokenizer extension
        vocabulary = analysis_results['vocabulary']
        
        # Prepare vocabulary file for Qwen extension
        extension_vocab = {}
        
        # Add all selected tokens with unique IDs
        token_id = self.qwen_vocab_size  # Start after Qwen vocab
        
        for word, info in vocabulary['words']:
            extension_vocab[word] = token_id
            token_id += 1
        
        for suffix, freq in vocabulary['suffixes']:
            if suffix not in extension_vocab:  # Avoid duplicates
                extension_vocab[suffix] = token_id
                token_id += 1
        
        for root, freq in vocabulary['roots']:
            if root not in extension_vocab:
                extension_vocab[root] = token_id
                token_id += 1
        
        for morpheme, info in vocabulary['complex_morphemes']:
            if morpheme not in extension_vocab:
                extension_vocab[morpheme] = token_id
                token_id += 1
        
        # Save extension vocabulary
        with open(output_path / 'qwen_turkish_extension_vocab.json', 'w', encoding='utf-8') as f:
            json.dump(extension_vocab, f, ensure_ascii=False, indent=2)
        
        # Save training vocabulary (word -> id mapping)
        with open(output_path / 'turkish_tokens_for_training.txt', 'w', encoding='utf-8') as f:
            for token in extension_vocab.keys():
                f.write(f"{token}\n")
        
        logger.info(f"Vocabulary analysis saved to {output_path}")
        logger.info(f"Extension vocabulary size: {len(extension_vocab)} tokens")
        logger.info(f"New total vocabulary size: {self.qwen_vocab_size + len(extension_vocab)}")
        
        return extension_vocab


def analyze_turkish_vocabulary(corpus_file: str = "analysis_results/high_quality_turkish_data.jsonl",
                              target_vocab_size: int = 40000):
    """Main function to analyze Turkish vocabulary"""
    
    analyzer = TurkishVocabularyAnalyzer()
    
    # Load Qwen tokenizer
    if not analyzer.load_qwen_tokenizer():
        logger.error("Cannot proceed without Qwen tokenizer")
        return None
    
    # Load corpus data
    corpus_data = []
    corpus_path = Path(corpus_file)
    
    if corpus_path.exists():
        logger.info(f"Loading corpus from {corpus_path}")
        with open(corpus_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    item = json.loads(line.strip())
                    corpus_data.append(item)
                except json.JSONDecodeError:
                    continue
    else:
        logger.error(f"Corpus file not found: {corpus_path}")
        return None
    
    if not corpus_data:
        logger.error("No corpus data loaded")
        return None
    
    logger.info(f"Loaded {len(corpus_data)} samples from corpus")
    
    # Perform vocabulary analysis
    logger.info("Starting vocabulary analysis...")
    analysis_results = analyzer.extract_vocabulary_candidates(corpus_data, target_vocab_size)
    
    # Save results
    extension_vocab = analyzer.save_vocabulary_analysis(analysis_results)
    
    # Print summary
    print("\n" + "="*60)
    print("TURKISH VOCABULARY ANALYSIS COMPLETED")
    print("="*60)
    print(f"Total words analyzed: {analysis_results['statistics']['total_words']:,}")
    print(f"Unique Turkish words: {analysis_results['statistics']['turkish_words']:,}")
    print(f"Selected for extension: {analysis_results['statistics']['new_tokens_identified']:,}")
    print(f"Current tokenization efficiency: {analysis_results['statistics']['current_efficiency']:.3f}")
    print(f"Expected improvement: {analysis_results['recommendations']['expected_token_reduction']}")
    print(f"New total vocab size: {analyzer.qwen_vocab_size + len(extension_vocab):,}")
    
    print("\nVocabulary breakdown:")
    vocab = analysis_results['vocabulary']
    print(f"- High-frequency words: {len(vocab['words'])}")
    print(f"- Turkish suffixes: {len(vocab['suffixes'])}")
    print(f"- Word roots: {len(vocab['roots'])}")
    print(f"- Complex morphemes: {len(vocab['complex_morphemes'])}")
    
    return analysis_results


if __name__ == "__main__":
    results = analyze_turkish_vocabulary(target_vocab_size=40000)
    if results:
        print("\nNext steps:")
        print("1. Review vocab_analysis/turkish_vocabulary_analysis.json")
        print("2. Use qwen_turkish_extension_vocab.json for tokenizer extension")
        print("3. Proceed with DoRA training configuration")