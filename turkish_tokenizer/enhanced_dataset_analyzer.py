"""
Enhanced Turkish Dataset Analyzer
Improved quality metrics and advanced filtering

Key Improvements:
- Advanced Turkish language detection
- Better quality scoring algorithm  
- Morphological complexity analysis
- Enhanced deduplication
- Memory-efficient processing
- Robust error handling
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from collections import Counter, defaultdict
import re
from typing import Dict, List, Tuple, Any
import logging
from datasets import load_dataset
from tqdm import tqdm
import hashlib
import gzip

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnhancedTurkishAnalyzer:
    """Enhanced analyzer with improved metrics"""
    
    def __init__(self, output_dir: str = "enhanced_analysis"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Enhanced Turkish patterns
        self.turkish_chars = set("çğıöşüâîûÇĞİÖŞÜ")
        self.vowels = set("aeiıoöuüAEIİOÖUÜ")
        self.consonants = set("bcçdfgğhjklmnprsştuvyzBCÇDFGĞHJKLMNPRSŞTUVYZ")
        
        # Morphological patterns
        self.suffixes = {
            'plural': ['ler', 'lar'],
            'possessive': ['im', 'ım', 'üm', 'um', 'in', 'ın', 'ün', 'un'],
            'case': ['de', 'da', 'den', 'dan', 'ye', 'ya', 'nin', 'nın'],
            'verb': ['mek', 'mak', 'yor', 'dı', 'di', 'mış', 'miş']
        }
        
        # Dataset configuration
        self.dataset_sources = {
            'local': {
                'turkish_quiz': '../data/raw/turkish_quiz_instruct.csv',
                'competition': '../data/processed/competition_dataset.json',
                'tr_mega': '../data/TR_MEGA_1_2000_Combined.jsonl',
                'synthetic_mega': '../data/synthetic/TR_MEGA_1_2000_Combined.jsonl',
                'llm_10k_v1': '../data/synthetic/turkish_llm_10k_dataset.jsonl.gz',
                'llm_10k_v3': '../data/synthetic/turkish_llm_10k_dataset_v3.jsonl.gz'
            },
            'huggingface': {
                'merve/turkish_instructions': {'column': 'instruction', 'limit': 5000},
                'TFLai/Turkish-Alpaca': {'column': 'instruction', 'limit': 5000},
                'malhajar/OpenOrca-tr': {'column': 'instruction', 'limit': 5000},
                'umarigan/turkish_corpus': {'column': 'knowledge', 'limit': 10000},
                'Huseyin/muspdf': {'column': 'text', 'limit': 15000},
                'tubitak/tuba-corpus': {'column': 'text', 'limit': 20000},
                'boun-pars/boun-corpus': {'column': 'text', 'limit': 10000},
                'selimfirat/bilkent-turkish-writings-dataset': {'column': 'text', 'limit': 12000}
            }
        }
    
    def enhanced_turkish_score(self, text: str) -> Dict[str, float]:
        """Enhanced Turkish language scoring"""
        if not text or len(text) < 10:
            return {'overall': 0.0, 'chars': 0.0, 'morphology': 0.0, 'syntax': 0.0}
        
        text_lower = text.lower()
        words = re.findall(r'\b\w+\b', text_lower)
        
        # Character-level analysis
        char_score = sum(1 for c in text_lower if c in self.turkish_chars) / len(text)
        
        # Morphological analysis
        morph_matches = 0
        total_words = len(words)
        
        for word in words:
            for category, suffixes in self.suffixes.items():
                if any(word.endswith(suffix) for suffix in suffixes):
                    morph_matches += 1
                    break
        
        morph_score = morph_matches / max(total_words, 1)
        
        # Syntactic patterns (Turkish word order SOV tendency)
        syntax_score = self._analyze_syntax(words)
        
        # Vowel harmony check
        harmony_score = self._check_vowel_harmony(words)
        
        # Overall score
        overall = (char_score * 0.3 + morph_score * 0.3 + 
                  syntax_score * 0.2 + harmony_score * 0.2)
        
        return {
            'overall': overall,
            'chars': char_score,
            'morphology': morph_score,
            'syntax': syntax_score,
            'harmony': harmony_score
        }
    
    def _analyze_syntax(self, words: List[str]) -> float:
        """Analyze Turkish syntactic patterns"""
        if len(words) < 3:
            return 0.5
        
        # Simple heuristics for Turkish syntax
        score = 0.0
        checks = 0
        
        # Check for typical Turkish patterns
        for i in range(len(words) - 1):
            current = words[i]
            next_word = words[i + 1]
            
            # Verb typically at the end
            if self._is_likely_verb(next_word) and i > len(words) * 0.6:
                score += 1.0
                checks += 1
            
            # Object before verb pattern
            if self._is_likely_object(current) and self._is_likely_verb(next_word):
                score += 1.0
                checks += 1
            
            checks += 1
        
        return score / max(checks, 1)
    
    def _is_likely_verb(self, word: str) -> bool:
        """Check if word is likely a Turkish verb"""
        verb_endings = ['yor', 'dı', 'di', 'du', 'dü', 'mış', 'miş', 'acak', 'ecek']
        return any(word.endswith(ending) for ending in verb_endings)
    
    def _is_likely_object(self, word: str) -> bool:
        """Check if word is likely an object (accusative case)"""
        object_endings = ['i', 'ı', 'ü', 'u', 'yi', 'yı', 'yü', 'yu']
        return any(word.endswith(ending) for ending in object_endings)
    
    def _check_vowel_harmony(self, words: List[str]) -> float:
        """Check vowel harmony compliance"""
        front_vowels = set('eiöü')
        back_vowels = set('aıou')
        
        harmony_violations = 0
        total_checks = 0
        
        for word in words:
            if len(word) < 4:  # Skip short words
                continue
            
            vowels = [c for c in word.lower() if c in self.vowels]
            if len(vowels) > 1:
                total_checks += 1
                
                # Check harmony within word
                front_count = sum(1 for v in vowels if v in front_vowels)
                back_count = sum(1 for v in vowels if v in back_vowels)
                
                # Strong violation if both front and back vowels
                if front_count > 0 and back_count > 0:
                    harmony_violations += 1
        
        return 1.0 - (harmony_violations / max(total_checks, 1))
    
    def advanced_quality_score(self, text: str) -> Dict[str, float]:
        """Advanced quality scoring with multiple dimensions"""
        
        if not text or len(text) < 20:
            return {'overall': 0.0, 'length': 0.0, 'diversity': 0.0, 'coherence': 0.0}
        
        # Length score (optimal range)
        length_score = min(1.0, len(text) / 100) if len(text) < 100 else min(1.0, 500 / len(text))
        
        # Lexical diversity
        words = text.split()
        unique_words = len(set(words))
        diversity_score = unique_words / max(len(words), 1)
        
        # Character diversity
        unique_chars = len(set(text.lower()))
        char_diversity = unique_chars / 35  # Turkish alphabet + punctuation
        
        # Coherence (sentence structure)
        sentences = re.split(r'[.!?]+', text)
        valid_sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        sentence_score = len(valid_sentences) / max(len(sentences), 1)
        
        # Readability (average word/sentence length)
        avg_word_len = np.mean([len(w) for w in words]) if words else 0
        readability = min(1.0, avg_word_len / 8)  # Optimal around 6-8 chars
        
        # Overall quality
        overall = (length_score * 0.2 + diversity_score * 0.3 + 
                  char_diversity * 0.2 + sentence_score * 0.2 + readability * 0.1)
        
        return {
            'overall': overall,
            'length': length_score,
            'diversity': diversity_score,
            'coherence': sentence_score,
            'readability': readability
        }
    
    def load_datasets_efficiently(self) -> List[Dict]:
        """Memory-efficient dataset loading"""
        all_data = []
        
        # Load local datasets
        for name, path in self.dataset_sources['local'].items():
            full_path = Path(__file__).parent / path
            
            if not full_path.exists():
                logger.warning(f"Dataset not found: {full_path}")
                continue
            
            logger.info(f"Loading {name}...")
            
            try:
                data_batch = []
                
                if path.endswith('.gz'):
                    with gzip.open(full_path, 'rt', encoding='utf-8') as f:
                        for line_num, line in enumerate(f):
                            if line_num >= 20000:  # Memory limit
                                break
                            try:
                                item = json.loads(line.strip())
                                text = self._extract_text(item)
                                if text and len(text) > 30:
                                    data_batch.append({
                                        'text': text,
                                        'source': name,
                                        'type': self._classify_type(item, name)
                                    })
                            except json.JSONDecodeError:
                                continue
                
                elif path.endswith('.jsonl'):
                    with open(full_path, 'r', encoding='utf-8') as f:
                        for line_num, line in enumerate(f):
                            if line_num >= 20000:
                                break
                            try:
                                item = json.loads(line.strip())
                                text = self._extract_text(item)
                                if text and len(text) > 30:
                                    data_batch.append({
                                        'text': text,
                                        'source': name,
                                        'type': self._classify_type(item, name)
                                    })
                            except json.JSONDecodeError:
                                continue
                
                elif path.endswith('.json'):
                    with open(full_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        if isinstance(data, list):
                            for item in data[:15000]:  # Limit per file
                                text = self._extract_text(item)
                                if text and len(text) > 30:
                                    data_batch.append({
                                        'text': text,
                                        'source': name,
                                        'type': self._classify_type(item, name)
                                    })
                
                elif path.endswith('.csv'):
                    df = pd.read_csv(full_path)
                    text_cols = [col for col in df.columns 
                               if any(keyword in col.lower() 
                                     for keyword in ['text', 'content', 'question', 'instruction'])]
                    
                    if text_cols and len(df) > 0:
                        for _, row in df.head(15000).iterrows():
                            for col in text_cols:
                                text = str(row[col])
                                if len(text) > 30:
                                    data_batch.append({
                                        'text': text,
                                        'source': name,
                                        'type': 'instruction' if 'instruct' in col.lower() else 'knowledge'
                                    })
                                    break
                
                all_data.extend(data_batch)
                logger.info(f"Loaded {len(data_batch)} samples from {name}")
                
            except Exception as e:
                logger.error(f"Error loading {name}: {e}")
        
        # Load HuggingFace datasets (with limits)
        for dataset_name, config in self.dataset_sources['huggingface'].items():
            try:
                logger.info(f"Loading HuggingFace: {dataset_name}")
                dataset = load_dataset(dataset_name, split='train', streaming=True)
                
                count = 0
                for item in dataset:
                    if count >= config['limit']:
                        break
                    
                    text = self._extract_text(item)
                    if text and len(text) > 30:
                        all_data.append({
                            'text': text,
                            'source': dataset_name,
                            'type': 'instruction' if 'instruction' in config['column'] else 'knowledge'
                        })
                        count += 1
                
                logger.info(f"Loaded {count} samples from {dataset_name}")
                
            except Exception as e:
                logger.warning(f"Could not load {dataset_name}: {e}")
        
        return all_data
    
    def _extract_text(self, item: Dict) -> str:
        """Extract text from various data formats"""
        if isinstance(item, str):
            return item
        
        # Try different field names
        for field in ['text', 'content', 'instruction', 'output', 'response', 'question', 'answer']:
            if field in item and item[field]:
                return str(item[field])
        
        # Try combination fields
        if 'instruction' in item and 'output' in item:
            instruction = str(item.get('instruction', ''))
            output = str(item.get('output', ''))
            if instruction and output:
                return f"{instruction} {output}"
        
        return ""
    
    def _classify_type(self, item: Dict, source: str) -> str:
        """Classify data type"""
        if isinstance(item, dict):
            if any(key in item for key in ['instruction', 'input', 'question']):
                return 'instruction'
        
        if any(keyword in source.lower() for keyword in ['corpus', 'knowledge', 'writing']):
            return 'knowledge'
        
        return 'instruction'
    
    def analyze_with_improvements(self) -> Dict:
        """Main analysis with all improvements"""
        logger.info("Starting enhanced Turkish dataset analysis...")
        
        # Load datasets efficiently
        all_data = self.load_datasets_efficiently()
        logger.info(f"Total samples loaded: {len(all_data)}")
        
        if not all_data:
            return {}
        
        # Enhanced scoring
        scored_data = []
        for item in tqdm(all_data, desc="Enhanced scoring"):
            text = item['text']
            
            turkish_scores = self.enhanced_turkish_score(text)
            quality_scores = self.advanced_quality_score(text)
            
            # Combined score
            overall_score = (turkish_scores['overall'] * 0.6 + quality_scores['overall'] * 0.4)
            
            item.update({
                'turkish_score': turkish_scores['overall'],
                'quality_score': quality_scores['overall'],
                'combined_score': overall_score,
                'detailed_scores': {**turkish_scores, **quality_scores}
            })
            
            scored_data.append(item)
        
        # Filter top quality (top 15% instead of 10% for more data)
        scored_data.sort(key=lambda x: x['combined_score'], reverse=True)
        top_samples = int(len(scored_data) * 0.15)
        high_quality = scored_data[:top_samples]
        
        # Enhanced deduplication
        deduplicated = self.enhanced_deduplication(high_quality)
        
        results = {
            'total_samples': len(all_data),
            'high_quality_samples': len(high_quality),
            'final_samples': len(deduplicated),
            'average_turkish_score': np.mean([d['turkish_score'] for d in deduplicated]),
            'average_quality_score': np.mean([d['quality_score'] for d in deduplicated]),
            'data': deduplicated[:50000]  # Limit output
        }
        
        # Save results
        self.save_enhanced_results(results, deduplicated)
        
        return results
    
    def enhanced_deduplication(self, data: List[Dict], threshold: float = 0.8) -> List[Dict]:
        """Enhanced deduplication with better similarity detection"""
        logger.info("Performing enhanced deduplication...")
        
        if not data:
            return []
        
        # Use text hashing for initial filtering
        seen_hashes = set()
        hash_filtered = []
        
        for item in data:
            text_hash = hashlib.md5(item['text'].encode()).hexdigest()
            if text_hash not in seen_hashes:
                seen_hashes.add(text_hash)
                hash_filtered.append(item)
        
        logger.info(f"Hash deduplication: {len(data)} -> {len(hash_filtered)}")
        
        # Semantic similarity for remaining items
        final_data = []
        
        for i, item in enumerate(tqdm(hash_filtered, desc="Similarity check")):
            is_duplicate = False
            
            # Check against already selected items
            for existing in final_data[-100:]:  # Check last 100 for efficiency
                similarity = self._calculate_text_similarity(item['text'], existing['text'])
                if similarity > threshold:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                final_data.append(item)
        
        logger.info(f"Final deduplication: {len(hash_filtered)} -> {len(final_data)}")
        return final_data
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate text similarity using word overlap"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1 & words2
        union = words1 | words2
        
        return len(intersection) / len(union)
    
    def save_enhanced_results(self, results: Dict, processed_data: List[Dict]):
        """Save enhanced analysis results"""
        
        # Save main results
        with open(self.output_dir / 'enhanced_analysis_report.json', 'w', encoding='utf-8') as f:
            # Remove data field for report
            report = {k: v for k, v in results.items() if k != 'data'}
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        # Save processed data
        with open(self.output_dir / 'high_quality_turkish_enhanced.jsonl', 'w', encoding='utf-8') as f:
            for item in processed_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        logger.info(f"Enhanced results saved to {self.output_dir}")


def main():
    """Run enhanced analysis"""
    analyzer = EnhancedTurkishAnalyzer()
    results = analyzer.analyze_with_improvements()
    
    if results:
        print("\n🎉 ENHANCED ANALYSIS COMPLETED")
        print(f"📊 Total samples: {results['total_samples']:,}")
        print(f"✅ High quality: {results['high_quality_samples']:,}")
        print(f"🎯 Final dataset: {results['final_samples']:,}")
        print(f"📈 Avg Turkish score: {results['average_turkish_score']:.3f}")
        print(f"⭐ Avg quality score: {results['average_quality_score']:.3f}")


if __name__ == "__main__":
    main()