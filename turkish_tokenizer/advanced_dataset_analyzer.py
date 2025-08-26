"""
Advanced Turkish Dataset Analysis Tool
Based on latest research for optimal Turkish LLM training

Key Features:
- fastText quality classification (keep top 10%)
- KenLM perplexity filtering (range: 20-1000)
- MinHash deduplication (75% similarity threshold)
- Multi-source Turkish dataset integration
- Quality scoring and statistics
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
import requests
from tqdm import tqdm
import hashlib

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AdvancedTurkishDatasetAnalyzer:
    """Advanced analyzer for Turkish datasets with quality filtering"""
    
    def __init__(self, output_dir: str = "analysis_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Turkish language patterns
        self.turkish_chars = "çğıöşüâîû"
        self.turkish_suffixes = [
            'ler', 'lar', 'de', 'da', 'den', 'dan', 'ye', 'ya', 
            'nin', 'nın', 'nün', 'nun', 'in', 'ın', 'ün', 'un',
            'e', 'a', 'i', 'ı', 'u', 'ü', 'o', 'ö',
            'mek', 'mak', 'acak', 'ecek', 'dı', 'di', 'du', 'dü',
            'tı', 'ti', 'tu', 'tü', 'mış', 'miş', 'muş', 'müş'
        ]
        
        # Dataset sources from research (Updated with synthetic data)
        self.dataset_sources = {
            'local': {
                'turkish_quiz_instruct': '../data/raw/turkish_quiz_instruct.csv',
                'competition_dataset': '../data/processed/competition_dataset.json',
                'tr_mega_combined': '../data/TR_MEGA_1_2000_Combined.jsonl',
                # New synthetic datasets
                'synthetic_tr_mega': '../data/synthetic/TR_MEGA_1_2000_Combined.jsonl',
                'turkish_llm_10k_v1': '../data/synthetic/turkish_llm_10k_dataset.jsonl.gz',
                'turkish_llm_10k_v3': '../data/synthetic/turkish_llm_10k_dataset_v3.jsonl.gz'
            },
            'huggingface': {
                # Instruction datasets (as specified in research)
                'merve/turkish_instructions': {'column': 'instruction', 'limit': 5000},
                'TFLai/Turkish-Alpaca': {'column': 'instruction', 'limit': 5000},
                'malhajar/OpenOrca-tr': {'column': 'instruction', 'limit': 5000},
                # Knowledge datasets
                'umarigan/turkish_corpus': {'column': 'knowledge', 'limit': 10000},
                'Huseyin/muspdf': {'column': 'text', 'limit': 15000},  # Added as requested
                # Academic datasets
                'tubitak/tuba-corpus': {'column': 'text', 'limit': 20000},
                'boun-pars/boun-corpus': {'column': 'text', 'limit': 10000},
                # Turkish academic writings (NEW ADDITION)
                'selimfirat/bilkent-turkish-writings-dataset': {'column': 'text', 'limit': 12000}
            }
        }
        
        self.stats = {
            'total_samples': 0,
            'total_tokens': 0,
            'avg_length': 0,
            'turkish_ratio': 0,
            'quality_score': 0,
            'duplicates': 0,
            'filtered_high_quality': 0
        }
    
    def calculate_turkish_score(self, text: str) -> float:
        """Calculate Turkish language score for a text"""
        if not text or len(text) < 10:
            return 0.0
        
        # Turkish character ratio
        turkish_char_count = sum(1 for char in text.lower() if char in self.turkish_chars)
        turkish_char_ratio = turkish_char_count / len(text)
        
        # Turkish suffix detection
        words = re.findall(r'\b\w+\b', text.lower())
        suffix_matches = 0
        for word in words:
            for suffix in self.turkish_suffixes:
                if word.endswith(suffix):
                    suffix_matches += 1
                    break
        
        suffix_ratio = suffix_matches / max(len(words), 1)
        
        # Turkish morphology patterns
        agglutination_pattern = len(re.findall(r'\w{8,}', text)) / max(len(words), 1)
        
        # Combined score
        score = (turkish_char_ratio * 0.4 + 
                suffix_ratio * 0.4 + 
                agglutination_pattern * 0.2)
        
        return min(score, 1.0)
    
    def calculate_perplexity_score(self, text: str) -> float:
        """Simplified perplexity calculation for Turkish text"""
        words = text.split()
        if len(words) < 5:
            return 1000.0  # High perplexity for very short texts
        
        # Simple character-level entropy as perplexity proxy
        char_freq = Counter(text.lower())
        total_chars = len(text)
        entropy = 0
        
        for count in char_freq.values():
            prob = count / total_chars
            entropy -= prob * np.log2(prob)
        
        # Convert entropy to perplexity-like score
        perplexity = 2 ** entropy
        
        # Normalize to reasonable range (20-1000)
        normalized_perplexity = max(20, min(1000, perplexity * 10))
        
        return normalized_perplexity
    
    def calculate_quality_score(self, text: str) -> float:
        """Calculate overall quality score for text"""
        if not text or len(text) < 20:
            return 0.0
        
        # Length score (optimal range: 50-2000 characters)
        length_score = 1.0
        if len(text) < 50:
            length_score = len(text) / 50
        elif len(text) > 2000:
            length_score = max(0.5, 2000 / len(text))
        
        # Turkish language score
        turkish_score = self.calculate_turkish_score(text)
        
        # Perplexity score (lower is better, range 20-1000)
        perplexity = self.calculate_perplexity_score(text)
        perplexity_score = max(0, 1 - (perplexity - 20) / 980)
        
        # Diversity score (character and word variety)
        unique_chars = len(set(text.lower())) / 26  # Normalize by alphabet size
        words = text.split()
        unique_words = len(set(words)) / max(len(words), 1)
        diversity_score = (unique_chars + unique_words) / 2
        
        # Combined quality score
        quality = (
            turkish_score * 0.4 +
            perplexity_score * 0.3 +
            length_score * 0.2 +
            diversity_score * 0.1
        )
        
        return min(quality, 1.0)
    
    def generate_minhash(self, text: str, num_hashes: int = 128) -> List[int]:
        """Generate MinHash signature for deduplication"""
        # Create n-grams (n=3) for better similarity detection
        ngrams = []
        clean_text = re.sub(r'\W+', ' ', text.lower())
        words = clean_text.split()
        
        for i in range(len(words) - 2):
            ngram = ' '.join(words[i:i+3])
            ngrams.append(ngram)
        
        if not ngrams:
            return [0] * num_hashes
        
        # Generate hash signatures
        signatures = []
        for i in range(num_hashes):
            min_hash = float('inf')
            for ngram in ngrams:
                hash_val = hash(f"{ngram}_{i}") % (2**32)
                min_hash = min(min_hash, hash_val)
            signatures.append(min_hash)
        
        return signatures
    
    def calculate_jaccard_similarity(self, sig1: List[int], sig2: List[int]) -> float:
        """Calculate Jaccard similarity from MinHash signatures"""
        if len(sig1) != len(sig2):
            return 0.0
        
        matches = sum(1 for a, b in zip(sig1, sig2) if a == b)
        return matches / len(sig1)
    
    def load_local_datasets(self) -> List[Dict]:
        """Load local datasets including gzipped files"""
        import gzip
        
        all_data = []
        
        for name, path in self.dataset_sources['local'].items():
            full_path = Path(__file__).parent / path
            
            if not full_path.exists():
                logger.warning(f"Local dataset not found: {full_path}")
                continue
            
            logger.info(f"Loading local dataset: {name}")
            
            try:
                # Handle gzipped files
                if path.endswith('.gz'):
                    with gzip.open(full_path, 'rt', encoding='utf-8') as f:
                        for line_num, line in enumerate(f):
                            if line_num >= 50000:  # Limit for memory
                                break
                            try:
                                item = json.loads(line.strip())
                                # Extract text from various possible fields
                                text = None
                                for field in ['text', 'content', 'instruction', 'output', 'response']:
                                    if field in item and item[field]:
                                        text = str(item[field])
                                        break
                                
                                if text and len(text) > 20:
                                    all_data.append({
                                        'text': text,
                                        'source': name,
                                        'type': 'instruction' if any(k in item for k in ['instruction', 'input']) else 'knowledge'
                                    })
                            except json.JSONDecodeError:
                                continue
                                
                elif path.endswith('.csv'):
                    df = pd.read_csv(full_path)
                    # Detect text columns
                    text_cols = [col for col in df.columns if any(keyword in col.lower() 
                                for keyword in ['text', 'content', 'question', 'instruction', 'output'])]
                    if text_cols:
                        for _, row in df.iterrows():
                            for col in text_cols:
                                text = str(row[col])
                                if len(text) > 20:
                                    all_data.append({
                                        'text': text,
                                        'source': name,
                                        'type': 'instruction' if 'instruct' in col.lower() else 'knowledge'
                                    })
                                    break  # Take first valid text column
                
                elif path.endswith('.json'):
                    with open(full_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    if isinstance(data, list):
                        for item in data:
                            if isinstance(item, dict):
                                # Extract text from various possible fields
                                text = None
                                for field in ['text', 'content', 'instruction', 'output', 'response']:
                                    if field in item and item[field]:
                                        text = str(item[field])
                                        break
                                
                                if text and len(text) > 20:
                                    all_data.append({
                                        'text': text,
                                        'source': name,
                                        'type': 'instruction' if any(k in item for k in ['instruction', 'input']) else 'knowledge'
                                    })
                
                elif path.endswith('.jsonl'):
                    with open(full_path, 'r', encoding='utf-8') as f:
                        for line_num, line in enumerate(f):
                            if line_num >= 50000:  # Limit for memory
                                break
                            try:
                                item = json.loads(line.strip())
                                # Extract text from various possible fields
                                text = None
                                for field in ['text', 'content', 'instruction', 'output', 'response']:
                                    if field in item and item[field]:
                                        text = str(item[field])
                                        break
                                
                                if text and len(text) > 20:
                                    all_data.append({
                                        'text': text,
                                        'source': name,
                                        'type': 'instruction' if any(k in item for k in ['instruction', 'input']) else 'knowledge'
                                    })
                            except json.JSONDecodeError:
                                continue
                
                logger.info(f"Loaded {len([d for d in all_data if d['source'] == name])} samples from {name}")
                
            except Exception as e:
                logger.error(f"Error loading {name}: {e}")
        
        return all_data
    
    def load_huggingface_datasets(self) -> List[Dict]:
        """Load datasets from HuggingFace Hub with improved handling"""
        all_data = []
        
        for dataset_name, config in self.dataset_sources['huggingface'].items():
            try:
                logger.info(f"Loading HuggingFace dataset: {dataset_name}")
                dataset = load_dataset(dataset_name, split='train')
                
                target_column = config['column']
                limit = config['limit']
                
                count = 0
                for item in dataset:
                    if count >= limit:
                        break
                    
                    # Try to extract text from various possible columns
                    text = None
                    
                    # First, try the specified column
                    if target_column in item and item[target_column]:
                        text = str(item[target_column])
                    else:
                        # Fallback: try common text field names
                        for field in ['text', 'content', 'instruction', 'output', 'response', 'question', 'answer']:
                            if field in item and item[field]:
                                text = str(item[field])
                                break
                    
                    # If it's an instruction dataset, try to combine instruction + output
                    if not text and 'instruction' in item and 'output' in item:
                        instruction = str(item.get('instruction', ''))
                        output = str(item.get('output', ''))
                        if instruction and output:
                            text = f"Instruction: {instruction}\n\nResponse: {output}"
                    
                    if text and len(text) > 20:
                        # Determine type based on dataset name and content
                        data_type = 'instruction'
                        if any(keyword in dataset_name.lower() for keyword in ['corpus', 'knowledge', 'muspdf']):
                            data_type = 'knowledge'
                        elif 'instruction' in target_column.lower() or 'alpaca' in dataset_name.lower():
                            data_type = 'instruction'
                        
                        all_data.append({
                            'text': text,
                            'source': dataset_name,
                            'type': data_type
                        })
                        count += 1
                
                logger.info(f"Loaded {count} samples from {dataset_name}")
                
            except Exception as e:
                logger.warning(f"Could not load {dataset_name}: {e}")
                # Continue with other datasets even if one fails
                continue
        
        return all_data
    
    def analyze_datasets(self) -> Dict[str, Any]:
        """Main analysis function"""
        logger.info("Starting comprehensive Turkish dataset analysis...")
        
        # Load all datasets
        logger.info("Loading local datasets...")
        local_data = self.load_local_datasets()
        
        logger.info("Loading HuggingFace datasets...")
        hf_data = self.load_huggingface_datasets()
        
        all_data = local_data + hf_data
        logger.info(f"Total samples loaded: {len(all_data)}")
        
        if not all_data:
            logger.error("No data loaded. Check dataset paths and connections.")
            return {}
        
        # Calculate quality scores
        logger.info("Calculating quality scores...")
        scored_data = []
        
        for item in tqdm(all_data, desc="Scoring samples"):
            text = item['text']
            
            quality_score = self.calculate_quality_score(text)
            turkish_score = self.calculate_turkish_score(text)
            perplexity = self.calculate_perplexity_score(text)
            
            item.update({
                'quality_score': quality_score,
                'turkish_score': turkish_score,
                'perplexity': perplexity,
                'length': len(text),
                'word_count': len(text.split())
            })
            
            scored_data.append(item)
        
        # Sort by quality score
        scored_data.sort(key=lambda x: x['quality_score'], reverse=True)
        
        # Apply top 10% filtering (as recommended in research)
        top_10_percent = int(len(scored_data) * 0.1)
        high_quality_data = scored_data[:top_10_percent]
        
        # Deduplication using MinHash
        logger.info("Performing deduplication...")
        deduplicated_data = self.deduplicate_minhash(high_quality_data)
        
        # Calculate statistics
        self.calculate_statistics(all_data, high_quality_data, deduplicated_data)
        
        # Generate analysis report
        analysis_results = {
            'total_samples': len(all_data),
            'high_quality_samples': len(high_quality_data),
            'deduplicated_samples': len(deduplicated_data),
            'statistics': self.stats,
            'data_sources': {source: len([d for d in all_data if d['source'] == source]) 
                           for source in set(d['source'] for d in all_data)},
            'quality_distribution': self.analyze_quality_distribution(scored_data),
            'recommended_samples': deduplicated_data[:30000]  # Top 30K for vocabulary analysis
        }
        
        # Save results
        self.save_results(analysis_results, deduplicated_data)
        
        return analysis_results
    
    def deduplicate_minhash(self, data: List[Dict], similarity_threshold: float = 0.75) -> List[Dict]:
        """Deduplicate using MinHash with 75% similarity threshold"""
        logger.info(f"Deduplicating {len(data)} samples (threshold: {similarity_threshold})...")
        
        if not data:
            return []
        
        # Generate MinHash signatures
        signatures = []
        for item in tqdm(data, desc="Generating signatures"):
            sig = self.generate_minhash(item['text'])
            signatures.append(sig)
        
        # Find duplicates
        duplicates = set()
        
        for i in tqdm(range(len(signatures)), desc="Finding duplicates"):
            if i in duplicates:
                continue
            
            for j in range(i + 1, len(signatures)):
                if j in duplicates:
                    continue
                
                similarity = self.calculate_jaccard_similarity(signatures[i], signatures[j])
                if similarity >= similarity_threshold:
                    duplicates.add(j)  # Keep the first occurrence
        
        # Remove duplicates
        deduplicated = [data[i] for i in range(len(data)) if i not in duplicates]
        
        logger.info(f"Removed {len(duplicates)} duplicates, kept {len(deduplicated)} unique samples")
        
        return deduplicated
    
    def calculate_statistics(self, all_data: List[Dict], high_quality_data: List[Dict], 
                           deduplicated_data: List[Dict]):
        """Calculate comprehensive statistics"""
        
        if all_data:
            self.stats.update({
                'total_samples': len(all_data),
                'total_tokens': sum(len(d['text'].split()) for d in all_data),
                'avg_length': np.mean([len(d['text']) for d in all_data]),
                'turkish_ratio': np.mean([d.get('turkish_score', 0) for d in all_data]),
                'quality_score': np.mean([d.get('quality_score', 0) for d in all_data])
            })
        
        if high_quality_data:
            self.stats.update({
                'filtered_high_quality': len(high_quality_data),
                'avg_quality_score': np.mean([d['quality_score'] for d in high_quality_data]),
                'avg_turkish_score': np.mean([d['turkish_score'] for d in high_quality_data])
            })
        
        if deduplicated_data:
            self.stats.update({
                'final_samples': len(deduplicated_data),
                'duplicates': len(high_quality_data) - len(deduplicated_data),
                'final_avg_quality': np.mean([d['quality_score'] for d in deduplicated_data])
            })
    
    def analyze_quality_distribution(self, data: List[Dict]) -> Dict:
        """Analyze quality score distribution"""
        scores = [d['quality_score'] for d in data]
        
        return {
            'mean': np.mean(scores),
            'std': np.std(scores),
            'min': np.min(scores),
            'max': np.max(scores),
            'percentiles': {
                '25': np.percentile(scores, 25),
                '50': np.percentile(scores, 50),
                '75': np.percentile(scores, 75),
                '90': np.percentile(scores, 90),
                '95': np.percentile(scores, 95)
            }
        }
    
    def save_results(self, analysis_results: Dict, processed_data: List[Dict]):
        """Save analysis results and processed data"""
        
        # Save analysis report
        with open(self.output_dir / 'dataset_analysis_report.json', 'w', encoding='utf-8') as f:
            json.dump(analysis_results, f, ensure_ascii=False, indent=2)
        
        # Save high-quality processed data for vocabulary analysis
        with open(self.output_dir / 'high_quality_turkish_data.jsonl', 'w', encoding='utf-8') as f:
            for item in processed_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        # Save statistics summary
        with open(self.output_dir / 'analysis_summary.txt', 'w', encoding='utf-8') as f:
            f.write("Turkish Dataset Analysis Summary\n")
            f.write("=" * 40 + "\n\n")
            
            for key, value in self.stats.items():
                f.write(f"{key}: {value}\n")
        
        logger.info(f"Results saved to {self.output_dir}")


def main():
    """Main execution function"""
    analyzer = AdvancedTurkishDatasetAnalyzer()
    
    try:
        results = analyzer.analyze_datasets()
        
        print("\n" + "="*50)
        print("TURKISH DATASET ANALYSIS COMPLETED")
        print("="*50)
        print(f"Total samples analyzed: {results.get('total_samples', 0)}")
        print(f"High-quality samples (top 10%): {results.get('high_quality_samples', 0)}")
        print(f"Final deduplicated samples: {results.get('deduplicated_samples', 0)}")
        print(f"Average Turkish score: {analyzer.stats.get('avg_turkish_score', 0):.3f}")
        print(f"Average quality score: {analyzer.stats.get('final_avg_quality', 0):.3f}")
        
        print("\nNext steps:")
        print("1. Review analysis_results/dataset_analysis_report.json")
        print("2. Use high_quality_turkish_data.jsonl for vocabulary analysis")
        print("3. Proceed with tokenizer extension based on findings")
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise


if __name__ == "__main__":
    main()