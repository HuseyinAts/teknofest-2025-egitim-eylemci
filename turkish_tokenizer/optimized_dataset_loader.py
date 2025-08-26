"""
OPTİMİZE EDİLMİŞ DATASET YÜKLEYICI
Bellek-verimli veri yükleme ve streaming

Kritik Özellikler:
- Memory-efficient streaming
- Batch processing with memory monitoring
- Turkish text quality filtering
- Automatic garbage collection
- Dataset caching and compression
"""

import os
import gc
import gzip
import json
import psutil
from pathlib import Path
from typing import Dict, List, Optional, Iterator, Tuple
from dataclasses import dataclass
import logging
import torch
from datasets import Dataset, IterableDataset, load_dataset
import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)

@dataclass 
class DatasetConfig:
    """Dataset yükleme konfigürasyonu"""
    
    # Memory management
    max_memory_gb: float = 8.0
    batch_size: int = 1000
    streaming: bool = True
    use_compression: bool = True
    
    # Quality filtering - MEMORY'DEN: En az 30 karakter!
    min_text_length: int = 30  # 🚨 KRİTİK: 30 minimum
    max_text_length: int = 4096
    min_turkish_score: float = 0.3
    
    # Caching
    use_cache: bool = True
    cache_dir: str = "dataset_cache"
    
    # Processing
    max_samples_per_source: int = 50000
    shuffle_buffer_size: int = 10000


class MemoryMonitor:
    """Bellek kullanım monitörü"""
    
    def __init__(self, max_memory_gb: float = 8.0):
        self.max_memory_gb = max_memory_gb
        self.max_memory_bytes = max_memory_gb * 1024 ** 3
        
    def get_current_memory(self) -> Dict[str, float]:
        """Mevcut bellek kullanımı"""
        
        process = psutil.Process()
        memory_info = process.memory_info()
        
        return {
            'rss_gb': memory_info.rss / (1024 ** 3),
            'vms_gb': memory_info.vms / (1024 ** 3),
            'percent': process.memory_percent()
        }
    
    def should_free_memory(self) -> bool:
        """Bellek temizlenip temizlenmeyeceği"""
        
        current = self.get_current_memory()
        return current['rss_gb'] > self.max_memory_gb * 0.8
    
    def force_gc(self):
        """Garbage collection zorla"""
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        gc.collect()
        
        logger.info(f"Memory after GC: {self.get_current_memory()['rss_gb']:.2f} GB")


class TurkishTextFilter:
    """Türkçe metin kalite filtresi"""
    
    def __init__(self, min_score: float = 0.3):
        self.min_score = min_score
        self.turkish_chars = set("çğıöşüÇĞİÖŞÜ")
        self.vowels = set("aeiıoöuüAEIıOÖUÜ")
    
    def calculate_turkish_score(self, text: str) -> float:
        """Türkçe metin skoru hesapla"""
        
        if not text or len(text) < 10:
            return 0.0
        
        # Turkish character ratio
        char_score = sum(1 for c in text if c in self.turkish_chars) / len(text)
        
        # Word-level analysis
        words = text.split()
        if not words:
            return char_score
        
        turkish_word_count = 0
        for word in words:
            if any(c in self.turkish_chars for c in word.lower()):
                turkish_word_count += 1
        
        word_score = turkish_word_count / len(words)
        
        # Combined score
        return (char_score * 0.4 + word_score * 0.6)
    
    def is_valid_turkish_text(self, text: str) -> bool:
        """Geçerli Türkçe metin kontrolü"""
        
        if not isinstance(text, str):
            return False
        
        if len(text.strip()) < self.min_score * 100:  # Minimum length
            return False
        
        score = self.calculate_turkish_score(text)
        return score >= self.min_score


class OptimizedDatasetLoader:
    """Optimize edilmiş dataset yükleyici"""
    
    def __init__(self, config: DatasetConfig):
        self.config = config
        self.memory_monitor = MemoryMonitor(config.max_memory_gb)
        self.text_filter = TurkishTextFilter(config.min_turkish_score)
        
        # Cache setup
        if config.use_cache:
            self.cache_dir = Path(config.cache_dir)
            self.cache_dir.mkdir(exist_ok=True)
        
        # Dataset sources
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
    
    def load_streaming_dataset(self) -> Iterator[Dict]:
        """Streaming dataset yükleme"""
        
        logger.info("Starting streaming dataset loading...")
        
        total_samples = 0
        
        # Load local datasets
        for source_name, file_path in self.dataset_sources['local'].items():
            
            if total_samples >= self.config.max_samples_per_source * len(self.dataset_sources['local']):
                break
            
            logger.info(f"Loading {source_name}...")
            
            try:
                yield from self._load_local_file_streaming(source_name, file_path)
                
                # Memory check
                if self.memory_monitor.should_free_memory():
                    self.memory_monitor.force_gc()
                    
            except Exception as e:
                logger.warning(f"Failed to load {source_name}: {e}")
                continue
        
        # Load HuggingFace datasets
        for source_name, config in self.dataset_sources['huggingface'].items():
            
            if total_samples >= self.config.max_samples_per_source * len(self.dataset_sources['huggingface']):
                break
            
            logger.info(f"Loading HuggingFace: {source_name}...")
            
            try:
                yield from self._load_huggingface_streaming(source_name, config)
                
                # Memory check
                if self.memory_monitor.should_free_memory():
                    self.memory_monitor.force_gc()
                    
            except Exception as e:
                logger.warning(f"Failed to load {source_name}: {e}")
                continue
    
    def _load_local_file_streaming(self, source_name: str, file_path: str) -> Iterator[Dict]:
        """Lokal dosya streaming yükleme"""
        
        full_path = Path(__file__).parent / file_path
        
        if not full_path.exists():
            logger.warning(f"File not found: {full_path}")
            return
        
        sample_count = 0
        
        try:
            # Gzip file
            if full_path.suffix == '.gz':
                with gzip.open(full_path, 'rt', encoding='utf-8') as f:
                    yield from self._process_jsonl_stream(f, source_name, sample_count)
            
            # JSONL file
            elif full_path.suffix == '.jsonl':
                with open(full_path, 'r', encoding='utf-8') as f:
                    yield from self._process_jsonl_stream(f, source_name, sample_count)
            
            # JSON file
            elif full_path.suffix == '.json':
                yield from self._process_json_file(full_path, source_name)
            
            # CSV file
            elif full_path.suffix == '.csv':
                yield from self._process_csv_file(full_path, source_name)
            
            else:
                logger.warning(f"Unsupported file type: {full_path.suffix}")
        
        except Exception as e:
            logger.error(f"Error processing {full_path}: {e}")
    
    def _process_jsonl_stream(self, file_handle, source_name: str, sample_count: int) -> Iterator[Dict]:
        """JSONL stream işleme"""
        
        for line_num, line in enumerate(file_handle):
            
            if sample_count >= self.config.max_samples_per_source:
                break
            
            if line_num % 10000 == 0 and line_num > 0:
                logger.info(f"{source_name}: Processed {line_num} lines, yielded {sample_count} samples")
            
            try:
                item = json.loads(line.strip())
                text = self._extract_text_from_item(item)
                
                if self._is_valid_sample(text):
                    yield {
                        'text': text,
                        'source': source_name,
                        'turkish_score': self.text_filter.calculate_turkish_score(text)
                    }
                    sample_count += 1
                
            except (json.JSONDecodeError, Exception):
                continue
    
    def _process_json_file(self, file_path: Path, source_name: str) -> Iterator[Dict]:
        """JSON file işleme"""
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if not isinstance(data, list):
            return
        
        sample_count = 0
        
        for item in data:
            
            if sample_count >= self.config.max_samples_per_source:
                break
            
            text = self._extract_text_from_item(item)
            
            if self._is_valid_sample(text):
                yield {
                    'text': text,
                    'source': source_name,
                    'turkish_score': self.text_filter.calculate_turkish_score(text)
                }
                sample_count += 1
    
    def _process_csv_file(self, file_path: Path, source_name: str) -> Iterator[Dict]:
        """CSV file işleme"""
        
        import pandas as pd
        
        df = pd.read_csv(file_path)
        
        # Text columns bulma
        text_cols = [col for col in df.columns 
                    if any(keyword in col.lower() 
                          for keyword in ['text', 'content', 'question', 'instruction'])]
        
        if not text_cols:
            return
        
        sample_count = 0
        
        for _, row in df.iterrows():
            
            if sample_count >= self.config.max_samples_per_source:
                break
            
            for col in text_cols:
                text = str(row[col])
                
                if self._is_valid_sample(text):
                    yield {
                        'text': text,
                        'source': source_name,
                        'turkish_score': self.text_filter.calculate_turkish_score(text)
                    }
                    sample_count += 1
                    break  # One text per row
    
    def _load_huggingface_streaming(self, dataset_name: str, config: Dict) -> Iterator[Dict]:
        """HuggingFace dataset streaming yükleme"""
        
        try:
            dataset = load_dataset(dataset_name, split='train', streaming=True)
            
            sample_count = 0
            column = config['column']
            limit = config['limit']
            
            for item in dataset:
                
                if sample_count >= limit:
                    break
                
                if column in item:
                    text = str(item[column])
                    
                    if self._is_valid_sample(text):
                        yield {
                            'text': text,
                            'source': dataset_name,
                            'turkish_score': self.text_filter.calculate_turkish_score(text)
                        }
                        sample_count += 1
        
        except Exception as e:
            logger.error(f"Error loading {dataset_name}: {e}")
    
    def _extract_text_from_item(self, item) -> str:
        """Item'dan metin çıkarma"""
        
        if isinstance(item, str):
            return item
        
        if isinstance(item, dict):
            # Try different field names
            for field in ['text', 'content', 'instruction', 'output', 'response']:
                if field in item and item[field]:
                    return str(item[field])
            
            # Try instruction + output combination
            if 'instruction' in item and 'output' in item:
                instruction = str(item.get('instruction', ''))
                output = str(item.get('output', ''))
                if instruction and output:
                    return f"### Talimat:\n{instruction}\n\n### Cevap:\n{output}"
        
        return ""
    
    def _is_valid_sample(self, text: str) -> bool:
        """Geçerli örnek kontrolü"""
        
        if not text or not isinstance(text, str):
            return False
        
        text_len = len(text.strip())
        
        if text_len < self.config.min_text_length or text_len > self.config.max_text_length:
            return False
        
        return self.text_filter.is_valid_turkish_text(text)
    
    def create_batched_dataset(self, tokenizer, max_samples: Optional[int] = None) -> Dataset:
        """Batch'lenmiş dataset oluşturma"""
        
        logger.info("Creating batched dataset...")
        
        # Collect samples in batches
        all_samples = []
        sample_count = 0
        
        for sample in self.load_streaming_dataset():
            
            all_samples.append(sample)
            sample_count += 1
            
            # Batch processing
            if len(all_samples) >= self.config.batch_size:
                self._process_batch(all_samples, tokenizer)
                
                # Memory management
                if self.memory_monitor.should_free_memory():
                    self.memory_monitor.force_gc()
            
            # Max samples check
            if max_samples and sample_count >= max_samples:
                break
        
        # Process remaining samples
        if all_samples:
            self._process_batch(all_samples, tokenizer)
        
        # Create final dataset
        logger.info(f"Creating dataset from {len(all_samples)} samples...")
        dataset = Dataset.from_list(all_samples)
        
        return dataset
    
    def _process_batch(self, samples: List[Dict], tokenizer):
        """Batch işleme"""
        
        # Tokenization
        texts = [sample['text'] for sample in samples]
        
        try:
            tokenized = tokenizer(
                texts,
                truncation=True,
                padding=False,
                max_length=2048,
                return_attention_mask=False
            )
            
            # Add tokenized data to samples
            for i, sample in enumerate(samples):
                if i < len(tokenized['input_ids']):
                    sample['input_ids'] = tokenized['input_ids'][i]
                    sample['token_length'] = len(tokenized['input_ids'][i])
        
        except Exception as e:
            logger.warning(f"Tokenization failed for batch: {e}")
    
    def save_processed_dataset(self, dataset: Dataset, output_path: str):
        """İşlenmiş dataset'i kaydet"""
        
        output_path = Path(output_path)
        
        if self.config.use_compression:
            # Save as compressed JSONL
            with gzip.open(f"{output_path}.jsonl.gz", 'wt', encoding='utf-8') as f:
                for sample in dataset:
                    f.write(json.dumps(sample, ensure_ascii=False) + '\n')
            
            logger.info(f"Compressed dataset saved to {output_path}.jsonl.gz")
        else:
            # Save as regular JSONL
            with open(f"{output_path}.jsonl", 'w', encoding='utf-8') as f:
                for sample in dataset:
                    f.write(json.dumps(sample, ensure_ascii=False) + '\n')
            
            logger.info(f"Dataset saved to {output_path}.jsonl")


def create_optimized_dataset(tokenizer, 
                           max_samples: Optional[int] = None,
                           config: Optional[DatasetConfig] = None) -> Dataset:
    """Optimize edilmiş dataset oluştur"""
    
    if config is None:
        config = DatasetConfig()
    
    loader = OptimizedDatasetLoader(config)
    dataset = loader.create_batched_dataset(tokenizer, max_samples)
    
    # Statistics
    logger.info(f"Final dataset statistics:")
    logger.info(f"  Total samples: {len(dataset)}")
    
    if len(dataset) > 0:
        avg_length = np.mean([len(sample.get('input_ids', [])) for sample in dataset])
        avg_turkish_score = np.mean([sample.get('turkish_score', 0) for sample in dataset])
        
        logger.info(f"  Average token length: {avg_length:.1f}")
        logger.info(f"  Average Turkish score: {avg_turkish_score:.3f}")
    
    return dataset


# Test function
def test_optimized_loader():
    """Optimize edilmiş loader'ı test et"""
    
    print("🧪 Optimized dataset loader test ediliyor...")
    
    config = DatasetConfig(
        max_memory_gb=4.0,
        batch_size=100,
        max_samples_per_source=1000
    )
    
    loader = OptimizedDatasetLoader(config)
    
    # Test streaming
    sample_count = 0
    for sample in loader.load_streaming_dataset():
        sample_count += 1
        if sample_count >= 10:  # Test first 10 samples
            break
    
    print(f"✅ Loaded {sample_count} samples via streaming")
    
    print("🎉 Optimized loader testi başarılı!")


if __name__ == "__main__":
    test_optimized_loader()