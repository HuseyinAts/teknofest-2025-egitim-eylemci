"""
Türkçe NLP Optimizasyon Modülü
Production-ready Türkçe metin işleme ve optimizasyon araçları
"""

import re
import ftfy
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import unicodedata
from collections import Counter
import hashlib
from functools import lru_cache

# Logging yapılandırması
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TextQuality(Enum):
    """Metin kalite seviyeleri"""
    EXCELLENT = "excellent"  # 90-100
    GOOD = "good"           # 75-89
    FAIR = "fair"           # 60-74
    POOR = "poor"           # 40-59
    VERY_POOR = "very_poor" # 0-39


@dataclass
class ProcessedText:
    """İşlenmiş metin veri yapısı"""
    original: str
    cleaned: str
    quality_score: float
    quality_level: TextQuality
    language: str
    metadata: Dict[str, Any]
    warnings: List[str]
    
    
class TurkishMorphologyAnalyzer:
    """Türkçe morfoloji analizi ve lemmatizasyon"""
    
    def __init__(self):
        # Türkçe ekler ve kökler
        self.suffixes = {
            'ler', 'lar', 'den', 'dan', 'de', 'da', 'e', 'a',
            'i', 'ı', 'u', 'ü', 'im', 'ım', 'um', 'üm',
            'sin', 'sın', 'sun', 'sün', 'iz', 'ız', 'uz', 'üz',
            'siniz', 'sınız', 'sunuz', 'sünüz', 'ler', 'lar',
            'li', 'lı', 'lu', 'lü', 'lik', 'lık', 'luk', 'lük',
            'ci', 'cı', 'cu', 'cü', 'çi', 'çı', 'çu', 'çü',
            'siz', 'sız', 'suz', 'süz', 'ca', 'ce', 'ça', 'çe',
            'mek', 'mak', 'dik', 'dık', 'duk', 'dük', 'tik', 'tık', 'tuk', 'tük',
            'miş', 'mış', 'muş', 'müş', 'ecek', 'acak',
            'yor', 'iyor', 'ıyor', 'uyor', 'üyor'
        }
        
        # Türkçe kökleri cache'le
        self._stem_cache = {}
        
    def lemmatize(self, word: str) -> str:
        """Kelimeyi kök haline getir"""
        if word in self._stem_cache:
            return self._stem_cache[word]
            
        original = word
        word = word.lower()
        
        # En uzun ek eşleşmesini bul
        for length in range(len(word), 0, -1):
            for start in range(len(word) - length + 1):
                suffix = word[start:]
                if suffix in self.suffixes and len(word[:start]) > 2:
                    stem = word[:start]
                    self._stem_cache[original] = stem
                    return stem
                    
        self._stem_cache[original] = word
        return word
        
    def analyze_morphology(self, text: str) -> Dict[str, Any]:
        """Morfolojik analiz yap"""
        words = text.split()
        stems = [self.lemmatize(w) for w in words]
        
        return {
            'word_count': len(words),
            'unique_stems': len(set(stems)),
            'stem_ratio': len(set(stems)) / len(words) if words else 0,
            'most_common_stems': Counter(stems).most_common(10)
        }


class TurkishNER:
    """Türkçe Adlandırılmış Varlık Tanıma"""
    
    def __init__(self):
        # Türkçe özel isimler için basit pattern'ler
        self.patterns = {
            'PERSON': [
                r'\b[A-ZÇĞİÖŞÜ][a-zçğıöşü]+\s+[A-ZÇĞİÖŞÜ][a-zçğıöşü]+\b',
                r'\b(Bay|Bayan|Sayın|Dr|Prof|Doç)\s+[A-ZÇĞİÖŞÜ][a-zçğıöşü]+\b'
            ],
            'LOCATION': [
                r'\b(İstanbul|Ankara|İzmir|Bursa|Antalya|Türkiye|Anadolu)\b',
                r'\b[A-ZÇĞİÖŞÜ][a-zçğıöşü]+\s+(İl|İlçe|Mahalle|Caddesi|Sokak)\b'
            ],
            'ORGANIZATION': [
                r'\b[A-ZÇĞİÖŞÜ][a-zçğıöşü]+\s+(A\.Ş\.|Ltd\.|Şirketi|Üniversitesi|Bakanlığı)\b',
                r'\b(TBMM|MEB|YÖK|TÜBİTAK|TOBB)\b'
            ],
            'DATE': [
                r'\b\d{1,2}\s+(Ocak|Şubat|Mart|Nisan|Mayıs|Haziran|Temmuz|Ağustos|Eylül|Ekim|Kasım|Aralık)\s+\d{4}\b',
                r'\b\d{1,2}[./]\d{1,2}[./]\d{4}\b'
            ]
        }
        
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Varlıkları çıkar"""
        entities = {}
        
        for entity_type, patterns in self.patterns.items():
            entities[entity_type] = []
            for pattern in patterns:
                matches = re.findall(pattern, text)
                entities[entity_type].extend(matches)
            entities[entity_type] = list(set(entities[entity_type]))
            
        return entities


class TurkishTextOptimizer:
    """Ana Türkçe metin optimizasyon sınıfı"""
    
    def __init__(self, enable_cache: bool = True):
        # Türkçe karakterler
        self.turkish_chars = set('abcçdefgğhıijklmnoöprsştuüvyzABCÇDEFGĞHIİJKLMNOÖPRSŞTUÜVYZ')
        self.turkish_specific = set('çğıöşüÇĞİÖŞÜ')
        
        # Alt modüller
        self.morphology = TurkishMorphologyAnalyzer()
        self.ner = TurkishNER()
        
        # Regex pattern'leri compile et (performans için)
        self.patterns = {
            'url': re.compile(r'https?://\S+|www\.\S+'),
            'email': re.compile(r'\S+@\S+\.\S+'),
            'html': re.compile(r'<[^>]+>'),
            'phone': re.compile(r'(\+90|0)?\s*\d{3}\s*\d{3}\s*\d{2}\s*\d{2}'),
            'multiple_spaces': re.compile(r'\s+'),
            'multiple_newlines': re.compile(r'\n{3,}'),
            'special_chars': re.compile(r'[^\w\s.,!?;:\-\'\"çğıöşüÇĞİÖŞÜ]'),
            'abbreviations': re.compile(r'\b(?:vb|vs|vd|yy|sy|sf|bkz|bk|Dr|Prof|Doç|Yrd|Av|Müh)\.')
        }
        
        # Cache
        self.enable_cache = enable_cache
        if enable_cache:
            self._cache = {}
            
        # İstatistikler
        self.stats = {
            'processed_count': 0,
            'total_quality': 0,
            'quality_distribution': Counter()
        }
        
    def _get_cache_key(self, text: str) -> str:
        """Cache key oluştur"""
        return hashlib.md5(text.encode()).hexdigest()
        
    def clean_text(self, text: str, preserve_structure: bool = False) -> str:
        """Metni temizle"""
        if not text:
            return ""
            
        # Encoding düzeltmesi
        text = ftfy.fix_text(text)
        
        # Unicode normalizasyon
        text = unicodedata.normalize('NFC', text)
        
        # URL ve email'leri placeholder ile değiştir
        url_placeholders = []
        for match in self.patterns['url'].finditer(text):
            url_placeholders.append(match.group())
            text = text.replace(match.group(), f'[URL_{len(url_placeholders)-1}]')
            
        email_placeholders = []
        for match in self.patterns['email'].finditer(text):
            email_placeholders.append(match.group())
            text = text.replace(match.group(), f'[EMAIL_{len(email_placeholders)-1}]')
            
        # HTML etiketlerini kaldır
        text = self.patterns['html'].sub('', text)
        
        # Özel karakterleri temizle (Türkçe karakterleri koru)
        if not preserve_structure:
            text = self.patterns['special_chars'].sub(' ', text)
            
        # Fazla boşlukları temizle
        text = self.patterns['multiple_spaces'].sub(' ', text)
        text = self.patterns['multiple_newlines'].sub('\n\n', text)
        
        # Baş ve sondaki boşlukları temizle
        text = text.strip()
        
        return text
        
    def calculate_quality_score(self, text: str) -> Tuple[float, Dict[str, float]]:
        """Detaylı kalite skoru hesapla"""
        if not text:
            return 0.0, {}
            
        scores = {}
        
        # 1. Uzunluk skoru (100-5000 ideal)
        text_len = len(text)
        if 100 <= text_len <= 5000:
            scores['length'] = 1.0
        elif text_len < 100:
            scores['length'] = text_len / 100
        else:
            scores['length'] = max(0, 1 - (text_len - 5000) / 10000)
            
        # 2. Türkçe karakter oranı
        turkish_ratio = self._calculate_turkish_ratio(text)
        scores['turkish'] = min(turkish_ratio * 20, 1.0)  # %5 = tam puan
        
        # 3. Kelime çeşitliliği
        words = text.lower().split()
        if words:
            unique_ratio = len(set(words)) / len(words)
            scores['diversity'] = unique_ratio
        else:
            scores['diversity'] = 0
            
        # 4. Cümle yapısı
        sentences = re.split(r'[.!?]+', text)
        if len(sentences) > 1:
            avg_sentence_len = sum(len(s.split()) for s in sentences) / len(sentences)
            if 5 <= avg_sentence_len <= 20:
                scores['structure'] = 1.0
            else:
                scores['structure'] = 0.5
        else:
            scores['structure'] = 0.3
            
        # 5. Noktalama dengesi
        punct_count = sum(1 for c in text if c in '.,!?;:')
        if words:
            punct_ratio = punct_count / len(words)
            if 0.05 <= punct_ratio <= 0.15:
                scores['punctuation'] = 1.0
            else:
                scores['punctuation'] = 0.5
        else:
            scores['punctuation'] = 0
            
        # 6. Büyük/küçük harf dengesi
        alpha_chars = [c for c in text if c.isalpha()]
        if alpha_chars:
            upper_ratio = sum(1 for c in alpha_chars if c.isupper()) / len(alpha_chars)
            if 0.02 <= upper_ratio <= 0.15:
                scores['capitalization'] = 1.0
            else:
                scores['capitalization'] = 0.5
        else:
            scores['capitalization'] = 0
            
        # Ağırlıklı ortalama
        weights = {
            'length': 0.2,
            'turkish': 0.25,
            'diversity': 0.2,
            'structure': 0.15,
            'punctuation': 0.1,
            'capitalization': 0.1
        }
        
        total_score = sum(scores.get(k, 0) * v for k, v in weights.items()) * 100
        
        return total_score, scores
        
    def _calculate_turkish_ratio(self, text: str) -> float:
        """Türkçe karakter oranını hesapla"""
        if not text:
            return 0.0
            
        char_count = sum(1 for c in text if c.isalpha())
        if char_count == 0:
            return 0.0
            
        turkish_count = sum(1 for c in text if c in self.turkish_specific)
        return turkish_count / char_count
        
    def detect_language(self, text: str) -> str:
        """Dil tespiti (basitleştirilmiş)"""
        # Türkçe özel karakter oranına bak
        if self._calculate_turkish_ratio(text) > 0.02:
            return 'tr'
            
        # Basit kelime kontrolü
        turkish_words = {'ve', 'bir', 'bu', 'için', 'ile', 'da', 'de', 'ki', 'ne', 'var', 'yok'}
        words = set(text.lower().split()[:50])  # İlk 50 kelime
        
        if len(words.intersection(turkish_words)) >= 3:
            return 'tr'
            
        return 'unknown'
        
    def process(self, text: str, 
                enable_morphology: bool = True,
                enable_ner: bool = True,
                use_cache: bool = True) -> ProcessedText:
        """Metni komple işle"""
        
        # Cache kontrolü
        if use_cache and self.enable_cache:
            cache_key = self._get_cache_key(text)
            if cache_key in self._cache:
                logger.debug(f"Cache hit for text hash: {cache_key}")
                return self._cache[cache_key]
                
        warnings = []
        metadata = {}
        
        # Temizle
        cleaned = self.clean_text(text)
        
        if not cleaned or len(cleaned) < 20:
            warnings.append("Text too short after cleaning")
            return ProcessedText(
                original=text,
                cleaned=cleaned,
                quality_score=0,
                quality_level=TextQuality.VERY_POOR,
                language='unknown',
                metadata={},
                warnings=warnings
            )
            
        # Dil tespiti
        language = self.detect_language(cleaned)
        if language != 'tr':
            warnings.append(f"Non-Turkish content detected: {language}")
            
        # Kalite skoru
        quality_score, score_details = self.calculate_quality_score(cleaned)
        metadata['score_details'] = score_details
        
        # Kalite seviyesi
        if quality_score >= 90:
            quality_level = TextQuality.EXCELLENT
        elif quality_score >= 75:
            quality_level = TextQuality.GOOD
        elif quality_score >= 60:
            quality_level = TextQuality.FAIR
        elif quality_score >= 40:
            quality_level = TextQuality.POOR
        else:
            quality_level = TextQuality.VERY_POOR
            
        # Morfoloji analizi
        if enable_morphology and language == 'tr':
            metadata['morphology'] = self.morphology.analyze_morphology(cleaned)
            
        # NER
        if enable_ner and language == 'tr':
            metadata['entities'] = self.ner.extract_entities(cleaned)
            
        # İstatistikleri güncelle
        self.stats['processed_count'] += 1
        self.stats['total_quality'] += quality_score
        self.stats['quality_distribution'][quality_level.value] += 1
        
        result = ProcessedText(
            original=text,
            cleaned=cleaned,
            quality_score=quality_score,
            quality_level=quality_level,
            language=language,
            metadata=metadata,
            warnings=warnings
        )
        
        # Cache'e ekle
        if use_cache and self.enable_cache:
            self._cache[cache_key] = result
            
        return result
        
    def get_statistics(self) -> Dict[str, Any]:
        """İstatistikleri döndür"""
        return {
            'total_processed': self.stats['processed_count'],
            'average_quality': self.stats['total_quality'] / self.stats['processed_count'] if self.stats['processed_count'] > 0 else 0,
            'quality_distribution': dict(self.stats['quality_distribution']),
            'cache_size': len(self._cache) if self.enable_cache else 0
        }
        
    def clear_cache(self):
        """Cache'i temizle"""
        if self.enable_cache:
            self._cache.clear()
            logger.info("Cache cleared")