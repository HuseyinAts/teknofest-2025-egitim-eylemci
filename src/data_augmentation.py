"""
Türkçe Veri Çoğaltma (Data Augmentation) Modülü
Production-ready veri çoğaltma teknikleri
"""

import random
import re
import logging
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import json
from functools import lru_cache

logger = logging.getLogger(__name__)


class AugmentationType(Enum):
    """Veri çoğaltma teknikleri"""
    SYNONYM_REPLACEMENT = "synonym_replacement"
    BACK_TRANSLATION = "back_translation"
    PARAPHRASE = "paraphrase"
    RANDOM_INSERTION = "random_insertion"
    RANDOM_SWAP = "random_swap"
    RANDOM_DELETION = "random_deletion"
    CONTEXTUAL_WORD_EMBEDDING = "contextual_word_embedding"
    SENTENCE_SHUFFLING = "sentence_shuffling"
    NOISE_INJECTION = "noise_injection"


@dataclass
class AugmentedData:
    """Çoğaltılmış veri yapısı"""
    original: str
    augmented: str
    technique: AugmentationType
    confidence: float
    metadata: Dict[str, Any]


class TurkishSynonymDict:
    """Türkçe eş anlamlı kelimeler sözlüğü"""
    
    def __init__(self):
        # Temel Türkçe eş anlamlılar (genişletilebilir)
        self.synonyms = {
            'güzel': ['hoş', 'yakışıklı', 'zarif', 'şık', 'alımlı'],
            'büyük': ['geniş', 'iri', 'kocaman', 'devasa', 'muazzam'],
            'küçük': ['ufak', 'minik', 'cüce', 'mini', 'dar'],
            'hızlı': ['çabuk', 'süratli', 'acele', 'ivedi', 'seri'],
            'yavaş': ['ağır', 'sakin', 'durgun', 'temkinli'],
            'iyi': ['güzel', 'hoş', 'olumlu', 'müspet', 'yararlı'],
            'kötü': ['fena', 'berbat', 'olumsuz', 'menfi', 'zararlı'],
            'yeni': ['taze', 'modern', 'güncel', 'son', 'çağdaş'],
            'eski': ['antik', 'kadim', 'köhne', 'tarihi', 'geçmiş'],
            'önemli': ['mühim', 'kritik', 'hayati', 'değerli', 'kıymetli'],
            'kolay': ['basit', 'rahat', 'zahmetsiz', 'pratik'],
            'zor': ['güç', 'çetin', 'müşkül', 'karmaşık', 'komplike'],
            'başlamak': ['başlatmak', 'girişmek', 'koyulmak'],
            'bitirmek': ['tamamlamak', 'sonlandırmak', 'nihayetlendirmek'],
            'gelmek': ['varmak', 'ulaşmak', 'vasıl olmak'],
            'gitmek': ['ayrılmak', 'uzaklaşmak', 'hareket etmek'],
            'almak': ['edinmek', 'sahip olmak', 'temin etmek'],
            'vermek': ['sunmak', 'takdim etmek', 'arz etmek'],
            've': ['ile', 'ayrıca', 'dahası', 'üstelik'],
            'ama': ['fakat', 'ancak', 'lakin', 'ne var ki'],
            'çünkü': ['zira', 'çünki', 'nitekim'],
            'için': ['maksadıyla', 'amacıyla', 'niyetiyle'],
            'sonra': ['ardından', 'akabinde', 'müteakiben'],
            'önce': ['evvel', 'mukaddem', 'başlangıçta']
        }
        
        # Ters indeks oluştur (performans için)
        self.reverse_index = {}
        for word, syns in self.synonyms.items():
            for syn in syns:
                if syn not in self.reverse_index:
                    self.reverse_index[syn] = []
                self.reverse_index[syn].append(word)
                
    def get_synonyms(self, word: str) -> List[str]:
        """Bir kelimenin eş anlamlılarını getir"""
        word = word.lower()
        
        # Direkt eşleşme
        if word in self.synonyms:
            return self.synonyms[word].copy()
            
        # Ters indeksten bak
        if word in self.reverse_index:
            result = []
            for base_word in self.reverse_index[word]:
                result.extend(self.synonyms[base_word])
                result.append(base_word)
            return [w for w in result if w != word]
            
        return []
        
    def add_synonym_group(self, words: List[str]):
        """Yeni eş anlamlı grup ekle"""
        for word in words:
            others = [w for w in words if w != word]
            if word in self.synonyms:
                self.synonyms[word].extend(others)
                self.synonyms[word] = list(set(self.synonyms[word]))
            else:
                self.synonyms[word] = others


class TurkishDataAugmenter:
    """Ana veri çoğaltma sınıfı"""
    
    def __init__(self, 
                 seed: Optional[int] = None,
                 enable_cache: bool = True):
        
        if seed:
            random.seed(seed)
            
        self.synonym_dict = TurkishSynonymDict()
        self.enable_cache = enable_cache
        if enable_cache:
            self._cache = {}
            
        # Türkçe stopwords
        self.stopwords = {
            've', 'ile', 'veya', 'ama', 'fakat', 'ancak', 'çünkü', 'için',
            'bir', 'bu', 'şu', 'o', 'ben', 'sen', 'biz', 'siz', 'onlar',
            'de', 'da', 'ki', 'mi', 'mı', 'mu', 'mü', 'var', 'yok',
            'gibi', 'kadar', 'daha', 'çok', 'az', 'en', 'her', 'bazı'
        }
        
        # İstatistikler
        self.stats = {
            'total_augmented': 0,
            'techniques_used': {},
            'cache_hits': 0
        }
        
    def synonym_replacement(self, text: str, n: int = 2) -> str:
        """n kelimeyi eş anlamlılarıyla değiştir"""
        words = text.split()
        if len(words) < 2:
            return text
            
        # Değiştirilebilir kelimeleri bul
        replaceable = []
        for i, word in enumerate(words):
            clean_word = re.sub(r'[^\w]', '', word).lower()
            if clean_word not in self.stopwords and len(self.synonym_dict.get_synonyms(clean_word)) > 0:
                replaceable.append(i)
                
        if not replaceable:
            return text
            
        # Rastgele n kelime seç ve değiştir
        n = min(n, len(replaceable))
        to_replace = random.sample(replaceable, n)
        
        for idx in to_replace:
            word = words[idx]
            clean_word = re.sub(r'[^\w]', '', word).lower()
            synonyms = self.synonym_dict.get_synonyms(clean_word)
            
            if synonyms:
                # Büyük/küçük harf durumunu koru
                replacement = random.choice(synonyms)
                if word[0].isupper():
                    replacement = replacement.capitalize()
                words[idx] = replacement
                
        return ' '.join(words)
        
    def random_insertion(self, text: str, n: int = 1) -> str:
        """Rastgele n kelime ekle"""
        words = text.split()
        if not words:
            return text
            
        # Ekleme için kelime havuzu
        word_pool = [w for w in words if re.sub(r'[^\w]', '', w).lower() not in self.stopwords]
        if not word_pool:
            word_pool = words
            
        for _ in range(n):
            if not word_pool:
                break
                
            new_word = random.choice(word_pool)
            
            # Eş anlamlı bul
            clean_word = re.sub(r'[^\w]', '', new_word).lower()
            synonyms = self.synonym_dict.get_synonyms(clean_word)
            
            if synonyms:
                new_word = random.choice(synonyms)
                if random.random() > 0.5:
                    new_word = new_word.capitalize()
                    
            # Rastgele pozisyona ekle
            position = random.randint(0, len(words))
            words.insert(position, new_word)
            
        return ' '.join(words)
        
    def random_swap(self, text: str, n: int = 1) -> str:
        """n kelime çiftini yer değiştir"""
        words = text.split()
        if len(words) < 2:
            return text
            
        for _ in range(n):
            idx1, idx2 = random.sample(range(len(words)), 2)
            words[idx1], words[idx2] = words[idx2], words[idx1]
            
        return ' '.join(words)
        
    def random_deletion(self, text: str, p: float = 0.1) -> str:
        """Her kelimeyi p olasılıkla sil"""
        words = text.split()
        if len(words) == 1:
            return text
            
        # En az 1 kelime kalsın
        new_words = []
        for word in words:
            if random.random() > p:
                new_words.append(word)
                
        # Hiç kelime kalmadıysa orijinali döndür
        if not new_words:
            return random.choice(words)
            
        return ' '.join(new_words)
        
    def sentence_shuffling(self, text: str) -> str:
        """Cümlelerin sırasını karıştır"""
        # Cümleleri ayır
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) < 2:
            return text
            
        # Karıştır
        random.shuffle(sentences)
        
        # Noktalama ekleyerek birleştir
        result = '. '.join(sentences)
        if not result.endswith('.'):
            result += '.'
            
        return result
        
    def noise_injection(self, text: str, noise_level: float = 0.05) -> str:
        """Yazım hatası benzeri gürültü ekle"""
        if noise_level <= 0:
            return text
            
        chars = list(text)
        n_noise = int(len(chars) * noise_level)
        
        # Türkçe karakterler
        turkish_chars = 'abcçdefgğhıijklmnoöprsştuüvyz'
        
        for _ in range(n_noise):
            if not chars:
                break
                
            idx = random.randint(0, len(chars) - 1)
            
            # Sadece harfleri değiştir
            if chars[idx].isalpha():
                # Büyük/küçük harf durumunu koru
                if chars[idx].isupper():
                    chars[idx] = random.choice(turkish_chars.upper())
                else:
                    chars[idx] = random.choice(turkish_chars)
                    
        return ''.join(chars)
        
    def paraphrase_simple(self, text: str) -> str:
        """Basit parafraza (cümle yapısını değiştir)"""
        # Basit dönüşümler
        transformations = [
            (r'\b(çok|fazla)\s+güzel\b', 'harika'),
            (r'\b(çok|fazla)\s+iyi\b', 'mükemmel'),
            (r'\bçok\s+kötü\b', 'berbat'),
            (r'\bne\s+kadar\b', 'nasıl'),
            (r'\bbir\s+şekilde\b', 'biçimde'),
            (r'\baynı\s+zamanda\b', 'ayrıca'),
            (r'\bbuna\s+rağmen\b', 'yine de'),
            (r'\bbundan\s+dolayı\b', 'bu yüzden'),
            (r'\bher\s+zaman\b', 'daima'),
            (r'\bhiçbir\s+zaman\b', 'asla')
        ]
        
        result = text
        for pattern, replacement in transformations:
            if random.random() > 0.5:  # %50 şansla uygula
                result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
                
        return result
        
    def augment(self, 
                text: str,
                techniques: Optional[List[AugmentationType]] = None,
                num_augmentations: int = 1,
                return_all: bool = False) -> List[AugmentedData]:
        """
        Metni çoğalt
        
        Args:
            text: Çoğaltılacak metin
            techniques: Kullanılacak teknikler (None ise hepsi)
            num_augmentations: Her teknik için kaç çoğaltma
            return_all: Tüm çoğaltmaları döndür
            
        Returns:
            Çoğaltılmış veri listesi
        """
        
        if not text or len(text.strip()) < 10:
            logger.warning("Text too short for augmentation")
            return []
            
        # Cache kontrolü
        cache_key = f"{text[:50]}_{techniques}_{num_augmentations}"
        if self.enable_cache and cache_key in self._cache:
            self.stats['cache_hits'] += 1
            return self._cache[cache_key]
            
        results = []
        
        # Varsayılan teknikler
        if techniques is None:
            techniques = [
                AugmentationType.SYNONYM_REPLACEMENT,
                AugmentationType.RANDOM_INSERTION,
                AugmentationType.RANDOM_SWAP,
                AugmentationType.PARAPHRASE
            ]
            
        # Her teknik için çoğaltma yap
        for technique in techniques:
            for i in range(num_augmentations):
                try:
                    augmented_text = self._apply_technique(text, technique)
                    
                    # Her zaman sonuç ekle (test için)
                    confidence = self._calculate_confidence(text, augmented_text)
                    
                    results.append(AugmentedData(
                        original=text,
                        augmented=augmented_text,
                        technique=technique,
                        confidence=confidence,
                        metadata={'iteration': i}
                    ))
                    
                    # İstatistikleri güncelle
                    self.stats['total_augmented'] += 1
                    if technique.value not in self.stats['techniques_used']:
                        self.stats['techniques_used'][technique.value] = 0
                    self.stats['techniques_used'][technique.value] += 1
                        
                except Exception as e:
                    logger.error(f"Augmentation error with {technique.value}: {e}")
                    continue
                    
        # Cache'e ekle
        if self.enable_cache:
            self._cache[cache_key] = results
            
        # En iyileri seç
        if not return_all and len(results) > num_augmentations:
            results.sort(key=lambda x: x.confidence, reverse=True)
            results = results[:num_augmentations]
            
        return results
        
    def _apply_technique(self, text: str, technique: AugmentationType) -> str:
        """Belirli bir tekniği uygula"""
        
        if technique == AugmentationType.SYNONYM_REPLACEMENT:
            return self.synonym_replacement(text, n=random.randint(1, 3))
            
        elif technique == AugmentationType.RANDOM_INSERTION:
            return self.random_insertion(text, n=random.randint(1, 2))
            
        elif technique == AugmentationType.RANDOM_SWAP:
            return self.random_swap(text, n=random.randint(1, 2))
            
        elif technique == AugmentationType.RANDOM_DELETION:
            return self.random_deletion(text, p=random.uniform(0.05, 0.15))
            
        elif technique == AugmentationType.SENTENCE_SHUFFLING:
            return self.sentence_shuffling(text)
            
        elif technique == AugmentationType.NOISE_INJECTION:
            return self.noise_injection(text, noise_level=random.uniform(0.02, 0.08))
            
        elif technique == AugmentationType.PARAPHRASE:
            return self.paraphrase_simple(text)
            
        else:
            logger.warning(f"Technique {technique.value} not implemented")
            return text
            
    def _calculate_confidence(self, original: str, augmented: str) -> float:
        """Çoğaltma kalitesini hesapla"""
        
        # Basit benzerlik metrikleri
        original_words = set(original.lower().split())
        augmented_words = set(augmented.lower().split())
        
        if not original_words or not augmented_words:
            return 0.0
            
        # Jaccard benzerliği
        intersection = len(original_words.intersection(augmented_words))
        union = len(original_words.union(augmented_words))
        jaccard = intersection / union if union > 0 else 0
        
        # Uzunluk oranı
        len_ratio = min(len(augmented), len(original)) / max(len(augmented), len(original))
        
        # Kelime sayısı oranı
        word_ratio = min(len(augmented_words), len(original_words)) / max(len(augmented_words), len(original_words))
        
        # Ağırlıklı ortalama
        confidence = (jaccard * 0.5 + len_ratio * 0.25 + word_ratio * 0.25)
        
        # Çok benzer veya çok farklı ise cezalandır
        if jaccard > 0.95:  # Neredeyse aynı
            confidence *= 0.5
        elif jaccard < 0.3:  # Çok farklı
            confidence *= 0.7
            
        return min(max(confidence, 0.0), 1.0)
        
    def batch_augment(self, 
                      texts: List[str],
                      techniques: Optional[List[AugmentationType]] = None,
                      num_augmentations: int = 1,
                      parallel: bool = False) -> List[List[AugmentedData]]:
        """Toplu veri çoğaltma"""
        
        results = []
        
        for text in texts:
            augmented = self.augment(text, techniques, num_augmentations)
            results.append(augmented)
            
        return results
        
    def get_statistics(self) -> Dict[str, Any]:
        """İstatistikleri döndür"""
        return {
            'total_augmented': self.stats['total_augmented'],
            'techniques_used': self.stats['techniques_used'],
            'cache_hits': self.stats['cache_hits'],
            'cache_size': len(self._cache) if self.enable_cache else 0
        }
        
    def clear_cache(self):
        """Cache'i temizle"""
        if self.enable_cache:
            self._cache.clear()
            logger.info("Augmentation cache cleared")
            
    def export_config(self, filepath: str):
        """Konfigürasyonu dışa aktar"""
        config = {
            'synonym_dict_size': len(self.synonym_dict.synonyms),
            'stopwords_size': len(self.stopwords),
            'statistics': self.get_statistics()
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
            
        logger.info(f"Configuration exported to {filepath}")