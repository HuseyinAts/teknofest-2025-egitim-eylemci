"""
Comprehensive Turkish NLP Integration Module
TEKNOFEST 2025 - Unified Turkish NLP with multiple libraries
"""

import os
import re
import json
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
from pathlib import Path

# Turkish NLP Libraries
try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

try:
    from transformers import (
        pipeline, 
        AutoTokenizer, 
        AutoModelForTokenClassification,
        AutoModelForSequenceClassification,
        AutoModelForQuestionAnswering
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    import nltk
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

# Import our custom modules
from .advanced_turkish_morphology import TurkishMorphologyAnalyzer, TurkishStemmer, TurkishLemmatizer
from .turkish_bpe_tokenizer import TurkishBPETokenizer, TokenizerConfig
from .zemberek_integration import ZemberekIntegration, ZemberekConfig, ZemberekMode

logger = logging.getLogger(__name__)


class NLPTask(Enum):
    """Supported NLP tasks"""
    TOKENIZATION = "tokenization"
    MORPHOLOGY = "morphology"
    LEMMATIZATION = "lemmatization"
    STEMMING = "stemming"
    POS_TAGGING = "pos_tagging"
    NER = "ner"
    SENTIMENT = "sentiment"
    TEXT_CLASSIFICATION = "classification"
    QUESTION_ANSWERING = "qa"
    SUMMARIZATION = "summarization"
    TRANSLATION = "translation"
    DEPENDENCY_PARSING = "dependency"
    WORD_EMBEDDINGS = "embeddings"


@dataclass
class NLPResult:
    """Unified result structure for NLP tasks"""
    task: NLPTask
    input_text: str
    output: Any
    confidence: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    processing_time: Optional[float] = None


class TurkishNLPPipeline:
    """Comprehensive Turkish NLP pipeline integrating multiple libraries"""
    
    def __init__(self, 
                 use_zemberek: bool = True,
                 use_transformers: bool = True,
                 use_spacy: bool = False,
                 cache_enabled: bool = True):
        
        self.use_zemberek = use_zemberek
        self.use_transformers = use_transformers and TRANSFORMERS_AVAILABLE
        self.use_spacy = use_spacy and SPACY_AVAILABLE
        self.cache = {} if cache_enabled else None
        
        # Initialize components
        self._init_morphology()
        self._init_tokenizers()
        self._init_models()
        self._init_resources()
        
        logger.info("Turkish NLP Pipeline initialized")
        
    def _init_morphology(self):
        """Initialize morphological analyzers"""
        # Custom morphology analyzer
        self.morphology_analyzer = TurkishMorphologyAnalyzer()
        self.stemmer = TurkishStemmer()
        self.lemmatizer = TurkishLemmatizer()
        
        # Zemberek integration
        if self.use_zemberek:
            try:
                self.zemberek = ZemberekIntegration(
                    ZemberekConfig(mode=ZemberekMode.REST_API)
                )
                logger.info("Zemberek integration enabled")
            except Exception as e:
                logger.warning(f"Zemberek initialization failed: {e}")
                self.use_zemberek = False
                
    def _init_tokenizers(self):
        """Initialize tokenizers"""
        # Custom BPE tokenizer
        self.bpe_tokenizer = None
        tokenizer_path = "models/turkish_bpe_tokenizer"
        
        if os.path.exists(tokenizer_path):
            try:
                self.bpe_tokenizer = TurkishBPETokenizer.load(tokenizer_path)
                logger.info("Custom BPE tokenizer loaded")
            except Exception as e:
                logger.warning(f"BPE tokenizer load failed: {e}")
                
        # Transformers tokenizers
        self.transformer_tokenizers = {}
        if self.use_transformers:
            try:
                # BERT Turkish
                self.transformer_tokenizers['bert'] = AutoTokenizer.from_pretrained(
                    "dbmdz/bert-base-turkish-cased"
                )
                # ELECTRA Turkish
                self.transformer_tokenizers['electra'] = AutoTokenizer.from_pretrained(
                    "dbmdz/electra-base-turkish-cased-discriminator"
                )
                logger.info("Transformer tokenizers loaded")
            except Exception as e:
                logger.warning(f"Transformer tokenizer load failed: {e}")
                
    def _init_models(self):
        """Initialize pre-trained models"""
        self.models = {}
        
        if self.use_transformers:
            try:
                # NER Model
                self.models['ner'] = pipeline(
                    "ner",
                    model="savasy/bert-base-turkish-ner-cased",
                    aggregation_strategy="simple"
                )
                
                # Sentiment Analysis
                self.models['sentiment'] = pipeline(
                    "sentiment-analysis",
                    model="savasy/bert-base-turkish-sentiment-cased"
                )
                
                # Text Classification
                self.models['classification'] = pipeline(
                    "text-classification",
                    model="dbmdz/bert-base-turkish-cased"
                )
                
                logger.info("Transformer models loaded")
                
            except Exception as e:
                logger.warning(f"Model loading failed: {e}")
                
        # SpaCy model
        if self.use_spacy:
            try:
                import spacy
                self.spacy_nlp = spacy.load("tr_core_news_trf")
                logger.info("SpaCy Turkish model loaded")
            except Exception as e:
                logger.warning(f"SpaCy model load failed: {e}")
                self.use_spacy = False
                
    def _init_resources(self):
        """Initialize linguistic resources"""
        # Turkish stopwords
        self.stopwords = set([
            "ve", "ile", "de", "da", "ki", "bu", "bir", "için", "olan",
            "olarak", "daha", "çok", "en", "gibi", "sonra", "kadar",
            "ama", "ancak", "her", "hem", "ya", "veya", "değil", "mi",
            "mu", "mı", "mü", "ise", "ne", "nasıl", "neden", "niçin",
            "kim", "kimin", "nerede", "nereye", "nereden", "hangi",
            "bütün", "tüm", "bazı", "hiç", "şey", "şu", "o", "ben",
            "sen", "biz", "siz", "onlar", "beni", "seni", "bizi", 
            "sizi", "onları", "benim", "senin", "bizim", "sizin",
            "onların", "böyle", "şöyle", "öyle", "bile", "rağmen",
            "dolayı", "zaman", "şimdi", "bugün", "dün", "yarın"
        ])
        
        # Turkish abbreviations
        self.abbreviations = {
            "Dr.": "Doktor",
            "Prof.": "Profesör",
            "Doç.": "Doçent",
            "Av.": "Avukat",
            "vb.": "ve benzeri",
            "vs.": "vesaire",
            "örn.": "örneğin",
            "bkz.": "bakınız",
            "yy.": "yüzyıl",
            "M.Ö.": "Milattan Önce",
            "M.S.": "Milattan Sonra",
            "T.C.": "Türkiye Cumhuriyeti"
        }
        
        # Entity type mappings
        self.entity_types = {
            "PER": "Kişi",
            "LOC": "Yer",
            "ORG": "Kurum",
            "DATE": "Tarih",
            "TIME": "Zaman",
            "MONEY": "Para",
            "PERCENT": "Yüzde",
            "MISC": "Diğer"
        }
        
    def tokenize(self, text: str, method: str = "best") -> List[str]:
        """Tokenize Turkish text"""
        import time
        start_time = time.time()
        
        tokens = []
        
        if method == "best" or method == "zemberek":
            if self.use_zemberek:
                tokens = self.zemberek.tokenize(text)
                
        if not tokens and (method == "best" or method == "bpe"):
            if self.bpe_tokenizer:
                tokens = self.bpe_tokenizer.tokenize(text)
                
        if not tokens and (method == "best" or method == "transformer"):
            if 'bert' in self.transformer_tokenizers:
                encoding = self.transformer_tokenizers['bert'].tokenize(text)
                tokens = encoding
                
        if not tokens and (method == "best" or method == "spacy"):
            if self.use_spacy:
                doc = self.spacy_nlp(text)
                tokens = [token.text for token in doc]
                
        if not tokens:
            # Fallback to simple tokenization
            tokens = self._simple_tokenize(text)
            
        processing_time = time.time() - start_time
        
        return NLPResult(
            task=NLPTask.TOKENIZATION,
            input_text=text,
            output=tokens,
            processing_time=processing_time
        )
        
    def _simple_tokenize(self, text: str) -> List[str]:
        """Simple regex-based tokenization"""
        # Handle Turkish punctuation and special cases
        text = re.sub(r'([.!?,;:])', r' \1 ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip().split()
        
    def analyze_morphology(self, text: str, use_zemberek_first: bool = True) -> NLPResult:
        """Perform morphological analysis"""
        import time
        start_time = time.time()
        
        analyses = []
        
        if use_zemberek_first and self.use_zemberek:
            try:
                analyses = self.zemberek.analyze_morphology(text)
            except Exception as e:
                logger.warning(f"Zemberek morphology failed: {e}")
                
        if not analyses:
            # Use custom morphology analyzer
            words = self._simple_tokenize(text)
            for word in words:
                analysis = self.morphology_analyzer.analyze(word)
                analyses.append(analysis)
                
        processing_time = time.time() - start_time
        
        return NLPResult(
            task=NLPTask.MORPHOLOGY,
            input_text=text,
            output=analyses,
            processing_time=processing_time
        )
        
    def lemmatize(self, text: str) -> NLPResult:
        """Lemmatize Turkish text"""
        import time
        start_time = time.time()
        
        words = self._simple_tokenize(text)
        lemmas = []
        
        for word in words:
            if self.use_zemberek:
                try:
                    analyses = self.zemberek.analyze_morphology(word)
                    if analyses:
                        lemmas.append(analyses[0].lemma)
                    else:
                        lemmas.append(self.lemmatizer.lemmatize(word))
                except:
                    lemmas.append(self.lemmatizer.lemmatize(word))
            else:
                lemmas.append(self.lemmatizer.lemmatize(word))
                
        processing_time = time.time() - start_time
        
        return NLPResult(
            task=NLPTask.LEMMATIZATION,
            input_text=text,
            output=lemmas,
            processing_time=processing_time
        )
        
    def stem(self, text: str) -> NLPResult:
        """Stem Turkish text"""
        import time
        start_time = time.time()
        
        words = self._simple_tokenize(text)
        stems = self.stemmer.stem_batch(words)
        
        processing_time = time.time() - start_time
        
        return NLPResult(
            task=NLPTask.STEMMING,
            input_text=text,
            output=stems,
            processing_time=processing_time
        )
        
    def extract_entities(self, text: str) -> NLPResult:
        """Extract named entities"""
        import time
        start_time = time.time()
        
        entities = []
        
        if 'ner' in self.models:
            try:
                ner_results = self.models['ner'](text)
                for entity in ner_results:
                    entities.append({
                        "text": entity['word'],
                        "type": entity['entity_group'],
                        "score": entity['score'],
                        "start": entity['start'],
                        "end": entity['end']
                    })
            except Exception as e:
                logger.warning(f"NER model failed: {e}")
                
        if not entities and self.use_zemberek:
            try:
                entities = self.zemberek.find_named_entities(text)
            except Exception as e:
                logger.warning(f"Zemberek NER failed: {e}")
                
        if not entities:
            # Fallback to pattern-based NER
            entities = self._pattern_based_ner(text)
            
        processing_time = time.time() - start_time
        
        return NLPResult(
            task=NLPTask.NER,
            input_text=text,
            output=entities,
            processing_time=processing_time
        )
        
    def _pattern_based_ner(self, text: str) -> List[Dict]:
        """Simple pattern-based NER"""
        entities = []
        
        # Person names (simplified)
        person_pattern = r'\b[A-ZÇĞİÖŞÜ][a-zçğıöşü]+\s+[A-ZÇĞİÖŞÜ][a-zçğıöşü]+\b'
        for match in re.finditer(person_pattern, text):
            entities.append({
                "text": match.group(),
                "type": "PER",
                "start": match.start(),
                "end": match.end()
            })
            
        # Locations with common suffixes
        location_suffixes = ["'de", "'da", "'den", "'dan", "'e", "'a", "'ye", "'ya"]
        for suffix in location_suffixes:
            pattern = rf'\b[A-ZÇĞİÖŞÜ][a-zçğıöşü]+{suffix}\b'
            for match in re.finditer(pattern, text):
                entities.append({
                    "text": match.group()[:-len(suffix)],
                    "type": "LOC",
                    "start": match.start(),
                    "end": match.end() - len(suffix)
                })
                
        return entities
        
    def analyze_sentiment(self, text: str) -> NLPResult:
        """Analyze sentiment of Turkish text"""
        import time
        start_time = time.time()
        
        sentiment = None
        confidence = 0.0
        
        if 'sentiment' in self.models:
            try:
                result = self.models['sentiment'](text)[0]
                sentiment = result['label']
                confidence = result['score']
            except Exception as e:
                logger.warning(f"Sentiment model failed: {e}")
                
        if sentiment is None:
            # Fallback to lexicon-based sentiment
            sentiment, confidence = self._lexicon_sentiment(text)
            
        processing_time = time.time() - start_time
        
        return NLPResult(
            task=NLPTask.SENTIMENT,
            input_text=text,
            output=sentiment,
            confidence=confidence,
            processing_time=processing_time
        )
        
    def _lexicon_sentiment(self, text: str) -> Tuple[str, float]:
        """Simple lexicon-based sentiment analysis"""
        positive_words = {
            "güzel", "iyi", "harika", "mükemmel", "başarılı", "mutlu",
            "sevindirici", "olumlu", "faydalı", "yararlı", "süper"
        }
        
        negative_words = {
            "kötü", "berbat", "rezalet", "başarısız", "mutsuz", "üzücü",
            "olumsuz", "zararlı", "korkunç", "fena", "rezil"
        }
        
        words = set(w.lower() for w in self._simple_tokenize(text))
        
        pos_count = len(words & positive_words)
        neg_count = len(words & negative_words)
        
        if pos_count > neg_count:
            return "POSITIVE", pos_count / (pos_count + neg_count)
        elif neg_count > pos_count:
            return "NEGATIVE", neg_count / (pos_count + neg_count)
        else:
            return "NEUTRAL", 0.5
            
    def remove_stopwords(self, text: str) -> List[str]:
        """Remove Turkish stopwords"""
        words = self._simple_tokenize(text.lower())
        return [w for w in words if w not in self.stopwords]
        
    def expand_abbreviations(self, text: str) -> str:
        """Expand Turkish abbreviations"""
        for abbr, expansion in self.abbreviations.items():
            text = text.replace(abbr, expansion)
        return text
        
    def analyze_complete(self, text: str) -> Dict[str, Any]:
        """Perform complete analysis of Turkish text"""
        results = {}
        
        # Tokenization
        tokens_result = self.tokenize(text)
        results['tokens'] = tokens_result.output
        
        # Morphology
        morph_result = self.analyze_morphology(text)
        results['morphology'] = morph_result.output
        
        # Lemmatization
        lemma_result = self.lemmatize(text)
        results['lemmas'] = lemma_result.output
        
        # Stemming
        stem_result = self.stem(text)
        results['stems'] = stem_result.output
        
        # Named entities
        ner_result = self.extract_entities(text)
        results['entities'] = ner_result.output
        
        # Sentiment
        sentiment_result = self.analyze_sentiment(text)
        results['sentiment'] = {
            'label': sentiment_result.output,
            'confidence': sentiment_result.confidence
        }
        
        # Statistics
        results['statistics'] = {
            'char_count': len(text),
            'word_count': len(results['tokens']),
            'sentence_count': text.count('.') + text.count('!') + text.count('?'),
            'stopword_count': len([w for w in results['tokens'] if w.lower() in self.stopwords])
        }
        
        return results


def test_turkish_nlp():
    """Test Turkish NLP pipeline"""
    print("Testing Turkish NLP Pipeline")
    print("=" * 50)
    
    # Initialize pipeline
    pipeline = TurkishNLPPipeline(
        use_zemberek=False,  # Set to True if Zemberek is available
        use_transformers=False,  # Set to True if models are downloaded
        use_spacy=False  # Set to True if spaCy model is installed
    )
    
    test_texts = [
        "Ankara Türkiye'nin başkentidir.",
        "Öğrenciler yarın sınava girecekler.",
        "Bu kitap çok güzel ve faydalı.",
        "İstanbul Boğazı'nda vapurla gezinti yaptık.",
        "TEKNOFEST 2025 yarışması için hazırlanıyoruz."
    ]
    
    for text in test_texts:
        print(f"\nText: {text}")
        print("-" * 40)
        
        # Complete analysis
        results = pipeline.analyze_complete(text)
        
        print(f"Tokens: {results['tokens']}")
        print(f"Lemmas: {results['lemmas']}")
        print(f"Stems: {results['stems']}")
        print(f"Entities: {results['entities']}")
        print(f"Sentiment: {results['sentiment']}")
        print(f"Statistics: {results['statistics']}")


if __name__ == "__main__":
    test_turkish_nlp()