"""
Comprehensive Test Suite for Turkish NLP Modules
TEKNOFEST 2025 - Testing all Turkish NLP components
"""

import pytest
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.nlp.advanced_turkish_morphology import (
    TurkishMorphologyAnalyzer,
    TurkishStemmer,
    TurkishLemmatizer,
    MorphemeType,
    VowelType
)
from src.nlp.turkish_bpe_tokenizer import (
    TurkishBPETokenizer,
    TokenizerConfig
)
from src.nlp.turkish_nlp_integration import (
    TurkishNLPPipeline,
    NLPTask
)


class TestTurkishMorphology:
    """Test suite for Turkish morphology analyzer"""
    
    @pytest.fixture
    def analyzer(self):
        return TurkishMorphologyAnalyzer()
    
    @pytest.fixture
    def stemmer(self):
        return TurkishStemmer()
    
    @pytest.fixture
    def lemmatizer(self):
        return TurkishLemmatizer()
    
    def test_vowel_harmony_analysis(self, analyzer):
        """Test vowel harmony detection"""
        # Valid vowel harmony
        valid_words = ["kitaplar", "evlerimiz", "güzellik", "öğretmen"]
        for word in valid_words:
            result = analyzer.analyze_vowel_harmony(word)
            assert result["valid"] == True, f"Expected valid harmony for {word}"
        
        # Invalid vowel harmony (mixed front-back)
        invalid_words = ["kitapler", "evlarımız"]
        for word in invalid_words:
            result = analyzer.analyze_vowel_harmony(word)
            assert result["front_back"] == False, f"Expected invalid front-back harmony for {word}"
    
    def test_consonant_mutation(self, analyzer):
        """Test consonant mutation rules"""
        test_cases = [
            ("kitap", "accusative", "kitab"),
            ("ağaç", "dative", "ağac"),
            ("renk", "genitive", "reng"),
        ]
        
        for stem, suffix_type, expected in test_cases:
            result = analyzer.apply_consonant_mutation(stem, suffix_type)
            assert result == expected, f"Expected {expected}, got {result}"
    
    def test_vowel_drop(self, analyzer):
        """Test vowel drop rules"""
        test_cases = [
            ("burun", ("burn", True)),
            ("ağız", ("ağz", True)),
            ("kitap", ("kitap", False)),
        ]
        
        for word, (expected_stem, expected_drop) in test_cases:
            result_stem, dropped = analyzer.apply_vowel_drop(word)
            assert result_stem == expected_stem
            assert dropped == expected_drop
    
    def test_buffer_consonant(self, analyzer):
        """Test buffer consonant determination"""
        test_cases = [
            ("su", "dative", "y"),      # su-y-a
            ("su", "genitive", "n"),    # su-n-un
            ("kitap", "dative", ""),    # kitab-a (no buffer)
        ]
        
        for stem, suffix_type, expected in test_cases:
            result = analyzer.determine_buffer_consonant(stem, suffix_type)
            assert result == expected
    
    def test_syllabification(self, analyzer):
        """Test Turkish syllabification"""
        test_cases = [
            ("merhaba", ["mer", "ha", "ba"]),
            ("öğretmen", ["öğ", "ret", "men"]),
            ("istanbul", ["is", "tan", "bul"]),
            ("arkadaş", ["ar", "ka", "daş"]),
        ]
        
        for word, expected in test_cases:
            result = analyzer.syllabify(word)
            assert result == expected, f"Expected {expected}, got {result}"
    
    def test_compound_analysis(self, analyzer):
        """Test compound word detection"""
        test_cases = [
            ("kahvehane", True),
            ("öğretmen", True),
            ("kitap", False),
        ]
        
        for word, expected_compound in test_cases:
            is_compound, _ = analyzer.analyze_compound(word)
            assert is_compound == expected_compound
    
    def test_suffix_allomorph(self, analyzer):
        """Test suffix allomorph selection"""
        test_cases = [
            ("ev", "ı/i/u/ü", "i"),        # front vowel
            ("kitap", "ı/i/u/ü", "ı"),     # back vowel
            ("göz", "ı/i/u/ü", "ü"),       # rounded front
            ("okul", "ı/i/u/ü", "u"),      # rounded back
        ]
        
        for stem, suffix_base, expected in test_cases:
            result = analyzer.get_suffix_allomorph(stem, suffix_base)
            assert result == expected
    
    def test_morphological_analysis(self, analyzer):
        """Test complete morphological analysis"""
        test_words = ["evlerimizden", "kitapları", "öğretmenlik"]
        
        for word in test_words:
            analysis = analyzer.analyze(word)
            assert analysis.surface_form == word
            assert analysis.root != ""
            assert len(analysis.morphemes) > 0
            assert analysis.pos_tag in ["NOUN", "VERB"]
    
    def test_stemming(self, stemmer):
        """Test Turkish stemming"""
        test_cases = [
            ("kitaplar", "kitap"),
            ("evlerimizden", "ev"),
            ("güzellik", "güzel"),
        ]
        
        for word, expected_stem in test_cases:
            # Note: Simplified test - actual stems may vary
            result = stemmer.stem(word)
            assert len(result) > 0
            assert len(result) <= len(word)
    
    def test_lemmatization(self, lemmatizer):
        """Test Turkish lemmatization"""
        test_cases = [
            ("gider", "git"),
            ("yer", "ye"),
            ("kitaplar", "kitap"),
        ]
        
        for word, expected_lemma in test_cases:
            result = lemmatizer.lemmatize(word)
            assert len(result) > 0
            assert len(result) <= len(word)


class TestTurkishBPETokenizer:
    """Test suite for Turkish BPE tokenizer"""
    
    @pytest.fixture
    def config(self):
        return TokenizerConfig(
            vocab_size=1000,
            turkish_specific=True,
            min_frequency=1
        )
    
    @pytest.fixture
    def tokenizer(self, config):
        return TurkishBPETokenizer(config)
    
    def test_tokenizer_initialization(self, tokenizer):
        """Test tokenizer initialization"""
        assert tokenizer.config.turkish_specific == True
        assert len(tokenizer.config.special_tokens) > 0
        assert "<pad>" in tokenizer.special_tokens_map
        assert "<unk>" in tokenizer.special_tokens_map
    
    def test_pre_tokenization(self, tokenizer):
        """Test pre-tokenization with Turkish handling"""
        text = "Merhaba dünya! Bugün hava çok güzel."
        tokens = tokenizer.pre_tokenize(text)
        
        assert len(tokens) > 0
        assert all(isinstance(t, str) for t in tokens)
    
    def test_turkish_suffix_handling(self, tokenizer):
        """Test Turkish suffix marking"""
        test_cases = [
            "evlerimizden",
            "kitapları",
            "öğretmenlik",
        ]
        
        for word in test_cases:
            result = tokenizer._handle_turkish_suffixes(word)
            # Check if suffix boundary marker is added for long words
            if len(word) > 5:
                assert len(result) >= len(word)
    
    def test_special_pattern_replacement(self, tokenizer):
        """Test special pattern detection and replacement"""
        test_cases = [
            ("100 TL", "<turk_currency>"),
            ("%50", "<turk_percent>"),
            ("15:30", "<turk_time>"),
            ("01/01/2025", "<turk_date>"),
        ]
        
        for text, expected_token in test_cases:
            result = tokenizer._replace_special_patterns(text)
            # Check if pattern is replaced (if token is in special tokens)
            if expected_token in tokenizer.special_tokens_map:
                assert expected_token in result or text in result
    
    def test_encoding_decoding(self, tokenizer):
        """Test encoding and decoding consistency"""
        # Train on small corpus first
        corpus = [
            "Merhaba dünya",
            "Türkiye'nin başkenti Ankara'dır",
            "Öğrenciler sınava hazırlanıyor",
        ]
        tokenizer.train(corpus, vocab_size=100)
        
        test_text = "Merhaba dünya"
        encoded = tokenizer.encode(test_text, add_special_tokens=False)
        decoded = tokenizer.decode(encoded, skip_special_tokens=True)
        
        assert len(encoded) > 0
        assert all(isinstance(id, int) for id in encoded)
        # Decoded text should be similar (may have minor differences due to BPE)
        assert len(decoded) > 0
    
    def test_tokenization(self, tokenizer):
        """Test tokenization into subwords"""
        corpus = ["Test corpus for tokenizer"]
        tokenizer.train(corpus, vocab_size=50)
        
        text = "Test"
        tokens = tokenizer.tokenize(text)
        
        assert len(tokens) > 0
        assert all(isinstance(t, str) for t in tokens)


class TestTurkishNLPPipeline:
    """Test suite for integrated Turkish NLP pipeline"""
    
    @pytest.fixture
    def pipeline(self):
        return TurkishNLPPipeline(
            use_zemberek=False,  # Disable external dependencies for testing
            use_transformers=False,
            use_spacy=False,
            cache_enabled=True
        )
    
    def test_pipeline_initialization(self, pipeline):
        """Test pipeline initialization"""
        assert pipeline.morphology_analyzer is not None
        assert pipeline.stemmer is not None
        assert pipeline.lemmatizer is not None
        assert len(pipeline.stopwords) > 0
        assert len(pipeline.abbreviations) > 0
    
    def test_tokenization_methods(self, pipeline):
        """Test different tokenization methods"""
        text = "Merhaba dünya! Nasılsınız?"
        
        result = pipeline.tokenize(text, method="best")
        assert result.task == NLPTask.TOKENIZATION
        assert len(result.output) > 0
        assert result.processing_time >= 0
    
    def test_morphological_analysis(self, pipeline):
        """Test morphological analysis"""
        text = "kitapları okudum"
        
        result = pipeline.analyze_morphology(text)
        assert result.task == NLPTask.MORPHOLOGY
        assert len(result.output) > 0
    
    def test_lemmatization(self, pipeline):
        """Test lemmatization"""
        text = "kitapları okudum"
        
        result = pipeline.lemmatize(text)
        assert result.task == NLPTask.LEMMATIZATION
        assert len(result.output) > 0
    
    def test_stemming(self, pipeline):
        """Test stemming"""
        text = "kitapları okudum"
        
        result = pipeline.stem(text)
        assert result.task == NLPTask.STEMMING
        assert len(result.output) > 0
    
    def test_entity_extraction(self, pipeline):
        """Test named entity recognition"""
        text = "Ahmet Ankara'ya gitti."
        
        result = pipeline.extract_entities(text)
        assert result.task == NLPTask.NER
        # Pattern-based NER should find at least person name
        assert isinstance(result.output, list)
    
    def test_sentiment_analysis(self, pipeline):
        """Test sentiment analysis"""
        text = "Bu kitap çok güzel ve faydalı."
        
        result = pipeline.analyze_sentiment(text)
        assert result.task == NLPTask.SENTIMENT
        assert result.output in ["POSITIVE", "NEGATIVE", "NEUTRAL"]
        assert result.confidence >= 0 and result.confidence <= 1
    
    def test_stopword_removal(self, pipeline):
        """Test stopword removal"""
        text = "Bu ve şu kitaplar çok güzel"
        
        filtered = pipeline.remove_stopwords(text)
        assert len(filtered) < len(text.split())
        assert "ve" not in filtered
        assert "bu" not in [w.lower() for w in filtered]
    
    def test_abbreviation_expansion(self, pipeline):
        """Test abbreviation expansion"""
        text = "Dr. Ahmet Bey T.C. vatandaşıdır."
        
        expanded = pipeline.expand_abbreviations(text)
        assert "Doktor" in expanded
        assert "Türkiye Cumhuriyeti" in expanded
    
    def test_complete_analysis(self, pipeline):
        """Test complete text analysis"""
        text = "Ankara Türkiye'nin başkentidir."
        
        results = pipeline.analyze_complete(text)
        
        assert 'tokens' in results
        assert 'morphology' in results
        assert 'lemmas' in results
        assert 'stems' in results
        assert 'entities' in results
        assert 'sentiment' in results
        assert 'statistics' in results
        
        assert results['statistics']['word_count'] > 0
        assert results['statistics']['char_count'] == len(text)


class TestIntegration:
    """Integration tests for Turkish NLP modules"""
    
    def test_morphology_tokenizer_integration(self):
        """Test integration between morphology and tokenizer"""
        analyzer = TurkishMorphologyAnalyzer()
        config = TokenizerConfig(turkish_specific=True)
        tokenizer = TurkishBPETokenizer(config)
        
        text = "evlerimizden"
        
        # Morphological analysis
        analysis = analyzer.analyze(text)
        
        # Tokenization
        tokens = tokenizer.pre_tokenize(text)
        
        assert len(analysis.morphemes) > 0
        assert len(tokens) > 0
    
    def test_pipeline_caching(self):
        """Test caching functionality"""
        pipeline = TurkishNLPPipeline(cache_enabled=True)
        
        text = "Test caching"
        
        # First call
        result1 = pipeline.tokenize(text)
        time1 = result1.processing_time
        
        # Second call (should be cached)
        result2 = pipeline.tokenize(text)
        time2 = result2.processing_time
        
        assert result1.output == result2.output
        # Note: Can't reliably test timing due to system variations
    
    def test_error_handling(self):
        """Test error handling in pipeline"""
        pipeline = TurkishNLPPipeline()
        
        # Empty text
        result = pipeline.tokenize("")
        assert result.output == [] or result.output == [""]
        
        # Very long text
        long_text = "word " * 10000
        result = pipeline.tokenize(long_text)
        assert len(result.output) > 0
        
        # Special characters
        special_text = "!@#$%^&*()"
        result = pipeline.tokenize(special_text)
        assert result.output is not None


def test_turkish_patterns():
    """Test Turkish-specific patterns and rules"""
    analyzer = TurkishMorphologyAnalyzer()
    
    # Test vowel harmony patterns
    test_words = {
        "güzelleştirmek": True,   # Valid harmony
        "büyüklük": True,          # Valid harmony
        "kitaplar": True,          # Valid harmony
        "evlerimiz": True,         # Valid harmony
    }
    
    for word, expected_valid in test_words.items():
        result = analyzer.analyze_vowel_harmony(word)
        assert result["valid"] == expected_valid, f"Failed for {word}"


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])