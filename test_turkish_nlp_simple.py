# -*- coding: utf-8 -*-
"""
Turkish NLP Demo - Simple Version
TEKNOFEST 2025
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from src.nlp.advanced_turkish_morphology import TurkishMorphologyAnalyzer
from src.nlp.turkish_bpe_tokenizer import TurkishBPETokenizer, TokenizerConfig
from src.nlp.turkish_nlp_integration import TurkishNLPPipeline

def test_morphology():
    print("\n" + "="*60)
    print("TURKISH MORPHOLOGY ANALYSIS TEST")
    print("="*60)
    
    analyzer = TurkishMorphologyAnalyzer()
    
    test_words = ["evlerimizden", "kitaplari", "ogretmenlik"]
    
    for word in test_words:
        print(f"\nWord: {word}")
        analysis = analyzer.analyze(word)
        print(f"  Root: {analysis.root}")
        print(f"  POS: {analysis.pos_tag}")
        
        harmony = analyzer.analyze_vowel_harmony(word)
        print(f"  Vowel Harmony: {'Valid' if harmony['valid'] else 'Invalid'}")
        
        syllables = analyzer.syllabify(word)
        print(f"  Syllables: {'-'.join(syllables)}")

def test_tokenizer():
    print("\n" + "="*60)
    print("TURKISH BPE TOKENIZER TEST")
    print("="*60)
    
    config = TokenizerConfig(vocab_size=500, turkish_specific=True, min_frequency=1)
    tokenizer = TurkishBPETokenizer(config)
    
    corpus = [
        "Merhaba dunya",
        "Turkiye'nin baskenti Ankara'dir",
        "Ogrenciler sinava hazirlaniyor"
    ]
    
    print("\nTraining tokenizer...")
    tokenizer.train(corpus, vocab_size=200)
    
    test_text = "TEKNOFEST 2025"
    tokens = tokenizer.tokenize(test_text)
    print(f"\nText: {test_text}")
    print(f"Tokens: {tokens}")

def test_pipeline():
    print("\n" + "="*60)
    print("TURKISH NLP PIPELINE TEST")
    print("="*60)
    
    pipeline = TurkishNLPPipeline(
        use_zemberek=False,
        use_transformers=False,
        use_spacy=False
    )
    
    text = "Ankara Turkiye'nin baskentidir"
    
    print(f"\nAnalyzing: {text}")
    results = pipeline.analyze_complete(text)
    
    print(f"Tokens: {results['tokens']}")
    print(f"Lemmas: {results['lemmas']}")
    print(f"Stems: {results['stems']}")
    print(f"Sentiment: {results['sentiment']['label']} ({results['sentiment']['confidence']:.2%})")
    print(f"Stats: Words={results['statistics']['word_count']}, Chars={results['statistics']['char_count']}")

if __name__ == "__main__":
    print("\nTEKNOFEST 2025 - TURKISH NLP MODULE TEST")
    print("="*60)
    
    try:
        test_morphology()
        test_tokenizer()
        test_pipeline()
        
        print("\n" + "="*60)
        print("ALL TESTS COMPLETED SUCCESSFULLY!")
        print("="*60)
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()