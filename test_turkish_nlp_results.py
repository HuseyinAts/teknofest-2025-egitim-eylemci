# -*- coding: utf-8 -*-
"""
Turkish NLP Test with File Output
TEKNOFEST 2025
"""

import sys
from pathlib import Path
import json

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from src.nlp.advanced_turkish_morphology import TurkishMorphologyAnalyzer
from src.nlp.turkish_bpe_tokenizer import TurkishBPETokenizer, TokenizerConfig
from src.nlp.turkish_nlp_integration import TurkishNLPPipeline

def run_tests():
    results = {}
    
    # Test 1: Morphology
    print("Testing morphology...")
    analyzer = TurkishMorphologyAnalyzer()
    
    morph_results = []
    test_words = ["evlerimizden", "kitapları", "öğretmenlik", "güzelleştirmek"]
    
    for word in test_words:
        analysis = analyzer.analyze(word)
        harmony = analyzer.analyze_vowel_harmony(word)
        syllables = analyzer.syllabify(word)
        
        morph_results.append({
            "word": word,
            "root": analysis.root,
            "pos": analysis.pos_tag,
            "morphemes": [m.surface for m in analysis.morphemes],
            "vowel_harmony": harmony["valid"],
            "syllables": syllables
        })
    
    results["morphology"] = morph_results
    
    # Test 2: Tokenizer
    print("Testing tokenizer...")
    config = TokenizerConfig(vocab_size=500, turkish_specific=True, min_frequency=1)
    tokenizer = TurkishBPETokenizer(config)
    
    corpus = [
        "Merhaba dünya",
        "Türkiye'nin başkenti Ankara'dır",
        "Öğrenciler sınava hazırlanıyor",
        "TEKNOFEST 2025 yarışması"
    ]
    
    tokenizer.train(corpus, vocab_size=200)
    
    tokenizer_results = []
    test_texts = ["TEKNOFEST 2025", "Türkçe NLP", "Eğitim teknolojileri"]
    
    for text in test_texts:
        tokens = tokenizer.tokenize(text)
        tokenizer_results.append({
            "text": text,
            "tokens": [str(t) for t in tokens]  # Convert to string to avoid encoding issues
        })
    
    results["tokenizer"] = tokenizer_results
    
    # Test 3: Pipeline
    print("Testing pipeline...")
    pipeline = TurkishNLPPipeline(
        use_zemberek=False,
        use_transformers=False,
        use_spacy=False
    )
    
    pipeline_results = []
    test_sentences = [
        "Ankara Türkiye'nin başkentidir.",
        "Öğrenciler yarın sınava girecekler.",
        "Bu kitap çok güzel ve faydalı."
    ]
    
    for sentence in test_sentences:
        analysis = pipeline.analyze_complete(sentence)
        
        pipeline_results.append({
            "text": sentence,
            "tokens": analysis["tokens"],
            "lemmas": analysis["lemmas"],
            "stems": analysis["stems"],
            "sentiment": {
                "label": analysis["sentiment"]["label"],
                "confidence": round(analysis["sentiment"]["confidence"], 3)
            },
            "word_count": analysis["statistics"]["word_count"],
            "entities": len(analysis["entities"])
        })
    
    results["pipeline"] = pipeline_results
    
    # Test 4: Special Features
    print("Testing special features...")
    special_results = {}
    
    # Consonant mutation
    mutations = []
    test_mutations = [("kitap", "kitab"), ("ağaç", "ağac"), ("renk", "reng")]
    for base, _ in test_mutations:
        mutated = analyzer.apply_consonant_mutation(base, "accusative")
        mutations.append({"base": base, "mutated": mutated})
    special_results["consonant_mutations"] = mutations
    
    # Vowel drop
    drops = []
    test_drops = ["burun", "ağız", "boyun"]
    for word in test_drops:
        dropped, did_drop = analyzer.apply_vowel_drop(word)
        drops.append({"word": word, "dropped": dropped, "did_drop": did_drop})
    special_results["vowel_drops"] = drops
    
    # Stopword removal
    text = "Bu ve şu kitaplar çok güzel"
    filtered = pipeline.remove_stopwords(text)
    special_results["stopword_removal"] = {
        "original": text,
        "filtered": filtered
    }
    
    results["special_features"] = special_results
    
    return results

def main():
    print("\nTEKNOFEST 2025 - Turkish NLP Module Test")
    print("="*60)
    print("Running tests...")
    
    try:
        results = run_tests()
        
        # Save results to file
        output_file = "turkish_nlp_test_results.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"\nTest completed successfully!")
        print(f"Results saved to: {output_file}")
        
        # Print summary
        print("\n" + "="*60)
        print("TEST SUMMARY:")
        print("="*60)
        print(f"Morphology tests: {len(results['morphology'])} words analyzed")
        print(f"Tokenizer tests: {len(results['tokenizer'])} texts tokenized")
        print(f"Pipeline tests: {len(results['pipeline'])} sentences analyzed")
        print(f"Special features: {len(results['special_features'])} features tested")
        
        # Display some results safely
        print("\nSample Results:")
        print("-"*40)
        
        # Show first morphology result
        if results['morphology']:
            first = results['morphology'][0]
            print(f"Morphology: {first['word']} -> Root: {first['root']}, POS: {first['pos']}")
        
        # Show first pipeline result
        if results['pipeline']:
            first = results['pipeline'][0]
            print(f"Pipeline: {first['text'][:30]}...")
            print(f"  Sentiment: {first['sentiment']['label']} ({first['sentiment']['confidence']:.1%})")
            print(f"  Words: {first['word_count']}, Entities: {first['entities']}")
        
        print("\nCheck turkish_nlp_test_results.json for complete results!")
        
    except Exception as e:
        print(f"\nError occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()