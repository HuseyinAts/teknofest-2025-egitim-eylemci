"""
Turkish NLP Demo - Quick Start
TEKNOFEST 2025 - Türkçe NLP Modüllerini Test Et
"""

import sys
import os
from pathlib import Path

# Set UTF-8 encoding for Windows
if sys.platform == 'win32':
    os.environ['PYTHONIOENCODING'] = 'utf-8'

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from src.nlp.advanced_turkish_morphology import (
    TurkishMorphologyAnalyzer, 
    TurkishStemmer, 
    TurkishLemmatizer
)
from src.nlp.turkish_bpe_tokenizer import TurkishBPETokenizer, TokenizerConfig
from src.nlp.turkish_nlp_integration import TurkishNLPPipeline

def demo_morphology():
    """Morfoloji analizi demo"""
    print("\n" + "="*60)
    print("TURKCE MORFOLOJI ANALIZI")
    print("="*60)
    
    analyzer = TurkishMorphologyAnalyzer()
    stemmer = TurkishStemmer()
    lemmatizer = TurkishLemmatizer()
    
    test_words = [
        "evlerimizden",
        "kitapları",
        "öğretmenlik",
        "güzelleştirmek",
        "Türkiye'nin"
    ]
    
    for word in test_words:
        print(f"\nKelime: {word}")
        print("-" * 40)
        
        # Morfolojik analiz
        analysis = analyzer.analyze(word)
        print(f"  Kok: {analysis.root}")
        print(f"  Sozcuk Turu: {analysis.pos_tag}")
        print(f"  Ekler: {[m.surface for m in analysis.morphemes]}")
        
        # Ünlü uyumu
        harmony = analyzer.analyze_vowel_harmony(word)
        print(f"  Unlu Uyumu: {'Gecerli' if harmony['valid'] else 'Gecersiz'}")
        if harmony['valid']:
            print(f"    - Kalinlik-Incelik: {'OK' if harmony['front_back'] else 'X'}")
            print(f"    - Duzluk-Yuvarliklik: {'OK' if harmony['rounded_unrounded'] else 'X'}")
        
        # Heceleme
        syllables = analyzer.syllabify(word)
        print(f"  Heceler: {'-'.join(syllables)}")
        
        # Kök ve lemma
        stem = stemmer.stem(word)
        lemma = lemmatizer.lemmatize(word)
        print(f"  Kök (Stem): {stem}")
        print(f"  Lemma: {lemma}")

def demo_tokenizer():
    """BPE Tokenizer demo"""
    print("\n" + "="*60)
    print("🔤 TÜRKÇE BPE TOKENIZER")
    print("="*60)
    
    # Tokenizer yapılandırması
    config = TokenizerConfig(
        vocab_size=1000,
        turkish_specific=True,
        min_frequency=1
    )
    
    tokenizer = TurkishBPETokenizer(config)
    
    # Mini corpus ile eğit
    corpus = [
        "Merhaba dünya! Bugün hava çok güzel.",
        "Türkiye'nin başkenti Ankara'dır.",
        "İstanbul Boğazı'nda vapurla gezinti yapmak çok keyifli.",
        "Öğrenciler sınavlara hazırlanıyor.",
        "Yapay zeka teknolojileri hızla gelişiyor.",
        "TEKNOFEST 2025 yarışmasına hazırlanıyoruz.",
        "Türkçe doğal dil işleme çalışmaları önem kazanıyor.",
        "Eğitim teknolojileri öğrenmeyi kolaylaştırıyor.",
    ]
    
    print("\n📊 Tokenizer Eğitimi Başlıyor...")
    tokenizer.train(corpus, vocab_size=500)
    
    # Test metinleri
    test_texts = [
        "TEKNOFEST 2025 için hazırız!",
        "Türkçe NLP çok güzel çalışıyor.",
        "15:30'da %50 indirim var.",
        "Dr. Ahmet Bey 100 TL ödedi."
    ]
    
    for text in test_texts:
        print(f"\n📝 Metin: {text}")
        print("-" * 40)
        
        # Tokenize et
        tokens = tokenizer.tokenize(text)
        print(f"  Tokenlar: {tokens}")
        
        # Encode/Decode
        token_ids = tokenizer.encode(text, add_special_tokens=True)
        print(f"  Token ID'ler: {token_ids[:10]}{'...' if len(token_ids) > 10 else ''}")
        
        decoded = tokenizer.decode(token_ids, skip_special_tokens=True)
        print(f"  Decoded: {decoded}")

def demo_pipeline():
    """Entegre pipeline demo"""
    print("\n" + "="*60)
    print("🚀 TÜRKÇE NLP PIPELINE - KOMPLE ANALİZ")
    print("="*60)
    
    # Pipeline başlat (harici bağımlılıklar olmadan)
    pipeline = TurkishNLPPipeline(
        use_zemberek=False,      # Zemberek kullanma (Java gerektirir)
        use_transformers=False,  # Transformer modelleri kullanma (büyük indirme gerektirir)
        use_spacy=False,        # spaCy kullanma (ek kurulum gerektirir)
        cache_enabled=True
    )
    
    # Test metinleri
    test_texts = [
        "Ankara Türkiye'nin başkentidir.",
        "Öğrenciler yarın sınava girecekler.",
        "Bu kitap çok güzel ve faydalı.",
        "İstanbul Boğazı'nda vapurla gezinti yaptık.",
        "TEKNOFEST 2025 yarışması için hazırlanıyoruz.",
        "Prof. Dr. Ahmet Yılmaz Ankara Üniversitesi'nde çalışıyor."
    ]
    
    for text in test_texts:
        print(f"\n📝 Metin: {text}")
        print("=" * 50)
        
        # Komple analiz
        results = pipeline.analyze_complete(text)
        
        print("\n🔤 Tokenlar:")
        print(f"  {results['tokens']}")
        
        print("\n📚 Lemma'lar:")
        print(f"  {results['lemmas']}")
        
        print("\n🌱 Kökler:")
        print(f"  {results['stems']}")
        
        print("\n👤 Varlıklar (Named Entities):")
        if results['entities']:
            for entity in results['entities']:
                print(f"  - {entity.get('text', 'N/A')} [{entity.get('type', 'N/A')}]")
        else:
            print("  (Varlık bulunamadı)")
        
        print("\n😊 Duygu Analizi:")
        sentiment = results['sentiment']
        print(f"  Duygu: {sentiment['label']}")
        print(f"  Güven: {sentiment['confidence']:.2%}")
        
        print("\n📊 İstatistikler:")
        stats = results['statistics']
        print(f"  Karakter: {stats['char_count']}")
        print(f"  Kelime: {stats['word_count']}")
        print(f"  Cümle: {stats['sentence_count']}")
        print(f"  Stopword: {stats['stopword_count']}")

def demo_special_features():
    """Özel özellikler demo"""
    print("\n" + "="*60)
    print("✨ ÖZEL ÖZELLİKLER")
    print("="*60)
    
    analyzer = TurkishMorphologyAnalyzer()
    pipeline = TurkishNLPPipeline(
        use_zemberek=False,
        use_transformers=False,
        use_spacy=False
    )
    
    print("\n1️⃣ ÜNSÜZ DEĞİŞİMİ (Consonant Mutation)")
    print("-" * 40)
    mutations = [
        ("kitap", "kitabı"),
        ("ağaç", "ağacı"),
        ("renk", "rengi"),
        ("köpek", "köpeği")
    ]
    
    for base, expected in mutations:
        mutated = analyzer.apply_consonant_mutation(base, "accusative")
        print(f"  {base} → {mutated} (beklenen: {expected})")
    
    print("\n2️⃣ ÜNLÜ DÜŞMESİ (Vowel Drop)")
    print("-" * 40)
    drops = [
        "burun", "ağız", "boyun", "oğul"
    ]
    
    for word in drops:
        dropped, did_drop = analyzer.apply_vowel_drop(word)
        if did_drop:
            print(f"  {word} → {dropped} ✅")
        else:
            print(f"  {word} (değişmedi)")
    
    print("\n3️⃣ STOPWORD TEMİZLEME")
    print("-" * 40)
    text = "Bu ve şu kitaplar çok güzel ama pahalı"
    filtered = pipeline.remove_stopwords(text)
    print(f"  Orijinal: {text}")
    print(f"  Temizlenmiş: {filtered}")
    
    print("\n4️⃣ KISALTMA AÇMA")
    print("-" * 40)
    text = "Dr. Ahmet Bey T.C. vatandaşı, Prof. Ayşe Hanım'la çalışıyor."
    expanded = pipeline.expand_abbreviations(text)
    print(f"  Orijinal: {text}")
    print(f"  Açılmış: {expanded}")
    
    print("\n5️⃣ BİRLEŞİK KELİME ANALİZİ")
    print("-" * 40)
    compounds = ["kahvehane", "başöğretmen", "hanımeli", "bilgisayar"]
    
    for word in compounds:
        is_compound, parts = analyzer.analyze_compound(word)
        if is_compound:
            print(f"  {word} → {' + '.join(parts)} ✅")
        else:
            print(f"  {word} (birleşik değil)")

def main():
    """Ana demo fonksiyonu"""
    print("\n" + "=" * 60)
    print("   TEKNOFEST 2025 - TÜRKÇE NLP DEMO   ")
    print("=" * 60)
    
    try:
        # Morfoloji demo
        demo_morphology()
        
        # Tokenizer demo
        demo_tokenizer()
        
        # Pipeline demo
        demo_pipeline()
        
        # Özel özellikler
        demo_special_features()
        
        print("\n" + "="*60)
        print("✅ TÜM TESTLER BAŞARIYLA TAMAMLANDI!")
        print("="*60)
        
        print("\n📌 Sonraki adımlar:")
        print("  1. Zemberek-NLP kurulumu için: pip install JPype1")
        print("  2. Transformer modelleri için: pip install transformers torch")
        print("  3. Daha büyük corpus ile tokenizer eğitimi")
        print("  4. Production için API entegrasyonu")
        
    except Exception as e:
        print(f"\n❌ Hata oluştu: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()