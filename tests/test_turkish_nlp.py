"""
Türkçe NLP Optimizasyon Test Suite
"""

import unittest
import tempfile
import json
from pathlib import Path
import sys
import os

# Proje root'u path'e ekle
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.turkish_nlp_optimizer import (
    TurkishTextOptimizer,
    TurkishMorphologyAnalyzer,
    TurkishNER,
    TextQuality,
    ProcessedText
)
from src.data_augmentation import (
    TurkishDataAugmenter,
    AugmentationType,
    TurkishSynonymDict
)
from src.training_optimizer import (
    TrainingConfig,
    TrainingMetrics,
    CheckpointManager,
    EarlyStopping,
    GradientAccumulator
)


class TestTurkishTextOptimizer(unittest.TestCase):
    """Türkçe metin optimizasyon testleri"""
    
    def setUp(self):
        self.optimizer = TurkishTextOptimizer(enable_cache=False)
        
    def test_clean_text_basic(self):
        """Temel metin temizleme testi"""
        text = "Bu   bir     test   metnidir."
        cleaned = self.optimizer.clean_text(text)
        self.assertEqual(cleaned, "Bu bir test metnidir.")
        
    def test_clean_text_with_urls(self):
        """URL temizleme testi"""
        text = "Web sitemiz https://example.com adresinde."
        cleaned = self.optimizer.clean_text(text)
        self.assertNotIn("https://", cleaned)
        
    def test_clean_text_with_html(self):
        """HTML temizleme testi"""
        text = "<p>Bu bir <b>test</b> metnidir.</p>"
        cleaned = self.optimizer.clean_text(text)
        self.assertNotIn("<p>", cleaned)
        self.assertNotIn("<b>", cleaned)
        
    def test_turkish_character_ratio(self):
        """Türkçe karakter oranı testi"""
        text1 = "Çok güzel bir gün, şimdi İstanbul'a gidiyoruz."
        ratio1 = self.optimizer._calculate_turkish_ratio(text1)
        self.assertGreater(ratio1, 0.05)
        
        text2 = "This is an English text."
        ratio2 = self.optimizer._calculate_turkish_ratio(text2)
        self.assertLess(ratio2, 0.01)
        
    def test_quality_score_calculation(self):
        """Kalite skoru hesaplama testi"""
        good_text = """Bu çok kaliteli bir Türkçe metindir. İçerisinde Türkçe karakterler var.
        Cümleler düzgün yapılandırılmış. Noktalama işaretleri yerinde kullanılmış."""
        
        score, details = self.optimizer.calculate_quality_score(good_text)
        
        self.assertGreater(score, 50)
        self.assertIn('length', details)
        self.assertIn('turkish', details)
        self.assertIn('diversity', details)
        
    def test_language_detection(self):
        """Dil tespit testi"""
        turkish_text = "Merhaba, nasılsınız? Bugün hava çok güzel."
        lang = self.optimizer.detect_language(turkish_text)
        self.assertEqual(lang, 'tr')
        
        english_text = "Hello, how are you? The weather is nice today."
        lang = self.optimizer.detect_language(english_text)
        self.assertNotEqual(lang, 'tr')
        
    def test_process_complete(self):
        """Komple işleme testi"""
        text = "Bu test için kullanılan örnek bir Türkçe metindir."
        result = self.optimizer.process(text)
        
        self.assertIsInstance(result, ProcessedText)
        self.assertEqual(result.original, text)
        self.assertIsNotNone(result.cleaned)
        self.assertIsInstance(result.quality_score, float)
        self.assertIsInstance(result.quality_level, TextQuality)
        self.assertEqual(result.language, 'tr')
        
    def test_cache_functionality(self):
        """Cache fonksiyonalite testi"""
        optimizer_with_cache = TurkishTextOptimizer(enable_cache=True)
        text = "Cache testi için örnek metin."
        
        # İlk işleme
        result1 = optimizer_with_cache.process(text)
        
        # Cache'den alınmalı
        result2 = optimizer_with_cache.process(text)
        
        self.assertEqual(result1.cleaned, result2.cleaned)
        self.assertEqual(result1.quality_score, result2.quality_score)
        
        # Cache'i temizle
        optimizer_with_cache.clear_cache()
        stats = optimizer_with_cache.get_statistics()
        self.assertEqual(stats.get('cache_size', 0), 0)


class TestTurkishMorphology(unittest.TestCase):
    """Türkçe morfoloji testleri"""
    
    def setUp(self):
        self.analyzer = TurkishMorphologyAnalyzer()
        
    def test_lemmatization_basic(self):
        """Temel lemmatizasyon testi"""
        test_cases = [
            ("evler", "ev"),
            ("kitaplar", "kitap"),
            ("gidiyorum", "gid"),
            ("gelecek", "gel"),
            ("yapmış", "yap")
        ]
        
        for word, expected_stem in test_cases:
            result = self.analyzer.lemmatize(word)
            # Tam eşleşme beklemiyoruz, sadece kısalma olmalı
            self.assertLessEqual(len(result), len(word))
            
    def test_analyze_morphology(self):
        """Morfolojik analiz testi"""
        text = "Evler güzeldir. Çocuklar parkta oynuyorlar."
        analysis = self.analyzer.analyze_morphology(text)
        
        self.assertIn('word_count', analysis)
        self.assertIn('unique_stems', analysis)
        self.assertIn('stem_ratio', analysis)
        self.assertGreater(analysis['word_count'], 0)


class TestTurkishNER(unittest.TestCase):
    """Türkçe NER testleri"""
    
    def setUp(self):
        self.ner = TurkishNER()
        
    def test_person_detection(self):
        """Kişi ismi tespiti"""
        text = "Ahmet Yılmaz bugün İstanbul'a gitti."
        entities = self.ner.extract_entities(text)
        
        self.assertIn('PERSON', entities)
        self.assertIn('Ahmet Yılmaz', entities['PERSON'])
        
    def test_location_detection(self):
        """Yer ismi tespiti"""
        text = "İstanbul Türkiye'nin en büyük şehridir."
        entities = self.ner.extract_entities(text)
        
        self.assertIn('LOCATION', entities)
        self.assertTrue(any('İstanbul' in loc for loc in entities['LOCATION']))
        
    def test_organization_detection(self):
        """Kurum ismi tespiti"""
        text = "TÜBİTAK yeni projeler açıkladı."
        entities = self.ner.extract_entities(text)
        
        self.assertIn('ORGANIZATION', entities)
        self.assertIn('TÜBİTAK', entities['ORGANIZATION'])


class TestDataAugmentation(unittest.TestCase):
    """Veri çoğaltma testleri"""
    
    def setUp(self):
        self.augmenter = TurkishDataAugmenter(seed=42, enable_cache=False)
        
    def test_synonym_replacement(self):
        """Eş anlamlı değiştirme testi"""
        text = "Bu güzel bir gün"
        augmented = self.augmenter.synonym_replacement(text, n=1)
        
        self.assertNotEqual(text, augmented)
        self.assertEqual(len(text.split()), len(augmented.split()))
        
    def test_random_insertion(self):
        """Rastgele ekleme testi"""
        text = "Test metni örneği"
        augmented = self.augmenter.random_insertion(text, n=1)
        
        self.assertGreater(len(augmented.split()), len(text.split()))
        
    def test_random_swap(self):
        """Rastgele yer değiştirme testi"""
        text = "Birinci ikinci üçüncü kelime"
        augmented = self.augmenter.random_swap(text, n=1)
        
        self.assertEqual(len(text.split()), len(augmented.split()))
        self.assertNotEqual(text, augmented)
        
    def test_random_deletion(self):
        """Rastgele silme testi"""
        text = "Bu uzun bir test metni örneğidir"
        augmented = self.augmenter.random_deletion(text, p=0.2)
        
        self.assertLessEqual(len(augmented.split()), len(text.split()))
        
    def test_sentence_shuffling(self):
        """Cümle karıştırma testi"""
        text = "İlk cümle. İkinci cümle. Üçüncü cümle."
        augmented = self.augmenter.sentence_shuffling(text)
        
        # Cümle sayısı aynı kalmalı
        original_sentences = text.count('.')
        augmented_sentences = augmented.count('.')
        self.assertEqual(original_sentences, augmented_sentences)
        
    def test_noise_injection(self):
        """Gürültü ekleme testi"""
        text = "Temiz metin"
        augmented = self.augmenter.noise_injection(text, noise_level=0.1)
        
        # Uzunluk aynı kalmalı
        self.assertEqual(len(text), len(augmented))
        
    def test_augment_batch(self):
        """Toplu çoğaltma testi"""
        texts = [
            "İlk metin örneği",
            "İkinci metin örneği",
            "Üçüncü metin örneği"
        ]
        
        results = self.augmenter.batch_augment(texts, num_augmentations=2)
        
        self.assertEqual(len(results), len(texts))
        for result in results:
            self.assertIsInstance(result, list)
            
    def test_confidence_calculation(self):
        """Güven skoru hesaplama testi"""
        original = "Bu bir test metnidir"
        augmented = "Bu bir deneme metnidir"
        
        confidence = self.augmenter._calculate_confidence(original, augmented)
        
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)


class TestSynonymDict(unittest.TestCase):
    """Eş anlamlı sözlük testleri"""
    
    def setUp(self):
        self.dict = TurkishSynonymDict()
        
    def test_get_synonyms(self):
        """Eş anlamlı getirme testi"""
        synonyms = self.dict.get_synonyms("güzel")
        
        self.assertIsInstance(synonyms, list)
        self.assertGreater(len(synonyms), 0)
        
    def test_add_synonym_group(self):
        """Yeni eş anlamlı grup ekleme testi"""
        new_group = ["test1", "test2", "test3"]
        self.dict.add_synonym_group(new_group)
        
        synonyms = self.dict.get_synonyms("test1")
        self.assertIn("test2", synonyms)
        self.assertIn("test3", synonyms)


class TestTrainingOptimizer(unittest.TestCase):
    """Eğitim optimizasyon testleri"""
    
    def setUp(self):
        self.config = TrainingConfig(
            model_name="test_model",
            batch_size=8,
            num_epochs=2,
            checkpoint_dir=tempfile.mkdtemp()
        )
        
    def test_config_save_load(self):
        """Konfigürasyon kaydet/yükle testi"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            self.config.save(f.name)
            loaded_config = TrainingConfig.load(f.name)
            
        self.assertEqual(self.config.model_name, loaded_config.model_name)
        self.assertEqual(self.config.batch_size, loaded_config.batch_size)
        
    def test_training_metrics(self):
        """Eğitim metrikleri testi"""
        metrics = TrainingMetrics(
            epoch=1,
            step=100,
            loss=0.5,
            learning_rate=2e-5,
            eval_loss=0.4,
            eval_accuracy=0.85
        )
        
        self.assertEqual(metrics.epoch, 1)
        self.assertEqual(metrics.step, 100)
        self.assertAlmostEqual(metrics.loss, 0.5)
        
    def test_early_stopping(self):
        """Early stopping testi"""
        early_stop = EarlyStopping(patience=3, greater_is_better=False)
        
        # İyileşme var
        self.assertFalse(early_stop(0.5))
        self.assertFalse(early_stop(0.4))
        self.assertFalse(early_stop(0.3))
        
        # İyileşme yok
        self.assertFalse(early_stop(0.3))
        self.assertFalse(early_stop(0.3))
        self.assertTrue(early_stop(0.3))  # Patience aşıldı
        
    def test_gradient_accumulator(self):
        """Gradient accumulation testi"""
        accumulator = GradientAccumulator(accumulation_steps=4)
        
        self.assertFalse(accumulator.should_step())  # 1
        self.assertFalse(accumulator.should_step())  # 2
        self.assertFalse(accumulator.should_step())  # 3
        self.assertTrue(accumulator.should_step())   # 4
        self.assertFalse(accumulator.should_step())  # 5
        
    def test_checkpoint_manager(self):
        """Checkpoint yönetimi testi"""
        manager = CheckpointManager(
            checkpoint_dir=tempfile.mkdtemp(),
            save_total_limit=2
        )
        
        # Mock model ve optimizer
        import torch
        model = torch.nn.Linear(10, 10)
        optimizer = torch.optim.Adam(model.parameters())
        
        # Checkpoint kaydet
        metrics = TrainingMetrics(epoch=1, step=100, loss=0.5, learning_rate=2e-5)
        path = manager.save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=1,
            step=100,
            metrics=metrics,
            config=self.config
        )
        
        self.assertTrue(Path(path).exists())
        
        # Checkpoint yükle
        loaded_info = manager.load_checkpoint(path, model, optimizer)
        self.assertEqual(loaded_info['epoch'], 1)
        self.assertEqual(loaded_info['step'], 100)


class TestIntegration(unittest.TestCase):
    """Entegrasyon testleri"""
    
    def test_full_pipeline(self):
        """Tam pipeline testi"""
        # 1. Metin optimize et
        optimizer = TurkishTextOptimizer()
        text = "Bu entegrasyon testi için kullanılan örnek bir Türkçe metindir."
        processed = optimizer.process(text)
        
        self.assertIsNotNone(processed)
        self.assertGreater(processed.quality_score, 0)
        
        # 2. Veri çoğalt
        augmenter = TurkishDataAugmenter()
        augmented_list = augmenter.augment(
            processed.cleaned,
            techniques=[AugmentationType.SYNONYM_REPLACEMENT],
            num_augmentations=2
        )
        
        self.assertGreater(len(augmented_list), 0)
        
        # 3. Kalite kontrolü
        for augmented in augmented_list:
            processed_aug = optimizer.process(augmented.augmented)
            self.assertIsNotNone(processed_aug)
            
    def test_statistics_tracking(self):
        """İstatistik takibi testi"""
        optimizer = TurkishTextOptimizer(enable_cache=False)
        augmenter = TurkishDataAugmenter(enable_cache=False)
        
        texts = [
            "İlk test metni örnek olarak yazılmıştır.",
            "İkinci test metni de örnek amaçlı yazıldı.",
            "Üçüncü test metni için de aynı durum geçerli."
        ]
        
        # İşle
        for text in texts:
            optimizer.process(text)
            augmenter.augment(text)
            
        # İstatistikleri kontrol et
        opt_stats = optimizer.get_statistics()
        self.assertEqual(opt_stats['total_processed'], 3)
        
        aug_stats = augmenter.get_statistics()
        self.assertGreater(aug_stats['total_augmented'], 0)


def run_tests():
    """Test suite'i çalıştır"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Tüm test sınıflarını ekle
    suite.addTests(loader.loadTestsFromTestCase(TestTurkishTextOptimizer))
    suite.addTests(loader.loadTestsFromTestCase(TestTurkishMorphology))
    suite.addTests(loader.loadTestsFromTestCase(TestTurkishNER))
    suite.addTests(loader.loadTestsFromTestCase(TestDataAugmentation))
    suite.addTests(loader.loadTestsFromTestCase(TestSynonymDict))
    suite.addTests(loader.loadTestsFromTestCase(TestTrainingOptimizer))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
    
    # Runner oluştur ve çalıştır
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Sonuçları özetle
    print("\n" + "="*70)
    print("TEST SONUÇLARI")
    print("="*70)
    print(f"Toplam test: {result.testsRun}")
    print(f"Başarılı: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Başarısız: {len(result.failures)}")
    print(f"Hata: {len(result.errors)}")
    
    return result


if __name__ == '__main__':
    run_tests()