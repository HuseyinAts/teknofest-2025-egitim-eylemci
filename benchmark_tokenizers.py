"""
Tokenizer Benchmark Script
Qwen3 vs Turkish Mixtral performans karşılaştırması
"""

import time
import psutil
import os
from typing import List, Dict, Any
import json
from pathlib import Path

# Test metinleri
TEST_TEXTS = {
    "kısa_turkce": "Merhaba, nasılsınız?",
    "orta_turkce": "Öğrencilerin kişiselleştirilmiş öğrenme yolculuğunda yapay zeka destekli adaptif değerlendirme sistemleri kullanılıyor.",
    "uzun_turkce": """Eğitim teknolojileri alanında gerçekleştirilen son araştırmalar, yapay zeka destekli öğrenme platformlarının 
    öğrenci başarısını %30-40 oranında artırdığını göstermektedir. Özellikle kişiselleştirilmiş öğrenme yolları, 
    adaptif değerlendirme sistemleri ve gerçek zamanlı geri bildirim mekanizmaları sayesinde öğrenciler kendi hızlarında 
    ve seviyelerine uygun içeriklerle çalışabilmektedir.""",
    "teknik_turkce": "Machine Learning ve Deep Learning algoritmaları kullanılarak NLP tabanlı bir AI sistemi geliştirildi.",
    "egitim_domain": "Lisans öğrencileri için hazırlanan bu PhD seviyesindeki içerik, MSc programlarında da kullanılabilir.",
    "mixed_lang": "Python programlama dilinde 'Merhaba Dünya' yazdırmak için print('Hello World') kullanılır.",
    "matematik": "x^2 + y^2 = r^2 denklemi bir çemberi ifade eder. Pi sayısı yaklaşık 3.14159'dur.",
    "tarih_saat": "Toplantı 15 Ocak 2024 tarihinde saat 14:30'da başlayacak ve %85 katılım bekleniyor."
}

class TokenizerBenchmark:
    """Tokenizer karşılaştırma sınıfı"""
    
    def __init__(self):
        self.results = {
            "qwen3": {},
            "turkish_mixtral": {}
        }
        self.qwen_tokenizer = None
        self.turkish_tokenizer = None
        
    def setup_tokenizers(self):
        """Tokenizer'ları yükle"""
        print("Tokenizer'lar yükleniyor...")
        
        # Qwen3 Tokenizer
        try:
            from qwen3_tiktoken_wrapper import Qwen3TiktokenTokenizer
            self.qwen_tokenizer = Qwen3TiktokenTokenizer()
            print("[OK] Qwen3 tokenizer yüklendi")
        except Exception as e:
            print(f"[ERROR] Qwen3 tokenizer yüklenemedi: {e}")
            
        # Turkish Mixtral Tokenizer
        try:
            from src.turkish_mixtral_tokenizer import TurkishMixtralTokenizer
            self.turkish_tokenizer = TurkishMixtralTokenizer()
            print("[OK] Turkish Mixtral tokenizer yüklendi")
        except Exception as e:
            print(f"[ERROR] Turkish Mixtral tokenizer yüklenemedi: {e}")
            
    def measure_memory(self) -> float:
        """Mevcut bellek kullanımını ölç (MB)"""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024
        
    def benchmark_tokenizer(self, tokenizer, name: str, text: str) -> Dict[str, Any]:
        """Tek bir tokenizer'ı benchmark et"""
        if tokenizer is None:
            return None
            
        # Bellek ölçümü başlangıç
        mem_start = self.measure_memory()
        
        # Tokenizasyon süresi
        start_time = time.time()
        
        try:
            # Tokenize
            if name == "qwen3":
                tokens = tokenizer.encode(text)
                token_strs = [str(t) for t in tokens[:10]]  # İlk 10 token
            else:  # turkish_mixtral
                tokens = tokenizer.encode(text)
                token_strs = tokenizer.tokenize(text)[:10]
                
            tokenization_time = time.time() - start_time
            
            # Decode süresi
            start_time = time.time()
            decoded = tokenizer.decode(tokens)
            decode_time = time.time() - start_time
            
            # Bellek ölçümü bitiş
            mem_end = self.measure_memory()
            
            return {
                "token_count": len(tokens),
                "tokenization_time_ms": tokenization_time * 1000,
                "decode_time_ms": decode_time * 1000,
                "memory_used_mb": mem_end - mem_start,
                "first_10_tokens": token_strs,
                "decode_match": text in decoded or decoded in text
            }
            
        except Exception as e:
            return {
                "error": str(e),
                "token_count": 0,
                "tokenization_time_ms": 0,
                "decode_time_ms": 0,
                "memory_used_mb": 0
            }
            
    def run_benchmarks(self):
        """Tüm benchmark'ları çalıştır"""
        print("\n" + "="*60)
        print("TOKENIZER BENCHMARK BAŞLIYOR")
        print("="*60)
        
        for text_name, text in TEST_TEXTS.items():
            print(f"\n[TEST] {text_name}")
            print(f"   Metin uzunluğu: {len(text)} karakter")
            
            # Qwen3 benchmark
            if self.qwen_tokenizer:
                result = self.benchmark_tokenizer(self.qwen_tokenizer, "qwen3", text)
                if result:
                    self.results["qwen3"][text_name] = result
                    print(f"   Qwen3: {result['token_count']} token, {result['tokenization_time_ms']:.2f}ms")
                    
            # Turkish Mixtral benchmark
            if self.turkish_tokenizer:
                result = self.benchmark_tokenizer(self.turkish_tokenizer, "turkish_mixtral", text)
                if result:
                    self.results["turkish_mixtral"][text_name] = result
                    print(f"   Turkish: {result['token_count']} token, {result['tokenization_time_ms']:.2f}ms")
                    
            # Karşılaştırma
            if text_name in self.results["qwen3"] and text_name in self.results["turkish_mixtral"]:
                qwen = self.results["qwen3"][text_name]
                turkish = self.results["turkish_mixtral"][text_name]
                
                if qwen["token_count"] > 0 and turkish["token_count"] > 0:
                    token_diff = ((qwen["token_count"] - turkish["token_count"]) / qwen["token_count"]) * 100
                    if qwen["tokenization_time_ms"] > 0:
                        time_diff = ((qwen["tokenization_time_ms"] - turkish["tokenization_time_ms"]) / qwen["tokenization_time_ms"]) * 100
                    else:
                        time_diff = 0
                    
                    print(f"   [KARSILASTIRMA]:")
                    print(f"      Token tasarrufu: {token_diff:.1f}%")
                    print(f"      Hız artışı: {time_diff:.1f}%")
                    
    def generate_report(self):
        """Detaylı rapor oluştur"""
        print("\n" + "="*60)
        print("DETAYLI KARŞILAŞTIRMA RAPORU")
        print("="*60)
        
        # Ortalama değerler
        if self.results["qwen3"] and self.results["turkish_mixtral"]:
            qwen_avg_tokens = sum(r["token_count"] for r in self.results["qwen3"].values()) / len(self.results["qwen3"])
            turkish_avg_tokens = sum(r["token_count"] for r in self.results["turkish_mixtral"].values()) / len(self.results["turkish_mixtral"])
            
            qwen_avg_time = sum(r["tokenization_time_ms"] for r in self.results["qwen3"].values()) / len(self.results["qwen3"])
            turkish_avg_time = sum(r["tokenization_time_ms"] for r in self.results["turkish_mixtral"].values()) / len(self.results["turkish_mixtral"])
            
            print(f"\n[ORTALAMA DEGERLER]:")
            print(f"{'Metrik':<30} {'Qwen3':>15} {'Turkish Mixtral':>15} {'Fark':>10}")
            print("-" * 70)
            print(f"{'Ortalama Token Sayısı':<30} {qwen_avg_tokens:>15.1f} {turkish_avg_tokens:>15.1f} {((qwen_avg_tokens-turkish_avg_tokens)/qwen_avg_tokens*100):>9.1f}%")
            print(f"{'Ortalama Tokenizasyon (ms)':<30} {qwen_avg_time:>15.2f} {turkish_avg_time:>15.2f} {((qwen_avg_time-turkish_avg_time)/qwen_avg_time*100):>9.1f}%")
            
        # Detaylı sonuçlar
        print(f"\n[DETAYLI SONUCLAR]:")
        print("-" * 70)
        
        for text_name in TEST_TEXTS.keys():
            print(f"\n[{text_name}]:")
            
            if text_name in self.results["qwen3"]:
                q = self.results["qwen3"][text_name]
                print(f"   Qwen3:")
                print(f"      Token sayısı: {q['token_count']}")
                print(f"      Tokenizasyon: {q['tokenization_time_ms']:.2f}ms")
                print(f"      Decode: {q['decode_time_ms']:.2f}ms")
                
            if text_name in self.results["turkish_mixtral"]:
                t = self.results["turkish_mixtral"][text_name]
                print(f"   Turkish Mixtral:")
                print(f"      Token sayısı: {t['token_count']}")
                print(f"      Tokenizasyon: {t['tokenization_time_ms']:.2f}ms")
                print(f"      Decode: {t['decode_time_ms']:.2f}ms")
                
        # Sonuçları JSON olarak kaydet
        with open("tokenizer_benchmark_results.json", "w", encoding="utf-8") as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        print(f"\n[KAYDEDILDI] Detaylı sonuçlar 'tokenizer_benchmark_results.json' dosyasına kaydedildi")
        
    def calculate_cost_impact(self):
        """Maliyet etkisi hesapla"""
        print("\n" + "="*60)
        print("MALİYET ETKİSİ ANALİZİ")
        print("="*60)
        
        if not (self.results["qwen3"] and self.results["turkish_mixtral"]):
            print("Yeterli veri yok")
            return
            
        # Ortalama token farkı
        total_qwen_tokens = sum(r["token_count"] for r in self.results["qwen3"].values())
        total_turkish_tokens = sum(r["token_count"] for r in self.results["turkish_mixtral"].values())
        
        token_reduction = (1 - total_turkish_tokens / total_qwen_tokens) * 100
        
        # API maliyet tahmini (OpenAI pricing benzeri)
        cost_per_1k_tokens = 0.002  # $0.002 per 1K tokens
        monthly_requests = 1_000_000
        avg_request_chars = 200  # Ortalama istek uzunluğu
        
        # Token tahminleri
        qwen_tokens_per_request = (total_qwen_tokens / sum(len(t) for t in TEST_TEXTS.values())) * avg_request_chars
        turkish_tokens_per_request = (total_turkish_tokens / sum(len(t) for t in TEST_TEXTS.values())) * avg_request_chars
        
        qwen_monthly_cost = (qwen_tokens_per_request * monthly_requests / 1000) * cost_per_1k_tokens
        turkish_monthly_cost = (turkish_tokens_per_request * monthly_requests / 1000) * cost_per_1k_tokens
        
        print(f"\n[MALIYET] Aylık API Maliyet Tahmini (1M request):")
        print(f"   Qwen3: ${qwen_monthly_cost:.2f}")
        print(f"   Turkish Mixtral: ${turkish_monthly_cost:.2f}")
        print(f"   Tasarruf: ${qwen_monthly_cost - turkish_monthly_cost:.2f} (%{token_reduction:.1f})")
        
        print(f"\n[PROJEKSIYON] Yıllık Projeksiyon:")
        print(f"   Tasarruf: ${(qwen_monthly_cost - turkish_monthly_cost) * 12:.2f}")
        
        print(f"\n[PERFORMANS] Performans Kazançları:")
        print(f"   Token Azalması: %{token_reduction:.1f}")
        print(f"   Context Window Verimliliği: %{token_reduction:.1f} daha fazla içerik")
        print(f"   Response Time İyileşmesi: ~%{token_reduction/2:.1f} (tahmini)")


def main():
    """Ana fonksiyon"""
    benchmark = TokenizerBenchmark()
    
    # Tokenizer'ları yükle
    benchmark.setup_tokenizers()
    
    if not benchmark.qwen_tokenizer and not benchmark.turkish_tokenizer:
        print("[ERROR] Hiçbir tokenizer yüklenemedi. Benchmark yapılamıyor.")
        return
        
    # Benchmark'ları çalıştır
    benchmark.run_benchmarks()
    
    # Rapor oluştur
    benchmark.generate_report()
    
    # Maliyet analizi
    benchmark.calculate_cost_impact()
    
    print("\n" + "="*60)
    print("BENCHMARK TAMAMLANDI")
    print("="*60)


if __name__ == "__main__":
    main()