#!/usr/bin/env python3
"""
Türkçe Tokenizer Proje Kurulumu
Bu script, Qwen3-8B için Türkçe tokenizer geliştirme projesinin
tüm klasör yapısını ve temel dosyalarını oluşturur.
"""

import os
import sys
from pathlib import Path

def create_directory_structure():
    """Proje klasör yapısını oluştur"""
    
    # Ana proje dizini
    base_dir = Path("c:/Users/husey/teknofest-2025-egitim-eylemci/turkish_tokenizer")
    
    # Oluşturulacak klasörler
    directories = [
        "data/raw",
        "data/processed", 
        "data/tokenizer_data",
        "vocab_analysis",
        "tokenizer_extension",
        "training",
        "evaluation",
        "models/qwen3_original",
        "models/qwen3_turkish",
        "scripts",
        "configs",
        "notebooks",
        "results",
        "logs"
    ]
    
    print("🚀 Türkçe Tokenizer Proje Yapısı Oluşturuluyor...")
    
    for directory in directories:
        dir_path = base_dir / directory
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"✅ Oluşturuldu: {directory}")
    
    # __init__.py dosyalarını oluştur
    python_dirs = ["vocab_analysis", "tokenizer_extension", "training", "evaluation"]
    for dir_name in python_dirs:
        init_file = base_dir / dir_name / "__init__.py"
        init_file.touch(exist_ok=True)
    
    print("\n📁 Proje klasör yapısı başarıyla oluşturuldu!")

def create_requirements_file():
    """Gerekli paketlerin listesini oluştur"""
    base_dir = Path("c:/Users/husey/teknofest-2025-egitim-eylemci/turkish_tokenizer")
    
    requirements = """# Türkçe Tokenizer Geliştirme İçin Gerekli Paketler

# Temel ML/NLP kütüphaneleri
torch>=2.0.0
transformers>=4.35.0
tokenizers>=0.15.0
datasets>=2.14.0

# Qwen3 modeli için
safetensors>=0.4.0
accelerate>=0.24.0

# Veri işleme
pandas>=2.0.0
numpy>=1.24.0
scipy>=1.10.0

# Türkçe NLP
zeyrek  # Türkçe morfotaktik analiz
turkish-stemmer  # Türkçe gövdeleme

# Görselleştirme ve analiz
matplotlib>=3.7.0
seaborn>=0.12.0
plotly>=5.15.0
wordcloud>=1.9.0

# Metin işleme
regex>=2023.8.8
sentencepiece>=0.1.99

# Progress bars ve utilities
tqdm>=4.65.0
rich>=13.5.0

# Jupyter notebook
jupyter>=1.0.0
ipykernel>=6.25.0

# Test ve kalite kontrolü
pytest>=7.4.0
black>=23.7.0
"""
    
    req_file = base_dir / "requirements.txt"
    with open(req_file, 'w', encoding='utf-8') as f:
        f.write(requirements)
    
    print("📋 requirements.txt dosyası oluşturuldu!")

def create_config_files():
    """Yapılandırma dosyalarını oluştur"""
    base_dir = Path("c:/Users/husey/teknofest-2025-egitim-eylemci/turkish_tokenizer")
    
    # Ana yapılandırma dosyası
    config_content = """# Türkçe Tokenizer Yapılandırması

# Model bilgileri
MODEL_NAME: "Qwen/Qwen2.5-8B"
MODEL_PATH: "models/qwen3_original"
TURKISH_MODEL_PATH: "models/qwen3_turkish"

# Veri yolları
DATA_RAW_PATH: "../data/raw/turkish_quiz_instruct.csv"
DATA_PROCESSED_PATH: "../data/processed/competition_dataset.jsonl"
TOKENIZER_DATA_PATH: "data/tokenizer_data"

# Tokenizer ayarları
ORIGINAL_VOCAB_SIZE: 151936
TURKISH_VOCAB_SIZE: 50000  # Eklenecek Türkçe token sayısı
TOTAL_VOCAB_SIZE: 201936

# Eğitim ayarları
BATCH_SIZE: 16
LEARNING_RATE: 5e-5
NUM_EPOCHS: 3
MAX_LENGTH: 2048

# Donanım ayarları
DEVICE: "cuda"  # veya "cpu"
MIXED_PRECISION: true
GRADIENT_CHECKPOINTING: true

# Türkçe özel ayarlar
TURKISH_SUFFIXES: ["-ler", "-lar", "-de", "-da", "-den", "-dan", "-ye", "-ya", "-nin", "-nın", "-nün", "-nun"]
MORPHEME_AWARE: true
VOWEL_HARMONY: true

# Değerlendirme ayarları
EVAL_BATCH_SIZE: 32
EVAL_STEPS: 500
SAVE_STEPS: 1000

# Çıktı ayarları
LOG_LEVEL: "INFO"
SAVE_TOTAL_LIMIT: 3
"""
    
    config_file = base_dir / "configs" / "config.yaml"
    with open(config_file, 'w', encoding='utf-8') as f:
        f.write(config_content)
    
    print("⚙️ config.yaml dosyası oluşturuldu!")

if __name__ == "__main__":
    print("🇹🇷 Qwen3-8B Türkçe Tokenizer Proje Kurulumu")
    print("=" * 50)
    
    create_directory_structure()
    create_requirements_file() 
    create_config_files()
    
    print("\n🎉 Proje kurulumu tamamlandı!")
    print("\n📋 Sıradaki adımlar:")
    print("1. cd turkish_tokenizer")
    print("2. pip install -r requirements.txt")
    print("3. python scripts/analyze_turkish_data.py")