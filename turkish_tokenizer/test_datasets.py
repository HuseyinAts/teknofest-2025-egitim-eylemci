"""
Test script to verify dataset configuration
"""

import sys
sys.path.append('.')

from advanced_dataset_analyzer import AdvancedTurkishDatasetAnalyzer

def test_dataset_config():
    print("🧪 Testing Updated Dataset Configuration...")
    
    analyzer = AdvancedTurkishDatasetAnalyzer()
    
    print("\n📂 LOCAL DATASETS:")
    for name, path in analyzer.dataset_sources['local'].items():
        print(f"  ✅ {name}")
        print(f"     Path: {path}")
    
    print(f"\n🌐 HUGGINGFACE DATASETS:")
    for name, config in analyzer.dataset_sources['huggingface'].items():
        print(f"  ✅ {name}")
        print(f"     Column: {config['column']}, Limit: {config['limit']}")
    
    total_datasets = len(analyzer.dataset_sources['local']) + len(analyzer.dataset_sources['huggingface'])
    print(f"\n🎯 TOTAL CONFIGURED DATASETS: {total_datasets}")
    
    # Check for specific datasets mentioned by user
    expected_datasets = [
        'merve/turkish_instructions',
        'TFLai/Turkish-Alpaca', 
        'malhajar/OpenOrca-tr',
        'umarigan/turkish_corpus',
        'Huseyin/muspdf',
        'selimfirat/bilkent-turkish-writings-dataset'  # NEW ADDITION
    ]
    
    print(f"\n✅ VERIFICATION - Required HuggingFace datasets:")
    for dataset in expected_datasets:
        if dataset in analyzer.dataset_sources['huggingface']:
            print(f"  ✅ {dataset} - INCLUDED")
        else:
            print(f"  ❌ {dataset} - MISSING")
    
    print(f"\n✅ VERIFICATION - Synthetic datasets:")
    synthetic_datasets = [name for name in analyzer.dataset_sources['local'].keys() if 'synthetic' in name or 'llm_10k' in name]
    if synthetic_datasets:
        for dataset in synthetic_datasets:
            print(f"  ✅ {dataset} - INCLUDED")
    else:
        print(f"  ⚠️ No synthetic datasets found in config")

if __name__ == "__main__":
    test_dataset_config()