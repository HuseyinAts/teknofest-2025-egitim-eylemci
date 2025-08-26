"""
Quick Test Runner for Turkish LLM Pipeline
Validates setup and runs basic functionality tests

Tests:
1. Environment and dependencies
2. Dataset availability and format
3. Qwen3-8B model access
4. Basic Turkish text processing
5. Memory and compute requirements
"""

import os
import sys
import json
import torch
from pathlib import Path
import importlib
import traceback
from typing import Dict, List, Tuple

def test_environment_setup() -> Tuple[bool, str]:
    """Test Python environment and dependencies"""
    
    print("🔍 Testing environment setup...")
    
    try:
        # Check Python version
        if sys.version_info < (3, 9):
            return False, f"Python 3.9+ required, found {sys.version}"
        
        # Check required packages
        required_packages = [
            'torch', 'transformers', 'datasets', 'peft',
            'numpy', 'pandas', 'tqdm', 'safetensors'
        ]
        
        missing_packages = []
        for package in required_packages:
            try:
                importlib.import_module(package)
            except ImportError:
                missing_packages.append(package)
        
        if missing_packages:
            return False, f"Missing packages: {', '.join(missing_packages)}"
        
        # Check torch GPU availability
        gpu_available = torch.cuda.is_available()
        gpu_count = torch.cuda.device_count() if gpu_available else 0
        
        print(f"   ✅ Python {sys.version.split()[0]}")
        print(f"   ✅ All required packages installed")
        print(f"   {'✅' if gpu_available else '⚠️ '} GPU: {gpu_count} device(s) available")
        
        return True, f"Environment OK (GPU: {gpu_available})"
        
    except Exception as e:
        return False, f"Environment test failed: {e}"


def test_dataset_availability() -> Tuple[bool, str]:
    """Test dataset files availability"""
    
    print("📊 Testing dataset availability...")
    
    try:
        data_dir = Path("../data")
        
        # Check for main data files
        expected_files = [
            "raw/turkish_quiz_instruct.csv",
            "processed/competition_dataset.json",
            "TR_MEGA_1_2000_Combined.jsonl"
        ]
        
        found_files = []
        missing_files = []
        
        for file_path in expected_files:
            full_path = data_dir / file_path
            if full_path.exists():
                found_files.append(file_path)
                size_mb = full_path.stat().st_size / (1024 * 1024)
                print(f"   ✅ {file_path} ({size_mb:.1f} MB)")
            else:
                missing_files.append(file_path)
                print(f"   ❌ {file_path} (not found)")
        
        if not found_files:
            return False, "No dataset files found"
        
        if missing_files:
            return True, f"Partial dataset available: {len(found_files)}/{len(expected_files)} files"
        
        return True, f"All datasets available: {len(found_files)} files"
        
    except Exception as e:
        return False, f"Dataset test failed: {e}"


def test_qwen_model_access() -> Tuple[bool, str]:
    """Test Qwen3-8B model access"""
    
    print("🤖 Testing Qwen3-8B model access...")
    
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        # Try to load tokenizer first (smaller download)
        print("   📥 Loading Qwen3-8B tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            "Qwen/Qwen3-8B",
            trust_remote_code=True
        )
        
        vocab_size = len(tokenizer.vocab)
        print(f"   ✅ Tokenizer loaded (vocab size: {vocab_size:,})")
        
        # Test tokenization
        test_text = "Merhaba dünya! Bu bir Türkçe test metnidir."
        tokens = tokenizer.tokenize(test_text)
        
        print(f"   ✅ Tokenization test: {len(tokens)} tokens for Turkish text")
        
        # Try to get model config (without loading full model)
        try:
            from transformers import AutoConfig
            config = AutoConfig.from_pretrained("Qwen/Qwen3-8B", trust_remote_code=True)
            print(f"   ✅ Model config accessible (hidden_size: {config.hidden_size})")
        except Exception as e:
            print(f"   ⚠️  Model config: {e}")
        
        return True, f"Qwen3-8B accessible (vocab: {vocab_size:,})"
        
    except Exception as e:
        return False, f"Qwen model test failed: {e}"


def test_turkish_processing() -> Tuple[bool, str]:
    """Test Turkish text processing capabilities"""
    
    print("🇹🇷 Testing Turkish text processing...")
    
    try:
        # Test Turkish text samples
        turkish_samples = [
            "Geliyorum eve, yarın görüşürüz.",
            "Ağaçların yeşilliği çok güzel.",
            "Teknoloji alanındaki yenilikler hayatımızı kolaylaştırıyor.",
            "Öğrencilerin başarısı için elimizden geleni yapacağız."
        ]
        
        # Basic Turkish character detection
        turkish_chars = set('çğıöşüâîû')
        
        results = []
        for text in turkish_samples:
            # Character analysis
            char_count = len(text)
            turkish_char_count = sum(1 for c in text.lower() if c in turkish_chars)
            turkish_ratio = turkish_char_count / char_count if char_count > 0 else 0
            
            # Word analysis
            words = text.split()
            avg_word_length = sum(len(word) for word in words) / len(words) if words else 0
            
            results.append({
                'text': text[:30] + "..." if len(text) > 30 else text,
                'turkish_ratio': turkish_ratio,
                'avg_word_length': avg_word_length,
                'words': len(words)
            })
        
        avg_turkish_ratio = sum(r['turkish_ratio'] for r in results) / len(results)
        avg_word_length = sum(r['avg_word_length'] for r in results) / len(results)
        
        print(f"   ✅ Turkish character ratio: {avg_turkish_ratio:.3f}")
        print(f"   ✅ Average word length: {avg_word_length:.1f}")
        print(f"   ✅ Processed {len(turkish_samples)} Turkish samples")
        
        return True, f"Turkish processing OK (ratio: {avg_turkish_ratio:.3f})"
        
    except Exception as e:
        return False, f"Turkish processing test failed: {e}"


def test_memory_requirements() -> Tuple[bool, str]:
    """Test memory and compute requirements"""
    
    print("💾 Testing memory and compute requirements...")
    
    try:
        import psutil
        
        # System memory
        memory = psutil.virtual_memory()
        memory_gb = memory.total / (1024**3)
        memory_available_gb = memory.available / (1024**3)
        
        print(f"   📊 System RAM: {memory_gb:.1f} GB total, {memory_available_gb:.1f} GB available")
        
        # GPU memory (if available)
        gpu_memory_info = ""
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                gpu_memory_info += f"GPU{i}: {gpu_memory:.1f}GB "
                print(f"   🎮 {torch.cuda.get_device_name(i)}: {gpu_memory:.1f} GB")
        else:
            print(f"   ⚠️  No GPU detected")
        
        # Estimate requirements
        estimated_model_memory = 16  # GB for Qwen3-8B in FP16
        estimated_training_memory = 24  # GB for training with LoRA
        
        sufficient_memory = memory_gb >= 32 and (
            not torch.cuda.is_available() or 
            any(torch.cuda.get_device_properties(i).total_memory / (1024**3) >= 20 
                for i in range(torch.cuda.device_count()))
        )
        
        status = "✅ Sufficient" if sufficient_memory else "⚠️  Limited"
        print(f"   {status} for training requirements")
        
        return sufficient_memory, f"Memory OK: {memory_gb:.1f}GB RAM, {gpu_memory_info}"
        
    except Exception as e:
        return False, f"Memory test failed: {e}"


def test_pipeline_imports() -> Tuple[bool, str]:
    """Test pipeline module imports"""
    
    print("🔧 Testing pipeline module imports...")
    
    try:
        # Test our custom modules
        modules_to_test = [
            'advanced_dataset_analyzer',
            'turkish_vocabulary_analyzer', 
            'qwen_turkish_extender',
            'advanced_turkish_trainer',
            'master_orchestrator'
        ]
        
        import_results = []
        
        for module_name in modules_to_test:
            try:
                module_path = Path(f"{module_name}.py")
                if module_path.exists():
                    # Basic syntax check by compiling
                    with open(module_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    compile(content, module_path, 'exec')
                    print(f"   ✅ {module_name}.py (syntax OK)")
                    import_results.append(True)
                else:
                    print(f"   ❌ {module_name}.py (not found)")
                    import_results.append(False)
                    
            except SyntaxError as e:
                print(f"   ❌ {module_name}.py (syntax error: {e})")
                import_results.append(False)
            except Exception as e:
                print(f"   ⚠️  {module_name}.py ({e})")
                import_results.append(False)
        
        success_count = sum(import_results)
        total_count = len(import_results)
        
        if success_count == total_count:
            return True, f"All {total_count} pipeline modules OK"
        else:
            return False, f"Only {success_count}/{total_count} pipeline modules OK"
        
    except Exception as e:
        return False, f"Pipeline import test failed: {e}"


def run_quick_test_suite() -> Dict:
    """Run complete quick test suite"""
    
    print("\n" + "="*60)
    print("🚀 TURKISH LLM PIPELINE - QUICK TEST SUITE")
    print("="*60)
    
    tests = [
        ("Environment Setup", test_environment_setup),
        ("Dataset Availability", test_dataset_availability), 
        ("Qwen Model Access", test_qwen_model_access),
        ("Turkish Processing", test_turkish_processing),
        ("Memory Requirements", test_memory_requirements),
        ("Pipeline Imports", test_pipeline_imports)
    ]
    
    results = {}
    passed_tests = 0
    total_tests = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        try:
            success, message = test_func()
            results[test_name] = {
                'success': success,
                'message': message
            }
            
            if success:
                passed_tests += 1
                print(f"   ✅ PASSED: {message}")
            else:
                print(f"   ❌ FAILED: {message}")
                
        except Exception as e:
            results[test_name] = {
                'success': False,
                'message': f"Test error: {e}"
            }
            print(f"   ❌ ERROR: {e}")
            print(f"   {traceback.format_exc()}")
    
    # Summary
    print(f"\n" + "="*60)
    print(f"TEST SUMMARY: {passed_tests}/{total_tests} tests passed")
    print("="*60)
    
    overall_success = passed_tests == total_tests
    
    if overall_success:
        print("🎉 ALL TESTS PASSED! Pipeline is ready for execution.")
        print("\nNext steps:")
        print("1. Run: python master_orchestrator.py")
        print("2. Or run individual stages as needed")
        print("3. Monitor progress in logs and output directories")
    else:
        print(f"⚠️  {total_tests - passed_tests} tests failed. Please address issues before running pipeline.")
        print("\nFailed tests:")
        for test_name, result in results.items():
            if not result['success']:
                print(f"   - {test_name}: {result['message']}")
    
    # Save test results
    results_summary = {
        'timestamp': str(torch.utils.data.get_worker_info()),
        'passed_tests': passed_tests,
        'total_tests': total_tests,
        'overall_success': overall_success,
        'detailed_results': results
    }
    
    with open('test_results.json', 'w', encoding='utf-8') as f:
        json.dump(results_summary, f, indent=2, default=str)
    
    print(f"\nDetailed results saved to: test_results.json")
    
    return results_summary


if __name__ == "__main__":
    # Run the quick test suite
    results = run_quick_test_suite()
    
    # Exit with appropriate code
    sys.exit(0 if results['overall_success'] else 1)