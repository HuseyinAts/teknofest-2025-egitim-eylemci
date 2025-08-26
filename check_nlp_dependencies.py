#!/usr/bin/env python3
"""
NLP Dependencies Checker
Verifies that all required NLP modules are installed and functional.
"""

import sys
import importlib
from typing import List, Tuple, Dict
import json
from datetime import datetime


class NLPDependencyChecker:
    """Check and validate NLP dependencies."""
    
    REQUIRED_MODULES = {
        # Core ML/NLP
        'torch': {
            'min_version': '2.1.0',
            'critical': True,
            'description': 'PyTorch deep learning framework'
        },
        'transformers': {
            'min_version': '4.36.0',
            'critical': True,
            'description': 'Hugging Face Transformers library'
        },
        'peft': {
            'min_version': '0.6.0',
            'critical': True,
            'description': 'Parameter-Efficient Fine-Tuning'
        },
        
        # Turkish NLP specific
        'zemberek': {
            'module_name': 'zemberek',
            'critical': True,
            'description': 'Turkish morphological analysis'
        },
        'turkish_morphology': {
            'module_name': 'turkish_morphology',
            'critical': True,
            'description': 'Turkish language morphology'
        },
        'nltk': {
            'min_version': '3.8.0',
            'critical': True,
            'description': 'Natural Language Toolkit'
        },
        'spacy': {
            'min_version': '3.7.0',
            'critical': False,
            'description': 'Industrial-strength NLP'
        },
        'TurkishStemmer': {
            'module_name': 'TurkishStemmer',
            'critical': True,
            'description': 'Turkish language stemming'
        }
    }
    
    TURKISH_NLP_DATA = {
        'nltk': [
            'punkt',
            'stopwords',
            'averaged_perceptron_tagger'
        ],
        'spacy': ['tr_core_news_md']
    }
    
    def __init__(self):
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'modules': {},
            'data': {},
            'errors': [],
            'warnings': []
        }
    
    def check_module(self, module_name: str, config: Dict) -> Tuple[bool, str]:
        """Check if a module is installed and meets version requirements."""
        actual_module = config.get('module_name', module_name)
        
        try:
            module = importlib.import_module(actual_module)
            
            # Check version if specified
            if 'min_version' in config:
                if hasattr(module, '__version__'):
                    version = module.__version__
                    if self._compare_versions(version, config['min_version']) < 0:
                        return False, f"Version {version} < required {config['min_version']}"
                    return True, f"Version {version}"
                else:
                    return True, "Version check not available"
            
            return True, "Installed"
            
        except ImportError as e:
            return False, f"Not installed: {str(e)}"
    
    def check_nltk_data(self) -> Dict[str, bool]:
        """Check if required NLTK data is downloaded."""
        import nltk
        results = {}
        
        for data_name in self.TURKISH_NLP_DATA['nltk']:
            try:
                nltk.data.find(f'tokenizers/{data_name}')
                results[data_name] = True
            except LookupError:
                results[data_name] = False
                # Try to download
                try:
                    nltk.download(data_name, quiet=True)
                    results[data_name] = True
                except Exception as e:
                    self.results['warnings'].append(f"Could not download NLTK data '{data_name}': {e}")
        
        return results
    
    def check_spacy_models(self) -> Dict[str, bool]:
        """Check if required spaCy models are installed."""
        try:
            import spacy
            results = {}
            
            for model_name in self.TURKISH_NLP_DATA['spacy']:
                try:
                    spacy.load(model_name)
                    results[model_name] = True
                except Exception:
                    results[model_name] = False
                    self.results['warnings'].append(
                        f"spaCy model '{model_name}' not installed. "
                        f"Install with: python -m spacy download {model_name}"
                    )
            
            return results
        except ImportError:
            return {}
    
    def test_turkish_nlp_functionality(self) -> bool:
        """Test basic Turkish NLP functionality."""
        test_text = "Merhaba, bu bir Türkçe test cümlesidir."
        
        tests_passed = []
        
        # Test tokenization with NLTK
        try:
            import nltk
            tokens = nltk.word_tokenize(test_text)
            tests_passed.append(('NLTK Tokenization', len(tokens) > 0))
        except Exception as e:
            tests_passed.append(('NLTK Tokenization', False))
            self.results['errors'].append(f"NLTK tokenization failed: {e}")
        
        # Test Turkish stemmer
        try:
            from TurkishStemmer import TurkishStemmer
            stemmer = TurkishStemmer()
            stemmed = stemmer.stem("kitaplar")
            tests_passed.append(('Turkish Stemming', stemmed == "kitap"))
        except Exception as e:
            tests_passed.append(('Turkish Stemming', False))
            self.results['errors'].append(f"Turkish stemming failed: {e}")
        
        # Test transformers tokenizer
        try:
            from transformers import AutoTokenizer
            # Try to load a Turkish model tokenizer
            tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-turkish-cased")
            tokens = tokenizer.tokenize(test_text)
            tests_passed.append(('Transformers Tokenization', len(tokens) > 0))
        except Exception as e:
            tests_passed.append(('Transformers Tokenization', False))
            self.results['warnings'].append(f"Transformers Turkish tokenization warning: {e}")
        
        return all(result for _, result in tests_passed)
    
    def _compare_versions(self, v1: str, v2: str) -> int:
        """Compare two version strings."""
        def normalize(v):
            parts = v.split('.')
            return [int(x) for x in parts[:3]]  # Compare major.minor.patch
        
        v1_parts = normalize(v1)
        v2_parts = normalize(v2)
        
        for i in range(3):
            if i >= len(v1_parts):
                v1_parts.append(0)
            if i >= len(v2_parts):
                v2_parts.append(0)
        
        if v1_parts < v2_parts:
            return -1
        elif v1_parts > v2_parts:
            return 1
        return 0
    
    def run_checks(self) -> Dict:
        """Run all dependency checks."""
        print("=" * 60)
        print("NLP Dependency Checker")
        print("=" * 60)
        
        # Check modules
        critical_missing = []
        for module_name, config in self.REQUIRED_MODULES.items():
            success, message = self.check_module(module_name, config)
            self.results['modules'][module_name] = {
                'installed': success,
                'message': message,
                'critical': config.get('critical', False),
                'description': config.get('description', '')
            }
            
            status = "✓" if success else "✗"
            critical_marker = " [CRITICAL]" if config.get('critical') else ""
            print(f"{status} {module_name}: {message}{critical_marker}")
            
            if not success and config.get('critical'):
                critical_missing.append(module_name)
        
        print("\n" + "-" * 60)
        
        # Check NLTK data
        if 'nltk' in self.results['modules'] and self.results['modules']['nltk']['installed']:
            print("\nChecking NLTK data...")
            nltk_results = self.check_nltk_data()
            self.results['data']['nltk'] = nltk_results
            for data_name, installed in nltk_results.items():
                status = "✓" if installed else "✗"
                print(f"  {status} {data_name}")
        
        # Check spaCy models
        if 'spacy' in self.results['modules'] and self.results['modules']['spacy']['installed']:
            print("\nChecking spaCy models...")
            spacy_results = self.check_spacy_models()
            self.results['data']['spacy'] = spacy_results
            for model_name, installed in spacy_results.items():
                status = "✓" if installed else "✗"
                print(f"  {status} {model_name}")
        
        print("\n" + "-" * 60)
        
        # Test functionality
        print("\nTesting Turkish NLP functionality...")
        functionality_ok = self.test_turkish_nlp_functionality()
        self.results['functionality_test'] = functionality_ok
        
        if functionality_ok:
            print("✓ Turkish NLP functionality tests passed")
        else:
            print("✗ Some Turkish NLP functionality tests failed")
        
        # Summary
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        
        if critical_missing:
            print(f"\n❌ CRITICAL: Missing required modules: {', '.join(critical_missing)}")
            print("\nInstall missing dependencies with:")
            print("  pip install -r requirements.txt")
            self.results['status'] = 'FAILED'
        else:
            print("\n✅ All critical NLP dependencies are installed")
            self.results['status'] = 'PASSED'
        
        if self.results['warnings']:
            print(f"\n⚠️  Warnings ({len(self.results['warnings'])}):")
            for warning in self.results['warnings'][:5]:
                print(f"  - {warning}")
        
        if self.results['errors']:
            print(f"\n❌ Errors ({len(self.results['errors'])}):")
            for error in self.results['errors'][:5]:
                print(f"  - {error}")
        
        # Save results to file
        with open('nlp_dependency_check_results.json', 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"\nDetailed results saved to: nlp_dependency_check_results.json")
        
        return self.results


def main():
    """Main entry point."""
    checker = NLPDependencyChecker()
    results = checker.run_checks()
    
    # Exit with appropriate code
    if results['status'] == 'FAILED':
        sys.exit(1)
    elif results['warnings'] or results['errors']:
        sys.exit(2)  # Exit with warning code
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()