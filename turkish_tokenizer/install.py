#!/usr/bin/env python3
"""
🚀 Turkish LLM Pipeline - One-Click Installation Script
Automatic setup for GitHub repository usage

Usage:
    python install.py
    python install.py --colab  # For Google Colab
    python install.py --test   # Run tests after installation
"""

import os
import sys
import subprocess
import platform
import argparse
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class TurkishLLMInstaller:
    """Complete installation and setup manager"""
    
    def __init__(self, colab_mode=False, run_tests=False):
        self.colab_mode = colab_mode
        self.run_tests = run_tests
        self.base_dir = Path(__file__).parent
        
        # Platform detection
        self.is_windows = platform.system() == "Windows"
        self.is_linux = platform.system() == "Linux" 
        self.is_mac = platform.system() == "Darwin"
        
        # Python executable
        self.python_cmd = sys.executable
        
        print("🇹🇷 Turkish LLM Pipeline - Installation Starting")
        print("=" * 60)
        print(f"📍 Platform: {platform.system()}")
        print(f"🐍 Python: {sys.version}")
        print(f"📁 Directory: {self.base_dir}")
        print(f"☁️  Colab Mode: {'Yes' if self.colab_mode else 'No'}")
        print("=" * 60)
    
    def check_python_version(self):
        """Python version kontrolü"""
        
        version = sys.version_info
        if version.major < 3 or (version.major == 3 and version.minor < 9):
            logger.error("❌ Python 3.9+ required!")
            logger.error(f"Current version: {version.major}.{version.minor}")
            return False
        
        logger.info(f"✅ Python version: {version.major}.{version.minor} (OK)")
        return True
    
    def install_requirements(self):
        """Requirements installation"""
        
        logger.info("📦 Installing requirements...")
        
        requirements_file = self.base_dir / "requirements.txt"
        
        if not requirements_file.exists():
            logger.error(f"❌ Requirements file not found: {requirements_file}")
            return False
        
        try:
            # Upgrade pip first
            subprocess.check_call([
                self.python_cmd, "-m", "pip", "install", "--upgrade", "pip"
            ])
            
            # Install requirements
            cmd = [self.python_cmd, "-m", "pip", "install", "-r", str(requirements_file)]
            
            if self.colab_mode:
                cmd.extend(["--quiet", "--no-warn-script-location"])
            
            subprocess.check_call(cmd)
            
            logger.info("✅ Requirements installed successfully")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Requirements installation failed: {e}")
            return False
    
    def setup_directories(self):
        """Gerekli dizinleri oluştur"""
        
        logger.info("📁 Setting up directories...")
        
        directories = [
            "analysis_results",
            "vocab_analysis", 
            "training_output",
            "dataset_cache",
            "logs"
        ]
        
        for dir_name in directories:
            dir_path = self.base_dir / dir_name
            dir_path.mkdir(exist_ok=True)
            logger.info(f"✅ Created: {dir_name}/")
        
        return True
    
    def check_gpu_availability(self):
        """GPU kontrolü"""
        
        logger.info("🔍 Checking GPU availability...")
        
        try:
            import torch
            
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                
                logger.info(f"✅ GPU detected: {gpu_name}")
                logger.info(f"✅ GPU memory: {gpu_memory:.1f}GB")
                logger.info(f"✅ GPU count: {gpu_count}")
                
                if "A100" in gpu_name:
                    logger.info("🎉 A100 GPU detected - optimal for training!")
                elif gpu_memory >= 24:
                    logger.info("✅ High-memory GPU - good for training")
                else:
                    logger.warning("⚠️ Low GPU memory - consider Colab Pro+")
                
                return True
            else:
                logger.warning("⚠️ No GPU detected - training will be slow")
                return False
                
        except ImportError:
            logger.error("❌ PyTorch not installed")
            return False
    
    def verify_installation(self):
        """Installation verification"""
        
        logger.info("🧪 Verifying installation...")
        
        # Test core imports
        test_imports = [
            "torch", "transformers", "datasets", "peft", 
            "numpy", "pandas", "tqdm"
        ]
        
        failed_imports = []
        
        for module in test_imports:
            try:
                __import__(module)
                logger.info(f"✅ {module}")
            except ImportError as e:
                logger.error(f"❌ {module}: {e}")
                failed_imports.append(module)
        
        if failed_imports:
            logger.error(f"❌ Failed imports: {failed_imports}")
            return False
        
        # Test Turkish LLM components
        try:
            from quick_test_runner import run_quick_tests
            logger.info("✅ Turkish LLM components importable")
            
            if self.run_tests:
                logger.info("🧪 Running quick tests...")
                test_results = run_quick_tests()
                if test_results.get('all_passed', False):
                    logger.info("✅ All tests passed")
                else:
                    logger.warning("⚠️ Some tests failed - check test_results.json")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Component test failed: {e}")
            return False
    
    def create_colab_setup(self):
        """Google Colab özel setup"""
        
        if not self.colab_mode:
            return True
        
        logger.info("☁️ Setting up Google Colab environment...")
        
        # Mount Google Drive
        try:
            from google.colab import drive
            drive.mount('/content/drive')
            logger.info("✅ Google Drive mounted")
        except:
            logger.warning("⚠️ Google Drive mount failed (may not be in Colab)")
        
        # Set up Colab-specific paths
        colab_dirs = [
            "/content/drive/MyDrive/turkish_llm_output",
            "/content/drive/MyDrive/dataset_cache",
            "/content/logs"
        ]
        
        for dir_path in colab_dirs:
            os.makedirs(dir_path, exist_ok=True)
            logger.info(f"✅ Colab directory: {dir_path}")
        
        return True
    
    def print_usage_guide(self):
        """Usage guide yazdır"""
        
        print("\n" + "🎯 INSTALLATION COMPLETED" + "\n")
        print("=" * 60)
        
        if self.colab_mode:
            print("☁️ GOOGLE COLAB USAGE:")
            print("""
# Quick Start (Colab Pro+ A100)
from colab_pro_a100_optimized_trainer import run_colab_pro_a100_training
results = run_colab_pro_a100_training()
print(f"Final Loss: {results['final_loss']:.4f}")

# Hybrid Ensemble (Maximum Success)
from hybrid_ensemble_trainer import run_ensemble_training
ensemble_results = run_ensemble_training()
            """)
        else:
            print("💻 LOCAL/SERVER USAGE:")
            print("""
# Complete Pipeline
python master_orchestrator.py --vocab-size 40000

# Quick Test
python quick_test_runner.py

# Individual Components
from final_master_trainer import run_final_master_training
results = run_final_master_training()
            """)
        
        print("\n📊 EXPECTED RESULTS:")
        print("- Final Loss: <1.5 (target)")
        print("- Token Reduction: 50-70%") 
        print("- Training Time: 6-10 hours (A100)")
        print("- Success Rate: 95%+ (ensemble)")
        
        print("\n📁 KEY FILES:")
        print("- README.md - Project overview")
        print("- GITHUB_USAGE_GUIDE.md - Detailed usage")
        print("- ULTRA_ANALIZ_RAPORU.md - Technical analysis")
        print("- quick_test_runner.py - Environment testing")
        
        print("\n🆘 SUPPORT:")
        print("- Check logs/ directory for detailed logs")
        print("- Run: python quick_test_runner.py")
        print("- Review GITHUB_USAGE_GUIDE.md for troubleshooting")
        
        print("\n✅ Ready to train Turkish LLM!")
        print("=" * 60)
    
    def run_installation(self):
        """Ana installation süreci"""
        
        # Step-by-step installation
        steps = [
            ("Python Version Check", self.check_python_version),
            ("Directory Setup", self.setup_directories),
            ("Requirements Installation", self.install_requirements),
            ("GPU Check", self.check_gpu_availability),
            ("Colab Setup", self.create_colab_setup),
            ("Installation Verification", self.verify_installation)
        ]
        
        for step_name, step_func in steps:
            logger.info(f"🔄 {step_name}...")
            
            try:
                success = step_func()
                if success:
                    logger.info(f"✅ {step_name} - OK")
                else:
                    logger.error(f"❌ {step_name} - FAILED")
                    return False
                    
            except Exception as e:
                logger.error(f"❌ {step_name} - ERROR: {e}")
                return False
        
        # Success
        self.print_usage_guide()
        return True


def main():
    """Main installation function"""
    
    parser = argparse.ArgumentParser(description="Turkish LLM Pipeline Installer")
    parser.add_argument("--colab", action="store_true", help="Google Colab mode")
    parser.add_argument("--test", action="store_true", help="Run tests after installation")
    
    args = parser.parse_args()
    
    # Run installation
    installer = TurkishLLMInstaller(
        colab_mode=args.colab,
        run_tests=args.test
    )
    
    success = installer.run_installation()
    
    if success:
        print("🎉 Installation completed successfully!")
        sys.exit(0)
    else:
        print("❌ Installation failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()