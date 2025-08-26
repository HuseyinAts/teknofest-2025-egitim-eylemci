"""
Turkish LLM Pipeline Setup Script
Quick environment preparation and dependency installation

Usage:
    python setup_turkish_pipeline.py
"""

import os
import sys
import subprocess
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def install_requirements():
    """Install required packages"""
    
    requirements = [
        "torch>=2.0.0",
        "transformers>=4.35.0", 
        "datasets>=2.14.0",
        "peft>=0.6.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "tqdm>=4.65.0",
        "safetensors>=0.3.0",
        "accelerate>=0.24.0",
        "sentencepiece>=0.1.99",
        "psutil>=5.9.0"
    ]
    
    logger.info("Installing required packages...")
    
    for package in requirements:
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", package
            ])
            logger.info(f"✅ {package} installed")
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Failed to install {package}: {e}")
            return False
    
    return True

def setup_directories():
    """Create necessary directories"""
    
    directories = [
        "analysis_results",
        "vocab_analysis", 
        "qwen3_turkish_extended",
        "training_output",
        "turkish_llm_pipeline"
    ]
    
    logger.info("Setting up directories...")
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        logger.info(f"✅ Directory created: {directory}")

def check_gpu():
    """Check GPU availability"""
    
    try:
        import torch
        gpu_available = torch.cuda.is_available()
        gpu_count = torch.cuda.device_count()
        
        if gpu_available:
            logger.info(f"✅ GPU detected: {gpu_count} device(s)")
            for i in range(gpu_count):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                logger.info(f"   GPU {i}: {gpu_name} ({gpu_memory:.1f}GB)")
        else:
            logger.warning("⚠️  No GPU detected. Training will be slow.")
            
        return gpu_available
        
    except ImportError:
        logger.error("❌ PyTorch not installed")
        return False

def main():
    """Main setup function"""
    
    print("🇹🇷 Turkish LLM Pipeline Setup")
    print("=" * 40)
    
    # Install requirements
    if not install_requirements():
        logger.error("Failed to install requirements")
        return False
    
    # Setup directories
    setup_directories()
    
    # Check GPU
    gpu_ok = check_gpu()
    
    # Final check
    print("\n" + "=" * 40)
    print("Setup completed!")
    
    if gpu_ok:
        print("✅ Ready for training")
        print("\nNext steps:")
        print("1. Run: python quick_test_runner.py")
        print("2. Run: python master_orchestrator.py")
    else:
        print("⚠️  GPU recommended for training")
        print("\nNext steps:")
        print("1. Install CUDA-compatible PyTorch")
        print("2. Run: python quick_test_runner.py")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)