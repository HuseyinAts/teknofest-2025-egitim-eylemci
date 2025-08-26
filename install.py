#!/usr/bin/env python3
"""
🚀 TEKNOFEST 2025 - Turkish LLM Training System
One-Click Installation Script

This script automatically sets up the complete Turkish LLM training environment
with all optimizations and dependencies.
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def print_header():
    """Print installation header"""
    print("🔥" * 80)
    print("🚀 TEKNOFEST 2025 - TURKISH LLM TRAINING SYSTEM")
    print("🔥" * 80)
    print("📦 One-Click Installation Script")
    print("🎯 Setting up complete Turkish LLM training environment...")
    print("🔥" * 80)
    print()

def check_python_version():
    """Check Python version compatibility"""
    print("🐍 Checking Python version...")
    
    if sys.version_info < (3, 9):
        print("❌ Error: Python 3.9+ required")
        print(f"   Current version: {sys.version}")
        print("   Please upgrade Python and try again.")
        sys.exit(1)
    
    print(f"✅ Python {sys.version.split()[0]} - Compatible")
    return True

def check_gpu():
    """Check GPU availability and type"""
    print("\n🎮 Checking GPU availability...")
    
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            
            print(f"✅ GPU detected: {gpu_name}")
            print(f"📊 GPU memory: {gpu_memory:.1f}GB")
            
            if "A100" in gpu_name:
                print("🚀 A100 detected - enabling tensor core optimizations")
                return "A100"
            elif gpu_memory >= 16:
                print("✅ High-memory GPU - suitable for training")
                return "HIGH_MEM"
            else:
                print("⚠️ Low-memory GPU - may need batch size adjustments")
                return "LOW_MEM"
        else:
            print("⚠️ No GPU detected - CPU training will be slow")
            return "CPU"
    except ImportError:
        print("⚠️ PyTorch not installed yet - will check after installation")
        return "UNKNOWN"

def install_pytorch():
    """Install PyTorch with CUDA support"""
    print("\n🔥 Installing PyTorch with CUDA support...")
    
    # Detect platform
    system = platform.system().lower()
    
    # PyTorch installation command
    if system == "linux" or "colab" in os.environ.get("HOSTNAME", "").lower():
        cmd = [
            sys.executable, "-m", "pip", "install", 
            "torch", "torchvision", "torchaudio", 
            "--index-url", "https://download.pytorch.org/whl/cu118"
        ]
    else:
        cmd = [
            sys.executable, "-m", "pip", "install", 
            "torch", "torchvision", "torchaudio"
        ]
    
    try:
        subprocess.run(cmd, check=True)
        print("✅ PyTorch installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ PyTorch installation failed: {e}")
        return False

def install_requirements():
    """Install all required packages"""
    print("\n📦 Installing required packages...")
    
    requirements_file = Path(__file__).parent / "requirements.txt"
    
    if not requirements_file.exists():
        print("❌ requirements.txt not found")
        return False
    
    try:
        cmd = [sys.executable, "-m", "pip", "install", "-r", str(requirements_file)]
        subprocess.run(cmd, check=True)
        print("✅ Requirements installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Requirements installation failed: {e}")
        return False

def install_optional_optimizations():
    """Install optional optimization packages"""
    print("\n⚡ Installing optional optimizations...")
    
    optional_packages = [
        ("git+https://github.com/Liuhong99/Sophia.git", "Sophia optimizer (real Hessian approximation)"),
        ("flash-attn", "Flash Attention (A100 optimization)"),
    ]
    
    for package, description in optional_packages:
        print(f"📦 Installing {description}...")
        try:
            cmd = [sys.executable, "-m", "pip", "install", package]
            subprocess.run(cmd, check=True, capture_output=True)
            print(f"✅ {description} installed")
        except subprocess.CalledProcessError:
            print(f"⚠️ {description} installation failed (optional)")

def setup_directories():
    """Create necessary directories"""
    print("\n📁 Setting up directories...")
    
    directories = [
        "checkpoints",
        "outputs", 
        "logs",
        "data",
        "models"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✅ Created directory: {directory}")

def verify_installation():
    """Verify that everything is installed correctly"""
    print("\n🔍 Verifying installation...")
    
    # Test imports
    test_imports = [
        ("torch", "PyTorch"),
        ("transformers", "Transformers"),
        ("datasets", "Datasets"),
        ("peft", "PEFT"),
        ("accelerate", "Accelerate")
    ]
    
    all_good = True
    
    for module_name, display_name in test_imports:
        try:
            __import__(module_name)
            print(f"✅ {display_name} - OK")
        except ImportError:
            print(f"❌ {display_name} - FAILED")
            all_good = False
    
    # Test GPU (if available)
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA - Available (GPU: {torch.cuda.get_device_name(0)})")
        else:
            print("⚠️ CUDA - Not available (CPU mode)")
    except:
        print("❌ CUDA - Check failed")
        all_good = False
    
    return all_good

def create_quick_start_script():
    """Create a quick start script"""
    print("\n📝 Creating quick start script...")
    
    script_content = '''#!/usr/bin/env python3
"""
🚀 TEKNOFEST 2025 - Turkish LLM Quick Start
"""

import sys
from pathlib import Path

# Add project to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

def main():
    print("🔥 TEKNOFEST 2025 - Turkish LLM Training")
    print("="*50)
    
    try:
        # Import the complete pipeline
        from turkish_tokenizer.colab_qwen3_turkish_complete import ColabQwen3TurkishPipeline
        
        # Initialize pipeline
        print("📦 Initializing Turkish LLM pipeline...")
        pipeline = ColabQwen3TurkishPipeline()
        
        # Run training
        print("🚀 Starting training...")
        results = pipeline.run_complete_pipeline()
        
        # Show results
        print("\\n🎉 Training completed!")
        print(f"📊 Success: {results.get('success', False)}")
        print(f"📊 Final Loss: {results.get('final_loss', 'N/A')}")
        print(f"⏱️ Training Time: {results.get('training_time_hours', 'N/A'):.2f}h")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Try running: python install.py")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
'''
    
    with open("quick_start.py", "w", encoding="utf-8") as f:
        f.write(script_content)
    
    print("✅ Created quick_start.py")

def print_success_message():
    """Print final success message"""
    print("\n" + "🎉" * 80)
    print("🏆 INSTALLATION COMPLETED SUCCESSFULLY!")
    print("🎉" * 80)
    print()
    print("🚀 Your Turkish LLM training system is ready!")
    print()
    print("📋 Next steps:")
    print("   1. Run: python quick_start.py")
    print("   2. Or use Google Colab with the provided notebooks")
    print("   3. Check README.md for detailed documentation")
    print()
    print("🇹🇷 Ready for TEKNOFEST 2025 Turkish LLM development!")
    print("🎉" * 80)

def main():
    """Main installation process"""
    print_header()
    
    # Step 1: Check Python version
    check_python_version()
    
    # Step 2: Check GPU
    gpu_type = check_gpu()
    
    # Step 3: Install PyTorch
    if not install_pytorch():
        print("❌ Installation failed at PyTorch step")
        sys.exit(1)
    
    # Step 4: Install requirements
    if not install_requirements():
        print("❌ Installation failed at requirements step")
        sys.exit(1)
    
    # Step 5: Install optional optimizations
    install_optional_optimizations()
    
    # Step 6: Setup directories
    setup_directories()
    
    # Step 7: Verify installation
    if not verify_installation():
        print("⚠️ Installation completed with some issues")
        print("💡 Check error messages above and try manual installation")
    
    # Step 8: Create quick start script
    create_quick_start_script()
    
    # Step 9: Success message
    print_success_message()

if __name__ == "__main__":
    main()