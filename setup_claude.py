#!/usr/bin/env python
"""Claude Code için hafif setup scripti"""

import subprocess
import sys

def install_package(package):
    """Tek bir paketi yükle"""
    try:
        print(f"Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"[OK] {package} installed")
    except subprocess.CalledProcessError:
        print(f"[FAIL] Failed to install {package}")

# Temel paketler
core_packages = [
    "pandas",
    "numpy", 
    "requests",
    "python-dotenv",
    "pyyaml"
]

print("Installing core packages...")
for package in core_packages:
    install_package(package)

print("\n[COMPLETE] Core setup complete!")
print("Note: Install ML packages (torch, transformers) in Google Colab")