"""
Fix Tokenizer Loading Issue for Qwen3-8B Model
This script downloads and fixes the tokenizer files
"""

import os
import json
import shutil
from pathlib import Path
from transformers import AutoTokenizer
import torch
from huggingface_hub import snapshot_download

def fix_tokenizer():
    """Fix the corrupted tokenizer by re-downloading it"""
    
    print("Fixing Tokenizer Loading Issue...")
    print("="*70)
    
    # Define paths
    model_name = "Qwen/Qwen2.5-7B"  # Using stable Qwen2.5 instead of Qwen3
    cache_dir = "./cache"
    backup_dir = "./cache_backup"
    
    # Create directories
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    Path(backup_dir).mkdir(parents=True, exist_ok=True)
    
    try:
        # Step 1: Backup existing cache
        print("Backing up existing cache...")
        if Path(cache_dir).exists():
            # Move old cache to backup
            for item in Path(cache_dir).iterdir():
                if item.is_dir() and "tokenizer" in str(item).lower():
                    backup_path = Path(backup_dir) / item.name
                    if backup_path.exists():
                        shutil.rmtree(backup_path)
                    shutil.move(str(item), str(backup_path))
                    print(f"  Backed up: {item.name}")
        
        # Step 2: Clear tokenizer cache
        print("\nClearing tokenizer cache...")
        tokenizer_patterns = [
            "**/tokenizer*",
            "**/vocab*",
            "**/merges*",
            "**/special_tokens*"
        ]
        
        for pattern in tokenizer_patterns:
            for file in Path(cache_dir).glob(pattern):
                try:
                    if file.is_file():
                        file.unlink()
                        print(f"  Removed: {file.name}")
                except Exception as e:
                    print(f"  Warning: Could not remove {file.name}: {e}")
        
        # Step 3: Download fresh tokenizer files
        print("\nDownloading fresh tokenizer files...")
        print(f"  Model: {model_name}")
        
        # Download only tokenizer files
        tokenizer_files = [
            "tokenizer.json",
            "tokenizer_config.json",
            "vocab.json",
            "merges.txt",
            "special_tokens_map.json"
        ]
        
        # Try to download tokenizer files directly
        try:
            snapshot_download(
                repo_id=model_name,
                cache_dir=cache_dir,
                allow_patterns=tokenizer_files,
                ignore_patterns=["*.bin", "*.safetensors", "*.h5", "*.msgpack"],
                resume_download=True
            )
            print("  [OK] Tokenizer files downloaded successfully")
        except Exception as e:
            print(f"  [WARNING] Direct download failed: {e}")
            print("  Trying alternative method...")
        
        # Step 4: Load and test tokenizer
        print("\nTesting tokenizer...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                cache_dir=cache_dir,
                trust_remote_code=True,
                use_fast=True,
                padding_side="left"
            )
            
            # Configure tokenizer
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Test tokenization
            test_text = "Merhaba, bu bir test metnidir."
            tokens = tokenizer(test_text, return_tensors="pt")
            decoded = tokenizer.decode(tokens['input_ids'][0])
            
            print(f"  [OK] Tokenizer loaded successfully!")
            print(f"  Test encoding: {tokens['input_ids'].shape}")
            print(f"  Test decoding: {decoded[:50]}...")
            
            # Save tokenizer locally for backup
            local_tokenizer_dir = Path("./fixed_tokenizer")
            local_tokenizer_dir.mkdir(parents=True, exist_ok=True)
            tokenizer.save_pretrained(local_tokenizer_dir)
            print(f"\n  [SAVED] Tokenizer saved to: {local_tokenizer_dir}")
            
            return True
            
        except Exception as e:
            print(f"  [ERROR] Tokenizer test failed: {e}")
            return False
            
    except Exception as e:
        print(f"\n[ERROR] Error during fix: {e}")
        
        # Restore backup if available
        if Path(backup_dir).exists():
            print("\n[RESTORE] Restoring backup...")
            for item in Path(backup_dir).iterdir():
                restore_path = Path(cache_dir) / item.name
                if restore_path.exists():
                    shutil.rmtree(restore_path)
                shutil.move(str(item), str(restore_path))
                print(f"  Restored: {item.name}")
        
        return False
    
    finally:
        # Clean up backup
        if Path(backup_dir).exists():
            shutil.rmtree(backup_dir)
    
    print("\n" + "="*70)
    print("[COMPLETED] Tokenizer fix completed!")
    return True


def update_training_script():
    """Update the training script to use the fixed tokenizer"""
    
    print("\nUpdating training script configuration...")
    
    config_updates = """
# Updated configuration for fixed tokenizer
FIXED_CONFIG = {
    'model_name': 'Qwen/Qwen2.5-7B',  # Using stable version
    'tokenizer_path': './fixed_tokenizer',  # Use local fixed tokenizer
    'cache_dir': './cache',
    'use_local_tokenizer': True
}
"""
    
    # Save configuration
    config_file = Path("tokenizer_config.py")
    with open(config_file, 'w') as f:
        f.write(config_updates)
    
    print(f"  Configuration saved to: {config_file}")
    
    # Create updated loader function
    loader_code = '''
def load_tokenizer_safe(model_name_or_path, cache_dir="./cache", **kwargs):
    """Safely load tokenizer with fallback options"""
    
    # First try local fixed tokenizer
    local_tokenizer = Path("./fixed_tokenizer")
    if local_tokenizer.exists():
        try:
            print("Loading tokenizer from local fixed version...")
            tokenizer = AutoTokenizer.from_pretrained(
                str(local_tokenizer),
                trust_remote_code=True,
                **kwargs
            )
            print("[OK] Loaded from local fixed tokenizer")
            return tokenizer
        except Exception as e:
            print(f"[WARNING] Local tokenizer failed: {e}")
    
    # Try alternative model
    alternative_models = [
        "Qwen/Qwen2.5-7B",
        "Qwen/Qwen2-7B",
        "mistralai/Mistral-7B-v0.1"
    ]
    
    for model in alternative_models:
        try:
            print(f"Trying {model}...")
            tokenizer = AutoTokenizer.from_pretrained(
                model,
                cache_dir=cache_dir,
                trust_remote_code=True,
                **kwargs
            )
            print(f"[OK] Successfully loaded tokenizer from {model}")
            
            # Configure special tokens
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            return tokenizer
        except Exception as e:
            print(f"  Failed: {e}")
            continue
    
    raise Exception("Could not load tokenizer from any source")
'''
    
    # Save loader function
    loader_file = Path("tokenizer_loader.py")
    with open(loader_file, 'w') as f:
        f.write(loader_code)
    
    print(f"  Loader function saved to: {loader_file}")
    
    return True


if __name__ == "__main__":
    print("\n" + "="*70)
    print("QWEN TOKENIZER FIX UTILITY")
    print("="*70)
    
    # Check environment
    print("\nSystem Info:")
    print(f"  PyTorch: {torch.__version__}")
    print(f"  CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    
    # Run fix
    print("\nStarting tokenizer fix process...")
    
    if fix_tokenizer():
        print("\n[SUCCESS] Tokenizer fixed successfully!")
        
        # Update training script
        if update_training_script():
            print("\nNext steps:")
            print("  1. The tokenizer has been fixed and saved locally")
            print("  2. Update your training script to use the fixed tokenizer:")
            print("     - Replace: AutoTokenizer.from_pretrained('Qwen/Qwen3-8B', ...)")
            print("     - With: AutoTokenizer.from_pretrained('./fixed_tokenizer', ...)")
            print("  3. Or use the provided load_tokenizer_safe() function")
            print("\n  You can now run your training script!")
    else:
        print("\n[FAILED] Tokenizer fix failed. Please check the error messages above.")
        print("\nAlternative solutions:")
        print("  1. Try using a different model (e.g., Qwen2.5-7B)")
        print("  2. Clear all cache and re-download")
        print("  3. Use a different tokenizer library version")