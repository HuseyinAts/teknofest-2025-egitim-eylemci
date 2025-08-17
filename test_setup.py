"""Setup'ın çalıştığını test et"""

def test_imports():
    """Temel import'ları test et"""
    try:
        import pandas as pd
        import numpy as np
        print("[OK] Pandas and NumPy imported")
        
        # Basit bir işlem
        df = pd.DataFrame({'test': [1, 2, 3]})
        print(f"[OK] Created DataFrame with {len(df)} rows")
        
        return True
    except ImportError as e:
        print(f"[ERROR] Import error: {e}")
        return False

if __name__ == "__main__":
    if test_imports():
        print("\n[SUCCESS] Setup successful!")
    else:
        print("\n[FAILED] Setup failed")