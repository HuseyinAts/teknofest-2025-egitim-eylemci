#!/usr/bin/env python
"""Download Turkish educational datasets"""

import os
import pandas as pd
from datasets import load_dataset

def download_turkish_quiz_dataset():
    """Download Turkish Quiz Instruct dataset from Hugging Face"""
    print("Downloading Turkish Quiz Instruct dataset...")
    
    try:
        # Load dataset
        dataset = load_dataset("Kamyar-zeinalipour/Turkish-Quiz-Instruct")
        
        # Get train split
        train_data = dataset['train']
        print(f"[OK] Loaded {len(train_data)} examples")
        
        # Convert to DataFrame
        df = pd.DataFrame(train_data)
        
        # Save to CSV
        output_dir = "data/raw"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "turkish_quiz_instruct.csv")
        
        df.to_csv(output_path, index=False, encoding='utf-8')
        print(f"[OK] Saved to {output_path}")
        
        # Show statistics
        print(f"\nDataset Statistics:")
        print(f"- Total examples: {len(df)}")
        print(f"- Columns: {list(df.columns)}")
        print(f"- File size: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")
        
        return df
        
    except Exception as e:
        print(f"[ERROR] Failed to download dataset: {e}")
        return None

if __name__ == "__main__":
    df = download_turkish_quiz_dataset()
    
    if df is not None:
        print("\n[SUCCESS] Data collection complete!")
        print("\nFirst example:")
        print(df.iloc[0].to_dict())