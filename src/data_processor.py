"""
Data Processing Module for TEKNOFEST 2025
Processes educational data from various sources
"""

import pandas as pd
import json
from pathlib import Path
import os
from typing import Dict, List, Optional, Any, Union
import re
import numpy as np
from src.container import singleton
from src.config import Settings

@singleton
class DataProcessor:
    """Process and prepare educational data for model training"""
    
    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings
        self.data_dir = Path("data")
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        
        # Create directories if they don't exist
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
    def clean_text(self, text: str) -> str:
        """Clean and normalize Turkish text"""
        if pd.isna(text):
            return ""
        
        # Basic cleaning
        text = str(text)
        text = text.strip()
        
        # Fix common Turkish character issues
        replacements = {
            'ý': 'ı',
            'þ': 'ş',
            'ð': 'ğ',
            'Ý': 'İ',
            'Þ': 'Ş',
            'Ð': 'Ğ'
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
            
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        return text
    
    def process_quiz_data(self) -> pd.DataFrame:
        """Process Turkish Quiz Instruct dataset"""
        quiz_path = self.raw_dir / "turkish_quiz_instruct.csv"
        
        if not quiz_path.exists():
            print(f"[WARNING] {quiz_path} not found")
            return pd.DataFrame()
        
        print(f"[INFO] Processing quiz data from {quiz_path}")
        df = pd.read_csv(quiz_path)
        
        # Clean text columns
        text_columns = ['content', 'multiple_questions', 'short_questions']
        for col in text_columns:
            if col in df.columns:
                df[col] = df[col].apply(self.clean_text)
        
        # Create instruction-response pairs for fine-tuning
        processed_data = []
        
        for _, row in df.iterrows():
            # Multiple choice questions
            if pd.notna(row.get('multiple_questions')):
                processed_data.append({
                    'instruction': f"Konu: {row.get('subject', 'Genel')} - {row.get('subsubtopic', '')}\\n{row.get('content', '')}",
                    'input': row['multiple_questions'].split('\\n')[0] if '\\n' in row['multiple_questions'] else row['multiple_questions'],
                    'output': row['multiple_questions'].split('Right Answer: ')[-1] if 'Right Answer:' in row['multiple_questions'] else '',
                    'type': 'multiple_choice',
                    'subject': row.get('subject', 'Genel')
                })
            
            # Short answer questions
            if pd.notna(row.get('short_questions')):
                processed_data.append({
                    'instruction': f"Konu: {row.get('subject', 'Genel')} - {row.get('subsubtopic', '')}\\n{row.get('content', '')}",
                    'input': row['short_questions'].split('\\n')[0] if '\\n' in row['short_questions'] else row['short_questions'],
                    'output': row['short_questions'].split('Right Answer: ')[-1] if 'Right Answer:' in row['short_questions'] else '',
                    'type': 'short_answer',
                    'subject': row.get('subject', 'Genel')
                })
        
        return pd.DataFrame(processed_data)
    
    def process_eba_pdfs(self) -> pd.DataFrame:
        """Process EBA PDF files (placeholder for future implementation)"""
        print("[INFO] EBA PDF processing not yet implemented")
        
        # Placeholder for PDF processing
        # Will need PyPDF2 or pdfplumber
        """
        import PyPDF2
        
        eba_dir = self.raw_dir / "eba"
        processed_data = []
        
        for pdf_file in eba_dir.glob("*.pdf"):
            with open(pdf_file, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text()
                
                # Process extracted text
                processed_data.append({
                    'content': self.clean_text(text),
                    'source': pdf_file.name
                })
        """
        
        return pd.DataFrame()
    
    def process_custom_data(self) -> pd.DataFrame:
        """Process custom educational data"""
        custom_data = []
        
        # Example custom data for Turkish education
        sample_topics = [
            {
                'instruction': 'Matematik konusunu açıkla',
                'input': 'Pisagor teoremi nedir?',
                'output': 'Pisagor teoremi, bir dik üçgende hipotenüsün karesinin, dik kenarların karelerinin toplamına eşit olduğunu belirtir: a² + b² = c²',
                'type': 'explanation',
                'subject': 'Matematik'
            },
            {
                'instruction': 'Türkçe dilbilgisi sorusu',
                'input': 'Fiil nedir ve türleri nelerdir?',
                'output': 'Fiil, bir işi, oluşu veya durumu bildiren sözcüktür. Türleri: 1) Yapısına göre (basit, türemiş, birleşik), 2) Anlamına göre (iş, oluş, durum)',
                'type': 'grammar',
                'subject': 'Türkçe'
            },
            {
                'instruction': 'Fen Bilimleri konusu',
                'input': 'Fotosentez nedir?',
                'output': 'Fotosentez, bitkilerin güneş ışığını kullanarak karbondioksit ve sudan besin ve oksijen ürettiği yaşamsal olaydır.',
                'type': 'explanation',
                'subject': 'Fen Bilimleri'
            }
        ]
        
        return pd.DataFrame(sample_topics)
    
    def create_training_dataset(self, sample_size: int = 10000) -> pd.DataFrame:
        """Create competition dataset with 10K samples"""
        print("[INFO] Creating training dataset...")
        
        datasets = []
        
        # 1. Process Turkish Quiz Instruct
        quiz_data = self.process_quiz_data()
        if not quiz_data.empty:
            datasets.append(quiz_data)
            print(f"[OK] Processed {len(quiz_data)} quiz examples")
        
        # 2. Process custom data
        custom_data = self.process_custom_data()
        if not custom_data.empty:
            datasets.append(custom_data)
            print(f"[OK] Added {len(custom_data)} custom examples")
        
        # 3. Process EBA PDFs (when available)
        eba_data = self.process_eba_pdfs()
        if not eba_data.empty:
            datasets.append(eba_data)
            print(f"[OK] Processed {len(eba_data)} EBA examples")
        
        # Combine all datasets
        if not datasets:
            print("[ERROR] No data to process")
            return pd.DataFrame()
        
        combined = pd.concat(datasets, ignore_index=True)
        print(f"[INFO] Total examples: {len(combined)}")
        
        # Sample for competition (or use all if less than sample_size)
        if len(combined) > sample_size:
            final_dataset = combined.sample(n=sample_size, random_state=42)
            print(f"[INFO] Sampled {sample_size} examples for competition")
        else:
            final_dataset = combined
            print(f"[INFO] Using all {len(final_dataset)} examples")
        
        # Save in multiple formats
        output_path = self.processed_dir / "competition_dataset"
        
        # JSON format
        final_dataset.to_json(
            f"{output_path}.json",
            orient='records',
            force_ascii=False,
            indent=2
        )
        print(f"[OK] Saved JSON to {output_path}.json")
        
        # CSV format
        final_dataset.to_csv(
            f"{output_path}.csv",
            index=False,
            encoding='utf-8'
        )
        print(f"[OK] Saved CSV to {output_path}.csv")
        
        # JSONL format (for fine-tuning)
        with open(f"{output_path}.jsonl", 'w', encoding='utf-8') as f:
            for _, row in final_dataset.iterrows():
                f.write(json.dumps(row.to_dict(), ensure_ascii=False) + '\\n')
        print(f"[OK] Saved JSONL to {output_path}.jsonl")
        
        return final_dataset
    
    def get_dataset_statistics(self, df: pd.DataFrame) -> Dict:
        """Get statistics about the dataset"""
        stats = {
            'total_examples': len(df),
            'subjects': df['subject'].value_counts().to_dict() if 'subject' in df.columns else {},
            'types': df['type'].value_counts().to_dict() if 'type' in df.columns else {},
            'avg_instruction_length': df['instruction'].str.len().mean() if 'instruction' in df.columns else 0,
            'avg_output_length': df['output'].str.len().mean() if 'output' in df.columns else 0,
        }
        return stats
    
    def process_raw_data(self, raw_data: Optional[Dict]) -> Optional[Dict]:
        """Process raw data and return processed version"""
        if raw_data is None:
            return None
        
        if not raw_data:
            return {}
        
        processed = raw_data.copy()
        
        # Process text fields
        if 'text' in processed:
            processed['text'] = self.clean_text(processed['text'])
        
        # Process scores if present
        if 'scores' in processed and isinstance(processed['scores'], list):
            processed['normalized_scores'] = self.normalize_scores(processed['scores'])
            processed['score_stats'] = self.calculate_statistics(processed['scores'])
        
        # Process grades if present
        if 'grades' in processed:
            processed['average_grade'] = np.mean(processed['grades'])
        
        return processed
    
    def validate_data(self, data: Dict) -> bool:
        """Validate data structure and content"""
        if not isinstance(data, dict):
            return False
        
        # Check required fields
        if 'student_id' not in data or not data['student_id']:
            return False
        
        # Validate grade
        if 'grade' in data:
            if not isinstance(data['grade'], (int, float)):
                return False
            if data['grade'] < 1 or data['grade'] > 12:
                return False
        
        # Validate score
        if 'score' in data:
            if not isinstance(data['score'], (int, float)):
                return False
            if data['score'] < 0:
                return False
        
        return True
    
    def normalize_scores(self, scores: List[float]) -> List[float]:
        """Normalize scores to 0-1 range"""
        if not scores:
            return []
        
        scores_array = np.array(scores)
        min_score = scores_array.min()
        max_score = scores_array.max()
        
        if max_score == min_score:
            return [0.5] * len(scores)
        
        normalized = (scores_array - min_score) / (max_score - min_score)
        return normalized.tolist()
    
    def calculate_statistics(self, data: List[float]) -> Dict[str, float]:
        """Calculate statistical measures for data"""
        if not data:
            return {
                'mean': 0,
                'median': 0,
                'std': 0,
                'min': 0,
                'max': 0,
                'count': 0
            }
        
        data_array = np.array(data)
        
        return {
            'mean': float(data_array.mean()),
            'median': float(np.median(data_array)),
            'std': float(data_array.std()),
            'min': float(data_array.min()),
            'max': float(data_array.max()),
            'count': len(data)
        }
    
    def process_batch(self, batch_data: List[Dict]) -> Optional[List[Dict]]:
        """Process a batch of data items"""
        if batch_data is None:
            return None
        
        processed_batch = []
        for item in batch_data:
            try:
                processed_item = self.process_raw_data(item)
                if processed_item is not None:
                    processed_batch.append(processed_item)
            except Exception as e:
                print(f"[WARNING] Error processing item: {e}")
                continue
        
        return processed_batch
    
    def load_from_file(self, file_path: str) -> Optional[Union[List, Dict]]:
        """Load data from a JSON file"""
        try:
            path = Path(file_path)
            if not path.exists():
                return None
            
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            return data
        except Exception as e:
            print(f"[ERROR] Failed to load file {file_path}: {e}")
            return None


def main():
    """Main function to run data processing"""
    processor = DataProcessor()
    
    # Create training dataset
    dataset = processor.create_training_dataset(sample_size=5000)  # Start with 5K for testing
    
    if not dataset.empty:
        # Show statistics
        stats = processor.get_dataset_statistics(dataset)
        print("\\n[STATISTICS]")
        print(f"Total examples: {stats['total_examples']}")
        print(f"Subjects: {stats['subjects']}")
        print(f"Types: {stats['types']}")
        print(f"Avg instruction length: {stats['avg_instruction_length']:.0f} chars")
        print(f"Avg output length: {stats['avg_output_length']:.0f} chars")
        
        # Show sample
        print("\\n[SAMPLE DATA]")
        sample = dataset.head(1).iloc[0]
        print(f"Instruction: {sample.get('instruction', '')[:100]}...")
        print(f"Input: {sample.get('input', '')[:100]}...")
        print(f"Output: {sample.get('output', '')[:100]}...")


if __name__ == "__main__":
    main()