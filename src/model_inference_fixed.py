"""
Model Inference Module for TEKNOFEST 2025
Handles loading and inference with the fine-tuned Qwen model
"""

import os
import sys
# Fix encoding for Windows
if sys.platform == 'win32':
    import locale
    if locale.getpreferredencoding().upper() != 'UTF-8':
        os.environ['PYTHONIOENCODING'] = 'utf-8'

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional, Dict, List
import yaml
from pathlib import Path
import logging
from dotenv import load_dotenv

# Load environment variables for Hugging Face token
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TeknofestModel:
    """Fine-tuned Qwen model for Turkish educational content"""
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        """Initialize the model with configuration"""
        self.config = self._load_config(config_path)
        self.model = None
        self.tokenizer = None
        self.device = None
        self._setup_model()
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML and override with environment variables."""
        config_file = Path(config_path)
        if not config_file.exists():
            config = {}
        else:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)

        # Ensure 'model' key exists
        if 'model' not in config:
            config['model'] = {}

        # Override with environment variables
        config['model']['base_model'] = os.getenv('MODEL_NAME', config['model'].get('base_model', 'Huseyin/qwen3-8b-turkish-teknofest2025-private'))
        config['model']['max_length'] = int(os.getenv('MODEL_MAX_LENGTH', config['model'].get('max_length', 2048)))
        config['model']['temperature'] = float(os.getenv('MODEL_TEMPERATURE', config['model'].get('temperature', 0.7)))
        config['model']['device'] = os.getenv('MODEL_DEVICE', config['model'].get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        # Disable 8-bit loading for compatibility
        config['model']['load_in_8bit'] = False
        
        return config
    
    def _setup_model(self):
        """Load the model and tokenizer"""
        model_config = self.config['model']
        model_name = model_config['base_model']
        
        logger.info(f"Loading model: {model_name}")
        
        # Set device
        self.device = model_config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Get HF token from environment
        hf_token = os.getenv("HUGGING_FACE_HUB_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
        if hf_token:
            logger.info(f"Using HF token: hf_...{hf_token[-4:]}")
        
        try:
            # Load tokenizer
            logger.info("Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True,
                token=hf_token if hf_token else None
            )
            
            # Set padding token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model without quantization for better compatibility
            logger.info("Loading model...")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32,
                token=hf_token if hf_token else None,
                device_map='auto' if self.device == 'cuda' else None,
                low_cpu_mem_usage=True
            )
            
            # Move to device if not using device_map
            if self.device == 'cpu':
                self.model = self.model.to(self.device)
            
            self.model.eval()
            logger.info(f"Model loaded successfully on {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            logger.error("Please check:")
            logger.error("1. Internet connection is available")
            logger.error("2. Hugging Face token is valid in .env file")
            logger.error("3. Model name is correct")
            logger.error("4. Sufficient memory is available")
            raise
    
    def generate_response(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: Optional[float] = None,
        top_p: float = 0.9,
        do_sample: bool = True
    ) -> str:
        """Generate response from the model"""
        
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        # Use config temperature if not specified
        if temperature is None:
            temperature = self.config['model'].get('temperature', 0.7)
        
        # Tokenize input
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.config['model']['max_length']
        )
        
        # Move inputs to device
        if self.device == 'cuda':
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode response
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Remove the input prompt from response
        if response.startswith(prompt):
            response = response[len(prompt):].strip()
        
        return response
    
    def generate_educational_content(
        self,
        subject: str,
        topic: str,
        grade_level: int,
        content_type: str = "explanation",
        learning_style: str = "visual"
    ) -> str:
        """Generate educational content based on parameters"""
        
        # Format prompt based on content type
        if content_type == "explanation":
            prompt = f"""Sen Türkçe eğitim asistanısın. {grade_level}. sınıf {subject} dersinden {topic} konusunu {learning_style} öğrenme stiline uygun şekilde açıkla.

Konu: {topic}
Açıklama:"""
        
        elif content_type == "quiz":
            prompt = f"""Sen Türkçe eğitim asistanısın. {grade_level}. sınıf {subject} dersinden {topic} konusu için çoktan seçmeli bir soru hazırla.

Konu: {topic}
Soru:"""
        
        elif content_type == "example":
            prompt = f"""Sen Türkçe eğitim asistanısın. {grade_level}. sınıf {subject} dersinden {topic} konusu için günlük hayattan bir örnek ver.

Konu: {topic}
Örnek:"""
        
        else:
            prompt = f"""Sen Türkçe eğitim asistanısın. {grade_level}. sınıf {subject} dersinden {topic} konusu hakkında bilgi ver.

Konu: {topic}
Bilgi:"""
        
        return self.generate_response(prompt)
    
    def answer_question(
        self,
        question: str,
        context: Optional[str] = None,
        subject: Optional[str] = None
    ) -> str:
        """Answer a student's question"""
        
        if context:
            prompt = f"""Sen Türkçe eğitim asistanısın. Aşağıdaki bağlam doğrultusunda soruyu cevapla.

Bağlam: {context}

Soru: {question}
Cevap:"""
        elif subject:
            prompt = f"""Sen Türkçe eğitim asistanısın. {subject} dersi kapsamında soruyu cevapla.

Soru: {question}
Cevap:"""
        else:
            prompt = f"""Sen Türkçe eğitim asistanısın. Öğrencinin sorusunu cevapla.

Soru: {question}
Cevap:"""
        
        return self.generate_response(prompt)
    
    def provide_feedback(
        self,
        student_answer: str,
        correct_answer: str,
        question: str
    ) -> Dict[str, str]:
        """Provide feedback on student's answer"""
        
        prompt = f"""Sen Türkçe eğitim asistanısın. Öğrencinin cevabını değerlendir ve geri bildirim ver.

Soru: {question}
Doğru Cevap: {correct_answer}
Öğrenci Cevabı: {student_answer}

Değerlendirme:"""
        
        feedback = self.generate_response(prompt, max_new_tokens=256)
        
        # Simple correctness check
        is_correct = student_answer.lower().strip() == correct_answer.lower().strip()
        
        return {
            'is_correct': is_correct,
            'feedback': feedback,
            'suggestion': 'Harika!' if is_correct else 'Tekrar dene!'
        }
    
    def create_study_material(
        self,
        topics: List[str],
        subject: str,
        grade_level: int,
        material_type: str = "summary"
    ) -> str:
        """Create study materials for given topics"""
        
        topics_str = ", ".join(topics)
        
        if material_type == "summary":
            prompt = f"""Sen Türkçe eğitim asistanısın. {grade_level}. sınıf {subject} dersinden aşağıdaki konuların özet çalışma notlarını hazırla.

Konular: {topics_str}

Çalışma Notları:"""
        
        elif material_type == "practice":
            prompt = f"""Sen Türkçe eğitim asistanısın. {grade_level}. sınıf {subject} dersinden aşağıdaki konular için alıştırma soruları hazırla.

Konular: {topics_str}

Alıştırma Soruları:"""
        
        else:
            prompt = f"""Sen Türkçe eğitim asistanısın. {grade_level}. sınıf {subject} dersinden aşağıdaki konular için çalışma materyali hazırla.

Konular: {topics_str}

Materyal:"""
        
        return self.generate_response(prompt, max_new_tokens=1024)


# Singleton instance
_model_instance = None


def get_model() -> TeknofestModel:
    """Get or create the model singleton instance"""
    global _model_instance
    if _model_instance is None:
        _model_instance = TeknofestModel()
    return _model_instance


# Testing function
if __name__ == "__main__":
    # Test the model
    print("Initializing TEKNOFEST model...")
    try:
        model = get_model()
        
        # Test explanation generation
        print("\n=== Test 1: Generate Explanation ===")
        response = model.generate_educational_content(
            subject="Matematik",
            topic="Pisagor Teoremi",
            grade_level=9,
            content_type="explanation",
            learning_style="visual"
        )
        print(f"Response: {response[:200]}...")
        
        # Test question answering
        print("\n=== Test 2: Answer Question ===")
        answer = model.answer_question(
            question="Pisagor teoremi nedir?",
            subject="Matematik"
        )
        print(f"Answer: {answer[:200]}...")
        
        # Test quiz generation
        print("\n=== Test 3: Generate Quiz ===")
        quiz = model.generate_educational_content(
            subject="Fizik",
            topic="Newton'un Hareket Yasaları",
            grade_level=10,
            content_type="quiz"
        )
        print(f"Quiz: {quiz[:200]}...")
        
        print("\n[SUCCESS] Model tests completed!")
    except Exception as e:
        print(f"\n[ERROR] Model test failed: {e}")
        print("\nTroubleshooting tips:")
        print("1. Check your internet connection")
        print("2. Verify HUGGING_FACE_HUB_TOKEN in .env file")
        print("3. Ensure you have sufficient disk space and memory")
        print("4. Try running: py -m pip install --upgrade transformers torch")