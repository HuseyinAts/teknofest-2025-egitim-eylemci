"""
TEKNOFEST 2025 - Model İndirme ve Optimizasyon
Bu script AI modelini indirir, optimize eder ve production'a hazırlar
"""

import os
import sys
import torch
import time
import json
from pathlib import Path
from typing import Optional, Dict, Any
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline
)
from huggingface_hub import snapshot_download, HfApi
import warnings
warnings.filterwarnings("ignore")

class ModelManager:
    """Model yönetimi ve optimizasyon"""
    
    def __init__(self):
        self.base_model = "Qwen/Qwen2.5-7B-Instruct"  # Güncel ve daha küçük model
        self.finetuned_model = "Huseyin/qwen3-8b-turkish-teknofest2025-private"
        self.cache_dir = Path("models")
        self.cache_dir.mkdir(exist_ok=True)
        
        # HuggingFace token from env
        self.hf_token = os.getenv("HUGGING_FACE_HUB_TOKEN", "")
        if not self.hf_token:
            print("⚠️ HUGGING_FACE_HUB_TOKEN bulunamadı!")
    
    def check_disk_space(self) -> bool:
        """Disk alanı kontrolü"""
        import shutil
        
        _, _, free = shutil.disk_usage("/")
        free_gb = free / (1024**3)
        
        print(f"💾 Boş disk alanı: {free_gb:.2f} GB")
        
        if free_gb < 20:
            print("❌ En az 20 GB boş alan gerekli!")
            return False
        return True
    
    def download_model(self, model_name: str = None) -> str:
        """Model indirme"""
        if model_name is None:
            model_name = self.base_model
        
        print(f"\n📥 Model indiriliyor: {model_name}")
        print("⏳ Bu işlem 10-30 dakika sürebilir...")
        
        model_path = self.cache_dir / model_name.replace("/", "_")
        
        if model_path.exists():
            print(f"✅ Model zaten mevcut: {model_path}")
            return str(model_path)
        
        try:
            # Download with progress
            snapshot_download(
                repo_id=model_name,
                cache_dir=str(self.cache_dir),
                local_dir=str(model_path),
                token=self.hf_token,
                resume_download=True,
                max_workers=4
            )
            
            print(f"✅ Model indirildi: {model_path}")
            return str(model_path)
            
        except Exception as e:
            print(f"❌ Model indirme hatası: {e}")
            return None
    
    def quantize_model(self, model_path: str, quantization: str = "int8") -> Any:
        """Model quantization for optimization"""
        print(f"\n🔧 Model optimize ediliyor ({quantization})...")
        
        try:
            if quantization == "int8":
                bnb_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    bnb_8bit_compute_dtype=torch.float16
                )
            elif quantization == "int4":
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch.float16
                )
            else:
                bnb_config = None
            
            # Load model with quantization
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )
            
            print(f"✅ Model optimize edildi!")
            
            # Calculate model size
            param_count = sum(p.numel() for p in model.parameters())
            size_mb = param_count * 2 / (1024**2)  # Assuming fp16
            print(f"📊 Model boyutu: {size_mb:.2f} MB")
            
            return model
            
        except Exception as e:
            print(f"❌ Quantization hatası: {e}")
            return None
    
    def create_pipeline(self, model, tokenizer) -> Any:
        """Create inference pipeline"""
        print("\n🚀 Inference pipeline oluşturuluyor...")
        
        try:
            pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=512,
                temperature=0.7,
                top_p=0.95,
                repetition_penalty=1.15
            )
            
            print("✅ Pipeline hazır!")
            return pipe
            
        except Exception as e:
            print(f"❌ Pipeline hatası: {e}")
            return None
    
    def test_model(self, pipe) -> bool:
        """Model test"""
        print("\n🧪 Model test ediliyor...")
        
        test_prompts = [
            "Matematik dersinde türev konusunu nasıl öğretirsin?",
            "10. sınıf fizik müfredatında hangi konular var?",
            "Öğrencilerin motivasyonunu nasıl artırabilirim?"
        ]
        
        for i, prompt in enumerate(test_prompts, 1):
            print(f"\n📝 Test {i}: {prompt}")
            
            try:
                start = time.time()
                response = pipe(prompt, max_new_tokens=100)
                elapsed = time.time() - start
                
                print(f"💬 Cevap: {response[0]['generated_text'][:200]}...")
                print(f"⏱️ Süre: {elapsed:.2f} saniye")
                
            except Exception as e:
                print(f"❌ Test hatası: {e}")
                return False
        
        return True
    
    def save_optimized_model(self, model, tokenizer, output_path: str = None):
        """Save optimized model"""
        if output_path is None:
            output_path = str(self.cache_dir / "optimized_model")
        
        print(f"\n💾 Optimize edilmiş model kaydediliyor: {output_path}")
        
        try:
            model.save_pretrained(output_path)
            tokenizer.save_pretrained(output_path)
            
            # Save config
            config = {
                "base_model": self.base_model,
                "optimization": "int8",
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "path": output_path
            }
            
            with open(Path(output_path) / "optimization_config.json", "w") as f:
                json.dump(config, f, indent=2)
            
            print("✅ Model kaydedildi!")
            return output_path
            
        except Exception as e:
            print(f"❌ Kaydetme hatası: {e}")
            return None
    
    def setup_model_serving(self):
        """Setup model serving with TorchServe or similar"""
        print("\n🌐 Model serving yapılandırılıyor...")
        
        serving_config = """
# TorchServe Configuration
inference_address=http://0.0.0.0:8080
management_address=http://0.0.0.0:8081
metrics_address=http://0.0.0.0:8082
number_of_netty_threads=4
job_queue_size=100
model_store=/models
model_snapshot={"name":"startup.cfg","modelCount":1,"models":{"teknofest_model":{"1.0":{"defaultVersion":true,"marName":"teknofest_model.mar","minWorkers":1,"maxWorkers":4,"batchSize":1,"maxBatchDelay":100,"responseTimeout":120}}}}
        """
        
        with open("model_serving_config.properties", "w") as f:
            f.write(serving_config)
        
        print("✅ Model serving config oluşturuldu!")
        
        # Create docker-compose for model serving
        docker_compose = """
version: '3.8'

services:
  model-server:
    image: pytorch/torchserve:latest-gpu
    container_name: teknofest-model-server
    ports:
      - "8080:8080"  # Inference
      - "8081:8081"  # Management
      - "8082:8082"  # Metrics
    volumes:
      - ./models:/models
      - ./model_serving_config.properties:/home/model-server/config.properties
    environment:
      - TS_CONFIG_FILE=/home/model-server/config.properties
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    restart: unless-stopped
        """
        
        with open("docker-compose.model.yml", "w") as f:
            f.write(docker_compose)
        
        print("✅ Docker compose model serving oluşturuldu!")

def create_model_api():
    """Create FastAPI wrapper for model"""
    
    api_code = '''"""
TEKNOFEST 2025 - Model API Service
FastAPI wrapper for AI model
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List
import torch
from transformers import pipeline
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="TEKNOFEST Model API", version="1.0.0")

# Global model instance
model_pipeline = None

class ModelRequest(BaseModel):
    prompt: str
    max_tokens: Optional[int] = 256
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.95

class ModelResponse(BaseModel):
    response: str
    tokens_used: int
    latency_ms: float
    model_name: str

@app.on_event("startup")
async def load_model():
    """Load model on startup"""
    global model_pipeline
    
    logger.info("Loading model...")
    try:
        model_pipeline = pipeline(
            "text-generation",
            model="models/optimized_model",
            device_map="auto",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        logger.info("Model loaded successfully!")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")

@app.get("/health")
async def health():
    """Health check"""
    return {
        "status": "healthy",
        "model_loaded": model_pipeline is not None,
        "cuda_available": torch.cuda.is_available()
    }

@app.post("/generate", response_model=ModelResponse)
async def generate_text(request: ModelRequest):
    """Generate text using the model"""
    if model_pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        start = time.time()
        
        # Generate response
        result = model_pipeline(
            request.prompt,
            max_new_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            do_sample=True
        )
        
        response_text = result[0]["generated_text"]
        # Remove the prompt from response
        if response_text.startswith(request.prompt):
            response_text = response_text[len(request.prompt):].strip()
        
        latency = (time.time() - start) * 1000
        
        return ModelResponse(
            response=response_text,
            tokens_used=len(response_text.split()),
            latency_ms=round(latency, 2),
            model_name="teknofest-model-v1"
        )
        
    except Exception as e:
        logger.error(f"Generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/batch_generate")
async def batch_generate(prompts: List[str], max_tokens: int = 256):
    """Batch text generation"""
    if model_pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    responses = []
    for prompt in prompts:
        try:
            result = model_pipeline(prompt, max_new_tokens=max_tokens)
            responses.append(result[0]["generated_text"])
        except Exception as e:
            responses.append(f"Error: {e}")
    
    return {"responses": responses}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8090)
'''
    
    with open("model_api_service.py", "w") as f:
        f.write(api_code)
    
    print("✅ Model API service oluşturuldu!")

def create_fallback_system():
    """Create fallback system for model failures"""
    
    fallback_code = '''"""
TEKNOFEST 2025 - Model Fallback System
Handles model failures gracefully
"""
import logging
from typing import Optional, Dict, Any
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class ModelProvider(ABC):
    """Abstract base class for model providers"""
    
    @abstractmethod
    async def generate(self, prompt: str, **kwargs) -> Dict[str, Any]:
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        pass

class PrimaryModel(ModelProvider):
    """Primary model (local)"""
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self._load_model()
    
    def _load_model(self):
        try:
            # Load your primary model here
            self.model = True  # Placeholder
            logger.info("Primary model loaded")
        except Exception as e:
            logger.error(f"Failed to load primary model: {e}")
            self.model = None
    
    async def generate(self, prompt: str, **kwargs) -> Dict[str, Any]:
        if not self.is_available():
            raise Exception("Primary model not available")
        
        # Generate with primary model
        return {
            "response": f"Primary model response to: {prompt}",
            "source": "primary"
        }
    
    def is_available(self) -> bool:
        return self.model is not None

class FallbackModel(ModelProvider):
    """Fallback model (API-based)"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.available = bool(self.api_key)
    
    async def generate(self, prompt: str, **kwargs) -> Dict[str, Any]:
        if not self.is_available():
            raise Exception("Fallback model not available")
        
        # Use OpenAI or another API as fallback
        import openai
        
        try:
            response = await openai.ChatCompletion.acreate(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=kwargs.get("max_tokens", 256)
            )
            
            return {
                "response": response.choices[0].message.content,
                "source": "fallback_api"
            }
        except Exception as e:
            logger.error(f"Fallback API error: {e}")
            raise
    
    def is_available(self) -> bool:
        return self.available

class RuleBasedFallback(ModelProvider):
    """Rule-based fallback (no model needed)"""
    
    def __init__(self):
        self.rules = self._load_rules()
    
    def _load_rules(self) -> Dict:
        """Load rule-based responses"""
        return {
            "matematik": "Matematik konusunda size yardımcı olabilirim.",
            "fizik": "Fizik dersi için çeşitli kaynaklarımız var.",
            "default": "Bu konuda size nasıl yardımcı olabilirim?"
        }
    
    async def generate(self, prompt: str, **kwargs) -> Dict[str, Any]:
        # Simple keyword matching
        prompt_lower = prompt.lower()
        
        for keyword, response in self.rules.items():
            if keyword in prompt_lower:
                return {
                    "response": response,
                    "source": "rule_based"
                }
        
        return {
            "response": self.rules["default"],
            "source": "rule_based"
        }
    
    def is_available(self) -> bool:
        return True  # Always available

class ModelOrchestrator:
    """Orchestrates multiple model providers with fallback"""
    
    def __init__(self):
        self.providers = [
            PrimaryModel("models/optimized_model"),
            FallbackModel(),
            RuleBasedFallback()  # Always available last resort
        ]
    
    async def generate(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Generate with fallback logic"""
        
        for i, provider in enumerate(self.providers):
            try:
                if provider.is_available():
                    logger.info(f"Using provider {i}: {provider.__class__.__name__}")
                    result = await provider.generate(prompt, **kwargs)
                    result["provider_index"] = i
                    return result
            except Exception as e:
                logger.warning(f"Provider {i} failed: {e}")
                continue
        
        # If all providers fail
        raise Exception("All model providers failed")
    
    def get_status(self) -> Dict[str, bool]:
        """Get status of all providers"""
        return {
            provider.__class__.__name__: provider.is_available()
            for provider in self.providers
        }

# Global orchestrator instance
orchestrator = ModelOrchestrator()

async def generate_with_fallback(prompt: str, **kwargs) -> Dict[str, Any]:
    """Main entry point for text generation with fallback"""
    return await orchestrator.generate(prompt, **kwargs)
'''
    
    with open("model_fallback_system.py", "w") as f:
        f.write(fallback_code)
    
    print("✅ Fallback system oluşturuldu!")

def main():
    """Ana fonksiyon"""
    
    print("🚀 TEKNOFEST 2025 - Model İndirme ve Optimizasyon")
    print("=" * 50)
    
    manager = ModelManager()
    
    # 1. Check disk space
    if not manager.check_disk_space():
        print("❌ Yetersiz disk alanı!")
        sys.exit(1)
    
    # 2. User selection
    print("\n📋 Model Seçenekleri:")
    print("1. Qwen2.5-7B-Instruct (Önerilen, 7GB)")
    print("2. Custom fine-tuned model")
    print("3. Sadece optimizasyon yap (model zaten var)")
    print("4. Test mode (model indirmeden)")
    
    choice = input("\nSeçiminiz (1-4): ").strip()
    
    if choice == "1":
        # Download and optimize base model
        model_path = manager.download_model()
        if model_path:
            # Load tokenizer
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                manager.base_model,
                trust_remote_code=True
            )
            
            # Quantize model
            model = manager.quantize_model(model_path, "int8")
            
            if model:
                # Create pipeline
                pipe = manager.create_pipeline(model, tokenizer)
                
                if pipe:
                    # Test model
                    if manager.test_model(pipe):
                        # Save optimized model
                        manager.save_optimized_model(model, tokenizer)
                        
                        # Setup serving
                        manager.setup_model_serving()
                        
                        # Create API service
                        create_model_api()
                        
                        # Create fallback system
                        create_fallback_system()
                        
                        print("\n" + "=" * 50)
                        print("✅ MODEL HAZIR!")
                        print("\n📝 Kullanım:")
                        print("  - Model API: python model_api_service.py")
                        print("  - Docker: docker-compose -f docker-compose.model.yml up")
                        print("  - Test: curl http://localhost:8090/health")
    
    elif choice == "2":
        # Download custom model
        model_name = input("Model adı (örn: username/model-name): ").strip()
        model_path = manager.download_model(model_name)
        # Continue with optimization...
    
    elif choice == "3":
        # Just optimize existing model
        model_path = input("Model path: ").strip()
        # Continue with optimization...
    
    elif choice == "4":
        # Test mode - create mock services
        print("\n🧪 Test mode - Mock servisler oluşturuluyor...")
        create_model_api()
        create_fallback_system()
        print("✅ Test servisleri hazır!")
    
    else:
        print("❌ Geçersiz seçim!")

if __name__ == "__main__":
    main()
