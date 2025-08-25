"""
Qwen3-8B Modelini Turkish Mixtral Tokenizer ile Eğitme Script'i
Bu script, Turkish tokenizer'ı kullanarak Qwen modelini fine-tune eder.
"""

import torch
import json
from pathlib import Path
from typing import List, Dict, Optional
import logging
from dataclasses import dataclass
from torch.utils.data import Dataset, DataLoader

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType

from src.qwen_turkish_tokenizer_adapter import QwenTurkishTokenizerAdapter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Model ve training konfigürasyonu"""
    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    turkish_tokenizer_path: str = "notebooks/turkish_mixtral_v3_fixed.model"
    use_4bit: bool = True
    use_lora: bool = True
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    max_length: int = 512
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    num_epochs: int = 3
    warmup_steps: int = 100
    output_dir: str = "./qwen-turkish-finetuned"
    

class TurkishDataset(Dataset):
    """Turkish tokenizer kullanan custom dataset"""
    
    def __init__(self, data_path: str, adapter: QwenTurkishTokenizerAdapter, max_length: int = 512):
        self.adapter = adapter
        self.max_length = max_length
        self.data = self._load_data(data_path)
        
    def _load_data(self, data_path: str) -> List[Dict]:
        """Veriyi yükle"""
        data = []
        
        # JSONL formatında veri oku
        if Path(data_path).exists():
            with open(data_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        item = json.loads(line.strip())
                        data.append(item)
                    except:
                        continue
        else:
            # Demo veri oluştur
            logger.warning(f"Veri dosyası bulunamadı: {data_path}. Demo veri kullanılıyor.")
            data = [
                {"instruction": "Python'da liste oluşturma", "response": "Python'da liste oluşturmak için köşeli parantez [] kullanılır. Örnek: my_list = [1, 2, 3, 4, 5]"},
                {"instruction": "Yapay zeka nedir?", "response": "Yapay zeka, makinelerin insan benzeri zeka göstermesini sağlayan teknolojilerin genel adıdır."},
                {"instruction": "Machine Learning ve Deep Learning farkı", "response": "Machine Learning, veriden öğrenen algoritmalar içerir. Deep Learning ise çok katmanlı sinir ağları kullanan ML'in bir alt dalıdır."},
                {"instruction": "Türkiye'nin başkenti neresidir?", "response": "Türkiye'nin başkenti Ankara'dır."},
                {"instruction": "Öğrenciler için etkili çalışma yöntemleri", "response": "Etkili çalışma için: düzenli program, kısa molalar, aktif öğrenme, tekrar ve uygulama önemlidir."},
            ]
            
        return data
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Instruction-response formatında birleştir
        if "instruction" in item and "response" in item:
            text = f"### Soru: {item['instruction']}\n### Cevap: {item['response']}"
        elif "text" in item:
            text = item["text"]
        else:
            text = str(item)
            
        # Turkish tokenizer ile işle
        inputs = self.adapter.prepare_input_for_model(text, max_length=self.max_length)
        
        # Labels'ı input_ids'in kopyası yap (causal LM için)
        inputs['labels'] = inputs['input_ids'].clone()
        
        return inputs


class TurkishTokenizerTrainer:
    """Qwen modelini Turkish tokenizer ile eğiten sınıf"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Adapter'ı oluştur
        self.adapter = QwenTurkishTokenizerAdapter(
            qwen_model_path=config.model_name,
            turkish_tokenizer_path=config.turkish_tokenizer_path
        )
        
        # Model ve tokenizer'ı yükle
        self._load_model()
        
    def _load_model(self):
        """Model'i yükle ve konfigüre et"""
        logger.info(f"Loading model: {self.config.model_name}")
        
        # 4-bit quantization config
        if self.config.use_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            )
        else:
            bnb_config = None
            
        # Model'i yükle
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            quantization_config=bnb_config,
            device_map="auto" if self.config.use_4bit else None,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )
        
        # LoRA uygula
        if self.config.use_lora:
            logger.info("Applying LoRA...")
            
            # Model'i LoRA için hazırla
            if self.config.use_4bit:
                self.model = prepare_model_for_kbit_training(self.model)
                
            lora_config = LoraConfig(
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", 
                               "gate_proj", "up_proj", "down_proj"],
                lora_dropout=self.config.lora_dropout,
                bias="none",
                task_type=TaskType.CAUSAL_LM
            )
            
            self.model = get_peft_model(self.model, lora_config)
            
            # İstatistikleri yazdır
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            all_params = sum(p.numel() for p in self.model.parameters())
            
            logger.info(f"Trainable params: {trainable_params:,} ({100 * trainable_params / all_params:.2f}%)")
            logger.info(f"All params: {all_params:,}")
            
        logger.info("✅ Model loaded successfully")
        
    def train(self, train_data_path: str, eval_data_path: Optional[str] = None):
        """Model'i eğit"""
        logger.info("Starting training...")
        
        # Dataset'leri oluştur
        train_dataset = TurkishDataset(train_data_path, self.adapter, self.config.max_length)
        eval_dataset = TurkishDataset(eval_data_path, self.adapter, self.config.max_length) if eval_data_path else None
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            warmup_steps=self.config.warmup_steps,
            num_train_epochs=self.config.num_epochs,
            learning_rate=self.config.learning_rate,
            fp16=True,
            logging_steps=10,
            save_strategy="steps",
            save_steps=100,
            evaluation_strategy="steps" if eval_dataset else "no",
            eval_steps=100 if eval_dataset else None,
            load_best_model_at_end=True if eval_dataset else False,
            report_to="none",
            remove_unused_columns=False,
            dataloader_drop_last=True,
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.adapter.qwen_tokenizer,
            mlm=False,
            pad_to_multiple_of=8
        )
        
        # Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
        )
        
        # Eğitimi başlat
        trainer.train()
        
        # Model'i kaydet
        trainer.save_model()
        logger.info(f"✅ Model saved to {self.config.output_dir}")
        
    def test_generation(self, prompt: str):
        """Test generation with Turkish tokenizer"""
        logger.info(f"\nTest prompt: {prompt}")
        
        # Input'u hazırla
        inputs = self.adapter.prepare_input_for_model(prompt, max_length=self.config.max_length)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.7,
                do_sample=True,
                top_p=0.95,
            )
            
        # Decode
        generated_text = self.adapter.decode_from_model(outputs[0])
        logger.info(f"Generated: {generated_text}")
        
        return generated_text


def main():
    """Ana fonksiyon"""
    print("="*60)
    print("QWEN + TURKISH TOKENIZER TRAINING")
    print("="*60)
    
    # Konfigürasyon
    config = ModelConfig(
        model_name="Qwen/Qwen2.5-1.5B-Instruct",  # Daha küçük model test için
        use_4bit=True,
        use_lora=True,
        batch_size=2,
        num_epochs=1,  # Test için 1 epoch
        output_dir="./qwen-turkish-test"
    )
    
    # Trainer'ı oluştur
    trainer = TurkishTokenizerTrainer(config)
    
    # Veri yolu (mevcut değilse demo veri kullanılacak)
    train_data_path = "data/processed/training_data.jsonl"
    
    # Eğitimi başlat
    trainer.train(train_data_path)
    
    # Test et
    test_prompts = [
        "Python'da for döngüsü nasıl kullanılır?",
        "Yapay zeka nedir ve nerelerde kullanılır?",
        "Öğrenciler için etkili çalışma yöntemleri nelerdir?"
    ]
    
    print("\n" + "="*60)
    print("GENERATION TESTS")
    print("="*60)
    
    for prompt in test_prompts:
        trainer.test_generation(prompt)
        print("-"*40)
        
    print("\n✅ Training completed successfully!")
    

if __name__ == "__main__":
    main()