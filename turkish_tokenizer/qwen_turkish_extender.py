"""
Qwen3-8B Turkish Tokenizer Extension
Extends Qwen3-8B tokenizer with 30K-50K Turkish-specific tokens

Key Features:
- Aggressive vocabulary expansion for Turkish-only use
- Preserves original Qwen3 knowledge while adding Turkish optimization
- Morphology-aware token integration
- Smart embedding initialization for new tokens
- DoRA-compatible model preparation
"""

import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    PreTrainedTokenizer,
    PreTrainedModel
)
from safetensors import safe_open
from safetensors.torch import save_file
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QwenTurkishTokenizerExtender:
    """Extends Qwen3-8B tokenizer with Turkish vocabulary"""
    
    def __init__(self, 
                 model_name: str = "Qwen/Qwen3-8B",
                 output_dir: str = "qwen3_turkish_extended"):
        
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.original_tokenizer = None
        self.extended_tokenizer = None
        self.original_model = None
        self.extended_model = None
        
        # Extension statistics
        self.extension_stats = {
            'original_vocab_size': 0,
            'added_tokens': 0,
            'new_vocab_size': 0,
            'embedding_dimension': 0,
            'model_parameters_before': 0,
            'model_parameters_after': 0
        }
    
    def load_original_assets(self) -> bool:
        """Load original Qwen3-8B tokenizer and model"""
        try:
            logger.info(f"Loading original Qwen3-8B model and tokenizer...")
            
            # Load tokenizer
            self.original_tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                use_fast=False
            )
            
            # Load model
            self.original_model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            
            # Update statistics
            self.extension_stats.update({
                'original_vocab_size': len(self.original_tokenizer.vocab),
                'embedding_dimension': self.original_model.config.hidden_size,
                'model_parameters_before': self.original_model.num_parameters()
            })
            
            logger.info(f"Original vocabulary size: {self.extension_stats['original_vocab_size']}")
            logger.info(f"Embedding dimension: {self.extension_stats['embedding_dimension']}")
            logger.info(f"Model parameters: {self.extension_stats['model_parameters_before']:,}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load original assets: {e}")
            return False
    
    def load_turkish_vocabulary(self, vocab_file: str) -> Dict[str, int]:
        """Load Turkish vocabulary for extension"""
        vocab_path = Path(vocab_file)
        
        if not vocab_path.exists():
            logger.error(f"Turkish vocabulary file not found: {vocab_path}")
            return {}
        
        try:
            with open(vocab_path, 'r', encoding='utf-8') as f:
                turkish_vocab = json.load(f)
            
            logger.info(f"Loaded {len(turkish_vocab)} Turkish tokens for extension")
            return turkish_vocab
            
        except Exception as e:
            logger.error(f"Failed to load Turkish vocabulary: {e}")
            return {}
    
    def create_extended_tokenizer(self, turkish_vocab: Dict[str, int]) -> PreTrainedTokenizer:
        """Create extended tokenizer with Turkish vocabulary"""
        
        logger.info("Creating extended tokenizer...")
        
        # Clone original tokenizer
        self.extended_tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            use_fast=False
        )
        
        # Get original vocabulary
        original_vocab = dict(self.extended_tokenizer.vocab)
        original_vocab_size = len(original_vocab)
        
        # Prepare new tokens (filter out existing ones)
        new_tokens = []
        current_max_id = max(original_vocab.values())
        
        for token, suggested_id in turkish_vocab.items():
            if token not in original_vocab:
                new_tokens.append(token)
        
        if not new_tokens:
            logger.warning("No new Turkish tokens to add!")
            return self.extended_tokenizer
        
        # Add new tokens to tokenizer
        logger.info(f"Adding {len(new_tokens)} new Turkish tokens...")
        
        added_tokens = self.extended_tokenizer.add_tokens(new_tokens)
        
        logger.info(f"Successfully added {added_tokens} new tokens")
        
        # Update statistics
        self.extension_stats.update({
            'added_tokens': added_tokens,
            'new_vocab_size': len(self.extended_tokenizer.vocab)
        })
        
        return self.extended_tokenizer
    
    def initialize_new_embeddings(self, 
                                 embedding_matrix: torch.Tensor,
                                 new_vocab_size: int) -> torch.Tensor:
        """Initialize embeddings for new Turkish tokens"""
        
        original_size, embedding_dim = embedding_matrix.shape
        added_tokens = new_vocab_size - original_size
        
        if added_tokens <= 0:
            return embedding_matrix
        
        logger.info(f"Initializing embeddings for {added_tokens} new tokens...")
        
        # Create new embedding matrix
        new_embedding_matrix = torch.zeros(
            (new_vocab_size, embedding_dim),
            dtype=embedding_matrix.dtype,
            device=embedding_matrix.device
        )
        
        # Copy original embeddings
        new_embedding_matrix[:original_size] = embedding_matrix
        
        # Initialize new token embeddings
        # Strategy 1: Use normal distribution around mean of original embeddings
        original_mean = embedding_matrix.mean(dim=0)
        original_std = embedding_matrix.std(dim=0)
        
        # Strategy 2: For Turkish tokens, use similar tokens if possible
        # Find tokens that might be morphologically similar
        original_vocab = {v: k for k, v in self.original_tokenizer.vocab.items()}
        
        for new_token_idx in range(original_size, new_vocab_size):
            if hasattr(self.extended_tokenizer, 'convert_ids_to_tokens'):
                new_token = self.extended_tokenizer.convert_ids_to_tokens(new_token_idx)
                
                # Try to find similar existing tokens for initialization
                similar_embeddings = []
                
                # Look for tokens with similar prefixes/suffixes
                for orig_idx in range(min(1000, original_size)):  # Check first 1000 for efficiency
                    orig_token = original_vocab.get(orig_idx, "")
                    
                    if (orig_token and len(orig_token) > 2 and 
                        (new_token.startswith(orig_token[:2]) or 
                         new_token.endswith(orig_token[-2:]) or
                         orig_token.startswith(new_token[:2]))):
                        similar_embeddings.append(embedding_matrix[orig_idx])
                
                if similar_embeddings:
                    # Average of similar embeddings + small noise
                    similar_stack = torch.stack(similar_embeddings)
                    base_embedding = similar_stack.mean(dim=0)
                    noise = torch.normal(0, 0.01, base_embedding.shape).to(base_embedding.device)
                    new_embedding_matrix[new_token_idx] = base_embedding + noise
                else:
                    # Fallback to statistical initialization
                    noise = torch.normal(original_mean, original_std * 0.1).to(embedding_matrix.device)
                    new_embedding_matrix[new_token_idx] = noise
            else:
                # Simple statistical initialization
                noise = torch.normal(original_mean, original_std * 0.1).to(embedding_matrix.device)
                new_embedding_matrix[new_token_idx] = noise
        
        logger.info("New token embeddings initialized successfully")
        return new_embedding_matrix
    
    def extend_model_embeddings(self) -> PreTrainedModel:
        """Extend model embeddings for new vocabulary"""
        
        logger.info("Extending model embeddings...")
        
        # Clone the model
        self.extended_model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        # Get current embeddings
        input_embeddings = self.extended_model.get_input_embeddings()
        output_embeddings = self.extended_model.get_output_embeddings()
        
        original_vocab_size = self.extension_stats['original_vocab_size']
        new_vocab_size = self.extension_stats['new_vocab_size']
        
        if new_vocab_size == original_vocab_size:
            logger.warning("No vocabulary extension needed")
            return self.extended_model
        
        # Extend input embeddings
        logger.info("Extending input embeddings...")
        original_input_weights = input_embeddings.weight.data
        
        new_input_weights = self.initialize_new_embeddings(
            original_input_weights, new_vocab_size
        )
        
        # Create new input embedding layer
        new_input_embeddings = torch.nn.Embedding(
            new_vocab_size, 
            self.extension_stats['embedding_dimension'],
            dtype=original_input_weights.dtype
        )
        new_input_embeddings.weight.data = new_input_weights
        
        # Extend output embeddings (lm_head)
        logger.info("Extending output embeddings...")
        original_output_weights = output_embeddings.weight.data
        
        new_output_weights = self.initialize_new_embeddings(
            original_output_weights, new_vocab_size
        )
        
        # Create new output embedding layer
        new_output_embeddings = torch.nn.Linear(
            self.extension_stats['embedding_dimension'],
            new_vocab_size,
            bias=False,
            dtype=original_output_weights.dtype
        )
        new_output_embeddings.weight.data = new_output_weights
        
        # Update model with new embeddings
        self.extended_model.set_input_embeddings(new_input_embeddings)
        self.extended_model.set_output_embeddings(new_output_embeddings)
        
        # Update model configuration
        self.extended_model.config.vocab_size = new_vocab_size
        
        # Update statistics
        self.extension_stats['model_parameters_after'] = self.extended_model.num_parameters()
        
        logger.info(f"Model embeddings extended successfully")
        logger.info(f"New model parameters: {self.extension_stats['model_parameters_after']:,}")
        
        return self.extended_model
    
    def save_extended_assets(self):
        """Save extended tokenizer and model"""
        
        logger.info(f"Saving extended assets to {self.output_dir}")
        
        # Save extended tokenizer
        tokenizer_path = self.output_dir / "tokenizer"
        self.extended_tokenizer.save_pretrained(tokenizer_path)
        logger.info(f"Extended tokenizer saved to {tokenizer_path}")
        
        # Save extended model
        model_path = self.output_dir / "model"
        self.extended_model.save_pretrained(
            model_path,
            safe_serialization=True,
            max_shard_size="5GB"
        )
        logger.info(f"Extended model saved to {model_path}")
        
        # Save extension statistics
        stats_path = self.output_dir / "extension_stats.json"
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(self.extension_stats, f, indent=2)
        
        # Save configuration for training
        config = {
            'model_name': self.model_name,
            'original_vocab_size': self.extension_stats['original_vocab_size'],
            'new_vocab_size': self.extension_stats['new_vocab_size'],
            'added_tokens': self.extension_stats['added_tokens'],
            'embedding_dimension': self.extension_stats['embedding_dimension'],
            'training_recommendations': {
                'learning_rate': 2e-4,
                'warmup_ratio': 0.1,
                'epochs': 10,
                'batch_size': 16,
                'gradient_accumulation_steps': 8,
                'lora_config': {
                    'r': 256,
                    'lora_alpha': 128,
                    'target_modules': [
                        'q_proj', 'k_proj', 'v_proj', 'o_proj',
                        'gate_proj', 'up_proj', 'down_proj'
                    ],
                    'use_dora': True,
                    'use_rslora': True
                }
            }
        }
        
        config_path = self.output_dir / "training_config.json"
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2)
        
        logger.info("Extension completed successfully!")
        
        return {
            'tokenizer_path': str(tokenizer_path),
            'model_path': str(model_path),
            'config_path': str(config_path),
            'stats': self.extension_stats
        }
    
    def validate_extension(self) -> Dict:
        """Validate the tokenizer extension"""
        
        logger.info("Validating tokenizer extension...")
        
        # Test tokenization
        test_texts = [
            "Merhaba dünya, nasılsınız?",
            "Türkçe dilindeki kelimeler ve morpholoji yapısı oldukça karmaşıktır.",
            "Geliyorum, gitmiştim, yapacağım gibi fiil çekimlerini anlayabilir mi?",
            "Ağaçların yeşilliği ve kuşların cıvıltısı çok güzeldir.",
            "Teknoloji alanındaki yenilikler hayatımızı kolaylaştırıyor."
        ]
        
        validation_results = {}
        
        for i, text in enumerate(test_texts):
            # Original tokenization
            orig_tokens = self.original_tokenizer.tokenize(text)
            orig_ids = self.original_tokenizer.encode(text)
            
            # Extended tokenization
            ext_tokens = self.extended_tokenizer.tokenize(text)
            ext_ids = self.extended_tokenizer.encode(text)
            
            # Calculate improvement
            token_reduction = (len(orig_tokens) - len(ext_tokens)) / len(orig_tokens) * 100
            
            validation_results[f"test_{i}"] = {
                'text': text,
                'original_tokens': len(orig_tokens),
                'extended_tokens': len(ext_tokens),
                'token_reduction_percent': token_reduction,
                'original_tokenization': orig_tokens[:10],  # First 10 tokens
                'extended_tokenization': ext_tokens[:10]
            }
        
        # Calculate average improvement
        avg_reduction = np.mean([
            result['token_reduction_percent'] 
            for result in validation_results.values()
        ])
        
        validation_summary = {
            'average_token_reduction': avg_reduction,
            'tests': validation_results,
            'vocabulary_stats': self.extension_stats
        }
        
        logger.info(f"Validation completed. Average token reduction: {avg_reduction:.1f}%")
        
        return validation_summary


def extend_qwen_tokenizer(
    turkish_vocab_file: str = "vocab_analysis/qwen_turkish_extension_vocab.json",
    output_dir: str = "qwen3_turkish_extended"
) -> Dict:
    """Main function to extend Qwen3-8B tokenizer with Turkish vocabulary"""
    
    logger.info("Starting Qwen3-8B Turkish tokenizer extension...")
    
    # Initialize extender
    extender = QwenTurkishTokenizerExtender(output_dir=output_dir)
    
    # Load original assets
    if not extender.load_original_assets():
        logger.error("Failed to load original Qwen3-8B assets")
        return {}
    
    # Load Turkish vocabulary
    turkish_vocab = extender.load_turkish_vocabulary(turkish_vocab_file)
    if not turkish_vocab:
        logger.error("Failed to load Turkish vocabulary")
        return {}
    
    # Create extended tokenizer
    extended_tokenizer = extender.create_extended_tokenizer(turkish_vocab)
    if not extended_tokenizer:
        logger.error("Failed to create extended tokenizer")
        return {}
    
    # Extend model embeddings
    extended_model = extender.extend_model_embeddings()
    if not extended_model:
        logger.error("Failed to extend model embeddings")
        return {}
    
    # Validate extension
    validation_results = extender.validate_extension()
    
    # Save extended assets
    save_results = extender.save_extended_assets()
    
    # Complete results
    results = {
        'extension_stats': extender.extension_stats,
        'validation': validation_results,
        'saved_assets': save_results
    }
    
    # Print summary
    print("\n" + "="*60)
    print("QWEN3-8B TURKISH TOKENIZER EXTENSION COMPLETED")
    print("="*60)
    print(f"Original vocabulary: {extender.extension_stats['original_vocab_size']:,}")
    print(f"Added Turkish tokens: {extender.extension_stats['added_tokens']:,}")
    print(f"New vocabulary size: {extender.extension_stats['new_vocab_size']:,}")
    print(f"Average token reduction: {validation_results['average_token_reduction']:.1f}%")
    print(f"Model parameters: {extender.extension_stats['model_parameters_after']:,}")
    print(f"\nExtended assets saved to: {output_dir}")
    
    return results


if __name__ == "__main__":
    # Run the extension
    results = extend_qwen_tokenizer()
    
    if results:
        print("\nNext steps:")
        print("1. Review extension_stats.json for detailed metrics")
        print("2. Use training_config.json for DoRA training setup")
        print("3. Start Turkish training with extended model")
    else:
        print("Extension failed. Check logs for details.")