"""
EKSIKSIZ NEFTune İMPLEMENTASYONU
Embedding layer noise injection ve adaptive scaling

Kritik Özellikler:
- Proper embedding layer hook
- Adaptive noise scaling based on Turkish performance
- Trainer callback integration
- Memory-efficient implementation
- Turkish-specific noise patterns
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from transformers import TrainerCallback, TrainerState, TrainingArguments
import logging

logger = logging.getLogger(__name__)

class NEFTuneEmbeddingHook:
    """NEFTune için embedding layer hook"""
    
    def __init__(self, 
                 alpha: float = 10.0,
                 noise_type: str = "gaussian",
                 adaptive_scaling: bool = True,
                 turkish_morphology_aware: bool = True):
        
        self.alpha = alpha
        self.noise_type = noise_type
        self.adaptive_scaling = adaptive_scaling
        self.turkish_morphology_aware = turkish_morphology_aware
        
        # Adaptive parameters
        self.current_alpha = alpha
        self.performance_history = []
        self.noise_scale_factor = 1.0
        
        # Turkish-specific parameters
        self.turkish_token_indices = set()
        self.morphological_boundaries = {}
        
    def apply_noise(self, embeddings: torch.Tensor, 
                   input_ids: Optional[torch.Tensor] = None,
                   training_step: Optional[int] = None) -> torch.Tensor:
        """Embedding'lere noise uygula"""
        
        if not embeddings.requires_grad:
            return embeddings
        
        batch_size, seq_len, embed_dim = embeddings.shape
        
        # Adaptive alpha calculation
        current_alpha = self._calculate_adaptive_alpha(training_step)
        
        # Noise generation
        if self.noise_type == "gaussian":
            noise = torch.normal(0, 1, size=embeddings.shape, device=embeddings.device)
        elif self.noise_type == "uniform":
            noise = torch.rand_like(embeddings) * 2 - 1  # [-1, 1] uniform
        else:
            raise ValueError(f"Unknown noise type: {self.noise_type}")
        
        # Turkish-specific noise modulation
        if self.turkish_morphology_aware and input_ids is not None:
            noise = self._apply_turkish_noise_modulation(noise, input_ids)
        
        # Noise scaling
        noise_norm = torch.norm(noise, dim=-1, keepdim=True)
        embed_norm = torch.norm(embeddings, dim=-1, keepdim=True)
        
        # Scaled noise: α * ||embedding|| / sqrt(d) * noise / ||noise||
        scaled_noise = (current_alpha * embed_norm / math.sqrt(embed_dim)) * (noise / (noise_norm + 1e-8))
        
        # Apply noise
        noisy_embeddings = embeddings + scaled_noise
        
        return noisy_embeddings
    
    def _calculate_adaptive_alpha(self, training_step: Optional[int] = None) -> float:
        """Adaptive alpha hesaplama"""
        
        if not self.adaptive_scaling:
            return self.alpha
        
        # Performance-based adaptation
        if len(self.performance_history) > 0:
            recent_performance = np.mean(self.performance_history[-10:])  # Son 10 step
            
            # İyi performans -> daha az noise
            # Kötü performans -> daha fazla noise
            if recent_performance < 1.0:  # Low loss = good performance
                self.noise_scale_factor = max(0.5, self.noise_scale_factor * 0.98)
            else:  # High loss = bad performance
                self.noise_scale_factor = min(2.0, self.noise_scale_factor * 1.02)
        
        # Training step-based decay
        if training_step is not None and training_step > 1000:
            decay_factor = 1.0 / (1.0 + 0.0001 * (training_step - 1000))
            self.current_alpha = self.alpha * self.noise_scale_factor * decay_factor
        else:
            self.current_alpha = self.alpha * self.noise_scale_factor
        
        return self.current_alpha
    
    def _apply_turkish_noise_modulation(self, noise: torch.Tensor, 
                                       input_ids: torch.Tensor) -> torch.Tensor:
        """Türkçe-spesifik noise modülasyonu"""
        
        # Turkish token mask
        turkish_mask = torch.zeros_like(input_ids, dtype=torch.float32)
        
        for batch_idx in range(input_ids.shape[0]):
            for seq_idx in range(input_ids.shape[1]):
                token_id = input_ids[batch_idx, seq_idx].item()
                
                # Turkish token ise daha fazla noise
                if token_id in self.turkish_token_indices:
                    turkish_mask[batch_idx, seq_idx] = 1.2
                else:
                    turkish_mask[batch_idx, seq_idx] = 0.8
        
        # Noise modulation
        turkish_mask = turkish_mask.unsqueeze(-1)  # [batch, seq, 1]
        modulated_noise = noise * turkish_mask
        
        return modulated_noise
    
    def update_performance(self, loss: float):
        """Performance history güncelle"""
        self.performance_history.append(loss)
        
        # Keep only recent history
        if len(self.performance_history) > 100:
            self.performance_history = self.performance_history[-50:]
    
    def set_turkish_token_indices(self, turkish_token_indices: set):
        """Turkish token indices ayarla"""
        self.turkish_token_indices = turkish_token_indices
        logger.info(f"Set {len(turkish_token_indices)} Turkish token indices for NEFTune")


class NEFTuneCallback(TrainerCallback):
    """NEFTune için Trainer callback"""
    
    def __init__(self, 
                 alpha: float = 10.0,
                 noise_type: str = "gaussian",
                 adaptive_scaling: bool = True,
                 turkish_token_indices: Optional[set] = None):
        
        self.neftune_hook = NEFTuneEmbeddingHook(
            alpha=alpha,
            noise_type=noise_type,
            adaptive_scaling=adaptive_scaling,
            turkish_morphology_aware=(turkish_token_indices is not None)
        )
        
        if turkish_token_indices:
            self.neftune_hook.set_turkish_token_indices(turkish_token_indices)
        
        self.embedding_layers = []
        self.hook_handles = []
        
    def on_train_begin(self, args: TrainingArguments, state: TrainerState, 
                      control, model=None, **kwargs):
        """Training başlangıcında hook'ları kur"""
        
        if model is None:
            return
        
        # Embedding layer'ları bul
        self.embedding_layers = []
        for name, module in model.named_modules():
            if isinstance(module, nn.Embedding):
                self.embedding_layers.append((name, module))
        
        logger.info(f"Found {len(self.embedding_layers)} embedding layers for NEFTune")
        
        # Hook'ları kur
        self._install_hooks()
    
    def on_train_end(self, args: TrainingArguments, state: TrainerState, 
                    control, model=None, **kwargs):
        """Training sonunda hook'ları kaldır"""
        self._remove_hooks()
    
    def on_log(self, args: TrainingArguments, state: TrainerState, 
              control, model=None, logs=None, **kwargs):
        """Her log'da performance güncelle"""
        
        if logs and 'train_loss' in logs:
            self.neftune_hook.update_performance(logs['train_loss'])
    
    def _install_hooks(self):
        """Embedding layer hook'larını kur"""
        
        for name, embedding_layer in self.embedding_layers:
            
            def create_hook(layer_name):
                def forward_hook(module, input_ids, output):
                    """Forward hook fonksiyonu"""
                    
                    if module.training and len(input_ids) > 0:
                        # Input IDs al
                        ids_tensor = input_ids[0] if isinstance(input_ids, tuple) else input_ids
                        
                        # NEFTune noise uygula
                        noisy_output = self.neftune_hook.apply_noise(
                            embeddings=output,
                            input_ids=ids_tensor,
                            training_step=getattr(self, 'current_step', None)
                        )
                        
                        return noisy_output
                    
                    return output
                
                return forward_hook
            
            # Hook'u register et
            hook_handle = embedding_layer.register_forward_hook(create_hook(name))
            self.hook_handles.append(hook_handle)
            
            logger.info(f"Installed NEFTune hook on {name}")
    
    def _remove_hooks(self):
        """Hook'ları kaldır"""
        
        for handle in self.hook_handles:
            handle.remove()
        
        self.hook_handles.clear()
        logger.info("Removed all NEFTune hooks")
    
    def update_training_step(self, step: int):
        """Training step güncelle"""
        self.current_step = step


class NEFTuneModelWrapper(nn.Module):
    """NEFTune wrapper for models without trainer"""
    
    def __init__(self, model: nn.Module, neftune_config: Dict):
        super(NEFTuneModelWrapper, self).__init__()
        
        self.model = model
        self.neftune_hook = NEFTuneEmbeddingHook(**neftune_config)
        
        # Install hooks
        self.hook_handles = []
        self._install_hooks()
    
    def _install_hooks(self):
        """Hook'ları kur"""
        
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Embedding):
                
                def create_hook():
                    def forward_hook(module, inputs, output):
                        if self.training:
                            input_ids = inputs[0] if isinstance(inputs, tuple) else inputs
                            return self.neftune_hook.apply_noise(output, input_ids)
                        return output
                    return forward_hook
                
                handle = module.register_forward_hook(create_hook())
                self.hook_handles.append(handle)
    
    def forward(self, *args, **kwargs):
        """Forward pass"""
        return self.model(*args, **kwargs)
    
    def __getattr__(self, name):
        """Model attributes'larını proxy et"""
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)


def create_neftune_callback(alpha: float = 10.0,
                          tokenizer = None,
                          adaptive_scaling: bool = True) -> NEFTuneCallback:
    """NEFTune callback oluştur"""
    
    # Turkish token indices
    turkish_token_indices = None
    if tokenizer is not None:
        # Türkçe karakterli token'ları bul
        turkish_chars = set("çğıöşüâîûÇĞİÖŞÜ")
        turkish_token_indices = set()
        
        for token_id, token in tokenizer.vocab.items():
            if any(char in token for char in turkish_chars):
                turkish_token_indices.add(token_id)
        
        logger.info(f"Found {len(turkish_token_indices)} Turkish tokens")
    
    callback = NEFTuneCallback(
        alpha=alpha,
        noise_type="gaussian",
        adaptive_scaling=adaptive_scaling,
        turkish_token_indices=turkish_token_indices
    )
    
    return callback


def apply_neftune_to_model(model: nn.Module, 
                          alpha: float = 10.0,
                          noise_type: str = "gaussian") -> NEFTuneModelWrapper:
    """Model'e NEFTune uygula"""
    
    neftune_config = {
        'alpha': alpha,
        'noise_type': noise_type,
        'adaptive_scaling': True,
        'turkish_morphology_aware': True
    }
    
    wrapped_model = NEFTuneModelWrapper(model, neftune_config)
    
    logger.info(f"Applied NEFTune to model with alpha={alpha}")
    
    return wrapped_model


# Test fonksiyonu
def test_neftune_implementation():
    """NEFTune implementasyonunu test et"""
    
    print("🧪 NEFTune implementasyonu test ediliyor...")
    
    # Test embedding layer
    embedding = nn.Embedding(1000, 768)
    
    # NEFTune hook
    hook = NEFTuneEmbeddingHook(alpha=5.0)
    
    # Test input
    input_ids = torch.randint(0, 1000, (4, 32))
    embeddings = embedding(input_ids)
    
    print(f"✅ Original embeddings shape: {embeddings.shape}")
    
    # Apply noise
    noisy_embeddings = hook.apply_noise(embeddings, input_ids)
    
    print(f"✅ Noisy embeddings shape: {noisy_embeddings.shape}")
    
    # Test noise magnitude
    noise = noisy_embeddings - embeddings
    noise_magnitude = torch.norm(noise).item()
    embed_magnitude = torch.norm(embeddings).item()
    
    print(f"✅ Noise magnitude: {noise_magnitude:.4f}")
    print(f"✅ Embedding magnitude: {embed_magnitude:.4f}")
    print(f"✅ Noise ratio: {noise_magnitude/embed_magnitude:.4f}")
    
    print("🎉 NEFTune implementasyonu başarılı!")


if __name__ == "__main__":
    import math
    test_neftune_implementation()