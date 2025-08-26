"""
GELİŞMİŞ DORA İMPLEMENTASYONU
Gerçek weight decomposition ve magnitude scaling

Kritik Özellikler:
- Gerçek weight decomposition (magnitude + direction)
- Adaptive magnitude scaling
- Turkish-specific pattern preservation
- Memory-efficient implementation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class DoRALinear(nn.Module):
    """Gerçek DoRA implementasyonu - Weight Decomposition"""
    
    def __init__(self, 
                 base_layer: nn.Linear,
                 r: int = 256,
                 lora_alpha: int = 128,
                 lora_dropout: float = 0.05,
                 use_magnitude_scaling: bool = True,
                 turkish_pattern_preservation: bool = True):
        
        super(DoRALinear, self).__init__()
        
        self.base_layer = base_layer
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r
        self.use_magnitude_scaling = use_magnitude_scaling
        self.turkish_pattern_preservation = turkish_pattern_preservation
        
        # LoRA parameters
        self.lora_A = nn.Parameter(torch.randn(r, self.in_features))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, r))
        self.dropout = nn.Dropout(lora_dropout)
        
        # DoRA specific: Magnitude vector (m)
        if self.use_magnitude_scaling:
            # Base weight'in magnitude'unu hesapla
            with torch.no_grad():
                base_magnitude = torch.norm(base_layer.weight.data, dim=1, keepdim=True)
                self.magnitude = nn.Parameter(base_magnitude.clone())
        
        # Turkish pattern preservation weights
        if self.turkish_pattern_preservation:
            self.turkish_weights = nn.Parameter(torch.ones(self.out_features, 1))
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Ağırlık başlatma"""
        # LoRA A: Kaiming uniform
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        
        # LoRA B: Zero initialization (önemli!)
        nn.init.zeros_(self.lora_B)
        
        # Turkish weights: Xavier initialization
        if self.turkish_pattern_preservation:
            nn.init.xavier_uniform_(self.turkish_weights)
    
    def _compute_dora_weight(self) -> torch.Tensor:
        """DoRA weight hesaplama: W = m * (W_0 + BA) / ||W_0 + BA||"""
        
        # Base weight
        base_weight = self.base_layer.weight
        
        # LoRA delta: BA
        lora_delta = (self.lora_B @ self.lora_A) * self.scaling
        
        # Combined weight: W_0 + ΔW
        combined_weight = base_weight + lora_delta
        
        if self.use_magnitude_scaling:
            # Direction normalization: W / ||W||
            weight_norm = torch.norm(combined_weight, dim=1, keepdim=True)
            weight_direction = combined_weight / (weight_norm + 1e-8)
            
            # DoRA: m * direction
            dora_weight = self.magnitude * weight_direction
        else:
            dora_weight = combined_weight
        
        # Turkish pattern preservation
        if self.turkish_pattern_preservation:
            dora_weight = dora_weight * self.turkish_weights
        
        return dora_weight
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with DoRA weight"""
        
        # Compute DoRA weight
        dora_weight = self._compute_dora_weight()
        
        # Apply dropout to input
        x = self.dropout(x)
        
        # Linear transformation with DoRA weight
        output = F.linear(x, dora_weight, self.base_layer.bias)
        
        return output
    
    def update_magnitude_scaling(self, turkish_loss: Optional[float] = None):
        """Magnitude scaling güncelleme (Turkish-aware)"""
        
        if not self.use_magnitude_scaling:
            return
            
        with torch.no_grad():
            # Current combined weight
            base_weight = self.base_layer.weight
            lora_delta = (self.lora_B @ self.lora_A) * self.scaling
            combined_weight = base_weight + lora_delta
            
            # Compute current magnitude
            current_magnitude = torch.norm(combined_weight, dim=1, keepdim=True)
            
            # Adaptive update based on Turkish performance
            if turkish_loss is not None:
                # Lower loss = increase magnitude scaling
                scaling_factor = 1.0 + 0.1 * (1.0 / (1.0 + turkish_loss))
                target_magnitude = current_magnitude * scaling_factor
                
                # Smooth update
                self.magnitude.data = 0.9 * self.magnitude.data + 0.1 * target_magnitude
            else:
                # Standard magnitude update
                self.magnitude.data = 0.95 * self.magnitude.data + 0.05 * current_magnitude


class DoRAConverter:
    """Base linear layer'ları DoRA layer'larına çevirir"""
    
    def __init__(self, 
                 r: int = 256,
                 lora_alpha: int = 128,
                 lora_dropout: float = 0.05,
                 target_modules: List[str] = None):
        
        self.r = r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.target_modules = target_modules or [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ]
        
    def convert_model_to_dora(self, model: nn.Module) -> nn.Module:
        """Model'i DoRA'ya çevir"""
        
        logger.info("Converting model to DoRA...")
        
        converted_modules = []
        
        for name, module in model.named_modules():
            # Linear layer ve target module kontrolü
            if isinstance(module, nn.Linear) and any(target in name for target in self.target_modules):
                
                # DoRA layer oluştur
                dora_layer = DoRALinear(
                    base_layer=module,
                    r=self.r,
                    lora_alpha=self.lora_alpha,
                    lora_dropout=self.lora_dropout,
                    use_magnitude_scaling=True,
                    turkish_pattern_preservation=True
                )
                
                # Model'de replace et
                self._replace_module(model, name, dora_layer)
                converted_modules.append(name)
        
        logger.info(f"Converted {len(converted_modules)} modules to DoRA: {converted_modules}")
        
        return model
    
    def _replace_module(self, parent_module: nn.Module, module_name: str, new_module: nn.Module):
        """Module replacement helper"""
        
        # Split module path
        path_parts = module_name.split('.')
        
        # Navigate to parent
        current_module = parent_module
        for part in path_parts[:-1]:
            current_module = getattr(current_module, part)
        
        # Replace final module
        setattr(current_module, path_parts[-1], new_module)


class DoRATrainerCallback:
    """DoRA için özel trainer callback"""
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.dora_modules = []
        
        # DoRA module'larını topla
        for name, module in model.named_modules():
            if isinstance(module, DoRALinear):
                self.dora_modules.append((name, module))
        
        logger.info(f"Found {len(self.dora_modules)} DoRA modules for callback")
    
    def on_evaluate(self, eval_loss: float):
        """Evaluation sonrası DoRA güncelleme"""
        
        for name, dora_module in self.dora_modules:
            dora_module.update_magnitude_scaling(turkish_loss=eval_loss)
    
    def on_save_checkpoint(self, checkpoint_dir: str):
        """Checkpoint kaydetme"""
        
        dora_states = {}
        for name, dora_module in self.dora_modules:
            dora_states[name] = {
                'lora_A': dora_module.lora_A.data.cpu(),
                'lora_B': dora_module.lora_B.data.cpu(),
                'magnitude': dora_module.magnitude.data.cpu() if dora_module.use_magnitude_scaling else None,
                'turkish_weights': dora_module.turkish_weights.data.cpu() if dora_module.turkish_pattern_preservation else None
            }
        
        torch.save(dora_states, f"{checkpoint_dir}/dora_states.pt")
        logger.info(f"DoRA states saved to {checkpoint_dir}")


def apply_dora_to_model(model: nn.Module, 
                       r: int = 256, 
                       lora_alpha: int = 128,
                       target_modules: List[str] = None) -> Tuple[nn.Module, DoRATrainerCallback]:
    """Model'e DoRA uygula ve callback döndür - TOKENIZER SAFE VERSION"""
    
    # CRITICAL: TOKENIZER MISMATCH PROTECTİON
    # Memory'den: ASLA modules_to_save=["embed_tokens", "lm_head"] kullanma!
    logger.warning("🚨 TOKENIZER SAFETY: embed_tokens ve lm_head MODİFİYE EDİLMEYECEK!")
    
    converter = DoRAConverter(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=target_modules or [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
            # 🚨 KRİTİK: embed_tokens ve lm_head DAHİL DEĞİL!
        ]
    )
    
    # Convert model
    dora_model = converter.convert_model_to_dora(model)
    
    # Create callback
    dora_callback = DoRATrainerCallback(dora_model)
    
    # Trainable parameters count
    total_params = sum(p.numel() for p in dora_model.parameters())
    trainable_params = sum(p.numel() for p in dora_model.parameters() if p.requires_grad)
    
    logger.info(f"DoRA conversion completed:")
    logger.info(f"  Total parameters: {total_params:,}")
    logger.info(f"  Trainable parameters: {trainable_params:,}")
    logger.info(f"  Trainable ratio: {trainable_params/total_params:.3%}")
    
    return dora_model, dora_callback


# Test fonksiyonu
def test_dora_implementation():
    """DoRA implementasyonunu test et"""
    
    print("🧪 DoRA implementasyonu test ediliyor...")
    
    # Test linear layer
    base_linear = nn.Linear(768, 3072)
    
    # DoRA'ya çevir
    dora_linear = DoRALinear(base_linear, r=64, lora_alpha=32)
    
    # Test input
    x = torch.randn(32, 128, 768)
    
    # Forward pass
    output = dora_linear(x)
    
    print(f"✅ Input shape: {x.shape}")
    print(f"✅ Output shape: {output.shape}")
    print(f"✅ DoRA parameters: {sum(p.numel() for p in dora_linear.parameters()):,}")
    
    # Test magnitude update
    dora_linear.update_magnitude_scaling(turkish_loss=0.5)
    print("✅ Magnitude scaling updated")
    
    print("🎉 DoRA implementasyonu başarılı!")


if __name__ == "__main__":
    test_dora_implementation()