#!/usr/bin/env python3
"""
🇹🇷 TURKISH VOWEL HARMONY ENGINE
Comprehensive vowel harmony validation with morphological boundary detection
TEKNOFEST 2025 - Turkish LLM Linguistic Enhancement
"""

import torch
import re
import logging
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass
import warnings
warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)

@dataclass
class VowelHarmonyConfig:
    """Turkish vowel harmony configuration"""
    enable_validation: bool = True
    harmony_weight: float = 0.1
    morphology_boundary_weight: float = 0.15
    strict_mode: bool = False  # Strict Turkish rules vs learned patterns

class TurkishVowelHarmonyEngine:
    """Complete Turkish vowel harmony validation and optimization engine"""
    
    def __init__(self, config: VowelHarmonyConfig = None):
        self.config = config or VowelHarmonyConfig()
        
        # Turkish vowel groups for harmony rules
        self.front_unrounded = {'e', 'i'}     # e-i harmony
        self.back_unrounded = {'a', 'ı'}      # a-ı harmony  
        self.front_rounded = {'ö', 'ü'}       # ö-ü harmony
        self.back_rounded = {'o', 'u'}        # o-u harmony
        
        self.all_vowels = self.front_unrounded | self.back_unrounded | self.front_rounded | self.back_rounded
        
        # Morphological suffixes that must follow harmony
        self.harmony_suffixes = {
            'possessive': ['ım', 'im', 'um', 'üm', 'ın', 'in', 'un', 'ün'],
            'case_locative': ['da', 'de', 'ta', 'te'],  
            'case_ablative': ['dan', 'den', 'tan', 'ten'],
            'case_dative': ['a', 'e', 'ya', 'ye'],
            'plural': ['lar', 'ler'],
            'verb_present': ['yor', 'iyor', 'ıyor', 'uyor', 'üyor'],
            'verb_past': ['di', 'dı', 'du', 'dü', 'ti', 'tı', 'tu', 'tü']
        }
        
        # Exceptions (loanwords, compound words)
        self.harmony_exceptions = {
            'elma', 'anne', 'baba', 'kalem', 'kitap', 'computer', 'internet'
        }
        
        logger.info("✅ Turkish Vowel Harmony Engine initialized")
        
    def analyze_word_harmony(self, word: str) -> Dict[str, any]:
        """Analyze vowel harmony compliance for a single word"""
        
        if not word or len(word) < 2:
            return {'valid': True, 'score': 1.0, 'violations': []}
            
        # Skip exceptions
        if word.lower() in self.harmony_exceptions:
            return {'valid': True, 'score': 1.0, 'violations': [], 'exception': True}
        
        vowels = []
        positions = []
        
        # Extract vowels and their positions
        for i, char in enumerate(word.lower()):
            if char in self.all_vowels:
                vowels.append(char)
                positions.append(i)
                
        if len(vowels) < 2:
            return {'valid': True, 'score': 1.0, 'violations': []}
        
        violations = []
        harmony_score = 0.0
        total_checks = 0
        
        # Check harmony between consecutive vowels
        for i in range(len(vowels) - 1):
            current_vowel = vowels[i]
            next_vowel = vowels[i + 1]
            total_checks += 1
            
            # Determine if harmony is maintained
            if self._check_harmony_pair(current_vowel, next_vowel):
                harmony_score += 1.0
            else:
                violations.append({
                    'position': positions[i + 1],
                    'expected_group': self._get_harmony_group(current_vowel),
                    'found_vowel': next_vowel,
                    'violation_type': 'vowel_harmony'
                })
        
        final_score = harmony_score / total_checks if total_checks > 0 else 1.0
        
        return {
            'valid': len(violations) == 0,
            'score': final_score,
            'violations': violations,
            'vowels': vowels,
            'total_vowels': len(vowels)
        }
    
    def _check_harmony_pair(self, vowel1: str, vowel2: str) -> bool:
        """Check if two consecutive vowels follow Turkish harmony rules"""
        
        # Same vowel group - always harmonious
        if self._get_harmony_group(vowel1) == self._get_harmony_group(vowel2):
            return True
            
        # Mixed groups - check specific rules
        group1 = self._get_harmony_group(vowel1)
        group2 = self._get_harmony_group(vowel2)
        
        # Compatible harmony patterns
        compatible_patterns = [
            ('back_unrounded', 'back_rounded'),    # a,ı + o,u
            ('front_unrounded', 'front_rounded'),  # e,i + ö,ü
            ('back_rounded', 'back_unrounded'),    # o,u + a,ı  
            ('front_rounded', 'front_unrounded')   # ö,ü + e,i
        ]
        
        return (group1, group2) in compatible_patterns
        
    def _get_harmony_group(self, vowel: str) -> str:
        """Get harmony group for a vowel"""
        if vowel in self.front_unrounded:
            return 'front_unrounded'
        elif vowel in self.back_unrounded:
            return 'back_unrounded'
        elif vowel in self.front_rounded:
            return 'front_rounded'
        elif vowel in self.back_rounded:
            return 'back_rounded'
        else:
            return 'unknown'
    
    def analyze_text_harmony(self, text: str) -> Dict[str, any]:
        """Analyze vowel harmony for entire text"""
        
        # Split into words (simple tokenization)
        words = re.findall(r'\b[a-zA-ZçğıöşüÇĞIÖŞÜ]+\b', text)
        
        total_words = len(words)
        if total_words == 0:
            return {'overall_score': 1.0, 'compliant_words': 0, 'violations': []}
        
        compliant_words = 0
        total_score = 0.0
        all_violations = []
        
        for word in words:
            analysis = self.analyze_word_harmony(word)
            total_score += analysis['score']
            
            if analysis['valid']:
                compliant_words += 1
            else:
                all_violations.extend([{
                    'word': word,
                    **violation
                } for violation in analysis['violations']])
        
        return {
            'overall_score': total_score / total_words,
            'compliant_words': compliant_words,
            'total_words': total_words,
            'compliance_rate': compliant_words / total_words,
            'violations': all_violations
        }
    
    def compute_harmony_loss(self, 
                           input_ids: torch.Tensor,
                           tokenizer,
                           predictions: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute vowel harmony loss for LLM training"""
        
        if not self.config.enable_validation:
            return torch.tensor(0.0, device=input_ids.device)
        
        batch_size = input_ids.shape[0]
        device = input_ids.device
        
        harmony_scores = []
        
        for i in range(batch_size):
            try:
                # Decode tokens to text
                tokens = input_ids[i].cpu().tolist()
                text = tokenizer.decode(tokens, skip_special_tokens=True)
                
                # Analyze harmony
                analysis = self.analyze_text_harmony(text)
                harmony_scores.append(analysis['overall_score'])
                
            except Exception as e:
                logger.debug(f"Harmony analysis error: {e}")
                harmony_scores.append(0.5)  # Neutral score
        
        # Convert to tensor
        harmony_tensor = torch.tensor(harmony_scores, device=device)
        
        # Harmony loss (1 - harmony_score, so better harmony = lower loss)
        harmony_loss = 1.0 - harmony_tensor.mean()
        
        return harmony_loss * self.config.harmony_weight
    
    def get_morphology_boundaries(self, word: str) -> List[Tuple[int, str]]:
        """Detect morphological boundaries in Turkish word"""
        
        boundaries = []
        word_lower = word.lower()
        
        # Check for common suffixes
        for suffix_type, suffixes in self.harmony_suffixes.items():
            for suffix in suffixes:
                if word_lower.endswith(suffix) and len(word) > len(suffix):
                    boundary_pos = len(word) - len(suffix)
                    boundaries.append((boundary_pos, suffix_type))
        
        return boundaries
    
    def validate_morphology_harmony(self, word: str) -> Dict[str, any]:
        """Validate harmony specifically at morphological boundaries"""
        
        boundaries = self.get_morphology_boundaries(word)
        
        if not boundaries:
            return {'valid': True, 'score': 1.0, 'boundary_violations': []}
        
        violations = []
        
        for boundary_pos, suffix_type in boundaries:
            stem = word[:boundary_pos]
            suffix = word[boundary_pos:]
            
            # Check harmony between stem and suffix
            stem_analysis = self.analyze_word_harmony(stem)
            suffix_analysis = self.analyze_word_harmony(suffix)
            
            if stem_analysis['vowels'] and suffix_analysis['vowels']:
                last_stem_vowel = stem_analysis['vowels'][-1]
                first_suffix_vowel = suffix_analysis['vowels'][0]
                
                if not self._check_harmony_pair(last_stem_vowel, first_suffix_vowel):
                    violations.append({
                        'boundary_position': boundary_pos,
                        'suffix_type': suffix_type,
                        'stem_vowel': last_stem_vowel,
                        'suffix_vowel': first_suffix_vowel
                    })
        
        score = 1.0 - (len(violations) / len(boundaries)) if boundaries else 1.0
        
        return {
            'valid': len(violations) == 0,
            'score': score,
            'boundary_violations': violations,
            'boundaries_checked': len(boundaries)
        }

# Integration functions for LLM training
def create_harmony_engine(enable_validation: bool = True,
                         harmony_weight: float = 0.1) -> TurkishVowelHarmonyEngine:
    """Create Turkish vowel harmony engine for LLM integration"""
    
    config = VowelHarmonyConfig(
        enable_validation=enable_validation,
        harmony_weight=harmony_weight
    )
    
    return TurkishVowelHarmonyEngine(config)

def compute_turkish_linguistic_loss(input_ids: torch.Tensor,
                                   tokenizer,
                                   harmony_engine: TurkishVowelHarmonyEngine) -> torch.Tensor:
    """Compute combined Turkish linguistic loss (harmony + morphology)"""
    
    harmony_loss = harmony_engine.compute_harmony_loss(input_ids, tokenizer)
    
    # Future: Add morphological loss computation here
    morphology_loss = torch.tensor(0.0, device=input_ids.device)
    
    total_loss = harmony_loss + morphology_loss
    
    return total_loss

# Testing
if __name__ == "__main__":
    print("🧪 Testing Turkish Vowel Harmony Engine...")
    
    engine = create_harmony_engine()
    
    # Test words
    test_words = [
        "kitap",      # Valid harmony
        "kalem",      # Valid harmony  
        "masada",     # Valid with suffix
        "evde",       # Valid with suffix
        "computer",   # Exception (loanword)
        "telefona"    # Test case
    ]
    
    for word in test_words:
        analysis = engine.analyze_word_harmony(word)
        morph_analysis = engine.validate_morphology_harmony(word)
        
        print(f"📝 {word}: harmony={analysis['score']:.2f}, morphology={morph_analysis['score']:.2f}")
    
    print("✅ Turkish Vowel Harmony Engine test complete!")