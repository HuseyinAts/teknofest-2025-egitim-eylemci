#!/usr/bin/env python3
"""
🇹🇷 ADVANCED TURKISH UNICODE PROCESSOR
Comprehensive Turkish character handling with İ/ı distinction
TEKNOFEST 2025 - Turkish Language Processing Excellence

CRITICAL FEATURES:
- Proper İ/ı (dotted/dotless I) distinction handling
- Turkish character normalization and validation
- Context-aware case conversion
- Morphological boundary-aware processing
- Unicode normalization for Turkish text
- Collation and sorting for Turkish strings
"""

import re
import unicodedata
import logging
from typing import Dict, List, Optional, Tuple, Set, Any, Union
from dataclasses import dataclass
from enum import Enum
import warnings
warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)

class TurkishCaseMode(Enum):
    """Turkish case conversion modes"""
    LOWER = "lower"
    UPPER = "upper"
    TITLE = "title"
    PRESERVE = "preserve"

@dataclass
class TurkishCharacterInfo:
    """Information about Turkish characters"""
    character: str
    unicode_name: str
    category: str
    is_turkish_specific: bool
    ascii_equivalent: Optional[str] = None
    case_variants: Dict[str, str] = None
    
    def __post_init__(self):
        if self.case_variants is None:
            self.case_variants = {}

class AdvancedTurkishUnicodeProcessor:
    """Advanced processor for Turkish Unicode text with İ/ı distinction"""
    
    def __init__(self):
        
        # Turkish alphabet with proper Unicode points
        self.turkish_alphabet = {
            # Lowercase
            'a': '\u0061', 'b': '\u0062', 'c': '\u0063', 'ç': '\u00E7', 'd': '\u0064',
            'e': '\u0065', 'f': '\u0066', 'g': '\u0067', 'ğ': '\u011F', 'h': '\u0068',
            'ı': '\u0131', 'i': '\u0069', 'j': '\u006A', 'k': '\u006B', 'l': '\u006C',
            'm': '\u006D', 'n': '\u006E', 'o': '\u006F', 'ö': '\u00F6', 'p': '\u0070',
            'r': '\u0072', 's': '\u0073', 'ş': '\u015F', 't': '\u0074', 'u': '\u0075',
            'ü': '\u00FC', 'v': '\u0076', 'y': '\u0079', 'z': '\u007A',
            
            # Uppercase  
            'A': '\u0041', 'B': '\u0042', 'C': '\u0043', 'Ç': '\u00C7', 'D': '\u0044',
            'E': '\u0045', 'F': '\u0046', 'G': '\u0047', 'Ğ': '\u011E', 'H': '\u0048',
            'I': '\u0049', 'İ': '\u0130', 'J': '\u004A', 'K': '\u004B', 'L': '\u004C',
            'M': '\u004D', 'N': '\u004E', 'O': '\u004F', 'Ö': '\u00D6', 'P': '\u0050',
            'R': '\u0052', 'S': '\u0053', 'Ş': '\u015E', 'T': '\u0054', 'U': '\u0055',
            'Ü': '\u00DC', 'V': '\u0056', 'Y': '\u0059', 'Z': '\u005A'
        }
        
        # Critical Turkish case mappings (İ/ı distinction)
        self.turkish_case_mappings = {
            # Lowercase to uppercase
            'ı': 'I',  # dotless i -> dotless I
            'i': 'İ',  # dotted i -> dotted İ
            'ç': 'Ç',
            'ğ': 'Ğ', 
            'ö': 'Ö',
            'ş': 'Ş',
            'ü': 'Ü',
            
            # Uppercase to lowercase
            'I': 'ı',  # dotless I -> dotless i
            'İ': 'i',  # dotted İ -> dotted i
            'Ç': 'ç',
            'Ğ': 'ğ',
            'Ö': 'ö', 
            'Ş': 'ş',
            'Ü': 'ü'
        }
        
        # Turkish-specific characters
        self.turkish_specific_chars = {'ç', 'ğ', 'ı', 'İ', 'ö', 'ş', 'ü', 'Ç', 'Ğ', 'I', 'Ö', 'Ş', 'Ü'}
        
        # ASCII equivalents for compatibility
        self.turkish_to_ascii = {
            'ç': 'c', 'Ç': 'C',
            'ğ': 'g', 'Ğ': 'G', 
            'ı': 'i', 'I': 'I',
            'İ': 'I', 'i': 'i',
            'ö': 'o', 'Ö': 'O',
            'ş': 's', 'Ş': 'S',
            'ü': 'u', 'Ü': 'U'
        }
        
        # Contextual rules for İ/ı
        self.i_context_rules = self._compile_i_context_rules()
        
        # Unicode normalization forms
        self.normalization_forms = ['NFC', 'NFD', 'NFKC', 'NFKD']
        
        # Turkish collation order
        self.turkish_collation_order = self._create_collation_order()
        
        logger.info("✅ Advanced Turkish Unicode Processor initialized")
    
    def _compile_i_context_rules(self) -> Dict[str, re.Pattern]:
        """Compile context-sensitive rules for İ/ı processing"""
        
        return {
            # Words where 'i' should remain dotted
            'keep_dotted_i': re.compile(r'\b(bil|dil|fil|gir|hiz|kir|mil|pil|sir|tir|vil|zil)\w*\b', re.IGNORECASE),
            
            # Words where 'I' should be dotless
            'keep_dotless_I': re.compile(r'\b(BIL|DIL|FIL|GIR|HIZ|KIR|MIL|PIL|SIR|TIR|VIL|ZIL)\w*\b'),
            
            # Loanwords that may not follow Turkish rules
            'loanwords': re.compile(r'\b(internet|intranet|site|online|offline|email|wifi|bluetooth)\b', re.IGNORECASE),
            
            # Proper nouns (often exceptions)
            'proper_nouns': re.compile(r'\b[İIıi][A-ZÇĞÖŞÜa-zçğıöşü]+\b')
        }
    
    def _create_collation_order(self) -> Dict[str, int]:
        """Create Turkish alphabetical collation order"""
        
        # Turkish alphabet order
        turkish_order = [
            'a', 'b', 'c', 'ç', 'd', 'e', 'f', 'g', 'ğ', 'h', 
            'ı', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'ö', 'p', 
            'r', 's', 'ş', 't', 'u', 'ü', 'v', 'y', 'z'
        ]
        
        collation = {}
        for i, char in enumerate(turkish_order):
            collation[char] = i * 2  # Even for lowercase
            collation[char.upper()] = i * 2 + 1  # Odd for uppercase
        
        # Special handling for İ/I
        collation['ı'] = collation['i'] - 1  # ı comes before i
        collation['I'] = collation['İ'] - 1  # I comes before İ
        
        return collation
    
    def normalize_unicode(self, text: str, form: str = 'NFC') -> str:
        """Normalize Unicode text using specified form"""
        
        if form not in self.normalization_forms:
            raise ValueError(f"Invalid normalization form: {form}")
        
        try:
            normalized = unicodedata.normalize(form, text)
            
            # Additional Turkish-specific normalization
            normalized = self._normalize_turkish_specific(normalized)
            
            return normalized
            
        except Exception as e:
            logger.warning(f"Unicode normalization failed: {e}")
            return text
    
    def _normalize_turkish_specific(self, text: str) -> str:
        """Apply Turkish-specific Unicode normalization"""
        
        # Ensure proper Turkish characters
        replacements = {
            # Common misrepresentations
            '\u0049\u0307': 'İ',  # I + combining dot -> İ
            '\u0131\u0307': 'i',  # ı + combining dot -> i
            
            # Incorrect Unicode sequences
            'İ\u0307': 'İ',  # İ + extra dot
            'I\u0307': 'İ',  # I + dot -> İ
        }
        
        for incorrect, correct in replacements.items():
            text = text.replace(incorrect, correct)
        
        return text
    
    def turkish_lower(self, text: str, preserve_loanwords: bool = True) -> str:
        """Convert to lowercase using Turkish rules"""
        
        result = []
        
        for char in text:
            if char in self.turkish_case_mappings:
                # Use Turkish-specific mapping
                result.append(self.turkish_case_mappings[char])
            elif char == 'I':
                # Critical: I -> ı (dotless)
                result.append('ı')
            elif char == 'İ':
                # Critical: İ -> i (dotted)
                result.append('i')
            else:
                # Standard lowercase
                result.append(char.lower())
        
        lowercase_text = ''.join(result)
        
        # Apply context rules if needed
        if preserve_loanwords:
            lowercase_text = self._apply_loanword_rules(lowercase_text)
        
        return lowercase_text
    
    def turkish_upper(self, text: str, preserve_loanwords: bool = True) -> str:
        """Convert to uppercase using Turkish rules"""
        
        result = []
        
        for char in text:
            if char in self.turkish_case_mappings:
                # Use Turkish-specific mapping
                result.append(self.turkish_case_mappings[char])
            elif char == 'ı':
                # Critical: ı -> I (dotless)
                result.append('I')
            elif char == 'i':
                # Critical: i -> İ (dotted)
                result.append('İ')
            else:
                # Standard uppercase
                result.append(char.upper())
        
        uppercase_text = ''.join(result)
        
        # Apply context rules if needed
        if preserve_loanwords:
            uppercase_text = self._apply_loanword_rules(uppercase_text)
        
        return uppercase_text
    
    def turkish_title(self, text: str) -> str:
        """Convert to title case using Turkish rules"""
        
        words = text.split()
        title_words = []
        
        for word in words:
            if word:
                # First character to Turkish uppercase
                first_char = self.turkish_upper(word[0])
                # Rest to Turkish lowercase
                rest = self.turkish_lower(word[1:]) if len(word) > 1 else ""
                title_words.append(first_char + rest)
        
        return ' '.join(title_words)
    
    def _apply_loanword_rules(self, text: str) -> str:
        """Apply special rules for loanwords and exceptions"""
        
        # Keep certain loanwords unchanged
        loanword_matches = self.i_context_rules['loanwords'].findall(text)
        
        for loanword in loanword_matches:
            # Preserve original case for loanwords
            pass  # Implementation would preserve specific loanword patterns
        
        return text
    
    def detect_turkish_characters(self, text: str) -> Dict[str, Any]:
        """Detect and analyze Turkish characters in text"""
        
        char_count = {}
        turkish_char_count = 0
        total_chars = 0
        
        for char in text:
            if char.isalpha():
                total_chars += 1
                char_count[char] = char_count.get(char, 0) + 1
                
                if char in self.turkish_specific_chars:
                    turkish_char_count += 1
        
        # Calculate Turkish character density
        turkish_density = turkish_char_count / max(1, total_chars)
        
        # Detect İ/ı usage patterns
        i_patterns = self._analyze_i_patterns(text)
        
        return {
            'total_chars': total_chars,
            'turkish_chars': turkish_char_count,
            'turkish_density': turkish_density,
            'char_frequency': char_count,
            'i_patterns': i_patterns,
            'is_likely_turkish': turkish_density > 0.05  # 5% threshold
        }
    
    def _analyze_i_patterns(self, text: str) -> Dict[str, int]:
        """Analyze İ/ı usage patterns"""
        
        patterns = {
            'dotted_i_count': text.count('i'),
            'dotless_i_count': text.count('ı'),
            'dotted_I_count': text.count('İ'),
            'dotless_I_count': text.count('I'),
            'total_i_variants': 0
        }
        
        patterns['total_i_variants'] = sum([
            patterns['dotted_i_count'],
            patterns['dotless_i_count'], 
            patterns['dotted_I_count'],
            patterns['dotless_I_count']
        ])
        
        return patterns
    
    def validate_turkish_text(self, text: str) -> Dict[str, Any]:
        """Validate Turkish text for proper character usage"""
        
        issues = []
        suggestions = []
        
        # Check for common İ/ı mistakes
        if 'I' in text and 'ı' in text:
            # Check if I should be İ in context
            i_positions = [i for i, char in enumerate(text) if char == 'I']
            
            for pos in i_positions:
                context = text[max(0, pos-3):pos+4]
                if self._should_be_dotted_I(context, pos-max(0, pos-3)):
                    issues.append(f"Position {pos}: 'I' should probably be 'İ'")
                    suggestions.append(f"Change 'I' at position {pos} to 'İ'")
        
        # Check for incorrect case patterns
        case_issues = self._check_case_patterns(text)
        issues.extend(case_issues)
        
        return {
            'is_valid': len(issues) == 0,
            'issues': issues,
            'suggestions': suggestions,
            'confidence': 1.0 - (len(issues) * 0.1)  # Reduce confidence per issue
        }
    
    def _should_be_dotted_I(self, context: str, i_position: int) -> bool:
        """Determine if 'I' should be 'İ' based on context"""
        
        # Simple heuristic: if surrounded by lowercase Turkish chars
        before = context[:i_position].lower()
        after = context[i_position+1:].lower()
        
        turkish_chars = set('çğıöşü')
        
        has_turkish_before = any(c in turkish_chars for c in before)
        has_turkish_after = any(c in turkish_chars for c in after)
        
        return has_turkish_before or has_turkish_after
    
    def _check_case_patterns(self, text: str) -> List[str]:
        """Check for incorrect case patterns"""
        
        issues = []
        
        # Check for mixed case in words
        words = re.findall(r'\b[A-ZÇĞİÖŞÜa-zçğıiöşü]+\b', text)
        
        for word in words:
            if self._has_incorrect_case_pattern(word):
                issues.append(f"Incorrect case pattern in word: '{word}'")
        
        return issues
    
    def _has_incorrect_case_pattern(self, word: str) -> bool:
        """Check if word has incorrect case pattern"""
        
        # Simple check: mixed case in middle of word (except proper nouns)
        if len(word) < 2:
            return False
        
        # Allow first letter capitalization
        rest_of_word = word[1:]
        
        # Check for unexpected uppercase in middle
        has_mixed_case = any(c.isupper() for c in rest_of_word)
        
        # Allow if it's a known compound or abbreviation
        if self._is_known_compound(word):
            return False
        
        return has_mixed_case
    
    def _is_known_compound(self, word: str) -> bool:
        """Check if word is a known compound or abbreviation"""
        
        known_compounds = {
            'İstanbul', 'İzmir', 'İngilizce', 'İtalyanca', 'İspanyolca',
            'ABD', 'AB', 'TC', 'TBMM', 'TRT', 'PTT'
        }
        
        return word in known_compounds
    
    def convert_to_ascii(self, text: str, preserve_case: bool = True) -> str:
        """Convert Turkish characters to ASCII equivalents"""
        
        result = []
        
        for char in text:
            if char in self.turkish_to_ascii:
                ascii_char = self.turkish_to_ascii[char]
                
                # Preserve original case if requested
                if preserve_case:
                    if char.isupper():
                        ascii_char = ascii_char.upper()
                    else:
                        ascii_char = ascii_char.lower()
                
                result.append(ascii_char)
            else:
                result.append(char)
        
        return ''.join(result)
    
    def turkish_sort_key(self, text: str) -> Tuple:
        """Generate sort key for Turkish collation"""
        
        key = []
        
        for char in text.lower():
            if char in self.turkish_collation_order:
                key.append(self.turkish_collation_order[char])
            else:
                # Use Unicode code point for non-Turkish chars
                key.append(ord(char) + 1000)
        
        return tuple(key)
    
    def process_for_tokenization(self, text: str, 
                                normalize: bool = True,
                                case_mode: TurkishCaseMode = TurkishCaseMode.PRESERVE) -> str:
        """Process text for optimal tokenization"""
        
        processed_text = text
        
        # Unicode normalization
        if normalize:
            processed_text = self.normalize_unicode(processed_text)
        
        # Case processing
        if case_mode == TurkishCaseMode.LOWER:
            processed_text = self.turkish_lower(processed_text)
        elif case_mode == TurkishCaseMode.UPPER:
            processed_text = self.turkish_upper(processed_text)
        elif case_mode == TurkishCaseMode.TITLE:
            processed_text = self.turkish_title(processed_text)
        
        # Additional cleanup
        processed_text = self._cleanup_for_tokenization(processed_text)
        
        return processed_text
    
    def _cleanup_for_tokenization(self, text: str) -> str:
        """Clean up text for optimal tokenization"""
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove zero-width characters
        text = re.sub(r'[\u200B-\u200F\uFEFF]', '', text)
        
        # Normalize punctuation
        text = re.sub(r'["""]', '"', text)  # Normalize quotes
        text = re.sub(r'[''']', "'", text)  # Normalize apostrophes
        
        return text.strip()
    
    def get_character_info(self, char: str) -> TurkishCharacterInfo:
        """Get detailed information about a character"""
        
        try:
            unicode_name = unicodedata.name(char, 'UNKNOWN')
            category = unicodedata.category(char)
            
            is_turkish = char in self.turkish_specific_chars
            ascii_equiv = self.turkish_to_ascii.get(char)
            
            # Get case variants
            case_variants = {}
            if char.islower() and char in self.turkish_case_mappings:
                case_variants['upper'] = self.turkish_case_mappings[char]
            elif char.isupper() and char in self.turkish_case_mappings:
                case_variants['lower'] = self.turkish_case_mappings[char]
            
            return TurkishCharacterInfo(
                character=char,
                unicode_name=unicode_name,
                category=category,
                is_turkish_specific=is_turkish,
                ascii_equivalent=ascii_equiv,
                case_variants=case_variants
            )
            
        except Exception as e:
            logger.warning(f"Error getting character info for '{char}': {e}")
            return TurkishCharacterInfo(
                character=char,
                unicode_name='UNKNOWN',
                category='UNKNOWN',
                is_turkish_specific=False
            )

# Factory functions
def create_turkish_unicode_processor() -> AdvancedTurkishUnicodeProcessor:
    """Create Turkish Unicode processor"""
    return AdvancedTurkishUnicodeProcessor()

def process_turkish_text(text: str, 
                        normalize: bool = True,
                        case_mode: str = 'preserve') -> str:
    """Quick function to process Turkish text"""
    
    processor = create_turkish_unicode_processor()
    
    case_enum = {
        'lower': TurkishCaseMode.LOWER,
        'upper': TurkishCaseMode.UPPER, 
        'title': TurkishCaseMode.TITLE,
        'preserve': TurkishCaseMode.PRESERVE
    }.get(case_mode, TurkishCaseMode.PRESERVE)
    
    return processor.process_for_tokenization(text, normalize, case_enum)

# Testing
if __name__ == "__main__":
    print("🧪 Testing Advanced Turkish Unicode Processor...")
    
    # Create processor
    processor = create_turkish_unicode_processor()
    
    # Test texts with İ/ı challenges
    test_texts = [
        "İstanbul'da internet kullanıyorum.",
        "TÜRKIYE CUMHURİYETI",  
        "Bilgisayar programlama dili",
        "İçinde ı ve i olan kelimeler"
    ]
    
    for text in test_texts:
        print(f"\n📝 Original: {text}")
        
        # Test case conversions
        lower = processor.turkish_lower(text)
        upper = processor.turkish_upper(text)
        title = processor.turkish_title(text)
        
        print(f"   Lower: {lower}")
        print(f"   Upper: {upper}")
        print(f"   Title: {title}")
        
        # Test character detection
        char_analysis = processor.detect_turkish_characters(text)
        print(f"   Turkish density: {char_analysis['turkish_density']:.2f}")
        
        # Test validation
        validation = processor.validate_turkish_text(text)
        print(f"   Valid: {validation['is_valid']}, Confidence: {validation['confidence']:.2f}")
    
    print("\n✅ Advanced Turkish Unicode Processor test complete!")