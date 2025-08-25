"""
Advanced Turkish Morphology Module
TEKNOFEST 2025 - Comprehensive Turkish NLP Support
"""

import re
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
import json
from pathlib import Path


class MorphemeType(Enum):
    """Turkish morpheme types"""
    ROOT = "ROOT"
    DERIVATIONAL = "DERIVATIONAL"
    INFLECTIONAL = "INFLECTIONAL"
    CASE = "CASE"
    POSSESSIVE = "POSSESSIVE"
    PLURAL = "PLURAL"
    TENSE = "TENSE"
    PERSON = "PERSON"
    NEGATIVE = "NEGATIVE"
    QUESTION = "QUESTION"
    CAUSATIVE = "CAUSATIVE"
    PASSIVE = "PASSIVE"
    REFLEXIVE = "REFLEXIVE"
    RECIPROCAL = "RECIPROCAL"
    ABILITY = "ABILITY"


class VowelType(Enum):
    """Turkish vowel classifications"""
    FRONT = ["e", "i", "ö", "ü"]
    BACK = ["a", "ı", "o", "u"]
    ROUNDED = ["o", "ö", "u", "ü"]
    UNROUNDED = ["a", "e", "ı", "i"]
    HIGH = ["ı", "i", "u", "ü"]
    LOW = ["a", "e", "o", "ö"]


@dataclass
class Morpheme:
    """Represents a Turkish morpheme"""
    surface: str
    morpheme_type: MorphemeType
    features: Dict[str, str] = field(default_factory=dict)
    lemma: Optional[str] = None


@dataclass
class MorphologicalAnalysis:
    """Complete morphological analysis of a Turkish word"""
    surface_form: str
    root: str
    morphemes: List[Morpheme]
    pos_tag: str
    features: Dict[str, str]
    derivations: List[str]
    is_compound: bool = False
    compound_parts: List[str] = field(default_factory=list)


class TurkishMorphologyAnalyzer:
    """Advanced Turkish morphological analyzer with comprehensive linguistic rules"""
    
    def __init__(self):
        self.vowels = set("aeıioöuü")
        self.consonants = set("bcçdfgğhjklmnprsştvyz")
        self.hard_consonants = set("çfhkpsşt")
        self.soft_consonants = set("bcdgğjlmnrvyz")
        
        # Initialize morphological rules
        self._init_suffix_rules()
        self._init_stem_alternations()
        self._init_compound_patterns()
        
    def _init_suffix_rules(self):
        """Initialize Turkish suffix attachment rules"""
        self.case_suffixes = {
            "nominative": "",
            "accusative": {"vowel": "yı/yi/yu/yü", "consonant": "ı/i/u/ü"},
            "dative": {"vowel": "ya/ye", "consonant": "a/e"},
            "locative": {"vowel": "da/de", "consonant": "ta/te"},
            "ablative": {"vowel": "dan/den", "consonant": "tan/ten"},
            "genitive": {"vowel": "nın/nin/nun/nün", "consonant": "ın/in/un/ün"},
            "instrumental": {"vowel": "yla/yle", "consonant": "la/le"}
        }
        
        self.possessive_suffixes = {
            "1sg": {"vowel": "m", "consonant": "ım/im/um/üm"},
            "2sg": {"vowel": "n", "consonant": "ın/in/un/ün"},
            "3sg": {"vowel": "sı/si/su/sü", "consonant": "ı/i/u/ü"},
            "1pl": {"vowel": "mız/miz/muz/müz", "consonant": "ımız/imiz/umuz/ümüz"},
            "2pl": {"vowel": "nız/niz/nuz/nüz", "consonant": "ınız/iniz/unuz/ünüz"},
            "3pl": {"vowel": "ları/leri", "consonant": "ları/leri"}
        }
        
        self.tense_suffixes = {
            "present": "yor",
            "past": "dı/di/du/dü/tı/ti/tu/tü",
            "future": "acak/ecek",
            "aorist": "ar/er/ır/ir/ur/ür",
            "necessitative": "malı/meli",
            "conditional": "sa/se"
        }
        
        self.derivational_suffixes = {
            "noun_to_verb": ["la", "le", "lan", "len", "laş", "leş"],
            "verb_to_noun": ["ma", "me", "mak", "mek", "ış", "iş", "uş", "üş"],
            "noun_to_adj": ["lı", "li", "lu", "lü", "sız", "siz", "suz", "süz"],
            "adj_to_noun": ["lık", "lik", "luk", "lük"],
            "causative": ["dır", "dir", "dur", "dür", "tır", "tir", "tur", "tür"],
            "passive": ["ıl", "il", "ul", "ül", "n"],
            "reflexive": ["ın", "in", "un", "ün"],
            "reciprocal": ["ış", "iş", "uş", "üş"],
            "ability": ["abil", "ebil"]
        }
        
    def _init_stem_alternations(self):
        """Initialize stem alternation rules (consonant and vowel changes)"""
        self.consonant_alternations = {
            # Consonant mutation rules (sertleşme/yumuşama)
            "p": "b",  # kitap -> kitabı
            "ç": "c",  # ağaç -> ağacı
            "t": "d",  # kanat -> kanadı
            "k": "ğ",  # köpek -> köpeği
            "nk": "ng", # renk -> rengi
        }
        
        self.vowel_drops = {
            # Vowel drop patterns (ünlü düşmesi)
            "burun": "burn",  # burun -> burnu
            "ağız": "ağz",    # ağız -> ağzı
            "boyun": "boyn",  # boyun -> boynu
            "oğul": "oğl",    # oğul -> oğlu
        }
        
        self.buffer_consonants = {
            # Buffer consonant rules (kaynaştırma harfleri)
            "n": ["genitive", "accusative", "possessive"],  # su -> suyun (with y-buffer)
            "y": ["dative", "accusative"],  # su -> suya
            "s": ["3sg_possessive"],  # su -> suyu -> suyusu
        }
        
    def _init_compound_patterns(self):
        """Initialize compound word patterns"""
        self.compound_patterns = [
            # Pattern: Noun + Noun compounds
            (r"(\w+)\s*(\w+)", ["NN"]),  # kahve + hane = kahvehane
            # Pattern: Adjective + Noun compounds  
            (r"(\w+)\s*(\w+)", ["JN"]),  # büyük + anne = büyükanne
            # Pattern: Verb + Verb compounds
            (r"(\w+)[ae]\s*(\w+)", ["VV"]),  # yaza + bil = yazabil
        ]
        
    def analyze_vowel_harmony(self, word: str) -> Dict[str, bool]:
        """Analyze vowel harmony in a Turkish word"""
        vowels_in_word = [char for char in word if char in self.vowels]
        
        if len(vowels_in_word) < 2:
            return {"valid": True, "front_back": True, "rounded_unrounded": True}
        
        # Check front-back harmony (kalınlık-incelik uyumu)
        front_vowels = set("eiöü")
        back_vowels = set("aıou")
        
        word_vowel_set = set(vowels_in_word)
        has_front = bool(word_vowel_set & front_vowels)
        has_back = bool(word_vowel_set & back_vowels)
        
        front_back_harmony = not (has_front and has_back)
        
        # Check rounded-unrounded harmony (düzlük-yuvarlaklık uyumu)
        rounded_harmony = True
        for i in range(len(vowels_in_word) - 1):
            current = vowels_in_word[i]
            next_vowel = vowels_in_word[i + 1]
            
            # Complex harmony rules
            if current in "ou" and next_vowel not in "au":
                rounded_harmony = False
            elif current in "öü" and next_vowel not in "eü":
                rounded_harmony = False
                
        return {
            "valid": front_back_harmony and rounded_harmony,
            "front_back": front_back_harmony,
            "rounded_unrounded": rounded_harmony,
            "vowel_sequence": vowels_in_word
        }
    
    def apply_consonant_mutation(self, stem: str, suffix_type: str) -> str:
        """Apply consonant mutation rules (ünsüz değişimi)"""
        if not stem:
            return stem
            
        last_char = stem[-1]
        
        # Check if mutation is needed based on suffix type
        mutation_triggers = ["accusative", "dative", "genitive", "possessive"]
        
        if suffix_type not in mutation_triggers:
            return stem
            
        # Apply hard to soft consonant changes
        if last_char in self.hard_consonants:
            for hard, soft in self.consonant_alternations.items():
                if stem.endswith(hard):
                    return stem[:-len(hard)] + soft
                    
        # Special case for -nk -> -ng
        if stem.endswith("nk"):
            return stem[:-2] + "ng"
            
        return stem
    
    def apply_vowel_drop(self, stem: str) -> Tuple[str, bool]:
        """Apply vowel drop rules (ünlü düşmesi)"""
        # Check common vowel drop patterns
        for full_form, dropped_form in self.vowel_drops.items():
            if stem == full_form:
                return dropped_form, True
                
        # Check second syllable vowel drop pattern
        syllables = self.syllabify(stem)
        if len(syllables) == 2:
            first_syl, second_syl = syllables
            # If second syllable is CV and ends with narrow vowel
            if len(second_syl) == 2 and second_syl[1] in "ıiu":
                return first_syl + second_syl[0], True
                
        return stem, False
    
    def determine_buffer_consonant(self, stem: str, suffix_type: str) -> str:
        """Determine if a buffer consonant is needed"""
        if not stem:
            return ""
            
        last_char = stem[-1]
        
        # Word ends with vowel
        if last_char in self.vowels:
            if suffix_type in ["dative", "accusative"]:
                return "y"
            elif suffix_type in ["genitive"]:
                return "n"
            elif suffix_type == "3sg_possessive":
                return "s"
                
        return ""
    
    def syllabify(self, word: str) -> List[str]:
        """Divide a Turkish word into syllables"""
        if not word:
            return []
            
        syllables = []
        current_syllable = ""
        
        for i, char in enumerate(word):
            current_syllable += char
            
            if char in self.vowels:
                # Look ahead for consonants
                j = i + 1
                while j < len(word) and word[j] in self.consonants:
                    # Turkish syllable structure rules
                    if j == i + 1:
                        # Single consonant goes to next syllable
                        if j + 1 < len(word) and word[j + 1] in self.vowels:
                            syllables.append(current_syllable)
                            current_syllable = ""
                            break
                        else:
                            current_syllable += word[j]
                    elif j == i + 2:
                        # Two consonants: first stays, second goes to next
                        current_syllable += word[i + 1]
                        syllables.append(current_syllable)
                        current_syllable = ""
                        break
                    j += 1
                    
                if j == len(word):
                    syllables.append(current_syllable)
                    current_syllable = ""
                    
        if current_syllable:
            if syllables and current_syllable[0] in self.consonants:
                syllables[-1] += current_syllable
            else:
                syllables.append(current_syllable)
                
        return syllables
    
    def analyze_compound(self, word: str) -> Tuple[bool, List[str]]:
        """Analyze if a word is a compound and split it"""
        # Common Turkish compound patterns
        compound_markers = [
            ("başkan", ["baş", "kan"]),  # false split to avoid
            ("kalemlik", ["kalem", "lik"]),
            ("öğretmen", ["öğret", "men"]),
        ]
        
        # Check against known compounds
        for compound, parts in compound_markers:
            if word == compound:
                return True, parts
                
        # Try to identify potential compound boundaries
        # This is simplified - real implementation would use dictionary
        potential_splits = []
        for i in range(3, len(word) - 2):
            left = word[:i]
            right = word[i:]
            
            # Check if both parts could be valid Turkish roots
            if self._is_valid_root(left) and self._is_valid_root(right):
                potential_splits.append([left, right])
                
        if potential_splits:
            # Return the most likely split (simplified heuristic)
            return True, potential_splits[0]
            
        return False, []
    
    def _is_valid_root(self, word: str) -> bool:
        """Check if a word segment could be a valid Turkish root"""
        if len(word) < 2:
            return False
            
        # Must contain at least one vowel
        if not any(char in self.vowels for char in word):
            return False
            
        # Check vowel harmony
        harmony = self.analyze_vowel_harmony(word)
        
        return harmony["valid"]
    
    def get_suffix_allomorph(self, stem: str, suffix_base: str) -> str:
        """Get the correct allomorph of a suffix based on vowel harmony"""
        if "/" not in suffix_base:
            return suffix_base
            
        # Get the last vowel of the stem
        last_vowel = None
        for char in reversed(stem):
            if char in self.vowels:
                last_vowel = char
                break
                
        if not last_vowel:
            return suffix_base.split("/")[0]
            
        # Determine the correct allomorph
        variants = suffix_base.split("/")
        
        # Front-back harmony
        if last_vowel in "eiöü":  # Front vowels
            if last_vowel in "ei":
                return variants[1] if len(variants) > 1 else variants[0]
            else:  # öü
                return variants[3] if len(variants) > 3 else variants[1]
        else:  # Back vowels aıou
            if last_vowel in "aı":
                return variants[0]
            else:  # ou
                return variants[2] if len(variants) > 2 else variants[0]
                
    def analyze(self, word: str) -> MorphologicalAnalysis:
        """Perform complete morphological analysis of a Turkish word"""
        # Initialize analysis
        morphemes = []
        features = {}
        derivations = []
        
        # Clean and normalize input
        word = word.lower().strip()
        
        # Check for compound
        is_compound, compound_parts = self.analyze_compound(word)
        
        # Simplified analysis (would use FST in production)
        # Try to identify root and suffixes
        root = self._extract_root(word)
        remaining = word[len(root):]
        
        # Add root morpheme
        morphemes.append(Morpheme(
            surface=root,
            morpheme_type=MorphemeType.ROOT,
            lemma=root
        ))
        
        # Analyze remaining suffixes
        if remaining:
            suffix_morphemes = self._analyze_suffixes(remaining, root)
            morphemes.extend(suffix_morphemes)
            
        # Extract features from morphemes
        for morpheme in morphemes:
            features.update(morpheme.features)
            
        # Determine POS tag
        pos_tag = self._determine_pos(morphemes)
        
        return MorphologicalAnalysis(
            surface_form=word,
            root=root,
            morphemes=morphemes,
            pos_tag=pos_tag,
            features=features,
            derivations=derivations,
            is_compound=is_compound,
            compound_parts=compound_parts
        )
    
    def _extract_root(self, word: str) -> str:
        """Extract the root from a Turkish word"""
        # Simplified root extraction
        # In production, this would use a comprehensive dictionary
        
        # Try progressively shorter substrings
        for i in range(len(word), 1, -1):
            candidate = word[:i]
            if self._is_valid_root(candidate):
                return candidate
                
        return word
    
    def _analyze_suffixes(self, suffix_string: str, root: str) -> List[Morpheme]:
        """Analyze the suffix chain"""
        morphemes = []
        
        # Simplified suffix analysis
        # Real implementation would use finite state transducer
        
        remaining = suffix_string
        current_stem = root
        
        while remaining:
            found_suffix = False
            
            # Try to match known suffixes
            for suffix_type, patterns in [
                ("plural", ["lar", "ler"]),
                ("possessive", ["ım", "im", "um", "üm", "m"]),
                ("case", ["ı", "i", "u", "ü", "a", "e", "da", "de", "dan", "den"]),
                ("tense", ["yor", "dı", "di", "du", "dü", "acak", "ecek"]),
            ]:
                for pattern in patterns:
                    if remaining.startswith(pattern):
                        morphemes.append(Morpheme(
                            surface=pattern,
                            morpheme_type=MorphemeType[suffix_type.upper()],
                            features={suffix_type: pattern}
                        ))
                        remaining = remaining[len(pattern):]
                        current_stem += pattern
                        found_suffix = True
                        break
                        
                if found_suffix:
                    break
                    
            if not found_suffix:
                # Unknown suffix, treat as derivational
                morphemes.append(Morpheme(
                    surface=remaining[0],
                    morpheme_type=MorphemeType.DERIVATIONAL,
                    features={"unknown": remaining[0]}
                ))
                remaining = remaining[1:]
                
        return morphemes
    
    def _determine_pos(self, morphemes: List[Morpheme]) -> str:
        """Determine part of speech from morpheme analysis"""
        # Simplified POS determination
        
        has_verb_suffix = any(
            m.morpheme_type in [MorphemeType.TENSE, MorphemeType.PERSON]
            for m in morphemes
        )
        
        if has_verb_suffix:
            return "VERB"
            
        has_case = any(m.morpheme_type == MorphemeType.CASE for m in morphemes)
        
        if has_case:
            return "NOUN"
            
        # Default to noun
        return "NOUN"
    
    def generate_surface_form(self, root: str, morphemes: List[Dict[str, str]]) -> str:
        """Generate surface form from root and morpheme specifications"""
        result = root
        current_stem = root
        
        for morpheme_spec in morphemes:
            suffix_type = morpheme_spec.get("type", "")
            
            # Apply phonological rules
            modified_stem = self.apply_consonant_mutation(current_stem, suffix_type)
            dropped_stem, vowel_dropped = self.apply_vowel_drop(modified_stem)
            
            if vowel_dropped:
                current_stem = dropped_stem
                
            # Add buffer consonant if needed
            buffer = self.determine_buffer_consonant(current_stem, suffix_type)
            if buffer:
                current_stem += buffer
                
            # Get correct suffix allomorph
            suffix_base = morpheme_spec.get("suffix", "")
            suffix = self.get_suffix_allomorph(current_stem, suffix_base)
            
            current_stem += suffix
            result = current_stem
            
        return result


class TurkishStemmer:
    """Rule-based Turkish stemmer"""
    
    def __init__(self):
        self.analyzer = TurkishMorphologyAnalyzer()
        
    def stem(self, word: str) -> str:
        """Extract the stem of a Turkish word"""
        analysis = self.analyzer.analyze(word)
        return analysis.root
    
    def stem_batch(self, words: List[str]) -> List[str]:
        """Stem multiple words efficiently"""
        return [self.stem(word) for word in words]


class TurkishLemmatizer:
    """Turkish lemmatizer with morphological analysis"""
    
    def __init__(self):
        self.analyzer = TurkishMorphologyAnalyzer()
        self.irregular_verbs = {
            # Irregular verb mappings
            "gider": "git",
            "yer": "ye",
            "der": "de",
        }
        
    def lemmatize(self, word: str, pos: Optional[str] = None) -> str:
        """Get the lemma of a Turkish word"""
        analysis = self.analyzer.analyze(word)
        
        # Check for irregular forms
        if analysis.root in self.irregular_verbs:
            return self.irregular_verbs[analysis.root]
            
        return analysis.root
    
    def lemmatize_batch(self, words: List[str]) -> List[str]:
        """Lemmatize multiple words efficiently"""
        return [self.lemmatize(word) for word in words]


def test_morphology_analyzer():
    """Test the Turkish morphology analyzer"""
    analyzer = TurkishMorphologyAnalyzer()
    
    test_words = [
        "evlerimizden",  # ev-ler-imiz-den (house-PL-POSS.1PL-ABL)
        "kitapları",     # kitap-lar-ı (book-PL-ACC/POSS.3)
        "öğretmenlik",   # öğretmen-lik (teacher-NESS)
        "yazabiliyorum", # yaz-abil-iyor-um (write-ABLE-PROG-1SG)
        "gelecekmiş",    # gel-ecek-miş (come-FUT-EVID)
    ]
    
    print("Turkish Morphological Analysis Tests")
    print("=" * 50)
    
    for word in test_words:
        print(f"\nAnalyzing: {word}")
        analysis = analyzer.analyze(word)
        
        print(f"  Root: {analysis.root}")
        print(f"  POS: {analysis.pos_tag}")
        print(f"  Morphemes: {[m.surface for m in analysis.morphemes]}")
        
        # Test vowel harmony
        harmony = analyzer.analyze_vowel_harmony(word)
        print(f"  Vowel Harmony: {'✓' if harmony['valid'] else '✗'}")
        
        # Test syllabification
        syllables = analyzer.syllabify(word)
        print(f"  Syllables: {'-'.join(syllables)}")


if __name__ == "__main__":
    test_morphology_analyzer()