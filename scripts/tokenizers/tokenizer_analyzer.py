"""
Turkish Text Tokenization Analyzer
Clean Code Implementation with SOLID Principles
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from enum import Enum
from abc import ABC, abstractmethod
import tiktoken
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TokenizerConstants:
    """Application constants - No magic numbers!"""
    TURKISH_CHARS = set('ğüşıöçĞÜŞİÖÇ')
    EFFICIENCY_THRESHOLD_HIGH = 0.3
    EFFICIENCY_THRESHOLD_LOW = 0.5
    MAX_DISPLAY_TOKENS = 10
    MAX_TEXT_PREVIEW = 100
    DEFAULT_ENCODING = "cl100k_base"
    
    # Display formatting
    SEPARATOR_WIDTH = 70
    SUBSECTION_WIDTH = 50


class TokenizerEfficiency(Enum):
    """Tokenizer efficiency levels for Turkish text"""
    HIGH = "✅ High - Efficient for Turkish"
    MODERATE = "≈ Moderate - Acceptable performance"
    LOW = "⚠️ Low - Inefficient for Turkish"


@dataclass(frozen=True)
class CharacterStatistics:
    """Character analysis statistics"""
    total_chars: int
    turkish_char_count: int
    turkish_char_ratio: float
    word_count: int
    average_word_length: float


@dataclass(frozen=True)
class TokenizationResult:
    """Complete tokenization analysis result"""
    text: str
    token_count: int
    tokens: List[int]
    tokens_per_word: float
    tokens_per_char: float
    character_stats: CharacterStatistics
    efficiency: TokenizerEfficiency


class ITokenizer(ABC):
    """Tokenizer interface - Dependency Inversion Principle"""
    
    @abstractmethod
    def encode(self, text: str) -> List[int]:
        """Encode text to tokens"""
        pass
    
    @abstractmethod
    def decode_token(self, token_id: int) -> Optional[str]:
        """Decode a single token"""
        pass
    
    @abstractmethod
    def get_encoding_name(self) -> str:
        """Get the encoding name"""
        pass


class TikTokenWrapper(ITokenizer):
    """TikToken implementation of ITokenizer"""
    
    def __init__(self, encoding_name: str = TokenizerConstants.DEFAULT_ENCODING):
        self._encoding_name = encoding_name
        self._tokenizer = tiktoken.get_encoding(encoding_name)
        logger.info(f"Initialized TikToken with encoding: {encoding_name}")
    
    def encode(self, text: str) -> List[int]:
        """Encode text to tokens"""
        return self._tokenizer.encode(text)
    
    def decode_token(self, token_id: int) -> Optional[str]:
        """Decode a single token safely"""
        try:
            token_bytes = self._tokenizer.decode_single_token_bytes(token_id)
            return token_bytes.decode('utf-8')
        except UnicodeDecodeError:
            return None
        except Exception as e:
            logger.warning(f"Error decoding token {token_id}: {e}")
            return None
    
    def get_encoding_name(self) -> str:
        """Get the encoding name"""
        return self._encoding_name


class CharacterAnalyzer:
    """Analyzes character statistics - Single Responsibility Principle"""
    
    @staticmethod
    def analyze(text: str) -> CharacterStatistics:
        """Analyze character statistics of text"""
        if not text:
            return CharacterStatistics(0, 0, 0.0, 0, 0.0)
        
        turkish_count = sum(1 for c in text if c in TokenizerConstants.TURKISH_CHARS)
        total_chars = len(text)
        words = text.split()
        word_count = len(words)
        
        return CharacterStatistics(
            total_chars=total_chars,
            turkish_char_count=turkish_count,
            turkish_char_ratio=turkish_count / total_chars if total_chars > 0 else 0.0,
            word_count=word_count,
            average_word_length=sum(len(w) for w in words) / word_count if word_count > 0 else 0.0
        )


class EfficiencyCalculator:
    """Calculates tokenization efficiency - Single Responsibility Principle"""
    
    @staticmethod
    def calculate(tokens_per_char: float) -> TokenizerEfficiency:
        """Calculate efficiency based on tokens per character ratio"""
        if tokens_per_char < TokenizerConstants.EFFICIENCY_THRESHOLD_HIGH:
            return TokenizerEfficiency.HIGH
        elif tokens_per_char > TokenizerConstants.EFFICIENCY_THRESHOLD_LOW:
            return TokenizerEfficiency.LOW
        else:
            return TokenizerEfficiency.MODERATE


class TokenizationAnalyzer:
    """Main analyzer class - Coordinates the analysis process"""
    
    def __init__(self, tokenizer: ITokenizer):
        """Initialize with tokenizer (Dependency Injection)"""
        self._tokenizer = tokenizer
        self._char_analyzer = CharacterAnalyzer()
        self._efficiency_calculator = EfficiencyCalculator()
        logger.info("TokenizationAnalyzer initialized")
    
    def analyze(self, text: str) -> TokenizationResult:
        """Perform complete tokenization analysis"""
        if not text:
            raise ValueError("Text cannot be empty")
        
        # Tokenize
        tokens = self._tokenizer.encode(text)
        
        # Analyze characters
        char_stats = self._char_analyzer.analyze(text)
        
        # Calculate metrics
        tokens_per_word = len(tokens) / char_stats.word_count if char_stats.word_count > 0 else 0
        tokens_per_char = len(tokens) / char_stats.total_chars if char_stats.total_chars > 0 else 0
        
        # Calculate efficiency
        efficiency = self._efficiency_calculator.calculate(tokens_per_char)
        
        return TokenizationResult(
            text=text,
            token_count=len(tokens),
            tokens=tokens,
            tokens_per_word=tokens_per_word,
            tokens_per_char=tokens_per_char,
            character_stats=char_stats,
            efficiency=efficiency
        )


class ResultFormatter:
    """Formats and displays results - Single Responsibility Principle"""
    
    def __init__(self, tokenizer: ITokenizer):
        self._tokenizer = tokenizer
    
    def format_result(self, result: TokenizationResult) -> str:
        """Format result as string"""
        lines = [
            f"Text Preview: {result.text[:TokenizerConstants.MAX_TEXT_PREVIEW]}{'...' if len(result.text) > TokenizerConstants.MAX_TEXT_PREVIEW else ''}",
            f"Total Tokens: {result.token_count}",
            f"Tokens per Word: {result.tokens_per_word:.2f}",
            f"Tokens per Character: {result.tokens_per_char:.3f}",
            f"Turkish Character Ratio: {result.character_stats.turkish_char_ratio:.1%}",
            f"Efficiency: {result.efficiency.value}"
        ]
        return "\n".join(lines)
    
    def format_token_breakdown(self, result: TokenizationResult, limit: int = None) -> str:
        """Format token breakdown"""
        if limit is None:
            limit = TokenizerConstants.MAX_DISPLAY_TOKENS
        
        lines = [f"Token Breakdown ({result.token_count} total):"]
        
        for i, token_id in enumerate(result.tokens[:limit]):
            token_str = self._tokenizer.decode_token(token_id)
            if token_str:
                lines.append(f"  [{i:3d}] {token_id:6d}: '{token_str}'")
            else:
                lines.append(f"  [{i:3d}] {token_id:6d}: <binary>")
        
        if result.token_count > limit:
            lines.append(f"  ... and {result.token_count - limit} more tokens")
        
        return "\n".join(lines)
    
    def print_result(self, result: TokenizationResult, show_tokens: bool = False):
        """Print formatted result"""
        print("\n" + "=" * TokenizerConstants.SEPARATOR_WIDTH)
        print(self.format_result(result))
        
        if show_tokens:
            print("\n" + "-" * TokenizerConstants.SUBSECTION_WIDTH)
            print(self.format_token_breakdown(result))
        
        print("=" * TokenizerConstants.SEPARATOR_WIDTH)


class TurkishTextBenchmark:
    """Benchmarks tokenizer on Turkish texts"""
    
    DEFAULT_TEST_TEXTS = [
        "Merhaba, Türkçe doğal dil işleme modelleri üzerinde çalışıyorum.",
        "İstanbul'un tarihi yarımadası, UNESCO Dünya Mirası Listesi'nde yer almaktadır.",
        "Yapay zeka ve makine öğrenmesi teknolojileri hızla gelişmektedir.",
        "Türkiye'nin coğrafi konumu, Avrupa ve Asya kıtaları arasında köprü görevi görmektedir.",
        "Günümüzde bilgisayarlı görü sistemleri birçok alanda kullanılmaktadır.",
        "Türkçe karakterler: ğ, ü, ş, ı, ö, ç içeren bir metin örneği."
    ]
    
    def __init__(self, analyzer: TokenizationAnalyzer, formatter: ResultFormatter):
        """Initialize with analyzer and formatter (Dependency Injection)"""
        self._analyzer = analyzer
        self._formatter = formatter
        self._results: List[TokenizationResult] = []
    
    def run(self, texts: Optional[List[str]] = None) -> List[TokenizationResult]:
        """Run benchmark on texts"""
        if texts is None:
            texts = self.DEFAULT_TEST_TEXTS
        
        logger.info(f"Running benchmark on {len(texts)} texts")
        self._results = []
        
        for i, text in enumerate(texts, 1):
            try:
                result = self._analyzer.analyze(text)
                self._results.append(result)
                
                print(f"\n{'='*TokenizerConstants.SEPARATOR_WIDTH}")
                print(f"TEST {i}/{len(texts)}")
                self._formatter.print_result(result, show_tokens=False)
                
            except Exception as e:
                logger.error(f"Error analyzing text {i}: {e}")
        
        self._print_summary()
        return self._results
    
    def _print_summary(self):
        """Print benchmark summary"""
        if not self._results:
            print("No results to summarize")
            return
        
        print("\n" + "=" * TokenizerConstants.SEPARATOR_WIDTH)
        print("BENCHMARK SUMMARY")
        print("=" * TokenizerConstants.SEPARATOR_WIDTH)
        
        avg_tokens = sum(r.token_count for r in self._results) / len(self._results)
        min_tokens = min(r.token_count for r in self._results)
        max_tokens = max(r.token_count for r in self._results)
        
        efficient_count = sum(1 for r in self._results if r.efficiency == TokenizerEfficiency.HIGH)
        moderate_count = sum(1 for r in self._results if r.efficiency == TokenizerEfficiency.MODERATE)
        inefficient_count = sum(1 for r in self._results if r.efficiency == TokenizerEfficiency.LOW)
        
        print(f"Total Tests: {len(self._results)}")
        print(f"Average Tokens: {avg_tokens:.1f}")
        print(f"Min/Max Tokens: {min_tokens}/{max_tokens}")
        print(f"\nEfficiency Distribution:")
        print(f"  {TokenizerEfficiency.HIGH.value}: {efficient_count}")
        print(f"  {TokenizerEfficiency.MODERATE.value}: {moderate_count}")
        print(f"  {TokenizerEfficiency.LOW.value}: {inefficient_count}")


def main():
    """Main entry point"""
    try:
        # Create components using dependency injection
        tokenizer = TikTokenWrapper()
        analyzer = TokenizationAnalyzer(tokenizer)
        formatter = ResultFormatter(tokenizer)
        benchmark = TurkishTextBenchmark(analyzer, formatter)
        
        # Run benchmark
        results = benchmark.run()
        
        # Interactive mode
        print("\n" + "=" * TokenizerConstants.SEPARATOR_WIDTH)
        print("INTERACTIVE MODE")
        print("=" * TokenizerConstants.SEPARATOR_WIDTH)
        
        while True:
            user_input = input("\nEnter Turkish text to analyze (or 'quit' to exit): ").strip()
            if user_input.lower() in ['quit', 'exit', 'q']:
                break
            
            if user_input:
                try:
                    result = analyzer.analyze(user_input)
                    formatter.print_result(result, show_tokens=True)
                except Exception as e:
                    print(f"Error: {e}")
        
        print("\nThank you for using Turkish Tokenization Analyzer!")
        
    except Exception as e:
        logger.error(f"Application error: {e}")
        raise


if __name__ == "__main__":
    main()
