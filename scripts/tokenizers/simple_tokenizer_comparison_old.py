import tiktoken
from transformers import AutoTokenizer
import os

def simple_tokenizer_comparison(text):
    """
    Compare tokenizers for Turkish text using simpler methods.
    This version uses tiktoken for GPT models which doesn't require authentication.
    """
    
    # Use tiktoken for GPT models (no auth required)
    gpt_tokenizer = tiktoken.get_encoding("cl100k_base")  # GPT-4 tokenizer
    gpt_tokens = gpt_tokenizer.encode(text)
    
    print(f"Text: {text[:100]}{'...' if len(text) > 100 else ''}")
    print(f"\nGPT-4 Tokenizer: {len(gpt_tokens)} tokens")
    
    # Show token details for analysis
    print("\nToken breakdown:")
    for token_id in gpt_tokens[:10]:  # Show first 10 tokens
        token_bytes = gpt_tokenizer.decode_single_token_bytes(token_id)
        try:
            token_str = token_bytes.decode('utf-8')
            print(f"  {token_id}: '{token_str}'")
        except:
            print(f"  {token_id}: [bytes: {token_bytes}]")
    
    if len(gpt_tokens) > 10:
        print(f"  ... and {len(gpt_tokens) - 10} more tokens")
    
    return len(gpt_tokens)

def character_based_analysis(text):
    """
    Analyze text characteristics for Turkish efficiency assessment.
    """
    turkish_chars = set('ğüşıöçĞÜŞİÖÇ')
    
    # Count Turkish-specific characters
    turkish_char_count = sum(1 for c in text if c in turkish_chars)
    total_chars = len(text)
    turkish_ratio = turkish_char_count / total_chars if total_chars > 0 else 0
    
    # Count words
    words = text.split()
    word_count = len(words)
    
    # Average word length
    avg_word_length = sum(len(word) for word in words) / word_count if word_count > 0 else 0
    
    print("\n" + "=" * 50)
    print("CHARACTER ANALYSIS")
    print("=" * 50)
    print(f"Total characters: {total_chars}")
    print(f"Turkish-specific characters: {turkish_char_count} ({turkish_ratio:.1%})")
    print(f"Word count: {word_count}")
    print(f"Average word length: {avg_word_length:.1f} characters")
    print(f"Tokens per word ratio: {len(gpt_tokens) / word_count:.2f}")
    
    # Efficiency assessment
    tokens_per_char = len(gpt_tokens) / total_chars
    print(f"\nTokens per character: {tokens_per_char:.3f}")
    
    if tokens_per_char > 0.5:
        print("⚠️ High token ratio - tokenizer may be inefficient for Turkish")
    elif tokens_per_char < 0.3:
        print("✅ Good token ratio - tokenizer is efficient")
    else:
        print("≈ Moderate token ratio")

def test_turkish_texts_simple():
    """Test multiple Turkish texts."""
    test_texts = [
        "Merhaba, Türkçe doğal dil işleme modelleri üzerinde çalışıyorum.",
        "İstanbul'un tarihi yarımadası, UNESCO Dünya Mirası Listesi'nde yer almaktadır.",
        "Yapay zeka ve makine öğrenmesi teknolojileri hızla gelişmektedir.",
        "Türkiye'nin coğrafi konumu, Avrupa ve Asya kıtaları arasında köprü görevi görmektedir.",
        "Günümüzde bilgisayarlı görü sistemleri birçok alanda kullanılmaktadır.",
        "Türkçe karakterler: ğ, ü, ş, ı, ö, ç içeren bir metin örneği."
    ]
    
    print("=" * 70)
    print("TURKISH TEXT TOKENIZATION ANALYSIS")
    print("=" * 70)
    
    all_tokens = []
    for i, text in enumerate(test_texts, 1):
        print(f"\n\nTest {i}:")
        print("-" * 50)
        token_count = simple_tokenizer_comparison(text)
        all_tokens.append(token_count)
        
        # Analyze the text
        global gpt_tokens
        gpt_tokenizer = tiktoken.get_encoding("cl100k_base")
        gpt_tokens = gpt_tokenizer.encode(text)
        character_based_analysis(text)
    
    # Summary
    print("\n\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    avg_tokens = sum(all_tokens) / len(all_tokens)
    print(f"Average token count: {avg_tokens:.1f}")
    print(f"Min tokens: {min(all_tokens)}")
    print(f"Max tokens: {max(all_tokens)}")

if __name__ == "__main__":
    # First, install tiktoken if not already installed
    try:
        import tiktoken
    except ImportError:
        print("Installing tiktoken...")
        os.system("py -m pip install tiktoken")
        import tiktoken
    
    # Run the analysis
    test_turkish_texts_simple()
    
    print("\n\n" + "=" * 70)
    print("CUSTOM TEXT TEST")
    print("=" * 70)
    print("\nYou can test your own Turkish text:")
    custom_text = input("Enter Turkish text (or press Enter to skip): ").strip()
    if custom_text:
        simple_tokenizer_comparison(custom_text)
        gpt_tokenizer = tiktoken.get_encoding("cl100k_base")
        gpt_tokens = gpt_tokenizer.encode(custom_text)
        character_based_analysis(custom_text)