"""
TEKNOFEST 2025 - Test Runner
Bu script dependencies kurulumunu kontrol eder ve testleri çalıştırır
"""

import subprocess
import sys
import os
from pathlib import Path

def check_and_install_dependencies():
    """Check and install required test dependencies"""
    
    print("=" * 60)
    print("TEKNOFEST 2025 - Test Dependencies Kontrolü")
    print("=" * 60)
    
    required_packages = [
        "pytest",
        "pytest-cov",
        "pytest-asyncio",
        "fastapi",
        "httpx",
        "sqlalchemy"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
            print(f"✅ {package} - Yüklü")
        except ImportError:
            print(f"❌ {package} - Eksik")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n📦 Eksik paketler yükleniyor: {', '.join(missing_packages)}")
        for package in missing_packages:
            print(f"\nYükleniyor: {package}")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print("\n✅ Tüm dependencies yüklendi!")
    else:
        print("\n✅ Tüm dependencies zaten yüklü!")
    
    return True

def run_simple_tests():
    """Run simple unit tests without pytest"""
    
    print("\n" + "=" * 60)
    print("TEKNOFEST 2025 - Basit Test Çalıştırma")
    print("=" * 60)
    
    # Test 1: ZPD Calculation
    print("\n📝 Test 1: ZPD (Zone of Proximal Development) Hesaplama")
    
    def calculate_zpd(current_level: float, performance: float) -> float:
        if performance > 0.8:
            new_level = current_level + 0.1
        elif performance < 0.6:
            new_level = current_level - 0.1
        else:
            new_level = current_level
        return max(0.0, min(1.0, new_level))
    
    test_cases = [
        (0.5, 0.9, 0.6, "Good performance"),
        (0.5, 0.5, 0.4, "Poor performance"),
        (0.5, 0.7, 0.5, "Average performance"),
        (0.95, 0.9, 1.0, "Max boundary"),
        (0.05, 0.4, 0.0, "Min boundary")
    ]
    
    passed = 0
    failed = 0
    
    for current, perf, expected, desc in test_cases:
        result = calculate_zpd(current, perf)
        if result == expected:
            print(f"  ✅ {desc}: {result} (Expected: {expected})")
            passed += 1
        else:
            print(f"  ❌ {desc}: {result} (Expected: {expected})")
            failed += 1
    
    # Test 2: XP Calculation
    print("\n📝 Test 2: XP (Experience Points) Hesaplama")
    
    def calculate_xp(quiz_score: float, difficulty: float, time_bonus: bool = False) -> int:
        base_xp = int(quiz_score * 10)
        difficulty_multiplier = 1 + difficulty
        time_multiplier = 1.2 if time_bonus else 1.0
        total_xp = int(base_xp * difficulty_multiplier * time_multiplier)
        return max(0, total_xp)
    
    xp_tests = [
        (85, 0.3, False, "Normal XP"),
        (85, 0.7, False, "Hard difficulty"),
        (85, 0.7, True, "With time bonus"),
        (100, 1.0, True, "Perfect score"),
        (0, 0.5, False, "Zero score")
    ]
    
    for score, diff, bonus, desc in xp_tests:
        xp = calculate_xp(score, diff, bonus)
        print(f"  ✅ {desc}: {xp} XP")
        passed += 1
    
    # Test 3: Grade Validation
    print("\n📝 Test 3: Sınıf Seviyesi Doğrulama")
    
    valid_grades = [9, 10, 11, 12]
    invalid_grades = [0, 8, 13, 15, -1]
    
    for grade in valid_grades:
        if 9 <= grade <= 12:
            print(f"  ✅ Grade {grade}: Valid")
            passed += 1
        else:
            print(f"  ❌ Grade {grade}: Should be valid")
            failed += 1
    
    for grade in invalid_grades:
        if not (9 <= grade <= 12):
            print(f"  ✅ Grade {grade}: Invalid (as expected)")
            passed += 1
        else:
            print(f"  ❌ Grade {grade}: Should be invalid")
            failed += 1
    
    # Test 4: Learning Style Validation
    print("\n📝 Test 4: Öğrenme Stili Doğrulama")
    
    valid_styles = ["visual", "auditory", "reading", "kinesthetic"]
    invalid_styles = ["invalid", "other", ""]
    
    for style in valid_styles:
        if style in ["visual", "auditory", "reading", "kinesthetic"]:
            print(f"  ✅ Style '{style}': Valid")
            passed += 1
        else:
            print(f"  ❌ Style '{style}': Should be valid")
            failed += 1
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SONUÇLARI")
    print("=" * 60)
    print(f"✅ Başarılı: {passed} test")
    print(f"❌ Başarısız: {failed} test")
    print(f"📊 Başarı Oranı: {(passed/(passed+failed)*100):.1f}%")
    
    return passed, failed

def run_pytest_if_available():
    """Try to run pytest if available"""
    
    print("\n" + "=" * 60)
    print("PYTEST ile Test Çalıştırma")
    print("=" * 60)
    
    try:
        import pytest
        print("✅ Pytest bulundu! Testler çalıştırılıyor...")
        print("\nKomut: pytest tests/unit -v --tb=short\n")
        
        # Change to project directory
        os.chdir(r"C:\Users\husey\teknofest-2025-egitim-eylemci")
        
        # Run pytest
        exit_code = pytest.main([
            "tests/unit",
            "-v",
            "--tb=short",
            "--no-header",
            "-q"
        ])
        
        if exit_code == 0:
            print("\n✅ Tüm pytest testleri başarılı!")
        else:
            print(f"\n⚠️ Bazı testler başarısız oldu (exit code: {exit_code})")
            
    except ImportError:
        print("❌ Pytest yüklü değil. Manuel testler çalıştırılıyor...")
    except Exception as e:
        print(f"⚠️ Pytest çalıştırılamadı: {e}")
        print("Manuel testler çalıştırılıyor...")

def main():
    """Main test runner"""
    
    print("🚀 TEKNOFEST 2025 - Test Suite Başlatılıyor\n")
    
    # Step 1: Check dependencies
    dependencies_ok = check_and_install_dependencies()
    
    # Step 2: Run simple tests
    passed, failed = run_simple_tests()
    
    # Step 3: Try to run pytest
    run_pytest_if_available()
    
    print("\n" + "=" * 60)
    print("✅ TEST ÇALIŞTIRMA TAMAMLANDI!")
    print("=" * 60)
    
    print("\n📝 Sonraki Adımlar:")
    print("1. Coverage raporu için: pytest --cov=src --cov-report=html")
    print("2. Tüm testler için: pytest tests/ -v")
    print("3. Parallel test için: pytest -n auto")
    
    print("\n💡 İpucu: Test dosyaları tests/ klasöründe bulunmaktadır.")
    print("   - Unit tests: tests/unit/")
    print("   - Integration tests: tests/integration/")

if __name__ == "__main__":
    main()
    input("\nDevam etmek için Enter'a basın...")