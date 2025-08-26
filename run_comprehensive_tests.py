#!/usr/bin/env python3
"""
Comprehensive Test Runner with Coverage Report
Target: Achieve 80%+ test coverage
"""

import os
import sys
import subprocess
import json
from pathlib import Path


def run_tests_with_coverage():
    """Run all tests with coverage measurement"""
    
    print("=" * 80)
    print("TEKNOFEST 2025 - Comprehensive Test Suite")
    print("Target Coverage: 80%+")
    print("=" * 80)
    
    # Test files to run
    test_files = [
        "tests/test_learning_path_agent_comprehensive.py",
        "tests/test_study_buddy_agent_comprehensive.py",
        "tests/test_api_endpoints_comprehensive.py",
        "tests/test_auth_comprehensive.py",
        "tests/test_database_operations_comprehensive.py",
        "tests/test_agents.py",
        "tests/test_api_endpoints.py",
        "tests/test_auth.py",
        "tests/test_database.py",
        "tests/test_data_processor.py",
        "tests/test_model_integration.py",
        "tests/test_core.py",
        "tests/test_error_handling.py",
        "tests/test_integration.py",
        "tests/test_security.py",
        "tests/test_turkish_nlp.py",
        "tests/test_gamification.py",
        "tests/test_irt_engine.py",
        "tests/test_offline_support.py",
        "tests/test_rate_limiting.py",
    ]
    
    # Run coverage erase first
    print("\n📊 Cleaning previous coverage data...")
    subprocess.run(["coverage", "erase"], capture_output=True)
    
    # Run tests with coverage
    print("\n🧪 Running comprehensive test suite...")
    
    test_results = {}
    failed_tests = []
    
    for test_file in test_files:
        if not Path(test_file).exists():
            print(f"⚠️  Skipping {test_file} (not found)")
            continue
            
        print(f"\n▶️  Running {test_file}...")
        
        result = subprocess.run(
            ["coverage", "run", "-a", "-m", "pytest", test_file, "-v", "--tb=short"],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            print(f"✅ {test_file} - PASSED")
            test_results[test_file] = "PASSED"
        else:
            print(f"❌ {test_file} - FAILED")
            test_results[test_file] = "FAILED"
            failed_tests.append(test_file)
            # Continue running other tests even if one fails
    
    # Generate coverage report
    print("\n📈 Generating coverage report...")
    
    # Text report
    print("\n" + "=" * 80)
    print("COVERAGE REPORT")
    print("=" * 80)
    
    result = subprocess.run(
        ["coverage", "report", "-m", "--skip-covered", "--skip-empty"],
        capture_output=True,
        text=True
    )
    
    print(result.stdout)
    
    # Parse coverage percentage from output
    coverage_percentage = 0
    for line in result.stdout.split('\n'):
        if line.startswith('TOTAL'):
            parts = line.split()
            if len(parts) >= 4:
                coverage_str = parts[-1].replace('%', '')
                try:
                    coverage_percentage = float(coverage_str)
                except ValueError:
                    pass
    
    # HTML report
    subprocess.run(["coverage", "html", "--skip-covered"], capture_output=True)
    print(f"\n📂 HTML coverage report generated in: htmlcov/index.html")
    
    # JSON report for detailed analysis
    subprocess.run(["coverage", "json"], capture_output=True)
    
    # Analyze coverage by module
    print("\n" + "=" * 80)
    print("MODULE COVERAGE ANALYSIS")
    print("=" * 80)
    
    with open("coverage.json", "r") as f:
        coverage_data = json.load(f)
    
    module_coverage = {}
    
    for file_path, data in coverage_data.get("files", {}).items():
        if "src/" in file_path and "__pycache__" not in file_path:
            module_name = file_path.replace("\\", "/").split("src/")[1]
            coverage_percent = data["summary"]["percent_covered"]
            module_coverage[module_name] = coverage_percent
    
    # Sort by coverage percentage
    sorted_modules = sorted(module_coverage.items(), key=lambda x: x[1])
    
    print("\n🔴 Modules with lowest coverage (need attention):")
    for module, coverage in sorted_modules[:10]:
        if coverage < 80:
            print(f"  {module:50} {coverage:6.2f}%")
    
    print("\n🟢 Modules with good coverage (80%+):")
    good_coverage_count = 0
    for module, coverage in sorted_modules:
        if coverage >= 80:
            good_coverage_count += 1
            if good_coverage_count <= 10:  # Show first 10
                print(f"  {module:50} {coverage:6.2f}%")
    
    if good_coverage_count > 10:
        print(f"  ... and {good_coverage_count - 10} more modules")
    
    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    
    total_tests = len(test_results)
    passed_tests = sum(1 for v in test_results.values() if v == "PASSED")
    
    print(f"\n📊 Test Results:")
    print(f"  Total test files run: {total_tests}")
    print(f"  Passed: {passed_tests}")
    print(f"  Failed: {len(failed_tests)}")
    
    if failed_tests:
        print(f"\n❌ Failed test files:")
        for test in failed_tests:
            print(f"    - {test}")
    
    print(f"\n📈 Coverage Results:")
    print(f"  Overall Coverage: {coverage_percentage:.2f}%")
    print(f"  Target Coverage: 80%")
    
    if coverage_percentage >= 80:
        print(f"\n🎉 SUCCESS! Target coverage of 80% achieved!")
    else:
        gap = 80 - coverage_percentage
        print(f"\n⚠️  Coverage gap: {gap:.2f}% below target")
        print(f"  Focus on modules with low coverage listed above")
    
    # Recommendations
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS TO REACH 80% COVERAGE")
    print("=" * 80)
    
    if coverage_percentage < 80:
        print("\n📝 Priority areas for additional testing:")
        
        critical_modules = [
            ("agents/", "AI agent functionality"),
            ("api/", "API endpoints and routes"),
            ("core/", "Core business logic"),
            ("database/", "Database operations"),
            ("turkish_nlp", "Turkish NLP components"),
        ]
        
        for module_prefix, description in critical_modules:
            low_coverage_in_module = [
                (m, c) for m, c in module_coverage.items() 
                if module_prefix in m and c < 80
            ]
            
            if low_coverage_in_module:
                print(f"\n  {description}:")
                for mod, cov in low_coverage_in_module[:3]:
                    print(f"    - {mod}: {cov:.2f}% (needs +{80-cov:.2f}%)")
    
    return coverage_percentage


def main():
    """Main function"""
    try:
        # Change to project root directory
        project_root = Path(__file__).parent
        os.chdir(project_root)
        
        # Add src to path
        sys.path.insert(0, str(project_root / "src"))
        
        # Run tests
        coverage = run_tests_with_coverage()
        
        # Exit with appropriate code
        if coverage >= 80:
            sys.exit(0)
        else:
            sys.exit(1)
            
    except Exception as e:
        print(f"\n❌ Error running tests: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()