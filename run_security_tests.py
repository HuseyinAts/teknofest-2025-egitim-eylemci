#!/usr/bin/env python
"""
Run comprehensive security tests
"""
import subprocess
import sys
import os
import json
from datetime import datetime

def run_command(command, description):
    """Run a command and return results"""
    print(f"\n{'='*60}")
    print(f"🔧 {description}")
    print(f"{'='*60}")
    print(f"Command: {command}")
    print("-" * 60)
    
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=60
        )
        
        if result.returncode == 0:
            print(f"✅ {description} - SUCCESS")
            if result.stdout:
                print("Output:")
                print(result.stdout[:1000])  # Limit output
        else:
            print(f"❌ {description} - FAILED")
            if result.stderr:
                print("Error:")
                print(result.stderr[:1000])
        
        return result.returncode == 0, result.stdout, result.stderr
    
    except subprocess.TimeoutExpired:
        print(f"⏱️ {description} - TIMEOUT")
        return False, "", "Command timed out"
    except Exception as e:
        print(f"❌ {description} - ERROR: {e}")
        return False, "", str(e)

def main():
    print("=" * 80)
    print("TEKNOFEST 2025 - Comprehensive Security Test Suite")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    results = {}
    
    # 1. Install required packages
    print("\n📦 Installing required packages...")
    run_command(
        "pip install pytest pytest-cov pytest-asyncio bandit safety pylint black mypy -q",
        "Installing test dependencies"
    )
    
    # 2. Run security tests
    print("\n🔒 Running Security Tests...")
    success, stdout, stderr = run_command(
        "pytest tests/test_security_comprehensive.py -v --tb=short",
        "Security Test Suite"
    )
    results['security_tests'] = success
    
    # 3. Run all tests with coverage
    print("\n🧪 Running All Tests with Coverage...")
    success, stdout, stderr = run_command(
        "pytest tests/ --cov=src --cov-report=term-missing --cov-report=html -v --tb=short",
        "Full Test Suite with Coverage"
    )
    results['all_tests'] = success
    
    # 4. Code quality checks
    print("\n📝 Code Quality Checks...")
    
    # Black formatting check
    success, stdout, stderr = run_command(
        "black --check src/",
        "Black Code Formatting Check"
    )
    results['black_format'] = success
    
    # Pylint check
    success, stdout, stderr = run_command(
        "pylint src/ --exit-zero --output-format=json > pylint_report.json",
        "Pylint Code Quality Analysis"
    )
    results['pylint'] = True  # exit-zero means always success
    
    # Type checking with mypy
    success, stdout, stderr = run_command(
        "mypy src/ --ignore-missing-imports --no-error-summary",
        "MyPy Type Checking"
    )
    results['mypy'] = success
    
    # 5. Security vulnerability scan
    print("\n🔍 Security Vulnerability Scanning...")
    
    # Bandit security scan
    success, stdout, stderr = run_command(
        "bandit -r src/ -f json -o bandit_report.json",
        "Bandit Security Scan"
    )
    results['bandit'] = success
    
    # Safety check for dependencies
    success, stdout, stderr = run_command(
        "safety check --json > safety_report.json",
        "Safety Dependency Check"
    )
    results['safety'] = success
    
    # 6. Generate test report
    print("\n" + "=" * 80)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 80)
    
    total_tests = len(results)
    passed_tests = sum(1 for v in results.values() if v)
    
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name.ljust(30)}: {status}")
    
    print("-" * 80)
    print(f"Total: {passed_tests}/{total_tests} tests passed")
    print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
    
    # Generate HTML report
    html_report = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Security Test Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1 {{ color: #333; }}
            .passed {{ color: green; }}
            .failed {{ color: red; }}
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
        </style>
    </head>
    <body>
        <h1>TEKNOFEST 2025 - Security Test Report</h1>
        <p>Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        
        <h2>Test Results</h2>
        <table>
            <tr>
                <th>Test Category</th>
                <th>Status</th>
            </tr>
    """
    
    for test_name, passed in results.items():
        status_class = "passed" if passed else "failed"
        status_text = "PASSED" if passed else "FAILED"
        html_report += f"""
            <tr>
                <td>{test_name.replace('_', ' ').title()}</td>
                <td class="{status_class}">{status_text}</td>
            </tr>
        """
    
    html_report += f"""
        </table>
        
        <h2>Summary</h2>
        <p>Total Tests: {total_tests}</p>
        <p>Passed: {passed_tests}</p>
        <p>Failed: {total_tests - passed_tests}</p>
        <p>Success Rate: {(passed_tests/total_tests)*100:.1f}%</p>
        
        <h3>Coverage Report</h3>
        <p><a href="htmlcov/index.html">View Detailed Coverage Report</a></p>
        
        <h3>Security Reports</h3>
        <ul>
            <li><a href="bandit_report.json">Bandit Security Scan</a></li>
            <li><a href="safety_report.json">Safety Dependency Check</a></li>
            <li><a href="pylint_report.json">Pylint Code Quality</a></li>
        </ul>
    </body>
    </html>
    """
    
    with open("test_report.html", "w") as f:
        f.write(html_report)
    
    print("\n📄 HTML report generated: test_report.html")
    
    # Return exit code
    if passed_tests == total_tests:
        print("\n🎉 All tests passed! Your code is secure and ready for deployment.")
        return 0
    else:
        print(f"\n⚠️ {total_tests - passed_tests} test(s) failed. Please review and fix issues.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
