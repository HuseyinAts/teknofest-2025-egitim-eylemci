"""
E2E test runner with reporting and parallel execution.
"""

import os
import sys
import pytest
import subprocess
from pathlib import Path
import json
from datetime import datetime


class E2ETestRunner:
    """E2E test runner with advanced features."""
    
    def __init__(self):
        self.test_dir = Path(__file__).parent
        self.report_dir = self.test_dir / 'reports'
        self.report_dir.mkdir(exist_ok=True)
    
    def install_dependencies(self):
        """Install E2E test dependencies."""
        print("Installing E2E test dependencies...")
        subprocess.run(
            [sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'],
            cwd=self.test_dir,
            check=True
        )
        
        # Install Playwright browsers
        print("Installing Playwright browsers...")
        subprocess.run(
            [sys.executable, '-m', 'playwright', 'install', 'chromium'],
            check=True
        )
    
    def start_services(self):
        """Start required services for E2E tests."""
        print("Starting services...")
        
        # Start Docker containers using docker-compose
        compose_file = self.test_dir.parent.parent / 'docker-compose.yml'
        if compose_file.exists():
            subprocess.run(
                ['docker-compose', '-f', str(compose_file), 'up', '-d'],
                check=True
            )
    
    def run_tests(self, parallel=True, verbose=True, markers=None):
        """Run E2E tests with specified options."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Build pytest command
        cmd = [sys.executable, '-m', 'pytest']
        
        # Add test directory
        cmd.append(str(self.test_dir))
        
        # Add parallel execution
        if parallel:
            cmd.extend(['-n', 'auto'])
        
        # Add verbosity
        if verbose:
            cmd.append('-v')
        
        # Add markers
        if markers:
            cmd.extend(['-m', markers])
        
        # Add reporting
        cmd.extend([
            '--html', f'{self.report_dir}/report_{timestamp}.html',
            '--self-contained-html',
            '--json-report',
            '--json-report-file', f'{self.report_dir}/report_{timestamp}.json',
            '--junit-xml', f'{self.report_dir}/junit_{timestamp}.xml'
        ])
        
        # Add coverage
        cmd.extend([
            '--cov=src',
            '--cov-report=html:htmlcov_e2e',
            '--cov-report=term'
        ])
        
        # Run tests
        print(f"Running E2E tests: {' '.join(cmd)}")
        result = subprocess.run(cmd)
        
        return result.returncode
    
    def generate_allure_report(self):
        """Generate Allure report for test results."""
        allure_dir = self.report_dir / 'allure'
        allure_dir.mkdir(exist_ok=True)
        
        # Run tests with Allure
        subprocess.run([
            sys.executable, '-m', 'pytest',
            str(self.test_dir),
            '--alluredir', str(allure_dir)
        ])
        
        # Generate Allure report
        subprocess.run([
            'allure', 'generate',
            str(allure_dir),
            '-o', str(self.report_dir / 'allure-report'),
            '--clean'
        ])
    
    def run_specific_test(self, test_name):
        """Run a specific test by name."""
        cmd = [
            sys.executable, '-m', 'pytest',
            '-v',
            '-k', test_name,
            str(self.test_dir)
        ]
        
        return subprocess.run(cmd).returncode
    
    def run_smoke_tests(self):
        """Run only smoke tests."""
        return self.run_tests(markers='smoke')
    
    def run_critical_tests(self):
        """Run only critical tests."""
        return self.run_tests(markers='critical')
    
    def cleanup(self):
        """Cleanup after tests."""
        print("Cleaning up...")
        
        # Stop Docker containers
        compose_file = self.test_dir.parent.parent / 'docker-compose.yml'
        if compose_file.exists():
            subprocess.run(
                ['docker-compose', '-f', str(compose_file), 'down'],
                check=True
            )
    
    def analyze_results(self):
        """Analyze test results and generate summary."""
        # Find latest JSON report
        json_reports = list(self.report_dir.glob('report_*.json'))
        if not json_reports:
            print("No test reports found")
            return
        
        latest_report = max(json_reports, key=lambda p: p.stat().st_mtime)
        
        with open(latest_report) as f:
            data = json.load(f)
        
        # Generate summary
        summary = {
            'total_tests': data['summary']['total'],
            'passed': data['summary'].get('passed', 0),
            'failed': data['summary'].get('failed', 0),
            'skipped': data['summary'].get('skipped', 0),
            'duration': data['duration'],
            'timestamp': datetime.now().isoformat()
        }
        
        # Save summary
        summary_file = self.report_dir / 'test_summary.json'
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Print summary
        print("\n" + "="*50)
        print("E2E TEST SUMMARY")
        print("="*50)
        print(f"Total Tests: {summary['total_tests']}")
        print(f"Passed: {summary['passed']}")
        print(f"Failed: {summary['failed']}")
        print(f"Skipped: {summary['skipped']}")
        print(f"Duration: {summary['duration']:.2f} seconds")
        print("="*50)
        
        return summary


def main():
    """Main entry point for E2E test runner."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run E2E tests')
    parser.add_argument('--install', action='store_true', help='Install dependencies')
    parser.add_argument('--parallel', action='store_true', help='Run tests in parallel')
    parser.add_argument('--smoke', action='store_true', help='Run smoke tests only')
    parser.add_argument('--critical', action='store_true', help='Run critical tests only')
    parser.add_argument('--test', help='Run specific test by name')
    parser.add_argument('--allure', action='store_true', help='Generate Allure report')
    parser.add_argument('--no-cleanup', action='store_true', help='Skip cleanup')
    
    args = parser.parse_args()
    
    runner = E2ETestRunner()
    
    try:
        # Install dependencies if requested
        if args.install:
            runner.install_dependencies()
        
        # Start services
        runner.start_services()
        
        # Run tests based on arguments
        if args.test:
            exit_code = runner.run_specific_test(args.test)
        elif args.smoke:
            exit_code = runner.run_smoke_tests()
        elif args.critical:
            exit_code = runner.run_critical_tests()
        elif args.allure:
            runner.generate_allure_report()
            exit_code = 0
        else:
            exit_code = runner.run_tests(parallel=args.parallel)
        
        # Analyze results
        runner.analyze_results()
        
    finally:
        # Cleanup unless skipped
        if not args.no_cleanup:
            runner.cleanup()
    
    sys.exit(exit_code)


if __name__ == '__main__':
    main()