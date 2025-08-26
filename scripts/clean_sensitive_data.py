#!/usr/bin/env python3
"""
Scan and clean sensitive data from repository
"""

import re
import os
import sys
from pathlib import Path
from typing import List, Dict, Tuple
import json
import argparse


# Patterns to detect sensitive data
SENSITIVE_PATTERNS = {
    "api_key": r'["\']?api[_-]?key["\']?\s*[:=]\s*["\']?([a-zA-Z0-9_\-]{20,})["\']?',
    "secret_key": r'["\']?secret[_-]?key["\']?\s*[:=]\s*["\']?([a-zA-Z0-9_\-]{20,})["\']?',
    "password": r'["\']?password["\']?\s*[:=]\s*["\']?([^"\'\s]{8,})["\']?',
    "token": r'["\']?token["\']?\s*[:=]\s*["\']?([a-zA-Z0-9_\-]{20,})["\']?',
    "aws_key": r'AKIA[A-Z0-9]{16}',
    "jwt": r'eyJ[A-Za-z0-9_\-]+\.eyJ[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+',
    "github_token": r'ghp_[a-zA-Z0-9]{36}',
    "huggingface_token": r'hf_[a-zA-Z0-9]{34}',
    "database_url": r'(postgresql|mysql|mongodb)://[^:]+:[^@]+@[^/]+/[^\s]+',
    "redis_url": r'redis://[^:]*:[^@]+@[^/]+/\d+',
    "smtp_password": r'smtp[_-]?password["\']?\s*[:=]\s*["\']?([^"\'\s]+)["\']?',
    "private_key": r'-----BEGIN (RSA |EC )?PRIVATE KEY-----',
    "ssh_key": r'ssh-rsa\s+[A-Za-z0-9+/]+',
}

# Files to skip
SKIP_FILES = {
    ".env", ".env.example", ".env.template", 
    "generate_secrets.py", "clean_sensitive_data.py",
    ".git", "__pycache__", "node_modules", ".venv", "venv",
    "*.pyc", "*.pyo", "*.pyd", "*.so", "*.dll", "*.dylib",
    "*.jpg", "*.jpeg", "*.png", "*.gif", "*.ico", "*.svg",
    "*.pdf", "*.doc", "*.docx", "*.xls", "*.xlsx",
    "*.zip", "*.tar", "*.gz", "*.rar", "*.7z",
    "*.bin", "*.exe", "*.dmg", "*.iso",
    "*.db", "*.sqlite", "*.sqlite3"
}

# Safe placeholder values
SAFE_PLACEHOLDERS = {
    "api_key": "YOUR_API_KEY_HERE",
    "secret_key": "CHANGE_THIS_SECRET_KEY",
    "password": "CHANGE_THIS_PASSWORD",
    "token": "YOUR_TOKEN_HERE",
    "aws_key": "YOUR_AWS_ACCESS_KEY",
    "jwt": "YOUR_JWT_TOKEN",
    "github_token": "ghp_YOUR_GITHUB_TOKEN",
    "huggingface_token": "hf_YOUR_HUGGINGFACE_TOKEN",
    "database_url": "postgresql://username:password@localhost:5432/dbname",
    "redis_url": "redis://localhost:6379/0",
    "smtp_password": "YOUR_SMTP_PASSWORD",
    "private_key": "YOUR_PRIVATE_KEY",
    "ssh_key": "YOUR_SSH_KEY",
}


def should_skip_file(file_path: Path) -> bool:
    """Check if file should be skipped."""
    # Skip hidden directories and common excluded paths
    for part in file_path.parts:
        if part.startswith('.') and part != '.':
            return True
        if part in SKIP_FILES:
            return True
    
    # Skip binary and media files
    if file_path.suffix.lower() in {
        '.pyc', '.pyo', '.pyd', '.so', '.dll', '.dylib',
        '.jpg', '.jpeg', '.png', '.gif', '.ico', '.svg',
        '.pdf', '.doc', '.docx', '.xls', '.xlsx',
        '.zip', '.tar', '.gz', '.rar', '.7z',
        '.bin', '.exe', '.dmg', '.iso',
        '.db', '.sqlite', '.sqlite3'
    }:
        return True
    
    # Skip specific files
    if file_path.name in SKIP_FILES:
        return True
    
    return False


def scan_file(file_path: Path) -> List[Tuple[str, int, str, str]]:
    """Scan a file for sensitive data."""
    findings = []
    
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
    except Exception as e:
        return findings
    
    for line_num, line in enumerate(lines, 1):
        for pattern_name, pattern in SENSITIVE_PATTERNS.items():
            matches = re.finditer(pattern, line, re.IGNORECASE)
            for match in matches:
                # Skip if it's already a placeholder
                if any(placeholder in match.group(0) for placeholder in [
                    "YOUR_", "CHANGE_THIS", "example", "placeholder", 
                    "xxx", "***", "...", "<", ">"
                ]):
                    continue
                
                findings.append((
                    pattern_name,
                    line_num,
                    match.group(0),
                    line.strip()
                ))
    
    return findings


def clean_file(file_path: Path, findings: List[Tuple[str, int, str, str]], dry_run: bool = True) -> bool:
    """Clean sensitive data from file."""
    if not findings:
        return False
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return False
    
    modified = False
    for pattern_name, line_num, match, _ in findings:
        if line_num <= len(lines):
            original_line = lines[line_num - 1]
            placeholder = SAFE_PLACEHOLDERS.get(pattern_name, "REDACTED")
            
            # Replace the sensitive data with placeholder
            new_line = original_line.replace(match, placeholder)
            
            if new_line != original_line:
                lines[line_num - 1] = new_line
                modified = True
    
    if modified and not dry_run:
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.writelines(lines)
            return True
        except Exception as e:
            print(f"Error writing {file_path}: {e}")
            return False
    
    return modified


def scan_repository(root_path: Path) -> Dict[Path, List[Tuple[str, int, str, str]]]:
    """Scan entire repository for sensitive data."""
    all_findings = {}
    
    # Define file extensions to scan
    scan_extensions = {
        '.py', '.js', '.ts', '.jsx', '.tsx', '.json', '.yaml', '.yml',
        '.env', '.conf', '.config', '.ini', '.toml', '.properties',
        '.sh', '.bash', '.zsh', '.fish', '.ps1', '.bat', '.cmd',
        '.md', '.txt', '.rst', '.xml', '.html', '.css'
    }
    
    for file_path in root_path.rglob('*'):
        if file_path.is_file() and not should_skip_file(file_path):
            # Check if we should scan this file type
            if file_path.suffix.lower() in scan_extensions or file_path.name in {
                'Dockerfile', 'docker-compose.yml', 'Makefile', 'Vagrantfile'
            }:
                findings = scan_file(file_path)
                if findings:
                    all_findings[file_path] = findings
    
    return all_findings


def create_report(findings: Dict[Path, List[Tuple[str, int, str, str]]], output_file: Path = None):
    """Create a report of findings."""
    report = {
        "total_files": len(findings),
        "total_issues": sum(len(f) for f in findings.values()),
        "findings": {}
    }
    
    for file_path, file_findings in findings.items():
        report["findings"][str(file_path)] = [
            {
                "type": pattern_name,
                "line": line_num,
                "preview": line_preview[:100] + "..." if len(line_preview) > 100 else line_preview
            }
            for pattern_name, line_num, _, line_preview in file_findings
        ]
    
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
    
    return report


def main():
    parser = argparse.ArgumentParser(description="Scan and clean sensitive data from repository")
    parser.add_argument(
        "--path", "-p",
        type=Path,
        default=Path.cwd(),
        help="Repository path to scan (default: current directory)"
    )
    parser.add_argument(
        "--clean", "-c",
        action="store_true",
        help="Clean found sensitive data (replace with placeholders)"
    )
    parser.add_argument(
        "--dry-run", "-d",
        action="store_true",
        help="Dry run - show what would be changed without modifying files"
    )
    parser.add_argument(
        "--report", "-r",
        type=Path,
        help="Save report to JSON file"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Quiet mode - only show summary"
    )
    
    args = parser.parse_args()
    
    print(f"Scanning repository: {args.path}")
    findings = scan_repository(args.path)
    
    if not findings:
        print("No sensitive data found!")
        return
    
    # Display findings
    total_issues = sum(len(f) for f in findings.values())
    print(f"\nWARNING: Found {total_issues} potential sensitive data issues in {len(findings)} files:")
    
    if not args.quiet:
        for file_path, file_findings in findings.items():
            print(f"\nFile: {file_path}:")
            for pattern_name, line_num, match, line_preview in file_findings[:5]:  # Show first 5
                print(f"   Line {line_num}: {pattern_name}")
                print(f"      {line_preview[:80]}...")
            if len(file_findings) > 5:
                print(f"   ... and {len(file_findings) - 5} more issues")
    
    # Create report if requested
    if args.report:
        report = create_report(findings, args.report)
        print(f"\nReport saved to: {args.report}")
    
    # Clean if requested
    if args.clean or args.dry_run:
        if args.dry_run:
            print("\nDRY RUN - No files will be modified")
        else:
            print("\nCleaning sensitive data...")
        
        cleaned_count = 0
        for file_path, file_findings in findings.items():
            if clean_file(file_path, file_findings, dry_run=args.dry_run):
                cleaned_count += 1
                if not args.quiet:
                    print(f"   {'Would clean' if args.dry_run else 'Cleaned'}: {file_path}")
        
        print(f"\n{'Would clean' if args.dry_run else 'Cleaned'} {cleaned_count} files")
    
    # Git recommendations
    print("\nRecommendations:")
    print("   1. Review all findings before cleaning")
    print("   2. Commit current changes before cleaning")
    print("   3. After cleaning, update .env.example with proper placeholders")
    print("   4. Consider using git-secrets or similar tools for pre-commit hooks")
    print("   5. If secrets were already pushed, rotate them immediately")
    
    # Check if secrets might be in git history
    git_dir = args.path / ".git"
    if git_dir.exists():
        print("\nWARNING: If any secrets were previously committed:")
        print("   - They may still exist in git history")
        print("   - Consider using BFG Repo-Cleaner or git filter-branch")
        print("   - Rotate all affected credentials immediately")


if __name__ == "__main__":
    main()