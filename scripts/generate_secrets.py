#!/usr/bin/env python3
"""
Generate secure secrets for production environment
"""

import secrets
import string
import sys
from pathlib import Path
from typing import Dict, Any
import argparse


def generate_secret_key(length: int = 64) -> str:
    """Generate a secure random secret key."""
    return secrets.token_hex(length // 2)


def generate_password(length: int = 32, include_special: bool = True) -> str:
    """Generate a secure random password."""
    alphabet = string.ascii_letters + string.digits
    if include_special:
        alphabet += "!@#$%^&*()_+-="
    return ''.join(secrets.choice(alphabet) for _ in range(length))


def generate_api_key(prefix: str = "tk", length: int = 32) -> str:
    """Generate an API key with prefix."""
    return f"{prefix}_{secrets.token_urlsafe(length)}"


def generate_database_password() -> str:
    """Generate a secure database password without problematic characters."""
    # Avoid characters that might cause issues in connection strings
    safe_chars = string.ascii_letters + string.digits + "_"
    return ''.join(secrets.choice(safe_chars) for _ in range(24))


def generate_all_secrets() -> Dict[str, Any]:
    """Generate all required secrets for the application."""
    return {
        # Core secrets
        "SECRET_KEY": generate_secret_key(64),
        "JWT_SECRET_KEY": generate_secret_key(64),
        
        # Database
        "DATABASE_PASSWORD": generate_database_password(),
        
        # Redis
        "REDIS_PASSWORD": generate_password(24, include_special=False),
        
        # API Keys
        "MCP_API_KEY": generate_api_key("mcp", 32),
        "INTERNAL_API_KEY": generate_api_key("internal", 32),
        
        # Email
        "SMTP_PASSWORD": generate_password(16),
        
        # AWS (if needed)
        "AWS_SECRET_ACCESS_KEY": generate_api_key("aws", 40),
        
        # Docker Registry
        "DOCKER_PASSWORD": generate_password(20),
        
        # Monitoring
        "GRAFANA_ADMIN_PASSWORD": generate_password(16),
        "PROMETHEUS_BASIC_AUTH_PASSWORD": generate_password(16),
    }


def create_env_file(secrets: Dict[str, Any], output_path: Path, template_path: Path = None):
    """Create .env file with generated secrets."""
    
    if template_path and template_path.exists():
        # Read template
        with open(template_path, 'r') as f:
            content = f.read()
        
        # Replace placeholders with actual secrets
        replacements = {
            "CHANGE_THIS_TO_64_CHAR_RANDOM_STRING": secrets["SECRET_KEY"],
            "CHANGE_THIS_TO_DIFFERENT_64_CHAR_RANDOM_STRING": secrets["JWT_SECRET_KEY"],
            "password@localhost": f"{secrets['DATABASE_PASSWORD']}@localhost",
            "CHANGE_THIS_TO_REDIS_PASSWORD": secrets["REDIS_PASSWORD"],
            "CHANGE_THIS_TO_MCP_API_KEY": secrets["MCP_API_KEY"],
            "YOUR_APP_PASSWORD": secrets["SMTP_PASSWORD"],
            "CHANGE_THIS_TO_AWS_SECRET": secrets["AWS_SECRET_ACCESS_KEY"],
            "CHANGE_THIS_TO_DOCKER_PASSWORD": secrets["DOCKER_PASSWORD"],
        }
        
        for old, new in replacements.items():
            content = content.replace(old, new)
        
        # Write updated content
        with open(output_path, 'w') as f:
            f.write(content)
    else:
        # Create minimal .env file
        with open(output_path, 'w') as f:
            f.write("# Auto-generated secure secrets\n")
            f.write("# Generated using generate_secrets.py\n\n")
            for key, value in secrets.items():
                f.write(f"{key}={value}\n")


def validate_existing_env(env_path: Path) -> list:
    """Validate existing .env file for security issues."""
    issues = []
    
    if not env_path.exists():
        return ["No .env file found"]
    
    with open(env_path, 'r') as f:
        lines = f.readlines()
    
    for line in lines:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        
        if '=' in line:
            key, value = line.split('=', 1)
            key = key.strip()
            value = value.strip()
            
            # Check for default/weak values
            weak_patterns = [
                "CHANGE_THIS", "YOUR_", "password", "secret", "123456",
                "admin", "default", "test", "demo", "example"
            ]
            
            for pattern in weak_patterns:
                if pattern.lower() in value.lower():
                    issues.append(f"{key} contains weak/default value: {pattern}")
            
            # Check key lengths
            if key in ["SECRET_KEY", "JWT_SECRET_KEY"] and len(value) < 32:
                issues.append(f"{key} is too short (min 32 chars)")
            
            # Check for exposed tokens
            if "TOKEN" in key and value and not value.startswith("hf_"):
                if len(value) < 20:
                    issues.append(f"{key} appears to be a weak token")
    
    return issues


def main():
    parser = argparse.ArgumentParser(description="Generate secure secrets for production")
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path(".env.production"),
        help="Output file path (default: .env.production)"
    )
    parser.add_argument(
        "--template", "-t",
        type=Path,
        default=Path(".env.example"),
        help="Template file path (default: .env.example)"
    )
    parser.add_argument(
        "--validate", "-v",
        type=Path,
        help="Validate existing .env file"
    )
    parser.add_argument(
        "--show", "-s",
        action="store_true",
        help="Show generated secrets (careful!)"
    )
    
    args = parser.parse_args()
    
    if args.validate:
        print(f"[VALIDATE] Validating {args.validate}...")
        issues = validate_existing_env(args.validate)
        if issues:
            print("[WARNING] Security issues found:")
            for issue in issues:
                print(f"   - {issue}")
            sys.exit(1)
        else:
            print("[SUCCESS] No security issues found!")
        return
    
    print("[GENERATE] Generating secure secrets...")
    secrets = generate_all_secrets()
    
    if args.show:
        print("\n[WARNING] Generated secrets (DO NOT SHARE):")
        for key, value in secrets.items():
            print(f"   {key}: {value}")
    
    # Create output directory if needed
    args.output.parent.mkdir(parents=True, exist_ok=True)
    
    # Create .env file
    create_env_file(secrets, args.output, args.template)
    
    print(f"\n[SUCCESS] Secrets generated and saved to {args.output}")
    print("\n[IMPORTANT] Important reminders:")
    print("   1. Never commit .env files with real secrets to git")
    print("   2. Add .env* to .gitignore")
    print("   3. Use different secrets for each environment")
    print("   4. Store production secrets in a secure vault")
    print("   5. Rotate secrets regularly")
    
    # Check if .gitignore exists and has .env
    gitignore_path = Path(".gitignore")
    if gitignore_path.exists():
        with open(gitignore_path, 'r') as f:
            gitignore_content = f.read()
        if ".env" not in gitignore_content:
            print("\n[WARNING] .env is not in .gitignore!")
            print("   Add these lines to .gitignore:")
            print("   .env")
            print("   .env.*")
            print("   !.env.example")


if __name__ == "__main__":
    main()