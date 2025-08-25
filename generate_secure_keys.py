#!/usr/bin/env python
"""
Generate secure environment variables for production
"""
import secrets
import string
import os

def generate_secure_key(length=32):
    """Generate a cryptographically secure random key"""
    return secrets.token_hex(length)

def generate_password(length=16):
    """Generate a secure random password"""
    alphabet = string.ascii_letters + string.digits + string.punctuation
    return ''.join(secrets.choice(alphabet) for _ in range(length))

def main():
    print("=" * 60)
    print("TEKNOFEST 2025 - Secure Environment Variables Generator")
    print("=" * 60)
    print()
    
    # Generate secure keys
    secret_key = generate_secure_key(32)
    jwt_secret_key = generate_secure_key(32)
    
    print("🔐 Generated Secure Keys:")
    print("-" * 60)
    print(f"SECRET_KEY={secret_key}")
    print(f"JWT_SECRET_KEY={jwt_secret_key}")
    print()
    
    # Database password suggestion
    db_password = generate_password(16)
    print("🔑 Suggested Database Password:")
    print("-" * 60)
    print(f"DATABASE_PASSWORD={db_password}")
    print()
    
    # Redis password suggestion
    redis_password = generate_secure_key(16)
    print("🔑 Suggested Redis Password:")
    print("-" * 60)
    print(f"REDIS_PASSWORD={redis_password}")
    print()
    
    # API keys
    mcp_api_key = generate_secure_key(24)
    print("🔑 Suggested API Keys:")
    print("-" * 60)
    print(f"MCP_API_KEY={mcp_api_key}")
    print()
    
    # Update .env file option
    response = input("Do you want to update .env file with these keys? (y/n): ")
    
    if response.lower() == 'y':
        env_file = '.env'
        
        # Read current .env
        with open(env_file, 'r') as f:
            lines = f.readlines()
        
        # Update keys
        updated_lines = []
        for line in lines:
            if line.startswith('SECRET_KEY='):
                updated_lines.append(f'SECRET_KEY={secret_key}\n')
            elif line.startswith('JWT_SECRET_KEY='):
                updated_lines.append(f'JWT_SECRET_KEY={jwt_secret_key}\n')
            else:
                updated_lines.append(line)
        
        # Write back
        with open(env_file, 'w') as f:
            f.writelines(updated_lines)
        
        print("✅ .env file updated successfully!")
    else:
        print("ℹ️ Please update your .env file manually with the generated keys.")
    
    print()
    print("⚠️ IMPORTANT SECURITY NOTES:")
    print("-" * 60)
    print("1. Never commit .env file to version control")
    print("2. Use different keys for each environment (dev/staging/prod)")
    print("3. Rotate keys periodically (every 3-6 months)")
    print("4. Store production keys in secure vault (AWS Secrets, etc.)")
    print("5. Enable audit logging for key usage")
    print()
    print("=" * 60)
    print("✅ Security configuration complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()
