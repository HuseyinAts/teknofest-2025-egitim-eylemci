#!/usr/bin/env python3
"""
Create Clean Architecture folder structure
"""

import os
from pathlib import Path

def create_clean_architecture():
    """Create all necessary folders for Clean Architecture"""
    
    base_path = Path("src")
    
    folders = [
        # Domain Layer
        "domain/entities",
        "domain/value_objects",
        "domain/services",
        "domain/interfaces",
        "domain/events",
        
        # Application Layer
        "application/services",
        "application/dtos",
        "application/use_cases",
        "application/interfaces",
        
        # Infrastructure Layer
        "infrastructure/persistence/models",
        "infrastructure/persistence/repositories",
        "infrastructure/external/services",
        "infrastructure/cache",
        "infrastructure/config",
        
        # Presentation Layer
        "presentation/api/v1/endpoints",
        "presentation/api/v1/requests",
        "presentation/api/v1/responses",
        "presentation/middleware",
        "presentation/validators",
        
        # Shared/Common
        "shared/constants",
        "shared/exceptions",
        "shared/utils",
        "shared/decorators"
    ]
    
    for folder in folders:
        folder_path = base_path / folder
        folder_path.mkdir(parents=True, exist_ok=True)
        
        # Create __init__.py
        init_file = folder_path / "__init__.py"
        if not init_file.exists():
            init_file.write_text('"""Package initialization"""')
        
        print(f"✅ Created: {folder_path}")
    
    print("\n🎉 Clean Architecture structure created successfully!")

if __name__ == "__main__":
    create_clean_architecture()
