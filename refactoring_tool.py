#!/usr/bin/env python3
"""
Clean Code Refactoring Tool
TEKNOFEST 2025 - Automated refactoring assistant
"""

import os
import ast
import json
import argparse
from pathlib import Path
from typing import List, Dict, Tuple
from collections import defaultdict
import re


class CodeAnalyzer:
    """Analyze Python code for Clean Code violations"""
    
    def __init__(self, project_path: str):
        self.project_path = Path(project_path)
        self.issues = []
        self.metrics = defaultdict(int)
    
    def analyze_file(self, file_path: Path) -> Dict:
        """Analyze a single Python file"""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        try:
            tree = ast.parse(content)
            
            issues = {
                'file': str(file_path.relative_to(self.project_path)),
                'violations': []
            }
            
            # Check for long methods
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    method_length = len(content.split('\n')[node.lineno-1:node.end_lineno])
                    if method_length > 20:
                        issues['violations'].append({
                            'type': 'LONG_METHOD',
                            'line': node.lineno,
                            'name': node.name,
                            'length': method_length,
                            'suggestion': f"Method '{node.name}' is {method_length} lines long. Consider splitting it into smaller methods."
                        })
                    
                    # Check for too many parameters
                    if len(node.args.args) > 4:
                        issues['violations'].append({
                            'type': 'TOO_MANY_PARAMETERS',
                            'line': node.lineno,
                            'name': node.name,
                            'count': len(node.args.args),
                            'suggestion': f"Method '{node.name}' has {len(node.args.args)} parameters. Consider using a parameter object."
                        })
            
            # Check for magic numbers
            magic_numbers = self._find_magic_numbers(content)
            for line_no, number in magic_numbers:
                issues['violations'].append({
                    'type': 'MAGIC_NUMBER',
                    'line': line_no,
                    'value': number,
                    'suggestion': f"Magic number '{number}' found. Define it as a named constant."
                })
            
            # Check for duplicate code patterns
            duplicates = self._find_duplicates(content)
            for dup in duplicates:
                issues['violations'].append({
                    'type': 'DUPLICATE_CODE',
                    'lines': dup['lines'],
                    'suggestion': "Duplicate code pattern detected. Consider extracting to a method."
                })
            
            return issues
            
        except SyntaxError as e:
            return {
                'file': str(file_path.relative_to(self.project_path)),
                'error': f"Syntax error: {e}"
            }
    
    def _find_magic_numbers(self, content: str) -> List[Tuple[int, str]]:
        """Find magic numbers in code"""
        magic_numbers = []
        lines = content.split('\n')
        
        # Patterns to find magic numbers (excluding 0, 1, -1)
        pattern = r'\b(?<!\.)\d+(?!\.\d)(?!\w)'
        
        for i, line in enumerate(lines, 1):
            # Skip comments and strings
            if '#' in line:
                line = line[:line.index('#')]
            
            matches = re.findall(pattern, line)
            for match in matches:
                num = int(match)
                if num not in [0, 1, -1, 100, 200, 404, 500]:  # Common acceptable numbers
                    if num > 1:
                        magic_numbers.append((i, match))
        
        return magic_numbers
    
    def _find_duplicates(self, content: str) -> List[Dict]:
        """Find duplicate code patterns"""
        duplicates = []
        lines = content.split('\n')
        
        # Simple duplicate detection (can be enhanced)
        line_groups = defaultdict(list)
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            if len(stripped) > 20 and not stripped.startswith('#'):
                line_groups[stripped].append(i)
        
        for line_content, occurrences in line_groups.items():
            if len(occurrences) > 2:
                duplicates.append({
                    'lines': occurrences,
                    'content': line_content[:50] + '...' if len(line_content) > 50 else line_content
                })
        
        return duplicates
    
    def analyze_project(self) -> Dict:
        """Analyze entire project"""
        all_issues = []
        
        for py_file in self.project_path.rglob('*.py'):
            if 'venv' not in str(py_file) and '__pycache__' not in str(py_file):
                issues = self.analyze_file(py_file)
                if issues.get('violations'):
                    all_issues.append(issues)
        
        return {
            'total_files': len(all_issues),
            'total_violations': sum(len(f['violations']) for f in all_issues),
            'files': all_issues
        }


class RefactoringGenerator:
    """Generate refactoring suggestions and templates"""
    
    @staticmethod
    def generate_constants_file(issues: Dict) -> str:
        """Generate constants file from magic numbers"""
        constants = set()
        
        for file_data in issues.get('files', []):
            for violation in file_data['violations']:
                if violation['type'] == 'MAGIC_NUMBER':
                    constants.add(violation['value'])
        
        content = '''"""
Application Constants
Auto-generated from magic numbers found in code
"""

class AppConstants:
    """Application-wide constants"""
'''
        for const in sorted(constants):
            const_name = f"DEFAULT_VALUE_{const}"
            content += f"    {const_name} = {const}\n"
        
        return content
    
    @staticmethod
    def generate_repository_interface() -> str:
        """Generate repository interface template"""
        return '''"""
Repository Interface Template
"""

from abc import ABC, abstractmethod
from typing import Optional, List, Generic, TypeVar

T = TypeVar('T')


class IRepository(ABC, Generic[T]):
    """Base repository interface"""
    
    @abstractmethod
    async def get_by_id(self, entity_id: str) -> Optional[T]:
        """Get entity by ID"""
        pass
    
    @abstractmethod
    async def get_all(self, limit: int = 100, offset: int = 0) -> List[T]:
        """Get all entities with pagination"""
        pass
    
    @abstractmethod
    async def add(self, entity: T) -> T:
        """Add new entity"""
        pass
    
    @abstractmethod
    async def update(self, entity: T) -> T:
        """Update existing entity"""
        pass
    
    @abstractmethod
    async def delete(self, entity_id: str) -> bool:
        """Delete entity by ID"""
        pass
'''
    
    @staticmethod
    def generate_service_template(service_name: str) -> str:
        """Generate service class template"""
        return f'''"""
{service_name} Service
Clean Architecture Service Layer
"""

from typing import List, Optional
from dataclasses import dataclass


@dataclass
class {service_name}Request:
    """Request DTO for {service_name}"""
    # Add fields here
    pass


@dataclass
class {service_name}Response:
    """Response DTO for {service_name}"""
    # Add fields here
    pass


class {service_name}Service:
    """Service layer for {service_name} operations"""
    
    def __init__(self, repository, logger):
        self._repository = repository
        self._logger = logger
    
    async def execute(self, request: {service_name}Request) -> {service_name}Response:
        """Execute {service_name} operation"""
        try:
            # Validate request
            self._validate_request(request)
            
            # Business logic here
            result = await self._process(request)
            
            # Return response
            return self._create_response(result)
            
        except Exception as e:
            self._logger.error(f"Error in {service_name}: {{e}}")
            raise
    
    def _validate_request(self, request: {service_name}Request) -> None:
        """Validate request data"""
        # Add validation logic
        pass
    
    async def _process(self, request: {service_name}Request):
        """Process business logic"""
        # Add processing logic
        pass
    
    def _create_response(self, result) -> {service_name}Response:
        """Create response from result"""
        return {service_name}Response()
'''


class ProjectRestructurer:
    """Restructure project to Clean Architecture"""
    
    def __init__(self, project_path: str):
        self.project_path = Path(project_path)
    
    def create_clean_structure(self):
        """Create Clean Architecture folder structure"""
        folders = [
            'src/domain/entities',
            'src/domain/value_objects',
            'src/domain/services',
            'src/domain/interfaces',
            'src/application/services',
            'src/application/dtos',
            'src/application/use_cases',
            'src/infrastructure/persistence',
            'src/infrastructure/external',
            'src/infrastructure/cache',
            'src/infrastructure/config',
            'src/presentation/api',
            'src/presentation/middleware',
            'src/presentation/validators',
            'src/presentation/responses',
            'src/shared/constants',
            'src/shared/exceptions',
            'src/shared/utils',
        ]
        
        for folder in folders:
            folder_path = self.project_path / folder
            folder_path.mkdir(parents=True, exist_ok=True)
            
            # Create __init__.py
            init_file = folder_path / '__init__.py'
            if not init_file.exists():
                init_file.write_text('"""Package initialization"""')
        
        print(f"✅ Created Clean Architecture structure in {self.project_path}")
    
    def generate_templates(self):
        """Generate template files"""
        templates = {
            'src/domain/interfaces/repositories.py': RefactoringGenerator.generate_repository_interface(),
            'src/shared/constants/app_constants.py': '',  # Will be filled by analyzer
            'src/shared/exceptions/base.py': self._generate_exception_template(),
        }
        
        for file_path, content in templates.items():
            full_path = self.project_path / file_path
            if not full_path.exists() and content:
                full_path.write_text(content)
                print(f"✅ Generated template: {file_path}")
    
    def _generate_exception_template(self) -> str:
        """Generate exception template"""
        return '''"""
Custom Exception Classes
"""


class ApplicationError(Exception):
    """Base application exception"""
    
    def __init__(self, message: str, code: str = None):
        self.message = message
        self.code = code or self.__class__.__name__
        super().__init__(self.message)


class DomainError(ApplicationError):
    """Domain layer exceptions"""
    pass


class ValidationError(ApplicationError):
    """Validation exceptions"""
    pass


class RepositoryError(ApplicationError):
    """Repository layer exceptions"""
    pass


class ServiceError(ApplicationError):
    """Service layer exceptions"""
    pass
'''


def main():
    parser = argparse.ArgumentParser(description='Clean Code Refactoring Tool')
    parser.add_argument('command', choices=['analyze', 'restructure', 'generate'],
                       help='Command to execute')
    parser.add_argument('--path', default='.',
                       help='Project path (default: current directory)')
    parser.add_argument('--output', default='refactoring_report.json',
                       help='Output file for analysis report')
    parser.add_argument('--service', help='Service name for template generation')
    
    args = parser.parse_args()
    
    if args.command == 'analyze':
        print("🔍 Analyzing project for Clean Code violations...")
        analyzer = CodeAnalyzer(args.path)
        results = analyzer.analyze_project()
        
        # Save report
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Print summary
        print(f"\n📊 Analysis Complete:")
        print(f"   Total files analyzed: {results['total_files']}")
        print(f"   Total violations found: {results['total_violations']}")
        print(f"\n📋 Report saved to: {args.output}")
        
        # Generate constants file if magic numbers found
        if any('MAGIC_NUMBER' in str(v) for f in results['files'] for v in f['violations']):
            constants = RefactoringGenerator.generate_constants_file(results)
            const_file = Path(args.path) / 'src' / 'shared' / 'constants' / 'generated_constants.py'
            const_file.parent.mkdir(parents=True, exist_ok=True)
            const_file.write_text(constants)
            print(f"✅ Generated constants file: {const_file}")
    
    elif args.command == 'restructure':
        print("🏗️ Restructuring project to Clean Architecture...")
        restructurer = ProjectRestructurer(args.path)
        restructurer.create_clean_structure()
        restructurer.generate_templates()
        print("✅ Project restructuring complete!")
    
    elif args.command == 'generate':
        if not args.service:
            print("❌ Error: --service name required for generate command")
            return
        
        print(f"📝 Generating service template for: {args.service}")
        template = RefactoringGenerator.generate_service_template(args.service)
        
        service_file = Path(args.path) / 'src' / 'application' / 'services' / f'{args.service.lower()}_service.py'
        service_file.parent.mkdir(parents=True, exist_ok=True)
        service_file.write_text(template)
        
        print(f"✅ Generated service template: {service_file}")


if __name__ == '__main__':
    main()
