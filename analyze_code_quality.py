#!/usr/bin/env python3
"""
Code Quality Analyzer
TEKNOFEST 2025 - Educational Technologies Platform

Analyzes code quality metrics and provides recommendations.
"""

import ast
import os
from pathlib import Path
from typing import Dict, List, Tuple, Any
import json
from datetime import datetime
from collections import defaultdict
import re


class CodeQualityAnalyzer:
    """Analyzes Python code for quality metrics."""
    
    def __init__(self, project_root: Path):
        """Initialize analyzer with project root."""
        self.project_root = project_root
        self.src_dir = project_root / "src"
        self.metrics = defaultdict(dict)
        self.issues = []
        self.recommendations = []
        
    def analyze_file(self, file_path: Path) -> Dict[str, Any]:
        """
        Analyze a single Python file.
        
        Args:
            file_path: Path to Python file
            
        Returns:
            Dictionary of metrics for the file
        """
        metrics = {
            "lines_of_code": 0,
            "functions": 0,
            "classes": 0,
            "complexity": 0,
            "max_function_length": 0,
            "avg_function_length": 0,
            "docstring_coverage": 0,
            "type_hint_coverage": 0,
            "magic_numbers": [],
            "long_functions": [],
            "missing_docstrings": [],
            "missing_type_hints": [],
            "duplicate_code": [],
            "code_smells": []
        }
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Count lines
            lines = content.split('\n')
            metrics["lines_of_code"] = len([l for l in lines if l.strip() and not l.strip().startswith('#')])
            
            # Parse AST
            tree = ast.parse(content)
            
            # Analyze AST
            metrics.update(self._analyze_ast(tree, content))
            
            # Find magic numbers
            metrics["magic_numbers"] = self._find_magic_numbers(content)
            
            # Find code smells
            metrics["code_smells"] = self._detect_code_smells(content)
            
        except Exception as e:
            self.issues.append(f"Error analyzing {file_path}: {e}")
            
        return metrics
    
    def _analyze_ast(self, tree: ast.AST, content: str) -> Dict[str, Any]:
        """Analyze AST for various metrics."""
        metrics = {
            "functions": 0,
            "classes": 0,
            "complexity": 0,
            "docstring_coverage": 0,
            "type_hint_coverage": 0,
            "long_functions": [],
            "missing_docstrings": [],
            "missing_type_hints": []
        }
        
        function_lengths = []
        total_functions = 0
        functions_with_docstrings = 0
        functions_with_type_hints = 0
        
        for node in ast.walk(tree):
            # Count classes
            if isinstance(node, ast.ClassDef):
                metrics["classes"] += 1
                if not ast.get_docstring(node):
                    metrics["missing_docstrings"].append(f"Class: {node.name}")
                    
            # Count and analyze functions
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                total_functions += 1
                metrics["functions"] += 1
                
                # Check function length
                if hasattr(node, 'lineno') and hasattr(node, 'end_lineno'):
                    func_length = node.end_lineno - node.lineno
                    function_lengths.append(func_length)
                    
                    if func_length > 20:  # Functions longer than 20 lines
                        metrics["long_functions"].append({
                            "name": node.name,
                            "lines": func_length
                        })
                
                # Check docstring
                if ast.get_docstring(node):
                    functions_with_docstrings += 1
                else:
                    metrics["missing_docstrings"].append(f"Function: {node.name}")
                
                # Check type hints
                has_return_type = node.returns is not None
                has_param_types = all(arg.annotation is not None for arg in node.args.args)
                
                if has_return_type and has_param_types:
                    functions_with_type_hints += 1
                else:
                    metrics["missing_type_hints"].append(node.name)
                
                # Calculate cyclomatic complexity
                metrics["complexity"] += self._calculate_complexity(node)
        
        # Calculate averages
        if function_lengths:
            metrics["avg_function_length"] = sum(function_lengths) / len(function_lengths)
            metrics["max_function_length"] = max(function_lengths)
        
        if total_functions > 0:
            metrics["docstring_coverage"] = (functions_with_docstrings / total_functions) * 100
            metrics["type_hint_coverage"] = (functions_with_type_hints / total_functions) * 100
            
        return metrics
    
    def _calculate_complexity(self, node: ast.AST) -> int:
        """Calculate cyclomatic complexity of a function."""
        complexity = 1  # Base complexity
        
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1
                
        return complexity
    
    def _find_magic_numbers(self, content: str) -> List[Dict[str, Any]]:
        """Find magic numbers in code."""
        magic_numbers = []
        
        # Regex to find numbers (excluding 0, 1, common values)
        pattern = r'\b(?!0\b|1\b|2\b|10\b|100\b|1000\b)\d+\.?\d*\b'
        
        for i, line in enumerate(content.split('\n'), 1):
            # Skip comments and strings
            if line.strip().startswith('#'):
                continue
                
            matches = re.finditer(pattern, line)
            for match in matches:
                # Check if it's not in a string
                if '"' not in line[:match.start()] or '"' not in line[match.end():]:
                    magic_numbers.append({
                        "line": i,
                        "value": match.group(),
                        "context": line.strip()
                    })
                    
        return magic_numbers
    
    def _detect_code_smells(self, content: str) -> List[Dict[str, Any]]:
        """Detect common code smells."""
        smells = []
        lines = content.split('\n')
        
        for i, line in enumerate(lines, 1):
            # Long lines
            if len(line) > 100:
                smells.append({
                    "type": "long_line",
                    "line": i,
                    "length": len(line),
                    "description": "Line exceeds 100 characters"
                })
            
            # TODO comments
            if 'TODO' in line or 'FIXME' in line or 'HACK' in line:
                smells.append({
                    "type": "todo_comment",
                    "line": i,
                    "description": f"Found TODO/FIXME/HACK comment"
                })
            
            # Broad exception handling
            if 'except Exception' in line or 'except:' in line:
                smells.append({
                    "type": "broad_exception",
                    "line": i,
                    "description": "Broad exception handling detected"
                })
            
            # Print statements (should use logging)
            if re.match(r'^\s*print\(', line):
                smells.append({
                    "type": "print_statement",
                    "line": i,
                    "description": "Use logging instead of print"
                })
                
        return smells
    
    def analyze_project(self) -> None:
        """Analyze entire project."""
        print("🔍 Analyzing code quality...")
        
        python_files = list(self.src_dir.glob("**/*.py"))
        
        for file_path in python_files:
            relative_path = file_path.relative_to(self.project_root)
            print(f"  Analyzing: {relative_path}")
            
            metrics = self.analyze_file(file_path)
            self.metrics[str(relative_path)] = metrics
            
        self._calculate_project_metrics()
        self._generate_recommendations()
        
    def _calculate_project_metrics(self) -> None:
        """Calculate project-wide metrics."""
        total_loc = sum(m["lines_of_code"] for m in self.metrics.values())
        total_functions = sum(m["functions"] for m in self.metrics.values())
        total_classes = sum(m["classes"] for m in self.metrics.values())
        
        avg_complexity = 0
        total_complexity = sum(m["complexity"] for m in self.metrics.values())
        if total_functions > 0:
            avg_complexity = total_complexity / total_functions
        
        # Calculate coverage
        files_with_metrics = [m for m in self.metrics.values() if m["functions"] > 0]
        
        avg_docstring_coverage = 0
        avg_type_hint_coverage = 0
        
        if files_with_metrics:
            avg_docstring_coverage = sum(m["docstring_coverage"] for m in files_with_metrics) / len(files_with_metrics)
            avg_type_hint_coverage = sum(m["type_hint_coverage"] for m in files_with_metrics) / len(files_with_metrics)
        
        self.project_metrics = {
            "total_files": len(self.metrics),
            "total_lines_of_code": total_loc,
            "total_functions": total_functions,
            "total_classes": total_classes,
            "average_complexity": round(avg_complexity, 2),
            "average_docstring_coverage": round(avg_docstring_coverage, 2),
            "average_type_hint_coverage": round(avg_type_hint_coverage, 2),
            "total_issues": len(self.issues)
        }
        
    def _generate_recommendations(self) -> None:
        """Generate recommendations based on analysis."""
        
        # Check docstring coverage
        if self.project_metrics["average_docstring_coverage"] < 80:
            self.recommendations.append({
                "priority": "HIGH",
                "category": "Documentation",
                "recommendation": f"Improve docstring coverage (currently {self.project_metrics['average_docstring_coverage']:.1f}%)"
            })
        
        # Check type hint coverage
        if self.project_metrics["average_type_hint_coverage"] < 80:
            self.recommendations.append({
                "priority": "HIGH",
                "category": "Type Safety",
                "recommendation": f"Add type hints (currently {self.project_metrics['average_type_hint_coverage']:.1f}% coverage)"
            })
        
        # Check complexity
        if self.project_metrics["average_complexity"] > 5:
            self.recommendations.append({
                "priority": "MEDIUM",
                "category": "Complexity",
                "recommendation": f"Reduce cyclomatic complexity (average: {self.project_metrics['average_complexity']:.1f})"
            })
        
        # Check for long functions
        total_long_functions = sum(len(m["long_functions"]) for m in self.metrics.values())
        if total_long_functions > 0:
            self.recommendations.append({
                "priority": "MEDIUM",
                "category": "Function Length",
                "recommendation": f"Refactor {total_long_functions} long functions (>20 lines)"
            })
        
        # Check for magic numbers
        total_magic_numbers = sum(len(m["magic_numbers"]) for m in self.metrics.values())
        if total_magic_numbers > 0:
            self.recommendations.append({
                "priority": "LOW",
                "category": "Magic Numbers",
                "recommendation": f"Extract {total_magic_numbers} magic numbers to constants"
            })
        
    def generate_report(self) -> None:
        """Generate detailed quality report."""
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "project_metrics": self.project_metrics,
            "recommendations": self.recommendations,
            "file_metrics": self.metrics,
            "issues": self.issues
        }
        
        # Save JSON report
        report_path = self.project_root / "code_quality_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Generate HTML report
        self._generate_html_report(report)
        
        print(f"\n📊 Reports generated:")
        print(f"  - JSON: {report_path}")
        print(f"  - HTML: {self.project_root / 'code_quality_report.html'}")
        
    def _generate_html_report(self, report: Dict) -> None:
        """Generate HTML report."""
        
        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Code Quality Report - TEKNOFEST 2025</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 10px; }}
        h1 {{ color: #333; border-bottom: 3px solid #007bff; padding-bottom: 10px; }}
        h2 {{ color: #555; margin-top: 30px; }}
        .metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin: 20px 0; }}
        .metric {{ background: #f8f9fa; padding: 15px; border-radius: 8px; border-left: 4px solid #007bff; }}
        .metric-value {{ font-size: 24px; font-weight: bold; color: #007bff; }}
        .metric-label {{ color: #666; font-size: 14px; margin-top: 5px; }}
        .recommendation {{ background: #fff3cd; padding: 10px; margin: 10px 0; border-radius: 5px; border-left: 4px solid #ffc107; }}
        .high {{ border-left-color: #dc3545; background: #f8d7da; }}
        .medium {{ border-left-color: #ffc107; background: #fff3cd; }}
        .low {{ border-left-color: #28a745; background: #d4edda; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background: #007bff; color: white; }}
        .timestamp {{ color: #666; font-size: 14px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Code Quality Report</h1>
        <p class="timestamp">Generated: {report['timestamp']}</p>
        
        <h2>Project Metrics</h2>
        <div class="metrics">
            <div class="metric">
                <div class="metric-value">{report['project_metrics']['total_files']}</div>
                <div class="metric-label">Total Files</div>
            </div>
            <div class="metric">
                <div class="metric-value">{report['project_metrics']['total_lines_of_code']:,}</div>
                <div class="metric-label">Lines of Code</div>
            </div>
            <div class="metric">
                <div class="metric-value">{report['project_metrics']['total_functions']}</div>
                <div class="metric-label">Functions</div>
            </div>
            <div class="metric">
                <div class="metric-value">{report['project_metrics']['total_classes']}</div>
                <div class="metric-label">Classes</div>
            </div>
            <div class="metric">
                <div class="metric-value">{report['project_metrics']['average_complexity']}</div>
                <div class="metric-label">Avg Complexity</div>
            </div>
            <div class="metric">
                <div class="metric-value">{report['project_metrics']['average_docstring_coverage']:.1f}%</div>
                <div class="metric-label">Docstring Coverage</div>
            </div>
            <div class="metric">
                <div class="metric-value">{report['project_metrics']['average_type_hint_coverage']:.1f}%</div>
                <div class="metric-label">Type Hint Coverage</div>
            </div>
        </div>
        
        <h2>Recommendations</h2>
        {"".join(f'<div class="recommendation {r["priority"].lower()}"><strong>[{r["priority"]}]</strong> {r["category"]}: {r["recommendation"]}</div>' for r in report['recommendations'])}
        
        <h2>Top Issues by File</h2>
        <table>
            <tr>
                <th>File</th>
                <th>Long Functions</th>
                <th>Magic Numbers</th>
                <th>Code Smells</th>
                <th>Complexity</th>
            </tr>
            {"".join(f'''<tr>
                <td>{file}</td>
                <td>{len(metrics['long_functions'])}</td>
                <td>{len(metrics['magic_numbers'])}</td>
                <td>{len(metrics['code_smells'])}</td>
                <td>{metrics['complexity']}</td>
            </tr>''' for file, metrics in list(report['file_metrics'].items())[:10])}
        </table>
    </div>
</body>
</html>"""
        
        html_path = self.project_root / "code_quality_report.html"
        html_path.write_text(html)
        
    def print_summary(self) -> None:
        """Print summary to console."""
        
        print("\n" + "="*60)
        print("CODE QUALITY ANALYSIS SUMMARY")
        print("="*60)
        
        print(f"\n📊 Project Metrics:")
        print(f"  • Files analyzed: {self.project_metrics['total_files']}")
        print(f"  • Total LOC: {self.project_metrics['total_lines_of_code']:,}")
        print(f"  • Functions: {self.project_metrics['total_functions']}")
        print(f"  • Classes: {self.project_metrics['total_classes']}")
        print(f"  • Avg Complexity: {self.project_metrics['average_complexity']:.1f}")
        print(f"  • Docstring Coverage: {self.project_metrics['average_docstring_coverage']:.1f}%")
        print(f"  • Type Hint Coverage: {self.project_metrics['average_type_hint_coverage']:.1f}%")
        
        if self.recommendations:
            print(f"\n🎯 Top Recommendations:")
            for rec in self.recommendations[:5]:
                icon = "🔴" if rec["priority"] == "HIGH" else "🟡" if rec["priority"] == "MEDIUM" else "🟢"
                print(f"  {icon} [{rec['priority']}] {rec['recommendation']}")
        
        # Calculate grade
        score = 0
        score += min(30, self.project_metrics['average_docstring_coverage'] * 0.3)
        score += min(30, self.project_metrics['average_type_hint_coverage'] * 0.3)
        score += max(0, 20 - self.project_metrics['average_complexity'] * 2)
        score += 20 if len(self.issues) == 0 else max(0, 20 - len(self.issues))
        
        grade = "A" if score >= 90 else "B" if score >= 80 else "C" if score >= 70 else "D" if score >= 60 else "F"
        
        print(f"\n📈 Overall Grade: {grade} ({score:.0f}/100)")
        
        if grade in ["A", "B"]:
            print("✅ Good code quality! Keep up the great work!")
        elif grade == "C":
            print("⚠️  Code quality is acceptable but could be improved.")
        else:
            print("❌ Code quality needs significant improvement.")


def main():
    """Main entry point."""
    project_root = Path(__file__).parent
    
    print("🔍 Code Quality Analyzer")
    print("="*60)
    
    analyzer = CodeQualityAnalyzer(project_root)
    analyzer.analyze_project()
    analyzer.generate_report()
    analyzer.print_summary()


if __name__ == "__main__":
    main()
