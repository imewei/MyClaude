#!/usr/bin/env python3
"""
Custom Analyzer Plugin
======================

Custom code analyzer that performs complexity analysis and provides metrics.

Features:
- Function complexity analysis
- Import analysis
- Code metrics (LOC, functions, classes)
- Recommendations for refactoring
"""

import sys
import ast
import re
from pathlib import Path
from typing import Dict, List, Any

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.plugin_base import CommandPlugin, PluginContext, PluginResult
from api.command_api import CommandAPI


class CustomAnalyzerPlugin(CommandPlugin):
    """Custom code analyzer plugin"""

    def load(self) -> bool:
        """Load plugin"""
        self.logger.info(f"Loading {self.metadata.name} plugin")
        return True

    def execute(self, context: PluginContext) -> PluginResult:
        """Execute code analysis"""
        work_dir = context.work_dir

        # Configuration
        max_complexity = self.get_config('max_complexity', 10)
        analyze_imports = self.get_config('analyze_imports', True)
        analyze_functions = self.get_config('analyze_functions', True)

        # Find Python files
        py_files = list(work_dir.rglob("*.py"))

        if not py_files:
            return CommandAPI.error_result(
                self.metadata.name,
                "No Python files found in directory"
            )

        # Analyze files
        analysis_results = {
            "total_files": len(py_files),
            "total_lines": 0,
            "total_functions": 0,
            "total_classes": 0,
            "complex_functions": [],
            "import_analysis": {},
            "recommendations": []
        }

        for py_file in py_files:
            try:
                file_analysis = self._analyze_file(py_file)
                analysis_results["total_lines"] += file_analysis["lines"]
                analysis_results["total_functions"] += file_analysis["functions"]
                analysis_results["total_classes"] += file_analysis["classes"]

                # Check complexity
                if analyze_functions:
                    for func in file_analysis["function_details"]:
                        if func["complexity"] > max_complexity:
                            analysis_results["complex_functions"].append({
                                "file": str(py_file.relative_to(work_dir)),
                                "function": func["name"],
                                "complexity": func["complexity"]
                            })

                # Import analysis
                if analyze_imports:
                    for imp in file_analysis["imports"]:
                        if imp not in analysis_results["import_analysis"]:
                            analysis_results["import_analysis"][imp] = 0
                        analysis_results["import_analysis"][imp] += 1

            except Exception as e:
                self.logger.warning(f"Error analyzing {py_file}: {e}")

        # Generate recommendations
        if analysis_results["complex_functions"]:
            analysis_results["recommendations"].append(
                f"Found {len(analysis_results['complex_functions'])} functions with "
                f"complexity > {max_complexity}. Consider refactoring."
            )

        if analysis_results["total_lines"] > 10000:
            analysis_results["recommendations"].append(
                "Large codebase detected. Consider modularization."
            )

        # Return results
        return CommandAPI.success_result(
            plugin_name=self.metadata.name,
            data=analysis_results,
            message=f"Analyzed {len(py_files)} Python files"
        )

    def _analyze_file(self, file_path: Path) -> Dict[str, Any]:
        """Analyze a single Python file"""
        content = file_path.read_text()
        lines = content.split('\n')

        # Parse AST
        try:
            tree = ast.parse(content)
        except SyntaxError:
            return {
                "lines": len(lines),
                "functions": 0,
                "classes": 0,
                "function_details": [],
                "imports": []
            }

        # Count elements
        functions = []
        classes = 0
        imports = []

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                complexity = self._calculate_complexity(node)
                functions.append({
                    "name": node.name,
                    "line": node.lineno,
                    "complexity": complexity
                })
            elif isinstance(node, ast.ClassDef):
                classes += 1
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.append(node.module)

        return {
            "lines": len(lines),
            "functions": len(functions),
            "classes": classes,
            "function_details": functions,
            "imports": list(set(imports))
        }

    def _calculate_complexity(self, node: ast.FunctionDef) -> int:
        """Calculate cyclomatic complexity of a function"""
        complexity = 1  # Base complexity

        for child in ast.walk(node):
            # Count decision points
            if isinstance(child, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1

        return complexity

    def get_command_info(self) -> dict:
        """Get command information"""
        return {
            "name": "custom-analyzer",
            "description": "Analyze code complexity and metrics",
            "usage": "/custom-analyzer [--max-complexity=N]",
            "arguments": [
                {
                    "name": "--max-complexity",
                    "description": "Maximum allowed complexity",
                    "default": 10
                }
            ],
            "examples": [
                {
                    "command": "/custom-analyzer",
                    "description": "Analyze code with default settings"
                },
                {
                    "command": "/custom-analyzer --max-complexity=15",
                    "description": "Analyze with custom complexity threshold"
                }
            ]
        }


def main():
    """Test plugin"""
    from core.plugin_base import PluginMetadata, PluginType

    metadata = PluginMetadata(
        name="custom-analyzer",
        version="1.0.0",
        plugin_type=PluginType.COMMAND,
        description="Custom code analyzer",
        author="Test"
    )

    plugin = CustomAnalyzerPlugin(metadata)
    plugin.load()

    context = PluginContext(
        plugin_name="custom-analyzer",
        command_name="custom-analyzer",
        work_dir=Path.cwd(),
        config={},
        framework_version="2.0.0"
    )

    result = plugin.execute(context)
    print(CommandAPI.format_output(result))

    return 0


if __name__ == "__main__":
    sys.exit(main())