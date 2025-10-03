#!/usr/bin/env python3
"""
AST Analyzer Utilities for Command Executors
Provides AST-based code analysis for Python, JavaScript, and other languages
"""

import ast
from pathlib import Path
from typing import Dict, List, Set, Optional, Any
from dataclasses import dataclass


@dataclass
class FunctionInfo:
    """Information about a function"""
    name: str
    lineno: int
    args: List[str]
    returns: Optional[str]
    docstring: Optional[str]
    is_async: bool
    decorators: List[str]


@dataclass
class ClassInfo:
    """Information about a class"""
    name: str
    lineno: int
    bases: List[str]
    methods: List[FunctionInfo]
    docstring: Optional[str]
    decorators: List[str]


@dataclass
class ImportInfo:
    """Information about an import"""
    module: str
    names: List[str]
    alias: Optional[str]
    lineno: int
    is_from_import: bool


class PythonASTAnalyzer:
    """Analyzer for Python AST"""

    def __init__(self, file_path: Path):
        self.file_path = file_path
        with open(file_path, 'r') as f:
            self.source = f.read()
        self.tree = ast.parse(self.source, filename=str(file_path))

    def get_functions(self) -> List[FunctionInfo]:
        """Extract all function definitions"""
        functions = []

        for node in ast.walk(self.tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                functions.append(FunctionInfo(
                    name=node.name,
                    lineno=node.lineno,
                    args=[arg.arg for arg in node.args.args],
                    returns=ast.unparse(node.returns) if node.returns else None,
                    docstring=ast.get_docstring(node),
                    is_async=isinstance(node, ast.AsyncFunctionDef),
                    decorators=[ast.unparse(d) for d in node.decorator_list]
                ))

        return functions

    def get_classes(self) -> List[ClassInfo]:
        """Extract all class definitions"""
        classes = []

        for node in ast.walk(self.tree):
            if isinstance(node, ast.ClassDef):
                methods = []
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        methods.append(FunctionInfo(
                            name=item.name,
                            lineno=item.lineno,
                            args=[arg.arg for arg in item.args.args],
                            returns=ast.unparse(item.returns) if item.returns else None,
                            docstring=ast.get_docstring(item),
                            is_async=isinstance(item, ast.AsyncFunctionDef),
                            decorators=[ast.unparse(d) for d in item.decorator_list]
                        ))

                classes.append(ClassInfo(
                    name=node.name,
                    lineno=node.lineno,
                    bases=[ast.unparse(base) for base in node.bases],
                    methods=methods,
                    docstring=ast.get_docstring(node),
                    decorators=[ast.unparse(d) for d in node.decorator_list]
                ))

        return classes

    def get_imports(self) -> List[ImportInfo]:
        """Extract all import statements"""
        imports = []

        for node in ast.walk(self.tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(ImportInfo(
                        module=alias.name,
                        names=[alias.name],
                        alias=alias.asname,
                        lineno=node.lineno,
                        is_from_import=False
                    ))
            elif isinstance(node, ast.ImportFrom):
                imports.append(ImportInfo(
                    module=node.module or '',
                    names=[alias.name for alias in node.names],
                    alias=None,
                    lineno=node.lineno,
                    is_from_import=True
                ))

        return imports

    def get_used_names(self) -> Set[str]:
        """Get all names used in the code"""
        used_names = set()

        for node in ast.walk(self.tree):
            if isinstance(node, ast.Name):
                used_names.add(node.id)
            elif isinstance(node, ast.Attribute):
                if isinstance(node.value, ast.Name):
                    used_names.add(node.value.id)

        return used_names

    def find_unused_imports(self) -> List[ImportInfo]:
        """Find imports that are not used in the code"""
        imports = self.get_imports()
        used_names = self.get_used_names()
        unused = []

        for imp in imports:
            # Check if any imported name is used
            is_used = False
            for name in imp.names:
                # Handle 'from module import *'
                if name == '*':
                    is_used = True
                    break

                # Check if the imported name or its alias is used
                check_name = imp.alias if imp.alias else name
                if check_name in used_names:
                    is_used = True
                    break

            if not is_used:
                unused.append(imp)

        return unused

    def find_undefined_names(self) -> Set[str]:
        """Find names that are used but not defined"""
        defined_names = set()
        used_names = set()

        # Collect defined names
        for node in ast.walk(self.tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                defined_names.add(node.name)
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        defined_names.add(target.id)
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        defined_names.add(alias.asname or alias.name)
                else:
                    for alias in node.names:
                        defined_names.add(alias.asname or alias.name)

        # Collect used names
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                used_names.add(node.id)

        # Exclude built-in names
        builtins = set(dir(__builtins__))
        undefined = (used_names - defined_names) - builtins

        return undefined

    def get_complexity(self, function_name: Optional[str] = None) -> Dict[str, int]:
        """
        Calculate cyclomatic complexity

        Args:
            function_name: Specific function to analyze (None for all)

        Returns:
            Dict mapping function names to complexity scores
        """
        complexities = {}

        for node in ast.walk(self.tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if function_name and node.name != function_name:
                    continue

                complexity = 1  # Base complexity

                # Count decision points
                for child in ast.walk(node):
                    if isinstance(child, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                        complexity += 1
                    elif isinstance(child, ast.BoolOp):
                        complexity += len(child.values) - 1

                complexities[node.name] = complexity

        return complexities

    def find_dead_code(self) -> List[Dict[str, Any]]:
        """Find potentially dead code (unreachable after return/raise)"""
        dead_code = []

        class DeadCodeFinder(ast.NodeVisitor):
            def __init__(self):
                self.in_function = False
                self.found_dead = []

            def visit_FunctionDef(self, node):
                self.in_function = True
                found_terminator = False

                for i, stmt in enumerate(node.body):
                    if found_terminator:
                        self.found_dead.append({
                            'function': node.name,
                            'lineno': stmt.lineno,
                            'type': 'unreachable_after_return'
                        })

                    if isinstance(stmt, (ast.Return, ast.Raise)):
                        found_terminator = True

                self.generic_visit(node)
                self.in_function = False

            visit_AsyncFunctionDef = visit_FunctionDef

        finder = DeadCodeFinder()
        finder.visit(self.tree)
        return finder.found_dead

    def get_function_calls(self) -> Dict[str, List[str]]:
        """Get all function calls made in the code"""
        calls = {}

        class CallVisitor(ast.NodeVisitor):
            def __init__(self):
                self.current_function = None
                self.call_graph = {}

            def visit_FunctionDef(self, node):
                old_function = self.current_function
                self.current_function = node.name
                self.call_graph[node.name] = []
                self.generic_visit(node)
                self.current_function = old_function

            visit_AsyncFunctionDef = visit_FunctionDef

            def visit_Call(self, node):
                if self.current_function:
                    if isinstance(node.func, ast.Name):
                        self.call_graph[self.current_function].append(node.func.id)
                    elif isinstance(node.func, ast.Attribute):
                        self.call_graph[self.current_function].append(
                            ast.unparse(node.func)
                        )
                self.generic_visit(node)

        visitor = CallVisitor()
        visitor.visit(self.tree)
        return visitor.call_graph

    def find_missing_docstrings(self) -> List[Dict[str, Any]]:
        """Find functions and classes without docstrings"""
        missing = []

        for node in ast.walk(self.tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if not ast.get_docstring(node):
                    missing.append({
                        'type': 'function' if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) else 'class',
                        'name': node.name,
                        'lineno': node.lineno
                    })

        return missing

    def to_dict(self) -> Dict[str, Any]:
        """Convert analysis to dictionary"""
        return {
            'file': str(self.file_path),
            'functions': [f.__dict__ for f in self.get_functions()],
            'classes': [c.__dict__ for c in self.get_classes()],
            'imports': [i.__dict__ for i in self.get_imports()],
            'unused_imports': [i.__dict__ for i in self.find_unused_imports()],
            'undefined_names': list(self.find_undefined_names()),
            'complexity': self.get_complexity(),
            'dead_code': self.find_dead_code(),
            'missing_docstrings': self.find_missing_docstrings()
        }


class CodeAnalyzer:
    """Multi-language code analyzer factory"""

    @staticmethod
    def analyze_file(file_path: Path) -> Dict[str, Any]:
        """
        Analyze a file based on its extension

        Args:
            file_path: Path to file

        Returns:
            Analysis results as dict
        """
        suffix = file_path.suffix.lower()

        if suffix == '.py':
            analyzer = PythonASTAnalyzer(file_path)
            return analyzer.to_dict()
        elif suffix in ['.js', '.jsx', '.ts', '.tsx']:
            # JavaScript/TypeScript analysis would go here
            return {'error': 'JavaScript/TypeScript analysis not yet implemented'}
        elif suffix in ['.rs']:
            # Rust analysis would go here
            return {'error': 'Rust analysis not yet implemented'}
        else:
            return {'error': f'Unsupported file type: {suffix}'}

    @staticmethod
    def analyze_directory(dir_path: Path, pattern: str = '*.py') -> List[Dict[str, Any]]:
        """
        Analyze all files in a directory matching a pattern

        Args:
            dir_path: Path to directory
            pattern: File pattern to match

        Returns:
            List of analysis results
        """
        results = []
        for file_path in dir_path.rglob(pattern):
            if file_path.is_file():
                try:
                    result = CodeAnalyzer.analyze_file(file_path)
                    results.append(result)
                except Exception as e:
                    results.append({
                        'file': str(file_path),
                        'error': str(e)
                    })
        return results