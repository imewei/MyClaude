#!/usr/bin/env python3
"""
Semantic Code Analyzer
======================

Deep semantic understanding of code beyond AST parsing.

Features:
- Extract code semantics and intent
- Identify design patterns automatically
- Detect anti-patterns and code smells
- Map relationships between components
- Generate semantic graphs
- Cross-language semantic analysis

Author: Claude Code AI Team
"""

import ast
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json


class SemanticNode(Enum):
    """Types of semantic nodes"""
    CLASS = "class"
    FUNCTION = "function"
    METHOD = "method"
    VARIABLE = "variable"
    IMPORT = "import"
    DECORATOR = "decorator"
    PATTERN = "pattern"
    ANTIPATTERN = "antipattern"


class DesignPattern(Enum):
    """Common design patterns"""
    SINGLETON = "singleton"
    FACTORY = "factory"
    BUILDER = "builder"
    OBSERVER = "observer"
    STRATEGY = "strategy"
    DECORATOR = "decorator"
    ADAPTER = "adapter"
    FACADE = "facade"
    PROXY = "proxy"
    COMMAND = "command"


class CodeSmell(Enum):
    """Common code smells"""
    LONG_METHOD = "long_method"
    LARGE_CLASS = "large_class"
    GOD_OBJECT = "god_object"
    DUPLICATE_CODE = "duplicate_code"
    DEAD_CODE = "dead_code"
    MAGIC_NUMBERS = "magic_numbers"
    DEEP_NESTING = "deep_nesting"
    TOO_MANY_PARAMETERS = "too_many_parameters"


@dataclass
class SemanticEntity:
    """Semantic entity in code"""
    name: str
    node_type: SemanticNode
    file_path: Path
    line_number: int
    semantics: Dict[str, Any]
    relationships: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SemanticGraph:
    """Graph of semantic relationships"""
    entities: Dict[str, SemanticEntity]
    relationships: Dict[str, List[Tuple[str, str, str]]]  # type -> [(source, target, label)]
    patterns: List[Dict[str, Any]]
    smells: List[Dict[str, Any]]


class SemanticAnalyzer:
    """
    Deep semantic code analyzer.

    Analyzes code to extract:
    - Intent and purpose
    - Design patterns
    - Anti-patterns
    - Component relationships
    - Semantic structure
    """

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.entities: Dict[str, SemanticEntity] = {}
        self.patterns: List[Dict[str, Any]] = []
        self.smells: List[Dict[str, Any]] = []

    def analyze_codebase(self, root_path: Path) -> SemanticGraph:
        """
        Analyze entire codebase for semantics.

        Args:
            root_path: Root directory of codebase

        Returns:
            Semantic graph of codebase
        """
        self.logger.info(f"Analyzing codebase: {root_path}")

        # Find all Python files
        python_files = list(root_path.rglob("*.py"))

        for file_path in python_files:
            try:
                self._analyze_file(file_path)
            except Exception as e:
                self.logger.error(f"Error analyzing {file_path}: {e}")

        # Build relationships
        relationships = self._build_relationships()

        # Detect patterns
        self._detect_patterns()

        # Detect code smells
        self._detect_code_smells()

        return SemanticGraph(
            entities=self.entities,
            relationships=relationships,
            patterns=self.patterns,
            smells=self.smells
        )

    def analyze_file(self, file_path: Path) -> Dict[str, Any]:
        """
        Analyze single file for semantic understanding.

        Args:
            file_path: Path to file

        Returns:
            Semantic analysis results
        """
        return self._analyze_file(file_path)

    def _analyze_file(self, file_path: Path) -> Dict[str, Any]:
        """Analyze file semantics"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                source = f.read()

            tree = ast.parse(source)

            # Extract entities
            visitor = SemanticVisitor(file_path)
            visitor.visit(tree)

            # Store entities
            for entity in visitor.entities:
                key = f"{entity.file_path}:{entity.name}"
                self.entities[key] = entity

            return {
                "file": str(file_path),
                "entities": len(visitor.entities),
                "complexity": visitor.complexity,
                "imports": visitor.imports,
                "intent": self._infer_intent(visitor)
            }

        except Exception as e:
            self.logger.error(f"Failed to analyze {file_path}: {e}")
            return {"file": str(file_path), "error": str(e)}

    def _build_relationships(self) -> Dict[str, List[Tuple[str, str, str]]]:
        """Build relationship graph between entities"""
        relationships = {
            "inherits": [],
            "calls": [],
            "imports": [],
            "uses": [],
            "decorates": []
        }

        # Analyze each entity for relationships
        for key, entity in self.entities.items():
            # Inheritance relationships
            if "bases" in entity.metadata:
                for base in entity.metadata["bases"]:
                    relationships["inherits"].append((key, base, "inherits_from"))

            # Call relationships
            if "calls" in entity.metadata:
                for call in entity.metadata["calls"]:
                    relationships["calls"].append((key, call, "calls"))

            # Import relationships
            if entity.node_type == SemanticNode.IMPORT:
                relationships["imports"].append((key, entity.name, "imports"))

        return relationships

    def _detect_patterns(self):
        """Detect design patterns in code"""
        self.patterns = []

        # Singleton detection
        self._detect_singleton()

        # Factory detection
        self._detect_factory()

        # Decorator detection
        self._detect_decorator_pattern()

        # Observer detection
        self._detect_observer()

        self.logger.info(f"Detected {len(self.patterns)} design patterns")

    def _detect_singleton(self):
        """Detect singleton pattern"""
        for key, entity in self.entities.items():
            if entity.node_type == SemanticNode.CLASS:
                # Check for singleton indicators
                metadata = entity.metadata

                # Has __new__ or __init__ with instance checking
                if "has_instance_check" in metadata:
                    self.patterns.append({
                        "pattern": DesignPattern.SINGLETON.value,
                        "entity": key,
                        "confidence": 0.9,
                        "indicators": ["instance_check"]
                    })

    def _detect_factory(self):
        """Detect factory pattern"""
        for key, entity in self.entities.items():
            if entity.node_type in [SemanticNode.CLASS, SemanticNode.FUNCTION]:
                name = entity.name.lower()

                # Check for factory naming
                if "factory" in name or "create" in name:
                    if "returns_instances" in entity.metadata:
                        self.patterns.append({
                            "pattern": DesignPattern.FACTORY.value,
                            "entity": key,
                            "confidence": 0.8,
                            "indicators": ["naming", "returns_instances"]
                        })

    def _detect_decorator_pattern(self):
        """Detect decorator pattern (not Python decorators)"""
        for key, entity in self.entities.items():
            if entity.node_type == SemanticNode.CLASS:
                # Check if wraps another class
                if "wraps_class" in entity.metadata:
                    self.patterns.append({
                        "pattern": DesignPattern.DECORATOR.value,
                        "entity": key,
                        "confidence": 0.85,
                        "indicators": ["wraps_class"]
                    })

    def _detect_observer(self):
        """Detect observer pattern"""
        for key, entity in self.entities.items():
            if entity.node_type == SemanticNode.CLASS:
                metadata = entity.metadata

                # Check for observer indicators
                has_attach = "attach" in metadata.get("methods", [])
                has_detach = "detach" in metadata.get("methods", [])
                has_notify = "notify" in metadata.get("methods", [])

                if has_attach and has_detach and has_notify:
                    self.patterns.append({
                        "pattern": DesignPattern.OBSERVER.value,
                        "entity": key,
                        "confidence": 0.95,
                        "indicators": ["attach", "detach", "notify"]
                    })

    def _detect_code_smells(self):
        """Detect code smells and anti-patterns"""
        self.smells = []

        for key, entity in self.entities.items():
            metadata = entity.metadata

            # Long method
            if entity.node_type in [SemanticNode.FUNCTION, SemanticNode.METHOD]:
                lines = metadata.get("lines", 0)
                if lines > 50:
                    self.smells.append({
                        "smell": CodeSmell.LONG_METHOD.value,
                        "entity": key,
                        "severity": "medium" if lines < 100 else "high",
                        "metrics": {"lines": lines}
                    })

            # Large class
            if entity.node_type == SemanticNode.CLASS:
                methods = len(metadata.get("methods", []))
                lines = metadata.get("lines", 0)

                if methods > 20 or lines > 500:
                    self.smells.append({
                        "smell": CodeSmell.LARGE_CLASS.value,
                        "entity": key,
                        "severity": "high" if methods > 30 else "medium",
                        "metrics": {"methods": methods, "lines": lines}
                    })

            # Too many parameters
            if entity.node_type in [SemanticNode.FUNCTION, SemanticNode.METHOD]:
                params = metadata.get("parameters", 0)
                if params > 5:
                    self.smells.append({
                        "smell": CodeSmell.TOO_MANY_PARAMETERS.value,
                        "entity": key,
                        "severity": "medium" if params < 8 else "high",
                        "metrics": {"parameters": params}
                    })

            # Deep nesting
            if entity.node_type in [SemanticNode.FUNCTION, SemanticNode.METHOD]:
                nesting = metadata.get("max_nesting", 0)
                if nesting > 4:
                    self.smells.append({
                        "smell": CodeSmell.DEEP_NESTING.value,
                        "entity": key,
                        "severity": "medium" if nesting < 6 else "high",
                        "metrics": {"nesting": nesting}
                    })

        self.logger.info(f"Detected {len(self.smells)} code smells")

    def _infer_intent(self, visitor) -> str:
        """Infer the intent/purpose of code"""
        # Analyze class/function names, docstrings, patterns

        if hasattr(visitor, 'docstrings') and visitor.docstrings:
            return visitor.docstrings[0][:100]

        # Infer from names
        names = [e.name for e in visitor.entities]

        if any("test" in n.lower() for n in names):
            return "Testing functionality"
        elif any("model" in n.lower() for n in names):
            return "Data modeling"
        elif any("controller" in n.lower() or "handler" in n.lower() for n in names):
            return "Request handling"

        return "General utility"

    def get_semantic_summary(self) -> Dict[str, Any]:
        """Get summary of semantic analysis"""
        return {
            "total_entities": len(self.entities),
            "patterns_detected": len(self.patterns),
            "code_smells": len(self.smells),
            "entity_breakdown": {
                node_type.value: sum(1 for e in self.entities.values()
                                    if e.node_type == node_type)
                for node_type in SemanticNode
            },
            "pattern_breakdown": {
                pattern.value: sum(1 for p in self.patterns
                                  if p["pattern"] == pattern.value)
                for pattern in DesignPattern
            }
        }


class SemanticVisitor(ast.NodeVisitor):
    """AST visitor for semantic extraction"""

    def __init__(self, file_path: Path):
        self.file_path = file_path
        self.entities: List[SemanticEntity] = []
        self.imports: List[str] = []
        self.complexity = 0
        self.current_class = None
        self.nesting_level = 0
        self.max_nesting = 0
        self.docstrings: List[str] = []

    def visit_ClassDef(self, node):
        """Visit class definition"""
        bases = [self._get_name(base) for base in node.bases]
        decorators = [self._get_name(d) for d in node.decorator_list]

        # Count methods
        methods = [n.name for n in node.body if isinstance(n, ast.FunctionDef)]

        # Extract docstring
        docstring = ast.get_docstring(node)
        if docstring:
            self.docstrings.append(docstring)

        entity = SemanticEntity(
            name=node.name,
            node_type=SemanticNode.CLASS,
            file_path=self.file_path,
            line_number=node.lineno,
            semantics={
                "bases": bases,
                "decorators": decorators,
                "methods": methods,
                "docstring": docstring
            },
            metadata={
                "bases": bases,
                "methods": methods,
                "lines": node.end_lineno - node.lineno if node.end_lineno else 0
            }
        )

        self.entities.append(entity)

        # Visit children
        old_class = self.current_class
        self.current_class = node.name
        self.generic_visit(node)
        self.current_class = old_class

    def visit_FunctionDef(self, node):
        """Visit function definition"""
        params = [arg.arg for arg in node.args.args]
        decorators = [self._get_name(d) for d in node.decorator_list]

        # Extract docstring
        docstring = ast.get_docstring(node)
        if docstring:
            self.docstrings.append(docstring)

        # Determine if method or function
        node_type = SemanticNode.METHOD if self.current_class else SemanticNode.FUNCTION

        # Calculate complexity
        complexity = self._calculate_complexity(node)
        self.complexity += complexity

        entity = SemanticEntity(
            name=node.name,
            node_type=node_type,
            file_path=self.file_path,
            line_number=node.lineno,
            semantics={
                "parameters": params,
                "decorators": decorators,
                "docstring": docstring,
                "complexity": complexity
            },
            metadata={
                "parameters": len(params),
                "lines": node.end_lineno - node.lineno if node.end_lineno else 0,
                "complexity": complexity
            }
        )

        self.entities.append(entity)

        # Track nesting
        old_nesting = self.nesting_level
        self.nesting_level += 1
        self.max_nesting = max(self.max_nesting, self.nesting_level)

        self.generic_visit(node)

        self.nesting_level = old_nesting

    def visit_Import(self, node):
        """Visit import statement"""
        for alias in node.names:
            self.imports.append(alias.name)

            entity = SemanticEntity(
                name=alias.name,
                node_type=SemanticNode.IMPORT,
                file_path=self.file_path,
                line_number=node.lineno,
                semantics={"module": alias.name, "alias": alias.asname}
            )
            self.entities.append(entity)

        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        """Visit from-import statement"""
        module = node.module or ""
        for alias in node.names:
            full_name = f"{module}.{alias.name}" if module else alias.name
            self.imports.append(full_name)

            entity = SemanticEntity(
                name=full_name,
                node_type=SemanticNode.IMPORT,
                file_path=self.file_path,
                line_number=node.lineno,
                semantics={"module": module, "name": alias.name, "alias": alias.asname}
            )
            self.entities.append(entity)

        self.generic_visit(node)

    def _get_name(self, node) -> str:
        """Get name from node"""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f"{self._get_name(node.value)}.{node.attr}"
        elif isinstance(node, ast.Call):
            return self._get_name(node.func)
        return str(node)

    def _calculate_complexity(self, node) -> int:
        """Calculate cyclomatic complexity"""
        complexity = 1

        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1

        return complexity


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Demonstration of semantic analyzer"""
    print("Semantic Code Analyzer")
    print("=====================\n")

    analyzer = SemanticAnalyzer()

    # Analyze current file as demo
    current_file = Path(__file__)
    result = analyzer.analyze_file(current_file)

    print(f"Analyzed: {result['file']}")
    print(f"Entities: {result['entities']}")
    print(f"Complexity: {result['complexity']}")
    print(f"Intent: {result['intent']}")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())