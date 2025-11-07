# Code Analysis Framework

**Version**: 1.0.3
**Category**: code-documentation
**Purpose**: Comprehensive code complexity analysis and structure extraction

## Overview

This framework provides AST-based code analysis to determine complexity, identify programming concepts, detect design patterns, and assess code structure across multiple languages.

## CodeAnalyzer Class

### Core Analysis Method

```python
import ast
import re
from typing import Dict, List, Tuple

class CodeAnalyzer:
    def analyze_complexity(self, code: str) -> Dict:
        """
        Analyze code complexity and structure

        Returns:
            Dictionary with complexity metrics, concepts, patterns, and difficulty level
        """
        analysis = {
            'complexity_score': 0,
            'concepts': [],
            'patterns': [],
            'dependencies': [],
            'difficulty_level': 'beginner'
        }

        # Parse code structure
        try:
            tree = ast.parse(code)

            # Analyze complexity metrics
            analysis['metrics'] = {
                'lines_of_code': len(code.splitlines()),
                'cyclomatic_complexity': self._calculate_cyclomatic_complexity(tree),
                'nesting_depth': self._calculate_max_nesting(tree),
                'function_count': len([n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]),
                'class_count': len([n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)])
            }

            # Identify concepts used
            analysis['concepts'] = self._identify_concepts(tree)

            # Detect design patterns
            analysis['patterns'] = self._detect_patterns(tree)

            # Extract dependencies
            analysis['dependencies'] = self._extract_dependencies(tree)

            # Determine difficulty level
            analysis['difficulty_level'] = self._assess_difficulty(analysis)

        except SyntaxError as e:
            analysis['parse_error'] = str(e)

        return analysis
```

### Concept Identification

```python
def _identify_concepts(self, tree) -> List[str]:
    """
    Identify programming concepts used in the code

    Detects: async/await, decorators, context managers, generators,
    comprehensions, lambda functions, exception handling
    """
    concepts = []

    for node in ast.walk(tree):
        # Async/await
        if isinstance(node, (ast.AsyncFunctionDef, ast.AsyncWith, ast.AsyncFor)):
            concepts.append('asynchronous programming')

        # Decorators
        elif isinstance(node, ast.FunctionDef) and node.decorator_list:
            concepts.append('decorators')

        # Context managers
        elif isinstance(node, ast.With):
            concepts.append('context managers')

        # Generators
        elif isinstance(node, ast.Yield):
            concepts.append('generators')

        # List/Dict/Set comprehensions
        elif isinstance(node, (ast.ListComp, ast.DictComp, ast.SetComp)):
            concepts.append('comprehensions')

        # Lambda functions
        elif isinstance(node, ast.Lambda):
            concepts.append('lambda functions')

        # Exception handling
        elif isinstance(node, ast.Try):
            concepts.append('exception handling')

    return list(set(concepts))
```

### Complexity Metrics

```python
def _calculate_cyclomatic_complexity(self, tree) -> int:
    """Calculate cyclomatic complexity (decision points)"""
    complexity = 1  # Base complexity

    for node in ast.walk(tree):
        # Add 1 for each decision point
        if isinstance(node, (ast.If, ast.While, ast.For, ast.And, ast.Or)):
            complexity += 1
        elif isinstance(node, ast.ExceptHandler):
            complexity += 1
        elif isinstance(node, ast.comprehension):
            complexity += 1

    return complexity

def _calculate_max_nesting(self, tree) -> int:
    """Calculate maximum nesting depth"""
    def get_depth(node, current_depth=0):
        max_depth = current_depth
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.With, ast.Try)):
                child_depth = get_depth(child, current_depth + 1)
                max_depth = max(max_depth, child_depth)
            else:
                child_depth = get_depth(child, current_depth)
                max_depth = max(max_depth, child_depth)
        return max_depth

    return get_depth(tree)
```

### Difficulty Assessment

```python
def _assess_difficulty(self, analysis: Dict) -> str:
    """
    Assess code difficulty level based on metrics

    Levels: beginner, intermediate, advanced, expert
    """
    metrics = analysis['metrics']
    concepts = analysis['concepts']

    # Scoring system
    score = 0

    # Complexity metrics
    if metrics['cyclomatic_complexity'] > 20:
        score += 3
    elif metrics['cyclomatic_complexity'] > 10:
        score += 2
    elif metrics['cyclomatic_complexity'] > 5:
        score += 1

    # Nesting depth
    if metrics['nesting_depth'] > 4:
        score += 2
    elif metrics['nesting_depth'] > 3:
        score += 1

    # Advanced concepts
    advanced_concepts = ['asynchronous programming', 'decorators', 'generators', 'metaclasses']
    score += len([c for c in concepts if c in advanced_concepts])

    # Determine level
    if score >= 8:
        return 'expert'
    elif score >= 5:
        return 'advanced'
    elif score >= 2:
        return 'intermediate'
    else:
        return 'beginner'
```

## Language-Specific Analyzers

### JavaScript/TypeScript Analyzer

```python
class JavaScriptAnalyzer:
    """Analyze JavaScript/TypeScript code structure"""

    def analyze_js_code(self, code: str) -> Dict:
        """
        Analyze JS/TS code using appropriate parser

        Note: Requires @babel/parser or typescript for full analysis
        """
        # Use regex-based analysis for basic cases
        # For production, use proper JS/TS parser

        analysis = {
            'functions': self._find_functions(code),
            'classes': self._find_classes(code),
            'imports': self._find_imports(code),
            'exports': self._find_exports(code),
            'async_patterns': self._find_async_patterns(code)
        }

        return analysis

    def _find_functions(self, code: str) -> List[str]:
        """Find function declarations"""
        patterns = [
            r'function\s+(\w+)',  # function foo()
            r'const\s+(\w+)\s*=\s*(?:async\s+)?\(',  # const foo = ()
            r'(\w+)\s*:\s*(?:async\s+)?\(',  # foo: ()
        ]

        functions = []
        for pattern in patterns:
            functions.extend(re.findall(pattern, code))

        return list(set(functions))
```

### Go Analyzer

```python
class GoAnalyzer:
    """Analyze Go code structure"""

    def analyze_go_code(self, code: str) -> Dict:
        """
        Analyze Go code structure

        Extracts: packages, types, functions, interfaces
        """
        analysis = {
            'package': self._find_package(code),
            'imports': self._find_imports(code),
            'types': self._find_types(code),
            'functions': self._find_functions(code),
            'methods': self._find_methods(code),
            'interfaces': self._find_interfaces(code)
        }

        return analysis

    def _find_package(self, code: str) -> str:
        """Extract package name"""
        match = re.search(r'package\s+(\w+)', code)
        return match.group(1) if match else 'main'

    def _find_types(self, code: str) -> List[Dict]:
        """Find type definitions"""
        pattern = r'type\s+(\w+)\s+(struct|interface)\s*\{'
        matches = re.finditer(pattern, code)

        types = []
        for match in matches:
            types.append({
                'name': match.group(1),
                'kind': match.group(2)
            })

        return types
```

## Common Pitfalls Detection

```python
def analyze_common_pitfalls(self, code: str) -> List[Dict]:
    """
    Identify common mistakes and anti-patterns

    Returns: List of issues with severity and explanations
    """
    issues = []

    # Check for common Python pitfalls
    pitfall_patterns = [
        {
            'pattern': r'except:',
            'issue': 'Bare except clause',
            'severity': 'high',
            'explanation': 'Catches ALL exceptions including system exits. Use specific exception types.'
        },
        {
            'pattern': r'def.*\(\s*\):.*global',
            'issue': 'Global variable usage',
            'severity': 'medium',
            'explanation': 'Makes code harder to test and reason about. Use parameters or class attributes.'
        },
        {
            'pattern': r'eval\(',
            'issue': 'Use of eval()',
            'severity': 'critical',
            'explanation': 'Security risk. Executes arbitrary code. Use ast.literal_eval() for safe evaluation.'
        },
        {
            'pattern': r'== None',
            'issue': 'Equality check with None',
            'severity': 'low',
            'explanation': 'Use "is None" instead of "== None" for identity check.'
        },
        {
            'pattern': r'def\s+\w+\([^)]*=\[\]',
            'issue': 'Mutable default argument',
            'severity': 'high',
            'explanation': 'Mutable defaults are shared across calls. Use None and initialize inside function.'
        }
    ]

    for pitfall in pitfall_patterns:
        if re.search(pitfall['pattern'], code):
            issues.append(pitfall)

    return issues
```

## Best Practices Checker

```python
def check_best_practices(self, code: str, language: str = 'python') -> Dict:
    """
    Check code against best practices

    Returns: Dictionary with scores and recommendations
    """
    checks = {
        'naming_conventions': self._check_naming(code, language),
        'documentation': self._check_documentation(code),
        'complexity': self._check_complexity(code),
        'type_hints': self._check_type_hints(code) if language == 'python' else None,
        'error_handling': self._check_error_handling(code)
    }

    # Calculate overall score
    scores = [v['score'] for v in checks.values() if v and 'score' in v]
    checks['overall_score'] = sum(scores) / len(scores) if scores else 0

    return checks

def _check_naming(self, code: str, language: str) -> Dict:
    """Check naming convention compliance"""
    if language == 'python':
        # PEP 8 naming conventions
        snake_case_pattern = r'^[a-z_][a-z0-9_]*$'
        class_pattern = r'^[A-Z][a-zA-Z0-9]*$'

        tree = ast.parse(code)
        issues = []

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                if not re.match(snake_case_pattern, node.name):
                    issues.append(f"Function '{node.name}' doesn't follow snake_case")
            elif isinstance(node, ast.ClassDef):
                if not re.match(class_pattern, node.name):
                    issues.append(f"Class '{node.name}' doesn't follow PascalCase")

        return {
            'score': max(0, 100 - len(issues) * 10),
            'issues': issues
        }

    return {'score': 100, 'issues': []}
```

## Usage Examples

### Basic Code Analysis

```python
analyzer = CodeAnalyzer()

code = '''
def fibonacci(n):
    """Calculate nth Fibonacci number"""
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
'''

analysis = analyzer.analyze_complexity(code)
print(f"Difficulty: {analysis['difficulty_level']}")
print(f"Concepts: {analysis['concepts']}")
print(f"Cyclomatic Complexity: {analysis['metrics']['cyclomatic_complexity']}")
```

### Pitfall Detection

```python
code_with_issues = '''
def process_data(data=[]):  # Mutable default!
    data.append(1)
    try:
        result = eval(user_input)  # Security risk!
    except:  # Bare except!
        pass
    return data
'''

pitfalls = analyzer.analyze_common_pitfalls(code_with_issues)
for issue in pitfalls:
    print(f"{issue['severity'].upper()}: {issue['issue']}")
    print(f"  {issue['explanation']}")
```

### Multi-Language Analysis

```python
# Python
py_analysis = CodeAnalyzer().analyze_complexity(python_code)

# JavaScript
js_analysis = JavaScriptAnalyzer().analyze_js_code(javascript_code)

# Go
go_analysis = GoAnalyzer().analyze_go_code(go_code)
```

## Integration Points

This framework integrates with:
- **Visualization Techniques**: Provides structure data for diagram generation
- **Learning Resources**: Identifies concepts that need explanation
- **Documentation Generation**: Extracts structure for API docs

## Performance Considerations

- **AST Parsing**: O(n) where n = lines of code
- **Concept Detection**: Single tree walk
- **Pitfall Detection**: Regex-based, fast for most codebases
- **Caching**: Consider caching parsed trees for large files

## Extension Points

To add new language support:
1. Create language-specific analyzer class
2. Implement code parsing (AST or regex-based)
3. Extract structure (functions, classes, types)
4. Map to common analysis format
5. Add to main analyzer dispatcher
