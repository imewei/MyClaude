# Test Generation Patterns

Comprehensive guide for AST parsing, test case generation algorithms, edge case identification, mocking strategies, and framework templates for automated test generation.

## AST-Based Code Analysis

### Python AST Parsing

```python
import ast
from typing import List, Dict, Any

class PythonCodeAnalyzer:
    """Analyze Python code using AST to extract testable units"""

    def analyze_module(self, file_path: str) -> Dict[str, Any]:
        """Extract functions, classes, and methods from Python module"""

        with open(file_path, 'r') as f:
            source = f.read()

        tree = ast.parse(source, filename=file_path)

        return {
            'functions': self.extract_functions(tree),
            'classes': self.extract_classes(tree),
            'imports': self.extract_imports(tree),
            'constants': self.extract_constants(tree)
        }

    def extract_functions(self, tree: ast.Module) -> List[Dict]:
        """Extract all function definitions"""
        functions = []

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Skip private and test functions
                if node.name.startswith('_') or node.name.startswith('test_'):
                    continue

                functions.append({
                    'name': node.name,
                    'args': self.extract_arguments(node.args),
                    'returns': self.extract_return_type(node.returns),
                    'decorators': [ast.unparse(d) for d in node.decorator_list],
                    'docstring': ast.get_docstring(node),
                    'is_async': isinstance(node, ast.AsyncFunctionDef),
                    'complexity': self.calculate_complexity(node),
                    'line_number': node.lineno
                })

        return functions

    def extract_arguments(self, args: ast.arguments) -> List[Dict]:
        """Extract function arguments with types"""
        arguments = []

        for arg in args.args:
            arguments.append({
                'name': arg.arg,
                'type': ast.unparse(arg.annotation) if arg.annotation else None,
                'default': None  # Handled separately
            })

        # Add defaults
        defaults = args.defaults
        if defaults:
            for i, default in enumerate(defaults):
                arg_index = len(arguments) - len(defaults) + i
                arguments[arg_index]['default'] = ast.unparse(default)

        return arguments

    def extract_classes(self, tree: ast.Module) -> List[Dict]:
        """Extract all class definitions"""
        classes = []

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                methods = []

                for item in node.body:
                    if isinstance(item, ast.FunctionDef):
                        methods.append({
                            'name': item.name,
                            'args': self.extract_arguments(item.args),
                            'is_property': any(
                                isinstance(d, ast.Name) and d.id == 'property'
                                for d in item.decorator_list
                            ),
                            'is_classmethod': any(
                                isinstance(d, ast.Name) and d.id == 'classmethod'
                                for d in item.decorator_list
                            ),
                            'is_staticmethod': any(
                                isinstance(d, ast.Name) and d.id == 'staticmethod'
                                for d in item.decorator_list
                            )
                        })

                classes.append({
                    'name': node.name,
                    'bases': [ast.unparse(base) for base in node.bases],
                    'methods': methods,
                    'docstring': ast.get_docstring(node),
                    'line_number': node.lineno
                })

        return classes

    def calculate_complexity(self, node: ast.FunctionDef) -> int:
        """Calculate cyclomatic complexity"""
        complexity = 1

        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1

        return complexity
```

### TypeScript AST Parsing

```typescript
import * as ts from 'typescript';
import * as fs from 'fs';

interface FunctionInfo {
  name: string;
  parameters: ParameterInfo[];
  returnType: string | null;
  isAsync: boolean;
  isExported: boolean;
  complexity: number;
}

interface ParameterInfo {
  name: string;
  type: string | null;
  optional: boolean;
  defaultValue: string | null;
}

class TypeScriptCodeAnalyzer {
  analyzeFile(filePath: string): { functions: FunctionInfo[] } {
    const sourceCode = fs.readFileSync(filePath, 'utf-8');
    const sourceFile = ts.createSourceFile(
      filePath,
      sourceCode,
      ts.ScriptTarget.Latest,
      true
    );

    const functions: FunctionInfo[] = [];

    const visit = (node: ts.Node) => {
      if (ts.isFunctionDeclaration(node) || ts.isFunctionExpression(node) || ts.isArrowFunction(node)) {
        const funcInfo = this.extractFunctionInfo(node);
        if (funcInfo) {
          functions.push(funcInfo);
        }
      }

      ts.forEachChild(node, visit);
    };

    visit(sourceFile);

    return { functions };
  }

  private extractFunctionInfo(node: ts.FunctionLikeDeclaration): FunctionInfo | null {
    const name = node.name?.getText() || 'anonymous';

    // Skip test functions
    if (name.startsWith('test') || name.includes('Test')) {
      return null;
    }

    const parameters = node.parameters.map(param => ({
      name: param.name.getText(),
      type: param.type ? param.type.getText() : null,
      optional: !!param.questionToken,
      defaultValue: param.initializer ? param.initializer.getText() : null
    }));

    const returnType = node.type ? node.type.getText() : null;

    const isAsync = !!(node.modifiers?.some(
      mod => mod.kind === ts.SyntaxKind.AsyncKeyword
    ));

    const isExported = !!(node.modifiers?.some(
      mod => mod.kind === ts.SyntaxKind.ExportKeyword
    ));

    return {
      name,
      parameters,
      returnType,
      isAsync,
      isExported,
      complexity: this.calculateComplexity(node)
    };
  }

  private calculateComplexity(node: ts.Node): number {
    let complexity = 1;

    const visit = (n: ts.Node) => {
      if (ts.isIfStatement(n) || ts.isWhileStatement(n) ||
          ts.isForStatement(n) || ts.isCatchClause(n)) {
        complexity++;
      }

      ts.forEachChild(n, visit);
    };

    visit(node);

    return complexity;
  }
}
```

## Test Case Generation Algorithms

### Algorithm 1: Happy Path Generation

```python
class HappyPathGenerator:
    """Generate happy path test cases"""

    def generate_test(self, func_info: Dict) -> str:
        """Generate basic happy path test"""

        test_name = f"test_{func_info['name']}_success"

        # Generate mock arguments
        args = self.generate_mock_args(func_info['args'])

        # Generate test code
        test_code = f"""
def {test_name}():
    '''Test {func_info['name']} with valid input'''
    # Arrange
    {self.generate_setup(func_info)}

    # Act
    result = {func_info['name']}({args})

    # Assert
    assert result is not None
    {self.generate_specific_assertions(func_info)}
"""

        return test_code

    def generate_mock_args(self, args: List[Dict]) -> str:
        """Generate mock arguments based on type hints"""

        mock_args = []

        for arg in args:
            arg_type = arg.get('type')

            if arg_type == 'str':
                mock_args.append(f"'{arg['name']}_value'")
            elif arg_type == 'int':
                mock_args.append('42')
            elif arg_type == 'float':
                mock_args.append('3.14')
            elif arg_type == 'bool':
                mock_args.append('True')
            elif arg_type == 'List':
                mock_args.append('[1, 2, 3]')
            elif arg_type == 'Dict':
                mock_args.append("{'key': 'value'}")
            else:
                # For custom types, create mock object
                mock_args.append(f"Mock(spec={arg_type})")

        return ', '.join(mock_args)

    def generate_specific_assertions(self, func_info: Dict) -> str:
        """Generate type-specific assertions"""

        return_type = func_info.get('returns')

        if return_type == 'bool':
            return 'assert isinstance(result, bool)'
        elif return_type == 'int':
            return 'assert isinstance(result, int)\n    assert result >= 0'
        elif return_type == 'str':
            return 'assert isinstance(result, str)\n    assert len(result) > 0'
        elif return_type == 'List':
            return 'assert isinstance(result, list)'
        elif return_type == 'Dict':
            return 'assert isinstance(result, dict)'
        else:
            return 'assert result is not None'
```

### Algorithm 2: Edge Case Generation

```python
class EdgeCaseGenerator:
    """Generate edge case test scenarios"""

    def generate_edge_cases(self, func_info: Dict) -> List[str]:
        """Generate comprehensive edge case tests"""

        tests = []

        # Empty input tests
        if self.accepts_collections(func_info):
            tests.append(self.generate_empty_input_test(func_info))

        # None/null tests
        if self.accepts_optional(func_info):
            tests.append(self.generate_none_input_test(func_info))

        # Boundary value tests
        if self.accepts_numeric(func_info):
            tests.extend(self.generate_boundary_tests(func_info))

        # Exception tests
        tests.append(self.generate_exception_test(func_info))

        return tests

    def generate_empty_input_test(self, func_info: Dict) -> str:
        """Test with empty collections"""

        return f"""
def test_{func_info['name']}_with_empty_input():
    '''Test {func_info['name']} handles empty input'''
    result = {func_info['name']}([])
    assert result is not None
"""

    def generate_none_input_test(self, func_info: Dict) -> str:
        """Test with None values"""

        return f"""
def test_{func_info['name']}_with_none():
    '''Test {func_info['name']} handles None input'''
    with pytest.raises((ValueError, TypeError)):
        {func_info['name']}(None)
"""

    def generate_boundary_tests(self, func_info: Dict) -> List[str]:
        """Test boundary values for numeric inputs"""

        tests = []

        # Zero test
        tests.append(f"""
def test_{func_info['name']}_with_zero():
    '''Test {func_info['name']} with zero value'''
    result = {func_info['name']}(0)
    assert result is not None
""")

        # Negative test
        tests.append(f"""
def test_{func_info['name']}_with_negative():
    '''Test {func_info['name']} with negative value'''
    result = {func_info['name']}(-1)
    assert result is not None
""")

        # Large value test
        tests.append(f"""
def test_{func_info['name']}_with_large_value():
    '''Test {func_info['name']} with large value'''
    result = {func_info['name']}(1000000)
    assert result is not None
""")

        return tests

    def generate_exception_test(self, func_info: Dict) -> str:
        """Test exception handling"""

        return f"""
def test_{func_info['name']}_handles_errors():
    '''Test {func_info['name']} error handling'''
    with pytest.raises(Exception):
        {func_info['name']}({self.generate_invalid_args(func_info)})
"""

    def generate_invalid_args(self, func_info: Dict) -> str:
        """Generate invalid arguments"""

        args = func_info.get('args', [])

        if not args:
            return ''

        # Return wrong type for first argument
        first_arg = args[0]
        arg_type = first_arg.get('type')

        if arg_type == 'str':
            return '123'  # Number instead of string
        elif arg_type in ['int', 'float']:
            return "'not_a_number'"  # String instead of number
        else:
            return 'None'
```

### Algorithm 3: Parametrized Test Generation

```python
class ParametrizedTestGenerator:
    """Generate parametrized tests for multiple scenarios"""

    def generate_parametrized_test(
        self,
        func_info: Dict,
        test_cases: List[Dict]
    ) -> str:
        """Generate pytest parametrized test"""

        # Build parameter string
        params = self.build_parameter_string(test_cases)

        # Build test cases
        cases = self.build_test_cases(test_cases)

        test_code = f"""
@pytest.mark.parametrize("{params}", [
{cases}
])
def test_{func_info['name']}_parametrized({params}):
    '''Parametrized test for {func_info['name']}'''
    result = {func_info['name']}(input_value)
    assert result == expected
"""

        return test_code

    def build_parameter_string(self, test_cases: List[Dict]) -> str:
        """Build parameter string for @pytest.mark.parametrize"""

        # Extract unique parameter names
        param_names = set()
        for case in test_cases:
            param_names.update(case.keys())

        return ', '.join(sorted(param_names))

    def build_test_cases(self, test_cases: List[Dict]) -> str:
        """Build test case tuples"""

        cases = []

        for case in test_cases:
            values = [repr(case[k]) for k in sorted(case.keys())]
            cases.append(f"    ({', '.join(values)})")

        return ',\n'.join(cases)

    def generate_test_cases_from_docstring(self, func_info: Dict) -> List[Dict]:
        """Extract test cases from function docstring"""

        docstring = func_info.get('docstring', '')

        if not docstring:
            return []

        # Parse docstring for examples
        # Example format:
        # Examples:
        #     >>> calculate_tax(100, 0.10)
        #     110.0
        #     >>> calculate_tax(200, 0.15)
        #     230.0

        examples = []

        lines = docstring.split('\n')
        for i, line in enumerate(lines):
            if '>>>' in line:
                # Extract input
                input_line = line.split('>>>')[1].strip()
                # Extract expected output (next line)
                if i + 1 < len(lines):
                    expected = lines[i + 1].strip()

                    examples.append({
                        'input': input_line,
                        'expected': expected
                    })

        return examples
```

## Mocking Strategies

### Auto-Mock Generation

```python
class MockGenerator:
    """Generate mock objects for dependencies"""

    def generate_mocks(self, func_info: Dict) -> str:
        """Generate mock setup for function dependencies"""

        # Analyze function body for external calls
        dependencies = self.extract_dependencies(func_info)

        mocks = []

        for dep in dependencies:
            if dep['type'] == 'function':
                mocks.append(self.generate_function_mock(dep))
            elif dep['type'] == 'class':
                mocks.append(self.generate_class_mock(dep))
            elif dep['type'] == 'module':
                mocks.append(self.generate_module_mock(dep))

        return '\n\n'.join(mocks)

    def generate_function_mock(self, dep: Dict) -> str:
        """Generate mock for external function"""

        return f"""
@pytest.fixture
def mock_{dep['name']}(monkeypatch):
    '''Mock {dep['name']} function'''
    mock_func = Mock(return_value={dep['default_return']})
    monkeypatch.setattr('{dep['module']}.{dep['name']}', mock_func)
    return mock_func
"""

    def generate_class_mock(self, dep: Dict) -> str:
        """Generate mock for external class"""

        methods = dep.get('methods', [])

        method_mocks = []
        for method in methods:
            method_mocks.append(
                f"    mock.{method}.return_value = {self.get_default_return(method)}"
            )

        return f"""
@pytest.fixture
def mock_{dep['name']}():
    '''Mock {dep['name']} class'''
    mock = Mock(spec={dep['name']})
{chr(10).join(method_mocks)}
    return mock
"""

    def generate_module_mock(self, dep: Dict) -> str:
        """Generate mock for entire module"""

        return f"""
@pytest.fixture(autouse=True)
def mock_{dep['name']}_module(monkeypatch):
    '''Mock {dep['name']} module'''
    mock_module = Mock()
    monkeypatch.setitem(sys.modules, '{dep['name']}', mock_module)
    return mock_module
"""

    def get_default_return(self, method_name: str) -> str:
        """Get sensible default return value based on method name"""

        if 'get' in method_name.lower():
            return "{'id': 1, 'name': 'Test'}"
        elif 'save' in method_name.lower() or 'create' in method_name.lower():
            return "True"
        elif 'delete' in method_name.lower():
            return "True"
        else:
            return "None"
```

## Framework Templates

### pytest Template

```python
PYTEST_TEMPLATE = '''
"""
Tests for {module_name}

Generated automatically by test-generate command.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock

from {module_path} import {imports}


class Test{class_name}:
    """Test suite for {class_name}"""

    @pytest.fixture
    def instance(self):
        """Create instance for testing"""
        return {class_name}()

    {test_methods}


# Standalone function tests
{function_tests}


# Parametrized tests
{parametrized_tests}


# Edge case tests
{edge_case_tests}


# Integration tests
{integration_tests}
'''
```

### Jest Template

```typescript
const JEST_TEMPLATE = `
/**
 * Tests for {moduleName}
 *
 * Generated automatically by test-generate command.
 */

import {{ {imports} }} from './{modulePath}';

describe('{moduleName}', () => {{
  {testSuites}
}});

{standaloneTests}
`;
```

### Test Method Template

```python
METHOD_TEST_TEMPLATE = '''
    def test_{method_name}_{scenario}(self, instance{fixtures}):
        """Test {method_name} - {scenario}"""
        # Arrange
        {setup}

        # Act
        result = instance.{method_name}({args})

        # Assert
        {assertions}
'''
```

## Test Coverage Analysis

```python
class CoverageGapAnalyzer:
    """Identify untested code and generate missing tests"""

    def analyze_coverage(self, coverage_file: str, source_dir: str) -> List[Dict]:
        """Find uncovered code and generate tests"""

        coverage_data = self.read_coverage(coverage_file)

        gaps = []

        for file_path, file_data in coverage_data['files'].items():
            missing_lines = file_data.get('missing_lines', [])

            if missing_lines:
                # Analyze source code
                analyzer = PythonCodeAnalyzer()
                code_info = analyzer.analyze_module(file_path)

                # Find uncovered functions
                uncovered = self.find_uncovered_functions(
                    code_info,
                    missing_lines
                )

                for func in uncovered:
                    # Generate tests for uncovered function
                    tests = self.generate_tests_for_function(func)

                    gaps.append({
                        'file': file_path,
                        'function': func['name'],
                        'missing_lines': func['missing_lines'],
                        'generated_tests': tests,
                        'priority': self.calculate_priority(func)
                    })

        # Sort by priority
        gaps.sort(key=lambda x: x['priority'], reverse=True)

        return gaps

    def calculate_priority(self, func: Dict) -> float:
        """Calculate test priority score"""

        score = 0.0

        # Higher complexity = higher priority
        score += func.get('complexity', 1) * 10

        # Public functions = higher priority
        if not func['name'].startswith('_'):
            score += 20

        # Missing coverage = higher priority
        coverage_pct = 1 - (len(func.get('missing_lines', [])) / max(func.get('total_lines', 1), 1))
        score += (1 - coverage_pct) * 30

        return score
```

## Best Practices

1. **Analyze code structure** before generating tests
2. **Generate happy path first**, then edge cases
3. **Use parametrized tests** for multiple scenarios
4. **Mock external dependencies** to isolate tests
5. **Generate meaningful test names** describing what is being tested
6. **Include docstrings** explaining test purpose
7. **Follow AAA pattern** (Arrange, Act, Assert)
8. **Generate assertions based on return types**
9. **Handle async functions** appropriately
10. **Maintain consistency** with existing test patterns
