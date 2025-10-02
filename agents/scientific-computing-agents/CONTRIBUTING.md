# Contributing to Scientific Computing Agents

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

---

## Table of Contents

1. [Getting Started](#getting-started)
2. [Code Standards](#code-standards)
3. [Testing Requirements](#testing-requirements)
4. [Documentation](#documentation)
5. [Pull Request Process](#pull-request-process)
6. [Issue Reporting](#issue-reporting)

---

## Getting Started

### Development Setup

1. **Fork and Clone**
   ```bash
   git fork https://github.com/your-org/scientific-computing-agents.git
   git clone https://github.com/YOUR-USERNAME/scientific-computing-agents.git
   cd scientific-computing-agents
   ```

2. **Create Virtual Environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt  # Development dependencies
   ```

4. **Run Tests**
   ```bash
   pytest tests/ -v
   ```

---

## Code Standards

### Python Style Guide

We follow **PEP 8** with some modifications:

- **Line length**: 100 characters (not 79)
- **Indentation**: 4 spaces
- **Quotes**: Double quotes for strings
- **Imports**: Grouped and sorted

### Type Hints

Use type hints for function signatures:

```python
from typing import Dict, List, Any, Optional

def process_data(
    input_data: Dict[str, Any],
    options: Optional[List[str]] = None
) -> AgentResult:
    ...
```

###Docstring Format

Use **Google-style docstrings**:

```python
def solve_problem(problem_data: Dict[str, Any]) -> AgentResult:
    """
    Solve a scientific computing problem.

    Args:
        problem_data: Dictionary containing problem specification with keys:
            - 'type': Problem type ('ode', 'pde', 'optimization')
            - 'parameters': Problem-specific parameters

    Returns:
        AgentResult containing solution data and metadata

    Raises:
        ValueError: If problem_data is invalid
        RuntimeError: If solver fails to converge

    Example:
        >>> agent = SomeAgent()
        >>> result = agent.solve_problem({'type': 'ode', ...})
        >>> solution = result.data['solution']
    """
    ...
```

### Code Organization

**Agent Structure**:
```python
class MyAgent(ComputationalMethodAgent):
    """Brief agent description.

    Detailed description of capabilities.
    """

    def __init__(self, config: Optional[Dict] = None):
        """Initialize agent with configuration."""
        metadata = AgentMetadata(
            name="MyAgent",
            version="1.0.0",
            description="...",
            author="...",
            capabilities=self.get_capabilities(),
            dependencies=[...]
        )
        super().__init__(metadata, config)

    def get_capabilities(self) -> List[Capability]:
        """Return agent capabilities."""
        return [...]

    def primary_method(self, data: Dict[str, Any]) -> AgentResult:
        """Main agent method."""
        ...
```

---

## Testing Requirements

### Unit Tests

**Every new feature requires tests**:

```python
import pytest
from agents.my_agent import MyAgent

def test_basic_functionality():
    """Test basic agent functionality."""
    agent = MyAgent()
    result = agent.some_method({'input': 'data'})

    assert result.success
    assert 'output' in result.data
    assert result.data['output'] > 0

def test_error_handling():
    """Test error handling."""
    agent = MyAgent()
    result = agent.some_method({'invalid': 'input'})

    assert not result.success
    assert len(result.errors) > 0
```

### Test Coverage

- **Minimum**: 80% code coverage
- **Target**: 90% code coverage
- Run coverage: `pytest --cov=agents --cov-report=html`

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_my_agent.py -v

# Run with coverage
pytest tests/ --cov=agents --cov-report=term-missing

# Run specific test
pytest tests/test_my_agent.py::test_basic_functionality -v
```

---

## Documentation

### Code Documentation

1. **Docstrings**: All public functions, classes, methods
2. **Type hints**: All function signatures
3. **Comments**: Complex logic, non-obvious decisions

### User Documentation

When adding features that affect users:

1. Update `docs/GETTING_STARTED.md` if needed
2. Add examples to `examples/` directory
3. Update `docs/API_REFERENCE.md`
4. Add entry to `docs/USER_GUIDE.md`

### Example Template

```python
"""
Example: Descriptive Title

Brief description of what this example demonstrates.

Key concepts:
- Concept 1
- Concept 2
- Concept 3
"""

import numpy as np
import matplotlib.pyplot as plt

# ... example code with clear comments ...

if __name__ == "__main__":
    main()
```

---

## Pull Request Process

### Before Submitting

1. **Code quality**:
   ```bash
   # Format code
   black agents/ tests/

   # Check style
   flake8 agents/ tests/

   # Type checking
   mypy agents/
   ```

2. **Tests pass**:
   ```bash
   pytest tests/ -v
   ```

3. **Documentation updated**:
   - Docstrings complete
   - User docs updated if needed
   - Example added if appropriate

### Creating a Pull Request

1. **Create branch**:
   ```bash
   git checkout -b feature/my-feature
   # or
   git checkout -b fix/issue-123
   ```

2. **Make changes**:
   - Write code
   - Add tests
   - Update documentation

3. **Commit**:
   ```bash
   git add .
   git commit -m "Add feature: brief description"
   ```

   **Commit message format**:
   - `Add feature: description` - New functionality
   - `Fix: description` - Bug fix
   - `Update: description` - Improvement to existing feature
   - `Docs: description` - Documentation only
   - `Test: description` - Test additions/fixes

4. **Push and create PR**:
   ```bash
   git push origin feature/my-feature
   ```

   Then create pull request on GitHub.

### PR Description Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Code refactoring

## Testing
- [ ] All tests pass
- [ ] New tests added
- [ ] Coverage maintained/improved

## Documentation
- [ ] Docstrings updated
- [ ] User documentation updated
- [ ] Example added (if appropriate)

## Checklist
- [ ] Code follows project style
- [ ] Self-reviewed code
- [ ] Tests pass locally
- [ ] Documentation updated
```

### Review Process

1. **Automated checks**: CI must pass
2. **Code review**: At least one approval required
3. **Testing**: Reviewer may test changes
4. **Merge**: Squash and merge after approval

---

## Issue Reporting

### Bug Reports

Use this template:

```markdown
**Description**
Clear description of the bug

**To Reproduce**
Steps to reproduce:
1. Import X
2. Call method Y
3. See error

**Expected Behavior**
What should happen

**Actual Behavior**
What actually happens

**Environment**
- Python version:
- OS:
- Package versions: (output of `pip list`)

**Additional Context**
Any other relevant information
```

### Feature Requests

```markdown
**Feature Description**
What feature would you like to see?

**Use Case**
Why is this feature needed?

**Proposed Solution**
How might this work?

**Alternatives**
Other approaches considered

**Additional Context**
Any other relevant information
```

---

## Development Guidelines

### Adding a New Agent

1. **Create agent file**: `agents/my_agent.py`
2. **Inherit from base**: `ComputationalMethodAgent`
3. **Implement required methods**:
   - `__init__()`
   - `get_capabilities()`
   - Primary computation method
4. **Add tests**: `tests/test_my_agent.py`
5. **Add example**: `examples/example_my_agent.py`
6. **Update documentation**

### Adding a New Feature

1. **Write tests first** (TDD approach)
2. **Implement feature**
3. **Update documentation**
4. **Add example if significant**
5. **Submit PR**

### Optimization Guidelines

Before optimizing:
1. **Profile first**: Use `PerformanceProfilerAgent`
2. **Measure baseline**: Record current performance
3. **Optimize**: Make changes
4. **Measure improvement**: Quantify speedup
5. **Document**: Update `docs/OPTIMIZATION_GUIDE.md`

See `docs/OPTIMIZATION_GUIDE.md` for detailed optimization strategies.

---

## Code of Conduct

### Our Standards

- **Respectful**: Treat everyone with respect
- **Collaborative**: Work together constructively
- **Inclusive**: Welcome diverse perspectives
- **Professional**: Maintain professional conduct

### Reporting Issues

Report code of conduct violations to: [email@example.com]

---

## Recognition

Contributors are recognized in:
- `CONTRIBUTORS.md` file
- Release notes
- Project documentation

---

## Questions?

- **Documentation**: Check `docs/` directory
- **Examples**: Browse `examples/` directory
- **Issues**: Search existing GitHub issues
- **Contact**: Open a discussion on GitHub

---

## Summary

**Quick Checklist**:
- [ ] Fork and clone repository
- [ ] Create feature branch
- [ ] Write code following style guide
- [ ] Add tests (80%+ coverage)
- [ ] Update documentation
- [ ] Run tests locally
- [ ] Create pull request
- [ ] Respond to review feedback

Thank you for contributing to Scientific Computing Agents! 🚀

---

**Last Updated**: 2025-09-30
**Version**: 1.0
