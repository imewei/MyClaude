# Contributing to Claude Code Command Executor Framework

> Guidelines for contributing to the project

Thank you for your interest in contributing! This document provides guidelines and instructions for contributing.

---

## Table of Contents

1. [Code of Conduct](#code-of-conduct)
2. [Getting Started](#getting-started)
3. [How to Contribute](#how-to-contribute)
4. [Development Setup](#development-setup)
5. [Coding Standards](#coding-standards)
6. [Testing Requirements](#testing-requirements)
7. [Pull Request Process](#pull-request-process)
8. [Community](#community)

---

## Code of Conduct

### Our Pledge

We pledge to make participation in our project a harassment-free experience for everyone, regardless of age, body size, disability, ethnicity, gender identity and expression, level of experience, nationality, personal appearance, race, religion, or sexual identity and orientation.

### Our Standards

**Positive behavior includes:**
- Using welcoming and inclusive language
- Being respectful of differing viewpoints
- Gracefully accepting constructive criticism
- Focusing on what is best for the community
- Showing empathy towards other community members

**Unacceptable behavior includes:**
- Trolling, insulting/derogatory comments, and personal attacks
- Public or private harassment
- Publishing others' private information
- Other conduct which could reasonably be considered inappropriate

### Enforcement

Instances of abusive, harassing, or otherwise unacceptable behavior may be reported by contacting the project team. All complaints will be reviewed and investigated promptly and fairly.

---

## Getting Started

### Ways to Contribute

- **Bug Reports**: Report bugs you encounter
- **Feature Requests**: Suggest new features
- **Code**: Submit bug fixes or new features
- **Documentation**: Improve or add documentation
- **Tutorials**: Create tutorials or guides
- **Plugins**: Develop plugins
- **Testing**: Help test new features
- **Community**: Help others in discussions

---

## How to Contribute

### Reporting Bugs

**Before submitting a bug report:**
1. Check existing issues to avoid duplicates
2. Verify the bug in the latest version
3. Collect diagnostic information

**Bug report should include:**
- Clear, descriptive title
- Steps to reproduce
- Expected vs actual behavior
- System information
- Error messages/logs
- Screenshots if applicable

**Template:**
```markdown
## Bug Description
Clear description of the bug

## Steps to Reproduce
1. Step 1
2. Step 2
3. ...

## Expected Behavior
What should happen

## Actual Behavior
What actually happens

## Environment
- OS: [e.g., macOS 14.0]
- Python Version: [e.g., 3.11]
- Framework Version: [e.g., 1.0.0]

## Additional Context
Any other relevant information

## Diagnostics
```bash
claude-commands diagnostics
```
```

### Suggesting Features

**Feature request should include:**
- Clear, descriptive title
- Use case / problem it solves
- Proposed solution
- Alternative solutions considered
- Additional context

**Template:**
```markdown
## Feature Description
Clear description of the feature

## Problem / Use Case
What problem does this solve?

## Proposed Solution
How should it work?

## Alternatives Considered
Other solutions you've considered

## Additional Context
Any other relevant information
```

---

## Development Setup

### Prerequisites

```bash
# Python 3.9+
python --version

# Git
git --version

# pip
pip --version
```

### Fork and Clone

```bash
# Fork on GitHub, then clone your fork
git clone https://github.com/YOUR-USERNAME/claude-commands.git
cd claude-commands

# Add upstream remote
git remote add upstream https://github.com/anthropics/claude-commands.git
```

### Install Dependencies

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install

# Run tests to verify setup
pytest
```

### Project Structure

```
claude-commands/
├── executors/          # Command executors
├── ai_features/        # Agents and AI
├── workflows/          # Workflow engine
├── plugins/            # Plugin system
├── ux/                 # User experience
├── tests/              # Test suites
├── docs/               # Documentation
└── final/              # Phase 7 documentation
```

---

## Coding Standards

### Python Style Guide

Follow PEP 8 with these specifics:

**Line Length**: 100 characters (not 79)

**Type Hints**: Required for all functions
```python
def process_data(items: List[str], count: int = 10) -> Dict[str, Any]:
    """Process items and return results."""
    pass
```

**Docstrings**: Required for all public classes and functions
```python
def analyze_code(file_path: str, options: Dict[str, Any]) -> Analysis:
    """
    Analyze code quality.

    Args:
        file_path: Path to file to analyze
        options: Analysis options

    Returns:
        Analysis: Analysis results

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If options are invalid
    """
    pass
```

**Imports**: Organized and typed
```python
# Standard library
import os
import sys
from pathlib import Path

# Third-party
import pytest
from rich.console import Console

# Local
from claude_commands.executor import BaseExecutor
from claude_commands.agents import BaseAgent
```

### Code Quality

Run quality checks before committing:

```bash
# Auto-fix code style
/check-code-quality --auto-fix

# Type checking
mypy claude_commands

# Linting
pylint claude_commands

# Security scanning
bandit -r claude_commands
```

### Commit Messages

Follow Conventional Commits format:

```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `style`: Formatting
- `refactor`: Code refactoring
- `test`: Tests
- `chore`: Maintenance

**Examples:**
```
feat(agents): add quantum computing agent

fix(executor): resolve parallel execution deadlock

docs(tutorial): add performance optimization tutorial

test(workflow): add workflow engine integration tests
```

---

## Testing Requirements

### Test Coverage

- **Minimum**: 80% overall
- **Target**: 90%+
- **Required**: 100% for critical paths

### Writing Tests

```python
# tests/unit/test_feature.py
import pytest
from claude_commands.feature import Feature

class TestFeature:
    """Test feature functionality"""

    @pytest.fixture
    def feature(self):
        """Feature fixture"""
        return Feature()

    def test_basic_functionality(self, feature):
        """Test basic functionality"""
        result = feature.process("input")
        assert result == "expected"

    def test_edge_case(self, feature):
        """Test edge case"""
        with pytest.raises(ValueError):
            feature.process(None)
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=claude_commands --cov-report=html

# Run specific test
pytest tests/unit/test_feature.py::TestFeature::test_basic_functionality

# Run with verbose output
pytest -v

# Run with debug output
pytest -vv --pdb
```

### Test Types

**Unit Tests**: Test individual functions/classes
```bash
pytest tests/unit/
```

**Integration Tests**: Test component integration
```bash
pytest tests/integration/
```

**End-to-End Tests**: Test complete workflows
```bash
pytest tests/e2e/
```

---

## Pull Request Process

### Before Submitting

1. **Create Branch**
```bash
# Update main
git checkout main
git pull upstream main

# Create feature branch
git checkout -b feature/your-feature-name
```

2. **Make Changes**
- Follow coding standards
- Add tests
- Update documentation
- Run quality checks

3. **Test**
```bash
# Run tests
pytest

# Check coverage
pytest --cov --cov-report=html

# Quality check
/check-code-quality --auto-fix

# Verify all tests pass
/run-all-tests
```

4. **Commit**
```bash
# Stage changes
git add .

# Commit with conventional message
git commit -m "feat(scope): description"

# Or use smart commit
/commit --ai-message --validate
```

### Submitting PR

1. **Push to Fork**
```bash
git push origin feature/your-feature-name
```

2. **Create Pull Request**
- Go to GitHub
- Click "New Pull Request"
- Select your branch
- Fill out PR template

**PR Template:**
```markdown
## Description
Clear description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation
- [ ] Performance improvement
- [ ] Refactoring

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests added/updated
- [ ] All tests passing
- [ ] Coverage ≥ 80%

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No breaking changes (or documented)
- [ ] Changelog updated

## Related Issues
Fixes #123
```

### Code Review Process

**Reviews check for:**
- Code quality and style
- Test coverage
- Documentation
- Performance impact
- Security implications
- Breaking changes

**Response time:**
- Initial review: Within 2 business days
- Follow-up: Within 1 business day

**Approval requirements:**
- At least 1 approval from maintainer
- All CI checks passing
- No unresolved comments

### After Review

**Address feedback:**
```bash
# Make changes
# Commit
git add .
git commit -m "fix: address review comments"
git push origin feature/your-feature-name
```

**Keep updated:**
```bash
# Rebase on main
git fetch upstream
git rebase upstream/main
git push origin feature/your-feature-name --force
```

---

## Development Guidelines

### Adding New Commands

1. **Create executor**
```python
# executors/implementations/my_command.py
from claude_commands.executor import BaseExecutor

class MyCommandExecutor(BaseExecutor):
    def execute(self, args):
        # Implementation
        pass
```

2. **Register command**
```python
# Register in command registry
CommandRegistry.register(
    name="my-command",
    executor=MyCommandExecutor,
    metadata=CommandMetadata(...)
)
```

3. **Add documentation**
```markdown
# my-command.md
Complete command documentation
```

4. **Add tests**
```python
# tests/unit/test_my_command.py
Test implementation
```

### Adding New Agents

1. **Create agent**
```python
# ai_features/agents/my_agent.py
from claude_commands.agents import BaseAgent

class MyAgent(BaseAgent):
    name = "MyAgent"
    expertise = ["domain"]

    def analyze(self, context):
        # Implementation
        pass
```

2. **Register agent**
```python
AGENT_REGISTRY["my_category"].append(MyAgent)
```

3. **Add tests**

### Adding Workflows

1. **Create YAML**
```yaml
# workflows/definitions/my-workflow.yml
name: My Workflow
steps:
  - name: step1
    command: some-command
```

2. **Test workflow**
```bash
# Test execution
/multi-agent-optimize --workflow=my-workflow
```

---

## Community

### Communication Channels

- **GitHub Issues**: Bug reports, features
- **GitHub Discussions**: Q&A, ideas
- **Documentation**: Guides and references

### Getting Help

1. Check documentation
2. Search existing issues
3. Ask in discussions
4. Create new issue if needed

### Recognition

Contributors are recognized in:
- CHANGELOG.md
- Repository contributors page
- Release notes

---

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

## Questions?

Contact the project maintainers or open a discussion on GitHub.

---

**Thank you for contributing!**

**Version**: 1.0.0 | **Last Updated**: September 2025