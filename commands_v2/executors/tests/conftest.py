#!/usr/bin/env python3
"""
Pytest Configuration and Shared Fixtures
========================================

Shared test fixtures for command executor framework testing.

Fixtures:
- temp_workspace: Temporary workspace for testing
- mock_git_repo: Mock git repository
- sample_python_project: Sample Python codebase
- sample_julia_project: Sample Julia codebase
- sample_jax_project: Sample JAX/ML codebase
- backup_system: Configured backup system
- dry_run_executor: Configured dry-run executor
- agent_orchestrator: Configured agent orchestrator
- cache_manager: Configured cache manager
"""

import sys
import os
import shutil
import tempfile
import json
from pathlib import Path
from typing import Dict, Any, List, Generator
from datetime import datetime

import pytest

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from executors.framework import (
    BaseCommandExecutor,
    ExecutionContext,
    ExecutionResult,
    ExecutionPhase,
    AgentType,
    CommandCategory,
    ValidationEngine,
    CacheManager,
    AgentOrchestrator,
)
from executors.safety_manager import (
    BackupSystem,
    DryRunExecutor,
    RollbackManager,
    ValidationPipeline,
    ChangeType,
    RiskLevel,
)
from executors.performance import PerformanceMonitor, ParallelExecutor


# ============================================================================
# Pytest Configuration Hooks
# ============================================================================

def pytest_configure(config):
    """Configure pytest with custom markers"""
    config.addinivalue_line(
        "markers", "unit: Unit tests for individual components"
    )
    config.addinivalue_line(
        "markers", "integration: Integration tests for full workflows"
    )
    config.addinivalue_line(
        "markers", "workflow: Real-world workflow tests"
    )
    config.addinivalue_line(
        "markers", "performance: Performance and benchmark tests"
    )
    config.addinivalue_line(
        "markers", "slow: Tests that take significant time"
    )
    config.addinivalue_line(
        "markers", "fast: Fast tests that can run frequently"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers automatically"""
    for item in items:
        # Add markers based on test location
        if "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        elif "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
            item.add_marker(pytest.mark.fast)
        elif "workflow" in str(item.fspath):
            item.add_marker(pytest.mark.workflow)
            item.add_marker(pytest.mark.slow)
        elif "performance" in str(item.fspath):
            item.add_marker(pytest.mark.performance)
            item.add_marker(pytest.mark.slow)


# ============================================================================
# Workspace Fixtures
# ============================================================================

@pytest.fixture
def temp_workspace() -> Generator[Path, None, None]:
    """
    Create temporary workspace for testing.

    Yields:
        Path to temporary directory
    """
    temp_dir = tempfile.mkdtemp(prefix="executor_test_")
    workspace = Path(temp_dir)

    try:
        yield workspace
    finally:
        # Cleanup
        if workspace.exists():
            shutil.rmtree(workspace, ignore_errors=True)


@pytest.fixture
def mock_git_repo(temp_workspace: Path) -> Path:
    """
    Create mock git repository for testing.

    Args:
        temp_workspace: Temporary workspace

    Returns:
        Path to git repository
    """
    git_dir = temp_workspace / "mock_repo"
    git_dir.mkdir(parents=True)

    # Initialize git
    (git_dir / ".git").mkdir()
    (git_dir / ".git" / "config").write_text("[core]\n\trepositoryformatversion = 0\n")

    # Create some files
    (git_dir / "README.md").write_text("# Mock Repository\n\nFor testing.")
    (git_dir / "src").mkdir()
    (git_dir / "src" / "__init__.py").write_text("")
    (git_dir / "src" / "main.py").write_text(
        "def main():\n    print('Hello, World!')\n\nif __name__ == '__main__':\n    main()\n"
    )

    # Create tests directory
    (git_dir / "tests").mkdir()
    (git_dir / "tests" / "__init__.py").write_text("")
    (git_dir / "tests" / "test_main.py").write_text(
        "def test_main():\n    assert True\n"
    )

    return git_dir


# ============================================================================
# Sample Project Fixtures
# ============================================================================

@pytest.fixture
def sample_python_project(temp_workspace: Path) -> Path:
    """
    Create sample Python project for testing.

    Args:
        temp_workspace: Temporary workspace

    Returns:
        Path to Python project
    """
    project = temp_workspace / "python_project"
    project.mkdir(parents=True)

    # Create package structure
    (project / "mypackage").mkdir()
    (project / "mypackage" / "__init__.py").write_text(
        '__version__ = "0.1.0"\n'
    )
    (project / "mypackage" / "core.py").write_text(
        '''"""Core module"""

def add(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b

def multiply(a: int, b: int) -> int:
    """Multiply two numbers"""
    return a * b

class Calculator:
    """Simple calculator class"""

    def __init__(self):
        self.history = []

    def calculate(self, operation: str, a: float, b: float) -> float:
        """Perform calculation"""
        if operation == "add":
            result = a + b
        elif operation == "subtract":
            result = a - b
        elif operation == "multiply":
            result = a * b
        elif operation == "divide":
            if b == 0:
                raise ValueError("Cannot divide by zero")
            result = a / b
        else:
            raise ValueError(f"Unknown operation: {operation}")

        self.history.append((operation, a, b, result))
        return result
'''
    )

    # Create utils module
    (project / "mypackage" / "utils.py").write_text(
        '''"""Utility functions"""

import logging

logger = logging.getLogger(__name__)

def format_result(value: float, precision: int = 2) -> str:
    """Format result with precision"""
    return f"{value:.{precision}f}"

def validate_input(value: str) -> bool:
    """Validate input string"""
    try:
        float(value)
        return True
    except ValueError:
        return False
'''
    )

    # Create tests
    (project / "tests").mkdir()
    (project / "tests" / "__init__.py").write_text("")
    (project / "tests" / "test_core.py").write_text(
        '''"""Tests for core module"""

import pytest
from mypackage.core import add, multiply, Calculator

def test_add():
    assert add(2, 3) == 5

def test_multiply():
    assert multiply(2, 3) == 6

def test_calculator():
    calc = Calculator()
    assert calc.calculate("add", 2, 3) == 5
    assert calc.calculate("multiply", 2, 3) == 6

def test_calculator_divide_by_zero():
    calc = Calculator()
    with pytest.raises(ValueError):
        calc.calculate("divide", 1, 0)
'''
    )

    # Create setup.py
    (project / "setup.py").write_text(
        '''from setuptools import setup, find_packages

setup(
    name="mypackage",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[],
    extras_require={
        "dev": ["pytest>=7.0.0", "black>=23.0.0", "mypy>=1.0.0"]
    }
)
'''
    )

    # Create requirements.txt
    (project / "requirements.txt").write_text(
        "numpy>=1.20.0\nscipy>=1.7.0\n"
    )

    # Create pyproject.toml
    (project / "pyproject.toml").write_text(
        '''[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"

[project]
name = "mypackage"
version = "0.1.0"
description = "Sample Python package"
authors = [{name = "Test Author", email = "test@example.com"}]
'''
    )

    return project


@pytest.fixture
def sample_julia_project(temp_workspace: Path) -> Path:
    """
    Create sample Julia project for testing.

    Args:
        temp_workspace: Temporary workspace

    Returns:
        Path to Julia project
    """
    project = temp_workspace / "julia_project"
    project.mkdir(parents=True)

    # Create src directory
    (project / "src").mkdir()
    (project / "src" / "MyPackage.jl").write_text(
        '''module MyPackage

export greet, calculate

function greet(name::String)
    println("Hello, $name!")
end

function calculate(x::Float64, y::Float64)
    return x + y
end

end # module
'''
    )

    # Create test directory
    (project / "test").mkdir()
    (project / "test" / "runtests.jl").write_text(
        '''using Test
using MyPackage

@testset "MyPackage Tests" begin
    @test calculate(2.0, 3.0) == 5.0
    @test calculate(10.0, -5.0) == 5.0
end
'''
    )

    # Create Project.toml
    (project / "Project.toml").write_text(
        '''name = "MyPackage"
uuid = "12345678-1234-1234-1234-123456789012"
authors = ["Test Author <test@example.com>"]
version = "0.1.0"
'''
    )

    return project


@pytest.fixture
def sample_jax_project(temp_workspace: Path) -> Path:
    """
    Create sample JAX/ML project for testing.

    Args:
        temp_workspace: Temporary workspace

    Returns:
        Path to JAX project
    """
    project = temp_workspace / "jax_project"
    project.mkdir(parents=True)

    # Create model module
    (project / "model.py").write_text(
        '''"""Simple JAX neural network model"""

import jax
import jax.numpy as jnp
from jax import grad, jit, vmap

def relu(x):
    return jnp.maximum(0, x)

def predict(params, x):
    """Forward pass"""
    W1, b1, W2, b2 = params
    hidden = relu(jnp.dot(x, W1) + b1)
    return jnp.dot(hidden, W2) + b2

def loss(params, x, y):
    """MSE loss"""
    pred = predict(params, x)
    return jnp.mean((pred - y) ** 2)

def update(params, x, y, lr=0.01):
    """Gradient descent step"""
    grads = grad(loss)(params, x, y)
    return [(p - lr * g) for p, g in zip(params, grads)]
'''
    )

    # Create training script
    (project / "train.py").write_text(
        '''"""Training script"""

import jax.numpy as jnp
from model import loss, update, predict

def train(params, X_train, y_train, epochs=100):
    """Train the model"""
    for epoch in range(epochs):
        params = update(params, X_train, y_train)
        if epoch % 10 == 0:
            l = loss(params, X_train, y_train)
            print(f"Epoch {epoch}, Loss: {l}")
    return params
'''
    )

    # Create requirements
    (project / "requirements.txt").write_text(
        "jax>=0.4.0\njaxlib>=0.4.0\noptax>=0.1.0\nflax>=0.7.0\n"
    )

    return project


# ============================================================================
# Component Fixtures
# ============================================================================

@pytest.fixture
def execution_context(temp_workspace: Path) -> ExecutionContext:
    """
    Create execution context for testing.

    Args:
        temp_workspace: Temporary workspace

    Returns:
        ExecutionContext instance
    """
    return ExecutionContext(
        command_name="test_command",
        work_dir=temp_workspace,
        args={"test_arg": "test_value"},
        dry_run=False,
        interactive=False,
        parallel=False,
        agents=[AgentType.AUTO]
    )


@pytest.fixture
def backup_system(temp_workspace: Path) -> BackupSystem:
    """
    Create configured backup system.

    Args:
        temp_workspace: Temporary workspace

    Returns:
        BackupSystem instance
    """
    backup_root = temp_workspace / "backups"
    return BackupSystem(backup_root=backup_root)


@pytest.fixture
def dry_run_executor() -> DryRunExecutor:
    """
    Create dry-run executor.

    Returns:
        DryRunExecutor instance
    """
    return DryRunExecutor()


@pytest.fixture
def agent_orchestrator() -> AgentOrchestrator:
    """
    Create agent orchestrator.

    Returns:
        AgentOrchestrator instance
    """
    return AgentOrchestrator()


@pytest.fixture
def cache_manager(temp_workspace: Path) -> CacheManager:
    """
    Create cache manager.

    Args:
        temp_workspace: Temporary workspace

    Returns:
        CacheManager instance
    """
    cache_dir = temp_workspace / "cache"
    return CacheManager(cache_dir=cache_dir)


@pytest.fixture
def validation_engine() -> ValidationEngine:
    """
    Create validation engine.

    Returns:
        ValidationEngine instance
    """
    return ValidationEngine()


@pytest.fixture
def validation_pipeline() -> ValidationPipeline:
    """
    Create validation pipeline.

    Returns:
        ValidationPipeline instance
    """
    return ValidationPipeline()


@pytest.fixture
def rollback_manager(backup_system: BackupSystem) -> RollbackManager:
    """
    Create rollback manager.

    Args:
        backup_system: Backup system

    Returns:
        RollbackManager instance
    """
    return RollbackManager(backup_system=backup_system)


# ============================================================================
# Mock Fixtures
# ============================================================================

@pytest.fixture
def mock_claude_client():
    """Mock Claude client for testing"""
    class MockClaudeClient:
        def __init__(self):
            self.call_count = 0

        def send_message(self, message: str) -> str:
            self.call_count += 1
            return f"Mock response to: {message}"

        def analyze_code(self, code: str) -> Dict[str, Any]:
            return {
                "issues": [],
                "suggestions": ["Consider adding type hints"],
                "metrics": {"complexity": 5, "quality_score": 85}
            }

    return MockClaudeClient()


@pytest.fixture
def mock_git_operations():
    """Mock git operations for testing"""
    class MockGit:
        def __init__(self):
            self.commits = []
            self.staged_files = []

        def add(self, files: List[str]):
            self.staged_files.extend(files)

        def commit(self, message: str) -> str:
            commit_hash = f"mock_{len(self.commits)}"
            self.commits.append({"hash": commit_hash, "message": message})
            self.staged_files = []
            return commit_hash

        def status(self) -> Dict[str, Any]:
            return {
                "modified": [],
                "staged": self.staged_files,
                "untracked": []
            }

    return MockGit()


# ============================================================================
# Performance Fixtures
# ============================================================================

@pytest.fixture
def performance_monitor() -> PerformanceMonitor:
    """
    Create performance monitor.

    Returns:
        PerformanceMonitor instance
    """
    return PerformanceMonitor()


@pytest.fixture
def parallel_executor() -> ParallelExecutor:
    """
    Create parallel executor.

    Returns:
        ParallelExecutor instance
    """
    return ParallelExecutor(max_workers=4)


# ============================================================================
# Data Fixtures
# ============================================================================

@pytest.fixture
def sample_code_with_issues() -> str:
    """Sample Python code with quality issues"""
    return '''
def calculate(x,y,z):
    if x==0:
        return y+z
    elif x==1:
        return y*z
    else:
        result=x+y+z
        return result

def process_data(data):
    results=[]
    for item in data:
        if item>0:
            results.append(item*2)
    return results
'''


@pytest.fixture
def sample_optimizable_code() -> str:
    """Sample code that can be optimized"""
    return '''
import numpy as np

def slow_operation(data):
    result = []
    for i in range(len(data)):
        for j in range(len(data)):
            result.append(data[i] * data[j])
    return result

def inefficient_loop(arr):
    total = 0
    for i in range(len(arr)):
        total += arr[i]
    return total
'''


@pytest.fixture
def sample_test_data() -> Dict[str, Any]:
    """Sample test data"""
    return {
        "users": [
            {"id": 1, "name": "Alice", "age": 30},
            {"id": 2, "name": "Bob", "age": 25},
            {"id": 3, "name": "Charlie", "age": 35}
        ],
        "metrics": {
            "performance": 0.85,
            "quality": 0.92,
            "coverage": 0.88
        }
    }


# ============================================================================
# Utility Functions
# ============================================================================

def create_file(path: Path, content: str):
    """Helper to create file with content"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def assert_file_exists(path: Path, message: str = ""):
    """Assert that file exists"""
    assert path.exists(), message or f"File does not exist: {path}"


def assert_file_contains(path: Path, text: str, message: str = ""):
    """Assert that file contains text"""
    assert_file_exists(path)
    content = path.read_text()
    assert text in content, message or f"File does not contain '{text}': {path}"


def count_files_recursive(directory: Path, pattern: str = "*") -> int:
    """Count files recursively"""
    return len(list(directory.rglob(pattern)))