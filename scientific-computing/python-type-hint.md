---
description: Add comprehensive type hints to Python/JAX code using typing and jax.typing for mypy compatibility with intelligent 23-agent optimization
category: python-scientific
argument-hint: "[--strict] [--jax-arrays] [--scientific] [--target=file|directory] [--agents=auto|python|scientific|ai|quality|all] [--orchestrate] [--intelligent] [--breakthrough] [--optimize] [--comprehensive]"
allowed-tools: "*"
model: inherit
tags: python, type-hints, mypy, jax, scientific-computing, type-safety
---

# Python Type Hints for Scientific Computing

Add comprehensive type hints to Python/JAX code for improved type safety and mypy compatibility.

## Quick Start

```bash
# Add basic type hints with intelligent agents
/python-type-hint --agents=auto --intelligent

# Strict typing with JAX arrays and agent orchestration
/python-type-hint --strict --jax-arrays --agents=python --orchestrate

# Scientific computing patterns with specialized agents
/python-type-hint --scientific --jax-arrays --agents=scientific --breakthrough

# Comprehensive type hinting with agent optimization
/python-type-hint --agents=all --comprehensive --optimize --intelligent
```

Adds comprehensive type hints to Python/JAX code with intelligent 23-agent optimization for type safety and mypy compatibility.

## Options

| Option | Description |
|--------|-------------|
| `--strict` | Enable strict mypy configuration and advanced type checking |
| `--jax-arrays` | Include JAX-specific array types and typing patterns |
| `--scientific` | Add scientific computing type patterns (NumPy, SciPy, etc.) |
| `--target=<path>` | Target specific file or directory for type hint addition |
| `--agents=<agents>` | Agent selection (auto, python, scientific, ai, quality, all) |
| `--orchestrate` | Enable advanced 23-agent orchestration with type hint intelligence |
| `--intelligent` | Enable intelligent agent selection based on Python typing analysis |
| `--breakthrough` | Enable breakthrough type safety optimization |
| `--optimize` | Apply type hint optimization with agent coordination |
| `--comprehensive` | Enable comprehensive type annotation with agent intelligence |

## Core Features

- **Function Signatures**: Add comprehensive type hints to all function parameters and return types
- **JAX Integration**: Use `jax.typing` for JAX-specific array types and shapes
- **Mypy Compatibility**: Ensure all type hints work with mypy static type checker
- **Scientific Patterns**: Support NumPy, SciPy, and scientific computing type patterns
- **Strict Mode**: Advanced type checking with strict mypy configuration
- **23-Agent Type Intelligence**: Multi-agent collaboration for optimal type hint implementation
- **Advanced Type Safety**: Agent-driven type safety analysis and optimization
- **Intelligent Mypy Integration**: Agent-coordinated mypy configuration and validation

## 23-Agent Intelligent Type Hinting System

### Intelligent Agent Selection (`--intelligent`)
**Auto-Selection Algorithm**: Analyzes Python type hinting requirements, code patterns, and type safety goals to automatically choose optimal agent combinations from the 23-agent library.

```bash
# Python Type Hinting Pattern Detection → Agent Selection
- Research Computing → research-intelligence-master + python-expert + code-quality-expert
- Production Systems → ai-systems-architect + python-expert + systems-architect
- Scientific Computing → scientific-computing-master + python-expert + code-quality-expert
- ML/AI Development → neural-networks-master + python-expert + jax-pro
- Code Quality Focus → code-quality-expert + python-expert + documentation-architect
```

### Core Python Type Hinting Agents

#### **`python-expert`** - Python Ecosystem Type Expert
- **Python Typing**: Deep expertise in Python typing system and advanced type annotations
- **JAX Integration**: JAX-specific type patterns with jax.typing and array types
- **Mypy Mastery**: Advanced mypy configuration and static type checking optimization
- **Package Ecosystem**: Python typing package coordination and best practices
- **Cross-Platform Compatibility**: Multi-platform Python type safety optimization

#### **`code-quality-expert`** - Code Quality & Type Safety
- **Type Safety Analysis**: Advanced type safety analysis and validation
- **Code Quality Standards**: Type hint quality standards and best practices
- **Static Analysis Integration**: Integration with mypy, pylint, and other static analyzers
- **Documentation Quality**: Type annotation documentation and readability
- **Maintainability Focus**: Long-term type safety and code maintainability

#### **`scientific-computing-master`** - Scientific Python Typing
- **Scientific Typing**: Scientific computing type patterns in Python/JAX/NumPy
- **Numerical Methods**: Advanced numerical method type annotations
- **Research Applications**: Academic and research-grade type safety
- **Domain Integration**: Cross-domain scientific computing type patterns
- **Mathematical Foundation**: Mathematical type safety and algorithmic typing

#### **`jax-pro`** - JAX Type Specialist
- **JAX Typing**: Advanced JAX-specific type annotations and patterns
- **Device Types**: GPU/TPU device-specific type annotations
- **Transformation Types**: JAX transformation type safety (jit, vmap, grad)
- **Array Types**: JAX array type systems and shape annotations
- **Performance Integration**: Type hints that enhance JAX performance

### Advanced Agent Selection Strategies

#### **`auto`** - Intelligent Agent Selection for Type Hinting
Automatically analyzes Python type hinting requirements and selects optimal agent combinations:
- **Code Pattern Analysis**: Detects Python coding patterns and type hint opportunities
- **Complexity Assessment**: Evaluates type complexity and annotation requirements
- **Agent Matching**: Maps type hinting needs to relevant agent expertise
- **Quality Balance**: Balances type safety with code readability and maintainability

#### **`python`** - Python-Specialized Type Hinting Team
- `python-expert` (Python typing lead)
- `code-quality-expert` (quality assurance)
- `jax-pro` (JAX integration)
- `scientific-computing-master` (scientific applications)

#### **`scientific`** - Scientific Computing Type Team
- `scientific-computing-master` (lead)
- `python-expert` (Python implementation)
- `code-quality-expert` (quality standards)
- `research-intelligence-master` (research methodology)

#### **`ai`** - AI/ML Python Type Team
- `neural-networks-master` (lead)
- `python-expert` (Python optimization)
- `jax-pro` (JAX type patterns)
- `ai-systems-architect` (production systems)

#### **`quality`** - Code Quality-Focused Type Team
- `code-quality-expert` (lead)
- `python-expert` (Python typing)
- `documentation-architect` (documentation)
- `systems-architect` (system integration)

#### **`all`** - Complete 23-Agent Type Hinting Ecosystem
Activates all relevant agents with intelligent orchestration for breakthrough type safety implementation.

### Advanced 23-Agent Type Hinting Examples

```bash
# Intelligent auto-selection for type hinting
/python-type-hint --agents=auto --intelligent --optimize

# Scientific computing type hinting with specialized agents
/python-type-hint --agents=scientific --breakthrough --orchestrate --scientific

# Production system type safety optimization
/python-type-hint --agents=quality --optimize --comprehensive --strict

# AI/ML type hinting development
/python-type-hint --agents=ai --breakthrough --orchestrate --jax-arrays

# Research-grade type safety analysis
/python-type-hint --agents=all --breakthrough --intelligent --comprehensive

# Complete 23-agent type hinting ecosystem
/python-type-hint --agents=all --orchestrate --breakthrough --intelligent --optimize
```

## Basic Type Hints Setup

```python
from typing import (
    Tuple, Optional, Union, List, Dict, Any, Callable, TypeVar,
    Generic, Protocol, overload, cast
)
from jax.typing import ArrayLike
import jax.numpy as jnp
import jax
import numpy as np
from numpy.typing import NDArray

# Type aliases for clarity
Array = jnp.ndarray
Float = Union[float, jnp.ndarray]
Int = Union[int, jnp.ndarray]

# Basic function with type hints
def linear_function(x: Array, slope: float, intercept: float) -> Array:
    """Linear function with type annotations."""
    return slope * x + intercept

# Function with optional parameters
def normalize_data(
    data: Array,
    axis: Optional[int] = None,
    keepdims: bool = False
) -> Array:
    """Normalize data along specified axis."""
    mean = jnp.mean(data, axis=axis, keepdims=keepdims)
    std = jnp.std(data, axis=axis, keepdims=keepdims)
    return (data - mean) / std

# Function returning multiple values
def train_test_split(
    X: Array,
    y: Array,
    test_size: float = 0.2
) -> Tuple[Array, Array, Array, Array]:
    """Split data into train and test sets."""
    n_samples = X.shape[0]
    n_test = int(n_samples * test_size)

    indices = jnp.arange(n_samples)
    test_indices = indices[:n_test]
    train_indices = indices[n_test:]

    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]
```

## JAX-Specific Type Hints

```python
import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike, DTypeLike
from typing import Union, Optional, Callable

# JAX array types
PRNGKey = jax.Array  # For random keys
Params = Dict[str, Array]  # Model parameters

# JAX transformation type hints
def apply_jit(func: Callable[[Array], Array]) -> Callable[[Array], Array]:
    """Apply JIT compilation to function."""
    return jax.jit(func)

def compute_gradient(
    func: Callable[[Array], Array],
    x: Array
) -> Array:
    """Compute gradient using JAX."""
    grad_func = jax.grad(func)
    return grad_func(x)

# Vectorized functions
def batch_function(
    func: Callable[[Array], Array]
) -> Callable[[Array], Array]:
    """Vectorize function over batch dimension."""
    return jax.vmap(func, in_axes=0, out_axes=0)

# PRNG key handling
def random_normal(
    key: PRNGKey,
    shape: Tuple[int, ...],
    dtype: DTypeLike = jnp.float32
) -> Array:
    """Generate random normal array."""
    return jax.random.normal(key, shape, dtype=dtype)

def split_key(key: PRNGKey, num: int = 2) -> Array:
    """Split PRNG key for multiple random operations."""
    return jax.random.split(key, num)

# Parameter initialization
def init_linear_params(
    key: PRNGKey,
    input_dim: int,
    output_dim: int
) -> Dict[str, Array]:
    """Initialize linear layer parameters."""
    k1, k2 = jax.random.split(key)

    return {
        'weights': jax.random.normal(k1, (output_dim, input_dim)) * 0.1,
        'bias': jax.random.normal(k2, (output_dim,)) * 0.1
    }
```

## Generic Types and Protocols

```python
from typing import TypeVar, Generic, Protocol, runtime_checkable
from abc import abstractmethod

# Generic type variables
T = TypeVar('T')
ArrayT = TypeVar('ArrayT', bound=Array)
NumericT = TypeVar('NumericT', int, float, complex)

# Generic container for scientific data
class DataContainer(Generic[T]):
    def __init__(self, data: T, metadata: Dict[str, Any]) -> None:
        self.data = data
        self.metadata = metadata

    def get_data(self) -> T:
        return self.data

    def get_metadata(self) -> Dict[str, Any]:
        return self.metadata

# Protocol for optimizers
@runtime_checkable
class Optimizer(Protocol):
    def update(
        self,
        params: Params,
        gradients: Params,
        learning_rate: float
    ) -> Params:
        ...

    def get_state(self) -> Dict[str, Any]:
        ...

# Protocol for models
@runtime_checkable
class Model(Protocol):
    def __call__(self, params: Params, x: Array) -> Array:
        ...

    def init_params(self, key: PRNGKey, input_shape: Tuple[int, ...]) -> Params:
        ...

# Implementation example
class SGDOptimizer:
    def __init__(self, momentum: float = 0.0) -> None:
        self.momentum = momentum
        self.velocity: Optional[Params] = None

    def update(
        self,
        params: Params,
        gradients: Params,
        learning_rate: float
    ) -> Params:
        if self.velocity is None:
            self.velocity = {k: jnp.zeros_like(v) for k, v in params.items()}

        updated_params = {}
        for key in params:
            self.velocity[key] = (
                self.momentum * self.velocity[key] - learning_rate * gradients[key]
            )
            updated_params[key] = params[key] + self.velocity[key]

        return updated_params

    def get_state(self) -> Dict[str, Any]:
        return {'momentum': self.momentum, 'velocity': self.velocity}
```

## Scientific Computing Type Patterns

```python
import numpy as np
import scipy.sparse as sp
from scipy import optimize
from numpy.typing import NDArray
from typing import Union, Tuple, Callable, Any

# NumPy array type aliases
FloatArray = NDArray[np.floating]
IntArray = NDArray[np.integer]
ComplexArray = NDArray[np.complexfloating]
BoolArray = NDArray[np.bool_]

# Sparse matrix types
SparseMatrix = Union[
    sp.csr_matrix, sp.csc_matrix, sp.coo_matrix,
    sp.bsr_matrix, sp.dia_matrix, sp.dok_matrix, sp.lil_matrix
]

# Optimization function types
ObjectiveFunction = Callable[[FloatArray], float]
GradientFunction = Callable[[FloatArray], FloatArray]
HessianFunction = Callable[[FloatArray], FloatArray]

# Scientific functions with type hints
def solve_ode(
    func: Callable[[float, FloatArray], FloatArray],
    y0: FloatArray,
    t_span: Tuple[float, float],
    t_eval: Optional[FloatArray] = None,
    method: str = 'RK45'
) -> Tuple[FloatArray, FloatArray]:
    """Solve ordinary differential equation."""
    from scipy.integrate import solve_ivp

    result = solve_ivp(func, t_span, y0, t_eval=t_eval, method=method)
    return result.t, result.y

def optimize_function(
    objective: ObjectiveFunction,
    x0: FloatArray,
    gradient: Optional[GradientFunction] = None,
    hessian: Optional[HessianFunction] = None,
    bounds: Optional[List[Tuple[float, float]]] = None
) -> optimize.OptimizeResult:
    """Optimize scalar function."""
    return optimize.minimize(
        objective, x0, jac=gradient, hess=hessian, bounds=bounds
    )

def fit_polynomial(
    x: FloatArray,
    y: FloatArray,
    degree: int
) -> Tuple[FloatArray, float]:
    """Fit polynomial to data."""
    coeffs = np.polyfit(x, y, degree)
    residual = np.sum((np.polyval(coeffs, x) - y) ** 2)
    return coeffs, residual

# Sparse matrix operations
def sparse_matrix_multiply(
    A: SparseMatrix,
    B: Union[SparseMatrix, FloatArray]
) -> Union[SparseMatrix, FloatArray]:
    """Multiply sparse matrices efficiently."""
    return A @ B

def solve_sparse_system(
    A: SparseMatrix,
    b: FloatArray,
    method: str = 'spsolve'
) -> FloatArray:
    """Solve sparse linear system."""
    if method == 'spsolve':
        return sp.linalg.spsolve(A, b)
    elif method == 'cg':
        result, info = sp.linalg.cg(A, b)
        if info != 0:
            raise RuntimeError(f"Conjugate gradient failed with info={info}")
        return result
    else:
        raise ValueError(f"Unknown method: {method}")
```

## Class-Based Type Hints

```python
from typing import ClassVar, Final
from dataclasses import dataclass
from abc import ABC, abstractmethod

@dataclass(frozen=True)
class ExperimentConfig:
    """Configuration for scientific experiment."""
    learning_rate: float
    batch_size: int
    num_epochs: int
    random_seed: int
    model_name: str

    # Class variable
    SUPPORTED_MODELS: ClassVar[List[str]] = ['mlp', 'cnn', 'transformer']

    def __post_init__(self) -> None:
        if self.model_name not in self.SUPPORTED_MODELS:
            raise ValueError(f"Unsupported model: {self.model_name}")

class BaseNeuralNetwork(ABC):
    """Abstract base class for neural networks."""

    def __init__(self, input_dim: int, output_dim: int) -> None:
        self.input_dim: Final[int] = input_dim
        self.output_dim: Final[int] = output_dim
        self._params: Optional[Params] = None

    @abstractmethod
    def forward(self, params: Params, x: Array) -> Array:
        """Forward pass through network."""
        pass

    @abstractmethod
    def init_params(self, key: PRNGKey) -> Params:
        """Initialize network parameters."""
        pass

    @property
    def params(self) -> Params:
        if self._params is None:
            raise ValueError("Parameters not initialized")
        return self._params

    @params.setter
    def params(self, value: Params) -> None:
        self._params = value

class MLP(BaseNeuralNetwork):
    """Multi-layer perceptron implementation."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        activation: Callable[[Array], Array] = jnp.tanh
    ) -> None:
        super().__init__(input_dim, output_dim)
        self.hidden_dims = hidden_dims
        self.activation = activation
        self.layer_dims = [input_dim] + hidden_dims + [output_dim]

    def forward(self, params: Params, x: Array) -> Array:
        """Forward pass through MLP."""
        h = x
        for i in range(len(self.layer_dims) - 1):
            w = params[f'w_{i}']
            b = params[f'b_{i}']
            h = h @ w.T + b
            if i < len(self.layer_dims) - 2:  # No activation on output layer
                h = self.activation(h)
        return h

    def init_params(self, key: PRNGKey) -> Params:
        """Initialize MLP parameters."""
        params = {}
        keys = jax.random.split(key, len(self.layer_dims) - 1)

        for i, (in_dim, out_dim) in enumerate(zip(self.layer_dims[:-1], self.layer_dims[1:])):
            k1, k2 = jax.random.split(keys[i])
            params[f'w_{i}'] = jax.random.normal(k1, (out_dim, in_dim)) * 0.1
            params[f'b_{i}'] = jax.random.normal(k2, (out_dim,)) * 0.1

        return params
```

## Advanced Type Annotations

```python
from typing import (
    Literal, Union, overload, TypedDict, NewType,
    NamedTuple, Concatenate, ParamSpec
)
from functools import partial

# Literal types for specific options
OptimizationMethod = Literal['adam', 'sgd', 'rmsprop']
ActivationFunction = Literal['relu', 'tanh', 'sigmoid', 'gelu']

# TypedDict for structured dictionaries
class ModelConfig(TypedDict):
    model_type: Literal['mlp', 'cnn']
    hidden_dims: List[int]
    activation: ActivationFunction
    dropout_rate: float

class TrainingConfig(TypedDict):
    optimizer: OptimizationMethod
    learning_rate: float
    batch_size: int
    num_epochs: int

# NewType for domain-specific types
UserId = NewType('UserId', int)
ExperimentId = NewType('ExperimentId', str)

# NamedTuple for structured return values
class TrainingResult(NamedTuple):
    final_params: Params
    train_losses: List[float]
    val_losses: List[float]
    training_time: float

# ParamSpec for decorators
P = ParamSpec('P')
R = TypeVar('R')

def log_execution(func: Callable[P, R]) -> Callable[P, R]:
    """Decorator to log function execution."""
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        print(f"Executing {func.__name__}")
        result = func(*args, **kwargs)
        print(f"Finished {func.__name__}")
        return result
    return wrapper

# Overloaded functions
@overload
def create_array(data: List[int]) -> Array: ...

@overload
def create_array(data: List[float]) -> Array: ...

@overload
def create_array(data: List[List[Union[int, float]]]) -> Array: ...

def create_array(data: Union[List[int], List[float], List[List[Union[int, float]]]]) -> Array:
    """Create JAX array from various input types."""
    return jnp.array(data)

# Generic factory function
def create_optimizer(
    optimizer_type: OptimizationMethod,
    learning_rate: float,
    **kwargs: Any
) -> Optimizer:
    """Factory function to create optimizers."""
    if optimizer_type == 'adam':
        return AdamOptimizer(learning_rate, **kwargs)
    elif optimizer_type == 'sgd':
        return SGDOptimizer(learning_rate, **kwargs)
    elif optimizer_type == 'rmsprop':
        return RMSPropOptimizer(learning_rate, **kwargs)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_type}")
```

## mypy Configuration

```ini
# mypy.ini
[mypy]
# Basic options
python_version = 3.9
warn_return_any = True
warn_unused_configs = True
disallow_untyped_defs = True

# Strictness options
disallow_any_generics = True
disallow_subclassing_any = True
disallow_untyped_calls = True
disallow_incomplete_defs = True
check_untyped_defs = True
disallow_untyped_decorators = True

# Error reporting
show_error_codes = True
show_column_numbers = True
pretty = True

# Warnings
warn_redundant_casts = True
warn_unused_ignores = True
warn_no_return = True
warn_unreachable = True

# Per-module options
[mypy-numpy.*]
ignore_missing_imports = True

[mypy-scipy.*]
ignore_missing_imports = True

[mypy-jax.*]
ignore_missing_imports = True

[mypy-matplotlib.*]
ignore_missing_imports = True
```

## Type Checking Integration

```python
# Type checking utilities
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    # Import only for type checking (not at runtime)
    from expensive_library import ExpensiveClass

def process_data(data: Any) -> Array:
    """Process data with runtime type checking."""
    if not isinstance(data, (list, tuple, np.ndarray)):
        raise TypeError(f"Expected list, tuple, or ndarray, got {type(data)}")

    # Runtime cast for type checker
    array_data = cast(Union[List[float], NDArray], data)
    return jnp.array(array_data)

# Runtime type validation
def validate_params(params: Any) -> Params:
    """Validate parameters at runtime."""
    if not isinstance(params, dict):
        raise TypeError("Parameters must be a dictionary")

    for key, value in params.items():
        if not isinstance(key, str):
            raise TypeError("Parameter keys must be strings")
        if not isinstance(value, jnp.ndarray):
            raise TypeError("Parameter values must be JAX arrays")

    return params

# Type-safe configuration loading
def load_config(config_path: str) -> ModelConfig:
    """Load and validate configuration."""
    import json

    with open(config_path, 'r') as f:
        raw_config = json.load(f)

    # Runtime validation of TypedDict
    required_keys = {'model_type', 'hidden_dims', 'activation', 'dropout_rate'}
    if not all(key in raw_config for key in required_keys):
        missing = required_keys - raw_config.keys()
        raise ValueError(f"Missing configuration keys: {missing}")

    return cast(ModelConfig, raw_config)
```

## Testing Type Annotations

```python
import pytest
from typing import Any

def test_type_annotations() -> None:
    """Test that functions work with proper types."""
    # Test array creation
    int_array = create_array([1, 2, 3])
    assert int_array.shape == (3,)
    assert int_array.dtype == jnp.int32

    float_array = create_array([1.0, 2.0, 3.0])
    assert float_array.shape == (3,)
    assert float_array.dtype == jnp.float32

    # Test parameter validation
    valid_params = {'w': jnp.array([1.0, 2.0]), 'b': jnp.array([0.5])}
    validated = validate_params(valid_params)
    assert validated == valid_params

    # Test invalid parameters
    with pytest.raises(TypeError):
        validate_params("not_a_dict")

    with pytest.raises(TypeError):
        validate_params({123: jnp.array([1.0])})  # Non-string key

def test_mypy_compliance() -> None:
    """Test that code passes mypy checks."""
    # This would typically be run as part of CI/CD
    import subprocess
    result = subprocess.run(['mypy', '.'], capture_output=True, text=True)
    assert result.returncode == 0, f"mypy errors: {result.stdout}"
```

## Agent-Enhanced Python Type Hinting Integration Patterns

### Complete Python Type Safety Development Workflow
```bash
# Intelligent Python type safety development pipeline
/python-type-hint --agents=auto --intelligent --optimize --comprehensive
/python-debug-prof --agents=quality --intelligent --suggest-opts
/check-code-quality --agents=python --language=python --analysis=basic
```

### Scientific Computing Type Safety Pipeline
```bash
# High-performance scientific Python type safety workflow
/python-type-hint --agents=scientific --breakthrough --orchestrate --scientific
/jax-init --agents=python --intelligent --optimize
/jax-essentials --agents=python --intelligent --operation=all
```

### Production Python Type Safety Infrastructure
```bash
# Large-scale production Python type safety optimization
/python-type-hint --agents=ai --optimize --comprehensive --strict
/generate-tests --agents=python --type=unit --coverage=95
/run-all-tests --agents=quality --scientific --coverage
```

## Related Commands

**Python Ecosystem Development**: Enhanced Python type safety development with agent intelligence
- `/jax-init --agents=auto` - Initialize JAX projects with proper imports and agent optimization
- `/python-debug-prof --agents=quality` - Profile typed Python/JAX code with quality agents
- `/jax-debug --agents=python` - Debug JAX code with type safety and Python agents

**Cross-Language Type Safety**: Multi-language type integration
- `/jax-essentials --agents=auto` - Learn JAX transformations with type hints
- `/julia-jit-like --agents=python` - Compare with Julia type optimization
- `/optimize --agents=python` - Python optimization with specialized agents

**Quality Assurance**: Python type safety validation and optimization
- `/generate-tests --agents=quality --type=unit` - Generate Python tests with type safety focus
- `/run-all-tests --agents=python --scientific` - Comprehensive Python testing with specialized agents
- `/check-code-quality --agents=auto --language=python` - Python code quality with agent optimization

ARGUMENTS: [--strict] [--jax-arrays] [--scientific] [--target=file|directory] [--agents=auto|python|scientific|ai|quality|all] [--orchestrate] [--intelligent] [--breakthrough] [--optimize] [--comprehensive]