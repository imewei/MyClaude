---
description: Add comprehensive type hints to Python/JAX code using typing and jax.typing for mypy compatibility
category: python-scientific
argument-hint: "[--strict] [--jax-arrays] [--scientific]"
allowed-tools: "*"
---

# /python-type-hint

Add comprehensive type hints to Python/JAX code using typing or jax.typing. Ensure compatibility with mypy and explain benefits for scientific workflows.

## Description

Transforms Python/JAX code to include comprehensive type annotations using the typing module and jax.typing. Ensures mypy compatibility and provides clear type safety for scientific computing workflows.

## Usage

```
/python-type-hint [--strict] [--jax-arrays] [--scientific]
```

## What it does

1. Add type hints to function signatures and variables
2. Use jax.typing for JAX-specific array types
3. Ensure mypy compatibility for static type checking
4. Provide generic types for scientific computing patterns
5. Document benefits for research reproducibility

## Example output

```python
from typing import Tuple, Optional, Union, List, Dict, Any, Callable, TypeVar
from jax.typing import ArrayLike
import jax.numpy as jnp
import jax
import numpy as np

# JAX-specific type hints
Array = jax.Array  # JAX array type
PRNGKey = jax.Array  # PRNG key type

# Generic types for scientific computing
T = TypeVar('T')
DType = Union[jnp.float32, jnp.float64, jnp.int32, jnp.int64]

# Basic function with type hints
def normalize_vector(x: ArrayLike, axis: int = -1) -> Array:
    """Normalize a vector along specified axis."""
    return x / jnp.linalg.norm(x, axis=axis, keepdims=True)

# Complex scientific function with comprehensive typing
def train_step(
    params: Dict[str, Array],
    batch: Tuple[Array, Array],
    optimizer_state: Any,
    learning_rate: float = 1e-3
) -> Tuple[Dict[str, Array], Any, float]:
    """Training step with full type annotations."""
    x, y = batch

    def loss_fn(p: Dict[str, Array]) -> float:
        logits = model_forward(p, x)
        return jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits, y))

    loss_value, grads = jax.value_and_grad(loss_fn)(params)
    updates, new_opt_state = optimizer.update(grads, optimizer_state, params)
    new_params = optax.apply_updates(params, updates)

    return new_params, new_opt_state, loss_value

# Higher-order function typing
def create_loss_fn(
    model_fn: Callable[[Dict[str, Array], Array], Array]
) -> Callable[[Dict[str, Array], Tuple[Array, Array]], float]:
    """Create loss function with proper typing."""
    def loss_fn(params: Dict[str, Array], batch: Tuple[Array, Array]) -> float:
        x, y = batch
        predictions = model_fn(params, x)
        return jnp.mean((predictions - y) ** 2)
    return loss_fn

# JAX transformation typing
@jax.jit
def jitted_function(x: Array, y: Array) -> Array:
    """JIT-compiled function with type hints."""
    return jnp.dot(x, y)

# Vectorized function typing
batch_process: Callable[[Array], Array] = jax.vmap(
    lambda x: jnp.sum(x ** 2)
)

# Class with type hints for scientific computing
class GaussianProcess:
    """Gaussian Process with comprehensive typing."""

    def __init__(
        self,
        kernel_fn: Callable[[Array, Array], Array],
        noise_var: float = 1e-6
    ) -> None:
        self.kernel_fn = kernel_fn
        self.noise_var = noise_var
        self.X_train: Optional[Array] = None
        self.y_train: Optional[Array] = None

    def fit(self, X: Array, y: Array) -> 'GaussianProcess':
        """Fit GP to training data."""
        self.X_train = X
        self.y_train = y
        return self

    def predict(
        self,
        X_test: Array
    ) -> Tuple[Array, Array]:
        """Predict with uncertainty quantification."""
        if self.X_train is None or self.y_train is None:
            raise ValueError("Model must be fitted first")

        # Implementation details...
        mean_pred = jnp.zeros(X_test.shape[0])
        var_pred = jnp.ones(X_test.shape[0])

        return mean_pred, var_pred

# Protocol for scientific models (Python 3.8+)
from typing import Protocol

class ScientificModel(Protocol):
    """Protocol for scientific computing models."""

    def forward(self, params: Dict[str, Array], x: Array) -> Array:
        """Forward pass through model."""
        ...

    def loss(self, params: Dict[str, Array], batch: Tuple[Array, Array]) -> float:
        """Compute loss on batch."""
        ...

# Advanced: Generic array operations
def matrix_multiply_generic(
    A: Array,
    B: Array,
    precision: Optional[jax.lax.Precision] = None
) -> Array:
    """Generic matrix multiplication with precision control."""
    return jnp.dot(A, B, precision=precision)

# Mypy configuration for JAX projects
"""
# mypy.ini configuration for scientific computing:
[mypy]
python_version = 3.9
warn_return_any = True
warn_unused_configs = True
disallow_untyped_defs = True
disallow_incomplete_defs = True

[mypy-jax.*]
ignore_missing_imports = False

[mypy-numpy.*]
ignore_missing_imports = False

[mypy-scipy.*]
ignore_missing_imports = True
"""

# Benefits of type hints in scientific computing:
"""
1. REPRODUCIBILITY: Clear function signatures ensure consistent usage
2. DEBUGGING: Early detection of shape/dtype mismatches
3. DOCUMENTATION: Self-documenting code for research collaboration
4. IDE SUPPORT: Better autocomplete and error detection
5. REFACTORING: Safe code changes with static analysis
6. ONBOARDING: Easier for new team members to understand code
7. TESTING: Type-guided test generation and property-based testing
"""

# Common type hint patterns for scientific computing
RealArray = Union[Array, np.ndarray]  # Accept both JAX and NumPy arrays
Shape = Tuple[int, ...]  # Array shape type
Optimizer = Any  # Placeholder for optimizer types
ModelParams = Dict[str, Union[Array, Dict[str, Array]]]  # Nested parameters

# Type checking utilities
def check_array_shape(arr: Array, expected_shape: Shape) -> Array:
    """Runtime shape checking with type hints."""
    if arr.shape != expected_shape:
        raise ValueError(f"Expected shape {expected_shape}, got {arr.shape}")
    return arr

# Integration with existing JAX ecosystem
def typed_jax_workflow() -> None:
    """Example of fully typed JAX workflow."""
    key: PRNGKey = jax.random.PRNGKey(42)
    data: Array = jax.random.normal(key, (100, 10))

    # Type-safe function composition
    normalized: Array = normalize_vector(data)
    processed: Array = batch_process(normalized)

    print(f"Result shape: {processed.shape}")
```

## Related Commands

- `/jax-init` - Initialize JAX projects with proper imports
- `/python-debug-prof` - Profile typed Python/JAX code
- `/jax-debug` - Debug JAX code with type safety