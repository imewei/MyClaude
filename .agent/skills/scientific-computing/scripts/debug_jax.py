#!/usr/bin/env python3
"""
JAX Debugging Utilities

Tools for debugging JAX code, identifying issues, and understanding errors.
"""

import jax
import jax.numpy as jnp
from typing import Any, Callable


def check_nan_inf(x: Any, name: str = "tensor") -> bool:
    """
    Check for NaN or Inf values in JAX arrays or pytrees.

    Args:
        x: JAX array or pytree
        name: Name for logging

    Returns:
        True if no NaN/Inf found, False otherwise
    """
    def check_array(arr):
        has_nan = jnp.any(jnp.isnan(arr))
        has_inf = jnp.any(jnp.isinf(arr))
        return has_nan, has_inf

    # Handle pytrees
    leaves = jax.tree_leaves(x)
    for i, leaf in enumerate(leaves):
        has_nan, has_inf = check_array(leaf)
        if has_nan or has_inf:
            print(f"⚠ {name} leaf {i}: NaN={has_nan}, Inf={has_inf}")
            print(f"  Shape: {leaf.shape}, dtype: {leaf.dtype}")
            return False

    print(f"✓ {name}: No NaN/Inf detected")
    return True


def print_pytree_structure(pytree: Any, name: str = "pytree", max_depth: int = 3):
    """
    Print structure and shapes of a pytree.

    Args:
        pytree: JAX pytree to inspect
        name: Name for the pytree
        max_depth: Maximum depth to print
    """
    print(f"\n{name} structure:")
    print("-" * 60)

    def print_node(node, prefix="", depth=0):
        if depth > max_depth:
            print(f"{prefix}...")
            return

        if isinstance(node, dict):
            for key, value in node.items():
                if isinstance(value, (dict, list, tuple)):
                    print(f"{prefix}{key}:")
                    print_node(value, prefix + "  ", depth + 1)
                elif isinstance(value, jnp.ndarray):
                    print(f"{prefix}{key}: shape={value.shape}, dtype={value.dtype}")
                else:
                    print(f"{prefix}{key}: {type(value).__name__}")
        elif isinstance(node, (list, tuple)):
            for i, item in enumerate(node):
                if isinstance(item, jnp.ndarray):
                    print(f"{prefix}[{i}]: shape={item.shape}, dtype={item.dtype}")
                else:
                    print(f"{prefix}[{i}]:")
                    print_node(item, prefix + "  ", depth + 1)
        elif isinstance(node, jnp.ndarray):
            print(f"{prefix}shape={node.shape}, dtype={node.dtype}")

    print_node(pytree)


def debug_jit(fn: Callable, *args, **kwargs):
    """
    Debug a JIT-compiled function by disabling JIT and adding checks.

    Args:
        fn: JIT-compiled function
        *args: Arguments for fn
        **kwargs: Keyword arguments for fn
    """
    print("\nDebugging JIT function...")
    print("-" * 60)

    # Run with JIT disabled
    print("Running with JIT disabled...")
    with jax.disable_jit():
        try:
            result = fn(*args, **kwargs)
            print("✓ Function runs successfully without JIT")

            # Check for NaN/Inf
            check_nan_inf(result, "Output")

            return result

        except Exception as e:
            print(f"✗ Error without JIT: {e}")
            raise

    # If we get here, function works without JIT
    print("\nRunning with JIT enabled...")
    try:
        result = fn(*args, **kwargs)
        print("✓ Function runs successfully with JIT")
        return result
    except Exception as e:
        print(f"✗ Error with JIT: {e}")
        print("\nThis is likely a tracer-related issue.")
        print("Common causes:")
        print("  - Python control flow (use jax.lax.cond instead)")
        print("  - Dynamic shapes (use static_argnums)")
        print("  - Side effects (ensure pure functions)")
        raise


def inspect_jaxpr(fn: Callable, *args, **kwargs):
    """
    Inspect the JAX intermediate representation (jaxpr) of a function.

    Args:
        fn: Function to inspect
        *args: Example arguments
        **kwargs: Example keyword arguments
    """
    print("\nJAXPR Inspection:")
    print("-" * 60)

    jaxpr = jax.make_jaxpr(fn)(*args, **kwargs)
    print(jaxpr)

    print("\nOperations used:")
    ops = set()
    for eqn in jaxpr.jaxpr.eqns:
        ops.add(str(eqn.primitive))

    for op in sorted(ops):
        print(f"  - {op}")


def check_gradient_flow(loss_fn: Callable, params: Any, *args, **kwargs):
    """
    Check gradient flow through parameters.

    Args:
        loss_fn: Loss function to check
        params: Parameters
        *args: Additional arguments for loss_fn
        **kwargs: Additional keyword arguments
    """
    print("\nChecking gradient flow...")
    print("-" * 60)

    # Compute gradients
    grad_fn = jax.grad(loss_fn)
    grads = grad_fn(params, *args, **kwargs)

    # Check each parameter
    def check_param(path, grad):
        grad_norm = jnp.linalg.norm(grad.flatten())
        grad_mean = jnp.mean(jnp.abs(grad))
        grad_max = jnp.max(jnp.abs(grad))

        has_nan = jnp.any(jnp.isnan(grad))
        has_inf = jnp.any(jnp.isinf(grad))

        status = "✓" if not (has_nan or has_inf) else "✗"
        print(f"{status} {'.'.join(map(str, path))}")
        print(f"    norm={grad_norm:.4f}, mean={grad_mean:.6f}, max={grad_max:.6f}")

        if has_nan:
            print(f"    ⚠ Contains NaN!")
        if has_inf:
            print(f"    ⚠ Contains Inf!")
        if grad_norm == 0:
            print(f"    ⚠ Zero gradient (no gradient flow)")

    # Check all parameters
    jax.tree_map_with_path(check_param, grads)


def monitor_training_step(train_step_fn: Callable, params: Any, batch: Any):
    """
    Monitor a training step for common issues.

    Args:
        train_step_fn: Training step function
        params: Model parameters
        batch: Training batch
    """
    print("\nMonitoring training step...")
    print("-" * 60)

    # Check inputs
    print("\n1. Input checks:")
    check_nan_inf(params, "Parameters")
    check_nan_inf(batch, "Batch")

    # Run training step
    print("\n2. Running training step...")
    try:
        result = train_step_fn(params, batch)
        print("✓ Training step completed")
    except Exception as e:
        print(f"✗ Training step failed: {e}")
        raise

    # Check outputs
    print("\n3. Output checks:")
    if isinstance(result, tuple):
        for i, item in enumerate(result):
            check_nan_inf(item, f"Output[{i}]")
    else:
        check_nan_inf(result, "Output")

    return result


def enable_nan_checking():
    """Enable automatic NaN checking in JAX."""
    print("Enabling NaN checking...")
    jax.config.update("jax_debug_nans", True)
    print("✓ NaN checking enabled")
    print("  (Will raise error on first NaN)")


def disable_nan_checking():
    """Disable automatic NaN checking in JAX."""
    print("Disabling NaN checking...")
    jax.config.update("jax_debug_nans", False)
    print("✓ NaN checking disabled")


def check_device_placement(pytree: Any):
    """Check which devices data is placed on."""
    print("\nDevice placement:")
    print("-" * 60)

    def check_array(path, arr):
        devices = arr.devices()
        print(f"{'.'.join(map(str, path))}: {[str(d) for d in devices]}")

    jax.tree_map_with_path(check_array, pytree)


# Example usage
if __name__ == '__main__':
    print("JAX Debugging Utilities Demo")
    print("=" * 60)

    # Example 1: Check for NaN/Inf
    print("\n1. NaN/Inf Detection")
    x_good = jnp.array([1.0, 2.0, 3.0])
    x_bad = jnp.array([1.0, jnp.nan, 3.0])

    check_nan_inf(x_good, "good_array")
    check_nan_inf(x_bad, "bad_array")

    # Example 2: PyTree structure
    print("\n2. PyTree Structure")
    params = {
        'layer1': {'w': jnp.ones((10, 5)), 'b': jnp.zeros(10)},
        'layer2': {'w': jnp.ones((10, 3)), 'b': jnp.zeros(3)},
    }
    print_pytree_structure(params, "model_params")

    # Example 3: Debug JIT
    print("\n3. Debug JIT Function")

    @jax.jit
    def good_fn(x):
        return x ** 2

    debug_jit(good_fn, jnp.array([1.0, 2.0, 3.0]))

    # Example 4: Gradient flow
    print("\n4. Check Gradient Flow")

    def loss_fn(params, x):
        h = params['layer1']['w'] @ x + params['layer1']['b']
        h = jax.nn.relu(h)
        out = params['layer2']['w'] @ h + params['layer2']['b']
        return jnp.mean(out ** 2)

    x = jnp.ones(5)
    check_gradient_flow(loss_fn, params, x)

    print("\n" + "=" * 60)
    print("Debugging demo complete!")
