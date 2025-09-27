---
description: Compute gradients using JAX automatic differentiation with value_and_grad and higher-order derivatives
category: jax-core
argument-hint: "[function_name] [--higher-order] [--value-and-grad]"
allowed-tools: "*"
---

# /jax-grad

Compute gradients of scalar-valued functions using JAX's automatic differentiation.

## Description

Sets up gradient computation using `jax.grad`, `jax.value_and_grad`, and handles higher-order derivatives. Essential for optimization and machine learning workflows.

## Usage

```
/jax-grad [function_name] [--higher-order] [--value-and-grad]
```

## What it does

1. Apply `jax.grad` to compute gradients of scalar functions
2. Use `jax.value_and_grad` when both function value and gradient are needed
3. Handle higher-order derivatives with nested `grad` calls
4. Provide gradient checking and debugging tips

## Example output

```python
# Basic gradient computation
def loss_fn(params, x, y):
    pred = jnp.dot(x, params)
    return jnp.mean((pred - y) ** 2)

# Compute gradient with respect to first argument (params)
grad_fn = jax.grad(loss_fn)
gradients = grad_fn(params, x, y)

# Get both value and gradient (more efficient than separate calls)
value_and_grad_fn = jax.value_and_grad(loss_fn)
loss_value, gradients = value_and_grad_fn(params, x, y)

# Specify argument position for gradient
grad_fn = jax.grad(loss_fn, argnums=0)  # gradient w.r.t. params
grad_fn = jax.grad(loss_fn, argnums=(0, 1))  # gradient w.r.t. params and x

# Higher-order derivatives
hessian_fn = jax.grad(jax.grad(loss_fn))  # Second derivative
hess = hessian_fn(params, x, y)

# For vector outputs, use jax.jacobian
jacobian_fn = jax.jacobian(vector_function)

# Gradient tips:
# - Function must return a scalar for jax.grad
# - Use jax.jacobian for vector-valued functions
# - Use has_aux=True if function returns auxiliary data
```

## Related Commands

- `/jax-jit` - JIT compile gradient functions for speed
- `/jax-ml-train` - Use gradients in training loops
- `/jax-optax-optimizer` - Apply gradients with optimizers