# Scientific & Mathematical Skills

JAX is heavily used in "AI for Science" (Physics, Biology, Chemistry). This reference covers numerical computing expertise essential for scientific applications.

## Numerical Stability

### Floating-Point Precision

```python
import jax.numpy as jnp

# Know your precision requirements
x_f32 = jnp.array([1.0], dtype=jnp.float32)   # 7 decimal digits
x_bf16 = jnp.array([1.0], dtype=jnp.bfloat16) # 3 decimal digits, same range as f32
x_f16 = jnp.array([1.0], dtype=jnp.float16)   # 3 decimal digits, limited range

# Mixed precision training pattern
def mixed_precision_step(params_f32, x_bf16, y_bf16):
    # Forward in lower precision
    params_bf16 = jax.tree.map(lambda p: p.astype(jnp.bfloat16), params_f32)
    loss, grads_bf16 = jax.value_and_grad(loss_fn)(params_bf16, x_bf16, y_bf16)

    # Accumulate gradients in higher precision
    grads_f32 = jax.tree.map(lambda g: g.astype(jnp.float32), grads_bf16)
    return loss, grads_f32
```

### NaN Debugging

```python
# Enable NaN checking globally
jax.config.update("jax_debug_nans", True)

# Common NaN sources and fixes
def stable_softmax(x):
    # Anti-pattern: can overflow
    # return jnp.exp(x) / jnp.exp(x).sum()

    # Stable: subtract max before exp
    x_max = jnp.max(x, axis=-1, keepdims=True)
    exp_x = jnp.exp(x - x_max)
    return exp_x / exp_x.sum(axis=-1, keepdims=True)

def stable_log_softmax(x):
    # Use JAX's stable implementation
    return jax.nn.log_softmax(x)

def stable_cross_entropy(logits, labels):
    # Avoid log(0) by using log_softmax
    log_probs = jax.nn.log_softmax(logits)
    return -jnp.sum(labels * log_probs, axis=-1)
```

### Avoiding Unstable Gradients

```python
# Problem: gradient of sqrt(x) at x=0 is inf
def unstable_norm(x):
    return jnp.sqrt(jnp.sum(x ** 2))

# Solution: add small epsilon
def stable_norm(x, eps=1e-8):
    return jnp.sqrt(jnp.sum(x ** 2) + eps)

# Problem: where() with NaN in unused branch still propagates
def unstable_safe_div(x, y):
    # NaN gradient even when y != 0
    return jnp.where(y != 0, x / y, 0.0)

# Solution: modify both branches to be valid
def stable_safe_div(x, y, eps=1e-8):
    safe_y = jnp.where(y != 0, y, 1.0)  # Avoid div by zero
    result = x / safe_y
    return jnp.where(y != 0, result, 0.0)
```

## Custom Primitives

Register custom primitives when specific operations are not efficiently supported or require custom derivative rules.

### Custom VJP (Vector-Jacobian Product)

```python
from jax import custom_vjp

@custom_vjp
def my_special_fn(x):
    """Forward pass - normal computation."""
    return jnp.exp(x) * jnp.sin(x)

def my_special_fn_fwd(x):
    """Forward pass that saves values for backward."""
    y = my_special_fn(x)
    return y, (x, y)  # Return result and residuals

def my_special_fn_bwd(res, g):
    """Backward pass using saved residuals."""
    x, y = res
    # Custom gradient computation
    grad_x = g * (jnp.exp(x) * jnp.sin(x) + jnp.exp(x) * jnp.cos(x))
    return (grad_x,)  # Tuple of gradients for each input

my_special_fn.defvjp(my_special_fn_fwd, my_special_fn_bwd)
```

### Custom JVP (Jacobian-Vector Product)

```python
from jax import custom_jvp

@custom_jvp
def my_fn(x):
    return jnp.sin(x)

@my_fn.defjvp
def my_fn_jvp(primals, tangents):
    x, = primals
    x_dot, = tangents
    primal_out = my_fn(x)
    tangent_out = jnp.cos(x) * x_dot  # Custom derivative
    return primal_out, tangent_out
```

### Stop Gradient

```python
# Prevent gradients through specific paths
def contrastive_loss(anchor, positive, negative):
    # Don't update negative examples
    neg_fixed = jax.lax.stop_gradient(negative)
    pos_sim = similarity(anchor, positive)
    neg_sim = similarity(anchor, neg_fixed)
    return -jnp.log(jnp.exp(pos_sim) / (jnp.exp(pos_sim) + jnp.exp(neg_sim)))

# Straight-through estimator for discrete operations
def straight_through_round(x):
    """Round in forward pass, pass gradients through in backward."""
    return x + jax.lax.stop_gradient(jnp.round(x) - x)
```

## Advanced Control Flow

Unlike Python loops, these primitives compile into the XLA graph—essential for performance in recurrent networks or ODE solvers.

### jax.lax.scan: Compiled Sequential Loops

```python
def rnn_cell(carry, x):
    h = carry
    new_h = jnp.tanh(jnp.dot(x, Wx) + jnp.dot(h, Wh) + b)
    return new_h, new_h  # (new_carry, output)

# Scan over sequence dimension
final_hidden, all_hiddens = jax.lax.scan(rnn_cell, initial_hidden, inputs)

# ODE solver with scan
def euler_step(y, t_dt):
    t, dt = t_dt
    dy = ode_fn(y, t)
    return y + dt * dy, y

ts = jnp.linspace(0, 1, 100)
dts = jnp.diff(ts)
_, trajectory = jax.lax.scan(euler_step, y0, (ts[:-1], dts))
```

### jax.lax.cond: Compiled Conditionals

```python
def safe_operation(x, threshold):
    def true_branch(x):
        return jnp.sqrt(x)

    def false_branch(x):
        return jnp.zeros_like(x)

    return jax.lax.cond(
        x > threshold,
        true_branch,
        false_branch,
        x
    )

# Nested conditionals
def classify(x):
    return jax.lax.cond(
        x < 0,
        lambda _: -1,
        lambda x: jax.lax.cond(x > 0, lambda _: 1, lambda _: 0, x),
        x
    )
```

### jax.lax.while_loop: Dynamic Iteration

```python
def newton_raphson(f, df, x0, tol=1e-6, max_iter=100):
    """Find root using Newton-Raphson method."""

    def cond_fn(state):
        x, fx, i = state
        return (jnp.abs(fx) > tol) & (i < max_iter)

    def body_fn(state):
        x, _, i = state
        fx = f(x)
        dfx = df(x)
        x_new = x - fx / dfx
        return x_new, f(x_new), i + 1

    x0_fx = f(x0)
    final_x, final_fx, iterations = jax.lax.while_loop(
        cond_fn, body_fn, (x0, x0_fx, 0)
    )
    return final_x
```

### jax.lax.fori_loop: Fixed Iteration Count

```python
def power_iteration(A, v, num_iters=100):
    """Compute dominant eigenvector."""

    def body_fn(i, v):
        v = A @ v
        return v / jnp.linalg.norm(v)

    return jax.lax.fori_loop(0, num_iters, body_fn, v)
```

### jax.lax.switch: Multi-way Branching

```python
def multi_activation(x, activation_type):
    """Apply different activations based on type."""

    def relu(x): return jnp.maximum(0, x)
    def tanh(x): return jnp.tanh(x)
    def sigmoid(x): return jax.nn.sigmoid(x)
    def gelu(x): return jax.nn.gelu(x)

    return jax.lax.switch(
        activation_type,
        [relu, tanh, sigmoid, gelu],
        x
    )
```

## Scientific Computing Patterns

### Differentiable Physics

```python
def simulate_pendulum(theta0, omega0, dt, num_steps):
    """Differentiable pendulum simulation."""

    def step(state, _):
        theta, omega = state
        # Physics: d²θ/dt² = -g/L * sin(θ)
        alpha = -9.81 / 1.0 * jnp.sin(theta)
        omega_new = omega + alpha * dt
        theta_new = theta + omega_new * dt
        return (theta_new, omega_new), theta_new

    _, trajectory = jax.lax.scan(step, (theta0, omega0), None, length=num_steps)
    return trajectory

# Can differentiate through the simulation!
d_trajectory_d_theta0 = jax.jacobian(simulate_pendulum)(0.5, 0.0, 0.01, 100)
```

### Implicit Differentiation

```python
from jax import custom_vjp

@custom_vjp
def fixed_point_solve(f, x0, num_iters=100):
    """Solve x = f(x) via iteration."""
    def body(i, x):
        return f(x)
    return jax.lax.fori_loop(0, num_iters, body, x0)

def fixed_point_solve_fwd(f, x0, num_iters):
    x_star = fixed_point_solve(f, x0, num_iters)
    return x_star, (f, x_star)

def fixed_point_solve_bwd(res, g):
    f, x_star = res
    # Implicit differentiation: solve (I - df/dx)^T v = g
    def vjp_residual(v):
        _, vjp_f = jax.vjp(f, x_star)
        return v - vjp_f(v)[0]

    v = fixed_point_solve(vjp_residual, g, num_iters=100)
    return (None, v, None)  # Gradients for (f, x0, num_iters)

fixed_point_solve.defvjp(fixed_point_solve_fwd, fixed_point_solve_bwd)
```

### Batched Linear Algebra

```python
# Solve many linear systems efficiently
def batched_solve(As, bs):
    """Solve Ax = b for batched A and b."""
    return jax.vmap(jnp.linalg.solve)(As, bs)

# Batched eigendecomposition
def batched_eigh(As):
    """Eigendecomposition for batched symmetric matrices."""
    return jax.vmap(jnp.linalg.eigh)(As)

# With gradients through eigendecomposition
@jax.jit
def differentiable_spectral_norm(A):
    eigenvalues, _ = jnp.linalg.eigh(A.T @ A)
    return jnp.sqrt(jnp.max(eigenvalues))
```
