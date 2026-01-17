#!/usr/bin/env python3
"""
Workflow Pattern 1: Quick JAX Prototyping

Demonstrates incremental addition of JAX transformations for rapid experimentation.
"""

import jax
import jax.numpy as jnp
import time


def example_quick_prototyping():
    """Complete example of quick JAX prototyping workflow"""

    print("=" * 60)
    print("Workflow Pattern 1: Quick JAX Prototyping")
    print("=" * 60)

    # Generate synthetic data
    rng = jax.random.PRNGKey(0)
    X = jax.random.normal(rng, (1000, 10))
    y = jax.random.normal(rng, (1000, 1))

    # Step 1: Pure function definition
    print("\nStep 1: Pure function definition")
    params = {'w': jnp.ones((10, 1)), 'b': jnp.zeros((1,))}

    def model_fn(params, x):
        """Simple linear model"""
        return x @ params['w'] + params['b']

    # Test single example
    pred = model_fn(params, X[0])
    print(f"Single prediction shape: {pred.shape}")

    # Step 2: Define loss function
    print("\nStep 2: Define loss function")

    def loss_fn(params, x, y):
        """Mean squared error loss"""
        pred = model_fn(params, x)
        return jnp.mean((pred - y) ** 2)

    # Test loss
    loss = loss_fn(params, X[0], y[0])
    print(f"Single example loss: {loss:.4f}")

    # Step 3: Add compilation with jit
    print("\nStep 3: Add JIT compilation")
    fast_loss_fn = jax.jit(loss_fn)

    # Benchmark speedup
    start = time.time()
    for _ in range(100):
        _ = loss_fn(params, X[0], y[0])
        jax.block_until_ready(_)
    naive_time = time.time() - start

    start = time.time()
    for _ in range(100):
        _ = fast_loss_fn(params, X[0], y[0])
        jax.block_until_ready(_)
    jit_time = time.time() - start

    print(f"Naive: {naive_time * 1000:.2f}ms")
    print(f"JIT: {jit_time * 1000:.2f}ms")
    print(f"Speedup: {naive_time / jit_time:.1f}x")

    # Step 4: Add vectorization with vmap
    print("\nStep 4: Add vectorization with vmap")

    # Manual batching (slow)
    def manual_batch_loss(params, X_batch, y_batch):
        losses = [loss_fn(params, x, y) for x, y in zip(X_batch, y_batch)]
        return jnp.mean(jnp.array(losses))

    # Automatic vectorization (fast)
    vectorized_loss_fn = jax.vmap(loss_fn, in_axes=(None, 0, 0))

    def batch_loss_fn(params, X_batch, y_batch):
        return jnp.mean(vectorized_loss_fn(params, X_batch, y_batch))

    # Benchmark
    start = time.time()
    _ = manual_batch_loss(params, X[:32], y[:32])
    jax.block_until_ready(_)
    manual_time = time.time() - start

    start = time.time()
    _ = batch_loss_fn(params, X[:32], y[:32])
    jax.block_until_ready(_)
    vmap_time = time.time() - start

    print(f"Manual batching: {manual_time * 1000:.2f}ms")
    print(f"Vmap batching: {vmap_time * 1000:.2f}ms")
    print(f"Speedup: {manual_time / vmap_time:.1f}x")

    # Step 5: Add differentiation with grad
    print("\nStep 5: Add gradient computation")

    grad_fn = jax.grad(batch_loss_fn)
    grads = grad_fn(params, X[:32], y[:32])

    print(f"Gradient w shape: {grads['w'].shape}")
    print(f"Gradient b shape: {grads['b'].shape}")
    print(f"Gradient w norm: {jnp.linalg.norm(grads['w']):.4f}")

    # Step 6: Combine all transformations
    print("\nStep 6: Combine all transformations")

    @jax.jit
    def fast_grad_fn(params, X_batch, y_batch):
        """Compiled gradient function with vectorization"""
        vectorized_loss = jax.vmap(loss_fn, in_axes=(None, 0, 0))
        batch_loss = lambda p: jnp.mean(vectorized_loss(p, X_batch, y_batch))
        return jax.grad(batch_loss)(params)

    # Final benchmark
    start = time.time()
    _ = grad_fn(params, X[:32], y[:32])
    jax.block_until_ready(_)
    base_grad_time = time.time() - start

    start = time.time()
    _ = fast_grad_fn(params, X[:32], y[:32])
    jax.block_until_ready(_)
    fast_grad_time = time.time() - start

    print(f"Base gradient: {base_grad_time * 1000:.2f}ms")
    print(f"Optimized gradient: {fast_grad_time * 1000:.2f}ms")
    print(f"Speedup: {base_grad_time / fast_grad_time:.1f}x")

    # Step 7: Simple training loop
    print("\nStep 7: Quick training loop")

    learning_rate = 0.01
    params = {'w': jnp.ones((10, 1)), 'b': jnp.zeros((1,))}

    for step in range(10):
        grads = fast_grad_fn(params, X[:32], y[:32])
        # SGD update
        params = jax.tree_map(lambda p, g: p - learning_rate * g, params, grads)

        if step % 3 == 0:
            loss = batch_loss_fn(params, X[:32], y[:32])
            print(f"Step {step}, Loss: {loss:.4f}")

    print("\n✓ Quick prototyping workflow complete!")
    print(f"Final parameters: w norm = {jnp.linalg.norm(params['w']):.4f}")


if __name__ == '__main__':
    example_quick_prototyping()
