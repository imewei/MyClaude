"""
XLA Optimization Patterns for JAX

Demonstrates techniques for analyzing and optimizing XLA compilation,
avoiding recompilation, and understanding HLO output.
"""

import jax
import jax.numpy as jnp
import os
from functools import partial


# =============================================================================
# Pattern 1: Avoiding Recompilation
# =============================================================================

# BAD: Different shapes cause recompilation
def bad_variable_shapes(data_list):
    """This recompiles for each unique shape."""
    results = []
    for data in data_list:
        result = jax.jit(lambda x: x ** 2)(data)  # Recompiles!
        results.append(result)
    return results


# GOOD: Pad to fixed shape
def good_fixed_shapes(data_list, max_len):
    """Pad all inputs to same shape - single compilation."""

    @jax.jit
    def compute(x, mask):
        result = x ** 2
        return jnp.where(mask, result, 0.0)

    results = []
    for data in data_list:
        # Pad to max_len
        padded = jnp.zeros(max_len).at[:len(data)].set(data)
        mask = jnp.arange(max_len) < len(data)
        results.append(compute(padded, mask))
    return results


# =============================================================================
# Pattern 2: Static vs Traced Arguments
# =============================================================================

# BAD: Recompiles for every unique value of `num_layers`
@jax.jit
def bad_dynamic_layers(x, num_layers):
    for _ in range(num_layers):  # num_layers is traced!
        x = jax.nn.relu(x @ jnp.eye(x.shape[-1]))
    return x


# GOOD: Mark control-flow-affecting args as static
@partial(jax.jit, static_argnums=(1,))
def good_static_layers(x, num_layers):
    for _ in range(num_layers):  # num_layers is static constant
        x = jax.nn.relu(x @ jnp.eye(x.shape[-1]))
    return x


# GOOD: Use static_argnames for clarity
@partial(jax.jit, static_argnames=('num_layers', 'activation'))
def configurable_network(x, num_layers, activation='relu'):
    act_fn = {'relu': jax.nn.relu, 'gelu': jax.nn.gelu}[activation]
    for _ in range(num_layers):
        x = act_fn(x @ jnp.eye(x.shape[-1]))
    return x


# =============================================================================
# Pattern 3: Inspecting Traced Computation
# =============================================================================

def inspect_jaxpr(fn, *example_args):
    """Print the JAX intermediate representation."""
    jaxpr = jax.make_jaxpr(fn)(*example_args)
    print("=" * 60)
    print("JAXPR (JAX Intermediate Representation)")
    print("=" * 60)
    print(jaxpr)
    return jaxpr


def inspect_hlo(fn, *example_args):
    """Get the HLO (XLA) representation."""
    computation = jax.xla_computation(fn)(*example_args)
    hlo_text = computation.as_hlo_text()
    print("=" * 60)
    print("HLO (XLA High Level Optimizer)")
    print("=" * 60)
    print(hlo_text[:2000])  # First 2000 chars
    if len(hlo_text) > 2000:
        print(f"\n... ({len(hlo_text) - 2000} more characters)")
    return computation


# =============================================================================
# Pattern 4: Fusion Optimization
# =============================================================================

def unfused_operations(x):
    """Multiple operations that should fuse."""
    y = x * 2
    z = y + 1
    w = jnp.sin(z)
    return w


def check_fusion(fn, *args):
    """Check if operations are fused in HLO."""
    computation = jax.xla_computation(fn)(*args)
    hlo = computation.as_hlo_text()

    # Look for fusion indicators
    fusion_count = hlo.count('fusion')
    print(f"Fusion operations found: {fusion_count}")

    # Check for separate kernels (bad)
    kernel_count = hlo.count('ROOT')
    print(f"Root operations (potential kernel launches): {kernel_count}")

    return hlo


# =============================================================================
# Pattern 5: Memory-Efficient Patterns
# =============================================================================

def memory_inefficient(x, weights_list):
    """Stores all intermediate activations."""
    activations = []
    for w in weights_list:
        x = jax.nn.relu(x @ w)
        activations.append(x)  # Memory grows linearly
    return x, activations


def memory_efficient_scan(x, weights_list):
    """Uses scan - only stores current activation."""
    weights_stacked = jnp.stack(weights_list)

    def layer_fn(carry, w):
        x = carry
        x_new = jax.nn.relu(x @ w)
        return x_new, None  # Don't store intermediate

    final_x, _ = jax.lax.scan(layer_fn, x, weights_stacked)
    return final_x


def gradient_checkpointing(x, weights_list):
    """Recompute activations during backward pass."""
    from jax.checkpoint import checkpoint

    def forward_layer(x, w):
        return jax.nn.relu(x @ w)

    for w in weights_list:
        # Checkpoint: don't store activation, recompute in backward
        x = checkpoint(forward_layer)(x, w)

    return x


# =============================================================================
# Pattern 6: Compilation Caching
# =============================================================================

def setup_compilation_cache(cache_dir="/tmp/jax_cache"):
    """Enable persistent compilation cache."""
    os.makedirs(cache_dir, exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", cache_dir)
    print(f"Compilation cache enabled at: {cache_dir}")


def monitor_compilations():
    """Enable compilation logging."""
    os.environ["JAX_LOG_COMPILES"] = "1"
    print("Compilation logging enabled. Watch for '[jax.jit]' messages.")


# =============================================================================
# Pattern 7: Donate Buffers for In-Place Updates
# =============================================================================

@partial(jax.jit, donate_argnums=(0,))
def inplace_update(state, update):
    """Donate `state` buffer - allows XLA to reuse memory.

    The original `state` array is no longer valid after this call.
    """
    return state + update


def demonstrate_donation():
    """Show buffer donation pattern."""
    state = jnp.zeros(1000000)  # 4MB

    # Without donation: allocates new 4MB for result
    # With donation: reuses state's memory

    for _ in range(100):
        update = jnp.ones(1000000) * 0.01
        state = inplace_update(state, update)
        # state buffer is reused each iteration

    return state


# =============================================================================
# Pattern 8: Debugging Slow Compilation
# =============================================================================

def profile_compilation(fn, *args):
    """Profile JIT compilation time."""
    import time

    # First call: includes compilation
    start = time.perf_counter()
    result1 = jax.jit(fn)(*args)
    result1.block_until_ready()
    compile_time = time.perf_counter() - start

    # Second call: execution only
    start = time.perf_counter()
    result2 = jax.jit(fn)(*args)
    result2.block_until_ready()
    exec_time = time.perf_counter() - start

    print(f"First call (compile + exec): {compile_time*1000:.2f}ms")
    print(f"Second call (exec only): {exec_time*1000:.2f}ms")
    print(f"Compilation overhead: {(compile_time - exec_time)*1000:.2f}ms")

    return result1


# =============================================================================
# Pattern 9: XLA Dump for Deep Analysis
# =============================================================================

def enable_xla_dump(dump_dir="/tmp/xla_dump"):
    """Enable XLA dump for detailed analysis."""
    os.makedirs(dump_dir, exist_ok=True)
    os.environ["XLA_FLAGS"] = (
        f"--xla_dump_to={dump_dir} "
        "--xla_dump_hlo_as_text "
        "--xla_dump_hlo_as_html "
        "--xla_dump_hlo_pass_re=.*"  # Dump all passes
    )
    print(f"XLA dumps will be saved to: {dump_dir}")
    print("After running, check .txt and .html files for HLO analysis")


# =============================================================================
# Demo
# =============================================================================

if __name__ == "__main__":
    print("XLA Optimization Patterns Demo")
    print("=" * 60)

    # Monitor compilations
    monitor_compilations()

    # Example function
    def example_fn(x, y):
        z = x @ y
        return jax.nn.softmax(z, axis=-1)

    x = jnp.ones((32, 64))
    y = jnp.ones((64, 128))

    # Inspect JAXPR
    print("\n1. JAXPR Inspection")
    inspect_jaxpr(example_fn, x, y)

    # Profile compilation
    print("\n2. Compilation Profiling")
    profile_compilation(example_fn, x, y)

    # Check fusion
    print("\n3. Fusion Analysis")
    check_fusion(unfused_operations, jnp.ones(1000))

    print("\n" + "=" * 60)
    print("Tips for XLA optimization:")
    print("1. Use static shapes when possible")
    print("2. Mark control-flow args as static_argnums")
    print("3. Use jax.lax.scan instead of Python loops")
    print("4. Enable compilation cache for faster restarts")
    print("5. Use buffer donation for in-place updates")
    print("6. Inspect HLO to find fusion breaks")
