"""
Pure Functional Patterns for JAX

Demonstrates JAX-first patterns for state management, PyTree manipulation,
and explicit RNG handling. These patterns are essential for writing
JIT-compatible, parallelizable code.
"""

import jax
import jax.numpy as jnp
import optax
from typing import NamedTuple, Dict, Any


# =============================================================================
# Pattern 1: Explicit State Passing (No Globals)
# =============================================================================

class TrainState(NamedTuple):
    """Immutable training state - all state is explicit."""
    params: Dict[str, Any]
    opt_state: Any
    step: int
    loss_history: jnp.ndarray
    rng_key: jnp.ndarray


def create_train_state(params, optimizer, rng_key, history_size=1000):
    """Create initial training state."""
    return TrainState(
        params=params,
        opt_state=optimizer.init(params),
        step=0,
        loss_history=jnp.zeros(history_size),
        rng_key=rng_key,
    )


def train_step(state: TrainState, batch, optimizer, loss_fn):
    """Pure training step - no side effects, all state explicit."""
    # Split RNG for this step
    rng_key, dropout_key = jax.random.split(state.rng_key)

    # Compute loss and gradients
    loss, grads = jax.value_and_grad(loss_fn)(
        state.params, batch, dropout_key
    )

    # Update parameters
    updates, new_opt_state = optimizer.update(grads, state.opt_state, state.params)
    new_params = optax.apply_updates(state.params, updates)

    # Update loss history (immutable update)
    history_idx = state.step % len(state.loss_history)
    new_history = state.loss_history.at[history_idx].set(loss)

    # Return new state (immutable)
    return TrainState(
        params=new_params,
        opt_state=new_opt_state,
        step=state.step + 1,
        loss_history=new_history,
        rng_key=rng_key,
    ), loss


# =============================================================================
# Pattern 2: PyTree Manipulation
# =============================================================================

def init_params(rng_key, layer_sizes):
    """Initialize nested parameter structure."""
    params = {}
    for i, (in_size, out_size) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        rng_key, w_key, b_key = jax.random.split(rng_key, 3)
        params[f'layer_{i}'] = {
            'w': jax.random.normal(w_key, (in_size, out_size)) * 0.01,
            'b': jnp.zeros(out_size),
        }
    return params


def count_params(params):
    """Count total parameters in nested structure."""
    leaves = jax.tree.leaves(params)
    return sum(leaf.size for leaf in leaves)


def clip_gradients(grads, max_norm):
    """Clip gradients by global norm using PyTree operations."""
    # Compute global norm
    leaves = jax.tree.leaves(grads)
    global_norm = jnp.sqrt(sum(jnp.sum(leaf ** 2) for leaf in leaves))

    # Clip factor
    clip_factor = jnp.minimum(1.0, max_norm / (global_norm + 1e-8))

    # Apply to all leaves
    return jax.tree.map(lambda g: g * clip_factor, grads)


def apply_weight_decay(params, decay_rate):
    """Apply weight decay to all parameters."""
    return jax.tree.map(lambda p: p * (1 - decay_rate), params)


def interpolate_params(params1, params2, alpha):
    """Linear interpolation between two parameter sets."""
    return jax.tree.map(
        lambda p1, p2: (1 - alpha) * p1 + alpha * p2,
        params1,
        params2
    )


def flatten_params(params):
    """Flatten nested params to single vector."""
    leaves, treedef = jax.tree.flatten(params)
    flat = jnp.concatenate([leaf.ravel() for leaf in leaves])
    shapes = [leaf.shape for leaf in leaves]
    return flat, (treedef, shapes)


def unflatten_params(flat, structure):
    """Restore nested structure from flat vector."""
    treedef, shapes = structure
    sizes = [int(jnp.prod(jnp.array(shape))) for shape in shapes]
    splits = jnp.cumsum(jnp.array(sizes[:-1]))
    leaves = jnp.split(flat, splits)
    leaves = [leaf.reshape(shape) for leaf, shape in zip(leaves, shapes)]
    return jax.tree.unflatten(treedef, leaves)


# =============================================================================
# Pattern 3: Explicit RNG Handling
# =============================================================================

def sample_batch(rng_key, data, batch_size):
    """Sample random batch without side effects."""
    n_samples = data.shape[0]
    indices = jax.random.choice(rng_key, n_samples, shape=(batch_size,), replace=False)
    return data[indices]


def dropout(rng_key, x, rate, training=True):
    """Dropout with explicit RNG."""
    if not training:
        return x
    keep_prob = 1 - rate
    mask = jax.random.bernoulli(rng_key, keep_prob, x.shape)
    return jnp.where(mask, x / keep_prob, 0)


def forward_with_dropout(params, x, rng_key, training=True):
    """Forward pass with multiple dropout layers, each with unique RNG."""
    # Split into keys for each layer
    keys = jax.random.split(rng_key, len(params))

    for i, (layer_key, (name, layer_params)) in enumerate(zip(keys, params.items())):
        # Linear transformation
        x = x @ layer_params['w'] + layer_params['b']

        # Activation (except last layer)
        if i < len(params) - 1:
            x = jax.nn.relu(x)
            x = dropout(layer_key, x, rate=0.1, training=training)

    return x


def parallel_random_init(rng_key, num_models, init_fn, *args):
    """Initialize multiple models in parallel with different seeds."""
    keys = jax.random.split(rng_key, num_models)
    return jax.vmap(init_fn, in_axes=(0,) + (None,) * len(args))(keys, *args)


# =============================================================================
# Pattern 4: Immutable Updates with .at[].set()
# =============================================================================

def update_row(matrix, idx, new_row):
    """Update single row immutably."""
    return matrix.at[idx].set(new_row)


def scatter_add(values, indices, updates):
    """Scatter-add updates at indices."""
    return values.at[indices].add(updates)


def masked_update(array, mask, new_values):
    """Update only where mask is True."""
    return jnp.where(mask, new_values, array)


def ring_buffer_append(buffer, idx, value):
    """Append to ring buffer at current position."""
    buffer_size = buffer.shape[0]
    write_idx = idx % buffer_size
    return buffer.at[write_idx].set(value), idx + 1


# =============================================================================
# Pattern 5: Functional Data Pipelines
# =============================================================================

def create_batch_iterator(data, batch_size, rng_key):
    """Create shuffled batch iterator as pure function.

    Returns a function that takes step number and returns batch.
    """
    n_samples = data.shape[0]
    n_batches = n_samples // batch_size

    # Pre-shuffle indices
    indices = jax.random.permutation(rng_key, n_samples)

    def get_batch(step):
        batch_idx = step % n_batches
        start = batch_idx * batch_size
        batch_indices = jax.lax.dynamic_slice(indices, (start,), (batch_size,))
        return data[batch_indices]

    return get_batch


def augment_batch(rng_key, batch, augment_fns):
    """Apply random augmentations functionally."""
    keys = jax.random.split(rng_key, len(augment_fns))

    for key, augment_fn in zip(keys, augment_fns):
        batch = augment_fn(key, batch)

    return batch


# =============================================================================
# Example: Complete Training Loop (Functional Style)
# =============================================================================

def functional_training_loop(
    init_params_fn,
    loss_fn,
    optimizer,
    data,
    num_steps,
    batch_size,
    rng_key,
):
    """Complete training loop with no side effects."""

    # Initialize
    init_key, train_key = jax.random.split(rng_key)
    params = init_params_fn(init_key)
    state = create_train_state(params, optimizer, train_key)

    # Create batch getter
    batch_fn = create_batch_iterator(data, batch_size, train_key)

    # JIT the training step
    @jax.jit
    def jit_train_step(state, batch):
        return train_step(state, batch, optimizer, loss_fn)

    # Training loop using lax.scan for efficiency
    def scan_step(state, step):
        batch = batch_fn(step)
        new_state, loss = jit_train_step(state, batch)
        return new_state, loss

    final_state, losses = jax.lax.scan(
        scan_step,
        state,
        jnp.arange(num_steps)
    )

    return final_state, losses


if __name__ == "__main__":
    # Demo: PyTree operations
    rng = jax.random.PRNGKey(42)
    params = init_params(rng, [784, 256, 128, 10])

    print(f"Total parameters: {count_params(params):,}")
    print(f"Parameter structure: {jax.tree.structure(params)}")

    # Flatten and unflatten
    flat, structure = flatten_params(params)
    print(f"Flattened shape: {flat.shape}")

    restored = unflatten_params(flat, structure)
    assert jax.tree.all(jax.tree.map(jnp.allclose, params, restored))
    print("Flatten/unflatten roundtrip: OK")
