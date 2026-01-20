"""
Custom MCMC Samplers with Blackjax

Demonstrates low-level control over MCMC inference including
custom kernels, mixed samplers, and parallel chains.
"""

import jax
import jax.numpy as jnp
from jax.scipy import stats
import blackjax
from functools import partial
from typing import NamedTuple, Callable, Dict, Any


# =============================================================================
# Pattern 1: Basic Custom NUTS Loop
# =============================================================================

def custom_nuts_sampler(log_prob_fn: Callable,
                        initial_position: jnp.ndarray,
                        num_warmup: int = 500,
                        num_samples: int = 1000,
                        rng_key: jnp.ndarray = None) -> Dict[str, Any]:
    """Full custom NUTS implementation with diagnostics."""

    if rng_key is None:
        rng_key = jax.random.PRNGKey(0)

    rng_key, warmup_key, sample_key = jax.random.split(rng_key, 3)

    # === WARMUP: Adapt step size and mass matrix ===
    warmup = blackjax.window_adaptation(
        blackjax.nuts,
        log_prob_fn,
        is_mass_matrix_diagonal=True,
        initial_step_size=1.0,
        target_acceptance_rate=0.8,
    )

    (state, kernel_params), warmup_info = warmup.run(
        warmup_key,
        initial_position,
        num_steps=num_warmup
    )

    print(f"Warmup complete:")
    print(f"  Step size: {kernel_params['step_size']:.4f}")
    print(f"  Mass matrix diag (first 5): {jnp.diag(kernel_params['inverse_mass_matrix'])[:5]}")

    # === SAMPLING ===
    nuts_kernel = blackjax.nuts(log_prob_fn, **kernel_params).step

    # Custom step function that collects diagnostics
    def step_fn(state, key):
        new_state, info = nuts_kernel(key, state)
        return new_state, {
            'position': new_state.position,
            'log_prob': new_state.logdensity,
            'is_divergent': info.is_divergent,
            'acceptance_rate': info.acceptance_rate,
            'num_integration_steps': info.num_integration_steps,
        }

    sample_keys = jax.random.split(sample_key, num_samples)
    final_state, samples = jax.lax.scan(step_fn, state, sample_keys)

    # Compute diagnostics
    n_divergent = jnp.sum(samples['is_divergent'])
    mean_accept = jnp.mean(samples['acceptance_rate'])
    mean_steps = jnp.mean(samples['num_integration_steps'])

    print(f"\nSampling complete ({num_samples} samples):")
    print(f"  Divergences: {n_divergent}")
    print(f"  Mean acceptance rate: {mean_accept:.3f}")
    print(f"  Mean integration steps: {mean_steps:.1f}")

    return {
        'samples': samples['position'],
        'log_probs': samples['log_prob'],
        'diagnostics': samples,
        'kernel_params': kernel_params,
    }


# =============================================================================
# Pattern 2: Mixed Gibbs + HMC Sampler
# =============================================================================

class MixedSamplerState(NamedTuple):
    """State for mixed Gibbs/HMC sampler."""
    continuous_position: jnp.ndarray
    discrete_position: jnp.ndarray
    hmc_state: Any  # Blackjax HMC state


def create_mixed_sampler(continuous_log_prob: Callable,
                         discrete_update: Callable,
                         hmc_params: Dict) -> Callable:
    """Create sampler that mixes HMC (continuous) with Gibbs (discrete).

    Args:
        continuous_log_prob: log p(continuous | discrete, data)
        discrete_update: function(key, discrete, continuous) -> new_discrete
        hmc_params: dict with 'step_size', 'inverse_mass_matrix', 'num_integration_steps'
    """

    hmc_kernel = blackjax.hmc(continuous_log_prob, **hmc_params).step

    def mixed_step(state: MixedSamplerState, key: jnp.ndarray) -> MixedSamplerState:
        key1, key2 = jax.random.split(key)

        # Step 1: HMC update for continuous parameters
        new_hmc_state, hmc_info = hmc_kernel(key1, state.hmc_state)

        # Step 2: Gibbs update for discrete parameters
        new_discrete = discrete_update(key2, state.discrete_position, new_hmc_state.position)

        return MixedSamplerState(
            continuous_position=new_hmc_state.position,
            discrete_position=new_discrete,
            hmc_state=new_hmc_state,
        ), hmc_info

    return mixed_step


# =============================================================================
# Pattern 3: Parallel Chains with Different Initializations
# =============================================================================

def run_parallel_chains(log_prob_fn: Callable,
                        initial_positions: jnp.ndarray,
                        num_warmup: int = 500,
                        num_samples: int = 1000,
                        rng_key: jnp.ndarray = None) -> Dict:
    """Run multiple chains in parallel using vmap.

    Args:
        initial_positions: Shape (n_chains, param_dim)
    """
    n_chains = initial_positions.shape[0]
    if rng_key is None:
        rng_key = jax.random.PRNGKey(0)

    chain_keys = jax.random.split(rng_key, n_chains)

    def run_single_chain(init_pos, key):
        warmup = blackjax.window_adaptation(blackjax.nuts, log_prob_fn)
        warmup_key, sample_key = jax.random.split(key)

        (state, params), _ = warmup.run(warmup_key, init_pos, num_steps=num_warmup)
        kernel = blackjax.nuts(log_prob_fn, **params).step

        def step(state, key):
            state, info = kernel(key, state)
            return state, (state.position, info.is_divergent)

        sample_keys = jax.random.split(sample_key, num_samples)
        _, (samples, divergent) = jax.lax.scan(step, state, sample_keys)

        return samples, divergent

    # Vectorize over chains
    all_samples, all_divergent = jax.vmap(run_single_chain)(initial_positions, chain_keys)

    # Shape: (n_chains, num_samples, param_dim)
    print(f"Ran {n_chains} chains, {num_samples} samples each")
    print(f"Total divergences: {jnp.sum(all_divergent)}")

    return {
        'samples': all_samples,
        'divergent': all_divergent,
    }


# =============================================================================
# Pattern 4: Parallel Chains with pmap (Multi-Device)
# =============================================================================

def run_pmap_chains(log_prob_fn: Callable,
                    initial_positions: jnp.ndarray,
                    num_warmup: int = 500,
                    num_samples: int = 1000,
                    rng_key: jnp.ndarray = None) -> jnp.ndarray:
    """Run chains across multiple devices using pmap.

    Args:
        initial_positions: Shape (n_devices, param_dim)
    """
    n_devices = jax.device_count()
    assert initial_positions.shape[0] == n_devices, \
        f"Need {n_devices} initial positions for {n_devices} devices"

    if rng_key is None:
        rng_key = jax.random.PRNGKey(0)

    device_keys = jax.random.split(rng_key, n_devices)

    @jax.pmap
    def run_chain_on_device(init_pos, key):
        warmup = blackjax.window_adaptation(blackjax.nuts, log_prob_fn)
        warmup_key, sample_key = jax.random.split(key)

        (state, params), _ = warmup.run(warmup_key, init_pos, num_steps=num_warmup)
        kernel = blackjax.nuts(log_prob_fn, **params).step

        def step(state, key):
            state, _ = kernel(key, state)
            return state, state.position

        sample_keys = jax.random.split(sample_key, num_samples)
        _, samples = jax.lax.scan(step, state, sample_keys)

        return samples

    # Run on all devices
    all_samples = run_chain_on_device(initial_positions, device_keys)

    print(f"Ran {n_devices} chains on {n_devices} devices")
    return all_samples  # Shape: (n_devices, num_samples, param_dim)


# =============================================================================
# Pattern 5: Adaptive Step Size During Sampling
# =============================================================================

def dual_averaging_step_size(log_prob_fn: Callable,
                              initial_position: jnp.ndarray,
                              target_accept: float = 0.8,
                              num_adapt: int = 100,
                              rng_key: jnp.ndarray = None) -> float:
    """Find optimal step size using dual averaging."""

    if rng_key is None:
        rng_key = jax.random.PRNGKey(0)

    # Initial guess
    step_size = 1.0
    log_step_size = jnp.log(step_size)
    log_step_size_bar = 0.0
    h_bar = 0.0

    # Dual averaging parameters
    gamma = 0.05
    t0 = 10
    kappa = 0.75
    mu = jnp.log(10 * step_size)

    # Simple HMC kernel for adaptation
    inverse_mass_matrix = jnp.eye(len(initial_position))

    state = blackjax.hmc.init(initial_position, log_prob_fn)

    for m in range(1, num_adapt + 1):
        rng_key, key = jax.random.split(rng_key)

        kernel = blackjax.hmc(
            log_prob_fn,
            step_size=jnp.exp(log_step_size),
            inverse_mass_matrix=inverse_mass_matrix,
            num_integration_steps=10,
        )

        state, info = kernel.step(key, state)
        accept_rate = info.acceptance_rate

        # Dual averaging update
        w = 1.0 / (m + t0)
        h_bar = (1 - w) * h_bar + w * (target_accept - accept_rate)
        log_step_size = mu - jnp.sqrt(m) / gamma * h_bar
        m_w = m ** (-kappa)
        log_step_size_bar = m_w * log_step_size + (1 - m_w) * log_step_size_bar

    final_step_size = jnp.exp(log_step_size_bar)
    print(f"Adapted step size: {final_step_size:.4f}")
    return final_step_size


# =============================================================================
# Pattern 6: Custom Kernel with Tempering
# =============================================================================

def tempered_log_prob(log_prob_fn: Callable, temperature: float) -> Callable:
    """Create tempered version of log-prob: log_prob / T."""
    def tempered(position):
        return log_prob_fn(position) / temperature
    return tempered


def parallel_tempering_step(log_prob_fn: Callable,
                             states: list,
                             temperatures: jnp.ndarray,
                             kernels: list,
                             key: jnp.ndarray) -> list:
    """Single step of parallel tempering.

    Run each chain at its temperature, then propose swaps.
    """
    n_temps = len(temperatures)
    keys = jax.random.split(key, n_temps + n_temps - 1)

    # Step each chain
    new_states = []
    for i, (state, kernel, k) in enumerate(zip(states, kernels, keys[:n_temps])):
        new_state, _ = kernel.step(k, state)
        new_states.append(new_state)

    # Propose swaps between adjacent temperatures
    for i in range(n_temps - 1):
        swap_key = keys[n_temps + i]

        # Swap acceptance probability
        log_prob_i = log_prob_fn(new_states[i].position)
        log_prob_j = log_prob_fn(new_states[i + 1].position)

        beta_i, beta_j = 1 / temperatures[i], 1 / temperatures[i + 1]

        log_accept = (beta_j - beta_i) * (log_prob_i - log_prob_j)

        if jax.random.uniform(swap_key) < jnp.exp(log_accept):
            # Swap positions
            new_states[i], new_states[i + 1] = new_states[i + 1], new_states[i]

    return new_states


# =============================================================================
# Demo
# =============================================================================

def demo():
    """Demonstrate custom MCMC patterns."""
    print("=" * 60)
    print("Custom MCMC Samplers Demo")
    print("=" * 60)

    # Simple target distribution
    def log_prob(position):
        # Correlated 2D Gaussian
        mu = jnp.array([0.0, 0.0])
        cov = jnp.array([[1.0, 0.8], [0.8, 1.0]])
        diff = position - mu
        return -0.5 * diff @ jnp.linalg.solve(cov, diff)

    # Run custom NUTS
    print("\n1. Custom NUTS Sampler")
    print("-" * 40)
    initial = jnp.array([5.0, -5.0])
    result = custom_nuts_sampler(log_prob, initial, num_warmup=200, num_samples=500)

    samples = result['samples']
    print(f"\nPosterior mean: {jnp.mean(samples, axis=0)}")
    print(f"Posterior std: {jnp.std(samples, axis=0)}")

    # Run parallel chains with vmap
    print("\n2. Parallel Chains (vmap)")
    print("-" * 40)
    n_chains = 4
    initial_positions = jax.random.normal(jax.random.PRNGKey(0), (n_chains, 2)) * 3
    parallel_result = run_parallel_chains(log_prob, initial_positions,
                                          num_warmup=200, num_samples=500)

    # Pool all chains
    pooled = parallel_result['samples'].reshape(-1, 2)
    print(f"Pooled posterior mean: {jnp.mean(pooled, axis=0)}")


if __name__ == "__main__":
    demo()
