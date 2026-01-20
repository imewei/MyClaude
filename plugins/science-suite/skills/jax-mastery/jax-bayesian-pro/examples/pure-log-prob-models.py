"""
Pure Log-Prob Models for JAX

Demonstrates how to write Bayesian models as pure functions without
any probabilistic programming language syntax - just vanilla JAX.
"""

import jax
import jax.numpy as jnp
from jax.scipy import stats
from typing import Dict, Any
from functools import partial


# =============================================================================
# Pattern 1: Basic Pure Log-Prob Model
# =============================================================================

def simple_regression_log_prob(params: Dict[str, jnp.ndarray],
                                data: Dict[str, jnp.ndarray]) -> float:
    """Pure log-probability for simple linear regression.

    No magic contexts, no hidden state. Just a function.
    """
    # Extract parameters
    beta = params['beta']          # Regression coefficients
    log_sigma = params['log_sigma']  # Log of noise std (unconstrained)
    sigma = jnp.exp(log_sigma)

    # Extract data
    X, y = data['X'], data['y']

    # Prior log-probabilities
    log_prior = (
        jnp.sum(stats.norm.logpdf(beta, 0, 10)) +  # beta ~ N(0, 10)
        stats.norm.logpdf(log_sigma, 0, 2)          # log_sigma ~ N(0, 2)
    )

    # Likelihood
    mu = X @ beta
    log_lik = jnp.sum(stats.norm.logpdf(y, mu, sigma))

    return log_prior + log_lik


# =============================================================================
# Pattern 2: Hierarchical Model as Pure Function
# =============================================================================

def hierarchical_log_prob(params: Dict[str, jnp.ndarray],
                          data: Dict[str, jnp.ndarray]) -> float:
    """Pure log-prob for hierarchical (random effects) model.

    Model:
        mu_0 ~ N(0, 10)
        sigma_0 ~ HalfNormal(5)
        mu_group[k] ~ N(mu_0, sigma_0)
        sigma ~ HalfNormal(2)
        y[i] ~ N(mu_group[group[i]], sigma)
    """
    # Parameters
    mu_0 = params['mu_0']
    log_sigma_0 = params['log_sigma_0']
    sigma_0 = jnp.exp(log_sigma_0)

    mu_group = params['mu_group']  # Shape: (n_groups,)

    log_sigma = params['log_sigma']
    sigma = jnp.exp(log_sigma)

    # Data
    y = data['y']
    group_idx = data['group_idx']

    # === PRIORS ===
    # Hyperpriors
    log_prior_mu_0 = stats.norm.logpdf(mu_0, 0, 10)

    # log_sigma_0 prior: transformed HalfNormal
    # If log_sigma_0 ~ N(0, 2), then sigma_0 ~ LogNormal
    # For HalfNormal(5), we use an adjustment
    log_prior_log_sigma_0 = stats.norm.logpdf(log_sigma_0, 0, 2)

    # Group-level priors
    log_prior_mu_group = jnp.sum(stats.norm.logpdf(mu_group, mu_0, sigma_0))

    # Observation noise prior
    log_prior_log_sigma = stats.norm.logpdf(log_sigma, 0, 1)

    # === LIKELIHOOD ===
    group_means = mu_group[group_idx]
    log_lik = jnp.sum(stats.norm.logpdf(y, group_means, sigma))

    # Total log probability
    log_prob = (
        log_prior_mu_0 +
        log_prior_log_sigma_0 +
        log_prior_mu_group +
        log_prior_log_sigma +
        log_lik
    )

    return log_prob


# =============================================================================
# Pattern 3: Vectorized Log-Prob for Batch Data
# =============================================================================

def single_observation_log_lik(params: Dict, obs: jnp.ndarray) -> float:
    """Log-likelihood for a single observation."""
    mu, sigma = params['mu'], jnp.exp(params['log_sigma'])
    return stats.norm.logpdf(obs, mu, sigma)


def batched_log_prob(params: Dict, data: jnp.ndarray) -> float:
    """Log-prob vectorized over batch dimension.

    This is efficient: vmap allows processing all data in parallel.
    """
    # Prior
    log_prior = (
        stats.norm.logpdf(params['mu'], 0, 10) +
        stats.norm.logpdf(params['log_sigma'], 0, 2)
    )

    # Vectorized likelihood
    log_liks = jax.vmap(partial(single_observation_log_lik, params))(data)
    log_lik = jnp.sum(log_liks)

    return log_prior + log_lik


# =============================================================================
# Pattern 4: Masked Log-Prob for Ragged Data
# =============================================================================

def masked_log_prob(params: Dict,
                    padded_data: jnp.ndarray,
                    mask: jnp.ndarray) -> float:
    """Log-prob with masking for variable-length sequences.

    Args:
        params: Model parameters
        padded_data: Shape (batch, max_len), padded with zeros
        mask: Shape (batch, max_len), True for valid entries
    """
    mu = params['mu']
    sigma = jnp.exp(params['log_sigma'])

    # Prior
    log_prior = (
        stats.norm.logpdf(mu, 0, 10) +
        stats.norm.logpdf(params['log_sigma'], 0, 2)
    )

    # Compute log-lik for all entries (including padding)
    all_log_liks = stats.norm.logpdf(padded_data, mu, sigma)

    # Zero out padded entries
    masked_log_liks = jnp.where(mask, all_log_liks, 0.0)

    # Sum only valid entries
    log_lik = jnp.sum(masked_log_liks)

    return log_prior + log_lik


# =============================================================================
# Pattern 5: Mixture Model Log-Prob
# =============================================================================

def mixture_log_prob(params: Dict, data: jnp.ndarray) -> float:
    """Log-prob for Gaussian mixture model.

    Uses log-sum-exp trick for numerical stability.
    """
    # Parameters
    log_weights = params['log_weights']  # (K,) unnormalized log weights
    mus = params['mus']                   # (K,) component means
    log_sigmas = params['log_sigmas']     # (K,) log component stds

    K = len(log_weights)
    sigmas = jnp.exp(log_sigmas)

    # Normalize weights
    log_weights_normalized = log_weights - jax.scipy.special.logsumexp(log_weights)

    # Prior
    log_prior = (
        jnp.sum(stats.norm.logpdf(mus, 0, 10)) +
        jnp.sum(stats.norm.logpdf(log_sigmas, 0, 2))
    )

    # Likelihood with log-sum-exp
    def single_obs_log_lik(x):
        # Log probability under each component
        log_component_probs = stats.norm.logpdf(x, mus, sigmas)
        # Weighted sum in log space
        return jax.scipy.special.logsumexp(log_weights_normalized + log_component_probs)

    log_lik = jnp.sum(jax.vmap(single_obs_log_lik)(data))

    return log_prior + log_lik


# =============================================================================
# Pattern 6: Non-Centered Parameterization
# =============================================================================

def funnel_log_prob_centered(params: Dict) -> float:
    """Centered parameterization - has funnel geometry."""
    tau = jnp.exp(params['log_tau'])  # tau > 0
    x = params['x']  # (n,)

    log_prior_tau = stats.halfnorm.logpdf(tau, scale=3) + params['log_tau']  # Jacobian
    log_prior_x = jnp.sum(stats.norm.logpdf(x, 0, tau))

    return log_prior_tau + log_prior_x


def funnel_log_prob_noncentered(params: Dict) -> float:
    """Non-centered parameterization - no funnel."""
    log_tau = params['log_tau']
    tau = jnp.exp(log_tau)
    z = params['z']  # (n,) - standard normal latent

    # Priors on transformed parameters
    log_prior_log_tau = stats.norm.logpdf(log_tau, 0, 1)
    log_prior_z = jnp.sum(stats.norm.logpdf(z, 0, 1))

    # The actual x = tau * z (deterministic transformation)
    # No need to include in log_prob since we're sampling z, not x

    return log_prior_log_tau + log_prior_z


# =============================================================================
# Pattern 7: Transformations and Jacobians
# =============================================================================

def transformed_params_log_prob(unconstrained_params: Dict, data: jnp.ndarray) -> float:
    """Log-prob with proper Jacobian corrections for transformations.

    We sample unconstrained parameters and transform to constrained space.
    Must include Jacobian of transformation in log-prob.
    """
    # Unconstrained -> Constrained transformations
    mu = unconstrained_params['mu']  # Already unconstrained

    # sigma > 0: use exp transform
    log_sigma = unconstrained_params['log_sigma']
    sigma = jnp.exp(log_sigma)

    # rho in (-1, 1): use tanh transform
    unconstrained_rho = unconstrained_params['unconstrained_rho']
    rho = jnp.tanh(unconstrained_rho)

    # === PRIORS IN CONSTRAINED SPACE ===
    # But we're sampling in unconstrained space, so add Jacobians

    # mu ~ N(0, 10) - no transform needed
    log_prior_mu = stats.norm.logpdf(mu, 0, 10)

    # sigma ~ HalfNormal(5)
    # Prior in constrained space + Jacobian of exp transform
    log_prior_sigma = stats.halfnorm.logpdf(sigma, scale=5) + log_sigma
    # The +log_sigma is |d(sigma)/d(log_sigma)| = exp(log_sigma) = sigma
    # in log space: log(sigma) = log_sigma

    # rho ~ Uniform(-1, 1) implies unconstrained_rho ~ Logistic(0, 1)
    # Jacobian of tanh: d(rho)/d(u) = 1 - tanh(u)^2
    log_prior_rho = -2 * jnp.log(jnp.cosh(unconstrained_rho))  # Logistic log-prob

    # === LIKELIHOOD ===
    log_lik = jnp.sum(stats.norm.logpdf(data, mu, sigma))

    return log_prior_mu + log_prior_sigma + log_prior_rho + log_lik


# =============================================================================
# Demo: Using with Blackjax
# =============================================================================

def demo_with_blackjax():
    """Show how to use pure log-prob with Blackjax."""
    import blackjax

    # Generate fake data
    rng = jax.random.PRNGKey(42)
    true_mu, true_sigma = 2.5, 1.5
    data = jax.random.normal(rng, shape=(100,)) * true_sigma + true_mu

    # Define log-prob (close over data)
    def log_prob(params):
        return batched_log_prob(params, data)

    # Initial parameters
    initial_params = {
        'mu': 0.0,
        'log_sigma': 0.0,
    }

    # Flatten for Blackjax (expects array, not dict)
    def pack(params):
        return jnp.array([params['mu'], params['log_sigma']])

    def unpack(arr):
        return {'mu': arr[0], 'log_sigma': arr[1]}

    def packed_log_prob(arr):
        return log_prob(unpack(arr))

    # Run NUTS
    warmup = blackjax.window_adaptation(blackjax.nuts, packed_log_prob)
    rng_key = jax.random.PRNGKey(0)
    (state, params), _ = warmup.run(rng_key, pack(initial_params), num_steps=500)

    kernel = blackjax.nuts(packed_log_prob, **params).step

    def step(state, key):
        state, info = kernel(key, state)
        return state, state.position

    keys = jax.random.split(rng_key, 1000)
    _, samples = jax.lax.scan(step, state, keys)

    # Unpack samples
    mu_samples = samples[:, 0]
    sigma_samples = jnp.exp(samples[:, 1])

    print(f"True mu: {true_mu}, Estimated: {jnp.mean(mu_samples):.3f} ± {jnp.std(mu_samples):.3f}")
    print(f"True sigma: {true_sigma}, Estimated: {jnp.mean(sigma_samples):.3f} ± {jnp.std(sigma_samples):.3f}")


if __name__ == "__main__":
    print("Pure Log-Prob Models Demo")
    print("=" * 50)
    demo_with_blackjax()
