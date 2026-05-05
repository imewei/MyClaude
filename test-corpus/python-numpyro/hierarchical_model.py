"""Hierarchical Bayesian regression with NumPyro and JAX."""

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoNormal
import arviz as az


def hierarchical_model(group_idx: jnp.ndarray, x: jnp.ndarray, y: jnp.ndarray) -> None:
    """Two-level hierarchical model with non-centered parameterization."""
    n_groups = int(group_idx.max()) + 1

    # Hyperpriors
    mu_alpha = numpyro.sample("mu_alpha", dist.Normal(0.0, 5.0))
    sigma_alpha = numpyro.sample("sigma_alpha", dist.HalfNormal(2.0))
    beta = numpyro.sample("beta", dist.Normal(0.0, 1.0))
    sigma = numpyro.sample("sigma", dist.HalfNormal(1.0))

    # Group-level intercepts (non-centered for better NUTS geometry)
    with numpyro.plate("groups", n_groups):
        alpha_raw = numpyro.sample("alpha_raw", dist.Normal(0.0, 1.0))
    alpha = mu_alpha + sigma_alpha * alpha_raw

    # Likelihood
    mu = alpha[group_idx] + beta * x
    numpyro.sample("obs", dist.Normal(mu, sigma), obs=y)


def fit_nuts(group_idx, x, y, num_samples=1000, num_warmup=500, num_chains=4):
    """Run NUTS inference and return an ArviZ-friendly chain."""
    kernel = NUTS(hierarchical_model, target_accept_prob=0.85)
    mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples, num_chains=num_chains)
    mcmc.run(jax.random.PRNGKey(0), group_idx, x, y)
    mcmc.print_summary()
    return az.from_numpyro(mcmc)


def fit_svi(group_idx, x, y, num_steps=2000):
    """Variational inference fallback for large datasets."""
    guide = AutoNormal(hierarchical_model)
    optimizer = numpyro.optim.Adam(step_size=1e-3)
    svi = SVI(hierarchical_model, guide, optimizer, loss=Trace_ELBO())
    svi_result = svi.run(jax.random.PRNGKey(0), num_steps, group_idx, x, y)
    return guide, svi_result.params
