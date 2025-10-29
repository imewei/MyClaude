#!/usr/bin/env python3
"""
Hierarchical Model Template

Template for hierarchical/multilevel Bayesian models with NumPyro.
Demonstrates partial pooling and group-level inference.

Usage:
    python hierarchical_model_template.py
"""

import jax.numpy as jnp
import jax.random as random
import numpyro
import numpyro.distributions as dist
from numpyro.infer import NUTS, MCMC, Predictive
import matplotlib.pyplot as plt


def hierarchical_model(group_idx, x, y=None):
    """
    Hierarchical linear regression with partial pooling.

    Model:
        y_ij ~ Normal(α_j + β_j * x_ij, σ)
        α_j ~ Normal(μ_α, σ_α)  # Group-level intercepts
        β_j ~ Normal(μ_β, σ_β)  # Group-level slopes

    Args:
        group_idx: Group identifiers (0, 1, 2, ...)
        x: Predictors
        y: Responses (None for prior/posterior predictive)

    TODO: Customize for your hierarchical structure
    """
    n_groups = len(jnp.unique(group_idx))

    # Global hyperpriors (population-level)
    mu_alpha = numpyro.sample('mu_alpha', dist.Normal(0, 10))
    sigma_alpha = numpyro.sample('sigma_alpha', dist.HalfNormal(5))

    mu_beta = numpyro.sample('mu_beta', dist.Normal(0, 10))
    sigma_beta = numpyro.sample('sigma_beta', dist.HalfNormal(5))

    # Group-level parameters (partial pooling)
    with numpyro.plate('groups', n_groups):
        alpha = numpyro.sample('alpha', dist.Normal(mu_alpha, sigma_alpha))
        beta = numpyro.sample('beta', dist.Normal(mu_beta, sigma_beta))

    # Observation-level noise
    sigma = numpyro.sample('sigma', dist.HalfNormal(5))

    # Likelihood
    mu = alpha[group_idx] + beta[group_idx] * x

    with numpyro.plate('data', len(x)):
        numpyro.sample('obs', dist.Normal(mu, sigma), obs=y)


def generate_hierarchical_data():
    """
    Generate synthetic hierarchical data.

    TODO: Replace with your actual data loading

    Returns:
        group_idx: Group identifiers
        x: Predictors
        y: Responses
    """
    rng_key = random.PRNGKey(42)

    # Setup
    n_groups = 5
    n_per_group = 20
    N = n_groups * n_per_group

    # Group assignments
    group_idx = jnp.repeat(jnp.arange(n_groups), n_per_group)

    # Predictors (same distribution across groups)
    key_x = random.split(rng_key)[0]
    x = random.normal(key_x, (N,))

    # True hierarchical structure
    true_mu_alpha = 2.0
    true_sigma_alpha = 1.5
    key_alpha = random.split(rng_key)[1]
    true_alpha = true_mu_alpha + true_sigma_alpha * random.normal(key_alpha, (n_groups,))

    true_mu_beta = 3.0
    true_sigma_beta = 0.8
    key_beta = random.split(rng_key)[2]
    true_beta = true_mu_beta + true_sigma_beta * random.normal(key_beta, (n_groups,))

    true_sigma = 0.5

    # Generate responses
    key_noise = random.split(rng_key)[3]
    y = true_alpha[group_idx] + true_beta[group_idx] * x + \
        true_sigma * random.normal(key_noise, (N,))

    print(f"Generated hierarchical data:")
    print(f"  Groups: {n_groups}")
    print(f"  Observations per group: {n_per_group}")
    print(f"  Total observations: {N}")
    print(f"  True population intercept: {true_mu_alpha:.2f} ± {true_sigma_alpha:.2f}")
    print(f"  True population slope: {true_mu_beta:.2f} ± {true_sigma_beta:.2f}")

    return group_idx, x, y, true_alpha, true_beta


def run_hierarchical_inference(group_idx, x, y):
    """Run MCMC inference for hierarchical model."""
    print("\nRunning hierarchical MCMC...")

    nuts_kernel = NUTS(hierarchical_model, target_accept_prob=0.9)  # Higher for hierarchical
    mcmc = MCMC(nuts_kernel, num_warmup=1000, num_samples=2000, num_chains=4)

    mcmc.run(random.PRNGKey(0), group_idx, x, y)
    mcmc.print_summary(prob=0.95)

    return mcmc


def visualize_shrinkage(mcmc, true_alpha, true_beta, n_groups):
    """
    Visualize hierarchical shrinkage effect.

    Shows how group estimates are "shrunk" toward population mean.
    """
    posterior_samples = mcmc.get_samples()

    # Extract estimates
    alpha_samples = posterior_samples['alpha']  # (n_samples, n_groups)
    beta_samples = posterior_samples['beta']

    alpha_mean = alpha_samples.mean(axis=0)
    beta_mean = beta_samples.mean(axis=0)

    mu_alpha_mean = posterior_samples['mu_alpha'].mean()
    mu_beta_mean = posterior_samples['mu_beta'].mean()

    # Plot shrinkage
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Intercept shrinkage
    axes[0].scatter(true_alpha, alpha_mean, s=100, alpha=0.6, label='Group estimates')
    axes[0].axhline(mu_alpha_mean, color='red', linestyle='--', linewidth=2,
                   label=f'Population mean ({mu_alpha_mean:.2f})')
    axes[0].plot([-5, 5], [-5, 5], 'k--', alpha=0.3, label='Perfect recovery')
    axes[0].set_xlabel('True Group Intercept')
    axes[0].set_ylabel('Estimated Group Intercept')
    axes[0].set_title('Hierarchical Shrinkage: Intercepts')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Slope shrinkage
    axes[1].scatter(true_beta, beta_mean, s=100, alpha=0.6, label='Group estimates')
    axes[1].axhline(mu_beta_mean, color='red', linestyle='--', linewidth=2,
                   label=f'Population mean ({mu_beta_mean:.2f})')
    axes[1].plot([0, 5], [0, 5], 'k--', alpha=0.3, label='Perfect recovery')
    axes[1].set_xlabel('True Group Slope')
    axes[1].set_ylabel('Estimated Group Slope')
    axes[1].set_title('Hierarchical Shrinkage: Slopes')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('hierarchical_shrinkage.png', dpi=150)
    print("✓ Saved: hierarchical_shrinkage.png")


def plot_group_fits(group_idx, x, y, mcmc, n_groups):
    """Plot fits for each group."""
    posterior_samples = mcmc.get_samples()

    alpha_mean = posterior_samples['alpha'].mean(axis=0)
    beta_mean = posterior_samples['beta'].mean(axis=0)

    fig, axes = plt.subplots(1, n_groups, figsize=(15, 3))

    for j in range(n_groups):
        ax = axes[j] if n_groups > 1 else axes

        # Data for this group
        group_mask = group_idx == j
        x_group = x[group_mask]
        y_group = y[group_mask]

        # Fit
        x_range = jnp.linspace(x_group.min(), x_group.max(), 100)
        y_pred = alpha_mean[j] + beta_mean[j] * x_range

        # Plot
        ax.scatter(x_group, y_group, alpha=0.6, s=30)
        ax.plot(x_range, y_pred, 'r-', linewidth=2)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(f'Group {j}')
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('group_fits.png', dpi=150)
    print("✓ Saved: group_fits.png")


def main():
    """Complete hierarchical modeling workflow."""
    print("="*70)
    print("HIERARCHICAL BAYESIAN MODELING WITH NUMPYRO")
    print("="*70)

    # Generate data
    group_idx, x, y, true_alpha, true_beta = generate_hierarchical_data()
    n_groups = len(jnp.unique(group_idx))

    # Run inference
    mcmc = run_hierarchical_inference(group_idx, x, y)

    # Visualize shrinkage
    print("\nVisualizing hierarchical shrinkage...")
    visualize_shrinkage(mcmc, true_alpha, true_beta, n_groups)

    # Plot group fits
    print("\nPlotting group-level fits...")
    plot_group_fits(group_idx, x, y, mcmc, n_groups)

    print("\n" + "="*70)
    print("HIERARCHICAL MODELING COMPLETE")
    print("="*70)
    print("\nKey concepts demonstrated:")
    print("  1. Partial pooling - groups share information")
    print("  2. Shrinkage - estimates pulled toward population mean")
    print("  3. Group-level variation - each group has own parameters")
    print("  4. Population-level inference - μ_α, μ_β, σ_α, σ_β")
    print("\nGenerated files:")
    print("  - hierarchical_shrinkage.png")
    print("  - group_fits.png")
    print("="*70)


if __name__ == "__main__":
    main()
