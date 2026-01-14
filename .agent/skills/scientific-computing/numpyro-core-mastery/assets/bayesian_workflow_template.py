#!/usr/bin/env python3
"""
Bayesian Workflow Template

A complete template for Bayesian inference with NumPyro.
Copy and customize for your projects.

Usage:
    python bayesian_workflow_template.py
"""

import jax.numpy as jnp
import jax.random as random
import numpyro
import numpyro.distributions as dist
from numpyro.infer import NUTS, MCMC, Predictive
from numpyro.diagnostics import summary
import matplotlib.pyplot as plt
import arviz as az


# ============================================================================
# STEP 1: DEFINE YOUR MODEL
# ============================================================================

def model(x, y=None):
    """
    Define your Bayesian model here.

    Args:
        x: Independent variable(s)
        y: Dependent variable (None for prior/posterior predictive)

    TODO: Customize this function for your domain
    """
    # Priors (TODO: Adjust based on domain knowledge)
    alpha = numpyro.sample('alpha', dist.Normal(0, 10))
    beta = numpyro.sample('beta', dist.Normal(0, 10))
    sigma = numpyro.sample('sigma', dist.HalfNormal(5))

    # Expected value (TODO: Replace with your model equation)
    mu = alpha + beta * x

    # Likelihood (TODO: Choose appropriate distribution)
    with numpyro.plate('data', len(x)):
        numpyro.sample('obs', dist.Normal(mu, sigma), obs=y)


# ============================================================================
# STEP 2: LOAD YOUR DATA
# ============================================================================

def load_data():
    """
    Load your data here.

    TODO: Replace with your data loading logic

    Returns:
        x: Independent variable(s)
        y: Dependent variable
    """
    # Example: Generate synthetic data (REPLACE THIS)
    rng_key = random.PRNGKey(42)
    N = 100

    key_x, key_noise = random.split(rng_key)
    x = random.uniform(key_x, (N,), minval=0, maxval=10)

    # True model: y = 2 + 3*x + noise
    true_alpha, true_beta, true_sigma = 2.0, 3.0, 0.5
    y = true_alpha + true_beta * x + true_sigma * random.normal(key_noise, (N,))

    return x, y


# ============================================================================
# STEP 3: PRIOR PREDICTIVE CHECK
# ============================================================================

def prior_predictive_check(x, y):
    """
    Check if prior generates reasonable predictions.

    Args:
        x: Independent variable
        y: Observed data (for comparison)
    """
    print("\n" + "="*70)
    print("PRIOR PREDICTIVE CHECK")
    print("="*70)

    prior_predictive = Predictive(model, num_samples=500)
    prior_samples = prior_predictive(random.PRNGKey(1), x, y=None)

    y_prior = prior_samples['obs']

    print(f"Prior predictive range: [{y_prior.min():.2f}, {y_prior.max():.2f}]")
    print(f"Observed data range:    [{y.min():.2f}, {y.max():.2f}]")

    # Visualize
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    for i in range(min(100, len(y_prior))):
        plt.plot(x, y_prior[i], 'C0', alpha=0.05)
    plt.scatter(x, y, c='red', s=20, alpha=0.6, label='Observed')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Prior Predictive Samples')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.hist(y_prior.flatten(), bins=50, alpha=0.5, density=True, label='Prior')
    plt.hist(y, bins=30, alpha=0.5, density=True, label='Observed')
    plt.xlabel('y')
    plt.ylabel('Density')
    plt.title('Prior vs Observed Distribution')
    plt.legend()

    plt.tight_layout()
    plt.savefig('prior_predictive_check.png', dpi=150)
    print("\n✓ Saved: prior_predictive_check.png")


# ============================================================================
# STEP 4: RUN MCMC INFERENCE
# ============================================================================

def run_inference(x, y):
    """
    Run MCMC inference.

    Args:
        x: Independent variable
        y: Dependent variable

    Returns:
        mcmc: MCMC object with results
    """
    print("\n" + "="*70)
    print("RUNNING MCMC INFERENCE")
    print("="*70)

    # Configure NUTS kernel
    nuts_kernel = NUTS(
        model,
        target_accept_prob=0.8,  # TODO: Increase to 0.9-0.95 if divergences
        max_tree_depth=10
    )

    # Configure MCMC
    mcmc = MCMC(
        nuts_kernel,
        num_warmup=1000,    # TODO: Adjust warmup
        num_samples=2000,   # TODO: Adjust samples
        num_chains=4,       # TODO: Adjust chains
        progress_bar=True
    )

    # Run inference
    rng_key = random.PRNGKey(0)  # TODO: Change seed for different runs
    mcmc.run(rng_key, x, y)

    return mcmc


# ============================================================================
# STEP 5: DIAGNOSTICS
# ============================================================================

def check_diagnostics(mcmc):
    """
    Check convergence diagnostics.

    Args:
        mcmc: MCMC object after run()

    Returns:
        bool: True if all diagnostics pass
    """
    print("\n" + "="*70)
    print("CONVERGENCE DIAGNOSTICS")
    print("="*70)

    posterior_samples = mcmc.get_samples()
    summary_dict = summary(posterior_samples, prob=0.95)

    all_good = True

    # Print summary
    mcmc.print_summary(prob=0.95)

    # Check R-hat
    print("\nR-hat Check:")
    for param, stats in summary_dict.items():
        r_hat = stats['r_hat']
        status = "✓" if r_hat < 1.01 else "✗ FAIL"
        print(f"  {param}: {r_hat:.4f} {status}")
        if r_hat > 1.01:
            all_good = False

    # Check ESS
    print("\nEffective Sample Size Check:")
    for param, stats in summary_dict.items():
        n_eff = stats['n_eff']
        status = "✓" if n_eff > 400 else "⚠ LOW"
        print(f"  {param}: {n_eff:.0f} {status}")
        if n_eff < 400:
            all_good = False

    # Check divergences
    print("\nDivergences Check:")
    divergences = mcmc.get_extra_fields()['diverging'].sum()
    print(f"  Total divergences: {divergences}")
    if divergences > 0:
        print("  ✗ Found divergences - increase target_accept_prob to 0.9")
        all_good = False
    else:
        print("  ✓ No divergences")

    if all_good:
        print("\n✓ All diagnostics passed!")
    else:
        print("\n⚠ Some diagnostics failed - see recommendations above")

    return all_good


# ============================================================================
# STEP 6: TRACE PLOTS
# ============================================================================

def plot_traces(mcmc):
    """
    Plot trace plots for visual convergence check.

    Args:
        mcmc: MCMC object after run()
    """
    print("\n" + "="*70)
    print("GENERATING TRACE PLOTS")
    print("="*70)

    idata = az.from_numpyro(mcmc)
    az.plot_trace(idata)
    plt.tight_layout()
    plt.savefig('trace_plots.png', dpi=150)
    print("✓ Saved: trace_plots.png")


# ============================================================================
# STEP 7: POSTERIOR PREDICTIVE CHECK
# ============================================================================

def posterior_predictive_check(mcmc, x, y):
    """
    Check model fit with posterior predictive.

    Args:
        mcmc: MCMC object after run()
        x: Independent variable
        y: Observed data
    """
    print("\n" + "="*70)
    print("POSTERIOR PREDICTIVE CHECK")
    print("="*70)

    posterior_samples = mcmc.get_samples()
    posterior_predictive = Predictive(model, posterior_samples)
    ppc_samples = posterior_predictive(random.PRNGKey(2), x, y=None)

    y_ppc = ppc_samples['obs']

    # Visualize
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.scatter(x, y, c='red', s=20, alpha=0.6, label='Observed')

    # Plot posterior predictive samples
    for i in range(min(100, len(y_ppc))):
        plt.plot(x, y_ppc[i], 'C0', alpha=0.02)

    # Plot mean and credible interval
    y_mean = y_ppc.mean(axis=0)
    y_lower = jnp.percentile(y_ppc, 2.5, axis=0)
    y_upper = jnp.percentile(y_ppc, 97.5, axis=0)

    sorted_idx = jnp.argsort(x)
    plt.plot(x[sorted_idx], y_mean[sorted_idx], 'b-', linewidth=2, label='Mean')
    plt.fill_between(x[sorted_idx], y_lower[sorted_idx], y_upper[sorted_idx],
                     alpha=0.3, color='blue', label='95% CI')

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Posterior Predictive Fit')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.hist(y_ppc.flatten(), bins=50, alpha=0.5, density=True, label='Posterior Predictive')
    plt.hist(y, bins=30, alpha=0.5, density=True, label='Observed')
    plt.xlabel('y')
    plt.ylabel('Density')
    plt.title('Posterior Predictive Check')
    plt.legend()

    plt.tight_layout()
    plt.savefig('posterior_predictive_check.png', dpi=150)
    print("✓ Saved: posterior_predictive_check.png")


# ============================================================================
# STEP 8: EXTRACT RESULTS
# ============================================================================

def extract_results(mcmc):
    """
    Extract and summarize results.

    Args:
        mcmc: MCMC object after run()

    Returns:
        dict: Results summary
    """
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)

    posterior_samples = mcmc.get_samples()

    results = {}

    for param_name, samples in posterior_samples.items():
        mean = samples.mean()
        std = samples.std()
        lower = jnp.percentile(samples, 2.5)
        upper = jnp.percentile(samples, 97.5)

        results[param_name] = {
            'mean': float(mean),
            'std': float(std),
            'ci_lower': float(lower),
            'ci_upper': float(upper)
        }

        print(f"\n{param_name}:")
        print(f"  Mean: {mean:.4f}")
        print(f"  Std:  {std:.4f}")
        print(f"  95% CI: [{lower:.4f}, {upper:.4f}]")

    return results


# ============================================================================
# MAIN WORKFLOW
# ============================================================================

def main():
    """
    Complete Bayesian workflow.
    """
    print("\n" + "="*70)
    print("BAYESIAN WORKFLOW WITH NUMPYRO")
    print("="*70)

    # Step 1: Load data
    print("\n1. Loading data...")
    x, y = load_data()
    print(f"   Loaded {len(x)} observations")

    # Step 2: Prior predictive check
    print("\n2. Prior predictive check...")
    prior_predictive_check(x, y)

    # Step 3: Run inference
    print("\n3. Running MCMC inference...")
    mcmc = run_inference(x, y)

    # Step 4: Check diagnostics
    print("\n4. Checking diagnostics...")
    diagnostics_pass = check_diagnostics(mcmc)

    # Step 5: Trace plots
    print("\n5. Generating trace plots...")
    plot_traces(mcmc)

    # Step 6: Posterior predictive check
    print("\n6. Posterior predictive check...")
    posterior_predictive_check(mcmc, x, y)

    # Step 7: Extract results
    print("\n7. Extracting results...")
    results = extract_results(mcmc)

    # Final summary
    print("\n" + "="*70)
    print("WORKFLOW COMPLETE")
    print("="*70)
    print("\nGenerated files:")
    print("  - prior_predictive_check.png")
    print("  - trace_plots.png")
    print("  - posterior_predictive_check.png")

    if diagnostics_pass:
        print("\n✓ All diagnostics passed - results are reliable!")
    else:
        print("\n⚠ Some diagnostics failed - review output above")

    print("\n" + "="*70)


if __name__ == "__main__":
    main()
