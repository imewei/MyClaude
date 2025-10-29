#!/usr/bin/env python3
"""
Workflow Pattern 4: Bayesian Inference with NumPyro

Demonstrates NumPyro integration with JAX for Bayesian modeling.

Note: For comprehensive NumPyro workflows, see the 'numpyro-core-mastery' skill.
"""

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, Predictive
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt


def generate_synthetic_data(n=100):
    """Generate synthetic regression data"""
    rng = jax.random.PRNGKey(0)
    x_key, noise_key = jax.random.split(rng)

    # True parameters
    true_w = 2.5
    true_b = 1.0
    true_sigma = 0.5

    # Generate data
    x = jax.random.uniform(x_key, (n,), minval=-3, maxval=3)
    y = true_w * x + true_b + true_sigma * jax.random.normal(noise_key, (n,))

    return x, y, (true_w, true_b, true_sigma)


def bayesian_linear_model(x, y=None):
    """Bayesian linear regression model"""

    # Priors
    w = numpyro.sample('w', dist.Normal(0, 10))
    b = numpyro.sample('b', dist.Normal(0, 10))
    sigma = numpyro.sample('sigma', dist.HalfNormal(5))

    # Likelihood
    mean = w * x + b
    with numpyro.plate('data', len(x)):
        numpyro.sample('y', dist.Normal(mean, sigma), obs=y)


def example_bayesian_inference():
    """Complete Bayesian inference workflow with NumPyro"""

    print("=" * 60)
    print("Workflow Pattern 4: Bayesian Inference with NumPyro")
    print("=" * 60)

    # Step 1: Generate synthetic data
    print("\nStep 1: Generate synthetic data")
    x_train, y_train, true_params = generate_synthetic_data(n=100)
    true_w, true_b, true_sigma = true_params

    print(f"Training data: {len(x_train)} points")
    print(f"True parameters: w={true_w:.2f}, b={true_b:.2f}, sigma={true_sigma:.2f}")

    # Step 2: MCMC inference with NUTS
    print("\nStep 2: Run MCMC inference with NUTS")

    # Setup NUTS kernel
    nuts_kernel = NUTS(bayesian_linear_model)

    # MCMC sampler (JAX automatically parallelizes chains)
    mcmc = MCMC(
        nuts_kernel,
        num_warmup=500,
        num_samples=1000,
        num_chains=4  # Parallel chains
    )

    # Run inference
    rng_key = jax.random.PRNGKey(0)
    print("Running MCMC... (this may take a moment)")
    mcmc.run(rng_key, x_train, y_train)

    # Get samples
    samples = mcmc.get_samples()
    print(f"\n✓ MCMC complete!")
    print(f"Samples shape: w={samples['w'].shape}, b={samples['b'].shape}")

    # Step 3: Posterior analysis
    print("\nStep 3: Posterior analysis")

    # Posterior means
    w_mean = jnp.mean(samples['w'])
    b_mean = jnp.mean(samples['b'])
    sigma_mean = jnp.mean(samples['sigma'])

    # Posterior std
    w_std = jnp.std(samples['w'])
    b_std = jnp.std(samples['b'])
    sigma_std = jnp.std(samples['sigma'])

    print(f"Posterior means:")
    print(f"  w = {w_mean:.3f} ± {w_std:.3f} (true: {true_w:.2f})")
    print(f"  b = {b_mean:.3f} ± {b_std:.3f} (true: {true_b:.2f})")
    print(f"  sigma = {sigma_mean:.3f} ± {sigma_std:.3f} (true: {true_sigma:.2f})")

    # 95% credible intervals
    w_lower = jnp.percentile(samples['w'], 2.5)
    w_upper = jnp.percentile(samples['w'], 97.5)
    print(f"\n95% credible interval for w: [{w_lower:.3f}, {w_upper:.3f}]")
    print(f"True w in interval: {w_lower <= true_w <= w_upper}")

    # Step 4: Posterior predictive sampling
    print("\nStep 4: Posterior predictive sampling")

    # Test points
    x_test = jnp.linspace(-3, 3, 50)

    # Posterior predictive
    predictive = Predictive(bayesian_linear_model, samples)
    predictions = predictive(jax.random.PRNGKey(1), x_test)

    print(f"Predictions shape: {predictions['y'].shape}")
    print(f"  {predictions['y'].shape[0]} posterior samples")
    print(f"  {predictions['y'].shape[1]} test points")

    # Step 5: Uncertainty quantification
    print("\nStep 5: Uncertainty quantification")

    # Predictive mean and uncertainty
    y_pred_mean = jnp.mean(predictions['y'], axis=0)
    y_pred_std = jnp.std(predictions['y'], axis=0)

    # Credible intervals
    y_lower = jnp.percentile(predictions['y'], 2.5, axis=0)
    y_upper = jnp.percentile(predictions['y'], 97.5, axis=0)

    print(f"Mean prediction at x=0: {y_pred_mean[25]:.3f} ± {y_pred_std[25]:.3f}")
    print(f"95% interval at x=0: [{y_lower[25]:.3f}, {y_upper[25]:.3f}]")

    # Step 6: Convergence diagnostics
    print("\nStep 6: Convergence diagnostics")

    # Print MCMC summary
    mcmc.print_summary()

    # Step 7: JAX optimizations for prediction
    print("\nStep 7: JAX-optimized predictions")

    # JIT-compiled prediction function
    @jax.jit
    @jax.vmap
    def predict_sample(w, b, x):
        """Vectorized prediction over posterior samples"""
        return w * x + b

    # Fast predictions
    x_test_single = jnp.array(1.0)
    predictions_fast = predict_sample(samples['w'], samples['b'], x_test_single)

    print(f"Prediction at x={x_test_single}:")
    print(f"  Mean: {jnp.mean(predictions_fast):.3f}")
    print(f"  Std: {jnp.std(predictions_fast):.3f}")

    # Step 8: Compare with point estimate
    print("\nStep 8: Compare with frequentist point estimate")

    # MLE using JAX
    @jax.jit
    def mle_loss(params, x, y):
        w, b = params
        pred = w * x + b
        return jnp.mean((pred - y) ** 2)

    # Optimize
    import optax
    optimizer = optax.adam(learning_rate=0.1)

    params = jnp.array([0.0, 0.0])  # Initial guess
    opt_state = optimizer.init(params)

    @jax.jit
    def mle_step(params, opt_state, x, y):
        loss, grads = jax.value_and_grad(mle_loss)(params, x, y)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    # Run optimization
    for _ in range(100):
        params, opt_state, loss = mle_step(params, opt_state, x_train, y_train)

    mle_w, mle_b = params
    print(f"MLE estimates: w={mle_w:.3f}, b={mle_b:.3f}")
    print(f"Bayesian estimates: w={w_mean:.3f}, b={b_mean:.3f}")
    print(f"Key difference: Bayesian provides uncertainty quantification!")

    # Step 9: Out-of-distribution uncertainty
    print("\nStep 9: Out-of-distribution uncertainty")

    # Predictions far from training data
    x_ood = jnp.array([-10.0, 10.0])
    pred_ood = Predictive(bayesian_linear_model, samples)
    ood_predictions = pred_ood(jax.random.PRNGKey(2), x_ood)

    for i, x_val in enumerate(x_ood):
        ood_mean = jnp.mean(ood_predictions['y'][:, i])
        ood_std = jnp.std(ood_predictions['y'][:, i])
        print(f"  x={x_val:.1f}: mean={ood_mean:.3f}, std={ood_std:.3f}")

    print("\nNote: Uncertainty increases far from training data!")

    # Step 10: Model comparison (optional)
    print("\nStep 10: Summary")
    print("-" * 60)
    print("Bayesian inference with NumPyro provides:")
    print("  ✓ Full posterior distribution over parameters")
    print("  ✓ Uncertainty quantification for predictions")
    print("  ✓ Credible intervals (not just point estimates)")
    print("  ✓ Principled handling of uncertainty")
    print("  ✓ JAX-accelerated sampling (parallel chains)")
    print("\n✓ Bayesian inference workflow complete!")


if __name__ == '__main__':
    # Suppress NumPyro warnings for cleaner output
    import warnings
    warnings.filterwarnings('ignore')

    example_bayesian_inference()
