"""Example: Adaptive Hybrid Streaming Optimizer with Parameter Normalization

This example demonstrates the hybrid_streaming method which combines Adam warmup
with Gauss-Newton refinement for superior convergence on ill-conditioned problems.

Features demonstrated:
- method='hybrid_streaming' for multi-scale parameters
- HybridStreamingConfig presets (aggressive, conservative, memory_optimized)
- Automatic parameter normalization
- ParameterNormalizer for direct usage
- Covariance transformation for normalized parameters
- Three-phase optimization pipeline (Phase 0/1/2)

Three Issues Solved:
1. Weak gradients (scale imbalance) -> Parameter normalization
2. Slow convergence -> Streaming Gauss-Newton
3. Crude covariance -> Exact J^T J + transform

Run this example:
    python examples/demos/hybrid_streaming_demo.py
"""

import time

import jax
import jax.numpy as jnp
import numpy as np

from nlsq import HybridStreamingConfig, curve_fit
from nlsq.parameter_normalizer import NormalizedModelWrapper, ParameterNormalizer


def multi_scale_model(x, amplitude, decay_rate, offset):
    """Model with parameters spanning 6 orders of magnitude.

    This model is challenging for standard optimizers because:
    - amplitude ~ O(1000)
    - decay_rate ~ O(0.001)
    - offset ~ O(100)

    The 10^6 scale difference causes ill-conditioned Jacobians.
    """
    return amplitude * jnp.exp(-decay_rate * x) + offset


def main():
    print("=" * 70)
    print("Adaptive Hybrid Streaming Optimizer with Parameter Normalization")
    print("=" * 70)
    print()

    # Generate synthetic data with multi-scale true parameters
    np.random.seed(42)
    n_samples = 50000
    x_data = jnp.linspace(0, 100, n_samples)

    # True parameters spanning 6 orders of magnitude
    true_amplitude = 1500.0  # O(1000)
    true_decay_rate = 0.005  # O(0.001)
    true_offset = 75.0  # O(100)
    true_params = jnp.array([true_amplitude, true_decay_rate, true_offset])

    # Generate noisy data
    y_true = multi_scale_model(x_data, true_amplitude, true_decay_rate, true_offset)
    noise = jax.random.normal(jax.random.PRNGKey(42), y_true.shape) * 10
    y_data = y_true + noise

    print(f"Dataset: {n_samples:,} samples")
    print(f"True parameters:")
    print(f"  amplitude   = {true_amplitude:.4f} (O(1000))")
    print(f"  decay_rate  = {true_decay_rate:.6f} (O(0.001))")
    print(f"  offset      = {true_offset:.4f} (O(100))")
    print(f"  Scale ratio = {true_amplitude / true_decay_rate:.0e}")
    print()

    # Initial guess
    p0 = [1000.0, 0.01, 50.0]

    # Parameter bounds
    bounds = ([0.1, 0.0001, -1000], [10000, 1.0, 1000])

    # =========================================================================
    # Example 1: Basic hybrid_streaming usage
    # =========================================================================
    print("-" * 70)
    print("Example 1: Basic hybrid_streaming Usage")
    print("-" * 70)
    print()

    start_time = time.time()
    popt1, pcov1 = curve_fit(
        multi_scale_model,
        x_data,
        y_data,
        p0=p0,
        bounds=bounds,
        method="hybrid_streaming",
        verbose=1,
    )
    elapsed1 = time.time() - start_time

    print()
    print(f"Results (elapsed: {elapsed1:.2f}s):")
    print(f"  amplitude   = {popt1[0]:.4f} (true: {true_amplitude:.4f})")
    print(f"  decay_rate  = {popt1[1]:.6f} (true: {true_decay_rate:.6f})")
    print(f"  offset      = {popt1[2]:.4f} (true: {true_offset:.4f})")
    print(f"  Uncertainties: {jnp.sqrt(jnp.diag(pcov1))}")
    print()

    # =========================================================================
    # Example 2: Using HybridStreamingConfig presets
    # =========================================================================
    print("-" * 70)
    print("Example 2: HybridStreamingConfig Presets")
    print("-" * 70)
    print()

    # Test different presets
    presets = [
        ("default", HybridStreamingConfig()),
        ("aggressive", HybridStreamingConfig.aggressive()),
        ("conservative", HybridStreamingConfig.conservative()),
        ("memory_optimized", HybridStreamingConfig.memory_optimized()),
    ]

    print("Preset Comparison:")
    print("-" * 50)
    print(f"{'Preset':<18} {'Warmup LR':<12} {'GN Tol':<12} {'Chunk Size'}")
    print("-" * 50)

    for name, config in presets:
        print(
            f"{name:<18} {config.warmup_learning_rate:<12.4f} "
            f"{config.gauss_newton_tol:<12.0e} {config.chunk_size}"
        )
    print()

    # Use conservative preset for high-quality solution
    print("Fitting with conservative preset (highest quality)...")
    config = HybridStreamingConfig.conservative()

    start_time = time.time()
    popt2, pcov2 = curve_fit(
        multi_scale_model,
        x_data,
        y_data,
        p0=p0,
        bounds=bounds,
        method="hybrid_streaming",
        verbose=0,
    )
    elapsed2 = time.time() - start_time

    print(f"Results (elapsed: {elapsed2:.2f}s):")
    print(f"  amplitude   = {popt2[0]:.4f}")
    print(f"  decay_rate  = {popt2[1]:.6f}")
    print(f"  offset      = {popt2[2]:.4f}")
    print()

    # =========================================================================
    # Example 3: Custom configuration
    # =========================================================================
    print("-" * 70)
    print("Example 3: Custom HybridStreamingConfig")
    print("-" * 70)
    print()

    custom_config = HybridStreamingConfig(
        # Phase 0: Normalization
        normalize=True,
        normalization_strategy="bounds",  # Use bounds-based normalization
        # Phase 1: Adam warmup
        warmup_iterations=200,
        warmup_learning_rate=0.005,
        loss_plateau_threshold=1e-5,
        gradient_norm_threshold=1e-4,
        # Phase 2: Gauss-Newton
        gauss_newton_max_iterations=50,
        gauss_newton_tol=1e-9,
        trust_region_initial=1.0,
        # Streaming
        chunk_size=5000,
        # Precision
        precision="auto",
    )

    print("Custom configuration:")
    print(f"  normalization_strategy = '{custom_config.normalization_strategy}'")
    print(f"  warmup_iterations      = {custom_config.warmup_iterations}")
    print(f"  warmup_learning_rate   = {custom_config.warmup_learning_rate}")
    print(f"  gauss_newton_tol       = {custom_config.gauss_newton_tol}")
    print(f"  chunk_size             = {custom_config.chunk_size}")
    print()

    start_time = time.time()
    popt3, pcov3 = curve_fit(
        multi_scale_model,
        x_data,
        y_data,
        p0=p0,
        bounds=bounds,
        method="hybrid_streaming",
        verbose=0,
    )
    elapsed3 = time.time() - start_time

    print(f"Results (elapsed: {elapsed3:.2f}s):")
    print(f"  amplitude   = {popt3[0]:.4f}")
    print(f"  decay_rate  = {popt3[1]:.6f}")
    print(f"  offset      = {popt3[2]:.4f}")
    print()

    # =========================================================================
    # Example 4: Direct ParameterNormalizer usage
    # =========================================================================
    print("-" * 70)
    print("Example 4: Direct ParameterNormalizer Usage")
    print("-" * 70)
    print()

    # Create normalizer with bounds
    p0_array = jnp.array([1000.0, 0.01, 50.0])
    bounds_array = (
        jnp.array([0.1, 0.0001, -1000]),
        jnp.array([10000, 1.0, 1000]),
    )

    normalizer = ParameterNormalizer(p0_array, bounds_array, strategy="bounds")

    print("ParameterNormalizer demo:")
    print(f"  Original p0: {p0_array}")

    # Normalize parameters
    normalized_p0 = normalizer.normalize(p0_array)
    print(f"  Normalized:  {normalized_p0}")

    # Denormalize back
    denormalized = normalizer.denormalize(normalized_p0)
    print(f"  Denormalized:{denormalized}")
    print()

    # Create wrapped model
    print("Creating NormalizedModelWrapper...")

    def simple_model(x, a, b):
        return a * jnp.exp(-b * x)

    simple_p0 = jnp.array([50.0, 0.5])
    simple_bounds = (jnp.array([10.0, 0.0]), jnp.array([100.0, 1.0]))
    simple_normalizer = ParameterNormalizer(simple_p0, simple_bounds, strategy="bounds")
    wrapped_model = NormalizedModelWrapper(simple_model, simple_normalizer)

    print(f"  Wrapped model created for optimization in normalized space")
    print()

    # =========================================================================
    # Example 5: Covariance transformation
    # =========================================================================
    print("-" * 70)
    print("Example 5: Covariance Transformation")
    print("-" * 70)
    print()

    print("When optimizing in normalized space, covariance must be transformed:")
    print()
    print("  Cov_original = J @ Cov_normalized @ J.T")
    print()
    print("  where J = normalizer.normalization_jacobian")
    print()

    # Demo with the normalizer
    J = normalizer.normalization_jacobian
    print(f"Normalization Jacobian (diagonal):")
    print(f"  J = diag({jnp.diag(J)})")
    print()

    # Simulate normalized covariance
    pcov_normalized = jnp.diag(jnp.array([0.01, 0.01, 0.01]))
    pcov_original = J @ pcov_normalized @ J.T

    print("Example covariance transformation:")
    print(f"  Cov_normalized diagonal: {jnp.diag(pcov_normalized)}")
    print(f"  Cov_original diagonal:   {jnp.diag(pcov_original)}")
    print()

    # =========================================================================
    # Example 6: Comparison with TRF (without normalization)
    # =========================================================================
    print("-" * 70)
    print("Example 6: Comparison - hybrid_streaming vs TRF")
    print("-" * 70)
    print()

    print("Fitting with TRF (no automatic normalization)...")
    start_time = time.time()
    try:
        popt_trf, pcov_trf = curve_fit(
            multi_scale_model,
            x_data,
            y_data,
            p0=p0,
            bounds=bounds,
            method="trf",
            verbose=0,
        )
        elapsed_trf = time.time() - start_time
        trf_success = True
    except Exception as e:
        elapsed_trf = time.time() - start_time
        trf_success = False
        print(f"  TRF failed: {e}")

    print()
    print("Comparison Results:")
    print("-" * 50)
    print(f"{'Method':<20} {'Time (s)':<12} {'Amplitude':<15} {'Decay Rate'}")
    print("-" * 50)

    print(
        f"{'hybrid_streaming':<20} {elapsed1:<12.2f} "
        f"{popt1[0]:<15.4f} {popt1[1]:.6f}"
    )

    if trf_success:
        print(
            f"{'TRF':<20} {elapsed_trf:<12.2f} "
            f"{popt_trf[0]:<15.4f} {popt_trf[1]:.6f}"
        )
    else:
        print(f"{'TRF':<20} {'FAILED':<12}")

    print(
        f"{'True values':<20} {'-':<12} "
        f"{true_amplitude:<15.4f} {true_decay_rate:.6f}"
    )
    print()

    # =========================================================================
    # Summary
    # =========================================================================
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    print()
    print("Key Takeaways:")
    print()
    print("1. Use method='hybrid_streaming' for multi-scale parameters")
    print("   (parameters differing by >1000x)")
    print()
    print("2. HybridStreamingConfig presets:")
    print("   - aggressive(): Speed priority")
    print("   - conservative(): Quality priority")
    print("   - memory_optimized(): Large datasets")
    print()
    print("3. Normalization strategies:")
    print("   - 'auto': Default, adapts to bounds/p0")
    print("   - 'bounds': Normalize to [0, 1] using bounds")
    print("   - 'p0': Scale by initial parameter magnitudes")
    print("   - 'none': No normalization")
    print()
    print("4. Three-phase optimization:")
    print("   - Phase 0: Parameter normalization")
    print("   - Phase 1: Adam warmup (fast basin location)")
    print("   - Phase 2: Gauss-Newton (accurate minimum + covariance)")
    print()
    print("5. Covariance is automatically transformed back to original scale")
    print()
    print("=" * 70)
    print("Example complete!")


if __name__ == "__main__":
    main()
