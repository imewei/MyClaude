#!/usr/bin/env python3
"""
Comprehensive diagnostics for NLSQ optimization results.

Usage:
    from diagnose_optimization import diagnose_result

    result = optimizer.fit()
    diagnose_result(result)

Provides detailed convergence analysis, warnings, and recommendations.
"""

import jax.numpy as jnp


def diagnose_result(result, verbose=True):
    """
    Comprehensive diagnostics for optimization result.

    Args:
        result: OptimizeResult from NLSQ CurveFit or StreamingOptimizer
        verbose: If True, print detailed diagnostics

    Returns:
        dict: Diagnostic metrics and warnings
    """

    diagnostics = {
        'warnings': [],
        'recommendations': [],
        'metrics': {}
    }

    if verbose:
        print("=" * 70)
        print("NLSQ OPTIMIZATION DIAGNOSTICS")
        print("=" * 70)

    # 1. Convergence Status
    if verbose:
        print("\n1. Convergence Status")
        print(f"   Success: {result.success}")
        print(f"   Message: {result.message}")
        print(f"   Iterations: {result.nfev}")

    diagnostics['metrics']['success'] = result.success
    diagnostics['metrics']['iterations'] = result.nfev

    if not result.success:
        diagnostics['warnings'].append("Optimization did not converge")
        diagnostics['recommendations'].append("See troubleshooting section below")

    # 2. Cost Function Analysis
    if verbose:
        print("\n2. Cost Function")

    cost_reduction = (result.initial_cost - result.cost) / result.initial_cost * 100
    diagnostics['metrics']['cost_reduction_pct'] = cost_reduction

    if verbose:
        print(f"   Initial cost: {result.initial_cost:.6e}")
        print(f"   Final cost:   {result.cost:.6e}")
        print(f"   Reduction:    {cost_reduction:.2f}%")

    # Assess cost reduction quality
    if cost_reduction < 10:
        diagnostics['warnings'].append(f"Poor cost reduction ({cost_reduction:.1f}%)")
        diagnostics['recommendations'].extend([
            "Check initial guess quality",
            "Verify model matches data",
            "Try different loss function"
        ])
        if verbose:
            print("   ⚠️  WARNING: Poor cost reduction (<10%)")
    elif cost_reduction > 99:
        if verbose:
            print("   ✓ Excellent cost reduction")

    # 3. Gradient Analysis
    if verbose:
        print("\n3. Gradient")

    grad_norm = jnp.linalg.norm(result.grad)
    grad_inf = jnp.max(jnp.abs(result.grad))

    diagnostics['metrics']['gradient_norm'] = float(grad_norm)
    diagnostics['metrics']['gradient_inf'] = float(grad_inf)

    if verbose:
        print(f"   ||gradient||₂: {grad_norm:.6e}")
        print(f"   ||gradient||∞: {grad_inf:.6e}")

    if grad_norm > 1e-4:
        diagnostics['warnings'].append(f"Large gradient norm ({grad_norm:.2e})")
        diagnostics['recommendations'].extend([
            "May not be at local minimum",
            "Try increasing max_nfev",
            "Check for numerical issues"
        ])
        if verbose:
            print("   ⚠️  WARNING: Large gradient norm")
    else:
        if verbose:
            print("   ✓ Small gradient (near critical point)")

    # 4. Jacobian Conditioning
    if verbose:
        print("\n4. Jacobian Conditioning")

    jac_condition = jnp.linalg.cond(result.jac)
    diagnostics['metrics']['jacobian_condition'] = float(jac_condition)

    if verbose:
        print(f"   Condition number: {jac_condition:.6e}")

    if jac_condition > 1e12:
        diagnostics['warnings'].append("Extremely ill-conditioned Jacobian")
        diagnostics['recommendations'].extend([
            "Problem likely unsolvable as-is",
            "Reduce number of parameters",
            "Add regularization"
        ])
        if verbose:
            print("   ⚠️  CRITICAL: Extremely ill-conditioned")
    elif jac_condition > 1e10:
        diagnostics['warnings'].append("Ill-conditioned Jacobian")
        diagnostics['recommendations'].extend([
            "Consider parameter scaling",
            "Check for redundant parameters"
        ])
        if verbose:
            print("   ⚠️  WARNING: Ill-conditioned")
    elif jac_condition > 1e8:
        if verbose:
            print("   ⚠️  Moderately conditioned (results may be sensitive)")
    else:
        if verbose:
            print("   ✓ Well-conditioned")

    # 5. Parameter Analysis
    if verbose:
        print("\n5. Parameters")
        print(f"   Values: {result.x}")

    diagnostics['metrics']['parameters'] = [float(p) for p in result.x]

    # Check for parameters at bounds
    if hasattr(result, 'active_mask'):
        active = result.active_mask
        if jnp.any(active != 0):
            at_bounds = jnp.where(active != 0)[0]
            diagnostics['warnings'].append(f"Parameters at bounds: {at_bounds.tolist()}")
            diagnostics['recommendations'].append("Consider relaxing bounds")
            if verbose:
                print(f"   ⚠️  WARNING: Some parameters at bounds")
                print(f"      Active constraints: {at_bounds.tolist()}")

    # 6. Residual Analysis
    if verbose:
        print("\n6. Residuals")

    residuals = result.fun
    residual_mean = jnp.mean(residuals)
    residual_std = jnp.std(residuals)
    residual_max = jnp.max(jnp.abs(residuals))

    diagnostics['metrics']['residual_mean'] = float(residual_mean)
    diagnostics['metrics']['residual_std'] = float(residual_std)
    diagnostics['metrics']['residual_max'] = float(residual_max)

    if verbose:
        print(f"   Mean:   {residual_mean:.6e}")
        print(f"   Std:    {residual_std:.6e}")
        print(f"   Max:    {residual_max:.6e}")
        print(f"   Range:  [{jnp.min(residuals):.6e}, {jnp.max(residuals):.6e}]")

    # Check for systematic bias
    if abs(residual_mean) > 0.1 * residual_std:
        diagnostics['warnings'].append("Systematic bias in residuals")
        diagnostics['recommendations'].extend([
            "Model may be misspecified",
            "Consider additional terms"
        ])
        if verbose:
            print("   ⚠️  WARNING: Systematic bias in residuals")

    # 7. Summary
    if verbose:
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)

        if not diagnostics['warnings']:
            print("✓ No issues detected - optimization looks good!")
        else:
            print(f"⚠️  {len(diagnostics['warnings'])} warning(s) detected:\n")
            for i, warning in enumerate(diagnostics['warnings'], 1):
                print(f"{i}. {warning}")

            if diagnostics['recommendations']:
                print(f"\nRecommendations:")
                for rec in set(diagnostics['recommendations']):  # Remove duplicates
                    print(f"  • {rec}")

        print("=" * 70)

    return diagnostics


def troubleshoot_nonconvergence(model, x, y, p0, result):
    """
    Specific diagnostics for non-convergent optimization.

    Args:
        model: The model function
        x: Independent variable data
        y: Dependent variable data
        p0: Initial parameter guess
        result: Failed OptimizeResult
    """

    print("=" * 70)
    print("NON-CONVERGENCE TROUBLESHOOTING")
    print("=" * 70)

    # Check 1: Initial guess quality
    print("\n1. Initial Guess Quality")
    initial_pred = model(x, p0)
    initial_residuals = y - initial_pred
    initial_sse = jnp.sum(initial_residuals**2)
    final_sse = result.cost

    print(f"   Initial SSE: {initial_sse:.6e}")
    print(f"   Final SSE:   {final_sse:.6e}")

    if initial_sse < final_sse:
        print("   ❌ PROBLEM: Final cost worse than initial!")
        print("      → Optimization diverged")
        print("      → Try better initial guess")
        print("      → Reduce step size (use 'lm' method)")
    elif final_sse > 0.9 * initial_sse:
        print("   ⚠️  Poor improvement from initial guess")
        print("      → Try different p0")
        print("      → Check model correctness")

    # Check 2: Data quality
    print("\n2. Data Quality")
    x_range = jnp.max(x) - jnp.min(x)
    y_range = jnp.max(y) - jnp.min(y)
    y_noise = jnp.std(y - jnp.mean(y))

    print(f"   X range: [{jnp.min(x):.2e}, {jnp.max(x):.2e}]")
    print(f"   Y range: [{jnp.min(y):.2e}, {jnp.max(y):.2e}]")
    print(f"   Y noise level: {y_noise:.6e}")

    # Check for NaN/Inf
    if jnp.any(jnp.isnan(x)) or jnp.any(jnp.isnan(y)):
        print("   ❌ PROBLEM: NaN values in data!")
        print("      → Clean data before fitting")

    if jnp.any(jnp.isinf(x)) or jnp.any(jnp.isinf(y)):
        print("   ❌ PROBLEM: Infinite values in data!")
        print("      → Remove or cap extreme values")

    # Check 3: Parameter scaling
    print("\n3. Parameter Scaling")
    param_scales = jnp.abs(result.x)
    print(f"   Parameter magnitudes: {param_scales}")

    if jnp.max(param_scales) / (jnp.min(param_scales) + 1e-10) > 1e6:
        print("   ⚠️  WARNING: Poorly scaled parameters")
        print("      → Normalize parameters to similar scales")
        print("      → Use parameter transformation")

    # Check 4: Suggested fixes
    print("\n4. Suggested Fixes")
    print("   Try these in order:")
    print("   1. Improve initial guess (use domain knowledge)")
    print("   2. Normalize/scale parameters")
    print("   3. Use robust loss function ('huber' or 'soft_l1')")
    print("   4. Increase max_nfev")
    print("   5. Try different algorithm ('trf' vs 'lm')")
    print("   6. Add parameter bounds")
    print("   7. Simplify model (remove parameters)")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    print("Import this module and use diagnose_result(result)")
    print("Example:")
    print("  from diagnose_optimization import diagnose_result")
    print("  result = optimizer.fit()")
    print("  diagnose_result(result)")
